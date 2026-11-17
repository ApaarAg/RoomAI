"use client";

import { useCallback, useEffect, useId, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { apiClient, ApiError } from "@/lib/api/client";
import { ClientLatencyMetrics, DetectionItem, DetectionResponse, ReadinessResponse } from "@/lib/api/types";
import { SpatialObject3D, unprojectAllDetections } from "@/lib/spatial/coordinates";
import { LiveStatusBadge } from "@/components/navigation/LiveStatusBadge";
import { Interactive3DStudioCanvas } from "@/components/spatial/Interactive3DStudioCanvas";

type LifecycleState =
  | { status: "IDLE" }
  | { status: "UPLOADING" }
  | { status: "INFERENCING"; startTime: number }
  | { status: "SUCCESS"; response: DetectionResponse; latency: ClientLatencyMetrics }
  | { status: "EMPTY_RESULT"; response: DetectionResponse; latency: ClientLatencyMetrics }
  | { status: "ERROR"; error: ApiError };

interface PresetItem {
  id: string;
  name: string;
  category: string;
  url: string;
  description: string;
}

const PRESETS: PresetItem[] = [
  {
    id: "bedroom-01",
    name: "Modern Bedroom Suite",
    category: "Indoor / Group A",
    url: "/assets/roomai/exhibit-01/roomai-bedroom-source.jpeg",
    description: "Multi-fixture room containing bed, nightstand, lamps, and windows.",
  },
  {
    id: "living-01",
    name: "Living Room & Kitchen",
    category: "Indoor / Group A",
    url: "/assets/roomai/result-viewer/01-rgb.png",
    description: "Open plan space with cabinetry, shelving, and home appliances.",
  },
  {
    id: "editorial-01",
    name: "Architectural Editorial",
    category: "Indoor / Group A",
    url: "/assets/roomai/exhibit-01/roomai-bedroom-editorial.jpeg",
    description: "High-contrast architectural space with layered perspective.",
  },
  {
    id: "pipeline-01",
    name: "Pipeline Scene Study",
    category: "Indoor / Group A",
    url: "/assets/roomai/pipeline/01_rgb_image.webp",
    description: "Standard benchmark image used for 3D depth unprojection.",
  },
];

const MODEL_CHOICES = [
  {
    id: "indoor_detector_a3_yolo11s_800px",
    label: "Model A3 (Indoor 800px)",
    tag: "PRIMARY / PRODUCTION",
    description: "Trained on 12 indoor furniture classes (chairs, tables, beds, cabinets, appliances).",
  },
  {
    id: "exterior_detector_b2_yolo11s",
    label: "Model B2 (Exterior 640px)",
    tag: "SPECIALIST",
    description: "Trained on architectural exterior components (facade, porch, rooflines, doors).",
  },
  {
    id: "parking_detector_c2_yolo11s",
    label: "Model C2 (Parking 640px)",
    tag: "SPECIALIST",
    description: "Trained on vehicles, outdoor surfaces, perimeter elements, and parking spaces.",
  },
];

export default function ModelStudioPage() {
  const [selectedModel, setSelectedModel] = useState<string>("indoor_detector_a3_yolo11s_800px");
  const [imageSrc, setImageSrc] = useState<string>(PRESETS[0].url);
  const [currentFile, setCurrentFile] = useState<File | Blob | null>(null);
  const [activePresetId, setActivePresetId] = useState<string>(PRESETS[0].id);
  const [imageDims, setImageDims] = useState<{ width: number; height: number }>({ width: 1280, height: 960 });
  const [confidence, setConfidence] = useState<number>(0.25);
  const [lifecycle, setLifecycle] = useState<LifecycleState>({ status: "IDLE" });
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  const [showSpatial3D, setShowSpatial3D] = useState<boolean>(true);
  const [readiness, setReadiness] = useState<ReadinessResponse | null>(null);

  // 3D Viewport & Reconstruction Pipeline States
  const [viewMode, setViewMode] = useState<"2d" | "3d" | "split">("3d");
  const [isReconstructing, setIsReconstructing] = useState<boolean>(false);
  const [reconProgress, setReconProgress] = useState<number>(0);
  const [reconStage, setReconStage] = useState<string>("");
  const [reconstructedGlbUrl, setReconstructedGlbUrl] = useState<string | null>(null);
  const [reconstructedPlyUrl, setReconstructedPlyUrl] = useState<string | null>(null);
  const [reconstructionJobId, setReconstructionJobId] = useState<string | null>(null);
  const [reconError, setReconError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const activeRequestIdRef = useRef<number>(0);
  const objectUrlRef = useRef<string | null>(null);
  const fileInputId = useId();

  const cancelInference = useCallback(() => {
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
    setLifecycle({ status: "IDLE" });
  }, []);

  const runInference = useCallback(async (fileToUse?: File | Blob, modelToUse?: string, confToUse?: number) => {
    let targetFile = fileToUse || currentFile;
    const targetModel = modelToUse || selectedModel;
    const targetConf = confToUse !== undefined ? confToUse : confidence;

    if (!targetFile) {
      try {
        const activePreset = PRESETS.find((p) => p.id === activePresetId) || PRESETS[0];
        const res = await fetch(activePreset.url);
        targetFile = await res.blob();
        setCurrentFile(targetFile);
      } catch {
        return;
      }
    }

    abortControllerRef.current?.abort();
    const controller = new AbortController();
    abortControllerRef.current = controller;

    const currentReqId = ++activeRequestIdRef.current;
    setLifecycle({ status: "UPLOADING" });

    const inferTimer = setTimeout(() => {
      if (activeRequestIdRef.current === currentReqId) {
        setLifecycle({ status: "INFERENCING", startTime: performance.now() });
      }
    }, 40);

    try {
      const { data, clientLatency } = await apiClient.detectImage(targetFile, {
        confidenceThreshold: targetConf,
        modelId: targetModel,
        signal: controller.signal,
      });

      clearTimeout(inferTimer);
      if (activeRequestIdRef.current !== currentReqId) return;

      if (data.detections.length === 0) {
        setLifecycle({ status: "EMPTY_RESULT", response: data, latency: clientLatency });
      } else {
        setLifecycle({ status: "SUCCESS", response: data, latency: clientLatency });
      }
    } catch (err: unknown) {
      clearTimeout(inferTimer);
      if (activeRequestIdRef.current !== currentReqId) return;

      if (err instanceof ApiError) {
        if (err.code === "REQUEST_CANCELLED") {
          setLifecycle({ status: "IDLE" });
          return;
        }
        setLifecycle({ status: "ERROR", error: err });
      } else {
        setLifecycle({
          status: "ERROR",
          error: new ApiError(
            err instanceof Error ? err.message : String(err),
            500,
            "UNEXPECTED_ERROR",
            "Verify FastAPI backend is running on http://localhost:8000"
          ),
        });
      }
    }
  }, [currentFile, confidence, selectedModel, activePresetId]);

  const runFull3DReconstruction = useCallback(async () => {
    let targetFile = currentFile;
    if (!targetFile) {
      const activePreset = PRESETS.find((p) => p.id === activePresetId) || PRESETS[0];
      try {
        const res = await fetch(activePreset.url);
        targetFile = await res.blob();
        setCurrentFile(targetFile);
      } catch {
        return;
      }
    }

    setIsReconstructing(true);
    setReconProgress(0.1);
    setReconStage("Uploading image payload to backend...");
    setReconError(null);

    try {
      // 1. Upload image to backend
      const uploadRes = await apiClient.uploadImage(targetFile);
      setReconProgress(0.28);
      setReconStage("Submitting 3D room reconstruction pipeline job...");

      // 2. Submit reconstruction job
      const jobRes = await apiClient.submitReconstructionJob([uploadRes.upload_id], {
        modelId: "roomai_composite_v1",
        refineDepth: true,
      });

      setReconstructionJobId(jobRes.job_id);
      setReconProgress(0.45);
      setReconStage("Estimating metric depth (Depth Anything V2 Small)...");

      // 3. Poll status until complete
      const pollInterval = setInterval(async () => {
        try {
          const statusRes = await apiClient.pollJobStatus(jobRes.job_id);
          setReconProgress(Math.max(0.4, statusRes.progress));
          setReconStage(statusRes.stage);

          if (statusRes.status === "COMPLETED") {
            clearInterval(pollInterval);
            setIsReconstructing(false);
            const glb = apiClient.getExportGlbUrl(jobRes.job_id);
            const ply = apiClient.getExportPlyUrl(jobRes.job_id);
            setReconstructedGlbUrl(glb);
            setReconstructedPlyUrl(ply);
            setViewMode("3d");
          } else if (statusRes.status === "FAILED") {
            clearInterval(pollInterval);
            setIsReconstructing(false);
            setReconError(statusRes.error?.message || "3D reconstruction computation failed.");
          }
        } catch (pollErr) {
          clearInterval(pollInterval);
          setIsReconstructing(false);
          setReconError(pollErr instanceof Error ? pollErr.message : "Polling status failed");
        }
      }, 1500);
    } catch (err: unknown) {
      setIsReconstructing(false);
      setReconError(err instanceof Error ? err.message : String(err));
    }
  }, [currentFile, activePresetId]);

  // Load preset blob, fetch hardware status, and run initial inference on mount
  useEffect(() => {
    let active = true;

    // Check backend hardware readiness
    apiClient.checkReadiness()
      .then((data) => {
        if (active) setReadiness(data);
      })
      .catch(() => undefined);

    // Fetch initial preset and run inference
    fetch(PRESETS[0].url)
      .then((res) => res.blob())
      .then((blob) => {
        if (active) {
          setCurrentFile(blob);
          runInference(blob, selectedModel, confidence);
        }
      })
      .catch(() => undefined);

    return () => {
      active = false;
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
      abortControllerRef.current?.abort();
    };
  }, []);

  const handleSelectPreset = useCallback(async (preset: PresetItem) => {
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current);
      objectUrlRef.current = null;
    }
    setActivePresetId(preset.id);
    setImageSrc(preset.url);
    setReconstructedGlbUrl(null);
    setReconstructedPlyUrl(null);
    setReconstructionJobId(null);
    setReconError(null);

    try {
      const res = await fetch(preset.url);
      const blob = await res.blob();
      setCurrentFile(blob);
      runInference(blob, selectedModel, confidence);
    } catch {
      // silently handle
    }
  }, [selectedModel, confidence, runInference]);

  const handleImageSelected = useCallback((file: File) => {
    if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) {
      setLifecycle({
        status: "ERROR",
        error: new ApiError(
          "Unsupported format. Please select JPEG, PNG, or WebP.",
          415,
          "UNSUPPORTED_MEDIA_TYPE",
          "Convert image to JPEG, PNG, or WebP."
        ),
      });
      return;
    }

    if (file.size > 20 * 1024 * 1024) {
      setLifecycle({
        status: "ERROR",
        error: new ApiError(
          `Image size (${(file.size / (1024 * 1024)).toFixed(1)}MB) exceeds 20MB limit.`,
          413,
          "FILE_TOO_LARGE",
          "Resize or compress image before submitting."
        ),
      });
      return;
    }

    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    const newUrl = URL.createObjectURL(file);
    objectUrlRef.current = newUrl;
    setImageSrc(newUrl);
    setCurrentFile(file);
    setActivePresetId("");
    setReconstructedGlbUrl(null);
    setReconstructedPlyUrl(null);
    setReconstructionJobId(null);
    setReconError(null);

    const img = new Image();
    img.onload = () => {
      setImageDims({ width: img.naturalWidth, height: img.naturalHeight });
    };
    img.src = newUrl;

    runInference(file, selectedModel, confidence);
  }, [selectedModel, confidence, runInference]);

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleImageSelected(file);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file) handleImageSelected(file);
  };

  const detections: DetectionItem[] = useMemo(() => {
    if (lifecycle.status === "SUCCESS") {
      return lifecycle.response.detections;
    }
    return [];
  }, [lifecycle]);

  const spatialObjects: SpatialObject3D[] = useMemo(() => {
    if (detections.length === 0) return [];
    return unprojectAllDetections(detections, {
      aspectRatio: imageDims.width / imageDims.height,
    });
  }, [detections, imageDims]);

  // Group detections by class for summary badges
  const classBreakdown = useMemo(() => {
    const counts: Record<string, number> = {};
    detections.forEach((d) => {
      counts[d.class_name] = (counts[d.class_name] || 0) + 1;
    });
    return Object.entries(counts).sort((a, b) => b[1] - a[1]);
  }, [detections]);

  const isBusy = lifecycle.status === "UPLOADING" || lifecycle.status === "INFERENCING";

  return (
    <div className="studio-root">
      {/* 1. Header Navigation */}
      <header className="studio-nav">
        <div className="nav-left">
          <Link href="/" className="wordmark" aria-label="ROOMAI — Home">
            ROOMAI<span>®</span>
          </Link>
          <span className="studio-crumb">// MODEL PLAYGROUND</span>
        </div>

        <div className="nav-center">
          <Link href="/" className="nav-link">
            ← HOME
          </Link>
          <Link href="/explore" className="nav-link">
            PIPELINE &amp; RESEARCH
          </Link>
          <a
            href="http://localhost:8000/docs"
            target="_blank"
            rel="noopener noreferrer"
            className="nav-link"
          >
            API DOCS ↗
          </a>
        </div>

        <div className="nav-right">
          <LiveStatusBadge showDetails={true} />
        </div>
      </header>

      {/* 2. Main Studio Container */}
      <main className="studio-main">
        {/* Top Headline Banner */}
        <section className="studio-header">
          <div className="header-meta">
            <span className="eyebrow-tag">ROOM INTELLIGENCE // REAL-TIME CV ENGINE</span>
            <h1>Trained Model Playground &amp; Spatial Inference</h1>
            <p className="header-description">
              Execute live GPU inference using our production-trained <strong>Model A3 (YOLO11s at 800px)</strong>.
              Upload any room photograph or select a benchmark preset to inspect real 2D bounding boxes,
              class confidences, execution latency, and provisional 3D camera-ray unprojections.
            </p>
          </div>

          {/* Model Selector Tabs */}
          <div className="model-selector-bar" role="tablist" aria-label="Model Architecture Selection">
            {MODEL_CHOICES.map((choice) => (
              <button
                key={choice.id}
                type="button"
                role="tab"
                aria-selected={selectedModel === choice.id}
                className={`model-tab-btn ${selectedModel === choice.id ? "is-active" : ""}`}
                onClick={() => {
                  setSelectedModel(choice.id);
                  if (currentFile && !isBusy) {
                    runInference(currentFile, choice.id, confidence);
                  }
                }}
                disabled={isBusy}
              >
                <div className="model-tab-top">
                  <span className="model-tab-label">{choice.label}</span>
                  <span className="model-tab-tag">{choice.tag}</span>
                </div>
                <p className="model-tab-desc">{choice.description}</p>
              </button>
            ))}
          </div>
        </section>

        {/* 3. Interactive Stage & Controls */}
        <section className="studio-workspace">
          {/* Preset Picker & Input Controls */}
          <div className="workspace-controls-card">
            <div className="preset-bar">
              <span className="preset-label">BENCHMARK PRESETS:</span>
              <div className="preset-pill-group">
                {PRESETS.map((p) => (
                  <button
                    key={p.id}
                    type="button"
                    className={`preset-pill ${activePresetId === p.id ? "is-selected" : ""}`}
                    onClick={() => handleSelectPreset(p)}
                    disabled={isBusy}
                  >
                    {p.name}
                  </button>
                ))}
              </div>
            </div>

            <div className="inference-action-bar">
              <div className="action-bar-left">
                <input
                  type="file"
                  ref={fileInputRef}
                  id={fileInputId}
                  accept="image/jpeg,image/png,image/webp"
                  style={{ display: "none" }}
                  onChange={handleFileInputChange}
                />
                <button
                  type="button"
                  className="action-pill primary"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isBusy}
                >
                  UPLOAD ROOM PHOTO
                </button>
                <span className="supported-formats-hint">JPEG, PNG, WEBP · UP TO 20MB</span>
              </div>

              <div className="action-bar-right">
                <div className="confidence-control">
                  <label htmlFor="studio-conf-slider">
                    CONFIDENCE: <strong>{Math.round(confidence * 100)}%</strong>
                  </label>
                  <input
                    id="studio-conf-slider"
                    type="range"
                    min="0.10"
                    max="0.85"
                    step="0.05"
                    value={confidence}
                    disabled={isBusy}
                    onChange={(e) => {
                      const newConf = parseFloat(e.target.value);
                      setConfidence(newConf);
                      if (currentFile && !isBusy) {
                        runInference(currentFile, selectedModel, newConf);
                      }
                    }}
                    className="threshold-slider"
                  />
                </div>

                {!isBusy ? (
                  <button
                    type="button"
                    className="action-pill run-button"
                    onClick={() => runInference()}
                  >
                    RUN INFERENCE ON GPU ➔
                  </button>
                ) : (
                  <button
                    type="button"
                    className="action-pill danger"
                    onClick={cancelInference}
                  >
                    CANCEL INFERENCE
                  </button>
                )}
              </div>
            </div>
          </div>

          {/* 4. Viewport Mode Controls & Full 3D Reconstruction Action */}
          <div className="studio-view-mode-bar">
            <div className="view-mode-toggles" role="tablist" aria-label="Visual representation modes">
              <button
                type="button"
                role="tab"
                aria-selected={viewMode === "2d"}
                className={`view-mode-btn ${viewMode === "2d" ? "is-active" : ""}`}
                onClick={() => setViewMode("2d")}
              >
                2D OPTICAL VIEW
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={viewMode === "3d"}
                className={`view-mode-btn ${viewMode === "3d" ? "is-active" : ""}`}
                onClick={() => setViewMode("3d")}
              >
                3D INTERACTIVE ROOM
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={viewMode === "split"}
                className={`view-mode-btn ${viewMode === "split" ? "is-active" : ""}`}
                onClick={() => setViewMode("split")}
              >
                SPLIT VIEW (2D + 3D)
              </button>
            </div>

            <div className="recon-actions-group">
              <button
                type="button"
                className="action-pill recon-btn"
                disabled={isReconstructing}
                onClick={runFull3DReconstruction}
                title="Execute full Depth Anything V2 + RANSAC 3D room reconstruction pipeline"
              >
                {isReconstructing ? "SYNTHESIZING 3D..." : "⚡ RECONSTRUCT 3D ROOM (GLTF/PLY)"}
              </button>

              {reconstructedGlbUrl && (
                <a
                  href={reconstructedGlbUrl}
                  download="roomai_scene.glb"
                  className="action-pill download-btn"
                  title="Download binary glTF 2.0 3D scene"
                >
                  ⬇ DOWNLOAD .GLB
                </a>
              )}

              {reconstructedPlyUrl && (
                <a
                  href={reconstructedPlyUrl}
                  download="roomai_point_cloud.ply"
                  className="action-pill download-btn"
                  title="Download ASCII PLY point cloud"
                >
                  ⬇ DOWNLOAD .PLY
                </a>
              )}
            </div>
          </div>

          {reconError && (
            <div className="telemetry-error-box" style={{ marginTop: 8 }}>
              <span className="error-title">RECONSTRUCTION NOTICE</span>
              <p className="error-msg">{reconError}</p>
              <button type="button" className="action-pill primary" onClick={runFull3DReconstruction}>
                RETRY 3D PIPELINE
              </button>
            </div>
          )}

          {/* 5. Visual Viewport: 2D Optical, 3D Interactive Room, or Split View */}
          {viewMode === "2d" && (
            <div
              className="studio-stage"
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
            >
              <div className="studio-stage-inner">
                <img
                  src={imageSrc}
                  alt="Room scene for Model A3 inference"
                  className="studio-image"
                  onLoad={(e) => {
                    const img = e.currentTarget;
                    setImageDims({ width: img.naturalWidth, height: img.naturalHeight });
                  }}
                />

                {/* SVG 2D Bounding Box Overlay */}
                {detections.length > 0 && (
                  <svg
                    className="studio-overlay"
                    viewBox={`0 0 ${imageDims.width} ${imageDims.height}`}
                    preserveAspectRatio="none"
                    aria-label="Object detection bounding boxes"
                  >
                    {detections.map((det, idx) => {
                      const b = det.bbox;
                      const isHovered = hoveredIdx === idx;
                      const strokeColor = isHovered ? "#E08365" : "#B47762";
                      const fillColor = isHovered ? "rgba(224, 131, 101, 0.3)" : "rgba(180, 119, 98, 0.14)";
                      const boxW = Math.max(1, b.x2 - b.x1);
                      const boxH = Math.max(1, b.y2 - b.y1);

                      return (
                        <g
                          key={`${det.class_name}-${idx}`}
                          className="studio-box-group"
                          onMouseEnter={() => setHoveredIdx(idx)}
                          onMouseLeave={() => setHoveredIdx(null)}
                        >
                          <rect
                            x={b.x1}
                            y={b.y1}
                            width={boxW}
                            height={boxH}
                            fill={fillColor}
                            stroke={strokeColor}
                            strokeWidth={isHovered ? 3 : 2}
                            vectorEffect="non-scaling-stroke"
                          />
                          {/* Label Badge */}
                          <g transform={`translate(${b.x1}, ${Math.max(22, b.y1 - 6)})`}>
                            <rect
                              x="0"
                              y="-18"
                              width={det.class_name.length * 8 + 48}
                              height="20"
                              fill="#0A0908"
                              stroke={strokeColor}
                              strokeWidth="1"
                              rx="2"
                            />
                            <text
                              x="6"
                              y="-4"
                              fill="#F2F0EA"
                              fontFamily="var(--mono)"
                              fontSize="11"
                              fontWeight="600"
                              letterSpacing="0.05em"
                            >
                              {det.class_name.toUpperCase()} {Math.round(det.confidence * 100)}%
                            </text>
                          </g>
                        </g>
                      );
                    })}
                  </svg>
                )}

                {/* Busy HUD Overlay */}
                {isBusy && (
                  <div className="studio-hud-overlay">
                    <div className="hud-spinner" />
                    <p className="hud-status">
                      {lifecycle.status === "UPLOADING"
                        ? "STREAMING IMAGE PAYLOAD..."
                        : `RUNNING ${selectedModel.toUpperCase()} INFERENCE ON NVIDIA RTX 4060 GPU...`}
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}

          {viewMode === "3d" && (
            <div className="studio-stage" style={{ minHeight: 560 }}>
              <Interactive3DStudioCanvas
                imageSrc={imageSrc}
                spatialObjects={spatialObjects}
                hoveredIdx={hoveredIdx}
                onHoverObject={setHoveredIdx}
                glbUrl={reconstructedGlbUrl}
                isReconstructing={isReconstructing}
                reconstructionProgress={reconProgress}
                reconstructionStage={reconStage}
              />
            </div>
          )}

          {viewMode === "split" && (
            <div className="split-stage-container">
              <div className="split-left-pane">
                <div className="studio-stage-inner">
                  <img
                    src={imageSrc}
                    alt="2D Optical Room View"
                    className="studio-image"
                  />
                  {detections.length > 0 && (
                    <svg
                      className="studio-overlay"
                      viewBox={`0 0 ${imageDims.width} ${imageDims.height}`}
                      preserveAspectRatio="none"
                      aria-label="2D object detection bounding boxes"
                    >
                      {detections.map((det, idx) => {
                        const b = det.bbox;
                        const isHovered = hoveredIdx === idx;
                        const strokeColor = isHovered ? "#E08365" : "#B47762";
                        const fillColor = isHovered ? "rgba(224, 131, 101, 0.3)" : "rgba(180, 119, 98, 0.14)";
                        const boxW = Math.max(1, b.x2 - b.x1);
                        const boxH = Math.max(1, b.y2 - b.y1);

                        return (
                          <g
                            key={`split-${det.class_name}-${idx}`}
                            className="studio-box-group"
                            onMouseEnter={() => setHoveredIdx(idx)}
                            onMouseLeave={() => setHoveredIdx(null)}
                          >
                            <rect
                              x={b.x1}
                              y={b.y1}
                              width={boxW}
                              height={boxH}
                              fill={fillColor}
                              stroke={strokeColor}
                              strokeWidth={isHovered ? 3 : 2}
                              vectorEffect="non-scaling-stroke"
                            />
                          </g>
                        );
                      })}
                    </svg>
                  )}
                </div>
              </div>

              <div className="split-right-pane">
                <Interactive3DStudioCanvas
                  imageSrc={imageSrc}
                  spatialObjects={spatialObjects}
                  hoveredIdx={hoveredIdx}
                  onHoverObject={setHoveredIdx}
                  glbUrl={reconstructedGlbUrl}
                  isReconstructing={isReconstructing}
                  reconstructionProgress={reconProgress}
                  reconstructionStage={reconStage}
                />
              </div>
            </div>
          )}

          {/* 5. Telemetry & Results Section */}
          <div className="studio-telemetry-panel">
            {lifecycle.status === "SUCCESS" && (
              <>
                <div className="telemetry-stat-grid">
                  <div className="telemetry-card">
                    <span className="telemetry-label">DETECTED OBJECTS</span>
                    <strong className="telemetry-value highlight">{lifecycle.response.detections.length}</strong>
                  </div>
                  <div className="telemetry-card">
                    <span className="telemetry-label">GPU FORWARD PASS</span>
                    <strong className="telemetry-value">
                      {lifecycle.response.inference.latency_ms.toFixed(1)} ms
                    </strong>
                  </div>
                  <div className="telemetry-card">
                    <span className="telemetry-label">TOTAL ROUNDTRIP</span>
                    <strong className="telemetry-value">
                      {lifecycle.latency.total_roundtrip_ms} ms
                    </strong>
                  </div>
                  <div className="telemetry-card">
                    <span className="telemetry-label">CACHE DIGEST</span>
                    <strong className="telemetry-value">
                      {lifecycle.response.inference.cached ? "LRU CACHED" : "FRESH GPU"}
                    </strong>
                  </div>
                  <div className="telemetry-card">
                    <span className="telemetry-label">DEVICE ACTIVE</span>
                    <strong className="telemetry-value">
                      {lifecycle.response.inference.device.toUpperCase()}
                    </strong>
                  </div>
                </div>

                {/* Class Breakdown Chips */}
                {classBreakdown.length > 0 && (
                  <div className="class-breakdown-strip">
                    <span className="breakdown-title">CLASS DISTRIBUTION:</span>
                    <div className="breakdown-chips">
                      {classBreakdown.map(([cls, count]) => (
                        <span key={cls} className="class-chip">
                          <strong>{count}x</strong> {cls}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Toggle 3D Metrics Table */}
                <div className="telemetry-table-header">
                  <div>
                    <h3>PROVISIONAL 3D OPTICAL CAMERA-RAY UNPROJECTION</h3>
                    <p>
                      Camera pinhole unprojection (FOV = 34&deg;) onto assumed scene plane (Z = 3.0m).
                      Hovering a table row highlights the 2D bounding box on the image stage.
                    </p>
                  </div>
                  <button
                    type="button"
                    className="action-pill subtle"
                    onClick={() => setShowSpatial3D(!showSpatial3D)}
                  >
                    {showSpatial3D ? "HIDE 3D RAY METRICS" : "SHOW 3D RAY METRICS"}
                  </button>
                </div>

                {showSpatial3D && spatialObjects.length > 0 && (
                  <div className="studio-table-wrapper">
                    <table className="spatial-table">
                      <thead>
                        <tr>
                          <th>#</th>
                          <th>OBJECT CLASS</th>
                          <th>CONF</th>
                          <th>POS X (m)</th>
                          <th>POS Y (m)</th>
                          <th>POS Z (m)</th>
                          <th>WIDTH (m)</th>
                          <th>HEIGHT (m)</th>
                          <th>NDC (X, Y)</th>
                        </tr>
                      </thead>
                      <tbody>
                        {spatialObjects.map((obj, i) => (
                          <tr
                            key={`spatial-${obj.label}-${i}`}
                            className={hoveredIdx === i ? "is-row-hovered" : ""}
                            onMouseEnter={() => setHoveredIdx(i)}
                            onMouseLeave={() => setHoveredIdx(null)}
                          >
                            <td>{i + 1}</td>
                            <td><strong>{obj.label}</strong></td>
                            <td>{Math.round(obj.confidence * 100)}%</td>
                            <td>{obj.position[0]}</td>
                            <td>{obj.position[1]}</td>
                            <td>{obj.position[2]}</td>
                            <td>{obj.dimensions[0]}</td>
                            <td>{obj.dimensions[1]}</td>
                            <td>{obj.ndc[0]}, {obj.ndc[1]}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </>
            )}

            {lifecycle.status === "EMPTY_RESULT" && (
              <div className="telemetry-empty-box">
                <span className="empty-title">NO OBJECTS DETECTED AT CONFIDENCE &gt;= {Math.round(confidence * 100)}%</span>
                <p>Try lowering the confidence threshold to 15% or select another room photograph.</p>
              </div>
            )}

            {lifecycle.status === "ERROR" && (
              <div className="telemetry-error-box">
                <span className="error-title">API COMMUNICATION ERROR: {lifecycle.error.code}</span>
                <p className="error-msg">{lifecycle.error.message}</p>
                {lifecycle.error.remediation && (
                  <p className="error-fix">Remediation: {lifecycle.error.remediation}</p>
                )}
                <button type="button" className="action-pill primary" onClick={() => runInference()}>
                  RETRY INFERENCE
                </button>
              </div>
            )}

            {lifecycle.status === "IDLE" && (
              <div className="telemetry-standby-box">
                <span>STANDBY // SELECT A PRESET OR UPLOAD AN IMAGE AND CLICK &quot;RUN INFERENCE ON GPU&quot;</span>
              </div>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}
