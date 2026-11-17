"use client";

import { useCallback, useEffect, useId, useMemo, useRef, useState } from "react";
import { apiClient, ApiError } from "@/lib/api/client";
import { ClientLatencyMetrics, DetectionItem, DetectionResponse } from "@/lib/api/types";
import { SpatialObject3D, unprojectAllDetections } from "@/lib/spatial/coordinates";

type LifecycleState =
  | { status: "IDLE" }
  | { status: "UPLOADING" }
  | { status: "INFERENCING"; startTime: number }
  | { status: "SUCCESS"; response: DetectionResponse; latency: ClientLatencyMetrics }
  | { status: "EMPTY_RESULT"; response: DetectionResponse; latency: ClientLatencyMetrics }
  | { status: "ERROR"; error: ApiError };

const PRESET_IMAGE_URL = "/assets/roomai/result-viewer/01-rgb.png";

export function LiveInferenceExhibit() {
  const [imageSrc, setImageSrc] = useState<string>(PRESET_IMAGE_URL);
  const [currentFile, setCurrentFile] = useState<File | Blob | null>(null);
  const [imageDims, setImageDims] = useState<{ width: number; height: number }>({ width: 1920, height: 1080 });
  const [confidence, setConfidence] = useState<number>(0.35);
  const [lifecycle, setLifecycle] = useState<LifecycleState>({ status: "IDLE" });
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  const [showSpatial3D, setShowSpatial3D] = useState<boolean>(false);

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

  const runInference = useCallback(async (fileToUse?: File | Blob, confToUse?: number) => {
    const targetFile = fileToUse || currentFile;
    const targetConf = confToUse !== undefined ? confToUse : confidence;
    if (!targetFile) return;

    // Abort previous in-flight request
    abortControllerRef.current?.abort();
    const controller = new AbortController();
    abortControllerRef.current = controller;

    const currentReqId = ++activeRequestIdRef.current;
    setLifecycle({ status: "UPLOADING" });

    const inferStartTimer = setTimeout(() => {
      if (activeRequestIdRef.current === currentReqId) {
        setLifecycle({ status: "INFERENCING", startTime: performance.now() });
      }
    }, 40);

    try {
      const { data, clientLatency } = await apiClient.detectImage(targetFile, {
        confidenceThreshold: targetConf,
        signal: controller.signal,
      });

      clearTimeout(inferStartTimer);

      if (activeRequestIdRef.current !== currentReqId) return;

      if (data.detections.length === 0) {
        setLifecycle({ status: "EMPTY_RESULT", response: data, latency: clientLatency });
      } else {
        setLifecycle({ status: "SUCCESS", response: data, latency: clientLatency });
      }
    } catch (err: unknown) {
      clearTimeout(inferStartTimer);
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
            "Check browser console and backend status."
          ),
        });
      }
    }
  }, [currentFile, confidence]);

  const handleImageSelected = useCallback((file: File) => {
    // Client-side validation: format and size
    if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) {
      setLifecycle({
        status: "ERROR",
        error: new ApiError(
          "Unsupported media type. Please select a JPEG, PNG, or WebP image.",
          415,
          "CLIENT_UNSUPPORTED_TYPE",
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
          "CLIENT_FILE_TOO_LARGE",
          "Resize image before uploading."
        ),
      });
      return;
    }

    // Revoke previous blob URL if any
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current);
    }
    const newUrl = URL.createObjectURL(file);
    objectUrlRef.current = newUrl;
    setImageSrc(newUrl);
    setCurrentFile(file);
    setLifecycle({ status: "IDLE" });

    // Measure natural dimensions and run inference immediately
    const img = new Image();
    img.onload = () => {
      setImageDims({ width: img.naturalWidth, height: img.naturalHeight });
    };
    img.src = newUrl;

    runInference(file, confidence);
  }, [confidence, runInference]);

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleImageSelected(file);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file) handleImageSelected(file);
  };

  // Load preset image blob and auto-run initial inference on mount
  useEffect(() => {
    let active = true;
    fetch(PRESET_IMAGE_URL)
      .then((res) => res.blob())
      .then((blob) => {
        if (active) {
          setCurrentFile(blob);
          runInference(blob, confidence);
        }
      })
      .catch(() => undefined);
    return () => {
      active = false;
      if (objectUrlRef.current) {
        URL.revokeObjectURL(objectUrlRef.current);
      }
      abortControllerRef.current?.abort();
    };
  }, []);

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

  const isBusy = lifecycle.status === "UPLOADING" || lifecycle.status === "INFERENCING";

  return (
    <div className="live-inference-panel">
      {/* Factual Technical Model Specification Panel */}
      <div className="model-technical-strip" aria-label="Model A3 Technical Specifications">
        <div className="spec-item">
          <span className="spec-label">MODEL</span>
          <span className="spec-value">A3 / YOLO11s</span>
        </div>
        <div className="spec-item">
          <span className="spec-label">INPUT</span>
          <span className="spec-value">Indoor RGB Image</span>
        </div>
        <div className="spec-item">
          <span className="spec-label">DETECTION</span>
          <span className="spec-value">12-Class Indoor Objects</span>
        </div>
        <div className="spec-item">
          <span className="spec-label">INFERENCE</span>
          <span className="spec-value">Live Server-Side Engine</span>
        </div>
        <div className="spec-item">
          <span className="spec-label">DEVICE</span>
          <span className="spec-value">CUDA / NVIDIA GPU</span>
        </div>
      </div>

      {/* Top Controls Toolbar */}
      <div className="inference-toolbar">
        <div className="toolbar-left">
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
            className="action-pill"
            onClick={() => fileInputRef.current?.click()}
            disabled={isBusy}
          >
            UPLOAD ROOM IMAGE
          </button>
          <button
            type="button"
            className="action-pill subtle"
            onClick={async () => {
              const res = await fetch(PRESET_IMAGE_URL);
              const blob = await res.blob();
              setCurrentFile(blob);
              setImageSrc(PRESET_IMAGE_URL);
              setImageDims({ width: 1920, height: 1080 });
              runInference(blob, confidence);
            }}
            disabled={isBusy}
          >
            USE PRESET PHOTO
          </button>
        </div>

        <div className="toolbar-right">
          <label className="threshold-label" htmlFor="confidence-threshold-slider">
            CONFIDENCE: <strong>{Math.round(confidence * 100)}%</strong>
          </label>
          <input
            id="confidence-threshold-slider"
            type="range"
            min="0.15"
            max="0.85"
            step="0.05"
            value={confidence}
            disabled={isBusy}
            onChange={(e) => {
              const newConf = parseFloat(e.target.value);
              setConfidence(newConf);
              if (currentFile && !isBusy) {
                runInference(currentFile, newConf);
              }
            }}
            className="threshold-slider"
          />

          {!isBusy ? (
            <button
              type="button"
              className="action-pill primary"
              onClick={() => runInference()}
            >
              RUN MODEL A3 (GPU) ➔
            </button>
          ) : (
            <button
              type="button"
              className="action-pill danger"
              onClick={cancelInference}
            >
              CANCEL
            </button>
          )}
        </div>
      </div>

      {/* Main Viewport Stage with 2D Detection Overlay */}
      <div
        className="inference-stage"
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
      >
        <div className="inference-image-wrapper">
          <img
            src={imageSrc}
            alt="Room scene for Model A3 inference"
            className="inference-image"
            onLoad={(e) => {
              const img = e.currentTarget;
              setImageDims({ width: img.naturalWidth, height: img.naturalHeight });
            }}
          />

          {/* SVG 2D Bounding Box Overlay */}
          {detections.length > 0 && (
            <svg
              className="inference-overlay"
              viewBox={`0 0 ${imageDims.width} ${imageDims.height}`}
              preserveAspectRatio="none"
              aria-label="Object detection bounding boxes"
            >
              {detections.map((det, idx) => {
                const b = det.bbox;
                const isHovered = hoveredIdx === idx;
                const strokeColor = isHovered ? "#B47762" : "#8B4937";
                const fillColor = isHovered ? "rgba(180, 119, 98, 0.28)" : "rgba(139, 73, 55, 0.12)";
                const boxW = Math.max(1, b.x2 - b.x1);
                const boxH = Math.max(1, b.y2 - b.y1);

                return (
                  <g
                    key={`${det.class_name}-${idx}`}
                    className="detection-box-group"
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
                    <g transform={`translate(${b.x1}, ${Math.max(20, b.y1 - 6)})`}>
                      <rect
                        x="0"
                        y="-16"
                        width={det.class_name.length * 7.5 + 46}
                        height="18"
                        fill="#0A0908"
                        stroke={strokeColor}
                        strokeWidth="1"
                        rx="2"
                      />
                      <text
                        x="6"
                        y="-3"
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

          {/* Truthful Status HUD */}
          {isBusy && (
            <div className="inference-hud-overlay">
              <div className="hud-spinner" />
              <p className="hud-status">
                {lifecycle.status === "UPLOADING"
                  ? "UPLOADING IMAGE PAYLOAD..."
                  : "RUNNING MODEL A3 INFERENCE ON GPU..."}
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Latency Telemetry & Spatial Inspector */}
      <div className="inference-telemetry-bar">
        {lifecycle.status === "SUCCESS" && (
          <div className="telemetry-grid">
            <div className="telemetry-item">
              <span>DETECTED OBJECTS</span>
              <strong>{lifecycle.response.detections.length}</strong>
            </div>
            <div className="telemetry-item">
              <span>GPU INFERENCE</span>
              <strong>{lifecycle.response.inference.latency_ms.toFixed(1)} ms</strong>
            </div>
            <div className="telemetry-item">
              <span>TOTAL ROUNDTRIP</span>
              <strong>{lifecycle.latency.total_roundtrip_ms} ms</strong>
            </div>
            <div className="telemetry-item">
              <span>CACHED</span>
              <strong>{lifecycle.response.inference.cached ? "YES (LRU)" : "NO (FRESH)"}</strong>
            </div>
            <div className="telemetry-item">
              <span>DEVICE</span>
              <strong>{lifecycle.response.inference.device.toUpperCase()}</strong>
            </div>
            <div className="telemetry-actions">
              <button
                type="button"
                className={`telemetry-toggle-btn${showSpatial3D ? " is-active" : ""}`}
                onClick={() => setShowSpatial3D(!showSpatial3D)}
              >
                {showSpatial3D ? "HIDE PROVISIONAL 3D RAY METRICS" : "INSPECT PROVISIONAL 3D RAY METRICS"}
              </button>
            </div>
          </div>
        )}

        {lifecycle.status === "EMPTY_RESULT" && (
          <div className="telemetry-empty">
            <span>NO OBJECTS DETECTED AT CONFIDENCE &gt;= {Math.round(confidence * 100)}%</span>
            <small>Try lowering the confidence threshold to 25% or upload another scene.</small>
          </div>
        )}

        {lifecycle.status === "ERROR" && (
          <div className="telemetry-error">
            <span className="error-title">API ERROR: {lifecycle.error.code}</span>
            <p className="error-msg">{lifecycle.error.message}</p>
            {lifecycle.error.remediation && (
              <p className="error-fix">Remediation: {lifecycle.error.remediation}</p>
            )}
            <button type="button" className="action-pill primary" onClick={() => runInference()}>
              RETRY
            </button>
          </div>
        )}

        {lifecycle.status === "IDLE" && (
          <div className="telemetry-idle">
            <span>MODEL A3 STANDBY · READY FOR SCENE INFERENCE</span>
          </div>
        )}
      </div>

      {/* 3D Spatial Unprojection Inspector */}
      {showSpatial3D && spatialObjects.length > 0 && (
        <div className="spatial-inspector-panel">
          <div className="inspector-header">
            <span>PROVISIONAL 3D CAMERA-RAY PROJECTION (ASSUMED DEPTH Z = 3.0m)</span>
            <small>Pinhole optical unprojection (FOV 34°). Metric depth is provisionally assumed at 3.0m; not sensor measurement.</small>
          </div>
          <div className="inspector-table-wrapper">
            <table className="spatial-table">
              <thead>
                <tr>
                  <th>OBJECT</th>
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
        </div>
      )}
    </div>
  );
}
