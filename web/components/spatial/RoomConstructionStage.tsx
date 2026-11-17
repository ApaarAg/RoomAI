"use client";

import { useRef, useState, useCallback, useEffect } from "react";
import Image from "next/image";
import { Room3DCanvas, SpatialObjectInfo } from "./Room3DCanvas";
import { SpatialObjectHUD } from "./SpatialObjectHUD";

export interface StageDefinition {
  step: string;
  name: string;
  kicker: string;
  headline: string;
  subhead: string;
  image: string;
  is3D: boolean;
}

export const CONSTRUCTION_STAGES: StageDefinition[] = [
  {
    step: "01",
    name: "RGB PHOTOGRAPH",
    kicker: "01 / RGB // OPTICAL BASELINE",
    headline: "THE SINGLE-VIEW CAPTURE.",
    subhead: "Standard 2D bedroom photograph captured under natural ambient indoor lighting.",
    image: "/img2/01-rgb.png",
    is3D: false,
  },
  {
    step: "02",
    name: "SCENE UNDERSTANDING",
    kicker: "02 / SCENE // GEOMETRIC PRIORS",
    headline: "SPATIAL OBJECT LOCALIZATION.",
    subhead: "Model A3 (YOLO11s @ 800px) predicts semantic class bounds across 12 indoor categories.",
    image: "/img2/02-scene-priors.png",
    is3D: false,
  },
  {
    step: "03",
    name: "METRIC DEPTH",
    kicker: "03 / DEPTH // DENSE RANGE MAP",
    headline: "DEPTH EMERGES FROM SHADOWS.",
    subhead: "Depth Anything V2 Metric Small generates calibrated physical distance (2.51m — 7.18m range).",
    image: "/img2/03-raw-depth.png",
    is3D: false,
  },
  {
    step: "04",
    name: "CAMERA-RAY UNPROJECTION",
    kicker: "04 / UNPROJECTION // OPTICAL RAYS",
    headline: "PIXELS LEAVE THE IMAGE PLANE.",
    subhead: "Pinhole geometry (34° optical FOV) unprojects 2D pixels along diverging perspective camera rays.",
    image: "/img2/04-depth-unprojection.png",
    is3D: true,
  },
  {
    step: "05",
    name: "POINT CLOUD",
    kicker: "05 / POINT FIELD // 40,000 VERTICES",
    headline: "SPATIAL COHERENCE FORMS.",
    subhead: "Dense 3D point cloud synthesized with true metric camera coordinates and floor plane alignment.",
    image: "/img2/05-point-cloud.png",
    is3D: true,
  },
  {
    step: "06",
    name: "INTERACTIVE 3D ROOM",
    kicker: "06 / SPATIAL ROOM // GLTF 2.0 DIGITAL ASSET",
    headline: "THE ROOM BECOMES A WORLD.",
    subhead: "RANSAC floor recovery and 3D bounding geometry envelopes enable full orbit, zoom, and spatial inspection.",
    image: "/img2/06-ply-glb-export.png",
    is3D: true,
  },
];

interface RoomConstructionStageProps {
  activeStageIndex: number;
  overallProgress: number; // 0.0 to 1.0
  onJumpToStage: (index: number) => void;
}

export function RoomConstructionStage({
  activeStageIndex,
  overallProgress,
  onJumpToStage,
}: RoomConstructionStageProps) {
  const [hoveredObject, setHoveredObject] = useState<SpatialObjectInfo | null>(null);
  const [isFreeOrbit, setIsFreeOrbit] = useState(false);
  const resetCameraRef = useRef<(() => void) | null>(null);
  const setPresetRef = useRef<((preset: "default" | "top" | "iso") => void) | null>(null);

  const stage = CONSTRUCTION_STAGES[activeStageIndex] || CONSTRUCTION_STAGES[0];
  const isFinalStage = activeStageIndex >= 5;
  const is3DActive = activeStageIndex >= 3;

  // Normalized progress within the 3D transition zone (stages 3 to 5)
  // Stage 3: progress in [0.50, 0.66], Stage 4: [0.66, 0.83], Stage 5: [0.83, 1.0]
  const stageProgress = Math.max(0, Math.min(1, (overallProgress - 0.5) / 0.5));

  const handleReset = () => {
    if (resetCameraRef.current) resetCameraRef.current();
  };

  const handlePreset = (preset: "default" | "top" | "iso") => {
    if (setPresetRef.current) setPresetRef.current(preset);
  };

  return (
    <div className={`room-construction-stage-plate ${isFinalStage ? "is-final-room" : ""}`}>
      {/* 1. Plate Topbar */}
      <div className="stage-plate-topbar">
        <div className="topbar-meta">
          <span className="topbar-index">SECTION 03 // 3D ROOM RECONSTRUCTION CONTINUUM</span>
          <span className={`topbar-status-badge ${isFinalStage ? "is-live-3d" : ""}`}>
            {isFinalStage
              ? "● 3D ROOM SYNTHESIZED — INTERACTIVE ORBIT ACTIVE"
              : "◈ SCROLL-DRIVEN TRANSFORMATION // REVERSIBLE CONTINUUM"}
          </span>
        </div>

        {/* 3D Camera Controls Toolbar (Active in Stage 06) */}
        {isFinalStage && (
          <div className="topbar-3d-toolbar" role="toolbar" aria-label="3D Camera Controls">
            <span className="toolbar-label">VIEW PRESETS:</span>
            <button
              type="button"
              className="toolbar-btn"
              onClick={() => handlePreset("default")}
              title="Standard perspective eye-level view"
            >
              PERSPECTIVE
            </button>
            <button
              type="button"
              className="toolbar-btn"
              onClick={() => handlePreset("top")}
              title="Top-down architectural plan view"
            >
              PLAN (TOP)
            </button>
            <button
              type="button"
              className="toolbar-btn"
              onClick={() => handlePreset("iso")}
              title="45-degree isometric spatial view"
            >
              ISO 45°
            </button>
            <button
              type="button"
              className="toolbar-btn btn-reset"
              onClick={handleReset}
              title="Reset camera to default origin"
            >
              RESET ⟲
            </button>
          </div>
        )}
      </div>

      {/* 2. Dynamic Narrative Headline (Synchronized with Scroll) */}
      <div className="stage-editorial-banner">
        <div className="banner-text">
          <div className="banner-kicker-row">
            <span className="banner-kicker">{stage.kicker}</span>
            <span className="banner-counter">
              STAGE 0{activeStageIndex + 1} / 06
            </span>
          </div>
          <h3 className="banner-headline">{stage.headline}</h3>
          <p className="banner-subhead">{stage.subhead}</p>
        </div>

        <div className="banner-actions">
          <button
            type="button"
            className="stage-step-btn"
            onClick={() => onJumpToStage(Math.max(0, activeStageIndex - 1))}
            disabled={activeStageIndex === 0}
            aria-label="Previous transformation stage"
          >
            ← PREV
          </button>
          <button
            type="button"
            className="stage-step-btn"
            onClick={() => onJumpToStage(Math.min(CONSTRUCTION_STAGES.length - 1, activeStageIndex + 1))}
            disabled={activeStageIndex === CONSTRUCTION_STAGES.length - 1}
            aria-label="Next transformation stage"
          >
            NEXT →
          </button>
        </div>
      </div>

      {/* 3. Visual Stage: 2D Stack Layer + 3D WebGL Canvas Layer */}
      <div className="stage-viewport-area">
        {/* Layer A: 2D Image & Depth Transition Stack */}
        <div
          className="stage-2d-layer"
          style={{
            opacity: activeStageIndex <= 3 ? 1 : Math.max(0, 1 - (overallProgress - 0.66) / 0.15),
            pointerEvents: activeStageIndex <= 3 ? "auto" : "none",
          }}
          aria-hidden={activeStageIndex >= 4}
        >
          {CONSTRUCTION_STAGES.map((s, idx) => (
            <div
              key={s.step}
              className={`stage-image-wrapper ${idx === activeStageIndex ? "is-visible" : "is-hidden"}`}
              style={{
                zIndex: idx + 1,
                opacity: idx === activeStageIndex ? 1 : 0,
                transform:
                  idx === activeStageIndex
                    ? "scale(1.0)"
                    : idx < activeStageIndex
                    ? "scale(1.02) translateY(-4px)"
                    : "scale(0.98) translateY(4px)",
              }}
            >
              <Image
                src={s.image}
                alt={s.headline}
                fill
                priority={idx <= 1}
                className="stage-image-content"
                sizes="(max-width: 1024px) 100vw, 1200px"
              />
            </div>
          ))}

          {/* Graticule Reticle Lines */}
          <div className="stage-cad-reticle" aria-hidden="true">
            <span className="reticle-corner tl" />
            <span className="reticle-corner tr" />
            <span className="reticle-corner bl" />
            <span className="reticle-corner br" />
            <div className="reticle-axis-x" />
            <div className="reticle-axis-y" />
          </div>
        </div>

        {/* Layer B: WebGL 3D Interactive Room Canvas */}
        <div
          className="stage-3d-layer"
          style={{
            opacity: activeStageIndex >= 3 ? 1 : 0,
            pointerEvents: is3DActive ? "auto" : "none",
          }}
        >
          <Room3DCanvas
            stageProgress={stageProgress}
            activeStageIndex={activeStageIndex}
            isInteractiveOrbit={isFreeOrbit || isFinalStage}
            onObjectHover={setHoveredObject}
            onResetViewRef={(fn) => {
              resetCameraRef.current = fn;
            }}
            onSetPresetRef={(fn) => {
              setPresetRef.current = fn;
            }}
          />
        </div>

        {/* Inspection HUD Overlay */}
        <SpatialObjectHUD hoveredObject={hoveredObject} activeStageIndex={activeStageIndex} />

        {/* Interaction Hint Pill (Shown in Stage 06) */}
        {isFinalStage && (
          <div className="stage-orbit-hint" role="note">
            <div className="hint-pill">
              <span className="hint-dot" />
              <span>LEFT DRAG TO ORBIT · RIGHT DRAG TO PAN · WHEEL TO ZOOM</span>
            </div>
          </div>
        )}
      </div>

      {/* 4. Bottom Technical Stage Navigator (01 through 06) */}
      <div className="stage-bottom-nav" role="tablist" aria-label="Reconstruction stages">
        <div className="stage-pills-row">
          {CONSTRUCTION_STAGES.map((s, idx) => (
            <button
              key={s.step}
              type="button"
              role="tab"
              aria-selected={activeStageIndex === idx}
              className={`stage-pill-button ${activeStageIndex === idx ? "is-active" : ""} ${
                idx < activeStageIndex ? "is-past" : ""
              }`}
              onClick={() => onJumpToStage(idx)}
            >
              <span className="pill-index">{s.step}</span>
              <strong className="pill-title">{s.name}</strong>
            </button>
          ))}
        </div>

        <div className="stage-progress-track">
          <div
            className="stage-progress-fill"
            style={{ width: `${Math.round(overallProgress * 100)}%` }}
          />
        </div>
      </div>
    </div>
  );
}
