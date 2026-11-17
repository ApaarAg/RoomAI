"use client";

import { SpatialObjectInfo } from "./Room3DCanvas";

interface SpatialObjectHUDProps {
  hoveredObject: SpatialObjectInfo | null;
  activeStageIndex: number;
}

export function SpatialObjectHUD({ hoveredObject, activeStageIndex }: SpatialObjectHUDProps) {
  // If user is inspecting a specific 3D segmented object
  if (hoveredObject) {
    return (
      <div className="spatial-inspection-hud is-hovered" role="region" aria-label="Spatial Object Inspection">
        <div className="hud-header">
          <div className="hud-tag">
            <span className="hud-indicator-dot" />
            <span>INSPECTING 3D OBJECT</span>
          </div>
          <span className="hud-id">{hoveredObject.id}</span>
        </div>

        <div className="hud-body">
          <div className="hud-title-row">
            <h4 className="hud-title">{hoveredObject.name}</h4>
            <span className="hud-category">{hoveredObject.category}</span>
          </div>

          <p className="hud-desc">{hoveredObject.description}</p>

          <div className="hud-metric-grid">
            <div className="hud-metric-cell">
              <span className="cell-label">3D BOUNDS (W×H×D)</span>
              <strong className="cell-value">
                {hoveredObject.dimensions.width.toFixed(2)}m × {hoveredObject.dimensions.height.toFixed(2)}m ×{" "}
                {hoveredObject.dimensions.depth.toFixed(2)}m
              </strong>
            </div>

            <div className="hud-metric-cell">
              <span className="cell-label">CAMERA SPACE [X, Y, Z]</span>
              <strong className="cell-value">
                [{hoveredObject.position.x > 0 ? "+" : ""}
                {hoveredObject.position.x.toFixed(2)}m,{" "}
                {hoveredObject.position.y > 0 ? "+" : ""}
                {hoveredObject.position.y.toFixed(2)}m,{" "}
                {hoveredObject.position.z.toFixed(2)}m]
              </strong>
            </div>

            <div className="hud-metric-cell">
              <span className="cell-label">MODEL CONFIDENCE</span>
              <strong className="cell-value">
                {(hoveredObject.confidence * 100).toFixed(1)}% <small>(Model A3 + Fusion)</small>
              </strong>
            </div>

            <div className="hud-metric-cell">
              <span className="cell-label">FLOOR RELATION</span>
              <strong className="cell-value">{hoveredObject.elevation}</strong>
            </div>
          </div>
        </div>

        <div className="hud-footer">
          <span>MODEL A3 INDOOR + FASTSAM SEGMENTATION + DEPTH ANYTHING V2</span>
        </div>
      </div>
    );
  }

  // When no object is hovered, display room-level spatial reconstruction telemetry
  return (
    <div className="spatial-inspection-hud is-idle" role="region" aria-label="Room Telemetry HUD">
      <div className="hud-header">
        <div className="hud-tag">
          <span className="hud-indicator-dot idle-dot" />
          <span>SPATIAL TELEMETRY // AI_013_010</span>
        </div>
        <span className="hud-id">
          {activeStageIndex >= 5 ? "INTERACTIVE 3D ROOM" : `STAGE 0${activeStageIndex + 1}`}
        </span>
      </div>

      <div className="hud-body">
        <div className="hud-title-row">
          <h4 className="hud-title">BEDROOM SPATIAL RECONSTRUCTION</h4>
          <span className="hud-category">RANSAC Plane + Metric Depth</span>
        </div>

        <div className="hud-metric-grid">
          <div className="hud-metric-cell">
            <span className="cell-label">ROOM DIMENSIONS (W × L × H)</span>
            <strong className="cell-value">2.54m × 4.26m × 2.21m</strong>
          </div>

          <div className="hud-metric-cell">
            <span className="cell-label">FLOOR ELEVATION / CAMERA</span>
            <strong className="cell-value">
              Y = +0.66m <small>(Camera H = 0.80m)</small>
            </strong>
          </div>

          <div className="hud-metric-cell">
            <span className="cell-label">METRIC DEPTH BOUNDS</span>
            <strong className="cell-value">2.51m — 7.18m (Metric V2)</strong>
          </div>

          <div className="hud-metric-cell">
            <span className="cell-label">SEGMENTED OBJECTS</span>
            <strong className="cell-value">5 Identified Geometry Envelopes</strong>
          </div>
        </div>
      </div>

      <div className="hud-footer">
        {activeStageIndex >= 5 ? (
          <span>HOVER OVER 3D FURNITURE BOXES TO INSPECT METRIC COORDINATES</span>
        ) : (
          <span>SCROLL DOWN TO PROGRESS TRANSFORMATION INTO 3D SPACE</span>
        )}
      </div>
    </div>
  );
}
