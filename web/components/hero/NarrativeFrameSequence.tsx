"use client";

import { useEffect, useMemo, useRef } from "react";
import manifest from "@/data/narrativeFrames.json";

type NarrativeFrame = {
  file: string;
  stage: string;
  progress: number;
  type: string;
};

const frames = manifest.sequence as NarrativeFrame[];
const sequenceRoot = "/assets/roomai/exhibit-01/sequence";

const stageAnnotations: Array<{ phase: string; title: string; detail: string; metric: string }> = [
  { phase: "01 / CAPTURE", title: "RGB SOURCE PLATE", detail: "Single-view architectural photograph · 1648 × 928 px", metric: "CAM: 34° FOV · RGB" },
  { phase: "01 / CAPTURE", title: "SCENE ANALYSIS", detail: "Structural saliency and geometric composition lines", metric: "COMPOSITION: 3-AXIS" },
  { phase: "02 / INFERENCE", title: "DEPTH EMERGENCE", detail: "Relative disparity estimation across room boundaries", metric: "DISPARITY ESTIMATE" },
  { phase: "02 / INFERENCE", title: "RAW DEPTH MAP", detail: "Dense non-metric depth estimation prior", metric: "GRADIENT: CONTINUOUS" },
  { phase: "03 / DISPLACEMENT", title: "DEPTH RELIEF SURFACE", detail: "Isometric micro-relief extrusion from pixel coordinates", metric: "SURFACE: DISPLACED" },
  { phase: "03 / DISPLACEMENT", title: "SPARSE POINT FIELD", detail: "Camera-space ray projection along optical center", metric: "DENSITY: 12.5K PTS" },
  { phase: "04 / PROJECTION", title: "DENSE POINT CLOUD", detail: "Luminance-coded spatial point volume", metric: "DENSITY: 64K PTS" },
  { phase: "04 / PROJECTION", title: "SPATIAL WIREFRAME", detail: "Planar boundary and room geometry alignment", metric: "GRID: BOUNDARY FIT" },
  { phase: "05 / GEOMETRY", title: "SURFACE TOPOLOGY", detail: "Coordinate grid mesh fitting to depth contours", metric: "TOPOLOGY: STRUCTURED" },
  { phase: "05 / GEOMETRY", title: "MATERIAL SYNTHESIS", detail: "Photographic re-projection onto spatial volume", metric: "ALBEDO + DISPLACEMENT" },
  { phase: "06 / EXPLORATION", title: "SEMANTIC PARSING", detail: "Spatial volume segmentation and bounding priors", metric: "SEMANTIC: BOUNDED" },
  { phase: "06 / EXPLORATION", title: "SPATIAL INTERACTION", detail: "Interactive viewport ready for spatial inspection", metric: "VIEWPORT: COHERENT" },
];

function clamp01(v: number) {
  return Math.min(1, Math.max(0, v));
}

export function NarrativeFrameSequence({ progress }: { progress: number }) {
  const frameRefs = useRef<(HTMLImageElement | null)[]>([]);
  const bounded = clamp01(progress);
  const upperIndex = frames.findIndex((frame) => frame.progress >= bounded);
  const nextIndex = upperIndex === -1 ? frames.length - 1 : upperIndex;
  const previousIndex = Math.max(0, nextIndex - 1);
  const previous = frames[previousIndex];
  const next = frames[nextIndex];
  const span = next.progress - previous.progress;
  const rawBlend = nextIndex === previousIndex || span <= 0 ? 0 : (bounded - previous.progress) / span;
  // Sinusoidal equal-energy crossfade to eliminate luminance dip
  const blend = Math.sin(clamp01(rawBlend) * (Math.PI / 2));
  const currentIndex = rawBlend >= 0.5 ? nextIndex : previousIndex;
  const currentFrame = frames[currentIndex];
  const annotation = stageAnnotations[currentIndex] ?? stageAnnotations[0];

  // Deep Obsidian Curtain as sequence approaches completion (0.84 -> 1.0)
  const curtainOpacity = clamp01((bounded - 0.84) / 0.16);
  // Subtle cinematic forward camera drift through the 12 states (1.0 -> 1.036)
  const cameraScale = 1.0 + bounded * 0.036;

  // Preload and decode all 12 narrative frames upfront
  useEffect(() => {
    frames.forEach((frame) => {
      const image = new Image();
      image.src = `${sequenceRoot}/${frame.file}`;
      void image.decode().catch(() => undefined);
    });
  }, []);

  // Update persistent frame opacity and visibility directly on DOM refs without re-rendering
  useEffect(() => {
    for (let i = 0; i < frames.length; i++) {
      const el = frameRefs.current[i];
      if (!el) continue;
      if (i === previousIndex && previousIndex === nextIndex) {
        el.style.opacity = "1";
        el.style.visibility = "visible";
      } else if (i === previousIndex) {
        el.style.opacity = String(1 - blend);
        el.style.visibility = "visible";
      } else if (i === nextIndex) {
        el.style.opacity = String(blend);
        el.style.visibility = "visible";
      } else {
        el.style.opacity = "0";
        el.style.visibility = "hidden";
      }
    }
  }, [previousIndex, nextIndex, blend]);

  return <div className="narrative-sequence" aria-label="Illustrative ROOMAI visual sequence">
    <div className="sequence-art" aria-hidden="true" style={{ transform: `scale(${cameraScale})` }}>
      {frames.map((frame, index) => (
        <img
          key={frame.file}
          ref={(el) => { frameRefs.current[index] = el; }}
          className={`sequence-frame frame-stage-${index}`}
          src={`${sequenceRoot}/${frame.file}`}
          alt=""
          loading={index < 3 ? "eager" : "lazy"}
          fetchPriority={index === 0 ? "high" : "auto"}
          decoding="async"
          style={{
            zIndex: index + 1,
            opacity: index === 0 ? 1 : 0,
            visibility: index <= 1 ? "visible" : "hidden",
          }}
        />
      ))}
      <div className="sequence-wash" />
      {/* Deep Obsidian Ambient Curtain Fade before unpinning */}
      <div
        className="sequence-curtain"
        style={{ opacity: curtainOpacity, pointerEvents: "none" }}
      />
      <div className="sequence-frame-id">ROOMAI / VISUAL STUDY 01</div>
      <div className="sequence-stage-mark">
        <span>STAGE</span>
        <strong>{String(currentIndex + 1).padStart(2, "0")}</strong>
        <i>/ 12</i>
      </div>
    </div>

    <div
      className="sequence-copy"
      style={{
        transform: `translate3d(0, ${-bounded * 20}px, 0)`,
        opacity: Math.max(0.12, 1 - curtainOpacity * 0.88),
      }}
    >
      <p className="sequence-kicker">
        SINGLE-VIEW SPATIAL RESEARCH <span>·</span> {annotation.phase}
      </p>
      <h1 id="hero-title">From a Single Image<br />to an <em>Interactive World.</em></h1>

      {/* Progressive editorial stage callout */}
      <div className="sequence-telemetry-badge" key={annotation.title}>
        <div className="telemetry-lead">
          <span className="telemetry-pulse" />
          <span className="telemetry-label">{annotation.title}</span>
        </div>
        <p className="telemetry-detail">{annotation.detail}</p>
        <span className="telemetry-metric">{annotation.metric}</span>
      </div>

      <p className="sequence-description">
        Reconstruct, understand, and explore indoor environments through depth estimation, spatial representations, and interactive 3D.
      </p>
      <div className="sequence-status">
        <span className="status-dot" />ILLUSTRATIVE VISUAL STUDY <span>·</span> NOT MODEL OUTPUT
      </div>
      <a className="sequence-link" href="#explore">EXPLORE THE VERIFIED DEPTH STUDY <span>↓</span></a>
    </div>

    <div className="sequence-footer">
      <span className="sequence-caption">
        {currentFrame.stage.toUpperCase()} <i>/ {currentFrame.type.toUpperCase()}</i>
      </span>
      <div
        className="sequence-progress"
        role="progressbar"
        aria-label="Visual sequence progress"
        aria-valuemin={1}
        aria-valuemax={12}
        aria-valuenow={currentIndex + 1}
      >
        <span style={{ transform: `scaleX(${bounded})` }} />
        <div>
          {frames.map((frame, index) => (
            <i key={frame.file} className={index === currentIndex ? "is-current" : ""} />
          ))}
        </div>
      </div>
      <span className="sequence-scroll">SCROLL TO TRACE THE IMAGE <b>↘</b></span>
    </div>
  </div>;
}
