"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { Pipeline } from "@/components/ui/Pipeline";
import { TechnicalLabel } from "@/components/ui/TechnicalLabel";

if (typeof window !== "undefined") {
  gsap.registerPlugin(ScrollTrigger);
}

const stages = [
  { label: "RGB IMAGE", file: "01_rgb_image.webp" },
  { label: "MODEL A3 DETECTOR", file: "02_hunyuan_scene_priors.webp" },
  { label: "METRIC DEPTH ESTIMATION", file: "03_raw_depth_concept.webp" },
  { label: "3D CAMERA UNPROJECTION", file: "04_depth_unprojection_concept.webp" },
  { label: "SPATIAL SCENE GRAPH", file: "05_point_cloud_concept.webp" },
  { label: "GLB / PLY EXPORT", file: "06_ply_glb_export_concept.webp" },
] as const;

const assetRoot = "/assets/roomai/pipeline";

export function PipelineExperience() {
  const journey = useRef<HTMLDivElement>(null);
  const layerRefs = useRef<(HTMLImageElement | null)[]>([]);
  const progressLine = useRef<HTMLSpanElement>(null);
  const listProgress = useRef<HTMLElement>(null);
  const stageCount = useRef<HTMLSpanElement>(null);
  const stageName = useRef<HTMLElement>(null);
  const stageIndicator = useRef<HTMLDivElement>(null);
  const scanCursor = useRef<HTMLDivElement>(null);

  const [isMobile, setIsMobile] = useState(false);
  const [mobileStage, setMobileStage] = useState(0);

  // Preload and decode all stage images upfront
  useEffect(() => {
    stages.forEach((stage) => {
      const img = new Image();
      img.src = `${assetRoot}/${stage.file}`;
      void img.decode().catch(() => undefined);
    });
  }, []);

  const updateStageDisplay = useCallback((activeStage: number, progressVal: number) => {
    const root = journey.current;
    const progressElement = progressLine.current;
    const listProgressElement = listProgress.current;
    const countElement = stageCount.current;
    const nameElement = stageName.current;
    const indicatorElement = stageIndicator.current;
    if (!root || !progressElement || !listProgressElement || !countElement || !nameElement || !indicatorElement) return;

    countElement.textContent = `${String(activeStage + 1).padStart(2, "0")} / 06`;
    nameElement.textContent = stages[activeStage].label;
    indicatorElement.setAttribute("aria-valuenow", String(activeStage + 1));
    indicatorElement.setAttribute("aria-label", `Pipeline stage ${activeStage + 1} of 6: ${stages[activeStage].label}`);

    progressElement.style.transform = `scaleX(${progressVal})`;
    listProgressElement.style.transform = `scaleY(${progressVal})`;

    const boxItems = Array.from(root.querySelectorAll<HTMLElement>(".architecture-flow li"));
    const sequenceItems = Array.from(root.querySelectorAll<HTMLElement>(".engine-process .pipeline li"));

    boxItems.forEach((item, index) => {
      const boxStage = Math.min(index, 4);
      item.classList.toggle("is-active", boxStage === activeStage || (index === 4 && activeStage >= 4));
      item.classList.toggle("is-past", boxStage < activeStage);
      item.classList.toggle("is-next", boxStage === activeStage + 1);
    });

    sequenceItems.forEach((item, index) => {
      const active = index === activeStage;
      item.classList.toggle("is-active", active);
      item.classList.toggle("is-past", index < activeStage);
      item.classList.toggle("is-next", index === activeStage + 1);
      if (active) item.setAttribute("aria-current", "step");
      else item.removeAttribute("aria-current");
    });
  }, []);

  // Mobile step navigation
  const setStepMobile = useCallback((targetIndex: number) => {
    const clamped = Math.max(0, Math.min(stages.length - 1, targetIndex));
    setMobileStage(clamped);

    for (let i = 0; i < stages.length; i++) {
      const el = layerRefs.current[i];
      if (!el) continue;
      if (i === clamped) {
        el.style.opacity = "1";
        el.style.visibility = "visible";
        el.style.transform = "none";
        el.style.filter = "none";
        el.style.maskImage = "none";
        el.style.webkitMaskImage = "none";
      } else {
        el.style.opacity = "0";
        el.style.visibility = "hidden";
        el.style.maskImage = "none";
        el.style.webkitMaskImage = "none";
      }
    }

    const cursor = scanCursor.current;
    if (cursor) cursor.style.opacity = "0";

    updateStageDisplay(clamped, clamped / (stages.length - 1));
  }, [updateStageDisplay]);

  useEffect(() => {
    const root = journey.current;
    const cursor = scanCursor.current;
    if (!root) return;

    const mobileCheck = () => {
      const mobile = window.innerWidth < 768;
      setIsMobile(mobile);
      return mobile;
    };

    mobileCheck();
    let previousActiveStage = -1;

    const update = (value: number) => {
      const progress = gsap.utils.clamp(0, 1, value);
      const scaled = progress * (stages.length - 1);
      const segment = Math.min(stages.length - 2, Math.floor(scaled));
      const rawMix = progress >= 1 ? 1 : scaled - segment;
      // Equal-energy optical crossfade curve (cos^2 + sin^2 = 1)
      const outgoingOpacity = Math.cos(rawMix * (Math.PI / 2));
      const incomingOpacity = Math.sin(rawMix * (Math.PI / 2));
      const activeStage = Math.min(stages.length - 1, segment + (rawMix >= 0.5 ? 1 : 0));

      for (let i = 0; i < stages.length; i++) {
        const el = layerRefs.current[i];
        if (!el) continue;

        if (i < segment) {
          // Completed past layers
          el.style.opacity = "0";
          el.style.visibility = "hidden";
          el.style.maskImage = "none";
          el.style.webkitMaskImage = "none";
        } else if (i === segment) {
          // Outgoing layer: smooth fade-out with micro-scale settling
          el.style.visibility = "visible";
          el.style.opacity = outgoingOpacity.toFixed(4);
          el.style.transform = `translate3d(0, 0, 0) scale(${(1.0 + 0.003 * rawMix).toFixed(4)})`;
          el.style.filter = `brightness(${(1.0 - 0.03 * rawMix).toFixed(3)})`;
          el.style.maskImage = "none";
          el.style.webkitMaskImage = "none";
        } else if (i === segment + 1) {
          // Incoming layer: soft feathered aperture reveal + optical crossfade + micro-scale arrival
          el.style.visibility = "visible";
          el.style.opacity = incomingOpacity.toFixed(4);
          el.style.transform = `translate3d(0, 0, 0) scale(${(0.997 + 0.003 * rawMix).toFixed(4)})`;
          el.style.filter = `brightness(${(0.97 + 0.03 * rawMix).toFixed(3)})`;

          if (rawMix > 0.015 && rawMix < 0.985) {
            const sweepX = rawMix * 100;
            // 20% soft optical gradient feather prevents harsh bisection of room landmarks
            const maskGrad = `linear-gradient(90deg, #000 0%, #000 ${Math.max(0, sweepX - 10).toFixed(1)}%, rgba(0,0,0,0) ${Math.min(100, sweepX + 10).toFixed(1)}%, rgba(0,0,0,0) 100%)`;
            el.style.maskImage = maskGrad;
            el.style.webkitMaskImage = maskGrad;
          } else {
            el.style.maskImage = "none";
            el.style.webkitMaskImage = "none";
          }
        } else {
          // Future layers
          el.style.opacity = "0";
          el.style.visibility = "hidden";
          el.style.maskImage = "none";
          el.style.webkitMaskImage = "none";
        }
      }

      // Soft optical scanning beam tracking the aperture centroid
      if (cursor) {
        cursor.style.opacity = rawMix > 0.02 && rawMix < 0.98 ? "1" : "0";
        cursor.style.transform = `translate3d(${rawMix * 100}%, 0, 0)`;
      }

      if (activeStage !== previousActiveStage) {
        previousActiveStage = activeStage;
        updateStageDisplay(activeStage, progress);
      }
    };

    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
    let timeline: gsap.core.Timeline | null = null;

    const buildTimeline = () => {
      timeline?.scrollTrigger?.kill();
      timeline?.kill();
      timeline = null;

      if (window.innerWidth < 768) {
        // Decoupled mobile: do not hijack scroll
        updateStageDisplay(0, 0);
        return;
      }

      update(0);
      if (reducedMotion.matches) return;

      const playhead = { value: 0 };
      timeline = gsap.timeline({
        scrollTrigger: {
          trigger: root,
          start: "top top+=84",
          end: "bottom bottom",
          scrub: 0.35,
          invalidateOnRefresh: true,
        },
      });
      timeline.fromTo(playhead, { value: 0 }, {
        value: 1,
        duration: 1,
        ease: "none",
        onUpdate: () => update(playhead.value),
      });
    };

    const onResize = () => {
      const mobile = mobileCheck();
      if (mobile) {
        timeline?.scrollTrigger?.kill();
        timeline?.kill();
        timeline = null;
      } else if (!timeline) {
        buildTimeline();
      }
    };

    buildTimeline();
    window.addEventListener("resize", onResize);
    reducedMotion.addEventListener("change", buildTimeline);

    const refresh = window.requestAnimationFrame(() => ScrollTrigger.refresh());
    return () => {
      window.cancelAnimationFrame(refresh);
      window.removeEventListener("resize", onResize);
      reducedMotion.removeEventListener("change", buildTimeline);
      timeline?.scrollTrigger?.kill();
      timeline?.kill();
    };
  }, [updateStageDisplay]);

  return <div className="engine-journey" ref={journey}>
    <div className="engine-composition">
      <figure className="engine-plate">
        <div className="plate-heading">
          <span>IMPLEMENTED SYSTEM / DATA PATH</span>
          <span>FIG. 02</span>
        </div>
        <div className="engine-drawing architecture-flow">
          <div className="pipeline-visual" id="pipeline-visual">
            {stages.map((stage, index) => (
              <img
                key={stage.file}
                ref={(el) => { layerRefs.current[index] = el; }}
                src={`${assetRoot}/${stage.file}`}
                data-stage={index}
                alt=""
                className="pipeline-image-layer"
                style={{
                  zIndex: index + 1,
                  opacity: index === 0 ? 1 : 0,
                  visibility: index <= 1 ? "visible" : "hidden",
                  transform: "translate3d(0, 0, 0)",
                }}
                fetchPriority={index === 0 ? "high" : "auto"}
                loading={index <= 2 ? "eager" : "lazy"}
                decoding="async"
              />
            ))}
            {/* Hairline optical scan cursor */}
            <div className="pipeline-scan-cursor" ref={scanCursor} aria-hidden="true" />

            <div className="pipeline-visual-heading">
              <span>VISUAL STUDY / CONCEPT IMAGES</span>
              <span>{isMobile ? "TOUCH CAROUSEL" : "SCROLL-SCRUBBED"}</span>
            </div>
            <div className="pipeline-stage-indicator" ref={stageIndicator} role="progressbar" aria-label="Pipeline stage" aria-valuemin={1} aria-valuemax={6} aria-valuenow={1}>
              <span ref={stageCount}>01 / 06</span>
              <strong ref={stageName}>RGB IMAGE</strong>
            </div>
            <div className="pipeline-image-progress" aria-hidden="true"><span ref={progressLine} /></div>
          </div>

          {/* Mobile tactile carousel controls */}
          {isMobile && (
            <div className="mobile-stage-controls" role="group" aria-label="Stage carousel controls">
              <button
                type="button"
                className="mobile-step-btn"
                onClick={() => setStepMobile(mobileStage - 1)}
                disabled={mobileStage === 0}
                aria-label="Previous pipeline stage"
              >
                ← PREV
              </button>
              <div className="mobile-step-dots">
                {stages.map((stg, i) => (
                  <button
                    key={stg.file}
                    type="button"
                    className={`mobile-dot${mobileStage === i ? " is-active" : ""}`}
                    onClick={() => setStepMobile(i)}
                    aria-label={`Jump to stage ${i + 1}: ${stg.label}`}
                  />
                ))}
              </div>
              <button
                type="button"
                className="mobile-step-btn"
                onClick={() => setStepMobile(mobileStage + 1)}
                disabled={mobileStage === stages.length - 1}
                aria-label="Next pipeline stage"
              >
                NEXT →
              </button>
            </div>
          )}

          <ol className="pipeline-boxes" aria-label="Implemented pipeline representation">
            <li className="is-active"><span className="architecture-index">01</span><strong>RGB IMAGE</strong><small>single-view input</small></li>
            <li><span className="architecture-index">02</span><strong>MODEL A3</strong><small>12-class spatial detector</small></li>
            <li><span className="architecture-index">03</span><strong>METRIC DEPTH</strong><small>Depth Anything V2</small></li>
            <li><span className="architecture-index">04</span><strong>UNPROJECT</strong><small>camera-space volumes</small></li>
            <li><span className="architecture-index">05</span><strong>EXPORT</strong><small>glTF 2.0 / PLY</small></li>
          </ol>
          <div className="architecture-foot"><span>RGB → DETECTION → METRIC DEPTH → 3D SPATIAL VOLUMES</span><span>ROOMAI PIPELINE</span></div>
        </div>
        <figcaption className="plate-heading plate-bottom"><span>ONE DATA PATH · SIX COMPUTATIONAL STAGES</span><span>PRODUCTION ENGINE</span></figcaption>
      </figure>
      <aside className="engine-process" aria-label="Computational sequence">
        <TechnicalLabel>COMPUTATIONAL SEQUENCE / 06 STAGES</TechnicalLabel>
        <div className="pipeline-list-wrap">
          <span className="pipeline-progress-track" aria-hidden="true"><i ref={listProgress} /></span>
          <Pipeline />
        </div>
      </aside>
    </div>
  </div>;
}
