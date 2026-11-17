"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { LiveInferenceExhibit } from "@/components/sections/LiveInferenceExhibit";

if (typeof window !== "undefined") {
  gsap.registerPlugin(ScrollTrigger);
}

const stages = [
  {
    step: "01",
    kicker: "01 / IMAGE",
    title: "THE ROOM",
    short: "IMAGE",
    label: "01 / IMAGE — THE ROOM",
    image: "01-rgb.png",
    description: "Single-view photograph of the indoor scene.",
  },
  {
    step: "02",
    kicker: "02 / PERCEPTION",
    title: "THE IMAGE IS ONLY THE BEGINNING.",
    short: "PERCEPTION",
    label: "02 / PERCEPTION — THE IMAGE IS ONLY THE BEGINNING.",
    image: "02-scene-priors.png",
    description: "Production YOLO11s trained on 12 indoor categories with calibrated bounding boxes.",
  },
  {
    step: "03",
    kicker: "03 / DEPTH",
    title: "DEPTH EMERGES.",
    short: "DEPTH",
    label: "03 / DEPTH — DEPTH EMERGES.",
    image: "03-raw-depth.png",
    description: "Depth Anything V2 metric depth estimation for indoor spatial understanding.",
  },
  {
    step: "04",
    kicker: "04 / UNPROJECTION",
    title: "THE IMAGE BECOMES SPACE.",
    short: "SPACE",
    label: "04 / UNPROJECTION — THE IMAGE BECOMES SPACE.",
    image: "04-depth-unprojection.png",
    description: "Depth-associated bounding boxes become camera-space 3D coordinates via 34° pinhole geometry.",
  },
  {
    step: "05",
    kicker: "05 / POINT CLOUD",
    title: "GEOMETRY EMERGES.",
    short: "POINT CLOUD",
    label: "05 / POINT CLOUD — GEOMETRY EMERGES.",
    image: "05-point-cloud.png",
    description: "Dense spatial representation derived from unprojected 3D coordinates.",
  },
  {
    step: "06",
    kicker: "06 / SPATIAL MODEL",
    title: "THE ROOM BECOMES A WORLD.",
    short: "WORLD",
    label: "06 / SPATIAL MODEL — THE ROOM BECOMES A WORLD.",
    image: "06-ply-glb-export.png",
    description: "Validated glTF 2.0 and ASCII PLY exports for downstream spatial inspection.",
  },
] as const;

const assetRoot = "/img2";
type ViewMode = "PIPELINE" | "DETECT";

export function ResultViewer() {
  const track = useRef<HTMLDivElement>(null);
  const layerRefs = useRef<(HTMLImageElement | null)[]>([]);
  const progressLine = useRef<HTMLSpanElement>(null);
  const readoutCount = useRef<HTMLSpanElement>(null);
  const readoutName = useRef<HTMLElement>(null);
  const kickerRef = useRef<HTMLSpanElement>(null);
  const headlineRef = useRef<HTMLHeadingElement>(null);
  const descRef = useRef<HTMLParagraphElement>(null);
  const scanCursor = useRef<HTMLDivElement>(null);

  const timelineRef = useRef<gsap.core.Timeline | null>(null);
  const animateProgressRef = useRef<(progress: number) => void>(() => undefined);
  const viewModeRef = useRef<ViewMode>("PIPELINE");

  const [activeStage, setActiveStage] = useState(0);
  const [viewMode, setViewMode] = useState<ViewMode>("PIPELINE");
  const [assetFailed, setAssetFailed] = useState(false);

  // Preload and decode all 6 result-viewer images upfront from /img2
  useEffect(() => {
    stages.forEach((stage) => {
      const img = new Image();
      img.src = `${assetRoot}/${stage.image}`;
      void img.decode().catch(() => undefined);
    });
  }, []);

  const changeMode = useCallback((mode: ViewMode) => {
    viewModeRef.current = mode;
    setViewMode(mode);
    if (mode === "PIPELINE") {
      window.requestAnimationFrame(() => {
        ScrollTrigger.refresh();
      });
    }
  }, []);

  // Deep-link auto-activation for #detect, #model, #model-a3, or ?mode=detect
  useEffect(() => {
    const handleHash = () => {
      if (typeof window === "undefined") return;
      const hash = window.location.hash.toLowerCase();
      const search = window.location.search.toLowerCase();
      if (
        hash === "#detect" ||
        hash === "#model" ||
        hash === "#model-a3" ||
        search.includes("mode=detect")
      ) {
        changeMode("DETECT");
        const target = document.getElementById("detect") || document.getElementById("result-viewer");
        if (target) {
          setTimeout(() => {
            target.scrollIntoView({ behavior: "smooth", block: "start" });
          }, 150);
        }
      } else if (hash === "#explore" || hash === "#result-viewer") {
        changeMode("PIPELINE");
      }
    };

    handleHash();
    window.addEventListener("hashchange", handleHash);
    return () => window.removeEventListener("hashchange", handleHash);
  }, [changeMode]);

  // Main scroll progress & visual stage interpolation
  useEffect(() => {
    const element = track.current;
    const progressElement = progressLine.current;
    const countElement = readoutCount.current;
    const nameElement = readoutName.current;
    const kickerElement = kickerRef.current;
    const headlineElement = headlineRef.current;
    const descElement = descRef.current;
    const cursor = scanCursor.current;

    if (!element) return;

    const motionQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
    let selectedStage = -1;

    const update = (value: number) => {
      const progress = gsap.utils.clamp(0, 1, value);
      const scaled = progress * (stages.length - 1);
      const segment = Math.min(stages.length - 2, Math.floor(scaled));
      const rawMix = progress >= 1 ? 1 : scaled - segment;

      // Equal-energy optical crossfade curve (cos^2 + sin^2 = 1)
      const outgoingOpacity = motionQuery.matches
        ? (rawMix >= 0.5 ? 0 : 1)
        : Math.cos(rawMix * (Math.PI / 2));
      const incomingOpacity = motionQuery.matches
        ? (rawMix >= 0.5 ? 1 : 0)
        : Math.sin(rawMix * (Math.PI / 2));
      const stage = Math.min(stages.length - 1, segment + (rawMix >= 0.5 ? 1 : 0));
      const restrained = motionQuery.matches;

      // Update image layers with continuous smooth transforms
      for (let i = 0; i < stages.length; i++) {
        const el = layerRefs.current[i];
        if (!el) continue;

        if (i < segment) {
          el.style.opacity = "0";
          el.style.visibility = "hidden";
          el.style.maskImage = "none";
          el.style.webkitMaskImage = "none";
        } else if (i === segment) {
          el.style.visibility = "visible";
          el.style.opacity = outgoingOpacity.toFixed(4);
          el.style.transform = restrained
            ? "none"
            : `translate3d(0, ${(-3 * rawMix).toFixed(2)}px, 0) scale(${(1.0 + 0.006 * rawMix).toFixed(4)})`;
          el.style.filter = restrained
            ? "none"
            : `brightness(${(1.0 - 0.02 * rawMix).toFixed(3)})`;
          el.style.maskImage = "none";
          el.style.webkitMaskImage = "none";
        } else if (i === segment + 1) {
          el.style.visibility = "visible";
          el.style.opacity = incomingOpacity.toFixed(4);
          el.style.transform = restrained
            ? "none"
            : `translate3d(0, ${(3 * (1 - rawMix)).toFixed(2)}px, 0) scale(${(0.994 + 0.006 * rawMix).toFixed(4)})`;
          el.style.filter = restrained
            ? "none"
            : `brightness(${(0.98 + 0.02 * rawMix).toFixed(3)})`;

          if (!restrained && rawMix > 0.02 && rawMix < 0.98) {
            const sweepX = rawMix * 100;
            const maskGrad = `linear-gradient(90deg, #000 0%, #000 ${Math.max(0, sweepX - 10).toFixed(1)}%, rgba(0,0,0,0) ${Math.min(100, sweepX + 10).toFixed(1)}%, rgba(0,0,0,0) 100%)`;
            el.style.maskImage = maskGrad;
            el.style.webkitMaskImage = maskGrad;
          } else {
            el.style.maskImage = "none";
            el.style.webkitMaskImage = "none";
          }
        } else {
          el.style.opacity = "0";
          el.style.visibility = "hidden";
          el.style.maskImage = "none";
          el.style.webkitMaskImage = "none";
        }
      }

      if (progressElement) {
        progressElement.style.transform = `scaleX(${progress})`;
      }

      if (cursor) {
        cursor.style.opacity = !restrained && rawMix > 0.02 && rawMix < 0.98 ? "0.9" : "0";
        cursor.style.transform = `translate3d(${rawMix * 100}%, 0, 0)`;
      }

      // Synchronize textual narrative in lockstep with the transformation stage
      if (stage !== selectedStage) {
        selectedStage = stage;
        setActiveStage(stage);

        if (kickerElement) kickerElement.textContent = stages[stage].kicker;
        if (headlineElement) headlineElement.textContent = stages[stage].title;
        if (descElement) descElement.textContent = stages[stage].description;
        if (countElement) countElement.textContent = `${String(stage + 1).padStart(2, "0")} / 06`;
        if (nameElement) nameElement.textContent = stages[stage].short;

        const pills = element.querySelectorAll<HTMLElement>(".stage-nav-pill");
        pills.forEach((pill, idx) => {
          pill.classList.toggle("is-active", idx === stage);
          pill.classList.toggle("is-past", idx < stage);
          pill.setAttribute("aria-selected", idx === stage ? "true" : "false");
        });
      }
    };

    let timeline: gsap.core.Timeline | null = null;

    const buildTimeline = () => {
      timeline?.scrollTrigger?.kill();
      timeline?.kill();
      timeline = null;

      if (window.innerWidth < 768) {
        // Mobile clean viewport: initialize first stage without hijacking vertical scroll
        update(0);
        return;
      }

      update(0);
      if (motionQuery.matches) return;

      const playhead = { value: 0 };
      timeline = gsap.timeline({
        scrollTrigger: {
          trigger: element,
          start: "top top+=84",
          end: "bottom bottom",
          scrub: 0.25,
          invalidateOnRefresh: true,
          onUpdate: (self) => {
            if (progressElement) {
              progressElement.style.transform = `scaleX(${self.progress})`;
            }
          },
        },
      });

      timeline.fromTo(
        playhead,
        { value: 0 },
        {
          value: 1,
          ease: "none",
          duration: 1,
          onUpdate: () => {
            update(playhead.value);
          },
        }
      );

      timelineRef.current = timeline;
    };

    buildTimeline();

    animateProgressRef.current = (targetProg) => {
      if (motionQuery.matches) {
        update(targetProg);
        return;
      }
      const dummy = { value: 0 };
      gsap.to(dummy, {
        value: targetProg,
        duration: 0.5,
        ease: "power2.inOut",
        onUpdate: () => update(dummy.value),
      });
    };

    motionQuery.addEventListener("change", buildTimeline);
    window.addEventListener("resize", buildTimeline);
    const refreshFrame = window.requestAnimationFrame(() => ScrollTrigger.refresh());

    return () => {
      window.cancelAnimationFrame(refreshFrame);
      motionQuery.removeEventListener("change", buildTimeline);
      window.removeEventListener("resize", buildTimeline);
      timeline?.scrollTrigger?.kill();
      timeline?.kill();
      timelineRef.current = null;
      animateProgressRef.current = () => undefined;
    };
  }, []);

  const goToStage = (targetIndex: number) => {
    if (viewModeRef.current !== "PIPELINE") {
      changeMode("PIPELINE");
    }
    const targetProg = targetIndex / (stages.length - 1);
    if (window.innerWidth < 768) {
      animateProgressRef.current(targetProg);
      return;
    }
    const trigger = timelineRef.current?.scrollTrigger;
    if (trigger) {
      const scrollStart = trigger.start;
      const scrollEnd = trigger.end;
      const targetScrollY = scrollStart + targetProg * (scrollEnd - scrollStart);
      window.scrollTo({ top: targetScrollY, behavior: "smooth" });
    } else {
      animateProgressRef.current(targetProg);
    }
  };

  const isDetectMode = viewMode === "DETECT";

  return (
    <div className={`result-scroll-track${isDetectMode ? " is-detect-mode" : ""}`} id="result-viewer" ref={track}>
      <div className="explore-frame result-viewer-frame">
        {/* Topline Bar with Mode Switcher & Technical Indicators */}
        <div className="result-topline">
          <div className="result-topline-meta">
            <span>SECTION 03 // ROOMAI TRANSFORMATION CONTINUUM</span>
            <span className={isDetectMode ? "runtime-badge" : "study-badge"}>
              {isDetectMode
                ? "● MODEL A3 RUNTIME DETECTOR (GPU ACCELERATED · FASTAPI CONNECTED)"
                : "◈ VISUAL STUDY · NOT RUNTIME MODEL OUTPUT"}
            </span>
          </div>

          <div className="result-mode-switch" role="group" aria-label="Viewer representation mode">
            <button
              type="button"
              className={`mode-switch-btn ${!isDetectMode ? "is-active" : ""}`}
              onClick={() => changeMode("PIPELINE")}
            >
              ◈ SCROLL TRANSFORMATION
            </button>
            <button
              type="button"
              className={`mode-switch-btn is-model-a3 ${isDetectMode ? "is-active" : ""}`}
              onClick={() => changeMode("DETECT")}
            >
              ● MODEL A3 DETECTOR (LIVE AI)
            </button>
          </div>
        </div>

        {/* Dynamic Transformation Header Block (Synchronized with Scroll) */}
        {!isDetectMode ? (
          <div className="result-transformation-header">
            <div className="result-header-text">
              <div className="result-kicker-row">
                <span className="result-stage-kicker" ref={kickerRef}>
                  {stages[activeStage].kicker}
                </span>
              </div>
              <h3 className="result-stage-headline" ref={headlineRef}>
                {stages[activeStage].title}
              </h3>
              <p className="result-stage-desc" ref={descRef}>
                {stages[activeStage].description}
              </p>
            </div>

            <div className="result-header-controls">
              <button
                type="button"
                className="result-step-button"
                onClick={() => goToStage(Math.max(0, activeStage - 1))}
                disabled={activeStage === 0}
                aria-label="Previous transformation stage"
              >
                ← PREV
              </button>
              <button
                type="button"
                className="result-step-button"
                onClick={() => goToStage(Math.min(stages.length - 1, activeStage + 1))}
                disabled={activeStage === stages.length - 1}
                aria-label="Next transformation stage"
              >
                NEXT →
              </button>
            </div>
          </div>
        ) : null}

        {/* Visual Stage: Pinned Canvas Stack or Live Inference Detector */}
        <div className={`result-stage${isDetectMode ? " is-live-detect" : ""}`}>
          {isDetectMode ? (
            <div className="detect-mode-wrapper">
              <div className="detect-mode-banner">
                <span>● MODEL A3 ACTIVE RUNTIME // REAL GPU INFERENCE (YOLO11s @ 800px)</span>
                <Link href="/model" className="detect-banner-btn">
                  OPEN FULLSCREEN MODEL PLAYGROUND ↗
                </Link>
              </div>
              <LiveInferenceExhibit />
            </div>
          ) : (
            <>
              <div className="axis-labels" aria-hidden="true">
                <span>X</span>
                <span>Y</span>
                <span>Z</span>
              </div>

              <div className="result-image-stack" aria-hidden="true">
                {stages.map((stage, index) => (
                  <img
                    key={stage.image}
                    ref={(el) => {
                      layerRefs.current[index] = el;
                    }}
                    src={`${assetRoot}/${stage.image}`}
                    data-stage={index}
                    className="result-image-layer"
                    style={{
                      zIndex: index + 1,
                    }}
                    alt={stage.label}
                    fetchPriority={index <= 1 ? "high" : "auto"}
                    loading="eager"
                    decoding="async"
                    onError={(e) => {
                      const img = e.currentTarget;
                      if (!img.dataset.retried) {
                        img.dataset.retried = "true";
                        img.src = `/assets/roomai/result-viewer/${stage.image}`;
                      } else {
                        setAssetFailed(true);
                      }
                    }}
                  />
                ))}
                <div className="result-scan-cursor" ref={scanCursor} aria-hidden="true" />
              </div>

              {assetFailed && (
                <div className="result-asset-error" role="status">
                  A supplied result frame could not be loaded.
                </div>
              )}

              <span className="result-source-note">VISUAL STUDY · NOT RUNTIME MODEL OUTPUT</span>

              <div className="scene-readout" aria-hidden="true">
                <span>
                  CAMERA <b>01 / BEDROOM PERSPECTIVE</b>
                </span>
                <span>
                  FOV <b>34° OPTICAL PINHOLE</b>
                </span>
                <span>
                  DEPTH <b>METRIC DENSE RECONSTRUCTION</b>
                </span>
              </div>

              <div className="result-view-readout" aria-live="polite">
                <span ref={readoutCount}>{`${String(activeStage + 1).padStart(2, "0")} / 06`}</span>
                <strong ref={readoutName}>{stages[activeStage].short}</strong>
              </div>

              <div className="result-image-progress" aria-hidden="true">
                <span ref={progressLine} />
              </div>
            </>
          )}
        </div>

        {/* Minimal Technical Progress Navigation (01 through 06) */}
        {!isDetectMode && (
          <div className="result-footer-nav" role="tablist" aria-label="Transformation sequence">
            {stages.map((stage, idx) => (
              <button
                key={stage.image}
                type="button"
                role="tab"
                aria-selected={activeStage === idx}
                className={`stage-nav-pill ${activeStage === idx ? "is-active" : ""} ${idx < activeStage ? "is-past" : ""}`}
                onClick={() => goToStage(idx)}
                aria-label={`Jump to stage ${idx + 1}: ${stage.short}`}
              >
                <span className="pill-step">{stage.step}</span>
                <strong className="pill-name">{stage.short}</strong>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
