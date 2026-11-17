"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { Container } from "@/components/layout/Container";
import { SectionLabel } from "@/components/ui/SectionLabel";
import { RoomConstructionStage, CONSTRUCTION_STAGES } from "./RoomConstructionStage";

export function RoomConstructionSection() {
  const trackRef = useRef<HTMLDivElement>(null);
  const [activeStage, setActiveStage] = useState(0);
  const [overallProgress, setOverallProgress] = useState(0);
  const isProgrammaticScroll = useRef(false);
  const unlockTimer = useRef<NodeJS.Timeout | null>(null);

  // Handle scroll progress within the track
  useEffect(() => {
    const track = trackRef.current;
    if (!track) return;

    const motionQuery = window.matchMedia("(prefers-reduced-motion: reduce)");

    const handleScroll = () => {
      if (window.innerWidth < 768 || motionQuery.matches) {
        return; // Mobile and reduced motion use manual pill navigation
      }

      if (isProgrammaticScroll.current) return;

      const rect = track.getBoundingClientRect();
      const windowHeight = window.innerHeight;
      const trackHeight = rect.height;

      // Track starts pinning when its top reaches near the navbar (80px)
      const topOffset = 80;
      const scrollDist = -rect.top + topOffset;
      const maxScroll = trackHeight - windowHeight;

      if (maxScroll <= 0) return;

      if (scrollDist < 0) {
        setOverallProgress(0);
        setActiveStage(0);
        return;
      }
      if (scrollDist > maxScroll) {
        setOverallProgress(1);
        setActiveStage(CONSTRUCTION_STAGES.length - 1);
        return;
      }

      const progress = Math.max(0, Math.min(1, scrollDist / maxScroll));
      setOverallProgress(progress);

      // Map progress to 6 stages [0, 5]
      const totalStages = CONSTRUCTION_STAGES.length;
      const calculatedStage = Math.min(
        totalStages - 1,
        Math.floor(progress * totalStages)
      );

      setActiveStage(calculatedStage);
    };

    window.addEventListener("scroll", handleScroll, { passive: true });
    handleScroll();

    return () => {
      window.removeEventListener("scroll", handleScroll);
      if (unlockTimer.current) clearTimeout(unlockTimer.current);
    };
  }, []);

  // Jump to specific stage
  const jumpToStage = useCallback((stageIndex: number) => {
    setActiveStage(stageIndex);
    const progress = stageIndex / (CONSTRUCTION_STAGES.length - 1);
    setOverallProgress(progress);

    const track = trackRef.current;
    if (!track) return;

    isProgrammaticScroll.current = true;
    if (unlockTimer.current) clearTimeout(unlockTimer.current);
    unlockTimer.current = setTimeout(() => {
      isProgrammaticScroll.current = false;
    }, 700);

    const rect = track.getBoundingClientRect();
    const currentScrollY = window.scrollY;
    const trackTop = currentScrollY + rect.top - 80;
    const maxScroll = rect.height - window.innerHeight;

    if (maxScroll > 0) {
      const targetY = trackTop + progress * maxScroll;
      window.scrollTo({ top: targetY, behavior: "smooth" });
    }
  }, []);

  return (
    <section
      className="section room-construction-section"
      id="construction"
      aria-labelledby="construction-title"
    >
      <div id="room-3d" className="construction-scroll-anchor" />

      <Container>
        <div className="section-transition-bridge" aria-hidden="true">
          <span className="bridge-line" />
          <span className="bridge-tag">
            SINGLE-VIEW TO SPATIAL GEOMETRY // 6 DISCRETE STAGES → GLTF 2.0 ASSET
          </span>
          <span className="bridge-line" />
        </div>

        <SectionLabel number="03">3D ROOM CONSTRUCTION</SectionLabel>

        <div className="section-heading split-heading">
          <h2 id="construction-title">
            THE IMAGE<br />
            <em>BECOMES SPACE.</em>
          </h2>
          <p>
            Depth, geometry, and spatial relationships emerge from a single view. Scrub the visual
            continuum below to watch an indoor photograph transform into an interactive 3D digital
            asset with metric floor plane recovery and segmented spatial volumes.
          </p>
        </div>
      </Container>

      {/* Scroll-Driven Pinned 3D Construction Track */}
      <div className="construction-scroll-track" ref={trackRef}>
        <div className="construction-sticky-viewport">
          <Container className="construction-stage-container">
            <RoomConstructionStage
              activeStageIndex={activeStage}
              overallProgress={overallProgress}
              onJumpToStage={jumpToStage}
            />
          </Container>
        </div>
      </div>

      <Container>
        <div className="construction-technical-footnote">
          <div className="footnote-tag">TECHNICAL ATTRIBUTION // SPATIAL PIPELINE</div>
          <p>
            <strong>SCIENTIFIC DISTINCTION:</strong> Model A3 is an indoor object detector
            providing 2D bounding boxes and semantic priors. The complete 3D spatial reconstruction
            demonstrated above uses RoomAI’s composite pipeline: Model A3 + SegFormer surface
            segmentation + Depth Anything V2 Metric estimation + RANSAC planar floor recovery on
            evaluated test split <code>ai_013_010-frame_0000</code>.
          </p>
        </div>
      </Container>
    </section>
  );
}
