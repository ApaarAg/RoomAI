"use client";

import { createContext, type ReactNode, useContext, useEffect, useState } from "react";
import gsap from "gsap";
import type { Representation } from "./RepresentationController";

export type NarrativeState = {
  masterProgress: number;
  depthProgress: number;
  representation: Representation;
  representationProgress: number;
  pointsProgress: number;
  structureProgress: number;
  worldProgress: number;
};

const clamp01 = (value: number) => Math.min(1, Math.max(0, value));

// These ranges reserve the later narrative stages without representing assets
// that have not been verified for this reconstruction.
export const narrativeRanges = {
  photograph: [0, 0.18],
  depth: [0.18, 0.38],
  points: [0.38, 0.58],
  structure: [0.58, 0.76],
  world: [0.76, 1],
} as const;

export const pointCloudAvailable = false;
export const spatialStructureAvailable = false;
export const worldAvailable = false;

const stageProgress = (value: number, range: readonly [number, number]) =>
  clamp01((value - range[0]) / (range[1] - range[0]));

export function resolveNarrativeState(masterProgress: number): NarrativeState {
  const master = clamp01(masterProgress);
  const depthProgress = stageProgress(master, narrativeRanges.depth);
  const pointsProgress = stageProgress(master, narrativeRanges.points);
  const structureProgress = stageProgress(master, narrativeRanges.structure);
  const worldProgress = stageProgress(master, narrativeRanges.world);
  const representation: Representation = master < 0.18 ? "REALITY"
    : master < 0.38 ? "DEPTH"
      : master < 0.58 ? "POINTS"
        : master < 0.76 ? "GEOMETRY" : "WORLD";
  const ranges: Record<Representation, readonly [number, number]> = {
    REALITY: narrativeRanges.photograph,
    DEPTH: narrativeRanges.depth,
    POINTS: narrativeRanges.points,
    GEOMETRY: narrativeRanges.structure,
    WORLD: narrativeRanges.world,
    HYBRID: narrativeRanges.depth,
  };
  return {
    masterProgress: master,
    depthProgress,
    representation,
    representationProgress: stageProgress(master, ranges[representation]),
    pointsProgress,
    structureProgress,
    worldProgress,
  };
}

const NarrativeContext = createContext<NarrativeState>(resolveNarrativeState(0));

export function useNarrative() {
  return useContext(NarrativeContext);
}

export function NarrativeController({ children }: { children: ReactNode | ((state: NarrativeState) => ReactNode) }) {
  const [state, setState] = useState<NarrativeState>(() => resolveNarrativeState(0));

  useEffect(() => {
    let lastProgress = Number.NaN;
    const hero = document.getElementById("hero-experience");
    if (!hero) return;

    const applyProgress = (value: number) => {
      if (Math.abs(value - lastProgress) < 0.0003) return;
      lastProgress = value;
      setState(resolveNarrativeState(value));
    };

    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
    let targetProgress = 0;
    let currentProgress = 0;
    let animId = 0;
    let lastTimestamp = 0;

    const lerpProgress = (timestamp: number) => {
      if (!lastTimestamp) lastTimestamp = timestamp;
      const deltaTime = Math.min(0.064, Math.max(0.001, (timestamp - lastTimestamp) / 1000));
      lastTimestamp = timestamp;

      const diff = targetProgress - currentProgress;
      if (Math.abs(diff) < 0.0002) {
        currentProgress = targetProgress;
        applyProgress(currentProgress);
        animId = 0;
        lastTimestamp = 0;
        return;
      }
      // Frame-rate independent time-based critically damped interpolation:
      // alpha = 1 - exp(-k * deltaTime). At 60Hz (dt=0.0167s), k=10.5 yields alpha ~ 0.16.
      const alpha = 1 - Math.exp(-10.5 * deltaTime);
      currentProgress += diff * alpha;
      applyProgress(currentProgress);
      animId = window.requestAnimationFrame(lerpProgress);
    };

    const calculateProgress = () => {
      const documentTop = hero.getBoundingClientRect().top + window.scrollY;
      const scrollRange = Math.max(1, hero.offsetHeight - window.innerHeight);
      return clamp01((window.scrollY - documentTop) / scrollRange);
    };

    const updateFromScroll = () => {
      const raw = calculateProgress();
      if (reducedMotion.matches) {
        applyProgress(raw);
        return;
      }
      targetProgress = raw;
      if (!animId) {
        lastTimestamp = 0;
        animId = window.requestAnimationFrame(lerpProgress);
      }
    };

    const setup = () => {
      if (animId) {
        window.cancelAnimationFrame(animId);
        animId = 0;
      }
      lastTimestamp = 0;
      const initial = calculateProgress();
      targetProgress = initial;
      currentProgress = initial;
      applyProgress(initial);
    };

    window.addEventListener("scroll", updateFromScroll, { passive: true });
    window.addEventListener("resize", setup);
    window.addEventListener("hashchange", setup);
    window.addEventListener("popstate", setup);
    reducedMotion.addEventListener("change", setup);
    setup();

    return () => {
      window.removeEventListener("scroll", updateFromScroll);
      window.removeEventListener("resize", setup);
      window.removeEventListener("hashchange", setup);
      window.removeEventListener("popstate", setup);
      reducedMotion.removeEventListener("change", setup);
      if (animId) window.cancelAnimationFrame(animId);
    };
  }, []);

  return <NarrativeContext.Provider value={state}>
    {typeof children === "function" ? children(state) : children}
  </NarrativeContext.Provider>;
}
