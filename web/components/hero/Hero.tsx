"use client";

import { useNarrative } from "@/components/3d/NarrativeController";
import { NarrativeFrameSequence } from "./NarrativeFrameSequence";

export function Hero() {
  const { masterProgress } = useNarrative();

  return <section className="hero narrative-hero" id="hero-experience" aria-labelledby="hero-title">
    <div className="hero-sticky">
      <NarrativeFrameSequence progress={masterProgress} />
    </div>
  </section>;
}
