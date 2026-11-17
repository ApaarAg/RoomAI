import { NarrativeController } from "@/components/3d/NarrativeController";
import { Navigation } from "@/components/navigation/Navigation";
import { Hero } from "@/components/hero/Hero";
import { TransformSection } from "@/components/sections/TransformSection";
import { EngineSection } from "@/components/sections/EngineSection";
import { RoomConstructionSection } from "@/components/spatial/RoomConstructionSection";
import { ExploreSection } from "@/components/sections/ExploreSection";
import { EvidenceSection } from "@/components/sections/EvidenceSection";
import { FutureSection } from "@/components/sections/FutureSection";
import { ProjectSection } from "@/components/sections/ProjectSection";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "ROOMAI — Spatial Reconstruction Experience",
  description: "Interactive architectural study of single-view indoor spatial reconstruction, depth unprojection, and point cloud synthesis.",
};

export default function ExplorePage() {
  return (
    <NarrativeController>
      <Navigation />
      <main id="top">
        <Hero />
        <TransformSection />
        <EngineSection />
        <RoomConstructionSection />
        <ExploreSection />
        <EvidenceSection />
        <FutureSection />
        <ProjectSection />
      </main>
    </NarrativeController>
  );
}
