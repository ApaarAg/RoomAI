import { Container } from "@/components/layout/Container";
import { SectionLabel } from "@/components/ui/SectionLabel";
import { ResultViewer } from "@/components/sections/ResultViewer";

export function ExploreSection() {
  return (
    <section className="section explore-section result-explore-section" id="explore" aria-labelledby="explore-title">
      <div id="detect" className="detect-scroll-anchor" />
      <Container>
        <SectionLabel number="04">LIVE SYSTEM // MODEL A3 RUNTIME</SectionLabel>
        <div className="section-heading split-heading">
          <h2 id="explore-title">
            REAL-TIME<br />
            <em>PERCEPTION ENGINE.</em>
          </h2>
          <p>
            Connect directly to our GPU-accelerated Model A3 inference service running on NVIDIA hardware. Upload custom indoor photography or run evaluation benchmarks with calibrated bounding boxes and optical camera-ray projections.
          </p>
        </div>
        <ResultViewer />
      </Container>
    </section>
  );
}
