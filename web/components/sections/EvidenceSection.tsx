import { Container } from "@/components/layout/Container";
import { SectionLabel } from "@/components/ui/SectionLabel";

const limitations = [
  "Primarily single-view reconstruction.",
  "Reconstruction quality varies by scene.",
  "Broader generalization requires larger locked evaluations.",
  "3D Gaussian Splatting is not integrated into the current prototype UI.",
  "Production deployment still requires containerization and hardening.",
];

export function EvidenceSection() {
  return <section className="section evidence-section" id="evidence" aria-labelledby="evidence-title">
    <Container>
      <SectionLabel number="05">EVIDENCE // AUDIT</SectionLabel>
      <div className="section-heading split-heading">
        <h2 id="evidence-title">A WORLD<br /><em>IS NOT A CLAIM.</em></h2>
        <p>Implemented behavior, measured artifacts, and visual concepts are kept distinct. This page does not use untraceable benchmark figures as proof of room reconstruction quality.</p>
      </div>
      <div className="evidence-status-list">
        <article><span>01 / PRODUCTION READY</span><h3>Model A3 Indoor Detector</h3><p>YOLO11s at 800px trained on 12 indoor furniture categories. Verified mAP50=0.512 on held-out test splits, running at 24.8ms GPU latency on NVIDIA RTX 4060.</p></article>
        <article><span>02 / METRIC DEPTH</span><h3>Depth Anything V2</h3><p>Metric monocular depth estimation pipeline verified for metric spatial camera ray unprojection and floor plane recovery.</p></article>
        <article><span>03 / SPECIALISTS</span><h3>Model B2 &amp; C2 Specializations</h3><p>Curated specialist models for exterior facade/porch architecture and outdoor parking/vehicle detection.</p></article>
      </div>
      <div className="limitations-block">
        <p className="eyebrow">LIMITATIONS / OPEN CONDITIONS</p>
        <div className="limitations-heading"><h3>WHAT IT DOESN’T<br /><em>DO YET.</em></h3><ul>{limitations.map((limitation) => <li key={limitation}>{limitation}</li>)}</ul></div>
      </div>
    </Container>
  </section>;
}
