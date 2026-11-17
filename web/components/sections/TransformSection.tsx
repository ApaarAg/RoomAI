import { Container } from "@/components/layout/Container";
import { SectionLabel } from "@/components/ui/SectionLabel";

export function TransformSection() {
  return <section className="section transform-section" id="transform" aria-labelledby="transform-title">
    <Container>
      <div className="section-transition-bridge" aria-hidden="true">
        <span className="bridge-line" />
        <span className="bridge-tag">CONTINUUM / 12 DISCRETE PHASES → COMPUTATIONAL ARCHITECTURE</span>
        <span className="bridge-line" />
      </div>
      <SectionLabel number="01">TRANSFORM</SectionLabel>
      <div className="section-heading split-heading">
        <h2 id="transform-title">ONE IMAGE.<br /><em>TWELVE STATES.</em></h2>
        <p>The pinned visual study above is scrubbed directly by scroll: analysis, depth, surface, points, structure, and exploration. Reverse the scroll to move back through the stages.</p>
      </div>
      <div className="transform-note"><span>VISUAL SEQUENCE / CONCEPTUAL</span><p>The staged images are illustrative. The separate depth exhibit below uses ROOMAI’s verified raw-depth artifact from the supplied bedroom photograph.</p><a href="#explore">VIEW THE DEPTH EXHIBIT <b>↘</b></a></div>
      <p className="section-footnote">SCROLL-SCRUBBED STORY <span>ILLUSTRATIVE ASSETS · RAW DEPTH VIEWER BELOW</span></p>
    </Container>
  </section>;
}
