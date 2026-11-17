import { Container } from "@/components/layout/Container";
import { SectionLabel } from "@/components/ui/SectionLabel";
import { futureCapabilities } from "@/data/siteContent";

export function FutureSection() {
  return <section className="section future-section" id="future" aria-labelledby="future-title">
    <Container>
      <SectionLabel number="06">RESEARCH // FUTURE</SectionLabel>
      <div className="future-layout">
        <div>
          <p className="eyebrow future-kicker">FUTURE DIRECTIONS / NOT CURRENT FEATURES</p>
          <h2 id="future-title">FROM RECONSTRUCTION<br /><em>TO SPATIAL INTELLIGENCE.</em></h2>
        </div>
        <ol className="future-list">{futureCapabilities.map((capability, index) => <li key={capability}><span>0{index + 1}</span>{capability}</li>)}</ol>
      </div>
    </Container>
  </section>;
}
