import { Container } from "@/components/layout/Container";
import { SectionLabel } from "@/components/ui/SectionLabel";
import { PipelineExperience } from "@/components/sections/PipelineExperience";

export function EngineSection() {
  return <section className="section engine-section" id="engine" aria-labelledby="engine-title">
    <Container>
      <SectionLabel number="02">ENGINE</SectionLabel>
      <div className="section-heading split-heading">
        <h2 id="engine-title">THE IMAGE<br /><em>IS ONLY THE BEGINNING.</em></h2>
        <p>A research pipeline connects learned scene priors to geometric representations and portable 3D formats.</p>
      </div>
      <PipelineExperience />
    </Container>
  </section>;
}
