import Link from "next/link";
import { Container } from "@/components/layout/Container";
import { projectLinks } from "@/data/siteContent";

export function ProjectSection() {
  return <>
    <section className="section project-section" id="project" aria-labelledby="project-title">
      <Container>
        <p className="project-kicker">ROOMAI® / PROJECT</p>
        <div className="project-layout">
          <div><h2 id="project-title">ROOMAI</h2><p>An AI-powered indoor spatial reconstruction system.</p></div>
          <div className="project-links" aria-label="Project references">
            {projectLinks.map((link) => (
              <a
                key={link.label}
                href={link.href}
                target={link.href.startsWith("http") ? "_blank" : undefined}
                rel={link.href.startsWith("http") ? "noopener noreferrer" : undefined}
              >
                <span>{link.label}</span>
                <small>{link.note}</small>
              </a>
            ))}
            <div className="project-link-unavailable"><span>MODEL A3</span><small>YOLO11s @ 800px</small></div>
          </div>
        </div>
        <div className="project-bottom"><span>ROOMAI PRODUCTION INFERENCE ENGINE</span><a href="#top">BACK TO TOP ↑</a></div>
      </Container>
    </section>
    <section className="final-statement" aria-label="ROOMAI conclusion">
      <Container>
        <p>I DIDN’T JUST SEE<br />A STILL IMAGE.</p>
        <p className="final-emphasis">I FOLLOWED IT<br /><em>INTO SPACE.</em></p>
        <Link className="explore-project" href="/model">LAUNCH TRAINED MODEL PLAYGROUND <span>→</span></Link>
      </Container>
    </section>
  </>;
}
