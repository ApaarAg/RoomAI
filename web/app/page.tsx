import Link from "next/link";
import type { Metadata } from "next";
import { SplineScene } from "@/components/ui/splite";
import { Spotlight } from "@/components/ui/spotlight";
import { Card } from "@/components/ui/card";
import { LiveStatusBadge } from "@/components/navigation/LiveStatusBadge";

export const metadata: Metadata = {
  title: "ROOMAI — Spatial Intelligence & 3D Scene Reconstruction",
  description:
    "Understand spaces. See the room differently. Single-view spatial reconstruction, metric camera-ray unprojection, and interactive 3D scene intelligence.",
};

export default function RoomAILandingPage() {
  return (
    <div className="first-page-root">
      {/* 1. Header Navigation */}
      <header className="first-page-nav">
        <Link href="/" className="wordmark" aria-label="ROOMAI — Home">
          ROOMAI<span>®</span>
        </Link>
        <nav className="nav-links" aria-label="Main Navigation">
          <Link href="/model" className="nav-link nav-highlight">
            <span>●</span> USE TRAINED MODEL
          </Link>
          <Link href="/explore#construction" className="nav-link">
            <span>03</span> 3D ROOM
          </Link>
          <Link href="/explore#detect" className="nav-link">
            <span>04</span> LIVE DETECTOR
          </Link>
          <Link href="/explore#engine" className="nav-link">
            <span>02</span> PIPELINE
          </Link>
          <Link href="/explore#evidence" className="nav-link">
            <span>05</span> AUDIT &amp; RESEARCH
          </Link>
        </nav>
        <div className="nav-status-group">
          <LiveStatusBadge showDetails={true} />
        </div>
      </header>

      {/* 2. Interactive 3D Hero Section */}
      <main className="spline-hero-section">
        <div className="spline-hero-grid">
          {/* Left Column: Technical Copy & Intentional CTAs */}
          <div className="spline-hero-copy">
            <span className="eyebrow-tag">ROOM INTELLIGENCE // SPATIAL RESEARCH</span>

            <h1>
              Understand spaces.
              <br />
              See the room <em>differently</em>.
            </h1>

            <p className="spline-hero-description">
              Single-view indoor spatial perception and metric camera-ray unprojection.
              Transforming standard indoor photographs into coherent 3D volumes,
              structured scene graphs, and dense spatial representations.
            </p>

            <div className="spline-cta-group">
              <Link href="/model" className="cta-button-primary">
                TRY TRAINED MODEL (A3 GPU) <span>→</span>
              </Link>
              <Link href="/explore#detect" className="cta-button-secondary">
                View Interactive Detector Demo
              </Link>
            </div>

            <div className="hero-meta-readout">
              <span>LATENCY: &lt;35MS</span>
              <span>FOV: 34° OPTICAL</span>
              <span>DEPTH: PROVISIONAL Z=3.0M</span>
              <span>MODEL: A3 INDOOR</span>
            </div>
          </div>

          {/* Right Column: Interactive Spline 3D Scene in Engineered Plate */}
          <div className="spline-visual-column">
            <Card className="spline-card-container">
              <Spotlight fill="rgba(180, 119, 98, 0.26)" size={520} />

              <div className="spline-plate-header">
                <span>PLATE 00 // INTERACTIVE 3D ENVIRONMENT</span>
                <span className="live-badge">LIVE SCENE</span>
              </div>

              <div className="spline-stage-wrapper">
                <SplineScene
                  scene="https://prod.spline.design/kZDDjO5HuC9GJUM2/scene.splinecode"
                  className="w-full h-full"
                />
              </div>

              <div className="spline-plate-footer">
                <span>LATENT ROOM GEOMETRY</span>
                <span>DRAG TO ORBIT // INTERACTIVE CANVAS</span>
              </div>
            </Card>
          </div>
        </div>
      </main>

      {/* 3. Technical Capabilities Strip */}
      <section className="first-page-strip" aria-label="System Capabilities">
        <div className="first-page-strip-inner">
          <div className="capability-card">
            <span className="capability-index">01 // CAMERA RAY UNPROJECTION</span>
            <h2 className="capability-title">Optical Perspective Projection</h2>
            <p className="capability-body">
              Unprojects 2D bounding boxes along optical camera rays with pinhole
              perspective modeling ($FOV=34^\circ$) onto a nominal scene depth plane.
            </p>
          </div>

          <div className="capability-card">
            <span className="capability-index">02 // LOW-LATENCY DETECTION</span>
            <h2 className="capability-title">Model A3 Perception</h2>
            <p className="capability-body">
              Production YOLO11s architecture evaluating 800px indoor scene captures
              at 23–35ms p95 latency on NVIDIA RTX 4060 GPU with calibrated confidences.
            </p>
            <Link href="/model" className="capability-link">
              Launch Model A3 Playground <span>→</span>
            </Link>
          </div>

          <div className="capability-card">
            <span className="capability-index">03 // DENSE 3D SYNTHESIS</span>
            <h2 className="capability-title">Spatial Point Cloud Export</h2>
            <p className="capability-body">
              Depth Anything V2 metric depth estimation producing validated
              PLY point fields and glTF 2.0 digital asset representations.
            </p>
            <Link href="/explore#construction" className="capability-link">
              Explore 3D Room Construction <span>→</span>
            </Link>
          </div>
        </div>
      </section>

      {/* 4. Minimalist Technical Footer */}
      <footer className="first-page-footer">
        <div>ROOMAI // SPATIAL UNDERSTANDING &amp; 3D SCENE RECONSTRUCTION</div>
        <div>
          <Link href="/model">LAUNCH TRAINED MODEL A3 PLAYGROUND →</Link>
        </div>
      </footer>
    </div>
  );
}
