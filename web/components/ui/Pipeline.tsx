import { pipeline } from "@/data/siteContent";

export function Pipeline() {
  return <ol className="pipeline" aria-label="Reconstruction pipeline">
    {pipeline.map((step, index) => <li key={step}>
      <span className="pipeline-index">0{index + 1}</span>
      <span className="pipeline-name">{step}</span>
      {index < pipeline.length - 1 ? <span className="pipeline-arrow" aria-hidden="true">↓</span> : null}
    </li>)}
  </ol>;
}
