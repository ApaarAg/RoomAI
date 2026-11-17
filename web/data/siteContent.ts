export const sections = [
  { id: "hero-experience", number: "01", label: "OVERVIEW" },
  { id: "engine", number: "02", label: "PIPELINE" },
  { id: "construction", number: "03", label: "3D ROOM" },
  { id: "explore", number: "04", label: "LIVE DETECTOR" },
  { id: "evidence", number: "05", label: "EVIDENCE" },
  { id: "future", number: "06", label: "RESEARCH" },
  { id: "project", number: "07", label: "PROJECT" },
] as const;

export const pipeline = [
  "RGB ROOM IMAGE",
  "MODEL A3 SPATIAL DETECTOR",
  "METRIC DEPTH ESTIMATION",
  "CAMERA RAY UNPROJECTION",
  "3D SPATIAL SCENE GRAPH",
  "GLTF 2.0 / PLY EXPORT",
] as const;

export const futureCapabilities = [
  "MULTI-VIEW RECONSTRUCTION",
  "SCENE UNDERSTANDING",
  "MESH RECONSTRUCTION",
  "CAD / SPATIAL DESIGN",
  "SPATIAL REASONING",
] as const;

export const projectLinks = [
  { label: "LIVE MODEL PLAYGROUND", href: "/model", note: "MODEL A3 RUNTIME ↗" },
  { label: "3D ROOM CONSTRUCTION", href: "/explore#construction", note: "SPATIAL 3D ↗" },
  { label: "FASTAPI BACKEND DOCS", href: "http://localhost:8000/docs", note: "OPENAPI SPECS ↗" },
  { label: "BENCHMARKS & AUDIT", href: "#evidence", note: "EVALUATION ↗" },
] as const;
