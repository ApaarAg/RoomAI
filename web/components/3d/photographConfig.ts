export const photographConfig = {
  imageUrl: "/assets/roomai/exhibit-01/roomai-bedroom-editorial.jpeg",
  imageAlt: "Editorially cleaned image of the original bedroom. The depth field shown later was inferred from the original photograph, not this edited version.",
  caption: "EDITORIAL RGB · DEPTH FROM ORIGINAL PHOTO",
  width: 3.12,
  height: 2.34,
  mountBorder: 0.1,
} as const;

export const depthFieldConfig = {
  // Provenance: upl_cee8234c7f93 → worldmirror1_baseline →
  // job_1794499397_4aa219ea → depth/view_00_raw.npy.
  browserAssetUrl: "/assets/roomai/exhibit-01/depth/view_00_raw.f32",
  metadataUrl: "/assets/roomai/exhibit-01/depth/view_00_raw.metadata.json",
  dimensions: [518, 518] as const,
  observedRange: [0.8537687063217163, 1.0215181112289429] as const,
  semantics: "raw model-native depth from the original, pre-edit photograph; metric units are unknown",
  // The model sees the 960px-wide centered square of the 1280×960 source.
  sourceCropX: [0.125, 0.875] as const,
  // WorldMirror's camera unprojection defines Z = raw depth: larger values are
  // farther along the model camera's forward axis. Browser +Z faces its camera,
  // so negate the display axis to preserve that proven near/far ordering.
  renderDirection: -1,
  displacementScale: 1.4,
  segments: [128, 96] as const,
} as const;
