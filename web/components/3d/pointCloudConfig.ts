export const pointCloudConfig = {
  // Provenance: upl_cee8234c7f93 → worldmirror1_baseline →
  // job_1794499397_4aa219ea → scene_reconstruction.ply.
  assetUrl: "/assets/roomai/exhibit-01/point-cloud/scene_reconstruction.ply",
  metadataUrl: "/assets/roomai/exhibit-01/point-cloud/scene_reconstruction.metadata.json",
  vertexCount: 61,
  // Points are already in RoomAI / Three-compatible coordinates.
  pointSize: 5,
  opacity: 0.92,
  // Audit Phase 4A.1: the same-job PLY is valid but has only 61 collapsed
  // vertices, so it must not be exhibited as a room-scale point field.
  availableForExhibit: false,
} as const;
