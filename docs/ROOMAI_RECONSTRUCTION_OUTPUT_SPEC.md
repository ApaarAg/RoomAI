# ROOMAI Reconstruction Output Specification

**Status:** Proposed Phase 4.1 contract; not implemented.  
**Purpose:** Make every reconstruction reproducible, geometrically interpretable, and safe to evaluate without claiming more than the model supports.

## Contract principles

- Keep the uploaded original immutable. Store derived, model-ready RGB separately and record all transforms between them.
- Preserve numeric depth, masks, camera metadata, and coordinate conventions as data. Heatmaps are visualizations only.
- Distinguish `metric_calibrated`, `metric_model_estimate`, `relative_inverse`, `scene_normalized`, and `unknown` depth. A model label or plausible value range does not prove physical meters.
- A single RGB view supports visible-side 2.5D geometry only. Do not infer hidden surfaces, complete meshes, or object shape without real evidence.
- Treat output as accepted only when input rights, model/checkpoint provenance, data checks, geometry checks, and export checks all pass.

## Canonical exhibit bundle

```text
exhibit-{id}/
├── source/
│   ├── rgb_original.<original-extension>      # exact uploaded bytes, immutable
│   └── rgb_decoded.png                        # orientation/color-normalized pixels
├── depth/
│   ├── depth_raw.npy                          # H×W little-endian float32, no display normalization
│   ├── valid_mask.png                          # H×W 0/255 binary mask
│   └── confidence.npy                          # optional H×W float32 model confidence
├── points/
│   └── point_cloud.ply                        # canonical full accepted sample, RGB uchar
├── geometry/
│   └── reconstruction.glb                     # optional; points and/or validated surfaces
├── diagnostics/
│   ├── depth_preview.png                       # derived visualization, never numeric source
│   ├── rgb_depth_overlay.png
│   └── pointcloud_views.png
└── manifest.json
```

Private intake/originals must remain outside public static serving. Runtime tiers are derived later from an accepted master; this Phase 4.1 contract does not authorize frontend work.

## Required fields

### Source RGB

| Field | Requirement |
|---|---|
| `source.image_id`, `source.sha256` | Stable ID and SHA-256 of original bytes |
| `source.path`, `source.mime_type`, `source.width_px`, `source.height_px` | Exact source identity and dimensions |
| `source.rights.status`, `source.rights.owner`, `source.rights.license`, `source.rights.evidence` | Explicit rights decision and evidence; `unknown` blocks public use |
| `decoded_rgb.sha256`, `decoded_rgb.color_space`, `decoded_rgb.orientation_applied` | Pixel identity after decode/orientation handling |
| `preprocessing.operations[]` | Ordered crop/resize/pad/color transforms, source and destination sizes, interpolation, and affine pixel mapping |

### Depth and camera

| Field | Requirement |
|---|---|
| `depth.path`, `depth.sha256`, `depth.shape`, `depth.dtype` | Full-resolution raw array and integrity |
| `depth.representation` | One of the explicit semantics below |
| `depth.unit`, `depth.scale_to_meters` | `m` and 1.0 only for metric outputs; otherwise null/unknown and no metric claim |
| `depth.invalid_value`, `depth.valid_mask_path`, `depth.valid_pixel_ratio` | Explicit invalid encoding and separate boolean validity mask |
| `depth.alignment_reference` | RGB image hash/decoded image and pixel transform to which depth is registered |
| `camera.model` | `pinhole` initially; document distortion if any |
| `camera.intrinsics.K` | 3×3 matrix in pixel coordinates for the exact decoded/depth resolution; record `fx,fy,cx,cy` too |
| `camera.intrinsics.source`, `camera.intrinsics.status` | `capture_metadata`, `dataset_calibration`, or `estimated`; status cannot imply calibration when estimated |
| `camera.coordinate_frame` | `opencv_camera`: +X right, +Y down, +Z forward |
| `camera.extrinsics` | Optional camera-to-world transform with named source/frame; absent means points remain camera-local |

`depth.representation` values:

- `metric_calibrated`: metric scale verified against independent sensor/known-distance references for this device/domain.
- `metric_model_estimate`: model trained for metric output, stored in meters, but still an estimate; publish measured error and uncertainty.
- `relative_inverse`: relative ordering only; higher/lower direction declared; unit and metric scale are null.
- `scene_normalized`: dimensionless canonical/model scale; scale calibration absent.
- `unknown`: no geometry acceptance.

Raw arrays must not be percentile-normalized. A display map is derived into `diagnostics/` and must include a manifest relation to the raw depth hash.

### Point cloud

- `points.path`, `points.sha256`, `points.count`, `points.color_encoding`, `points.sample_method`, `points.source_pixel_mapping` are required.
- Canonical point units equal the declared depth unit; coordinate frame is explicit.
- RGB colors are sampled from the aligned decoded RGB at the source pixel that generated each point. Values in PLY are uchar 0–255; API arrays may use float [0,1].
- Include only finite XYZ values marked valid by the depth/confidence mask. Keep sample selection deterministic and record seed/voxel settings.
- For depth-unprojected camera points, verify `Z == depth[u,v]` and reproject with K to the generating pixel within the documented rounding tolerance.
- Include no fabricated point noise, smoothed geometry, fitted hidden walls, or invented furniture surfaces.

### Geometry and GLB

- `geometry.kind`: `point_cloud`, `depth_surface_partial`, or `mesh_validated`. Do not call a point cloud a mesh.
- Optional object boxes are labelled `semantic_bounds` with detector provenance and are not part of room surface geometry.
- GLB points use primitive mode `POINTS`, float32 `POSITION`, and normalized or uchar normalized `COLOR_0` when color is available. If the viewer path cannot support point colors, that limitation must be tested and the PLY remains the canonical colored source.
- Any triangle surface must identify the source depth samples, edge/discontinuity rejection rule, connected components, and surface validation result. Do not close unseen surfaces.
- GLB positions must be meters. Convert OpenCV camera basis to glTF’s right-handed Y-up camera basis with a documented transform (for this basis, `diag(1,-1,-1)`), applied once. Record the exact source-to-export 4×4 matrix in `asset.extras` or the sidecar manifest. PLY has no portable units/frame metadata; manifest is authoritative.
- Store metadata in glTF `asset.extras`/root-node `extras`: schema version, manifest ID, units, coordinate frame, source hash, model ID, and transform. Do not rely on a viewer to infer units from raw vertex bounds.
- Any reduced GLB tier must use deterministic indices/voxelization and record the canonical PLY hash and source index mapping. PLY and GLB need not have the same count only when that intentional reduction is declared.

## Model, run, and manifest provenance

`manifest.json` must include:

```json
{
  "schema_version": "roomai.reconstruction/1.0",
  "exhibit_id": "<stable-id>",
  "status": "candidate|accepted|rejected",
  "source": {"sha256": "...", "rights": {"status": "..."}},
  "model": {
    "adapter_id": "roomai_composite_v1",
    "model_name": "...",
    "checkpoint_sha256": "...",
    "config_sha256": "...",
    "code_revision": "...",
    "software_versions": {"python": "...", "torch": "...", "transformers": "..."}
  },
  "preprocessing": {"input_shape": [0, 0], "operations": []},
  "depth": {"representation": "metric_model_estimate", "unit": "m", "scale_to_meters": 1.0},
  "camera": {"coordinate_frame": "opencv_camera", "intrinsics": {"units": "pixels", "K": []}},
  "outputs": {"depth": {}, "valid_mask": {}, "point_cloud": {}, "glb": {}},
  "validation": {"status": "pass|fail", "checks": [], "metrics": {}},
  "limitations": ["single-view; occluded surfaces unavailable"]
}
```

Use null for unavailable values and an explicit reason. Never silently fill missing calibration, license, or checkpoint fields with defaults and then mark the bundle accepted.

## Storage/format requirements

- Preserve original file bytes. Keep working decoded RGB lossless (PNG) for reproducibility.
- Store depth as float32 `.npy` or lossless `.npz`; mask as lossless single-channel PNG. Do not store false-color images as depth.
- PLY is a canonical interchange export with XYZ float and RGB uchar. GLB is an optimized runtime/export view, not the sole evidence source.
- Hash every source and output. Do not rewrite originals when optimizing; make a new derived artifact and record the transformation.
