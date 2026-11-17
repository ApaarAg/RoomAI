# ROOMAI Backend Reconstruction Output Contract

**Contract version:** `1.0.0`  
**Canonical entrypoint:** `uvicorn api.main:app`  
**Canonical writer:** `services.job_service.JobService._persist_and_validate_bundle`

## Contract guarantees

- Every new API reconstruction job preserves its uploaded source bytes and file hash. The decoded RGB used by inference is also retained losslessly as PNG with dimensions, color-space label, and a separate hash.
- Raw predicted depth is stored as float32 NumPy data. A normalized/false-color image is never substituted for the raw depth artifact.
- A validity mask is stored as a single-channel PNG. For the current metric-trained indoor estimator, valid depth is finite and `0.05 < z < 30 m`; other model semantics use finite positive values. The mask is always paired with the raw map and source dimensions.
- Camera intrinsics use pixel units: `fx`, `fy`, `cx`, `cy`; principal points are pixel-center coordinates. A fallback estimate is marked `estimated`. Values supplied by a client are `provided_unverified`; the API does not treat a client assertion as proof of calibration. A verified calibration is marked only if an evidence file is retained and its recorded SHA-256 matches.
- Coordinate frame for canonical unprojection is OpenCV camera coordinates: origin at camera center, `+X` right, `+Y` down, `+Z` forward. Depth semantics separately state whether values are metric model estimates, relative inverse depth, or scene-normalized output.
- The canonical point cloud contains `xyz` float32, `rgb` float32 in `[0,1]`, and row-major source pixel indices. RGB is sampled from the corresponding decoded RGB pixel. Invalid values are excluded before export.
- PLY preserves canonical XYZ/RGB. GLB is a point-cloud runtime export with `POSITION`/`COLOR_0`, never represented as a complete mesh. Semantic bounds remain scene metadata.
- A computation’s job status and an asset’s acceptance status are distinct. Job status remains `COMPLETED` after successful inference/export. The asset may be `REVIEW_REQUIRED` or `REJECTED`. A rejected asset cannot be downloaded from the export endpoint.
- The canonical pixel-to-XYZ acceptance path is single-view. Multi-view submissions preserve every original/decoded input and reconstruction telemetry, but do not claim registered raw depth or validated extrinsics; their output is marked `multiview_unverified` and cannot be accepted as this contract's canonical exhibit without a separately validated registration path.

## Job bundle layout

```text
artifacts/jobs/{job_id}/
├── inputs/view_00.png                    # lossless RGB inference input
├── source/view_00.{jpg,png,webp}          # exact uploaded bytes when submitted via upload API
├── rgb/view_00.png                       # decoded RGB used by inference
├── depth_raw.npy                         # aligned float32 raw depth, when available
├── validity_mask.png                     # valid-pixel mask, when a 2D depth map exists
├── model_native_depth.npy                 # preserved only if a model depth tensor is not a 2D aligned map
├── model_native_outputs.npz               # optional Hunyuan raw depth/points/confidence/normals/camera
├── point_cloud_canonical.npz              # xyz, rgb, source_pixel_indices
├── model_native_geometry.npz              # optional unverified model-native points for forensics
├── scene_reconstruction.ply                # canonical colored point set
├── scene_reconstruction.glb                # deterministic desktop point tier (40k default)
├── scene_reconstruction_mobile.glb         # deterministic mobile point tier (15k default)
├── scene_graph.json                        # semantic scene summary, no large arrays
├── validation_report.json                 # automated checks + optional Khronos result
└── exhibit_manifest.json                  # provenance, semantics, hashes, transforms, metrics, acceptance
```

Files are omitted when the corresponding model output does not exist; null paths include a reason in validation/manifest data. The job store continues to use its existing `artifacts` and `results_summary` JSON fields; no database migration is required.

## Manifest fields

The versioned manifest contains:

```json
{
  "schema": "roomai.exhibit_manifest",
  "schema_version": "1.0.0",
  "job_id": "...",
  "model_id": "roomai_composite_v1",
  "model_name": "...",
  "model_version": "...",
  "source": {"source_original_path": "...", "source_original_sha256": "..."},
  "decoded_rgb": {"path": "rgb/view_00.png", "sha256": "...", "width": 518, "height": 518},
  "preprocessing": {"transforms": [], "camera_intrinsics_after_preprocessing": {}},
  "camera": {"intrinsics": {"fx": 0, "fy": 0, "cx": 0, "cy": 0, "units": "pixels"}},
  "coordinate_frame": "opencv_camera_x_right_y_down_z_forward",
  "depth": {"semantics": "metric_model_estimate", "units": "m", "scale_status": "..."},
  "geometry": {"representation": "colored_single_view_point_cloud", "complete_mesh": false},
  "model_provenance": {"checkpoint_digests": [], "license_status": "..."},
  "runtime": {"python": "...", "packages": {}},
  "code_provenance": [],
  "validation": {"is_valid": false, "errors": []},
  "acceptance": {"status": "REVIEW_REQUIRED", "reasons": []},
  "assets": {}
}
```

Unavailable data is null or omitted with a reason. A missing intrinsic, unit, checkpoint digest, or license record is never silently filled and represented as verified.

## Sampling and export semantics

- Canonical default ceiling: 200,000 valid points; configurable using `canonical_max_points`.
- Desktop GLB default: 40,000 points; accepts existing `downsample_points` and `desktop_max_points` options.
- Mobile GLB default: 15,000 points; configurable using `mobile_max_points`.
- Sampling uses `numpy.default_rng(seed).choice`, sorted source indices, with default seed `0`. Sampling configuration is recorded in the manifest. PLY/GLB parity is checked against the canonical array; GLB RGB is byte-quantized and compared with that quantization tolerance.
- Detection bounding boxes are not emitted as reconstructed room geometry. No triangulation, hidden surface creation, or scale correction is applied.

## API compatibility

Existing job request fields and response fields remain valid. `camera_intrinsics` is an optional additive request field containing numeric pixel-unit `fx/fy/cx/cy`. `JobStatusResponse.asset_acceptance` is optional and additive. Existing routes and database columns are unchanged. The duplicate `app/api` stack is not modified or implied to satisfy this contract.

## Khronos glTF validation

The worker runs the internal binary/accessor/color validator on every exported GLB. If Node and the official `gltf-validator` package are available, it also invokes:

```powershell
node scripts/validate_glb_khronos.mjs artifacts/jobs/{job_id}/scene_reconstruction.glb
```

Install `gltf-validator` only in the release/validation environment (`npm install --no-save gltf-validator`). When unavailable, the manifest report says so; a release still requires the external check. A Khronos-reported error rejects the asset.

## Reproduction command

In the `roomai-hunyuan` environment:

```powershell
python scripts/run_phase4_2_hypersim_smoke.py
```

This runs one held-out synthetic Hypersim frame with the dataset camera matrix and stores results under `asset-production/roomai/01-review/`. It is a diagnostic, not a real-photo or production exhibit.
