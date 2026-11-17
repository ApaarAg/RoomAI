# ROOMAI Phase 4.3 — Khronos GLB Validation Report

**Status:** Desktop and mobile point-cloud GLBs pass the pinned Khronos validator and RoomAI structural/parity checks. This is a file-integrity result; it does not approve the reconstruction as a production exhibit.

## Validator setup

- Validator: official Khronos `gltf-validator` npm package, pinned to `2.0.0-dev.3.10`.
- Reproducibility: exact version in `validation/gltf-validator/package.json` and integrity-resolved dependency in `validation/gltf-validator/package-lock.json`.
- Install from the repository root by running `npm ci --ignore-scripts` in `validation/gltf-validator/`.
- Validation entry point: `scripts/validate_glb_khronos.mjs`.
- The deployed reconstruction job now invokes the external validator for **both** tiers. A missing Node runtime, package, output report, or failed validation marks the asset validation failed; external validation is not silently skipped.
- The job writes complete Khronos JSON reports as separate assets and includes their paths and SHA-256 digests in `exhibit_manifest.json`.

The package/version is listed by the [official Khronos Validator npm package](https://www.npmjs.com/package/gltf-validator); the [Khronos project documentation](https://github.com/KhronosGroup/glTF-Validator) describes GLB, accessor and binary-buffer checks.

## Initial failure and correction

The validator exposed a real issue in the previous exporter: tightly packed RGB `COLOR_0` values had a 3-byte per-vertex step. Khronos reported `MESH_PRIMITIVE_ACCESSOR_UNALIGNED` because vertex attributes must meet four-byte alignment.

The exporter now stores normalized unsigned-byte RGB in a 4-byte stride (three color bytes plus one unused pad byte), while keeping the glTF accessor as `VEC3`. Point coordinates, color values, tier counts, deterministic sampling, and the point-cloud representation are unchanged. The decoder and internal range validator understand the stride. The original GLBs were not overwritten.

## Final validation evidence

Source reconstruction: `asset-production/roomai/01-review/phase4_3-pipeline-ai_013_010-frame_0000-run02/point_cloud_canonical.npz` (200,000 canonical points, from the real composite inference output on a Hypersim synthetic render).

Reproducible isolated export check: `asset-production/roomai/01-review/phase4_3-glb-validation-final-run02/`.

Integrated job evidence and complete per-tier reports: `asset-production/roomai/01-review/phase4_3-pipeline-ai_013_010-frame_0000-run02/`.

| Check | Desktop | Mobile |
|---|---:|---:|
| Outcome | **PASS** | **PASS** |
| Khronos version | 2.0.0-dev.3.10 | 2.0.0-dev.3.10 |
| Khronos errors / warnings | 0 / 0 | 0 / 0 |
| File size | 640,964 bytes | 240,964 bytes |
| `POSITION` count | 40,000 | 15,000 |
| `COLOR_0` count | 40,000 | 15,000 |
| Primitive | `POINTS` (mode 0) | `POINTS` (mode 0) |
| Position type | float `VEC3` | float `VEC3` |
| Color type | normalized unsigned-byte `VEC3`, stride 4 | normalized unsigned-byte `VEC3`, stride 4 |
| Position data bytes | 480,000 | 180,000 |
| Color data bytes | 160,000 | 60,000 |
| Total declared GLB buffer | 640,000 | 240,000 |
| Node matrix/translation/rotation/scale | none (identity) | none (identity) |
| Coordinate metadata | OpenCV camera, meters (model estimate), point cloud, not a surface mesh | same |

The position accessor min/max agree with the values read back from the GLB. Desktop bounds are `min=[-2.24434,-2.87061,2.51414]`, `max=[4.08983,1.13996,7.16078]`; mobile bounds are `min=[-2.23682,-2.87061,2.51414]`, `max=[4.08983,1.13996,7.15661]`. The mobile differences are expected from deterministic subsampling. The RoomAI PLY/GLB parity check also passed for both deterministic subsets.

The GLB metadata says `units: m` because the source is the metric-trained model estimate. It does **not** mean the scene has independent metric calibration; the manifest continues to classify it as a model estimate and leaves exhibit acceptance at `REVIEW_REQUIRED`.

## Report files

- Integrated: `phase4_3-pipeline-ai_013_010-frame_0000-run02/khronos_desktop_report.json` and `khronos_mobile_report.json`; both are hash-listed in the exhibit manifest.
- Standalone: `phase4_3-glb-validation-final-run02/khronos_desktop.json`, `khronos_mobile.json`, and `glb_validation_summary.json`.
- Initial failure evidence is retained in the earlier `phase4_3-glb-validation-ai_013_010-run01/` directory and documents the corrected alignment error.

## Acceptance boundary

**GLB serialization and glTF conformance: PASS.**  
**Geometry/provenance/scale for public exhibit use: REVIEW_REQUIRED.** The GLBs are point clouds, not a complete room mesh. Their depth scale has not passed the scene-level gate and no rights-cleared real-camera reference test has been supplied.
