# ROOMAI Reconstruction Validation Plan — Phase 4.1

**Status:** Proposed test gates; no backend code changed.  
**Goal:** Prove that one run’s RGB, depth, mask, intrinsics, points, geometry, and GLB describe the same scene with honest scale semantics.

## Validation pipeline

```text
Rights + source identity
  -> decode/preprocess transform checks
  -> depth semantics and numeric integrity
  -> RGB/depth registration
  -> camera/coordinate reprojection checks
  -> point distribution and bounds checks
  -> GLB/PLY parity and format checks
  -> visual review + independent accuracy benchmark
  -> reproducible manifest + accept/reject decision
```

Validation results are fail-closed. `COMPLETED` means computation ended; `ACCEPTED` is a separate state earned only after all required gates pass.

## Automated checks

| Gate | Checks | Fail conditions / output |
|---|---|---|
| Rights and provenance | Rights evidence exists; source SHA matches immutable original; upload ID, job/run ID, model/config/checkpoint hash and code revision are joined. | Missing/unclear public-use permission blocks public acceptance. Missing checkpoint/config hash blocks reproducibility. |
| RGB/depth registration | `depth.shape == mask.shape == decoded_rgb.height/width` after recorded transform; transformation chain maps pixel centers correctly; sample mask/depth overlay and edge alignment; all resizes/crops update K. | Shape mismatch, unknown crop, unmatched input hash, or >2 px median reprojection/edge displacement on calibration fixture fails. For real scenes without labels, produce a visual overlay and mark registration `review_required`. |
| Depth integrity | dtype/endianness/shape; finite ratio; valid mask definition; positive/in-range values for valid pixels; invalid sentinel only outside mask; stable min/median/max and percentile stats. | NaN/Inf in valid pixels, empty mask, range contradiction, false-color input, or mismatched units fails. Out-of-range values are reported, not silently clipped into valid data. |
| Depth semantics/scale | Model/config and output-head semantics agree with `depth.representation`; no post-hoc scale fit for metric acceptance. Compare against independent GT without scale fitting. Report scale-fit only as a separate relative-shape diagnostic. | Relative/canonical output labelled meters fails. A model name containing “Metric” is insufficient. Unknown scale is rejected for metric geometry claims. |
| Unprojection consistency | For sampled valid pixels, compute expected XYZ from K/depth; compare to point cloud; reproject XYZ to pixel centers; round-trip camera/world/export transforms. Check axis handedness and determinant. | XYZ depth differs from source depth beyond numeric tolerance, reprojection exceeds 1 px in untransformed fixtures, reflection/duplicate transform detected, or frame metadata absent. |
| Point-cloud integrity | Count >0; finite XYZ; RGB count equals XYZ count and channels within range; deterministic sample hash; duplicate ratio; robust 1st–99th percentile extents; covariance eigenvalues/PCA; compare point Z histogram to raw depth. | Empty/non-finite/misaligned colors fail. Flag collapse if fewer than two robust axes have meaningful spread or covariance rank is <2 for a room sample. A truly planar single surface may pass as `partial_surface`, not as a room volume. |
| Bounds and scale | Report AABB, robust extents, centroid, depth range, camera origin, and room-layout dimensions. Compare estimates with known camera/calibration/measurement when available. | Negative/zero spans, impossible units, extreme dimensions, or layout metadata not grounded in points fail or require review. Do not use generic room ranges as proof of metric accuracy. |
| GLB integrity | Validate GLB header, chunk boundaries/alignment, buffer lengths, bufferViews/accessors/counts/strides, finite positions, bounds, supported primitive modes, material/color attributes, and transform metadata. Run Khronos glTF Validator in CI/release validation. | Any invalid accessor/buffer, undeclared units/frame, color loss where required, or export falsely described as mesh fails. |
| PLY/GLB parity | Parse both files. For full parity, compare identical XYZ/RGB arrays. For a declared reduced GLB, reproduce deterministic subset/voxel set from PLY and compare points/colors after the documented coordinate transform and quantization tolerance. | Unexplained count differences, random unreproducible subset, color mismatch, coordinate/bounds mismatch, or box meshes masquerading as surfaces fails. |
| Reconstruction completeness | Record `point_cloud`, `partial_depth_surface`, or `validated_mesh`; count occlusion/invalid pixels; identify holes and view coverage. | Single view must not be accepted as a complete room mesh. Missing surfaces are reported as unobserved, not synthesized. |
| Visual quality | Render RGB, raw-depth preview, point cloud from front/side/top, and GLB in at least two independent viewers. Reviewer checks recognizable structure, no collapsed axis, no displaced color, no obvious floating/tilted room, and no invented faces. | Severe collapse, alignment drift, export mismatch, or materially misleading surfaces rejects the run even if containers parse. |

## Provisional acceptance thresholds

These are starting thresholds for discussion, not claims that one successful frame qualifies a model:

### Metric model estimate

- Held-out indoor benchmark: at least 100 frames across at least 5 distinct scenes, split by scene (not adjacent frames only).
- Unscaled AbsRel ≤ 0.20 and δ1 ≥ 0.70 as initial quality gates; publish RMSE and median absolute error too.
- Median `GT depth / predicted depth` in `[0.90, 1.10]` on each scene for meter-scale acceptance; no per-frame scale fitting to pass.
- At least 95% finite valid depth coverage on designated indoor regions, with failures/invalid coverage reported.
- Real-camera validation: at least 3 rights-cleared indoor photos with known sensor intrinsics and independent measured distances/depth references. Test outside model training/validation images.

The one-frame Hypersim diagnostic has AbsRel 0.1379 but a GT/predicted median ratio of 0.8561, so it passes only the provisional AbsRel criterion and fails the provisional absolute-scale gate. It is synthetic-render evidence only.

### Point/geometry gates

- At least 10,000 valid points for the desktop exhibit candidate after filtering; mobile LOD is validated later and not considered canonical.
- All points finite; RGB aligned/count-matched; minimum 1,000 unique points after deduplication.
- For a room-volume claim, at least two robust extents nontrivial and covariance rank ≥2; for a full-volume/mesh claim, rank ≥3 with connected surfaces and independent surface validation. Thresholds derive from fixture dimensions and noise tolerance, not hard-coded generic room dimensions.
- Room layout must be marked `uncalibrated` unless dimensions are checked against independent measured references.

### Export gates

- PLY parser checks fields, row count, finite XYZ, RGB bounds, and units/frame from manifest.
- GLB passes Khronos validator and project semantic checks; points carry `POSITION` and `COLOR_0`; buffer/accessor ranges and bounds are correct.
- GLB uses a deterministic declared subset or full PLY content. Repeated runs with same inputs/config/model produce the same point sampling and manifest hashes, apart from explicitly nondeterministic inference fields.
- Round-trip rendering from GLB agrees visually and numerically with canonical PLY and RGB source.

## Test matrix

| Test | Data | Purpose | Product evidence? |
|---|---|---|---|
| Unit: depth-to-XYZ | Synthetic numeric arrays with known K/points/masks | Formula, axis, units, reproject/intrinsic transform | No |
| Unit: format exporters | Small known colored point sets and optional mesh | PLY/GLB geometry/color/accessor parity | No |
| Controlled held-out benchmark | Hypersim CC BY-SA scene-held-out RGB + exact depth/K; record render provenance | Numeric depth, alignment, coordinate and metric-scale checks | No; synthetic render only |
| Research-only cross-domain benchmark | ScanNet only if terms permit this research run | Diagnostic generalization only; never commercial/internal commercial development under recorded terms | No |
| Rights-cleared real-room validation | User-owned/licensed camera photo, actual K and measured reference distances/depth | Validate real-photo domain, camera geometry, scale and visual suitability | Candidate only; requires license record |
| End-to-end job/API | Same immutable fixture through deployed `api.main` upload/job/export routes | Persistence, state/failure behavior, manifest and export storage | Only after source/model rights gates |
| Visual review | RGB/depth/points/GLB orthographic views | Detect collapsed, shifted, noisy or misleading geometry | Candidate review |

## Required test run record

Every run stores the input ID/hash, rights/license, model/checkpoint/config/code hashes, environment, preprocessing transforms, raw depth and mask hashes, K/extrinsics/frame, all metrics, point counts/bounds/PCA, PLY/GLB parser results, visual review links, and final `accepted/rejected` status.

## Current baseline from the controlled diagnostic

The composite adapter completed in 11.731 seconds in `roomai-hunyuan` (RTX 4060 Laptop GPU), returned a 518×518 finite depth map and a 200,000-point colored PLY, with a valid GLB container. Depth was not physically ground truth: unscaled AbsRel 0.1379, RMSE 0.7211 m, scale ratio 0.8561. Camera K was guessed at fx/fy=1000 while the held-out frame K was fx=447.896/fy=598.134. GLB had positions only and six meshes, including five detection boxes. This is a useful smoke test, not acceptance.

The P0 Hunyuan exhibit remains rejected based on the asset report: nearly degenerate geometry, no raw depth, incorrect metric metadata, no confidence/camera provenance, and rights/checkpoint concerns.
