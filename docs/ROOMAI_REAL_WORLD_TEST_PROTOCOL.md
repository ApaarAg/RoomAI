# ROOMAI Real-World Reconstruction Intake and Test Protocol

**Purpose:** Evaluate the current monocular depth estimate against an actual indoor camera image, known camera parameters, and independent physical references. No real-camera evaluation has been run yet. Do not fill unknown fields with inferred values.

## Required intake package

Provide one folder per scene containing the original capture and an evidence sidecar. For a model acceptance decision, use at least three independent rooms/scenes; one image is only an intake/shakedown run.

### 1. Original RGB image

- Original, unedited JPEG/PNG/HEIF converted only if the original file is also retained. Include the exact file used for evaluation and its SHA-256.
- Record original dimensions, color profile, EXIF orientation, capture date, camera make/model, lens, focal setting, digital zoom, and whether any crop, resize, stabilization, HDR, or image enhancement was applied.
- Do not remove EXIF before archiving the original. Store a separate orientation-normalized decode only when the transformation is recorded.
- Capture a normally lit furnished room with visible planar structure. Avoid people, family photos, addresses, screens, mirrors, glass-heavy views, and private documents. Do not use AI-generated imagery as evidence.

### 2. Rights and use permission

Include an explicit written permission record from the photographer/rightsholder that covers:

- internal processing for reconstruction evaluation;
- storage/retention of the image and derived depth/geometry;
- whether the image and any derived result may appear in a public ROOMAI website, portfolio, awards submission, marketing or social post;
- attribution, expiration, territory and any restrictions.

Internal testing permission does not imply public display permission. If public-use scope is not explicit, mark provenance `REVIEW_REQUIRED` and do not publish the source or derivative.

### 3. Camera calibration

Provide calibration for the exact camera/lens/resolution/focus configuration where available:

- pixel-unit intrinsic matrix `K` (`fx`, `fy`, `cx`, `cy`) for the original image dimensions;
- lens distortion model and coefficients;
- calibration method/date and calibration target/dataset;
- file and SHA-256 for a calibration result, manufacturer/device record, or calibration-board capture set.

If only manufacturer metadata or an estimated focal length is available, record that evidence and mark it `uncalibrated` or `provided_unverified`. Do not invent intrinsics. If no intrinsics are available, record `null`; the pipeline's centered focal heuristic may be used only as a separately labeled diagnostic.

### 4. Physical reference distances

Record at least three independently measured distances in the scene, preferably spanning near, middle and far depth and not all parallel. For each measurement include:

- measured value and unit;
- instrument/method (e.g. laser rangefinder or tape measure), date, and stated uncertainty;
- the two physical endpoints and how they are visible in the image;
- endpoint pixel coordinates in the unmodified RGB image, with a marked-up copy stored separately;
- whether the measurement is line-of-sight, horizontal, vertical, or another dimension.

Do not estimate a measurement from a photo or from the model output. If a point cannot be located reliably in the RGB, mark that reference unusable rather than moving it to fit the reconstruction.

### 5. Optional independent depth reference

An aligned RGB-D capture, depth camera frame, calibrated LiDAR scan, or surveyed room scan can provide stronger pixel-wise evaluation. Include the original raw depth, its scale/unit encoding, validity mask, depth-camera intrinsics, depth-to-color extrinsics, timestamps, registration/calibration procedure, and source license. Preserve raw data and distinguish measured depth from interpolated or hole-filled depth. Do not treat Depth Anything or another monocular prediction as ground truth.

## Intake sidecar template

Create `scene_intake.json` next to the source files and replace every `null` only with evidence-backed values:

```json
{
  "schema": "roomai.real_world_intake",
  "schema_version": "1.0",
  "scene_id": "",
  "rgb": {
    "filename": "",
    "sha256": "",
    "width": null,
    "height": null,
    "original_format": "",
    "orientation": "",
    "camera_make_model": "",
    "lens": "",
    "focal_setting": null,
    "digital_zoom": null,
    "edits_or_transforms": []
  },
  "rights": {
    "rightsholder": "",
    "permission_evidence_file": "",
    "internal_evaluation": null,
    "public_website_display": null,
    "marketing_or_awards_display": null,
    "attribution": null,
    "territory": null,
    "expiration_or_restrictions": null
  },
  "camera_calibration": {
    "status": "unknown",
    "intrinsics_pixel_units": null,
    "distortion_model": null,
    "distortion_coefficients": null,
    "evidence_file": null,
    "evidence_sha256": null,
    "calibration_resolution": null
  },
  "reference_distances": [
    {
      "id": "",
      "distance": null,
      "unit": "",
      "method": "",
      "uncertainty": null,
      "endpoint_a_pixel_uv": null,
      "endpoint_b_pixel_uv": null,
      "measurement_geometry": "",
      "evidence_file": null
    }
  ],
  "independent_depth": {
    "available": false,
    "filename": null,
    "sha256": null,
    "units": null,
    "scale_to_meters": null,
    "validity_mask_filename": null,
    "depth_intrinsics": null,
    "depth_to_rgb_extrinsics": null,
    "registration_evidence": null
  }
}
```

## Evaluation procedure

1. Freeze and hash the supplied RGB, calibration, permission and measurement records. Preserve originals unchanged.
2. Verify dimensions, color/orientation handling, intrinsics resolution, distortion assumptions and any RGB/depth registration. Record every crop/resize and transformed `K`.
3. Run the existing Depth Anything V2 Metric Indoor Small pipeline without fitting a scale factor. Persist raw float depth and mask.
4. If independent depth is supplied, align using the documented calibration only; report AbsRel, RMSE in meters, median absolute error, δ1 (`max(pred/GT, GT/pred) < 1.25`), coverage and per-range/per-region metrics. Do not scale-fit the prediction for the primary metrics.
5. For each physical reference, unproject both marked endpoints with the verified camera model and compare the reconstructed separation to the independent measured value. Report signed and absolute error plus measurement uncertainty. A known-distance scale correction may be reported only as a separate diagnostic and never used to claim uncalibrated prediction accuracy.
6. Record point count, finite ratio, AABB, robust p01–p99 extents, covariance eigenvalues/PCA, collapse flag, and RGB-to-XYZ reprojection error. Produce source/depth/point-cloud views for human review.
7. Compare the scene-level results to predeclared thresholds; record failures and invalid regions instead of excluding them silently. Keep each scene's result separate before aggregate statistics.
8. Mark the output `REVIEW_REQUIRED` until rights, camera evidence, physical references, geometry and human visual review all pass.

## Release-quality evidence target

The current validation plan requires at least three rights-cleared, independent real indoor scenes with known intrinsics and measured reference distances. Independent registered depth is optional for intake but strongly preferred for pixel-wise accuracy metrics. Use multiple scenes and never adjacent frames from one room as a substitute for independent rooms.

## Still needed from the project owner

1. Original indoor RGB file(s), unchanged.
2. Written permission that explicitly covers internal evaluation and states whether public showcase use is allowed.
3. Camera/lens calibration evidence at the capture resolution, or an explicit statement that it is unavailable.
4. At least three measured, image-locatable distances per scene with units, method and uncertainty.
5. Optional independent registered depth/scan and its rights/calibration evidence.
