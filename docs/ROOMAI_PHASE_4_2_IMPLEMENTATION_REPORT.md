# ROOMAI Phase 4.2 — Reconstruction Pipeline Remediation

**Status:** Implemented on the deployed `uvicorn api.main:app` path; focused tests and one controlled benchmark passed. The benchmark bundle is marked `REVIEW_REQUIRED`, not production accepted.  
**Scope:** Reconstruction backend, artifact persistence, validation, and tests. No frontend code, existing source assets, or `app/api` duplicate implementation was modified.

## Baseline and preservation

The workspace had extensive pre-existing modifications and untracked files, including the deployed API tree and artifact store; it was not a clean Git checkout. Before editing, SHA-256 hashes and copies of each initially targeted file were recorded under:

```text
artifacts/phase4_2_baseline-20261112-003351/
```

The pre-existing `tests/test_model_adapters.py`, whose Hunyuan capability expectation contradicted implementation, was copied into that backup as well. New job IDs are checked before writing so an existing job directory is never overwritten. Original uploaded file bytes are copied into each new job bundle, and decoded RGB is stored losslessly as PNG. Existing artifacts were not deleted or replaced.

## Architecture

```text
uvicorn api.main:app
  POST /api/v1/uploads -> artifacts/uploads/{upload_id}.{ext}
  POST /api/v1/jobs/reconstruct
    -> api/routers/jobs.py (adds source paths and optional pixel-unit K)
    -> services/job_service.py
       -> models.adapters.registry
          -> roomai_composite_v1 (canonical single-view path)
             -> SceneFusionService + Depth Anything V2 Metric Indoor Small
       -> utils/reconstruction_contract.py
          -> lossless source/depth/mask persistence
          -> deterministic camera-aware depth unprojection
          -> numerical, alignment, collapse and reprojection validation
       -> utils/export_3d.py
          -> canonical colored PLY, deterministic colored desktop/mobile GLBs
          -> internal GLB/PLY structural validation and parity checks
       -> exhibit_manifest.json + validation_report.json
       -> job status COMPLETED + separate asset_acceptance state
```

The parallel `app/api` implementation remains untouched. Hunyuan remains staged and is not an approved production source.

## Implemented changes

- **Source and depth evidence:** preserve original upload bytes and SHA-256 where the request comes through the upload API; save the exact decoded RGB as PNG with its own hash; persist raw float32 depth without visualization normalization and a pixel validity mask. Hunyuan confidence, raw depth, raw points, normals, camera prediction, and the model-to-RoomAI transform are retained in a native `.npz` when available.
- **Camera semantics:** API requests can optionally provide `fx/fy/cx/cy` in pixel units. Omitted intrinsics use an explicit estimated centered focal heuristic. Provided intrinsics are marked `provided_unverified`; they become `calibrated` only when a calibration evidence file is present, its hash matches, and the internal caller marks that evidence verified. Hunyuan center crop and resize are recorded and K is transformed accordingly.
- **Depth semantics:** manifests distinguish `metric_model_estimate`, `relative_inverse`, and `scene_normalized`. Only the default metric-trained estimator is unprojected into the canonical production candidate cloud; model estimates are still not described as ground truth. Hunyuan is always `scene_normalized`, carries no meter units, and remains rejected.
- **Point cloud:** single-view metric-estimate XYZ is regenerated from the saved depth map, the selected pixel-unit K, and corresponding source RGB pixels. Invalid/non-finite depth is excluded. Deterministic sampling saves source pixel indices. The manifest records bounds, robust extents, covariance eigenvalues, point counts, and reprojection error.
- **Exports:** PLY includes XYZ, RGB, units, camera frame, and a `colored_point_cloud_not_mesh` marker. Desktop and mobile GLBs contain matching `POSITION` and `COLOR_0` arrays, use deterministic samples (40k/15k defaults), identify units/frame, and contain only point primitives. Semantic detection boxes no longer appear as reconstructed room surfaces.
- **Validation and job state:** structural and numerical checks run before completion. A computation may be `COMPLETED` while its asset is `REVIEW_REQUIRED` or `REJECTED`. Rejected output cannot be downloaded through the GLB/PLY route. The status response exposes the separate acceptance state.
- **Hunyuan:** capability claims for metric depth, mesh GLB, and 3DGS were corrected; available weights are `STAGED`, not production-ready. Raw outputs and checkpoint SHA-256 are recorded for diagnosis; mismatch against the configured expected digest is a validation error. Product use remains blocked for geometry and licensing review.
- **Reproducibility:** manifests include hashes for input/output files, runtime package versions, pipeline code, model config, and default composite checkpoints. `scripts/run_phase4_2_hypersim_smoke.py` reproduces the controlled diagnostic. `scripts/validate_glb_khronos.mjs` invokes the official Khronos validator when its npm package is installed; the job worker reports validator availability/results.

## Files changed

- `api/routers/jobs.py`, `api/schemas.py`
- `services/job_service.py`, `services/scene_fusion_service.py`, `services/hunyuan_mirror_service.py`
- `models/adapters/roomai_composite.py`, `models/adapters/hunyuan_mirror_adapter.py`, `models/scene3d_model.py`
- `utils/export_3d.py`
- `utils/reconstruction_contract.py` (new)
- `scripts/run_phase4_2_hypersim_smoke.py` (new)
- `scripts/validate_glb_khronos.mjs` (new)
- `tests/test_reconstruction_contract.py` (new)
- `tests/test_model_adapters.py` (capability assertion corrected)
- `docs/ROOMAI_BACKEND_OUTPUT_CONTRACT.md` (new)
- `docs/ROOMAI_RECONSTRUCTION_TEST_REPORT.md` (new)

## Validation evidence

- `py_compile` passed for all changed Python files; `node --check scripts/validate_glb_khronos.mjs` passed.
- Focused tests: **22 passed** across the new contract tests, job lifecycle, model adapters, and scene fusion integration. The exact final count is recorded in `ROOMAI_RECONSTRUCTION_TEST_REPORT.md`.
- Held-out Hypersim test frame `ai_013_010/frame_0000` ran through the composite adapter and artifact writer in 16.2 seconds on the available RTX 4060 Laptop GPU. Raw depth was 518×518 with 100% finite/valid pixels; 200,000 canonical colored points were produced. PLY/desktop GLB/mobile GLB passed internal structure and parity checks. The point cloud was not collapsed and XYZ-to-pixel maximum reprojection error was approximately `3.1e-5` px.
- Depth metrics were unscaled AbsRel `0.1379`, RMSE `0.7211 m`, and ground-truth/prediction median ratio `0.8561`. No scale fitting was applied. This fails the provisional absolute-scale tolerance and therefore remains review-only.
- Visual review found room structure recognizable from multiple orthographic views, with expected single-view holes and sparse/ambiguous surfaces. No faces or hidden room surfaces were synthesized.
- The official Khronos validator package is not installed in this environment. Its runner is integrated, but the external validation step reports unavailable; internal GLB checks passed.
- The pre-existing FastAPI `TestClient` suite stalled at its first `/health` request in this environment before producing an assertion result. Direct job, adapter, exporter, and reconstruction tests passed. This harness issue is noted rather than counted as a pass.

## Remaining limits and release gates

1. There is no rights-cleared **real camera** room photograph with independent distance references in the workspace. Hypersim is a synthetic render and is not product evidence.
2. Monocular absolute scale has not met the project’s acceptance threshold. Preserve its output as a model estimate until scene-held-out and real-room checks pass.
3. Ordinary API-supplied K is not calibration proof. It remains `provided_unverified` unless accompanied by verifiable calibration evidence.
4. Multi-view fusion remains experimental; the canonical aligned depth-to-pixel point bundle is currently limited to one view. Multi-view output is preserved but rejected/reviewed rather than falsely labeled aligned.
5. Hunyuan remains rejected until point/depth/confidence/camera consistency, checkpoint provenance, commercial rights, and geometry quality pass review.
6. The optional Khronos validator must be installed in the release environment and run before asset release.

No complete mesh or hidden room geometry is claimed.
