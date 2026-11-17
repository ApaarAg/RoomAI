# ROOMAI Phase 4.3 — Final Validation and Real-World Reconstruction Readiness

**Decision: ROOMAI is not ready for production asset generation.** The GLB format gate and mocked HTTP workflow pass. The metric-depth model fails the current multi-scene quality gate, the local evaluation sample is smaller than the planned acceptance set, and no rights-cleared real-camera image with calibration and measured references has been supplied. The generated reconstruction remains a synthetic diagnostic marked `REVIEW_REQUIRED`; it is not an approved exhibit.

## Acceptance matrix

| Gate | Status | Evidence / reason |
|---|---|---|
| Pinned external Khronos validator | **PASS** | `gltf-validator@2.0.0-dev.3.10`, exact npm lockfile, integrated into the job workflow |
| Desktop GLB structure and conformance | **PASS** | 40,000 POSITION/COLOR_0 points; Khronos 0 errors / 0 warnings; internal structure and PLY parity pass |
| Mobile GLB structure and conformance | **PASS** | 15,000 POSITION/COLOR_0 points; Khronos 0 errors / 0 warnings; internal structure and PLY parity pass |
| External validator failure handling | **PASS** | Missing/unavailable/failed validator now makes validation fail; reports are separately persisted and hash-listed |
| FastAPI health/upload/job/status/results/artifact routes | **PASS** | 24 focused tests passed; local Windows socket setup was run outside the restricted sandbox |
| Worker job-failure persistence | **PASS** | Isolated JobService test records `FAILED` and exception details; expensive model inference is not used in HTTP route tests |
| Depth-map numeric validity on available sample | **PASS** | 20/20 benchmark frames had finite valid predictions over all valid reference pixels; 100% coverage on this sample |
| Point-cloud numerical integrity and alignment | **PASS** | All evaluated clouds noncollapsed by the project rank check; source pixel reprojection max `3.05e-5 px` |
| Depth/scale accuracy on multi-scene synthetic benchmark | **FAIL** | Mean AbsRel `0.31979`; mean RMSE `1.87934 m`; the validation scene `ai_016_001` has mean AbsRel `0.48653` and mean δ1 `0.54306` |
| Planned held-out benchmark size | **BLOCKED** | 20 frames over two populated scenes were available; target is 100 frames across five scenes. Other standardized test/validation scene directories are empty. |
| Real-camera validation | **BLOCKED** | No documented rights-cleared indoor photo with camera calibration and independently measured distances is available in the workspace |
| Hunyuan geometry/camera consistency | **FAIL** | Raw synthetic diagnostic has near-zero-scale points, median point-Z/depth ratio `0.01593`, predicted `fy=2,590,000 px`, and reprojection errors hundreds of thousands to millions of pixels under both examined coordinate hypotheses |
| Hunyuan checkpoint identity | **PASS** | Checkpoint SHA-256 matches the configured expected digest |
| Hunyuan production/legal approval | **REVIEW_REQUIRED** | Keep disabled: geometry/camera heads fail; license eligibility depends on deployment territory, product/user scale and distribution obligations |
| Production exhibit / full room mesh | **BLOCKED** | No exhibit is accepted; generated data is a single-view point cloud and must not be described as a complete room mesh |

## Work completed

### GLB validation

The pinned Khronos validator uncovered misaligned three-byte `COLOR_0` records. The exporter now writes color values with a four-byte stride. New desktop/mobile outputs pass Khronos, internal accessors/range checks, and deterministic PLY parity. Full report: [ROOMAI_GLB_VALIDATION_REPORT.md](ROOMAI_GLB_VALIDATION_REPORT.md).

### HTTP test stall and route coverage

The stall reproduces even for a minimal FastAPI app. The stack blocks while Windows AnyIO/Starlette creates a local socket pair for its TestClient portal; outside the restricted sandbox `/health` returns 200. Current dependency versions are FastAPI 0.134.0, Starlette 1.0.1, HTTPX 0.28.1 and AnyIO 4.13.0. No compatibility error occurred in the permitted test run. Tests cover health, readiness, models, upload, validation failures, reconstruction submission, status/results, export retrieval, submission failure and worker failure. Details: [ROOMAI_HTTP_TEST_REPORT.md](ROOMAI_HTTP_TEST_REPORT.md).

### Held-out Depth Anything V2 evaluation

The evaluation ran the real local **Depth Anything V2 Metric Indoor Small** checkpoint on 10 evenly sampled frames from each of two standardized Hypersim scenes: `test/ai_013_010` and `val/ai_016_001`. Scene/reference intrinsics and metric planar-Z depth are provided by the synthetic dataset. No per-frame scale fit was applied.

| Scene | Frames | AbsRel mean | RMSE mean (m) | δ1 mean | Median GT/pred scale ratio | Valid coverage mean | Median robust XYZ extent (m) |
|---|---:|---:|---:|---:|---:|---:|---|
| `test/ai_013_010` | 10 | 0.15305 | 0.64568 | 0.87828 | 0.89483 | 1.00000 | 5.712 × 3.836 × 4.541 |
| `val/ai_016_001` | 10 | 0.48653 | 3.11300 | 0.54306 | 0.88489 | 1.00000 | 9.154 × 6.898 × 14.143 |
| Frame-weighted aggregate | 20 | 0.31979 | 1.87934 | 0.71067 | 0.89483 median of per-frame ratios | 1.00000 | — |

Frame-level median GT/pred scale ratios vary from `0.56707` to `2.44207`, despite complete predicted-depth coverage. The metric-trained checkpoint therefore does not deliver stable, independently calibrated scale across these scenes. The model’s nominal “metric” output must remain described as an estimate.

Point bounds/PCA and image reprojection are recorded for every frame in `asset-production/roomai/01-review/phase4_3-depth-evaluation-run01/depth_benchmark_report.json`. Visual comparison for both scenes is stored alongside the report. The benchmark is synthetic-render evidence; it does not establish performance on camera photographs.

### Hunyuan investigation

The configured checkpoint at `results/final_execution/checkpoints/exp_05_expanded_lora_rank8_final.ckpt` hashes to `2472ebadcf06ead0276c2e7fbd7467a6010c1a5cd591aa066e7ea8c57cec699f`, matching `config/model_config.yaml`.

On the same synthetic frame, native heads were finite and full resolution: points `[1,1,518,518,3]`, depth `[1,1,518,518,1]`, point confidence `[1,1,518,518]`, normals `[1,1,518,518,3]`. They do not describe consistent scene geometry: the transformed XYZ span is only about `0.0368 × 0.0877 × 0.0586` in native units; median point `|Z|` is `0.01505`, while depth-head median is `0.94465`. This is a native-scale mismatch, not proof that either head is in meters.

The model does return `camera_params`, `camera_poses`, and `camera_intrs`; however, the current service asks for a non-existent `cam` key and records `camera: null`. The service also drops `depth_conf` and `normals_conf`, and its `confidence_threshold` parameter is not applied. The direct camera head predicted `fx=261.73 px` and `fy=2,590,000 px` on a 518×518 frame whose dataset intrinsics are `fx=447.90`, `fy=598.13`. Reprojection median errors were approximately 9.59 million pixels assuming camera-frame points and 449,066 pixels assuming world-frame points. Coordinate semantics remain unresolved; neither hypothesis is acceptable.

The complete raw model heads and forensics are under `asset-production/roomai/01-review/phase4_3-hunyuan-investigation-run03/`. No Hunyuan output is enabled for production. The local Tencent community license has territory, distribution and scale eligibility terms; verify legal applicability to the actual service/deployment before any product use.

## Reproducible evidence locations

- Latest integrated composite bundle: `asset-production/roomai/01-review/phase4_3-pipeline-ai_013_010-frame_0000-run02/`.
- Full multi-scene evaluation: `asset-production/roomai/01-review/phase4_3-depth-evaluation-run01/`.
- Full Hunyuan raw head capture: `asset-production/roomai/01-review/phase4_3-hunyuan-investigation-run03/`.
- Standalone desktop/mobile GLB tier and full structure validation: `asset-production/roomai/01-review/phase4_3-glb-validation-final-run02/`.
- Phase 4.3 scripts: `scripts/evaluate_phase4_3_depth.py`, `scripts/investigate_phase4_3_hunyuan.py`, `scripts/validate_phase4_3_glb_tiers.py`.

## Required next input

Use the intake procedure in [ROOMAI_REAL_WORLD_TEST_PROTOCOL.md](ROOMAI_REAL_WORLD_TEST_PROTOCOL.md). The project owner must provide original indoor RGB image(s), explicit evaluation/public-display permission scope, available camera calibration, and at least three measured distances with units, endpoints, method and uncertainty per scene. An independent registered depth capture is optional but preferred.

Until those inputs are evaluated and the scene-held-out benchmark is expanded, **do not approve ROOMAI for production asset generation or begin website development**.
