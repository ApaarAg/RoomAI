# ROOMAI Reconstruction Pipeline Audit — Phase 4.1

**Audit date:** 2026-11-12  
**Disposition:** The current default composite path can produce a plausible single-view, metric-depth point-cloud estimate. It is not yet a reproducible, camera-calibrated, production-accepted reconstruction pipeline. The Hunyuan path that produced the P0 photo exhibit is not acceptable as-is.  
**Scope:** Read-only code/configuration audit plus one controlled diagnostic run. No backend source, existing source artifact, or existing job database was modified. No frontend work was started.

## Executive assessment

There are two different answers depending on what “reconstruction” means:

- **Visible-side 2.5D point cloud:** realistically achievable with the default `roomai_composite_v1` path. It estimates depth, unprojects pixels, and can export a colored PLY. A one-frame controlled run produced coherent room-like structure and passed the project’s basic PLY/GLB container checks.
- **A calibrated, complete 3D room or production exhibit:** not established. Single-view depth cannot recover occluded surfaces. The default path uses guessed focal lengths and does not preserve its raw depth or explicit validity mask in job artifacts. Its “metric” output is a learned estimate, not a camera-calibrated measurement. The current GLB omits point colors and includes detection boxes as geometry. The Hunyuan exhibit’s point map is collapsed and its output scale was mislabeled.

The pipeline can be remediated without replacing its working detector, segmenter, depth estimator, or asynchronous job flow. The minimum work is to establish one canonical output contract, record the true preprocessing/camera metadata, persist raw arrays and provenance, validate geometry before marking a job complete, and make exports represent the same colored point set. Keep Hunyuan out of product acceptance until its point output, scale, checkpoint lineage, and license are resolved.

## Evidence and scope limits

The four requested Phase 4 documents were read. The deployed entrypoint is confirmed by `docs/DEPLOYMENT.md`: `uvicorn api.main:app`. Its reconstruction route is `api/routers/jobs.py`, which calls `services/job_service.py` and the registry in `models/adapters/`. The artifacts audited in Phase 4 P0 came from this path.

There is also a second, overlapping stack under `app/api`, `app/workers`, and `app/services`. `app/api/main.py` is not the deployment entrypoint documented by the project. It uses a separate `GPUWorker`, model registry, reconstruction service, and exporter. Its existence is a maintainability risk: behavior and output claims can differ depending on which entrypoint is launched.

No local uploaded camera photograph has a documented public-use license. The living-room upload in the P0 report has unverified rights; the other paired source is an internal-proprietary schematic. For a controlled test only, the default composite path was run on Hypersim `ai_013_010`, held-out test frame 0000. This is a V-Ray synthetic render, not a photograph and not website evidence. The local Phase 9.1 licensing audit records Hypersim as CC BY-SA 3.0. Test outputs were kept in the review directory and not promoted.

## Current architecture

```text
Deployment: uvicorn api.main:app
  POST /api/v1/uploads
    -> services/upload_service.py
    -> artifacts/uploads/{upload_id}.*
  POST /api/v1/jobs/reconstruct
    -> api/routers/jobs.py
    -> services/job_service.py (SQLite queue + worker)
    -> models.adapters.registry.RECONSTRUCTION_REGISTRY
       ├─ default roomai_composite_v1
       │   └─ models/adapters/roomai_composite.py
       │       └─ services/scene_fusion_service.py
       │           ├─ YOLO11s A3 indoor object detector
       │           ├─ YOLO11n-Seg instance segmenter
       │           ├─ Depth Anything V2 Metric Indoor Small
       │           ├─ pixel-depth unprojection to colored XYZ
       │           └─ RANSAC/layout and object-depth heuristics
       └─ explicit hunyuan_world_mirror_1b
           └─ models/adapters/hunyuan_mirror_adapter.py
               └─ services/hunyuan_mirror_service.py
                   └─ Tencent HunyuanWorld-Mirror 1.0
                      camera/depth/point/normal heads; optional GS head
    -> services/job_service.py
       ├─ utils/export_3d.py -> ASCII PLY + custom GLB
       ├─ shallow container validators
       └─ scene_graph.json (point_cloud/depth_map/heatmap excluded)
```

`app/api/main.py` is a parallel implementation and is not the active route used for the P0 job records. It has a second inference engine and exporter. Remediation should first select and document the canonical service path; avoid silently porting half the fix into the inactive stack.

## Exact model and algorithm paths

### Default composite (`roomai_composite_v1`)

The adapter’s capability description names YOLO11s A3, YOLO11n-Seg, Depth Anything V2 Metric Indoor Small, RANSAC layout estimation, and SIFT/Umeyama for multi-view. For a single view, the deployed path is:

1. Decode RGB upload to PIL.
2. Run the A3 Ultralytics YOLO11s detector (`checkpoints/indoor_detector_a3_yolo11s_800px_best.pt`, configured `imgsz: 800`).
3. Run Ultralytics YOLO11n-Seg (`checkpoints/yolo11n-seg.pt`, `imgsz: 640`). Despite a `SceneGraph.model_name` string mentioning “FastSAM + SegFormer-B1”, the implementation and config load YOLO-Seg. That model-name string is inaccurate provenance.
4. Load the local Depth Anything V2 Metric Indoor Small checkpoint through Transformers `AutoImageProcessor` and `AutoModelForDepthEstimation` (`checkpoints/depth_anything_v2_metric_indoor_small`). The processor’s predicted map is bicubic-resized to the original input width/height. `raw_depth` is retained in memory as float32; the returned display map is separately normalized.
5. The model creates points with `X=(u-cx)Z/fx`, `Y=(v-cy)Z/fy`, `Z=depth`, samples RGB at the same pixel indices, and randomly downsamples to at most 200,000 points.
6. Scene fusion assigns depth to detected objects, estimates room layout/floor plane, and derives relationships. Multi-view uses a separate SIFT/FLANN feature alignment and Umeyama-style rigid/similarity alignment path; it is classified experimental.
7. `services/job_service.py` exports a full ASCII PLY and a GLB point sample capped at 40,000 plus object boxes.

Depth Anything V2 Metric Indoor Small is a **metric-trained monocular estimator** and its raw output is interpreted as meters by this code. That does not make its result camera-calibrated ground truth. Accuracy and absolute scale must be measured on held-out data and real rooms. By contrast, `scene3d` configured with Depth Anything V2 Small is relative/inverse depth and must never be labeled meters.

### Hunyuan (`hunyuan_world_mirror_1b`)

The active root adapter loads Tencent HunyuanWorld-Mirror 1.0, a ViT-L/14/DINOv2-backed model with camera, depth, point-map, and normal heads. The configured checkpoint is `results/final_execution/checkpoints/exp_05_expanded_lora_rank8_final.ckpt`; config contains an expected SHA, but the adapter does not enforce it and job rows do not record the loaded checkpoint digest. The immutable model registry and version manifests point at different checkpoint paths, so registry/version provenance is not a reliable record of what the job loaded.

The input is center-cropped to a square and resized to 518×518. The adapter extracts predicted depth, points, normals, and RGB. It transforms points and normals with `diag(1,-1,-1)` (a length-preserving 180° X-axis rotation from the declared Hunyuan/OpenGL basis to the RoomAI/OpenCV basis), then filters points using depth `0.1 < d < 25`. It does not use `pts3d_conf`, does not compare point-map Z to the depth head, and does not consume the model’s predicted camera matrices/intrinsics. It labels the raw depth `metric` and fills camera intrinsics with `fx=fy=1000, cx=cy=0.5`.

The local Hunyuan depth-semantics audit says the model predicts canonical scene-normalized values near unit scale; it is not physically metric without calibration. The P0 job’s depth summary around 0.9–1.02 and its PLY XYZ bounds near 0.1 are incompatible representations. The source transform only changes signs and preserves lengths; it cannot itself compress a room by an order of magnitude. Therefore the saved collapse is already present in the model point map or an earlier tensor/checkpoint/preprocessing issue. The job did not preserve raw `pts3d`, confidence, depth, camera prediction, or checkpoint hash, so a narrower model-side cause cannot be proven from that run.

The P0 Hunyuan PLY is finite but near-degenerate; the GLB is a `POINTS` primitive containing positions only. The adapter’s confidence threshold argument is unused. `supports_metric_depth` and `supports_mesh_glb` capability flags overstate the actual outputs. The current adapter enables Gaussian splatting only for a Stage 2 path and does not serialize a Gaussian scene as a production asset. Model/version manifests identify Hunyuan weights as non-commercial research; product use requires rights review in addition to technical acceptance.

## Root causes and confirmed defects

| Finding | Evidence / root cause | Effect |
|---|---|---|
| P0 Hunyuan point cloud collapses | PLY bounds are near zero and orthographic views show a narrow diagonal sheet. The applied basis transform is orthonormal and no metric scale is applied. Raw point/depth/confidence tensors were not persisted, so head/checkpoint/preprocessing cause is not separable. | Geometry cannot be used for hero; arbitrary scaling would conceal the defect. |
| Hunyuan depth falsely labeled metric | Raw canonical scene-normalized depth is copied into `depth_min_m/max_m`; `depth_mode="metric"` is hard-coded. | Metadata and measurements are invalid. |
| No raw depth or mask in job bundle | Job serializer explicitly excludes `point_cloud`, `depth_map`, and `heatmap_image`; it writes no `.npy/.npz` depth or validity mask. Hunyuan results expose a transient depth map but no persistent depth artifact. | Alignment, validity, reprojection, and model-head diagnosis cannot be reproduced after the job. |
| Camera metadata is ambiguous/guessed | Defaults are `fx=fy=1000`, `cx=cy=0.5`; code heuristically treats principal points ≤1 as normalized fractions and multiplies by width/height. There is no capture calibration or recorded crop/resize transform. | Wrong lateral dimensions and untraceable reprojection; metadata value convention is unclear. |
| Observed composite frame uses wrong focal lengths | On Hypersim test frame, actual K from the dataset manifest is fx=447.896, fy=598.134, cx=cy=259 px; pipeline unprojection uses fx=fy=1000. | X/Y dimensions are compressed relative to the calibrated frame. The point cloud’s metric width/height and room layout are not trustworthy. |
| RGB preservation is incomplete | Job service saves decoded images as JPEG quality 95 in the job input directory. The upload may remain separately, but the job bundle does not preserve original bytes/hash plus a full decode/preprocessing transform. | Exact input-to-output reproduction and image-depth alignment provenance are weakened. |
| GLB drops point colors | `utils/export_3d.export_scene_glb` writes `POSITION` only for the point primitive; it does not write `COLOR_0`. | GLB loses the observed RGB texture signal present in PLY. |
| PLY and GLB represent different subsets | PLY includes up to 200,000 points; GLB randomly samples up to 40,000 and adds object boxes. Subsampling is nondeterministic. | Exports are not exact point/color equivalents and cannot be hash-reproduced. |
| Detected boxes can look like reconstructed geometry | GLB exports object extents as triangle box meshes; they are detector-derived bounds, not actual furniture surfaces. | Viewer may imply unsupported surfaces/shape. |
| Validators check containers, not geometric truth | PLY validation checks header/row count; GLB validation checks magic/version/length/JSON parsing. Neither validates finite vertices, transforms, point colors, units, collapsed geometry, alignment, nor cross-export consistency. Job service records validation flags but does not fail the job on false values. | Broken geometry can be marked `COMPLETED`. |
| Duplicate pipeline stacks | `api.main` is documented deployment; `app.api.main` has separate GPU worker, model/inference/export stack. The latter hardcodes metric mode and similar intrinsics defaults. | Fixes/tests can diverge between launch targets. |
| Composite model description is inaccurate | Scene graph string says FastSAM + SegFormer-B1 while config/implementation use YOLO11n-Seg. | Provenance is false even when inference succeeds. |

## Controlled pipeline test

The `roomai-hunyuan` environment was used (Python 3.10.20, PyTorch 2.11.0+cu128, Transformers 4.49.0, huggingface-hub 0.36.2, Ultralytics 8.4.66; RTX 4060 Laptop GPU). The base environment’s Transformers import failed because its `huggingface-hub` was incompatible; the project environment works.

Input: `research/3d_reconstruction/data/ground_truth/standardized/hypersim/test/ai_013_010/frame_0000_rgb.png`, 518×518, V-Ray synthetic render, paired metric planar-Z depth and camera calibration. No source transform was applied. This is a held-out synthetic diagnostic, not a photograph, product proof, or approved exhibit.

| Measure | Result |
|---|---:|
| End-to-end composite adapter time | 11.731 s |
| Predicted depth | 518×518 float32, 100% finite and within current validity range |
| Unscaled depth AbsRel | 0.1379 |
| Unscaled depth RMSE | 0.7211 m |
| GT/predicted median-depth ratio | 0.8561 (14.4% scale discrepancy; no scale fit applied) |
| Point cloud | 200,000 finite RGB points; no collapse on this frame |
| PLY | 6,589,828 bytes; structural validator pass; 200,000 vertices |
| GLB | 484,376 bytes; structural validator pass; 6 meshes (point set plus five boxes); point attributes only `POSITION` |
| Intrinsics | pipeline fx/fy=1000; frame GT fx=447.896/fy=598.134 pixels |

Visual review shows the rendered depth edges broadly follow the RGB structure and the cloud is room-like, not degenerate. The view is still a single depth surface with occlusion holes/shape ambiguity, not a complete room mesh. This one synthetic sample demonstrates that the default route can produce useful-looking 2.5D geometry; it does not establish accuracy on real photographs or qualify the exhibit. The 14.4% scale discrepancy and focal mismatch fail the proposed calibrated-geometry gate.

QA-only outputs and raw predictions are under `asset-production/roomai/01-review/pipeline-test-ai_013_010-frame_0000-run2/`. They are diagnostics; do not promote them to production. The first attempted run was interrupted during slow model initialization and left an empty review directory only.

## Can the current pipeline meet the website requirement?

**Not reliably as-is.** The default composite pipeline is the strongest remediation starting point: it already emits a full-resolution depth estimate and a colored camera-view point set, with a successful single-frame controlled run. But it has guessed camera intrinsics, no durable depth/mask/provenance bundle, a weak validator, lossy job input re-encoding, and a colorless/downsampled GLB. It can support the creative transformation as an honestly described visible-side point representation after remediation and acceptance testing.

The Hunyuan branch is not a production fallback today: the known P0 result is collapsed, raw diagnostics are absent, scale is mislabeled, camera predictions/confidence are ignored, checkpoint identity is not captured, and the listed weight license is non-commercial research. Do not “repair” that PLY by scaling, smoothing, or adding guessed geometry.

Neither pipeline can infer hidden room surfaces from one RGB image without additional views or priors. A valid output should be labelled `single_view_depth_point_cloud` / `partial_2_5d`; a full surface mesh is optional and must be omitted unless actual depth-surface triangulation passes geometry checks. Object boxes stay semantic bounds, not furniture meshes.

## Minimum proposed backend changes — approval required before implementation

1. Establish one canonical deployed pipeline/SceneGraph schema and make the inactive `app.api` stack explicitly non-production or bring it under the same tests later.
2. Preserve original upload bytes and decoded RGB hash; record crop/resize/color/orientation transforms and the selected model/checkpoint digest.
3. Preserve raw float32 depth, an explicit validity/confidence mask, raw-depth semantics, unit/scale/calibration status, and exact camera K with pixel units. Never call canonical/relative depth “meters”.
4. Centralize camera-frame math and transform K for every image operation. Use capture K when available; otherwise store estimated intrinsics and downgrade geometry/scale status.
5. For Hunyuan, retain raw depth/point/confidence/camera outputs; assert tensor layouts, compare point-map optical-axis Z with its depth head, use confidence, and mark native scale normalized until calibrated. Gate/disable its product mode until output/weight license passes.
6. Preserve existing `roomai_composite_v1` flow, but make deterministic point sampling and generate one canonical colored point array used by PLY and GLB. Add `COLOR_0` in GLB. Store detector boxes as clearly labelled bounds, not surface meshes.
7. Replace container-only acceptance with fail-closed data, calibration, collapse, bounds, export-parity, and visual gates. A validation failure must not become a completed production job.
8. Write a versioned exhibit manifest with input/output hashes, model/config/code versions, license, transformations, coordinate frame, depth scale, intrinsics, validation metrics, and asset links.

No backend implementation has started. This audit and the companion spec/plan are review documents only. Approval is requested for the minimum changes above before any code is modified.
