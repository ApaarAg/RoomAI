# ROOMAI Phase 4 P0 — Asset Validation Report

**Audit date:** 2026-11-11  
**Disposition:** **BLOCKED — no production exhibit selected.**  
**Scope:** read-only inspection of existing RGB, depth, PLY, GLB, job records, source manifests and relevant pipeline metadata. Candidate visualizations were written only to the review folder. No source file was modified, and no web app or FastAPI code was changed.

## Executive result

The existing artifacts contain traceable RGB-to-job pairings, and several PLY/GLB files pass structural checks. None of the candidate bundles is suitable for the P0 hero:

- The photo-like upload `upl_31972c1ea491630e` is paired by the jobs database to Hunyuan jobs, but its point cloud collapses into a narrow, nearly planar diagonal field. The required dense aligned depth map was not persisted. The job's `metric` label is contradicted by the local Hunyuan depth-semantics audit, which identifies canonical scene-normalized output. Public display rights are not recorded.
- The composite jobs pair with `artifacts/sample_room.jpg`, an 800×600 flat-color schematic rather than an interior photograph. The dataset manifest calls it **Internal Proprietary**. Its metric-depth heatmap is only a visualization; a complete depth raster is not recoverable from the saved scene graph, and camera intrinsics conflict across records.
- One remaining uploaded image has no completed reconstruction job. Test/benchmark reconstructions either map to an ADE validation set or multi-view ScanNet research assets and do not meet the single-view, cleared-interior requirement.

Therefore no source RGB/depth was promoted into runtime storage, no desktop/mobile LOD buffers were generated, and no GLB optimization was attempted. Scaling, smoothing, recoloring, or synthesizing missing depth would hide the actual output defects and is not an acceptable asset fix.

## Candidate-to-reconstruction map

| RGB candidate | Verified pairing | Existing output contents | Disposition |
|---|---|---|---|
| `artifacts/uploads/upl_31972c1ea491630e.png` (518×518 living-room image) | Job database records `upload_ids: [upl_31972c1ea491630e]` for completed `hunyuan_world_mirror_1b` jobs, including `job_1794406427_0fb76456`. Archived `view_00.jpg` is the same 518×518 content after JPEG conversion (mean absolute RGB delta ≈ 1.20/0.93/1.35). | PLY: 40,000 colored points. GLB: 40,000 `POINTS`, positions only. No object/surface mesh. Summary says 0.891–1.022 “m”; research audit says Hunyuan output is normalized and not metric without a separate scale. | **Not selected:** visibly collapsed point field, no saved aligned depth map, inaccurate metric metadata, no public-rights record. |
| `artifacts/uploads/upl_a51d932cc9a0d3c3.jpg` = `artifacts/sample_room.jpg` (800×600 schematic) | Exact SHA-256 match between the uploaded image and `sample_room.jpg`; the jobs DB links that upload ID to completed `roomai_composite_v1` jobs, including `job_1794406363_3c3f098c`. | PLY: 200,000 colored points. GLB: 40,000 uncolored `POINTS` plus two box meshes. Metric depth values exist as 200,000 point samples, but no per-sample image UV mapping. Saved `depth_map` field is a truncated NumPy representation string; the PNG is a color heatmap. | **Not selected:** source is a diagram rather than the photographic interior in the storyboard; depth/image alignment and intrinsics are unresolved; manifest says Internal Proprietary. |
| `artifacts/uploads/upl_33e89c330cc3079a.jpg` (320×240) | No matching completed job/upload ID found in local job records. | No paired depth, PLY, or GLB found. | **Excluded:** unmatched source and no usable reconstruction. |
| `artifacts/phase_7_1_e2e/scene_1..5` | Validation report identifies `ADE_val_00000001.jpg` … `ADE_val_00000005.jpg` in the local ADE validation dataset. | PLY/GLB validation outputs; scenes include non-room/outdoor content. | **Excluded:** evaluation samples, not the cleared RoomAI interior exhibit; dataset display rights not approved. |
| `artifacts/phase_7_3_multiview/multiview_fused_scene.*` | Multiview registration telemetry; source is multiple ScanNet views, not one RGB image. | Fused PLY and GLB. | **Excluded:** wrong single-view story and ScanNet research-use restrictions. |

The `artifacts/final_backend_verification/` files are from the `benchmark_final_backend.py` pipeline run on `artifacts/sample_room.jpg`, not the Hunyuan upload. The benchmark identifies A3 + FastSAM + SegFormer + Depth Anything V2 Metric. This is a separate export/run from the database job PLY (40,000 versus 200,000 vertices); they must not be silently mixed as one immutable exhibit bundle.

The separate benchmark export is `artifacts/final_backend_verification/reconstructed_pointcloud.ply` (40,000 colored points; SHA-256 `a4fab8649453630114bf98d71c9da1e3152dfa7b61261a873dc92ce7f63c4bf4`) and `reconstructed_scene.glb` (40,000 point positions plus two box meshes, no point-color attribute; SHA-256 `0c95a0617b77eb03c26c88c9a266799b6ffadb39845af185f12f13c4be2e2550`). Its `metric_depth_heatmap.png` (SHA-256 `3c9685b996f971be9b39ccb4c7577210e15b3db04da6f856e2fa7ed3b3e06cd2`) is a visualization only. The full raw depth raster was not preserved in the scene graph. These benchmark files remain distinct from `job_1794406363_3c3f098c`'s 200,000-point PLY and 40,000-point GLB.

Across the local job database, the two uploaded candidate IDs account for 50 job rows: 28 composite rows (22 completed, 6 failed) and 20 Hunyuan rows (15 completed, 5 failed), plus two test records. Completed composite runs contain 22 distinct PLY and 22 distinct GLB hashes; completed Hunyuan rows repeat one PLY/GLB hash pair despite inconsistent depth summaries. Repeated runs do not create additional independent exhibits or resolve the missing provenance/depth/rights gates.

## Technical audit findings

### Candidate A — Hunyuan output for the living-room image

- **Provenance:** upload ID, completed job, archived input path, model ID, and output paths are present in the local SQLite job database. The upload-to-job relationship is established. The historical checkpoint file hash is absent, so exact weight revision at inference time is unknown.
- **PLY:** ASCII PLY 1.0; declared and actual row count both 40,000; XYZ and uchar RGB fields; all coordinate values finite; all color channels within 0–255. Bounds are X `[-0.0014, 0.0268]`, Y `[0.0164, 0.0936]`, Z `[-0.0269, 0.0151]` in stored coordinates. The point distribution has one dominant PCA direction and looks like a narrow diagonal sheet rather than the visible room.
- **GLB:** valid GLB 2.0 header/chunks; one point primitive (`mode=POINTS`) with 40,000 `POSITION` values; no `COLOR_0`, no surfaces, and no object meshes. It is structurally valid, but it is not a usable colored room asset.
- **Depth and scale:** the job `scene_graph.json` has a depth range and a `depth_mode=metric` label but no dense depth map. The inference code filters/exports point coordinates while not saving the depth raster. The current local source audit, [`PHASE7_1_WORLD_MIRROR_DEPTH_SEMANTICS.md`](</F:/RoomAI/research/3d_reconstruction/reports/PHASE7_1_WORLD_MIRROR_DEPTH_SEMANTICS.md>), classifies pretrained HunyuanWorld-Mirror depth/points as canonical scene-normalized, not direct meters. The reconstruction wrapper's stored intrinsics (`fx/fy=1000`, `cx/cy=0.5` for 518×518) mix pixel and normalized conventions. Do not use these metadata for metric measurements or a reprojection.
- **Visual inspection:** [Hunyuan point-cloud orthographic audit](/F:/RoomAI/asset-production/roomai/01-review/exhibit-01/hunyuan_pointcloud_orthographic_audit.png) shows the output is not a recognizable room volume. This image is an audit visualization, not a product asset.

### Candidate B — composite output for the schematic source

- **Provenance:** `sample_room.jpg` and `upl_a51d932cc9a0d3c3.jpg` have the same SHA-256. The completed job database row points to that upload ID and archived input. Source provenance is established, but public display permission is not.
- **PLY:** ASCII PLY 1.0; 200,000 declared and actual vertices; XYZ plus RGB; all finite and colors in range. Bounds are X `[-1.2782, 1.2324]`, Y `[-0.8651, 0.7995]`, Z `[2.3322, 3.3866]`. The cloud forms surfaces around the flat rectangular schematic, as shown in [composite point-cloud audit](/F:/RoomAI/asset-production/roomai/01-review/exhibit-01/composite_pointcloud_orthographic_audit.png).
- **GLB:** valid GLB 2.0; 40,000 point vertices with `POSITION` only; no point colors. Two extra 8-vertex/36-index triangle box meshes encode detected boxes, not reconstructed furniture surfaces. Do not represent them as actual chair/appliance geometry.
- **Depth:** the 800×600 `metric_depth_heatmap.png` is RGB false-color output, not numeric depth. Scene graph JSON's top-level `depth_map` is a 479-character NumPy repr containing `...`, not the full raster. `point_cloud.depth`, XYZ and RGB each have 200,000 entries, but the saved data does not carry per-point pixel UVs. The scene graph stores `fx/fy=1000` and `cx/cy=0.5`, while the dataset manifest lists `cx=400`, `cy=300`. This conflict prevents a trustworthy aligned depth-texture export from the saved artifacts.
- **Rights/creative fit:** the local dataset manifest marks this image source `Internal Proprietary`. The image itself is a flat-color block diagram, not a photograph of a room. It cannot be the opening photograph without changing the approved creative concept.

### Depth outputs found

1. **Hunyuan job:** only depth mode/range metadata are saved with the job. No per-pixel raw depth file is present in its job directory.
2. **Composite benchmark:** a colorized heatmap and incomplete/truncated `depth_map` representation are saved. The point-cloud depth samples lack the pixel mapping and the camera intrinsics are inconsistent.
3. **Other depth visualizations:** `artifacts/depth_visualizations/*heatmap.png` and `artifacts/frontend_integration/ui_relative_depth_heatmap.png` are rendered visualizations. Their source-run pairing is not established for the living-room upload, and they are not numeric depth textures.
4. **Research depth labels:** metric depth arrays under research datasets are training/evaluation data; they are not paired to the RoomAI upload/reconstruction candidates and cannot be substituted into this exhibit.

## Validation performed

| Check | Result | Evidence / limitation |
|---|---|---|
| Job DB upload ID → archived input → GLB/PLY paths | **PASS for two candidate pairs** | Read-only SQLite query. The two user-upload IDs map to separate model/job families; 50 job rows are mostly repeated runs of those same two IDs, not 50 independent scenes. |
| RGB file properties and metadata | **PASS for file integrity; RIGHTS BLOCKED** | Pillow read/size/EXIF checks. `upl_319...png` is RGB 518×518 with no EXIF rights metadata; `sample_room.jpg` is 800×600 and hash-identical to `upl_a51...jpg`. Rights cannot be inferred from a local file. |
| PLY structural/data checks | **PASS for candidate files** | Header, declared/actual rows, XYZ/RGB shape, finite coordinates, color bounds, bounding boxes and covariance/PCA calculated with Python/NumPy. Structural validity does not imply visual suitability. |
| GLB container/accessor checks | **PASS for candidate files** | Read-only `scripts/inspect_3d_artifacts.py` and independent GLB header/chunk/accessor parsing. `POSITION` counts/bounds are valid; `COLOR_0` absence and primitive contents are recorded. |
| Image/depth alignment | **BLOCKED** | No complete raw, registered depth raster for either eligible-looking source. Color heatmaps cannot replace numeric depth; point samples lack saved UV mapping. |
| Coordinate system / metric scale | **BLOCKED for Hunyuan; CONFLICT for composite metadata** | Hunyuan scale semantics contradict the job's metric label. Composite graph and dataset manifest principal points disagree. No guessed coordinate repair was applied. |
| Geometry validity for hero | **FAIL for Hunyuan; visual mismatch for composite** | Hunyuan cloud is near-degenerate. Composite output follows a schematic, with no reconstructed surfaces; two GLB boxes are detections. |
| Browser visual QA | **PARTIAL / NOT PASSED** | Orthographic previews generated from original PLY points and colors for inspection only. No point LOD, GLB optimization, texture conversion, or browser runtime asset was created because no exhibit passed selection. |
| Rights | **BLOCKED** | Hunyuan image has no license record. Composite image is labeled Internal Proprietary. Public-use permission requires owner confirmation. |

The old `scripts/validate_3d_exports.py` was **not** run: it performs a new model inference and writes to `artifacts/export_validation/`, which would change existing artifacts and is unnecessary for this read-only candidate audit.

## Production-ready assets

**None.** The existing RGB, PLY and GLB files are verified candidate evidence, not approved production assets. No approved source RGB/depth pair, desktop/mobile point cloud, or optimized browser GLB exists as a result of this P0 batch. The three orthographic PNGs under `asset-production/roomai/01-review/exhibit-01/` are QA-only previews and must not ship.

## Remaining blockers

1. **Public image rights:** confirm that `upl_31972c1ea491630e.png` is cleared for public website use, or provide a cleared replacement. `sample_room.jpg` is marked Internal Proprietary and is not a suitable photographic room regardless.
2. **Usable output for a real interior:** the Hunyuan reconstruction associated with the photo does not pass visual geometry review. It needs a corrected/verified model output; arbitrary uniform scaling does not repair the flattened structure.
3. **Aligned depth:** provide or regenerate an exact full-resolution depth raster for the chosen image, with pixel registration, validity mask, units/scale semantics, camera intrinsics and coordinate convention saved alongside the reconstruction. For Hunyuan, explicitly label normalized depth unless a defensible metric calibration is available.
4. **Metadata integrity:** current wrappers write misleading Hunyuan metric values and inconsistent intrinsics. Fixing the inference/export data contract is backend/model-pipeline work and remains outside this asset-only batch.
5. **Checkpoint provenance:** the job records model ID but not the historical checkpoint hash. Record the checkpoint hash for any replacement run before approving its output.

No further P0 asset creation should proceed until a cleared photographic source and a compatible reconstruction/depth package pass these gates. This report does not authorize full website implementation.
