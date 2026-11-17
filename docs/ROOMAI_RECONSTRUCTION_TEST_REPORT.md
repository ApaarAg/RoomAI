# ROOMAI Reconstruction Pipeline — Phase 4.2 Test Report

**Run date:** 2026-11-12  
**Python environment:** `roomai-hunyuan` (Python 3.10.20, Torch 2.11.0+cu128, RTX 4060 Laptop GPU)  
**Overall:** Implementation checks pass; the controlled synthetic benchmark remains review-only. A real camera image and independent real-room reference remain outstanding.

## Automated results

Commands:

```powershell
python -m py_compile utils/reconstruction_contract.py utils/export_3d.py services/job_service.py services/scene_fusion_service.py services/hunyuan_mirror_service.py models/adapters/hunyuan_mirror_adapter.py models/adapters/roomai_composite.py api/routers/jobs.py api/schemas.py tests/test_reconstruction_contract.py scripts/run_phase4_2_hypersim_smoke.py tests/test_model_adapters.py
node --check scripts/validate_glb_khronos.mjs
python -m pytest tests/test_reconstruction_contract.py tests/test_job_lifecycle.py tests/test_model_adapters.py tests/test_scene_fusion_integration.py -q
```

Results:

- Python compilation: passed.
- Node syntax check: passed.
- Focused tests: **22 passed** in approximately 16.0 seconds on the final run.
- Covered original-byte/lossless-decoded-RGB preservation, additive API camera/acceptance fields, known-K unprojection/reprojection, resize/crop/pad K transforms, invalid-depth filtering, mask alignment, deterministic sampling, collapse detection, RGB correspondence, PLY/GLB color attributes and parity, deterministic GLB bytes, manifest creation, evidence-gated calibration, Hunyuan raw-output retention/rejection, independent job/asset status, rejected export blocking, worker failure state, existing job-store lifecycle, adapter metadata, and scene-fusion integration.

## Controlled held-out benchmark

Reproduce with:

```powershell
python scripts/run_phase4_2_hypersim_smoke.py
```

The runner refuses to overwrite its output directory. It uses the held-out Hypersim test frame `ai_013_010/frame_0000`, its original RGB, depth reference, and camera calibration `.npz`. The dataset audit records the source as CC BY-SA 3.0. It is a V-Ray synthetic render, **not a camera photograph and not a product exhibit**.

Final run directory:

```text
asset-production/roomai/01-review/phase4_2-hypersim-ai_013_010-frame_0000-run08/
```

Results from `pipeline_test_report.json`:

| Check | Result |
|---|---:|
| Adapter and artifact writer elapsed | 16.2 s |
| Prediction shape | 518 × 518 float32 |
| Finite / valid depth | 100% / 100% |
| Unscaled AbsRel | 0.1378659 |
| Unscaled RMSE | 0.7211257 m |
| Ground truth / prediction median ratio | 0.8560577 |
| Scale fitting | None |
| Canonical colored point count | 200,000 |
| Point bounds spans | 6.3884 × 4.0443 × 4.6701 (model-estimated meters) |
| Collapsed geometry check | Passed |
| XYZ→pixel max reprojection error | `3.1e-5` px |
| PLY | Valid, 200,000 colored vertices |
| Desktop GLB | Valid internally, 40,000 colored points, 600,952 bytes |
| Mobile GLB | Valid internally, 15,000 colored points, 225,948 bytes |
| PLY / desktop / mobile parity | Passed |
| Automated asset disposition | `REVIEW_REQUIRED` |

The median scale ratio is outside the provisional `[0.90, 1.10]` acceptance interval. The point cloud’s visible room surfaces are recognizable in front/top/side QA views, while single-view occlusion holes and incomplete surfaces remain. No mesh or hidden surfaces were created.

Visual inspection files:

- `asset-production/roomai/01-review/phase4_2-hypersim-ai_013_010-frame_0000-run08/depth_comparison_qa.png`
- `asset-production/roomai/01-review/phase4_2-hypersim-ai_013_010-frame_0000-run08/pointcloud_orthographic_qa.png`

Machine-readable records:

- `pipeline_test_report.json`
- `exhibit_manifest.json`
- `validation_report.json`

## Khronos validator

The worker attempted `scripts/validate_glb_khronos.mjs`. Node is available, but the official `gltf-validator` package is not installed. The report records this as `unavailable`; internal GLB header, chunk, buffer view, accessor, attribute, and point-parity checks passed. Run the Khronos validator in the release environment before accepting/releasing a GLB.

## Existing API test harness

The command below stalled on the first liveness test before it returned a response or assertion:

```powershell
python -m pytest tests/test_api_endpoints.py -vv --tb=short
```

The test is blocked at `TestClient.get('/health')`; interrupting the run produced no assertion output. This route does not invoke reconstruction, and the focused API schema/job tests and backend worker tests pass. Treat HTTP-level `TestClient` verification as unresolved environment/test-harness work, not as passed.

## Not tested or not accepted

- No rights-cleared real camera photograph with independent physical distance references was available. The older living-room upload still has unverified rights.
- Real-world metric accuracy, sensor calibration quality, and production visual acceptance remain unverified.
- Hunyuan production use remains blocked by its normalized scale/geometry and unresolved commercial license. The available candidate weights are marked staged, not approved.
- The duplicate `app/api` implementation was not changed and is not covered by this contract.
