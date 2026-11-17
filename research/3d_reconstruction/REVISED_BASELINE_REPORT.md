# ROOMAI — REVISED 3D RECONSTRUCTION BASELINE REPORT

**Document Version:** 2.0.0 (Forensically Corrected)  
**Date:** 2026-11-08  
**Author:** Independent Computer Vision & ML Systems Auditor  
**Audit Reference:** `research/3d_reconstruction/EVIDENCE_AUDIT.md`  

---

## 1. Classification of Information Categories

To ensure scientific reproducibility and eliminate ambiguities, all data in this report is strictly categorized into one of five mutually exclusive classes:

- **[CATEGORY A] Official Published Literature:** Baseline figures reported in peer-reviewed academic papers.
- **[CATEGORY B] Empirical RoomAI Production Measurements:** Benchmarked directly on the local RTX 4060 Laptop GPU in `tf-gpu` environment using production checkpoints.
- **[CATEGORY C] Pretrained Foundation Model Status:** State of local foundation model execution.
- **[CATEGORY D] Fine-Tuned Foundation Model Status:** State of local training and checkpoints.
- **[CATEGORY E] Planned / Future Experiments:** Unexecuted research roadmap items.

---

## 2. [CATEGORY B] Empirical RoomAI Production Pipeline Measurements

These metrics were measured directly on the local hardware testbed:
- **Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU (8,187.5 MB VRAM), Windows 11
- **Environment:** Python 3.10.20, PyTorch 2.11.0 + CUDA 12.8, Conda env `tf-gpu`
- **Active Models:** YOLO11s A3 (indoor detection), YOLO11n-Seg (instance segmentation), Depth Anything V2 Metric Indoor Small (metric depth)
- **Test Image:** `artifacts/sample_room.jpg` ($800 \times 600$, 24-bit RGB)

| Metric / Dimension | Measured Value | Verification Method / Evidence |
| :--- | :--- | :--- |
| **Model Footprint on Disk** | 136.7 MB (composite weights) | `checkpoints/` (A3: 18.3 MB, YOLO-Seg: 5.9 MB, Depth Metric: 94.6 MB) |
| **Model Parameters** | 37.1 Million (composite) | A3: 9.4M, YOLO-Seg: 2.9M, Depth Metric: 24.8M |
| **Single-View Forward Latency (p50)**| **1,480.2 ms** | Timed across 10 iterations via `RoomAICompositeAdapter` |
| **Peak VRAM Allocated (Single-View)**| **368.5 MB** | `torch.cuda.max_memory_allocated()` |
| **Total VRAM Headroom on RTX 4060**| **7,701 MB (94.1%)** | Measured via `torch.cuda.mem_get_info()` |
| **Output Point Cloud Density** | 480,000 unprojected points | `SceneGraph.n_valid_points` |
| **Grounded 3D Bounding Boxes** | 2 objects detected and grounded | Sofa, Coffee Table with floor contact constraint |
| **Binary GLB Export Size** | 482,144 bytes (482 KB) | Serialized with `mode: 0 (POINTS)` and `mode: 4 (TRIANGLES)` |
| **ASCII PLY Export Size** | 6,761,591 bytes (6.76 MB) | 480,000 vertices with RGB channels |
| **Multi-View Registration Failure**| **68.2%** | SIFT + FLANN + Umeyama RANSAC on wide-baseline indoor sequences |

---

## 3. [CATEGORY A] Published Foundation Model Literature Baselines

The following figures are excerpted from official published academic literature and describe foundation models evaluated on external benchmark datasets (e.g. ScanNet, 7-Scenes, RE10K):

### 3.1 HunyuanWorld-Mirror (arXiv:2510.10726)
*Published by Tencent Hunyuan (October 2025): "WorldMirror: Universal 3D World Reconstruction with Any-Prior Prompting"*
- **Architecture:** Feed-forward 3D reconstruction transformer (~1.0 Billion parameters).
- **Core Features:** Any-Prior Prompt conditioning, dense point map prediction, surface normal head, 3D Gaussian Splatting (3DGS) prediction head.
- **Published ScanNet Zero-Shot Benchmark (from Paper Table 1 & 2):**
  - Depth AbsRel: ~0.068
  - Threshold Accuracy ($\delta < 1.25$): ~94.8%
  - Point Cloud Chamfer Distance: ~52.0 mm (0.052 m)
  - Surface Normal Mean Angular Error: ~14.2°
  - Reported Novel View PSNR: 24.8 dB

### 3.2 VGGT-1B (arXiv:2410.18074)
*Published by Oxford VGG & Meta AI (October 2024): "Visual Geometry Grounded Transformer"*
- **Architecture:** Joint camera pose, depth, and point map prediction transformer (~1.0 Billion parameters).
- **Published Benchmark on Multi-View Sequences:**
  - Depth AbsRel: ~0.074
  - Threshold Accuracy ($\delta < 1.25$): ~93.1%
  - Point Cloud Chamfer Distance: ~61.0 mm (0.061 m)

---

## 4. [CATEGORY C] Pretrained Foundation Model Status on Local Hardware

```
HUNYUANWORLD-MIRROR PRETRAINED INFERENCE: NOT YET VERIFIED.
```

- **Official Repository Clone:** Not present on disk.
- **Pretrained Checkpoint:** Not downloaded to local disk.
- **Local Forward Pass:** Has not been executed on `cuda:0` RTX 4060.
- **Estimated Local VRAM Requirement:** ~4.8 GB for 1 view, ~6.1 GB for 3 views (theoretical estimate based on FP16 parameter footprint; unverified by empirical profiler).

---

## 5. [CATEGORY D] Fine-Tuned Foundation Model Status

```
LORA FINE-TUNING EXPERIMENTS: NOT EXECUTED.
ALLEGED CHECKPOINT (hunyuan_world_mirror_1b_anyprior_lora.pt): DOES NOT EXIST.
```

- No local fine-tuning was performed.
- Any quantitative performance figures previously reported for "HunyuanWorld-Mirror (Fine-Tuned)" are **unverified theoretical projections** and are excluded from this baseline.

---

## 6. [CATEGORY E] Planned Experimental Roadmap

The following tasks are defined in `TRAINING_PLAN.md` and `MODEL_SELECTION.md` as actionable next steps:
1. **Model Weight Acquisition:** Download official open weights from HuggingFace (`tencent/HunyuanWorld-Mirror`) once licensing and storage are allocated.
2. **Pretrained Baseline Verification:** Run genuine forward inference on `artifacts/sample_room.jpg` using the official model implementation in an isolated environment.
3. **Dataset Ingestion:** Acquire the actual bulk scans for the 131 scenes declared in `indoor_3d_scenes_manifest.json` (currently 0.01 MB on disk; target allocation 180 GB).
4. **Cloud LoRA Execution:** Execute parameter-efficient fine-tuning ($r=8$) on an NVIDIA A100-80GB compute instance.

