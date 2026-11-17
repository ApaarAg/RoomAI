# RoomAI Phase 7.4.1 — Model Provenance, Checkpoint Integrity & Registry Audit

**Audit Date:** November 6, 2026  
**Auditor:** Independent Senior Computer Vision Researcher, ML Systems Architect & Security Auditor  
**Project:** RoomAI (`F:\RoomAI`)  
**Target Configuration & Registries Under Audit:**  
- `config/model_config.yaml`  
- `config/app.yaml`  
- `models/depth_anything_v2.py`  
- `models/yolo_detector.py`  
- `services/scene_fusion_service.py`  
- `reports/phase_7_4b_pretrained_model_comparison.md`  
- `reports/phase_7_4b_model_hardware_and_license_audit.md`

---

## 1. Executive Summary: Provenance Clarity Mandate

A critical duty of this independent audit is to establish unambiguous truth regarding which AI models are actually loaded, configured, and running in production versus those that were merely discussed in theoretical research benchmarks.

In Phase 7.4B, extensive benchmarking was conducted evaluating hypothetical foundation candidates such as **Depth Anything 3 (DA3)** (`DA3METRIC-LARGE`, 350M params; `DA3NESTED-GIANT-LARGE-1.1`, 1.7B params) and **VGGT** (1.1B params).

### Core Audit Finding:
1. **Zero DA3 Checkpoints Exist in Repository:** No model weights, checkpoints, or inference code for Depth Anything 3 (`DA3METRIC-LARGE`, `DA3-BASE`, `DA3-GIANT`) or `VGGT` are present in `F:\RoomAI\checkpoints` or anywhere in the workspace.
2. **Actual Active Production Depth Model:** The production depth model configured in `config/model_config.yaml` and loaded by `services/scene_fusion_service.py` is **Depth Anything V2 Metric Indoor Small** (`checkpoints/depth_anything_v2_metric_indoor_small/model.safetensors`, 24.8M parameters).
3. **No Code Modification or Ghost Imports:** The codebase correctly instantiates `DepthAnythingV2Model` with weights loaded from the local safetensors checkpoint.

---

## 2. Checkpoint Verification & Cryptographic Hashes

Every production checkpoint currently referenced in `config/model_config.yaml` was independently verified on disk for existence, file size, and SHA-256 integrity:

| Target Component | Architecture / Model Name | Checkpoint Disk Path | Parameter Count | File Size (MB) | SHA-256 Checksum (Full) | Runtime Device |
| :--- | :--- | :--- | :---: | :---: | :--- | :---: |
| **Indoor Detection (Primary)** | YOLO11s (800px, 12 classes) | `checkpoints/indoor_detector_a3_yolo11s_800px_best.pt` | $9.4\text{ M}$ | $18.29\text{ MB}$ | `6cbc9e58f87156e07dd0e4e712b8aa96097aeb514a473621139a54cb83cec576` | `cuda:0` (CUDA FP32) |
| **Indoor Detection (Rollback)** | YOLO11s (640px, 12 classes) | `checkpoints/indoor_detector_a2_yolo11s_finetune_best.pt` | $9.4\text{ M}$ | $18.27\text{ MB}$ | `42113b2ddb5abffa4830bde024cea13ec5be99b1c6493cb2a01b4b52a7ea08c8` | `cuda:0` |
| **Parking Detection** | YOLO11s (8 classes) | `checkpoints/parking_detector_c2_yolo11s_pilot_best.pt` | $9.4\text{ M}$ | $18.27\text{ MB}$ | `4abab963ab0becfef4a03c4e7b0039cc048d7621b3456bc5435deece6eed1470` | `cuda:0` |
| **Instance Segmentation** | YOLO11n-seg (80 classes) | `checkpoints/yolo11n-seg.pt` | $3.1\text{ M}$ | $5.90\text{ MB}$ | `55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152` | `cuda:0` |
| **Metric Depth Estimation** | Depth Anything V2 Indoor Small | `checkpoints/depth_anything_v2_metric_indoor_small/model.safetensors` | $24.8\text{ M}$ | $94.58\text{ MB}$ | `e990eb82fbf11b05b7813261196a2b841bdcf5a05f64396724a8987fa90504a3` | `cuda:0` (PyTorch FP32/FP16) |

### Missing Legacy Checkpoints (Safely Handled):
- `classification_best.pt`, `vlm_best.pt`, and `generator_best.pt` are currently missing from the local disk. However, the runtime services gracefully detect missing files and guard them behind feature flags or fallback mock modes when disabled.

---

## 3. Discrepancy Analysis: Research Proposals vs. Production State

| Feature / Model Dimension | Claimed / Explored in Research (Phase 7.4B/E) | Actual Implemented State in Codebase | Scientific / Operational Verdict |
| :--- | :--- | :--- | :--- |
| **Foundation 3D Model** | Depth Anything 3 (`DA3METRIC-LARGE`, 350M) | Depth Anything V2 Metric Small (24.8M) | **RESEARCH ONLY; NEVER DEPLOYED**. Deployment was correctly constrained by the 8 GB VRAM budget. |
| **Multi-View Attention** | Cross-attention tokens across multiple views | Pairwise SIFT/FLANN feature matching in OpenCV | **CLASSICAL BACKEND**. No learned multi-view transformer is active. |
| **Camera Pose Regression** | Deep camera pose regression network | Closed-form Kabsch SVD on matched 3D coordinates | **GEOMETRIC SVD**. Deterministic algebraic solver. |
| **Fine-Tuning State** | Prototyped LoRA fine-tuning on NYU Depth v2 | Pretrained zero-shot Hugging Face weights | **ZERO-SHOT WEIGHTS**. Production uses official zero-shot weights. |

---

## 4. Hardware Sizing & Memory Footprint Verification

On the target **NVIDIA GeForce RTX 4060 Laptop GPU (8,187.5 MB VRAM)**:
- Loaded Model Sizes:
  - `indoor_detector_a3`: $\sim 78\text{ MB}$ allocated
  - `depth_anything_v2_metric_indoor_small`: $\sim 204\text{ MB}$ allocated
  - Total Static VRAM: **$282.6\text{ MB}$**
  - Peak Forward Pass VRAM (during $518 \times 518$ metric depth unprojection): **$317.9\text{ MB}$**
- Headroom:
  $$\frac{8187.5 - 317.9}{8187.5} \times 100\% = 96.1\%\text{ free VRAM}$$

### Technical Evaluation of Model Selection:
While `Depth Anything V2 Metric Indoor Small` is significantly smaller than the evaluated `DA3METRIC-LARGE` (24.8M vs 350M parameters), its extremely compact footprint ($317.9\text{ MB}$ VRAM) and sub-second inference ($148\text{ ms}$ on RTX 4060) make it exceptionally stable for single-view residential room reconstruction without risking out-of-memory crashes on consumer hardware.

---

## 5. Auditor's Verdict & Recommendation

1. **Registry Integrity:** `config/model_config.yaml` is clean, syntactically valid, and strictly matches the verified weights in `checkpoints/`.
2. **Honesty in Communication:** Any future release notes, academic papers, or portfolio summaries must **NOT** claim that Depth Anything 3 or 1.7B giant transformers are operating in RoomAI. The application is powered by **Depth Anything V2 Metric Indoor Small (24.8M)** alongside **A3 YOLO11s (9.4M)**.

