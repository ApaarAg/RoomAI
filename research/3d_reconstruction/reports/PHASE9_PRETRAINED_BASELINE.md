# ROOMAI — PHASE 9: PRETRAINED WORLDMIRROR-1.0 BASELINE BENCHMARK REPORT

**Author:** Senior 3D Computer Vision Research Scientist & Foundation Model Optimization Engineer  
**Date:** 2026-11-08  
**Project:** RoomAI (`F:\RoomAI`)  
**Status:** COMPLETE & EMPIRICALLY VERIFIED  
**Artifact Hash:** `dbc9c9b0d4fae796ce5dd142bdde7e491057e484db313fef0397cac0c33970e5`  
**JSON Evidence:** `research/3d_reconstruction/experiments/phase9/pretrained_baseline_evaluation.json`

---

## 1. EXECUTIVE SUMMARY & CHECKPOINT IDENTITY

Before conducting controlled fine-tuning experiments in Phase 9, an independent, rigorous ground-truth evaluation was executed for the immutable production candidate checkpoint `production_model.ckpt`.

### Verified Checkpoint Identity
- **File Name:** `production_model.ckpt`
- **File Location:** `research/3d_reconstruction/checkpoints/production_candidate/production_model.ckpt`
- **File Size:** 2,839,183,923 bytes (2,707.66 MB)
- **SHA-256 Digest:** `dbc9c9b0d4fae796ce5dd142bdde7e491057e484db313fef0397cac0c33970e5`
- **Base Architecture:** WorldMirror-1.0 (Visual Geometry Transformer backbone: 24 `frame_blocks` + 24 `global_blocks`, patch size 14, embedding dimension 1024, DPT multi-task prediction heads for Depth, Points, and Normals; total parameters: 1,223,736,047).
- **Inference Precision:** `torch.bfloat16` on NVIDIA GeForce RTX 4060 Laptop GPU (8,188 MB VRAM).

### Key Empirical Findings
1. **Zero Prediction Failures:** Evaluated 175 held-out indoor frames (100 Track A Hypersim + 75 Track B ScanNet) with **0.0% failure rate** (all frames yielded valid depth, point cloud, and normal predictions exceeding the 100-pixel validity threshold).
2. **Domain-Specific Scaling:** WorldMirror-1.0 operates in a canonical scale space:
   - On ScanNet (real sensor depth, range 0.5m–5.0m), the median scale factor $s^*$ is **$2.52$**, yielding a scaled depth AbsRel of **$0.2730$** ($\delta_1 = 59.6\%$).
   - On Hypersim (synthetic architectural render, range 0.8m–25.0m), the median scale factor $s^*$ is **$4.93$**, yielding a scaled depth AbsRel of **$0.5469$** ($\delta_1 = 29.3\%$).
3. **Point Cloud Accuracy:**
   - ScanNet 3D point cloud Sim(3) alignment error is **$0.4633\text{ m}$** (MAE) across real physical rooms.
   - Hypersim 3D point cloud Sim(3) alignment error is **$3.2401\text{ m}$** (reflecting large architectural distances up to 25 meters).
4. **Surface Normal Precision:** Mean angular error is **$61.53^\circ$** on Hypersim and **$62.94^\circ$** on ScanNet, with **$26.4\%$** of Hypersim surface normals within $30^\circ$ of ground truth.
5. **Hardware Footprint:** Peak VRAM consumption was **$3,379.8\text{ MB}$** (41.3% of 8 GB VRAM), with an average inference latency of **$377.8\text{ ms}$** per frame (2.65 FPS).

---

## 2. EVALUATION METHODOLOGY & DUAL-TRACK FIREWALL

To guarantee strict compliance with commercial and non-commercial licensing constraints while testing both in-domain fidelity and cross-domain generalization, the evaluation dataset was divided into two quarantined tracks:

```
                                  [ EVALUATION FIREWALL ]
                                             |
            +--------------------------------+--------------------------------+
            |                                                                 |
     [ TRACK A: HYPERSIM ]                                            [ TRACK B: SCANNET ]
  - Commercial Candidate                                          - Non-Commercial Research
  - License: CC BY-SA 3.0                                          - License: Custom Academic
  - Scene: ai_016_001 (Held-out)                                   - Scenes: scene0030_00/01/02
  - Frames: 100 validation frames                                  - Frames: 75 validation frames
  - Synthetic ray-traced ground truth                              - Sensor RGB-D ground truth
```

### Strict Test-Set Quarantine
The frozen test set scenes:
- `ai_013_010` (Hypersim: 100 test frames)
- `scene0011_00` (ScanNet: 178 test frames)
- `scene0011_01` (ScanNet: 206 test frames)
Total **484 test frames** remain strictly **quarantined and unread** during baseline evaluation and checkpoint selection.

---

## 3. TRACK A: HYPERSIM BASELINE RESULTS (Commercial Candidate)

Evaluated on 100 consecutive frames of held-out validation scene `ai_016_001`.

| Metric Category | Metric | Mean | Median | Std Dev |
| :--- | :--- | :---: | :---: | :---: |
| **Data Integrity** | Total Frames | 100 | 100 | — |
| | Valid Frames | 100 | 100 | — |
| | Failure Rate | **0.00%** | **0.00%** | — |
| | Valid Pixel Ratio | 92.86% | 95.96% | 8.09% |
| **Raw Metric Depth** | Raw AbsRel | **0.7452** | 0.7517 | 0.1257 |
| | Raw RMSE | **7.0159 m** | 6.4904 m | 2.0021 m |
| | Raw $\delta < 1.25$ | 0.0388 | 0.0000 | 0.0783 |
| | Raw $\delta < 1.25^2$ | 0.0842 | 0.0000 | 0.1396 |
| **Scale-Aligned Depth** | Scaled AbsRel | **0.5469** | 0.4655 | 0.3177 |
| | Scaled RMSE | **4.6727 m** | 4.6813 m | 1.1281 m |
| | Scaled $\delta < 1.25$ | **0.2926** | 0.2458 | 0.1468 |
| | Scaled $\delta < 1.25^2$ | 0.5307 | 0.5243 | 0.1650 |
| | Alignment Scale $s^*$ | 4.9346 | 4.4305 | 2.3614 |
| **3D Point Cloud** | Raw Euclidean MAE | 5.7572 m | 5.2904 m | 2.1721 m |
| | Raw Euclidean RMSE | 7.3973 m | 6.8224 m | 2.0654 m |
| | Sim(3) Umeyama MAE | **3.2401 m** | 3.3076 m | 0.8650 m |
| | Sim(3) Umeyama RMSE | 4.0093 m | 4.0397 m | 0.9735 m |
| | Sim(3) Scale Factor | 7.8209 | 8.0203 | 2.0497 |
| **Surface Normals** | Mean Angular Error | **61.53°** | 59.54° | 14.40° |
| | Median Angular Error | 67.60° | 75.97° | 22.30° |
| | Normals $< 11.25^\circ$ Acc | 7.28% | 6.40% | 6.30% |
| | Normals $< 22.50^\circ$ Acc | 20.25% | 19.18% | 10.88% |
| | Normals $< 30.00^\circ$ Acc | **26.45%** | 25.65% | 12.22% |

---

## 4. TRACK B: SCANNET BASELINE RESULTS (Non-Commercial Research)

Evaluated across 75 representative frames sampled uniformly from three distinct real-world indoor sequences (`scene0030_00`, `scene0030_01`, `scene0030_02`).

### 4.1 Track B Overall Aggregates

| Metric Category | Metric | Mean | Median | Std Dev |
| :--- | :--- | :---: | :---: | :---: |
| **Data Integrity** | Total Frames | 75 | 75 | — |
| | Valid Frames | 75 | 75 | — |
| | Failure Rate | **0.00%** | **0.00%** | — |
| | Valid Pixel Ratio | 89.42% | 90.54% | 3.45% |
| **Raw Metric Depth** | Raw AbsRel | **0.5556** | 0.5611 | 0.0559 |
| | Raw RMSE | **1.5285 m** | 1.5069 m | 0.2412 m |
| | Raw $\delta < 1.25$ | 0.0513 | 0.0296 | 0.0588 |
| | Raw $\delta < 1.25^2$ | 0.1364 | 0.1288 | 0.0964 |
| **Scale-Aligned Depth** | Scaled AbsRel | **0.2730** | 0.2770 | 0.1004 |
| | Scaled RMSE | **0.6618 m** | 0.6381 m | 0.2257 m |
| | Scaled $\delta < 1.25$ | **0.5959** | 0.5724 | 0.1852 |
| | Scaled $\delta < 1.25^2$ | 0.8068 | 0.8195 | 0.1262 |
| | Alignment Scale $s^*$ | 2.5163 | 2.5157 | 0.3388 |
| **3D Point Cloud** | Raw Euclidean MAE | 1.4612 m | 1.4520 m | 0.2522 m |
| | Sim(3) Umeyama MAE | **0.4633 m** | 0.4655 m | 0.1542 m |
| | Sim(3) Umeyama RMSE | 0.5349 m | 0.5213 m | 0.1647 m |
| | Sim(3) Scale Factor | 2.8085 | 2.7630 | 0.3605 |
| **Surface Normals** | Mean Angular Error | **62.94°** | 62.58° | 5.22° |
| | Normals $< 30.00^\circ$ Acc | 11.62% | 11.23% | 3.95% |

### 4.2 Per-Scene Breakdown (ScanNet Validation)

| Scene ID | Frame Count | Raw AbsRel | Scaled AbsRel | Scaled RMSE | Scaled $\delta_1$ | Sim(3) MAE | Normal MAE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `scene0030_00` | 25 | $0.5531 \pm 0.055$ | **0.2707** | 0.6137 m | 61.11% | 0.4277 m | 61.91° |
| `scene0030_01` | 25 | $0.5564 \pm 0.057$ | **0.2860** | 0.7287 m | 55.08% | 0.5066 m | 64.20° |
| `scene0030_02` | 25 | $0.5572 \pm 0.056$ | **0.2622** | 0.6430 m | 62.57% | 0.4557 m | 62.70° |
| **Scene-Weighted Average** | — | **0.5556** | **0.2730** | **0.6618 m** | **59.59%** | **0.4633 m** | **62.94°** |

**Consistency Observation:** Across all three independent ScanNet camera paths, the pretrained model exhibits remarkable metric stability: Scaled AbsRel varies by less than $\pm 0.015$ across scenes ($0.2622$ to $0.2860$).

---

## 5. HARDWARE & LATENCY TELEMETRY

- **Device:** NVIDIA GeForce RTX 4060 Laptop GPU (8,188 MB dedicated VRAM)
- **Host Memory:** 24 GB DDR5 System RAM
- **Operating System:** Windows 11 Home (Build 26100)
- **CUDA Runtime:** 12.8, PyTorch 2.1.2 (cu121)
- **Peak VRAM Allocated:** **$3,379.75\text{ MB}$** (41.28% of 8,188 MB capacity)
- **Inference Latency Profile:**
  - Initial warm-up frame: $1,668.9\text{ ms}$
  - Steady-state frame latency: **$351.8\text{ ms}$** (median), $377.8\text{ ms}$ (mean)
  - Throughput: **$2.65\text{ FPS}$**

---

## 6. PRETRAINED REFERENCE WINNER & SELECTION CRITERIA

In accordance with Phase 9 protocol, `production_model.ckpt` is established as the **immutable Reference Winner**.

### Gate Requirements for Candidate Checkpoint Promotion
A newly fine-tuned candidate checkpoint (from Experiment A or Experiment B) will replace `production_model.ckpt` **if and only if** all of the following conditions are simultaneously met:

1. **Track A (Hypersim) Improvement:**
   - Candidate Scaled AbsRel must improve by at least **5.0% relative** ($< 0.5195$).
   - Point cloud Sim(3) MAE must improve ($< 3.24\text{ m}$).
2. **Track B (ScanNet) Generalization Guardrail:**
   - Candidate Scaled AbsRel on ScanNet must **not regress by more than 10.0% relative** (must remain $\le 0.3003$).
3. **Surface Geometry Guardrail:**
   - Surface normal MAE must not regress by more than $3.0^\circ$ ($< 64.5^\circ$ on Hypersim, $< 65.9^\circ$ on ScanNet).
4. **Reliability & Stability Guardrail:**
   - Zero NaN / Inf predictions, failure rate must remain **$0.0\%$**.
5. **Physical Checkpoint Integrity:**
   - Resumable optimizer state and model weights serialized with verifiable SHA-256 digest.

If any candidate violates any of these guardrails, `production_model.ckpt` will remain the sole production checkpoint.

