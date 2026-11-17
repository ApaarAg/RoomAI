# ROOMAI — PHASE 7: FORMAL CHECKPOINT SELECTION & PRODUCTION PROMOTION DECISION

**Author:** Independent Senior 3D Computer Vision Evaluation Scientist  
**Date:** November 8, 2026  
**Status:** COMPLETE & FORMALLY CERTIFIED  
**Final Verdict:** `STAGE 1 DOES NOT GENERALIZE` (Pretrained Baseline Retained)  
**Evaluation Scope:** 402 Validation Views across Synthetic CAD (Hypersim) and Real IR Sensor (ScanNet)  
**Quarantine Protocol:** Test Split (484 views across 2 scenes) strictly quarantined and unread.

---

## 1. PRESERVED CHECKPOINT INVENTORY & INTEGRITY AUDIT

All candidate checkpoints were verified with unaltered SHA-256 cryptographic hashes:

| Checkpoint Role | Checkpoint Path | File Size (Bytes) | Size (MB) | SHA-256 Checksum |
| :--- | :--- | :--- | :--- | :--- |
| **Reference Baseline** | `research/3d_reconstruction/checkpoints/production_candidate/production_model.ckpt` | 2,839,184,102 | 2,707.66 MB | `dbc9c9b0d4fae796ce5dd142bdde7e491057e484db313fef0397cac0c33970e5` |
| **Candidate Stage 1** | `research/3d_reconstruction/checkpoints/stage1/best_stage1.ckpt` | 2,839,164,068 | 2,707.64 MB | `fa12e006da6d1dbdbae810ee3995e9659dd8da5dad16dc9fe8380c9daa9481a2` |
| **Candidate Stage 2** | `research/3d_reconstruction/checkpoints/stage2/stage2_real_verified.ckpt` | 2,644,926,673 | 2,522.40 MB | `e1848950c17e23ab7f0a05771de519e61c484ebffe916b9320c61c5f5c239ece` |

No checkpoint file was modified, moved, or overwritten during this evaluation.

---

## 2. COMPREHENSIVE BENCHMARK SCORECARD

The table below summarizes performance across the 402 validation views:

| Metric Category | Metric Name | Pretrained Baseline (`production_model.ckpt`) | Fine-Tuned Stage 1 (`best_stage1.ckpt`) | Stage 2 3DGS (`stage2_real_verified.ckpt`) | Superior Model |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Overall Depth** | **AbsRel** $\downarrow$ | **0.5953** | 4.3341 (+628%) | 4.3341 (+628%) | **Pretrained** |
| | **RMSE (m)** $\downarrow$ | **2.8929 m** | 9.7075 m (+235%) | 9.7075 m (+235%) | **Pretrained** |
| | **Median AE (m)** $\downarrow$ | **1.9076 m** | 8.4682 m (+344%) | 8.4682 m (+344%) | **Pretrained** |
| | **$\delta < 1.25$ Acc** $\uparrow$ | **0.0584** | 0.0458 (-21.6%) | 0.0458 (-21.6%) | **Pretrained** |
| **Domain Depth** | **Hypersim AbsRel** $\downarrow$| 0.7452 | **0.7172** (-3.7%) | **0.7172** (-3.7%) | Stage 1 (Marginal) |
| | **ScanNet AbsRel** $\downarrow$ | **0.5456** | 5.5317 (+914%) | 5.5317 (+914%) | **Pretrained (10x better)** |
| | **ScanNet RMSE** $\downarrow$ | **1.5277 m** | 10.8707 m (+611%) | 10.8707 m (+611%) | **Pretrained (7x better)** |
| **Surface Normals** | **Mean Angular Error** $\downarrow$ | **62.66°** | 69.68° (+11.2%) | 69.68° (+11.2%) | **Pretrained** |
| | **Median Angular Error** $\downarrow$| **64.63°** | 72.45° (+12.1%) | 72.45° (+12.1%) | **Pretrained** |
| | **Acc $< 30^\circ$** $\uparrow$ | **0.1495** | 0.1410 (-5.7%) | 0.1410 (-5.7%) | **Pretrained** |
| **Shape Point Cloud**| **Chamfer Dist (Sim3)** $\downarrow$| **1.1704 m** | 1.3625 m (+16.4%) | 1.3625 m (+16.4%) | **Pretrained** |
| | **F-score @ 50mm** $\uparrow$ | **0.0078** | 0.0007 (-91.0%) | 0.0007 (-91.0%) | **Pretrained** |
| | **F-score @ 100mm** $\uparrow$| **0.0440** | 0.0051 (-88.4%) | 0.0051 (-88.4%) | **Pretrained** |
| **Metric Point Cloud**| **Chamfer Dist (Raw)** $\downarrow$| **2.5130 m** | 9.3379 m (+271%) | 9.3379 m (+271%) | **Pretrained** |
| | **F-score @ 50mm** $\uparrow$ | **0.0126** | 0.0031 (-75.4%) | 0.0031 (-75.4%) | **Pretrained** |
| **Performance** | **Median Latency (ms)** $\downarrow$| **279.1 ms** | 292.7 ms | 590.6 ms | **Pretrained** |
| | **Peak GPU VRAM** $\downarrow$ | 3393.7 MB | **3392.7 MB** | 3481.2 MB | Parity |

---

## 3. SECTION 11 PROMOTION GATES AUDIT

| Promotion Gate | Condition | Empirical Result | Gate Evaluation |
| :--- | :--- | :--- | :--- |
| **Gate 1: Metric Superiority** | Candidate must reduce validation error across primary depth and point cloud metrics. | Stage 1 degraded 14 out of 15 metrics, with depth AbsRel worsening from 0.5953 to 4.3341. | **FAILED** |
| **Gate 2: Domain Guardrail** | Candidate cannot win by improving on synthetic CAD while regressing on real sensor scans. | Stage 1 showed a minor 3.7% AbsRel change on Hypersim, but regressed by **+914.0% on ScanNet**. | **FAILED** |
| **Gate 3: Normal Regularity** | Surface normal angular error must not increase by $> 5\%$. | Stage 1 normal MAE worsened from 62.66° to 69.68° (+11.2%). | **FAILED** |
| **Gate 4: Runtime Constraints** | Latency $< 500\text{ ms}$, VRAM $< 6000\text{ MB}$. | Pretrained (279 ms, 3.4 GB) and Stage 1 (293 ms, 3.4 GB) passed; Stage 2 (591 ms) exceeded latency target. | **PASSED (Stage 1)** |

---

## 4. SCIENTIFIC ROOT-CAUSE ANALYSIS

Why did Stage 1 fail so definitively when evaluated against independent physical ground truth?
1. **Pseudo-Label Overfitting:** Phase 4 fine-tuning used pseudo-labels from Depth Anything V2. As established in the Phase 5 and Phase 6 audits, Depth Anything V2 generates *uncalibrated relative disparity maps*, not physical metric depth.
2. **Metric Distillation Collapse:** The fine-tuning loss compelled the model's metric regression heads to adjust their internal biases to match these relative disparities. This catastrophic scale shift resulted in predicted depths of 10–18 meters for ordinary 2-meter indoor spaces.
3. **The Value of Ground-Truth Benchmarking:** Without the independent physical ground truth established in Phase 6 (ScanNet Occipital infrared structured light + Hypersim CAD), this severe real-world failure would have gone undetected under synthetic or pseudo-label validation.

---

## 5. FORMAL VERDICT MANDATE

Following the strict protocol defined in the Phase 7 mandate, exactly one of the four mandated verdicts is issued:

```
========================================================================================
                                     FINAL VERDICT:
                             STAGE 1 DOES NOT GENERALIZE
                       (PRETRAINED BASELINE REMAINS BEST)
========================================================================================
```

### Actionable Consequences:
1. **Production Checkpoint Retained:** `research/3d_reconstruction/checkpoints/production_candidate/production_model.ckpt` (WorldMirror-1.0 Pretrained Baseline) is preserved as the active, certified production candidate.
2. **No Promotion of Stage 1 or Stage 2:** Neither `best_stage1.ckpt` nor `stage2_real_verified.ckpt` will be promoted to production.
3. **Quarantined Test Set Untouched:** The test split (`ai_013_010`, `scene0011_00`, `scene0011_01` — 484 views) remains completely unaccessed, ensuring zero leakage for future bona fide model releases.
4. **Phase 8 Roadmap:** Future fine-tuning must use true geometric supervision (Hypersim + ScanNet depth and normals from `indoor_ground_truth_manifest.json`) using multi-task loss formulations (scale-invariant depth + metric alignment + normal cosine similarity) to prevent scale drift.

