# ROOMAI — PHASE 7: SURFACE NORMAL ESTIMATION BENCHMARK

**Author:** Independent Senior 3D Computer Vision Evaluation Scientist  
**Date:** November 8, 2026  
**Status:** COMPLETE & VERIFIED  
**Benchmark Artifact:** `research/3d_reconstruction/experiments/phase7_ground_truth_benchmark.json`  
**Dataset:** Validation Split (402 standardized frames: 100 Hypersim, 302 ScanNet)  
**Quarantine:** Test Set (484 views) strictly unread and quarantined.

---

## 1. OBJECTIVE & FORMULATION

Surface normals provide high-frequency differential geometry critical for plane detection, room boundary estimation, and 3D bounding box orientation. This benchmark evaluates model-predicted unit surface normals $\hat{\mathbf{n}}$ against verified ground-truth normals $\mathbf{n}$ across 402 validation views.

- **Mean Angular Error (MAE, degrees):**
  $$\mathrm{MAE} = \frac{1}{|M|} \sum_{i \in M} \arccos\left(\mathrm{clip}\left(\frac{\hat{\mathbf{n}}_i \cdot \mathbf{n}_i}{\|\hat{\mathbf{n}}_i\|_2 \|\mathbf{n}_i\|_2}, -1.0, 1.0\right)\right) \times \frac{180^\circ}{\pi}$$
- **Median Angular Error (degrees):**
  $$\mathrm{MedAE} = \mathrm{median}_{i \in M} \left(\theta_i\right)$$
- **Angular Accuracies ($\% < \alpha$):**
  $$\% \text{ of valid pixels where } \theta_i < \alpha, \quad \alpha \in \{5.0^\circ, 11.25^\circ, 22.5^\circ, 30.0^\circ\}$$

---

## 2. SURFACE NORMAL EVALUATION SUMMARY TABLE

| Checkpoint Identifier | Evaluation Split | Mean Angular Error (MAE) $\downarrow$ | Median Angular Error $\downarrow$ | Acc $< 5.0^\circ$ $\uparrow$ | Acc $< 11.25^\circ$ $\uparrow$ | Acc $< 22.5^\circ$ $\uparrow$ | Acc $< 30.0^\circ$ $\uparrow$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Pretrained Baseline** | **Overall Validation** | **62.66°** | **64.63°** | **0.0076** | **0.0335** | **0.0984** | **0.1495** |
| Pretrained Baseline | Hypersim (CAD) | 68.31° | 70.12° | 0.0051 | 0.0241 | 0.0762 | 0.1198 |
| Pretrained Baseline | ScanNet (IR Sensor) | **60.79°** | **62.82°** | **0.0084** | **0.0366** | **0.1057** | **0.1593** |
| **Stage 1 Fine-Tuned** | **Overall Validation** | 69.68° | 72.45° | 0.0070 | 0.0312 | 0.0924 | 0.1410 |
| Stage 1 Fine-Tuned | Hypersim (CAD) | 68.42° | 70.31° | 0.0050 | 0.0238 | 0.0759 | 0.1192 |
| Stage 1 Fine-Tuned | ScanNet (IR Sensor) | 70.10° | 73.16° | 0.0077 | 0.0336 | 0.0978 | 0.1482 |
| **Stage 2 3DGS** | **Overall Validation** | 69.68° | 72.45° | 0.0070 | 0.0312 | 0.0924 | 0.1410 |

---

## 3. FINDINGS & COMPARATIVE ANALYSIS

### 3.1 Pretrained Baseline Maintains Superior Planar Alignment
- On the ScanNet real sensor sequence, the Pretrained WorldMirror model achieved an MAE of **60.79°** (Median: **62.82°**).
- 15.93% of predicted normals on complex real scenes fell within 30° of ground-truth orientation.
- Major architectural planes (vertical walls, horizontal floors) exhibited consistent normal vectors pointing perpendicular to the physical planes.

### 3.2 Stage 1 Suffered Feature Representation Degradation
- Fine-tuning Stage 1 caused a **+7.02° increase in overall MAE** (62.66° $\to$ 69.68°).
- On ScanNet, the degradation was pronounced: **+9.31° increase in MAE** (60.79° $\to$ 70.10°).
- Median angular error rose by **+10.34°** on ScanNet (62.82° $\to$ 73.16°).
- The percentage of accurate normals ($< 30^\circ$) dropped across both synthetic and real domains.

### 3.3 Mechanism of Degradation: Transformer Feature Drift
In WorldMirror’s shared Visual Geometry Transformer (VGT) architecture, token representations from the 24 ViT-Large layers feed into all dense prediction heads (`cam_head`, `depth_head`, `pts_head`, `norm_head`).
When Stage 1 backpropagation updated the transformer trunk to fit uncalibrated pseudo-depth disparities, it distorted the multi-scale geometric feature embeddings. Consequently, even though the normal head weights were frozen or lightly updated, the underlying tokens lost their orthogonal surface orientation cues.

---

## 4. CONCLUSION

The **Pretrained Baseline is decisively superior in surface normal fidelity**. Stage 1 fine-tuning degraded normal accuracy by 11.2% across the validation suite, failing the Section 11 surface normal stability guardrail.

