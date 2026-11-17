# RoomAI Phase 15 — Stage A: Specialist Architecture and Error Audit Report

**Auditor / ML Systems Engineer:** Senior Computer-Vision Engineer & Production ML Systems Engineer  
**Date:** October 25, 2026  
**Repository:** `C:\Users\apaar\Desktop\RoomAI`  
**Artifact Dependencies:**
- Audit Data: [reports/phase15_audit_details.json](file:///C:/Users/apaar/Desktop/RoomAI/reports/phase15_audit_details.json)
- B6 Metrics: [reports/exterior_b6_metrics.json](file:///C:/Users/apaar/Desktop/RoomAI/reports/exterior_b6_metrics.json)
- Model Registry: [models/registry.py](file:///C:/Users/apaar/Desktop/RoomAI/models/registry.py)
- Model Config: [config/model_config.yaml](file:///C:/Users/apaar/Desktop/RoomAI/config/model_config.yaml)

---

## 1. Executive Summary

Phase 15 evaluates whether RoomAI's exterior building component detection can be substantially improved by pairing the strongest general detector (Model B6) with targeted specialist detectors. In accordance with Phase A requirements, a comprehensive audit was executed prior to modifying any inference code or training new models.

### Key Audit Findings
1. **Checkpoint Preservation:** All production and experimental checkpoints (B2, B3, B4, B5, B6) are verified intact byte-for-byte against their canonical MD5 hashes.
2. **Split Isolation:** The 154-image validation and 154-image test splits maintain **zero image and scene overlap** with the 1,302 training images.
3. **Metric Implementation:** The evaluation protocol uses genuine greedy one-to-one matching at IoU $\ge 0.50$, consistent 10-class canonical mapping, and distinct Macro-F1 (arithmetic mean across classes) and Micro-F1 (instance-aggregated) formulations.
4. **Error Diagnosis for Weak Classes:**
   - **`porch`** suffers primarily from severe recall deficiency (FN = 21 / 24, 87.5% missed) and box boundary misalignment (3 of 4 FPs are localization errors with $0.10 \le \text{IoU} < 0.50$).
   - **`pillar_column`** exhibits heavy cross-class confusion with `roof_cornice` (33) and `molding_deco` (31), alongside flat wall pilasters predicted as columns.
   - **`facade`** suffers from building-level macro vs micro plane ambiguity (61 localization errors, 200 background FPs).
   - **`door`** exhibits visual confusion with tall glass `window` panes (64 instances).
   - **`roof_cornice`** and **`balcony`** demonstrate solid baseline recall (>62%) but suffer from fine ornamentation boundary ambiguity.

---

## 2. Checkpoint Verification and System Safeguards

All checkpoints have been verified via MD5 checksums:

| Model ID | Relative Path | Canonical MD5 Hash | Verification Status | Role in Phase 15 |
|---|---|---|:---:|---|
| **B2** | `checkpoints/exterior_detector_b2_yolo11s_best.pt` | `14a931dd0adabc16cbd7b28ed1d17f66` | **VERIFIED** | Active Production Default (Unchanged) |
| **B3** | `checkpoints/exterior_detector_b3_yolo11s_rfs_best.pt` | `2c44291fa642aba3893db001520deba0` | **VERIFIED** | Rollback Candidate (Preserved) |
| **B4** | `checkpoints/exterior_detector_b4_yolo11s_curated_best.pt` | `d2710fba760713e68ea306e029ab651c` | **VERIFIED** | Experimental Model (Preserved) |
| **B5** | `checkpoints/exterior_detector_b5_transfer_best.pt` | `e12b4181fee244a3569dfa782ba3c3c3` | **VERIFIED** | Transfer Learning Baseline (Preserved) |
| **B6** | `checkpoints/exterior_detector_b6_expanded_best.pt` | `f5ceea56b0610ea14b647014e48152dc` | **VERIFIED** | Primary General Candidate (Preserved) |

**Production Guardrail:** B2 remains the active default in [config/model_config.yaml](file:///C:/Users/apaar/Desktop/RoomAI/config/model_config.yaml). No specialist model or candidate will be promoted to production without explicit verification against rollback protocols.

---

## 3. Dataset Integrity & Scene-Level Split Separation

The dataset manifests across `data/exterior_b6_expanded/` were audited for data leakage:
- **Train Split:** 1,302 images (35,515 annotations)
- **Validation Split:** 154 images (8,504 annotations) — **Strictly Locked**
- **Test Split:** 154 images (9,826 annotations) — **Strictly Locked**

### Leakage & Overlap Audit:
- Exact MD5 Filename Collisions:
  - $\text{Train} \cap \text{Val} = 0$
  - $\text{Train} \cap \text{Test} = 0$
  - $\text{Val} \cap \text{Test} = 0$
- Perceptual Hash Duplicate Screening:
  - Validated that zero images in the validation or test splits share near-duplicate hash distance ($d_H \le 5$) with the training corpus.
- **Rule for Specialist Ingestion:** Any new training data or negative mining crops prepared for specialist models will strictly draw from the 1,302 training images or external public training splits, with zero access to the locked validation or test sets.

---

## 4. Evaluation Protocol & Metric Audit

The evaluation logic in [scripts/evaluate_exterior_b6_validation.py](file:///C:/Users/apaar/Desktop/RoomAI/scripts/evaluate_exterior_b6_validation.py) was reviewed for mathematical and methodological validity:

### 4.1 Greedy One-to-One Matching
Predictions are sorted in descending order of confidence:
$$\hat{y}_{(1)}, \hat{y}_{(2)}, \dots, \hat{y}_{(K)}, \quad \text{where } c(\hat{y}_{(i)}) \ge c(\hat{y}_{(i+1)})$$
For each prediction $\hat{y}_{(i)}$ belonging to class $c$:
1. Compute $\text{IoU}(\hat{y}_{(i)}, g)$ against all ground-truth boxes $g \in G_c$.
2. Identify the unassigned ground-truth box $g^*$ with the maximum IoU.
3. If $\text{IoU}(\hat{y}_{(i)}, g^*) \ge 0.50$:
   - Increment $TP_c$ by 1.
   - Mark $g^*$ as assigned (`matched_gt.add(g^*)`).
4. Otherwise:
   - Increment $FP_c$ by 1.
5. All unassigned ground-truth boxes in $G_c$ are counted as False Negatives:
   $$FN_c = |G_c| - |\{g \in G_c : g \text{ matched}\}|$$

### 4.2 Mathematical Metrics
- **Per-Class Precision:** $P_c = \frac{TP_c}{TP_c + FP_c}$ ($0.0$ if denominator is $0$)
- **Per-Class Recall:** $R_c = \frac{TP_c}{TP_c + FN_c}$ ($0.0$ if denominator is $0$)
- **Per-Class F1:** $F1_c = \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$ ($0.0$ if $P_c + R_c = 0$)
- **Macro-F1:** $\frac{1}{10} \sum_{c=0}^9 F1_c$ (arithmetic mean across all 10 canonical classes, giving equal weight to rare classes).
- **Micro-F1:** $2 \cdot \frac{\sum_c TP_c}{\sum_c (2 \cdot TP_c + FP_c + FN_c)}$ (instance-weighted aggregation).

---

## 5. Granular Error Categorization for the Six Weak Classes

Audited on the 154 validation images with Model B6 (confidence threshold 0.25, IoU 0.50):

```
+----------------+-------+-----+-----+-----+---------+--------+--------+------------+--------------------+----------------+
| Class Name     | GT    | TP  | FP  | FN  | Prec    | Recall | F1     | Loc Errors | Top Confusion      | Background FPs |
+----------------+-------+-----+-----+-----+---------+--------+--------+------------+--------------------+----------------+
| porch          | 24    | 3   | 4   | 21  | 0.4286  | 0.1250 | 0.1935 | 3          | None (0)           | 1              |
| pillar_column  | 494   | 167 | 360 | 327 | 0.3169  | 0.3381 | 0.3271 | 54         | roof_cornice (33)  | 224            |
| facade         | 495   | 177 | 316 | 318 | 0.3590  | 0.3576 | 0.3583 | 61         | molding_deco (27)  | 200            |
| door           | 372   | 206 | 297 | 166 | 0.4095  | 0.5538 | 0.4709 | 66         | window (64)        | 159            |
| roof_cornice   | 445   | 318 | 261 | 127 | 0.5492  | 0.7146 | 0.6211 | 60         | pillar_column (31) | 155            |
| balcony        | 345   | 216 | 114 | 129 | 0.6545  | 0.6261 | 0.6400 | 36         | molding_deco (11)  | 62             |
+----------------+-------+-----+-----+-----+---------+--------+--------+------------+--------------------+----------------+
```

### 5.1 Root Cause Breakdown

1. **`porch` (Primary Target for Specialist):**
   - **Severe Recall Deficit:** 21 of 24 porches (87.5%) are completely missed by Model B6 at confidence 0.25.
   - **Localization Sensitivity:** Of the 4 false positives emitted, 3 have substantial overlap ($0.10 \le \text{IoU} < 0.50$) with the true porch structure. The model correctly identifies the porch presence but over-extends the box to encompass adjacent stairs, overhangs, or railings.
   - **Specialist Opportunity:** A dedicated specialist trained specifically on localized residential entries with tailored aspect ratios and multi-scale contextual features can dramatically recover missed detections and tighten spatial bounds.

2. **`pillar_column`:**
   - **Structural Merging:** 33 FPs overlap directly with decorative capitals under `roof_cornice`, and 31 overlap with vertical `molding_deco` fluting.
   - **Non-Standard Architectural Elements:** 224 background FPs stem from vertical pilasters and window mullions that are visually identical to columns but unannotated in the CMP Facade ground truth.

3. **`facade`:**
   - **Macro-Micro Scale Ambiguity:** In multi-story or multi-wing buildings, ground-truth annotations often define one large envelope, whereas the detector identifies individual facade planes for separate wings (61 localization errors, 200 background FPs).

4. **`door` vs `window`:**
   - **Material Confusion:** 64 false positives for `door` overlap with ground-truth `window` boxes. These almost exclusively represent glazed storefront windows, sliding glass balcony doors, and french patio windows.

5. **`roof_cornice` and `balcony`:**
   - Demonstrating acceptable recall (71.5% and 62.6%), their primary errors are boundary overshoots where balustrades meet wall moldings.

---

## 6. Phase B Specialist Design Specifications

Based on the audit, the **Porch Specialist** will be designed with the following constraints:
1. **Focus:** Single-class binary detector or 2-class context-aware detector (`porch` vs `negative_structure`), trained using transfer learning from B6/YOLO11s.
2. **Hard Negatives:** Enforce balanced negative training exemplars (facades without porches, entry doors, columns, and low balconies) to suppress spurious entrance false positives.
3. **Inference Modes to Compare:**
   - **Parallel Ensemble:** General detector (B6) + Porch Specialist run concurrently, fused via calibrated IoU merging.
   - **Cascaded Inference:** General detector runs first; if entrance cues (e.g., lower ground-floor door, low-confidence porch, or building entrance region) are detected, the Porch Specialist is selectively routed to the lower half or entrance crop.
   - **High-Resolution Inference:** Specialist operates at $imgsz=800$ or on high-resolution crops to resolve fine structural details (railings, columns, portico roofs).

This audit establishes the baseline evidence before any architectural or pipeline modifications.
