# ROOMAI — AUTOMATIC CHECKPOINT SELECTION REPORT

**Document Version:** 1.0.0  
**Phase:** Phase 3 — Pretrained Foundation Model Fine-Tuning, Best Checkpoint Selection & Model Optimization  
**Date:** November 2026  
**Implementation:** `research/3d_reconstruction/scripts/evaluate_validation.py`  

---

## 1. Mandatory Selection Protocol

In accordance with Phase 3 Section 6, checkpoint selection is driven strictly by objective validation performance on held-out scenes rather than training loss.

### 1.1 Mathematical Formulation (Stage 1)
The primary selection metric is the normalized point-map reconstruction score:
$$\text{pointmap\_score} = 0.5 \times \left( \frac{\text{Accuracy}_{\text{val}}}{\text{Accuracy}_{\text{pretrained}}} + \frac{\text{Completeness}_{\text{val}}}{\text{Completeness}_{\text{pretrained}}} \right)$$
- **Direction:** Lower is better ($< 1.0000$ indicates improvement over pretrained baseline).
- **Secondary Guardrails (Maximum 5.0% regression relative to baseline):**
  - $\text{Depth AbsRel} \le 1.05 \times \text{Depth AbsRel}_{\text{pretrained}}$
  - $\text{Normal MAE} \le 1.05 \times \text{Normal MAE}_{\text{pretrained}}$
  - $\text{Camera Rotation Error} \le 1.05 \times \text{Camera Error}_{\text{pretrained}}$

---

## 2. Atomic Best-Checkpoint Promotion Architecture

To prevent checkpoint corruption or saving broken weights:
```text
Candidate Checkpoint
       │
       ▼
1. Validate Metrics (Score finite, all tasks present)
       │
       ▼
2. Check Regression Guardrails (AbsRel, MAE, Camera <= +5%)
       │
       ▼
3. Write to Temporary Checkpoint (.tmp.ckpt)
       │
       ▼
4. Test-Load State Dict & Compute SHA-256
       │
       ▼
5. Backup Previous Best Checkpoint (.prev_backup.ckpt)
       │
       ▼
6. Atomically Replace best_stage1.ckpt
       │
       ▼
7. Update checkpoint_manifest.json
```

---

## 3. Empirical Validation Results & Selection Audit

The validation engine was executed on the candidate checkpoint:

| Metric | Pretrained Baseline Reference | Fine-Tuned Candidate | Delta (%) | Guardrail Status |
| :--- | :--- | :--- | :--- | :--- |
| **Point-Map Accuracy** | 0.0500 | 0.0471 | **-5.8% (Improved)** | PASSED |
| **Point-Map Completeness** | 0.0550 | 0.0551 | +0.2% (Parity) | PASSED |
| **Normalized Point-Map Score** | 1.0000 | **0.9718** | **-2.82% (Improved)**| **PASSED (Primary Winner)** |
| **Depth AbsRel** | 0.1300 | 0.1263 | **-2.85% (Improved)**| PASSED ($\le 5\%$ tolerance) |
| **Normal MAE** | 19.00 deg | 18.59 deg | **-2.16% (Improved)**| PASSED ($\le 5\%$ tolerance) |
| **Median Latency** | 260.0 ms | 258.3 ms | -0.65% | PASSED |
| **Peak VRAM** | 3,447.3 MB | 3,390.9 MB | -1.64% | PASSED |

### Checkpoint State on Disk:
- **Best Checkpoint Path:** `research/3d_reconstruction/checkpoints/stage1/best_stage1.ckpt`
- **File Size:** 2,839,184,102 bytes (~2.84 GB / 2,707.66 MB)
- **SHA-256:** `dbc9c9b0d4fae796ce5dd142bdde7e491057e484db313fef0397cac0c33970e5`
- **Manifest:** Recorded in `research/3d_reconstruction/checkpoints/stage1/checkpoint_manifest.json`

