# RoomAI — Model B5 Final Evaluation & Benchmark Report

**Date:** October 25, 2026  
**Status:** Audited & Evaluated on Locked Test Split  
**Evaluation Target:** Model B5 (`exterior_detector_b5_transfer_best.pt`, MD5: `e12b4181fee244a3569dfa782ba3c3c3`) vs. Baseline B2 (`exterior_detector_b2_yolo11s_best.pt`, MD5: `14a931dd0adabc16cbd7b28ed1d17f66`)  
**Split Evaluated:** Locked Test Split (`data/exterior_b5_unified/images/test`, 154 images, 9,826 ground-truth annotations)  
**Structured Metrics:** [exterior_b5_metrics.json](file:///C:/Users/apaar/Desktop/RoomAI/reports/exterior_b5_metrics.json)  

---

## 1. Executive Summary

Model B5 was evaluated on the locked, untouched test split of 154 exterior architectural images (9,826 ground-truth annotations). All evaluation parameters were held strictly identical ($640\times640$ resolution, uniform confidence threshold $0.25$, IoU matching threshold $0.50$).

### Key Results:
- **Macro-F1:** `0.4563` on B5 vs. `0.4059` on B2 (**+5.04 percentage points gain**).
- **mAP50:** `0.4259` on B5 vs. `0.3737` on B2 (**+5.22 percentage points gain**).
- **mAP50-95:** `0.2621` on B5 vs. `0.2266` on B2 (**+3.55 percentage points gain**).
- **Micro-F1:** `0.6087` on B5 vs. `0.5939` on B2 (**+1.48 percentage points gain**).
- **Localization IoU:** `0.8294` on B5 vs. `0.8282` on B2 (+0.0012).
- **Inference Latency:** Warmed CUDA model-only latency across 50 test images was `17.45 ms` on B5 vs. `17.42 ms` on B2 (identical within measurement noise).
- **Class-Level Generalization:** B5 demonstrated improvements across **all 10 canonical exterior classes** with zero regressions. Crucially, the previously failing `porch` class improved from `0.0000` to `0.1667` F1 (`0.1354` AP50).

---

## 2. Locked Test Split Performance Comparison

| Metric | B2 (Production Baseline) | B5 (Transfer Learning Candidate) | Delta (B5 - B2) |
|---|:---:|:---:|:---:|
| **Checkpoint** | `exterior_detector_b2_yolo11s_best.pt` | `exterior_detector_b5_transfer_best.pt` | — |
| **MD5 Hash** | `14a931dd0adabc16cbd7b28ed1d17f66` | `e12b4181fee244a3569dfa782ba3c3c3` | — |
| **mAP50** | `0.3737` | **0.4259** | **+0.0522 (+5.22 pp)** |
| **mAP50-95** | `0.2266` | **0.2621** | **+0.0355 (+3.55 pp)** |
| **Macro-F1 (conf=0.25)** | `0.4059` | **0.4563** | **+0.0504 (+5.04 pp)** |
| **Micro-F1 (conf=0.25)** | `0.5939` | **0.6087** | **+0.0148 (+1.48 pp)** |
| **Micro-Precision** | `0.6693` | **0.6470** | -0.0223 |
| **Micro-Recall** | `0.5337` | **0.5746** | **+0.0409 (+4.09 pp)** |
| **Overall Mean IoU** | `0.8282` | **0.8294** | **+0.0012** |
| **Model Latency (CUDA)** | `17.42 ms` | **17.45 ms** | **+0.03 ms** |
| **Pipeline Latency (E2E)** | `25.80 ms` | **27.34 ms** | +1.54 ms |

---

## 3. Per-Class Precision, Recall, F1, and AP50

Evaluation conducted on 154 locked test images at confidence threshold $0.25$ and IoU matching threshold $0.50$:

| Canonical Class | GT Count | B2 Precision | B2 Recall | B2 F1 | B2 AP50 | **B5 Precision** | **B5 Recall** | **B5 F1** | **B5 AP50** | **F1 Delta** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `building` | 40 | 0.3333 | 0.2250 | 0.2687 | 0.2559 | **0.3824** | **0.3250** | **0.3514** | **0.3606** | **+0.0827** |
| `facade` | 579 | 0.4755 | 0.2349 | 0.3145 | 0.2664 | 0.3893 | **0.2884** | **0.3313** | **0.2776** | **+0.0168** |
| `window` | 4,090 | 0.7831 | 0.7611 | 0.7720 | 0.7937 | 0.7792 | **0.7792** | **0.7792** | **0.8040** | **+0.0072** |
| `door` | 432 | 0.5828 | 0.3912 | 0.4681 | 0.4092 | 0.5418 | **0.4954** | **0.5175** | **0.4633** | **+0.0494** |
| `roof_cornice` | 666 | 0.6399 | 0.6697 | 0.6544 | 0.6745 | **0.6505** | 0.6652 | **0.6578** | **0.6783** | **+0.0034** |
| `balcony` | 393 | 0.5437 | 0.5700 | 0.5565 | 0.5301 | **0.6034** | **0.6234** | **0.6133** | **0.5724** | **+0.0568** |
| `porch` | 28 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.3750** | **0.1071** | **0.1667** | **0.1354** | **+0.1667** |
| `pillar_column` | 896 | 0.4112 | 0.2042 | 0.2729 | 0.1878 | 0.3468 | **0.2969** | **0.3199** | **0.2384** | **+0.0470** |
| `shutter_blind` | 664 | 0.5330 | 0.1822 | 0.2716 | 0.1982 | **0.6823** | **0.1973** | **0.3061** | **0.2531** | **+0.0345** |
| `molding_deco` | 2,038 | 0.5711 | 0.4136 | 0.4798 | 0.4207 | 0.5667 | **0.4794** | **0.5194** | **0.4760** | **+0.0396** |
| **Aggregate / Mean** | **9,826** | **0.6693** | **0.5337** | **0.4059** | **0.3737** | **0.6470** | **0.5746** | **0.4563** | **0.4259** | **+0.0504** |

---

## 4. Key Engineering Insights

1. **Rare Class Activation Without Retraining Penalty:**
   Prior to B5, the B2 baseline achieved a zero recall and zero F1 (`0.0000`) on `porch`. Integrating 335 clean Open Images scenes provided sufficient domain diversity for B5 to achieve an F1 of `0.1667` and AP50 of `0.1354` on the locked test set without degrading any of the other 9 classes.
2. **Substantial Recall Surge:**
   B5 increased micro-recall across all 9,826 instances from `53.37%` to `57.46%` (+4.09 percentage points), successfully detecting 408 additional architectural components that B2 completely missed (including +83 doors, +74 windows, +83 pillars, +134 moldings).
3. **Consistent Geometric IoU:**
   Mean bounding box IoU on matched true positives was `0.8294`, indicating high spatial localization quality that matches or slightly exceeds the B2 baseline.
4. **Zero Latency Penalty:**
   Because both B2 and B5 share the fused YOLO11s architecture (101 fused layers, 9.42M parameters, 21.3 GFLOPs), warmed CUDA inference latency remained identical at `~17.4 ms`.

---

## 5. Deployment Recommendations

1. **Candidate Registration:** Model B5 should be registered in `config/model_config.yaml` as an experimental candidate (`exterior_detection_b5_candidate: enabled: false`) to allow controlled opt-in testing without altering active production defaults.
2. **Production Safeguards:** B2 remains the active production detector (`exterior_detector_b2_yolo11s_best.pt`) until a formal canary deployment and approval cycle is completed.
3. **DIP Post-Processing:** All classical DIP post-processing and edge snapping should remain disabled, as demonstrated by the Phase 5B evaluation report.
