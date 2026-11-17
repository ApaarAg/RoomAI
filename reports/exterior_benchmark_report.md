# RoomAI Group B (Exterior & Building Components) Benchmark Report

**Date:** 2026-10-24 07:53:21
**Test Set:** 134 held-out exterior architecture images
**Confidence Threshold:** 0.25

## 1. Overall Model & Ensemble Comparison

| Model / Architecture | Precision | Recall | F1 Score | Latency (Mean) | Steady FPS |
|---|---|---|---|---|---|
| **Model B1 (YOLO11n)** | 0.9512 | 0.0040 | **0.0080** | 51.17 ms | 19.5 FPS |
| **Model B2 Pilot (YOLO11s)** | 0.8253 | 0.3144 | **0.4554** | 32.09 ms | 31.2 FPS |
| **Fine-Tuned Model B2 (YOLO11s)** | 0.7972 | 0.4833 | **0.6018** | 29.61 ms | 33.8 FPS |
| **Ensemble (WBF)** | 0.7888 | 0.4889 | **0.6037** | 51.47 ms | 19.4 FPS |
| **Ensemble (NMS)** | 0.7987 | 0.4796 | **0.5993** | 50.65 ms | 19.7 FPS |

## 2. Key Class Performance (F1 Score Comparison)

| Class Name | Model B1 (Pilot) | Model B2 (Pilot) | Model B2 (Fine-Tuned) | Ensemble (WBF) | Ensemble (NMS) |
|---|---|---|---|---|---|
| `building` | 0.0000 | 0.1250 | **0.3784** | 0.3784 | 0.3784 |
| `facade` | 0.0503 | 0.2698 | **0.3059** | 0.3162 | 0.3063 |
| `window` | 0.0119 | 0.6535 | **0.7962** | 0.7957 | 0.7953 |
| `door` | 0.0000 | 0.3618 | **0.4179** | 0.4241 | 0.4122 |
| `roof_cornice` | 0.0000 | 0.4995 | **0.6865** | 0.6875 | 0.6848 |
| `balcony` | 0.0000 | 0.2639 | **0.5562** | 0.5581 | 0.5562 |
| `porch` | 0.0000 | 0.0000 | **0.0000** | 0.0000 | 0.0000 |
| `pillar_column` | 0.0000 | 0.0801 | **0.2310** | 0.2511 | 0.2321 |
| `shutter_blind` | 0.0000 | 0.1103 | **0.2613** | 0.2411 | 0.2333 |
| `molding_deco` | 0.0000 | 0.3236 | **0.4484** | 0.4554 | 0.4442 |

## 3. Engineering Ensemble Recommendation

> [!TIP]
> **RETAIN ENSEMBLE**: Weighted Box Fusion (WBF) achieves F1 of 0.6037 (+0.19% improvement over best single detector Model B2). Ensemble fusion is empirically justified and retained.
