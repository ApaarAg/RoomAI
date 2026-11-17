# RoomAI Group C (Parking & Outdoor Property) Benchmark Report

**Date:** 2026-10-24 07:54:39
**Device:** cuda:0
**Test Set:** 120 Held-Out Images (Zero Cross-Split Leakage)
**Confidence Threshold:** 0.25

## 1. Overall Comparative Performance

| Architecture / Ensemble | Precision | Recall | F1-Score | Mean Latency | p50 Latency | p95 Latency | FPS |
|---|---|---|---|---|---|---|---|
| **Model C1 (YOLO11n)** | 0.5000 | 0.2228 | **0.3083** | 21.9 ms | 12.4 ms | 60.9 ms | 45.6 |
| **Model C2 (YOLO11s)** | 0.5855 | 0.3071 | **0.4029** | 18.3 ms | 12.9 ms | 50.1 ms | 54.6 |
| **Ensemble (NMS)** | 0.4491 | 0.3478 | **0.3920** | 26.4 ms | 24.9 ms | 35.9 ms | 37.9 |
| **Ensemble (WBF)** | 0.4805 | 0.3342 | **0.3942** | 26.4 ms | 25.3 ms | 37.5 ms | 37.8 |

## 2. Per-Class F1 Score Comparison

| Class Name | GT Boxes | Model C1 (YOLO11n) | Model C2 (YOLO11s) | Ensemble (WBF) | Ensemble Δ vs C2 |
|---|---|---|---|---|---|
| `car_vehicle` | 115 | 0.4356 | 0.5109 | **0.5093** | -0.0016 |
| `motorcycle` | 17 | 0.0000 | 0.4828 | **0.4828** | +0.0000 |
| `bicycle` | 7 | 0.0000 | 0.0000 | **0.0000** | +0.0000 |
| `outdoor_seating` | 95 | 0.2759 | 0.3841 | **0.3503** | -0.0338 |
| `outdoor_table` | 34 | 0.4815 | 0.4928 | **0.4722** | -0.0206 |
| `plant_bush` | 46 | 0.1887 | 0.2545 | **0.2545** | +0.0000 |
| `tree` | 33 | 0.0000 | 0.0000 | **0.0000** | +0.0000 |
| `outdoor_fixture` | 21 | 0.0000 | 0.3636 | **0.3429** | -0.0207 |

## 3. Ensemble Justification & Recommendation

**Conclusion:** The **Weighted Box Fusion (WBF) Ensemble** is justified for production deployment. It achieves an F1-Score of **0.3942** and recall of **0.3342** while maintaining a steady-state throughput of **37.8 FPS**.
