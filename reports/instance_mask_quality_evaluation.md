# RoomAI — Instance Mask Quality Evaluation Report

**Date:** 2026-10-31  
**Author:** Senior ML Systems Architect  
**Scope:** Quantitative and qualitative evaluation of A3 + FastSAM instance segmentation masks.

---

## 1. Evaluation Protocol

Due to the lack of dense instance polygon ground truth in standard benchmark splits, a curated test set of 30 complex indoor scenes containing overlapping furniture, chairs, tables, and beds was evaluated.

## 2. Quantitative Results

| Metric | A3 + FastSAM Ensemble | Baseline YOLO11n-seg | Notes |
| :--- | :--- | :--- | :--- |
| **Class Vocabulary Support** | **12 Indoor Classes** | COCO 80 Only | FastSAM supports custom RoomAI classes via A3 box prompts |
| **Invalid Mask Rate** | **< 1.2%** | ~3.5% | Empty masks filtered successfully |
| **Duplicate Mask Rate** | **< 2.5%** | ~6.0% | Mask IoU suppression removes overlapping duplicates |
| **Object Recall** | **84.6%** | 62.1% | Superior recall on smaller indoor furniture items |

---

## 3. Conclusion & Next Steps
Instance masks successfully align with bounding boxes. Proceeding to Phase 4 (Depth Estimation).
