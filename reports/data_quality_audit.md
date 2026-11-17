# RoomAI Data Quality & Annotation Audit Report

**Date:** 2026-10-24  
**Author:** RoomAI Senior Computer Vision Engineer & Data Quality Lead  
**Datasets Audited:** `indoor_canonical`, `exterior_canonical`, `parking_canonical`, `deepfurniture`  
**Data Directory:** `data/`  

---

## 1. Executive Summary

A comprehensive, automated data quality audit was executed across all canonical datasets. The audit inspected:
- **Split contamination & duplicate leakage:** Verified whether identical images or scene-correlated frames cross train/val/test boundaries.
- **Class frequency distribution:** Measured representation equity and identified severe tail-class deficits.
- **Bounding box geometry integrity:** Screened for out-of-bounds coordinates, degenerate boxes ($w \le 0$ or $h \le 0$), tiny sub-pixel boxes ($<0.1\%$ image area), and giant boxes ($>90\%$ area).
- **Label completeness:** Verified that every image has an associated label file and vice-versa.

---

## 2. Dataset Overview & Split Inventory

| Dataset | Ontology Scope | Total Images | Train Split | Val Split | Test Split | Total Bounding Boxes | Empty Images |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`indoor_canonical`** | 12 indoor furniture classes | 3,689 | 2,985 (80.9%) | 554 (15.0%) | 150 (4.1%) | 28,606 | 0 (0.0%) |
| **`exterior_canonical`** | 10 building facade classes | 726 | 458 (63.1%) | 134 (18.5%) | 134 (18.5%) | 49,890 | 0 (0.0%) |
| **`parking_canonical`** | 8 outdoor parking classes | 750 | 510 (68.0%) | 120 (16.0%) | 120 (16.0%) | 2,416 | 0 (0.0%) |
| **`deepfurniture`** | 11 furniture classes (baseline) | 1,000 | 700 (70.0%) | 150 (15.0%) | 150 (15.0%) | 6,314 | 0 (0.0%) |

---

## 3. Split Contamination & Leakage Analysis

### Exact Duplicate Image Leakage (MD5 Hash Collision)

Every image file was hashed using MD5 checksums across all splits:
- **`indoor_canonical`:** **0 exact duplicates** across splits.
- **`exterior_canonical`:** **0 exact duplicates** across splits.
- **`parking_canonical`:** **0 exact duplicates** across splits.
- **`deepfurniture`:** **0 exact duplicates** across splits.

### Scene / Property Correlation Leakage

In real-estate and architectural datasets, consecutive video frames or photos taken from the same room/property can artificially inflate test metrics if split randomly:
- **`indoor_canonical`:** **0 scene groups leaked to test**. DeepFurniture scene hashes (`df_L3D...`) and HomeObjects scene stems are partitioned strictly by scene group.
- **`exterior_canonical`:** **1 scene prefix group** (`cmp_`) exhibits overlapping sequence stems across train and test. Specifically, synthetic CMP architectural elevations share similar texture baselines.
- **`parking_canonical`:** **0 correlated scene leakage groups**.

---

## 4. Class Distribution & Imbalance Analysis

### Detailed Frequency Table

| Class Name | `indoor_canonical` | `exterior_canonical` | `parking_canonical` | `deepfurniture` |
| :--- | :--- | :--- | :--- | :--- |
| **cabinet#shelf / building** | 5,312 (18.6%) | 2,810 (5.6%) | — | 1,196 (18.9%) |
| **table / facade** | 3,845 (13.4%) | 1,420 (2.8%) | — | 895 (14.2%) |
| **chair#stool / window** | 4,920 (17.2%) | **19,995 (40.1%)** | — | 1,040 (16.5%) |
| **lamp / door** | 2,140 (7.5%) | 2,150 (4.3%) | — | 480 (7.6%) |
| **door / roof_cornice** | 1,890 (6.6%) | 7,850 (15.7%) | — | 350 (5.5%) |
| **bed / balcony** | 1,120 (3.9%) | 2,410 (4.8%) | — | 290 (4.6%) |
| **sofa / porch** | 1,480 (5.2%) | **6 (0.01%)** | — | 410 (6.5%) |
| **plant / pillar_column** | 1,650 (5.8%) | 172 (0.34%) | — | 385 (6.1%) |
| **decoration / shutter_blind**| 3,110 (10.9%) | 6,840 (13.7%) | — | 710 (11.2%) |
| **curtain / molding_deco** | 1,210 (4.2%) | 6,237 (12.5%) | — | 315 (5.0%) |
| **home-appliance** | 1,420 (5.0%) | — | — | 243 (3.8%) |
| **window** | 509 (1.8%) | — | — | — |
| **car_vehicle** | — | — | 760 (31.5%) | — |
| **motorcycle** | — | — | 310 (12.8%) | — |
| **bicycle** | — | — | 81 (3.4%) | — |
| **outdoor_seating** | — | — | 280 (11.6%) | — |
| **outdoor_table** | — | — | 215 (8.9%) | — |
| **plant_bush** | — | — | 320 (13.2%) | — |
| **tree** | — | — | 260 (10.8%) | — |
| **outdoor_fixture** | — | — | 190 (7.9%) | — |
| **Total Boxes** | **28,606** | **49,890** | **2,416** | **6,314** |
| **Imbalance Ratio (Max:Min)**| **10.44 : 1** | **3,332.50 : 1** | **9.38 : 1** | **6.83 : 1** |

> [!WARNING]
> **Severe Imbalance Warning (Group B):**  
> `exterior_canonical` suffers from an extreme imbalance ratio of **3,332.5 : 1**. `window` contains 19,995 annotations, whereas `porch` contains only 6 instances, and `pillar_column` contains only 172. This explains why the detector collapsed to 0.000 mAP on `porch` and 0.172 mAP on `pillar_column`.

---

## 5. Bounding Box Geometry & Anomaly Audit

The coordinates of all 87,226 bounding boxes were inspected for geometric anomalies:

| Anomaly Type | `indoor_canonical` | `exterior_canonical` | `parking_canonical` | `deepfurniture` | Severity / Action |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Out-of-bounds Coordinates** ($x, y, w, h \notin [0, 1]$) | **0** | **0** | **0** | **0** | None |
| **Degenerate Boxes** ($w \le 0$ or $h \le 0$) | **0** | **0** | **0** | **0** | None |
| **Invalid Class IDs** | **0** | **0** | **0** | **0** | None |
| **Missing Image or Label Pairings** | **0** | **0** | **0** | **0** | None |
| **Tiny Boxes** ($w \times h < 0.001$, $<0.1\%$ area) | 699 (2.4%) | **6,451 (12.9%)** | 289 (12.0%) | 116 (1.8%) | Low-res edge fragments |
| **Giant Boxes** ($w \times h > 0.90$, $>90\%$ area) | 2 (0.01%) | **291 (0.58%)** | 95 (3.9%) | 0 (0.0%) | Full-frame facade boxes |

### Key Anomaly Findings

1. **Tiny Architectural Annotations in Exterior:**  
   $12.9\%$ of exterior boxes ($6,451$ instances) are smaller than $0.1\%$ of the total image area (e.g. tiny individual window panes, small molding segments). At $640 \times 640$ training resolution, an object occupying $<0.1\%$ area corresponds to fewer than $15 \times 15$ pixels, which causes feature downsampling collapse in the YOLO P5 stride-32 stage.
2. **Overlapping Giant Boxes in Exterior Facades:**  
   291 exterior images feature a bounding box covering $>90\%$ of the image labeled as `facade` or `building`, which completely encompasses all the smaller `window` and `door` boxes. This creates severe hierarchical box competition during Non-Maximum Suppression (NMS).
3. **Window Label Absence in Test Set of Group A:**  
   `window` is present in `indoor_canonical` training (382 boxes) and validation (127 boxes), but has 0 boxes in the held-out test split because the test split was preserved strictly from the original 11-class DeepFurniture benchmark. This causes the test mAP50 for `window` to register as 0.000 despite achieving $0.569$ mAP50 on validation.

---

## 6. Actionable Data Remediation Plan

1. **Exterior Re-balancing (Priority 1):**  
   - Apply **Repeat Factor Sampling (RFS)** and Copy-Paste data augmentation to boost `porch` (currently 6 instances) and `pillar_column` (172 instances) to at least 500 effective training instances.
2. **Exterior Box Hierarchy Separation (Priority 2):**  
   - Suppress giant $>90\%$ `facade` boxes during fine-grained feature matching, or treat `building` vs `facade` as a global scene attribute rather than a competing object detection box.
3. **Canonical Window Harmonization in Indoor Test Split (Priority 3):**  
   - Curate 50 verified window annotations in the held-out test split using HomeObjects-3K held-out scenes to enable fair, balanced evaluation of all 12 classes on test.
