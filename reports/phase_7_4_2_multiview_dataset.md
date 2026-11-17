# RoomAI Phase 7.4.2 — Multi-View Dataset Investigation & Provenance Audit

**Date:** November 6, 2026  
**Author:** Senior Multi-View Geometry Engineer & Scientific Reproducibility Auditor  
**Project:** RoomAI (`F:\RoomAI`)  
**Scope:** Exhaustive Investigation of Local and Candidate Multi-View Datasets  
**Status:** **INVESTIGATION CONCLUDED; NO LOCAL DATASET AVAILABLE; NO-GO DECLARED**

---

## 1. Executive Summary: The Non-Negotiable Mandate of Scientific Honesty

Section 6 and Section 13 of the Phase 7.4.2 mandate establish an unequivocal rule of scientific integrity:
> *"Do not represent different ADE20K images as multiple views of one room. Do not use synthetic image shifts (np.roll). If genuine multi-view data is unavailable, report NO-GO for multi-view rather than manufacturing a substitute."*

This audit report documents the comprehensive investigation conducted across all workspace drives, directories, and candidate repositories to establish whether legitimate multi-view indoor sequences exist locally, and details the requirements for legitimate future acquisition.

---

## 2. Exhaustive Local Directory Audit

Every directory in `F:\RoomAI` and system storage was inspected for image sequences, camera trajectory logs, and RGB-D depth streams:

| Local Directory | Discovered Contents | Multi-View Suitability | Audit Finding |
| :--- | :--- | :---: | :--- |
| `ScanNet-master/` | C++ source tools, shaders, benchmark scripts (`Alignment`, `BenchmarkScripts`, `Calibrate`) | **NONE (Code Only)** | Contains utility code and evaluation tools, but **zero raw RGB-D scan data** (`.sens` files). |
| `archive (2)/ADEChallengeData2016` | 2,000 validation + 20,210 training images with semantic PNG masks | **NONE (Single Images)** | Every image is an independent, uncalibrated photograph from a unique viewpoint and scene. |
| `data/source/homeobjects-3K` | 3,000 YOLO detection images of living rooms and bedrooms | **NONE (Single Images)** | Independent web-scraped images for 2D object detection; no camera pose information. |
| `data/deepfurniture` | Cropped furniture images for classification and bounding boxes | **NONE (Single Objects)** | Isolated single furniture crops without multi-camera baselines. |
| `data/indoor_canonical` | Curated Open Images + ADE20K subsets for A3 YOLO11s detector training | **NONE (Detection Only)** | 2D bounding-box detection splits without multi-view pairs. |
| `RoomAI_Data/fiftyone_zoo` | Open Images v7 image pools | **NONE (Detection Only)** | Single-image detection datasets. |
| `artifacts/phase_7_3_multiview/` | Historical outputs from Phase 7.3 | **INVALID (Flawed Historical Run)** | Evaluated arbitrary combinations of unrelated ADE20K scenes (`abbey` + `airplane_cabin`). |

**Conclusion of Local Audit:** There are **zero** genuine multi-view indoor sequences on the local workstation.

---

## 3. External Multi-View Candidate Evaluation

To prepare for future acquisition, four prominent multi-view indoor datasets were evaluated against RoomAI requirements:

### 3.1 ScanNet / ScanNet++ (Princeton / Stanford / TUM)
- **Modality:** Real-world iPad RGB-D handheld video sequences with sensor camera poses, calibrated intrinsics, and 3D surface meshes.
- **Suitability for RoomAI:** Exceptional (Gold Standard for indoor 3D reconstruction).
- **Access Requirements:** Requires an institutional academic email, submission of a signed Terms of Use agreement, and API key authorization to access the 1.2 TB download script (`download-scannet.py`).
- **Local Availability:** Not downloaded.

### 3.2 Hypersim (Apple)
- **Modality:** Photorealistic synthetic indoor architectural scenes with perfect ground-truth metric depth, surface normals, and camera trajectories.
- **Suitability for RoomAI:** High (ideal for controlled synthetic benchmarking).
- **Access Requirements:** Publicly hosted on Apple AWS S3; requires AWS CLI and hundreds of gigabytes of disk storage.
- **Local Availability:** Not downloaded.

### 3.3 HM3D (Habitat-Matterport 3D)
- **Modality:** 1,000 high-resolution photorealistic 3D scans of real residential homes.
- **Suitability for RoomAI:** Very High for virtual fly-through multi-view generation.
- **Access Requirements:** Requires Hugging Face license approval and Habitat-Sim rendering framework.
- **Local Availability:** Not downloaded.

### 3.4 Replica Dataset (Meta AI)
- **Modality:** 18 photorealistic indoor environments with dense 3D meshes.
- **Suitability for RoomAI:** Moderate; requires offline rendering via Habitat-Sim to produce image sequences.
- **Local Availability:** Not downloaded.

---

## 4. Invalidation of Historical Multi-View Benchmarks

The following historical benchmarks are formally declared **INVALID AND DEPRECATED**:
1. **Phase 7.3 Multi-View Benchmark (`scripts/benchmark_phase_7_3_multiview.py`):**  
   Evaluated unrelated images (`ADE_val_00000001` [abbey], `00000002` [abbey], `00000005` [airplane_cabin]). The reported registration metrics were spurious matches between unrelated environments.
2. **Phase 7.4 Multi-View Benchmark (`scripts/benchmark_phase_7_4_evaluation.py`):**  
   Evaluated `np.roll()` cyclical pixel wraps. The claimed $93.3\%$ registration success and $0.110\text{ m}$ RMSE were artifacts of matching an image to a cyclically translated copy of itself.

---

## 5. Auditor's Verdict & Sub-Gate D/E Ruling

In strict compliance with the Phase 7.4.2 specification:
1. **No Substitutes Manufactured:** No synthetic shifts or fabricated image pairings were created.
2. **Ruling:** **Sub-Gate D (Genuine Multi-View Registration) and Sub-Gate E (Multi-View Fused Reconstruction) are declared NO-GO**.
3. **Product Guidance:** RoomAI must operate exclusively in **Single-View 3D Reconstruction Mode** for all live demonstrations and portfolio presentations until an officially licensed multi-camera dataset is downloaded and validated.

