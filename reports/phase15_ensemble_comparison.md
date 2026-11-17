# RoomAI Phase 15 — Multi-Model Specialist Architecture & Ensemble Comparison Report

**Author**: Senior Computer Vision Engineer & ML Systems Auditor  
**Date**: October 25, 2026  
**Repository**: `C:/Users/apaar/Desktop/RoomAI`  
**Evaluation Scope**: 154 Locked Validation Scenes (Canonical Exterior Architecture)  
**Hardware Environment**: NVIDIA GeForce RTX 4060 Laptop GPU (8,188 MiB VRAM), PyTorch 2.5.1+cu121, Python 3.10 (Conda `tf-gpu`)

---

## 1. Executive Summary & Objective

In **Phase 15**, we conducted a rigorous empirical evaluation to determine whether combining RoomAI's strongest general exterior object detector (**B6**) with a dedicated, class-specific **Porch Specialist** improves exterior-object detection over general-detector baselines (**B2** production default, **B6** candidate), or whether the operational costs (GPU latency, memory footprint, architectural complexity) outweigh the empirical gains.

### Key Findings
1. **Porch Detection Gains:**
   - General baseline **B2** completely failed on porches in validation scenes: **0.00% F1** (0 TP, 0 FP, 24 FN).
   - Expanded general detector **B6** detected 2 true positives with 4 false positives (**F1 = 0.1333**, Precision = 0.3333, Recall = 0.0833).
   - **Parallel Ensemble (WBF)** lifted recall to 3 TP (Recall = 0.1250) but emitted 10 false positives due to independent non-contextual proposals (**Porch F1 = 0.1622**, Precision = 0.2308).
   - **Cascaded Inference** achieved the highest porch F1 among multi-model combinations (**Porch F1 = 0.1765**, Precision = 0.3000, Recall = 0.1250) by invoking the specialist only when general contextual entrance cues were present, reducing false alarms by 30%.
2. **System Overhead & Latency Penalty:**
   - Multi-model inference incurs a **+53% to +58% median latency penalty** (B6: 21.10 ms vs Cascaded: 32.31 ms vs Parallel: 33.42 ms) and increases peak allocated VRAM from 146.9 MB to 182.9 MB.
   - Overall validation Macro-F1 improvement for cascaded inference was a modest **+0.0042** (0.5166 → 0.5208) because porches comprise only 0.34% of all annotated exterior objects (24 of 6,997 instances).
3. **Unexpected Digital Image Processing (DIP) Superiority:**
   - Applying **Contrast Limited Adaptive Histogram Equalization (CLAHE)** as an image pre-processing ablation to the single general B6 model achieved **Porch F1 = 0.1935** (Precision = 0.4286, Recall = 0.1250) and an overall **Macro-F1 of 0.5244**, outperforming *all* multi-model ensemble architectures at only **22.13 ms** median latency (+1.03 ms / +4.9% overhead).
4. **Specialist Expansion Justification (Stage G):**
   - Further single-class specialists for `pillar_column`, `door`, or `facade` are **unjustified** at this time. Adding sequential specialists would push end-to-end latency >60–80 ms while failing to resolve underlying annotation ambiguities and class overlap.

---

## 2. Checkpoint Provenance & Cryptographic Verification

All existing baselines and newly trained candidate checkpoints were audited and verified byte-for-byte using cryptographic MD5 hashes prior to evaluation:

| Model ID | Checkpoint Path | Architecture | Role | File Size (Bytes) | MD5 Checksum |
|---|---|---|---|---|---|
| **B2** | [exterior_detector_b2_yolo11s_best.pt](file:///C:/Users/apaar/Desktop/RoomAI/checkpoints/exterior_detector_b2_yolo11s_best.pt) | YOLO11s (10-class) | Production Default (Preserved) | 19,166,810 | `14a931dd0adabc16cbd7b28ed1d17f66` |
| **B5** | [exterior_detector_b5_transfer_best.pt](file:///C:/Users/apaar/Desktop/RoomAI/checkpoints/exterior_detector_b5_transfer_best.pt) | YOLO11s (10-class) | Transfer Baseline | 19,166,810 | `e12b4181fee244a3569dfa782ba3c3c3` |
| **B6** | [exterior_detector_b6_expanded_best.pt](file:///C:/Users/apaar/Desktop/RoomAI/checkpoints/exterior_detector_b6_expanded_best.pt) | YOLO11s (10-class) | General Candidate (Expanded) | 19,166,810 | `f5ceea56b0610ea14b647014e48152dc` |
| **Porch Spec** | [exterior_specialist_porch_yolo11s_best.pt](file:///C:/Users/apaar/Desktop/RoomAI/checkpoints/exterior_specialist_porch_yolo11s_best.pt) | YOLO11s (1-class) | Porch Specialist Candidate | 19,159,130 | `4261fc1ad5f6fd9ac29c0499e2adddfd` |

None of the baseline checkpoints were retrained or modified during Phase 15.

---

## 3. Architecture Comparison Matrix

All 6 evaluated architectures were evaluated on the identical 154 locked validation scenes (6,997 ground truth instances across 10 canonical exterior classes). Evaluation parameters: IoU match threshold = 0.50, General confidence threshold = 0.25, Specialist threshold = 0.20, Image resolution = 640×640.

| Metric / Dimension | Arch 1: B2 General Only | Arch 2: B6 General Only | Arch 3: Parallel Ensemble | Arch 4: Cascaded Inference | Arch 5: High-Res Crop | Arch 6: DIP CLAHE + B6 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Underlying Models** | B2 Only | B6 Only | B6 + Porch Specialist | B6 + Porch Specialist | B6 + Porch Specialist | B6 Only + CLAHE |
| **Macro-F1 (All 10 Classes)** | 0.4632 | 0.5166 | 0.5194 | **0.5208** | 0.5157 | **0.5244** |
| **Micro-F1 (All Objects)** | 0.6340 | 0.6540 | 0.6538 | 0.6539 | 0.6538 | 0.6499 |
| **Micro-Precision** | 0.6985 | 0.6874 | 0.6859 | 0.6865 | 0.6864 | 0.6841 |
| **Micro-Recall** | 0.5804 | 0.6237 | 0.6247 | 0.6244 | 0.6243 | 0.6190 |
| **Overall Mean IoU (TP)** | 0.8195 | 0.8152 | 0.8149 | 0.8151 | 0.8149 | 0.8153 |
| **Porch Precision** | 0.0000 | 0.3333 | 0.2308 | 0.3000 | 0.2500 | **0.4286** |
| **Porch Recall** | 0.0000 | 0.0833 | **0.1250** | **0.1250** | 0.0833 | **0.1250** |
| **Porch F1 Score** | 0.0000 | 0.1333 | 0.1622 | **0.1765** | 0.1250 | **0.1935** |
| **Porch TP / FP / FN** | 0 / 0 / 24 | 2 / 4 / 22 | 3 / 10 / 21 | 3 / 7 / 21 | 2 / 6 / 22 | 3 / 4 / 21 |
| **Median Latency (Warm)** | 20.62 ms | 21.10 ms | 33.42 ms | 32.31 ms | 35.35 ms | **22.13 ms** |
| **p95 Latency (Warm)** | 77.45 ms | 36.26 ms | 50.99 ms | 51.40 ms | 60.49 ms | 36.69 ms |
| **Peak GPU VRAM Allocated**| 110.18 MB | 146.93 MB | 182.94 MB | 182.94 MB | 182.96 MB | 182.88 MB |
| **Peak GPU VRAM Reserved** | 136.00 MB | 178.00 MB | 214.00 MB | 214.00 MB | 214.00 MB | 214.00 MB |
| **Model Cold-Load Time** | 74.57 ms | 120.37 ms | 185.12 ms | 185.12 ms | 185.12 ms | 120.37 ms |

---

## 4. Per-Class Performance Breakdown

Detailed class-level precision, recall, and F1 across the key architectures:

### 4.1 Arch 2: B6 General Only
```
Class            GT Count      TP      FP      FN   Precision     Recall         F1   Mean IoU
building               40      18      19      22      0.4865     0.4500     0.4675     0.8256
facade                495     144     132     351      0.5217     0.2909     0.3735     0.8123
window               3845    2950     935     895      0.7593     0.7672     0.7633     0.8291
door                  372     169     169     203      0.5000     0.4543     0.4761     0.7899
roof_cornice          445     296     162     149      0.6463     0.6652     0.6556     0.7719
balcony               345     216      95     129      0.6945     0.6261     0.6585     0.8174
porch                  24       2       4      22      0.3333     0.0833     0.1333     0.7816
pillar_column         494     151     224     343      0.4027     0.3057     0.3475     0.7788
shutter_blind         901     393     238     508      0.6228     0.4362     0.5127     0.8166
molding_deco           36      25      15      11      0.6250     0.6944     0.6579     0.7963
Macro-Average:                                         0.5592     0.4773     0.5166
```

### 4.2 Arch 4: Cascaded Specialist (B6 + Porch Specialist)
```
Class            GT Count      TP      FP      FN   Precision     Recall         F1   Mean IoU
building               40      18      19      22      0.4865     0.4500     0.4675     0.8256
facade                495     144     132     351      0.5217     0.2909     0.3735     0.8123
window               3845    2950     935     895      0.7593     0.7672     0.7633     0.8291
door                  372     169     169     203      0.5000     0.4543     0.4761     0.7899
roof_cornice          445     296     162     149      0.6463     0.6652     0.6556     0.7719
balcony               345     216      95     129      0.6945     0.6261     0.6585     0.8174
porch                  24       3       7      21      0.3000     0.1250     0.1765     0.7770
pillar_column         494     151     224     343      0.4027     0.3057     0.3475     0.7788
shutter_blind         901     393     238     508      0.6228     0.4362     0.5127     0.8166
molding_deco           36      25      15      11      0.6250     0.6944     0.6579     0.7963
Macro-Average:                                         0.5559     0.4815     0.5208
```

### 4.3 Arch 6: Digital Image Processing (CLAHE) + B6 General
```
Class            GT Count      TP      FP      FN   Precision     Recall         F1   Mean IoU
building               40      18      20      22      0.4737     0.4500     0.4615     0.8242
facade                495     141     136     354      0.5090     0.2848     0.3653     0.8105
window               3845    2943     932     902      0.7595     0.7654     0.7624     0.8297
door                  372     167     166     205      0.5015     0.4489     0.4738     0.7891
roof_cornice          445     298     165     147      0.6436     0.6697     0.6564     0.7731
balcony               345     217      98     128      0.6889     0.6290     0.6576     0.8179
porch                  24       3       4      21      0.4286     0.1250     0.1935     0.7801
pillar_column         494     155     236     339      0.3964     0.3138     0.3503     0.7789
shutter_blind         901     396     244     505      0.6188     0.4395     0.5143     0.8169
molding_deco           36      25      13      11      0.6579     0.6944     0.6757     0.7981
Macro-Average:                                         0.5678     0.4821     0.5244
```

---

## 5. Architectural Deep-Dive & Fusion Mechanics

### 5.1 Mode B: Parallel Ensemble vs Mode C: Cascaded Inference
In **Parallel Ensemble (Mode B)**, both models run unconditionally on the full 640×640 image. Coordinate fusion via Weighted Boxes Fusion (WBF) with weights $[0.8, 1.2]$ (favoring the specialist for porch) successfully combined overlapping porch boxes and added a consensus bonus (+0.05). However, when the specialist evaluated scenes without porches, it hallucinated porticos over open driveways, ground-floor retail entrances, and basement stairwells, emitting **10 false positives** ($P = 0.2308$).

In **Cascaded Inference (Mode C)**, the specialist is gated by the general detector:
$$\text{Trigger Specialist} \iff (\exists \, \text{door} \ge 0.20) \lor (\exists \, \text{pillar\_column} \ge 0.20) \lor (\exists \, \text{porch} \ge 0.10)$$
By requiring an entrance cue or a weak porch hypothesis from the general detector, Cascaded Inference suppressed 3 false-positive hallucinations on scenes devoid of entrance structures, lifting Precision to **0.3000** and Porch F1 to **0.1765**.

### 5.2 Mode D: High-Resolution Crop Inference Failure
Hypothesis: Cropping the lower half ($y \ge 0.35 \times H$) around entrance cues and scaling to 640×640 would provide fine-grained textural detail for small columns and pediments.  
**Empirical Finding**: Porch F1 dropped to **0.1250** (P = 0.2500, R = 0.0833).  
**Root Cause**: Porches are defined by architectural context relative to the whole building (the roofline above, upper windows, foundation elevation). Removing the upper building facade caused the detector to confuse basement egress wells and plain doorway awnings with porches.

### 5.3 Cross-Class Conflicts and Provenance Tracking
Every detection emitted by `SpecialistEnsembleDetector` contains an explicit `model_provenance` attribute in `DetectionItem`:
- `general_b6`: Standard general detector output.
- `specialist_porch`: Emitted solely by the Porch specialist.
- `ensemble_consensus_wbf`: Fused box where both models agreed.

Cross-class containment rules were strictly enforced:
- A `porch` contained within a large `facade` or `building` box is preserved as a nested architectural component.
- An overlapping `balcony` and `porch` at the same ground level triggers IoU-based resolution, preferring the porch if ground-adjacent and the balcony if suspended above floor level.

---

## 6. Error Gallery & Visual Analysis

Visual inspection of annotated scenes in [reports/phase15_error_gallery](file:///C:/Users/apaar/Desktop/RoomAI/reports/phase15_error_gallery/) reveals four distinct error modes:

1. **Recessed & Shadowed Porticos (`oi_curated_porch_val_0110.jpg`)**:
   - Deep overhangs cast steep shadows over the front entrance.
   - B6 general detector fails to detect the entrance completely (FN).
   - The Porch Specialist detects the columns and portico floor with confidence 0.41, successfully recovering the TP.
   - CLAHE pre-processing similarly lifts contrast in the shadowed recess, enabling B6 to detect the structure directly.
2. **Double-Storey Verandas (`oi_curated_porch_val_0111.jpg`)**:
   - The ground floor is an open porch while the second floor is a covered balcony.
   - Ground truth annotates only the ground-floor entrance as porch.
   - Parallel ensemble encloses the entire two-storey structure in a single porch box, triggering a localization FP ($0.25 < \text{IoU} < 0.50$).
   - Cascaded inference with deduplication truncates the box at the first-floor ceiling line, producing an accurate match.
3. **Window Canopies and Carports (`oi_curated_porch_val_0112.jpg`)**:
   - Independent specialist inference mistook an ornamental metal window canopy for a mini-porch (FP).
   - Cascaded routing did not suppress this because a nearby side door was present.
4. **Single-Step Stoops (`oi_curated_porch_val_0113.jpg`)**:
   - Ambiguous architectural definitions between a minimal concrete stoop and a porch.
   - Annotation in ground truth omitted the stoop, but the specialist detected it at confidence 0.28 (labeled FP against ground truth, though visually plausible).

---

## 7. Stage G — Expansion Justification for Additional Specialists

We formally audited whether additional single-class specialists should be trained for `pillar_column`, `door`, `balcony`, `roof_cornice`, or `facade`:

| Candidate Class | Val GT Count | B6 F1 | Dominant Error Mode | Specialist Justification | Next Recommended Action |
|---|:---:|:---:|---|:---:|---|
| **pillar_column** | 494 | 0.3475 | 224 FP (flat pilasters, window mullions) | **Unjustified** | Requires 3D depth/stereo cues, not a 2D bounding box specialist. Open Images lacks distinct pilaster labels. |
| **door** | 372 | 0.4761 | Confusion with full-length french windows | **Unjustified** | Shared visual features with windows. Specialist would suffer same transparency confusion. |
| **facade** | 495 | 0.3735 | Bounding box cannot represent concave wall boundaries | **Unjustified** | Bounding box detection is fundamentally ill-suited for facades; semantic segmentation is required. |
| **roof_cornice**| 445 | 0.6556 | High baseline performance | **Unjustified** | Baseline is already strong (0.6556 F1); diminishing returns. |
| **balcony** | 345 | 0.6585 | High baseline performance | **Unjustified** | Baseline is strong (0.6585 F1); minimal room for improvement. |

**Architectural Recommendation:** Do **NOT** train individual specialists for these classes. Stacking $N$ single-class models would linearly scale GPU memory and inference latency ($N \times 12\text{ ms} > 60\text{ ms}$), completely degrading interactive Streamlit performance without solving the root causes (pilaster depth vs flat molding, glass door vs window).

---

## 8. Stage H — Production Integration & Safety Verification

1. **Production Default Preserved:**
   - **Model B2** remains the production default (Index 0 in `ui/analysis_view.py`).
   - The Cascaded Specialist Ensemble and Model B6 are registered as optional **Experimental Profiles**.
2. **Backward Compatibility & Rollback:**
   - The entire multi-model pipeline is disableable and configurable via `config/model_config.yaml` and UI selectbox.
   - If specialist checkpoints are missing, the service fails gracefully with descriptive warnings or falls back to the general detector without crashing.
3. **Deterministic Mock Mode:**
   - Mock mode remains strictly isolated and deterministic (`mock_mode=True`), returning verified mock detections without GPU or disk dependencies.
4. **Automated Test Suite:**
   - Created [tests/test_specialist_ensemble.py](file:///C:/Users/apaar/Desktop/RoomAI/tests/test_specialist_ensemble.py) covering 8 comprehensive test cases (checksums, production defaults, IoU matching, mock mode, same-class deduplication, provenance tracking, missing checkpoint handling, and real CUDA inference).
   - Ran complete regression test suite: **100 passed in 33.69s**.
5. **Streamlit Upload-to-Result Workflow:**
   - Simulated full end-to-end user image upload and inference through `DetectionService` with `specialist_exterior_cascaded`. Successful real inference completed on CUDA, generating annotated visualization and timing telemetry.

---

## 9. Final Decision & Recommendations

### Final Decision: Option 2
> **The multi-model specialist ensemble improvement is too small and costs too much latency and memory to justify replacing the production default.**

### Evidence Basis
1. **Marginal Overall Utility (+0.0042 Macro-F1):**  
   Cascaded inference marginally improves Porch F1 from 0.1333 to 0.1765 (+0.0432) and Macro-F1 from 0.5166 to 0.5208 (+0.0042). This tiny macro gain is due to class sparsity (porches represent <0.4% of objects in exterior scenes).
2. **Severe Latency Overhead (+53% to +58%):**  
   Executing dual YOLO11s forward passes increases median per-image latency from **21.10 ms to 32.31 ms**, reducing maximum continuous throughput from 47.4 FPS to 30.9 FPS.
3. **DIP CLAHE Outperforms Multi-Model Ensemble:**  
   Single-model B6 with CLAHE achieves higher Porch F1 (**0.1935** vs 0.1765) and higher Macro-F1 (**0.5244** vs 0.5208) at **22.13 ms** (only +1.03 ms cost).

### Recommended Next Actions
1. **Keep B2 as Production Default; Maintain Cascaded Specialist as an Experimental Profile.**
2. **Adopt CLAHE Pre-processing in the Exterior Pipeline** as an optional image-enhancement toggle to assist shadow-recessed porticos without multi-model overhead.
3. **Prioritize Targeted Data Expansion over Architecture Complexity:** Before training any additional specialist models, collect 500+ high-resolution, multi-angle entrance images with explicit annotations distinguishing true 3D porticos from flat pilasters and window awnings.
