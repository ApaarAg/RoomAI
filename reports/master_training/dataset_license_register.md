# ROOMAI MASTER TRAINING — DATASET LICENSE REGISTER & LEGAL COMPLIANCE

**Author:** Principal 3D Vision Research Scientist, Dataset Engineer & Legal Compliance Lead  
**Date:** 2026-11-08  
**Project:** RoomAI (`F:\RoomAI`)  
**Phase:** Phase 01 (Dataset Acquisition & Licensing Audit)  
**Status:** COMPLETE & LEGALLY QUARANTINED  

---

## 1. DUAL-TRACK ARCHITECTURAL POLICY

To protect RoomAI's production deployment while enabling rigorous benchmark validation against real-world indoor environments, all datasets are strictly partitioned into two isolated operational tracks:

```
[ ROOMAI DATASET SEPARATION FRAMEWORK ]

┌────────────────────────────────────────────────────────┐
│ TRACK A: COMMERCIAL CANDIDATE PIPELINE                 │
│ Permitted in production model training & deployment    │
│ Datasets: Hypersim (Apple Inc.)                        │
│ License: Creative Commons Attribution-ShareAlike 3.0   │
└────────────────────────────────────────────────────────┘
                           ▲
                           │ ZERO WEIGHT / DATA LEAKAGE (STRICT FIREWALL)
                           ▼
┌────────────────────────────────────────────────────────┐
│ TRACK B: NON-COMMERCIAL RESEARCH & BENCHMARK TRACK     │
│ Restricted to internal validation & generalization     │
│ Datasets: ScanNet v2 (Stanford), ARKitScenes (Apple),  │
│           ScanNet++ (TUM)                              │
│ License: Non-Commercial Research Agreements Only       │
└────────────────────────────────────────────────────────┘
```

---

## 2. DETAILED DATASET LICENSING AUDIT

### 2.1 Dataset A: Hypersim (Track A — Production Candidate)
- **Licensor:** Apple Inc.
- **Official Repository:** `https://github.com/apple/ml-hypersim`
- **Applicable License:** **Creative Commons Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0)**
- **Commercial Usability:** **PERMITTED**
  - Section 3 allows commercial reproduction, distribution, and public performance of the work and derivative works.
- **Legal Compliance Obligations:**
  1. **Attribution (BY):** RoomAI must provide proper citation and credit to Apple Inc. and original 3D asset creators (Evermotion).
  2. **ShareAlike (SA):** If RoomAI creates and distributes an altered, adapted, or derivative dataset or database derived from Hypersim, that derivative database must be released under the same or compatible license.
  3. **Model Weight Consideration:** To avoid downstream license contamination of proprietary software, any model weights trained on Hypersim are documented with full provenance.
- **Current Acquisition Status:** 5 scenes (231 frames, 3.69 GB physical archives) verified on disk. Staged expansion catalog defined for 25 additional scenes.

### 2.2 Dataset B: ARKitScenes (Track B — Non-Commercial Research)
- **Licensor:** Apple Inc.
- **Official Repository:** `https://github.com/apple/ARKitScenes`
- **Applicable License:** **Apple Custom Non-Commercial Research License**
- **Commercial Usability:** **PROHIBITED**
  - Grants rights for personal, non-commercial research, and educational purposes only.
  - Expressly excludes commercial deployment. Commercial licensing requires formal written agreement via `ARKitScenes-license@group.apple.com`.
- **Operational Quarantine:** Data from ARKitScenes may only be used as a zero-shot generalization test set in Track B. No model trained on ARKitScenes weights will ever be deployed into the RoomAI production application.

### 2.3 Dataset C: ScanNet v2 (Track B — Non-Commercial Research)
- **Licensor:** Stanford University, Princeton University, Technical University of Munich
- **Official Repository:** `https://www.scan-net.org/`
- **Applicable License:** **ScanNet Terms of Use (Non-Commercial Research)**
- **Commercial Usability:** **STRICTLY PROHIBITED**
  - Clause 2: "The dataset may be used for non-commercial research and educational purposes only."
  - Prohibits commercial redistribution, commercial model training, or commercial inference services.
- **Operational Quarantine:** 8 sequences (859 standardized frames) exist on disk. They are quarantined strictly for held-out cross-dataset benchmark evaluation.

### 2.4 Dataset D: ScanNet++ (Track B — Non-Commercial Research)
- **Licensor:** Technical University of Munich (TUM)
- **Official Repository:** `https://scannetpp.mlsg.cit.tum.de/`
- **Applicable License:** **TUM Research Agreement**
- **Commercial Usability:** **STRICTLY PROHIBITED**
  - Access requires institutional registration and click-through agreement restricting use to academic non-commercial research.
- **Operational Quarantine:** Kept in Track B catalog. Not downloaded for production model training.

---

## 3. SUMMARY OF COMPLIANCE REGISTER

| Dataset | Provider | License Type | Commercial Rights | Track Assignment | Permitted in Production Checkpoint? |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Hypersim** | Apple Inc. | CC BY-SA 3.0 | **YES** (with SA & BY) | Track A | **YES** |
| **ScanNet v2** | Stanford / TUM | Academic Agreement | **NO** (Research Only) | Track B | **NO** (Benchmark Only) |
| **ARKitScenes** | Apple Inc. | Custom Non-Commercial | **NO** (Research Only) | Track B | **NO** (Benchmark Only) |
| **ScanNet++** | TUM | Academic Agreement | **NO** (Research Only) | Track B | **NO** (Benchmark Only) |

---

## 4. VERIFICATION SIGN-OFF
This licensing register establishes legal clarity. No Track B data will be utilized for training candidate checkpoints destined for the RoomAI production application.

