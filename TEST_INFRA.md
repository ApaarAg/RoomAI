# E2E Test Infra: RoomAI Product Experience

## Test Philosophy
- Opaque-box and requirement-driven derived from ORIGINAL_REQUEST.md.
- Verification methodology: Category-Partition + Boundary Value Analysis (BVA) + Pairwise Combinatorial Testing + Real-World Workload Testing on genuine ADE20K validation images.
- Non-regression: Must preserve 100% pass rate on all 158 existing unit/integration tests (`python -m pytest -q`).

## Feature Inventory
| # | Feature | Source (Requirement) | Tier 1 | Tier 2 | Tier 3 |
|---|---------|---------------------|:------:|:------:|:------:|
| F1 | Workspace Consolidation (3 Workspaces) | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ |
| F2 | Product Identity, Header & Badges | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ |
| F3 | Image Upload & SHA-256 State Invalidation | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ |
| F4 | Object Detection (A3/B2/C2, Metrics & Checkpoints) | ORIGINAL_REQUEST §R2 | 5 | 5 | ✓ |
| F5 | Vision Q&A (VLM Availability & Honest Error) | ORIGINAL_REQUEST §R2 | 5 | 5 | ✓ |
| F6 | 3D Scene Reconstruction (Metric vs Relative Depth) | ORIGINAL_REQUEST §R2 | 5 | 5 | ✓ |
| F7 | Interactive 3D Viewer & PLY/GLB Export Clarity | ORIGINAL_REQUEST §R2 | 5 | 5 | ✓ |
| F8 | Real-Mode Enforcement & GPU VRAM Compliance | ORIGINAL_REQUEST §R3 | 5 | 5 | ✓ |

## Test Architecture
- Test runner: `C:\Users\apaar\miniconda3\envs\tf-gpu\python.exe -m pytest -q`
- E2E Test Suite location: `tests/test_e2e_product_experience.py`
- Pass/Fail semantics: Exit code 0, 0 failures, 0 regressions across all 158 tests + all new E2E test cases.

## Real-World Application Scenarios (Tier 4)
| # | Scenario | Features Exercised | Complexity |
|---|----------|--------------------|------------|
| 1 | Living Room Real Perception (`ADE_val_00001531.jpg`) | F1, F3, F4 (A3), F6, F7, F8 | High |
| 2 | Bedroom Real Perception (`ADE_val_00000118.jpg`) | F1, F3, F4 (A3), F6, F7, F8 | High |
| 3 | Kitchen Real Perception (`ADE_val_00000459.jpg`) | F1, F3, F4 (A3), F6, F7, F8 | High |
| 4 | Dining Room Real Perception (`ADE_val_00000314.jpg`) | F1, F3, F4 (A3), F6, F7, F8 | High |
| 5 | Home Office Real Perception (`ADE_val_00000420.jpg`) | F1, F3, F4 (A3), F6, F7, F8 | High |

## Coverage Thresholds
- Tier 1: >= 5 per feature (>= 40 tests)
- Tier 2: >= 5 per feature (boundary/error handling)
- Tier 3: Pairwise combinations across upload, detection, VLM, and 3D reconstruction
- Tier 4: >= 5 realistic room scenarios on genuine ADE20K validation data
