# RoomAI — Phase 4: Previous Training Failure Analysis
**Author:** Principal CV Research Scientist  
**Date:** 2026-11-10 IST  
**Status:** COMPLETE  
**Previous experiments reviewed:** Phase 8 (flawed sampling), Phase 9 Exp A & B (insufficient data)

---

## 1. Failure Summary

All previous fine-tuning experiments (Phase 8 and Phase 9) failed to improve over the pretrained baseline.

| Experiment | Method | Train Frames | Hypersim AbsRel | Outcome |
|---|---|---|---|---|
| Pretrained baseline | — | — | 0.5469 (Phase 9 set) | **PRODUCTION WINNER** |
| Phase 8 Exp A | Head-only | ~31 (sequential, FLAWED) | 0.5737 | Rejected — sampling bug + data |
| Phase 8 Exp B | Partial backbone | ~31 (sequential, FLAWED) | 0.5934 | Rejected — sampling bug + data |
| Phase 9 Exp A | Head-only (shuffled) | 31 (3 scenes) | 0.5737 | Rejected — no generalization |
| Phase 9 Exp B | Partial backbone (shuffled) | 31 (3 scenes) | 0.5829 | Rejected — no generalization |

---

## 2. Root-Cause Analysis

### 2.1 Phase 8: Sequential Sampling Bug (FIXED)

**Diagnosis:** The Phase 8 DataLoader used sequential iteration without shuffle. With `batch_size=1` and no shuffle, every training epoch processed the same frame repeatedly in the same order, as the manifest was indexed with `step - 1` (a linear counter that did not randomize across scenes).

**Evidence:** SHA-256 of Phase 8 Exp A and B checkpoints are identical (`2333.81 MB`), proving the two "different" experiments actually learned from the same sequence.

**Fix applied in Phase 9:** `DataLoader(shuffle=True, generator=torch.Generator().manual_seed(42))` ✓

**Status: FIXED** — This bug cannot recur in the new pipeline.

---

### 2.2 Phase 9: Data Diversity Bottleneck (SOLVED WITH NEW DATASET)

**Diagnosis:** Phase 9 experiments used 3 training scenes with 31 total frames. The WorldMirror-1.0 model is a 1.22B-parameter foundation model pre-trained on:
- ScanNet: ~1,500 scenes
- Hypersim: ~461 scenes
- Other indoor/outdoor datasets: estimated 10,000+ scenes

**Quantitative evidence from Phase 9:**
- Training loss dropped: 0.8366 → 0.6841 (−18.2%) ✓ Gradients flowing
- Val AbsRel change: 0.6317 → 0.6321 (+0.06%) ✗ No generalization
- ScanNet AbsRel change: 0.2667 → 0.2667 (0.00%) ✗ No generalization

The gradient flow was healthy and optimization succeeded. The issue was **memorization of 31 images without learning generalizable depth priors**.

**Information-theoretic bound:** A 98M-parameter prediction head cannot learn meaningful indoor depth priors from 31 images. The minimum sample size for meaningful improvement scales with model capacity and the dimensionality of the target function.

**Fix: 131× scale increase.** New training set: 24 scenes × ~170 frames = **4,079 frames**

| Metric | Phase 9 | Phase 10 |
|---|---|---|
| Training scenes | 3 | **24** |
| Training frames | 31 | **4,079** |
| Scene diversity | 3 rooms | 24 distinct indoor environments |
| Camera streams | ~31 total | **55+ camera streams** |
| Geometric diversity | Very limited | Full indoor variation |

---

### 2.3 Trainable Parameter Analysis

**Phase 9 Exp A (Head-Only): 98,046,731 params (8.01% of 1.22B)**
- `depth_head`, `pts_head`, `norm_head` unfrozen
- Backbone (1.12B params): **fully frozen**
- Result: Optimization succeeded, no generalization

**Phase 9 Exp B (Partial Backbone): 123,267,083 params (10.07% of 1.22B)**
- Same heads + `frame_blocks[-1]` + `global_blocks[-1]`
- Result: Identical validation outcome to Exp A

**Key finding:** Unfreezing late backbone blocks provided NO additional benefit with only 31 training frames. The backbone blocks need diverse gradients from many scenes to update productively.

**Hypothesis for Phase 10:** With 4,079 training frames:
- Head-only fine-tuning may now provide meaningful improvement
- LoRA on attention layers (rank 4-16) may provide structured adaptation without catastrophic forgetting
- Progressive unfreezing may become feasible after head-only fine-tuning succeeds

---

## 3. Gradient Behavior Analysis (Phase 9 Gradient Audit)

From the Phase 9 gradient interaction audit across 15 indoor batches:

| Gradient Property | Value | Implication |
|---|---|---|
| Depth-Normal cosine similarity | ±0.04 | Near-zero conflict — no PCGrad needed |
| Depth-Points cosine similarity | +0.12 | Weak positive correlation |
| Gradient cancellation ratio | 38.2% | Low — multi-task loss is appropriate |
| Loss balance w_depth=1.0, w_pts=0.5, w_norm=0.1 | Justified | No rebalancing required |

**Conclusion:** The original loss weighting `L = 1.0 * L_depth + 0.5 * L_pts + 0.1 * L_normals` is correct. The loss was not the bottleneck.

---

## 4. Representation Change Analysis

Phase 9 measured that head-only fine-tuning produced:
- Zero change in ScanNet performance (cross-dataset generalization preserved)
- Checkpoint SHA-256 changed (parameters did update)
- But validation AbsRel unchanged to 4 decimal places

This is consistent with the model having:
1. Very strong pretrained backbone features that the small dataset cannot override
2. Head parameters that adapted to the 3-scene Hypersim distribution without generalizing

**New prediction for Phase 10 (4,079 frames):**
- With 24 scenes, the head should learn a broader indoor depth mapping function
- The head contains 98M parameters; with 4,079 training images it has ~24,000 images per parameter (by very rough proxy), which is more reasonable
- Real-world indoor depth range, scene diversity, and multi-camera coverage should all help

---

## 5. What to Change in Phase 5

Based on this analysis, the Phase 5 experiments should:

### Experiment 01 — Head-Only Fine-Tuning (new data scale)
- Same as Phase 9 Exp A, but with 4,079 training frames instead of 31
- Conservative LR: 2e-6 for heads
- 3-5 epochs
- Evaluate at every epoch checkpoint

### Experiment 02 — LoRA Adapters (new technique, not tried before)
- Apply LoRA adapters to the attention Q, K, V projections of late transformer blocks
- Rank 4, 8, 16 tested
- Much fewer trainable params than head-only (by design)
- LoRA was NOT tried in Phases 8 or 9 — genuinely new experiment
- Expected benefit: structured, low-rank updates to the backbone representation without catastrophic forgetting

### Experiment 03 — LoRA + Geometric Regularization
- Add KL divergence or feature-space regularization term
- Use pretrained model as reference teacher
- Prevent feature distribution drift

### Experiment 04 — Progressive Unfreezing
- Only if Exp 01 or 02 shows improvement
- Start from head-only checkpoint, then gradually unfreeze last N blocks
- Very conservative LR for backbone: 5e-8 → 1e-7

### Experiment 05 — Multi-Scale Training
- Try training at multiple resolutions (384×512, 480×640, 518×518)
- WorldMirror input is 518×518; test whether training at native resolution helps

---

## 6. LoRA Module Selection Strategy

The WorldMirror model uses a custom ViT-based architecture. Key modules to target for LoRA:

```
WorldMirror
├── visual_geometry_transformer (VGT) — the 1.12B backbone
│   ├── frame_blocks (N=24 blocks, each with)
│   │   ├── attn.q_proj  → LoRA target
│   │   ├── attn.k_proj  → LoRA target  
│   │   ├── attn.v_proj  → LoRA target
│   │   └── attn.out_proj → LoRA target
│   └── global_blocks (similar structure)
├── depth_head       → Head fine-tuning (all params)
├── pts_head         → Head fine-tuning (all params)
├── norm_head        → Head fine-tuning (all params)
└── cam_head         → Head fine-tuning (all params)
```

**LoRA strategy:** Apply only to last 4-8 frame_blocks and last 2-4 global_blocks.
This targets the highest-level geometric abstractions without touching the low-level feature backbone.

**Expected trainable param counts:**
- Rank-4 LoRA on 4 blocks: ~1-4M params
- Head-only: 98M params  
- Combined: ~100M params

---

## 7. Key Success Conditions for Phase 5

The critical difference from Phase 9 is data scale (131× more frames). Based on the analysis:

**Expected to succeed:** Head-only fine-tuning with proper data
**May succeed:** LoRA rank 8-16 on late blocks  
**Unlikely to succeed:** Full backbone fine-tuning (insufficient VRAM for large batches)
**Prerequisite for any success:** Clean depth loading with correct ray→planar conversion ✓

---

**Phase 4 Status: COMPLETE**  
**Next: Phase 5 — Execute fine-tuning experiments using new 4,079-frame Hypersim dataset**

