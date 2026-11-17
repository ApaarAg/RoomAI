# RoomAI Model Registry — Immutable Artifact Freeze

## Registry Specification

**Version**: 1.0.0
**Frozen At**: 2026-11-10T00:00:00Z
**Registry File**: `models/registry/model_registry.yaml`
**Artifact Manifest**: `models/manifests/artifact_manifest.json`

## Registered Models

### 1. WorldMirror-1.0 Baseline (Reference)

| Property | Value |
|----------|-------|
| **Model ID** | `worldmirror1_baseline` |
| **Alias** | `stage1_reference` |
| **Version** | `1.0.0-pretrained` |
| **Status** | `REFERENCE` |
| **Role** | Baseline and rollback foundation model |
| **Checkpoint** | `research/3d_reconstruction/checkpoints/production_candidate/production_model.ckpt` |
| **SHA-256** | `dbc9c9b0d4fae796ce5dd142bdde7e491057e484db313fef0397cac0c33970e5` |
| **File Size** | 2,839,184,102 bytes (2.64 GB) |
| **Source Commit** | `c61300e47ce06db2f02c8e2551aae68be7ae3a0b` (HunyuanWorld-Mirror) |
| **Dataset Manifest** | `86679e6aae422e57` (indoor_verified_manifest.json) |

**Architecture**:
- Backbone: DINOv2 ViT-L/14 (909.11M params, frozen)
- Heads: Camera (216.17M), Points (32.65M), Depth (32.65M), Normals (32.65M)
- **Total**: 1,223.25M parameters
- **GS Head**: Disabled
- **Trainable Partition**: None (fully pretrained)

**Capabilities**:
- Single-view: ✅
- Multi-view: ✅ (up to 6 views)
- Metric depth: ✅
- Surface normals: ✅
- 3D Gaussian Splatting: ❌
- GLB export: ✅
- PLY export: ✅

**Evaluation Metrics** (from candidate_manifest.json):

| Dataset | AbsRel | RMSE (m) | Median AE (m) | δ<1.25 | Normal MAE (°) | Chamfer (m) |
|---------|--------|----------|---------------|--------|----------------|-------------|
| Overall | 0.5953 | 2.8929 | 1.9076 | 0.0584 | 62.66 | 1.1704 |
| ScanNet | 0.5456 | 1.5277 | - | - | - | 0.4850 |
| Hypersim | 0.7452 | 7.0159 | - | - | - | 3.2401 |

**Known Limitations**:
- Pretrained baseline does not generalize to indoor domain (AbsRel > 0.5)
- No Gaussian Splatting head
- Sim3 scale ambiguity on real scans (4.07×)
- Normal estimation unreliable (MAE > 60°)

---

### 2. WorldMirror-1.0 Stage 1 (Validation Candidate)

| Property | Value |
|----------|-------|
| **Model ID** | `worldmirror1_lora_r8` |
| **Alias** | `stage1_candidate` |
| **Version** | `1.0.0-stage1-r8` |
| **Status** | `VALIDATION_CANDIDATE` |
| **Role** | Best reported Stage 1 candidate |
| **Checkpoint** | `research/3d_reconstruction/checkpoints/stage1/best_stage1.ckpt` |
| **SHA-256** | `fa12e006da6d1dbdbae810ee3995e9659dd8da5dad16dc9fe8380c9daa9481a2` |
| **File Size** | 2,839,164,068 bytes (2.64 GB) |
| **Source Commit** | `c61300e47ce06db2f02c8e2551aae68be7ae3a0b` |
| **Dataset Manifest** | `86679e6aae422e57` (indoor_verified_manifest.json) |
| **Training Config** | `research/3d_reconstruction/configs/finetune_stage1.yaml` |

**Architecture**:
- Backbone: DINOv2 ViT-L/14 (909.11M params, **frozen**)
- Heads: Camera (216.17M), Points (32.65M), Depth (32.65M), Normals (32.65M)
- **Total**: 1,223.25M parameters
- **GS Head**: Disabled
- **Trainable Partition**: **314.14M params** (geometric heads only, NOT LoRA)

**Critical Finding**: This is a **fully merged model**, NOT a separable LoRA adapter.
- No `lora.*` or `adapter.*` keys in state_dict
- All 314.14M head parameters fine-tuned end-to-end
- Cannot be separated into base + adapter

**Reported Metrics** (pilot split, NOT final test):

| Metric | Value |
|--------|-------|
| Hypersim AbsRel | 0.4526 |
| ScanNet AbsRel | 0.2835 |
| PointMap Score | 2.8035 |

**Validation Split Metrics** (indoor_verified_manifest.json, 3 views):

| Metric | Value |
|--------|-------|
| PointMap Accuracy (m) | 2.9590 |
| Depth AbsRel | 1.2720 |
| Depth RMSE (m) | 5.2839 |
| Normal MAE (°) | 67.46 |
| Latency (ms) | 275.95 |
| Peak VRAM (MB) | 3390.88 |

**Held-Out Test Results** (Phase 7/9 evaluation):
- ScanNet AbsRel: **5.53** (generalization failure)
- Overall AbsRel: **4.33**
- Status: **REJECTED — Does not generalize**

**Certification Status**: ⚠️ **NOT PERMANENTLY CERTIFIED**
- Reported metrics were on pilot split
- Final locked test split NOT evaluated under frozen protocol
- Must pass frozen test evaluation before production certification

---

### 3. WorldMirror-1.0 Stage 2 (Experimental)

| Property | Value |
|----------|-------|
| **Model ID** | `worldmirror1_stage2` |
| **Alias** | `stage2_experimental` |
| **Version** | `1.0.0-stage2-3dgs` |
| **Status** | `EXPERIMENTAL` |
| **Role** | Optional multi-view 3D Gaussian Splatting |
| **Checkpoint** | `research/3d_reconstruction/checkpoints/stage2/stage2_real_verified.ckpt` |
| **SHA-256** | `e1848950c17e23ab7f0a05771de519e61c484ebffe916b9320c61c5f5c239ece` |
| **File Size** | 2,644,926,673 bytes (2.46 GB) |
| **Source Commit** | `c61300e47ce06db2f02c8e2551aae68be7ae3a0b` |
| **Training Config** | `research/3d_reconstruction/configs/finetune_stage2.yaml` |

**Architecture**:
- Stage 1 weights (frozen): 1,223.25M params
- GS Head: 32.67M params
- GS Renderer: 0.30M params
- **Total**: 1,256.22M parameters

**Evaluation**:
- Mean Gaussians: 146,207
- Latency: 590.6 ms
- Status: **REJECTED — Inherits Stage 1 distortion**

**Constraint**: Stage 2 MUST remain optional. Never merge into Stage 1 pipeline.

---

## Checkpoint Integrity Audit

### Verified Checkpoints

| Checkpoint | Size (bytes) | SHA-256 | Status |
|------------|-------------|---------|--------|
| production_model.ckpt | 2,839,184,102 | dbc9c9b0d4fae796ce5dd142bdde7e491057e484db313fef0397cac0c33970e5 | ✅ Verified |
| best_stage1.ckpt | 2,839,164,068 | fa12e006da6d1dbdbae810ee3995e9659dd8da5dad16dc9fe8380c9daa9481a2 | ✅ Verified |
| stage2_real_verified.ckpt | 2,644,926,673 | e1848950c17e23ab7f0a05771de519e61c484ebffe916b9320c61c5f5c239ece | ✅ Verified |

### Duplicate Detection

| File | SHA-256 | Note |
|------|---------|------|
| smoke_test_stage1.tmp.ckpt | dbc9c9b0d4fae796ce5dd142bdde7e491057e484db313fef0397cac0c33970e5 | **Byte-identical copy** of production_model.ckpt |

---

## LoRA Parameter Partition Reconciliation

**Claimed**: "LoRA Rank 8 adapter with ~8M trainable params"

**Actual** (from checkpoint inspection):
| Component | Parameters | Trainable |
|-----------|------------|-----------|
| visual_geometry_transformer (backbone) | 909,112,320 | ❌ Frozen |
| cam_head | 216,174,610 | ✅ Fine-tuned |
| pts_head | 32,654,628 | ✅ Fine-tuned |
| depth_head | 32,654,562 | ✅ Fine-tuned |
| norm_head | 32,654,628 | ✅ Fine-tuned |
| **Total Trainable** | **314,138,428** | **314M (not 8M)** |

**Conclusion**: No LoRA adapter exists in the checkpoint. This is a full merged model with all geometric heads fine-tuned. The "LoRA Rank 8" designation refers to the training configuration intent, not the checkpoint structure.

---

## Aliases (Immutable)

```yaml
stage1_reference: "worldmirror1_baseline"      # Never change silently
stage1_candidate: "worldmirror1_lora_r8"       # Requires test split eval
stage2_experimental: "worldmirror1_stage2"     # Always optional
```

**Rule**: Aliases must be updated explicitly with audit trail. Never auto-promote.

---

## Artifact Manifest Summary

**File**: `models/manifests/artifact_manifest.json`

| Artifact ID | Model | Type | SHA-256 | Size |
|-------------|-------|------|---------|------|
| wm1_baseline_ckpt | baseline | checkpoint | dbc9c9b0... | 2.64 GB |
| wm1_stage1_ckpt | stage1 | checkpoint | fa12e006... | 2.64 GB |
| wm1_stage2_ckpt | stage2 | checkpoint | e1848950... | 2.46 GB |
| wm1_pretrained_config | baseline | config | (pending) | 421 B |
| stage1_training_config | stage1 | config | (pending) | - |
| stage2_training_config | stage2 | config | (pending) | - |
| indoor_verified_manifest | all | dataset | 86679e6a... | - |

**Total**: 8 artifacts, 8.32 GB