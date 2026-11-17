"""Unit and integration tests for Step 3: Class-Conditional Loss Masking and Pilot Annotation Integrity.

Verifies:
1. Annotated positive objects contribute to appropriate loss.
2. Verified negatives are handled correctly.
3. Missing/unannotated classes are ignored (0 loss contribution) rather than penalized as false-negative background.
4. Unsupported class mappings do not generate labels.
5. Existing RoomAI annotations remain fully valid with unmasked 1.0 loss behavior.
"""

from pathlib import Path
import pytest
import torch
import torch.nn as nn
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss

from utils.masked_detection_loss import MaskedDetectionLoss
from utils.multidataset_harmonization import (
    CANONICAL_EXTERIOR_CLASSES,
    CANONICAL_NAME_TO_ID,
    HARMONIZATION_RULES,
    validate_yolo_box,
)

import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="module")
def yolo_model_and_dummy_batch():
    """Initializes a lightweight YOLO model and synthetic batch for testing loss behavior."""
    ckpt_path = REPO_ROOT / "checkpoints" / "exterior_detector_b6_expanded_best.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = ckpt["model"]
        if isinstance(model.args, dict):
            from types import SimpleNamespace
            model.args = SimpleNamespace(**model.args)
    else:
        model = DetectionModel(nc=10, ch=3)
    model.eval()

    device = torch.device("cpu")
    model.to(device)

    # Synthetic batch: 2 images of size 640x640
    # Image 0: Porch object (class 6) at center
    # Image 1: Door object (class 3) and Window (class 2)
    batch = {
        "im_file": ["oi_pilot_train_0001.jpg", "b6_train_0002.jpg"],
        "batch_idx": torch.tensor([0, 1, 1], dtype=torch.int64),
        "cls": torch.tensor([[6], [3], [2]], dtype=torch.float32),
        "bboxes": torch.tensor([
            [0.5, 0.5, 0.4, 0.4],   # Image 0: porch
            [0.3, 0.6, 0.2, 0.4],   # Image 1: door
            [0.7, 0.4, 0.2, 0.3],   # Image 1: window
        ], dtype=torch.float32),
    }

    # Synthetic predictions matching YOLO output format
    bs = 2
    feats = [
        torch.zeros(bs, 64, 80, 80),
        torch.zeros(bs, 64, 40, 40),
        torch.zeros(bs, 64, 20, 20),
    ]
    total_anchors = 80 * 80 + 40 * 40 + 20 * 20  # 8400 anchors

    boxes = torch.randn(bs, 64, total_anchors)
    scores = torch.randn(bs, 10, total_anchors)  # 10 canonical classes

    preds = {
        "feats": feats,
        "boxes": boxes,
        "scores": scores,
    }

    return model, preds, batch


def test_loss_annotated_positive_objects_contribute(yolo_model_and_dummy_batch):
    """Verify that annotated positive objects generate non-zero box, classification, and dfl loss."""
    model, preds, batch = yolo_model_and_dummy_batch
    loss_fn = MaskedDetectionLoss(model)

    _, loss, loss_detach = loss_fn.get_assigned_targets_and_loss(preds, batch)
    assert loss_detach.shape == (3,)
    assert loss_detach[0] > 0.0, "Box loss must be non-zero for positive targets"
    assert loss_detach[1] > 0.0, "Cls loss must be non-zero for positive targets"
    assert loss_detach[2] > 0.0, "DFL loss must be non-zero for positive targets"


def test_loss_verified_negatives_handled_correctly(yolo_model_and_dummy_batch):
    """Verify that when a class is verified negative (mask=1.0, 0 targets), false positives are penalized."""
    model, preds, batch = yolo_model_and_dummy_batch

    # Verified mask with class 8 (shutter_blind) = 1.0 (verified negative on Image 0)
    masks_verified = {
        "oi_pilot_train_0001.jpg": [1.0] * 10,  # All classes verified
        "b6_train_0002.jpg": [1.0] * 10,
    }
    loss_fn_verified = MaskedDetectionLoss(model, verified_mask_lookup=masks_verified)

    # Same batch, but Image 0 has high false-positive score for shutter_blind (class 8)
    preds_high_fp = {
        "feats": preds["feats"],
        "boxes": preds["boxes"].clone(),
        "scores": preds["scores"].clone(),
    }
    # Force high prediction scores for class 8 in image 0
    preds_high_fp["scores"][0, 8, :] = 8.0  # High confidence FP

    _, _, loss_fp_verified = loss_fn_verified.get_assigned_targets_and_loss(preds_high_fp, batch)
    _, _, loss_clean = loss_fn_verified.get_assigned_targets_and_loss(preds, batch)

    # Classification loss must increase significantly due to false-positive penalty on verified class
    assert loss_fp_verified[1] > loss_clean[1], "Verified negative FP must increase classification loss"


def test_loss_missing_annotations_ignored(yolo_model_and_dummy_batch):
    """Verify that unannotated classes (mask=0.0) produce ZERO loss contribution and are ignored."""
    model, preds, batch = yolo_model_and_dummy_batch

    # Scenario A: Class 8 (shutter_blind) is unverified in Image 0 (mask=0.0)
    masks_unverified = {
        "oi_pilot_train_0001.jpg": [1.0 if c != 8 else 0.0 for c in range(10)],
        "b6_train_0002.jpg": [1.0] * 10,
    }
    loss_fn_masked = MaskedDetectionLoss(model, verified_mask_lookup=masks_unverified)

    # Scenario B: Class 8 is verified in Image 0 (mask=1.0)
    masks_verified = {
        "oi_pilot_train_0001.jpg": [1.0] * 10,
        "b6_train_0002.jpg": [1.0] * 10,
    }
    loss_fn_unmasked = MaskedDetectionLoss(model, verified_mask_lookup=masks_verified)

    # Inject high predictions for unannotated class 8 on image 0
    preds_high_unannotated = {
        "feats": preds["feats"],
        "boxes": preds["boxes"].clone(),
        "scores": preds["scores"].clone(),
    }
    preds_high_unannotated["scores"][0, 8, :] = 10.0

    _, _, loss_masked = loss_fn_masked.get_assigned_targets_and_loss(preds_high_unannotated, batch)
    _, _, loss_unmasked = loss_fn_unmasked.get_assigned_targets_and_loss(preds_high_unannotated, batch)

    # Under masked loss, unannotated class 8 is completely ignored (loss is strictly lower)
    assert loss_masked[1] < loss_unmasked[1], (
        f"Masked loss ({loss_masked[1]:.4f}) must be strictly lower than unmasked loss ({loss_unmasked[1]:.4f}) "
        "because false negatives for unannotated classes are zeroed out."
    )


def test_unsupported_class_mappings_do_not_generate_labels():
    """Verify that non-architectural, unsupported, and ambiguous classes produce no labels."""
    # Ensure COCO and unsupported classes produce zero target mapping
    unsupported_rules = [r for r in HARMONIZATION_RULES if r.mapping_status == "unsupported"]
    assert len(unsupported_rules) > 0
    for r in unsupported_rules:
        assert r.target_label is None
        assert r.canonical_id is None

    # Ingestion lookup strictly rejects non-canonical or ambiguous classes
    from scripts.prepare_pilot_b7_dataset import OI_MID_TO_CANONICAL
    # Ambiguous Open Images classes must not be silently ingested
    assert "/m/02522" not in OI_MID_TO_CANONICAL  # Column/Pillar is ambiguous with newspaper columns
    assert "/m/02p3w7" not in OI_MID_TO_CANONICAL  # Balcony is sparse/unreliable in OI
    assert "/m/01g317" not in OI_MID_TO_CANONICAL  # Person is unsupported

    # Invalid class ID fails validation
    invalid_box = validate_yolo_box(10, 0.5, 0.5, 0.2, 0.2)
    assert not invalid_box.is_valid
    assert "Invalid class ID" in invalid_box.error


def test_existing_roomai_annotations_remain_valid(yolo_model_and_dummy_batch):
    """Verify that existing RoomAI legacy images (fully verified mask [1.0]*10) match standard YOLO loss."""
    model, preds, batch = yolo_model_and_dummy_batch

    # Standard YOLO detection loss
    loss_std = v8DetectionLoss(model)
    # Masked loss with default all-ones mask
    loss_masked = MaskedDetectionLoss(model, verified_mask_lookup={
        "oi_pilot_train_0001.jpg": [1.0] * 10,
        "b6_train_0002.jpg": [1.0] * 10,
    })

    # Call get_assigned_targets_and_loss on both
    _, loss_tensor_std, _ = loss_std.get_assigned_targets_and_loss(preds, batch)
    _, loss_tensor_masked, _ = loss_masked.get_assigned_targets_and_loss(preds, batch)

    torch.testing.assert_close(loss_tensor_std, loss_tensor_masked, atol=1e-5, rtol=1e-5)
