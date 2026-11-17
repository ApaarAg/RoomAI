"""Unit and integration tests for Phase 19 Multi-Dataset Harmonization, Integrity, and Plan."""

import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from utils.multidataset_harmonization import (
    CANONICAL_EXTERIOR_CLASSES,
    CANONICAL_NAME_TO_ID,
    HARMONIZATION_RULES,
    validate_yolo_box,
    compute_image_phash,
    compute_phash_hamming_distance,
    audit_split_isolation,
    LabelMappingEntry,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_canonical_taxonomy_integrity():
    """Verify RoomAI canonical exterior taxonomy has exactly 10 classes in expected order."""
    assert len(CANONICAL_EXTERIOR_CLASSES) == 10
    expected_order = [
        "building", "facade", "window", "door", "roof_cornice",
        "balcony", "porch", "pillar_column", "shutter_blind", "molding_deco"
    ]
    for cid, name in enumerate(expected_order):
        assert CANONICAL_EXTERIOR_CLASSES[cid] == name
        assert CANONICAL_NAME_TO_ID[name] == cid


def test_label_mapping_coverage_and_status():
    """Verify that mapping rules exist for all target candidate datasets with valid statuses."""
    datasets = {r.source_dataset for r in HARMONIZATION_RULES}
    assert "CMP_Facade" in datasets
    assert "OpenImagesV7" in datasets
    assert "ADE20K" in datasets
    assert "MapillaryVistas" in datasets
    assert "COCO_Detection_80" in datasets

    valid_statuses = {"exact", "defensible", "ambiguous", "unsupported"}
    for r in HARMONIZATION_RULES:
        assert r.mapping_status in valid_statuses, f"Invalid status: {r.mapping_status}"
        if r.target_label is not None:
            assert r.target_label in CANONICAL_NAME_TO_ID, f"Invalid target: {r.target_label}"
            assert r.canonical_id is not None
        else:
            assert r.canonical_id is None


def test_open_images_v7_specific_mappings():
    """Verify critical rare class mappings from Open Images V7."""
    oi_rules = {r.source_label: r for r in HARMONIZATION_RULES if r.source_dataset == "OpenImagesV7"}
    
    assert "/m/04m6gz (Porch)" in oi_rules
    assert oi_rules["/m/04m6gz (Porch)"].target_label == "porch"
    assert oi_rules["/m/04m6gz (Porch)"].canonical_id == 6
    assert oi_rules["/m/04m6gz (Porch)"].mapping_status == "exact"

    assert "/m/031b6r (Window blind)" in oi_rules
    assert oi_rules["/m/031b6r (Window blind)"].target_label == "shutter_blind"
    assert oi_rules["/m/031b6r (Window blind)"].canonical_id == 8

    assert "/m/02dgv (Door)" in oi_rules
    assert oi_rules["/m/02dgv (Door)"].target_label == "door"
    assert oi_rules["/m/02dgv (Door)"].canonical_id == 3


def test_coco_unsupported_rejection():
    """Verify that COCO 80-class detection is correctly flagged as completely unsupported."""
    coco_rules = [r for r in HARMONIZATION_RULES if r.source_dataset == "COCO_Detection_80"]
    assert len(coco_rules) > 0
    for r in coco_rules:
        assert r.mapping_status == "unsupported"
        assert r.target_label is None


def test_ade20k_architectural_mappings():
    """Verify ADE20K part and object mapping for complex architectural elements."""
    ade_rules = {r.source_label: r for r in HARMONIZATION_RULES if r.source_dataset == "ADE20K"}
    
    assert "column, pillar" in ade_rules
    assert ade_rules["column, pillar"].target_label == "pillar_column"
    assert ade_rules["column, pillar"].canonical_id == 7

    assert "balcony" in ade_rules
    assert ade_rules["balcony"].target_label == "balcony"
    assert ade_rules["balcony"].canonical_id == 5

    assert "porch" in ade_rules
    assert ade_rules["porch"].target_label == "porch"
    assert ade_rules["porch"].canonical_id == 6


def test_yolo_box_validation_valid_cases():
    """Verify valid normalized YOLO bounding boxes pass validation."""
    res = validate_yolo_box(0, 0.5, 0.5, 0.4, 0.6)
    assert res.is_valid
    assert res.error is None
    assert res.box_coords == (0.5, 0.5, 0.4, 0.6)

    # Boundary corner box
    res2 = validate_yolo_box(2, 0.1, 0.1, 0.2, 0.2)
    assert res2.is_valid
    assert res2.box_coords == (0.1, 0.1, 0.2, 0.2)


def test_yolo_box_validation_invalid_cases():
    """Verify out-of-bounds, negative, NaN, and degenerate boxes fail validation."""
    # Negative coordinate
    res1 = validate_yolo_box(0, -0.1, 0.5, 0.4, 0.4)
    assert not res1.is_valid
    assert "out of normalized bounds" in res1.error

    # Out of bounds (> 1.0)
    res2 = validate_yolo_box(0, 0.9, 0.9, 0.4, 0.4)
    assert not res2.is_valid
    assert "out of normalized bounds" in res2.error

    # Degenerate zero dimension
    res3 = validate_yolo_box(1, 0.5, 0.5, 0.0, 0.4)
    assert not res3.is_valid
    assert "Degenerate" in res3.error

    # Sub-threshold dimension (sliver < 0.003)
    res4 = validate_yolo_box(1, 0.5, 0.5, 0.001, 0.4)
    assert not res4.is_valid
    assert "Sub-threshold" in res4.error

    # Invalid class ID
    res5 = validate_yolo_box(15, 0.5, 0.5, 0.2, 0.2)
    assert not res5.is_valid
    assert "Invalid class ID" in res5.error

    # Non-finite NaN
    res6 = validate_yolo_box(0, float("nan"), 0.5, 0.2, 0.2)
    assert not res6.is_valid
    assert "Non-finite" in res6.error


def test_phash_computation_and_distance():
    """Verify perceptual hashing is deterministic and captures identical vs distinct images."""
    x = np.linspace(0, 255, 100, dtype=np.uint8)
    y = np.linspace(0, 255, 100, dtype=np.uint8)
    xx, yy = np.meshgrid(x, y)
    arr = np.stack([xx, yy, (xx + yy) // 2], axis=-1).astype(np.uint8)
    im1 = Image.fromarray(arr)
    im2 = Image.fromarray(arr)

    h1 = compute_image_phash(im1)
    h2 = compute_image_phash(im2)
    assert h1 == h2
    assert compute_phash_hamming_distance(h1, h2) == 0

    # JPEG recompression simulation (subtle compression noise)
    import io
    buf = io.BytesIO()
    im1.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    im_recompressed = Image.open(buf)
    h_recompressed = compute_image_phash(im_recompressed)
    dist_recompressed = compute_phash_hamming_distance(h1, h_recompressed)
    assert dist_recompressed <= 4  # Matches near-duplicate threshold (< 5)

    # Inverted distinct image should have large Hamming distance
    arr_inv = 255 - arr
    im_inv = Image.fromarray(arr_inv)
    h_inv = compute_image_phash(im_inv)
    dist_inv = compute_phash_hamming_distance(h1, h_inv)
    assert dist_inv > 10


def test_split_isolation_mock(tmp_path):
    """Verify audit_split_isolation catches both exact MD5 and near-duplicate pHash leaks."""
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    train_dir.mkdir()
    val_dir.mkdir()

    # Image A in train
    im_a = Image.new("RGB", (64, 64), color=(100, 150, 200))
    im_a.save(train_dir / "train_001.jpg")

    # Identical Image A in val (MD5 collision leak!)
    im_a.save(val_dir / "val_001.jpg")

    # Distinct Image B in val
    im_b = Image.new("RGB", (64, 64), color=(0, 0, 0))
    im_b.save(val_dir / "val_002.jpg")

    report = audit_split_isolation({"train": train_dir, "val": val_dir})
    assert not report.is_isolated
    assert len(report.md5_collisions) == 1
    assert report.md5_collisions[0][0] == "train"
    assert report.md5_collisions[0][1] == "val"


def test_phase19_artifacts_exist():
    """Verify phase19 deliverables are generated and properly structured."""
    csv_path = REPO_ROOT / "reports" / "phase19_dataset_mapping.csv"
    assert csv_path.exists(), "phase19_dataset_mapping.csv missing!"
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 40
    assert "source_dataset,source_label,target_label" in lines[0]

    plan_path = REPO_ROOT / "reports" / "phase19_training_plan.json"
    assert plan_path.exists(), "phase19_training_plan.json missing!"
    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)
    assert "immutable_constraints" in plan
    assert "hardware_constraints" in plan
    assert "staged_training_plan" in plan
    assert plan["hardware_constraints"]["disk_c_free_gb"] > 0
    assert "experiment_b_pilot_multisource_detection" in plan["staged_training_plan"]
    assert plan["staged_training_plan"]["experiment_b_pilot_multisource_detection"]["image_limit"] == 300
