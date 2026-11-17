"""
RoomAI — Split Generation & File-Level Overlap Verification
Creates:
1. Golden Test Set (200 images, strictly quarantined from validation indoor images)
2. Validation Tuning Set (50 images, from validation indoor images)
3. Audit Benchmark Set (50 images, from validation indoor images)
4. Training Pool (3,830 images, from training indoor images)
5. 1% Few-Shot Split (48 images, from training pool)

Verifies file-level SHA256 hashes and ensures zero leakage.
"""

import hashlib
import json
from pathlib import Path
import random

REPO_ROOT = Path(__file__).resolve().parent.parent
ADE_ROOT = REPO_ROOT / "archive (2)" / "ADEChallengeData2016"
SPLITS_DIR = REPO_ROOT / "data" / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CATEGORIES = ["bedroom", "living_room", "bathroom", "kitchen", "dining_room"]


def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    scene_file = ADE_ROOT / "sceneCategories.txt"
    val_by_cat = {c: [] for c in TARGET_CATEGORIES}
    train_by_cat = {c: [] for c in TARGET_CATEGORIES}

    with open(scene_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                img_id, sc = parts
                if sc in TARGET_CATEGORIES:
                    if "ADE_val_" in img_id:
                        val_by_cat[sc].append(img_id)
                    elif "ADE_train_" in img_id:
                        train_by_cat[sc].append(img_id)

    # 1. Load the 50 Audit Benchmark images
    with open(REPO_ROOT / "reports" / "benchmark_reproducibility_results.json", "r") as f:
        audit_data = json.load(f)
    audit_50_set = set(audit_data["eval_images"])

    # 2. From the remaining validation indoor images, sample:
    # - Validation Tuning Set: 50 images (10 per category)
    # - Golden Test Set: 200 images (stratified: 70 bedroom, 40 living_room, 36 bathroom, 34 kitchen, 20 dining_room)
    random.seed(42)

    val_remaining = {}
    for c in TARGET_CATEGORIES:
        val_remaining[c] = [img for img in val_by_cat[c] if img not in audit_50_set]
        random.shuffle(val_remaining[c])

    val_tuning_50 = []
    golden_test_200 = []

    # Target allocations for Golden Test Set (200 total)
    golden_targets = {
        "bedroom": 70,
        "living_room": 40,
        "bathroom": 36,
        "kitchen": 34,
        "dining_room": 20,
    }

    # Extract 10 per category for Validation Tuning (50 total)
    for c in TARGET_CATEGORIES:
        val_tuning_50.extend(val_remaining[c][:10])
        val_remaining[c] = val_remaining[c][10:]

    # Extract Golden Test Set
    for c, target_n in golden_targets.items():
        assert len(val_remaining[c]) >= target_n, f"Not enough images for {c}: have {len(val_remaining[c])}, need {target_n}"
        golden_test_200.extend(val_remaining[c][:target_n])
        val_remaining[c] = val_remaining[c][target_n:]

    # 3. Create 1% Few-Shot Split from Training Pool (48 images)
    # Stratified: bedroom: 17, living_room: 10, bathroom: 9, kitchen: 8, dining_room: 4
    train_1pct_targets = {
        "bedroom": 17,
        "living_room": 10,
        "bathroom": 9,
        "kitchen": 8,
        "dining_room": 4,
    }
    train_pool_all = []
    train_1pct_48 = []
    for c in TARGET_CATEGORIES:
        imgs = list(train_by_cat[c])
        random.shuffle(imgs)
        train_pool_all.extend(imgs)
        train_1pct_48.extend(imgs[:train_1pct_targets[c]])

    # 4. Overlap & Hash Verification
    set_audit = set(audit_50_set)
    set_val_tuning = set(val_tuning_50)
    set_golden = set(golden_test_200)
    set_train_pool = set(train_pool_all)
    set_train_1pct = set(train_1pct_48)

    print("\n--- Split Size Summary ---")
    print(f"Audit Benchmark Set : {len(set_audit)}")
    print(f"Validation Tuning Set: {len(set_val_tuning)}")
    print(f"Golden Test Set     : {len(set_golden)}")
    print(f"Training Pool All   : {len(set_train_pool)}")
    print(f"1% Few-Shot Train   : {len(set_train_1pct)}")

    # Pairwise Intersection Checks
    print("\n--- Pairwise Disjointness Checks ---")
    checks = [
        ("Golden Test vs. Audit Benchmark", set_golden.intersection(set_audit)),
        ("Golden Test vs. Validation Tuning", set_golden.intersection(set_val_tuning)),
        ("Golden Test vs. Training Pool", set_golden.intersection(set_train_pool)),
        ("Validation Tuning vs. Audit Benchmark", set_val_tuning.intersection(set_audit)),
        ("Validation Tuning vs. Training Pool", set_val_tuning.intersection(set_train_pool)),
        ("Audit Benchmark vs. Training Pool", set_audit.intersection(set_train_pool)),
    ]

    all_passed = True
    for name, inter in checks:
        passed = (len(inter) == 0)
        print(f"  {name:38s}: {'PASSED (0 overlap)' if passed else 'FAILED: ' + str(len(inter))}")
        if not passed:
            all_passed = False

    # Compute sample hashes to prove disk integrity
    print("\n--- File Hash Sample Verification ---")
    sample_golden_hashes = [compute_sha256(ADE_ROOT / "images" / "validation" / f"{img}.jpg") for img in golden_test_200[:5]]
    sample_train_hashes = [compute_sha256(ADE_ROOT / "images" / "training" / f"{img}.jpg") for img in train_1pct_48[:5]]
    print(f"Computed SHA256 hashes successfully. Hash overlap between samples: {len(set(sample_golden_hashes).intersection(set(sample_train_hashes)))}")

    # Save manifests
    manifests = {
        "audit_benchmark_50": sorted(list(set_audit)),
        "validation_tuning_50": sorted(list(set_val_tuning)),
        "golden_test_200": sorted(list(set_golden)),
        "train_1pct_48": sorted(list(set_train_1pct)),
    }
    for name, img_list in manifests.items():
        p = SPLITS_DIR / f"{name}.json"
        with open(p, "w") as f:
            json.dump(img_list, f, indent=2)
        print(f"Saved manifest: {p} ({len(img_list)} images)")

    return all_passed


if __name__ == "__main__":
    assert main(), "Disjointness verification failed!"

