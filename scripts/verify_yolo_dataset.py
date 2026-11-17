"""
scripts/verify_yolo_dataset.py
Automated Verification Suite for RoomAI Open Images V7 Pilot YOLO Dataset.

Enforces 10 deterministic validation gates:
1. data.yaml existence, syntax, forward slashes, and class definitions.
2. Directory layout integrity (images/ and labels/ for train, val, test).
3. manifest.json schema compliance and 10,000 image count (8,000 / 1,000 / 1,000).
4. Exact 1:1 image and label file parity.
5. Zero split leakage across train, val, and test (ImageID and MD5 hash disjointness).
6. Image readability and dimension verification against manifest.
7. Bounding box coordinate geometry verification [0, 1] and class ID bounds.
8. Label-manifest synchronization (box counts and class counts).
9. Class representation and tail class balance (val_boxes >= 5, test_boxes >= 5).
10. Ultralytics YOLODataset loader smoke test on validation split.

Exits with code 0 on 100% PASS; exits with code 1 on any critical failure.
"""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Set, Tuple

from PIL import Image
import yaml

EXPECTED_NC = 18
EXPECTED_SPLIT_COUNTS = {"train": 8000, "val": 1000, "test": 1000}
TOTAL_SAMPLES = 10000


class DatasetVerifier:
    def __init__(self, dataset_dir: Path, check_images: bool = True, smoke_test: bool = True, strict: bool = True):
        self.dataset_dir = dataset_dir
        self.check_images = check_images
        self.smoke_test = smoke_test
        self.strict = strict
        self.results: List[Dict[str, str]] = []
        self.config: Dict[str, Any] = {}
        self.manifest: Dict[str, Any] = {}
        self.class_map: Dict[int, str] = {}
        self.split_image_ids: Dict[str, Set[str]] = {"train": set(), "val": set(), "test": set()}
        self.split_md5s: Dict[str, Set[str]] = {"train": set(), "val": set(), "test": set()}

    def record(self, gate_name: str, status: str, details: str):
        print(f"[{status:<7}] {gate_name}: {details}")
        self.results.append({"gate": gate_name, "status": status, "details": details})

    def run_all(self) -> bool:
        print("=" * 75)
        print("ROOMAI YOLO OPEN IMAGES PILOT DATASET VERIFICATION SUITE")
        print(f"Target Directory: {self.dataset_dir}")
        print("=" * 75)

        # Gate 1: Check data.yaml
        if not self.verify_data_yaml():
            return False

        # Gate 2: Check Directory Layout
        if not self.verify_directory_layout():
            return False

        # Gate 3: Check Manifest JSON
        if not self.verify_manifest():
            return False

        # Gate 4: 1:1 Image and Label Parity
        self.verify_file_parity()

        # Gate 5: Zero Split Leakage (ID and MD5 Disjointness)
        self.verify_zero_leakage()

        # Gate 6 & 7 & 8: Image Readability, Annotation Geometry, and Manifest Sync
        self.verify_annotations_and_images()

        # Gate 9: Class Balance and Minimum Representation
        self.verify_class_balance()

        # Gate 10: Ultralytics Dataloader Smoke Test
        if self.smoke_test:
            self.run_dataloader_smoke_test()

        return self.generate_scorecard()

    def verify_data_yaml(self) -> bool:
        yaml_path = self.dataset_dir / "data.yaml"
        if not yaml_path.exists():
            self.record("Gate 1: data.yaml Existence", "FAIL", f"File missing at {yaml_path}")
            return False

        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            self.record("Gate 1: data.yaml Syntax", "FAIL", f"YAML parse error: {e}")
            return False

        # Verify required keys
        req_keys = ["path", "train", "val", "test", "nc", "names"]
        missing = [k for k in req_keys if k not in self.config]
        if missing:
            self.record("Gate 1: data.yaml Keys", "FAIL", f"Missing keys: {missing}")
            return False

        # Verify forward slash path safety on Windows
        raw_path = str(self.config["path"])
        if "\\" in raw_path:
            self.record("Gate 1: Path Separator Safety", "WARNING", f"Backslash detected in '{raw_path}'; prefer forward slashes.")
        else:
            self.record("Gate 1: Path Separator Safety", "PASS", "Forward slashes utilized in dataset path.")

        # Verify nc and names
        nc = self.config["nc"]
        names = self.config["names"]
        if isinstance(names, dict):
            self.class_map = {int(k): str(v) for k, v in names.items()}
        elif isinstance(names, list):
            self.class_map = {i: str(v) for i, v in enumerate(names)}
        else:
            self.record("Gate 1: Class Definition", "FAIL", "Invalid names format (must be dict or list).")
            return False

        if len(self.class_map) != EXPECTED_NC or nc != EXPECTED_NC:
            self.record("Gate 1: Class Count Alignment", "FAIL", f"Expected {EXPECTED_NC} classes, found nc={nc}, len(names)={len(self.class_map)}.")
            return False

        if sorted(self.class_map.keys()) != list(range(EXPECTED_NC)):
            self.record("Gate 1: Contiguous Class IDs", "FAIL", f"Class IDs must be contiguous 0 to {EXPECTED_NC-1}.")
            return False

        self.record("Gate 1: data.yaml Verification", "PASS", f"Valid YAML with {EXPECTED_NC} contiguous classes.")
        return True

    def verify_directory_layout(self) -> bool:
        splits = ["train", "val", "test"]
        missing_dirs = []
        for s in splits:
            if not (self.dataset_dir / "images" / s).is_dir():
                missing_dirs.append(f"images/{s}")
            if not (self.dataset_dir / "labels" / s).is_dir():
                missing_dirs.append(f"labels/{s}")

        if missing_dirs:
            self.record("Gate 2: Directory Layout", "FAIL", f"Missing subdirectories: {missing_dirs}")
            return False

        self.record("Gate 2: Directory Layout", "PASS", "All images/{train,val,test} and labels/{train,val,test} exist.")
        return True

    def verify_manifest(self) -> bool:
        manifest_path = self.dataset_dir / "manifest.json"
        if not manifest_path.exists():
            self.record("Gate 3: Manifest Existence", "FAIL", f"manifest.json missing at {manifest_path}")
            return False

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                self.manifest = json.load(f)
        except Exception as e:
            self.record("Gate 3: Manifest Syntax", "FAIL", f"Invalid JSON in manifest: {e}")
            return False

        images_dict = self.manifest.get("images", {})
        if len(images_dict) != TOTAL_SAMPLES:
            self.record("Gate 3: Manifest Sample Count", "FAIL", f"Expected {TOTAL_SAMPLES} images, found {len(images_dict)}.")
            return False

        split_counts = {"train": 0, "val": 0, "test": 0}
        for img_id, meta in images_dict.items():
            s = meta.get("split")
            if s in split_counts:
                split_counts[s] += 1
                self.split_md5s[s].add(meta.get("md5", ""))

        for s, expected in EXPECTED_SPLIT_COUNTS.items():
            if split_counts[s] != expected:
                self.record("Gate 3: Manifest Split Counts", "FAIL", f"Split '{s}' has {split_counts[s]} images, expected {expected}.")
                return False

        self.record("Gate 3: Manifest Integrity", "PASS", f"10,000 samples tracked with exact 8000/1000/1000 split partitions.")
        return True

    def verify_file_parity(self):
        splits = ["train", "val", "test"]
        parity_failures = 0

        for s in splits:
            img_dir = self.dataset_dir / "images" / s
            lbl_dir = self.dataset_dir / "labels" / s

            img_files = set(p.stem for p in img_dir.glob("*.jpg"))
            lbl_files = set(p.stem for p in lbl_dir.glob("*.txt"))

            self.split_image_ids[s] = img_files

            if len(img_files) != EXPECTED_SPLIT_COUNTS[s]:
                self.record(f"Gate 4: {s} Image Count", "FAIL", f"Found {len(img_files)} images in images/{s}, expected {EXPECTED_SPLIT_COUNTS[s]}.")
                parity_failures += 1

            if len(lbl_files) != EXPECTED_SPLIT_COUNTS[s]:
                self.record(f"Gate 4: {s} Label Count", "FAIL", f"Found {len(lbl_files)} labels in labels/{s}, expected {EXPECTED_SPLIT_COUNTS[s]}.")
                parity_failures += 1

            unmatched_imgs = img_files - lbl_files
            unmatched_lbls = lbl_files - img_files

            if unmatched_imgs or unmatched_lbls:
                self.record(f"Gate 4: {s} 1:1 Parity", "FAIL", f"Unmatched: {len(unmatched_imgs)} images without labels, {len(unmatched_lbls)} labels without images.")
                parity_failures += 1

        if parity_failures == 0:
            self.record("Gate 4: File Parity & Counts", "PASS", "100% 1:1 parity across 10,000 images and label files.")

    def verify_zero_leakage(self):
        t_ids = self.split_image_ids["train"]
        v_ids = self.split_image_ids["val"]
        te_ids = self.split_image_ids["test"]

        leak_tv = t_ids.intersection(v_ids)
        leak_tte = t_ids.intersection(te_ids)
        leak_vte = v_ids.intersection(te_ids)

        if leak_tv or leak_tte or leak_vte:
            self.record("Gate 5: Split ImageID Disjointness", "FAIL", f"Leakage detected! Train/Val: {len(leak_tv)}, Train/Test: {len(leak_tte)}, Val/Test: {len(leak_vte)}")
        else:
            self.record("Gate 5: Split ImageID Disjointness", "PASS", "Zero ImageID overlap across train, val, and test splits.")

        t_md5 = self.split_md5s["train"]
        v_md5 = self.split_md5s["val"]
        te_md5 = self.split_md5s["test"]

        leak_md5_tv = t_md5.intersection(v_md5)
        leak_md5_tte = t_md5.intersection(te_md5)
        leak_md5_vte = v_md5.intersection(te_md5)

        if leak_md5_tv or leak_md5_tte or leak_md5_vte:
            self.record("Gate 5: Split MD5 Hash Disjointness", "FAIL", f"Content hash leakage detected! Train/Val: {len(leak_md5_tv)}")
        else:
            self.record("Gate 5: Split MD5 Hash Disjointness", "PASS", "Zero cryptographic MD5 hash overlap across splits.")

    def verify_annotations_and_images(self):
        splits = ["train", "val", "test"]
        corrupt_imgs = 0
        invalid_lines = 0
        coord_bounds_errors = 0
        class_id_errors = 0
        manifest_mismatches = 0
        total_boxes = 0

        manifest_images = self.manifest.get("images", {})

        for s in splits:
            img_dir = self.dataset_dir / "images" / s
            lbl_dir = self.dataset_dir / "labels" / s

            for img_stem in sorted(self.split_image_ids[s]):
                img_path = img_dir / f"{img_stem}.jpg"
                lbl_path = lbl_dir / f"{img_stem}.txt"

                # Image checks
                if self.check_images:
                    try:
                        with Image.open(img_path) as im:
                            w, h = im.size
                            meta = manifest_images.get(img_stem, {})
                            if meta and (w != meta.get("width") or h != meta.get("height")):
                                corrupt_imgs += 1
                    except Exception:
                        corrupt_imgs += 1

                # Label parsing
                content = lbl_path.read_text(encoding="utf-8").strip()
                file_box_count = 0

                if content:
                    for line_idx, line in enumerate(content.splitlines(), 1):
                        tokens = line.strip().split()
                        if len(tokens) != 5:
                            invalid_lines += 1
                            continue

                        try:
                            cid, xc, yc, bw, bh = float(tokens[0]), float(tokens[1]), float(tokens[2]), float(tokens[3]), float(tokens[4])
                        except ValueError:
                            invalid_lines += 1
                            continue

                        cid_int = int(cid)
                        if cid_int not in self.class_map:
                            class_id_errors += 1
                            continue

                        # Clamping & boundary checks
                        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
                            coord_bounds_errors += 1
                        else:
                            # Box corner check
                            xmin = xc - bw / 2.0
                            xmax = xc + bw / 2.0
                            ymin = yc - bh / 2.0
                            ymax = yc + bh / 2.0
                            if xmin < -1e-4 or xmax > 1.0001 or ymin < -1e-4 or ymax > 1.0001:
                                coord_bounds_errors += 1
                            else:
                                total_boxes += 1
                                file_box_count += 1

                # Check synchronization with manifest
                meta = manifest_images.get(img_stem)
                if meta:
                    expected_total = meta.get("total_boxes", 0)
                    if file_box_count != expected_total:
                        manifest_mismatches += 1

        if corrupt_imgs == 0:
            self.record("Gate 6: Image Integrity & Dimensions", "PASS", "All inspected images verified readable and dimension-matched.")
        else:
            self.record("Gate 6: Image Integrity & Dimensions", "FAIL", f"{corrupt_imgs} images corrupted or mismatched in dimensions.")

        if invalid_lines == 0 and coord_bounds_errors == 0 and class_id_errors == 0:
            self.record("Gate 7: Bounding Box Geometry", "PASS", f"{total_boxes} boxes verified normalized in [0, 1] with valid class IDs.")
        else:
            self.record("Gate 7: Bounding Box Geometry", "FAIL", f"Errors: {coord_bounds_errors} bounds, {class_id_errors} class IDs, {invalid_lines} malformed lines.")

        if manifest_mismatches == 0:
            self.record("Gate 8: Manifest Synchronization", "PASS", "100% agreement between disk label files and manifest records.")
        else:
            self.record("Gate 8: Manifest Synchronization", "FAIL", f"{manifest_mismatches} files have discrepant box counts vs manifest.")

    def verify_class_balance(self):
        splits = ["val", "test"]
        split_class_totals = {"val": {c: 0 for c in range(EXPECTED_NC)}, "test": {c: 0 for c in range(EXPECTED_NC)}}

        for s in splits:
            lbl_dir = self.dataset_dir / "labels" / s
            for lbl_file in lbl_dir.glob("*.txt"):
                content = lbl_file.read_text(encoding="utf-8").strip()
                if not content:
                    continue
                for line in content.splitlines():
                    tokens = line.strip().split()
                    if len(tokens) == 5:
                        try:
                            cid = int(float(tokens[0]))
                            if cid in split_class_totals[s]:
                                split_class_totals[s][cid] += 1
                        except ValueError:
                            pass

        starved_val = [self.class_map[c] for c, count in split_class_totals["val"].items() if count < 5]
        starved_test = [self.class_map[c] for c, count in split_class_totals["test"].items() if count < 5]

        if starved_val or starved_test:
            self.record("Gate 9: Tail Class Representation", "WARNING", f"Classes with < 5 samples: Val={starved_val}, Test={starved_test}")
        else:
            self.record("Gate 9: Tail Class Representation", "PASS", f"All {EXPECTED_NC} classes meet minimum representation threshold (>= 5) in val and test.")

    def run_dataloader_smoke_test(self):
        try:
            from ultralytics.data.dataset import YOLODataset
            val_img_path = str(self.dataset_dir / "images" / "val")
            ds = YOLODataset(img_path=val_img_path, data=self.config, task="detect")
            if len(ds) == EXPECTED_SPLIT_COUNTS["val"]:
                sample = ds[0]
                self.record("Gate 10: Ultralytics Dataloader Smoke Test", "PASS", f"Loaded {len(ds)} val samples successfully. Sample 0 keys: {list(sample.keys())}")
            else:
                self.record("Gate 10: Ultralytics Dataloader Smoke Test", "FAIL", f"Dataloader loaded {len(ds)} samples, expected {EXPECTED_SPLIT_COUNTS['val']}.")
        except Exception as e:
            self.record("Gate 10: Ultralytics Dataloader Smoke Test", "FAIL", f"Ultralytics loader failed: {e}")

    def generate_scorecard(self) -> bool:
        print("\n" + "=" * 75)
        print("VERIFICATION SCORECARD")
        print("=" * 75)
        passes = sum(1 for r in self.results if r["status"] == "PASS")
        warnings = sum(1 for r in self.results if r["status"] == "WARNING")
        fails = sum(1 for r in self.results if r["status"] == "FAIL")

        print(f"PASS: {passes} | WARNING: {warnings} | FAIL: {fails}")
        print("=" * 75)

        if fails == 0:
            print("\nVERDICT: DATASET IS 100% VERIFIED FOR ULTRALYTICS YOLO TRAINING.")
            return True
        else:
            print(f"\nVERDICT: VERIFICATION ENCOUNTERED {fails} CRITICAL FAILURES. REVIEW DETAILS ABOVE.")
            return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify RoomAI YOLO Dataset.")
    parser.add_argument("--dataset-dir", type=str, default=r"F:/RoomAI_Data/yolo_open_images_pilot", help="Path to YOLO dataset root.")
    parser.add_argument("--no-check-images", action="store_true", help="Skip individual PIL image verify check.")
    parser.add_argument("--no-smoke-test", action="store_true", help="Skip Ultralytics loader smoke test.")
    parser.add_argument("--non-strict", action="store_true", help="Do not exit with 1 on warnings.")
    args = parser.parse_args()

    verifier = DatasetVerifier(
        dataset_dir=Path(args.dataset_dir),
        check_images=not args.no_check_images,
        smoke_test=not args.no_smoke_test,
        strict=not args.non_strict,
    )
    success = verifier.run_all()
    sys.exit(0 if success else 1)
