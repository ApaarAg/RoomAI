"""
RoomAI — Phase E Split Integrity & File-Level Overlap Verification
Produces a rigorous file-level overlap report proving:
  1. The 200 Golden Test images are disjoint from the 50 benchmark images.
  2. Neither Golden Test nor benchmark images appear in fine-tuning data.
  3. No test image is used for hyperparameter selection, threshold tuning, or early stopping
     (i.e., Golden Test is disjoint from the validation tuning set).
  4. Fine-tuning (train) images are disjoint from all evaluation images.
  5. Confirms whether fine-tuning images originate from the ADE20K *training* split
     (already seen by the pretrained checkpoint) vs the *validation* split (unseen).

Outputs:
  reports/phase_e_split_integrity_report.json
  reports/phase_e_split_integrity_report.md
"""

import json
from pathlib import Path
from collections import Counter

REPO_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = REPO_ROOT / "data" / "splits"
REPORTS_DIR = REPO_ROOT / "reports"
ADE_ROOT = REPO_ROOT / "archive (2)" / "ADEChallengeData2016"

SPLIT_FILES = {
    "golden_test_200": SPLITS_DIR / "golden_test_200.json",
    "audit_benchmark_50": SPLITS_DIR / "audit_benchmark_50.json",
    "validation_tuning_50": SPLITS_DIR / "validation_tuning_50.json",
    "train_1pct_48": SPLITS_DIR / "train_1pct_48.json",
}


def load_ids(path):
    with open(path, "r") as f:
        return [str(x).strip() for x in json.load(f)]


def split_origin(img_id):
    """Return 'training' if ADE_train_*, 'validation' if ADE_val_*, else 'unknown'."""
    if img_id.startswith("ADE_train_"):
        return "training"
    if img_id.startswith("ADE_val_"):
        return "validation"
    return "unknown"


def file_exists_on_disk(img_id):
    origin = split_origin(img_id)
    folder = "training" if origin == "training" else "validation"
    img_path = ADE_ROOT / "images" / folder / f"{img_id}.jpg"
    ann_path = ADE_ROOT / "annotations" / folder / f"{img_id}.png"
    return img_path.exists() and ann_path.exists()


def main():
    print("=" * 70)
    print("RoomAI Phase E Split Integrity & File-Level Overlap Verification")
    print("=" * 70)

    splits = {name: load_ids(p) for name, p in SPLIT_FILES.items()}

    # 1. Counts
    counts = {name: len(ids) for name, ids in splits.items()}
    print("\n[1] Split counts:")
    for k, v in counts.items():
        print(f"    {k:28s}: {v}")

    # 2. Within-split duplicates
    dup_report = {}
    for name, ids in splits.items():
        c = Counter(ids)
        dups = {k: v for k, v in c.items() if v > 1}
        dup_report[name] = dups
        if dups:
            print(f"    [WARNING] {name} has internal duplicates: {dups}")
        else:
            print(f"    {name:28s}: 0 internal duplicates")

    # 3. Pairwise overlaps
    names = list(splits.keys())
    overlap_matrix = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            set_a = set(splits[a])
            set_b = set(splits[b])
            inter = sorted(set_a & set_b)
            key = f"{a} INTERSECT {b}"
            overlap_matrix[key] = {
                "intersection_size": len(inter),
                "intersection": inter,
            }
            print(f"    {key}: {len(inter)} overlapping images")

    # 4. Origin (training vs validation split) of each partition
    origin_report = {}
    for name, ids in splits.items():
        origins = Counter(split_origin(x) for x in ids)
        origin_report[name] = dict(origins)
        print(f"    {name:28s} origin: {dict(origins)}")

    # 5. Disk existence check
    existence_report = {}
    for name, ids in splits.items():
        missing = [x for x in ids if not file_exists_on_disk(x)]
        existence_report[name] = {"missing_count": len(missing), "missing": missing[:10]}
        if missing:
            print(f"    [WARNING] {name}: {len(missing)} files missing on disk")
        else:
            print(f"    {name:28s}: all files present on disk")

    # 6. Integrity verdict
    golden = set(splits["golden_test_200"])
    benchmark = set(splits["audit_benchmark_50"])
    val_tuning = set(splits["validation_tuning_50"])
    train = set(splits["train_1pct_48"])

    golden_vs_benchmark = len(golden & benchmark)
    golden_vs_val = len(golden & val_tuning)
    golden_vs_train = len(golden & train)
    benchmark_vs_train = len(benchmark & train)
    benchmark_vs_val = len(benchmark & val_tuning)
    train_vs_val = len(train & val_tuning)

    verdict = {
        "golden_disjoint_from_benchmark": golden_vs_benchmark == 0,
        "golden_disjoint_from_validation_tuning": golden_vs_val == 0,
        "golden_disjoint_from_finetune_train": golden_vs_train == 0,
        "benchmark_disjoint_from_finetune_train": benchmark_vs_train == 0,
        "benchmark_disjoint_from_validation_tuning": benchmark_vs_val == 0,
        "finetune_train_disjoint_from_validation_tuning": train_vs_val == 0,
        "all_evaluation_images_disjoint_from_train": len((golden | benchmark | val_tuning) & train) == 0,
    }
    print("\n[Verdict]")
    for k, v in verdict.items():
        print(f"    {k}: {v}")

    # 7. Pretraining overlap determination
    train_origins = Counter(split_origin(x) for x in splits["train_1pct_48"])
    golden_origins = Counter(split_origin(x) for x in splits["golden_test_200"])
    val_origins = Counter(split_origin(x) for x in splits["validation_tuning_50"])
    benchmark_origins = Counter(split_origin(x) for x in splits["audit_benchmark_50"])

    pretraining_overlap = {
        "finetune_train_from_ade_training_split": train_origins.get("training", 0),
        "finetune_train_from_ade_validation_split": train_origins.get("validation", 0),
        "golden_test_from_ade_training_split": golden_origins.get("training", 0),
        "golden_test_from_ade_validation_split": golden_origins.get("validation", 0),
        "validation_tuning_from_ade_training_split": val_origins.get("training", 0),
        "validation_tuning_from_ade_validation_split": val_origins.get("validation", 0),
        "benchmark_from_ade_training_split": benchmark_origins.get("training", 0),
        "benchmark_from_ade_validation_split": benchmark_origins.get("validation", 0),
        "interpretation": (
            "SegFormer-B1 (nvidia/segformer-b1-finetuned-ade-512-512) was pretrained on the "
            "ADE20K *training* split (20,210 images). Fine-tuning images are drawn from that "
            "SAME training split, so they were already seen during pretraining. The Golden Test "
            "Set, validation tuning set, and benchmark set are all drawn from the ADE20K "
            "*validation* split, which the pretrained checkpoint never observed. Therefore the "
            "pilot measures in-distribution continued training on seen data, NOT independent "
            "generalization to a novel domain. Generalization can only be claimed on the "
            "validation-split-derived evaluation sets."
        ),
    }

    report = {
        "split_counts": counts,
        "internal_duplicates": dup_report,
        "pairwise_overlaps": overlap_matrix,
        "origin_per_split": origin_report,
        "disk_existence": existence_report,
        "integrity_verdict": verdict,
        "pretraining_overlap_analysis": pretraining_overlap,
    }

    out_json = REPORTS_DIR / "phase_e_split_integrity_report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nJSON report saved to: {out_json}")

    # Markdown report
    md = []
    md.append("# RoomAI — Phase E Split Integrity & File-Level Overlap Report")
    md.append("")
    md.append(f"**Generated by:** `scripts/verify_phase_e_split_integrity.py`")
    md.append(f"**ADE20K root:** `archive (2)/ADEChallengeData2016`")
    md.append("")
    md.append("## 1. Split Counts")
    md.append("")
    md.append("| Split | Image Count |")
    md.append("|---|---|")
    for k, v in counts.items():
        md.append(f"| `{k}` | {v} |")
    md.append("")
    md.append("## 2. Internal Duplicates")
    md.append("")
    any_dup = False
    for name, dups in dup_report.items():
        if dups:
            any_dup = True
            md.append(f"- `{name}`: {dups}")
    if not any_dup:
        md.append("No internal duplicates detected in any split.")
    md.append("")
    md.append("## 3. Pairwise File-Level Overlap Matrix")
    md.append("")
    md.append("| Pair | Overlap Count | Overlapping IDs |")
    md.append("|---|---|---|")
    for key, val in overlap_matrix.items():
        ids = ", ".join(val["intersection"]) if val["intersection"] else "—"
        md.append(f"| {key} | {val['intersection_size']} | {ids} |")
    md.append("")
    md.append("## 4. ADE20K Split Origin (training vs validation)")
    md.append("")
    md.append("| Split | ADE_train_* (pretraining-seen) | ADE_val_* (pretraining-unseen) |")
    md.append("|---|---|---|")
    for name, origins in origin_report.items():
        md.append(f"| `{name}` | {origins.get('training', 0)} | {origins.get('validation', 0)} |")
    md.append("")
    md.append("## 5. Disk Existence Verification")
    md.append("")
    all_present = True
    for name, info in existence_report.items():
        if info["missing_count"] > 0:
            all_present = False
            md.append(f"- `{name}`: {info['missing_count']} missing (e.g. {info['missing']})")
    if all_present:
        md.append("All split image + annotation files verified present on disk.")
    md.append("")
    md.append("## 6. Integrity Verdict")
    md.append("")
    md.append("| Check | Pass |")
    md.append("|---|---|")
    for k, v in verdict.items():
        mark = "✅" if v else "❌"
        md.append(f"| {k} | {mark} |")
    md.append("")
    md.append("## 7. ADE20K Pretraining Overlap Determination")
    md.append("")
    md.append(pretraining_overlap["interpretation"])
    md.append("")
    md.append("| Partition | From ADE20K training split (seen by pretrained checkpoint) | From ADE20K validation split (unseen) |")
    md.append("|---|---|---|")
    md.append(f"| Fine-tuning train (1%) | {pretraining_overlap['finetune_train_from_ade_training_split']} | {pretraining_overlap['finetune_train_from_ade_validation_split']} |")
    md.append(f"| Golden Test (200) | {pretraining_overlap['golden_test_from_ade_training_split']} | {pretraining_overlap['golden_test_from_ade_validation_split']} |")
    md.append(f"| Validation tuning (50) | {pretraining_overlap['validation_tuning_from_ade_training_split']} | {pretraining_overlap['validation_tuning_from_ade_validation_split']} |")
    md.append(f"| Benchmark (50) | {pretraining_overlap['benchmark_from_ade_training_split']} | {pretraining_overlap['benchmark_from_ade_validation_split']} |")
    md.append("")
    md.append("## 8. Conclusion")
    md.append("")
    all_pass = all(verdict.values())
    if all_pass:
        md.append("All data-isolation checks **PASS**. The Golden Test Set is strictly disjoint from the benchmark, validation tuning, and fine-tuning partitions. No evaluation image is reused for training, hyperparameter selection, threshold tuning, or early stopping.")
    else:
        md.append("One or more data-isolation checks **FAILED**. The split must be redesigned before any training run.")
    md.append("")
    md.append("However, because the fine-tuning images originate from the ADE20K **training** split — the exact corpus used to pretrain `nvidia/segformer-b1-finetuned-ade-512-512` — the pilot does **not** measure independent domain generalization. It measures continued/refreshed training on data the checkpoint has already optimized against. Generalization claims are restricted to the validation-split-derived Golden Test, validation tuning, and benchmark sets, which the pretrained checkpoint never observed.")

    out_md = REPORTS_DIR / "phase_e_split_integrity_report.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"Markdown report saved to: {out_md}")


if __name__ == "__main__":
    main()
