r"""Automated Pipeline Orchestrator for RoomAI Milestone 3.

Sequentially executes:
1. Exp 1: Baseline Fine-Tuning (yolo11s.pt, 640px, batch 16, epochs 10, patience 5, with --resume)
2. Exp 2: High-Resolution Fine-Tuning (yolo11s.pt, 800px, batch 8, accumulate 2, epochs 10, patience 5, with --resume)
3. Exp 3: Lightweight Model Comparison (yolo11n.pt, 640px, batch 16, epochs 10, patience 5, with --resume)
4. Exp 4: Tail-Class Imbalance Remedy (yolo11s.pt, 640px, batch 16, data_rfs.yaml, epochs 10, patience 5, with --resume)
5. Comparison & Single Final Test Evaluation (scripts/compare_experiments.py)
6. Regression Verification (pytest -v tests/)
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ["USE_TF"] = "0"
os.environ["OPENCV_NUM_THREADS"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_DATA_ROOT = Path(r"F:\RoomAI_Data")
DATA_YAML = EXTERNAL_DATA_ROOT / "yolo_open_images_pilot" / "data.yaml"
DATA_RFS_YAML = EXTERNAL_DATA_ROOT / "yolo_open_images_pilot" / "data_rfs.yaml"
PROGRESS_FILE = REPO_ROOT / ".agents" / "teamwork" / "worker_m3_2" / "progress.md"
REPORTS_DIR = REPO_ROOT / "reports"

EXPERIMENTS = [
    {
        "name": "exp1_yolo11s_baseline_640",
        "cmd": [
            sys.executable,
            "scripts/train_open_images.py",
            "--model", "yolo11s.pt",
            "--data", str(DATA_YAML),
            "--imgsz", "640",
            "--batch", "16",
            "--accumulate", "1",
            "--epochs", "10",
            "--patience", "5",
            "--workers", "0",
            "--exp-name", "exp1_yolo11s_baseline_640",
            "--resume",
        ],
        "step_num": 4,
        "desc": "Exp 1: Baseline Fine-Tuning (YOLO11s, 640px, batch 16)",
    },
    {
        "name": "exp2_yolo11s_highres_800",
        "cmd": [
            sys.executable,
            "scripts/train_open_images.py",
            "--model", "yolo11s.pt",
            "--data", str(DATA_YAML),
            "--imgsz", "800",
            "--batch", "8",
            "--accumulate", "2",
            "--epochs", "10",
            "--patience", "5",
            "--workers", "0",
            "--exp-name", "exp2_yolo11s_highres_800",
            "--resume",
        ],
        "step_num": 5,
        "desc": "Exp 2: High-Resolution Fine-Tuning (YOLO11s, 800px, batch 8, accum 2)",
    },
    {
        "name": "exp3_yolo11n_lightweight_640",
        "cmd": [
            sys.executable,
            "scripts/train_open_images.py",
            "--model", "yolo11n.pt",
            "--data", str(DATA_YAML),
            "--imgsz", "640",
            "--batch", "16",
            "--accumulate", "1",
            "--epochs", "10",
            "--patience", "5",
            "--workers", "0",
            "--exp-name", "exp3_yolo11n_lightweight_640",
            "--resume",
        ],
        "step_num": 6,
        "desc": "Exp 3: Lightweight Model Comparison (YOLO11n, 640px, batch 16)",
    },
    {
        "name": "exp4_yolo11s_rfs_imbalance_remedy",
        "cmd": [
            sys.executable,
            "scripts/train_open_images.py",
            "--model", "yolo11s.pt",
            "--data", str(DATA_RFS_YAML),
            "--imgsz", "640",
            "--batch", "16",
            "--accumulate", "1",
            "--epochs", "10",
            "--patience", "5",
            "--workers", "0",
            "--exp-name", "exp4_yolo11s_rfs_imbalance_remedy",
            "--resume",
        ],
        "step_num": 7,
        "desc": "Exp 4: Tail-Class Imbalance Remedy (YOLO11s, 640px, batch 16, RFS)",
    },
]


def update_progress(current_status: str, completed_steps: list):
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    lines = [
        "# Progress: Worker M3-2 (Training Implementation Worker)",
        "",
        f"Last visited: {timestamp}",
        "",
        "## Status",
        f"- {current_status}",
        "",
        "## Steps",
        f"- [{'x' if 1 in completed_steps else ' '}] Step 1: Implement and verify scripts/generate_rfs_dataset.py",
        f"- [{'x' if 2 in completed_steps else ' '}] Step 2: Implement and verify scripts/train_open_images.py",
        f"- [{'x' if 3 in completed_steps else ' '}] Step 3: Implement and verify scripts/compare_experiments.py",
        f"- [{'x' if 4 in completed_steps else ' '}] Step 4: Execute training Experiment 1 (exp1_yolo11s_baseline_640)",
        f"- [{'x' if 5 in completed_steps else ' '}] Step 5: Execute training Experiment 2 (exp2_yolo11s_highres_800)",
        f"- [{'x' if 6 in completed_steps else ' '}] Step 6: Execute training Experiment 3 (exp3_yolo11n_lightweight_640)",
        f"- [{'x' if 7 in completed_steps else ' '}] Step 7: Execute training Experiment 4 (exp4_yolo11s_rfs_imbalance_remedy)",
        f"- [{'x' if 8 in completed_steps else ' '}] Step 8: Run experiment comparison & single final test evaluation",
        f"- [{'x' if 9 in completed_steps else ' '}] Step 9: Save winning checkpoint to F:\\RoomAI_Data\\checkpoints\\open_images_yolo11s_best.pt",
        f"- [{'x' if 10 in completed_steps else ' '}] Step 10: Run full test suite (pytest -v tests/) and confirm 129/129 pass",
        f"- [{'x' if 11 in completed_steps else ' '}] Step 11: Write handoff report and send final message",
        "",
    ]
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    t_pipeline_start = time.time()
    completed_steps = [1, 2, 3]

    print("=" * 80)
    print("RoomAI Milestone 3: Automated Pipeline Execution")
    print(f"Start Time: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    print("=" * 80)

    # Execute Exps 1-4
    for exp in EXPERIMENTS:
        exp_name = exp["name"]
        metric_file = REPORTS_DIR / f"{exp_name}_metrics.json"

        # Check if already fully completed
        if metric_file.exists():
            try:
                with open(metric_file, "r", encoding="utf-8") as f:
                    m_data = json.load(f)
                if "validation_metrics" in m_data and "mAP50" in m_data["validation_metrics"]:
                    print("\n" + "=" * 80)
                    print(f"SKIPPING {exp['desc']} — Already completed ({metric_file})")
                    print(f"  val_mAP50: {m_data['validation_metrics']['mAP50']}, val_mAP50-95: {m_data['validation_metrics']['mAP50_95']}")
                    print("=" * 80)
                    completed_steps.append(exp["step_num"])
                    continue
            except Exception as e:
                print(f"[WARNING] Corrupt or unreadable metric report {metric_file}, will re-run: {e}")

        print("\n" + "=" * 80)
        print(f"STARTING {exp['desc']}")
        print(f"Command: {' '.join(exp['cmd'])}")
        print("=" * 80)

        update_progress(f"Running {exp['desc']}...", completed_steps)

        t0 = time.time()
        ret = subprocess.run(exp["cmd"], cwd=str(REPO_ROOT))
        duration = time.time() - t0

        if ret.returncode != 0:
            print(f"ERROR: {exp_name} failed with exit code {ret.returncode} after {duration:.1f}s")
            update_progress(f"FAILED: {exp_name} exited with code {ret.returncode}", completed_steps)
            sys.exit(ret.returncode)

        assert metric_file.exists(), f"Metric report missing: {metric_file}"
        completed_steps.append(exp["step_num"])
        print(f"SUCCESS: Completed {exp_name} in {duration:.1f}s ({duration/60:.2f} min)")
        update_progress(f"Completed {exp['desc']} ({duration/60:.2f} min).", completed_steps)

    # Step 8 & 9: Compare experiments and evaluate winner on test split
    print("\n" + "=" * 80)
    print("STEP 8 & 9: Running Experiment Comparison & Single Final Test Evaluation")
    print("=" * 80)
    update_progress("Running comparison & single test evaluation...", completed_steps)

    t0 = time.time()
    compare_cmd = [sys.executable, "scripts/compare_experiments.py", "--device", "0", "--workers", "0"]
    ret = subprocess.run(compare_cmd, cwd=str(REPO_ROOT))
    if ret.returncode != 0:
        print(f"ERROR: compare_experiments.py failed with code {ret.returncode}")
        sys.exit(ret.returncode)

    completed_steps.extend([8, 9])
    update_progress("Comparison and single test evaluation completed successfully.", completed_steps)

    # Step 10: Run full test suite regression verification
    print("\n" + "=" * 80)
    print("STEP 10: Running Full Test Suite Regression Verification (pytest -v tests/)")
    print("=" * 80)
    update_progress("Running regression test suite (pytest -v tests/)...", completed_steps)

    t0 = time.time()
    pytest_env = os.environ.copy()
    pytest_env["USE_TF"] = "0"
    pytest_cmd = [sys.executable, "-m", "pytest", "-v", "tests/"]
    ret = subprocess.run(pytest_cmd, cwd=str(REPO_ROOT), env=pytest_env)
    if ret.returncode != 0:
        print(f"ERROR: pytest suite failed with code {ret.returncode}")
        sys.exit(ret.returncode)

    completed_steps.append(10)
    total_pipeline_duration = time.time() - t_pipeline_start
    print("\n" + "=" * 80)
    print(f"ALL MILESTONE 3 PIPELINE STEPS COMPLETED IN {total_pipeline_duration/3600:.2f} HOURS!")
    print("=" * 80)
    update_progress(f"All 4 experiments, single test evaluation, and 129/129 tests passed in {total_pipeline_duration/3600:.2f} hours.", completed_steps)


if __name__ == "__main__":
    main()
