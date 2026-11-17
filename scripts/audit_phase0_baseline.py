import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import yaml

REPO_ROOT = Path(r"C:\Users\apaar\Desktop\RoomAI").resolve()

# 1. Environment Details
python_exe = sys.executable
python_ver = sys.version
conda_env = os.environ.get("CONDA_DEFAULT_ENV", "unknown")

# GPU & Torch
try:
    import torch
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "N/A"
    vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if cuda_avail else 0.0
    torch_ver = torch.__version__
except Exception as e:
    cuda_avail = False
    device_name = str(e)
    vram_gb = 0.0
    torch_ver = "error"

# Dependencies
def get_pkg_version(name):
    try:
        import importlib.metadata
        return importlib.metadata.version(name)
    except Exception:
        try:
            mod = __import__(name)
            return getattr(mod, "__version__", "unknown")
        except Exception:
            return "not_installed"

pkgs = ["ultralytics", "torchvision", "pydantic", "pyyaml", "opencv-python", "pillow", "streamlit", "psutil", "numpy", "pytest"]
pkg_versions = {p: get_pkg_version(p) for p in pkgs}

# System resources
import psutil
ram_total_gb = round(psutil.virtual_memory().total / (1024**3), 2)
ram_avail_gb = round(psutil.virtual_memory().available / (1024**3), 2)
disk_free_gb = round(psutil.disk_usage(str(REPO_ROOT)).free / (1024**3), 2)

# 2. Git Status
def run_git(cmd):
    try:
        res = subprocess.run(["git"] + cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception as e:
        return f"git error: {e}"

git_branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
git_commit = run_git(["rev-parse", "HEAD"])
git_status_raw = run_git(["status", "--porcelain"])

tracked_modified = []
untracked = []
for line in git_status_raw.splitlines():
    if not line:
        continue
    code = line[:2]
    fname = line[3:].strip()
    if code == "??":
        untracked.append(fname)
    else:
        tracked_modified.append((code, fname))

# 3. Checkpoints Inspection
def hash_file(p: Path):
    h_sha = hashlib.sha256()
    h_md5 = hashlib.md5()
    with open(p, "rb") as f:
        while chunk := f.read(1024 * 1024 * 8):
            h_sha.update(chunk)
            h_md5.update(chunk)
    return h_sha.hexdigest().upper(), h_md5.hexdigest().lower()

ckpts_dir = REPO_ROOT / "checkpoints"
ckpt_records = []
if ckpts_dir.exists():
    for cp in sorted(ckpts_dir.glob("*.pt")):
        size_bytes = cp.stat().st_size
        sha256_val, md5_val = hash_file(cp)
        ckpt_records.append({
            "name": cp.name,
            "path": str(cp.relative_to(REPO_ROOT)),
            "size_bytes": size_bytes,
            "md5": md5_val,
            "sha256": sha256_val
        })

# 4. Config Parse Validity
config_dir = REPO_ROOT / "config"
config_validity = {}
if config_dir.exists():
    for cfg_p in sorted(config_dir.glob("*.yaml")):
        try:
            with open(cfg_p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            config_validity[cfg_p.name] = {"valid": True, "top_keys": list(data.keys()) if isinstance(data, dict) else type(data).__name__}
        except Exception as e:
            config_validity[cfg_p.name] = {"valid": False, "error": str(e)}

# 5. Pytest Discovery
pytest_collect = subprocess.run([python_exe, "-m", "pytest", "--collect-only", "-q"], cwd=str(REPO_ROOT), capture_output=True, text=True)
collect_out = pytest_collect.stdout
collect_lines = [l for l in collect_out.splitlines() if l and not l.startswith("=")]

baseline_data = {
    "repository_root": str(REPO_ROOT),
    "environment": {
        "python_executable": python_exe,
        "python_version": python_ver,
        "conda_default_env": conda_env,
        "platform": platform.platform(),
        "cuda_available": cuda_avail,
        "gpu_device": device_name,
        "vram_total_gb": vram_gb,
        "ram_total_gb": ram_total_gb,
        "ram_available_gb": ram_avail_gb,
        "disk_free_gb": disk_free_gb,
        "torch_version": torch_ver,
        "packages": pkg_versions
    },
    "git": {
        "branch": git_branch,
        "commit": git_commit,
        "tracked_modified_count": len(tracked_modified),
        "tracked_modified": tracked_modified,
        "untracked_count": len(untracked),
        "untracked_sample": untracked[:30]
    },
    "checkpoints": ckpt_records,
    "config_parse_validity": config_validity,
    "pytest_discovery": {
        "exit_code": pytest_collect.returncode,
        "summary_line": collect_lines[-1] if collect_lines else "unknown",
        "collected_items_sample": collect_lines[:25]
    }
}

out_json = REPO_ROOT / "reports" / "phase0_hardening_baseline_report.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(baseline_data, f, indent=2)

print(f"Baseline audit JSON saved to: {out_json}")
print(f"Git Branch: {git_branch} (Commit: {git_commit[:8]})")
print(f"Python: {python_exe} ({conda_env})")
print(f"GPU: {device_name} (CUDA={cuda_avail}, VRAM={vram_gb} GB)")
print(f"Tracked modified files: {len(tracked_modified)}")
print(f"Untracked files: {len(untracked)}")
print(f"Checkpoints: {len(ckpt_records)}")
print(f"Pytest collection: {collect_lines[-1] if collect_lines else 'None'}")

