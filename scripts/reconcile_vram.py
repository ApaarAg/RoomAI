"""
RoomAI — Comprehensive VRAM & Memory Accounting Suite
Reconciles:
1. Checkpoint file size on disk.
2. Parameter count & theoretical parameter memory (fp32).
3. PyTorch static allocated & reserved memory.
4. Dynamic inference peak memory.
5. nvidia-smi process memory footprint.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image

# Suppress TensorFlow probing
os.environ["TRANSFORMERS_NO_TF"] = "1"
sys.modules["tensorflow"] = None

import torch
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation, AutoModelForDepthEstimation
from ultralytics import YOLO, FastSAM

REPO_ROOT = Path(__file__).resolve().parent.parent


def get_nvidia_smi_memory():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            text=True
        )
        pid = os.getpid()
        for line in out.strip().split("\n"):
            parts = line.strip().split(",")
            if len(parts) == 2 and int(parts[0].strip()) == pid:
                return float(parts[1].strip())
    except Exception as e:
        return None
    return None


def main():
    print("===============================================================")
    print("RoomAI VRAM Accounting & Memory Footprint Reconciliation")
    print("===============================================================")

    models_info = [
        {
            "name": "Model A3 (YOLO11s 800px)",
            "disk_path": REPO_ROOT / "checkpoints" / "indoor_detector_a3_yolo11s_800px_best.pt",
            "loader": lambda: YOLO(str(REPO_ROOT / "checkpoints" / "indoor_detector_a3_yolo11s_800px_best.pt")).to("cuda"),
        },
        {
            "name": "FastSAM-s",
            "disk_path": Path("FastSAM-s.pt"),
            "loader": lambda: FastSAM("FastSAM-s.pt").to("cuda"),
        },
        {
            "name": "SegFormer-B1",
            "disk_path": None,  # In HuggingFace hub cache
            "hf_id": "nvidia/segformer-b1-finetuned-ade-512-512",
            "loader": lambda: AutoModelForSemanticSegmentation.from_pretrained("nvidia/segformer-b1-finetuned-ade-512-512").to("cuda"),
        },
        {
            "name": "Depth Anything V2 Metric Small",
            "disk_path": None,
            "hf_id": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
            "loader": lambda: AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf").to("cuda"),
        },
    ]

    # Baseline CUDA context memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _ = torch.zeros(1, device="cuda")
    base_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    base_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    base_smi = get_nvidia_smi_memory()

    print(f"CUDA Context Baseline: Allocated = {base_allocated:.1f} MB | Reserved = {base_reserved:.1f} MB | nvidia-smi = {base_smi} MB\n")

    loaded_models = []
    reconciliation = []

    for m in models_info:
        torch.cuda.reset_peak_memory_stats()
        alloc_before = torch.cuda.memory_allocated() / (1024 ** 2)

        model_obj = m["loader"]()
        loaded_models.append(model_obj)

        alloc_after = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved_now = torch.cuda.memory_reserved() / (1024 ** 2)
        smi_now = get_nvidia_smi_memory()

        # Parameter count
        if hasattr(model_obj, "model") and hasattr(model_obj.model, "parameters"):
            n_params = sum(p.numel() for p in model_obj.model.parameters())
        elif hasattr(model_obj, "parameters"):
            n_params = sum(p.numel() for p in model_obj.parameters())
        else:
            n_params = 0

        param_mb_fp32 = (n_params * 4) / (1024 ** 2)

        # File size
        if m["disk_path"] and m["disk_path"].exists():
            disk_mb = m["disk_path"].stat().st_size / (1024 ** 2)
        else:
            disk_mb = param_mb_fp32  # Approximate for HF cached safetensors

        reconciliation.append({
            "name": m["name"],
            "parameters_m": round(n_params / 1e6, 2),
            "disk_mb": round(disk_mb, 1),
            "param_mem_fp32_mb": round(param_mb_fp32, 1),
            "static_allocated_delta_mb": round(alloc_after - alloc_before, 1),
            "cumulative_allocated_mb": round(alloc_after, 1),
            "cumulative_reserved_mb": round(reserved_now, 1),
            "nvidia_smi_process_mb": smi_now,
        })

    print(f"{'Model Name':30s} | {'Params':7s} | {'Disk MB':7s} | {'Param MB':8s} | {'Alloc dMB':9s} | {'Cumul Alloc':11s} | {'Cumul Res':9s} | {'smi MB':6s}")
    print("-" * 115)
    for r in reconciliation:
        print(f"{r['name']:30s} | {r['parameters_m']:5.2f} M | {r['disk_mb']:7.1f} | {r['param_mem_fp32_mb']:8.1f} | {r['static_allocated_delta_mb']:9.1f} | {r['cumulative_allocated_mb']:11.1f} | {r['cumulative_reserved_mb']:9.1f} | {str(r['nvidia_smi_process_mb']):6s}")

    # Now run 1 inference cycle to capture dynamic memory
    torch.cuda.reset_peak_memory_stats()
    dummy_np = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_img = Image.new("RGB", (640, 480), color="white")

    proc_sem = AutoImageProcessor.from_pretrained("nvidia/segformer-b1-finetuned-ade-512-512")
    proc_depth = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf")

    # Forward passes
    _ = loaded_models[0](dummy_np, imgsz=800, verbose=False)
    _ = loaded_models[1](dummy_np, bboxes=[[10, 10, 100, 100]], verbose=False)
    in_s = proc_sem(images=dummy_img, return_tensors="pt").to("cuda")
    with torch.no_grad():
        _ = loaded_models[2](**in_s)
    in_d = proc_depth(images=dummy_img, return_tensors="pt").to("cuda")
    with torch.no_grad():
        _ = loaded_models[3](**in_d)

    torch.cuda.synchronize()
    dynamic_peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
    dynamic_peak_res = torch.cuda.max_memory_reserved() / (1024 ** 2)
    smi_peak = get_nvidia_smi_memory()

    print("\n--- Unified Inference Peak Memory Breakdown ---")
    print(f"Total Static Model Weights in VRAM : {reconciliation[-1]['cumulative_allocated_mb']:.1f} MB")
    print(f"Peak Dynamic Inference Activations : {dynamic_peak_alloc - reconciliation[-1]['cumulative_allocated_mb']:.1f} MB")
    print(f"Total PyTorch Peak Allocated Memory: {dynamic_peak_alloc:.1f} MB ({dynamic_peak_alloc/1024:.2f} GB)")
    print(f"Total PyTorch Peak Reserved Memory : {dynamic_peak_res:.1f} MB ({dynamic_peak_res/1024:.2f} GB)")
    if smi_peak is not None:
        print(f"Total Process Memory (nvidia-smi)  : {smi_peak} MB ({smi_peak/1024:.2f} GB)")
        print(f"Windows WDDM & CUDA Driver Overhead: {smi_peak - dynamic_peak_res:.1f} MB")
    else:
        print("Total Process Memory (nvidia-smi)  : unavailable (process not matched in nvidia-smi compute-apps query)")

    # Save reconciliation report
    report = {
        "cuda_context_baseline": {
            "allocated_mb": round(base_allocated, 1),
            "reserved_mb": round(base_reserved, 1),
            "nvidia_smi_mb": base_smi,
        },
        "per_model_reconciliation": reconciliation,
        "unified_inference_peak": {
            "total_static_model_weights_mb": round(reconciliation[-1]['cumulative_allocated_mb'], 1),
            "peak_dynamic_inference_activations_mb": round(dynamic_peak_alloc - reconciliation[-1]['cumulative_allocated_mb'], 1),
            "total_pytorch_peak_allocated_mb": round(dynamic_peak_alloc, 1),
            "total_pytorch_peak_allocated_gb": round(dynamic_peak_alloc / 1024, 2),
            "total_pytorch_peak_reserved_mb": round(dynamic_peak_res, 1),
            "total_pytorch_peak_reserved_gb": round(dynamic_peak_res / 1024, 2),
            "nvidia_smi_process_mb": smi_peak,
        },
    }
    out_json = REPO_ROOT / "reports" / "vram_reconciliation_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nVRAM reconciliation report saved to: {out_json}")


if __name__ == "__main__":
    main()
