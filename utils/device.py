"""Hardware and device inspection utility for RoomAI.

Accurately detects GPU availability, device names, and VRAM allocations
without forcing a hard CUDA dependency on systems without a dedicated GPU.
"""

from typing import Any, Dict


def get_device_info() -> Dict[str, Any]:
    """Inspects the local hardware and returns GPU and memory specifications."""
    info: Dict[str, Any] = {
        "cuda_available": False,
        "device_name": "CPU",
        "device_count": 0,
        "cuda_version": "N/A",
        "pytorch_version": "N/A",
        "total_vram_gb": 0.0,
        "allocated_vram_gb": 0.0,
        "allocated_vram_mb": 0.0,
        "peak_vram_gb": 0.0,
        "peak_vram_mb": 0.0,
        "free_vram_gb": 0.0,
        "reserved_vram_mb": 0.0,
        "recommended_device": "cpu",
    }

    try:
        import torch

        info["pytorch_version"] = torch.__version__
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["device_count"] = torch.cuda.device_count()
            info["device_name"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = getattr(torch.version, "cuda", "Unknown") or "Unknown"

            total_bytes = torch.cuda.get_device_properties(0).total_memory
            allocated_bytes = torch.cuda.memory_allocated(0)
            reserved_bytes = torch.cuda.memory_reserved(0)
            max_allocated_bytes = torch.cuda.max_memory_allocated(0)

            total_gb = total_bytes / (1024**3)
            allocated_gb = allocated_bytes / (1024**3)
            free_gb = (total_bytes - reserved_bytes) / (1024**3)

            info["total_vram_gb"] = round(total_gb, 2)
            info["allocated_vram_gb"] = round(allocated_gb, 3)
            info["allocated_vram_mb"] = round(allocated_bytes / (1024**2), 1)
            info["peak_vram_gb"] = round(max_allocated_bytes / (1024**3), 3)
            info["peak_vram_mb"] = round(max_allocated_bytes / (1024**2), 1)
            info["free_vram_gb"] = round(free_gb, 2)
            info["reserved_vram_mb"] = round(reserved_bytes / (1024**2), 1)
            info["recommended_device"] = "cuda"
    except ImportError:
        pass
    except Exception as exc:
        info["error"] = str(exc)

    return info


def get_optimal_device() -> str:
    """Returns 'cuda' if a compatible GPU is available, else 'cpu'."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


get_device_string = get_optimal_device
