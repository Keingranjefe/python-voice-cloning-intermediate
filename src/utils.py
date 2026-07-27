"""Utilidades comunes del proyecto."""

import torch
from pathlib import Path


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def print_gpu_info():
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"VRAM libre: {torch.cuda.mem_get_info()[0] / 1024**3:.1f} GB")
    else:
        print("No se detectó GPU CUDA")


def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)
