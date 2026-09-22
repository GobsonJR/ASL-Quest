from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def describe_device(device: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": None,
        "cuda_version": torch.version.cuda,
        "gpu_memory_gb": None,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(props.total_memory / (1024**3), 2)
    return info


def print_device_info() -> torch.device:
    device = get_device()
    info = describe_device(device)
    print("=" * 60)
    print(f"Device: {info['device']}")
    print(f"CUDA available: {info['cuda_available']}")
    if info["gpu_name"]:
        print(f"GPU: {info['gpu_name']}")
        print(f"CUDA: {info['cuda_version']}")
        print(f"GPU memory: {info['gpu_memory_gb']} GB")
    else:
        print("WARNING: CUDA GPU not detected. Training will use CPU.")
    print("=" * 60)
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
