"""
Training utilities for University-FineTuned-Assistant.
Handles hardware detection, memory estimation, and dataset formatting.
"""

import sys
import torch
from typing import Dict, Any, Tuple


def get_hardware_info() -> Dict[str, Any]:
    """Detect and return detailed system hardware capabilities."""
    info = {
        "platform": sys.platform,
        "device": "cpu",
        "device_name": "CPU",
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "gpu_count": 0,
        "vram_gb": 0.0,
        "supports_4bit": False,
        "recommended_dtype": torch.float32
    }

    if info["cuda_available"]:
        info["device"] = "cuda"
        info["gpu_count"] = torch.cuda.device_count()
        info["device_name"] = torch.cuda.get_device_name(0)
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        info["vram_gb"] = round(vram_bytes / (1024 ** 3), 2)
        info["supports_4bit"] = True
        info["recommended_dtype"] = torch.float16
    elif info["mps_available"]:
        info["device"] = "mps"
        info["device_name"] = "Apple Silicon (MPS)"
        info["supports_4bit"] = False
        info["recommended_dtype"] = torch.float16
    else:
        info["device"] = "cpu"
        info["device_name"] = "Standard Laptop CPU"
        info["supports_4bit"] = False
        info["recommended_dtype"] = torch.float32

    return info


def print_hardware_summary(hw_info: Dict[str, Any]):
    """Print clean, transparent hardware diagnostics."""
    print("=" * 60)
    print("             HARDWARE & ENVIRONMENT DETECTION")
    print("=" * 60)
    print(f"Device Mode:          {hw_info['device'].upper()}")
    print(f"Hardware Name:        {hw_info['device_name']}")
    if hw_info['cuda_available']:
        print(f"VRAM Capacity:        {hw_info['vram_gb']} GB")
        print(f"4-bit Quantization:   Supported (via bitsandbytes)")
    elif hw_info['mps_available']:
        print(f"Acceleration:         Apple Metal Performance Shaders (MPS)")
    else:
        print(f"GPU Status:           None detected. Running in lightweight CPU mode.")
        print("Note on CPU Training: Fine-tuning on CPU works for small models (e.g. 0.5B),")
        print("                      but will take several minutes to an hour depending on CPU.")
    print("=" * 60 + "\n")


def print_trainable_parameters(model) -> Tuple[int, int, float]:
    """Prints and returns trainable parameters vs total parameters."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    pct = 100 * trainable_params / all_param if all_param > 0 else 0
    print(f"[*] Trainable parameters: {trainable_params:,}")
    print(f"[*] Total parameters:     {all_param:,}")
    print(f"[*] Trainable percentage: {pct:.4f}%")
    return trainable_params, all_param, pct
