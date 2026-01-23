from __future__ import annotations

import os
from typing import Tuple

import torch


def select_device(runtime_device: str) -> torch.device:
    requested = str(runtime_device).strip().lower()
    if requested in {"auto", ""}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def configure_tf32(device: torch.device, enabled: bool) -> None:
    if device.type != "cuda" or not enabled:
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    print("TF32 enabled")


def maybe_compile_model(model: torch.nn.Module, enabled: bool, mode: str) -> torch.nn.Module:
    if not enabled:
        return model
    try:
        compiled = torch.compile(model, mode=str(mode).strip() or "default")
        print(f"torch.compile enabled (mode={mode})")
        return compiled
    except Exception as exc:
        print(f"torch.compile failed ({exc}); continuing without compile")
        return model


def setup_amp(
    device: torch.device, *, use_amp: bool, amp_dtype: str
) -> Tuple[bool, torch.dtype, torch.cuda.amp.GradScaler]:
    enabled = bool(use_amp and device.type == "cuda")
    dtype_key = str(amp_dtype).strip().lower()
    chosen_dtype = torch.float16 if dtype_key in {"fp16", "float16"} else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=enabled and chosen_dtype == torch.float16)
    if enabled:
        print(f"AMP enabled (dtype={dtype_key})")
    return enabled, chosen_dtype, scaler


def set_num_threads_from_slurm(*, default: int = 1) -> int:
    threads = int(os.getenv("SLURM_CPUS_PER_TASK", str(default)))
    torch.set_num_threads(threads)
    return threads

