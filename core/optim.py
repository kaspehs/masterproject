from __future__ import annotations

from typing import Any, Tuple

import math

import torch
import torch.optim as optim

from core.lr_schedules import WarmupCosineLrSchedule, WarmupExponentialLrSchedule


def setup_optimizer_and_scheduler(
    model: torch.nn.Module,
    *,
    optim_cfg: Any,
    scheduler_cfg: Any,
    epochs: int,
) -> Tuple[optim.Optimizer, Any]:
    optimizer_type = str(optim_cfg.optimizer).lower()
    lr = float(optim_cfg.lr)
    weight_decay = float(optim_cfg.weight_decay)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters remain after applying freeze settings.")
    if optimizer_type == "adamw":
        opt = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "adam":
        opt = optim.Adam(trainable_params, lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer '{optim_cfg.optimizer}'. Use 'adam' or 'adamw'.")

    max_lr = float(scheduler_cfg.max_lr)
    warmup_fraction = getattr(scheduler_cfg, "warmup_fraction", None)
    if warmup_fraction is not None:
        warmup_fraction = float(warmup_fraction)
        if warmup_fraction < 0.0 or warmup_fraction > 1.0:
            raise ValueError("scheduler.warmup_fraction must be between 0 and 1.")
        if warmup_fraction == 0.0:
            scheduler_warmup_steps = 0
        else:
            scheduler_warmup_steps = int(max(1, math.ceil(float(epochs) * warmup_fraction)))
    else:
        scheduler_warmup_steps = int(scheduler_cfg.warmup_steps)
    decay_steps = int(scheduler_cfg.decay_steps)
    min_lr = float(getattr(scheduler_cfg, "min_lr", 0.02 * max_lr))

    scheduler_type = scheduler_cfg.scheduler_type.lower() if hasattr(scheduler_cfg, "scheduler_type") else "cosine"
    if warmup_fraction is not None:
        decay_steps = max(1, int(epochs) - int(scheduler_warmup_steps))

    if scheduler_type == "cosine":
        lr_scheduler = WarmupCosineLrSchedule(max_lr, min_lr, scheduler_warmup_steps, decay_steps)
    elif scheduler_type == "exponential":
        lr_scheduler = WarmupExponentialLrSchedule(max_lr, min_lr, scheduler_warmup_steps, epochs)
    else:
        raise ValueError(f"Unknown scheduler_type '{scheduler_type}'. Use 'cosine' or 'exponential'.")

    return opt, lr_scheduler
