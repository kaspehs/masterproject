from __future__ import annotations

from typing import Any, Tuple

import torch
import torch.optim as optim

from ODE_pinn_helper import WarmupCosineLrSchedule, WarmupExponentialLrSchedule


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
    if optimizer_type == "adamw":
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "adam":
        opt = optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer '{optim_cfg.optimizer}'. Use 'adam' or 'adamw'.")

    max_lr = float(scheduler_cfg.max_lr)
    scheduler_warmup_steps = int(scheduler_cfg.warmup_steps)
    decay_steps = int(scheduler_cfg.decay_steps)
    min_lr = float(getattr(scheduler_cfg, "min_lr", 0.02 * max_lr))

    scheduler_type = scheduler_cfg.scheduler_type.lower() if hasattr(scheduler_cfg, "scheduler_type") else "cosine"
    if scheduler_type == "cosine":
        lr_scheduler = WarmupCosineLrSchedule(max_lr, min_lr, scheduler_warmup_steps, decay_steps)
    elif scheduler_type == "exponential":
        lr_scheduler = WarmupExponentialLrSchedule(max_lr, min_lr, scheduler_warmup_steps, epochs)
    else:
        raise ValueError(f"Unknown scheduler_type '{scheduler_type}'. Use 'cosine' or 'exponential'.")

    return opt, lr_scheduler

