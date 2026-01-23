from __future__ import annotations

import math


class WarmupCosineLrSchedule:
    def __init__(self, max_lr: float, min_lr: float, warmup_steps: int, decay_steps: int) -> None:
        """
        max_lr: peak learning rate after warmup
        min_lr: final learning rate at the end of cosine decay
        warmup_steps: number of linear warmup steps
        decay_steps: number of cosine decay steps
        """
        self.max_lr = float(max_lr)
        self.min_lr = float(min_lr)
        self.warmup_steps = int(warmup_steps)
        self.decay_steps = int(decay_steps)

    def get_lr(self, step: int) -> float:
        if step <= self.warmup_steps:
            return self.max_lr * step / max(self.warmup_steps, 1)

        t = step - self.warmup_steps
        if t >= self.decay_steps:
            return self.min_lr

        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * t / max(self.decay_steps, 1)))
        return self.min_lr + (self.max_lr - self.min_lr) * cosine_decay


class WarmupExponentialLrSchedule:
    def __init__(self, max_lr: float, min_lr: float, warmup_steps: int, total_steps: int) -> None:
        if min_lr <= 0 or max_lr <= 0:
            raise ValueError("Learning rates must be positive for exponential schedule")
        if total_steps <= 0:
            raise ValueError("total_steps must be positive for exponential schedule")
        self.max_lr = float(max_lr)
        self.min_lr = float(min_lr)
        self.warmup_steps = int(max(warmup_steps, 0))
        self.total_steps = int(max(total_steps, self.warmup_steps + 1))
        self.decay_steps = max(self.total_steps - self.warmup_steps, 1)
        ratio = self.min_lr / self.max_lr
        self.decay_base = ratio ** (1.0 / self.decay_steps)

    def get_lr(self, step: int) -> float:
        if self.warmup_steps > 0 and step <= self.warmup_steps:
            return self.max_lr * step / self.warmup_steps
        t = min(max(step - self.warmup_steps, 0), self.decay_steps)
        lr = self.max_lr * (self.decay_base**t)
        return max(min(lr, self.max_lr), self.min_lr)

