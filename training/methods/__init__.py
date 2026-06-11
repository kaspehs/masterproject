from __future__ import annotations

from typing import Callable

from training.training_utils import Config

Trainer = Callable[[Config, str], None]


def get_trainer(method: str) -> Trainer:
    key = str(method or "").strip().lower()
    if key in {"correction", "hnn", "phnn", "td_correction"}:
        from training.methods.correction.trainer import train as train_correction

        return train_correction
    if key in {"standalone", "latent_rnn", "scratch_latent_rnn"}:
        from training.methods.standalone.trainer import train as train_standalone

        return train_standalone
    raise ValueError(f"Unknown method '{method}'. Expected 'correction' or 'standalone'.")
