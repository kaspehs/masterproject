from __future__ import annotations

from typing import Callable

from training.training_utils import Config

Trainer = Callable[[Config, str], None]


def get_trainer(method: str) -> Trainer:
    key = str(method or "").strip().lower()
    if key in {"hnn", "phnn"}:
        from training.methods.hnn.trainer import train as train_hnn

        return train_hnn
    if key in {"latent_rnn", "scratch_latent_rnn"}:
        from training.methods.latent_rnn.trainer import train as train_latent_rnn

        return train_latent_rnn
    raise ValueError(f"Unknown method '{method}'. Expected 'hnn'/'phnn' or 'latent_rnn'.")
