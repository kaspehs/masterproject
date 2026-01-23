from __future__ import annotations

from typing import Callable

from HNN_helper import Config

Trainer = Callable[[Config, str], None]


def get_trainer(method: str) -> Trainer:
    key = str(method or "").strip().lower()
    if key in {"hnn", "phnn"}:
        from methods.hnn.trainer import train as train_hnn

        return train_hnn
    if key in {"pinn"}:
        from methods.pinn.trainer import train as train_pinn

        return train_pinn
    if key in {"vpinn", "weakpinn", "windowed_vpinn"}:
        from methods.vpinn.trainer import train as train_vpinn

        return train_vpinn
    raise ValueError(f"Unknown method '{method}'. Expected 'hnn', 'pinn', or 'vpinn'.")
