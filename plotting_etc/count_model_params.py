"""Count trainable parameters for each saved HNN model."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from HNN_helper import PHVIV, parse_config

MODELS = [
    ("MLP", Path("models/pirate_smoke_0122-125008.pt")),
]


def load_model(model_path: Path):
    ckpt = torch.load(model_path, map_location="cpu")
    cfg_raw = ckpt.get("config", {})
    if not isinstance(cfg_raw, dict):
        if hasattr(cfg_raw, "__dict__"):
            cfg_raw = dict(cfg_raw.__dict__)
        else:
            raise TypeError(f"Unsupported config type in checkpoint: {type(cfg_raw)}")
    cfg = parse_config(cfg_raw)
    model, _ = PHVIV.from_config(
        dt=ckpt.get("dt", 1e-3),
        cfg=dict(cfg.model.__dict__),
        arch_cfg=dict(cfg.architecture.__dict__),
        device=torch.device("cpu"),
    )
    incompatible = model.load_state_dict(ckpt["model_state"], strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            f"[warn] {model_path.name}: missing_keys={incompatible.missing_keys}, "
            f"unexpected_keys={incompatible.unexpected_keys}"
        )
    return model


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    for name, path in MODELS:
        model = load_model(path)
        total = count_params(model)
        print(f"{name}: {total:,} trainable parameters")


if __name__ == "__main__":
    main()
