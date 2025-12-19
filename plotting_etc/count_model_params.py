"""Count trainable parameters for each saved HNN model."""

from __future__ import annotations

from pathlib import Path

import torch

from HNN_helper import PHVIV, parse_config

MODELS = [
    ("MLP", Path("models/mlp_final1_1129-235240.pt")),
    ("ResNet", Path("models/residual_final1_1130-115801.pt")),
    ("PirateNet", Path("models/pirate_final1_1130-085457.pt")),
]


def load_model(model_path: Path):
    ckpt = torch.load(model_path, map_location="cpu")
    cfg = parse_config(ckpt["config"])
    model, _ = PHVIV.from_config(
        dt=ckpt.get("dt", 1e-3),
        cfg=dict(cfg.model.__dict__),
        arch_cfg=dict(cfg.architecture.__dict__),
        device=torch.device("cpu"),
    )
    model.load_state_dict(ckpt["model_state"])
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
