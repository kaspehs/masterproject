from __future__ import annotations

import argparse
from pathlib import Path

from HNN_helper import load_config, parse_config
from methods import get_trainer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train models (HNN/PINN) from YAML configuration.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    raw_cfg = load_config(args.config)
    cfg = parse_config(raw_cfg)
    config_name = args.config.stem
    trainer = get_trainer(getattr(cfg, "method", "hnn"))
    trainer(cfg, config_name)


if __name__ == "__main__":
    main()

