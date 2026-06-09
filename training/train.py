from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train models from YAML configuration.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    from training.training_utils import load_config, parse_config
    from training.methods import get_trainer

    raw_cfg = load_config(args.config)
    cfg = parse_config(raw_cfg)
    config_name = args.config.stem
    trainer = get_trainer(getattr(cfg, "method", "hnn"))
    trainer(cfg, config_name)


if __name__ == "__main__":
    main()
