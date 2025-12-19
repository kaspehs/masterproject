"""
Generate multiple TD-model data series for different initial conditions.
"""

from __future__ import annotations

import numpy as np
import argparse
import json
from itertools import product
from pathlib import Path
from typing import Iterable

from simulate_td_model_cf import simulate_td_model_cf


def run_batch(
    amplitude_factors: Iterable[float],
    fhat_values: Iterable[float],
    output_dir: Path,
    integrator: str = "rk4",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata: list[dict[str, float | str | int]] = []

    for idx, (a_factor, fhat) in enumerate(product(amplitude_factors, fhat_values), start=1):
        fname = output_dir / f"series_{idx:03d}_A{a_factor:.2f}_fhat{fhat:.3f}.npz"
        simulate_td_model_cf(
            A_factor=a_factor,
            fhat=fhat,
            output_path=fname,
            plot=False,
            seed=idx,
            verbose=False,
            integrator=integrator,
        )
        metadata.append(
            {
                "index": idx,
                "file": fname.name,
                "A_factor": a_factor,
                "fhat": fhat,
            }
        )

    meta_path = output_dir / "metadata.json"
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"Generated {len(metadata)} series. Metadata saved to {meta_path}.")


def main():
    parser = argparse.ArgumentParser(description="Generate multiple TD-model series.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "generated_series",
        help="Directory to store generated .npz files.",
    )
    parser.add_argument(
        "--integrator",
        choices=("rk4", "euler"),
        default="rk4",
        help="Time-integration method to use.",
    )
    args = parser.parse_args()

    amplitude_factors = [0.1, 0.3, 0.7, 0.9]
    fhat_values = [0.05, 0.10, 0.20, 0.25]
    run_batch(amplitude_factors, fhat_values, args.output_dir, integrator=args.integrator)


if __name__ == "__main__":
    main()
