"""
Generate multiple TD-model data series for different initial conditions.
"""

from __future__ import annotations

import numpy as np
import json
from itertools import product
from pathlib import Path
from typing import Iterable

from simulate_td_model_cf import Ca, D, K, M, U as U_DEFAULT, rho, simulate_td_model_cf


def run_batch(
    amplitude_factors: Iterable[float],
    fhat_values: Iterable[float],
    reduced_velocity_values: Iterable[float],
    output_dir: Path,
    *,
    integrator: str = "rk4",
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    seed: int = 1234,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    test_dir = output_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    if train_fraction <= 0.0 or val_fraction <= 0.0 or (train_fraction + val_fraction) >= 1.0:
        raise ValueError("train/val fractions must be positive and sum to less than 1.")
    rng = np.random.default_rng(int(seed))
    metadata: list[dict[str, float | str | int]] = []
    idx = 1

    amp_list = list(amplitude_factors)
    fhat_list = list(fhat_values)

    for ur_val in reduced_velocity_values:
        combos = list(product(amp_list, fhat_list))
        rng.shuffle(combos)
        n_total = len(combos)
        n_train = int(np.floor(n_total * train_fraction))
        n_val = int(np.floor(n_total * val_fraction))
        n_test = n_total - n_train - n_val
        if n_train == 0 or n_val == 0 or n_test == 0:
            raise ValueError("Split sizes are too small; adjust fractions or add more samples.")

        split_map = {
            "train": (train_dir, combos[:n_train]),
            "val": (val_dir, combos[n_train : n_train + n_val]),
            "test": (test_dir, combos[n_train + n_val :]),
        }

        for split_name, (split_dir, split_combos) in split_map.items():
            for a_factor, fhat in split_combos:
                fname = split_dir / f"series_{idx:03d}_A{a_factor:.2f}_fhat{fhat:.3f}_Ur{ur_val:.2f}.npz"
                flow_speed = reduced_velocity_to_flow_speed(ur_val)
                simulate_td_model_cf(
                    A_factor=a_factor,
                    fhat=fhat,
                    U=flow_speed,
                    output_path=fname,
                    plot=False,
                    seed=idx,
                    verbose=False,
                    integrator=integrator,
                )
                metadata.append(
                    {
                        "index": idx,
                        "split": split_name,
                        "file": fname.name,
                        "A_factor": a_factor,
                        "fhat": fhat,
                        "U_r": float(ur_val),
                        "U": float(flow_speed),
                    }
                )
                idx += 1

    meta_path = output_dir / "metadata.json"
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"Generated {len(metadata)} series. Metadata saved to {meta_path}.")


def reduced_velocity_to_flow_speed(ur_val: float) -> float:
    m_eff = M + D**2 * np.pi / 4.0 * rho * Ca
    return float(ur_val * D / (2.0 * np.pi) / np.sqrt(m_eff / K))


def default_reduced_velocity() -> float:
    m_eff = M + D**2 * np.pi / 4.0 * rho * Ca
    return float(2.0 * np.pi * U_DEFAULT / D * np.sqrt(m_eff / K))


def main():
    output_dir = Path(__file__).parent / "generated_series_Ur"
    integrator = "rk4"
    train_fraction = 0.7
    val_fraction = 0.15
    split_seed = 1234

    amplitude_factors = [-1.0, -0.5, 0.0, 0.5, 1.0]
    fhat_values = [-0.25, -0.15, -0.05, 0.05, 0.15, 0.25]
    reduced_velocity_values = [
        2.0,
        4.0,
        6.0,
        8.0,
        10.0,
        12.0,
    ]
    run_batch(
        amplitude_factors,
        fhat_values,
        reduced_velocity_values,
        output_dir,
        integrator=integrator,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        seed=split_seed,
    )


if __name__ == "__main__":
    main()
