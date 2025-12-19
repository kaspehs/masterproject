"""Aggregate multiple training runs per group and plot mean/std metrics."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


@dataclass
class LogGroup:
    label: str
    paths: list[str]


LOG_GROUPS: list[LogGroup] = [
    LogGroup(
        label="MLP Model",
        paths=[
            "HNNruns/mlp_final1_1129-235240",
            "HNNruns/mlp_final2_1130-023755",
            "HNNruns/mlp_final3_1129-235240",
            "HNNruns/mlp_final4_1130-101504",
            "HNNruns/mlp_final5_1130-083338",
        ],
    ),
    LogGroup(
        label="ResNet model",
        paths=[
            "HNNruns/residual_final1_1130-115801",
            "HNNruns/residual_final2_1130-123142",
            "HNNruns/residual_final3_1130-130911",
            "HNNruns/residual_final4_1130-102027",
            "HNNruns/residual_final5_1130-115308",
            ],
    ),
    LogGroup(
        label="PirateNet model",
        paths=[
            "HNNruns/pirate_final1_1130-085457",
            "HNNruns/pirate_final2_1130-052213",
            "HNNruns/pirate_final3_1130-023755",
            "HNNruns/pirate_final4_1129-235240",
            "HNNruns/pirate_final5_1129-235240",
            ],
    ),
]

METRICS = {
    "Validation": [
        ("val/rel_rmse_y", "NRMSEy", 1e-3, 1e-1),
        ("val/rel_rmse_force_total", "NRMSEf", 1e-2, 2e-2),
        ("val/force_spectral_rel_error_second_half", "Relative Spectral Error", None, None),
        ("val/force_fatigue_damage_rel_error", "Relative Fatigue Error", None, None),
    ],
    "Training": [
        ("train/residual_loss", "Residual Loss", 1e-3, 1e0),
    ],
}

SMOOTH_WINDOW = 25
STD_WINDOW = 50
OUTPUT_DIR = Path("figs")
SUMMARY_CSV = OUTPUT_DIR / "metric_summary.csv"
EPS = 1e-12


def load_scalar(log_path: str, tag: str) -> tuple[np.ndarray, np.ndarray]:
    ea = EventAccumulator(log_path, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    if tag not in tags:
        raise ValueError(f"Tag '{tag}' not found in {log_path}. Available: {tags}")
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events], dtype=float)
    vals = np.array([e.value for e in events], dtype=float)
    idx = np.argsort(steps)
    return steps[idx], vals[idx]


def smooth(values: np.ndarray, window: int, poly: int = 2) -> np.ndarray:
    n = len(values)
    if n <= 2:
        return values
    win = min(window, n - (1 - n % 2))
    if win < 3:
        return values
    if win % 2 == 0:
        win -= 1
    return savgol_filter(values, window_length=win, polyorder=poly)


def running_std(values: np.ndarray, window: int) -> np.ndarray:
    n = len(values)
    half = window // 2
    out = np.zeros_like(values)
    for i in range(n):
        l = max(0, i - half)
        r = min(n, i + half + 1)
        out[i] = np.std(values[l:r])
    return out


def aggregate_group(paths: Sequence[str], metric: str) -> tuple[np.ndarray, np.ndarray]:
    series = []
    length = None
    steps_ref = None
    for path in paths:
        steps, vals = load_scalar(path, metric)
        if length is None or len(steps) < length:
            length = len(steps)
            steps_ref = steps[:length]
        series.append(vals)
    trimmed = np.stack([vals[:length] for vals in series], axis=0)
    return steps_ref, trimmed


def plot_metric(section: str, metric: str, title: str, vmin: float | None, vmax: float | None):
    fig, ax = plt.subplots(figsize=(8, 5))
    for group in LOG_GROUPS:
        try:
            steps, values = aggregate_group(group.paths, metric)
        except ValueError as e:
            print(e)
            continue
        safe_vals = np.clip(np.abs(values), EPS, None)
        log_vals = np.log(safe_vals)
        log_mean = smooth(log_vals.mean(axis=0), SMOOTH_WINDOW)
        log_std = smooth(log_vals.std(axis=0), STD_WINDOW)
        mean = np.exp(log_mean)
        lower = np.exp(log_mean - log_std)
        upper = np.exp(log_mean + log_std)
        ax.plot(steps, mean, linewidth=2, label=group.label)
        ax.fill_between(steps, lower, upper, alpha=0.2)
        final_value = values[:, -1]
        final_mean = float(np.mean(final_value))
        final_std = float(np.std(final_value))
        with SUMMARY_CSV.open("a") as fh:
            fh.write(f"{section},{metric},{group.label},{final_mean:.6e},{final_std:.6e}\n")
    if vmin is not None or vmax is not None:
        ax.set_ylim(bottom=vmin, top=vmax)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(title)
    ax.set_title(f"{section}: {title}")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"agg_{metric.replace('/', '_')}.png"
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / fname, dpi=200)
    plt.close(fig)


def main():
    for section, items in METRICS.items():
        for metric, title, vmin, vmax in items:
            plot_metric(section, metric, title, vmin, vmax)


if __name__ == "__main__":
    main()
