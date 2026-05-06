"""Plot smoothed aggregate validation errors from TensorBoard logs.

Edit the config block below, then run:

    python plotting_etc/plot_tb_validation_aggregate.py
"""

from __future__ import annotations

import os
import tempfile
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_CACHE_ROOT = Path(tempfile.gettempdir()) / "masterproject_plot_cache"
(_CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
(_CACHE_ROOT / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
from tensorboard.util import tensor_util


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Root folder containing TensorBoard run directories. Discovery is recursive,
# so this works for layouts like logs/group_name/run_name/events.out.tfevents...
LOG_ROOT = Path("logs")

# Leave empty to auto-discover runs under LOG_ROOT. If set, these paths are used
# exactly, relative to the repository root or absolute.
RUN_DIRS: list[str] = []

# Number of discovered runs to include. Set to None to use all discovered runs.
NUM_RUNS: int | None = None

# Used only when RUN_DIRS is empty.
RUN_NAME_CONTAINS: str | None = None
SORT_RUNS_BY = "mtime_desc"  # "mtime_desc", "mtime_asc", or "name"

OUTPUT_DIR = Path("figs/tensorboard_validation")
OUTPUT_BASENAME = "aggregate_validation_seen_vs_unseen"
DPI = 300

PLOT_TITLE: str | None = None
X_LABEL = "Epoch"
Y_LABEL = "Aggregate validation error"
Y_SCALE = "log"  # "linear" or "log"
GRID_ALPHA = 0.25
FIGSIZE = (10.0, 7.0)
SHARE_X_AXIS = True

# Smoothing is applied across logged validation points, not raw epoch integers.
# Set to 1 to disable smoothing.
SMOOTH_WINDOW_POINTS = 7

# Draw per-run rolling std bands around each smoothed line. Since each run is
# plotted separately, this is temporal variation within a run, not cross-run std.
PLOT_ROLLING_STD_BANDS = True
BAND_STD_MULTIPLIER = 1.0
BAND_ALPHA = 0.10
LINE_WIDTH = 1.8

# Log-space aggregation is usually better for positive error metrics spanning
# orders of magnitude. Set False for arithmetic smoothing/std in linear space.
USE_LOG_SPACE_SMOOTHING = True
EPS = 1e-12

SAVE_PNG = True
SAVE_PDF = False
SAVE_CSV = True
SHOW_FIGURE = False


@dataclass(frozen=True)
class MetricConfig:
    tag: str
    label: str
    title: str


METRICS: list[MetricConfig] = [
    MetricConfig(
        tag="val_seen/Aggregate validation error",
        label="val_seen",
        title="Seen validation",
    ),
    MetricConfig(
        tag="val_unseen/Aggregate validation error",
        label="val_unseen",
        title="Unseen validation",
    ),
]


# ---------------------------------------------------------------------------
# TensorBoard loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScalarSeries:
    steps: np.ndarray
    values: np.ndarray


def _event_files(run_dir: Path) -> list[Path]:
    return sorted(run_dir.rglob("events.out.tfevents*"))


def _run_mtime(run_dir: Path) -> float:
    event_files = _event_files(run_dir)
    if event_files:
        return max(path.stat().st_mtime for path in event_files)
    return run_dir.stat().st_mtime


def discover_run_dirs() -> list[Path]:
    if RUN_DIRS:
        return [Path(path).expanduser() for path in RUN_DIRS]

    run_dirs = sorted({path.parent for path in LOG_ROOT.rglob("events.out.tfevents*")})
    if RUN_NAME_CONTAINS:
        run_dirs = [path for path in run_dirs if RUN_NAME_CONTAINS in str(path)]

    if SORT_RUNS_BY == "mtime_desc":
        run_dirs.sort(key=_run_mtime, reverse=True)
    elif SORT_RUNS_BY == "mtime_asc":
        run_dirs.sort(key=_run_mtime)
    elif SORT_RUNS_BY == "name":
        run_dirs.sort(key=lambda path: str(path))
    else:
        raise ValueError(f"Unsupported SORT_RUNS_BY={SORT_RUNS_BY!r}")

    if NUM_RUNS is not None:
        run_dirs = run_dirs[: int(NUM_RUNS)]
    return run_dirs


def _summary_value_to_float(value) -> float | None:
    if value.HasField("tensor"):
        arr = tensor_util.make_ndarray(value.tensor)
        if arr.size != 1:
            return None
        return float(arr.reshape(-1)[0])
    return float(value.simple_value)


def load_selected_scalars(run_dir: Path, tags: Iterable[str]) -> dict[str, ScalarSeries]:
    tags_set = set(tags)
    by_tag: dict[str, dict[int, float]] = {tag: {} for tag in tags_set}
    event_files = _event_files(run_dir)
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found under {run_dir}")

    for event_file in event_files:
        try:
            for event in EventFileLoader(str(event_file)).Load():
                if not event.summary.value:
                    continue
                step = int(event.step)
                for value in event.summary.value:
                    if value.tag not in tags_set:
                        continue
                    scalar_value = _summary_value_to_float(value)
                    if scalar_value is None or not np.isfinite(scalar_value):
                        continue
                    by_tag[value.tag][step] = scalar_value
        except Exception as exc:  # pragma: no cover - protects against partial event files.
            print(f"Warning: failed to read {event_file}: {exc}")

    series: dict[str, ScalarSeries] = {}
    for tag, values_by_step in by_tag.items():
        if not values_by_step:
            continue
        steps = np.asarray(sorted(values_by_step), dtype=float)
        values = np.asarray([values_by_step[int(step)] for step in steps], dtype=float)
        series[tag] = ScalarSeries(steps=steps, values=values)
    return series


# ---------------------------------------------------------------------------
# Aggregation and plotting
# ---------------------------------------------------------------------------


def smooth_1d(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size <= 2:
        return values
    window = min(int(window), int(values.size))
    if window <= 1:
        return values
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def rolling_std_1d(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size <= 2:
        return np.zeros_like(values)
    half = int(window) // 2
    out = np.zeros_like(values)
    for idx in range(values.size):
        left = max(0, idx - half)
        right = min(values.size, idx + half + 1)
        out[idx] = float(np.std(values[left:right]))
    return out


def smooth_series(series: ScalarSeries) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if USE_LOG_SPACE_SMOOTHING:
        work_values = np.log(np.clip(series.values, EPS, None))
        center = smooth_1d(work_values, SMOOTH_WINDOW_POINTS)
        spread = rolling_std_1d(work_values, SMOOTH_WINDOW_POINTS)
        lower = np.exp(center - BAND_STD_MULTIPLIER * spread)
        upper = np.exp(center + BAND_STD_MULTIPLIER * spread)
        smoothed = np.exp(center)
    else:
        smoothed = smooth_1d(series.values, SMOOTH_WINDOW_POINTS)
        spread = rolling_std_1d(series.values, SMOOTH_WINDOW_POINTS)
        lower = smoothed - BAND_STD_MULTIPLIER * spread
        upper = smoothed + BAND_STD_MULTIPLIER * spread
        if Y_SCALE == "log":
            lower = np.clip(lower, EPS, None)
            upper = np.clip(upper, EPS, None)
    return series.steps, smoothed, lower, upper


def save_metric_csv(
    output_path: Path,
    rows: list[tuple[str, str, str, float, float, float, float]],
) -> None:
    with output_path.open("w") as fh:
        writer = csv.writer(fh)
        writer.writerow(["run", "tag", "label", "epoch", "smoothed", "lower", "upper"])
        for run_label, tag, metric_label, step, smoothed, lower, upper in rows:
            writer.writerow(
                [
                    run_label,
                    tag,
                    metric_label,
                    f"{step:.10g}",
                    f"{smoothed:.10g}",
                    f"{lower:.10g}",
                    f"{upper:.10g}",
                ]
            )


def run_label(run_dir: Path) -> str:
    if run_dir.parent == LOG_ROOT:
        return run_dir.name
    return str(run_dir.relative_to(LOG_ROOT)) if run_dir.is_relative_to(LOG_ROOT) else run_dir.name


def plot(run_dirs: list[Path]) -> None:
    if not run_dirs:
        raise ValueError(f"No TensorBoard runs found under {LOG_ROOT}")

    tags = [metric.tag for metric in METRICS]
    print("Using runs:")
    for run_dir in run_dirs:
        print(f"  - {run_dir}")

    per_run = {run_dir: load_selected_scalars(run_dir, tags) for run_dir in run_dirs}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(METRICS), 1, figsize=FIGSIZE, sharex=SHARE_X_AXIS)
    axes_arr = np.atleast_1d(axes)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    csv_rows: list[tuple[str, str, str, float, float, float, float]] = []

    for ax, metric in zip(axes_arr, METRICS):
        plotted_count = 0
        for run_idx, (run_dir, tag_series) in enumerate(per_run.items()):
            if metric.tag not in tag_series:
                print(f"Warning: {metric.tag!r} missing in {run_dir}.")
                continue

            steps, smoothed, lower, upper = smooth_series(tag_series[metric.tag])
            label = run_label(run_dir)
            color = colors[run_idx % len(colors)]

            ax.plot(steps, smoothed, linewidth=LINE_WIDTH, color=color, label=label)
            if PLOT_ROLLING_STD_BANDS:
                ax.fill_between(steps, lower, upper, color=color, alpha=BAND_ALPHA, linewidth=0)

            plotted_count += 1
            if SAVE_CSV:
                csv_rows.extend(
                    (label, metric.tag, metric.label, float(step), float(value), float(lo), float(hi))
                    for step, value, lo, hi in zip(steps, smoothed, lower, upper)
                )

        if plotted_count == 0:
            print(f"Warning: skipping {metric.tag!r}; no selected runs contain it.")
        ax.set_title(metric.title)
        ax.set_ylabel(Y_LABEL)
        ax.set_yscale(Y_SCALE)
        ax.grid(alpha=GRID_ALPHA, which="both")

    axes_arr[-1].set_xlabel(X_LABEL)
    if PLOT_TITLE:
        fig.suptitle(PLOT_TITLE)
    handles: list = []
    labels: list[str] = []
    for ax in axes_arr:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            break
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(3, len(labels)), frameon=False)
    if SAVE_CSV:
        save_metric_csv(OUTPUT_DIR / f"{OUTPUT_BASENAME}_individual_runs.csv", csv_rows)
    top = 0.92 if PLOT_TITLE or handles else 1.0
    fig.tight_layout(rect=(0.0, 0.0, 1.0, top))

    if SAVE_PNG:
        out_path = OUTPUT_DIR / f"{OUTPUT_BASENAME}.png"
        fig.savefig(out_path, dpi=DPI)
        print(f"Saved {out_path}")
    if SAVE_PDF:
        out_path = OUTPUT_DIR / f"{OUTPUT_BASENAME}.pdf"
        fig.savefig(out_path)
        print(f"Saved {out_path}")
    if SHOW_FIGURE:
        plt.show()
    plt.close(fig)


def main() -> None:
    plot(discover_run_dirs())


if __name__ == "__main__":
    main()
