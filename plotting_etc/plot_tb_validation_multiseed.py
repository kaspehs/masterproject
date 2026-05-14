"""Plot mean ± std across multiple seeds for validation aggregate errors.

Each SEED_GROUP_DIRS entry is a folder whose subdirectories are individual seed
runs. One shaded mean±std line is drawn per group. Pass multiple group dirs to
overlay several lines in the same subplots.

Edit the config block below, then run:

    python plotting_etc/plot_tb_validation_multiseed.py
"""

from __future__ import annotations

import os
import re
import tempfile
import csv
import json
from dataclasses import dataclass, field
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
import matplotlib.ticker as mticker
import numpy as np
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
from tensorboard.util import tensor_util


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Each entry is a folder whose subdirectories are seed runs.
# Multiple entries produce multiple lines in each subplot.
SEED_GROUP_DIRS: list[str] = [
    "logs/mean/multi_seed",
    "logs/fhat/multi_seed",
    "logs/combined/multi_seed",
]

# One label per entry in SEED_GROUP_DIRS.  Used in the plot legend.
# Leave empty to use folder names.
SEED_GROUP_LABELS: list[str] = [
    "Final mean correction model",
    "Final frequency correction model",
    "Final combined correction model",
]

# Optional LaTeX labels for table rows, one per SEED_GROUP_DIRS entry.
# When provided, these are written verbatim (no escaping) — use raw LaTeX here.
# When empty or shorter than SEED_GROUP_DIRS, falls back to latex_escape(label).
SEED_GROUP_LATEX_LABELS: list[str] = [
    r"Final mean correction model",
    r"Final frequency correction model",
    r"Final combined correction model",
]

# Optional legend labels shown in the plot (plain text or matplotlib mathtext).
# When empty or shorter than SEED_GROUP_DIRS, falls back to SEED_GROUP_LABELS.
SEED_GROUP_LEGEND_LABELS: list[str] = [
    "Mean correction",
    "Frequency correction",
    "Combined correction",
]

OUTPUT_DIR = Path("figs/tensorboard_validation")
OUTPUT_BASENAME = "multiseed_validation_split_rows"
DPI = 300

PLOT_TITLE: str | None = None
X_LABEL = "Epoch"
X_LIMITS: tuple[float, float] | None = (0.0, 500.0)
X_LIMIT_MARGIN = 5.0
SHOW_SUBPLOT_TITLES = True
Y_LABEL_FONT_SIZE = 14.4
Y_LABEL_ROTATION = 0
Y_LABEL_PAD = 11
Y_SCALE = "log"  # "linear" or "log"
GRID_ALPHA = 0.25
FIGSIZE = (10.0, 9.0)
SHARE_X_AXIS = True

# Smoothing is applied to the cross-seed mean curve (in the same space used
# for aggregation).  Set to 1 to disable.
SMOOTH_WINDOW_POINTS = 3

PREFER_ASYNC_VALIDATION_JSON = True

# Cross-seed std bands — always meaningful here since std is across seeds.
PLOT_STD_BANDS = True
BAND_STD_MULTIPLIER = 1.0
BAND_ALPHA = 0.30
LINE_WIDTH = 1.8
# Marker at the actual (epoch, value) of the best individual-seed result —
# the checkpoint used as the final model.
PLOT_BEST_INDIVIDUAL_MARKER = True
BEST_INDIVIDUAL_MARKER_SIZE = 34

SELECTION_SPLIT_TAG_PREFIX = "val"
SELECTION_SCALAR_NAME = "Aggregate validation error"

PLOT_BASELINE_LINES = True
BASELINE_LABEL = "Baseline VIVANA-TD"
BASELINE_COLOR = "0.35"
BASELINE_LINE_WIDTH = 1.4
PLOT_BASELINE_Y_TICKS = True
BASELINE_ERRORS_BY_METRIC: dict[str, float] = {
    "combined": 0.534,
}
BASELINE_ERRORS_BY_SPLIT: dict[str, float | None] = {
    "val_seen": 0.534,
    "val_surrogate": 0.889,
    "val": 0.712,
}

# Baseline (VIVANA-TD) values for the four component errors, per split.
# Keys are (split_tag_prefix, scalar_name). Set a value to None to omit that line.
COMPONENT_BASELINE_ERRORS: dict[tuple[str, str], float | None] = {
    ("val_seen",       "Displacement std relative error"):        1.35,
    ("val_seen",       "Dominant frequency relative error"):      0.152,
    ("val_seen",       "Force std relative error"):               0.447,
    ("val_seen",       "Force dominant frequency relative error"): 0.188,
    ("val_surrogate",  "Displacement std relative error"):        0.471,
    ("val_surrogate",  "Dominant frequency relative error"):      0.163,
    ("val_surrogate",  "Force std relative error"):               2.74,
    ("val_surrogate",  "Force dominant frequency relative error"):0.180,
}

# Log-space aggregation across seeds — better for positive error metrics.
USE_LOG_SPACE_SMOOTHING = True
EPS = 1e-12

SAVE_PNG = True
SAVE_PDF = False
SAVE_CSV = True
SAVE_LATEX_TABLES = True
PRINT_LATEX_TABLES = True

LATEX_MEAN_TABLE_CAPTION = (
    "Multi-seed mean performance at the best-epoch checkpoint selected from the "
    "mean aggregate validation curve. "
    "Values are geometric means across seeds. "
    "Lower is better."
)
LATEX_MEAN_TABLE_LABEL = "tab:multiseed_mean_performance"

LATEX_STD_TABLE_CAPTION = (
    "Multi-seed standard deviation of performance at the best-epoch checkpoint. "
    "Values are standard deviations computed in log-space across seeds. "
    "Lower indicates more consistent training."
)
LATEX_STD_TABLE_LABEL = "tab:multiseed_std_performance"

LATEX_BOLD_BEST_PER_METRIC = True
LATEX_TABLE_COLSEP_PT = 3
LATEX_INCLUDE_BASELINE_ROW = True
LATEX_INCLUDE_BEST_EPOCH = False
LATEX_BEST_EPOCH_HEADER = "Epoch"
SHOW_FIGURE = False


@dataclass(frozen=True)
class MetricConfig:
    scalar_name: str
    label: str
    title: str
    y_label: str


@dataclass(frozen=True)
class SplitConfig:
    tag_prefix: str
    label: str
    linestyle: str = "-"


@dataclass(frozen=True)
class TableColumnConfig:
    split_tag_prefix: str
    scalar_name: str
    label: str
    latex_header: str


@dataclass(frozen=True)
class SelectedCheckpoint:
    step: float
    value: float


@dataclass
class SeedGroup:
    label: str           # plain-text identifier (used as internal key)
    latex_label: str     # raw LaTeX label for table rows
    legend_label: str    # label shown in the plot legend
    seed_dirs: list[Path] = field(default_factory=list)


SPLITS: list[SplitConfig] = [
    SplitConfig(tag_prefix="val_seen", label="Seen training trajectories", linestyle="-"),
    SplitConfig(tag_prefix="val_surrogate", label="Surrogate validation points", linestyle="-"),
    SplitConfig(tag_prefix="val", label="Combined validation", linestyle="-"),
]

METRICS: list[MetricConfig] = [
    MetricConfig(
        scalar_name="Aggregate validation error",
        label="combined",
        title="Combined aggregate error",
        y_label=r"$\bar{\varepsilon}$",
    ),
]

# Four component errors plotted in the per-split detail figures.
COMPONENT_METRICS: list[MetricConfig] = [
    MetricConfig(
        scalar_name="Displacement std relative error",
        label="disp_std",
        title="Displacement std error",
        y_label=r"$\varepsilon_{\sigma}^{y}$",
    ),
    MetricConfig(
        scalar_name="Dominant frequency relative error",
        label="disp_freq",
        title="Displacement frequency error",
        y_label=r"$\varepsilon_{\omega}^{y}$",
    ),
    MetricConfig(
        scalar_name="Force std relative error",
        label="force_std",
        title="Force std error",
        y_label=r"$\varepsilon_{\sigma}^{F}$",
    ),
    MetricConfig(
        scalar_name="Force dominant frequency relative error",
        label="force_freq",
        title="Force frequency error",
        y_label=r"$\varepsilon_{\omega}^{F}$",
    ),
]

COMPONENT_FIGSIZE = (10.0, 12.0)

TABLE_COLUMNS: list[TableColumnConfig] = [
    TableColumnConfig(
        split_tag_prefix="val_seen",
        scalar_name="Aggregate displacement error",
        label="seen_disp",
        latex_header=r"Seen $\bar{\varepsilon}_y$",
    ),
    TableColumnConfig(
        split_tag_prefix="val_seen",
        scalar_name="Aggregate force error",
        label="seen_force",
        latex_header=r"Seen $\bar{\varepsilon}_F$",
    ),
    TableColumnConfig(
        split_tag_prefix="val_surrogate",
        scalar_name="Aggregate displacement error",
        label="surrogate_disp",
        latex_header=r"Surr. $\bar{\varepsilon}_y$",
    ),
    TableColumnConfig(
        split_tag_prefix="val_surrogate",
        scalar_name="Aggregate force error",
        label="surrogate_force",
        latex_header=r"Surr. $\bar{\varepsilon}_F$",
    ),
    TableColumnConfig(
        split_tag_prefix="val_surrogate",
        scalar_name="Aggregate validation error",
        label="surrogate_combined",
        latex_header=r"Surr. $\bar{\varepsilon}$",
    ),
    TableColumnConfig(
        split_tag_prefix="val",
        scalar_name="Aggregate validation error",
        label="total_combined",
        latex_header=r"Total $\bar{\varepsilon}$",
    ),
]

BASELINE_TABLE_ERRORS: dict[tuple[str, str], float | None] = {
    ("val_seen", "Aggregate displacement error"): 0.752,
    ("val_seen", "Aggregate force error"): 0.316,
    ("val_surrogate", "Aggregate displacement error"): 0.317,
    ("val_surrogate", "Aggregate force error"): 1.46,
    ("val_surrogate", "Aggregate validation error"): BASELINE_ERRORS_BY_SPLIT.get("val_surrogate"),
    ("val", "Aggregate validation error"): BASELINE_ERRORS_BY_SPLIT.get("val"),
}


# ---------------------------------------------------------------------------
# Scalar loading (unchanged from plot_tb_validation_aggregate.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScalarSeries:
    steps: np.ndarray
    values: np.ndarray


def _event_files(run_dir: Path) -> list[Path]:
    return sorted(run_dir.rglob("events.out.tfevents*"))


def _async_result_files(run_dir: Path) -> list[Path]:
    return sorted((run_dir / "async_validation" / "results").glob("epoch_*.json"))


def _metric_from_async_payload(payload: dict, tag: str) -> float | None:
    split_tag, sep, metric_name = tag.partition("/")
    if not sep:
        return None
    split_results = payload.get("split_results")
    if not isinstance(split_results, dict):
        return None
    split_payload = split_results.get(split_tag)
    if not isinstance(split_payload, dict):
        return None
    metrics = split_payload.get("val_metrics")
    if not isinstance(metrics, dict) or metric_name not in metrics:
        return None
    try:
        value = float(metrics[metric_name])
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _series_from_values_by_step(by_tag: dict[str, dict[int, float]]) -> dict[str, ScalarSeries]:
    series: dict[str, ScalarSeries] = {}
    for tag, values_by_step in by_tag.items():
        if not values_by_step:
            continue
        steps = np.asarray(sorted(values_by_step), dtype=float)
        values = np.asarray([values_by_step[int(step)] for step in steps], dtype=float)
        series[tag] = ScalarSeries(steps=steps, values=values)
    return series


def _summary_value_to_float(value) -> float | None:
    if value.HasField("tensor"):
        arr = tensor_util.make_ndarray(value.tensor)
        if arr.size != 1:
            return None
        return float(arr.reshape(-1)[0])
    return float(value.simple_value)


def load_selected_scalars_from_async_json(run_dir: Path, tags: Iterable[str]) -> dict[str, ScalarSeries]:
    tags_set = set(tags)
    by_tag: dict[str, dict[int, float]] = {tag: {} for tag in tags_set}
    result_files = _async_result_files(run_dir)
    if not result_files:
        return {}

    for result_file in result_files:
        try:
            payload = json.loads(result_file.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"Warning: failed to read {result_file}: {exc}")
            continue
        try:
            step = int(payload.get("epoch", result_file.stem.removeprefix("epoch_")))
        except (TypeError, ValueError):
            print(f"Warning: async validation result {result_file} has no usable epoch.")
            continue
        for tag in tags_set:
            value = _metric_from_async_payload(payload, tag)
            if value is not None:
                by_tag[tag][step] = value

    return _series_from_values_by_step(by_tag)


def load_selected_scalars(run_dir: Path, tags: Iterable[str]) -> dict[str, ScalarSeries]:
    tags_set = set(tags)
    if PREFER_ASYNC_VALIDATION_JSON:
        series = load_selected_scalars_from_async_json(run_dir, tags_set)
        if series:
            missing = sorted(tags_set - set(series))
            if missing:
                print(
                    f"Warning: async validation JSON in {run_dir} is missing "
                    f"{len(missing)} requested tag(s); falling back to TensorBoard event files."
                )
            else:
                return series

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
        except Exception as exc:
            print(f"Warning: failed to read {event_file}: {exc}")

    return _series_from_values_by_step(by_tag)


# ---------------------------------------------------------------------------
# Seed-group discovery
# ---------------------------------------------------------------------------


def discover_seed_dirs(group_dir: Path) -> list[Path]:
    """Return all run dirs (containing event files or async_validation) under group_dir."""
    # Prefer dirs that have async_validation results
    async_dirs = sorted({p.parent.parent.parent for p in group_dir.rglob("async_validation/results/epoch_*.json")})
    if async_dirs:
        return async_dirs
    # Fall back to dirs containing TensorBoard event files
    return sorted({p.parent for p in group_dir.rglob("events.out.tfevents*")})


def load_groups() -> list[SeedGroup]:
    groups: list[SeedGroup] = []
    for idx, group_dir_str in enumerate(SEED_GROUP_DIRS):
        group_dir = Path(group_dir_str)
        label = (
            SEED_GROUP_LABELS[idx]
            if idx < len(SEED_GROUP_LABELS) and SEED_GROUP_LABELS[idx]
            else group_dir.name
        )
        latex_label = (
            SEED_GROUP_LATEX_LABELS[idx]
            if idx < len(SEED_GROUP_LATEX_LABELS) and SEED_GROUP_LATEX_LABELS[idx]
            else latex_escape(label)
        )
        legend_label = (
            SEED_GROUP_LEGEND_LABELS[idx]
            if idx < len(SEED_GROUP_LEGEND_LABELS) and SEED_GROUP_LEGEND_LABELS[idx]
            else label
        )
        seed_dirs = discover_seed_dirs(group_dir)
        if not seed_dirs:
            print(f"Warning: no seed runs found in {group_dir}; skipping group {label!r}.")
            continue
        print(f"Group {label!r}: {len(seed_dirs)} seed(s)")
        for sd in seed_dirs:
            print(f"  - {sd}")
        groups.append(SeedGroup(label=label, latex_label=latex_label, legend_label=legend_label, seed_dirs=seed_dirs))
    return groups


# ---------------------------------------------------------------------------
# Cross-seed aggregation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AggSeries:
    steps: np.ndarray      # common epoch grid
    mean: np.ndarray       # cross-seed (geometric) mean
    lower: np.ndarray      # mean - 1*std in the aggregation space
    upper: np.ndarray      # mean + 1*std in the aggregation space
    std: np.ndarray        # cross-seed std (in linear space for the std table)


def aggregate_across_seeds(
    seed_series_list: list[dict[str, ScalarSeries]],
    tags: list[str],
) -> dict[str, AggSeries]:
    """Aggregate per-seed ScalarSeries across seeds for each tag."""
    result: dict[str, AggSeries] = {}
    for tag in tags:
        per_seed_series = [s[tag] for s in seed_series_list if tag in s]
        if not per_seed_series:
            continue

        # Common epoch grid: intersection of all seeds' step sets
        step_sets = [set(s.steps.tolist()) for s in per_seed_series]
        common_steps = sorted(step_sets[0].intersection(*step_sets[1:]))
        if not common_steps:
            # Fall back to union with interpolation
            all_steps: set[float] = set()
            for s in per_seed_series:
                all_steps.update(s.steps.tolist())
            common_steps = sorted(all_steps)

        grid = np.asarray(common_steps, dtype=float)

        # Interpolate each seed to the common grid
        interpolated = np.stack(
            [np.interp(grid, s.steps, s.values) for s in per_seed_series],
            axis=0,
        )  # shape: (n_seeds, n_epochs)

        if USE_LOG_SPACE_SMOOTHING:
            log_vals = np.log(np.clip(interpolated, EPS, None))
            log_mean = log_vals.mean(axis=0)
            log_std = log_vals.std(axis=0, ddof=0)
            mean_vals = np.exp(log_mean)
            lower = np.exp(log_mean - BAND_STD_MULTIPLIER * log_std)
            upper = np.exp(log_mean + BAND_STD_MULTIPLIER * log_std)
            # Std for table: back-transform to std of original values via delta method
            std_vals = mean_vals * log_std
        else:
            mean_vals = interpolated.mean(axis=0)
            std_dev = interpolated.std(axis=0, ddof=0)
            lower = mean_vals - BAND_STD_MULTIPLIER * std_dev
            upper = mean_vals + BAND_STD_MULTIPLIER * std_dev
            if Y_SCALE == "log":
                lower = np.clip(lower, EPS, None)
                upper = np.clip(upper, EPS, None)
            std_vals = std_dev

        result[tag] = AggSeries(steps=grid, mean=mean_vals, lower=lower, upper=upper, std=std_vals)
    return result


def smooth_agg_series(agg: AggSeries) -> AggSeries:
    """Apply temporal smoothing to an already-aggregated series."""
    if SMOOTH_WINDOW_POINTS <= 1:
        return agg
    if USE_LOG_SPACE_SMOOTHING:
        log_mean = np.log(np.clip(agg.mean, EPS, None))
        log_mean_smooth = smooth_1d(log_mean, SMOOTH_WINDOW_POINTS)
        log_lower = np.log(np.clip(agg.lower, EPS, None))
        log_upper = np.log(np.clip(agg.upper, EPS, None))
        log_lower_smooth = smooth_1d(log_lower, SMOOTH_WINDOW_POINTS)
        log_upper_smooth = smooth_1d(log_upper, SMOOTH_WINDOW_POINTS)
        log_std = np.log(np.clip(agg.std, EPS, None))
        log_std_smooth = smooth_1d(log_std, SMOOTH_WINDOW_POINTS)
        return AggSeries(
            steps=agg.steps,
            mean=np.exp(log_mean_smooth),
            lower=np.exp(log_lower_smooth),
            upper=np.exp(log_upper_smooth),
            std=np.exp(log_std_smooth),
        )
    else:
        mean_smooth = smooth_1d(agg.mean, SMOOTH_WINDOW_POINTS)
        lower_smooth = smooth_1d(agg.lower, SMOOTH_WINDOW_POINTS)
        upper_smooth = smooth_1d(agg.upper, SMOOTH_WINDOW_POINTS)
        std_smooth = smooth_1d(agg.std, SMOOTH_WINDOW_POINTS)
        if Y_SCALE == "log":
            lower_smooth = np.clip(lower_smooth, EPS, None)
            upper_smooth = np.clip(upper_smooth, EPS, None)
        return AggSeries(
            steps=agg.steps,
            mean=mean_smooth,
            lower=lower_smooth,
            upper=upper_smooth,
            std=std_smooth,
        )


# ---------------------------------------------------------------------------
# Smoothing helpers (unchanged)
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


# ---------------------------------------------------------------------------
# Best-epoch selection
# ---------------------------------------------------------------------------


def interpolated_point_at_step(steps: np.ndarray, values: np.ndarray, step: float) -> tuple[float, float] | None:
    if steps.size == 0 or values.size == 0:
        return None
    step = float(step)
    if not np.isfinite(step):
        return None
    if step < float(steps[0]) or step > float(steps[-1]):
        return None
    value = float(np.interp(step, steps, values))
    if not np.isfinite(value):
        return None
    return step, value


def selected_checkpoint_for_group(agg_series: dict[str, AggSeries]) -> SelectedCheckpoint | None:
    tag = f"{SELECTION_SPLIT_TAG_PREFIX}/{SELECTION_SCALAR_NAME}"
    if tag not in agg_series:
        print(f"Warning: selection metric {tag!r} missing; using no selected checkpoint.")
        return None
    agg = smooth_agg_series(agg_series[tag])
    if agg.steps.size == 0:
        return None
    finite = np.isfinite(agg.mean)
    if not np.any(finite):
        return None
    finite_indices = np.flatnonzero(finite)
    best_idx = int(finite_indices[int(np.argmin(agg.mean[finite]))])
    return SelectedCheckpoint(
        step=float(agg.steps[best_idx]),
        value=float(agg.mean[best_idx]),
    )


# ---------------------------------------------------------------------------
# LaTeX / CSV helpers
# ---------------------------------------------------------------------------


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def format_latex_value(value: float, *, bold: bool = False) -> str:
    if not np.isfinite(float(value)):
        return "--"
    if float(value) == 0.0:
        body = "0"
    else:
        exponent = int(np.floor(np.log10(abs(float(value)))))
        mantissa = float(value) / (10.0**exponent)
        mantissa_text = f"{mantissa:.3g}"
        body = rf"{mantissa_text}{{\cdot}}10^{{{exponent}}}"
    return rf"$\mathbf{{{body}}}$" if bold else rf"${body}$"


def format_latex_epoch(value: float | None) -> str:
    if value is None or not np.isfinite(float(value)):
        return "--"
    if np.isclose(float(value), round(float(value)), rtol=0.0, atol=1e-9):
        return str(int(round(float(value))))
    return f"{float(value):.3g}"


def make_latex_performance_table(
    table_values: dict[str, dict[str, float]],
    selected_epochs: dict[str, float] | None,
    caption: str,
    label: str,
) -> str:
    """Build a LaTeX table from {row_label -> {column_label -> value}}."""
    column_headers = {column.label: column.latex_header for column in TABLE_COLUMNS}
    column_order = [column.label for column in TABLE_COLUMNS]
    selected_epochs = selected_epochs or {}

    value_column_spec = " ".join("c" for _ in column_order)
    if LATEX_INCLUDE_BEST_EPOCH:
        column_spec = "c c | " + value_column_spec if column_order else "c c"
    else:
        column_spec = "c | " + value_column_spec if column_order else "c"

    best_by_metric: dict[str, float] = {}
    if LATEX_BOLD_BEST_PER_METRIC:
        for column_label in column_order:
            values = [
                row[column_label]
                for row in table_values.values()
                if column_label in row and np.isfinite(row[column_label])
            ]
            if values:
                best_by_metric[column_label] = min(values)

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        rf"\setlength{{\tabcolsep}}{{{LATEX_TABLE_COLSEP_PT}pt}}",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        (
            "Variant"
            + (f" & {LATEX_BEST_EPOCH_HEADER}" if LATEX_INCLUDE_BEST_EPOCH else "")
            + (
                " & "
                + " & ".join(column_headers.get(col, latex_escape(col)) for col in column_order)
                if column_order
                else ""
            )
            + r" \\"
        ),
        r"\midrule",
    ]

    for row_label, row_metrics in table_values.items():
        row = [row_label]  # already raw LaTeX (caller is responsible for escaping if needed)
        if LATEX_INCLUDE_BEST_EPOCH:
            row.append(format_latex_epoch(selected_epochs.get(row_label)))
        for col in column_order:
            if col not in row_metrics:
                row.append("--")
                continue
            value = row_metrics[col]
            best_value = best_by_metric.get(col)
            bold = best_value is not None and np.isclose(value, best_value, rtol=1e-9, atol=1e-12)
            row.append(format_latex_value(value, bold=bold))
        lines.append(" & ".join(row) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{{latex_escape(caption)}}}",
            rf"\label{{{label}}}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def baseline_error_for_table_column(column: TableColumnConfig) -> float | None:
    value = BASELINE_TABLE_ERRORS.get((column.split_tag_prefix, column.scalar_name))
    if value is not None:
        return float(value)
    if column.scalar_name == "Aggregate validation error":
        split_value = BASELINE_ERRORS_BY_SPLIT.get(column.split_tag_prefix)
        if split_value is not None:
            return float(split_value)
        metric_value = BASELINE_ERRORS_BY_METRIC.get("combined")
        return float(metric_value) if metric_value is not None else None
    return None


def baseline_value_for(split: SplitConfig, metric: MetricConfig) -> float | None:
    split_value = BASELINE_ERRORS_BY_SPLIT.get(split.tag_prefix)
    if split_value is not None:
        return float(split_value)
    metric_value = BASELINE_ERRORS_BY_METRIC.get(metric.label)
    return float(metric_value) if metric_value is not None else None


def save_metric_csv(
    output_path: Path,
    rows: list[tuple],
) -> None:
    with output_path.open("w") as fh:
        writer = csv.writer(fh)
        writer.writerow(["group", "split", "tag", "metric", "epoch", "mean", "lower", "upper", "std"])
        for row in rows:
            writer.writerow([f"{v:.10g}" if isinstance(v, float) else v for v in row])


def add_baseline_y_tick(ax, baseline_value: float) -> None:
    baseline_value = float(baseline_value)
    if not np.isfinite(baseline_value) or baseline_value <= 0.0:
        return
    ymin, ymax = ax.get_ylim()
    low, high = sorted((float(ymin), float(ymax)))
    ticks = [
        float(tick)
        for tick in ax.get_yticks()
        if np.isfinite(float(tick)) and low <= float(tick) <= high
    ]
    if not any(np.isclose(tick, baseline_value, rtol=1e-6, atol=1e-12) for tick in ticks):
        ticks.append(baseline_value)
    ax.set_yticks(sorted(ticks))
    log_formatter = mticker.LogFormatterSciNotation(base=10)

    def formatter(value: float, pos: int | None = None) -> str:
        if np.isclose(float(value), baseline_value, rtol=1e-6, atol=1e-12):
            exponent = int(np.floor(np.log10(baseline_value)))
            mantissa = baseline_value / (10.0**exponent)
            return rf"${mantissa:.3g}\times 10^{{{exponent}}}$"
        return log_formatter(value, pos)

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(formatter))
    ax.set_ylim(ymin, ymax)


# ---------------------------------------------------------------------------
# Main plot routine
# ---------------------------------------------------------------------------


def plot(groups: list[SeedGroup]) -> None:
    if not groups:
        raise ValueError("No seed groups to plot.")
    if len(METRICS) != 1:
        raise ValueError("This plotter expects exactly one aggregate metric.")

    metric = METRICS[0]
    plot_tags = [f"{split.tag_prefix}/{metric.scalar_name}" for split in SPLITS]
    table_tags = [f"{col.split_tag_prefix}/{col.scalar_name}" for col in TABLE_COLUMNS]
    all_tags = sorted({*plot_tags, *table_tags})

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(SPLITS), 1, figsize=FIGSIZE, sharex=SHARE_X_AXIS)
    axes_arr = np.atleast_1d(axes)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    csv_rows: list[tuple] = []
    mean_table_values: dict[str, dict[str, float]] = {}
    std_table_values: dict[str, dict[str, float]] = {}
    selected_epochs: dict[str, float] = {}
    selected_checkpoints: dict[str, SelectedCheckpoint] = {}
    # group.label -> tag -> (epoch, value) of the best individual seed at that split
    individual_best_points: dict[str, dict[str, tuple[float, float]]] = {}

    # Optional baseline row in mean table
    if LATEX_INCLUDE_BASELINE_ROW:
        baseline_row: dict[str, float] = {}
        for col in TABLE_COLUMNS:
            v = baseline_error_for_table_column(col)
            if v is not None and np.isfinite(v):
                baseline_row[col.label] = v
        if baseline_row:
            mean_table_values[BASELINE_LABEL] = baseline_row
            std_table_values[BASELINE_LABEL] = {}

    for group_idx, group in enumerate(groups):
        # Load all seeds
        per_seed: list[dict[str, ScalarSeries]] = []
        for seed_dir in group.seed_dirs:
            per_seed.append(load_selected_scalars(seed_dir, all_tags))

        agg_series = aggregate_across_seeds(per_seed, all_tags)

        # Find the winning seed+epoch: lowest val/Aggregate validation error across all seeds.
        # Then record each plot tag's value for that seed at that epoch.
        if PLOT_BEST_INDIVIDUAL_MARKER:
            sel_tag = f"{SELECTION_SPLIT_TAG_PREFIX}/{SELECTION_SCALAR_NAME}"
            best_seed_idx: int | None = None
            best_indiv_epoch: float | None = None
            best_indiv_value = float("inf")
            for s_idx, seed_series in enumerate(per_seed):
                if sel_tag not in seed_series:
                    continue
                s = seed_series[sel_tag]
                min_idx = int(np.argmin(s.values))
                if float(s.values[min_idx]) < best_indiv_value:
                    best_indiv_value = float(s.values[min_idx])
                    best_indiv_epoch = float(s.steps[min_idx])
                    best_seed_idx = s_idx
            if best_seed_idx is not None and best_indiv_epoch is not None:
                print(
                    f"Group {group.label!r}: best individual seed={best_seed_idx}, "
                    f"epoch={best_indiv_epoch:g}, {sel_tag}={best_indiv_value:.6g}"
                )
                pts: dict[str, tuple[float, float]] = {}
                for tag in plot_tags:
                    seed_series = per_seed[best_seed_idx]
                    if tag not in seed_series:
                        continue
                    pt = interpolated_point_at_step(
                        seed_series[tag].steps, seed_series[tag].values, best_indiv_epoch
                    )
                    if pt is not None:
                        pts[tag] = pt
                individual_best_points[group.label] = pts

        # Determine best checkpoint from smoothed mean selection metric
        checkpoint = selected_checkpoint_for_group(agg_series)
        if checkpoint is None:
            print(f"Warning: could not determine best checkpoint for group {group.label!r}.")
            continue
        selected_checkpoints[group.label] = checkpoint
        # Use latex_label as key so table rows and epoch lookup are consistent
        selected_epochs[group.latex_label] = checkpoint.step
        print(
            f"Group {group.label!r}: best epoch={checkpoint.step:g}, "
            f"{SELECTION_SPLIT_TAG_PREFIX}/{SELECTION_SCALAR_NAME}={checkpoint.value:.6g}"
        )

        # Build table rows from mean and std at the selected epoch
        mean_row: dict[str, float] = {}
        std_row: dict[str, float] = {}
        for col in TABLE_COLUMNS:
            tag = f"{col.split_tag_prefix}/{col.scalar_name}"
            if tag not in agg_series:
                print(f"Warning: table metric {tag!r} missing for group {group.label!r}.")
                continue
            agg = smooth_agg_series(agg_series[tag])
            point_mean = interpolated_point_at_step(agg.steps, agg.mean, checkpoint.step)
            point_std = interpolated_point_at_step(agg.steps, agg.std, checkpoint.step)
            if point_mean is not None:
                mean_row[col.label] = point_mean[1]
            if point_std is not None:
                std_row[col.label] = point_std[1]
        mean_table_values[group.latex_label] = mean_row
        std_table_values[group.latex_label] = std_row

    # Draw plots
    for ax, split in zip(axes_arr, SPLITS):
        baseline_value = baseline_value_for(split, metric)
        if PLOT_BASELINE_LINES and baseline_value is not None and np.isfinite(float(baseline_value)):
            ax.axhline(
                float(baseline_value),
                color=BASELINE_COLOR,
                linestyle="--",
                linewidth=BASELINE_LINE_WIDTH,
                label=BASELINE_LABEL,
            )

        for group_idx, group in enumerate(groups):
            color = colors[group_idx % len(colors)]
            tag = f"{split.tag_prefix}/{metric.scalar_name}"

            per_seed = [load_selected_scalars(sd, [tag]) for sd in group.seed_dirs]
            agg_series = aggregate_across_seeds(per_seed, [tag])
            if tag not in agg_series:
                print(f"Warning: {tag!r} missing for group {group.label!r}.")
                continue

            agg = smooth_agg_series(agg_series[tag])
            ax.plot(
                agg.steps,
                agg.mean,
                linewidth=LINE_WIDTH,
                color=color,
                linestyle=split.linestyle,
                label=group.legend_label,
                zorder=2,
            )
            if PLOT_STD_BANDS:
                ax.fill_between(agg.steps, agg.lower, agg.upper, color=color, alpha=BAND_ALPHA, linewidth=0)

            if PLOT_BEST_INDIVIDUAL_MARKER:
                pt = individual_best_points.get(group.label, {}).get(tag)
                if pt is not None:
                    ax.scatter(pt[0], pt[1], color=color, s=BEST_INDIVIDUAL_MARKER_SIZE, zorder=4)

            if SAVE_CSV:
                for step, mean_v, lo, hi, std_v in zip(
                    agg.steps, agg.mean, agg.lower, agg.upper, agg.std
                ):
                    csv_rows.append((
                        group.label, split.label, tag, metric.label,
                        float(step), float(mean_v), float(lo), float(hi), float(std_v),
                    ))

        if SHOW_SUBPLOT_TITLES:
            ax.set_title(split.label)
        ax.set_ylabel(
            metric.y_label,
            fontsize=Y_LABEL_FONT_SIZE,
            rotation=Y_LABEL_ROTATION,
            labelpad=Y_LABEL_PAD,
            va="center",
        )
        ax.set_yscale(Y_SCALE)
        if PLOT_BASELINE_Y_TICKS and baseline_value is not None:
            add_baseline_y_tick(ax, float(baseline_value))
        ax.grid(alpha=GRID_ALPHA, which="both")
        if X_LIMITS is not None:
            ax.set_xlim(X_LIMITS[0] - X_LIMIT_MARGIN, X_LIMITS[1] + X_LIMIT_MARGIN)

    axes_arr[-1].set_xlabel(X_LABEL)
    if PLOT_TITLE:
        fig.suptitle(PLOT_TITLE)
    handles, labels = [], []
    for ax in axes_arr:
        h, l = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, l
            break
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), frameon=False)

    if SAVE_CSV:
        save_metric_csv(OUTPUT_DIR / f"{OUTPUT_BASENAME}_data.csv", csv_rows)

    if SAVE_LATEX_TABLES:
        mean_tex = make_latex_performance_table(
            mean_table_values, selected_epochs, LATEX_MEAN_TABLE_CAPTION, LATEX_MEAN_TABLE_LABEL
        )
        std_tex = make_latex_performance_table(
            std_table_values, selected_epochs, LATEX_STD_TABLE_CAPTION, LATEX_STD_TABLE_LABEL
        )
        (OUTPUT_DIR / f"{OUTPUT_BASENAME}_mean_table.tex").write_text(mean_tex + "\n")
        (OUTPUT_DIR / f"{OUTPUT_BASENAME}_std_table.tex").write_text(std_tex + "\n")
        if PRINT_LATEX_TABLES:
            print("\n--- Mean table ---\n")
            print(mean_tex)
            print("\n--- Std table ---\n")
            print(std_tex)

    top = 0.92 if PLOT_TITLE or handles else 1.0
    fig.tight_layout(rect=(0.0, 0.0, 1.0, top))

    if SAVE_PNG:
        out_path = OUTPUT_DIR / f"{OUTPUT_BASENAME}.png"
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0.03)
        print(f"Saved {out_path}")
    if SAVE_PDF:
        out_path = OUTPUT_DIR / f"{OUTPUT_BASENAME}.pdf"
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.03)
        print(f"Saved {out_path}")
    if SHOW_FIGURE:
        plt.show()
    plt.close(fig)


def plot_component_errors(groups: list[SeedGroup], split: SplitConfig) -> None:
    """Plot the four component error curves for a single split, one subplot each."""
    component_tags = [f"{split.tag_prefix}/{m.scalar_name}" for m in COMPONENT_METRICS]
    sel_tag = f"{SELECTION_SPLIT_TAG_PREFIX}/{SELECTION_SCALAR_NAME}"
    all_tags = sorted({*component_tags, sel_tag})

    # Pre-load all seeds per group and find the best individual seed point per tag
    per_group_seeds: dict[str, list[dict[str, ScalarSeries]]] = {}
    indiv_best: dict[str, dict[str, tuple[float, float]]] = {}
    for group in groups:
        per_seed = [load_selected_scalars(sd, all_tags) for sd in group.seed_dirs]
        per_group_seeds[group.label] = per_seed

        if PLOT_BEST_INDIVIDUAL_MARKER:
            best_seed_idx: int | None = None
            best_epoch: float | None = None
            best_value = float("inf")
            for s_idx, seed_series in enumerate(per_seed):
                if sel_tag not in seed_series:
                    continue
                s = seed_series[sel_tag]
                min_idx = int(np.argmin(s.values))
                if float(s.values[min_idx]) < best_value:
                    best_value = float(s.values[min_idx])
                    best_epoch = float(s.steps[min_idx])
                    best_seed_idx = s_idx
            if best_seed_idx is not None and best_epoch is not None:
                pts: dict[str, tuple[float, float]] = {}
                for tag in component_tags:
                    winner = per_seed[best_seed_idx]
                    if tag not in winner:
                        continue
                    pt = interpolated_point_at_step(winner[tag].steps, winner[tag].values, best_epoch)
                    if pt is not None:
                        pts[tag] = pt
                indiv_best[group.label] = pts

    fig, axes = plt.subplots(len(COMPONENT_METRICS), 1, figsize=COMPONENT_FIGSIZE, sharex=SHARE_X_AXIS)
    axes_arr = np.atleast_1d(axes)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ax, metric in zip(axes_arr, COMPONENT_METRICS):
        tag = f"{split.tag_prefix}/{metric.scalar_name}"

        baseline_value = COMPONENT_BASELINE_ERRORS.get((split.tag_prefix, metric.scalar_name))
        if PLOT_BASELINE_LINES and baseline_value is not None and np.isfinite(float(baseline_value)):
            ax.axhline(
                float(baseline_value),
                color=BASELINE_COLOR,
                linestyle="--",
                linewidth=BASELINE_LINE_WIDTH,
                label=BASELINE_LABEL,
            )

        for group_idx, group in enumerate(groups):
            color = colors[group_idx % len(colors)]
            per_seed = per_group_seeds[group.label]
            agg_series = aggregate_across_seeds(per_seed, [tag])
            if tag not in agg_series:
                print(f"Warning: {tag!r} missing for group {group.label!r}.")
                continue

            agg = smooth_agg_series(agg_series[tag])
            ax.plot(
                agg.steps, agg.mean,
                linewidth=LINE_WIDTH, color=color,
                linestyle=split.linestyle, label=group.legend_label, zorder=2,
            )
            if PLOT_STD_BANDS:
                ax.fill_between(agg.steps, agg.lower, agg.upper, color=color, alpha=BAND_ALPHA, linewidth=0)

            if PLOT_BEST_INDIVIDUAL_MARKER:
                pt = indiv_best.get(group.label, {}).get(tag)
                if pt is not None:
                    ax.scatter(pt[0], pt[1], color=color, s=BEST_INDIVIDUAL_MARKER_SIZE, zorder=4)

        if SHOW_SUBPLOT_TITLES:
            ax.set_title(metric.title)
        ax.set_ylabel(
            metric.y_label,
            fontsize=Y_LABEL_FONT_SIZE,
            rotation=Y_LABEL_ROTATION,
            labelpad=Y_LABEL_PAD,
            va="center",
        )
        ax.set_yscale(Y_SCALE)
        if PLOT_BASELINE_Y_TICKS and baseline_value is not None and np.isfinite(float(baseline_value)):
            add_baseline_y_tick(ax, float(baseline_value))
        ax.grid(alpha=GRID_ALPHA, which="both")
        if X_LIMITS is not None:
            ax.set_xlim(X_LIMITS[0] - X_LIMIT_MARGIN, X_LIMITS[1] + X_LIMIT_MARGIN)

    axes_arr[-1].set_xlabel(X_LABEL)
    if PLOT_TITLE:
        fig.suptitle(PLOT_TITLE)
    handles, labels = [], []
    for ax in axes_arr:
        h, l = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, l
            break
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), frameon=False)

    top = 0.92 if PLOT_TITLE or handles else 1.0
    fig.tight_layout(rect=(0.0, 0.0, 1.0, top))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    basename = f"{OUTPUT_BASENAME}_{split.tag_prefix}_components"
    if SAVE_PNG:
        out_path = OUTPUT_DIR / f"{basename}.png"
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0.03)
        print(f"Saved {out_path}")
    if SAVE_PDF:
        out_path = OUTPUT_DIR / f"{basename}.pdf"
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.03)
        print(f"Saved {out_path}")
    if SHOW_FIGURE:
        plt.show()
    plt.close(fig)


def main() -> None:
    groups = load_groups()
    plot(groups)
    split_by_tag = {s.tag_prefix: s for s in SPLITS}
    plot_component_errors(groups, split_by_tag["val_surrogate"])
    plot_component_errors(groups, split_by_tag["val_seen"])


if __name__ == "__main__":
    main()
