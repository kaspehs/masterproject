"""Plot smoothed displacement, force, and combined aggregate validation errors.

Edit the config block below, then run:

    python plotting_etc/plot_tb_validation_aggregate.py
"""

from __future__ import annotations

import os
import re
import tempfile
import csv
import json
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
import matplotlib.ticker as mticker
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

# Optional legend labels. RUN_LABELS is applied in the selected run order after
# discovery/sorting. When RUN_LABELS_BY_RUN is non-empty, it also acts as an
# allowlist: only matching runs are included. Keys can be run paths or default
# run labels.
RUN_LABELS: list[str] = []
RUN_LABELS_BY_RUN: dict[str, str] = {
    "combined/loss_ablation/sharedABLATION_onestep": r"Residual loss",
    "combined/loss_ablation/sharedABLATION_mse": r"+ MSE loss",
    "combined/loss_ablation/sharedABLATION_std": r"+ Std loss",
    "combined/loss_ablation/sharedABLATION_psd": r"+ PSD loss",
    "combined/loss_ablation/sharedABLATION_freq": r"+ Frequency loss",
    "combined/loss_ablation/sharedABLATION_std_freq": r"+ Std and Frequency losses",
    "combined/loss_ablation/sharedABLATION_std_psd": r"+ Std and PSD losses",
}

# Number of discovered runs to include. Set to None to use all discovered runs.
NUM_RUNS: int | None = None

# Used only when RUN_DIRS is empty.
RUN_NAME_CONTAINS: str | None = "combined/loss_ablation"
SORT_RUNS_BY = "mtime_desc"  # "mtime_desc", "mtime_asc", or "name"

OUTPUT_DIR = Path("figs/tensorboard_validation")
OUTPUT_BASENAME = "shared_combo_loss_ablation_validation_split_rows"
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

# Smoothing is applied across logged validation points, not raw epoch integers.
# Set to 1 to disable smoothing.
SMOOTH_WINDOW_POINTS = 3

# Async validation writes one JSON summary per validation epoch. Prefer those
# over mirrored TensorBoard event shards when available; they are the canonical
# epoch/value sequence and avoid stale/duplicated event-file artifacts.
PREFER_ASYNC_VALIDATION_JSON = True

# Draw per-run rolling std bands around each smoothed line. Since each run is
# plotted separately, this is temporal variation within a run, not cross-run std.
PLOT_ROLLING_STD_BANDS = False
BAND_STD_MULTIPLIER = 1.0
BAND_ALPHA = 0.10
LINE_WIDTH = 1.8
HIGHLIGHT_RUN_LABELS: set[str] = set()
HIGHLIGHT_RUN_KEYS: set[str] = set()
HIGHLIGHT_LINE_WIDTH = LINE_WIDTH
HIGHLIGHT_COLOR = "black"
HIGHLIGHT_MARKER_SIZE = 34
PLOT_FINAL_MARKERS = True
PLOT_FINAL_VALUE_LABELS = False
SELECTION_SPLIT_TAG_PREFIX = "val"
SELECTION_SCALAR_NAME = "Aggregate validation error"
# Optional manual checkpoint overrides. Leave empty to select the minimum
# combined-validation aggregate error for each run.
SELECTED_EPOCHS_BY_RUN: dict[str, float] = {}
OMIT_INITIAL_BEST_RUNS_FROM_PLOT = True
INITIAL_BEST_STEP_ATOL = 1e-9
FINAL_MARKER_SIZE = 34
FINAL_LABEL_FONT_SIZE = 8
FINAL_LABEL_X_OFFSET_POINTS = 5
FINAL_LABEL_RIGHT_MARGIN_FRACTION = 0.08

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

# Log-space aggregation is usually better for positive error metrics spanning
# orders of magnitude. Set False for arithmetic smoothing/std in linear space.
USE_LOG_SPACE_SMOOTHING = True
EPS = 1e-12

SAVE_PNG = True
SAVE_PDF = False
SAVE_CSV = True
SAVE_LATEX_TABLE = True
PRINT_LATEX_TABLE = True
LATEX_TABLE_CAPTION = (
    "Shared combined-correction loss ablation performance at the selected best-performing checkpoints. "
    "Displacement and force aggregates are means of their frequency and standard-deviation errors; "
    "the total aggregate is the mean of all four component errors. "
    "Lower values indicate better performance."
)
LATEX_TABLE_LABEL = "tab:shared_combo_loss_ablation_performance"
LATEX_BOLD_BEST_PER_METRIC = True
LATEX_TABLE_COLSEP_PT = 3
LATEX_INCLUDE_BASELINE_ROW = True
LATEX_INCLUDE_BEST_EPOCH = True
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
    is_initial: bool


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

# Fill component baseline entries here if/when those baseline evaluations are available.
# The aggregate baseline values below come from BASELINE_ERRORS_BY_SPLIT.
BASELINE_TABLE_ERRORS: dict[tuple[str, str], float | None] = {
    ("val_seen", "Aggregate displacement error"): 0.752,
    ("val_seen", "Aggregate force error"): 0.316,
    ("val_surrogate", "Aggregate displacement error"): 0.317,
    ("val_surrogate", "Aggregate force error"): 1.46,
    ("val_surrogate", "Aggregate validation error"): BASELINE_ERRORS_BY_SPLIT.get("val_surrogate"),
    ("val", "Aggregate validation error"): BASELINE_ERRORS_BY_SPLIT.get("val"),
}


# ---------------------------------------------------------------------------
# Scalar loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScalarSeries:
    steps: np.ndarray
    values: np.ndarray


def _event_files(run_dir: Path) -> list[Path]:
    return sorted(run_dir.rglob("events.out.tfevents*"))


def _async_result_files(run_dir: Path) -> list[Path]:
    return sorted((run_dir / "async_validation" / "results").glob("epoch_*.json"))


def _run_mtime(run_dir: Path) -> float:
    event_files = _event_files(run_dir)
    if event_files:
        return max(path.stat().st_mtime for path in event_files)
    return run_dir.stat().st_mtime


def _run_key_candidates(run_dir: Path, default_label: str | None = None) -> tuple[str, ...]:
    default_label = default_label or default_run_label(run_dir)
    candidates = [str(run_dir), default_label, run_dir.name]
    if run_dir.is_relative_to(LOG_ROOT):
        rel_parts = run_dir.relative_to(LOG_ROOT).parts
        for start in range(len(rel_parts)):
            candidates.append(str(Path(*rel_parts[start:])))
    return tuple(dict.fromkeys(candidates))


def _key_matches_run(candidate: str, key: str) -> bool:
    if candidate == key:
        return True
    if candidate.endswith(f"/{key}"):
        return True
    # Timestamped training runs use the config stem followed by "_MMDD-HHMMSS".
    timestamped_key = rf"{re.escape(key)}_\d{{4}}-\d{{6}}"
    return (
        re.fullmatch(timestamped_key, candidate) is not None
        or re.search(rf"/{timestamped_key}(?:/|$)", candidate) is not None
    )


def _matching_run_label_key(run_dir: Path, default_label: str | None = None) -> str | None:
    for key in RUN_LABELS_BY_RUN:
        aliases = tuple(dict.fromkeys((key, Path(key).name)))
        if any(
            _key_matches_run(candidate, alias)
            for alias in aliases
            for candidate in _run_key_candidates(run_dir, default_label)
        ):
            return key
    return None


def _run_label_order_key(run_dir: Path) -> tuple[int, int, str]:
    label_order = {key: idx for idx, key in enumerate(RUN_LABELS_BY_RUN)}
    default_label = default_run_label(run_dir)
    matched_key = _matching_run_label_key(run_dir, default_label)
    if matched_key in label_order:
        return (0, label_order[matched_key], default_label)
    return (1, len(label_order), default_label)


def _run_matches_label_config(run_dir: Path) -> bool:
    if not RUN_LABELS_BY_RUN:
        return True
    return _matching_run_label_key(run_dir) is not None


def discover_run_dirs() -> list[Path]:
    if RUN_DIRS:
        run_dirs = [Path(path).expanduser() for path in RUN_DIRS]
        return sorted(run_dirs, key=_run_label_order_key) if RUN_LABELS_BY_RUN else run_dirs

    run_dirs = sorted({path.parent for path in LOG_ROOT.rglob("events.out.tfevents*")})
    if RUN_NAME_CONTAINS:
        run_dirs = [path for path in run_dirs if RUN_NAME_CONTAINS in str(path)]
    if RUN_LABELS_BY_RUN:
        run_dirs = [path for path in run_dirs if _run_matches_label_config(path)]

    if SORT_RUNS_BY == "mtime_desc":
        run_dirs.sort(key=_run_mtime, reverse=True)
    elif SORT_RUNS_BY == "mtime_asc":
        run_dirs.sort(key=_run_mtime)
    elif SORT_RUNS_BY == "name":
        run_dirs.sort(key=lambda path: str(path))
    else:
        raise ValueError(f"Unsupported SORT_RUNS_BY={SORT_RUNS_BY!r}")

    if RUN_LABELS_BY_RUN:
        run_dirs.sort(key=_run_label_order_key)

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


def load_selected_scalars_from_async_json(run_dir: Path, tags: Iterable[str]) -> dict[str, ScalarSeries]:
    tags_set = set(tags)
    by_tag: dict[str, dict[int, float]] = {tag: {} for tag in tags_set}
    result_files = _async_result_files(run_dir)
    if not result_files:
        return {}

    for result_file in result_files:
        try:
            payload = json.loads(result_file.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - protects against partial files.
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


def _series_from_values_by_step(by_tag: dict[str, dict[int, float]]) -> dict[str, ScalarSeries]:
    series: dict[str, ScalarSeries] = {}
    for tag, values_by_step in by_tag.items():
        if not values_by_step:
            continue
        steps = np.asarray(sorted(values_by_step), dtype=float)
        values = np.asarray([values_by_step[int(step)] for step in steps], dtype=float)
        series[tag] = ScalarSeries(steps=steps, values=values)
    return series


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
        except Exception as exc:  # pragma: no cover - protects against partial event files.
            print(f"Warning: failed to read {event_file}: {exc}")

    return _series_from_values_by_step(by_tag)


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
    rows: list[tuple[str, str, str, str, float, float, float, float]],
) -> None:
    with output_path.open("w") as fh:
        writer = csv.writer(fh)
        writer.writerow(["run", "split", "tag", "metric", "epoch", "smoothed", "lower", "upper"])
        for run_label, split_label, tag, metric_label, step, smoothed, lower, upper in rows:
            writer.writerow(
                [
                    run_label,
                    split_label,
                    tag,
                    metric_label,
                    f"{step:.10g}",
                    f"{smoothed:.10g}",
                    f"{lower:.10g}",
                    f"{upper:.10g}",
                ]
            )


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
    table_values: dict[str, dict[str, tuple[float, float]]],
    selected_epochs: dict[str, float] | None = None,
) -> str:
    column_headers = {column.label: column.latex_header for column in TABLE_COLUMNS}
    column_order = [column.label for column in TABLE_COLUMNS]
    selected_epochs = selected_epochs or {}
    value_column_spec = " ".join("c" for _ in column_order)
    if LATEX_INCLUDE_BEST_EPOCH:
        column_spec = "c c" if not column_order else "c c | " + value_column_spec
    else:
        column_spec = "c" if not column_order else "c | " + value_column_spec
    best_by_metric: dict[str, float] = {}
    if LATEX_BOLD_BEST_PER_METRIC:
        for column_label in column_order:
            values = [
                metric_values[column_label][1]
                for metric_values in table_values.values()
                if column_label in metric_values and np.isfinite(metric_values[column_label][1])
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
                + " & ".join(column_headers.get(column_label, latex_escape(column_label)) for column_label in column_order)
                if column_order
                else ""
            )
            + r" \\"
        ),
        r"\midrule",
    ]

    for run_label, metric_values in table_values.items():
        row = [latex_escape(run_label)]
        if LATEX_INCLUDE_BEST_EPOCH:
            row.append(format_latex_epoch(selected_epochs.get(run_label)))
        for column_label in column_order:
            if column_label not in metric_values:
                row.append("--")
                continue
            value = metric_values[column_label][1]
            best_value = best_by_metric.get(column_label)
            bold = best_value is not None and np.isclose(value, best_value, rtol=1e-9, atol=1e-12)
            row.append(format_latex_value(value, bold=bold))
        lines.append(" & ".join(row) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{{latex_escape(LATEX_TABLE_CAPTION)}}}",
            rf"\label{{{LATEX_TABLE_LABEL}}}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def save_latex_table(output_path: Path, latex_table: str) -> None:
    output_path.write_text(latex_table + "\n")


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


def baseline_table_row() -> dict[str, tuple[float, float]]:
    values: dict[str, tuple[float, float]] = {}
    for column in TABLE_COLUMNS:
        baseline_value = baseline_error_for_table_column(column)
        if baseline_value is None or not np.isfinite(float(baseline_value)):
            continue
        values[column.label] = (float("nan"), float(baseline_value))
    return values


def default_run_label(run_dir: Path) -> str:
    if run_dir.parent == LOG_ROOT:
        return run_dir.name
    return str(run_dir.relative_to(LOG_ROOT)) if run_dir.is_relative_to(LOG_ROOT) else run_dir.name


def run_label(run_dir: Path, run_idx: int | None = None) -> str:
    default_label = default_run_label(run_dir)
    if run_idx is not None and run_idx < len(RUN_LABELS) and RUN_LABELS[run_idx]:
        return RUN_LABELS[run_idx]

    matched_key = _matching_run_label_key(run_dir, default_label)
    if matched_key is not None:
        return RUN_LABELS_BY_RUN[matched_key]
    return default_label


def is_highlighted_run(run_dir: Path, label: str) -> bool:
    default_label = default_run_label(run_dir)
    return (
        label in HIGHLIGHT_RUN_LABELS
        or default_label in HIGHLIGHT_RUN_KEYS
        or str(run_dir) in HIGHLIGHT_RUN_KEYS
    )


def selected_epoch_override(run_dir: Path, label: str) -> float | None:
    default_label = default_run_label(run_dir)
    for key in (str(run_dir), default_label, label):
        if key in SELECTED_EPOCHS_BY_RUN:
            return float(SELECTED_EPOCHS_BY_RUN[key])
    return None


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


def selected_checkpoint_for_run(
    run_dir: Path,
    label: str,
    tag_series: dict[str, ScalarSeries],
) -> SelectedCheckpoint | None:
    tag = f"{SELECTION_SPLIT_TAG_PREFIX}/{SELECTION_SCALAR_NAME}"
    if tag not in tag_series:
        print(f"Warning: selection metric {tag!r} missing in {run_dir}; using no selected checkpoint.")
        return None

    steps, smoothed, _lower, _upper = smooth_series(tag_series[tag])
    if steps.size == 0:
        return None
    first_step = float(steps[0])
    override_step = selected_epoch_override(run_dir, label)
    if override_step is not None:
        point = interpolated_point_at_step(steps, smoothed, override_step)
        if point is not None:
            step, value = point
            return SelectedCheckpoint(
                step=step,
                value=value,
                is_initial=np.isclose(step, first_step, rtol=0.0, atol=INITIAL_BEST_STEP_ATOL),
            )
        print(f"Warning: selected epoch override {override_step:g} outside logged range for {run_dir}; using minimum.")

    finite = np.isfinite(smoothed)
    if not np.any(finite):
        return None
    finite_indices = np.flatnonzero(finite)
    best_idx = int(finite_indices[int(np.argmin(smoothed[finite]))])
    best_step = float(steps[best_idx])
    best_value = float(smoothed[best_idx])
    if not np.isfinite(best_step) or not np.isfinite(best_value):
        return None
    return SelectedCheckpoint(
        step=best_step,
        value=best_value,
        is_initial=np.isclose(best_step, first_step, rtol=0.0, atol=INITIAL_BEST_STEP_ATOL),
    )


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


def baseline_value_for(split: SplitConfig, metric: MetricConfig) -> float | None:
    split_value = BASELINE_ERRORS_BY_SPLIT.get(split.tag_prefix)
    if split_value is not None:
        return float(split_value)
    metric_value = BASELINE_ERRORS_BY_METRIC.get(metric.label)
    return float(metric_value) if metric_value is not None else None


def plot(run_dirs: list[Path]) -> None:
    if not run_dirs:
        raise ValueError(f"No TensorBoard runs found under {LOG_ROOT}")
    if len(METRICS) != 1:
        raise ValueError("This plotter expects exactly one aggregate metric when plotting splits as rows.")

    metric = METRICS[0]
    plot_tags = [f"{split.tag_prefix}/{metric.scalar_name}" for split in SPLITS]
    table_tags = [f"{column.split_tag_prefix}/{column.scalar_name}" for column in TABLE_COLUMNS]
    tags = sorted({*plot_tags, *table_tags})
    print("Using runs:")
    for run_idx, run_dir in enumerate(run_dirs):
        print(f"  - {run_dir} as {run_label(run_dir, run_idx)!r}")

    per_run = {run_dir: load_selected_scalars(run_dir, tags) for run_dir in run_dirs}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(SPLITS), 1, figsize=FIGSIZE, sharex=SHARE_X_AXIS)
    axes_arr = np.atleast_1d(axes)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    csv_rows: list[tuple[str, str, str, str, float, float, float, float]] = []
    table_values: dict[str, dict[str, tuple[float, float]]] = {}
    selected_epochs: dict[str, float] = {}
    selected_checkpoints: dict[Path, SelectedCheckpoint] = {}
    omitted_initial_best_plot_runs: set[Path] = set()
    baseline_values = baseline_table_row() if LATEX_INCLUDE_BASELINE_ROW else {}
    if baseline_values:
        table_values[BASELINE_LABEL] = baseline_values

    for run_idx, (run_dir, tag_series) in enumerate(per_run.items()):
        label_base = run_label(run_dir, run_idx)
        selected_checkpoint = selected_checkpoint_for_run(run_dir, label_base, tag_series)
        if selected_checkpoint is None:
            continue
        selected_checkpoints[run_dir] = selected_checkpoint
        selected_step = selected_checkpoint.step
        selected_value = selected_checkpoint.value
        selected_epochs[label_base] = selected_step
        print(
            f"Selected {label_base!r}: epoch={selected_step:g}, "
            f"{SELECTION_SPLIT_TAG_PREFIX}/{SELECTION_SCALAR_NAME}={selected_value:.6g}"
        )
        if OMIT_INITIAL_BEST_RUNS_FROM_PLOT and selected_checkpoint.is_initial:
            omitted_initial_best_plot_runs.add(run_dir)
            print(f"  - Keeping {label_base!r} in the table but omitting it from the plot because its best epoch is initial.")
        for column in TABLE_COLUMNS:
            tag = f"{column.split_tag_prefix}/{column.scalar_name}"
            if tag not in tag_series:
                print(f"Warning: table metric {tag!r} missing in {run_dir}.")
                continue
            steps, smoothed, _lower, _upper = smooth_series(tag_series[tag])
            marker_point = interpolated_point_at_step(steps, smoothed, selected_step)
            if marker_point is None:
                print(f"Warning: selected epoch {selected_step:g} outside range for table metric {tag!r} in {run_dir}.")
                continue
            table_values.setdefault(label_base, {})[column.label] = marker_point

    for ax, split in zip(axes_arr, SPLITS):
        plotted_count = 0
        marker_steps: list[float] = []
        baseline_value = baseline_value_for(split, metric)
        if PLOT_BASELINE_LINES and baseline_value is not None and np.isfinite(float(baseline_value)):
            ax.axhline(
                float(baseline_value),
                color=BASELINE_COLOR,
                linestyle="--",
                linewidth=BASELINE_LINE_WIDTH,
                label=BASELINE_LABEL,
            )

        for run_idx, (run_dir, tag_series) in enumerate(per_run.items()):
            label_base = run_label(run_dir, run_idx)
            if run_dir in omitted_initial_best_plot_runs:
                continue
            highlighted = is_highlighted_run(run_dir, label_base)
            color = HIGHLIGHT_COLOR if highlighted else colors[run_idx % len(colors)]
            line_width = HIGHLIGHT_LINE_WIDTH if highlighted else LINE_WIDTH
            marker_size = HIGHLIGHT_MARKER_SIZE if highlighted else FINAL_MARKER_SIZE
            zorder = 4 if highlighted else 2
            tag = f"{split.tag_prefix}/{metric.scalar_name}"
            if tag not in tag_series:
                print(f"Warning: {tag!r} missing in {run_dir}.")
                continue

            steps, smoothed, lower, upper = smooth_series(tag_series[tag])

            ax.plot(
                steps,
                smoothed,
                linewidth=line_width,
                color=color,
                linestyle=split.linestyle,
                label=label_base,
                zorder=zorder,
            )
            if PLOT_ROLLING_STD_BANDS:
                ax.fill_between(steps, lower, upper, color=color, alpha=BAND_ALPHA, linewidth=0)

            if PLOT_FINAL_MARKERS or PLOT_FINAL_VALUE_LABELS:
                selected_checkpoint = selected_checkpoints.get(run_dir)
                marker_point = (
                    interpolated_point_at_step(steps, smoothed, selected_checkpoint.step)
                    if selected_checkpoint is not None
                    else None
                )
                if marker_point is not None:
                    marker_step, marker_value = marker_point
                    if PLOT_FINAL_MARKERS:
                        ax.scatter(
                            marker_step,
                            marker_value,
                            color=color,
                            s=marker_size,
                            zorder=zorder + 1,
                        )
                    if PLOT_FINAL_VALUE_LABELS:
                        ax.annotate(
                            f"{marker_value:.3g}",
                            xy=(marker_step, marker_value),
                            xytext=(FINAL_LABEL_X_OFFSET_POINTS, 0),
                            textcoords="offset points",
                            va="center",
                            fontsize=FINAL_LABEL_FONT_SIZE,
                            color=color,
                            annotation_clip=False,
                        )
                    marker_steps.append(marker_step)

            plotted_count += 1
            if SAVE_CSV:
                csv_rows.extend(
                    (
                        label_base,
                        split.label,
                        tag,
                        metric.label,
                        float(step),
                        float(value),
                        float(lo),
                        float(hi),
                    )
                    for step, value, lo, hi in zip(steps, smoothed, lower, upper)
                )

        if plotted_count == 0:
            print(f"Warning: skipping {split.tag_prefix!r}; no selected runs contain {split.tag_prefix}/{metric.scalar_name}.")
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
        elif marker_steps:
            xmin, xmax = ax.get_xlim()
            max_final_step = max(marker_steps)
            width = max(xmax - xmin, 1.0)
            ax.set_xlim(xmin, max(xmax, max_final_step + FINAL_LABEL_RIGHT_MARGIN_FRACTION * width))

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
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), frameon=False)
    if SAVE_CSV:
        save_metric_csv(OUTPUT_DIR / f"{OUTPUT_BASENAME}_individual_runs.csv", csv_rows)
    if SAVE_LATEX_TABLE and table_values:
        latex_table = make_latex_performance_table(table_values, selected_epochs)
        save_latex_table(OUTPUT_DIR / f"{OUTPUT_BASENAME}_performance_table.tex", latex_table)
        if PRINT_LATEX_TABLE:
            print("\nLaTeX performance table:\n")
            print(latex_table)
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


def main() -> None:
    plot(discover_run_dirs())


if __name__ == "__main__":
    main()
