"""Plot smoothed displacement, force, and combined aggregate validation errors.

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
# discovery/sorting. RUN_LABELS_BY_RUN can key by run path or default run label.
RUN_LABELS: list[str] = []
RUN_LABELS_BY_RUN: dict[str, str] = {
    "fhat/loss_ablation/ABLATION_onestep": "Residual loss only",
    "fhat/loss_ablation/ABLATION_mse": "+ MSE loss",
    "fhat/loss_ablation/ABLATION_std": "+ Std loss",
    "fhat/loss_ablation/ABLATION_std_freq": "+ Std and Frequency loss",
    "fhat/loss_ablation/ABLATION_std_psd": "+ Std and PSD loss",
}

# Number of discovered runs to include. Set to None to use all discovered runs.
NUM_RUNS: int | None = None

# Used only when RUN_DIRS is empty.
RUN_NAME_CONTAINS: str | None = "fhat/loss_ablation"
SORT_RUNS_BY = "mtime_desc"  # "mtime_desc", "mtime_asc", or "name"

OUTPUT_DIR = Path("figs/tensorboard_validation")
OUTPUT_BASENAME = "aggregate_validation_seen_components"
DPI = 300

PLOT_TITLE: str | None = None
X_LABEL = "Epoch"
X_LIMITS: tuple[float, float] | None = (0.0, 500.0)
X_LIMIT_MARGIN = 5.0
SHOW_SUBPLOT_TITLES = False
Y_LABEL_FONT_SIZE = 14.4
Y_LABEL_ROTATION = 0
Y_LABEL_PAD = 11
Y_SCALE = "log"  # "linear" or "log"
GRID_ALPHA = 0.25
FIGSIZE = (10.0, 9.0)
SHARE_X_AXIS = True

# Smoothing is applied across logged validation points, not raw epoch integers.
# Set to 1 to disable smoothing.
SMOOTH_WINDOW_POINTS = 7

# Draw per-run rolling std bands around each smoothed line. Since each run is
# plotted separately, this is temporal variation within a run, not cross-run std.
PLOT_ROLLING_STD_BANDS = False
BAND_STD_MULTIPLIER = 1.0
BAND_ALPHA = 0.10
LINE_WIDTH = 1.8
PLOT_FINAL_MARKERS = True
PLOT_FINAL_VALUE_LABELS = False
FINAL_MARKER_STEPS_BY_RUN: dict[str, float] = {
    "mean_loss_ablation/ABLATION_onestep": 161.0,
}
FINAL_MARKER_SIZE = 34
FINAL_LABEL_FONT_SIZE = 8
FINAL_LABEL_X_OFFSET_POINTS = 5
FINAL_LABEL_RIGHT_MARGIN_FRACTION = 0.08

PLOT_BASELINE_LINES = True
BASELINE_LABEL = "Baseline VIVANA-TD"
BASELINE_LINE_WIDTH = 1.4
PLOT_BASELINE_Y_TICKS = True
BASELINE_ERRORS_BY_METRIC: dict[str, float] = {
    "displacement": 0.752,
    "force": 0.316,
    "combined": 0.534,
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
    "Ablation performance at the selected best-performing checkpoints. "
    "The aggregate metric is the mean of the four component errors. "
    "Lower values indicate better performance."
)
LATEX_TABLE_LABEL = "tab:loss_ablation_performance"
LATEX_BOLD_BEST_PER_METRIC = True
LATEX_TABLE_COLSEP_PT = 3
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
class TableMetricConfig:
    scalar_name: str
    label: str
    latex_header: str


SPLITS: list[SplitConfig] = [
    SplitConfig(tag_prefix="val_seen", label="seen", linestyle="-"),
]


METRICS: list[MetricConfig] = [
    MetricConfig(
        scalar_name="Aggregate displacement error",
        label="displacement",
        title="Displacement aggregate error",
        y_label=r"$\bar{\varepsilon}_y$",
    ),
    MetricConfig(
        scalar_name="Aggregate force error",
        label="force",
        title="Force aggregate error",
        y_label=r"$\bar{\varepsilon}_F$",
    ),
    MetricConfig(
        scalar_name="Aggregate validation error",
        label="combined",
        title="Combined aggregate error",
        y_label=r"$\bar{\varepsilon}$",
    ),
]

TABLE_METRICS: list[TableMetricConfig] = [
    TableMetricConfig(
        scalar_name="Dominant frequency relative error",
        label="disp_frequency",
        latex_header=r"$\varepsilon_{f,y}$",
    ),
    TableMetricConfig(
        scalar_name="Displacement std relative error",
        label="disp_std",
        latex_header=r"$\varepsilon_{\sigma,y}$",
    ),
    TableMetricConfig(
        scalar_name="Force dominant frequency relative error",
        label="force_frequency",
        latex_header=r"$\varepsilon_{f,F}$",
    ),
    TableMetricConfig(
        scalar_name="Force std relative error",
        label="force_std",
        latex_header=r"$\varepsilon_{\sigma,F}$",
    ),
    TableMetricConfig(
        scalar_name="Aggregate validation error",
        label="combined",
        latex_header=r"$\bar{\varepsilon}$",
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


def _run_label_order_key(run_dir: Path) -> tuple[int, int, str]:
    label_order = {key: idx for idx, key in enumerate(RUN_LABELS_BY_RUN)}
    default_label = default_run_label(run_dir)
    for key in (str(run_dir), default_label):
        if key in label_order:
            return (0, label_order[key], default_label)
    return (1, len(label_order), default_label)


def discover_run_dirs() -> list[Path]:
    if RUN_DIRS:
        run_dirs = [Path(path).expanduser() for path in RUN_DIRS]
        return sorted(run_dirs, key=_run_label_order_key) if RUN_LABELS_BY_RUN else run_dirs

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


def make_latex_performance_table(
    table_values: dict[str, dict[str, tuple[float, float]]],
) -> str:
    metric_headers = {metric.label: metric.latex_header for metric in TABLE_METRICS}
    metric_order = [metric.label for metric in TABLE_METRICS]
    best_by_metric: dict[str, float] = {}
    if LATEX_BOLD_BEST_PER_METRIC:
        for metric_label in metric_order:
            values = [
                metric_values[metric_label][1]
                for metric_values in table_values.values()
                if metric_label in metric_values
            ]
            if values:
                best_by_metric[metric_label] = min(values)

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        rf"\setlength{{\tabcolsep}}{{{LATEX_TABLE_COLSEP_PT}pt}}",
        r"\begin{tabular}{c c c c c | c}",
        r"\toprule",
        "Variant & "
        + " & ".join(metric_headers.get(metric_label, latex_escape(metric_label)) for metric_label in metric_order)
        + r" \\",
        r"\midrule",
    ]

    for run_label, metric_values in table_values.items():
        row = [latex_escape(run_label)]
        for metric_label in metric_order:
            if metric_label not in metric_values:
                row.append("--")
                continue
            value = metric_values[metric_label][1]
            best_value = best_by_metric.get(metric_label)
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


def default_run_label(run_dir: Path) -> str:
    if run_dir.parent == LOG_ROOT:
        return run_dir.name
    return str(run_dir.relative_to(LOG_ROOT)) if run_dir.is_relative_to(LOG_ROOT) else run_dir.name


def run_label(run_dir: Path, run_idx: int | None = None) -> str:
    default_label = default_run_label(run_dir)
    if run_idx is not None and run_idx < len(RUN_LABELS) and RUN_LABELS[run_idx]:
        return RUN_LABELS[run_idx]

    for key in (str(run_dir), default_label):
        label = RUN_LABELS_BY_RUN.get(key)
        if label:
            return label
    return default_label


def final_marker_point(run_dir: Path, label: str, steps: np.ndarray, values: np.ndarray) -> tuple[float, float] | None:
    if steps.size == 0 or values.size == 0:
        return None
    marker_step = None
    default_label = default_run_label(run_dir)
    for key in (str(run_dir), default_label, label):
        if key in FINAL_MARKER_STEPS_BY_RUN:
            marker_step = float(FINAL_MARKER_STEPS_BY_RUN[key])
            break

    if marker_step is None:
        marker_step = float(steps[-1])
        marker_value = float(values[-1])
    else:
        if marker_step < float(steps[0]) or marker_step > float(steps[-1]):
            print(f"Warning: marker step {marker_step:g} outside logged range for {run_dir}; using final point.")
            marker_step = float(steps[-1])
            marker_value = float(values[-1])
        else:
            marker_value = float(np.interp(marker_step, steps, values))

    if not np.isfinite(marker_step) or not np.isfinite(marker_value):
        return None
    return marker_step, marker_value


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


def plot(run_dirs: list[Path]) -> None:
    if not run_dirs:
        raise ValueError(f"No TensorBoard runs found under {LOG_ROOT}")

    plot_tags = [f"{split.tag_prefix}/{metric.scalar_name}" for split in SPLITS for metric in METRICS]
    table_tags = [f"{split.tag_prefix}/{metric.scalar_name}" for split in SPLITS for metric in TABLE_METRICS]
    tags = sorted({*plot_tags, *table_tags})
    print("Using runs:")
    for run_idx, run_dir in enumerate(run_dirs):
        print(f"  - {run_dir} as {run_label(run_dir, run_idx)!r}")

    per_run = {run_dir: load_selected_scalars(run_dir, tags) for run_dir in run_dirs}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(METRICS), 1, figsize=FIGSIZE, sharex=SHARE_X_AXIS)
    axes_arr = np.atleast_1d(axes)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    csv_rows: list[tuple[str, str, str, str, float, float, float, float]] = []
    table_values: dict[str, dict[str, tuple[float, float]]] = {}

    for run_idx, (run_dir, tag_series) in enumerate(per_run.items()):
        label_base = run_label(run_dir, run_idx)
        for split in SPLITS:
            for table_metric in TABLE_METRICS:
                tag = f"{split.tag_prefix}/{table_metric.scalar_name}"
                if tag not in tag_series:
                    print(f"Warning: table metric {tag!r} missing in {run_dir}.")
                    continue
                steps, smoothed, _lower, _upper = smooth_series(tag_series[tag])
                marker_point = final_marker_point(run_dir, label_base, steps, smoothed)
                if marker_point is None:
                    continue
                table_values.setdefault(label_base, {})[table_metric.label] = marker_point

    for ax, metric in zip(axes_arr, METRICS):
        plotted_count = 0
        marker_steps: list[float] = []
        baseline_value = BASELINE_ERRORS_BY_METRIC.get(metric.label)
        if PLOT_BASELINE_LINES and baseline_value is not None and np.isfinite(float(baseline_value)):
            ax.axhline(
                float(baseline_value),
                color="black",
                linestyle="--",
                linewidth=BASELINE_LINE_WIDTH,
                label=BASELINE_LABEL,
            )

        for run_idx, (run_dir, tag_series) in enumerate(per_run.items()):
            color = colors[run_idx % len(colors)]
            label_base = run_label(run_dir, run_idx)
            for split in SPLITS:
                tag = f"{split.tag_prefix}/{metric.scalar_name}"
                if tag not in tag_series:
                    print(f"Warning: {tag!r} missing in {run_dir}.")
                    continue

                steps, smoothed, lower, upper = smooth_series(tag_series[tag])
                label = f"{label_base} ({split.label})" if len(SPLITS) > 1 else label_base

                ax.plot(
                    steps,
                    smoothed,
                    linewidth=LINE_WIDTH,
                    color=color,
                    linestyle=split.linestyle,
                    label=label,
                )
                if PLOT_ROLLING_STD_BANDS:
                    ax.fill_between(steps, lower, upper, color=color, alpha=BAND_ALPHA, linewidth=0)

                if PLOT_FINAL_MARKERS or PLOT_FINAL_VALUE_LABELS:
                    marker_point = final_marker_point(run_dir, label_base, steps, smoothed)
                    if marker_point is not None:
                        marker_step, marker_value = marker_point
                        if PLOT_FINAL_MARKERS:
                            ax.scatter(
                                marker_step,
                                marker_value,
                                color=color,
                                s=FINAL_MARKER_SIZE,
                                zorder=3,
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
            metric_tags = ", ".join(f"{split.tag_prefix}/{metric.scalar_name}" for split in SPLITS)
            print(f"Warning: skipping {metric.scalar_name!r}; no selected runs contain any of: {metric_tags}.")
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
        latex_table = make_latex_performance_table(table_values)
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
