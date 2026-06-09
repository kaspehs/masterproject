from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

_LOCAL_CACHE_DIR = (Path(__file__).resolve().parent / ".plot_cache").resolve()
_MPL_CACHE_DIR = _LOCAL_CACHE_DIR / "matplotlib"
_XDG_CACHE_DIR = _LOCAL_CACHE_DIR / "xdg"
_MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_XDG_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


PARAMETER_NAMES = ("Cv", "Cd", "Ca", "fhat0", "fhat_min", "fhat_max")
PARAMETER_PLOT_ORDER = ("Cv", "fhat_max", "Cd", "fhat0", "Ca", "fhat_min")
DEFAULT_METRIC_COLUMNS = (
    "displacement_std_relative_error",
    "dominant_frequency_relative_error",
    "force_dominant_frequency_relative_error",
    "force_std_relative_error",
)
FULL_SERIES_NRMSE_Y_COLUMN_CANDIDATES = (
    "full_series_nrmse_y",
    "full_series_NRSME_y",
    "full_series_NRMSE_y",
    "full_series_displacement_nrmse",
    "full_series_displacement_NRMSE",
    "full_series_rel_rmse_y",
    "full_series_relative_rmse_y",
)
ONE_STEP_NRMSE_Y_COLUMN_CANDIDATES = (
    "one_step_nrmse_y",
    "one_step_NRMSE_y",
    "one_step_displacement_nrmse",
    "one_step_displacement_NRMSE",
    "one_step_rel_rmse_y",
    "one_step_relative_rmse_y",
)
REQUIRED_RUN_FILENAMES = ("run_config.json", "best_result.json", "history.csv")
METRIC_AXIS_LABELS = {
    "displacement_std_relative_error": r"$\epsilon_{\sigma}^{y}$",
    "dominant_frequency_relative_error": r"$\epsilon_{\omega}^{y}$",
    "force_dominant_frequency_relative_error": r"$\epsilon_{\omega}^{f}$",
    "force_std_relative_error": r"$\epsilon_{\sigma}^{f}$",
    "full_series_nrmse_y": r"$\mathrm{NRMSE}_{y,\mathrm{full}}$",
    "full_series_NRSME_y": r"$\mathrm{NRMSE}_{y,\mathrm{full}}$",
    "full_series_NRMSE_y": r"$\mathrm{NRMSE}_{y,\mathrm{full}}$",
    "full_series_displacement_nrmse": r"$\mathrm{NRMSE}_{y,\mathrm{full}}$",
    "full_series_displacement_NRMSE": r"$\mathrm{NRMSE}_{y,\mathrm{full}}$",
    "full_series_rel_rmse_y": r"$\mathrm{NRMSE}_{y,\mathrm{full}}$",
    "full_series_relative_rmse_y": r"$\mathrm{NRMSE}_{y,\mathrm{full}}$",
}
PARAMETER_AXIS_LABELS = {
    "Cv": r"$C_v$",
    "Cd": r"$C_d$",
    "Ca": r"$C_a$",
    "fhat0": r"$\hat{f}_0$",
    "fhat_min": r"$\hat{f}_{\min}$",
    "fhat_max": r"$\hat{f}_{\max}$",
}
GENERATION_BEST_LABEL = "Best candidate in generation"
BEST_SO_FAR_LABEL = "Best candidate overall"
GENERATION_MEAN_LABEL = "Generation mean"
GENERATION_WORST_LABEL = "Generation worst"
BASELINE_LABEL = "VIVANA-TD baseline"
PARAMETER_Y_AXIS_FONT_SCALE = 1.0
METRIC_Y_AXIS_FONT_SCALE = 1.0

FIGURE_PREFIX = "fig_07_vivana_td_ga"
FIGURE_DPI = 300
SAVE_PNG_PREVIEW = False
ERROR_SCALE = 100.0

BASE_FONT_SIZE = 8
AXIS_LABEL_FONT_SIZE = 9
TICK_FONT_SIZE = 8
LEGEND_FONT_SIZE = 8
PANEL_LABEL_FONT_SIZE = 9
SPINE_COLOR = "0.65"
SPINE_LINE_WIDTH = 0.6
GRID_COLOR = "0.88"
GRID_MINOR_COLOR = "0.94"
THESIS_FIGSIZE_SINGLE = (5.85, 3.1)
THESIS_FIGSIZE_1X2 = (5.85, 2.9)
THESIS_FIGSIZE_2X2 = (5.85, 4.8)
THESIS_FIGSIZE_3X2 = (5.85, 6.4)

LINE_STYLES = {
    GENERATION_BEST_LABEL: {"color": "#0072B2", "marker": "o", "linestyle": "-", "linewidth": 1.25, "markersize": 3.0},
    BEST_SO_FAR_LABEL: {"color": "#009E73", "marker": "s", "linestyle": "-", "linewidth": 1.45, "markersize": 3.0},
    GENERATION_MEAN_LABEL: {"color": "0.35", "marker": None, "linestyle": "-", "linewidth": 1.1, "markersize": 0.0},
    GENERATION_WORST_LABEL: {"color": "0.62", "marker": None, "linestyle": ":", "linewidth": 1.0, "markersize": 0.0},
    BASELINE_LABEL: {"color": "0.25", "marker": None, "linestyle": "--", "linewidth": 1.0, "markersize": 0.0},
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_history(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows found in {path}.")
    return rows


def _apply_thesis_rcparams() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Latin Modern Roman", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": BASE_FONT_SIZE,
            "axes.labelsize": AXIS_LABEL_FONT_SIZE,
            "axes.titlesize": PANEL_LABEL_FONT_SIZE,
            "axes.linewidth": SPINE_LINE_WIDTH,
            "xtick.labelsize": TICK_FONT_SIZE,
            "ytick.labelsize": TICK_FONT_SIZE,
            "legend.fontsize": LEGEND_FONT_SIZE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": FIGURE_DPI,
        }
    )


def _apply_axes_style(ax: plt.Axes, *, minor_grid: bool = False) -> None:
    ax.grid(True, which="major", color=GRID_COLOR, linewidth=0.5, alpha=0.75)
    if minor_grid:
        ax.grid(True, which="minor", color=GRID_MINOR_COLOR, linewidth=0.35, alpha=0.45)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(SPINE_LINE_WIDTH)
        spine.set_edgecolor(SPINE_COLOR)


def _style_for(label: str) -> dict[str, Any]:
    return dict(LINE_STYLES.get(label, LINE_STYLES[GENERATION_BEST_LABEL]))


def _plot_line(ax: plt.Axes, x: np.ndarray, y: np.ndarray, label: str, **kwargs: Any) -> None:
    style = _style_for(label)
    style.update(kwargs)
    marker = style.pop("marker", None)
    markersize = float(style.pop("markersize", 0.0))
    ax.plot(
        x,
        y,
        marker=marker,
        markersize=markersize,
        label=label,
        **style,
    )


def _dedup_handles_labels(handles: list[Any], labels: list[str]) -> tuple[list[Any], list[str]]:
    seen: set[str] = set()
    out_handles: list[Any] = []
    out_labels: list[str] = []
    for handle, label in zip(handles, labels):
        if not label or label in seen:
            continue
        seen.add(label)
        out_handles.append(handle)
        out_labels.append(label)
    return out_handles, out_labels


def _legend_above(fig: plt.Figure, handles: list[Any], labels: list[str], *, ncol: int) -> None:
    handles, labels = _dedup_handles_labels(handles, labels)
    if not handles:
        return
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=int(ncol),
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        columnspacing=1.2,
        handletextpad=0.6,
    )


def _scaled_error(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr * ERROR_SCALE


def _objective_scale(loss_mode: str | None) -> float:
    return ERROR_SCALE if str(loss_mode or "mae").strip().lower() == "mae" else 1.0


def _objective_ylabel(loss_mode: str | None) -> str:
    return r"Objective [%]" if _objective_scale(loss_mode) == ERROR_SCALE else "Objective"


def _metric_ylabel(column: str) -> str:
    label = METRIC_AXIS_LABELS.get(column, column.replace("_", " "))
    return f"{label} [%]"


def _figure_path(output_dir: Path, suffix: str) -> Path:
    return output_dir / f"{FIGURE_PREFIX}_{suffix}.pdf"


def _available_columns(history_rows: list[dict[str, Any]]) -> set[str]:
    columns: set[str] = set()
    for row in history_rows:
        columns.update(row.keys())
    return columns


def _first_present(candidates: tuple[str, ...], columns: set[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _resolve_metric_columns(history_rows: list[dict[str, Any]]) -> tuple[str, ...]:
    columns = _available_columns(history_rows)
    metric_columns = [column for column in DEFAULT_METRIC_COLUMNS if column in columns]

    full_series_nrmse_y = _first_present(FULL_SERIES_NRMSE_Y_COLUMN_CANDIDATES, columns)
    one_step_nrmse_y = _first_present(ONE_STEP_NRMSE_Y_COLUMN_CANDIDATES, columns)
    if full_series_nrmse_y is not None and full_series_nrmse_y not in metric_columns:
        metric_columns.insert(0, full_series_nrmse_y)
    if one_step_nrmse_y is not None:
        metric_columns = [column for column in metric_columns if column != one_step_nrmse_y]

    if not metric_columns:
        raise ValueError("No GA metric columns were found in history.csv.")
    return tuple(metric_columns)


def _is_run_dir(path: Path) -> bool:
    return path.is_dir() and all((path / filename).is_file() for filename in REQUIRED_RUN_FILENAMES)


def _seed_sort_key(path: Path) -> tuple[int, str]:
    suffix = path.name.removeprefix("seed_")
    try:
        return (0, f"{int(suffix):09d}")
    except ValueError:
        return (1, path.name)


def _discover_seed_run_dirs(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted((child for child in path.iterdir() if _is_run_dir(child)), key=_seed_sort_key)


def _to_bool(value: Any) -> bool:
    text = str(value).strip().lower()
    return text in {"1", "true", "yes"}


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _generation_rows(history_rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in history_rows:
        generation = int(row["generation"])
        if generation < 0:
            continue
        grouped.setdefault(generation, []).append(row)
    if not grouped:
        raise ValueError("No non-negative generations found in history.")
    return dict(sorted(grouped.items()))


def _best_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    objectives = np.asarray([_to_float(row["objective"]) for row in rows], dtype=float)
    return rows[int(np.nanargmin(objectives))]


def _candidate_signature(row: dict[str, Any]) -> tuple[float, ...]:
    return tuple(round(_to_float(row[name]), 12) for name in PARAMETER_NAMES)


def build_generation_summary(history_rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped = _generation_rows(history_rows)
    baseline_row = next((row for row in history_rows if int(row["generation"]) == -1), None)
    metric_columns = _resolve_metric_columns(history_rows)

    generations: list[int] = []
    generation_best_objective: list[float] = []
    generation_mean_objective: list[float] = []
    generation_worst_objective: list[float] = []
    best_so_far_objective: list[float] = []
    cache_hit_fraction: list[float] = []
    failure_fraction: list[float] = []
    unique_fraction: list[float] = []

    generation_best_params = {name: [] for name in PARAMETER_NAMES}
    best_so_far_params = {name: [] for name in PARAMETER_NAMES}
    generation_best_metrics = {name: [] for name in metric_columns}
    best_so_far_metrics = {name: [] for name in metric_columns}

    running_best_row: dict[str, Any] | None = None
    running_best_objective = float("inf")

    for generation, rows in grouped.items():
        generations.append(generation)
        objectives = np.asarray([_to_float(row["objective"]) for row in rows], dtype=float)
        best_idx = int(np.nanargmin(objectives))
        best_row = rows[best_idx]
        best_objective = float(objectives[best_idx])

        generation_best_objective.append(best_objective)
        generation_mean_objective.append(float(np.nanmean(objectives)))
        generation_worst_objective.append(float(np.nanmax(objectives)))

        if best_objective < running_best_objective or running_best_row is None:
            running_best_row = best_row
            running_best_objective = best_objective
        best_so_far_objective.append(running_best_objective)

        total = max(len(rows), 1)
        cache_hit_fraction.append(sum(_to_bool(row["cache_hit"]) for row in rows) / total)
        failure_fraction.append(sum(_to_bool(row["failure"]) for row in rows) / total)
        unique_fraction.append(len({_candidate_signature(row) for row in rows}) / total)

        for name in PARAMETER_NAMES:
            generation_best_params[name].append(_to_float(best_row[name]))
            best_so_far_params[name].append(_to_float(running_best_row[name]))
        for name in metric_columns:
            generation_best_metrics[name].append(_to_float(best_row[name]))
            best_so_far_metrics[name].append(_to_float(running_best_row[name]))

    return {
        "baseline_row": baseline_row,
        "metric_columns": metric_columns,
        "generations": np.asarray(generations, dtype=int),
        "generation_best_objective": np.asarray(generation_best_objective, dtype=float),
        "generation_mean_objective": np.asarray(generation_mean_objective, dtype=float),
        "generation_worst_objective": np.asarray(generation_worst_objective, dtype=float),
        "best_so_far_objective": np.asarray(best_so_far_objective, dtype=float),
        "cache_hit_fraction": np.asarray(cache_hit_fraction, dtype=float),
        "failure_fraction": np.asarray(failure_fraction, dtype=float),
        "unique_fraction": np.asarray(unique_fraction, dtype=float),
        "generation_best_params": {name: np.asarray(values, dtype=float) for name, values in generation_best_params.items()},
        "best_so_far_params": {name: np.asarray(values, dtype=float) for name, values in best_so_far_params.items()},
        "generation_best_metrics": {name: np.asarray(values, dtype=float) for name, values in generation_best_metrics.items()},
        "best_so_far_metrics": {name: np.asarray(values, dtype=float) for name, values in best_so_far_metrics.items()},
    }


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output_path = path.with_suffix(".pdf")
    fig.savefig(output_path, format="pdf", dpi=FIGURE_DPI, bbox_inches="tight", pad_inches=0.02)
    if SAVE_PNG_PREVIEW:
        fig.savefig(output_path.with_suffix(".png"), dpi=FIGURE_DPI, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def _add_panel_label(ax: plt.Axes, index: int) -> None:
    ax.text(
        0.02,
        0.98,
        f"({chr(ord('a') + index)})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=PANEL_LABEL_FONT_SIZE,
    )


def _metric_grid(metric_columns: tuple[str, ...]) -> tuple[plt.Figure, np.ndarray, int, int]:
    n_metrics = len(metric_columns)
    if n_metrics <= 1:
        rows, cols, figsize = 1, 1, THESIS_FIGSIZE_SINGLE
    elif n_metrics <= 2:
        rows, cols, figsize = 1, 2, THESIS_FIGSIZE_1X2
    elif n_metrics <= 4:
        rows, cols, figsize = 2, 2, THESIS_FIGSIZE_2X2
    else:
        rows, cols, figsize = 3, 2, THESIS_FIGSIZE_3X2
    fig, axes = plt.subplots(rows, cols, figsize=figsize, sharex=True, squeeze=False)
    axes_flat = axes.reshape(-1)
    for ax in axes_flat[n_metrics:]:
        ax.set_visible(False)
    return fig, axes_flat, rows, cols


def _is_bottom_row(index: int, n_metrics: int, cols: int) -> bool:
    return index + cols >= n_metrics


def _set_parameter_ylim_from_values(ax: plt.Axes, values: list[Any]) -> None:
    arrays = [np.asarray(value, dtype=float).reshape(-1) for value in values]
    finite_values = np.concatenate([arr[np.isfinite(arr)] for arr in arrays if arr.size])
    if finite_values.size == 0:
        return
    ymin = float(np.min(finite_values))
    ymax = float(np.max(finite_values))
    if ymin == ymax:
        pad = 0.05 * max(abs(ymin), 1.0)
        ax.set_ylim(ymin - pad, ymax + pad)
        return
    ax.set_ylim(ymin, ymax)


def _format_seed_label(seed_value: Any, fallback: str) -> str:
    if seed_value is None:
        return fallback.replace("_", " ")
    return f"Seed {int(seed_value)}"


def _scaled_fontsize(rc_key: str, scale: float) -> float:
    base_size = plt.rcParams.get(rc_key, 10.0)
    if isinstance(base_size, str):
        base_size = FontProperties(size=base_size).get_size_in_points()
    return float(base_size) * float(scale)


def plot_objective_evolution(
    summary: dict[str, Any],
    output_path: Path,
    baseline_objective: float | None,
    loss_mode: str | None,
) -> None:
    generations = summary["generations"]
    scale = _objective_scale(loss_mode)
    fig, ax = plt.subplots(figsize=THESIS_FIGSIZE_SINGLE)
    _plot_line(ax, generations, summary["generation_best_objective"] * scale, GENERATION_BEST_LABEL)
    _plot_line(ax, generations, summary["best_so_far_objective"] * scale, BEST_SO_FAR_LABEL)
    _plot_line(ax, generations, summary["generation_mean_objective"] * scale, GENERATION_MEAN_LABEL, alpha=0.9)
    _plot_line(ax, generations, summary["generation_worst_objective"] * scale, GENERATION_WORST_LABEL, alpha=0.75)
    if baseline_objective is not None and np.isfinite(baseline_objective):
        style = _style_for(BASELINE_LABEL)
        ax.axhline(float(baseline_objective) * scale, label=BASELINE_LABEL, **{k: v for k, v in style.items() if k not in {"marker", "markersize"}})
    ax.set_xlabel("Generation")
    ax.set_ylabel(_objective_ylabel(loss_mode))
    ax.set_yscale("log")
    _apply_axes_style(ax, minor_grid=True)
    handles, labels = ax.get_legend_handles_labels()
    _legend_above(fig, handles, labels, ncol=3)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91), pad=0.35)
    _save_figure(fig, output_path)


def plot_parameter_evolution(
    summary: dict[str, Any],
    output_path: Path,
    baseline_params: dict[str, float],
    bounds: dict[str, tuple[float, float]] | dict[str, list[float]],
) -> None:
    generations = summary["generations"]
    fig, axes = plt.subplots(3, 2, figsize=THESIS_FIGSIZE_3X2, sharex=True)
    axes_flat = axes.reshape(-1)
    ylabel_fontsize = _scaled_fontsize("axes.labelsize", PARAMETER_Y_AXIS_FONT_SCALE)

    for parameter_idx, (ax, name) in enumerate(zip(axes_flat, PARAMETER_PLOT_ORDER)):
        gen_best = summary["generation_best_params"][name]
        best_so_far = summary["best_so_far_params"][name]
        _plot_line(ax, generations, gen_best, GENERATION_BEST_LABEL)
        _plot_line(ax, generations, best_so_far, BEST_SO_FAR_LABEL)
        ylim_values: list[Any] = [gen_best, best_so_far]
        if name in baseline_params:
            style = _style_for(BASELINE_LABEL)
            baseline_value = float(baseline_params[name])
            ax.axhline(baseline_value, label=BASELINE_LABEL, **{k: v for k, v in style.items() if k not in {"marker", "markersize"}})
            ylim_values.append([baseline_value])
        _set_parameter_ylim_from_values(ax, ylim_values)
        _apply_axes_style(ax)
        ax.set_ylabel(PARAMETER_AXIS_LABELS.get(name, name), fontsize=ylabel_fontsize)
        _add_panel_label(ax, parameter_idx)

    axes_flat[-2].set_xlabel("Generation")
    axes_flat[-1].set_xlabel("Generation")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    _legend_above(fig, handles, labels, ncol=3)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93), pad=0.35, w_pad=0.7, h_pad=0.55)
    _save_figure(fig, output_path)


def plot_metric_evolution(summary: dict[str, Any], output_path: Path) -> None:
    generations = summary["generations"]
    metric_columns = tuple(summary["metric_columns"])
    fig, axes_flat, _, cols = _metric_grid(metric_columns)
    ylabel_fontsize = _scaled_fontsize("axes.labelsize", METRIC_Y_AXIS_FONT_SCALE)

    for metric_idx, (ax, name) in enumerate(zip(axes_flat, metric_columns)):
        _plot_line(ax, generations, _scaled_error(summary["generation_best_metrics"][name]), GENERATION_BEST_LABEL)
        _plot_line(ax, generations, _scaled_error(summary["best_so_far_metrics"][name]), BEST_SO_FAR_LABEL)
        ax.set_ylabel(_metric_ylabel(name), fontsize=ylabel_fontsize)
        if _is_bottom_row(metric_idx, len(metric_columns), cols):
            ax.set_xlabel("Generation")
        ax.set_yscale("log")
        _apply_axes_style(ax, minor_grid=True)
        _add_panel_label(ax, metric_idx)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    _legend_above(fig, handles, labels, ncol=2)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93), pad=0.35, w_pad=0.8, h_pad=0.55)
    _save_figure(fig, output_path)


def plot_multi_seed_metric_evolution(
    seed_runs: list[dict[str, Any]],
    output_path: Path,
    baseline_metrics: dict[str, float],
) -> None:
    metric_columns = tuple(seed_runs[0]["summary"]["metric_columns"])
    fig, axes_flat, _, cols = _metric_grid(metric_columns)
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    ylabel_fontsize = _scaled_fontsize("axes.labelsize", METRIC_Y_AXIS_FONT_SCALE)

    for seed_idx, seed_run in enumerate(seed_runs):
        summary = seed_run["summary"]
        generations = summary["generations"]
        color = colors[seed_idx % len(colors)] if colors else None
        for metric_idx, (ax, name) in enumerate(zip(axes_flat, metric_columns)):
            ax.plot(
                generations,
                _scaled_error(summary["best_so_far_metrics"][name]),
                marker="o",
                linewidth=1.1,
                markersize=2.8,
                color=color,
                label=seed_run["label"] if metric_idx == 0 else None,
            )

    for metric_idx, (ax, name) in enumerate(zip(axes_flat, metric_columns)):
        if name in baseline_metrics and np.isfinite(baseline_metrics[name]):
            style = _style_for(BASELINE_LABEL)
            ax.axhline(float(baseline_metrics[name]) * ERROR_SCALE, label=BASELINE_LABEL, **{k: v for k, v in style.items() if k not in {"marker", "markersize"}})
        ax.set_ylabel(_metric_ylabel(name), fontsize=ylabel_fontsize)
        if _is_bottom_row(metric_idx, len(metric_columns), cols):
            ax.set_xlabel("Generation")
        ax.set_yscale("log")
        _apply_axes_style(ax, minor_grid=True)
        _add_panel_label(ax, metric_idx)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    _legend_above(fig, handles, labels, ncol=min(max(len(seed_runs) + 1, 2), 3))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91), pad=0.35, w_pad=0.8, h_pad=0.55)
    _save_figure(fig, output_path)


def plot_diagnostics(summary: dict[str, Any], output_path: Path) -> None:
    generations = summary["generations"]
    fig, axes = plt.subplots(1, 2, figsize=THESIS_FIGSIZE_1X2, sharex=True)

    axes[0].plot(generations, summary["unique_fraction"], marker="o", markersize=2.8, linewidth=1.1, color="#0072B2", label="Unique")
    axes[0].plot(generations, 1.0 - summary["cache_hit_fraction"], marker="s", markersize=2.8, linewidth=1.1, color="#009E73", label="New eval.")
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Fraction")
    axes[0].set_ylim(0.0, 1.05)
    _apply_axes_style(axes[0])
    _add_panel_label(axes[0], 0)

    relative_gap = (
        summary["generation_mean_objective"] - summary["generation_best_objective"]
    ) / np.maximum(summary["generation_mean_objective"], 1.0e-12)
    axes[1].plot(generations, relative_gap, marker="o", markersize=2.8, linewidth=1.1, color="#0072B2", label="Rel. gap")
    axes[1].plot(generations, summary["failure_fraction"], marker="s", markersize=2.8, linewidth=1.1, color="#009E73", label="Failures")
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Fraction")
    axes[1].set_ylim(bottom=0.0)
    _apply_axes_style(axes[1])
    _add_panel_label(axes[1], 1)

    handles: list[Any] = []
    labels: list[str] = []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    _legend_above(fig, handles, labels, ncol=4)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90), pad=0.35, w_pad=0.8)
    _save_figure(fig, output_path)


def plot_metric_tradeoff_scatter(summary: dict[str, Any], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=THESIS_FIGSIZE_SINGLE)
    x = summary["generation_best_metrics"]["displacement_std_relative_error"]
    y = summary["generation_best_metrics"]["force_std_relative_error"]
    c = summary["generations"]
    scatter = ax.scatter(_scaled_error(x), _scaled_error(y), c=c, cmap="viridis", s=28, linewidths=0.0, rasterized=True)
    ax.plot(_scaled_error(x), _scaled_error(y), color="0.65", linewidth=0.8, alpha=0.8)
    ax.set_xlabel(_metric_ylabel("displacement_std_relative_error"))
    ax.set_ylabel(_metric_ylabel("force_std_relative_error"))
    _apply_axes_style(ax)
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Generation")
    colorbar.ax.tick_params(labelsize=TICK_FONT_SIZE)
    fig.tight_layout(pad=0.35)
    _save_figure(fig, output_path)


def plot_multi_seed_objective_evolution(seed_runs: list[dict[str, Any]], output_path: Path, loss_mode: str | None) -> None:
    fig, ax = plt.subplots(figsize=THESIS_FIGSIZE_SINGLE)
    scale = _objective_scale(loss_mode)
    for seed_run in seed_runs:
        summary = seed_run["summary"]
        generations = summary["generations"]
        ax.plot(
            generations,
            summary["best_so_far_objective"] * scale,
            marker="o",
            markersize=2.8,
            linewidth=1.25,
            label=seed_run["label"],
        )
        ax.plot(
            generations,
            summary["generation_best_objective"] * scale,
            linewidth=0.8,
            alpha=0.32,
            color=ax.lines[-1].get_color(),
        )

    ax.set_xlabel("Generation")
    ax.set_ylabel(_objective_ylabel(loss_mode))
    ax.set_yscale("log")
    _apply_axes_style(ax, minor_grid=True)
    handles, labels = ax.get_legend_handles_labels()
    _legend_above(fig, handles, labels, ncol=min(max(len(seed_runs), 2), 3))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91), pad=0.35)
    _save_figure(fig, output_path)


def plot_multi_seed_parameter_evolution(
    seed_runs: list[dict[str, Any]],
    output_path: Path,
    baseline_params: dict[str, float],
    bounds: dict[str, tuple[float, float]] | dict[str, list[float]],
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=THESIS_FIGSIZE_3X2, sharex=True)
    axes_flat = axes.reshape(-1)
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    ylabel_fontsize = _scaled_fontsize("axes.labelsize", PARAMETER_Y_AXIS_FONT_SCALE)

    for seed_idx, seed_run in enumerate(seed_runs):
        summary = seed_run["summary"]
        generations = summary["generations"]
        color = colors[seed_idx % len(colors)] if colors else None
        for parameter_idx, (ax, name) in enumerate(zip(axes_flat, PARAMETER_PLOT_ORDER)):
            ax.plot(
                generations,
                summary["best_so_far_params"][name],
                marker="o",
                linewidth=1.1,
                markersize=2.8,
                color=color,
                label=seed_run["label"] if parameter_idx == 0 else None,
            )

    for parameter_idx, (ax, name) in enumerate(zip(axes_flat, PARAMETER_PLOT_ORDER)):
        ylim_values = [seed_run["summary"]["best_so_far_params"][name] for seed_run in seed_runs]
        if name in baseline_params:
            style = _style_for(BASELINE_LABEL)
            baseline_value = float(baseline_params[name])
            ax.axhline(baseline_value, label=BASELINE_LABEL, **{k: v for k, v in style.items() if k not in {"marker", "markersize"}})
            ylim_values.append([baseline_value])
        _set_parameter_ylim_from_values(ax, ylim_values)
        ax.set_ylabel(PARAMETER_AXIS_LABELS.get(name, name), fontsize=ylabel_fontsize)
        _apply_axes_style(ax)
        _add_panel_label(ax, parameter_idx)

    axes_flat[-2].set_xlabel("Generation")
    axes_flat[-1].set_xlabel("Generation")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    _legend_above(fig, handles, labels, ncol=min(max(len(seed_runs) + 1, 2), 3))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91), pad=0.35, w_pad=0.7, h_pad=0.55)
    _save_figure(fig, output_path)


def plot_multi_seed_final_objectives(seed_runs: list[dict[str, Any]], output_path: Path, loss_mode: str | None) -> None:
    labels = [seed_run["label"] for seed_run in seed_runs]
    scale = _objective_scale(loss_mode)
    best_objectives = np.asarray([seed_run["best_objective"] for seed_run in seed_runs], dtype=float) * scale
    baseline_objectives = np.asarray([seed_run["baseline_objective"] for seed_run in seed_runs], dtype=float) * scale

    fig, ax = plt.subplots(figsize=THESIS_FIGSIZE_SINGLE)
    x = np.arange(len(labels), dtype=float)
    ax.bar(x, best_objectives, width=0.62, color="#009E73", label="Optimized")
    ax.scatter(x, baseline_objectives, marker="D", s=28, color="0.25", label=BASELINE_LABEL, zorder=3)
    ax.set_xticks(x, labels)
    ax.set_xlabel("Seed")
    ax.set_ylabel(_objective_ylabel(loss_mode))
    ax.set_yscale("log")
    _apply_axes_style(ax, minor_grid=True)
    handles, labels = ax.get_legend_handles_labels()
    _legend_above(fig, handles, labels, ncol=2)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91), pad=0.35)
    _save_figure(fig, output_path)


def plot_multi_seed_relative_improvement(seed_runs: list[dict[str, Any]], output_path: Path) -> None:
    labels = [seed_run["label"] for seed_run in seed_runs]
    relative_improvement = 100.0 * np.asarray([seed_run["relative_improvement"] for seed_run in seed_runs], dtype=float)

    fig, ax = plt.subplots(figsize=THESIS_FIGSIZE_SINGLE)
    x = np.arange(len(labels), dtype=float)
    ax.bar(x, relative_improvement, width=0.62, color="#009E73")
    ax.set_xticks(x, labels)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Relative improvement [%]")
    _apply_axes_style(ax)
    fig.tight_layout(pad=0.35)
    _save_figure(fig, output_path)


def plot_single_run(run_dir: Path, output_dir: Path) -> None:
    run_config = _load_json(run_dir / "run_config.json")
    best_result = _load_json(run_dir / "best_result.json")
    history_rows = _load_history(run_dir / "history.csv")
    summary = build_generation_summary(history_rows)
    loss_mode = str(run_config.get("loss_mode", "mae"))

    plot_objective_evolution(
        summary,
        _figure_path(output_dir, "objective_evolution"),
        baseline_objective=_to_float(best_result.get("baseline_objective")),
        loss_mode=loss_mode,
    )
    plot_parameter_evolution(
        summary,
        _figure_path(output_dir, "parameter_evolution"),
        baseline_params={key: _to_float(value) for key, value in run_config.get("baseline_params", {}).items()},
        bounds=run_config.get("bounds", {}),
    )
    plot_metric_evolution(summary, _figure_path(output_dir, "metric_evolution"))
    plot_diagnostics(summary, _figure_path(output_dir, "diagnostics"))
    plot_metric_tradeoff_scatter(summary, _figure_path(output_dir, "metric_tradeoff_scatter"))


def plot_multi_seed_runs(run_dir: Path, output_dir: Path) -> None:
    seed_run_dirs = _discover_seed_run_dirs(run_dir)
    if not seed_run_dirs:
        raise ValueError(f"No seed run directories found in {run_dir}.")

    seed_runs: list[dict[str, Any]] = []
    shared_baseline_params: dict[str, float] = {}
    shared_baseline_metrics: dict[str, float] = {}
    shared_bounds: dict[str, tuple[float, float]] | dict[str, list[float]] = {}
    shared_loss_mode: str | None = None
    for seed_run_dir in seed_run_dirs:
        run_config = _load_json(seed_run_dir / "run_config.json")
        best_result = _load_json(seed_run_dir / "best_result.json")
        history_rows = _load_history(seed_run_dir / "history.csv")
        summary = build_generation_summary(history_rows)
        if shared_loss_mode is None:
            shared_loss_mode = str(run_config.get("loss_mode", "mae"))
        if not shared_baseline_params:
            shared_baseline_params = {
                key: _to_float(value) for key, value in run_config.get("baseline_params", {}).items()
            }
        if not shared_baseline_metrics:
            baseline_row = summary.get("baseline_row") or {}
            shared_baseline_metrics = {name: _to_float(baseline_row.get(name)) for name in summary["metric_columns"]}
        if not shared_bounds:
            shared_bounds = run_config.get("bounds", {})
        random_seed_value = run_config.get("random_seed")
        label = _format_seed_label(random_seed_value, seed_run_dir.name)
        seed_runs.append(
            {
                "label": label,
                "summary": summary,
                "best_objective": _to_float(best_result.get("best_objective")),
                "baseline_objective": _to_float(best_result.get("baseline_objective")),
                "relative_improvement": _to_float(best_result.get("relative_improvement")),
            }
        )

    plot_multi_seed_objective_evolution(
        seed_runs,
        _figure_path(output_dir, "objective_evolution_all_seeds"),
        loss_mode=shared_loss_mode,
    )
    plot_multi_seed_parameter_evolution(
        seed_runs,
        _figure_path(output_dir, "parameter_evolution_all_seeds"),
        baseline_params=shared_baseline_params,
        bounds=shared_bounds,
    )
    plot_multi_seed_metric_evolution(
        seed_runs,
        _figure_path(output_dir, "metric_evolution_all_seeds"),
        baseline_metrics=shared_baseline_metrics,
    )
    plot_multi_seed_final_objectives(
        seed_runs,
        _figure_path(output_dir, "final_objective_by_seed"),
        loss_mode=shared_loss_mode,
    )
    plot_multi_seed_relative_improvement(seed_runs, _figure_path(output_dir, "relative_improvement_by_seed"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Vivana-TD GA optimization history.")
    parser.add_argument(
        "run_dir",
        nargs="?",
        default="vivana_cfd_data_pipeline/outputs/analysis/vivana_td_ga",
        help="Single run directory, or a parent directory containing multiple seed_* run directories.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory for PDF plots. Defaults to <run_dir>/plots.",
    )
    parser.add_argument(
        "--png-preview",
        action="store_true",
        help="Also write PNG previews next to the thesis-style PDF figures.",
    )
    return parser.parse_args()


def main() -> None:
    global SAVE_PNG_PREVIEW
    args = parse_args()
    SAVE_PNG_PREVIEW = bool(args.png_preview)
    _apply_thesis_rcparams()
    run_dir = Path(args.run_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else run_dir / "plots"
    seed_run_dirs = _discover_seed_run_dirs(run_dir)
    if seed_run_dirs:
        plot_multi_seed_runs(run_dir, output_dir)
        print(f"Wrote multi-seed plots to {output_dir}")
        return
    if _is_run_dir(run_dir):
        plot_single_run(run_dir, output_dir)
        print(f"Wrote plots to {output_dir}")
        return
    raise ValueError(f"{run_dir} is neither a single run directory nor a parent directory with seed runs.")


if __name__ == "__main__":
    main()
