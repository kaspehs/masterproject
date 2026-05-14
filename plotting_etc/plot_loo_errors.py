"""Plot LOO (leave-one-Ur-out) rollout errors as a function of reduced velocity.

Edit LOO_EXPERIMENTS to overlay multiple sets of LOO models on the same axes.
VIVANA-TD pure baseline is always shown as a dashed black line.
"""
from __future__ import annotations

import os
import re
import sys
import csv
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

_CACHE_ROOT = Path(tempfile.gettempdir()) / "masterproject_plot_cache"
(_CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
(_CACHE_ROOT / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from HNN_helper import (
    AGGREGATE_VALIDATION_ERROR_KEY,
    DISP_STD_REL_ERROR_KEY,
    DOMINANT_FREQ_REL_ERROR_KEY,
    FORCE_DOMINANT_FREQ_REL_ERROR_KEY,
    FORCE_STD_REL_ERROR_KEY,
    PHVIV,
    compute_validation_metrics,
    dominant_frequency,
    load_td_correction_trajectories,
    parse_config,
    resolve_td_correction_params,
    resolve_td_memory_config,
    resolve_td_n_memory_torch,
    structural_step_constant_force_torch,
    td_baseline_step_torch,
)
from methods.hnn.trainer import (
    _td_correction_state_rollout,
    _td_flow_feature_from_traj,
)

# ── configuration ──────────────────────────────────────────────────────────────

DATA_DIR = ROOT / "CFD_Data" / "npz_exports_td_burnin_trimmed_alltimeseries"

LOO_EXPERIMENTS: list[dict[str, Any]] = [
    {
        "label": "Force correction",
        "model_dir": ROOT / "models" / "mean" / "loo",
        "color": "tab:orange",
        "marker": "s",
    },
    {
        "label": "Frequency correction",
        "model_dir": ROOT / "models" / "fhat" / "loo",
        "color": "tab:blue",
        "marker": "o",
    },
    {
        "label": "Combined correction",
        "model_dir": ROOT / "models" / "combined" / "loo",
        "color": "tab:green",
        "marker": "^",
    },
]

OUTPUT_DIR = ROOT / "figs" / "loo_errors"
AGGREGATE_OUTPUT_BASENAME = "loo_aggregate_error"
COMPONENT_OUTPUT_BASENAME = "loo_component_errors"
SUMMARY_TABLE_PATH = OUTPUT_DIR / "loo_errors_summary.csv"
DETAIL_BY_UR_TABLE_PATH = OUTPUT_DIR / "loo_errors_by_ur.csv"
DETAIL_BY_CASE_TABLE_PATH = OUTPUT_DIR / "loo_errors_by_case.csv"
DPI = 300

X_LABEL = r"Reduced velocity $U_r$"
BASE_FONT_SIZE = 9.0
AXIS_LABEL_FONT_SIZE = 10.0
TITLE_FONT_SIZE = 10.0
TICK_FONT_SIZE = 9.0
LEGEND_FONT_SIZE = 9.0
Y_LABEL_FONT_SIZE = 11.0
Y_LABEL_ROTATION = 0
Y_LABEL_PAD = 11
Y_SCALE = "log"
GRID_ALPHA = 0.25
LINE_WIDTH = 1.6
MARKER_SIZE = 5
X_LIMIT_MARGIN = 0.35
BASELINE_LABEL = "Baseline VIVANA-TD"
BASELINE_COLOR = "0.35"
BASELINE_LINE_WIDTH = 1.4
AGGREGATE_FIGSIZE = (7.0, 3.2)
COMPONENT_FIGSIZE = (7.0, 8.0)
SAVE_PNG = True
SAVE_PDF = False
SHOW_FIGURE = False

plt.rcParams.update({
    "font.size": BASE_FONT_SIZE,
    "axes.labelsize": AXIS_LABEL_FONT_SIZE,
    "axes.titlesize": TITLE_FONT_SIZE,
    "xtick.labelsize": TICK_FONT_SIZE,
    "ytick.labelsize": TICK_FONT_SIZE,
    "legend.fontsize": LEGEND_FONT_SIZE,
})

# Downsampling factor applied to raw CFD trajectories before evaluation.
# Must match the reduction_factor used during training.
REDUCTION_FACTOR = 20

# Match the spectral notebook's Block 9a LOO entries by default.
# Use "final" to reproduce the previous behavior that preferred final/final.pt.
LOO_CHECKPOINT_PREFERENCE = "best_val"  # "best_val" | "final"

# Mapping: checkpoint suffix → (Ur float value, NPZ glob pattern)
UR_MAP: dict[str, tuple[float, str]] = {
    "ur2":   (2.0,  "comb_Ur2__*.npz"),
    "ur4":   (4.0,  "comb_Ur4__*.npz"),
    "ur5":   (5.0,  "comb_Ur5__*.npz"),
    "ur575": (5.75, "comb_Ur575__*.npz"),
    "ur7":   (7.0,  "comb_Ur7__*.npz"),
    "ur8":   (8.0,  "comb_Ur8__*.npz"),
    "ur10":  (10.0, "comb_Ur10__*.npz"),
}

METRICS: list[tuple[str, str]] = [
    (DISP_STD_REL_ERROR_KEY,            r"$\varepsilon_{\sigma}^{y}$"),
    (DOMINANT_FREQ_REL_ERROR_KEY,       r"$\varepsilon_{\omega}^{y}$"),
    (FORCE_STD_REL_ERROR_KEY,           r"$\varepsilon_{\sigma}^{F}$"),
    (FORCE_DOMINANT_FREQ_REL_ERROR_KEY, r"$\varepsilon_{\omega}^{F}$"),
    (AGGREGATE_VALIDATION_ERROR_KEY,    r"$\bar{\varepsilon}$"),
]

AGGREGATE_METRIC = (AGGREGATE_VALIDATION_ERROR_KEY, r"$\bar{\varepsilon}$")
COMPONENT_METRICS: list[tuple[str, str, str]] = [
    (DISP_STD_REL_ERROR_KEY,            r"$\varepsilon_{\sigma}^{y}$", "Displacement std error"),
    (DOMINANT_FREQ_REL_ERROR_KEY,       r"$\varepsilon_{\omega}^{y}$", "Displacement frequency error"),
    (FORCE_STD_REL_ERROR_KEY,           r"$\varepsilon_{\sigma}^{F}$", "Force std error"),
    (FORCE_DOMINANT_FREQ_REL_ERROR_KEY, r"$\varepsilon_{\omega}^{F}$", "Force frequency error"),
]

TABLE_METRIC_LABELS: dict[str, str] = {
    DISP_STD_REL_ERROR_KEY: "disp_std_rel_error",
    DOMINANT_FREQ_REL_ERROR_KEY: "disp_freq_rel_error",
    FORCE_STD_REL_ERROR_KEY: "force_std_rel_error",
    FORCE_DOMINANT_FREQ_REL_ERROR_KEY: "force_freq_rel_error",
    AGGREGATE_VALIDATION_ERROR_KEY: "aggregate_error",
}

_LOO_NAME_RE = re.compile(r"LOO(ur\d+)(?:_|/)", re.IGNORECASE)
_UR_ORDER = sorted(UR_MAP.keys(), key=lambda s: UR_MAP[s][0])

# ── helpers ────────────────────────────────────────────────────────────────────

def _find_loo_checkpoints(model_dir: Path) -> dict[str, Path]:
    """Return {ur_suffix: checkpoint_path} according to LOO_CHECKPOINT_PREFERENCE."""
    final_result: dict[str, Path] = {}
    final_root = model_dir / "final"
    for pt in sorted(final_root.glob("*/final.pt")):
        m = _LOO_NAME_RE.search(pt.as_posix())
        if m is None:
            continue
        suffix = m.group(1).lower()
        if suffix in UR_MAP:
            final_result[suffix] = pt

    best_result: dict[str, Path] = {}
    for pt in sorted(model_dir.glob("*.pt")):
        m = _LOO_NAME_RE.search(pt.name)
        if m is None:
            continue
        suffix = m.group(1).lower()
        if suffix in UR_MAP:
            best_result[suffix] = pt

    preference = str(LOO_CHECKPOINT_PREFERENCE).strip().lower()
    if preference not in {"best_val", "final"}:
        raise ValueError("LOO_CHECKPOINT_PREFERENCE must be 'best_val' or 'final'.")
    primary = best_result if preference == "best_val" else final_result
    fallback = final_result if preference == "best_val" else best_result
    result = dict(primary)
    for suffix, path in fallback.items():
        result.setdefault(suffix, path)
    return result


def _load_model(path: Path, device: torch.device) -> tuple[PHVIV, dict]:
    ckpt = torch.load(path, map_location=device)
    config = parse_config(ckpt["config"])
    model_cfg = asdict(config.model)
    arch_cfg = asdict(config.architecture)
    if "predict_sigma" in ckpt:
        model_cfg["use_stochastic_process_noise"] = bool(ckpt["predict_sigma"])
    if "correction_mode" in ckpt:
        model_cfg["correction_mode"] = ckpt["correction_mode"]
    if "td_force_input_source" in ckpt:
        model_cfg["use_td_force_input"] = ckpt["td_force_input_source"]
    elif "use_td_force_input" in ckpt:
        model_cfg["use_td_force_input"] = ckpt["use_td_force_input"]
    if "use_td_fhat_input" in ckpt:
        model_cfg["use_td_fhat_input"] = bool(ckpt["use_td_fhat_input"])
    if "input_configs" in ckpt:
        model_cfg["input_configs"] = ckpt["input_configs"]
    if "use_phi_input" in ckpt:
        model_cfg["use_phi_input"] = bool(ckpt["use_phi_input"])
    if "phi_input_source" in ckpt:
        model_cfg["phi_input_source"] = ckpt["phi_input_source"]
    if "use_sigma_inputs" in ckpt:
        model_cfg["use_sigma_inputs"] = bool(ckpt["use_sigma_inputs"])
    if "shared_td_correction_trunk" in ckpt:
        arch_cfg["shared_td_correction_trunk"] = bool(ckpt["shared_td_correction_trunk"])

    model, _ = PHVIV.from_config(
        dt=float(ckpt["dt"]),
        cfg=model_cfg,
        arch_cfg=arch_cfg,
        device=device,
    )
    if "correction_mode" in ckpt:
        setattr(model, "correction_mode", ckpt["correction_mode"])
    if "td_force_input_source" in ckpt:
        setattr(model, "td_force_input_source", ckpt["td_force_input_source"])
    if "fhat_bound_multiplier" in ckpt:
        setattr(model, "fhat_bound_multiplier", float(ckpt["fhat_bound_multiplier"]))
    if "fhat_correction_bounds" in ckpt:
        setattr(model, "fhat_correction_bounds", ckpt["fhat_correction_bounds"])
    if "force_zero_output" in ckpt:
        setattr(model, "force_zero_output", bool(ckpt["force_zero_output"]))
    if "random_phase_training" in ckpt:
        setattr(model, "random_phase_training", bool(ckpt["random_phase_training"]))

    state = {
        k.removeprefix("_orig_mod.").removeprefix("module."): v
        for k, v in ckpt["model_state"].items()
    }
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, ckpt


def _load_trajs(
    suffix: str,
    reduction_factor: int = 1,
    *,
    ur_source: str = "stored",
    td_params: dict[str, float] | None = None,
    td_memory_cfg: dict[str, Any] | None = None,
) -> list[dict[str, np.ndarray]]:
    _, glob_pat = UR_MAP[suffix]
    paths = sorted(DATA_DIR.glob(glob_pat))
    if not paths:
        raise FileNotFoundError(f"No files matching '{glob_pat}' in {DATA_DIR}")
    return load_td_correction_trajectories(
        paths=paths,
        reduce_time=reduction_factor > 1,
        reduction_factor=reduction_factor,
        ur_source=ur_source,
        td_params=td_params,
        td_memory_cfg=td_memory_cfg,
    )


def _avg_metrics(per_traj: list[dict[str, float]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, _ in METRICS:
        vals = [m[key] for m in per_traj if key in m and np.isfinite(float(m[key]))]
        result[key] = float(np.mean(vals)) if vals else float("nan")
    return result


def _effective_reduced_velocity(trajs: list[dict], *, diameter: float) -> float:
    """Mean effective-mass reduced velocity for a held-out U_r group."""
    values: list[float] = []
    for traj in trajs:
        flow_speed = np.asarray(traj["flow_speed"], dtype=float).reshape(-1)
        finite_flow = flow_speed[np.isfinite(flow_speed)]
        if finite_flow.size == 0:
            continue
        stiffness = float(np.asarray(traj["stiffness_n_m"], dtype=float).reshape(()))
        effective_mass = float(np.asarray(traj["effective_mass_kg"], dtype=float).reshape(()))
        if not (
            np.isfinite(stiffness)
            and np.isfinite(effective_mass)
            and np.isfinite(diameter)
            and stiffness > 0.0
            and effective_mass > 0.0
            and diameter > 0.0
        ):
            continue
        omega_n = float(np.sqrt(stiffness / effective_mass))
        f_n = omega_n / (2.0 * np.pi)
        values.append(float(np.mean(finite_flow) / (f_n * diameter)))
    if not values:
        raise ValueError("Could not compute a finite effective reduced velocity from the loaded trajectories.")
    return float(np.mean(values))


def _trajectory_effective_reduced_velocity(traj: dict, *, diameter: float) -> float:
    flow_speed = np.asarray(traj["flow_speed"], dtype=float).reshape(-1)
    finite_flow = flow_speed[np.isfinite(flow_speed)]
    if finite_flow.size == 0:
        return float("nan")
    stiffness = float(np.asarray(traj["stiffness_n_m"], dtype=float).reshape(()))
    effective_mass = float(np.asarray(traj["effective_mass_kg"], dtype=float).reshape(()))
    if not (
        np.isfinite(stiffness)
        and np.isfinite(effective_mass)
        and np.isfinite(diameter)
        and stiffness > 0.0
        and effective_mass > 0.0
        and diameter > 0.0
    ):
        return float("nan")
    natural_frequency_hz = float(np.sqrt(stiffness / effective_mass) / (2.0 * np.pi))
    return float(np.mean(finite_flow) / (natural_frequency_hz * diameter))


def _finite_mean(values: list[float]) -> float:
    finite = [float(v) for v in values if np.isfinite(float(v))]
    return float(np.mean(finite)) if finite else float("nan")


def _format_table_float(value: float) -> str:
    value = float(value)
    if not np.isfinite(value):
        return "nan"
    if value == 0.0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e3:
        return f"{value:.3e}"
    return f"{value:.4g}"


def _format_ur_tick(value: float) -> str:
    return f"{float(value):.2f}".rstrip("0").rstrip(".")


def _as_scalar(value: Any, default: float = float("nan")) -> float:
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return float(default)
    finite = arr[np.isfinite(arr)]
    return float(finite[0]) if finite.size else float(default)


def _dominant_frequency_or_nan(values: np.ndarray, dt: float) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size < 2 or not np.isfinite(float(dt)) or float(dt) <= 0.0:
        return float("nan")
    try:
        return float(dominant_frequency(arr, float(dt)))
    except Exception:
        return float("nan")


def _rollout_diagnostic_values(
    *,
    traj: dict,
    y_pred: np.ndarray,
    force_pred: np.ndarray,
    dt: float,
    diameter: float,
) -> dict[str, float]:
    y_true = np.asarray(traj["y"], dtype=float).reshape(-1)
    force_true = np.asarray(traj["force_per_m"], dtype=float).reshape(-1)
    y_pred_arr = np.asarray(y_pred, dtype=float).reshape(-1)
    force_pred_arr = np.asarray(force_pred, dtype=float).reshape(-1)
    min_len_y = min(y_true.size, y_pred_arr.size)
    min_len_force = min(force_true.size, force_pred_arr.size)
    y_true_aligned = y_true[:min_len_y]
    y_pred_aligned = y_pred_arr[:min_len_y]
    force_true_aligned = force_true[:min_len_force]
    force_pred_aligned = force_pred_arr[:min_len_force]

    stiffness = _as_scalar(traj.get("stiffness_n_m"))
    effective_mass = _as_scalar(traj.get("effective_mass_kg"))
    natural_frequency_hz = (
        float(np.sqrt(stiffness / effective_mass) / (2.0 * np.pi))
        if np.isfinite(stiffness) and np.isfinite(effective_mass) and stiffness > 0.0 and effective_mass > 0.0
        else float("nan")
    )

    disp_freq_true_hz = _dominant_frequency_or_nan(y_true_aligned, dt)
    disp_freq_pred_hz = _dominant_frequency_or_nan(y_pred_aligned, dt)
    force_freq_true_hz = _dominant_frequency_or_nan(force_true_aligned, dt)
    force_freq_pred_hz = _dominant_frequency_or_nan(force_pred_aligned, dt)

    def _freq_ratio(freq_hz: float) -> float:
        return (
            float(freq_hz / natural_frequency_hz)
            if np.isfinite(freq_hz) and np.isfinite(natural_frequency_hz) and natural_frequency_hz > 0.0
            else float("nan")
        )

    return {
        "n_samples": int(min_len_y),
        "n_force_samples": int(min_len_force),
        "dt_s": float(dt),
        "duration_s": float(dt) * float(max(0, min_len_y - 1)),
        "diameter_m": float(diameter),
        "natural_frequency_hz": float(natural_frequency_hz),
        "disp_std_true": float(np.std(y_true_aligned)) if min_len_y else float("nan"),
        "disp_std_pred": float(np.std(y_pred_aligned)) if min_len_y else float("nan"),
        "force_std_true": float(np.std(force_true_aligned)) if min_len_force else float("nan"),
        "force_std_pred": float(np.std(force_pred_aligned)) if min_len_force else float("nan"),
        "disp_dominant_freq_true_hz": disp_freq_true_hz,
        "disp_dominant_freq_pred_hz": disp_freq_pred_hz,
        "force_dominant_freq_true_hz": force_freq_true_hz,
        "force_dominant_freq_pred_hz": force_freq_pred_hz,
        "disp_dominant_freq_ratio_true": _freq_ratio(disp_freq_true_hz),
        "disp_dominant_freq_ratio_pred": _freq_ratio(disp_freq_pred_hz),
        "force_dominant_freq_ratio_true": _freq_ratio(force_freq_true_hz),
        "force_dominant_freq_ratio_pred": _freq_ratio(force_freq_pred_hz),
        "trajectory_effective_ur": _trajectory_effective_reduced_velocity(traj, diameter=float(diameter)),
        "trajectory_model_ur_mean": float(np.mean(np.asarray(traj["ur"], dtype=float))),
        "trajectory_stored_ur_mean": float(np.mean(np.asarray(traj["ur_stored"], dtype=float))),
        "trajectory_label_ur_mean": (
            float(np.mean(np.asarray(traj["ur_label"], dtype=float)))
            if traj.get("ur_label") is not None
            else float("nan")
        ),
    }


def _rows_for_series(
    *,
    method: str,
    ur_values: list[float],
    metric_arrays: dict[str, list[float]],
    held_out_suffixes: list[str] | None = None,
    held_out_ur_labels: list[float] | None = None,
    checkpoint_paths: list[Path | None] | None = None,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for idx, ur_value in enumerate(ur_values):
        checkpoint_path = None if checkpoint_paths is None or idx >= len(checkpoint_paths) else checkpoint_paths[idx]
        row: dict[str, float | str] = {
            "method": method,
            "held_out_suffix": "" if held_out_suffixes is None or idx >= len(held_out_suffixes) else held_out_suffixes[idx],
            "held_out_ur_label": (
                float("nan") if held_out_ur_labels is None or idx >= len(held_out_ur_labels) else float(held_out_ur_labels[idx])
            ),
            "held_out_ur_effective_group": float(ur_value),
            "checkpoint_path": "" if checkpoint_path is None else str(checkpoint_path),
            "checkpoint_name": "" if checkpoint_path is None else checkpoint_path.name,
            "checkpoint_preference": str(LOO_CHECKPOINT_PREFERENCE),
        }
        for metric_key, _ in METRICS:
            values = metric_arrays.get(metric_key, [])
            row[TABLE_METRIC_LABELS[metric_key]] = (
                float(values[idx]) if idx < len(values) else float("nan")
            )
        rows.append(row)
    return rows


def _rows_for_case_details(
    *,
    method: str,
    held_out_suffix: str,
    held_out_ur_label: float,
    held_out_ur_effective_group: float,
    checkpoint_path: Path | None,
    metrics_by_case: list[dict[str, float | str]],
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for idx, case_metrics in enumerate(metrics_by_case):
        row: dict[str, float | str] = {
            "method": method,
            "held_out_suffix": held_out_suffix,
            "held_out_ur_label": float(held_out_ur_label),
            "held_out_ur_effective_group": float(held_out_ur_effective_group),
            "case_index": int(idx),
            "checkpoint_path": "" if checkpoint_path is None else str(checkpoint_path),
            "checkpoint_name": "" if checkpoint_path is None else checkpoint_path.name,
            "checkpoint_preference": str(LOO_CHECKPOINT_PREFERENCE),
        }
        row.update(case_metrics)
        rows.append(row)
    return rows


def _summary_rows(detail_rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    method_order: list[str] = []
    for row in detail_rows:
        method = str(row["method"])
        if method not in method_order:
            method_order.append(method)

    rows: list[dict[str, float | str]] = []
    for method in method_order:
        method_rows = [row for row in detail_rows if str(row["method"]) == method]
        out: dict[str, float | str] = {
            "method": method,
            "num_ur": len(method_rows),
        }
        for metric_key, _ in METRICS:
            label = TABLE_METRIC_LABELS[metric_key]
            out[label] = _finite_mean([float(row[label]) for row in method_rows])
        rows.append(out)
    return rows


def _write_csv(rows: list[dict[str, float | str]], output_path: Path) -> None:
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_summary_table(rows: list[dict[str, float | str]]) -> None:
    if not rows:
        return
    metric_labels = [TABLE_METRIC_LABELS[key] for key, _ in METRICS]
    headers = ["method", "num_ur", *metric_labels]
    rendered_rows = [
        [
            str(row["method"]),
            str(row["num_ur"]),
            *[_format_table_float(float(row[label])) for label in metric_labels],
        ]
        for row in rows
    ]
    widths = [
        max(len(header), *(len(row[idx]) for row in rendered_rows))
        for idx, header in enumerate(headers)
    ]
    print("\nLOO mean errors across held-out U_r values")
    print("  ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rendered_rows:
        print("  ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)))


def _plot_metric_curves(
    ax: plt.Axes,
    *,
    metric_key: str,
    y_label: str,
    title: str | None,
    td_ur_values: list[float],
    td_metric_arrays: dict[str, list[float]],
    experiment_results: list[dict],
    all_ur_ticks: list[float],
    show_xlabel: bool,
) -> None:
    if td_ur_values:
        ax.plot(
            td_ur_values,
            td_metric_arrays[metric_key],
            color=BASELINE_COLOR,
            linestyle="--",
            marker="x",
            linewidth=BASELINE_LINE_WIDTH,
            markersize=MARKER_SIZE,
            label=BASELINE_LABEL,
            zorder=2,
        )
    for res in experiment_results:
        ax.plot(
            res["ur_values"],
            res["metrics"][metric_key],
            color=res["color"],
            marker=res["marker"],
            linestyle="-",
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            label=res["label"],
            zorder=3,
        )

    if title:
        ax.set_title(title, fontsize=TITLE_FONT_SIZE)
    ax.set_ylabel(
        y_label,
        fontsize=Y_LABEL_FONT_SIZE,
        rotation=Y_LABEL_ROTATION,
        labelpad=Y_LABEL_PAD,
        va="center",
    )
    ax.set_yscale(Y_SCALE)
    ax.set_xticks(all_ur_ticks)
    tick_labels = [_format_ur_tick(tick) for tick in all_ur_ticks]
    if all_ur_ticks:
        ax.set_xlim(float(all_ur_ticks[0]) - X_LIMIT_MARGIN, float(all_ur_ticks[-1]) + X_LIMIT_MARGIN)
    ax.grid(alpha=GRID_ALPHA, which="both")
    if show_xlabel:
        ax.set_xlabel(X_LABEL)
        ax.set_xticklabels(tick_labels)
    else:
        ax.set_xticklabels([])


def _add_shared_legend(fig: plt.Figure, axes: list[plt.Axes]) -> bool:
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper center",
                ncol=min(4, len(labels)),
                frameon=False,
                fontsize=LEGEND_FONT_SIZE,
                handlelength=2.0,
                columnspacing=1.6,
                markerscale=0.9,
            )
            return True
    return False


def _save_plot(fig: plt.Figure, basename: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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


def _plot_aggregate_error(
    *,
    td_ur_values: list[float],
    td_metric_arrays: dict[str, list[float]],
    experiment_results: list[dict],
    all_ur_ticks: list[float],
) -> None:
    metric_key, y_label = AGGREGATE_METRIC
    fig, ax = plt.subplots(1, 1, figsize=AGGREGATE_FIGSIZE)
    _plot_metric_curves(
        ax,
        metric_key=metric_key,
        y_label=y_label,
        title=None,
        td_ur_values=td_ur_values,
        td_metric_arrays=td_metric_arrays,
        experiment_results=experiment_results,
        all_ur_ticks=all_ur_ticks,
        show_xlabel=True,
    )
    has_legend = _add_shared_legend(fig, [ax])
    top = 0.86 if has_legend else 1.0
    fig.tight_layout(rect=(0.0, 0.0, 1.0, top))
    _save_plot(fig, AGGREGATE_OUTPUT_BASENAME)


def _plot_component_errors(
    *,
    td_ur_values: list[float],
    td_metric_arrays: dict[str, list[float]],
    experiment_results: list[dict],
    all_ur_ticks: list[float],
) -> None:
    fig, axes = plt.subplots(len(COMPONENT_METRICS), 1, figsize=COMPONENT_FIGSIZE, sharex=True)
    axes_arr = list(np.atleast_1d(axes))
    for idx, (ax, (metric_key, y_label, title)) in enumerate(zip(axes_arr, COMPONENT_METRICS)):
        _plot_metric_curves(
            ax,
            metric_key=metric_key,
            y_label=y_label,
            title=title,
            td_ur_values=td_ur_values,
            td_metric_arrays=td_metric_arrays,
            experiment_results=experiment_results,
            all_ur_ticks=all_ur_ticks,
            show_xlabel=idx == len(axes_arr) - 1,
        )
    has_legend = _add_shared_legend(fig, axes_arr)
    top = 0.92 if has_legend else 1.0
    fig.tight_layout(rect=(0.0, 0.0, 1.0, top))
    _save_plot(fig, COMPONENT_OUTPUT_BASENAME)


def _eval_loo_model(
    model: PHVIV,
    ckpt: dict,
    trajs: list[dict],
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, float | str]]]:
    hnn_cfg = dict(ckpt["config"].get("hnn", {}))
    td_params = resolve_td_correction_params(hnn_cfg)
    td_memory_cfg = resolve_td_memory_config(hnn_cfg)
    mass_key = "dry_mass_kg" if str(hnn_cfg.get("td_mass_source", "dry")) == "dry" else "effective_mass_kg"
    mean_active = bool(ckpt.get("mean_active", False))
    fhat_active = bool(ckpt.get("fhat_active", False))
    predict_sigma = bool(ckpt.get("predict_sigma", False))
    td_force_src = str(ckpt.get("td_force_input_source", "none"))
    fhat_bound_mult = float(ckpt.get("fhat_bound_multiplier", 1.0))
    input_scaling = str(getattr(model, "input_scaling_mode", "current"))

    per_traj: list[dict[str, float]] = []
    detail_rows: list[dict[str, float | str]] = []
    with torch.no_grad():
        for traj in trajs:
            mass = float(np.asarray(traj[mass_key]).reshape(()))
            damping = float(np.asarray(traj["damping_c"]).reshape(()))
            stiffness = float(np.asarray(traj["stiffness_n_m"]).reshape(()))
            t_np = np.asarray(traj["t"], dtype=float)
            dt = float(t_np[1] - t_np[0])

            y_t = torch.from_numpy(np.ascontiguousarray(traj["y"])).float().to(device)
            dy_t = torch.from_numpy(np.ascontiguousarray(traj["dy"])).float().to(device)
            td_ctx = torch.from_numpy(np.ascontiguousarray(traj["td_context"])).float().to(device)

            ur_np = _td_flow_feature_from_traj(traj, input_scaling_mode=input_scaling, diameter=float(model.D))
            ur_t = torch.from_numpy(np.ascontiguousarray(ur_np)).float().unsqueeze(1).to(device)

            z0 = torch.cat([y_t[0:1].unsqueeze(1), dy_t[0:1].unsqueeze(1) * mass], dim=1)
            steps = int(y_t.shape[0] - 1)

            z_seq, force_seq, _, _, _ = _td_correction_state_rollout(
                model=model,
                z0=z0,
                ur0=ur_t[0:1],
                td_context0=td_ctx[0:1],
                steps=steps,
                dt=dt,
                structural_mass=torch.full((1, 1), mass, dtype=torch.float32, device=device),
                damping_c=torch.full((1, 1), damping, dtype=torch.float32, device=device),
                stiffness=torch.full((1, 1), stiffness, dtype=torch.float32, device=device),
                td_params=td_params,
                td_memory_cfg=td_memory_cfg,
                mean_active=mean_active,
                sigma_active=predict_sigma,
                fhat_active=fhat_active,
                td_force_input_source=td_force_src,
                fhat_bound_multiplier=fhat_bound_mult,
                rollout_stochastic=False,
                rollout_seed=0,
            )

            y_pred = z_seq[0, :, 0].cpu().numpy()
            vel_pred = (z_seq[0, :, 1] / mass).cpu().numpy()
            force_np = force_seq[0, :, 0].cpu().numpy()
            if force_np.size:
                force_np = np.concatenate([force_np[:1], force_np], axis=0)

            rollout = {
                "y_norm": y_pred / float(model.D),
                "p_norm": vel_pred / (np.sqrt(stiffness / mass) * float(model.D)),
                "force_total": force_np,
            }
            metrics = compute_validation_metrics(
                model=model,
                y_data_t=y_t,
                val_vel=dy_t,
                reduced_velocity=ur_t[:, 0],
                m_eff=mass,
                dt=dt,
                t=t_np,
                y_data_raw=np.asarray(traj["y"], dtype=float),
                force_data=np.asarray(traj["force_per_m"], dtype=float),
                D=float(model.D),
                k=stiffness,
                device=device,
                rollout=rollout,
            )
            per_traj.append(metrics)
            detail = {
                "trajectory_name": str(traj.get("name", "")),
                "source_name": str(traj.get("source_name", "")),
                "reduction_factor": int(_as_scalar(traj.get("reduction_factor"), default=1.0)),
                "reduction_offset": int(_as_scalar(traj.get("reduction_offset"), default=0.0)),
                "mass_source": str(hnn_cfg.get("td_mass_source", "dry")),
                "mass_kg": float(mass),
                "damping_c": float(damping),
                "stiffness_n_m": float(stiffness),
                "td_memory_mode": str(td_memory_cfg.get("mode", "")),
                "mean_active": int(mean_active),
                "fhat_active": int(fhat_active),
                "predict_sigma": int(predict_sigma),
                "td_force_input_source": td_force_src,
            }
            detail.update(_rollout_diagnostic_values(
                traj=traj,
                y_pred=y_pred,
                force_pred=force_np,
                dt=dt,
                diameter=float(model.D),
            ))
            for key, _ in METRICS:
                detail[TABLE_METRIC_LABELS[key]] = float(metrics.get(key, float("nan")))
            detail_rows.append(detail)

    return _avg_metrics(per_traj), detail_rows


def _eval_td_baseline(
    trajs: list[dict],
    td_params: dict[str, float],
    td_memory_cfg: dict[str, Any],
    mass_key: str,
    rho: float,
    diameter: float,
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, float | str]]]:
    per_traj: list[dict[str, float]] = []
    detail_rows: list[dict[str, float | str]] = []
    with torch.no_grad():
        for traj in trajs:
            mass = float(np.asarray(traj[mass_key]).reshape(()))
            damping = float(np.asarray(traj["damping_c"]).reshape(()))
            stiffness = float(np.asarray(traj["stiffness_n_m"]).reshape(()))
            t_np = np.asarray(traj["t"], dtype=float)
            dt = float(t_np[1] - t_np[0])

            y_t = torch.from_numpy(np.ascontiguousarray(traj["y"])).float().to(device)
            dy_t = torch.from_numpy(np.ascontiguousarray(traj["dy"])).float().to(device)
            ur_t = torch.from_numpy(np.ascontiguousarray(traj["ur"])).float().unsqueeze(1).to(device)
            td_ctx = torch.from_numpy(np.ascontiguousarray(traj["td_context"])).float().to(device)

            mass_t = torch.full((1, 1), mass, dtype=torch.float32, device=device)
            damp_t = torch.full((1, 1), damping, dtype=torch.float32, device=device)
            stiff_t = torch.full((1, 1), stiffness, dtype=torch.float32, device=device)

            z_cur = torch.cat([y_t[0:1].unsqueeze(1), dy_t[0:1].unsqueeze(1) * mass], dim=1)
            td_ctx_cur = td_ctx[0:1].clone()
            y_hist = [float(y_t[0].item())]
            f_hist: list[float] = []

            for _ in range(int(y_t.shape[0] - 1)):
                velocity = z_cur[:, 1:2] / mass
                step_params = dict(td_params)
                step_params["n_memory"] = resolve_td_n_memory_torch(
                    td_params,
                    dt=dt,
                    flow_speed=td_ctx_cur[:, 4:5],
                    diameter=diameter,
                    memory_cfg=td_memory_cfg,
                )
                td_force_next, td_ctx_next = td_baseline_step_torch(
                    velocity=velocity,
                    acceleration=td_ctx_cur[:, 0:1],
                    td_context=td_ctx_cur,
                    dt=dt,
                    rho=rho,
                    diameter=diameter,
                    params=step_params,
                )
                y_next, v_next, a_next = structural_step_constant_force_torch(
                    y=z_cur[:, 0:1],
                    velocity=velocity,
                    force=td_force_next,
                    dt=dt,
                    mass=mass_t,
                    damping_c=damp_t,
                    stiffness=stiff_t,
                )
                z_cur = torch.cat([y_next, v_next * mass], dim=1)
                td_ctx_cur = td_ctx_next.clone()
                td_ctx_cur[:, 0:1] = a_next
                y_hist.append(float(y_next[0, 0].item()))
                f_hist.append(float(td_force_next[0, 0].item()))

            y_pred = np.array(y_hist, dtype=float)
            force_np = np.array(f_hist, dtype=float)
            if force_np.size:
                force_np = np.concatenate([force_np[:1], force_np], axis=0)

            rollout = {
                "y_norm": y_pred / diameter,
                "p_norm": np.zeros_like(y_pred),
                "force_total": force_np,
            }
            metrics = compute_validation_metrics(
                model=None,
                y_data_t=y_t,
                val_vel=dy_t,
                reduced_velocity=ur_t[:, 0],
                m_eff=mass,
                dt=dt,
                t=t_np,
                y_data_raw=np.asarray(traj["y"], dtype=float),
                force_data=np.asarray(traj["force_per_m"], dtype=float),
                D=diameter,
                k=stiffness,
                device=device,
                rollout=rollout,
            )
            per_traj.append(metrics)
            detail = {
                "trajectory_name": str(traj.get("name", "")),
                "source_name": str(traj.get("source_name", "")),
                "reduction_factor": int(_as_scalar(traj.get("reduction_factor"), default=1.0)),
                "reduction_offset": int(_as_scalar(traj.get("reduction_offset"), default=0.0)),
                "mass_source": "dry" if mass_key == "dry_mass_kg" else "effective",
                "mass_kg": float(mass),
                "damping_c": float(damping),
                "stiffness_n_m": float(stiffness),
                "td_memory_mode": str(td_memory_cfg.get("mode", "")),
                "mean_active": 0,
                "fhat_active": 0,
                "predict_sigma": 0,
                "td_force_input_source": "none",
            }
            detail.update(_rollout_diagnostic_values(
                traj=traj,
                y_pred=y_pred,
                force_pred=force_np,
                dt=dt,
                diameter=float(diameter),
            ))
            for key, _ in METRICS:
                detail[TABLE_METRIC_LABELS[key]] = float(metrics.get(key, float("nan")))
            detail_rows.append(detail)

    return _avg_metrics(per_traj), detail_rows


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    device = torch.device("cpu")

    # Physical params for baseline (extracted from first loaded model)
    ref_td_params: dict[str, float] | None = None
    ref_td_memory_cfg: dict[str, Any] | None = None
    ref_mass_key: str | None = None
    ref_rho: float | None = None
    ref_diameter: float | None = None
    baseline_trajs: dict[str, list[dict]] = {}
    effective_ur_by_suffix: dict[str, float] = {}

    # Evaluate LOO experiments
    experiment_results: list[dict] = []
    detailed_case_rows: list[dict[str, float | str]] = []
    for exp in LOO_EXPERIMENTS:
        label = str(exp["label"])
        model_dir = Path(exp["model_dir"])
        print(f"\nEvaluating '{label}' from {model_dir.name}/")
        checkpoints = _find_loo_checkpoints(model_dir)
        if not checkpoints:
            print(f"  No LOO checkpoints found.")
            continue

        ur_values: list[float] = []
        held_out_suffixes: list[str] = []
        held_out_ur_labels: list[float] = []
        checkpoint_paths: list[Path | None] = []
        metric_arrays: dict[str, list[float]] = {key: [] for key, _ in METRICS}

        for suffix in _UR_ORDER:
            if suffix not in checkpoints:
                continue
            ur_val = UR_MAP[suffix][0]
            print(f"  Ur={ur_val:g} ...", end=" ", flush=True)
            model, ckpt = _load_model(checkpoints[suffix], device)
            hnn_cfg = dict(ckpt["config"].get("hnn", {}))
            td_params = resolve_td_correction_params(hnn_cfg)
            td_memory_cfg = resolve_td_memory_config(hnn_cfg)
            td_mass_source = str(hnn_cfg.get("td_mass_source", "dry")).strip().lower()
            if td_mass_source not in {"dry", "effective"}:
                raise ValueError(f"Unsupported hnn.td_mass_source={td_mass_source!r}; expected 'dry' or 'effective'.")

            if ref_td_params is None:
                ref_td_params = td_params
                ref_td_memory_cfg = td_memory_cfg
                ref_mass_key = "dry_mass_kg" if td_mass_source == "dry" else "effective_mass_kg"
                ref_rho = float(getattr(parse_config(ckpt["config"]).model, "rho", 1000.0))
                ref_diameter = float(model.D)

            trajs = _load_trajs(
                suffix,
                reduction_factor=REDUCTION_FACTOR,
                ur_source=td_mass_source,
                td_params=td_params,
                td_memory_cfg=td_memory_cfg,
            )
            baseline_trajs[suffix] = trajs
            ur_eff = _effective_reduced_velocity(trajs, diameter=float(model.D))
            effective_ur_by_suffix[suffix] = ur_eff

            metrics, case_metrics = _eval_loo_model(model, ckpt, trajs, device)
            ur_values.append(ur_eff)
            held_out_suffixes.append(suffix)
            held_out_ur_labels.append(ur_val)
            checkpoint_paths.append(checkpoints[suffix])
            for key, _ in METRICS:
                metric_arrays[key].append(metrics.get(key, float("nan")))
            detailed_case_rows.extend(
                _rows_for_case_details(
                    method=label,
                    held_out_suffix=suffix,
                    held_out_ur_label=ur_val,
                    held_out_ur_effective_group=ur_eff,
                    checkpoint_path=checkpoints[suffix],
                    metrics_by_case=case_metrics,
                )
            )
            print(f"done (U_r,eff={ur_eff:.4g})")

        experiment_results.append({
            "label": label,
            "color": exp.get("color", "steelblue"),
            "marker": exp.get("marker", "o"),
            "ur_values": ur_values,
            "held_out_suffixes": held_out_suffixes,
            "held_out_ur_labels": held_out_ur_labels,
            "checkpoint_paths": checkpoint_paths,
            "metrics": metric_arrays,
        })

    # Evaluate VIVANA-TD baseline — one point per unique (suffix, rf) combination
    # Use mean across all experiment rfs to get one baseline value per Ur
    td_ur_values: list[float] = []
    td_held_out_suffixes: list[str] = []
    td_held_out_ur_labels: list[float] = []
    td_metric_arrays: dict[str, list[float]] = {key: [] for key, _ in METRICS}
    if (
        ref_td_params is not None
        and ref_td_memory_cfg is not None
        and ref_mass_key is not None
        and ref_rho is not None
        and ref_diameter is not None
    ):
        print("\nEvaluating VIVANA-TD baseline...")
        for suffix in _UR_ORDER:
            if suffix not in baseline_trajs:
                continue
            ur_val = effective_ur_by_suffix.get(suffix)
            if ur_val is None:
                ur_val = _effective_reduced_velocity(baseline_trajs[suffix], diameter=ref_diameter)
            print(f"  Ur={ur_val:g} ...", end=" ", flush=True)
            metrics, case_metrics = _eval_td_baseline(
                baseline_trajs[suffix],
                ref_td_params,
                ref_td_memory_cfg,
                ref_mass_key,
                ref_rho,
                ref_diameter,
                device,
            )
            td_ur_values.append(ur_val)
            td_held_out_suffixes.append(suffix)
            td_held_out_ur_labels.append(UR_MAP[suffix][0])
            for key, _ in METRICS:
                td_metric_arrays[key].append(metrics.get(key, float("nan")))
            detailed_case_rows.extend(
                _rows_for_case_details(
                    method=BASELINE_LABEL,
                    held_out_suffix=suffix,
                    held_out_ur_label=UR_MAP[suffix][0],
                    held_out_ur_effective_group=ur_val,
                    checkpoint_path=None,
                    metrics_by_case=case_metrics,
                )
            )
            print("done")
    else:
        print("\nSkipping VIVANA-TD baseline (no model loaded to extract physical params).")

    detail_rows: list[dict[str, float | str]] = []
    if td_ur_values:
        detail_rows.extend(
            _rows_for_series(
                method=BASELINE_LABEL,
                ur_values=td_ur_values,
                metric_arrays=td_metric_arrays,
                held_out_suffixes=td_held_out_suffixes,
                held_out_ur_labels=td_held_out_ur_labels,
                checkpoint_paths=[None] * len(td_ur_values),
            )
        )
    for res in experiment_results:
        detail_rows.extend(
            _rows_for_series(
                method=str(res["label"]),
                ur_values=list(res["ur_values"]),
                metric_arrays=res["metrics"],
                held_out_suffixes=list(res["held_out_suffixes"]),
                held_out_ur_labels=list(res["held_out_ur_labels"]),
                checkpoint_paths=list(res["checkpoint_paths"]),
            )
        )
    summary_rows = _summary_rows(detail_rows)
    _write_csv(summary_rows, SUMMARY_TABLE_PATH)
    _write_csv(detail_rows, DETAIL_BY_UR_TABLE_PATH)
    _write_csv(detailed_case_rows, DETAIL_BY_CASE_TABLE_PATH)
    _print_summary_table(summary_rows)
    if detail_rows:
        print(f"\nSaved summary → {SUMMARY_TABLE_PATH}")
        print(f"Saved per-U_r details → {DETAIL_BY_UR_TABLE_PATH}")
    if detailed_case_rows:
        print(f"Saved per-case details → {DETAIL_BY_CASE_TABLE_PATH}")

    all_ur_ticks = sorted(effective_ur_by_suffix.values()) or sorted({v for v, _ in UR_MAP.values()})
    _plot_aggregate_error(
        td_ur_values=td_ur_values,
        td_metric_arrays=td_metric_arrays,
        experiment_results=experiment_results,
        all_ur_ticks=all_ur_ticks,
    )
    _plot_component_errors(
        td_ur_values=td_ur_values,
        td_metric_arrays=td_metric_arrays,
        experiment_results=experiment_results,
        all_ur_ticks=all_ur_ticks,
    )


if __name__ == "__main__":
    main()
