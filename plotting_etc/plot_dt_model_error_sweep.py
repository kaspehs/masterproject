"""plot_dt_model_error_sweep.py

Sweep over timestep sizes for Vivana-TD and trained Section 7 models,
evaluated on CFD NPZ data. The simulation at the smallest downsampling factor
is the per-model reference; absolute relative errors at coarser timesteps
are plotted in compact thesis-style panels.

Edit the CONFIGURATION section to set data paths, model checkpoints, and
sweep factors before running.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

# ── matplotlib isolation ─────────────────────────────────────────────────────
_CACHE_ROOT = Path(tempfile.gettempdir()) / "masterproject_plot_cache"
(_CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
(_CACHE_ROOT / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit these values before running.
# ═══════════════════════════════════════════════════════════════════════════════

# Directory containing CFD NPZ files; all *.npz files are loaded directly.
DATA_DIR: Path = ROOT / "vivana_cfd_data_pipeline" / "generated" / "td_burnin_trimmed_alltimeseries"

# Use one representative time series per reduced velocity.  When several NPZ
# segments exist for the same Ur, keep the lowest perturbation variant by name
# (e.g. __1Hydro before __2Hydro / __3Hydro).
ONE_SERIES_PER_REDUCED_VELOCITY: bool = True
EXCLUDED_SERIES_NAME_PREFIXES: tuple[str, ...] = ()

# Set DT_SWEEP_MODE to "downsampling_factor" for an integer factor sweep or
# "dt" for an absolute dt sweep (values in seconds).  In both modes the
# smallest value defines the reference (error = 0).
DT_SWEEP_MODE: str = "downsampling_factor"  # "downsampling_factor" or "dt"

# Used when DT_SWEEP_MODE = "downsampling_factor".
# Provide ≥ 2 distinct positive integers in any order — they are sorted internally.
#DOWNSAMPLING_FACTORS: tuple[int, ...] = (1, 2, 5, 10, 20, 30, 40)
DOWNSAMPLING_FACTORS: tuple[int, ...] = (1, 2, 5, 10, 20, 40)
#DOWNSAMPLING_FACTORS: tuple[int, ...] = (10, 20, 30, 40)

# Used when DT_SWEEP_MODE = "dt".
# Values are in seconds; the smallest is the reference dt.
# For each series the closest integer downsampling factor is used:
#   factor = max(1, round(dt_target / dt_base))
DT_VALUES: tuple[float, ...] = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5)

# Correction-model checkpoints and display labels.
# "path" may be relative to the repo root or absolute.
CORRECTION_MODEL_SPECS: list[dict[str, Any]] = [
    {
        "label": "Force correction",
        "path": ROOT / "models" / "mean" / "multi_seed" / "best_seed.pt",
    },
    {
        "label": "Frequency correction",
        "path": ROOT / "models" / "fhat" / "multi_seed" / "best_seed.pt",
    },
    {
        "label": "Combined correction",
        "path": ROOT / "models" / "combined" / "multi_seed" / "best_seed.pt",
    },
    {
        "label": "Standalone model",
        "path": ROOT / "models" / "latentrnn" / "best_seed.pt",
    },
]
# Vivana-TD pure baseline appearance.
VIVANA_TD_LABEL: str = "VIVANA-TD baseline"

# Structural mass source used in rollouts: "dry" or "effective".
# Vivana-TD force already includes added mass (Fca), so use dry structural mass.
MASS_SOURCE: str = "dry"

# TD memory handling used for all rollouts in this dt sweep.  Keep tau fixed
# relative to the TD reference period and resolve n_memory=tau/dt at each step.
TD_MEMORY_TAU_SPEC: str = "tau_over_tref:4"
TD_FORCE_PHASE_CONVENTION: str = "current"  # "current" matches old vforce_CF; "next" is the previous model-stepwise default.

# The Standalone model encoder was trained on trajectories reduced by this
# factor. Keep the latent history on this grid even when sweeping rollout dt.
LATENT_RNN_ENCODER_DOWNSAMPLING_FACTOR: int = 20

# Output directory (created if missing).
OUTPUT_DIR: Path = ROOT / "figs" / "dt_model_error_sweep"

# Figure resolution.
FIGURE_DPI: int = 300
SAVE_PNG_PREVIEW: bool = False

# The NPZ exports are already burn-in/trimmed segments.  Use their first stored
# CFD/TD state as the rollout initial condition.
ROLLOUT_START_SECONDS: float = 0.0
METRIC_SKIP_SECONDS: float = 0.0
METRIC_WINDOW_AFTER_CONVERGENCE_SECONDS: float = 100.0
MAX_ROLLOUT_SECONDS: float = 600.0
ROLLOUT_DTYPE: torch.dtype = torch.float64

# Metric windows can optionally be trimmed to the first steady-state onset.
# Keep this disabled when evaluating on the already burn-in-trimmed training
# NPZ segments: the stored segment length/dt defines the evaluation window.
USE_STEADY_STATE_CONVERGENCE_WINDOW: bool = False
STEADY_STATE_N_CYCLES: int = 10
STEADY_STATE_AMP_REL_TOL: float = 0.05

# Override each selected NPZ's stored initial state with the same low-perturbation
# IC, while keeping its parameters and time grid. Disabled for training-segment
# evaluation so each rollout starts from the stored reference state.
USE_SYNTHETIC_INITIAL_CONDITION: bool = False
SYNTHETIC_DISPLACEMENT_OVER_D: float = 0.1
SYNTHETIC_VELOCITY: float = 0.0
SYNTHETIC_THETA: float = 0.0
SYNTHETIC_SIGMA_DISPLACEMENT_OVER_D: float = 0.1

# Also write a companion plot where the x-axis is reduced velocity and the
# timestep/downsampling setting is held fixed.
GENERATE_REDUCED_VELOCITY_PLOT: bool = True
REDUCED_VELOCITY_SWEEP_VALUE: float | None = 20.0
REDUCED_VELOCITY_SOURCE: str = "effective"  # "effective" or "stored"

# ═══════════════════════════════════════════════════════════════════════════════

# 2×2 subplot specs in panel order top-left → bottom-right.
# Each entry: (signal_key, metric, title_unused).
# signal_key : "displacement" or "force"  (keys in the rollout output dict)
# metric     : "dominant_freq" or "std"
SUBPLOT_SPECS: tuple[tuple[str, str, str], ...] = (
    ("displacement", "dominant_freq", ""),
    ("displacement", "std",           ""),
    ("force",        "dominant_freq", ""),
    ("force",        "std",           ""),
)

# Y-axis symbols for relative timestep errors, plotted in percent.
_YLABEL_SYMBOLS: dict[tuple[str, str], str] = {
    ("force",        "std"):           r"$\varepsilon^F_{\sigma}$ [%]",
    ("force",        "dominant_freq"): r"$\varepsilon^F_{\omega}$ [%]",
    ("displacement", "std"):           r"$\varepsilon^y_{\sigma}$ [%]",
    ("displacement", "dominant_freq"): r"$\varepsilon^y_{\omega}$ [%]",
}

MODEL_STYLES: dict[str, dict[str, Any]] = {
    "VIVANA-TD baseline": {
        "color": "0.45",
        "linestyle": "--",
        "linewidth": 1.35,
        "marker": "o",
    },
    "Force correction": {
        "color": "#0072B2",
        "linestyle": "-",
        "linewidth": 1.35,
        "marker": "o",
    },
    "Frequency correction": {
        "color": "#D55E00",
        "linestyle": "-",
        "linewidth": 1.35,
        "marker": "o",
    },
    "Combined correction": {
        "color": "#009E73",
        "linestyle": "-",
        "linewidth": 1.35,
        "marker": "o",
    },
    "Standalone model": {
        "color": "#882255",
        "linestyle": "-",
        "linewidth": 1.35,
        "marker": "o",
    },
}

THESIS_FIGSIZE_2X2: tuple[float, float] = (5.85, 4.8)
BASE_FONT_SIZE: int = 8
AXIS_LABEL_FONT_SIZE: int = 9
TICK_FONT_SIZE: int = 8
LEGEND_FONT_SIZE: int = 8
PANEL_LABEL_FONT_SIZE: int = 9
SPINE_COLOR: str = "0.65"
SPINE_LINE_WIDTH: float = 0.6
GRID_COLOR: str = "0.88"
GRID_MINOR_COLOR: str = "0.94"
ERROR_SCALE: float = 100.0

# ─────────────────────────────────────────────────────────────────────────────
# Project imports (after ROOT is on sys.path).
# ─────────────────────────────────────────────────────────────────────────────

from tqdm import tqdm  # noqa: E402

from training.training_utils import dominant_frequency  # noqa: E402
from vivana_cfd_data_pipeline.scripts.training_npz_loader import load_series as _load_npz_series  # noqa: E402
from vivana_cfd_data_pipeline.helpers.model_rollouts import (  # noqa: E402
    LoadedTrainingModel,
    _find_steady_state_onset,
    load_trained_model_sources,
    simulate_checkpoint_series_rollout,
    simulate_vivana_td_stepwise,
)

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_sweep_x_values() -> tuple[float, ...]:
    """Return sorted sweep values (factors or dt seconds) from configuration."""
    if DT_SWEEP_MODE == "downsampling_factor":
        unique = sorted({int(f) for f in DOWNSAMPLING_FACTORS})
        if len(unique) < 2:
            raise SystemExit("DOWNSAMPLING_FACTORS must contain at least two distinct positive integers.")
        if any(f <= 0 for f in unique):
            raise SystemExit("All DOWNSAMPLING_FACTORS must be positive integers.")
        return tuple(float(f) for f in unique)
    elif DT_SWEEP_MODE == "dt":
        unique = sorted({float(v) for v in DT_VALUES})
        if len(unique) < 2:
            raise SystemExit("DT_VALUES must contain at least two distinct positive values.")
        if any(v <= 0 for v in unique):
            raise SystemExit("All DT_VALUES must be positive.")
        return tuple(unique)
    else:
        raise SystemExit(
            f"DT_SWEEP_MODE must be 'downsampling_factor' or 'dt', got {DT_SWEEP_MODE!r}."
        )


def _factor_for_x(base_dt: float, x_value: float) -> int:
    """Resolve the integer downsampling factor for a given sweep x-value."""
    if DT_SWEEP_MODE == "downsampling_factor":
        return max(1, int(round(x_value)))
    return max(1, int(round(x_value / base_dt)))


def _dt_for_x(base_dt: float, x_value: float) -> float:
    if DT_SWEEP_MODE == "downsampling_factor":
        return float(base_dt) * float(_factor_for_x(base_dt, x_value))
    return float(x_value)


def _sweep_axis_label() -> str:
    if DT_SWEEP_MODE == "downsampling_factor":
        return "Downsampling factor"
    return r"$\Delta t$ [s]"


def _format_x_tick(value: float) -> str:
    if DT_SWEEP_MODE == "downsampling_factor":
        return str(int(round(value)))
    return f"{value:g}"


def _output_token(value: float) -> str:
    return _format_x_tick(value).replace(".", "p").replace("-", "m")


def _sweep_descriptor(value: float) -> str:
    if DT_SWEEP_MODE == "downsampling_factor":
        return f"downsampling factor {_format_x_tick(value)}"
    return f"dt = {_format_x_tick(value)} s"


def _series_reduced_velocity(series: dict[str, Any]) -> float:
    if REDUCED_VELOCITY_SOURCE == "effective":
        preferred = series.get("ur_effective")
        fallback = series.get("ur")
    elif REDUCED_VELOCITY_SOURCE == "stored":
        preferred = series.get("ur")
        fallback = series.get("ur_effective")
    else:
        raise SystemExit(
            f"REDUCED_VELOCITY_SOURCE must be 'effective' or 'stored', got {REDUCED_VELOCITY_SOURCE!r}."
        )

    for raw in (preferred, fallback):
        if raw is None:
            continue
        arr = np.asarray(raw, dtype=float).reshape(-1)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            return float(np.mean(finite))
    return float("nan")


def _format_ur_tick(value: float) -> str:
    return f"{float(value):.3g}"


def _series_dt(series: dict[str, Any]) -> float:
    time = np.asarray(series["time"], dtype=float).reshape(-1)
    return float(np.median(np.diff(time))) if time.size >= 2 else float("nan")


def _series_rollout_mass(series: dict[str, Any]) -> float:
    if MASS_SOURCE == "dry":
        return float(series["dry_mass"])
    if MASS_SOURCE == "effective":
        return float(series["effective_mass"])
    raise SystemExit(f"MASS_SOURCE must be 'dry' or 'effective', got {MASS_SOURCE!r}.")


def _series_natural_period(series: dict[str, Any]) -> float:
    stiffness = float(series["stiffness"])
    mass = _series_rollout_mass(series)
    if not (np.isfinite(stiffness) and stiffness > 0.0 and np.isfinite(mass) and mass > 0.0):
        return float("nan")
    omega_n = float(np.sqrt(stiffness / mass))
    return float(2.0 * np.pi / omega_n) if omega_n > 0.0 else float("nan")


def _downsample_series(series: dict[str, Any], factor: int) -> dict[str, Any]:
    """Return a shallow copy of series with every `factor`-th sample (index 0 is always kept)."""
    if factor <= 1:
        return dict(series)
    time = np.asarray(series["time"], dtype=float).reshape(-1)
    idx = np.arange(0, time.size, factor, dtype=int)
    if idx.size < 2:
        raise ValueError(
            f"Series '{series.get('name', '<unknown>')}' is too short to "
            f"downsample by factor {factor} (only {idx.size} sample(s) remain)."
        )
    time_aligned_keys = (
        "time", "displacement", "velocity", "acceleration",
        "force_total", "force_per_m", "force_td_stored", "td_context",
    )
    reduced = dict(series)
    for key in time_aligned_keys:
        arr_raw = reduced.get(key)
        if arr_raw is None:
            continue
        arr = np.asarray(arr_raw)
        if arr.ndim >= 1 and arr.shape[0] == time.size:
            reduced[key] = arr[idx].copy()
    for key in ("ur", "ur_effective"):
        ur_raw = reduced.get(key)
        if isinstance(ur_raw, np.ndarray) and ur_raw.ndim >= 1 and ur_raw.shape[0] == time.size:
            reduced[key] = ur_raw[idx].copy()
    return reduced


def _rollout_series_at_dt(series: dict[str, Any], dt: float) -> dict[str, Any]:
    """Return a generated rollout shell using `series` parameters and first state."""
    dt_value = float(dt)
    if not (np.isfinite(dt_value) and dt_value > 0.0):
        raise ValueError(f"Need positive finite rollout dt, got {dt!r}.")
    max_seconds = float(MAX_ROLLOUT_SECONDS)
    n = max(2, int(np.floor(max_seconds / dt_value)) + 1)
    source_time = np.asarray(series["time"], dtype=float).reshape(-1)
    t0 = float(source_time[0]) if source_time.size else 0.0
    time = t0 + np.arange(n, dtype=float) * dt_value

    out = dict(series)
    out["time"] = time
    for key in ("displacement", "velocity", "acceleration", "force_total", "force_per_m", "force_td_stored"):
        arr = np.asarray(series[key], dtype=float).reshape(-1)
        fill = float(arr[0]) if arr.size else 0.0
        out[key] = np.full((n,), fill, dtype=float)
    ctx0 = np.asarray(series["td_context"], dtype=float).reshape(-1, 5)[0]
    out["td_context"] = np.repeat(ctx0.reshape(1, 5), n, axis=0)
    for key in ("ur", "ur_effective"):
        raw = series.get(key)
        if raw is None:
            continue
        arr = np.asarray(raw, dtype=float).reshape(-1)
        if arr.size:
            out[key] = np.full((n,), float(arr[0]), dtype=float)
    return out


def _trim_series_relative_start(series: dict[str, Any], start_seconds: float) -> dict[str, Any]:
    """Return a shallow copy starting `start_seconds` after this series begins."""
    if start_seconds <= 0.0:
        return dict(series)
    time = np.asarray(series["time"], dtype=float).reshape(-1)
    if time.size < 2:
        raise ValueError(f"Series '{series.get('name', '<unknown>')}' is too short to trim.")
    start_time = float(time[0]) + float(start_seconds)
    start_idx = int(np.searchsorted(time, start_time, side="left"))
    if time.size - start_idx < 2:
        raise ValueError(
            f"Series '{series.get('name', '<unknown>')}' has fewer than two samples after "
            f"{start_seconds:g} s rollout-start trim."
        )
    time_aligned_keys = (
        "time", "displacement", "velocity", "acceleration",
        "force_total", "force_per_m", "force_td_stored", "td_context",
    )
    trimmed = dict(series)
    for key in time_aligned_keys:
        arr_raw = trimmed.get(key)
        if arr_raw is None:
            continue
        arr = np.asarray(arr_raw)
        if arr.ndim >= 1 and arr.shape[0] == time.size:
            trimmed[key] = arr[start_idx:].copy()
    for key in ("ur", "ur_effective"):
        arr_raw = trimmed.get(key)
        if isinstance(arr_raw, np.ndarray) and arr_raw.ndim >= 1 and arr_raw.shape[0] == time.size:
            trimmed[key] = arr_raw[start_idx:].copy()
    return trimmed


def _wrap_angle(angle: float) -> float:
    return float(np.arctan2(np.sin(float(angle)), np.cos(float(angle))))


def _apply_synthetic_initial_condition(series: dict[str, Any]) -> dict[str, Any]:
    """Set a common low-perturbation rollout IC while preserving case parameters/time grid."""
    if not USE_SYNTHETIC_INITIAL_CONDITION:
        return dict(series)

    out = dict(series)
    diameter = float(out["diameter"])
    stiffness = float(out["stiffness"])
    mass = _series_rollout_mass(out)
    if not (np.isfinite(diameter) and diameter > 0.0):
        raise ValueError(f"Series '{series.get('name', '<unknown>')}' has invalid diameter.")
    if not (np.isfinite(stiffness) and stiffness > 0.0 and np.isfinite(mass) and mass > 0.0):
        raise ValueError(f"Series '{series.get('name', '<unknown>')}' has invalid stiffness/mass.")

    omega_n = float(np.sqrt(stiffness / mass))
    y0 = float(SYNTHETIC_DISPLACEMENT_OVER_D) * diameter
    dy0 = float(SYNTHETIC_VELOCITY)
    ddy0 = -omega_n * omega_n * y0
    sig_dy0 = max(float(SYNTHETIC_SIGMA_DISPLACEMENT_OVER_D) * diameter * omega_n, 1.0e-12)
    sig_ddy0 = max(float(SYNTHETIC_SIGMA_DISPLACEMENT_OVER_D) * diameter * omega_n * omega_n, 1.0e-12)

    flow_speed = float(np.asarray(out["td_context"], dtype=float).reshape(-1, 5)[0, 4])
    speed_mag = float(np.sqrt(max(flow_speed * flow_speed + dy0 * dy0, 1.0e-12)))
    projection = flow_speed / speed_mag
    dy_r = dy0 * projection
    ddy_r = ddy0 * projection
    phi_dy0 = float(np.arctan2(-ddy_r / sig_ddy0, dy_r / sig_dy0))
    phi_vy0 = _wrap_angle(phi_dy0 - float(SYNTHETIC_THETA))

    for key, value in (
        ("displacement", y0),
        ("velocity", dy0),
        ("acceleration", ddy0),
        ("force_per_m", 0.0),
        ("force_total", 0.0),
        ("force_td_stored", 0.0),
    ):
        arr_raw = out.get(key)
        if arr_raw is None:
            continue
        arr = np.asarray(arr_raw, dtype=float).copy()
        if arr.ndim >= 1 and arr.shape[0] >= 1:
            arr.reshape(-1)[0] = float(value)
            out[key] = arr

    ctx = np.asarray(out["td_context"], dtype=float).copy()
    if ctx.ndim != 2 or ctx.shape[0] < 1 or ctx.shape[1] < 5:
        raise ValueError(f"Series '{series.get('name', '<unknown>')}' has invalid td_context shape.")
    ctx[0, 0] = ddy0
    ctx[0, 1] = phi_vy0
    ctx[0, 2] = sig_dy0
    ctx[0, 3] = sig_ddy0
    out["td_context"] = ctx
    out["synthetic_initial_condition"] = {
        "y0": y0,
        "dy0": dy0,
        "ddy0": ddy0,
        "theta0": float(SYNTHETIC_THETA),
        "phi_vy0": phi_vy0,
        "sig_dy0": sig_dy0,
        "sig_ddy0": sig_ddy0,
        "omega_n": omega_n,
    }
    return out


def _select_one_series_per_reduced_velocity(all_series: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not ONE_SERIES_PER_REDUCED_VELOCITY:
        return list(all_series)

    def _variant_rank(series: dict[str, Any]) -> tuple[int, str]:
        name = str(series.get("name", ""))
        marker = "__"
        hydro = "Hydro"
        variant = 1
        if marker in name and name.endswith(hydro):
            tail = name.rsplit(marker, 1)[1]
            token = tail[: -len(hydro)]
            if token:
                try:
                    variant = int(token)
                except ValueError:
                    variant = 1
        return variant, name

    grouped: dict[float, dict[str, Any]] = {}
    for series in all_series:
        ur = _series_reduced_velocity(series)
        if not np.isfinite(ur):
            key = float(len(grouped))
        else:
            key = float(round(ur, 6))
        current = grouped.get(key)
        if current is None:
            grouped[key] = series
            continue
        if _variant_rank(series) < _variant_rank(current):
            grouped[key] = series
    return [grouped[key] for key in sorted(grouped)]


def _run_all_models(
    series: dict[str, Any],
    sources: list[LoadedTrainingModel],
    td_params: dict[str, float],
    *,
    pbar: "tqdm | None",
    dtype: torch.dtype = ROLLOUT_DTYPE,
    latent_encoder_series: dict[str, Any] | None = None,
) -> dict[str, dict[str, np.ndarray] | None]:
    """Run Vivana-TD and all correction models on `series`.

    Returns a dict mapping each model label to {"displacement": ..., "force": ...},
    or None if that model's rollout failed.  Updates `pbar` by one per model when provided.
    """
    results: dict[str, dict[str, np.ndarray] | None] = {}

    # Vivana-TD baseline
    try:
        out = simulate_vivana_td_stepwise(
            series,
            td_params=td_params,
            mass_source=MASS_SOURCE,
            td_memory_tau_s=TD_MEMORY_TAU_SPEC,
            dtype=dtype,
            force_phase_convention=TD_FORCE_PHASE_CONVENTION,
        )
        results[VIVANA_TD_LABEL] = {
            "displacement": np.asarray(out["displacement_td"], dtype=float),
            "force":        np.asarray(out["force_td"],        dtype=float),
        }
    except Exception as exc:
        results[VIVANA_TD_LABEL] = None
        tqdm.write(f"  [warn] Vivana-TD failed for '{series.get('name', '?')}': {exc}")
    if pbar is not None:
        pbar.update(1)

    # Correction models
    for source in sources:
        try:
            out = simulate_checkpoint_series_rollout(
                source,
                series,
                mass_source=MASS_SOURCE,
                td_memory_tau_s=TD_MEMORY_TAU_SPEC,
                dtype=dtype,
                force_phase_convention=TD_FORCE_PHASE_CONVENTION,
                latent_encoder_series=(
                    latent_encoder_series
                    if getattr(source, "kind", "") == "latent_rnn"
                    else None
                ),
            )
            results[source.label] = {
                "displacement": np.asarray(out["displacement"], dtype=float),
                "force":        np.asarray(out["force"],        dtype=float),
                "evaluation_start_idx": np.asarray([int(out.get("evaluation_start_idx", 0))], dtype=int),
            }
        except Exception as exc:
            results[source.label] = None
            tqdm.write(f"  [warn] {source.label} failed for '{series.get('name', '?')}': {exc}")
        if pbar is not None:
            pbar.update(1)

    return results


def _rollout_steady_state_onset(
    result: dict[str, np.ndarray] | None,
    *,
    dt: float,
    period_s: float,
) -> int | None:
    if not USE_STEADY_STATE_CONVERGENCE_WINDOW or result is None:
        return 0
    if not (np.isfinite(dt) and dt > 0.0 and np.isfinite(period_s) and period_s > 0.0):
        return None
    displacement = np.asarray(result.get("displacement", []), dtype=float).reshape(-1)
    if displacement.size < 2:
        return None
    values = displacement
    samples_per_cycle = max(1, int(round(float(period_s) / max(float(dt), 1.0e-12))))
    window = max(int(STEADY_STATE_N_CYCLES) * samples_per_cycle, 4)
    step = max(1, samples_per_cycle // 2)
    if values.size < 2 * window + 1:
        return None
    for index in range(0, values.size - 2 * window, step):
        segment_1 = values[index : index + window]
        segment_2 = values[index + window : index + 2 * window]
        amp_1 = float(np.std(segment_1))
        amp_2 = float(np.std(segment_2))
        if amp_1 < 1.0e-8 and amp_2 < 1.0e-8:
            return int(index)
        if amp_1 > 1.0e-8 and abs(amp_2 - amp_1) / amp_1 < float(STEADY_STATE_AMP_REL_TOL):
            return int(index)
    return None


def _metric_window(
    result: dict[str, np.ndarray] | None,
    *,
    dt: float,
    period_s: float,
    label: str,
    series_name: str,
    ur_value: float,
    x_label: str,
) -> tuple[int, int] | None:
    if result is None:
        return None
    n = np.asarray(result.get("displacement", []), dtype=float).reshape(-1).size
    if n < 2:
        return None
    eval_start = int(np.asarray(result.get("evaluation_start_idx", [0]), dtype=int).reshape(-1)[0])
    eval_start = max(0, min(eval_start, n - 2))
    if not USE_STEADY_STATE_CONVERGENCE_WINDOW:
        return eval_start, int(n)

    onset = _rollout_steady_state_onset(result, dt=dt, period_s=period_s)
    if onset is None:
        tqdm.write(
            f"  [warn] No convergence for {series_name} (Ur={float(ur_value):.4g}) / "
            f"{label} / {x_label} within {MAX_ROLLOUT_SECONDS:g} s."
        )
        return None
    onset = max(int(onset), eval_start)
    window_samples = max(2, int(round(float(METRIC_WINDOW_AFTER_CONVERGENCE_SECONDS) / float(dt))))
    end = int(onset) + window_samples
    if end > n:
        tqdm.write(
            f"  [warn] {series_name} (Ur={float(ur_value):.4g}) / {label} / {x_label} converged too late for "
            f"{METRIC_WINDOW_AFTER_CONVERGENCE_SECONDS:g} s metric window."
        )
        return None
    return int(onset), int(end)


def _abs_rel_error(
    pred: np.ndarray,
    ref: np.ndarray,
    metric: str,
    *,
    dt_pred: float,
    dt_ref: float,
    pred_window: tuple[int, int] | None = None,
    ref_window: tuple[int, int] | None = None,
) -> float:
    """Return absolute relative error (fraction, not percent) between `pred` and `ref` for the given metric."""
    pred_arr = np.asarray(pred, dtype=float)
    ref_arr  = np.asarray(ref,  dtype=float)
    if pred_window is None or ref_window is None:
        return float("nan")
    pred_start, pred_end = pred_window
    ref_start, ref_end = ref_window
    pred_arr = pred_arr[max(0, pred_start) : min(int(pred_end), pred_arr.size)]
    ref_arr = ref_arr[max(0, ref_start) : min(int(ref_end), ref_arr.size)]
    if pred_arr.size < 2 or ref_arr.size < 2:
        return float("nan")

    if metric == "std":
        std_ref = float(np.std(ref_arr))
        if not (np.isfinite(std_ref) and std_ref > 0.0):
            return float("nan")
        return abs(float(np.std(pred_arr)) - std_ref) / std_ref

    if metric == "dominant_freq":
        f_pred = dominant_frequency(pred_arr, dt_pred)
        f_ref  = dominant_frequency(ref_arr,  dt_ref)
        if not (np.isfinite(f_ref) and f_ref > 0.0 and np.isfinite(f_pred)):
            return float("nan")
        return abs(f_pred - f_ref) / f_ref

    raise ValueError(f"Unknown metric: {metric!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Main sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep(
    all_series: list[dict[str, Any]],
    sources: list[LoadedTrainingModel],
    td_params: dict[str, float],
    x_values: tuple[float, ...],
) -> tuple[
    dict[str, dict[tuple[str, str], dict[str, list[float]]]],
    dict[str, dict[tuple[str, str], list[list[tuple[float, float]]]]],
    list[str],
]:
    """Run the dt sweep and aggregate errors across all series.

    Returns:
        aggregate  — {label: {(signal, metric): {"mean": [...], "max": [...]}}}
        per_series — {label: {(signal, metric): [[(ur, err), ...] per x]}}
        model_labels — ordered list [Vivana-TD, corr1, corr2, corr3]
    """
    model_labels = [VIVANA_TD_LABEL] + [s.label for s in sources]
    n_models = len(model_labels)

    prepared_series: list[dict[str, Any]] = []
    for series in all_series:
        try:
            trimmed = _trim_series_relative_start(series, ROLLOUT_START_SECONDS)
            prepared_series.append(_apply_synthetic_initial_condition(trimmed))
        except Exception as exc:
            tqdm.write(f"  [warn] Skipping '{series.get('name', '?')}': {exc}")
    if not prepared_series:
        raise SystemExit("No series remain after applying the rollout-start trim.")

    total_ticks = len(prepared_series) * len(x_values) * n_models

    # collected[label][(sig, metric)][xi] = [per-series errors at x_values[xi]]
    collected: dict[str, dict[tuple[str, str], list[list[float]]]] = {
        label: {
            (sig, metric): [[] for _ in x_values]
            for sig, metric, _ in SUBPLOT_SPECS
        }
        for label in model_labels
    }
    per_series: dict[str, dict[tuple[str, str], list[list[tuple[float, float]]]]] = {
        label: {
            (sig, metric): [[] for _ in x_values]
            for sig, metric, _ in SUBPLOT_SPECS
        }
        for label in model_labels
    }

    with tqdm(total=total_ticks, desc="Sweep", unit="run") as pbar:
        for series in prepared_series:
            name = str(series.get("name", "<unknown>"))
            pbar.set_postfix_str(f"series={name}")
            ur = _series_reduced_velocity(series)
            base_dt = _series_dt(series)
            period_s = _series_natural_period(series)
            latent_encoder_series = _downsample_series(series, LATENT_RNN_ENCODER_DOWNSAMPLING_FACTOR)

            # ── Reference at the finest x-value ───────────────────────────
            # Use the stored trimmed NPZ series itself: native dt, native
            # length, and its first sample as the rollout initial condition.
            factor_gt = _factor_for_x(base_dt, x_values[0])
            gt_series = _downsample_series(series, factor_gt)
            dt_gt = _series_dt(gt_series)
            gt_results = _run_all_models(
                gt_series,
                sources,
                td_params,
                pbar=None,
                dtype=ROLLOUT_DTYPE,
                latent_encoder_series=latent_encoder_series,
            )
            gt_windows = {
                label: _metric_window(
                    gt_results.get(label),
                    dt=dt_gt,
                    period_s=period_s,
                    label=label,
                    series_name=name,
                    ur_value=ur,
                    x_label=f"x={_format_x_tick(x_values[0])}",
                )
                for label in model_labels
            }
            pbar.update(n_models)

            # Reference x: comparing each model to itself -> 0 when converged.
            for sig, metric, _ in SUBPLOT_SPECS:
                for label in model_labels:
                    gt = gt_results.get(label)
                    err = (
                        _abs_rel_error(
                            gt[sig],
                            gt[sig],
                            metric,
                            dt_pred=dt_gt,
                            dt_ref=dt_gt,
                            pred_window=gt_windows[label],
                            ref_window=gt_windows[label],
                        )
                        if gt is not None else float("nan")
                    )
                    collected[label][(sig, metric)][0].append(err)
                    per_series[label][(sig, metric)][0].append((ur, err))

            # ── Coarser x-values ──────────────────────────────────────────
            for xi in range(1, len(x_values)):
                pbar.set_postfix_str(f"series={name}, x={_format_x_tick(x_values[xi])}")
                factor_ds = _factor_for_x(base_dt, x_values[xi])
                ds_series = _downsample_series(series, factor_ds)
                dt_ds = _series_dt(ds_series)
                pred_results = _run_all_models(
                    ds_series,
                    sources,
                    td_params,
                    pbar=pbar,
                    dtype=ROLLOUT_DTYPE,
                    latent_encoder_series=latent_encoder_series,
                )
                pred_windows = {
                    label: _metric_window(
                        pred_results.get(label),
                        dt=dt_ds,
                        period_s=period_s,
                        label=label,
                        series_name=name,
                        ur_value=ur,
                        x_label=f"x={_format_x_tick(x_values[xi])}",
                    )
                    for label in model_labels
                }

                for sig, metric, _ in SUBPLOT_SPECS:
                    for label in model_labels:
                        gt   = gt_results.get(label)
                        pred = pred_results.get(label)
                        err = (
                            _abs_rel_error(
                                pred[sig],
                                gt[sig],
                                metric,
                                dt_pred=dt_ds,
                                dt_ref=dt_gt,
                                pred_window=pred_windows[label],
                                ref_window=gt_windows[label],
                            )
                            if (gt is not None and pred is not None) else float("nan")
                        )
                        collected[label][(sig, metric)][xi].append(err)
                        per_series[label][(sig, metric)][xi].append((ur, err))

    # Aggregate: mean and max across series for every (label, xi, signal, metric)
    aggregate: dict[str, dict[tuple[str, str], dict[str, list[float]]]] = {}
    for label in model_labels:
        aggregate[label] = {}
        for sig, metric, _ in SUBPLOT_SPECS:
            means: list[float] = []
            maxes: list[float] = []
            for xi in range(len(x_values)):
                finite_errors = [
                    e for e in collected[label][(sig, metric)][xi]
                    if np.isfinite(e)
                ]
                means.append(float(np.mean(finite_errors)) if finite_errors else float("nan"))
                maxes.append(float(np.max(finite_errors))  if finite_errors else float("nan"))
            aggregate[label][(sig, metric)] = {"mean": means, "max": maxes}

    return aggregate, per_series, model_labels


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _apply_thesis_rcparams() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({
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
    })


def _model_style(label: str) -> dict[str, Any]:
    return dict(MODEL_STYLES.get(label, {"color": "0.2", "linestyle": "-", "linewidth": 1.35, "marker": "o"}))


def _apply_axes_style(ax: Any) -> None:
    ax.grid(True, which="major", color=GRID_COLOR, linewidth=0.5, alpha=0.75)
    ax.grid(True, which="minor", color=GRID_MINOR_COLOR, linewidth=0.35, alpha=0.45)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(SPINE_LINE_WIDTH)
        spine.set_edgecolor(SPINE_COLOR)


def _add_panel_label(ax: Any, index: int) -> None:
    ax.text(
        0.02,
        0.96,
        f"({chr(ord('a') + int(index))})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=PANEL_LABEL_FONT_SIZE,
    )


def _scaled_errors(values: list[float]) -> list[float]:
    return [
        float(value) * ERROR_SCALE if np.isfinite(value) else float("nan")
        for value in values
    ]


def _save_figure(fig: Any, output_path: Path) -> None:
    fig.savefig(
        output_path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.02,
        dpi=FIGURE_DPI,
    )
    if SAVE_PNG_PREVIEW:
        fig.savefig(
            output_path.with_suffix(".png"),
            dpi=FIGURE_DPI,
            bbox_inches="tight",
            pad_inches=0.03,
        )


def _plot_error_sweep_stat(
    aggregate: dict[str, dict[tuple[str, str], dict[str, list[float]]]],
    model_labels: list[str],
    x_values: tuple[float, ...],
    *,
    stat_key: str,
    output_suffix: str,
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _apply_thesis_rcparams()
    output_path = OUTPUT_DIR / f"fig_07_dt_model_error_sweep_{DT_SWEEP_MODE}_{output_suffix}.pdf"

    # Skip the reference point (index 0, always 0% error) — start from index 1.
    x_vals = list(x_values)[1:]

    def _tick_label(value: float, _pos: Any) -> str:
        for tick in x_vals:
            if np.isclose(value, tick):
                return _format_x_tick(tick)
        return ""

    fig, axes = plt.subplots(2, 2, figsize=THESIS_FIGSIZE_2X2, sharex=True)

    for panel_index, (ax, (sig, metric, _)) in enumerate(zip(axes.flat, SUBPLOT_SPECS)):
        for label in model_labels:
            stats = aggregate[label][(sig, metric)]
            style = _model_style(label)
            ax.plot(
                x_vals,
                _scaled_errors(stats[stat_key][1:]),
                marker=style["marker"],
                markersize=3.2,
                linewidth=style["linewidth"],
                color=style["color"],
                linestyle=style["linestyle"],
                label=label,
            )
        if panel_index >= 2:
            ax.set_xlabel(_sweep_axis_label())
        ax.set_ylabel(
            _YLABEL_SYMBOLS[(sig, metric)],
            labelpad=6,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(FixedLocator(x_vals))
        ax.xaxis.set_major_formatter(FuncFormatter(_tick_label))
        _apply_axes_style(ax)
        _add_panel_label(ax, panel_index)

    model_handles = [
        Line2D(
            [0], [0],
            color=_model_style(label)["color"],
            linestyle=_model_style(label)["linestyle"],
            marker=_model_style(label)["marker"],
            markersize=3.2,
            linewidth=_model_style(label)["linewidth"],
            label=label,
        )
        for label in model_labels
    ]
    fig.legend(
        handles=model_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=3,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        columnspacing=1.2,
        handletextpad=0.6,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88), pad=0.35, w_pad=0.8, h_pad=0.6)
    _save_figure(fig, output_path)
    plt.close(fig)
    return output_path


def plot_error_sweep(
    aggregate: dict[str, dict[tuple[str, str], dict[str, list[float]]]],
    model_labels: list[str],
    x_values: tuple[float, ...],
) -> list[Path]:
    return [
        _plot_error_sweep_stat(
            aggregate,
            model_labels,
            x_values,
            stat_key="mean",
            output_suffix="mean",
        ),
        _plot_error_sweep_stat(
            aggregate,
            model_labels,
            x_values,
            stat_key="max",
            output_suffix="maximum",
        ),
    ]


def _resolve_reduced_velocity_sweep_value(x_values: tuple[float, ...]) -> tuple[float, int]:
    selected = float(x_values[-1] if REDUCED_VELOCITY_SWEEP_VALUE is None else REDUCED_VELOCITY_SWEEP_VALUE)
    matches = [idx for idx, value in enumerate(x_values) if np.isclose(value, selected)]
    if not matches:
        available = ", ".join(_format_x_tick(value) for value in x_values)
        raise SystemExit(
            f"REDUCED_VELOCITY_SWEEP_VALUE={selected:g} is not in the configured sweep values: {available}."
        )
    if matches[0] == 0:
        raise SystemExit(
            "The reduced-velocity plot needs a non-reference sweep value; "
            f"{_format_x_tick(selected)} is the finest reference point with zero error."
        )
    return float(x_values[matches[0]]), matches[0]


def _group_ur_error_points(points: list[tuple[float, float]]) -> dict[str, list[float]]:
    grouped: dict[float, list[float]] = {}
    for ur, error in points:
        if not (np.isfinite(ur) and np.isfinite(error)):
            continue
        grouped.setdefault(float(round(ur, 6)), []).append(float(error))

    ur_values = sorted(grouped)
    means: list[float] = []
    maxes: list[float] = []
    for ur in ur_values:
        values = np.asarray(grouped[ur], dtype=float)
        means.append(float(np.mean(values)))
        maxes.append(float(np.max(values)))
    return {"ur": ur_values, "mean": means, "max": maxes}


def plot_reduced_velocity_error_sweep(
    per_series: dict[str, dict[tuple[str, str], list[list[tuple[float, float]]]]],
    model_labels: list[str],
    x_values: tuple[float, ...],
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    sweep_value, sweep_index = _resolve_reduced_velocity_sweep_value(x_values)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _apply_thesis_rcparams()
    output_path = (
        OUTPUT_DIR
        / f"fig_07_dt_model_error_sweep_by_reduced_velocity_{DT_SWEEP_MODE}_{_output_token(sweep_value)}.pdf"
    )

    grouped_stats = {
        label: {
            (sig, metric): _group_ur_error_points(per_series[label][(sig, metric)][sweep_index])
            for sig, metric, _ in SUBPLOT_SPECS
        }
        for label in model_labels
    }
    ur_ticks = sorted(
        {
            ur
            for label in model_labels
            for sig, metric, _ in SUBPLOT_SPECS
            for ur in grouped_stats[label][(sig, metric)]["ur"]
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=THESIS_FIGSIZE_2X2, sharex=True)

    for panel_index, (ax, (sig, metric, _)) in enumerate(zip(axes.flat, SUBPLOT_SPECS)):
        for label in model_labels:
            stats = grouped_stats[label][(sig, metric)]
            if not stats["ur"]:
                continue
            style = _model_style(label)
            ax.plot(
                stats["ur"],
                _scaled_errors(stats["mean"]),
                marker=style["marker"],
                markersize=3.2,
                linewidth=style["linewidth"],
                color=style["color"],
                linestyle=style["linestyle"],
                label=label,
            )
        if panel_index >= 2:
            ax.set_xlabel(r"Reduced velocity $U_r$")
        ax.set_ylabel(
            _YLABEL_SYMBOLS[(sig, metric)],
            labelpad=6,
        )
        ax.set_yscale("log")
        if ur_ticks and len(ur_ticks) <= 16:
            ax.set_xticks(ur_ticks, [_format_ur_tick(tick) for tick in ur_ticks])
        ax.margins(x=0.04)
        _apply_axes_style(ax)
        _add_panel_label(ax, panel_index)

    model_handles = [
        Line2D(
            [0], [0],
            color=_model_style(label)["color"],
            linestyle=_model_style(label)["linestyle"],
            marker=_model_style(label)["marker"],
            markersize=3.2,
            linewidth=_model_style(label)["linewidth"],
            label=label,
        )
        for label in model_labels
    ]
    fig.legend(
        handles=model_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=3,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        columnspacing=1.2,
        handletextpad=0.6,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88), pad=0.35, w_pad=0.8, h_pad=0.6)
    _save_figure(fig, output_path)
    plt.close(fig)
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    x_values = _get_sweep_x_values()

    print(f"Data dir   : {DATA_DIR}")
    print(f"Mode       : {DT_SWEEP_MODE}")
    npz_paths = sorted(DATA_DIR.glob("*.npz"))
    if not npz_paths:
        raise SystemExit(f"No *.npz files found in {DATA_DIR}")
    print(f"NPZ files  : {len(npz_paths)}")

    print("Loading series...")
    all_series: list[dict[str, Any]] = []
    for path in tqdm(npz_paths, desc="Loading", unit="file"):
        try:
            all_series.append(_load_npz_series(path))
        except Exception as exc:
            tqdm.write(f"  [warn] Could not load {path.name}: {exc}")
    if not all_series:
        raise SystemExit("No series could be loaded.")
    if EXCLUDED_SERIES_NAME_PREFIXES:
        original_count = len(all_series)
        all_series = [
            series for series in all_series
            if not str(series.get("name", "")).startswith(EXCLUDED_SERIES_NAME_PREFIXES)
        ]
        print(f"Excluded   : {original_count - len(all_series)} NPZ files by name prefix")
    if ONE_SERIES_PER_REDUCED_VELOCITY:
        original_count = len(all_series)
        all_series = _select_one_series_per_reduced_velocity(all_series)
        print(f"One per Ur : {len(all_series)} selected from {original_count} NPZ files")

    raw_dt = _series_dt(all_series[0])
    print(f"Raw dt     : {raw_dt:.6g} s")
    print(f"Sweep ({DT_SWEEP_MODE}): {', '.join(_format_x_tick(x) for x in x_values)}")
    print(f"TD memory  : {TD_MEMORY_TAU_SPEC} (n_memory=round(tau/dt) per rollout step)")
    print(f"TD force phase: {TD_FORCE_PHASE_CONVENTION}")
    print(f"Latent RNN encoder downsampling: {LATENT_RNN_ENCODER_DOWNSAMPLING_FACTOR}x")
    print(f"Rollout dtype: {ROLLOUT_DTYPE}")
    print(f"Rollout grid: stored trimmed NPZ time grids; coarser runs use downsampled NPZ samples")
    print(f"Rollout IC : stored reference first sample after {ROLLOUT_START_SECONDS:g} s trim")
    if USE_SYNTHETIC_INITIAL_CONDITION:
        print(
            "Synthetic IC: "
            f"y0={SYNTHETIC_DISPLACEMENT_OVER_D:g}D, "
            f"dy0={SYNTHETIC_VELOCITY:g}, "
            f"theta0={SYNTHETIC_THETA:g}, "
            f"sig_dy={SYNTHETIC_SIGMA_DISPLACEMENT_OVER_D:g}D*omega_n, "
            f"sig_ddy={SYNTHETIC_SIGMA_DISPLACEMENT_OVER_D:g}D*omega_n^2"
        )
    if USE_STEADY_STATE_CONVERGENCE_WINDOW:
        print(f"Metric window: {METRIC_WINDOW_AFTER_CONVERGENCE_SECONDS:g} s after convergence")
        print(f"Max rollout: {MAX_ROLLOUT_SECONDS:g} s")
        print(
            "Metric trim: steady-state onset from displacement std "
            f"({STEADY_STATE_AMP_REL_TOL * 100:g}% over {STEADY_STATE_N_CYCLES:g} periods)"
        )
    else:
        print("Metric window: full stored trimmed NPZ segment")

    print("Loading correction models...")
    corr_specs = [
        {"path": str(spec["path"]), "label": spec["label"]}
        for spec in CORRECTION_MODEL_SPECS
    ]
    sources = load_trained_model_sources(corr_specs, repo_root=ROOT, device="cpu")
    for source in sources:
        model = getattr(source, "model", None)
        if hasattr(model, "to"):
            model.to(dtype=ROLLOUT_DTYPE)
    print(f"  Loaded: {', '.join(s.label for s in sources)}")

    # Use td_params from the first correction model for the Vivana-TD baseline.
    td_params: dict[str, float] = dict(sources[0].base_td_params)

    print("Running sweep...")
    aggregate, per_series, model_labels = run_sweep(all_series, sources, td_params, x_values)

    print("Plotting...")
    output_paths = plot_error_sweep(aggregate, model_labels, x_values)
    for output_path in output_paths:
        print(f"Saved to   : {output_path}")
    if GENERATE_REDUCED_VELOCITY_PLOT:
        ur_output_path = plot_reduced_velocity_error_sweep(per_series, model_labels, x_values)
        sweep_value, _ = _resolve_reduced_velocity_sweep_value(x_values)
        print(f"Saved Ur   : {ur_output_path} ({_sweep_descriptor(sweep_value)})")


if __name__ == "__main__":
    main()
