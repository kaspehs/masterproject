from __future__ import annotations

import glob
import json
from itertools import product
from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    tqdm = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from HNN_helper import resolve_td_memory_config, resolve_td_n_memory

try:
    from td_hidden_state import (
        build_paramsets_from_values,
        compute_force_spread_history,
        compute_theta_series as shared_compute_theta_series,
        initial_hidden_sigmas as shared_initial_hidden_sigmas,
        initial_phi_dy as shared_initial_phi_dy,
        paramset_id as shared_paramset_id,
        replay_hidden_state_with_cfd_motion as shared_replay_hidden_state_with_cfd_motion,
        wrap_phase as shared_wrap_phase,
    )
except ModuleNotFoundError:
    try:
        from Data_Gen.td_hidden_state import (
            build_paramsets_from_values,
            compute_force_spread_history,
            compute_theta_series as shared_compute_theta_series,
            initial_hidden_sigmas as shared_initial_hidden_sigmas,
            initial_phi_dy as shared_initial_phi_dy,
            paramset_id as shared_paramset_id,
            replay_hidden_state_with_cfd_motion as shared_replay_hidden_state_with_cfd_motion,
            wrap_phase as shared_wrap_phase,
        )
    except ModuleNotFoundError:
        from CFD_Data.td_hidden_state import (
            build_paramsets_from_values,
            compute_force_spread_history,
            compute_theta_series as shared_compute_theta_series,
            initial_hidden_sigmas as shared_initial_hidden_sigmas,
            initial_phi_dy as shared_initial_phi_dy,
            paramset_id as shared_paramset_id,
            replay_hidden_state_with_cfd_motion as shared_replay_hidden_state_with_cfd_motion,
            wrap_phase as shared_wrap_phase,
        )


# Case selection
THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR if (THIS_DIR / "npz_exports").exists() else THIS_DIR.parent / "CFD_Data"
INPUT_NPZS: list[Path] | None = None  # None -> use all files matching INPUT_NPZ_GLOB
#INPUT_NPZ_GLOB = str(THIS_DIR / "npz_exports" / "comb_Ur7__2Hydro.npz")
INPUT_NPZ_GLOB = str(DATA_DIR / "npz_exports" / "*.npz")
OUTPUT_DIR = DATA_DIR / "td_burnin_analysis"
OVERWRITE = True
MULTI_CASE_ONLY_SPREAD_PLOT = False  # If True, only generate the phase spread vs burn-in plot for multiple cases, skip single-case endpoint mismatch and trajectory overlay plots.
SKIP_INVALID_CASES = True
SHOW_PROGRESS = True
'''
# TD parameter sweep
CV_VALUES = [1.2]
CD_VALUES = [1.1]
CA_VALUES = [1.0]

FHAT_MIN_VALUES = [0.11]
FHAT0_VALUES = [0.18]
FHAT_MAX_VALUES = [0.26]
'''

CV_VALUES = [1.03]
CD_VALUES = [1.02]
CA_VALUES = [1.95]

FHAT_MIN_VALUES = [0.142]
FHAT0_VALUES = [0.142]
FHAT_MAX_VALUES = [0.173]

DAMPING_C_VALUES = [0.0]
N_MEMORY = 500
INTEGRATOR = "rk4_coupled"
TD_MEMORY_MODE = "tau_over_tref"
TD_TAU_OVER_TREF = 4.0
TD_MEMORY_TAU_S: float | None = None

# Burn-in analysis configs
THETA0_VALUES = np.linspace(-np.pi / 2, np.pi / 2, 7, endpoint=True)

COMPARISON_WINDOW_SECONDS = 10.0
PHASE_WRAP = "principal"
RERUN_BURNIN_WINDOWS_SECONDS = [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0)]
SPREAD_PLOT_YMIN = 1.0e-12
COMBINED_THETA_REL_STD_THRESHOLD = 1.0e-2
COMBINED_THETA_PERSISTENCE_SECONDS = 1.0
COMBINED_THETA_MAX_SECONDS = 100.0
COMBINED_THETA_FIG_DPI = 300
COMBINED_THETA_TITLE_FONTSIZE = 12
COMBINED_THETA_LABEL_FONTSIZE = 11
COMBINED_THETA_TICK_FONTSIZE = 10
COMBINED_THETA_LEGEND_FONTSIZE = 10

# Hidden-state initialization
SIGMA_INIT_MODE = "lookahead_rms"  # "zero", "local_rms", or "lookahead_rms"
SIGMA_INIT_WINDOW_SECONDS: float | str | None = "tau"  # "tau" -> use the resolved tau seconds


def _as_list(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)
    return [value]


def _format_float(value: float) -> str:
    text = f"{float(value):.6g}"
    return text.replace("+", "")


def _paramset_id(params: dict[str, float]) -> str:
    return shared_paramset_id(params)


def _wrap_phase(values: np.ndarray, mode: str) -> np.ndarray:
    return shared_wrap_phase(values, mode=mode)


def _compute_theta_series(
    *,
    dy: np.ndarray,
    ddy: np.ndarray,
    phi_vy: np.ndarray,
    sig_dy_loc: np.ndarray,
    sig_ddy_loc: np.ndarray,
    flow_speed_m_s: float,
) -> np.ndarray:
    return shared_compute_theta_series(
        dy=dy,
        ddy=ddy,
        phi_vy=phi_vy,
        sig_dy_loc=sig_dy_loc,
        sig_ddy_loc=sig_ddy_loc,
        flow_speed_m_s=flow_speed_m_s,
        mode=PHASE_WRAP,
    )


def _initial_phi_dy(
    *,
    dy0: float,
    ddy0: float,
    sig_dy_loc0: float,
    sig_ddy_loc0: float,
    flow_speed_m_s: float,
) -> float:
    return shared_initial_phi_dy(
        dy0=dy0,
        ddy0=ddy0,
        sig_dy_loc0=sig_dy_loc0,
        sig_ddy_loc0=sig_ddy_loc0,
        flow_speed_m_s=flow_speed_m_s,
    )


def _td_memory_config() -> dict[str, object]:
    return resolve_td_memory_config(
        {
            "td_memory_mode": TD_MEMORY_MODE,
            "td_tau_over_tref": TD_TAU_OVER_TREF,
            "td_memory_tau_s": TD_MEMORY_TAU_S,
        }
    )


def _resolve_case_td_memory(
    *,
    case: dict[str, np.ndarray | float | str],
    params: dict[str, float],
) -> tuple[int, float]:
    params_with_memory = dict(params)
    params_with_memory.setdefault("n_memory", float(N_MEMORY))
    dt_value = float(case["dt_dim"])
    n_memory_value = resolve_td_n_memory(
        params_with_memory,
        dt=dt_value,
        flow_speed=float(case["flow_speed_m_s"]),
        diameter=float(case["diameter_m"]),
        memory_cfg=_td_memory_config(),
    )
    return max(1, int(round(float(n_memory_value)))), float(n_memory_value) * dt_value


def _resolve_sigma_init_window_seconds(tau_s: float) -> float | None:
    window = SIGMA_INIT_WINDOW_SECONDS
    if window is None:
        return None
    if isinstance(window, str):
        key = window.strip().lower()
        if key == "tau":
            return float(tau_s)
        raise ValueError("SIGMA_INIT_WINDOW_SECONDS must be None, a number, or 'tau'.")
    return float(window)


def _circular_std(values: np.ndarray) -> float:
    angles = np.asarray(values, dtype=float)
    resultant = np.abs(np.mean(np.exp(1j * angles)))
    resultant = np.clip(resultant, 1.0e-12, 1.0)
    return float(np.sqrt(-2.0 * np.log(resultant)))


def _circular_mean(values: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    angles = np.asarray(values, dtype=float)
    valid = np.isfinite(angles)
    unit_vectors = np.where(valid, np.exp(1j * angles), 0.0 + 0.0j)
    counts = np.sum(valid, axis=axis)
    vector_sum = np.sum(unit_vectors, axis=axis)

    if axis is None:
        if int(counts) <= 0:
            return float("nan")
        mean_vector = vector_sum / float(counts)
        if np.abs(mean_vector) <= 1.0e-12:
            return float("nan")
        return float(np.angle(mean_vector))

    counts_arr = np.asarray(counts)
    mean_angles = np.full(counts_arr.shape, np.nan, dtype=float)
    nonzero_mask = counts_arr > 0
    if np.any(nonzero_mask):
        mean_vector = np.zeros(counts_arr.shape, dtype=complex)
        mean_vector[nonzero_mask] = vector_sum[nonzero_mask] / counts_arr[nonzero_mask]
        stable_mask = nonzero_mask & (np.abs(mean_vector) > 1.0e-12)
        mean_angles[stable_mask] = np.angle(mean_vector[stable_mask])
    return mean_angles


def _load_case(npz_path: Path) -> dict[str, np.ndarray | float | str]:
    if not npz_path.exists():
        raise FileNotFoundError(f"Input CFD case '{npz_path}' not found.")

    data = np.load(npz_path, allow_pickle=True)
    required_arrays = ["time_dim", "y_disp_dim", "y_vel_dim", "y_acc_dim"]
    required_scalars = [
        "dt_dim",
        "flow_speed_m_s",
        "stiffness_n_m",
        "dry_mass_kg",
        "diameter_m",
        "rho_kg_m3",
    ]
    missing = [key for key in required_arrays + required_scalars if key not in data.files]
    if missing:
        raise KeyError(f"CFD case '{npz_path}' is missing required keys: {missing}")
    has_force_per_m = "y_force_per_m_dim" in data.files
    has_force_total = "y_force_dim" in data.files
    if not has_force_per_m and not has_force_total:
        raise KeyError(
            f"CFD case '{npz_path}' is missing both 'y_force_per_m_dim' and 'y_force_dim'."
        )
    if not has_force_per_m and "span_m" not in data.files:
        raise KeyError(
            f"CFD case '{npz_path}' needs 'span_m' when only 'y_force_dim' is available."
        )

    time_dim = np.asarray(data["time_dim"], dtype=float)
    y_dim = np.asarray(data["y_disp_dim"], dtype=float)
    dy_dim = np.asarray(data["y_vel_dim"], dtype=float)
    ddy_dim = np.asarray(data["y_acc_dim"], dtype=float)
    if has_force_per_m:
        force_dim = np.asarray(data["y_force_per_m_dim"], dtype=float)
    else:
        span_m = float(np.asarray(data["span_m"]).reshape(()))
        force_dim = np.asarray(data["y_force_dim"], dtype=float) / span_m
    if not (len(time_dim) == len(y_dim) == len(dy_dim) == len(ddy_dim) == len(force_dim)):
        raise ValueError("Dimensional CFD channels must have the same length.")
    if len(time_dim) < 4:
        raise ValueError("Need at least 4 CFD samples for the burn-in analysis.")

    keep_mask = np.ones(len(time_dim), dtype=bool)
    dup_mask = np.diff(time_dim) <= 0.0
    if np.any(dup_mask):
        if np.any(np.diff(time_dim) < 0.0):
            raise ValueError("CFD time vector must be nondecreasing.")
        keep_mask[1:] = ~dup_mask
        time_dim = time_dim[keep_mask]
        y_dim = y_dim[keep_mask]
        dy_dim = dy_dim[keep_mask]
        ddy_dim = ddy_dim[keep_mask]
        force_dim = force_dim[keep_mask]

    dt_dim = float(np.asarray(data["dt_dim"]).reshape(()))
    diffs = np.diff(time_dim)
    if np.any(diffs <= 0.0):
        raise ValueError("CFD time vector must be strictly increasing after duplicate removal.")
    dt_median = float(np.median(diffs))
    if dt_median <= 0.0:
        raise ValueError("Median CFD time step must be positive.")
    # Use the observed median step in the analysis. The exported CFD series can
    # contain tiny floating-point jitter, and exact equality with dt_dim is not
    # important for the burn-in scan as long as time is monotone and near-uniform.
    dt_dim = dt_median

    return {
        "case_name": npz_path.stem,
        "source_path": str(npz_path),
        "time_dim": time_dim,
        "y_dim": y_dim,
        "dy_dim": dy_dim,
        "ddy_dim": ddy_dim,
        "force_dim": force_dim,
        "force_std_dim": float(np.std(force_dim)),
        "dt_dim": dt_dim,
        "flow_speed_m_s": float(np.asarray(data["flow_speed_m_s"]).reshape(())),
        "stiffness_n_m": float(np.asarray(data["stiffness_n_m"]).reshape(())),
        "dry_mass_kg": float(np.asarray(data["dry_mass_kg"]).reshape(())),
        "diameter_m": float(np.asarray(data["diameter_m"]).reshape(())),
        "rho_kg_m3": float(np.asarray(data["rho_kg_m3"]).reshape(())),
        "ur_label": (
            float(np.asarray(data["U_r_label_scalar"]).reshape(()))
            if "U_r_label_scalar" in data.files
            else (
                float(np.asarray(data["label_ur"]).reshape(()))
                if "label_ur" in data.files
                else float("nan")
            )
        ),
        "ur_computed": (
            float(np.asarray(data["U_r_computed_scalar"]).reshape(()))
            if "U_r_computed_scalar" in data.files
            else (
                float(np.asarray(data["computed_ur"]).reshape(()))
                if "computed_ur" in data.files
                else float("nan")
            )
        ),
        "structural_frequency_hz": (
            float(np.asarray(data["structural_frequency_hz"]).reshape(()))
            if "structural_frequency_hz" in data.files
            else None
        ),
    }


def _build_paramsets() -> list[dict[str, float]]:
    return build_paramsets_from_values(
        cv_values=CV_VALUES,
        cd_values=CD_VALUES,
        ca_values=CA_VALUES,
        damping_c_values=DAMPING_C_VALUES,
        fhat_min_values=FHAT_MIN_VALUES,
        fhat0_values=FHAT0_VALUES,
        fhat_max_values=FHAT_MAX_VALUES,
    )


def _candidate_start_indices(time_dim: np.ndarray, dt_dim: float, eval_idx: int) -> list[int]:
    step_samples = int(round(float(BURNIN_STEP_SECONDS) / float(dt_dim)))
    if step_samples <= 0:
        raise ValueError("BURNIN_STEP_SECONDS is too small relative to dt_dim.")

    eval_time = float(time_dim[eval_idx])
    max_burnin_seconds = eval_time - float(time_dim[0]) if MAX_BURNIN_SECONDS is None else float(MAX_BURNIN_SECONDS)
    if max_burnin_seconds < float(MIN_BURNIN_SECONDS):
        raise ValueError("MAX_BURNIN_SECONDS must be >= MIN_BURNIN_SECONDS.")

    starts: list[int] = []
    burnin = float(MIN_BURNIN_SECONDS)
    while burnin <= max_burnin_seconds + 0.5 * dt_dim:
        target_time = eval_time - burnin
        start_idx = int(np.searchsorted(time_dim, target_time, side="left"))
        start_idx = max(0, min(start_idx, eval_idx - 1))
        if eval_idx - start_idx >= 2:
            starts.append(start_idx)
        burnin += step_samples * dt_dim

    unique_starts = sorted(set(starts))
    if len(unique_starts) < 2:
        raise ValueError("Need at least two valid candidate burn-in starts.")
    return unique_starts


def _run_single_sim(
    *,
    case: dict[str, np.ndarray | float | str],
    params: dict[str, float],
    start_idx: int,
    eval_idx: int,
    theta0: float,
    n_memory: int,
    tau_s: float,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    time_dim = case["time_dim"]
    y_dim = case["y_dim"]
    dy_dim = case["dy_dim"]
    ddy_dim = case["ddy_dim"]

    start_time = float(time_dim[start_idx])
    eval_time = float(time_dim[eval_idx])
    burnin_seconds = eval_time - start_time
    sig_dy_loc0, sig_ddy_loc0 = _initial_hidden_sigmas(
        case=case,
        start_idx=start_idx,
        flow_speed_m_s=float(case["flow_speed_m_s"]),
        n_memory=int(n_memory),
        tau_s=float(tau_s),
    )
    phi_dy0 = _initial_phi_dy(
        dy0=float(dy_dim[start_idx]),
        ddy0=float(ddy_dim[start_idx]),
        sig_dy_loc0=sig_dy_loc0,
        sig_ddy_loc0=sig_ddy_loc0,
        flow_speed_m_s=float(case["flow_speed_m_s"]),
    )
    phi_vy0 = float(_wrap_phase(np.asarray([phi_dy0 - float(theta0)]), PHASE_WRAP)[0])
    sim = _replay_hidden_state_with_cfd_motion(
        time=time_dim[start_idx : eval_idx + 1],
        y=y_dim[start_idx : eval_idx + 1],
        dy=dy_dim[start_idx : eval_idx + 1],
        ddy=ddy_dim[start_idx : eval_idx + 1],
        flow_speed_m_s=float(case["flow_speed_m_s"]),
        rho_kg_m3=float(case["rho_kg_m3"]),
        diameter_m=float(case["diameter_m"]),
        params=params,
        phi_vy0=float(phi_vy0),
        sig_dy_loc0=sig_dy_loc0,
        sig_ddy_loc0=sig_ddy_loc0,
        n_memory=int(n_memory),
    )

    phi_vy_eval = float(sim["phi_vy"][-1])
    theta_series = _compute_theta_series(
        dy=sim["dy"],
        ddy=sim["ddy"],
        phi_vy=sim["phi_vy"],
        sig_dy_loc=sim["sig_dy_loc"],
        sig_ddy_loc=sim["sig_ddy_loc"],
        flow_speed_m_s=float(case["flow_speed_m_s"]),
    )
    theta_eval = float(theta_series[-1])
    row = {
        "case_name": str(case["case_name"]),
        "Cv": float(params["Cv"]),
        "Cd": float(params["Cd"]),
        "Ca": float(params["Ca"]),
        "C": float(params["C"]),
        "fhat_min": float(params["fhat_min"]),
        "fhat0": float(params["fhat0"]),
        "fhat_max": float(params["fhat_max"]),
        "theta0": float(theta0),
        "phi_dy0": phi_dy0,
        "phi_vy0": float(phi_vy0),
        "start_idx": int(start_idx),
        "start_time": start_time,
        "eval_idx": int(eval_idx),
        "eval_time": eval_time,
        "burnin_seconds": burnin_seconds,
        "phi_vy_eval": phi_vy_eval,
        "phi_vy_eval_wrapped": float(_wrap_phase(np.asarray([phi_vy_eval]), PHASE_WRAP)[0]),
        "theta_eval": theta_eval,
        "theta_eval_wrapped": float(_wrap_phase(np.asarray([theta_eval]), PHASE_WRAP)[0]),
        "n_memory": int(n_memory),
        "tau_s": float(tau_s),
        "F_total_eval": float(sim["F_total"][-1]),
        "F_total_eval_per_m": float(sim["F_total"][-1]),
        "Fy_eval": float(sim["Fy"][-1]),
        "sig_dy_loc0": sig_dy_loc0,
        "sig_ddy_loc0": sig_ddy_loc0,
        "y_eval_td": float(sim["y"][-1]),
        "dy_eval_td": float(sim["dy"][-1]),
        "ddy_eval_td": float(sim["ddy"][-1]),
        "y_eval_cfd": float(y_dim[eval_idx]),
        "dy_eval_cfd": float(dy_dim[eval_idx]),
        "ddy_eval_cfd": float(ddy_dim[eval_idx]),
    }
    row["y_err_eval"] = row["y_eval_td"] - row["y_eval_cfd"]
    row["dy_err_eval"] = row["dy_eval_td"] - row["dy_eval_cfd"]
    row["ddy_err_eval"] = row["ddy_eval_td"] - row["ddy_eval_cfd"]
    return row, sim


def _replay_hidden_state_with_cfd_motion(
    *,
    time: np.ndarray,
    y: np.ndarray,
    dy: np.ndarray,
    ddy: np.ndarray,
    flow_speed_m_s: float,
    rho_kg_m3: float,
    diameter_m: float,
    params: dict[str, float],
    phi_vy0: float,
    sig_dy_loc0: float,
    sig_ddy_loc0: float,
    n_memory: int,
) -> dict[str, np.ndarray]:
    return shared_replay_hidden_state_with_cfd_motion(
        time=time,
        y=y,
        dy=dy,
        ddy=ddy,
        flow_speed_m_s=float(flow_speed_m_s),
        rho_kg_m3=float(rho_kg_m3),
        diameter_m=float(diameter_m),
        params=params,
        phi_vy0=float(phi_vy0),
        sig_dy_loc0=float(sig_dy_loc0),
        sig_ddy_loc0=float(sig_ddy_loc0),
        n_memory=int(n_memory),
    )


def _initial_hidden_sigmas(
    *,
    case: dict[str, np.ndarray | float | str],
    start_idx: int,
    flow_speed_m_s: float,
    n_memory: int,
    tau_s: float | None = None,
) -> tuple[float, float]:
    window_seconds = _resolve_sigma_init_window_seconds(float(tau_s)) if tau_s is not None else SIGMA_INIT_WINDOW_SECONDS
    return shared_initial_hidden_sigmas(
        case_like=case,
        start_idx=start_idx,
        flow_speed_m_s=flow_speed_m_s,
        n_memory=int(n_memory),
        mode=SIGMA_INIT_MODE,
        window_seconds=window_seconds,
    )


def _plot_phase_vs_burnin(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for theta0, group in df.groupby("theta0", sort=True):
        group = group.sort_values("burnin_seconds")
        ax.plot(
            group["burnin_seconds"].to_numpy(),
            group["theta_eval_wrapped"].to_numpy(),
            marker="o",
            linewidth=1.2,
            markersize=3.0,
            label=f"theta0={theta0:.3f}",
        )
    ax.set_xlabel("Burn-in length [s]")
    ax.set_ylabel("Wrapped theta at evaluation [rad]")
    ax.set_title("Evaluation theta vs burn-in length")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, ncol=1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_phase_spread(
    case: dict[str, np.ndarray | float | str],
    df: pd.DataFrame,
    out_path: Path,
) -> None:
    grouped = _phase_spread_summary(case, df)
    _plot_phase_spread_summary(grouped, out_path=out_path)


def _phase_spread_summary(
    case: dict[str, np.ndarray | float | str],
    df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    force_std_ref = max(float(case["force_std_dim"]), np.finfo(float).eps)
    for burnin, group in df.groupby("burnin_seconds", sort=True):
        theta_vals = group["theta_eval_wrapped"].to_numpy(dtype=float)
        force_vals = group["F_total_eval"].to_numpy(dtype=float)
        rows.append(
            {
                "burnin_seconds": float(burnin),
                "theta_circular_std": _circular_std(theta_vals),
                "cos_theta_std": float(np.std(np.cos(theta_vals))),
                "force_total_rel_std": float(np.std(force_vals) / force_std_ref),
            }
        )
    grouped = pd.DataFrame(rows).sort_values("burnin_seconds").reset_index(drop=True)
    grouped["case_name"] = str(case["case_name"])
    return grouped


def _plot_phase_spread_summary(grouped: pd.DataFrame, *, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8, 11), sharex=True)
    panel_specs = [
        ("theta_circular_std", "Circular std of wrapped theta [rad]", "Theta spread vs burn-in length"),
        ("cos_theta_std", "Std of cos(theta) [-]", "cos(theta) spread vs burn-in length"),
        (
            "force_total_rel_std",
            "Std of predicted force per unit length / std(CFD force per unit length) [-]",
            "Relative force-per-unit-length spread vs burn-in length",
        ),
    ]
    for ax, (column, ylabel, title) in zip(axes, panel_specs):
        ax.plot(grouped["burnin_seconds"], grouped[column], marker="o", linewidth=1.5)
        min_col = f"{column}_min"
        max_col = f"{column}_max"
        if min_col in grouped.columns and max_col in grouped.columns:
            ax.fill_between(
                grouped["burnin_seconds"].to_numpy(dtype=float),
                grouped[min_col].to_numpy(dtype=float),
                grouped[max_col].to_numpy(dtype=float),
                alpha=0.16,
            )
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        positive = grouped[column].to_numpy(dtype=float)
        positive = positive[np.isfinite(positive) & (positive > 0.0)]
        if positive.size > 0:
            ax.set_yscale("log")
            ax.set_ylim(bottom=max(float(np.min(positive)) * 0.8, float(SPREAD_PLOT_YMIN)))
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Burn-in length [s]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _time_spread_analysis(
    *,
    case: dict[str, np.ndarray | float | str],
    params: dict[str, float],
) -> dict[str, object]:
    resolved_n_memory, resolved_tau_s = _resolve_case_td_memory(case=case, params=params)
    spread = compute_force_spread_history(
        case_payload=case,
        params=params,
        theta0_values=np.asarray(THETA0_VALUES, dtype=float),
        sigma_init_mode=SIGMA_INIT_MODE,
        sigma_init_window_seconds=_resolve_sigma_init_window_seconds(float(resolved_tau_s)),
        n_memory=int(resolved_n_memory),
        progress=_progress if SHOW_PROGRESS else None,
        progress_desc=f"{case['case_name']} theta0",
        return_force_stack=True,
    )
    time_dim = np.asarray(spread["time_dim"], dtype=float).reshape(-1)
    time_from_start_s = time_dim - float(time_dim[0])
    theta_stack = np.asarray(spread["theta_stack"], dtype=float)
    theta0_values = np.asarray(spread["theta0_values"], dtype=float).reshape(-1)
    force_total_rel_std = np.asarray(spread["force_total_rel_std"], dtype=float).reshape(-1)
    force_total_stack = np.asarray(spread["force_total_stack"], dtype=float)

    rows: list[dict[str, float | str]] = []
    for idx, time_value in enumerate(time_from_start_s):
        theta_vals = theta_stack[:, idx] if theta_stack.ndim == 2 and theta_stack.shape[1] == time_from_start_s.size else np.asarray([], dtype=float)
        rows.append(
            {
                "time_from_start_s": float(time_value),
                "theta_circular_std": _circular_std(theta_vals) if theta_vals.size > 0 else float("nan"),
                "cos_theta_std": float(np.std(np.cos(theta_vals))) if theta_vals.size > 0 else float("nan"),
                "force_total_rel_std": float(force_total_rel_std[idx]),
                "case_name": str(case["case_name"]),
                "ur_label": float(case["ur_label"]) if np.isfinite(float(case["ur_label"])) else float("nan"),
                "ur_computed": float(case["ur_computed"]) if np.isfinite(float(case["ur_computed"])) else float("nan"),
                "flow_speed_m_s": float(case["flow_speed_m_s"]),
                "stiffness_n_m": float(case["stiffness_n_m"]),
                "dry_mass_kg": float(case["dry_mass_kg"]),
                "rho_kg_m3": float(case["rho_kg_m3"]),
                "diameter_m": float(case["diameter_m"]),
                "Ca": float(params["Ca"]),
            }
        )
    summary = pd.DataFrame(rows)

    theta_traj_rows: list[dict[str, float | str]] = []
    if theta_stack.ndim == 2 and theta_stack.shape[1] == time_from_start_s.size:
        for theta_idx, theta0 in enumerate(theta0_values):
            for time_idx, time_value in enumerate(time_from_start_s):
                theta_traj_rows.append(
                    {
                        "time_from_start_s": float(time_value),
                        "theta0": float(theta0),
                        "theta": float(theta_stack[theta_idx, time_idx]),
                        "case_name": str(case["case_name"]),
                    }
                )
    theta_traj_df = pd.DataFrame(theta_traj_rows)
    return {
        "summary": summary,
        "theta_trajectories": theta_traj_df,
        "time_from_start_s": np.asarray(time_from_start_s, dtype=float),
        "theta_stack": np.asarray(theta_stack, dtype=float),
        "theta0_values": np.asarray(theta0_values, dtype=float),
        "force_total_stack": np.asarray(force_total_stack, dtype=float),
        "resolved_n_memory": int(resolved_n_memory),
        "resolved_tau_s": float(resolved_tau_s),
    }


def _plot_time_spread_summary(summary: pd.DataFrame, *, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8, 11), sharex=True)
    panel_specs = [
        ("theta_circular_std", "Circular std of wrapped theta [rad]", "Theta spread vs time from start"),
        ("cos_theta_std", "Std of cos(theta) [-]", "cos(theta) spread vs time from start"),
        (
            "force_total_rel_std",
            "Std of predicted force per unit length / std(CFD force per unit length) [-]",
            "Relative force-per-unit-length spread vs time from start",
        ),
    ]
    x = summary["time_from_start_s"].to_numpy(dtype=float)
    for ax, (column, ylabel, title) in zip(axes, panel_specs):
        values = summary[column].to_numpy(dtype=float)
        ax.plot(x, values, linewidth=1.5)
        positive = values[np.isfinite(values) & (values > 0.0)]
        if positive.size > 0:
            ax.set_yscale("log")
            ax.set_ylim(bottom=max(float(np.min(positive)) * 0.8, float(SPREAD_PLOT_YMIN)))
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time from series start [s]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_theta_trajectories(
    *,
    time_from_start_s: np.ndarray,
    theta_stack: np.ndarray,
    theta0_values: np.ndarray,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for idx, theta0 in enumerate(np.asarray(theta0_values, dtype=float).reshape(-1)):
        ax.plot(
            np.asarray(time_from_start_s, dtype=float),
            np.asarray(theta_stack[idx], dtype=float),
            linewidth=1.0,
            label=rf"$\theta_0={theta0:.3f}$",
        )
    ax.set_xlabel("Time from series start [s]")
    ax.set_ylabel(r"Wrapped $\theta$ [rad]")
    ax.set_title(r"Wrapped $\theta$ vs time from series start")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, ncol=1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _nominal_ur_key(case_name: str) -> str:
    match = re.search(r"Ur([^_]+)", str(case_name))
    return str(match.group(1)) if match else str(case_name)


def _nominal_ur_sort_value(case_name: str) -> float:
    raw = _nominal_ur_key(case_name)
    if raw == "575":
        return 5.75
    try:
        return float(raw)
    except ValueError:
        return float("inf")


def _stored_ur_display_value(summary: pd.DataFrame) -> float:
    if summary.empty:
        return float("nan")
    if "ur_computed" not in summary.columns:
        return float("nan")
    value = float(summary["ur_computed"].iloc[0])
    return value if np.isfinite(value) else float("nan")


def _estimate_burnin_time_from_summary(summary: pd.DataFrame) -> float:
    if summary.empty:
        return float("nan")
    time_arr = summary["time_from_start_s"].to_numpy(dtype=float)
    rel_std_arr = summary["force_total_rel_std"].to_numpy(dtype=float)
    finite_mask = np.isfinite(time_arr) & np.isfinite(rel_std_arr)
    time_arr = time_arr[finite_mask]
    rel_std_arr = rel_std_arr[finite_mask]
    if time_arr.size < 2:
        return float("nan")
    below = rel_std_arr < float(COMBINED_THETA_REL_STD_THRESHOLD)
    fail_prefix = np.concatenate(([0], np.cumsum(~below, dtype=int)))
    persistence_seconds = float(COMBINED_THETA_PERSISTENCE_SECONDS)
    for idx in range(time_arr.size):
        if not below[idx]:
            continue
        target_time = float(time_arr[idx]) + persistence_seconds
        end_idx = int(np.searchsorted(time_arr, target_time, side="left"))
        end_idx = min(end_idx, time_arr.size - 1)
        if fail_prefix[end_idx + 1] == fail_prefix[idx]:
            return float(time_arr[idx])
    return float(time_arr[-1])


def _select_theta_trajectory_bundles(
    bundles: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]],
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]]:
    if not bundles:
        return []
    chosen: dict[str, tuple[float, tuple[str, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]]] = {}
    for bundle in bundles:
        case_name, _time, _theta_stack, _theta0_values, summary = bundle
        ur_key = _nominal_ur_key(str(case_name))
        burnin_time = _estimate_burnin_time_from_summary(summary)
        current = chosen.get(ur_key)
        if current is None or burnin_time > current[0]:
            chosen[ur_key] = (burnin_time, bundle)
    selected = [item[1] for item in chosen.values()]
    selected.sort(key=lambda item: (_nominal_ur_sort_value(str(item[0])), str(item[0])))
    return selected


def _plot_theta_trajectories_all_cases(
    *,
    case_theta_trajectories: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]],
    out_path: Path,
) -> None:
    selected_bundles = _select_theta_trajectory_bundles(case_theta_trajectories)
    if not selected_bundles:
        raise ValueError("Need at least one case theta trajectory bundle.")
    n_cases = len(selected_bundles)
    fig_height = max(1.8 * n_cases, 6.0)
    fig, axes = plt.subplots(n_cases, 1, figsize=(12.5, fig_height), sharex=False)
    if n_cases == 1:
        axes = [axes]

    for ax, (case_name, time_from_start_s, theta_stack, theta0_values, summary) in zip(axes, selected_bundles):
        burnin_time = _estimate_burnin_time_from_summary(summary)
        stored_ur = _stored_ur_display_value(summary)
        time_arr = np.asarray(time_from_start_s, dtype=float)
        theta_arr = np.asarray(theta_stack, dtype=float)
        if COMBINED_THETA_MAX_SECONDS is None:
            mask = np.ones_like(time_arr, dtype=bool)
        else:
            mask = time_arr <= float(COMBINED_THETA_MAX_SECONDS)
        for idx, theta0 in enumerate(np.asarray(theta0_values, dtype=float).reshape(-1)):
            ax.plot(
                time_arr[mask],
                theta_arr[idx][mask],
                linewidth=0.9,
                alpha=0.9,
                label=rf"$\theta_0={theta0:.3f}$",
            )
        if theta_arr.ndim == 2 and theta_arr.shape[1] == time_arr.size:
            theta_mean = _circular_mean(theta_arr[:, mask], axis=0)
            ax.plot(
                time_arr[mask],
                np.asarray(theta_mean, dtype=float),
                color="black",
                linewidth=2.0,
                alpha=0.95,
                label="Circular mean" if ax is axes[0] else "_nolegend_",
                zorder=3,
            )
        if np.isfinite(burnin_time):
            ax.axvline(float(burnin_time), color="black", linestyle="--", linewidth=1.0, alpha=0.9)
        title = str(case_name)
        if np.isfinite(stored_ur):
            title = f"{case_name} | " + rf"$U_{{r}}={stored_ur:.2f}$"
        ax.set_title(title, loc="left", fontsize=COMBINED_THETA_TITLE_FONTSIZE)
        ax.set_ylabel(r"$\theta$ [rad]", fontsize=COMBINED_THETA_LABEL_FONTSIZE)
        ax.grid(True, alpha=0.3)
        if COMBINED_THETA_MAX_SECONDS is not None:
            ax.set_xlim(0.0, float(COMBINED_THETA_MAX_SECONDS))
        ax.tick_params(axis="both", labelsize=COMBINED_THETA_TICK_FONTSIZE)
    axes[-1].set_xlabel("Time from series start [s]", fontsize=COMBINED_THETA_LABEL_FONTSIZE)
    handles, labels = axes[0].get_legend_handles_labels()
    legend_ncol = max(1, len(labels))
    fig.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(0.06, 0.965, 0.88, 0.06),
        mode="expand",
        ncol=legend_ncol,
        fontsize=COMBINED_THETA_LEGEND_FONTSIZE,
        frameon=False,
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=int(COMBINED_THETA_FIG_DPI), bbox_inches="tight")
    plt.close(fig)


def _combine_time_spread_summaries(summaries: list[pd.DataFrame]) -> pd.DataFrame:
    if not summaries:
        raise ValueError("Need at least one time-spread summary to combine.")
    return pd.concat(summaries, ignore_index=True).sort_values(["case_name", "time_from_start_s"]).reset_index(drop=True)


def _plot_time_spread_all_cases(combined: pd.DataFrame, *, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
    panel_specs = [
        ("theta_circular_std", "Circular std of wrapped theta [rad]", "Theta spread vs time from start"),
        ("cos_theta_std", "Std of cos(theta) [-]", "cos(theta) spread vs time from start"),
        (
            "force_total_rel_std",
            "Std of predicted force per unit length / std(CFD force per unit length) [-]",
            "Relative force-per-unit-length spread vs time from start",
        ),
    ]
    case_groups = list(combined.groupby("case_name", sort=True))
    for ax, (column, ylabel, title) in zip(axes, panel_specs):
        positive_values = []
        for case_name, group in case_groups:
            values = group[column].to_numpy(dtype=float)
            ax.plot(
                group["time_from_start_s"].to_numpy(dtype=float),
                values,
                linewidth=1.3,
                label=str(case_name),
            )
            positive_values.append(values[np.isfinite(values) & (values > 0.0)])
        nonempty_positive_values = [vals for vals in positive_values if vals.size > 0]
        if nonempty_positive_values:
            positive = np.concatenate(nonempty_positive_values)
            if positive.size > 0:
                ax.set_yscale("log")
                ax.set_ylim(bottom=max(float(np.min(positive)) * 0.8, float(SPREAD_PLOT_YMIN)))
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time from series start [s]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_combined_time_spread_outputs(
    *,
    summaries: list[pd.DataFrame],
    theta_trajectory_bundles: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]],
    out_dir: Path,
) -> None:
    combined = _combine_time_spread_summaries(summaries)
    out_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_dir / "phase_spread_vs_time_all_cases.csv", index=False)
    _plot_time_spread_all_cases(
        combined,
        out_path=out_dir / "phase_spread_vs_time_all_cases.png",
    )
    _plot_theta_trajectories_all_cases(
        case_theta_trajectories=theta_trajectory_bundles,
        out_path=out_dir / "theta_vs_time_all_cases.png",
    )


def _combine_phase_spread_summaries(summaries: list[pd.DataFrame]) -> pd.DataFrame:
    if not summaries:
        raise ValueError("Need at least one phase spread summary to combine.")
    return pd.concat(summaries, ignore_index=True).sort_values(["case_name", "burnin_seconds"]).reset_index(drop=True)


def _plot_phase_spread_all_cases(combined: pd.DataFrame, *, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
    panel_specs = [
        ("theta_circular_std", "Circular std of wrapped theta [rad]", "Theta spread vs burn-in length"),
        ("cos_theta_std", "Std of cos(theta) [-]", "cos(theta) spread vs burn-in length"),
        (
            "force_total_rel_std",
            "Std of predicted force per unit length / std(CFD force per unit length) [-]",
            "Relative force-per-unit-length spread vs burn-in length",
        ),
    ]

    case_groups = list(combined.groupby("case_name", sort=True))
    for ax, (column, ylabel, title) in zip(axes, panel_specs):
        positive_values = []
        for case_name, group in case_groups:
            values = group[column].to_numpy(dtype=float)
            ax.plot(
                group["burnin_seconds"].to_numpy(dtype=float),
                values,
                marker="o",
                linewidth=1.3,
                markersize=3.0,
                label=str(case_name),
            )
            positive_values.append(values[np.isfinite(values) & (values > 0.0)])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        nonempty_positive_values = [vals for vals in positive_values if vals.size > 0]
        if nonempty_positive_values:
            positive = np.concatenate(nonempty_positive_values)
            if positive.size > 0:
                ax.set_yscale("log")
                ax.set_ylim(bottom=max(float(np.min(positive)) * 0.8, float(SPREAD_PLOT_YMIN)))
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Burn-in length [s]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_combined_phase_spread_outputs(
    *,
    summaries: list[pd.DataFrame],
    out_dir: Path,
) -> None:
    combined = _combine_phase_spread_summaries(summaries)
    out_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_dir / "phase_spread_vs_burnin_all_cases.csv", index=False)
    _plot_phase_spread_all_cases(
        combined,
        out_path=out_dir / "phase_spread_vs_burnin_all_cases.png",
    )


def _plot_endpoint_mismatch(df: pd.DataFrame, out_path: Path) -> None:
    metrics = ["y_err_eval", "dy_err_eval", "ddy_err_eval"]
    titles = ["|y error| at evaluation", "|dy error| at evaluation", "|ddy error| at evaluation"]
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    grouped = df.groupby("burnin_seconds", sort=True)
    for ax, metric, title in zip(axes, metrics, titles):
        summary = grouped[metric].agg(
            mean_abs=lambda s: np.mean(np.abs(s.to_numpy())),
            min_abs=lambda s: np.min(np.abs(s.to_numpy())),
            max_abs=lambda s: np.max(np.abs(s.to_numpy())),
        ).reset_index()
        x = summary["burnin_seconds"].to_numpy()
        ax.plot(x, summary["mean_abs"].to_numpy(), color="tab:blue", linewidth=1.8, label="mean |err|")
        ax.fill_between(
            x,
            summary["min_abs"].to_numpy(),
            summary["max_abs"].to_numpy(),
            color="tab:blue",
            alpha=0.18,
            label="min/max |err|",
        )
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[-1].set_xlabel("Burn-in length [s]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _selected_burnins(df: pd.DataFrame) -> list[float]:
    values = sorted(df["burnin_seconds"].unique())
    selected: list[float] = []
    for window_start, window_end in RERUN_BURNIN_WINDOWS_SECONDS:
        in_window = [
            float(value)
            for value in values
            if float(window_start) <= float(value) <= float(window_end)
        ]
        if not in_window:
            continue
        selected_value = min(in_window, key=lambda value: (abs(value - float(window_end)), value))
        if not any(np.isclose(selected_value, existing) for existing in selected):
            selected.append(selected_value)
    if selected:
        return selected
    if len(values) <= 3:
        return values
    return [values[0], values[len(values) // 2], values[-1]]


def _comparison_window(case: dict[str, np.ndarray | float | str], eval_time: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    time_dim = np.asarray(case["time_dim"], dtype=float)
    y_dim = np.asarray(case["y_dim"], dtype=float)
    dy_dim = np.asarray(case["dy_dim"], dtype=float)
    ddy_dim = np.asarray(case["ddy_dim"], dtype=float)
    t0 = max(float(time_dim[0]), eval_time - float(COMPARISON_WINDOW_SECONDS))
    mask = time_dim >= t0
    return time_dim[mask], y_dim[mask], dy_dim[mask], ddy_dim[mask]


def _rerun_for_selected_burnins(
    *,
    case: dict[str, np.ndarray | float | str],
    params: dict[str, float],
    df: pd.DataFrame,
) -> dict[float, list[tuple[float, dict[str, np.ndarray]]]]:
    selected = _selected_burnins(df)
    resolved_n_memory, resolved_tau_s = _resolve_case_td_memory(case=case, params=params)
    reruns: dict[float, list[tuple[float, dict[str, np.ndarray]]]] = {}
    for burnin in selected:
        subset = df[np.isclose(df["burnin_seconds"], burnin)].sort_values("theta0")
        runs: list[tuple[float, dict[str, np.ndarray]]] = []
        for row in subset.itertuples(index=False):
            _, sim = _run_single_sim(
                case=case,
                params=params,
                start_idx=int(row.start_idx),
                eval_idx=int(row.eval_idx),
                theta0=float(row.theta0),
                n_memory=int(resolved_n_memory),
                tau_s=float(resolved_tau_s),
            )
            runs.append((float(row.theta0), sim))
        reruns[float(burnin)] = runs
    return reruns


def _plot_trajectory_overlay(
    *,
    case: dict[str, np.ndarray | float | str],
    df: pd.DataFrame,
    reruns: dict[float, list[tuple[float, dict[str, np.ndarray]]]],
    out_path: Path,
) -> None:
    eval_time = float(df["eval_time"].iloc[0])
    cfd_t, cfd_y, cfd_dy, cfd_ddy = _comparison_window(case, eval_time)
    burnins = list(reruns.keys())
    fig, axes = plt.subplots(len(burnins), 3, figsize=(15, 4.0 * len(burnins)), sharex=False)
    if len(burnins) == 1:
        axes = np.asarray([axes])

    for row_idx, burnin in enumerate(burnins):
        for col_idx, (series, ylabel) in enumerate(
            [(cfd_y, "y [m]"), (cfd_dy, "dy [m/s]"), (cfd_ddy, "ddy [m/s^2]")]
        ):
            ax = axes[row_idx, col_idx]
            ax.plot(cfd_t, series, color="black", linewidth=2.0, label="CFD")
            for theta0, sim in reruns[burnin]:
                abs_time = df.loc[np.isclose(df["burnin_seconds"], burnin), "start_time"].iloc[0] + sim["time"]
                mask = abs_time >= max(float(abs_time[0]), eval_time - float(COMPARISON_WINDOW_SECONDS))
                if col_idx == 0:
                    values = sim["y"][mask]
                elif col_idx == 1:
                    values = sim["dy"][mask]
                else:
                    values = sim["ddy"][mask]
                ax.plot(abs_time[mask], values, linewidth=1.0, alpha=0.85, label=f"TD theta0={theta0:.2f}")
            ax.axvline(eval_time, color="tab:red", linestyle="--", linewidth=1.0)
            if row_idx == 0:
                ax.set_title(["Displacement", "Velocity", "Acceleration"][col_idx])
            if col_idx == 0:
                ax.set_ylabel(f"Burn-in {burnin:.2f}s\n{ylabel}")
            else:
                ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_phi_trajectories(
    *,
    case: dict[str, np.ndarray | float | str],
    df: pd.DataFrame,
    reruns: dict[float, list[tuple[float, dict[str, np.ndarray]]]],
    out_path: Path,
) -> None:
    eval_time = float(df["eval_time"].iloc[0])
    burnins = list(reruns.keys())
    fig, axes = plt.subplots(len(burnins), 1, figsize=(11, 3.2 * len(burnins)), sharex=False)
    if len(burnins) == 1:
        axes = [axes]

    for ax, burnin in zip(axes, burnins):
        start_time = float(df.loc[np.isclose(df["burnin_seconds"], burnin), "start_time"].iloc[0])
        for theta0, sim in reruns[burnin]:
            abs_time = start_time + sim["time"]
            mask = abs_time >= max(float(abs_time[0]), eval_time - float(COMPARISON_WINDOW_SECONDS))
            theta_series = _compute_theta_series(
                dy=sim["dy"],
                ddy=sim["ddy"],
                phi_vy=sim["phi_vy"],
                sig_dy_loc=sim["sig_dy_loc"],
                sig_ddy_loc=sim["sig_ddy_loc"],
                flow_speed_m_s=float(case["flow_speed_m_s"]),
            )
            ax.plot(
                abs_time[mask],
                theta_series[mask],
                linewidth=1.2,
                label=f"theta0={theta0:.2f}",
            )
        ax.axvline(eval_time, color="tab:red", linestyle="--", linewidth=1.0)
        ax.set_ylabel(f"Burn-in {burnin:.2f}s\nwrapped theta [rad]")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Absolute CFD time [s]")
    axes[0].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_total_predicted_force_trajectories(
    *,
    df: pd.DataFrame,
    reruns: dict[float, list[tuple[float, dict[str, np.ndarray]]]],
    out_path: Path,
) -> None:
    eval_time = float(df["eval_time"].iloc[0])
    burnins = list(reruns.keys())
    fig, axes = plt.subplots(len(burnins), 1, figsize=(11, 3.2 * len(burnins)), sharex=False)
    if len(burnins) == 1:
        axes = [axes]

    for ax, burnin in zip(axes, burnins):
        start_time = float(df.loc[np.isclose(df["burnin_seconds"], burnin), "start_time"].iloc[0])
        for theta0, sim in reruns[burnin]:
            abs_time = start_time + sim["time"]
            mask = abs_time >= max(float(abs_time[0]), eval_time - float(COMPARISON_WINDOW_SECONDS))
            total_predicted_force = sim["Fca"] + sim["Fcv"] + sim["Fdy"]
            ax.plot(
                abs_time[mask],
                total_predicted_force[mask],
                linewidth=1.2,
                label=f"theta0={theta0:.2f}",
            )
        ax.axvline(eval_time, color="tab:red", linestyle="--", linewidth=1.0)
        ax.set_ylabel(f"Burn-in {burnin:.2f}s\npredicted force [N/m]")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Absolute CFD time [s]")
    axes[0].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def load_burnin_scan(param_dir: Path) -> pd.DataFrame:
    csv_path = Path(param_dir) / "burnin_scan.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Burn-in scan CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Burn-in scan CSV is empty: {csv_path}")
    return df.sort_values(["burnin_seconds", "theta0"]).reset_index(drop=True)


def generate_timeseries_outputs(
    *,
    case: dict[str, np.ndarray | float | str],
    params: dict[str, float],
    df: pd.DataFrame,
    out_dir: Path,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reruns = _rerun_for_selected_burnins(case=case, params=params, df=df)
    _plot_trajectory_overlay(
        case=case,
        df=df,
        reruns=reruns,
        out_path=out_dir / "trajectory_overlay_short_mid_long.png",
    )
    _plot_phi_trajectories(
        case=case,
        df=df,
        reruns=reruns,
        out_path=out_dir / "phi_traj_short_mid_long.png",
    )
    _plot_total_predicted_force_trajectories(
        df=df,
        reruns=reruns,
        out_path=out_dir / "predicted_force_traj_short_mid_long.png",
    )


def _write_run_metadata(
    out_path: Path,
    *,
    case: dict[str, np.ndarray | float | str],
    params: dict[str, float],
    num_samples: int,
) -> None:
    resolved_n_memory, resolved_tau_s = _resolve_case_td_memory(case=case, params=params)
    td_memory_cfg = _td_memory_config()
    payload = {
        "case_name": case["case_name"],
        "input_npz": str(case["source_path"]),
        "integrator": INTEGRATOR,
        "sigma_init_mode": SIGMA_INIT_MODE,
        "sigma_init_window_seconds": SIGMA_INIT_WINDOW_SECONDS,
        "comparison_window_seconds": float(COMPARISON_WINDOW_SECONDS),
        "phase_wrap": PHASE_WRAP,
        "td_memory_mode": str(td_memory_cfg["mode"]),
        "td_tau_over_tref": float(td_memory_cfg["tau_over_tref"]),
        "td_memory_tau_s": None if td_memory_cfg["tau_s"] is None else float(td_memory_cfg["tau_s"]),
        "n_memory_fallback": int(N_MEMORY),
        "n_memory": int(resolved_n_memory),
        "tau_s": float(resolved_tau_s),
        "analysis_mode": "time_spread_from_start",
        "num_samples": int(num_samples),
        "theta0_values": [float(x) for x in np.asarray(THETA0_VALUES, dtype=float)],
        "dt_dim": float(case["dt_dim"]),
        "flow_speed_m_s": float(case["flow_speed_m_s"]),
        "stiffness_n_m": float(case["stiffness_n_m"]),
        "dry_mass_kg": float(case["dry_mass_kg"]),
        "diameter_m": float(case["diameter_m"]),
        "rho_kg_m3": float(case["rho_kg_m3"]),
        "force_std_dim": float(case["force_std_dim"]),
        "structural_frequency_hz": case["structural_frequency_hz"],
        "params": params,
    }
    out_path.write_text(json.dumps(payload, indent=2))


def _resolve_input_npzs() -> list[Path]:
    if INPUT_NPZS is not None:
        paths = [Path(path).resolve() for path in INPUT_NPZS]
    else:
        paths = sorted(Path(path).resolve() for path in glob.glob(INPUT_NPZ_GLOB) if Path(path).suffix == ".npz")
    if not paths:
        raise FileNotFoundError("No CFD .npz files selected for burn-in analysis.")
    return paths


def _progress(iterable, *, total: int | None = None, desc: str = ""):
    if SHOW_PROGRESS and tqdm is not None:
        return tqdm(iterable, total=total, desc=desc, leave=False)
    return iterable


def main() -> None:
    if str(INTEGRATOR).lower() not in {"rk4", "rk4_coupled"}:
        raise ValueError("INTEGRATOR must be 'rk4' or 'rk4_coupled'.")
    if SIGMA_INIT_MODE not in {"zero", "local_rms", "lookahead_rms"}:
        raise ValueError("SIGMA_INIT_MODE must be 'zero', 'local_rms', or 'lookahead_rms'.")

    paramsets = _build_paramsets()
    input_npzs = _resolve_input_npzs()
    multi_case_mode = len(input_npzs) > 1
    combined_phase_spread_by_param = {_paramset_id(params): [] for params in paramsets}
    combined_theta_trajectories_by_param = {_paramset_id(params): [] for params in paramsets}

    for npz_path in _progress(input_npzs, total=len(input_npzs), desc="Cases"):
        try:
            case = _load_case(npz_path)
        except Exception as exc:
            if SKIP_INVALID_CASES:
                print(f"Skipping case {npz_path.name}: {exc}")
                continue
            raise

        base_out_dir = OUTPUT_DIR.resolve() / str(case["case_name"])
        base_out_dir.mkdir(parents=True, exist_ok=True)
        if SHOW_PROGRESS:
            print(
                f"Case {case['case_name']}: {len(np.asarray(case['time_dim'], dtype=float))} time samples, "
                f"{len(np.asarray(THETA0_VALUES, dtype=float))} theta0 values, "
                f"{len(paramsets)} parameter set(s)"
            )

        for params in _progress(paramsets, total=len(paramsets), desc=f"{case['case_name']} paramsets"):
            param_dir = base_out_dir / _paramset_id(params)
            if param_dir.exists() and not bool(OVERWRITE):
                raise FileExistsError(f"Output directory already exists: {param_dir}")
            param_dir.mkdir(parents=True, exist_ok=True)

            analysis = _time_spread_analysis(case=case, params=params)
            phase_spread_summary = pd.DataFrame(analysis["summary"]).sort_values("time_from_start_s").reset_index(drop=True)
            theta_traj_df = pd.DataFrame(analysis["theta_trajectories"]).sort_values(["theta0", "time_from_start_s"]).reset_index(drop=True)
            phase_spread_summary.to_csv(param_dir / "phase_spread_vs_time.csv", index=False)
            theta_traj_df.to_csv(param_dir / "theta_vs_time.csv", index=False)
            combined_phase_spread_by_param[_paramset_id(params)].append(phase_spread_summary)
            _plot_time_spread_summary(
                phase_spread_summary,
                out_path=param_dir / "phase_spread_vs_time.png",
            )
            if not (multi_case_mode and MULTI_CASE_ONLY_SPREAD_PLOT):
                _plot_theta_trajectories(
                    time_from_start_s=np.asarray(analysis["time_from_start_s"], dtype=float),
                    theta_stack=np.asarray(analysis["theta_stack"], dtype=float),
                    theta0_values=np.asarray(analysis["theta0_values"], dtype=float),
                    out_path=param_dir / "theta_vs_time.png",
                )
            combined_theta_trajectories_by_param[_paramset_id(params)].append(
                (
                    str(case["case_name"]),
                    np.asarray(analysis["time_from_start_s"], dtype=float),
                    np.asarray(analysis["theta_stack"], dtype=float),
                    np.asarray(analysis["theta0_values"], dtype=float),
                    phase_spread_summary.copy(),
                )
            )
            _write_run_metadata(
                param_dir / "run_metadata.json",
                case=case,
                params=params,
                num_samples=int(np.asarray(case["time_dim"], dtype=float).size),
            )
            print(f"Wrote burn-in analysis to {param_dir}")

    if multi_case_mode:
        combined_root = OUTPUT_DIR.resolve() / "all_cases"
        for params in paramsets:
            param_id = _paramset_id(params)
            summaries = combined_phase_spread_by_param[param_id]
            if summaries:
                _write_combined_time_spread_outputs(
                    summaries=summaries,
                    theta_trajectory_bundles=combined_theta_trajectories_by_param[param_id],
                    out_dir=combined_root / param_id,
                )
                print(f"Wrote combined burn-in spread analysis to {combined_root / param_id}")


if __name__ == "__main__":
    main()
