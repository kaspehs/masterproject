from __future__ import annotations

import importlib
import math
from typing import Any

import numpy as np

FORCE_MAPPING_NRMSE_KEY = "Force mapping NRMSE"
DOMINANT_FREQ_REL_ERROR_KEY = "Dominant frequency relative error"
DISP_STD_REL_ERROR_KEY = "Displacement std relative error"
FORCE_DOMINANT_FREQ_REL_ERROR_KEY = "Force dominant frequency relative error"
FORCE_STD_REL_ERROR_KEY = "Force std relative error"


VALIDATION_ERROR_KEYS = [
    (FORCE_MAPPING_NRMSE_KEY, FORCE_MAPPING_NRMSE_KEY),
    (DISP_STD_REL_ERROR_KEY, DISP_STD_REL_ERROR_KEY),
    (DOMINANT_FREQ_REL_ERROR_KEY, DOMINANT_FREQ_REL_ERROR_KEY),
    (FORCE_DOMINANT_FREQ_REL_ERROR_KEY, FORCE_DOMINANT_FREQ_REL_ERROR_KEY),
    (FORCE_STD_REL_ERROR_KEY, FORCE_STD_REL_ERROR_KEY),
]

RK4_REFERENCE_METRICS = {
    DISP_STD_REL_ERROR_KEY,
    DOMINANT_FREQ_REL_ERROR_KEY,
}

_SIMA_ANALYSIS: Any | None = None
_SIMA_SERIES_CACHE: dict[str, dict | None] = {}
_SIMA_WARNED_CASES: set[str] = set()
DEFAULT_FIG_SAVE_DPI = 300
CFD_COLOR = "black"
BASELINE_COLOR = "0.45"
IMPROVED_COLOR = "0.20"
SIMA_COLOR = "#009E73"
PLOT_FONT_SCALE = 1.0
AXIS_LABEL_FONT_MULTIPLIER = 1.0


def _scaled_fontsize(size: float) -> float:
    return float(size) * PLOT_FONT_SCALE


def _axis_label_fontsize(size: float = 9) -> float:
    return _scaled_fontsize(size) * AXIS_LABEL_FONT_MULTIPLIER


def _get_plt():
    import matplotlib.pyplot as plt

    return plt


def _get_signal():
    import scipy.signal as signal

    return signal


def _apply_thesis_plot_style(plt) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Latin Modern Roman", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
        }
    )


def _get_hnn_metric_helpers():
    from training.training_utils import dominant_frequency, relative_error

    return dominant_frequency, relative_error


def _save_figure(fig, save_path, *, dpi: int = DEFAULT_FIG_SAVE_DPI) -> None:
    if save_path is None:
        return
    from pathlib import Path

    target = Path(save_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=int(dpi), bbox_inches="tight", pad_inches=0.02)


def _series_flow_speed(series: dict, *, expected_length: int | None = None) -> np.ndarray:
    if "flow_speed" in series:
        flow_speed = np.asarray(series["flow_speed"], dtype=float).reshape(-1)
    elif "td_context" in series:
        td_context = np.asarray(series["td_context"], dtype=float)
        if td_context.ndim != 2 or td_context.shape[1] < 5:
            raise ValueError("td_context must have shape (n, >=5) to recover flow speed.")
        flow_speed = np.asarray(td_context[:, 4], dtype=float).reshape(-1)
    else:
        raise KeyError("Series does not contain flow_speed or td_context.")

    if expected_length is not None and flow_speed.size != int(expected_length):
        if flow_speed.size == 1:
            flow_speed = np.full((int(expected_length),), float(flow_speed[0]), dtype=float)
        else:
            raise ValueError(f"Expected flow-speed history of length {expected_length}, got {flow_speed.size}.")
    return flow_speed


def force_per_m_to_coefficient(force_per_m: np.ndarray, *, rho: float, diameter: float, flow_speed: np.ndarray | float) -> np.ndarray:
    force_arr = np.asarray(force_per_m, dtype=float).reshape(-1)
    flow_arr = np.asarray(flow_speed, dtype=float).reshape(-1)
    if flow_arr.size == 1:
        flow_arr = np.full(force_arr.shape, float(flow_arr[0]), dtype=float)
    elif flow_arr.size != force_arr.size:
        raise ValueError(f"Force and flow-speed lengths must match; got {force_arr.size} and {flow_arr.size}.")

    denom = 0.5 * float(rho) * float(diameter) * flow_arr * flow_arr
    coeff = np.full(force_arr.shape, np.nan, dtype=float)
    valid = np.isfinite(force_arr) & np.isfinite(denom) & (np.abs(denom) > 0.0)
    coeff[valid] = force_arr[valid] / denom[valid]
    return coeff


def series_force_coefficient(series: dict, force_per_m: np.ndarray) -> np.ndarray:
    force_arr = np.asarray(force_per_m, dtype=float).reshape(-1)
    flow_speed = _series_flow_speed(series, expected_length=force_arr.size)
    return force_per_m_to_coefficient(
        force_arr,
        rho=float(series["rho"]),
        diameter=float(series["diameter"]),
        flow_speed=flow_speed,
    )


def _force_mapping_nrmse(force_pred: np.ndarray, force_true: np.ndarray) -> float:
    force_pred = np.asarray(force_pred, dtype=float).reshape(-1)
    force_true = np.asarray(force_true, dtype=float).reshape(-1)
    n = min(force_pred.size, force_true.size)
    if n < 1:
        return float("nan")
    pred = force_pred[:n]
    truth = force_true[:n]
    denom = float(np.sqrt(np.mean((truth - np.mean(truth)) ** 2)))
    if not np.isfinite(denom) or denom <= 0.0:
        return float("nan")
    return float(np.sqrt(np.mean((pred - truth) ** 2)) / denom)


def build_dataset_accessors(
    *,
    load_series_fn,
    iter_npz_files_fn,
    iter_all_npz_files_fn,
    split_dirs,
    max_files_per_split,
):
    def iter_npz_files(root, split):
        return iter_npz_files_fn(
            root,
            split,
            split_dirs=split_dirs,
            max_files_per_split=max_files_per_split,
        )

    def iter_all_npz_files(root, split_dirs_override=None):
        active_split_dirs = split_dirs if split_dirs_override is None else split_dirs_override
        return iter_all_npz_files_fn(
            root,
            split_dirs=active_split_dirs,
            max_files_per_split=max_files_per_split,
        )

    return load_series_fn, iter_npz_files, iter_all_npz_files


def resolve_td_params(overrides: dict | None = None) -> dict[str, float]:
    cfg = dict(overrides or {})
    try:
        td_hidden = importlib.import_module("vivana_cfd_data_pipeline.vivana_td.td_hidden_state")
        burnin = importlib.import_module("vivana_cfd_data_pipeline.scripts.analyze_vivana_td_burnin")
    except ModuleNotFoundError:
        td_hidden = importlib.import_module("td_hidden_state")
        burnin = importlib.import_module("analyze_vivana_td_burnin")
    defaults = td_hidden.build_single_paramset_from_burnin_config()
    out = {
        "Cv": float(cfg.get("td_cv", defaults["Cv"])),
        "Cd": float(cfg.get("td_cd", defaults["Cd"])),
        "Ca": float(cfg.get("td_ca", defaults["Ca"])),
        "fhat0": float(cfg.get("td_fhat0", defaults["fhat0"])),
        "fhat_min": float(cfg.get("td_fhat_min", defaults["fhat_min"])),
        "fhat_max": float(cfg.get("td_fhat_max", defaults["fhat_max"])),
        "n_memory": float(cfg.get("td_n_memory", getattr(burnin, "N_MEMORY", 500))),
    }
    if not (out["fhat_min"] <= out["fhat0"] <= out["fhat_max"]):
        raise ValueError("Require td_fhat_min <= td_fhat0 <= td_fhat_max.")
    if out["n_memory"] < 1.0:
        raise ValueError("td_n_memory must be >= 1.")
    return out


def _wrap_angle_rad(angle: float) -> float:
    angle = float(angle)
    return float(math.atan2(math.sin(angle), math.cos(angle)))


def td_baseline_step_numpy(*, velocity, acceleration, td_context, dt, rho, diameter, params):
    ddy = float(td_context[0])
    phi_vy = _wrap_angle_rad(td_context[1])
    sig_dy = float(td_context[2])
    sig_ddy = float(td_context[3])
    flow_speed = float(td_context[4])
    n_memory = max(1.0, float(params["n_memory"]))
    dt = float(dt)
    velocity = float(velocity)
    acceleration = float(acceleration)

    speed_mag = math.sqrt(max(flow_speed * flow_speed + velocity * velocity, 1.0e-12))
    projection = flow_speed / speed_mag
    dy_r = velocity * projection
    ddy_r = ddy * projection

    sig_dy_next = math.sqrt(max(((n_memory - 1.0) / n_memory) * (sig_dy * sig_dy) + (dy_r * dy_r) / n_memory, 1.0e-12))
    sig_ddy_next = math.sqrt(max(((n_memory - 1.0) / n_memory) * (sig_ddy * sig_ddy) + (ddy_r * ddy_r) / n_memory, 1.0e-12))

    cos_phi_dy = dy_r / max(sig_dy_next, 1.0e-12)
    sin_phi_dy = -ddy_r / max(sig_ddy_next, 1.0e-12)
    phi_dy = math.atan2(sin_phi_dy, cos_phi_dy)

    theta = math.atan2(math.sin(phi_dy - phi_vy), math.cos(phi_dy - phi_vy))
    if theta <= 0.0:
        fhat = float(params["fhat0"]) + (float(params["fhat0"]) - float(params["fhat_min"])) * math.sin(theta)
    else:
        fhat = float(params["fhat0"]) + (float(params["fhat_max"]) - float(params["fhat0"])) * math.sin(theta)
    omega_vy = 2.0 * math.pi * fhat * speed_mag / float(diameter)
    phi_vy_next = _wrap_angle_rad(phi_vy + dt * omega_vy)

    fdy = -0.5 * float(rho) * float(diameter) * float(params["Cd"]) * speed_mag * velocity
    fcv = 0.5 * float(rho) * float(diameter) * float(params["Cv"]) * speed_mag * flow_speed * math.cos(phi_vy_next)
    fca = -0.25 * float(rho) * float(params["Ca"]) * math.pi * (float(diameter) ** 2) * acceleration
    force_total = fca + fcv + fdy

    next_context = np.asarray([acceleration, phi_vy_next, sig_dy_next, sig_ddy_next, flow_speed], dtype=float)
    return float(force_total), next_context


def structural_step_constant_force_numpy(*, y, velocity, force, dt, mass, damping_c, stiffness):
    y = float(y)
    velocity = float(velocity)
    force = float(force)
    dt = float(dt)
    mass = float(mass)
    damping_c = float(damping_c)
    stiffness = float(stiffness)

    def accel(y_state, v_state):
        return (force - damping_c * v_state - stiffness * y_state) / mass

    k1_y = velocity
    k1_v = accel(y, velocity)
    y2 = y + 0.5 * dt * k1_y
    v2 = velocity + 0.5 * dt * k1_v
    k2_y = v2
    k2_v = accel(y2, v2)
    y3 = y + 0.5 * dt * k2_y
    v3 = velocity + 0.5 * dt * k2_v
    k3_y = v3
    k3_v = accel(y3, v3)
    y4 = y + dt * k3_y
    v4 = velocity + dt * k3_v
    k4_y = v4
    k4_v = accel(y4, v4)

    y_next = y + (dt / 6.0) * (k1_y + 2.0 * k2_y + 2.0 * k3_y + k4_y)
    v_next = velocity + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
    a_next = accel(y_next, v_next)
    return float(y_next), float(v_next), float(a_next)


def _vivana_coupled_diagnostics(*, y, velocity, phi_vy, q_dy, q_ddy, dt, mass, damping, stiffness, rho, diameter, span, flow_speed, params):
    sig_dy = math.sqrt(max(float(q_dy), 0.0))
    sig_ddy = math.sqrt(max(float(q_ddy), 0.0))
    phi_vy_value = _wrap_angle_rad(phi_vy)
    speed_mag = math.sqrt(max(float(flow_speed) * float(flow_speed) + float(velocity) * float(velocity), 1.0e-12))
    projection = float(flow_speed) / speed_mag
    dy_r = float(velocity) * projection

    force_drag_per_m = -0.5 * float(rho) * float(diameter) * float(params["Cd"]) * speed_mag * float(velocity)
    force_cv_per_m = 0.5 * float(rho) * float(diameter) * float(params["Cv"]) * speed_mag * float(flow_speed) * math.cos(phi_vy_value)
    added_mass_coeff_per_m = 0.25 * float(rho) * float(params["Ca"]) * math.pi * float(diameter) ** 2
    acceleration = (force_cv_per_m + force_drag_per_m - float(damping) * float(velocity) - float(stiffness) * float(y)) / max(float(mass) + added_mass_coeff_per_m, 1.0e-12)
    ddy_r = acceleration * projection

    cos_phi_dy = dy_r / max(sig_dy, 1.0e-12)
    sin_phi_dy = -ddy_r / max(sig_ddy, 1.0e-12)
    phi_dy = math.atan2(sin_phi_dy, cos_phi_dy)
    theta = math.atan2(math.sin(phi_dy - phi_vy_value), math.cos(phi_dy - phi_vy_value))
    if theta <= 0.0:
        fhat = float(params["fhat0"]) + (float(params["fhat0"]) - float(params["fhat_min"])) * math.sin(theta)
    else:
        fhat = float(params["fhat0"]) + (float(params["fhat_max"]) - float(params["fhat0"])) * math.sin(theta)
    omega_vy = 2.0 * math.pi * fhat * speed_mag / float(diameter)

    force_added_mass_per_m = -added_mass_coeff_per_m * acceleration
    force_total_per_m = force_cv_per_m + force_drag_per_m + force_added_mass_per_m
    tau = max(float(params["n_memory"]) * float(dt), 1.0e-12)
    q_dy_dot = (dy_r * dy_r - float(q_dy)) / tau
    q_ddy_dot = (ddy_r * ddy_r - float(q_ddy)) / tau
    return {
        "acceleration": float(acceleration),
        "force_total_per_m": float(force_total_per_m),
        "force_total": float(force_total_per_m * float(span)),
        "omega_vy": float(omega_vy),
        "q_dy_dot": float(q_dy_dot),
        "q_ddy_dot": float(q_ddy_dot),
    }


def _simulate_vivana_rk4_coupled(*, time, initial_displacement, initial_velocity, initial_acceleration, initial_phi_vy, initial_sig_dy, initial_sig_ddy, initial_force_per_m, mass, damping, stiffness, rho, diameter, span, flow_speed, params):
    time_arr = np.asarray(time, dtype=float).reshape(-1)
    if time_arr.size < 2:
        raise ValueError("Need at least two time samples for Vivana RK4-coupled rollout.")
    dt = float(np.median(np.diff(time_arr)))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("Need a positive finite dt for Vivana RK4-coupled rollout.")

    n = time_arr.size
    displacement = np.empty((n,), dtype=float)
    velocity = np.empty((n,), dtype=float)
    acceleration = np.empty((n,), dtype=float)
    force_per_m = np.empty((n,), dtype=float)
    force_total = np.empty((n,), dtype=float)
    phi_vy = np.empty((n,), dtype=float)
    q_dy = np.empty((n,), dtype=float)
    q_ddy = np.empty((n,), dtype=float)

    displacement[0] = float(initial_displacement)
    velocity[0] = float(initial_velocity)
    acceleration[0] = float(initial_acceleration)
    force_per_m[0] = float(initial_force_per_m)
    force_total[0] = float(initial_force_per_m) * float(span)
    phi_vy[0] = _wrap_angle_rad(initial_phi_vy)
    q_dy[0] = max(float(initial_sig_dy) ** 2, 0.0)
    q_ddy[0] = max(float(initial_sig_ddy) ** 2, 0.0)

    for idx in range(n - 1):
        state = np.asarray([displacement[idx], velocity[idx], phi_vy[idx], q_dy[idx], q_ddy[idx]], dtype=float)

        def rhs(state_vec):
            diag = _vivana_coupled_diagnostics(
                y=float(state_vec[0]),
                velocity=float(state_vec[1]),
                phi_vy=float(state_vec[2]),
                q_dy=float(state_vec[3]),
                q_ddy=float(state_vec[4]),
                dt=dt,
                mass=mass,
                damping=damping,
                stiffness=stiffness,
                rho=rho,
                diameter=diameter,
                span=span,
                flow_speed=flow_speed,
                params=params,
            )
            deriv = np.asarray([float(state_vec[1]), diag["acceleration"], diag["omega_vy"], diag["q_dy_dot"], diag["q_ddy_dot"]], dtype=float)
            return deriv, diag

        k1, diag1 = rhs(state)
        k2, _ = rhs(state + 0.5 * dt * k1)
        k3, _ = rhs(state + 0.5 * dt * k2)
        k4, _ = rhs(state + dt * k3)
        next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        next_state[2] = _wrap_angle_rad(next_state[2])
        next_state[3] = max(float(next_state[3]), 0.0)
        next_state[4] = max(float(next_state[4]), 0.0)

        displacement[idx + 1] = float(next_state[0])
        velocity[idx + 1] = float(next_state[1])
        phi_vy[idx + 1] = float(next_state[2])
        q_dy[idx + 1] = float(next_state[3])
        q_ddy[idx + 1] = float(next_state[4])
        next_diag = _vivana_coupled_diagnostics(
            y=float(next_state[0]),
            velocity=float(next_state[1]),
            phi_vy=float(next_state[2]),
            q_dy=float(next_state[3]),
            q_ddy=float(next_state[4]),
            dt=dt,
            mass=mass,
            damping=damping,
            stiffness=stiffness,
            rho=rho,
            diameter=diameter,
            span=span,
            flow_speed=flow_speed,
            params=params,
        )
        acceleration[idx + 1] = next_diag["acceleration"]
        force_per_m[idx + 1] = diag1["force_total_per_m"]
        force_total[idx + 1] = diag1["force_total"]

    return {
        "time": time_arr,
        "displacement": displacement,
        "velocity": velocity,
        "acceleration": acceleration,
        "force_per_m": force_per_m,
        "force_total": force_total,
        "phi_vy": phi_vy,
        "sig_dy": np.sqrt(np.clip(q_dy, a_min=0.0, a_max=None)),
        "sig_ddy": np.sqrt(np.clip(q_ddy, a_min=0.0, a_max=None)),
    }


def _resolve_structural_params(series: dict, *, mass_source: str = "dry") -> tuple[float, float, float]:
    source = str(mass_source).strip().lower()
    if source == "dry":
        mass = float(series["dry_mass"])
    elif source == "effective":
        mass = float(series["effective_mass"])
    else:
        raise ValueError("mass_source must be 'dry' or 'effective'.")
    damping = float(series["damping"])
    stiffness = float(series["stiffness"])
    return mass, damping, stiffness


def simulate_structural_response_with_force_rk4(series: dict, force_series: np.ndarray, *, mass_source: str = "dry") -> dict[str, np.ndarray]:
    time = np.asarray(series["time"], dtype=float)
    displacement_true = np.asarray(series["displacement"], dtype=float)
    velocity_true = np.asarray(series["velocity"], dtype=float)
    applied_force = np.asarray(force_series, dtype=float).reshape(-1)

    mass, damping, stiffness = _resolve_structural_params(series, mass_source=mass_source)
    n = time.size
    if applied_force.size != n:
        raise ValueError(f"Expected force series of length {n}, got {applied_force.size}.")

    displacement = np.empty((n,), dtype=float)
    velocity = np.empty((n,), dtype=float)
    acceleration = np.empty((n,), dtype=float)
    displacement[0] = float(displacement_true[0])
    velocity[0] = float(velocity_true[0])
    acceleration[0] = (applied_force[0] - damping * velocity[0] - stiffness * displacement[0]) / mass

    for idx in range(n - 1):
        dt = float(time[idx + 1] - time[idx])
        y_next, v_next, a_next = structural_step_constant_force_numpy(
            y=displacement[idx],
            velocity=velocity[idx],
            force=applied_force[idx],
            dt=dt,
            mass=mass,
            damping_c=damping,
            stiffness=stiffness,
        )
        displacement[idx + 1] = y_next
        velocity[idx + 1] = v_next
        acceleration[idx + 1] = a_next

    return {"displacement": displacement, "velocity": velocity, "acceleration": acceleration}


def simulate_structural_response_with_force_newmark(series: dict, force_series: np.ndarray, *, mass_source: str = "dry", beta: float = 0.25, gamma: float = 0.5) -> dict[str, np.ndarray]:
    time = np.asarray(series["time"], dtype=float)
    displacement_true = np.asarray(series["displacement"], dtype=float)
    velocity_true = np.asarray(series["velocity"], dtype=float)
    applied_force = np.asarray(force_series, dtype=float).reshape(-1)
    mass, damping, stiffness = _resolve_structural_params(series, mass_source=mass_source)
    n = time.size
    if applied_force.size != n:
        raise ValueError(f"Expected force series of length {n}, got {applied_force.size}.")
    if beta <= 0.0:
        raise ValueError("NEWMARK_BETA must be positive.")

    displacement = np.empty((n,), dtype=float)
    velocity = np.empty((n,), dtype=float)
    acceleration = np.empty((n,), dtype=float)
    displacement[0] = float(displacement_true[0])
    velocity[0] = float(velocity_true[0])
    acceleration[0] = (applied_force[0] - damping * velocity[0] - stiffness * displacement[0]) / mass

    for idx in range(n - 1):
        dt = float(time[idx + 1] - time[idx])
        a0 = 1.0 / (beta * dt * dt)
        a1 = gamma / (beta * dt)
        a2 = 1.0 / (beta * dt)
        a3 = 1.0 / (2.0 * beta) - 1.0
        a4 = gamma / beta - 1.0
        a5 = dt * (gamma / (2.0 * beta) - 1.0)
        k_eff = stiffness + a0 * mass + a1 * damping
        rhs = (
            applied_force[idx + 1]
            + mass * (a0 * displacement[idx] + a2 * velocity[idx] + a3 * acceleration[idx])
            + damping * (a1 * displacement[idx] + a4 * velocity[idx] + a5 * acceleration[idx])
        )
        y_next = rhs / k_eff
        a_next = a0 * (y_next - displacement[idx]) - a2 * velocity[idx] - a3 * acceleration[idx]
        v_next = velocity[idx] + dt * ((1.0 - gamma) * acceleration[idx] + gamma * a_next)
        displacement[idx + 1] = y_next
        velocity[idx + 1] = v_next
        acceleration[idx + 1] = a_next

    return {"displacement": displacement, "velocity": velocity, "acceleration": acceleration}


def simulate_vivana_td(series: dict, *, td_params: dict[str, float], mass_source: str = "dry") -> dict[str, np.ndarray]:
    time = np.asarray(series["time"], dtype=float)
    displacement_true = np.asarray(series["displacement"], dtype=float)
    velocity_true = np.asarray(series["velocity"], dtype=float)
    td_context = np.asarray(series["td_context"], dtype=float)

    source = str(mass_source).strip().lower()
    if source == "dry":
        mass = float(series["dry_mass"])
    elif source == "effective":
        mass = float(series["effective_mass"])
    else:
        raise ValueError("mass_source must be 'dry' or 'effective'.")

    damping = float(series["damping"])
    stiffness = float(series["stiffness"])
    rho = float(series["rho"])
    diameter = float(series["diameter"])
    span = float(series["span"])

    coupled = _simulate_vivana_rk4_coupled(
        time=time,
        initial_displacement=float(displacement_true[0]),
        initial_velocity=float(velocity_true[0]),
        initial_acceleration=float(td_context[0, 0]),
        initial_phi_vy=float(td_context[0, 1]),
        initial_sig_dy=float(td_context[0, 2]),
        initial_sig_ddy=float(td_context[0, 3]),
        initial_force_per_m=float(series["force_td_stored"][0]),
        mass=mass,
        damping=damping,
        stiffness=stiffness,
        rho=rho,
        diameter=diameter,
        span=span,
        flow_speed=float(td_context[0, 4]),
        params=td_params,
    )

    forced_truth_rk4 = simulate_structural_response_with_force_rk4(series, series["force_total"], mass_source=mass_source)
    forced_truth_newmark = simulate_structural_response_with_force_newmark(series, series["force_total"], mass_source=mass_source)
    return {
        "displacement_td": np.asarray(coupled["displacement"], dtype=float),
        "velocity_td": np.asarray(coupled["velocity"], dtype=float),
        "force_td": np.asarray(coupled["force_per_m"], dtype=float),
        "force_td_total": np.asarray(coupled["force_total"], dtype=float),
        "displacement_rk4_truth_force": forced_truth_rk4["displacement"],
        "velocity_rk4_truth_force": forced_truth_rk4["velocity"],
        "acceleration_rk4_truth_force": forced_truth_rk4["acceleration"],
        "displacement_newmark_truth_force": forced_truth_newmark["displacement"],
        "velocity_newmark_truth_force": forced_truth_newmark["velocity"],
        "acceleration_newmark_truth_force": forced_truth_newmark["acceleration"],
    }


def _get_sima_analysis():
    global _SIMA_ANALYSIS
    if _SIMA_ANALYSIS is None:
        _SIMA_ANALYSIS = importlib.import_module("vivana_cfd_data_pipeline.analysis.compare_sima_vs_vivana_td")
    return _SIMA_ANALYSIS


def _resolve_sima_condition_set(series: dict) -> str | None:
    sima_analysis = _get_sima_analysis()
    legacy_stiffness = {
        condition_set: float(stiffness_per_python * 4.0)
        for condition_set, stiffness_per_python in sima_analysis.PYTHON_STIFFNESS_BY_RUN.items()
    }

    case_name = str(series["name"]).strip()
    if case_name.startswith("ConditionSet_"):
        return case_name

    stiffness = float(series.get("stiffness", float("nan")))
    if np.isfinite(stiffness):
        condition_set, reference_stiffness = min(
            legacy_stiffness.items(),
            key=lambda item: abs(float(item[1]) - stiffness),
        )
        tolerance = max(0.5, 0.05 * abs(float(reference_stiffness)))
        if abs(float(reference_stiffness) - stiffness) <= tolerance:
            return str(condition_set)

    ur = float(series.get("ur", float("nan")))
    if np.isfinite(ur):
        target_by_condition = {
            "ConditionSet_1": 2.0,
            "ConditionSet_2": 4.0,
            "ConditionSet_3": 5.0,
            "ConditionSet_4": 5.75,
            "ConditionSet_5": 7.0,
            "ConditionSet_6": 8.0,
            "ConditionSet_7": 10.0,
        }
        condition_set, reference_ur = min(target_by_condition.items(), key=lambda item: abs(float(item[1]) - ur))
        if abs(float(reference_ur) - ur) <= 0.3:
            return str(condition_set)

    return None


def load_sima_series_for_npz(series: dict) -> dict | None:
    sima_analysis = _get_sima_analysis()
    case_name = str(series["name"])
    if case_name in _SIMA_SERIES_CACHE:
        return _SIMA_SERIES_CACHE[case_name]

    condition_set = _resolve_sima_condition_set(series)
    if condition_set is None:
        if case_name not in _SIMA_WARNED_CASES:
            print(f"Skipping SIMA overlay for {case_name}: could not resolve a matching ConditionSet from the NPZ stiffness/U_r.")
            _SIMA_WARNED_CASES.add(case_name)
        _SIMA_SERIES_CACHE[case_name] = None
        return None

    try:
        previous_window = (
            float(sima_analysis.SIM_START_TIME_S),
            float(sima_analysis.HIDDEN_STATE_BURNIN_SECONDS),
            sima_analysis.SIM_DURATION_S,
        )
        sima_analysis.SIM_START_TIME_S = 0.0
        sima_analysis.HIDDEN_STATE_BURNIN_SECONDS = 0.0
        sima_analysis.SIM_DURATION_S = None
        with sima_analysis.h5py.File(sima_analysis.SIMA_H5_PATH, "r") as handle:
            sima_case = sima_analysis._load_sima_case(
                handle,
                sima_analysis.SIMA_H5_PATH,
                {"condition_set": condition_set, "case_name": case_name},
            )
            python_params = sima_analysis._build_python_params(sima_case)
            sima_forces = sima_analysis._load_sima_force_rollout(handle, sima_case, python_params)
    except Exception as exc:
        if case_name not in _SIMA_WARNED_CASES:
            print(f"Skipping SIMA overlay for {case_name}: {exc}")
            _SIMA_WARNED_CASES.add(case_name)
        _SIMA_SERIES_CACHE[case_name] = None
        return None
    finally:
        if "previous_window" in locals():
            sima_analysis.SIM_START_TIME_S = previous_window[0]
            sima_analysis.HIDDEN_STATE_BURNIN_SECONDS = previous_window[1]
            sima_analysis.SIM_DURATION_S = previous_window[2]

    reference_flow_speed = _series_flow_speed(series)
    finite_reference_speed = reference_flow_speed[np.isfinite(reference_flow_speed)]
    sima_flow_speed_value = float(np.median(finite_reference_speed)) if finite_reference_speed.size else float("nan")
    sima_time = np.asarray(sima_case.time, dtype=float)

    sima_series = {
        "name": case_name,
        "condition_set": condition_set,
        "time": sima_time,
        "displacement": np.asarray(sima_case.y, dtype=float),
        "velocity": np.asarray(sima_case.dy, dtype=float),
        "acceleration": np.asarray(sima_case.ddy, dtype=float),
        "force_per_m": np.asarray(sima_forces["hydrodynamic_total"], dtype=float),
        "force_reported_per_m": np.asarray(sima_forces["hydrodynamic_total"], dtype=float),
        "force_added_mass_per_m": np.asarray(sima_forces.get("added_mass_per_m", np.zeros_like(sima_forces["hydrodynamic_total"])), dtype=float),
        "force_morison_per_m": np.asarray(sima_forces["morison_total"], dtype=float),
        "force_cross_flow_per_m": np.asarray(sima_forces["cross_flow_total"], dtype=float),
        "flow_speed": np.full(sima_time.shape, sima_flow_speed_value, dtype=float),
        "rho": float(series["rho"]),
        "diameter": float(series["diameter"]),
        "span": float(series["span"]),
        "ur_effective": float(series["ur_effective"]),
    }
    _SIMA_SERIES_CACHE[case_name] = sima_series
    return sima_series


def compute_psd_welch(time: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    signal = _get_signal()
    time = np.asarray(time, dtype=float).reshape(-1)
    values = np.asarray(values, dtype=float).reshape(-1)
    n = min(time.size, values.size)
    if n < 4:
        raise ValueError("Need at least four samples to compute a PSD.")
    time = time[:n]
    values = values[:n]
    dt = float(np.median(np.diff(time)))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("Need a positive finite dt to compute a PSD.")
    fs = 1.0 / dt
    values_centered = values - float(np.mean(values))
    nperseg = n
    noverlap = 0 if nperseg < 16 else min(nperseg // 2, nperseg - 1)
    nfft = max(8 * nperseg, nperseg)
    freqs, psd = signal.welch(
        values_centered,
        fs=fs,
        window="boxcar",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend="constant",
        scaling="density",
    )
    return freqs, psd


def suggest_psd_xlim(*freq_psd_pairs: tuple[np.ndarray, np.ndarray], harmonics: float = 2.0, min_xmax: float = 0.1) -> float:
    dominant_freqs = []
    nyquist_candidates = []
    for freqs, psd in freq_psd_pairs:
        freqs = np.asarray(freqs, dtype=float).reshape(-1)
        psd = np.asarray(psd, dtype=float).reshape(-1)
        n = min(freqs.size, psd.size)
        if n < 2:
            continue
        freqs = freqs[:n]
        psd = psd[:n]
        positive = freqs > 0.0
        if not np.any(positive):
            continue
        freqs_pos = freqs[positive]
        psd_pos = psd[positive]
        dominant_freqs.append(float(freqs_pos[int(np.argmax(psd_pos))]))
        nyquist_candidates.append(float(freqs[-1]))
    if not dominant_freqs:
        return float(min_xmax)
    dominant = max(dominant_freqs)
    nyquist = max(nyquist_candidates) if nyquist_candidates else dominant * harmonics
    xmax = max(min_xmax, harmonics * dominant)
    return float(min(xmax, nyquist))


def normalize_psd_area(freqs: np.ndarray, psd: np.ndarray) -> np.ndarray:
    freqs = np.asarray(freqs, dtype=float).reshape(-1)
    psd = np.asarray(psd, dtype=float).reshape(-1)
    n = min(freqs.size, psd.size)
    if n < 2:
        return psd[:n]
    freqs = freqs[:n]
    psd = np.clip(psd[:n], a_min=0.0, a_max=None)
    area = float(np.trapz(psd, freqs))
    if not np.isfinite(area) or area <= 0.0:
        return psd
    return psd / area


def dominant_frequency_from_signal(time: np.ndarray, values: np.ndarray, *, reference_frequency_hz: float | None = None, reference_peak_min_relative_height: float = 0.85) -> float:
    del reference_frequency_hz, reference_peak_min_relative_height
    time = np.asarray(time, dtype=float).reshape(-1)
    values = np.asarray(values, dtype=float).reshape(-1)
    n = min(time.size, values.size)
    if n < 4:
        return float("nan")
    time = time[:n]
    values = values[:n]
    dt = float(np.median(np.diff(time)))
    if not np.isfinite(dt) or dt <= 0.0:
        return float("nan")
    centered = values - float(np.mean(values))
    if np.allclose(centered, 0.0):
        return float("nan")

    freqs = np.fft.rfftfreq(centered.size, d=dt)
    power = np.abs(np.fft.rfft(centered)) ** 2
    if freqs.size < 2 or power.size < 2:
        return float("nan")
    power = np.asarray(power, dtype=float)
    power[0] = 0.0
    positive = np.isfinite(freqs) & np.isfinite(power) & (freqs > 0.0)
    if np.count_nonzero(positive) < 1:
        return float("nan")
    valid_indices = np.flatnonzero(positive)
    peak_index = int(valid_indices[int(np.argmax(power[positive]))])
    peak_power = float(power[peak_index])
    if not np.isfinite(peak_power) or peak_power <= 0.0:
        return float("nan")

    interpolated_index = float(peak_index)
    if 1 <= peak_index < (power.size - 1):
        y_prev = float(power[peak_index - 1])
        y_peak = float(power[peak_index])
        y_next = float(power[peak_index + 1])
        denom = y_prev - 2.0 * y_peak + y_next
        if np.isfinite(denom) and abs(denom) > 1.0e-18:
            delta = 0.5 * (y_prev - y_next) / denom
            if np.isfinite(delta):
                interpolated_index += float(np.clip(delta, -1.0, 1.0))

    df = float(freqs[1] - freqs[0])
    if not np.isfinite(df) or df <= 0.0:
        return float(freqs[peak_index])
    dominant_frequency_hz = interpolated_index * df
    return float(max(dominant_frequency_hz, 0.0))


def displacement_peak_amplitudes(time: np.ndarray, displacement: np.ndarray, dominant_frequency_hz: float) -> np.ndarray:
    signal = _get_signal()
    time = np.asarray(time, dtype=float).reshape(-1)
    displacement = np.asarray(displacement, dtype=float).reshape(-1)
    abs_disp = np.abs(displacement)
    if abs_disp.size == 0:
        return np.asarray([], dtype=float)
    dt = float(np.median(np.diff(time))) if time.size >= 2 else 1.0
    if np.isfinite(dominant_frequency_hz) and dominant_frequency_hz > 0.0 and np.isfinite(dt) and dt > 0.0:
        period = 1.0 / dominant_frequency_hz
        min_distance = max(1, int(round(0.35 * period / dt)))
    else:
        min_distance = 1
    prominence = max(1.0e-12, 0.05 * float(np.std(abs_disp)))
    peaks, _ = signal.find_peaks(abs_disp, distance=min_distance, prominence=prominence)
    amplitudes = abs_disp[peaks]
    if amplitudes.size == 0:
        amplitudes = np.asarray([float(np.max(abs_disp))], dtype=float)
    return amplitudes.astype(float)


def instantaneous_phase_lag_deg_samples(time: np.ndarray, reference: np.ndarray, target: np.ndarray, frequency_hz: float) -> np.ndarray:
    signal = _get_signal()
    time = np.asarray(time, dtype=float).reshape(-1)
    reference = np.asarray(reference, dtype=float).reshape(-1)
    target = np.asarray(target, dtype=float).reshape(-1)
    n = min(time.size, reference.size, target.size)
    if n < 16 or not np.isfinite(frequency_hz) or frequency_hz <= 0.0:
        return np.asarray([], dtype=float)

    time = time[:n]
    reference = reference[:n] - float(np.mean(reference[:n]))
    target = target[:n] - float(np.mean(target[:n]))
    dt = float(np.median(np.diff(time)))
    if not np.isfinite(dt) or dt <= 0.0:
        return np.asarray([], dtype=float)
    sample_rate_hz = 1.0 / dt
    nyquist_hz = 0.5 * sample_rate_hz

    low_hz = max(0.05, 0.65 * float(frequency_hz))
    high_hz = min(0.95 * nyquist_hz, 1.35 * float(frequency_hz))
    if high_hz <= low_hz:
        low_hz = max(0.05, 0.50 * float(frequency_hz))
        high_hz = min(0.95 * nyquist_hz, 1.50 * float(frequency_hz))
    if high_hz <= low_hz:
        return np.asarray([], dtype=float)

    try:
        sos = signal.butter(4, [low_hz, high_hz], btype="bandpass", fs=sample_rate_hz, output="sos")
        reference_filtered = signal.sosfiltfilt(sos, reference)
        target_filtered = signal.sosfiltfilt(sos, target)
    except ValueError:
        return np.asarray([], dtype=float)

    reference_analytic = signal.hilbert(reference_filtered)
    target_analytic = signal.hilbert(target_filtered)
    reference_amp = np.abs(reference_analytic)
    target_amp = np.abs(target_analytic)
    reference_threshold = max(1.0e-12, 0.05 * float(np.sqrt(np.mean(reference_amp ** 2))))
    target_threshold = max(1.0e-12, 0.05 * float(np.sqrt(np.mean(target_amp ** 2))))

    lag_rad = np.angle(target_analytic) - np.angle(reference_analytic)
    lag_deg = np.degrees(np.angle(np.exp(1j * lag_rad)))

    edge_trim = int(np.ceil(sample_rate_hz / max(float(frequency_hz), 1.0e-9)))
    mask = np.isfinite(lag_deg) & np.isfinite(reference_amp) & np.isfinite(target_amp)
    mask &= reference_amp >= reference_threshold
    mask &= target_amp >= target_threshold
    if 2 * edge_trim < n:
        mask[:edge_trim] = False
        mask[-edge_trim:] = False
    return np.asarray(lag_deg[mask], dtype=float)


def compute_summary_metrics(time: np.ndarray, displacement: np.ndarray, velocity: np.ndarray, force: np.ndarray, *, stiffness: float, effective_mass: float) -> dict:
    time = np.asarray(time, dtype=float).reshape(-1)
    displacement = np.asarray(displacement, dtype=float).reshape(-1)
    velocity = np.asarray(velocity, dtype=float).reshape(-1)
    force = np.asarray(force, dtype=float).reshape(-1)
    force_dominant_frequency_hz = dominant_frequency_from_signal(time, force)
    dominant_frequency_hz = dominant_frequency_from_signal(time, displacement, reference_frequency_hz=force_dominant_frequency_hz)
    if np.isfinite(stiffness) and stiffness > 0.0 and np.isfinite(effective_mass) and effective_mass > 0.0:
        natural_frequency_hz = float(np.sqrt(stiffness / effective_mass) / (2.0 * np.pi))
        dominant_frequency_ratio = float(dominant_frequency_hz / natural_frequency_hz) if natural_frequency_hz > 0.0 else float("nan")
        force_dominant_frequency_ratio = float(force_dominant_frequency_hz / natural_frequency_hz) if natural_frequency_hz > 0.0 else float("nan")
    else:
        natural_frequency_hz = float("nan")
        dominant_frequency_ratio = float("nan")
        force_dominant_frequency_ratio = float("nan")
    return {
        "disp_std": float(np.std(displacement)),
        "force_std": float(np.std(force)),
        "dominant_frequency_hz": float(dominant_frequency_hz),
        "force_dominant_frequency_hz": float(force_dominant_frequency_hz),
        "natural_frequency_hz": float(natural_frequency_hz),
        "dominant_frequency_ratio": float(dominant_frequency_ratio),
        "force_dominant_frequency_ratio": float(force_dominant_frequency_ratio),
        "peak_amplitudes": displacement_peak_amplitudes(time, displacement, dominant_frequency_hz),
        "force_peak_amplitudes": displacement_peak_amplitudes(time, force, force_dominant_frequency_hz),
        "phase_force_displacement_deg_samples": instantaneous_phase_lag_deg_samples(time, displacement, force, dominant_frequency_hz),
        "phase_force_velocity_deg_samples": instantaneous_phase_lag_deg_samples(time, velocity, force, dominant_frequency_hz),
    }


def global_generation_grid(summary_series: list[dict], *, transient_seconds: float) -> tuple[float, float, np.ndarray, np.ndarray]:
    if not summary_series:
        raise ValueError("Need at least one CFD series to build the common Vivana-TD generation grid.")
    dt_values = []
    durations = []
    for series in summary_series:
        time = np.asarray(series["time"], dtype=float).reshape(-1)
        if time.size < 2:
            continue
        dt_values.append(float(np.min(np.diff(time))))
        durations.append(float(time[-1] - time[0]))
    if not dt_values or not durations:
        raise ValueError("Could not determine generation dt/duration from the CFD series.")
    generation_dt = float(np.min(dt_values))
    generation_duration_s = float(np.max(durations) + transient_seconds)
    n_steps = int(np.ceil(generation_duration_s / generation_dt)) + 1
    time_full = generation_dt * np.arange(n_steps, dtype=float)
    keep_mask = time_full >= transient_seconds
    return generation_dt, generation_duration_s, time_full, keep_mask


def resolve_td_params_for_dt(params: dict[str, float], *, dt: float, td_memory_tau_s: float | str | None, flow_speed: float | None = None, diameter: float | None = None) -> dict[str, float]:
    resolved = dict(params)
    if td_memory_tau_s is None:
        return resolved
    dt_value = float(dt)
    if not np.isfinite(dt_value) or dt_value <= 0.0:
        raise ValueError("dt must be positive and finite when resolving n_memory from tau.")
    if isinstance(td_memory_tau_s, str):
        tau_mode = td_memory_tau_s.strip().lower()
        if tau_mode == "auto":
            tau_over_tref = 2.0
        elif tau_mode.startswith("auto:"):
            tau_over_tref = float(tau_mode.split(":", 1)[1])
        elif tau_mode.startswith("tau_over_tref:"):
            tau_over_tref = float(tau_mode.split(":", 1)[1])
        else:
            raise ValueError("td_memory_tau_s must be None, a positive number, 'auto', or 'tau_over_tref:<value>'.")
        flow_speed_value = float(flow_speed) if flow_speed is not None else float("nan")
        diameter_value = float(diameter) if diameter is not None else float("nan")
        fhat0 = float(params.get("fhat0", float("nan")))
        if not np.isfinite(flow_speed_value) or abs(flow_speed_value) <= 0.0:
            raise ValueError("Need a finite non-zero flow speed to resolve td_memory_tau_s from tau/T_ref.")
        if not np.isfinite(diameter_value) or diameter_value <= 0.0:
            raise ValueError("Need a finite positive diameter to resolve td_memory_tau_s from tau/T_ref.")
        if not np.isfinite(fhat0) or fhat0 <= 0.0:
            raise ValueError("Need a finite positive fhat0 to resolve td_memory_tau_s from tau/T_ref.")
        if not np.isfinite(tau_over_tref) or tau_over_tref <= 0.0:
            raise ValueError("tau/T_ref must be positive and finite when td_memory_tau_s is string-configured.")
        tau_value = tau_over_tref / (fhat0 * abs(flow_speed_value) / diameter_value)
    else:
        tau_value = float(td_memory_tau_s)
    if not np.isfinite(tau_value) or tau_value <= 0.0:
        raise ValueError("td_memory_tau_s must be positive and finite when provided.")
    resolved["n_memory"] = max(1.0, tau_value / dt_value)
    return resolved


def generate_vivana_summary_rollout(template: dict, target_ur: float, *, generation_dt: float, generation_duration_s: float, transient_seconds: float, td_params: dict[str, float], mass_source: str = "dry", initial_state: dict | None = None, td_memory_tau_s: float | str | None = None) -> dict:
    flow_speed_hist = np.asarray(template["td_context"][:, 4], dtype=float).reshape(-1)
    finite_speed = flow_speed_hist[np.isfinite(flow_speed_hist)]
    if finite_speed.size == 0:
        raise ValueError(f"Template {template['name']} does not have a valid flow-speed history.")
    target_ur = float(target_ur)
    if not np.isfinite(target_ur) or target_ur <= 0.0:
        raise ValueError("target_ur must be positive and finite.")

    effective_mass = float(template["effective_mass"])
    diameter = float(template["diameter"])
    rho = float(template["rho"])
    damping = float(template["damping"])
    dry_mass = float(template["dry_mass"])
    span = float(template["span"])
    if not (np.isfinite(effective_mass) and effective_mass > 0.0 and np.isfinite(diameter) and diameter > 0.0):
        raise ValueError(f"Template {template['name']} is missing valid effective-mass or diameter data.")

    flow_speed_const = float(np.median(finite_speed))
    target_natural_frequency_hz = flow_speed_const / (target_ur * diameter)
    target_stiffness = effective_mass * (2.0 * np.pi * target_natural_frequency_hz) ** 2

    source = str(mass_source).strip().lower()
    if source == "dry":
        mass = dry_mass
    elif source == "effective":
        mass = effective_mass
    else:
        raise ValueError("mass_source must be 'dry' or 'effective'.")

    td_params_effective = resolve_td_params_for_dt(
        td_params,
        dt=generation_dt,
        td_memory_tau_s=td_memory_tau_s,
        flow_speed=flow_speed_const,
        diameter=diameter,
    )

    n_steps = int(np.ceil(generation_duration_s / generation_dt)) + 1
    time_full = generation_dt * np.arange(n_steps, dtype=float)
    displacement = np.empty((n_steps,), dtype=float)
    velocity = np.empty((n_steps,), dtype=float)
    acceleration = np.empty((n_steps,), dtype=float)
    force_per_m = np.empty((n_steps,), dtype=float)
    force_total = np.empty((n_steps,), dtype=float)
    flow_speed = np.full((n_steps,), flow_speed_const, dtype=float)
    if initial_state is None:
        displacement[0] = float(template["displacement"][0])
        velocity[0] = float(template["velocity"][0])
        ctx = np.asarray(template["td_context"][0], dtype=float).reshape(-1)[:5].copy()
        force_per_m[0] = float(template["force_td_stored"][0])
    else:
        displacement[0] = float(initial_state["displacement"])
        velocity[0] = float(initial_state["velocity"])
        ctx = np.asarray(initial_state["td_context"], dtype=float).reshape(-1)[:5].copy()
        force_per_m[0] = float(initial_state.get("force", template["force_td_stored"][0]))
    ctx[4] = flow_speed_const
    acceleration[0] = float(ctx[0])
    force_total[0] = force_per_m[0] * float(span)

    for idx in range(n_steps - 1):
        td_force_next, ctx_next = td_baseline_step_numpy(
            velocity=velocity[idx],
            acceleration=acceleration[idx],
            td_context=ctx,
            dt=generation_dt,
            rho=rho,
            diameter=diameter,
            params=td_params_effective,
        )
        y_next, v_next, a_next = structural_step_constant_force_numpy(
            y=displacement[idx],
            velocity=velocity[idx],
            force=td_force_next,
            dt=generation_dt,
            mass=mass,
            damping_c=damping,
            stiffness=target_stiffness,
        )
        displacement[idx + 1] = y_next
        velocity[idx + 1] = v_next
        acceleration[idx + 1] = a_next
        force_per_m[idx + 1] = float(td_force_next)
        force_total[idx + 1] = float(td_force_next) * float(span)
        ctx = np.asarray(ctx_next, dtype=float).reshape(-1)[:5].copy()
        ctx[0] = a_next

    keep_mask = time_full >= transient_seconds
    if np.count_nonzero(keep_mask) < 4:
        raise ValueError("Trimmed Vivana-TD rollout is too short to analyze.")

    return {
        "time": time_full[keep_mask] - transient_seconds,
        "displacement": displacement[keep_mask],
        "velocity": velocity[keep_mask],
        "force": force_per_m[keep_mask],
        "force_coefficient": force_per_m_to_coefficient(
            force_per_m[keep_mask],
            rho=rho,
            diameter=diameter,
            flow_speed=flow_speed[keep_mask],
        ),
        "force_total": force_total[keep_mask],
        "flow_speed": flow_speed[keep_mask],
        "rho": float(rho),
        "diameter": float(diameter),
        "stiffness": float(target_stiffness),
        "effective_mass": float(effective_mass),
        "reference_mass": float(mass),
        "ur_effective": float(target_ur),
        "final_state": {
            "displacement": float(displacement[-1]),
            "velocity": float(velocity[-1]),
            "force": float(force_per_m[-1]),
            "td_context": np.asarray(ctx, dtype=float).reshape(-1)[:5].copy(),
        },
    }


def sorted_group_stats(grouped: dict[float, list[float]]) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    urs = np.asarray(sorted(grouped.keys()), dtype=float)
    values = [np.asarray(grouped[float(ur)], dtype=float) for ur in urs]
    means = np.asarray([float(np.mean(v)) for v in values], dtype=float)
    return urs, means, values


def _normalize_grouped_plot_specs(
    grouped_or_specs,
    *,
    default_label: str,
    default_alpha: float,
    default_size: float,
    default_marker: str | None,
    default_linewidth: float = 1.8,
) -> list[dict[str, object]]:
    palette = [
        BASELINE_COLOR,
        IMPROVED_COLOR,
        "tab:pink",
        "tab:cyan",
        "tab:olive",
        "tab:gray",
    ]

    def _is_grouped_dict(candidate) -> bool:
        if not isinstance(candidate, dict):
            return False
        if not candidate:
            return True
        sample_key = next(iter(candidate.keys()))
        return isinstance(sample_key, (float, int, np.floating, np.integer))

    if _is_grouped_dict(grouped_or_specs):
        return [
            {
                "grouped": grouped_or_specs,
                "label": default_label,
                "color": palette[0],
                "alpha": float(default_alpha),
                "size": float(default_size),
                "marker": default_marker,
                "show_points": True,
                "linewidth": float(default_linewidth),
                "linestyle": "-",
            }
        ]

    specs: list[dict[str, object]] = []
    if grouped_or_specs is None:
        return specs
    for idx, raw_spec in enumerate(list(grouped_or_specs)):
        if isinstance(raw_spec, dict) and "grouped" in raw_spec:
            grouped = dict(raw_spec.get("grouped", {}))
            label = str(raw_spec.get("label", f"{default_label} {idx + 1}"))
            color = str(raw_spec.get("color", palette[idx % len(palette)]))
            alpha = float(raw_spec.get("alpha", default_alpha))
            size = float(raw_spec.get("size", default_size))
            marker = raw_spec.get("marker", default_marker)
            show_points = bool(raw_spec.get("show_points", True))
            linewidth = float(raw_spec.get("linewidth", default_linewidth))
            linestyle = str(raw_spec.get("linestyle", "-"))
            specs.append(
                {
                    "grouped": grouped,
                    "label": label,
                    "color": color,
                    "alpha": alpha,
                    "size": size,
                    "marker": marker,
                    "show_points": show_points,
                    "linewidth": linewidth,
                    "linestyle": linestyle,
                    "markerfacecolor": raw_spec.get("markerfacecolor"),
                    "markeredgecolor": raw_spec.get("markeredgecolor"),
                    "markeredgewidth": float(raw_spec.get("markeredgewidth", 0.8)),
                    "plot_band": bool(raw_spec.get("plot_band", True)),
                    "plot_mean": bool(raw_spec.get("plot_mean", True)),
                    "plot_box": bool(raw_spec.get("plot_box", True)),
                    "show_box_mean": bool(raw_spec.get("show_box_mean", True)),
                    "use_anchor_urs": bool(raw_spec.get("use_anchor_urs", True)),
                }
            )
        else:
            raise ValueError("Grouped plot specs must be a grouped dict or a list of dicts with a 'grouped' key.")
    return specs


def plot_scalar_metric(split: str, metric_name: str, ylabel: str, cfd_grouped: dict, viv_grouped: dict, sima_grouped: dict | None = None, *legacy_args, vivana_fine_ur_step: float = 0.1, vivana_summary_mass_source: str = "dry", save_path=None, save_dpi: int = DEFAULT_FIG_SAVE_DPI):
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(12, 5.5))
    fig.suptitle(
        f"{split} | {metric_name} vs U_r | CFD/SIMA x-axis uses matched-case U_r | simulation mass={vivana_summary_mass_source} | normalization mass=effective",
        fontsize=_scaled_fontsize(14),
    )

    def _plot_grouped(
        grouped: dict,
        *,
        color: str,
        label: str,
        alpha: float,
        size: float,
        marker: str | None,
        show_points: bool = True,
        linewidth: float = 1.8,
        linestyle: str = "-",
        markerfacecolor: str | None = None,
        markeredgecolor: str | None = None,
        markeredgewidth: float = 0.8,
    ) -> None:
        urs, means, values = sorted_group_stats(grouped)
        if urs.size == 0:
            return
        if show_points:
            for ur, series_values in zip(urs, values):
                scatter_kwargs = {
                    "marker": marker or "o",
                    "alpha": alpha,
                    "s": size,
                    "linewidths": markeredgewidth,
                }
                if markerfacecolor == "none":
                    scatter_kwargs["facecolors"] = "none"
                    scatter_kwargs["edgecolors"] = markeredgecolor or color
                else:
                    scatter_kwargs["color"] = markerfacecolor or color
                    scatter_kwargs["edgecolors"] = markeredgecolor or color
                ax.scatter(np.full(series_values.shape, ur), series_values, **scatter_kwargs)
        plot_kwargs = {"color": color, "linewidth": linewidth, "label": label, "linestyle": linestyle}
        if marker is not None:
            plot_kwargs["marker"] = marker
            plot_kwargs["markersize"] = max(3.0, 0.45 * float(size) ** 0.5)
            plot_kwargs["markeredgewidth"] = markeredgewidth
            plot_kwargs["markeredgecolor"] = markeredgecolor or color
            plot_kwargs["markerfacecolor"] = markerfacecolor or color
        ax.plot(urs, means, **plot_kwargs)

    _plot_grouped(
        cfd_grouped,
        color=CFD_COLOR,
        label="CFD",
        alpha=0.35,
        size=18,
        marker="o",
        linestyle="none",
        linewidth=0.0,
        markerfacecolor="none",
        markeredgecolor="black",
        markeredgewidth=0.8,
    )
    if sima_grouped:
        _plot_grouped(sima_grouped, color=SIMA_COLOR, label="SIMA", alpha=0.28, size=18, marker="s")
    # Ignore legacy positional RK4 input from older notebook cells.
    if legacy_args:
        pass
    for viv_spec in _normalize_grouped_plot_specs(
        viv_grouped,
        default_label=f"Vivana-TD (U_r step={vivana_fine_ur_step:g})",
        default_alpha=0.20,
        default_size=12,
        default_marker=None,
    ):
        _plot_grouped(
            viv_spec["grouped"],
            color=str(viv_spec["color"]),
            label=str(viv_spec["label"]),
            alpha=float(viv_spec["alpha"]),
            size=float(viv_spec["size"]),
            marker=viv_spec["marker"],
            show_points=bool(viv_spec.get("show_points", True)),
            linewidth=float(viv_spec["linewidth"]),
            linestyle=str(viv_spec["linestyle"]),
            markerfacecolor=viv_spec.get("markerfacecolor"),
            markeredgecolor=viv_spec.get("markeredgecolor"),
            markeredgewidth=float(viv_spec.get("markeredgewidth", 0.8)),
        )

    ax.set_ylabel(ylabel, fontsize=_axis_label_fontsize())
    ax.set_xlabel(r"$U_r$", fontsize=_axis_label_fontsize())
    ax.tick_params(axis="both", labelsize=_scaled_fontsize(10))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=_scaled_fontsize(10))
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)
    _save_figure(fig, save_path, dpi=save_dpi)
    plt.show()


def _plot_scalar_metric_axis(
    ax,
    *,
    metric_name: str,
    ylabel: str,
    cfd_grouped: dict,
    viv_grouped,
    sima_grouped: dict | None = None,
    vivana_fine_ur_step: float = 0.1,
    show_xlabel: bool = True,
):
    def _plot_grouped(
        grouped: dict,
        *,
        color: str,
        label: str,
        alpha: float,
        size: float,
        marker: str | None,
        show_points: bool = True,
        plot_mean: bool = True,
        linewidth: float = 1.8,
        linestyle: str = "-",
        markerfacecolor: str | None = None,
        markeredgecolor: str | None = None,
        markeredgewidth: float = 0.8,
    ) -> None:
        urs, means, values = sorted_group_stats(grouped)
        if urs.size == 0:
            return
        if show_points:
            for point_idx, (ur, series_values) in enumerate(zip(urs, values)):
                scatter_kwargs = {
                    "marker": marker or "o",
                    "alpha": alpha,
                    "s": size,
                    "linewidths": markeredgewidth,
                    "label": label if point_idx == 0 and not plot_mean else None,
                }
                if markerfacecolor == "none":
                    scatter_kwargs["facecolors"] = "none"
                    scatter_kwargs["edgecolors"] = markeredgecolor or color
                else:
                    scatter_kwargs["color"] = markerfacecolor or color
                    scatter_kwargs["edgecolors"] = markeredgecolor or color
                ax.scatter(np.full(series_values.shape, ur), series_values, **scatter_kwargs)
        if not plot_mean:
            return
        plot_kwargs = {"color": color, "linewidth": linewidth, "label": label, "linestyle": linestyle}
        if marker is not None:
            plot_kwargs["marker"] = marker
            plot_kwargs["markersize"] = max(3.0, 0.45 * float(size) ** 0.5)
            plot_kwargs["markeredgewidth"] = markeredgewidth
            plot_kwargs["markeredgecolor"] = markeredgecolor or color
            plot_kwargs["markerfacecolor"] = markerfacecolor or color
        ax.plot(urs, means, **plot_kwargs)

    _plot_grouped(
        cfd_grouped,
        color=CFD_COLOR,
        label="CFD",
        alpha=1.0,
        size=32,
        marker="o",
        show_points=True,
        plot_mean=False,
        markerfacecolor="none",
        markeredgecolor="black",
        markeredgewidth=1.1,
    )
    if sima_grouped:
        _plot_grouped(sima_grouped, color=SIMA_COLOR, label="SIMA", alpha=0.28, size=18, marker="s")
    for viv_spec in _normalize_grouped_plot_specs(
        viv_grouped,
        default_label=f"Vivana-TD (U_r step={vivana_fine_ur_step:g})",
        default_alpha=0.20,
        default_size=12,
        default_marker=None,
    ):
        _plot_grouped(
            viv_spec["grouped"],
            color=str(viv_spec["color"]),
            label=str(viv_spec["label"]),
            alpha=float(viv_spec["alpha"]),
            size=float(viv_spec["size"]),
            marker=viv_spec["marker"],
            show_points=bool(viv_spec.get("show_points", True)),
            plot_mean=bool(viv_spec.get("plot_mean", True)),
            linewidth=float(viv_spec["linewidth"]),
            linestyle=str(viv_spec["linestyle"]),
            markerfacecolor=viv_spec.get("markerfacecolor"),
            markeredgecolor=viv_spec.get("markeredgecolor"),
            markeredgewidth=float(viv_spec.get("markeredgewidth", 0.8)),
        )

    ax.set_ylabel(ylabel, fontsize=_axis_label_fontsize())
    ax.set_xlabel(r"$U_r$" if show_xlabel else "", fontsize=_axis_label_fontsize())
    ax.tick_params(axis="both", labelsize=_scaled_fontsize(8), labelbottom=show_xlabel)
    ax.grid(True, alpha=0.22, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_scalar_metric_grid(split: str, metric_specs: list[dict], *, vivana_fine_ur_step: float = 0.1, vivana_summary_mass_source: str = "dry", figsize: tuple[float, float] = (5.85, 6.3), save_path=None, save_dpi: int = DEFAULT_FIG_SAVE_DPI):
    del split, vivana_summary_mass_source
    plt = _get_plt()
    _apply_thesis_plot_style(plt)
    if len(metric_specs) != 4:
        raise ValueError("plot_scalar_metric_grid expects exactly 4 metric specs.")

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    x_axis_specs = []
    for metric_spec in metric_specs:
        x_axis_specs.append({"grouped": metric_spec["cfd_grouped"]})
        if metric_spec.get("sima_grouped"):
            x_axis_specs.append({"grouped": metric_spec["sima_grouped"]})
        viv_specs = _normalize_grouped_plot_specs(
            metric_spec["viv_grouped"],
            default_label="",
            default_alpha=0.0,
            default_size=0.0,
            default_marker=None,
        )
        x_axis_specs.extend(viv_specs)
    x_axis_urs = _grouped_ur_values_from_specs(x_axis_specs)

    legend_handles = None
    legend_labels = None
    for idx, (ax, metric_spec) in enumerate(zip(np.asarray(axes).reshape(-1), metric_specs)):
        _plot_scalar_metric_axis(
            ax,
            metric_name=str(metric_spec["metric_name"]),
            ylabel=str(metric_spec["ylabel"]),
            cfd_grouped=dict(metric_spec["cfd_grouped"]),
            viv_grouped=metric_spec["viv_grouped"],
            sima_grouped=metric_spec.get("sima_grouped"),
            vivana_fine_ur_step=vivana_fine_ur_step,
            show_xlabel=idx == len(metric_specs) - 1,
        )
        panel_label = str(metric_spec.get("panel_label", f"({chr(ord('a') + idx)})"))
        ax.text(0.015, 0.95, panel_label, transform=ax.transAxes, ha="left", va="top", fontsize=_scaled_fontsize(9))
        _apply_reduced_velocity_x_axis(ax, x_axis_urs)
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    if legend_handles and legend_labels:
        fig.legend(legend_handles, legend_labels, loc="upper center", ncol=max(1, len(legend_labels)), fontsize=_scaled_fontsize(8), frameon=False)
    plt.tight_layout()
    fig.subplots_adjust(top=0.91, hspace=0.10)
    _save_figure(fig, save_path, dpi=save_dpi)
    plt.show()


def boxplot_width(urs: np.ndarray, fallback: float) -> float:
    urs = np.asarray(urs, dtype=float)
    if urs.size >= 2:
        return max(0.02, 0.55 * float(np.min(np.diff(urs))))
    return float(fallback)


def _grouped_ur_values_from_specs(specs: list[dict], *, anchor_urs: np.ndarray | None = None) -> np.ndarray:
    values = []
    if anchor_urs is not None:
        values.extend(np.asarray(anchor_urs, dtype=float).reshape(-1).tolist())
    for spec in specs:
        grouped = dict(spec.get("grouped", {}))
        values.extend(float(key) for key in grouped.keys())
    values = np.asarray(values, dtype=float)
    return np.unique(values[np.isfinite(values)])


def _apply_reduced_velocity_x_axis(ax, ur_values: np.ndarray) -> None:
    from matplotlib import ticker as mticker

    ur_values = np.asarray(ur_values, dtype=float).reshape(-1)
    ur_values = ur_values[np.isfinite(ur_values)]
    if ur_values.size:
        x_min = float(np.min(ur_values))
        x_max = float(np.max(ur_values))
        span = max(x_max - x_min, 1.0)
        margin = 0.05 * span
        ax.set_xlim(x_min - margin, x_max + margin)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6))


def _nearest_grouped_values(grouped: dict, anchor_urs: np.ndarray | None = None) -> tuple[np.ndarray, list[np.ndarray]]:
    keys = np.asarray(sorted(grouped.keys()), dtype=float)
    if keys.size == 0:
        return keys, []
    if anchor_urs is None:
        selected_urs = keys
        selected_keys = keys
    else:
        selected_urs = np.asarray(anchor_urs, dtype=float)
        selected_keys = np.asarray([keys[int(np.argmin(np.abs(keys - ur)))] for ur in selected_urs], dtype=float)
    values = [
        np.asarray(grouped[float(key)], dtype=float)[np.isfinite(grouped[float(key)])]
        for key in selected_keys
    ]
    return selected_urs, values


def _plot_peak_boxplot_axis(ax, specs: list[dict], *, ylabel: str, panel_label: str, anchor_urs: np.ndarray | None = None) -> tuple[list, list[str]]:
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    active_specs = []
    for spec in specs:
        grouped = dict(spec.get("grouped", {}))
        if grouped:
            active_specs.append({**spec, "grouped": grouped})
    if not active_specs:
        return [], []

    reference_urs = np.asarray(anchor_urs if anchor_urs is not None else sorted(active_specs[0]["grouped"].keys()), dtype=float)
    if reference_urs.size >= 2:
        min_spacing = float(np.min(np.diff(np.unique(reference_urs))))
    else:
        min_spacing = 1.0
    box_width = 0.12 * min_spacing
    box_spec_indices = [idx for idx, spec in enumerate(active_specs) if bool(spec.get("plot_box", True))]
    box_offsets = {}
    if box_spec_indices:
        offsets = (np.arange(len(box_spec_indices), dtype=float) - 0.5 * (len(box_spec_indices) - 1)) * 1.35 * box_width
        box_offsets = {spec_idx: float(offset) for spec_idx, offset in zip(box_spec_indices, offsets)}

    legend_handles = []
    legend_labels = []
    for spec_idx, spec in enumerate(active_specs):
        color = str(spec.get("color", "0.3"))
        facecolor = str(spec.get("facecolor", color))
        box_alpha = float(spec.get("box_alpha", 0.28 if facecolor != "none" else 1.0))
        markerfacecolor = str(spec.get("markerfacecolor", color))
        markeredgecolor = str(spec.get("markeredgecolor", color))
        label = str(spec.get("label", f"series {spec_idx + 1}"))
        plot_box = bool(spec.get("plot_box", True))
        spec_anchor_urs = reference_urs if bool(spec.get("use_anchor_urs", True)) else None
        urs, values = _nearest_grouped_values(spec["grouped"], spec_anchor_urs)
        positions = []
        box_values = []
        for ur, vals in zip(urs, values):
            vals = np.asarray(vals, dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            position_offset = float(box_offsets.get(spec_idx, 0.0)) if plot_box else 0.0
            positions.append(float(ur) + position_offset)
            box_values.append(vals)
        if not box_values:
            continue
        if plot_box:
            ax.boxplot(
                box_values,
                positions=positions,
                widths=box_width,
                patch_artist=True,
                manage_ticks=False,
                showfliers=False,
                boxprops={"facecolor": facecolor, "edgecolor": color, "alpha": box_alpha, "linewidth": 0.9},
                whiskerprops={"color": color, "linewidth": 0.8},
                capprops={"color": color, "linewidth": 0.8},
                medianprops={"color": "black", "linewidth": 0.9},
            )
        if bool(spec.get("show_box_mean", True)):
            means = np.asarray([float(np.mean(vals)) for vals in box_values], dtype=float)
            ax.plot(
                positions,
                means,
                color=color,
                linewidth=float(spec.get("linewidth", 1.2)),
                linestyle=str(spec.get("linestyle", "-")),
                marker=spec.get("marker", None),
                markersize=3.0,
                markerfacecolor=markerfacecolor,
                markeredgecolor=markeredgecolor,
                markeredgewidth=float(spec.get("markeredgewidth", 0.8)),
            )
        if plot_box:
            legend_handles.append(mpatches.Patch(facecolor=facecolor, edgecolor=color, alpha=box_alpha))
        else:
            legend_handles.append(
                Line2D([0], [0], color=color, linestyle=str(spec.get("linestyle", "-")), linewidth=float(spec.get("linewidth", 1.2)))
            )
        legend_labels.append(label)

    ax.set_ylabel(ylabel, fontsize=_axis_label_fontsize())
    ax.tick_params(axis="both", labelsize=_scaled_fontsize(8))
    ax.grid(True, alpha=0.22, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0.015, 0.95, panel_label, transform=ax.transAxes, ha="left", va="top", fontsize=_scaled_fontsize(9))
    _apply_reduced_velocity_x_axis(ax, _grouped_ur_values_from_specs(active_specs, anchor_urs=reference_urs))
    return legend_handles, legend_labels


def plot_peak_amplitude_boxplot_grid(displacement_specs: list[dict], force_specs: list[dict], *, anchor_urs: np.ndarray | None = None, figsize: tuple[float, float] = (5.85, 4.0), save_path=None, save_dpi: int = DEFAULT_FIG_SAVE_DPI):
    plt = _get_plt()
    _apply_thesis_plot_style(plt)
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    disp_handles, disp_labels = _plot_peak_boxplot_axis(
        axes[0],
        displacement_specs,
        ylabel=r"Peak $|y/D|$",
        panel_label="(a)",
        anchor_urs=anchor_urs,
    )
    _plot_peak_boxplot_axis(
        axes[1],
        force_specs,
        ylabel=r"Peak $|C_F|$",
        panel_label="(b)",
        anchor_urs=anchor_urs,
    )
    axes[0].set_xlabel("")
    axes[1].set_xlabel(r"$U_r$", fontsize=_axis_label_fontsize())
    if disp_handles and disp_labels:
        fig.legend(disp_handles, disp_labels, loc="upper center", ncol=max(1, len(disp_labels)), fontsize=_scaled_fontsize(8), frameon=False)
    plt.tight_layout()
    fig.subplots_adjust(top=0.86, hspace=0.12)
    _save_figure(fig, save_path, dpi=save_dpi)
    plt.show()


def plot_peak_boxplot(split: str, cfd_grouped: dict, viv_grouped: dict, sima_grouped: dict | None = None, *legacy_args, vivana_fine_ur_step: float = 0.1, vivana_summary_mass_source: str = "dry"):
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.suptitle(
        f"{split} | Displacement peak amplitude vs reduced velocity | CFD/SIMA x-axis uses matched-case U_r | simulation mass={vivana_summary_mass_source} | normalization mass=effective",
        fontsize=14,
    )

    def _plot_peak_summary(grouped: dict, *, color: str, label: str, marker: str | None, linewidth: float = 2.0, linestyle: str = "-") -> None:
        urs, _, values = sorted_group_stats(grouped)
        if urs.size == 0:
            return
        mean_vals = np.asarray([float(np.mean(v)) for v in values], dtype=float)
        min_vals = np.asarray([float(np.min(v)) for v in values], dtype=float)
        max_vals = np.asarray([float(np.max(v)) for v in values], dtype=float)
        std_vals = np.asarray([float(np.std(v)) for v in values], dtype=float)
        ax.fill_between(urs, min_vals, max_vals, color=color, alpha=0.10)
        ax.fill_between(urs, mean_vals - std_vals, mean_vals + std_vals, color=color, alpha=0.22)
        plot_kwargs = {"color": color, "linewidth": linewidth, "label": label, "linestyle": linestyle}
        if marker is not None:
            plot_kwargs["marker"] = marker
            plot_kwargs["markersize"] = 5
        ax.plot(urs, mean_vals, **plot_kwargs)

    _plot_peak_summary(cfd_grouped, color="tab:blue", label="CFD", marker="o")
    if sima_grouped:
        _plot_peak_summary(sima_grouped, color="tab:green", label="SIMA", marker="s")
    # Ignore legacy positional RK4 input from older notebook cells.
    if legacy_args:
        pass
    for viv_spec in _normalize_grouped_plot_specs(
        viv_grouped,
        default_label=f"Vivana-TD (U_r step={vivana_fine_ur_step:g})",
        default_alpha=0.20,
        default_size=12,
        default_marker=None,
        default_linewidth=2.0,
    ):
        _plot_peak_summary(
            viv_spec["grouped"],
            color=str(viv_spec["color"]),
            label=str(viv_spec["label"]),
            marker=viv_spec["marker"],
            linewidth=float(viv_spec["linewidth"]),
            linestyle=str(viv_spec["linestyle"]),
        )

    ax.set_ylabel("Peak |displacement|")
    ax.set_xlabel("Reduced velocity, U_r")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)
    plt.show()


def plot_phase_lag_summary(split: str, cfd_force_disp_grouped: dict, viv_force_disp_grouped: dict, cfd_force_vel_grouped: dict, viv_force_vel_grouped: dict, sima_force_disp_grouped: dict | None = None, sima_force_vel_grouped: dict | None = None, rk4_force_disp_grouped: dict | None = None, rk4_force_vel_grouped: dict | None = None, *, vivana_fine_ur_step: float = 0.1, vivana_summary_mass_source: str = "dry"):
    plt = _get_plt()
    fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=True)
    fig.suptitle(
        f"{split} | Phase lag vs reduced velocity | CFD/SIMA x-axis uses matched-case U_r | simulation mass={vivana_summary_mass_source} | normalization mass=effective",
        fontsize=14,
    )

    def _wrap_phase_deg(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        return ((values + 180.0) % 360.0) - 180.0

    def _circular_summary(grouped: dict):
        urs = np.asarray(sorted(grouped.keys()), dtype=float)
        if urs.size == 0:
            empty = np.asarray([])
            return urs, empty, empty, empty, empty, empty
        mean_vals = []
        min_vals = []
        max_vals = []
        lower_std = []
        upper_std = []
        for ur in urs:
            vals = _wrap_phase_deg(np.asarray(grouped[float(ur)], dtype=float).reshape(-1))
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                mean_vals.append(float("nan"))
                min_vals.append(float("nan"))
                max_vals.append(float("nan"))
                lower_std.append(float("nan"))
                upper_std.append(float("nan"))
                continue
            angles = np.deg2rad(vals)
            resultant = np.mean(np.exp(1j * angles))
            mean_angle = float(np.angle(resultant))
            mean_deg = float(np.rad2deg(mean_angle))
            deltas_deg = np.rad2deg(np.angle(np.exp(1j * (angles - mean_angle))))
            resultant_length = float(np.clip(np.abs(resultant), 1.0e-12, 1.0))
            std_deg = float(np.rad2deg(np.sqrt(max(0.0, -2.0 * np.log(resultant_length)))))
            mean_vals.append(mean_deg)
            min_vals.append(float(np.clip(mean_deg + np.min(deltas_deg), -180.0, 180.0)))
            max_vals.append(float(np.clip(mean_deg + np.max(deltas_deg), -180.0, 180.0)))
            lower_std.append(float(np.clip(mean_deg - std_deg, -180.0, 180.0)))
            upper_std.append(float(np.clip(mean_deg + std_deg, -180.0, 180.0)))
        return (
            urs,
            np.asarray(mean_vals, dtype=float),
            np.asarray(min_vals, dtype=float),
            np.asarray(max_vals, dtype=float),
            np.asarray(lower_std, dtype=float),
            np.asarray(upper_std, dtype=float),
        )

    def _plot_phase_summary(ax, grouped: dict, *, color: str, label: str, marker: str | None, linewidth: float = 2.0, linestyle: str = "-") -> None:
        urs, mean_vals, min_vals, max_vals, lower_std, upper_std = _circular_summary(grouped)
        if urs.size == 0:
            return
        mask = np.isfinite(mean_vals) & np.isfinite(min_vals) & np.isfinite(max_vals) & np.isfinite(lower_std) & np.isfinite(upper_std)
        urs = urs[mask]
        mean_vals = mean_vals[mask]
        min_vals = min_vals[mask]
        max_vals = max_vals[mask]
        lower_std = lower_std[mask]
        upper_std = upper_std[mask]
        if urs.size == 0:
            return
        ax.fill_between(urs, min_vals, max_vals, color=color, alpha=0.10)
        ax.fill_between(urs, lower_std, upper_std, color=color, alpha=0.22)
        plot_kwargs = {"color": color, "linewidth": linewidth, "label": label, "linestyle": linestyle}
        if marker is not None:
            plot_kwargs["marker"] = marker
            plot_kwargs["markersize"] = 5
        ax.plot(urs, mean_vals, **plot_kwargs)

    axes[0].set_title("Force-displacement phase lag")
    _plot_phase_summary(axes[0], cfd_force_disp_grouped, color="tab:blue", label="CFD", marker="o")
    if sima_force_disp_grouped:
        _plot_phase_summary(axes[0], sima_force_disp_grouped, color="tab:green", label="SIMA", marker="s")
    if rk4_force_disp_grouped:
        _plot_phase_summary(axes[0], rk4_force_disp_grouped, color="tab:red", label="RK4 + ground-truth force", marker="^")
    for viv_spec in _normalize_grouped_plot_specs(
        viv_force_disp_grouped,
        default_label=f"Vivana-TD (U_r step={vivana_fine_ur_step:g})",
        default_alpha=0.20,
        default_size=12,
        default_marker=None,
        default_linewidth=2.0,
    ):
        _plot_phase_summary(
            axes[0],
            viv_spec["grouped"],
            color=str(viv_spec["color"]),
            label=str(viv_spec["label"]),
            marker=viv_spec["marker"],
            linewidth=float(viv_spec["linewidth"]),
            linestyle=str(viv_spec["linestyle"]),
        )
    axes[0].set_ylabel("Phase lag [deg]")
    axes[0].set_ylim(-180.0, 180.0)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].set_title("Force-velocity phase lag")
    _plot_phase_summary(axes[1], cfd_force_vel_grouped, color="tab:blue", label="CFD", marker="o")
    if sima_force_vel_grouped:
        _plot_phase_summary(axes[1], sima_force_vel_grouped, color="tab:green", label="SIMA", marker="s")
    if rk4_force_vel_grouped:
        _plot_phase_summary(axes[1], rk4_force_vel_grouped, color="tab:red", label="RK4 + ground-truth force", marker="^")
    for viv_spec in _normalize_grouped_plot_specs(
        viv_force_vel_grouped,
        default_label=f"Vivana-TD (U_r step={vivana_fine_ur_step:g})",
        default_alpha=0.20,
        default_size=12,
        default_marker=None,
        default_linewidth=2.0,
    ):
        _plot_phase_summary(
            axes[1],
            viv_spec["grouped"],
            color=str(viv_spec["color"]),
            label=str(viv_spec["label"]),
            marker=viv_spec["marker"],
            linewidth=float(viv_spec["linewidth"]),
            linestyle=str(viv_spec["linestyle"]),
        )
    axes[1].set_ylabel("Phase lag [deg]")
    axes[1].set_xlabel("Reduced velocity, U_r")
    axes[1].set_ylim(-180.0, 180.0)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best")
    plt.tight_layout()
    fig.subplots_adjust(top=0.92)
    plt.show()


def compute_validation_style_error_metrics(*, time: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, force_true: np.ndarray | None = None, force_pred: np.ndarray | None = None) -> dict[str, float]:
    dominant_frequency, relative_error = _get_hnn_metric_helpers()
    time = np.asarray(time, dtype=float).reshape(-1)
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    force_true_arr = None if force_true is None else np.asarray(force_true, dtype=float).reshape(-1)
    force_pred_arr = None if force_pred is None else np.asarray(force_pred, dtype=float).reshape(-1)

    min_len_y = min(y_true.size, y_pred.size)
    min_len_force = 0 if force_true_arr is None or force_pred_arr is None else min(force_true_arr.size, force_pred_arr.size)
    dt = float(np.median(np.diff(time))) if time.size >= 2 else float("nan")

    metrics = {
        FORCE_MAPPING_NRMSE_KEY: float("nan"),
        DISP_STD_REL_ERROR_KEY: float("nan"),
        DOMINANT_FREQ_REL_ERROR_KEY: float("nan"),
        FORCE_DOMINANT_FREQ_REL_ERROR_KEY: float("nan"),
        FORCE_STD_REL_ERROR_KEY: float("nan"),
    }

    if min_len_force >= 1:
        metrics[FORCE_MAPPING_NRMSE_KEY] = float(_force_mapping_nrmse(force_pred_arr[:min_len_force], force_true_arr[:min_len_force]))

    if min_len_y > 1 and np.isfinite(dt) and dt > 0.0:
        y_true_aligned = y_true[:min_len_y]
        y_pred_aligned = y_pred[:min_len_y]
        true_dom = dominant_frequency(y_true_aligned, dt)
        pred_dom = dominant_frequency(y_pred_aligned, dt)
        dom_rel = relative_error(pred_dom, true_dom)
        if np.isfinite(dom_rel):
            metrics[DOMINANT_FREQ_REL_ERROR_KEY] = abs(float(dom_rel))

        true_std = float(np.std(y_true_aligned))
        pred_std = float(np.std(y_pred_aligned))
        std_rel = relative_error(pred_std, true_std)
        if np.isfinite(std_rel):
            metrics[DISP_STD_REL_ERROR_KEY] = abs(float(std_rel))

    if min_len_force > 1 and np.isfinite(dt) and dt > 0.0:
        force_true_aligned = force_true_arr[:min_len_force]
        force_pred_aligned = force_pred_arr[:min_len_force]
        force_true_dom = dominant_frequency(force_true_aligned, dt)
        force_pred_dom = dominant_frequency(force_pred_aligned, dt)
        force_dom_rel = relative_error(force_pred_dom, force_true_dom)
        if np.isfinite(force_dom_rel):
            metrics[FORCE_DOMINANT_FREQ_REL_ERROR_KEY] = abs(float(force_dom_rel))

        force_true_std = float(np.std(force_true_aligned))
        force_pred_std = float(np.std(force_pred_aligned))
        force_std_rel = relative_error(force_pred_std, force_true_std)
        if np.isfinite(force_std_rel):
            metrics[FORCE_STD_REL_ERROR_KEY] = abs(float(force_std_rel))

    return metrics


def _group_plot_stats(grouped_errors: dict[float, list[float]]):
    urs, _, values = sorted_group_stats(grouped_errors)
    if urs.size == 0:
        raise ValueError("No finite grouped values available to plot.")
    mean_vals = np.asarray([float(np.mean(v)) for v in values], dtype=float)
    min_vals = np.asarray([float(np.min(v)) for v in values], dtype=float)
    max_vals = np.asarray([float(np.max(v)) for v in values], dtype=float)
    std_vals = np.asarray([float(np.std(v)) for v in values], dtype=float)
    positive_values = np.concatenate([arr[np.isfinite(arr) & (arr > 0.0)] for arr in values if arr.size > 0]) if values else np.asarray([], dtype=float)
    return urs, mean_vals, min_vals, max_vals, std_vals, positive_values


def _plot_validation_error_band_axis(ax, metric_label: str, grouped_errors, *legacy_args, show_xlabel: bool = True):
    if legacy_args:
        pass

    error_specs = _normalize_grouped_plot_specs(
        grouped_errors,
        default_label="Vivana-TD baseline",
        default_alpha=0.20,
        default_size=12,
        default_marker="o",
        default_linewidth=2.0,
    )

    positive_arrays: list[np.ndarray] = []
    error_plot_stats: list[dict[str, object]] = []
    for error_spec in error_specs:
        urs, mean_vals, min_vals, max_vals, std_vals, positive_values = _group_plot_stats(error_spec["grouped"])
        error_plot_stats.append(
            {
                "urs": urs,
                "mean_vals": mean_vals,
                "min_vals": min_vals,
                "max_vals": max_vals,
                "std_vals": std_vals,
                "label": str(error_spec["label"]),
                "color": str(error_spec["color"]),
                "marker": error_spec["marker"],
                "linewidth": float(error_spec["linewidth"]),
                "linestyle": str(error_spec["linestyle"]),
                "alpha": float(error_spec.get("alpha", 0.20)),
                "size": float(error_spec.get("size", 12.0)),
                "markerfacecolor": error_spec.get("markerfacecolor"),
                "markeredgecolor": error_spec.get("markeredgecolor"),
                "markeredgewidth": float(error_spec.get("markeredgewidth", 0.8)),
                "plot_band": bool(error_spec.get("plot_band", True)),
                "plot_mean": bool(error_spec.get("plot_mean", True)),
                "show_points": bool(error_spec.get("show_points", False)),
            }
        )
        if positive_values.size > 0:
            positive_arrays.append(positive_values)

    if not error_plot_stats:
        ax.set_xlabel(r"$U_r$" if show_xlabel else "", fontsize=_axis_label_fontsize())
        ax.set_ylabel(metric_label, fontsize=_axis_label_fontsize())
        ax.tick_params(axis="both", labelsize=_scaled_fontsize(10), labelbottom=show_xlabel)
        ax.grid(True, alpha=0.25)
        ax.text(0.5, 0.5, "No finite values", transform=ax.transAxes, ha="center", va="center", fontsize=_scaled_fontsize(11))
        return

    merged_positive = np.concatenate(positive_arrays) if positive_arrays else np.asarray([], dtype=float)
    floor = float(max(1.0e-12, 0.5 * np.min(merged_positive))) if merged_positive.size > 0 else 1.0e-12

    for stats in error_plot_stats:
        if stats["show_points"]:
            grouped = next(
                (spec["grouped"] for spec in error_specs if str(spec["label"]) == stats["label"]),
                {},
            )
            for point_idx, ur in enumerate(stats["urs"]):
                vals = np.asarray(grouped.get(float(ur), []), dtype=float).reshape(-1)
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue
                scatter_kwargs = {
                    "marker": stats["marker"] or "o",
                    "alpha": stats["alpha"],
                    "s": stats["size"],
                    "linewidths": stats["markeredgewidth"],
                    "label": stats["label"] if point_idx == 0 and not stats["plot_mean"] else None,
                }
                if stats["markerfacecolor"] == "none":
                    scatter_kwargs["facecolors"] = "none"
                    scatter_kwargs["edgecolors"] = stats["markeredgecolor"] or stats["color"]
                else:
                    scatter_kwargs["color"] = stats["markerfacecolor"] or stats["color"]
                    scatter_kwargs["edgecolors"] = stats["markeredgecolor"] or stats["color"]
                ax.scatter(np.full(vals.shape, ur), np.maximum(vals, floor), **scatter_kwargs)
        if stats["plot_band"]:
            ax.fill_between(
                stats["urs"],
                np.maximum(stats["min_vals"], floor),
                np.maximum(stats["max_vals"], floor),
                color=stats["color"],
                alpha=0.10,
            )
            ax.fill_between(
                stats["urs"],
                np.maximum(stats["mean_vals"] - stats["std_vals"], floor),
                np.maximum(stats["mean_vals"] + stats["std_vals"], floor),
                color=stats["color"],
                alpha=0.22,
            )
        if not stats["plot_mean"]:
            continue
        plot_kwargs = {
            "color": stats["color"],
            "linewidth": stats["linewidth"],
            "linestyle": stats["linestyle"],
            "label": stats["label"],
        }
        if stats["marker"] is not None:
            plot_kwargs["marker"] = stats["marker"]
            plot_kwargs["markersize"] = max(3.0, float(stats["size"]) ** 0.5)
            plot_kwargs["markeredgewidth"] = stats["markeredgewidth"]
            plot_kwargs["markeredgecolor"] = stats["markeredgecolor"] or stats["color"]
            plot_kwargs["markerfacecolor"] = stats["markerfacecolor"] or stats["color"]
        ax.plot(stats["urs"], np.maximum(stats["mean_vals"], floor), **plot_kwargs)

    ax.set_xlabel(r"$U_r$" if show_xlabel else "", fontsize=_axis_label_fontsize())
    ax.set_ylabel(metric_label, fontsize=_axis_label_fontsize())
    ax.tick_params(axis="both", labelsize=_scaled_fontsize(8), labelbottom=show_xlabel)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.22, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_validation_error_band(title_prefix: str, metric_label: str, grouped_errors, *legacy_args, save_path=None, save_dpi: int = DEFAULT_FIG_SAVE_DPI):
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(12, 5.5))
    fig.suptitle(f"{title_prefix} | {metric_label} vs U_r", fontsize=_scaled_fontsize(14))
    _plot_validation_error_band_axis(ax, metric_label, grouped_errors, *legacy_args)
    ax.legend(loc="best", fontsize=_scaled_fontsize(10))
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)
    _save_figure(fig, save_path, dpi=save_dpi)
    plt.show()


def plot_validation_error_band_grid(title_prefix: str, metric_specs: list[dict], *, figsize: tuple[float, float] = (5.85, 6.3), save_path=None, save_dpi: int = DEFAULT_FIG_SAVE_DPI):
    del title_prefix
    plt = _get_plt()
    _apply_thesis_plot_style(plt)
    if len(metric_specs) != 4:
        raise ValueError("plot_validation_error_band_grid expects exactly 4 metric specs.")

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

    legend_handles = None
    legend_labels = None
    for idx, (ax, metric_spec) in enumerate(zip(np.asarray(axes).reshape(-1), metric_specs)):
        _plot_validation_error_band_axis(
            ax,
            str(metric_spec.get("ylabel", metric_spec["metric_label"])),
            metric_spec["grouped_errors"],
            show_xlabel=idx == len(metric_specs) - 1,
        )
        panel_label = str(metric_spec.get("panel_label", f"({chr(ord('a') + idx)})"))
        ax.text(0.015, 0.95, panel_label, transform=ax.transAxes, ha="left", va="top", fontsize=_scaled_fontsize(9))
        if legend_handles is None:
            handles, labels = ax.get_legend_handles_labels()
            if handles and labels:
                legend_handles, legend_labels = handles, labels

    if legend_handles and legend_labels:
        fig.legend(legend_handles, legend_labels, loc="upper center", ncol=max(1, len(legend_labels)), fontsize=_scaled_fontsize(8), frameon=False)
    plt.tight_layout()
    fig.subplots_adjust(top=0.91, hspace=0.10)
    _save_figure(fig, save_path, dpi=save_dpi)
    plt.show()
