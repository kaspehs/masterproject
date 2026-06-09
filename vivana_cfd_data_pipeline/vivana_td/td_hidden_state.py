from __future__ import annotations

import importlib
from itertools import product

import numpy as np

try:
    from vivana_td_model import (
        compute_theta_series as shared_compute_theta_series,
        replay_prescribed_motion as shared_replay_prescribed_motion,
        wrap_phase as shared_wrap_phase,
    )
except ModuleNotFoundError:
    from vivana_cfd_data_pipeline.vivana_td.vivana_td_model import (
        compute_theta_series as shared_compute_theta_series,
        replay_prescribed_motion as shared_replay_prescribed_motion,
        wrap_phase as shared_wrap_phase,
    )


def _as_list(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)
    return [value]


def _format_float(value: float) -> str:
    text = f"{float(value):.6g}"
    return text.replace("+", "")


def paramset_id(params: dict[str, float]) -> str:
    return (
        f"cv{_format_float(params['Cv'])}"
        f"_cd{_format_float(params['Cd'])}"
        f"_ca{_format_float(params['Ca'])}"
        f"_c{_format_float(params['C'])}"
        f"_fhat0{_format_float(params['fhat0'])}"
        f"_band{_format_float(params['fhat_min'])}-{_format_float(params['fhat_max'])}"
    )


def build_paramsets_from_values(
    *,
    cv_values,
    cd_values,
    ca_values,
    damping_c_values,
    fhat_min_values,
    fhat0_values,
    fhat_max_values,
) -> list[dict[str, float]]:
    paramsets: list[dict[str, float]] = []
    for cv, cd, ca, c, fhat_min, fhat0, fhat_max in product(
        _as_list(cv_values),
        _as_list(cd_values),
        _as_list(ca_values),
        _as_list(damping_c_values),
        _as_list(fhat_min_values),
        _as_list(fhat0_values),
        _as_list(fhat_max_values),
    ):
        values = {
            "Cv": float(cv),
            "Cd": float(cd),
            "Ca": float(ca),
            "C": float(c),
            "fhat_min": float(fhat_min),
            "fhat0": float(fhat0),
            "fhat_max": float(fhat_max),
        }
        if not (values["fhat_min"] <= values["fhat0"] <= values["fhat_max"]):
            raise ValueError(
                "Invalid parameter set: require fhat_min <= fhat0 <= fhat_max, "
                f"got {values}"
            )
        paramsets.append(values)
    return paramsets


def build_single_paramset_from_burnin_config() -> dict[str, float]:
    try:
        burnin = importlib.import_module("analyze_vivana_td_burnin")
    except ModuleNotFoundError:
        burnin = importlib.import_module("vivana_cfd_data_pipeline.scripts.analyze_vivana_td_burnin")

    paramsets = build_paramsets_from_values(
        cv_values=burnin.CV_VALUES,
        cd_values=burnin.CD_VALUES,
        ca_values=burnin.CA_VALUES,
        damping_c_values=burnin.DAMPING_C_VALUES,
        fhat_min_values=burnin.FHAT_MIN_VALUES,
        fhat0_values=burnin.FHAT0_VALUES,
        fhat_max_values=burnin.FHAT_MAX_VALUES,
    )
    if len(paramsets) != 1:
        raise ValueError(
            "Expected exactly one active TD parameter set in analyze_vivana_td_burnin.py, "
            f"found {len(paramsets)}."
        )
    return paramsets[0]


def wrap_phase(values: np.ndarray, mode: str = "principal") -> np.ndarray:
    if mode != "principal":
        raise ValueError("Only mode='principal' is supported.")
    return shared_wrap_phase(values)


def initial_phi_dy(
    dy0: float,
    ddy0: float,
    sig_dy_loc0: float,
    sig_ddy_loc0: float,
    flow_speed_m_s: float,
) -> float:
    speed_mag = float(np.sqrt(float(flow_speed_m_s) ** 2 + float(dy0) ** 2))
    projection = float(flow_speed_m_s) / max(speed_mag, np.finfo(float).eps)
    dy_r0 = float(dy0) * projection
    ddy_r0 = float(ddy0) * projection
    cos_phi_dy0 = dy_r0 / (float(sig_dy_loc0) + np.spacing(1.0))
    sin_phi_dy0 = -ddy_r0 / (float(sig_ddy_loc0) + np.spacing(1.0))
    return float(np.angle(cos_phi_dy0 + 1j * sin_phi_dy0))


def compute_theta_series(
    dy: np.ndarray,
    ddy: np.ndarray,
    phi_vy: np.ndarray,
    sig_dy_loc: np.ndarray,
    sig_ddy_loc: np.ndarray,
    flow_speed_m_s: float,
    mode: str = "principal",
) -> np.ndarray:
    if mode != "principal":
        raise ValueError("Only mode='principal' is supported.")
    return shared_compute_theta_series(
        dy=dy,
        ddy=ddy,
        phi_vy=phi_vy,
        sig_dy_loc=sig_dy_loc,
        sig_ddy_loc=sig_ddy_loc,
        flow_speed=float(flow_speed_m_s),
    )


def _extract_case_channel(case_like: dict[str, object], *keys: str) -> np.ndarray:
    for key in keys:
        if key in case_like:
            return np.asarray(case_like[key], dtype=float)
    raise KeyError(f"Missing required case channel. Tried keys: {keys}")


def initial_hidden_sigmas(
    case_like: dict[str, object],
    start_idx: int,
    flow_speed_m_s: float,
    n_memory: int,
    mode: str,
    window_seconds: float | None,
) -> tuple[float, float]:
    if mode == "zero":
        return 0.0, 0.0
    if mode not in {"local_rms", "lookahead_rms"}:
        raise ValueError("mode must be 'zero', 'local_rms', or 'lookahead_rms'.")

    y_vel = _extract_case_channel(case_like, "dy_dim", "y_vel_dim")
    y_acc = _extract_case_channel(case_like, "ddy_dim", "y_acc_dim")
    if "dt_dim" not in case_like:
        raise KeyError("case_like must contain dt_dim.")
    dt_dim = float(np.asarray(case_like["dt_dim"]).reshape(()))

    if window_seconds is None:
        window_samples = max(1, int(n_memory))
    else:
        window_samples = max(1, int(round(float(window_seconds) / dt_dim)))

    if mode == "lookahead_rms":
        end_hist_idx = min(int(y_vel.shape[0]), int(start_idx) + window_samples)
        dy_hist = np.asarray(y_vel[int(start_idx) : end_hist_idx], dtype=float)
        ddy_hist = np.asarray(y_acc[int(start_idx) : end_hist_idx], dtype=float)
    else:
        start_hist_idx = max(0, int(start_idx) - window_samples + 1)
        dy_hist = np.asarray(y_vel[start_hist_idx : start_idx + 1], dtype=float)
        ddy_hist = np.asarray(y_acc[start_hist_idx : start_idx + 1], dtype=float)
    if dy_hist.size == 0 or ddy_hist.size == 0:
        return 0.0, 0.0

    speed_mag = np.sqrt(float(flow_speed_m_s) ** 2 + dy_hist**2)
    local_projection = float(flow_speed_m_s) / np.maximum(speed_mag, np.finfo(float).eps)
    dy_r_hist = dy_hist * local_projection
    ddy_r_hist = ddy_hist * local_projection

    sig_dy_loc0 = float(np.sqrt(np.mean(dy_r_hist**2)))
    sig_ddy_loc0 = float(np.sqrt(np.mean(ddy_r_hist**2)))
    return sig_dy_loc0, sig_ddy_loc0


def replay_hidden_state_with_cfd_motion(
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
    n_memory: int = 500,
) -> dict[str, np.ndarray]:
    return shared_replay_prescribed_motion(
        time=time,
        y=y,
        dy=dy,
        ddy=ddy,
        params=params,
        phi_vy0=phi_vy0,
        sig_dy_loc0=sig_dy_loc0,
        sig_ddy_loc0=sig_ddy_loc0,
        relative_time=True,
        flow_speed_m_s=float(flow_speed_m_s),
        rho_kg_m3=float(rho_kg_m3),
        diameter_m=float(diameter_m),
        n_memory=int(n_memory),
    )


def compute_force_spread_history(
    case_payload: dict[str, object],
    params: dict[str, float],
    theta0_values: np.ndarray,
    sigma_init_mode: str,
    sigma_init_window_seconds: float | None,
    n_memory: int,
    progress=None,
    progress_desc: str = "",
    return_force_stack: bool = False,
) -> dict[str, np.ndarray | float]:
    time_dim = _extract_case_channel(case_payload, "time_dim")
    y_dim = _extract_case_channel(case_payload, "y_dim", "y_disp_dim")
    dy_dim = _extract_case_channel(case_payload, "dy_dim", "y_vel_dim")
    ddy_dim = _extract_case_channel(case_payload, "ddy_dim", "y_acc_dim")
    force_dim = _extract_case_channel(
        case_payload,
        "force_per_m_dim",
        "y_force_per_m_dim",
        "force_dim",
        "y_force_dim",
    )
    flow_speed_m_s = float(np.asarray(case_payload["flow_speed_m_s"]).reshape(()))
    rho_kg_m3 = float(np.asarray(case_payload["rho_kg_m3"]).reshape(()))
    diameter_m = float(np.asarray(case_payload["diameter_m"]).reshape(()))

    sig_dy_loc0, sig_ddy_loc0 = initial_hidden_sigmas(
        case_like=case_payload,
        start_idx=0,
        flow_speed_m_s=flow_speed_m_s,
        n_memory=int(n_memory),
        mode=sigma_init_mode,
        window_seconds=sigma_init_window_seconds,
    )
    phi_dy0 = initial_phi_dy(
        dy0=float(dy_dim[0]),
        ddy0=float(ddy_dim[0]),
        sig_dy_loc0=sig_dy_loc0,
        sig_ddy_loc0=sig_ddy_loc0,
        flow_speed_m_s=flow_speed_m_s,
    )

    theta0_iterable = np.asarray(theta0_values, dtype=float)
    if progress is not None:
        theta0_iterable = progress(
            theta0_iterable,
            total=int(np.asarray(theta0_values, dtype=float).size),
            desc=progress_desc,
        )

    force_total_stack = []
    theta_stack = []
    for theta0 in theta0_iterable:
        phi_vy0 = float(wrap_phase(np.asarray([phi_dy0 - float(theta0)]))[0])
        sim = replay_hidden_state_with_cfd_motion(
            time=time_dim,
            y=y_dim,
            dy=dy_dim,
            ddy=ddy_dim,
            flow_speed_m_s=flow_speed_m_s,
            rho_kg_m3=rho_kg_m3,
            diameter_m=diameter_m,
            params=params,
            phi_vy0=phi_vy0,
            sig_dy_loc0=sig_dy_loc0,
            sig_ddy_loc0=sig_ddy_loc0,
            n_memory=int(n_memory),
        )
        force_total_stack.append(np.asarray(sim["F_total"], dtype=float))
        if return_force_stack:
            theta_stack.append(
                np.asarray(
                    compute_theta_series(
                        dy=np.asarray(sim["dy"], dtype=float),
                        ddy=np.asarray(sim["ddy"], dtype=float),
                        phi_vy=np.asarray(sim["phi_vy"], dtype=float),
                        sig_dy_loc=np.asarray(sim["sig_dy_loc"], dtype=float),
                        sig_ddy_loc=np.asarray(sim["sig_ddy_loc"], dtype=float),
                        flow_speed_m_s=flow_speed_m_s,
                        mode="principal",
                    ),
                    dtype=float,
                )
            )

    force_total_stack = np.asarray(force_total_stack, dtype=float)
    force_std_ref = max(float(np.std(force_dim)), np.finfo(float).eps)
    force_total_rel_std = np.std(force_total_stack, axis=0) / force_std_ref
    return {
        "time_dim": np.asarray(time_dim, dtype=float),
        "force_total_rel_std": np.asarray(force_total_rel_std, dtype=float),
        "force_std_ref": float(force_std_ref),
        "phi_dy0": float(phi_dy0),
        "sig_dy_loc0": float(sig_dy_loc0),
        "sig_ddy_loc0": float(sig_ddy_loc0),
        "theta0_values": np.asarray(theta0_values, dtype=float) if return_force_stack else np.asarray([], dtype=float),
        "force_total_stack": np.asarray(force_total_stack, dtype=float) if return_force_stack else np.asarray([], dtype=float),
        "theta_stack": np.asarray(theta_stack, dtype=float) if return_force_stack else np.asarray([], dtype=float),
    }


def detect_burnin_start_index(
    time_dim: np.ndarray,
    rel_std: np.ndarray,
    threshold: float,
    persistence_seconds: float,
) -> int | None:
    time_dim = np.asarray(time_dim, dtype=float).reshape(-1)
    rel_std = np.asarray(rel_std, dtype=float).reshape(-1)
    if time_dim.size != rel_std.size:
        raise ValueError("time_dim and rel_std must have the same length.")
    if time_dim.size < 2:
        return None
    diffs = np.diff(time_dim)
    if np.any(diffs <= 0.0):
        raise ValueError("time_dim must be strictly increasing.")

    below = np.isfinite(rel_std) & (rel_std < float(threshold))
    fail_prefix = np.concatenate(([0], np.cumsum(~below, dtype=int)))
    persistence_seconds = float(persistence_seconds)

    for idx in range(time_dim.size):
        if not below[idx]:
            continue
        target_time = float(time_dim[idx]) + persistence_seconds
        end_idx = int(np.searchsorted(time_dim, target_time, side="left"))
        if end_idx >= time_dim.size:
            continue
        num_failures = int(fail_prefix[end_idx + 1] - fail_prefix[idx])
        if num_failures == 0:
            return idx
    return None
