from __future__ import annotations

import numpy as np

try:
    from utils import vforce_CF
except ModuleNotFoundError:
    try:
        from Data_Gen.utils import vforce_CF
    except ModuleNotFoundError:
        from CFD_Data.utils import vforce_CF


DEFAULT_PARAMS: dict[str, float | str] = {
    "M": 3.0,
    "C": 0.0e-4,
    "K": 7.4 / 4.0,
    "rho": 1.0,
    "U": 1.0,
    "D": 1.0,
    "nsteps": 200,
    "n_memory": 500,
    "Cv": 1.2,
    "Cd": 1.1,
    "Ca": 1.0,
    "fhat0": 0.18,
    "fhat_min": 0.11,
    "fhat_max": 0.26,
    "dt": 0.001,
    "T": 200.0,
    "A0": 0.1,
    "fhat_init": 0.17,
    "integrator": "forward_euler",
}

PYTHON_INTEGRATORS = (
    ("forward_euler", "Forward Euler"),
    ("rk4", "RK4"),
    ("rk4_coupled", "RK4-coupled"),
    ("newmark_beta", "Newmark-beta"),
)
VALID_INTEGRATORS = tuple(key for key, _ in PYTHON_INTEGRATORS)


def wrap_phase(values: np.ndarray | float) -> np.ndarray:
    return (np.asarray(values, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def compute_theta_series(
    dy: np.ndarray,
    ddy: np.ndarray,
    phi_vy: np.ndarray,
    sig_dy_loc: np.ndarray,
    sig_ddy_loc: np.ndarray,
    flow_speed: float,
) -> np.ndarray:
    dy_arr = np.asarray(dy, dtype=float)
    ddy_arr = np.asarray(ddy, dtype=float)
    phi_arr = np.asarray(phi_vy, dtype=float)
    sig_dy_arr = np.asarray(sig_dy_loc, dtype=float)
    sig_ddy_arr = np.asarray(sig_ddy_loc, dtype=float)
    speed_mag = np.sqrt(float(flow_speed) ** 2 + dy_arr**2)
    projection = float(flow_speed) / np.maximum(speed_mag, np.finfo(float).eps)
    dy_r = dy_arr * projection
    ddy_r = ddy_arr * projection
    cos_phi_dy = dy_r / (sig_dy_arr + np.spacing(1.0))
    sin_phi_dy = -ddy_r / (sig_ddy_arr + np.spacing(1.0))
    phi_dy = np.angle(cos_phi_dy + 1j * sin_phi_dy)
    return wrap_phase(phi_dy - phi_arr)


def _local_cf_kinematics(U: float, dy: float, ddy: float) -> tuple[float, float, float]:
    abs_v = float(np.sqrt(U**2 + dy**2))
    normal_y = float(U / (abs_v + np.spacing(1.0)))
    return abs_v, float(dy * normal_y), float(ddy * normal_y)


def _synchronized_fhat(
    fhat0: float,
    fhat_min: float,
    fhat_max: float,
    dy_r: float,
    ddy_r: float,
    phi_vy: float,
    sig_dy_r: float,
    sig_ddy_r: float,
) -> float:
    cos_phi_dy = dy_r / (sig_dy_r + np.spacing(1.0))
    sin_phi_dy = -ddy_r / (sig_ddy_r + np.spacing(1.0))
    phi_dy = float(np.angle(complex(cos_phi_dy, sin_phi_dy)))
    theta = float(np.angle(complex(np.cos(phi_dy - phi_vy), np.sin(phi_dy - phi_vy))))
    if theta <= 0.0:
        return float(fhat0 + (fhat0 - fhat_min) * np.sin(theta))
    return float(fhat0 + (fhat_max - fhat0) * np.sin(theta))


def vforce_cf(
    Cv: float,
    Cd: float,
    Ca: float,
    fhat0: float,
    fhat_min: float,
    fhat_max: float,
    dt: float,
    N: int,
    rho: float,
    U: float,
    D: float,
    dy: float,
    ddy: float,
    phi_vy: float,
    sig_dy_r: float,
    sig_ddy_r: float,
) -> tuple[float, float, float, float, float, float, float]:
    return vforce_CF(
        float(Cv),
        float(Cd),
        float(Ca),
        float(fhat0),
        float(fhat_min),
        float(fhat_max),
        float(dt),
        int(N),
        float(rho),
        float(U),
        float(D),
        float(dy),
        float(ddy),
        float(phi_vy),
        float(sig_dy_r),
        float(sig_ddy_r),
    )


def _structural_acceleration(y: float, dy: float, force: float, model: dict[str, float | str]) -> float:
    return float((-float(model["C"]) * dy - float(model["K"]) * y + force) / float(model["M"]))


def _coupled_diagnostics(
    y: float,
    dy: float,
    phi_vy: float,
    q_dy: float,
    q_ddy: float,
    model: dict[str, float | str],
) -> dict[str, float]:
    sig_dy = float(np.sqrt(max(q_dy, 0.0)))
    sig_ddy = float(np.sqrt(max(q_ddy, 0.0)))
    abs_v, _, _ = _local_cf_kinematics(float(model["U"]), dy, 0.0)
    Fdy = float(-0.5 * float(model["rho"]) * float(model["D"]) * float(model["Cd"]) * abs_v * dy)
    Fcv = float(
        0.5
        * float(model["rho"])
        * float(model["D"])
        * float(model["Cv"])
        * abs_v
        * float(model["U"])
        * np.cos(phi_vy)
    )
    added_mass_coeff = float(0.25 * float(model["rho"]) * float(model["Ca"]) * np.pi * float(model["D"]) ** 2)
    ddy = float((-float(model["C"]) * dy - float(model["K"]) * y + Fcv + Fdy) / (float(model["M"]) + added_mass_coeff))
    _, dy_r, ddy_r = _local_cf_kinematics(float(model["U"]), dy, ddy)
    fhat = _synchronized_fhat(
        float(model["fhat0"]),
        float(model["fhat_min"]),
        float(model["fhat_max"]),
        dy_r,
        ddy_r,
        phi_vy,
        sig_dy,
        sig_ddy,
    )
    omega_vy = float(2.0 * np.pi * fhat * abs_v / float(model["D"]))
    Fca = float(-added_mass_coeff * ddy)
    Fy = float(Fcv + Fdy + Fca)
    tau = float(model["n_memory"]) * float(model["dt"])
    q_dy_dot = float((dy_r**2 - q_dy) / tau)
    q_ddy_dot = float((ddy_r**2 - q_ddy) / tau)
    return {
        "ddy": ddy,
        "Fy": Fy,
        "Fcv": Fcv,
        "Fdy": Fdy,
        "Fca": Fca,
        "omega_vy": omega_vy,
        "q_dy_dot": q_dy_dot,
        "q_ddy_dot": q_ddy_dot,
    }


def _step_forward_euler(y: float, dy: float, dt: float, force: float, model: dict[str, float | str]) -> tuple[float, float, float]:
    ddy = _structural_acceleration(y, dy, force, model)
    y_next = float(y + dt * dy)
    dy_next = float(dy + dt * ddy)
    ddy_next = _structural_acceleration(y_next, dy_next, force, model)
    return y_next, dy_next, ddy_next


def _step_rk4(y: float, dy: float, dt: float, force: float, model: dict[str, float | str]) -> tuple[float, float, float]:
    def rhs(state: np.ndarray) -> np.ndarray:
        return np.asarray(
            [state[1], _structural_acceleration(float(state[0]), float(state[1]), force, model)],
            dtype=float,
        )

    state = np.asarray([y, dy], dtype=float)
    k1 = rhs(state)
    k2 = rhs(state + 0.5 * dt * k1)
    k3 = rhs(state + 0.5 * dt * k2)
    k4 = rhs(state + dt * k3)
    y_next, dy_next = state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    ddy_next = _structural_acceleration(float(y_next), float(dy_next), force, model)
    return float(y_next), float(dy_next), ddy_next


def _step_newmark_beta(
    y: float,
    dy: float,
    ddy: float,
    dt: float,
    force: float,
    model: dict[str, float | str],
    beta: float = 0.25,
    gamma: float = 0.5,
) -> tuple[float, float, float]:
    denom = float(model["M"]) + gamma * dt * float(model["C"]) + beta * dt**2 * float(model["K"])
    rhs = (
        force
        - float(model["C"]) * (dy + dt * (1.0 - gamma) * ddy)
        - float(model["K"]) * (y + dt * dy + dt**2 * (0.5 - beta) * ddy)
    )
    ddy_next = float(rhs / denom)
    dy_next = float(dy + dt * ((1.0 - gamma) * ddy + gamma * ddy_next))
    y_next = float(y + dt * dy + dt**2 * ((0.5 - beta) * ddy + beta * ddy_next))
    return y_next, dy_next, ddy_next


def _step_rk4_coupled(
    y: float,
    dy: float,
    phi_vy: float,
    q_dy: float,
    q_ddy: float,
    dt: float,
    model: dict[str, float | str],
) -> tuple[np.ndarray, dict[str, float]]:
    def rhs(state: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        diag = _coupled_diagnostics(
            float(state[0]),
            float(state[1]),
            float(state[2]),
            float(state[3]),
            float(state[4]),
            model,
        )
        deriv = np.asarray(
            [
                state[1],
                diag["ddy"],
                diag["omega_vy"],
                diag["q_dy_dot"],
                diag["q_ddy_dot"],
            ],
            dtype=float,
        )
        return deriv, diag

    state = np.asarray([y, dy, phi_vy, q_dy, q_ddy], dtype=float)
    k1, diag1 = rhs(state)
    k2, _ = rhs(state + 0.5 * dt * k1)
    k3, _ = rhs(state + 0.5 * dt * k2)
    k4, _ = rhs(state + dt * k3)
    next_state = state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    next_state[3] = max(float(next_state[3]), 0.0)
    next_state[4] = max(float(next_state[4]), 0.0)
    return np.asarray(next_state, dtype=float), diag1


def _integrate_step(
    y: float,
    dy: float,
    ddy: float,
    dt: float,
    force: float,
    model: dict[str, float | str],
) -> tuple[float, float, float]:
    integrator = str(model["integrator"])
    if integrator == "forward_euler":
        return _step_forward_euler(y, dy, dt, force, model)
    if integrator == "rk4":
        return _step_rk4(y, dy, dt, force, model)
    if integrator == "newmark_beta":
        return _step_newmark_beta(y, dy, ddy, dt, force, model)
    raise ValueError(f"Unknown integrator: {integrator}")


def run_simulation(
    params: dict[str, float | str] | None = None,
    initial_state: dict[str, float] | None = None,
) -> dict[str, np.ndarray | dict[str, float | str]]:
    model: dict[str, float | str] = dict(DEFAULT_PARAMS)
    if params:
        model.update(params)

    integrator = str(model["integrator"])
    if integrator not in VALID_INTEGRATORS:
        raise ValueError(f"integrator must be one of {VALID_INTEGRATORS}")

    dt = float(model["dt"])
    total_time = float(model["T"])
    n_samples = max(2, int(np.ceil(total_time / dt)))

    time = np.arange(n_samples, dtype=float) * dt
    y = np.zeros(n_samples, dtype=float)
    dy = np.zeros(n_samples, dtype=float)
    ddy = np.zeros(n_samples, dtype=float)
    Fy = np.zeros(n_samples, dtype=float)
    Fcv = np.zeros(n_samples, dtype=float)
    Fdy = np.zeros(n_samples, dtype=float)
    Fca = np.zeros(n_samples, dtype=float)
    phi_vy = np.zeros(n_samples, dtype=float)
    sig_dy_loc = np.zeros(n_samples, dtype=float)
    sig_ddy_loc = np.zeros(n_samples, dtype=float)
    q_dy_loc = np.zeros(n_samples, dtype=float)
    q_ddy_loc = np.zeros(n_samples, dtype=float)

    if initial_state:
        time += float(initial_state.get("time_offset", 0.0))
        y[0] = float(initial_state.get("y", 0.0))
        dy[0] = float(initial_state.get("dy", 0.0))
        ddy[0] = float(initial_state.get("ddy", 0.0))
        phi_vy[0] = float(initial_state.get("phi_vy", 0.0))
        sig_dy_loc[0] = float(initial_state.get("sig_dy_loc", 0.0))
        sig_ddy_loc[0] = float(initial_state.get("sig_ddy_loc", 0.0))
        q_dy_loc[0] = sig_dy_loc[0] ** 2
        q_ddy_loc[0] = sig_ddy_loc[0] ** 2
    else:
        omega_osc = float(2.0 * np.pi * float(model["fhat_init"]) * float(model["U"]) / float(model["D"]))
        y[0] = float(model["A0"]) * np.sin(omega_osc * time[0])
        dy[0] = omega_osc * float(model["A0"]) * np.cos(omega_osc * time[0])
        ddy[0] = -(omega_osc**2) * float(model["A0"]) * np.sin(omega_osc * time[0])

    for idx in range(n_samples - 1):
        if integrator == "rk4_coupled":
            next_state, diag = _step_rk4_coupled(
                float(y[idx]),
                float(dy[idx]),
                float(phi_vy[idx]),
                float(q_dy_loc[idx]),
                float(q_ddy_loc[idx]),
                dt,
                model,
            )
            y[idx + 1], dy[idx + 1], phi_vy[idx + 1], q_dy_loc[idx + 1], q_ddy_loc[idx + 1] = next_state
            next_diag = _coupled_diagnostics(
                float(y[idx + 1]),
                float(dy[idx + 1]),
                float(phi_vy[idx + 1]),
                float(q_dy_loc[idx + 1]),
                float(q_ddy_loc[idx + 1]),
                model,
            )
            ddy[idx + 1] = float(next_diag["ddy"])
            Fy[idx + 1] = float(diag["Fy"])
            Fcv[idx + 1] = float(diag["Fcv"])
            Fdy[idx + 1] = float(diag["Fdy"])
            Fca[idx + 1] = float(diag["Fca"])
            sig_dy_loc[idx + 1] = float(np.sqrt(q_dy_loc[idx + 1]))
            sig_ddy_loc[idx + 1] = float(np.sqrt(q_ddy_loc[idx + 1]))
            continue

        (
            Fy[idx + 1],
            phi_vy[idx + 1],
            sig_dy_loc[idx + 1],
            sig_ddy_loc[idx + 1],
            Fca[idx + 1],
            Fcv[idx + 1],
            Fdy[idx + 1],
        ) = vforce_cf(
            float(model["Cv"]),
            float(model["Cd"]),
            float(model["Ca"]),
            float(model["fhat0"]),
            float(model["fhat_min"]),
            float(model["fhat_max"]),
            dt,
            int(model["n_memory"]),
            float(model["rho"]),
            float(model["U"]),
            float(model["D"]),
            float(dy[idx]),
            float(ddy[idx]),
            float(phi_vy[idx]),
            float(sig_dy_loc[idx]),
            float(sig_ddy_loc[idx]),
        )
        q_dy_loc[idx + 1] = sig_dy_loc[idx + 1] ** 2
        q_ddy_loc[idx + 1] = sig_ddy_loc[idx + 1] ** 2
        y[idx + 1], dy[idx + 1], ddy[idx + 1] = _integrate_step(
            float(y[idx]),
            float(dy[idx]),
            float(ddy[idx]),
            dt,
            float(Fy[idx + 1]),
            model,
        )

    return {
        "params": model,
        "time": time,
        "y": y,
        "dy": dy,
        "ddy": ddy,
        "Fy": Fy,
        "Fcv": Fcv,
        "Fdy": Fdy,
        "Fca": Fca,
        "phi_vy": phi_vy,
        "sig_dy_loc": sig_dy_loc,
        "sig_ddy_loc": sig_ddy_loc,
    }


def build_phase_burn_in_state_from_displacement(
    *,
    time_full: np.ndarray,
    y_full: np.ndarray,
    dt: float,
    start_window_idx: int,
    compare_start_idx: int,
    end_idx: int,
    python_params: dict[str, float | str],
) -> dict[str, float]:
    disp = np.asarray(y_full[start_window_idx:end_idx], dtype=float)
    start_index = int(compare_start_idx - start_window_idx)
    vel = np.gradient(disp, dt)
    acc = np.gradient(vel, dt)

    phi_vy = 0.0
    sig_dy_loc = 0.0
    sig_ddy_loc = 0.0
    for idx in range(start_index):
        _, phi_vy, sig_dy_loc, sig_ddy_loc, *_ = vforce_cf(
            float(python_params["Cv"]),
            float(python_params["Cd"]),
            float(python_params["Ca"]),
            float(python_params["fhat0"]),
            float(python_params["fhat_min"]),
            float(python_params["fhat_max"]),
            dt,
            int(python_params["n_memory"]),
            float(python_params["rho"]),
            float(python_params["U"]),
            float(python_params["D"]),
            float(vel[idx]),
            float(acc[idx]),
            float(phi_vy),
            float(sig_dy_loc),
            float(sig_ddy_loc),
        )

    return {
        "time_offset": float(np.asarray(time_full, dtype=float)[compare_start_idx]),
        "y": float(disp[start_index]),
        "dy": float(vel[start_index]),
        "ddy": float(acc[start_index]),
        "phi_vy": float(phi_vy),
        "sig_dy_loc": float(sig_dy_loc),
        "sig_ddy_loc": float(sig_ddy_loc),
    }


def replay_prescribed_motion(
    *,
    time: np.ndarray,
    y: np.ndarray,
    dy: np.ndarray,
    ddy: np.ndarray,
    params: dict[str, float | str] | dict[str, float],
    phi_vy0: float,
    sig_dy_loc0: float,
    sig_ddy_loc0: float,
    relative_time: bool = False,
    flow_speed_m_s: float | None = None,
    rho_kg_m3: float | None = None,
    diameter_m: float | None = None,
    n_memory: int | None = None,
) -> dict[str, np.ndarray]:
    time_arr = np.asarray(time, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    dy_arr = np.asarray(dy, dtype=float)
    ddy_arr = np.asarray(ddy, dtype=float)
    if not (time_arr.size == y_arr.size == dy_arr.size == ddy_arr.size):
        raise ValueError("time, y, dy, and ddy must have the same length.")
    if time_arr.size < 2:
        raise ValueError("Need at least two samples for prescribed-motion replay.")

    resolved_U = float(params["U"]) if "U" in params else float(flow_speed_m_s)
    resolved_rho = float(params["rho"]) if "rho" in params else float(rho_kg_m3)
    resolved_D = float(params["D"]) if "D" in params else float(diameter_m)
    resolved_n_memory = int(params["n_memory"]) if "n_memory" in params else int(n_memory)

    n_samples = int(time_arr.size)
    Fy = np.zeros(n_samples, dtype=float)
    Fcv = np.zeros(n_samples, dtype=float)
    Fdy = np.zeros(n_samples, dtype=float)
    Fca = np.zeros(n_samples, dtype=float)
    phi_vy = np.zeros(n_samples, dtype=float)
    sig_dy_loc = np.zeros(n_samples, dtype=float)
    sig_ddy_loc = np.zeros(n_samples, dtype=float)

    phi_vy[0] = float(phi_vy0)
    sig_dy_loc[0] = float(sig_dy_loc0)
    sig_ddy_loc[0] = float(sig_ddy_loc0)

    for idx in range(n_samples - 1):
        dt = float(time_arr[idx + 1] - time_arr[idx])
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"Non-positive dt detected at index {idx}: {dt}")
        (
            Fy[idx + 1],
            phi_vy[idx + 1],
            sig_dy_loc[idx + 1],
            sig_ddy_loc[idx + 1],
            Fca[idx + 1],
            Fcv[idx + 1],
            Fdy[idx + 1],
        ) = vforce_cf(
            float(params["Cv"]),
            float(params["Cd"]),
            float(params["Ca"]),
            float(params["fhat0"]),
            float(params["fhat_min"]),
            float(params["fhat_max"]),
            dt,
            resolved_n_memory,
            resolved_rho,
            resolved_U,
            resolved_D,
            float(dy_arr[idx]),
            float(ddy_arr[idx]),
            float(phi_vy[idx]),
            float(sig_dy_loc[idx]),
            float(sig_ddy_loc[idx]),
        )

    out_time = time_arr - float(time_arr[0]) if relative_time else time_arr
    return {
        "time": np.asarray(out_time, dtype=float),
        "y": y_arr,
        "dy": dy_arr,
        "ddy": ddy_arr,
        "Fy": Fy,
        "F_total": np.asarray(Fca + Fcv + Fdy, dtype=float),
        "Fcv": Fcv,
        "Fdy": Fdy,
        "Fca": Fca,
        "phi_vy": phi_vy,
        "sig_dy_loc": sig_dy_loc,
        "sig_ddy_loc": sig_ddy_loc,
    }


def standardize_rollout(
    raw_result: dict[str, np.ndarray | dict[str, float | str]],
) -> dict[str, np.ndarray | dict[str, float | str] | str]:
    params = dict(raw_result["params"])  # type: ignore[arg-type]
    dy = np.asarray(raw_result["dy"], dtype=float)
    ddy = np.asarray(raw_result["ddy"], dtype=float)
    phi_vy = np.asarray(raw_result["phi_vy"], dtype=float)
    sig_dy = np.asarray(raw_result["sig_dy_loc"], dtype=float)
    sig_ddy = np.asarray(raw_result["sig_ddy_loc"], dtype=float)
    theta = compute_theta_series(
        dy=dy,
        ddy=ddy,
        phi_vy=phi_vy,
        sig_dy_loc=sig_dy,
        sig_ddy_loc=sig_ddy,
        flow_speed=float(params["U"]),
    )
    return {
        "integrator": str(params["integrator"]),
        "params": params,
        "time": np.asarray(raw_result["time"], dtype=float),
        "y": np.asarray(raw_result["y"], dtype=float),
        "dy": dy,
        "ddy": ddy,
        "force_total": np.asarray(raw_result["Fy"], dtype=float),
        "force_total_compare": np.asarray(raw_result["Fcv"], dtype=float) + np.asarray(raw_result["Fdy"], dtype=float),
        "force_cv": np.asarray(raw_result["Fcv"], dtype=float),
        "force_drag": np.asarray(raw_result["Fdy"], dtype=float),
        "force_added_mass": np.asarray(raw_result["Fca"], dtype=float),
        "phi_vy": phi_vy,
        "sig_dy": sig_dy,
        "sig_ddy": sig_ddy,
        "theta": np.asarray(theta, dtype=float),
    }


__all__ = [
    "DEFAULT_PARAMS",
    "PYTHON_INTEGRATORS",
    "VALID_INTEGRATORS",
    "build_phase_burn_in_state_from_displacement",
    "compute_theta_series",
    "replay_prescribed_motion",
    "run_simulation",
    "standardize_rollout",
    "vforce_cf",
    "wrap_phase",
]
