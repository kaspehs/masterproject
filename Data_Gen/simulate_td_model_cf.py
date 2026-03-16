"""
Reusable simulator for the TD cross-flow model plus batch generation helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

try:
    from utils import vforce_CF
except ModuleNotFoundError:
    from Data_Gen.utils import vforce_CF

# Base physical parameters
M = 16.79            # mass kg
zeta = 0.01          # structural damping
K = 1218.0           # stiffness N/m
rho = 1000.0         # fluid density (kg/m3)
U = 0.65             # flow speed (m/s)
D = 0.1              # cylinder diameter (m)
C = 1e-4             # damping (overrides 2*zeta*np.sqrt(M*K))
n_memory = 500       # timesteps for instantaneous velocity calculation

# Empirical force coefficients in TD model
Cv = 1.2             # vortex shedding coefficient
Cd = 1.2             # drag coefficient
Ca = 1.0             # added mass coefficient in still water

# Synchronization model parameters
fhat0 = 0.144        # centre of synchronization
fhat_min = 0.08
fhat_max = 0.206

T = 20.0
dt = 0.0001


def _validate_integrator(integrator: str) -> str:
    integrator = str(integrator).strip().lower()
    if integrator not in {"euler", "rk4"}:
        raise ValueError("integrator must be either 'euler' or 'rk4'.")
    return integrator


def _acceleration(*, y_val: float, dy_val: float, force_val: float, mass: float, damping_c: float, stiffness: float) -> float:
    return (1.0 / mass) * (-damping_c * dy_val - stiffness * y_val + force_val)


def _rk4_step(
    *,
    y_val: float,
    dy_val: float,
    force_val: float,
    dt_val: float,
    mass: float,
    damping_c: float,
    stiffness: float,
) -> tuple[float, float]:
    def acc_local(y_state: float, dy_state: float) -> float:
        return _acceleration(
            y_val=y_state,
            dy_val=dy_state,
            force_val=force_val,
            mass=mass,
            damping_c=damping_c,
            stiffness=stiffness,
        )

    k1_y = dy_val
    k1_v = acc_local(y_val, dy_val)

    y_mid = y_val + 0.5 * dt_val * k1_y
    v_mid = dy_val + 0.5 * dt_val * k1_v
    k2_y = v_mid
    k2_v = acc_local(y_mid, v_mid)

    y_mid = y_val + 0.5 * dt_val * k2_y
    v_mid = dy_val + 0.5 * dt_val * k2_v
    k3_y = v_mid
    k3_v = acc_local(y_mid, v_mid)

    y_end = y_val + dt_val * k3_y
    v_end = dy_val + dt_val * k3_v
    k4_y = v_end
    k4_v = acc_local(y_end, v_end)

    y_next = y_val + (dt_val / 6.0) * (k1_y + 2.0 * k2_y + 2.0 * k3_y + k4_y)
    v_next = dy_val + (dt_val / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
    return y_next, v_next


def _save_legacy_payload(output_path: Path, *, time: np.ndarray, y: np.ndarray, force_total: np.ndarray, hamiltonian: np.ndarray, dy: np.ndarray, u_r: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, a=time, b=y, c=force_total, d=hamiltonian, e=dy, U_r=u_r)


def simulate_td_model_cf_custom_init(
    *,
    dt: float,
    T: float,
    U: float,
    M: float,
    K: float,
    C: float,
    rho: float,
    D: float,
    Cv: float,
    Cd: float,
    Ca: float,
    fhat0: float,
    fhat_min: float,
    fhat_max: float,
    n_memory: int,
    integrator: str = "rk4",
    y0: float,
    dy0: float,
    ddy0: float,
    phi_vy0: float = 0.0,
    sig_dy_loc0: float = 0.0,
    sig_ddy_loc0: float = 0.0,
    output_path: str | Path | None = None,
    plot: bool = False,
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if T < 0.0:
        raise ValueError("T must be non-negative.")
    if M <= 0.0:
        raise ValueError("M must be positive.")
    if K <= 0.0:
        raise ValueError("K must be positive.")
    if rho <= 0.0:
        raise ValueError("rho must be positive.")
    if D <= 0.0:
        raise ValueError("D must be positive.")
    if n_memory < 1:
        raise ValueError("n_memory must be >= 1.")
    if fhat_min > fhat0 or fhat0 > fhat_max:
        raise ValueError("Require fhat_min <= fhat0 <= fhat_max.")

    integrator = _validate_integrator(integrator)

    n_steps = int(round(float(T) / float(dt)))
    if n_steps < 1:
        n_steps = 1
    n_samples = n_steps + 1

    time = np.arange(n_samples, dtype=float) * float(dt)
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

    y[0] = float(y0)
    dy[0] = float(dy0)
    ddy[0] = float(ddy0)
    phi_vy[0] = float(phi_vy0)
    sig_dy_loc[0] = float(sig_dy_loc0)
    sig_ddy_loc[0] = float(sig_ddy_loc0)

    u_r = float(2.0 * np.pi * U / D * np.sqrt((M + D**2 * np.pi / 4.0 * rho) / K))
    if verbose:
        print(f"Reduced velocity: {u_r:.3f}, damping C={C:.3e}, samples={n_samples}")

    for i in range(n_samples - 1):
        (
            Fy[i + 1],
            phi_vy[i + 1],
            sig_dy_loc[i + 1],
            sig_ddy_loc[i + 1],
            Fca[i + 1],
            Fcv[i + 1],
            Fdy[i + 1],
        ) = vforce_CF(
            Cv,
            Cd,
            Ca,
            fhat0,
            fhat_min,
            fhat_max,
            float(dt),
            int(n_memory),
            float(rho),
            float(U),
            float(D),
            float(dy[i]),
            float(ddy[i]),
            float(phi_vy[i]),
            float(sig_dy_loc[i]),
            float(sig_ddy_loc[i]),
        )

        if integrator == "rk4":
            y_next, dy_next = _rk4_step(
                y_val=float(y[i]),
                dy_val=float(dy[i]),
                force_val=float(Fy[i + 1]),
                dt_val=float(dt),
                mass=float(M),
                damping_c=float(C),
                stiffness=float(K),
            )
        else:
            y_next = float(y[i]) + float(dt) * float(dy[i])
            dy_next = float(dy[i]) + float(dt) * float(ddy[i])

        y[i + 1] = y_next
        dy[i + 1] = dy_next
        ddy[i + 1] = _acceleration(
            y_val=float(y_next),
            dy_val=float(dy_next),
            force_val=float(Fy[i + 1]),
            mass=float(M),
            damping_c=float(C),
            stiffness=float(K),
        )

    h = 0.5 * float(K) * y**2 + 0.5 * (float(M) + float(D) ** 2 / 4.0 * float(rho) * np.pi * float(Ca)) * dy**2
    f_total = Fcv + Fdy + Fca

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out,
            time=time,
            y=y,
            dy=dy,
            ddy=ddy,
            Fy=Fy,
            F_total=f_total,
            Fca=Fca,
            Fcv=Fcv,
            Fdy=Fdy,
            phi_vy=phi_vy,
            sig_dy_loc=sig_dy_loc,
            sig_ddy_loc=sig_ddy_loc,
            H=h,
            U_r=u_r,
        )

    if plot:
        _plot_diagnostics(time, y, dy, Fy, Fca, Fcv, Fdy)

    return {
        "time": time,
        "y": y,
        "dy": dy,
        "ddy": ddy,
        "Fy": Fy,
        "F_total": f_total,
        "Fca": Fca,
        "Fcv": Fcv,
        "Fdy": Fdy,
        "phi_vy": phi_vy,
        "sig_dy_loc": sig_dy_loc,
        "sig_ddy_loc": sig_ddy_loc,
        "H": h,
        "U_r": np.asarray(u_r, dtype=float),
    }


def simulate_td_model_cf(
    A_factor: float = 0.6,
    fhat: float = 0.15,
    dt: float = dt,
    T: float = T,
    U: float = U,
    output_path: str | Path | None = "data.npz",
    plot: bool = False,
    seed: int | None = None,
    verbose: bool = False,
    integrator: str = "rk4",
) -> Dict[str, np.ndarray]:
    """
    Simulate the TD model for a single set of initial conditions.

    Args:
        A_factor: multiplier applied to D to set the displacement amplitude.
        fhat: normalized frequency used for the initial harmonic displacement.
        dt: timestep size.
        T: total simulation time.
        U: flow speed used to set reduced velocity.
        output_path: where to store the npz file; set to None to skip saving.
        plot: whether to show diagnostic plots.
        seed: optional RNG seed kept for backward compatibility; currently unused.
        verbose: print reduced velocity and damping info when True.
        integrator: "rk4" (default) for Runge-Kutta 4 or "euler" for explicit Euler.

    Returns:
        Dictionary with time, displacement, force, Hamiltonian, velocity, etc.
    """
    if A_factor <= 0.0:
        raise ValueError("A_factor must be positive.")
    if fhat <= 0.0:
        raise ValueError("fhat must be positive.")
    if seed is not None:
        # Retained for compatibility with old callers that passed a seed.
        np.random.default_rng(seed)

    omega_osc = 2.0 * np.pi * float(fhat) * float(U) / float(D)
    amplitude = float(A_factor) * float(D)
    y0 = amplitude
    dy0 = omega_osc * amplitude
    ddy0 = 0.0

    full = simulate_td_model_cf_custom_init(
        dt=float(dt),
        T=float(T),
        U=float(U),
        M=float(M),
        K=float(K),
        C=float(C),
        rho=float(rho),
        D=float(D),
        Cv=float(Cv),
        Cd=float(Cd),
        Ca=float(Ca),
        fhat0=float(fhat0),
        fhat_min=float(fhat_min),
        fhat_max=float(fhat_max),
        n_memory=int(n_memory),
        integrator=integrator,
        y0=y0,
        dy0=dy0,
        ddy0=ddy0,
        phi_vy0=0.0,
        sig_dy_loc0=0.0,
        sig_ddy_loc0=0.0,
        output_path=None,
        plot=False,
        verbose=verbose,
    )

    sl = slice(1, -1)
    trimmed = {
        "time": np.asarray(full["time"][sl], dtype=float),
        "y": np.asarray(full["y"][sl], dtype=float),
        "dy": np.asarray(full["dy"][sl], dtype=float),
        "Fy": np.asarray(full["Fy"][sl], dtype=float),
        "F_total": np.asarray(full["F_total"][sl], dtype=float),
        "Fca": np.asarray(full["Fca"][sl], dtype=float),
        "Fcv": np.asarray(full["Fcv"][sl], dtype=float),
        "Fdy": np.asarray(full["Fdy"][sl], dtype=float),
        "H": np.asarray(full["H"][sl], dtype=float),
        "U_r": float(np.asarray(full["U_r"]).reshape(())),
    }

    if output_path is not None:
        _save_legacy_payload(
            Path(output_path),
            time=trimmed["time"],
            y=trimmed["y"],
            force_total=trimmed["F_total"],
            hamiltonian=trimmed["H"],
            dy=trimmed["dy"],
            u_r=trimmed["U_r"],
        )

    if plot:
        _plot_diagnostics(
            trimmed["time"],
            trimmed["y"],
            trimmed["dy"],
            trimmed["Fy"],
            trimmed["Fca"],
            trimmed["Fcv"],
            trimmed["Fdy"],
        )

    return trimmed


def _plot_diagnostics(time, y, dy, Fy, Fca, Fcv, Fdy):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4))
    plt.plot(time, Fy, label="Force (N)")
    plt.plot(time, y * 100, label=r"Displacement $\times 10^2$ (m)")
    plt.title("Cross-flow force and displacement")
    plt.xlabel("time (sec)")
    plt.ylabel("Simulation")
    plt.legend()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(time, Fy, label="Force (N)")
    plt.plot(time, Fca, label="Fca (N)")
    plt.plot(time, Fcv, label="Fcv (N)")
    plt.plot(time, Fdy, label="Fd (N)")
    plt.xlim([12, 14])
    plt.title("Force breakdown")
    plt.xlabel("time (sec)")
    plt.ylabel("Simulation")
    plt.legend()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(time, dy * 100, label="vel ×10 (m/s)")
    plt.plot(time, Fcv + Fdy, label="Fcv+Fdy (N)")
    plt.plot(time, Fca, label="Fca (N)")
    plt.xlim([12, 14])
    plt.title("Velocity vs. forces")
    plt.xlabel("time (sec)")
    plt.ylabel("Simulation")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    simulate_td_model_cf(plot=True, verbose=True)
