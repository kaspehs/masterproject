"""
Reusable simulator for the TD cross-flow model plus batch generation helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
try:
    from utils import vforce_CF
except:
    pass
try:
    from Data_Gen.utils import vforce_CF
except:
    pass

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

def simulate_td_model_cf(
    A_factor: float = 0.6,
    fhat: float = 0.15,
    dt: float = dt,
    T: float = T,
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
        output_path: where to store the npz file; set to None to skip saving.
        plot: whether to show diagnostic plots.
        seed: optional RNG seed for the initial vortex shedding phase.
        verbose: print reduced velocity and damping info when True.
        integrator: "rk4" (default) for Runge-Kutta 4 or "euler" for explicit Euler.

    Returns:
        Dictionary with time, displacement, force, Hamiltonian, velocity, etc.
    
    if A_factor <= 0.0:
        raise ValueError("A_factor must be positive.")
    if fhat <= 0.0:
        raise ValueError("fhat must be positive.")
        """
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if T <= 0.0:
        raise ValueError("T must be positive.")

    integrator = integrator.lower()
    if integrator not in {"euler", "rk4"}:
        raise ValueError("integrator must be either 'euler' or 'rk4'.")

    rng = np.random.default_rng(seed)

    N = int(np.ceil(T / dt))
    time = np.zeros(N)
    y = np.zeros(N)
    dy = np.zeros(N)
    ddy = np.zeros(N)
    Fy = np.zeros(N)
    Fcv = np.zeros(N)
    Fdy = np.zeros(N)
    Fca = np.zeros(N)

    phi_vy = np.zeros(N)
    #phi_vy[0] = 2.0 * np.pi * rng.random()
    sig_dy_loc = np.zeros(N)
    sig_ddy_loc = np.zeros(N)

    A = A_factor * D
    omega_osc = 2.0 * np.pi * fhat * U / D

    y[0] = A #* np.sin(omega_osc * time[0])
    dy[0] = omega_osc * A * np.cos(omega_osc * time[0])
    ddy[0] = -omega_osc**2 * A * np.sin(omega_osc * time[0])

    def acceleration(y_val: float, dy_val: float, force_val: float) -> float:
        return (1.0 / M) * (-C * dy_val - K * y_val + force_val)

    def rk4_step(y_val: float, dy_val: float, force_val: float, dt_val: float) -> tuple[float, float]:
        def acc_local(y_state: float, dy_state: float) -> float:
            return acceleration(y_state, dy_state, force_val)

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

    if verbose:
        U_r = 2 * np.pi * U / D * np.sqrt((M + D**2 * np.pi / 4.0 * rho) / K)
        print(f"Reduced velocity: {U_r:.3f}, damping C={C:.3e}")

    for i in range(N - 1):
        time[i] = i * dt
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
            dt,
            n_memory,
            rho,
            U,
            D,
            dy[i],
            ddy[i],
            phi_vy[i],
            sig_dy_loc[i],
            sig_ddy_loc[i],
        )

        if integrator == "rk4":
            y_next, dy_next = rk4_step(y[i], dy[i], Fy[i + 1], dt)
        else:
            y_next = y[i] + dt * dy[i]
            dy_next = dy[i] + dt * ddy[i]

        y[i + 1] = y_next
        dy[i + 1] = dy_next
        ddy[i + 1] = acceleration(y_next, dy_next, Fy[i + 1])

    # truncate the last element to keep shapes consistent with original script
    time = time[1:-1]
    y = y[1:-1]
    dy = dy[1:-1]
    Fy = Fy[1:-1]
    Fcv = Fcv[1:-1]
    Fdy = Fdy[1:-1]
    Fca = Fca[1:-1]

    H = 0.5 * K * y**2 + 0.5 * (M + D**2 / 4.0 * rho * np.pi * Ca) * dy**2
    F_total = Fcv + Fdy

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_path, a=time, b=y, c=F_total, d=H)

    if plot:
        _plot_diagnostics(time, y, dy, Fy, Fca, Fcv, Fdy)

    return {
        "time": time,
        "y": y,
        "dy": dy,
        "Fy": Fy,
        "F_total": F_total,
        "Fca": Fca,
        "Fcv": Fcv,
        "Fdy": Fdy,
        "H": H,
    }


def _plot_diagnostics(time, y, dy, Fy, Fca, Fcv, Fdy):
    fig = plt.figure(figsize=(7, 4))
    plt.plot(time, Fy, label="Force (N)")
    plt.plot(time, y * 100, label=r"Displacement $\times 10^2$ (m)")
    plt.title("Cross-flow force and displacement")
    plt.xlabel("time (sec)")
    plt.ylabel("Simulation")
    plt.legend()
    plt.show()

    fig = plt.figure(figsize=(7, 4))
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

    fig = plt.figure(figsize=(7, 4))
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
