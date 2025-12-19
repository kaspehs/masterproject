# integrate_viv_run.py
"""
Ground-truth integrator for the coupled VIV ODEs using your ODEConfig.

Edit the CONFIG section below to control:
- solver tolerances/method
- sampling density
- steady-state trimming
- saving
- plotting
"""

from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
import pathlib
import matplotlib.pyplot as plt

# ---- Import your config helpers (must be in same folder) ----
from legacy_code.ODE import ODEConfig, build_ode_setup, print_ode_summary

# =========================
# ======= CONFIG ==========
# =========================
# Use default ODEConfig() or override any parameter here, e.g.:
# cfg = ODEConfig(Vr=6.0, zeta=0.003, CL0=0.6)
cfg = ODEConfig()

# Integration + sampling controls
RTOL: float = 1e-9
ATOL: float = 1e-12          # scalar; applied to all states
METHOD: str = "Radau"        # "Radau" | "BDF" | "LSODA"
SAMPLES_PER_PERIOD: int = 1000

# Which period to use for sampling/steady trimming
# "auto" -> uses the higher of (omegany, omegaf)
# "structural" -> uses omegany; "vortex" -> uses omegaf
DOMINANT: str = "auto"       # "auto" | "structural" | "vortex"

# Keep only the last N dominant periods (0 = keep everything)
STEADY_PERIODS: int = 0

# Saving options
SAVE_FORMAT: str = "npz"     # "" (disable) | "npz" | "csv"
OUTPUT_BASENAME: str = "viv_ground_truth"

# Plotting
DO_PLOT: bool = True
# =========================
# ===== END CONFIG ========
# =========================


# ------------------ RHS and Jacobian ------------------ #
def viv_rhs(t: float, z: np.ndarray, p: dict[str, float]) -> np.ndarray:
    """
    State order: z = [y, yd, q, qd]
      my*ydd + cy*yd + ky*y + k3y*y**3 = Kl*q
      qdd + cq*(q**2 - 1)*qd + kq*q = Kc*yd
    """
    y, yd, q, qd = z
    my = p["my"]; cy = p["cy"]; ky = p["ky"]; k3y = p["k3y"]; Kl = p["Kl"]
    cq = p["cq"]; kq = p["kq"]; Kc = p["Kc"]

    ydd = (Kl*q - cy*yd - ky*y - k3y*y**3) / my
    qdd = - cq*(q**2 - 1.0)*qd - kq*q + Kc*ydd
    return np.array([yd, ydd, qd, qdd], dtype=float)

def viv_jac(t: float, z: np.ndarray, p: dict[str, float]) -> np.ndarray:
    """
    Analytic Jacobian df/dz (helps Radau/BDF).
    Order: [y, yd, q, qd]
    """
    y, yd, q, qd = z
    my = p["my"]; cy = p["cy"]; ky = p["ky"]; k3y = p["k3y"]; Kl = p["Kl"]
    cq = p["cq"]; kq = p["kq"]; Kc = p["Kc"]

    J = np.zeros((4, 4))
    # d/dz of dy = yd
    J[0, 1] = 1.0
    # d/dz of yd_dot = ydd
    J[1, 0] = -(ky + 3.0*k3y*y**2)/my
    J[1, 1] = -cy/my
    J[1, 2] = Kl/my
    # d/dz of dq = qd
    J[2, 3] = 1.0
    # Changes compared to velocity coupling
    J[3,0] = Kc * (-(ky + 3*k3y*y**2)/my)          # new term (was 0)
    J[3,1] = Kc * (-cy/my)                         # new term (was Kc)
    J[3,2] = -(2*cq*q)*qd - kq + Kc*(Kl/my)        # modified (added coupling)
    J[3,3] = -cq*(q**2 - 1.0)                      # unchanged

    return J

# ------------------ Helpers ------------------ #
def detect_keep_mask_by_periods(t: np.ndarray, keep_last_periods: int, T: float) -> np.ndarray:
    if keep_last_periods <= 0:
        return np.ones_like(t, dtype=bool)
    t_keep_start = t[-1] - keep_last_periods * T
    return t >= t_keep_start


def main():
    # Build parameters from your config
    setup = build_ode_setup(cfg)
    print_ode_summary(setup)

    p = setup["ode_params"]
    T_min = setup["T_MIN_CONST"]
    T_max = setup["T_MAX_CONST"]

    # Initial condition: [y, yd, q, qd]
    y0 = cfg.A0 * cfg.D
    q0 = cfg.q0
    z0 = np.array([y0, 0.0, q0, 0.0], dtype=float)

    # Dominant frequency selection
    if DOMINANT == "structural":
        omega_dom = setup["omegany"]
    elif DOMINANT == "vortex":
        omega_dom = setup["omegaf"]
    else:
        # auto: choose the higher frequency (shorter period)
        omega_dom = max(setup["omegany"], setup["omegaf"])
    T_dom = 2.0 * np.pi / omega_dom

    # Uniform evaluation grid (consistent sampling)
    n_periods = max(1, int(np.ceil((T_max - T_min) / T_dom)))
    N_eval = int(np.ceil(n_periods * SAMPLES_PER_PERIOD))
    t_eval = np.linspace(T_min, T_max, N_eval + 1)

    # Integrate
    sol = solve_ivp(
        fun=lambda t, z: viv_rhs(t, z, p),
        t_span=(T_min, T_max),
        y0=z0,
        method=METHOD,
        t_eval=t_eval,
        rtol=RTOL,
        atol=np.full(4, ATOL),
        jac=lambda t, z: viv_jac(t, z, p),
    )
    if not sol.success:
        raise RuntimeError(sol.message)

    t_all = sol.t
    z_all = sol.y.T  # (M, 4)

    # Optional steady tail
    keep = detect_keep_mask_by_periods(t_all, STEADY_PERIODS, T_dom)
    t_keep = t_all[keep]
    z_keep = z_all[keep]

    # Save if requested
    if SAVE_FORMAT:
        outbase = pathlib.Path(OUTPUT_BASENAME)
        if SAVE_FORMAT == "npz":
            np.savez_compressed(
                outbase.with_suffix(".npz"),
                t=t_keep,
                y=z_keep[:, 0],
                yd=z_keep[:, 1],
                q=z_keep[:, 2],
                qd=z_keep[:, 3],
                params=p,
                config=cfg.__dict__,
            )
            print(f"Saved {outbase.with_suffix('.npz')}")
        elif SAVE_FORMAT == "csv":
            data = np.column_stack([t_keep, z_keep])
            header = "t,y,yd,q,qd"
            np.savetxt(outbase.with_suffix(".csv"), data, delimiter=",", header=header, comments="")
            print(f"Saved {outbase.with_suffix('.csv')}")

    # Plot
    if DO_PLOT:
        # Plot y and q on shared time axis
        plt.figure()
        plt.plot(t_keep, z_keep[:, 0]/cfg.D, label="y/D")
        plt.plot(t_keep, z_keep[:, 2], label="q", linestyle="--")
        plt.xlabel("t [s]")
        plt.ylabel("state")
        plt.title("VIV ground truth (steady segment)" if STEADY_PERIODS > 0 else "VIV ground truth")
        plt.legend()
        plt.tight_layout()
        plt.grid()
        plt.show()

if __name__ == "__main__":
    main()
