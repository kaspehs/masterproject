"""Evaluate a trained HNN across an initial-condition grid and plot phase error maps."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from HNN_helper import PHVIV, parse_config, compute_velocity_numpy, rollout_model
from Data_Gen.simulate_td_model_cf import (
    M,
    K,
    rho,
    U,
    D,
    C,
    n_memory,
    Cv,
    Cd,
    Ca,
    fhat0,
    fhat_min,
    fhat_max,
)
from Data_Gen.utils import vforce_CF


MODEL_PATH = Path("models/residual_final1_1130-115801.pt")
GRID_Q = np.linspace(-0.1, 0.1, 10)
GRID_V = np.linspace(-1.0, 1.0, 10)
SIM_FINE_DT = 1e-4
SIM_T = 10.0
DEVICE = torch.device("cpu")


def load_model(model_path: Path) -> tuple[PHVIV, dict[str, float], float]:
    ckpt = torch.load(model_path, map_location=DEVICE)
    cfg = parse_config(ckpt["config"])
    data_path = Path(cfg.data.file)
    if not data_path.is_absolute():
        data_path = (model_path.parent.parent / data_path).resolve()
    base = np.load(data_path)
    time_base = np.asarray(base["a"])
    model_dt = float(time_base[1] - time_base[0]) if time_base.size > 1 else SIM_FINE_DT
    model_dict = asdict(cfg.model)
    arch_dict = asdict(cfg.architecture)
    model, derived = PHVIV.from_config(dt=model_dt, cfg=model_dict, arch_cfg=arch_dict, device=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, derived, model_dt


def simulate_ground_truth(q0: float, v0: float):
    dt = SIM_FINE_DT
    T = SIM_T
    N = int(np.ceil(T / dt)) + 2
    time = np.zeros(N)
    y = np.zeros(N)
    dy = np.zeros(N)
    ddy = np.zeros(N)
    Fy = np.zeros(N)
    Fcv = np.zeros(N)
    Fdy = np.zeros(N)
    Fca = np.zeros(N)
    phi_vy = np.zeros(N)
    sig_dy = np.zeros(N)
    sig_ddy = np.zeros(N)

    y[0] = q0
    dy[0] = v0

    def acceleration(y_val: float, dy_val: float, force_val: float) -> float:
        return (1.0 / M) * (-C * dy_val - K * y_val + force_val)

    def rk4_step(y_val: float, dy_val: float, force_val: float, dt_val: float):
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

    for i in range(N - 1):
        time[i] = i * dt
        Fy[i + 1], phi_vy[i + 1], sig_dy[i + 1], sig_ddy[i + 1], Fca[i+1], Fcv[i+1], Fdy[i+1] = vforce_CF(
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
            sig_dy[i],
            sig_ddy[i],
        )
        y_next, dy_next = rk4_step(y[i], dy[i], Fy[i + 1], dt)
        y[i + 1] = y_next
        dy[i + 1] = dy_next
        ddy[i + 1] = acceleration(y_next, dy_next, Fy[i + 1])

    time = time[1:-1]
    disp = y[1:-1]
    force = Fcv[1:-1] + Fdy[1:-1]
    return time, disp, force


def downsample_series(time: np.ndarray, disp: np.ndarray, force: np.ndarray, target_dt: float):
    if time.size < 2:
        return time, disp, force
    fine_dt = time[1] - time[0]
    step = max(1, int(round(target_dt / fine_dt)))
    return time[::step], disp[::step], force[::step]


def evaluate_initial_condition(model, derived, q0: float, v0: float, target_dt: float):
    t_fine, disp_fine, force_fine = simulate_ground_truth(q0, v0)
    t, disp_true, force_true = downsample_series(t_fine, disp_fine, force_fine, target_dt)
    dt = target_dt if t.size > 1 else target_dt
    vel_true = compute_velocity_numpy(disp_true, dt)
    y_tensor = torch.from_numpy(disp_true).float().to(DEVICE)
    vel_tensor = torch.from_numpy(vel_true).float().to(DEVICE)
    rollout = rollout_model(
        model,
        y_tensor,
        vel_tensor,
        derived["m_eff"],
        dt,
        t,
        derived["D"],
        derived["k"],
        DEVICE,
    )
    disp_pred = rollout["y_norm"] * derived["D"]
    force_pred = rollout["force_total"]
    min_len = min(len(disp_true), len(disp_pred))
    disp_slice = disp_true[:min_len]
    force_slice = force_true[:min_len]
    disp_rmse = float(np.sqrt(np.mean((disp_pred[:min_len] - disp_slice) ** 2)))
    force_rmse = float(np.sqrt(np.mean((force_pred[:min_len] - force_slice) ** 2)))
    disp_range = float(np.max(disp_slice) - np.min(disp_slice))
    force_range = float(np.max(force_slice) - np.min(force_slice))
    if disp_range <= 1e-9:
        disp_range = 1.0
    if force_range <= 1e-9:
        force_range = 1.0
    return disp_rmse / disp_range, force_rmse / force_range


def plot_error_field(
    q_vals: np.ndarray,
    v_vals: np.ndarray,
    errors: np.ndarray,
    name: str,
    vmin: float | None = None,
    vmax: float | None = None,
):
    fig, ax = plt.subplots(figsize=(6, 5))
    q_mesh, v_mesh = np.meshgrid(q_vals, v_vals, indexing="ij")
    mesh = ax.pcolormesh(q_mesh, v_mesh, errors, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xlabel("Initial displacement q0")
    ax.set_ylabel("Initial velocity v0")
    ax.set_title(f"Initial-condition error ({name})")
    fig.colorbar(mesh, ax=ax, label="RMSE")
    fig.tight_layout()
    out = Path("figs") / f"phase_error_{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    model, derived, model_dt = load_model(MODEL_PATH)
    disp_errors = np.zeros((GRID_Q.size, GRID_V.size))
    force_errors = np.zeros_like(disp_errors)
    for i, q0 in enumerate(GRID_Q):
        for j, v0 in enumerate(GRID_V):
            disp_rmse, force_rmse = evaluate_initial_condition(model, derived, q0, v0, model_dt)
            disp_errors[i, j] = disp_rmse
            force_errors[i, j] = force_rmse
            print(f"q={q0:.3f}, v={v0:.3f} -> disp={disp_rmse:.3e}, force={force_rmse:.3e}")

    disp_lims = (0.0, 0.5)
    force_lims = (0.0, 0.2)
    plot_error_field(GRID_Q, GRID_V, disp_errors, "disp", vmin=disp_lims[0], vmax=disp_lims[1])
    plot_error_field(GRID_Q, GRID_V, force_errors, "force", vmin=force_lims[0], vmax=force_lims[1])


if __name__ == "__main__":
    main()
