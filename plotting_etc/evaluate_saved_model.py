"""Offline evaluation script: load a saved HNN, simulate TD ground truth, compare."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from HNN_helper import Config, PHVIV, compute_velocity_numpy, rollout_model, parse_config
from Data_Gen.simulate_td_model_cf import simulate_td_model_cf


def load_checkpoint(model_path: Path) -> tuple[dict, Config]:
    ckpt = torch.load(model_path, map_location="cpu")
    raw_cfg = ckpt["config"]
    config = parse_config(raw_cfg) 
    return ckpt, config


def build_model(config: Config, dt: float, device: torch.device) -> tuple[PHVIV, dict[str, float]]:
    model_dict = asdict(config.model)
    arch_dict = asdict(config.architecture)
    model, derived = PHVIV.from_config(dt=dt, cfg=model_dict, arch_cfg=arch_dict, device=device)
    return model, derived


def evaluate_series(
    model: PHVIV,
    derived: dict[str, float],
    t: np.ndarray,
    disp: np.ndarray,
    force: np.ndarray,
    reduced_velocity: float,
    smoothing_cfg,
    device: torch.device,
) -> dict[str, float]:
    dt = float(t[1] - t[0]) if t.size > 1 else derived["D"]
    vel = compute_velocity_numpy(
        disp,
        dt,
        use_savgol=smoothing_cfg.use_savgol_smoothing,
        savgol_window=smoothing_cfg.window_length,
        savgol_polyorder=smoothing_cfg.polyorder,
    )
    y_tensor = torch.from_numpy(disp).float().to(device)
    vel_tensor = torch.from_numpy(vel).float().to(device)
    rollout = rollout_model(
        model,
        y_tensor,
        vel_tensor,
        reduced_velocity,
        derived["m_eff"],
        dt,
        t,
        derived["D"],
        derived["k"],
        device,
    )
    y_pred = rollout["y_norm"] * derived["D"]
    force_pred = rollout["force_total"]
    min_len = min(len(y_pred), len(disp))
    disp_rmse = float(np.sqrt(np.mean((y_pred[:min_len] - disp[:min_len]) ** 2)))
    force_rmse = float(np.sqrt(np.mean((force_pred[:min_len] - force[:min_len]) ** 2)))
    return {
        "disp_rmse": disp_rmse,
        "force_rmse": force_rmse,
    }


MODEL_PATH = Path("models/residual_final1_1130-115801.pt")
CASES = [
    (0.2, 0.0),
    (0.0, 0.10),
    (0.0, 0.15),
    (0.0, 0.20),
    (0.0, 0.25),
]
SIM_DT = 0.0001
SIM_T = 10.0
SIM_INTEGRATOR = "rk4"


def simulate_series(a_factor: float, fhat: float, dt: float, T: float, integrator: str, reduction_factor: int = 100):
    sim = simulate_td_model_cf(
        A_factor=a_factor,
        fhat=fhat,
        dt=dt,
        T=T,
        output_path=None,
        plot=False,
        verbose=False,
        integrator=integrator,
    )
    return (
        sim["time"][::reduction_factor],
        sim["y"][::reduction_factor],
        sim["F_total"][::reduction_factor],
        float(sim["U_r"]),
    )


def main():
    reduction_factor = 100
    ckpt, cfg = load_checkpoint(MODEL_PATH)
    device = torch.device("cpu")
    model_dt = SIM_DT*reduction_factor
    sample_time = np.arange(0, SIM_T, model_dt)
    dt = float(sample_time[1] - sample_time[0]) if sample_time.size > 1 else SIM_DT
    model, derived = build_model(cfg, dt=dt, device=device)
    state = ckpt["model_state"]
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    if any(k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            f"[warn] {MODEL_PATH.name}: missing_keys={incompatible.missing_keys}, "
            f"unexpected_keys={incompatible.unexpected_keys}"
        )
    model.eval()

    smoothing_cfg = cfg.smoothing
    results = []
    for amplitude, freq in CASES:
        t, disp, force, ur_val = simulate_series(amplitude, freq, SIM_DT, SIM_T, SIM_INTEGRATOR)
        metrics = evaluate_series(model, derived, t, disp, force, ur_val, smoothing_cfg, device)
        entry = {
            "amplitude": amplitude,
            "frequency": freq,
            **metrics,
        }
        results.append(entry)
        print(
            f"A={amplitude:.2f}, fhat={freq:.3f} -> "
            f"disp RMSE={metrics['disp_rmse']:.4e}, force RMSE={metrics['force_rmse']:.4e}"
        )

    mean_disp = float(np.mean([r["disp_rmse"] for r in results]))
    mean_force = float(np.mean([r["force_rmse"] for r in results]))
    print(f"\nAverage RMSE over {len(results)} simulations: disp={mean_disp:.4e}, force={mean_force:.4e}")


if __name__ == "__main__":
    main()
