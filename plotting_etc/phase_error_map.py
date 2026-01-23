"""Evaluate a trained HNN on cached TD simulations and plot error maps."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys
import json
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from HNN_helper import PHVIV, parse_config, compute_velocity_numpy, rollout_model


MODEL_PATH = Path("models/pirate_smoke_0122-125008.pt")
DEVICE = torch.device("cpu")
LOGGED_RUNS_DIR = Path("Data_Gen/groundtruth_runs_100hz")
LOGGER_HZ = 100.0
STEADY_STATE_WINDOW_S: float | None = None  # use full series; set e.g. 2.0 for last 2s

# Overlay training cases (A_factor, fhat) on the maps.
PLOT_TRAINING_POINTS = True
TRAINING_POINTS_LABEL = "trained on"
TRAINING_POINTS_SIZE = 18
TRAINING_POINTS_COLOR = "white"
TRAINING_POINTS_EDGE = "black"
TRAINING_POINTS_LINEWIDTH = 0.5

# Colormap limits
AUTO_COLOR_LIMITS = True
DISP_LIMITS = (0.0, 0.5)
FORCE_LIMITS = (0.0, 0.2)
AUTO_LIMIT_Q_LOW = 0.0
AUTO_LIMIT_Q_HIGH = 0.98


def _iter_logged_runs(runs_dir: Path, *, logger_hz: float) -> list[Path]:
    pattern = f"*log{int(logger_hz):d}Hz.npz"
    files = sorted(runs_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No logged runs found in {runs_dir} matching '{pattern}'.")
    return files


def _infer_dt_from_run(run_path: Path) -> float:
    with np.load(run_path) as run:
        if "dt_logger" in run:
            return float(np.asarray(run["dt_logger"]))
        time = np.asarray(run["time"])
    if time.size < 2:
        raise ValueError(f"Run {run_path} has too few samples to infer dt.")
    return float(time[1] - time[0])


def load_model(model_path: Path, *, dt: float) -> tuple[PHVIV, dict[str, float], float]:
    ckpt = torch.load(model_path, map_location=DEVICE)
    cfg = parse_config(ckpt["config"])
    model_dict = asdict(cfg.model)
    arch_dict = asdict(cfg.architecture)
    model, derived = PHVIV.from_config(dt=float(dt), cfg=model_dict, arch_cfg=arch_dict, device=DEVICE)
    incompatible = model.load_state_dict(ckpt["model_state"], strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            f"[warn] {model_path.name}: missing_keys={incompatible.missing_keys}, "
            f"unexpected_keys={incompatible.unexpected_keys}"
        )
    model.eval()
    return model, derived, float(dt)


def _load_config_from_checkpoint(model_path: Path):
    ckpt = torch.load(model_path, map_location="cpu")
    raw_cfg = ckpt.get("config", {})
    if not isinstance(raw_cfg, dict):
        if hasattr(raw_cfg, "__dict__"):
            raw_cfg = dict(raw_cfg.__dict__)
        else:
            raise TypeError(f"Unsupported config type in checkpoint: {type(raw_cfg)}")
    return parse_config(raw_cfg)


def _load_training_points_from_config(cfg) -> tuple[np.ndarray, np.ndarray]:
    if not getattr(cfg.data, "use_generated_train_series", False):
        return np.array([]), np.array([])
    series_dir = Path(getattr(cfg.data, "train_series_dir", ""))
    if not series_dir:
        return np.array([]), np.array([])
    if not series_dir.is_absolute():
        series_dir = (ROOT_DIR / series_dir).resolve()
    meta_path = series_dir / "metadata.json"
    if not meta_path.exists():
        return np.array([]), np.array([])
    with meta_path.open("r", encoding="utf-8") as fh:
        meta = json.load(fh) or []
    amps: list[float] = []
    fhats: list[float] = []
    for entry in meta:
        try:
            amps.append(float(entry["A_factor"]))
            fhats.append(float(entry["fhat"]))
        except Exception:
            continue
    return np.asarray(amps, dtype=float), np.asarray(fhats, dtype=float)


def _maybe_slice_steady_state(
    time: np.ndarray,
    disp: np.ndarray,
    force: np.ndarray,
    *,
    window_s: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if window_s is None:
        return time, disp, force
    if time.size == 0:
        return time, disp, force
    t_end = float(time[-1])
    mask = time >= (t_end - float(window_s))
    if not np.any(mask):
        return time, disp, force
    return time[mask], disp[mask], force[mask]


def evaluate_run(
    model: PHVIV,
    derived: dict[str, float],
    *,
    time: np.ndarray,
    disp_true: np.ndarray,
    force_true: np.ndarray,
) -> tuple[float, float]:
    dt = float(time[1] - time[0]) if time.size > 1 else 1.0
    vel_true = compute_velocity_numpy(disp_true, dt)
    y_tensor = torch.from_numpy(disp_true).float().to(DEVICE)
    vel_tensor = torch.from_numpy(vel_true).float().to(DEVICE)
    rollout = rollout_model(
        model,
        y_tensor,
        vel_tensor,
        derived["m_eff"],
        dt,
        time,
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
    disp_std = float(np.std(disp_slice))
    force_std = float(np.std(force_slice))
    if disp_std <= 1e-9:
        disp_std = 1.0
    if force_std <= 1e-9:
        force_std = 1.0
    return disp_rmse / disp_std, force_rmse / force_std


def plot_error_field(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    errors: np.ndarray,
    name: str,
    *,
    x_label: str,
    y_label: str,
    overlay_points: tuple[np.ndarray, np.ndarray] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
):
    fig, ax = plt.subplots(figsize=(6, 5))
    x_mesh, y_mesh = np.meshgrid(x_vals, y_vals, indexing="ij")
    mesh = ax.pcolormesh(x_mesh, y_mesh, errors, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    if overlay_points is not None:
        px, py = overlay_points
        if px.size and py.size:
            ax.scatter(
                px,
                py,
                s=TRAINING_POINTS_SIZE,
                c=TRAINING_POINTS_COLOR,
                edgecolors=TRAINING_POINTS_EDGE,
                linewidths=TRAINING_POINTS_LINEWIDTH,
                marker="o",
                label=TRAINING_POINTS_LABEL,
            )
            ax.legend(loc="upper right", frameon=True, framealpha=0.9)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"Error map ({name})")
    fig.colorbar(mesh, ax=ax, label="RMSE / std(ground truth)")
    fig.tight_layout()
    out = Path("figs") / f"phase_error_{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def _auto_limits(errors: np.ndarray, *, q_low: float, q_high: float) -> tuple[float | None, float | None]:
    finite = errors[np.isfinite(errors)]
    if finite.size == 0:
        return None, None
    q_low = float(np.clip(q_low, 0.0, 1.0))
    q_high = float(np.clip(q_high, 0.0, 1.0))
    if q_high < q_low:
        q_low, q_high = q_high, q_low
    vmin = float(np.quantile(finite, q_low))
    vmax = float(np.quantile(finite, q_high))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
    if vmax <= vmin:
        return None, None
    return vmin, vmax


def main():
    run_files = _iter_logged_runs(LOGGED_RUNS_DIR, logger_hz=LOGGER_HZ)
    model_dt = _infer_dt_from_run(run_files[0])
    model, derived, _ = load_model(MODEL_PATH, dt=model_dt)
    if PLOT_TRAINING_POINTS:
        cfg = _load_config_from_checkpoint(MODEL_PATH)
        training_points = _load_training_points_from_config(cfg)
    else:
        training_points = None

    accum: Dict[Tuple[float, float], Tuple[float, float, int]] = {}
    amps_set: set[float] = set()
    fhats_set: set[float] = set()

    for run_path in run_files:
        with np.load(run_path) as run:
            amp = float(np.asarray(run["A_factor"])) if "A_factor" in run else float("nan")
            fhat = float(np.asarray(run["fhat"])) if "fhat" in run else float("nan")
            time = np.asarray(run["time"])
            disp = np.asarray(run["y"])
            force = np.asarray(run["F_total"])

        if not np.isfinite(amp) or not np.isfinite(fhat):
            print(f"[warn] Skipping {run_path.name}: missing A_factor/fhat metadata.")
            continue

        time, disp, force = _maybe_slice_steady_state(time, disp, force, window_s=STEADY_STATE_WINDOW_S)
        disp_rmse, force_rmse = evaluate_run(model, derived, time=time, disp_true=disp, force_true=force)
        amps_set.add(amp)
        fhats_set.add(fhat)
        key = (amp, fhat)
        prev = accum.get(key)
        if prev is None:
            accum[key] = (disp_rmse, force_rmse, 1)
        else:
            d_sum, f_sum, n = prev
            accum[key] = (d_sum + disp_rmse, f_sum + force_rmse, n + 1)
        print(f"A={amp:.3f}, fhat={fhat:.5f} -> disp={disp_rmse:.3e}, force={force_rmse:.3e}")

    amps = np.array(sorted(amps_set), dtype=float)
    fhats = np.array(sorted(fhats_set), dtype=float)
    if amps.size == 0 or fhats.size == 0:
        raise RuntimeError("No valid runs found to build an error map.")

    disp_errors = np.full((amps.size, fhats.size), np.nan, dtype=float)
    force_errors = np.full_like(disp_errors, np.nan)
    for i, amp in enumerate(amps):
        for j, fhat in enumerate(fhats):
            d_sum, f_sum, n = accum.get((float(amp), float(fhat)), (np.nan, np.nan, 0))
            if n > 0:
                disp_errors[i, j] = d_sum / n
                force_errors[i, j] = f_sum / n

    if AUTO_COLOR_LIMITS:
        disp_lims = _auto_limits(disp_errors, q_low=AUTO_LIMIT_Q_LOW, q_high=AUTO_LIMIT_Q_HIGH)
        force_lims = _auto_limits(force_errors, q_low=AUTO_LIMIT_Q_LOW, q_high=AUTO_LIMIT_Q_HIGH)
    else:
        disp_lims = DISP_LIMITS
        force_lims = FORCE_LIMITS
    plot_error_field(
        amps,
        fhats,
        disp_errors,
        "disp",
        x_label="Amplitude factor (A_factor)",
        y_label="Normalized frequency (fhat)",
        overlay_points=training_points,
        vmin=disp_lims[0],
        vmax=disp_lims[1],
    )
    plot_error_field(
        amps,
        fhats,
        force_errors,
        "force",
        x_label="Amplitude factor (A_factor)",
        y_label="Normalized frequency (fhat)",
        overlay_points=training_points,
        vmin=force_lims[0],
        vmax=force_lims[1],
    )


if __name__ == "__main__":
    main()
