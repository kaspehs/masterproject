"""Generate a surface plot of the learned force network over displacement/momentum grid."""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from HNN_helper import PHVIV, parse_config


def iter_logged_runs(runs_dir: Path, *, logger_hz: float) -> tuple[list[Path], float]:
    pattern = f"*log{int(logger_hz):d}Hz.npz"
    files = sorted(runs_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No logged runs found in {runs_dir} matching '{pattern}'.")
    return files, float(logger_hz)


def load_model(ckpt_path: Path, device: torch.device) -> tuple[PHVIV, dict[str, float]]:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_raw = ckpt.get("config", {})
    if not isinstance(cfg_raw, dict):
        if hasattr(cfg_raw, "__dict__"):
            cfg_raw = dict(cfg_raw.__dict__)
        else:
            raise TypeError(f"Unsupported config type in checkpoint: {type(cfg_raw)}")
    cfg = parse_config(cfg_raw)
    dt = float(ckpt.get("dt", 0.01))
    model_dict = asdict(cfg.model)
    arch_dict = asdict(cfg.architecture)
    model, derived = PHVIV.from_config(dt=dt, cfg=model_dict, arch_cfg=arch_dict, device=device)
    state = ckpt["model_state"]
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    if any(k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            f"[warn] {ckpt_path.name}: missing_keys={incompatible.missing_keys}, "
            f"unexpected_keys={incompatible.unexpected_keys}"
        )
    model.eval()
    return model, derived


def evaluate_force_surface(
    model: PHVIV,
    q_values: np.ndarray,
    p_values: np.ndarray,
    reduced_velocity: float,
    device: torch.device,
):
    QQ, PP = np.meshgrid(q_values, p_values, indexing="ij")
    grid = np.stack([QQ, PP], axis=-1)
    with torch.no_grad():
        inputs = torch.from_numpy(grid.reshape(-1, 2)).float().to(device)
        forces = model.u_theta(inputs, reduced_velocity=reduced_velocity).cpu().numpy().reshape(QQ.shape)
    return QQ, PP, forces


def plot_heatmap(
    QQ: np.ndarray,
    PP: np.ndarray,
    forces: np.ndarray,
    save_path: Path | None = None,
    limit_path: tuple[np.ndarray, np.ndarray] | None = None,
):
    fig, ax = plt.subplots(figsize=(6, 5))
    finite = forces[np.isfinite(forces)]
    if finite.size:
        max_abs = np.max(np.abs(finite))
        if max_abs <= 0:
            max_abs = 1.0
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-max_abs, vmax=max_abs)
    else:
        norm = None
    mesh = ax.pcolormesh(QQ, PP, forces, shading="auto", cmap=FORCE_CMAP, norm=norm)
    ax.set_xlabel("Displacement q")
    ax.set_ylabel("Momentum p")
    # no title for cleaner comparison
    if limit_path is not None:
        q_path, p_path = limit_path
        ax.plot(q_path, p_path, color="black", linewidth=1.0, linestyle="--")
    fig.colorbar(mesh, ax=ax, label="Force")
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        print(f"Saved heatmap to {save_path}")
    else:
        plt.show()
    plt.close(fig)


CKPT_PATH = Path("models/pirate_smoke_0122-125008.pt")
LOGGED_RUNS_DIR = Path("Data_Gen/groundtruth_runs_100hz")
LOGGER_HZ = 100.0
STEADY_STATE_WINDOW_S = 4.0
FORCE_CMAP = LinearSegmentedColormap.from_list(
    "force_diverging",
    [
        (0.0, "blue"),
        (0.5, "0.8"),
        (1.0, "red"),
    ],
)


def main():
    device = torch.device("cpu")
    model, derived = load_model(CKPT_PATH, device)
    q_range = np.linspace(-derived["D"], derived["D"], 100)
    p_scale = np.sqrt(derived["k"] * derived["m_eff"]) * derived["D"]
    p_range = np.linspace(-p_scale, p_scale, 100)
    save_path = Path("figs") / f"force_heatmap_{CKPT_PATH.stem}.png"
    q_edges = np.linspace(q_range.min(), q_range.max(), q_range.size + 1)
    p_edges = np.linspace(p_range.min(), p_range.max(), p_range.size + 1)

    runs_dir = LOGGED_RUNS_DIR
    run_files, _ = iter_logged_runs(runs_dir, logger_hz=LOGGER_HZ)
    print(f"Using {len(run_files)} logged runs from {runs_dir}")
    with np.load(run_files[0]) as run:
        if "U_r" not in run:
            raise KeyError(f"{run_files[0].name} is missing reduced velocity 'U_r'.")
        ur_val = float(np.asarray(run["U_r"]).reshape(-1)[0])
    QQ, PP, forces = evaluate_force_surface(model, q_range, p_range, ur_val, device)
    sums = np.zeros_like(QQ)
    counts = np.zeros_like(QQ)
    squared = np.zeros_like(QQ)
    limit_path = None

    for run_path in run_files:
        with np.load(run_path) as run:
            q_vals = np.asarray(run["y"])
            p_vals = np.asarray(run["dy"]) * float(derived["m_eff"])
            f_vals = np.asarray(run["F_total"])
            time = np.asarray(run["time"])
        if limit_path is None:
            mask = time >= (time[-1] - float(STEADY_STATE_WINDOW_S))
            if np.any(mask):
                limit_path = (q_vals[mask], p_vals[mask])
        q_idx = np.digitize(q_vals, q_edges) - 1
        p_idx = np.digitize(p_vals, p_edges) - 1
        valid = (q_idx >= 0) & (q_idx < q_range.size) & (p_idx >= 0) & (p_idx < p_range.size)
        np.add.at(sums, (q_idx[valid], p_idx[valid]), f_vals[valid])
        np.add.at(squared, (q_idx[valid], p_idx[valid]), f_vals[valid] ** 2)
        np.add.at(counts, (q_idx[valid], p_idx[valid]), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(counts > 0, sums / counts, np.nan)
        var = np.where(counts > 0, squared / counts - mean**2, np.nan)
        std = np.sqrt(np.clip(var, a_min=0.0, a_max=None))
    if limit_path is None:
        limit_path = (np.array([]), np.array([]))
    plot_heatmap(QQ, PP, forces, save_path=save_path, limit_path=limit_path)
    plot_heatmap(QQ, PP, mean, save_path=Path("figs") / "force_heatmap_groundtruth.png", limit_path=limit_path)
    plot_heatmap(QQ, PP, std, save_path=Path("figs") / "force_std_groundtruth.png", limit_path=limit_path)
    diff = forces - mean
    plot_heatmap(
        QQ,
        PP,
        diff,
        save_path=Path("figs") / f"force_heatmap_diff_{CKPT_PATH.stem}.png",
        limit_path=limit_path,
    )

if __name__ == "__main__":
    main()
