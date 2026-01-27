"""Evaluate a trained HNN on cached TD simulations and plot error maps."""

from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
import sys
import json
import tempfile
from typing import Dict, Optional, Tuple, cast

import numpy as np
import torch

if "MPLCONFIGDIR" not in os.environ:
    mpl_dir = Path(tempfile.gettempdir()) / f"mplconfig-{os.getpid()}"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from HNN_helper import PHVIV, Residual, parse_config, rollout_model
from architectures import ODEPirateNet


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return float(default)
    return float(value)


def _env_optional_float(name: str) -> float | None:
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    if value.lower() in {"none", "null"}:
        return None
    return float(value)


def _resolve_under_root(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (ROOT_DIR / path).resolve()


MODEL_PATH = _resolve_under_root(Path(os.environ.get("PHASE_MODEL_PATH", "models/pirate_smoke_0122-125008.pt")))
LOGGED_RUNS_DIR = _resolve_under_root(Path(os.environ.get("PHASE_RUNS_DIR", "Data_Gen/groundtruth_runs_100hz")))
LOGGER_HZ = _env_float("PHASE_LOGGER_HZ", 100.0)
STEADY_STATE_WINDOW_S: float | None = _env_optional_float("PHASE_STEADY_STATE_WINDOW_S")
EVAL_BATCH_SIZE = int(os.environ.get("PHASE_EVAL_BATCH_SIZE", "256"))
PRINT_PER_RUN = os.environ.get("PHASE_PRINT_PER_RUN", "").strip().lower() in {"1", "true", "yes"}

DEVICE = torch.device(
    os.environ.get("PHASE_DEVICE", "").strip()
    or ("cuda" if torch.cuda.is_available() else "cpu")
)

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

def _normalize_state_dict_keys(state: dict) -> dict:
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    if any(k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    return state


def _load_checkpoint(model_path: Path) -> tuple[dict, object, str]:
    ckpt = torch.load(model_path, map_location=DEVICE)
    cfg = parse_config(ckpt.get("config", {}))
    method = str(getattr(cfg, "method", ckpt.get("method", "hnn"))).strip().lower()
    return ckpt, cfg, method


class ForceMLP(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, depth: int, activation: str) -> None:
        super().__init__()
        act = str(activation).strip().lower()
        if act == "tanh":
            act_cls = torch.nn.Tanh
        elif act == "relu":
            act_cls = torch.nn.ReLU
        elif act == "gelu":
            act_cls = torch.nn.GELU
        elif act in {"silu", "swish"}:
            act_cls = torch.nn.SiLU
        else:
            raise ValueError("activation must be one of: tanh, relu, gelu, silu")
        layers: list[torch.nn.Module] = []
        in_features = int(input_dim)
        for _ in range(max(1, int(depth))):
            layers.append(torch.nn.Linear(in_features, int(hidden_dim)))
            layers.append(act_cls())
            in_features = int(hidden_dim)
        layers.append(torch.nn.Linear(in_features, int(output_dim)))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _build_vpinn_force_model(cfg: object, *, input_dim: int, output_dim: int) -> torch.nn.Module:
    vp = dict(getattr(cfg, "vpinn", {}) or {})
    use_arch_cfg = bool(vp.get("use_architecture_config", False))
    if not use_arch_cfg:
        return ForceMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=int(vp.get("hidden_dim", 128)),
            depth=int(vp.get("depth", 3)),
            activation=str(vp.get("activation", "tanh")),
        )

    arch = getattr(cfg, "architecture", None)
    if arch is None:
        raise ValueError("vpinn.use_architecture_config is True but config has no 'architecture:' block.")
    net_type = str(getattr(arch, "force_net_type", "residual")).strip().lower()
    if net_type == "pirate":
        pirate_kwargs = {}
        pirate_kwargs.update(getattr(getattr(cfg, "model", None), "pirate_force_kwargs", {}) or {})
        pirate_kwargs.update(getattr(arch, "pirate_force_kwargs", {}) or {})
        pirate_kwargs.setdefault("depth", 2)
        pirate_kwargs.setdefault("fourier_features", 64)
        pirate_kwargs.setdefault("sigma", 1.0)
        pirate_kwargs.setdefault("activation", "tanh")
        return ODEPirateNet(
            input_size=int(input_dim),
            output_size=int(output_dim),
            **pirate_kwargs,
        )
    if net_type == "residual":
        cfg_res = dict(getattr(arch, "residual_kwargs", {}) or {})
        hidden = int(cfg_res.get("hidden", 128))
        layers = int(cfg_res.get("layers", 2))
        activation = str(cfg_res.get("activation", "gelu"))
        layers_list: list[torch.nn.Module] = [torch.nn.Linear(int(input_dim), hidden)]
        for _ in range(max(1, layers)):
            layers_list.append(Residual(hidden, activation=activation))
        layers_list.append(torch.nn.Linear(hidden, int(output_dim)))
        return torch.nn.Sequential(*layers_list)
    if net_type == "mlp":
        cfg_mlp = dict(getattr(arch, "mlp_kwargs", {}) or {})
        hidden = int(cfg_mlp.get("hidden", 128))
        layers = int(cfg_mlp.get("layers", 2))
        activation = str(cfg_mlp.get("activation", "gelu")).strip().lower()
        act_mod: torch.nn.Module
        if activation == "tanh":
            act_mod = torch.nn.Tanh()
        elif activation == "relu":
            act_mod = torch.nn.ReLU()
        elif activation == "gelu":
            act_mod = torch.nn.GELU()
        elif activation in {"silu", "swish"}:
            act_mod = torch.nn.SiLU()
        else:
            raise ValueError("activation must be one of: tanh, relu, gelu, silu")
        modules: list[torch.nn.Module] = []
        in_features = int(input_dim)
        for _ in range(max(1, layers)):
            modules.append(torch.nn.Linear(in_features, hidden))
            modules.append(act_mod)
            in_features = hidden
        modules.append(torch.nn.Linear(in_features, int(output_dim)))
        return torch.nn.Sequential(*modules)
    raise ValueError("architecture.force_net_type must be one of: residual, mlp, pirate")


def _m_eff_from_cfg(cfg: object) -> float:
    model_cfg = getattr(cfg, "model", None)
    rho = float(getattr(model_cfg, "rho", 1000.0))
    D = float(getattr(model_cfg, "D", 0.1))
    Ca = float(getattr(model_cfg, "Ca", 1.0))
    structural_mass = float(getattr(model_cfg, "structural_mass", 16.79))
    m_a = 0.25 * float(np.pi) * D * D * rho * Ca
    return structural_mass + m_a


def load_model(model_path: Path, *, dt: float) -> tuple[object, dict[str, float], float, str]:
    ckpt, cfg, method = _load_checkpoint(model_path)
    state = _normalize_state_dict_keys(ckpt["model_state"])
    if method in {"hnn", "phnn"}:
        model_dict = asdict(cfg.model)
        arch_dict = asdict(cfg.architecture)
        model, derived = PHVIV.from_config(dt=float(dt), cfg=model_dict, arch_cfg=arch_dict, device=DEVICE)
        incompatible = model.load_state_dict(state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            print(
                f"[warn] {model_path.name}: missing_keys={incompatible.missing_keys}, "
                f"unexpected_keys={incompatible.unexpected_keys}"
            )
        model.eval()
        return model, derived, float(dt), method

    if method == "vpinn":
        # VPINN checkpoint stores a pointwise force network; rollout uses the known ODE.
        model_cfg = getattr(cfg, "model", None)
        d = 1
        force_model = _build_vpinn_force_model(cfg, input_dim=2 * d, output_dim=d).to(DEVICE)
        incompatible = force_model.load_state_dict(state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            print(
                f"[warn] {model_path.name}: missing_keys={incompatible.missing_keys}, "
                f"unexpected_keys={incompatible.unexpected_keys}"
            )
        force_model.eval()
        derived = {
            "m_eff": _m_eff_from_cfg(cfg),
            "k": float(getattr(model_cfg, "k", 1.0)),
            "c": float(getattr(model_cfg, "damping_c", 0.0)),
            "D": float(getattr(model_cfg, "D", 1.0)),
        }
        return force_model, derived, float(dt), method

    raise ValueError(f"Unsupported method '{method}' in checkpoint/config.")


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
    dy: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    if window_s is None:
        return time, disp, force, dy
    if time.size == 0:
        return time, disp, force, dy
    t_end = float(time[-1])
    mask = time >= (t_end - float(window_s))
    if not np.any(mask):
        return time, disp, force, dy
    dy_sliced = None if dy is None else cast(np.ndarray, dy)[mask]
    return time[mask], disp[mask], force[mask], dy_sliced


def evaluate_run(
    model: PHVIV,
    derived: dict[str, float],
    *,
    time: np.ndarray,
    disp_true: np.ndarray,
    force_true: np.ndarray,
) -> tuple[float, float]:
    dt = float(time[1] - time[0]) if time.size > 1 else 1.0
    # NOTE: rollout_model only uses the initial velocity value (vel[0]) to
    # construct the initial momentum. Computing a full velocity time-series is
    # therefore wasted work during large sweeps. The main() function uses the
    # faster batched evaluator below; keep this for debugging.
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


def _eval_rollout_nrmse_batch(
    *,
    model: PHVIV,
    derived: dict[str, float],
    dt: float,
    disp_true: torch.Tensor,  # (B, T)
    force_true: torch.Tensor,  # (B, T)
    v0: torch.Tensor,  # (B,)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fast batched rollout evaluation.

    Matches the semantics used for rollout evaluation in this script:
    - Initial state uses (y0, p0=m_eff*v0)
    - Compare y_pred[k] and F_pred[k]=u_theta(state[k]) to ground truth at sample k
    - Advance with RK4 using model.step_rk4
    """
    if disp_true.ndim != 2 or force_true.ndim != 2:
        raise ValueError("disp_true and force_true must have shape (B, T)")
    if disp_true.shape != force_true.shape:
        raise ValueError("disp_true and force_true must have the same shape")
    if v0.ndim != 1 or v0.shape[0] != disp_true.shape[0]:
        raise ValueError("v0 must have shape (B,)")

    B, T = disp_true.shape
    m_eff = float(derived["m_eff"])

    y0 = disp_true[:, 0]
    state = torch.stack((y0, v0 * m_eff), dim=1)  # (B, 2)

    sum_sq_y = torch.zeros((B,), device=disp_true.device, dtype=torch.float32)
    sum_sq_f = torch.zeros((B,), device=disp_true.device, dtype=torch.float32)

    disp_std = torch.std(disp_true.float(), dim=1)
    force_std = torch.std(force_true.float(), dim=1)
    disp_std = torch.where(disp_std > 1e-9, disp_std, torch.ones_like(disp_std))
    force_std = torch.where(force_std > 1e-9, force_std, torch.ones_like(force_std))

    t_dummy = torch.zeros((B,), device=disp_true.device, dtype=disp_true.dtype)
    dt_t = float(dt)

    with torch.inference_mode():
        for k in range(int(T)):
            y_pred = state[:, 0]
            f_pred = model.u_theta(state).squeeze(-1)
            dy = (y_pred - disp_true[:, k]).float()
            df = (f_pred - force_true[:, k]).float()
            sum_sq_y = sum_sq_y + dy * dy
            sum_sq_f = sum_sq_f + df * df
            state = model.step_rk4(state, t_dummy, dt_t)

    rmse_y = torch.sqrt(sum_sq_y / float(max(T, 1)))
    rmse_f = torch.sqrt(sum_sq_f / float(max(T, 1)))
    nrmse_y = (rmse_y / disp_std).detach().cpu().numpy()
    nrmse_f = (rmse_f / force_std).detach().cpu().numpy()
    return nrmse_y, nrmse_f


def _eval_vpinn_rollout_nrmse_batch(
    *,
    force_model: torch.nn.Module,
    derived: dict[str, float],
    dt: float,
    disp_true: torch.Tensor,  # (B, T)
    force_true: torch.Tensor,  # (B, T)
    v0: torch.Tensor,  # (B,)
) -> tuple[np.ndarray, np.ndarray]:
    """
    VPINN rollout with known dynamics:
        x' = v
        v' = (f_theta(x,v) - c v - k x) / m
    """
    if disp_true.ndim != 2 or force_true.ndim != 2:
        raise ValueError("disp_true and force_true must have shape (B, T)")
    if disp_true.shape != force_true.shape:
        raise ValueError("disp_true and force_true must have the same shape")
    if v0.ndim != 1 or v0.shape[0] != disp_true.shape[0]:
        raise ValueError("v0 must have shape (B,)")

    B, T = disp_true.shape
    m_eff = float(derived["m_eff"])
    c = float(derived.get("c", 0.0))
    k = float(derived.get("k", 0.0))
    dt_t = float(dt)

    x = disp_true[:, 0]
    v = v0

    sum_sq_x = torch.zeros((B,), device=disp_true.device, dtype=torch.float32)
    sum_sq_f = torch.zeros((B,), device=disp_true.device, dtype=torch.float32)

    disp_std = torch.std(disp_true.float(), dim=1)
    force_std = torch.std(force_true.float(), dim=1)
    disp_std = torch.where(disp_std > 1e-9, disp_std, torch.ones_like(disp_std))
    force_std = torch.where(force_std > 1e-9, force_std, torch.ones_like(force_std))

    def f_theta(xi: torch.Tensor, vi: torch.Tensor) -> torch.Tensor:
        inp = torch.stack((xi, vi), dim=1)
        return force_model(inp).squeeze(-1)

    def accel(xi: torch.Tensor, vi: torch.Tensor) -> torch.Tensor:
        return (f_theta(xi, vi) - c * vi - k * xi) / m_eff

    with torch.inference_mode():
        for t_idx in range(int(T)):
            f0 = f_theta(x, v)
            dx = (x - disp_true[:, t_idx]).float()
            df = (f0 - force_true[:, t_idx]).float()
            sum_sq_x = sum_sq_x + dx * dx
            sum_sq_f = sum_sq_f + df * df

            k1_x = v
            k1_v = (f0 - c * v - k * x) / m_eff

            x2 = x + 0.5 * dt_t * k1_x
            v2 = v + 0.5 * dt_t * k1_v
            k2_x = v2
            k2_v = accel(x2, v2)

            x3 = x + 0.5 * dt_t * k2_x
            v3 = v + 0.5 * dt_t * k2_v
            k3_x = v3
            k3_v = accel(x3, v3)

            x4 = x + dt_t * k3_x
            v4 = v + dt_t * k3_v
            k4_x = v4
            k4_v = accel(x4, v4)

            x = x + (dt_t / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
            v = v + (dt_t / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)

    rmse_x = torch.sqrt(sum_sq_x / float(max(T, 1)))
    rmse_f = torch.sqrt(sum_sq_f / float(max(T, 1)))
    nrmse_x = (rmse_x / disp_std).detach().cpu().numpy()
    nrmse_f = (rmse_f / force_std).detach().cpu().numpy()
    return nrmse_x, nrmse_f


def _load_run(
    run_path: Path,
    *,
    prefer_dy: bool = True,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    with np.load(run_path) as run:
        amp = float(np.asarray(run["A_factor"])) if "A_factor" in run else float("nan")
        fhat = float(np.asarray(run["fhat"])) if "fhat" in run else float("nan")
        time = np.asarray(run["time"])
        disp = np.asarray(run["y"])
        force = np.asarray(run["F_total"])
        dy: Optional[np.ndarray] = None
        if prefer_dy and "dy" in run:
            dy = np.asarray(run["dy"])
    return amp, fhat, time, disp, force, dy


def _print_nrmse_summary(name: str, errors: np.ndarray) -> None:
    finite = errors[np.isfinite(errors)]
    if finite.size == 0:
        print(f"{name}: NRMSE summary: no finite values")
        return
    print(
        f"{name}: NRMSE summary over {finite.size} grid points: "
        f"mean={float(np.mean(finite)):.4e}, min={float(np.min(finite)):.4e}, max={float(np.max(finite)):.4e}"
    )


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
    fig.colorbar(mesh, ax=ax, label="NRMSE (RMSE / std(ground truth))")
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
    if DEVICE.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    run_files = _iter_logged_runs(LOGGED_RUNS_DIR, logger_hz=LOGGER_HZ)
    model_dt = _infer_dt_from_run(run_files[0])
    model, derived, _, method = load_model(MODEL_PATH, dt=model_dt)
    print(f"Loaded checkpoint method='{method}' on device='{DEVICE}'", flush=True)
    if PLOT_TRAINING_POINTS:
        cfg = _load_config_from_checkpoint(MODEL_PATH)
        training_points = _load_training_points_from_config(cfg)
    else:
        training_points = None

    accum: Dict[Tuple[float, float], Tuple[float, float, int]] = {}
    amps_set: set[float] = set()
    fhats_set: set[float] = set()

    # Group runs by (length, dt) so we can evaluate rollouts in batches.
    buckets: dict[
        tuple[int, float],
        list[tuple[float, float, np.ndarray, np.ndarray, np.ndarray, Optional[float]]],
    ] = {}
    for run_path in run_files:
        amp, fhat, time, disp, force, dy = _load_run(run_path, prefer_dy=True)
        if not np.isfinite(amp) or not np.isfinite(fhat):
            print(f"[warn] Skipping {run_path.name}: missing A_factor/fhat metadata.")
            continue
        time, disp, force, dy = _maybe_slice_steady_state(
            time, disp, force, window_s=STEADY_STATE_WINDOW_S, dy=dy
        )
        if time.size < 2:
            print(f"[warn] Skipping {run_path.name}: too few samples.")
            continue
        dt = float(time[1] - time[0])
        dt_key = float(np.round(dt, 12))
        dy0 = float(dy[0]) if dy is not None and dy.size else None
        buckets.setdefault((int(time.size), dt_key), []).append((amp, fhat, time, disp, force, dy0))

    for (T, dt_key), runs in sorted(buckets.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        for start in range(0, len(runs), int(EVAL_BATCH_SIZE)):
            batch = runs[start : start + int(EVAL_BATCH_SIZE)]
            amps_np = np.asarray([b[0] for b in batch], dtype=float)
            fhats_np = np.asarray([b[1] for b in batch], dtype=float)
            disp_batch = np.stack([b[3] for b in batch], axis=0).astype(np.float32, copy=False)
            force_batch = np.stack([b[4] for b in batch], axis=0).astype(np.float32, copy=False)

            v0_list: list[float] = []
            for _amp, _fhat, t_np, y_np, _f_np, dy0 in batch:
                if dy0 is not None:
                    v0_list.append(float(dy0))
                elif y_np.size >= 2:
                    v0_list.append(float((y_np[1] - y_np[0]) / float(t_np[1] - t_np[0])))
                else:
                    v0_list.append(0.0)
            v0_np = np.asarray(v0_list, dtype=np.float32)

            disp_true_t = torch.from_numpy(disp_batch).to(DEVICE, non_blocking=(DEVICE.type == "cuda"))
            force_true_t = torch.from_numpy(force_batch).to(DEVICE, non_blocking=(DEVICE.type == "cuda"))
            v0_t = torch.from_numpy(v0_np).to(DEVICE, non_blocking=(DEVICE.type == "cuda"))

            if method in {"hnn", "phnn"}:
                disp_nrmse, force_nrmse = _eval_rollout_nrmse_batch(
                    model=cast(PHVIV, model),
                    derived=derived,
                    dt=float(dt_key),
                    disp_true=disp_true_t,
                    force_true=force_true_t,
                    v0=v0_t,
                )
            elif method == "vpinn":
                disp_nrmse, force_nrmse = _eval_vpinn_rollout_nrmse_batch(
                    force_model=cast(torch.nn.Module, model),
                    derived=derived,
                    dt=float(dt_key),
                    disp_true=disp_true_t,
                    force_true=force_true_t,
                    v0=v0_t,
                )
            else:
                raise ValueError(f"Unsupported method '{method}'")

            for amp, fhat, dn, fn in zip(amps_np, fhats_np, disp_nrmse, force_nrmse):
                amps_set.add(float(amp))
                fhats_set.add(float(fhat))
                key = (float(amp), float(fhat))
                prev = accum.get(key)
                if prev is None:
                    accum[key] = (float(dn), float(fn), 1)
                else:
                    d_sum, f_sum, n = prev
                    accum[key] = (d_sum + float(dn), f_sum + float(fn), n + 1)
                if PRINT_PER_RUN:
                    print(f"A={amp:.3f}, fhat={fhat:.5f} -> disp_nrmse={dn:.3e}, force_nrmse={fn:.3e}")

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

    _print_nrmse_summary("displacement", disp_errors)
    _print_nrmse_summary("force_total", force_errors)

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
