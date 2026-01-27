from __future__ import annotations

import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils
from torch.utils.data import DataLoader, Dataset

from core.logging import setup_writer
from core.optim import setup_optimizer_and_scheduler
from core.runtime import (
    configure_tf32,
    maybe_compile_model,
    select_device,
    set_num_threads_from_slurm,
    setup_amp,
)
from HNN_helper import (
    Config,
    GradNormBalancer,
    Residual,
    compute_velocity_numpy,
    create_window_mask,
    create_zoom_mask,
    log_displacement_plots,
    log_force_plots,
    resample_uniform_series,
)
from architectures import ODEPirateNet


class ForceMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        depth: int,
        activation: str,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        act = activation.strip().lower()
        if act == "tanh":
            act_cls: type[nn.Module] = nn.Tanh
        elif act == "relu":
            act_cls = nn.ReLU
        elif act == "gelu":
            act_cls = nn.GELU
        elif act == "silu":
            act_cls = nn.SiLU
        else:
            raise ValueError("activation must be one of: tanh, relu, gelu, silu")

        layers: list[nn.Module] = []
        in_features = int(input_dim)
        for _ in range(int(depth)):
            layers.append(nn.Linear(in_features, int(hidden_dim)))
            layers.append(act_cls())
            in_features = int(hidden_dim)
        layers.append(nn.Linear(in_features, int(output_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _activation_from_string(name: str) -> nn.Module:
    key = str(name).strip().lower()
    if key == "tanh":
        return nn.Tanh()
    if key == "relu":
        return nn.ReLU()
    if key == "gelu":
        return nn.GELU()
    if key in {"silu", "swish"}:
        return nn.SiLU()
    raise ValueError("activation must be one of: tanh, relu, gelu, silu")


def _build_force_model(config: Config, *, input_dim: int, output_dim: int) -> nn.Module:
    vp = dict(config.vpinn or {})
    use_arch_cfg = bool(vp.get("use_architecture_config", False))
    if not use_arch_cfg:
        net_hidden = int(vp.get("hidden_dim", 128))
        net_depth = int(vp.get("depth", 3))
        activation = str(vp.get("activation", "tanh"))
        return ForceMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=net_hidden,
            depth=net_depth,
            activation=activation,
        )

    if config.architecture is None:
        raise ValueError("vpinn.use_architecture_config is True but config has no 'architecture:' block.")
    arch = config.architecture
    net_type = str(getattr(arch, "force_net_type", "residual")).strip().lower()

    if net_type == "pirate":
        pirate_kwargs = {}
        pirate_kwargs.update(getattr(config.model, "pirate_force_kwargs", {}) or {})
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
        cfg = dict(getattr(arch, "residual_kwargs", {}) or {})
        hidden = int(cfg.get("hidden", 128))
        layers = int(cfg.get("layers", 2))
        activation = str(cfg.get("activation", "gelu"))
        layers_list: list[nn.Module] = [nn.Linear(int(input_dim), hidden)]
        for _ in range(max(1, layers)):
            layers_list.append(Residual(hidden, activation=activation))
        layers_list.append(nn.Linear(hidden, int(output_dim)))
        return nn.Sequential(*layers_list)

    if net_type == "mlp":
        cfg = dict(getattr(arch, "mlp_kwargs", {}) or {})
        hidden = int(cfg.get("hidden", 128))
        layers = int(cfg.get("layers", 2))
        activation = _activation_from_string(str(cfg.get("activation", "gelu")))
        modules: list[nn.Module] = []
        in_features = int(input_dim)
        for _ in range(max(1, layers)):
            modules.append(nn.Linear(in_features, hidden))
            modules.append(activation)
            in_features = hidden
        modules.append(nn.Linear(in_features, int(output_dim)))
        return nn.Sequential(*modules)

    raise ValueError("architecture.force_net_type must be one of: residual, mlp, pirate")


def _vpinn_force(model: nn.Module, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return model(torch.cat([x, v], dim=-1))


def rollout_rk4(
    *,
    model: nn.Module,
    x0: torch.Tensor,
    v0: torch.Tensor,
    steps: int,
    dt: float,
    m: torch.Tensor,
    c: torch.Tensor,
    k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Differentiable RK4 rollout of:
        x' = v
        v' = (f_theta(x,v) - c v - k x) / m

    Returns:
        x_seq, v_seq, f_seq with shape (B, steps+1, d)
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    B, d = x0.shape
    m = m.view(1, d)
    c = c.view(1, d)
    k = k.view(1, d)

    x = x0
    v = v0
    xs = [x]
    vs = [v]
    fs = [_vpinn_force(model, x, v)]

    dt_t = x0.new_tensor(float(dt))
    half = x0.new_tensor(0.5)
    sixth = x0.new_tensor(1.0 / 6.0)

    def accel(xi: torch.Tensor, vi: torch.Tensor) -> torch.Tensor:
        fi = _vpinn_force(model, xi, vi)
        return (fi - c * vi - k * xi) / m

    for _ in range(int(steps)):
        k1_x = v
        k1_v = accel(x, v)

        x2 = x + half * dt_t * k1_x
        v2 = v + half * dt_t * k1_v
        k2_x = v2
        k2_v = accel(x2, v2)

        x3 = x + half * dt_t * k2_x
        v3 = v + half * dt_t * k2_v
        k3_x = v3
        k3_v = accel(x3, v3)

        x4 = x + dt_t * k3_x
        v4 = v + dt_t * k3_v
        k4_x = v4
        k4_v = accel(x4, v4)

        x = x + (dt_t * sixth) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
        v = v + (dt_t * sixth) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)

        xs.append(x)
        vs.append(v)
        fs.append(_vpinn_force(model, x, v))

    return torch.stack(xs, dim=1), torch.stack(vs, dim=1), torch.stack(fs, dim=1)


def _m_eff_from_model_cfg(model_cfg: Any) -> float:
    rho = float(getattr(model_cfg, "rho", 1000.0))
    D = float(getattr(model_cfg, "D", 0.1))
    Ca = float(getattr(model_cfg, "Ca", 1.0))
    structural_mass = float(getattr(model_cfg, "structural_mass", 16.79))
    m_a = 0.25 * math.pi * D * D * rho * Ca
    return structural_mass + m_a


def _read_timeseries_npz(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    with np.load(path) as data:
        if "time" in data:
            t = np.asarray(data["time"])
            x = np.asarray(data["y"])
            f = np.asarray(data["F_total"])
            if "dy" in data:
                v = np.asarray(data["dy"])
            elif "v" in data:
                v = np.asarray(data["v"])
            else:
                v = None
        else:
            t = np.asarray(data["a"])
            x = np.asarray(data["b"])
            f = np.asarray(data["c"])
            # Some generators store velocity as "e" (legacy) or "dy".
            if "dy" in data:
                v = np.asarray(data["dy"])
            elif "e" in data:
                v = np.asarray(data["e"])
            elif "v" in data:
                v = np.asarray(data["v"])
            else:
                v = None
    if t.ndim != 1:
        raise ValueError(f"{path} must contain 1D time array.")
    if x.ndim not in (1, 2) or f.ndim not in (1, 2):
        raise ValueError(f"{path} must contain 1D or 2D displacement/force arrays.")
    if x.shape[0] != t.size or f.shape[0] != t.size:
        raise ValueError(f"{path} has mismatched lengths (t={t.size}, x={x.shape[0]}, f={f.shape[0]}).")
    if x.ndim == 1:
        x = x[:, None]
    if f.ndim == 1:
        f = f[:, None]
    if x.shape != f.shape:
        raise ValueError(f"{path} has mismatched shapes (x={x.shape}, f={f.shape}).")
    if t.size < 2:
        raise ValueError(f"{path} is too short.")
    dt = float(t[1] - t[0])
    if not np.allclose(np.diff(t), dt, rtol=1e-6, atol=1e-9):
        raise ValueError(f"{path} time vector is not uniform.")
    if v is not None:
        v = np.asarray(v)
        if v.ndim not in (1, 2) or v.shape[0] != t.size:
            raise ValueError(f"{path} has invalid 'dy' shape (expected first dim {t.size}).")
        if v.ndim == 1:
            v = v[:, None]
        if v.shape != x.shape:
            raise ValueError(f"{path} has mismatched shapes (x={x.shape}, dy={v.shape}).")
    return t, x, f, v


def _infer_dt_target_from_data_cfg(data_cfg: Any) -> Optional[float]:
    data_path = Path(getattr(data_cfg, "file", ""))
    if not data_path:
        return None
    if not data_path.is_absolute():
        data_path = (Path.cwd() / data_path).resolve()
    if not data_path.exists():
        return None
    with np.load(data_path) as base:
        if "a" not in base:
            return None
        t = np.asarray(base["a"])
    if t.ndim != 1 or t.size < 2:
        return None
    if bool(getattr(data_cfg, "reduce_time", False)):
        rf = int(getattr(data_cfg, "reduction_factor", 1))
        rf = max(1, rf)
        t = t[::rf]
    if t.size < 2:
        return None
    return float(t[1] - t[0])

def _maybe_reduce_time(
    t: np.ndarray,
    x: np.ndarray,
    f: np.ndarray,
    v: Optional[np.ndarray],
    *,
    enabled: bool,
    reduction_factor: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if not enabled:
        return t, x, f, v
    rf = max(1, int(reduction_factor))
    t2 = t[::rf]
    x2 = x[::rf]
    f2 = f[::rf]
    v2 = v[::rf] if v is not None else None
    if t2.size < 2:
        raise ValueError("reduce_time produced too few samples")
    return t2, x2, f2, v2


def _load_trajectory(
    *,
    path: Path,
    dt_target: Optional[float],
    velocity_source: str,
    smoothing_cfg: Any,
    reduce_time: bool,
    reduction_factor: int,
    cut_start_seconds: float,
) -> tuple[dict[str, Any], float]:
    t, x, f_meas, v_file = _read_timeseries_npz(path)
    t, x, f_meas, v_file = _maybe_reduce_time(
        t,
        x,
        f_meas,
        v_file,
        enabled=reduce_time,
        reduction_factor=reduction_factor,
    )
    dt = float(t[1] - t[0])

    if dt_target is not None and not np.isclose(dt, float(dt_target), rtol=1e-9, atol=1e-12):
        t_in = t
        x, t = _resample_uniform_nd(t_in, x, float(dt_target))
        f_meas, t_force = _resample_uniform_nd(t_in, f_meas, float(dt_target))
        if not np.allclose(t, t_force, rtol=1e-9, atol=1e-12):
            raise ValueError(f"{path.name}: resampled x and f landed on different time grids")
        if v_file is not None:
            v_file, t_vel = _resample_uniform_nd(t_in, v_file, float(dt_target))
            if not np.allclose(t, t_vel, rtol=1e-9, atol=1e-12):
                raise ValueError(f"{path.name}: resampled v landed on different time grid")
        dt = float(dt_target)

    cut_start_seconds = max(0.0, float(cut_start_seconds))
    if cut_start_seconds > 0.0:
        t0 = float(t[0])
        mask = t >= (t0 + cut_start_seconds)
        t = t[mask]
        x = x[mask]
        f_meas = f_meas[mask]
        if v_file is not None:
            v_file = v_file[mask]
        if t.size < 2:
            raise ValueError(f"{path.name}: too few samples remain after cut_start_seconds={cut_start_seconds}.")

    if velocity_source == "file":
        if v_file is None:
            raise ValueError(f"{path.name} has no 'dy' but vpinn.velocity_source='file'.")
        v = v_file
    elif velocity_source == "compute":
        if x.shape[1] != 1:
            raise ValueError(
                "vpinn.velocity_source='compute' currently supports only d=1 displacement. "
                "Provide dy in-file for d>1 or extend compute_velocity_numpy to ND."
            )
        v_1d = compute_velocity_numpy(
            x[:, 0],
            dt,
            use_savgol=smoothing_cfg.use_savgol_smoothing,
            savgol_window=smoothing_cfg.window_length,
            savgol_polyorder=smoothing_cfg.polyorder,
        )
        v = v_1d[:, None]
    else:
        raise ValueError("vpinn.velocity_source must be 'compute' or 'file'.")

    traj = {
        "name": path.name,
        "t": torch.from_numpy(t.astype(np.float32)),
        "x": torch.from_numpy(x.astype(np.float32)),
        "v": torch.from_numpy(np.asarray(v, dtype=np.float32)),
        "f": torch.from_numpy(f_meas.astype(np.float32)),
    }
    return traj, dt


def _resample_uniform_nd(t: np.ndarray, y: np.ndarray, target_dt: float) -> tuple[np.ndarray, np.ndarray]:
    if y.ndim != 2:
        raise ValueError("y must be 2D for _resample_uniform_nd")
    ys: list[np.ndarray] = []
    t_out: Optional[np.ndarray] = None
    for j in range(y.shape[1]):
        yj, tj = resample_uniform_series(t, y[:, j], target_dt)
        ys.append(np.asarray(yj))
        if t_out is None:
            t_out = np.asarray(tj)
        else:
            if t_out.shape != np.asarray(tj).shape or not np.allclose(t_out, tj, rtol=1e-9, atol=1e-12):
                raise ValueError("Resample produced inconsistent time grids across dimensions")
    assert t_out is not None
    return np.stack(ys, axis=1), t_out


def _split_by_trajectory(
    trajectories: list[dict[str, Any]],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not (0.0 <= val_fraction < 1.0):
        raise ValueError("vpinn.val_fraction must be in [0, 1).")
    if not trajectories:
        raise ValueError("No trajectories to split.")
    if val_fraction == 0.0:
        return trajectories, []
    rng = np.random.default_rng(int(seed))
    idx = np.arange(len(trajectories))
    rng.shuffle(idx)
    n_val = int(round(val_fraction * len(trajectories)))
    n_val = min(max(n_val, 1), len(trajectories) - 1)
    val_idx = set(idx[:n_val].tolist())
    train = [traj for i, traj in enumerate(trajectories) if i not in val_idx]
    val = [traj for i, traj in enumerate(trajectories) if i in val_idx]
    return train, val


class WindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        trajectories: list[dict[str, Any]],
        *,
        window_intervals: int,
        stride: int,
    ) -> None:
        if window_intervals < 1:
            raise ValueError("vpinn.window_M must be >= 1")
        if stride < 1:
            raise ValueError("vpinn.stride must be >= 1")
        self.trajectories = trajectories
        self.M = int(window_intervals)
        self.M1 = self.M + 1
        self.stride = int(stride)

        traj_ids: list[np.ndarray] = []
        starts: list[np.ndarray] = []
        for traj_id, traj in enumerate(self.trajectories):
            x = traj["x"]
            length = int(x.shape[0])
            if length < self.M1:
                continue
            start_idx = np.arange(0, length - self.M1 + 1, self.stride, dtype=np.int32)
            if start_idx.size == 0:
                continue
            traj_ids.append(np.full_like(start_idx, traj_id, dtype=np.int32))
            starts.append(start_idx)
        if traj_ids:
            self._traj_ids = np.concatenate(traj_ids, axis=0)
            self._starts = np.concatenate(starts, axis=0)
        else:
            self._traj_ids = np.zeros((0,), dtype=np.int32)
            self._starts = np.zeros((0,), dtype=np.int32)

    def __len__(self) -> int:  # type: ignore[override]
        return int(self._traj_ids.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # type: ignore[override]
        traj = self.trajectories[int(self._traj_ids[idx])]
        start = int(self._starts[idx])
        end = start + self.M1
        x = traj["x"][start:end]
        v = traj["v"][start:end]
        f = traj["f"][start:end]
        return x, v, f


def _prepare_trajectories(config: Config) -> tuple[list[dict[str, Any]], float]:
    data_cfg = config.data
    smoothing_cfg = config.smoothing
    vp = dict(config.vpinn or {})

    velocity_source = str(vp.get("velocity_source", "compute")).strip().lower()
    dt_target = vp.get("dt_target", None)
    if dt_target is None:
        dt_target = _infer_dt_target_from_data_cfg(data_cfg)
    dt_target = None if dt_target is None else float(dt_target)

    if data_cfg.use_generated_train_series:
        series_dir = Path(data_cfg.train_series_dir)
        if not series_dir.is_absolute():
            series_dir = (Path.cwd() / series_dir).resolve()
        if not series_dir.exists():
            raise FileNotFoundError(f"Training series directory '{series_dir}' does not exist.")
        files = sorted(series_dir.glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"No '.npz' files found in '{series_dir}'.")
        sources: list[Path] = list(files)
    else:
        data_path = Path(data_cfg.file)
        if not data_path.is_absolute():
            data_path = (Path.cwd() / data_path).resolve()
        sources = [data_path]

    trajectories: list[dict[str, Any]] = []
    dt_ref: Optional[float] = None
    cut_start_seconds = float(getattr(data_cfg, "cut_start_seconds", 0.0))
    for path in sources:
        traj, dt = _load_trajectory(
            path=path,
            dt_target=dt_target,
            velocity_source=velocity_source,
            smoothing_cfg=smoothing_cfg,
            reduce_time=False,
            reduction_factor=1,
            cut_start_seconds=cut_start_seconds,
        )
        if dt_ref is None:
            dt_ref = dt
        elif not np.isclose(dt, float(dt_ref), rtol=1e-9, atol=1e-12):
            raise ValueError(f"{path} has dt={dt} but expected dt={dt_ref}.")
        trajectories.append(traj)

    if dt_ref is None:
        raise ValueError("No trajectories loaded.")
    return trajectories, float(dt_ref)


def _test_functions(M: int, dt: float, *, include_quadratic: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    M1 = int(M) + 1
    tau = torch.linspace(0.0, 1.0, M1, dtype=torch.float32)
    T = float(M) * float(dt)
    w_list = [torch.ones_like(tau), tau]
    wdot_list = [torch.zeros_like(tau), torch.full_like(tau, 1.0 / float(T))]
    if include_quadratic:
        w_list.append(tau**2)
        wdot_list.append(2.0 * tau / float(T))
    w = torch.stack(w_list, dim=0)  # (L, M1)
    wdot = torch.stack(wdot_list, dim=0)  # (L, M1)
    alpha = torch.ones((M1,), dtype=torch.float32)
    alpha[0] = 0.5
    alpha[-1] = 0.5
    return w, wdot, alpha


def _as_diag_param(value: Union[float, Sequence[float]], d: int, device: torch.device, name: str) -> torch.Tensor:
    if isinstance(value, (list, tuple, np.ndarray)):
        vec = torch.as_tensor(value, dtype=torch.float32, device=device).reshape(-1)
        if int(vec.numel()) != int(d):
            raise ValueError(f"vpinn.{name} must have length {d}, got {int(vec.numel())}.")
        return vec
    return torch.full((int(d),), float(value), dtype=torch.float32, device=device)


def _weak_residual(
    *,
    x: torch.Tensor,
    v: torch.Tensor,
    f_pred: torch.Tensor,
    m: torch.Tensor,
    c: torch.Tensor,
    k: torch.Tensor,
    dt: float,
    w: torch.Tensor,
    wdot: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    m = m.view(1, 1, -1)
    c = c.view(1, 1, -1)
    k = k.view(1, 1, -1)
    mv = m * v
    cv_kx_minus_f = c * v + k * x - f_pred
    vM = v[:, -1, :].unsqueeze(1)
    v0 = v[:, 0, :].unsqueeze(1)
    wM = w[:, -1].view(1, -1, 1)
    w0 = w[:, 0].view(1, -1, 1)
    boundary = (m * vM) * wM - (m * v0) * w0

    ww = w.unsqueeze(0).unsqueeze(-1)  # (1, L, M1, 1)
    wwdot = wdot.unsqueeze(0).unsqueeze(-1)  # (1, L, M1, 1)
    alpha_w = alpha.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # (1, 1, M1, 1)

    term = cv_kx_minus_f.unsqueeze(1) * ww - mv.unsqueeze(1) * wwdot
    trap = float(dt) * torch.sum(alpha_w * term, dim=2)  # (B, L, d)
    return boundary + trap


def _evaluate_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    non_blocking: bool,
    dt: float,
    m: torch.Tensor,
    c: torch.Tensor,
    k: torch.Tensor,
    wf: float,
    ww: float,
    use_force_loss: bool,
    use_weak_loss: bool,
    w: torch.Tensor,
    wdot: torch.Tensor,
    alpha: torch.Tensor,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> dict[str, float]:
    model.eval()
    loss_f_sum = 0.0
    loss_w_sum = 0.0
    count = 0
    with torch.no_grad():
        for x_win, v_win, f_meas in loader:
            x_win = x_win.to(device, non_blocking=non_blocking)
            v_win = v_win.to(device, non_blocking=non_blocking)
            f_meas = f_meas.to(device, non_blocking=non_blocking)

            B, M1, d = x_win.shape
            inp = torch.cat([x_win, v_win], dim=-1)

            with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
                flat = inp.reshape(B * M1, -1)
                f_pred = model(flat).reshape(B, M1, d)
                loss_f = F.mse_loss(f_pred, f_meas)
                if use_weak_loss:
                    R = _weak_residual(
                        x=x_win,
                        v=v_win,
                        f_pred=f_pred,
                        m=m,
                        c=c,
                        k=k,
                        dt=dt,
                        w=w,
                        wdot=wdot,
                        alpha=alpha,
                    )
                    loss_w = torch.mean(R.pow(2))
                else:
                    loss_w = loss_f.new_tensor(0.0)
            loss_f_sum += float(loss_f.detach().cpu())
            loss_w_sum += float(loss_w.detach().cpu())
            count += 1
    denom = max(count, 1)
    mean_lf = loss_f_sum / denom
    mean_lw = loss_w_sum / denom
    wf_eff = float(wf) if use_force_loss else 0.0
    ww_eff = float(ww) if use_weak_loss else 0.0
    return {"loss_data": mean_lf, "loss_physics": mean_lw, "loss_total": wf_eff * mean_lf + ww_eff * mean_lw}


def _log_rollout_validation(
    *,
    writer: Any,
    epoch: int,
    model: nn.Module,
    traj: dict[str, Any],
    dt: float,
    m: torch.Tensor,
    c: torch.Tensor,
    k: torch.Tensor,
    D: float,
    middle_time_plot: Sequence[float],
    device: torch.device,
) -> None:
    x_true_t = traj["x"].to(device)
    v_true_t = traj["v"].to(device)
    f_true_t = traj["f"].to(device)
    t_np = traj["t"].detach().cpu().numpy()
    if x_true_t.ndim != 2:
        return
    d = int(x_true_t.shape[-1])
    if d < 1:
        return
    if d > 1:
        print("vpinn rollout validation: d>1 detected; logging only the first DOF.")

    steps = int(x_true_t.shape[0] - 1)
    if steps < 1:
        return

    x_seq, v_seq, f_seq = rollout_rk4(
        model=model,
        x0=x_true_t[0:1, :],
        v0=v_true_t[0:1, :],
        steps=steps,
        dt=dt,
        m=m,
        c=c,
        k=k,
    )
    x_pred = x_seq[0, :, 0].detach().cpu().numpy()
    v_pred = v_seq[0, :, 0].detach().cpu().numpy()
    f_pred = f_seq[0, :, 0].detach().cpu().numpy()
    x_true = x_true_t[:, 0].detach().cpu().numpy()
    f_true = f_true_t[:, 0].detach().cpu().numpy()

    disp_std = float(np.std(x_true))
    if disp_std <= 0.0:
        disp_std = 1.0
    rel_rmse_y = float(np.sqrt(np.mean((x_pred - x_true) ** 2))) / disp_std
    writer.add_scalar("val/rollout_nrmse_y", rel_rmse_y, epoch)

    force_std = float(np.std(f_true))
    if force_std <= 0.0:
        force_std = 1.0
    rel_rmse_force = float(np.sqrt(np.mean((f_pred - f_true) ** 2))) / force_std
    writer.add_scalar("val/rollout_nrmse_force_total", rel_rmse_force, epoch)

    with torch.no_grad():
        f_on_data = _vpinn_force(model, x_true_t, v_true_t)[:, 0].detach().cpu().numpy()
    rel_rmse_force_on_data = float(np.sqrt(np.mean((f_on_data - f_true) ** 2))) / force_std
    writer.add_scalar("val/force_mapping_nrmse_on_data", rel_rmse_force_on_data, epoch)

    y_true_norm = x_true / float(D)
    y_pred_norm = x_pred / float(D)
    freq = float(torch.sqrt(k[0] / m[0]).detach().cpu())
    denom = float(freq * float(D)) if freq > 0 else 1.0
    p_pred_norm = v_pred / denom

    zoom_mask = create_zoom_mask(t_np)
    middle_mask = create_window_mask(t_np, middle_time_plot)
    middle_window = (float(middle_time_plot[0]), float(middle_time_plot[1]))
    log_displacement_plots(
        writer,
        epoch,
        t_np,
        y_true_norm,
        y_pred_norm,
        p_pred_norm,
        zoom_mask,
        middle_mask,
        middle_window,
    )
    log_force_plots(
        writer,
        epoch,
        t_np,
        f_pred,
        np.zeros_like(f_pred),
        f_pred,
        f_true,
        zoom_mask,
        middle_mask,
        middle_window,
        include_physical_drag=False,
    )


def train(config: Config, config_name: str) -> None:
    vp = dict(config.vpinn or {})
    runtime_cfg = config.runtime
    precision_cfg = config.precision
    compile_cfg = config.compile
    training_cfg = config.training
    optim_cfg = config.optim
    monitoring_cfg = config.monitoring

    window_M = int(vp.get("window_M", 50))
    stride = int(vp.get("stride", 1))
    val_fraction = float(vp.get("val_fraction", 0.1))
    split_seed = int(vp.get("split_seed", 0))
    wf = float(vp.get("wf", 1.0))
    ww = float(vp.get("ww", 1.0))
    use_force_loss = bool(vp.get("use_force_loss", True))
    use_weak_loss = bool(vp.get("use_weak_loss", True))
    include_quadratic = bool(vp.get("include_quadratic_test", False))
    if not (use_force_loss or use_weak_loss):
        raise ValueError("vpinn must enable at least one of: use_force_loss, use_weak_loss.")

    use_gradnorm = bool(vp.get("use_gradnorm", False))
    gradnorm_alpha = float(vp.get("gradnorm_alpha", 0.9))
    gradnorm_eps = float(vp.get("gradnorm_eps", 1e-8))
    gradnorm_min_weight = float(vp.get("gradnorm_min_weight", 0.1))
    gradnorm_max_weight = float(vp.get("gradnorm_max_weight", 10.0))
    gradnorm_update_every_steps = int(vp.get("gradnorm_update_every_steps", 1))
    gradnorm_update_every_steps = max(1, gradnorm_update_every_steps)

    if bool(vp.get("use_rollout_loss", False)):
        raise ValueError(
            "vpinn.use_rollout_loss is not supported. "
            "Rollout-based checks are intended for validation only; "
            "use monitoring.rollout_every_epochs to log rollout validation."
        )

    device = select_device(os.getenv("TRAIN_DEVICE", str(runtime_cfg.device)))
    print(f"Using device: {device}")
    configure_tf32(device, bool(precision_cfg.use_tf32))
    set_num_threads_from_slurm(default=1)
    non_blocking = device.type == "cuda"

    trajectories, dt = _prepare_trajectories(config)
    validation_only_data_file = bool(vp.get("validation_only_data_file", False))
    if validation_only_data_file:
        train_trajs, val_trajs = trajectories, []
    else:
        train_trajs, val_trajs = _split_by_trajectory(trajectories, val_fraction=val_fraction, seed=split_seed)
    if not train_trajs:
        raise ValueError("Empty training split. Reduce vpinn.val_fraction or provide more trajectories.")

    use_data_file_for_validation = bool(vp.get("use_data_file_for_validation", False))
    if use_data_file_for_validation:
        data_path = Path(config.data.file)
        if not data_path.is_absolute():
            data_path = (Path.cwd() / data_path).resolve()
        val_reduce_time = bool(getattr(config.data, "reduce_time", False))
        val_reduction_factor = int(getattr(config.data, "reduction_factor", 1))
        cut_start_seconds = float(getattr(config.data, "cut_start_seconds", 0.0))
        val_traj, val_dt = _load_trajectory(
            path=data_path,
            dt_target=dt,
            velocity_source=str(vp.get("velocity_source", "compute")).strip().lower(),
            smoothing_cfg=config.smoothing,
            reduce_time=val_reduce_time,
            reduction_factor=val_reduction_factor,
            cut_start_seconds=cut_start_seconds,
        )
        if val_dt != dt:
            raise ValueError(f"Validation data dt={val_dt} does not match training dt={dt}.")
        if val_trajs is None:
            val_trajs = []
        # Put `data.npz` first so rollout validation uses it by default.
        val_trajs = [val_traj] + list(val_trajs)

    train_dataset = WindowDataset(train_trajs, window_intervals=window_M, stride=stride)
    val_dataset = WindowDataset(val_trajs, window_intervals=window_M, stride=stride) if val_trajs else None
    if len(train_dataset) == 0:
        raise ValueError("No windows available for training. Reduce vpinn.window_M or check data lengths.")

    batch_size = int(training_cfg.batch_size)
    epochs = int(training_cfg.epochs)
    max_grad_norm = float(training_cfg.max_grad_norm)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(runtime_cfg.num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(runtime_cfg.num_workers),
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )
        if val_dataset is not None and len(val_dataset) > 0
        else None
    )

    d = int(train_trajs[0]["x"].shape[-1])
    m = _as_diag_param(vp.get("m", _m_eff_from_model_cfg(config.model)), d, device, "m")
    c = _as_diag_param(vp.get("c", getattr(config.model, "damping_c", 1e-4)), d, device, "c")
    k = _as_diag_param(vp.get("k", getattr(config.model, "k", 1218.0)), d, device, "k")

    input_dim = 2 * d
    output_dim = d
    model = _build_force_model(config, input_dim=input_dim, output_dim=output_dim)
    model = model.to(device)
    model = maybe_compile_model(model, bool(compile_cfg.use_compile), str(compile_cfg.compile_mode))

    gradnorm_balancer: Optional[GradNormBalancer] = None
    gradnorm_last_force = None
    gradnorm_last_weak = None
    if use_gradnorm and use_force_loss and use_weak_loss:
        gradnorm_balancer = GradNormBalancer(
            model,
            ["force", "weak"],
            alpha=gradnorm_alpha,
            eps=gradnorm_eps,
            min_weight=gradnorm_min_weight,
            max_weight=gradnorm_max_weight,
        )
    elif use_gradnorm and not (use_force_loss and use_weak_loss):
        print("vpinn.use_gradnorm is True but only one loss is enabled; skipping GradNorm.")

    opt, lr_scheduler = setup_optimizer_and_scheduler(
        model,
        optim_cfg=optim_cfg,
        scheduler_cfg=optim_cfg.scheduler,
        epochs=epochs,
    )

    amp_enabled, amp_dtype, scaler = setup_amp(
        device, use_amp=bool(precision_cfg.use_amp), amp_dtype=str(precision_cfg.amp_dtype)
    )

    w, wdot, alpha = _test_functions(window_M, dt, include_quadratic=include_quadratic)
    w = w.to(device)
    wdot = wdot.to(device)
    alpha = alpha.to(device)

    writer, run_name = setup_writer(config.logging.run_dir_root, config_name)

    use_lr_scheduler = bool(optim_cfg.use_lr_scheduler)
    base_lr = float(optim_cfg.lr)

    log_every = int(getattr(monitoring_cfg, "log_every_epochs", 1))
    print_every = int(getattr(monitoring_cfg, "print_every_epochs", 1))
    validate_every = int(getattr(monitoring_cfg, "validate_every_epochs", 1))
    rollout_every = int(getattr(monitoring_cfg, "rollout_every_epochs", 0))
    rollout_max_trajs = int(getattr(monitoring_cfg, "rollout_max_trajectories", 1))
    middle_time_plot = getattr(config.data, "middle_time_plot", [0.0, 1.0])
    if len(middle_time_plot) != 2:
        middle_time_plot = [0.0, 1.0]
    D_val = float(getattr(config.model, "D", 1.0))

    for epoch in range(epochs):
        model.train()
        if use_lr_scheduler:
            for group in opt.param_groups:
                group["lr"] = lr_scheduler.get_lr(epoch)

        loss_f_sum = torch.zeros((), device=device)
        loss_w_sum = torch.zeros((), device=device)
        loss_sum = torch.zeros((), device=device)
        grad_norm_sum = torch.zeros((), device=device)
        gradnorm_force_w_sum = torch.zeros((), device=device)
        gradnorm_weak_w_sum = torch.zeros((), device=device)
        gradnorm_count = 0
        batches = 0

        for step, (x_win, v_win, f_meas) in enumerate(train_loader):
            x_win = x_win.to(device, non_blocking=non_blocking)
            v_win = v_win.to(device, non_blocking=non_blocking)
            f_meas = f_meas.to(device, non_blocking=non_blocking)

            B, M1, d = x_win.shape
            inp = torch.cat([x_win, v_win], dim=-1)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
                flat = inp.reshape(B * M1, -1)
                f_pred = model(flat).reshape(B, M1, d)
                loss_f = F.mse_loss(f_pred, f_meas)
                if use_weak_loss:
                    R = _weak_residual(
                        x=x_win,
                        v=v_win,
                        f_pred=f_pred,
                        m=m,
                        c=c,
                        k=k,
                        dt=dt,
                        w=w,
                        wdot=wdot,
                        alpha=alpha,
                    )
                    loss_w = torch.mean(R.pow(2))
                else:
                    loss_w = loss_f.new_tensor(0.0)

                wf_eff = float(wf) if use_force_loss else 0.0
                ww_eff = float(ww) if use_weak_loss else 0.0
                if gradnorm_balancer is not None:
                    do_update = (step % gradnorm_update_every_steps) == 0 or gradnorm_last_force is None
                    if do_update:
                        weights = gradnorm_balancer.update({"force": loss_f.float(), "weak": loss_w.float()})
                        gradnorm_last_force = weights["force"]
                        gradnorm_last_weak = weights["weak"]
                    w_force = gradnorm_last_force
                    w_weak = gradnorm_last_weak
                    assert w_force is not None and w_weak is not None
                    gradnorm_force_w_sum = gradnorm_force_w_sum + w_force.detach()
                    gradnorm_weak_w_sum = gradnorm_weak_w_sum + w_weak.detach()
                    gradnorm_count += 1
                    loss = wf_eff * w_force * loss_f + ww_eff * w_weak * loss_w
                else:
                    loss = wf_eff * loss_f + ww_eff * loss_w

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
            else:
                loss.backward()
            grad_norm = nn_utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            if scaler.is_enabled():
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()

            batches += 1
            loss_sum = loss_sum + loss.detach()
            loss_f_sum = loss_f_sum + loss_f.detach()
            loss_w_sum = loss_w_sum + loss_w.detach()
            if isinstance(grad_norm, torch.Tensor):
                grad_norm_sum = grad_norm_sum + grad_norm.detach()
            else:
                grad_norm_sum = grad_norm_sum + torch.tensor(float(grad_norm), device=device)

        denom = float(max(batches, 1))
        metrics = {
            "loss_total": float((loss_sum / denom).detach().cpu()),
            "loss_data": float((loss_f_sum / denom).detach().cpu()),
            "loss_physics": float((loss_w_sum / denom).detach().cpu()),
            "grad_norm": float((grad_norm_sum / denom).detach().cpu()),
            "lr": float(opt.param_groups[0]["lr"]) if opt.param_groups else base_lr,
        }
        if gradnorm_count > 0:
            metrics["gradnorm_weight_data"] = float((gradnorm_force_w_sum / float(gradnorm_count)).detach().cpu())
            metrics["gradnorm_weight_physics"] = float((gradnorm_weak_w_sum / float(gradnorm_count)).detach().cpu())

        if (epoch % max(1, log_every)) == 0 or epoch == (epochs - 1):
            for k_name, v_value in metrics.items():
                writer.add_scalar(f"train/{k_name}", v_value, epoch)

        if (epoch % max(1, print_every)) == 0 or epoch == (epochs - 1):
            print(
                f"Epoch {epoch}: loss={metrics['loss_total']:.4e}, "
                f"Ldata={metrics['loss_data']:.4e}, Lphys={metrics['loss_physics']:.4e}, lr={metrics['lr']:.3e}"
            )

        if val_loader is not None and validate_every > 0 and ((epoch % validate_every) == 0 or epoch == (epochs - 1)):
            val_metrics = _evaluate_epoch(
                model=model,
                loader=val_loader,
                device=device,
                non_blocking=non_blocking,
                dt=dt,
                m=m,
                c=c,
                k=k,
                wf=wf,
                ww=ww,
                use_force_loss=use_force_loss,
                use_weak_loss=use_weak_loss,
                w=w,
                wdot=wdot,
                alpha=alpha,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )
            for k_name, v_value in val_metrics.items():
                writer.add_scalar(f"val/{k_name}", v_value, epoch)

        if rollout_every > 0 and ((epoch % rollout_every) == 0 or epoch == (epochs - 1)):
            candidates = val_trajs if val_trajs else train_trajs
            for traj in candidates[: max(1, rollout_max_trajs)]:
                _log_rollout_validation(
                    writer=writer,
                    epoch=epoch,
                    model=model,
                    traj=traj,
                    dt=dt,
                    m=m,
                    c=c,
                    k=k,
                    D=D_val,
                    middle_time_plot=middle_time_plot,
                    device=device,
                )

    writer.add_text("vpinn/config_vpinn", json.dumps(vp, indent=2, sort_keys=True), 0)

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{run_name}.pt"
    state_source: nn.Module = model
    if hasattr(model, "_orig_mod"):
        state_source = getattr(model, "_orig_mod")
    torch.save(
        {
            "model_state": state_source.state_dict(),
            "config": asdict(config),
            "run_name": run_name,
            "dt": dt,
            "method": "vpinn",
        },
        model_path,
    )
    print(f"Saved final model to {model_path}")

    writer.flush()
    writer.close()
