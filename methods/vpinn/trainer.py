from __future__ import annotations

import json
import math
import os
import time
import subprocess
import sys
import warnings
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
    DOMINANT_FREQ_REL_ERROR_KEY,
    DISP_SPECTRAL_SHAPE_ERROR_KEY,
    DISP_ROLLOUT_NRMSE_KEY,
    FORCE_SPECTRAL_SHAPE_ERROR_KEY,
    FORCE_MAPPING_NRMSE_KEY,
    FORCE_ROLLOUT_NRMSE_KEY,
    GradNormBalancer,
    MEAN_DISP_AMP_REL_ERROR_KEY,
    Residual,
    compute_velocity_numpy,
    create_window_mask,
    create_zoom_mask,
    dominant_frequency,
    format_loss_vs_ur_text,
    log_final_rollout_errors_vs_ur,
    log_loss_vs_ur,
    log_displacement_plots,
    log_force_plots,
    mean_displacement_amplitude,
    relative_error,
    resolve_middle_time_plot,
    resample_uniform_series,
    spectral_relative_error,
)
from architectures import ODEPirateNet, TemporalConvForceNet

_MISSING_METADATA_WARNED: set[str] = set()


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


class TemporalBackboneWithHead(nn.Module):
    """
    Temporal encoder + pointwise head.

    The backbone consumes (B, T, C_in) and outputs (B, T, C_mid).
    The head is then applied independently at each timestep to map C_mid -> C_out.
    """

    def __init__(self, backbone: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

        # Introspection hooks used by VPINN sequence logic.
        self.is_tcn_force_model = True
        self.history_len = int(getattr(backbone, "history_len", 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if x.ndim == 2:
            x = x.unsqueeze(1)
            squeeze_time = True
        elif x.ndim == 3:
            squeeze_time = False
        else:
            raise ValueError("TemporalBackboneWithHead expects input shape (N,C) or (B,T,C).")

        h = self.backbone(x)
        if h.ndim != 3:
            raise ValueError("Temporal backbone must return shape (B,T,C_mid).")

        B, T, C = h.shape
        y = self.head(h.reshape(B * T, C)).reshape(B, T, -1)
        if squeeze_time:
            y = y.squeeze(1)
        return y


class ScaledForceWrapper(nn.Module):
    """
    Wrap a force network so VPINN sees well-conditioned inputs and produces forces
    in physical units.

    The VPINN force model takes s = [x, v, U_r] and returns f. Here we apply:
      x_tilde = x / x_scale
      v_tilde = v / v_scale
      U_tilde = U_r / ur_scale
      f       = f_scale * f_net([x_tilde, v_tilde, U_tilde])
    """

    def __init__(
        self,
        base: nn.Module,
        *,
        d: int,
        x_scale: float,
        v_scale: torch.Tensor,
        ur_scale: float,
        f_scale: torch.Tensor,
    ) -> None:
        super().__init__()
        self.base = base
        self.d = int(d)
        if self.d < 1:
            raise ValueError("ScaledForceWrapper requires d >= 1")

        # Place scaling buffers on the same device as the wrapped model so forward()
        # never mixes CPU/GPU tensors.
        try:
            base_device = next(base.parameters()).device
        except StopIteration:
            base_device = v_scale.device

        x_scale = float(x_scale)
        if not np.isfinite(x_scale) or x_scale == 0.0:
            raise ValueError(f"x_scale must be finite and non-zero, got {x_scale}")
        ur_scale = float(ur_scale)
        if not np.isfinite(ur_scale) or ur_scale == 0.0:
            raise ValueError(f"ur_scale must be finite and non-zero, got {ur_scale}")

        self.register_buffer("x_scale", torch.tensor(x_scale, dtype=torch.float32, device=base_device))
        self.register_buffer("ur_scale", torch.tensor(ur_scale, dtype=torch.float32, device=base_device))
        self.register_buffer("v_scale", torch.as_tensor(v_scale, dtype=torch.float32).to(base_device).reshape(-1))
        self.register_buffer("f_scale", torch.as_tensor(f_scale, dtype=torch.float32).to(base_device).reshape(-1))

    def forward(self, s: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        d = self.d
        if int(s.shape[-1]) != int(2 * d + 1):
            raise ValueError(f"Expected input dim {2*d+1} ([x,v,U_r]) but got {int(s.shape[-1])}.")
        x = s[..., :d]
        v = s[..., d : 2 * d]
        ur = s[..., 2 * d :]
        if self.v_scale.numel() == 1:
            v_scale = self.v_scale
        else:
            v_scale = self.v_scale.view(*([1] * (v.ndim - 1)), -1)
        if self.f_scale.numel() == 1:
            f_scale = self.f_scale
        else:
            f_scale = self.f_scale.view(*([1] * (v.ndim - 1)), -1)

        x_tilde = x / self.x_scale
        v_tilde = v / v_scale
        ur_tilde = ur / self.ur_scale
        s_tilde = torch.cat([x_tilde, v_tilde, ur_tilde], dim=-1)
        f_tilde = self.base(s_tilde)
        return f_tilde * f_scale


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


def _build_arch_pointwise_head(
    *,
    config: Config,
    arch: Any,
    net_type: str,
    input_dim: int,
    output_dim: int,
) -> nn.Module:
    key = str(net_type).strip().lower()

    if key == "pirate":
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

    if key == "residual":
        cfg = dict(getattr(arch, "residual_kwargs", {}) or {})
        hidden = int(cfg.get("hidden", 128))
        layers = int(cfg.get("layers", 2))
        activation = str(cfg.get("activation", "gelu"))
        layers_list: list[nn.Module] = [nn.Linear(int(input_dim), hidden)]
        for _ in range(max(1, layers)):
            layers_list.append(Residual(hidden, activation=activation))
        layers_list.append(nn.Linear(hidden, int(output_dim)))
        return nn.Sequential(*layers_list)

    if key == "mlp":
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

    raise ValueError("architecture.force_net_type must be one of: residual, mlp, pirate, tcn")


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
    if net_type not in {"residual", "mlp", "pirate", "tcn"}:
        raise ValueError("architecture.force_net_type must be one of: residual, mlp, pirate, tcn")

    tcn_cfg = dict(getattr(arch, "tcn_kwargs", {}) or {})
    use_tcn_backbone = bool(
        tcn_cfg.get("enabled", tcn_cfg.get("use_as_backbone", False))
        or vp.get("use_tcn_backbone", False)
    )

    if net_type == "tcn":
        # Number of *previous* timesteps of context used for the first target in a window.
        default_history = int(vp.get("window_M", 50))
        return TemporalConvForceNet(
            input_size=int(input_dim),
            output_size=int(output_dim),
            hidden_channels=int(tcn_cfg.get("hidden", 128)),
            levels=int(tcn_cfg.get("levels", 4)),
            dilation_start=int(tcn_cfg.get("dilation_start", 1)),
            kernel_size=int(tcn_cfg.get("kernel_size", 3)),
            dropout=float(tcn_cfg.get("dropout", 0.0)),
            activation=str(tcn_cfg.get("activation", "gelu")),
            history_len=int(tcn_cfg.get("history_len", default_history)),
        )

    if use_tcn_backbone:
        default_history = int(vp.get("window_M", 50))
        head_input_dim = int(tcn_cfg.get("head_input_dim", tcn_cfg.get("hidden", 128)))
        if head_input_dim < 1:
            raise ValueError("architecture.tcn_kwargs.head_input_dim must be >= 1.")
        backbone = TemporalConvForceNet(
            input_size=int(input_dim),
            output_size=int(head_input_dim),
            hidden_channels=int(tcn_cfg.get("hidden", 128)),
            levels=int(tcn_cfg.get("levels", 4)),
            dilation_start=int(tcn_cfg.get("dilation_start", 1)),
            kernel_size=int(tcn_cfg.get("kernel_size", 3)),
            dropout=float(tcn_cfg.get("dropout", 0.0)),
            activation=str(tcn_cfg.get("activation", "gelu")),
            history_len=int(tcn_cfg.get("history_len", default_history)),
        )
        head = _build_arch_pointwise_head(
            config=config,
            arch=arch,
            net_type=net_type,
            input_dim=int(head_input_dim),
            output_dim=int(output_dim),
        )
        return TemporalBackboneWithHead(backbone=backbone, head=head)

    return _build_arch_pointwise_head(
        config=config,
        arch=arch,
        net_type=net_type,
        input_dim=int(input_dim),
        output_dim=int(output_dim),
    )


def _unwrap_force_model(model: nn.Module) -> nn.Module:
    current: nn.Module = model
    seen: set[int] = set()
    while True:
        obj_id = id(current)
        if obj_id in seen:
            break
        seen.add(obj_id)
        if isinstance(current, ScaledForceWrapper):
            current = current.base
            continue
        if hasattr(current, "_orig_mod"):
            maybe = getattr(current, "_orig_mod")
            if isinstance(maybe, nn.Module):
                current = maybe
                continue
        if hasattr(current, "module"):
            maybe = getattr(current, "module")
            if isinstance(maybe, nn.Module):
                current = maybe
                continue
        break
    return current


def _is_tcn_force_model(model: nn.Module) -> bool:
    base = _unwrap_force_model(model)
    return bool(getattr(base, "is_tcn_force_model", False))


def _tcn_history_len(model: nn.Module) -> int:
    base = _unwrap_force_model(model)
    return max(0, int(getattr(base, "history_len", 0)))


def _configured_validation_history_len(config: Config) -> int:
    """
    Resolve TCN history requirement from config (without constructing the model).
    """
    vp = dict(config.vpinn or {})
    if not bool(vp.get("use_architecture_config", False)):
        return 0
    arch = getattr(config, "architecture", None)
    if arch is None:
        return 0
    net_type = str(getattr(arch, "force_net_type", "residual")).strip().lower()
    cfg = dict(getattr(arch, "tcn_kwargs", {}) or {})
    use_tcn_backbone = bool(
        cfg.get("enabled", cfg.get("use_as_backbone", False))
        or vp.get("use_tcn_backbone", False)
    )
    if net_type != "tcn" and not use_tcn_backbone:
        return 0
    default_history = int(vp.get("window_M", 50))
    return max(0, int(cfg.get("history_len", default_history)))


def _vpinn_force_sequence(
    model: nn.Module,
    x: torch.Tensor,
    v: torch.Tensor,
    ur: torch.Tensor,
) -> torch.Tensor:
    """
    Unified VPINN force call.

    Inputs:
      x, v, ur with shape (B, T, d)
    Returns:
      f with shape (B, T, d)
    """
    if x.ndim != 3 or v.ndim != 3 or ur.ndim != 3:
        raise ValueError("x, v, ur must have shape (B, T, d)")
    inp = torch.cat([x, v, ur], dim=-1)
    B, T, d = x.shape
    if _is_tcn_force_model(model):
        out = model(inp)
        if out.ndim != 3:
            raise ValueError("TCN force model must return shape (B, T, d).")
        return out
    flat = inp.reshape(B * T, -1)
    return model(flat).reshape(B, T, d)


def _vpinn_force(model: nn.Module, x: torch.Tensor, v: torch.Tensor, ur: torch.Tensor) -> torch.Tensor:
    """
    Pointwise force with shape-preserving fallback for sequence models.

    Inputs:
      x, v, ur with shape (N, d)
    Returns:
      f with shape (N, d)
    """
    if x.ndim != 2 or v.ndim != 2 or ur.ndim != 2:
        raise ValueError("x, v, ur must have shape (N, d)")
    if _is_tcn_force_model(model):
        return _vpinn_force_sequence(
            model,
            x.unsqueeze(0),
            v.unsqueeze(0),
            ur.unsqueeze(0),
        ).squeeze(0)
    return model(torch.cat([x, v, ur], dim=-1))


def _vpinn_force_on_trajectory(
    model: nn.Module,
    x: torch.Tensor,
    v: torch.Tensor,
    ur: torch.Tensor,
) -> torch.Tensor:
    """
    Force evaluation over a single trajectory with optional left context padding.

    Inputs:
      x, v, ur with shape (T, d)
    Returns:
      f with shape (T, d)
    """
    if not _is_tcn_force_model(model):
        return _vpinn_force(model, x, v, ur)

    context = _tcn_history_len(model)
    if context <= 0:
        return _vpinn_force(model, x, v, ur)

    x_pad = torch.cat([x[0:1, :].expand(context, -1), x], dim=0)
    v_pad = torch.cat([v[0:1, :].expand(context, -1), v], dim=0)
    ur_pad = torch.cat([ur[0:1, :].expand(context, -1), ur], dim=0)
    f_pad = _vpinn_force_sequence(
        model,
        x_pad.unsqueeze(0),
        v_pad.unsqueeze(0),
        ur_pad.unsqueeze(0),
    ).squeeze(0)
    return f_pad[context:, :]


def _force_mapping_nrmse_over_trajs(
    *,
    model: nn.Module,
    val_trajs: Sequence[dict[str, Any]],
    device: torch.device,
) -> Optional[dict[str, float]]:
    if not val_trajs:
        return None
    model.eval()
    values_force: list[torch.Tensor] = []
    with torch.no_grad():
        for traj in val_trajs:
            x_true = traj["x"].to(device)
            v_true = traj["v"].to(device)
            f_true = traj["f"].to(device)
            ur_true = traj["ur"].to(device)
            f0_true = traj.get("f0", None)
            if f0_true is not None:
                f0_true = f0_true.to(device)
            val_start_idx = max(0, int(traj.get("val_start_idx", 0)))
            if x_true.ndim != 2:
                continue
            if val_start_idx >= int(x_true.shape[0]):
                continue
            f_pred = _vpinn_force_on_trajectory(model, x_true, v_true, ur_true)
            f_pred = f_pred[val_start_idx:, :]
            f_true = f_true[val_start_idx:, :]
            if f0_true is not None:
                f0_true = f0_true[val_start_idx:, :]
            if f_pred.ndim == 1:
                f_pred = f_pred.unsqueeze(-1)
            if f_true.ndim == 1:
                f_true = f_true.unsqueeze(-1)
            f_pred0 = f_pred[..., 0]
            f_true0 = f_true[..., 0]

            if f0_true is not None:
                f0_vec = f0_true[..., 0]
                f_pred_force = f_pred0 * f0_vec
                f_true_force = f_true0 * f0_vec
                force_std = torch.std(f_true_force)
                if force_std <= 0.0:
                    force_std = f_true_force.new_tensor(1.0)
                nrmse_force = torch.sqrt(torch.mean((f_pred_force - f_true_force) ** 2)) / force_std
                values_force.append(nrmse_force.detach())
            else:
                force_std = torch.std(f_true0)
                if force_std <= 0.0:
                    force_std = f_true0.new_tensor(1.0)
                nrmse_force = torch.sqrt(torch.mean((f_pred0 - f_true0) ** 2)) / force_std
                values_force.append(nrmse_force.detach())

    if not values_force:
        return None
    return {
        FORCE_MAPPING_NRMSE_KEY: float(torch.mean(torch.stack(values_force)).detach().cpu())
    }


def rollout_rk4(
    *,
    model: nn.Module,
    x0: torch.Tensor,
    v0: torch.Tensor,
    ur0: torch.Tensor,
    steps: int,
    dt: float,
    substeps: int = 1,
    m: torch.Tensor,
    c: torch.Tensor,
    k: torch.Tensor,
    f0: Optional[torch.Tensor] = None,
    x_hist_init: Optional[torch.Tensor] = None,
    v_hist_init: Optional[torch.Tensor] = None,
    ur_hist_init: Optional[torch.Tensor] = None,
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
    substeps = int(substeps)
    if substeps < 1:
        raise ValueError("substeps must be >= 1")
    B, d = x0.shape
    m = m.view(1, d)
    c = c.view(1, d)
    k = k.view(1, d)

    use_tcn = _is_tcn_force_model(model)
    x = x0
    v = v0
    xs = [x]
    vs = [v]

    if use_tcn:
        context = _tcn_history_len(model)
        hist_len = context + 1
        provided_history = any(t is not None for t in (x_hist_init, v_hist_init, ur_hist_init))
        if provided_history and not all(t is not None for t in (x_hist_init, v_hist_init, ur_hist_init)):
            raise ValueError("x_hist_init, v_hist_init, and ur_hist_init must be provided together.")
        if provided_history:
            x_hist = x_hist_init
            v_hist = v_hist_init
            ur_hist = ur_hist_init
            assert x_hist is not None and v_hist is not None and ur_hist is not None
            if x_hist.ndim != 3 or v_hist.ndim != 3 or ur_hist.ndim != 3:
                raise ValueError("History tensors must have shape (B, T_hist, d).")
            expected_shape = (B, hist_len, d)
            if tuple(x_hist.shape) != expected_shape:
                raise ValueError(f"x_hist_init must have shape {expected_shape}, got {tuple(x_hist.shape)}.")
            if tuple(v_hist.shape) != expected_shape:
                raise ValueError(f"v_hist_init must have shape {expected_shape}, got {tuple(v_hist.shape)}.")
            if tuple(ur_hist.shape) != expected_shape:
                raise ValueError(f"ur_hist_init must have shape {expected_shape}, got {tuple(ur_hist.shape)}.")
            x_hist = x_hist.clone()
            v_hist = v_hist.clone()
            ur_hist = ur_hist.clone()
        else:
            # Fallback cold start from repeated initial state.
            x_hist = x.unsqueeze(1).repeat(1, hist_len, 1)
            v_hist = v.unsqueeze(1).repeat(1, hist_len, 1)
            ur_hist = ur0.unsqueeze(1).repeat(1, hist_len, 1)

        def force_from_history(x_curr: torch.Tensor, v_curr: torch.Tensor) -> torch.Tensor:
            x_in = x_hist.clone()
            v_in = v_hist.clone()
            x_in[:, -1, :] = x_curr
            v_in[:, -1, :] = v_curr
            f_seq = _vpinn_force_sequence(model, x_in, v_in, ur_hist)
            return f_seq[:, -1, :]

        fs = [force_from_history(x, v)]
    else:
        fs = [_vpinn_force(model, x, v, ur0)]

    dt_t = x0.new_tensor(float(dt))
    dt_sub = dt_t / float(substeps)
    half = x0.new_tensor(0.5)
    sixth = x0.new_tensor(1.0 / 6.0)

    def accel(xi: torch.Tensor, vi: torch.Tensor) -> torch.Tensor:
        if use_tcn:
            ci = force_from_history(xi, vi)
        else:
            ci = _vpinn_force(model, xi, vi, ur0)
        fi = ci if f0 is None else ci * f0
        return (fi - c * vi - k * xi) / m

    for _ in range(int(steps)):
        for _sub in range(substeps):
            k1_x = v
            k1_v = accel(x, v)

            x2 = x + half * dt_sub * k1_x
            v2 = v + half * dt_sub * k1_v
            k2_x = v2
            k2_v = accel(x2, v2)

            x3 = x + half * dt_sub * k2_x
            v3 = v + half * dt_sub * k2_v
            k3_x = v3
            k3_v = accel(x3, v3)

            x4 = x + dt_sub * k3_x
            v4 = v + dt_sub * k3_v
            k4_x = v4
            k4_v = accel(x4, v4)

            x = x + (dt_sub * sixth) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
            v = v + (dt_sub * sixth) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)

            if use_tcn:
                x_hist = torch.cat([x_hist[:, 1:, :], x.unsqueeze(1)], dim=1)
                v_hist = torch.cat([v_hist[:, 1:, :], v.unsqueeze(1)], dim=1)

        xs.append(x)
        vs.append(v)
        if use_tcn:
            fs.append(force_from_history(x, v))
        else:
            fs.append(_vpinn_force(model, x, v, ur0))

    return torch.stack(xs, dim=1), torch.stack(vs, dim=1), torch.stack(fs, dim=1)


def _prune_async_processes(processes: list[subprocess.Popen]) -> list[subprocess.Popen]:
    return [proc for proc in processes if proc.poll() is None]


def _launch_async_validation(
    *,
    processes: list[subprocess.Popen],
    max_concurrent: int,
    checkpoint_path: Path,
    epoch: int,
    writer: SummaryWriter,
    async_device: str,
    async_num_workers: int,
    async_num_threads: int,
    rollout_every_epochs: int,
    cycle_validation_rollout: bool,
    do_losses: bool,
    do_rollout: bool,
) -> list[subprocess.Popen]:
    processes = _prune_async_processes(processes)
    if max_concurrent > 0 and len(processes) >= max_concurrent:
        return processes
    script_path = Path(__file__).resolve().parents[2] / "async_validate.py"
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(async_num_threads)
    env["MKL_NUM_THREADS"] = str(async_num_threads)
    env["OPENBLAS_NUM_THREADS"] = str(async_num_threads)
    env["NUMEXPR_NUM_THREADS"] = str(async_num_threads)
    args = [
        sys.executable,
        str(script_path),
        "--checkpoint",
        str(checkpoint_path),
        "--log-dir",
        str(writer.log_dir),
        "--epoch",
        str(epoch + 1),
        "--device",
        str(async_device),
        "--num-threads",
        str(async_num_threads),
        "--num-workers",
        str(async_num_workers),
        "--rollout-every",
        str(int(rollout_every_epochs)),
        "--cycle-rollout",
        "1" if cycle_validation_rollout else "0",
        "--do-losses",
        "1" if do_losses else "0",
        "--do-rollout",
        "1" if do_rollout else "0",
    ]
    processes.append(subprocess.Popen(args, env=env))
    return processes


def _m_eff_from_model_cfg(model_cfg: Any) -> float:
    rho = float(getattr(model_cfg, "rho", 1000.0))
    D = float(getattr(model_cfg, "D", 0.1))
    Ca = float(getattr(model_cfg, "Ca", 1.0))
    structural_mass = float(getattr(model_cfg, "structural_mass", 16.79))
    m_a = 0.25 * math.pi * D * D * rho * Ca
    return structural_mass + m_a


def _read_timeseries_npz(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], float]:
    with np.load(path) as data:
        ur_raw = data["U_r"] if "U_r" in data else None
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
    if ur_raw is None:
        raise ValueError(f"{path} is missing reduced velocity 'U_r'.")
    ur_arr = np.asarray(ur_raw, dtype=float)
    if ur_arr.ndim == 0:
        ur_val = float(ur_arr)
    else:
        ur_flat = ur_arr.reshape(-1)
        if ur_flat.shape[0] != t.size:
            raise ValueError(f"{path} reduced velocity length must match time series length.")
        ur_val = float(ur_flat[0])
        if not np.allclose(ur_flat, ur_val, rtol=1e-6, atol=1e-9):
            raise ValueError(f"{path} reduced velocity must be constant within a series.")
    return t, x, f, v, ur_val


def _load_metadata_map(meta_path: Path) -> dict[str, float]:
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file '{meta_path}' not found.")
    with meta_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Metadata file '{meta_path}' must contain a list.")
    mapping: dict[str, float] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        file_name = item.get("file")
        if not file_name:
            continue
        if "U" not in item:
            raise KeyError(f"Metadata entry for '{file_name}' is missing 'U'.")
        mapping[str(file_name)] = float(item["U"])
    if not mapping:
        raise ValueError(f"No valid entries found in metadata file '{meta_path}'.")
    return mapping


def _try_load_metadata_map(meta_path: Path) -> Optional[dict[str, float]]:
    """
    Best-effort metadata loading for force-coefficient mode.
    Returns None when metadata is unavailable/invalid so callers can fall back
    to U_r-based F0 construction.
    """
    try:
        return _load_metadata_map(meta_path)
    except Exception as exc:
        key = str(meta_path)
        if key not in _MISSING_METADATA_WARNED:
            _MISSING_METADATA_WARNED.add(key)
            warnings.warn(
                f"Could not load metadata file '{meta_path}'. "
                f"Falling back to U_r-based F0 conversion. Reason: {type(exc).__name__}: {exc}"
            )
        return None


def _finite_std_or_one(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return 1.0
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 1.0
    std = float(np.std(finite))
    if not np.isfinite(std) or std <= 0.0:
        return 1.0
    return std


def _safe_rel_rmse(pred: np.ndarray, target: np.ndarray, denom: float) -> float:
    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    t = np.asarray(target, dtype=np.float64).reshape(-1)
    n = int(min(p.size, t.size))
    if n <= 0:
        return float("nan")
    p = p[:n]
    t = t[:n]
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(t)):
        return float("inf")
    diff = p - t
    if not np.all(np.isfinite(diff)):
        return float("inf")
    scale = float(np.max(np.abs(diff)))
    if not np.isfinite(scale):
        return float("inf")
    if scale == 0.0:
        rmse = 0.0
    else:
        rmse = scale * float(np.sqrt(np.mean((diff / scale) ** 2)))
    denom = float(denom)
    if not np.isfinite(denom) or denom <= 0.0:
        denom = 1.0
    out = rmse / denom
    if not np.isfinite(out):
        return float("inf")
    return float(out)


def _f0_from_reduced_velocity(
    ur_val: float,
    *,
    rho: float,
    D: float,
    m_eff: float,
    k: float,
) -> float:
    ur = float(ur_val)
    if not np.isfinite(ur):
        raise ValueError(f"Invalid reduced velocity value for F0 conversion: {ur}")
    if m_eff <= 0.0 or k <= 0.0:
        raise ValueError(f"Invalid model parameters for F0 conversion: m_eff={m_eff}, k={k}")
    fn_hz = math.sqrt(float(k) / float(m_eff)) / (2.0 * math.pi)
    U_val = ur * float(fn_hz) * float(D)
    f0 = 0.5 * float(rho) * float(D) * (float(U_val) ** 2)
    if not np.isfinite(f0) or f0 <= 0.0:
        raise ValueError(f"Invalid F0 from U_r conversion: {f0}")
    return float(f0)


def _normalize_ur_filter(values: Any, *, key: str) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return np.empty((0,), dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{key} must contain only finite numeric values.")
    return arr


def _ur_in_filter(ur_value: float, ur_filter: np.ndarray, *, tol: float) -> bool:
    if ur_filter.size == 0:
        return False
    return bool(np.any(np.isclose(float(ur_value), ur_filter, rtol=0.0, atol=float(tol))))


def _read_reduced_velocity_from_npz(path: Path) -> float:
    with np.load(path) as data:
        if "U_r" not in data:
            raise ValueError(f"{path} is missing reduced velocity 'U_r'.")
        ur_arr = np.asarray(data["U_r"], dtype=float)
    if ur_arr.ndim == 0:
        return float(ur_arr)
    ur_flat = ur_arr.reshape(-1)
    if ur_flat.size == 0:
        raise ValueError(f"{path} has an empty reduced velocity array.")
    ur_val = float(ur_flat[0])
    if not np.allclose(ur_flat, ur_val, rtol=1e-6, atol=1e-9):
        raise ValueError(f"{path} reduced velocity must be constant within a series.")
    return ur_val


def _filter_paths_by_ur(
    paths: Sequence[Path],
    *,
    include_ur: np.ndarray | None,
    exclude_ur: np.ndarray | None,
    tol: float,
) -> list[Path]:
    filtered: list[Path] = []
    for path in paths:
        ur_val = _read_reduced_velocity_from_npz(path)
        if include_ur is not None and include_ur.size > 0 and not _ur_in_filter(ur_val, include_ur, tol=tol):
            continue
        if exclude_ur is not None and exclude_ur.size > 0 and _ur_in_filter(ur_val, exclude_ur, tol=tol):
            continue
        filtered.append(path)
    return filtered


def _infer_dt_target_from_data_cfg(data_cfg: Any) -> Optional[float]:
    def _dt_from_npz(path: Path) -> Optional[float]:
        if not path.exists():
            return None
        with np.load(path) as base:
            if "a" in base:
                t = np.asarray(base["a"])
            elif "time" in base:
                t = np.asarray(base["time"])
            else:
                return None
        if t.ndim != 1 or t.size < 2:
            return None
        return float(t[1] - t[0])

    # Prefer generated training-series dt when available.
    if bool(getattr(data_cfg, "use_generated_train_series", False)):
        series_dir = Path(getattr(data_cfg, "train_series_dir", ""))
        if series_dir and not series_dir.is_absolute():
            series_dir = (Path.cwd() / series_dir).resolve()
        if series_dir.exists():
            candidates: list[Path] = []
            train_dir = series_dir / "train"
            val_dir = series_dir / "val"
            if train_dir.exists() or val_dir.exists():
                if train_dir.exists():
                    candidates.extend(sorted(train_dir.glob("*.npz")))
                if val_dir.exists():
                    candidates.extend(sorted(val_dir.glob("*.npz")))
            else:
                candidates.extend(sorted(series_dir.glob("*.npz")))
            for path in candidates:
                dt_val = _dt_from_npz(path)
                if dt_val is not None:
                    return dt_val

    # Fallback to data file dt (legacy single-series mode).
    data_path = Path(getattr(data_cfg, "file", ""))
    if not data_path:
        return None
    if not data_path.is_absolute():
        data_path = (Path.cwd() / data_path).resolve()
    dt_val = _dt_from_npz(data_path)
    if dt_val is None:
        return None
    if bool(getattr(data_cfg, "reduce_time", False)):
        rf = max(1, int(getattr(data_cfg, "reduction_factor", 1)))
        dt_val = float(dt_val * rf)
    return dt_val

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
    rf_requested = max(1, int(reduction_factor))
    # Keep at least two samples after decimation whenever possible.
    rf = min(rf_requested, max(1, int(t.size) - 1))
    if rf < rf_requested:
        warnings.warn(
            "Reduction factor was too large for the current trajectory; "
            f"using step={rf} instead of {rf_requested} to keep at least two samples."
        )
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
    force_representation: str,
    f0_lookup: Optional[dict[str, float]],
    rho: float,
    D: float,
    m_eff: float,
    k: float,
    preserve_prefix_for_history: bool = False,
    min_history_context: int = 0,
) -> tuple[dict[str, Any], float]:
    t, x, f_meas, v_file, ur_val = _read_timeseries_npz(path)
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
    min_history_context = max(0, int(min_history_context))
    t0 = float(t[0])
    validation_start_idx = int(np.searchsorted(t, t0 + cut_start_seconds, side="left"))
    if preserve_prefix_for_history:
        if validation_start_idx < min_history_context:
            available_s = float(validation_start_idx * dt)
            needed_s = float(min_history_context * dt)
            raise ValueError(
                f"{path.name}: validation start at t={t0 + cut_start_seconds:.6g}s "
                f"(index={validation_start_idx}) is too early for TCN history_len={min_history_context}. "
                f"Need at least {needed_s:.6g}s ({min_history_context} samples) before validation start, "
                f"but only {available_s:.6g}s are available."
            )
        if (int(t.shape[0]) - validation_start_idx) < 2:
            raise ValueError(
                f"{path.name}: too few validation samples remain after cut_start_seconds={cut_start_seconds}."
            )
    else:
        if validation_start_idx > 0:
            mask = np.arange(t.shape[0]) >= validation_start_idx
            t = t[mask]
            x = x[mask]
            f_meas = f_meas[mask]
            if v_file is not None:
                v_file = v_file[mask]
        validation_start_idx = 0
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

    force_representation = str(force_representation).strip().lower()
    if force_representation not in {"force", "coefficient"}:
        raise ValueError("vpinn.force_representation must be one of: force, coefficient.")
    f0_val = None
    if force_representation == "coefficient":
        if f0_lookup is not None and path.name in f0_lookup:
            U_val = float(f0_lookup[path.name])
            f0_val = 0.5 * float(rho) * float(D) * float(U_val) ** 2
        else:
            if f0_lookup is not None and path.name not in f0_lookup:
                warnings.warn(
                    f"Metadata missing U for '{path.name}'. Falling back to U_r-based F0 conversion."
                )
            f0_val = _f0_from_reduced_velocity(
                ur_val,
                rho=float(rho),
                D=float(D),
                m_eff=float(m_eff),
                k=float(k),
            )
        if not np.isfinite(f0_val) or f0_val <= 0.0:
            raise ValueError(f"Invalid F0 for '{path.name}': {f0_val}")
        f_meas = f_meas / float(f0_val)

    ur_series = np.full((t.shape[0], 1), float(ur_val), dtype=np.float32)
    traj = {
        "name": path.name,
        "t": torch.from_numpy(t.astype(np.float32)),
        "x": torch.from_numpy(x.astype(np.float32)),
        "v": torch.from_numpy(np.asarray(v, dtype=np.float32)),
        "f": torch.from_numpy(f_meas.astype(np.float32)),
        "ur": torch.from_numpy(ur_series),
        # First timestep included in validation targets (prefix before this can be used as TCN history).
        "val_start_idx": int(validation_start_idx),
    }
    if f0_val is not None:
        f0_series = np.full((t.shape[0], 1), float(f0_val), dtype=np.float32)
        traj["f0"] = torch.from_numpy(f0_series)
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


class WindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        trajectories: list[dict[str, Any]],
        *,
        window_intervals: int,
        stride: int,
        history_context: int = 0,
        return_scale: bool = False,
    ) -> None:
        if window_intervals < 1:
            raise ValueError("vpinn.window_M must be >= 1")
        if stride < 1:
            raise ValueError("vpinn.stride must be >= 1")
        if int(history_context) < 0:
            raise ValueError("history_context must be >= 0")
        self.trajectories = trajectories
        self.M = int(window_intervals)
        self.M1 = self.M + 1
        self.stride = int(stride)
        self.context = int(history_context)
        self.window_total = self.M1 + self.context
        self._return_scale = bool(return_scale)
        self._return_f0 = any("f0" in traj for traj in self.trajectories)

        traj_ids: list[np.ndarray] = []
        starts: list[np.ndarray] = []
        for traj_id, traj in enumerate(self.trajectories):
            x = traj["x"]
            length = int(x.shape[0])
            val_start_idx = max(0, int(traj.get("val_start_idx", 0)))
            min_start = max(0, val_start_idx - self.context)
            max_start = length - self.window_total
            if max_start < min_start:
                continue
            start_idx = np.arange(min_start, max_start + 1, self.stride, dtype=np.int32)
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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:  # type: ignore[override]
        traj = self.trajectories[int(self._traj_ids[idx])]
        start = int(self._starts[idx])
        end = start + self.window_total
        x = traj["x"][start:end]
        v = traj["v"][start:end]
        f = traj["f"][start:end]
        ur = traj["ur"][start:end]
        items: list[torch.Tensor] = [x, v, f, ur]
        if self._return_scale:
            scale = float(traj.get("scale", 1.0))
            items.append(torch.tensor(scale, dtype=torch.float32))
        if self._return_f0:
            f0 = traj["f0"][start:end]
            items.append(f0)
        return tuple(items)  # type: ignore[return-value]


def _prepare_trajectories(config: Config) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    data_cfg = config.data
    smoothing_cfg = config.smoothing
    vp = dict(config.vpinn or {})

    velocity_source = str(vp.get("velocity_source", "compute")).strip().lower()
    force_representation = str(vp.get("force_representation", "force")).strip().lower()
    if force_representation not in {"force", "coefficient"}:
        raise ValueError("vpinn.force_representation must be one of: force, coefficient.")
    dt_target = vp.get("dt_target", None)
    if dt_target is None:
        dt_target = _infer_dt_target_from_data_cfg(data_cfg)
    dt_target = None if dt_target is None else float(dt_target)
    train_include_ur = _normalize_ur_filter(vp.get("train_include_ur"), key="vpinn.train_include_ur")
    train_exclude_ur = _normalize_ur_filter(vp.get("train_exclude_ur"), key="vpinn.train_exclude_ur")
    train_ur_filter_tol = float(vp.get("train_ur_filter_tol", 1e-6))
    if train_ur_filter_tol < 0.0:
        raise ValueError("vpinn.train_ur_filter_tol must be non-negative.")

    f0_lookup: Optional[dict[str, float]] = None
    if force_representation == "coefficient":
        if data_cfg.use_generated_train_series:
            meta_path = Path(data_cfg.train_series_dir) / "metadata.json"
        else:
            meta_path = Path(data_cfg.file).resolve().parent / "metadata.json"
        f0_lookup = _try_load_metadata_map(meta_path)
    m_eff_cfg = float(_m_eff_from_model_cfg(config.model))
    k_cfg = float(getattr(config.model, "k", 1218.0))

    if data_cfg.use_generated_train_series:
        series_dir = Path(data_cfg.train_series_dir)
        if not series_dir.is_absolute():
            series_dir = (Path.cwd() / series_dir).resolve()
        if not series_dir.exists():
            raise FileNotFoundError(f"Training series directory '{series_dir}' does not exist.")
        train_dir = series_dir / "train"
        val_dir = series_dir / "val"
        if train_dir.exists() or val_dir.exists():
            if not train_dir.exists() or not val_dir.exists():
                raise FileNotFoundError(f"Expected both train/ and val/ under '{series_dir}'.")
            train_files = sorted(train_dir.glob("*.npz"))
            val_files = sorted(val_dir.glob("*.npz"))
            if not train_files:
                raise FileNotFoundError(f"No '.npz' files found in '{train_dir}'.")
            if not val_files:
                raise FileNotFoundError(f"No '.npz' files found in '{val_dir}'.")
            sources: list[Path] = list(train_files)
            val_sources: list[Path] = list(val_files)
        else:
            files = sorted(series_dir.glob("*.npz"))
            if not files:
                raise FileNotFoundError(f"No '.npz' files found in '{series_dir}'.")
            sources = list(files)
            val_sources = []
    else:
        data_path = Path(data_cfg.file)
        if not data_path.is_absolute():
            data_path = (Path.cwd() / data_path).resolve()
        sources = [data_path]
        val_sources = []

    if data_cfg.use_generated_train_series:
        has_include = train_include_ur is not None and train_include_ur.size > 0
        has_exclude = train_exclude_ur is not None and train_exclude_ur.size > 0
        if has_include or has_exclude:
            print(
                "Applying VPINN training U_r filter: "
                f"include={None if train_include_ur is None else train_include_ur.tolist()}, "
                f"exclude={None if train_exclude_ur is None else train_exclude_ur.tolist()}, "
                f"tol={train_ur_filter_tol:g}"
            )
        sources = _filter_paths_by_ur(
            sources,
            include_ur=train_include_ur,
            exclude_ur=train_exclude_ur,
            tol=train_ur_filter_tol,
        )
        if not sources:
            raise ValueError(
                "No VPINN training series left after U_r filtering. "
                f"include={vp.get('train_include_ur')}, "
                f"exclude={vp.get('train_exclude_ur')}, "
                f"series_dir='{getattr(data_cfg, 'train_series_dir', '')}'."
            )
    val_history_context = _configured_validation_history_len(config)

    trajectories: list[dict[str, Any]] = []
    val_trajectories: list[dict[str, Any]] = []
    dt_ref: Optional[float] = None
    train_cut_start_seconds = float(
        data_cfg.cut_start_seconds_train
        if getattr(data_cfg, "cut_start_seconds_train", None) is not None
        else getattr(data_cfg, "cut_start_seconds", 0.0)
    )
    val_cut_start_seconds = float(
        data_cfg.cut_start_seconds_val
        if getattr(data_cfg, "cut_start_seconds_val", None) is not None
        else getattr(data_cfg, "cut_start_seconds", 0.0)
    )
    for path in sources:
        traj, dt = _load_trajectory(
            path=path,
            dt_target=dt_target,
            velocity_source=velocity_source,
            smoothing_cfg=smoothing_cfg,
            reduce_time=False,
            reduction_factor=1,
            cut_start_seconds=train_cut_start_seconds,
            force_representation=force_representation,
            f0_lookup=f0_lookup,
            rho=float(getattr(config.model, "rho", 1000.0)),
            D=float(getattr(config.model, "D", 0.1)),
            m_eff=m_eff_cfg,
            k=k_cfg,
        )
        if dt_ref is None:
            dt_ref = dt
        elif not np.isclose(dt, float(dt_ref), rtol=1e-9, atol=1e-12):
            raise ValueError(f"{path} has dt={dt} but expected dt={dt_ref}.")
        trajectories.append(traj)
    for path in val_sources:
        traj, dt = _load_trajectory(
            path=path,
            dt_target=dt_target,
            velocity_source=velocity_source,
            smoothing_cfg=smoothing_cfg,
            reduce_time=False,
            reduction_factor=1,
            cut_start_seconds=val_cut_start_seconds,
            force_representation=force_representation,
            f0_lookup=f0_lookup,
            rho=float(getattr(config.model, "rho", 1000.0)),
            D=float(getattr(config.model, "D", 0.1)),
            m_eff=m_eff_cfg,
            k=k_cfg,
            preserve_prefix_for_history=(val_history_context > 0),
            min_history_context=val_history_context,
        )
        if dt_ref is None:
            dt_ref = dt
        elif not np.isclose(dt, float(dt_ref), rtol=1e-9, atol=1e-12):
            raise ValueError(f"{path} has dt={dt} but expected dt={dt_ref}.")
        val_trajectories.append(traj)

    if dt_ref is None:
        raise ValueError("No trajectories loaded.")
    return trajectories, val_trajectories, float(dt_ref)


def _test_functions(
    M: int,
    dt: float,
    *,
    num_poly: int,
    num_sine: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    M1 = int(M) + 1
    tau = torch.linspace(0.0, 1.0, M1, dtype=torch.float32)
    T = float(M) * float(dt)
    w_list: list[torch.Tensor] = []
    wdot_list: list[torch.Tensor] = []

    num_poly = int(num_poly)
    num_sine = int(num_sine)
    if num_poly < 0 or num_sine < 0:
        raise ValueError("num_poly and num_sine must be >= 0.")

    if num_poly > 0:
        for degree in range(num_poly):
            w_list.append(tau**degree)
            if degree == 0:
                wdot_list.append(torch.zeros_like(tau))
            else:
                wdot_list.append(float(degree) * tau ** (degree - 1) / float(T))

    if num_sine > 0:
        for k in range(1, num_sine + 1):
            phase = float(k) * math.pi * tau
            w_list.append(torch.sin(phase))
            wdot_list.append((float(k) * math.pi / float(T)) * torch.cos(phase))

    if not w_list:
        raise ValueError("At least one test function is required (num_poly + num_sine must be > 0).")
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
    f0: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    m = m.view(1, 1, -1)
    c = c.view(1, 1, -1)
    k = k.view(1, 1, -1)
    if f0 is not None:
        f0 = f0.to(device=x.device, dtype=x.dtype)
        if f0.ndim == 2:
            f0 = f0.unsqueeze(-1)
        if f0.ndim != 3:
            raise ValueError("f0 must have shape (B, M1, 1) or (B, M1).")
        mv = (m * v) / f0
        cv_kx_minus_f = (c * v + k * x) / f0 - f_pred
    else:
        mv = m * v
        cv_kx_minus_f = c * v + k * x - f_pred
    mvM = mv[:, -1, :].unsqueeze(1)
    mv0 = mv[:, 0, :].unsqueeze(1)
    wM = w[:, -1].view(1, -1, 1)
    w0 = w[:, 0].view(1, -1, 1)
    boundary = mvM * wM - mv0 * w0

    ww = w.unsqueeze(0).unsqueeze(-1)  # (1, L, M1, 1)
    wwdot = wdot.unsqueeze(0).unsqueeze(-1)  # (1, L, M1, 1)
    alpha_w = alpha.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # (1, 1, M1, 1)

    term = cv_kx_minus_f.unsqueeze(1) * ww - mv.unsqueeze(1) * wwdot
    trap = float(dt) * torch.sum(alpha_w * term, dim=2)  # (B, L, d)
    return boundary + trap


def _per_traj_force_rms(traj: dict[str, Any]) -> float:
    f = torch.as_tensor(traj["f"], dtype=torch.float32)
    rms = torch.sqrt(torch.mean(f.pow(2)))
    val = float(rms.detach().cpu())
    return val if np.isfinite(val) and val > 0.0 else 1.0


def _per_traj_residual_rms(
    traj: dict[str, Any],
    *,
    dt: float,
    m: torch.Tensor,
    c: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    wdot: torch.Tensor,
    alpha: torch.Tensor,
    window_M: int,
    stride: int,
    eps: float,
) -> float:
    x = torch.as_tensor(traj["x"], dtype=torch.float32)
    v = torch.as_tensor(traj["v"], dtype=torch.float32)
    f = torch.as_tensor(traj["f"], dtype=torch.float32)
    f0 = None
    if "f0" in traj:
        f0 = torch.as_tensor(traj["f0"], dtype=torch.float32)
    length = int(x.shape[0])
    M1 = int(window_M) + 1
    if length < M1:
        return 1.0
    loss_sum = 0.0
    count = 0
    for start in range(0, length - M1 + 1, int(stride)):
        end = start + M1
        x_win = x[start:end].unsqueeze(0)
        v_win = v[start:end].unsqueeze(0)
        f_win = f[start:end].unsqueeze(0)
        f0_win = None
        if f0 is not None:
            f0_win = f0[start:end].unsqueeze(0)
        R = _weak_residual(
            x=x_win,
            v=v_win,
            f_pred=f_win,
            m=m,
            c=c,
            k=k,
            dt=dt,
            w=w,
            wdot=wdot,
            alpha=alpha,
            f0=f0_win,
        )
        loss_w = torch.mean(R.pow(2))
        loss_sum += float(loss_w.detach().cpu())
        count += 1
    mean_loss = loss_sum / float(max(count, 1))
    scale = math.sqrt(max(mean_loss, float(eps)))
    return scale if np.isfinite(scale) and scale > 0.0 else 1.0


def _apply_per_traj_scale(
    trajectories: list[dict[str, Any]],
    *,
    mode: str,
    dt: float,
    m: torch.Tensor,
    c: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    wdot: torch.Tensor,
    alpha: torch.Tensor,
    window_M: int,
    stride: int,
    eps: float,
) -> None:
    if not trajectories:
        return
    mode = str(mode).strip().lower()
    m_cpu = m.detach().cpu()
    c_cpu = c.detach().cpu()
    k_cpu = k.detach().cpu()
    w_cpu = w.detach().cpu()
    wdot_cpu = wdot.detach().cpu()
    alpha_cpu = alpha.detach().cpu()
    for traj in trajectories:
        if mode == "force_rms":
            scale = _per_traj_force_rms(traj)
        elif mode == "residual_rms":
            scale = _per_traj_residual_rms(
                traj,
                dt=dt,
                m=m_cpu,
                c=c_cpu,
                k=k_cpu,
                w=w_cpu,
                wdot=wdot_cpu,
                alpha=alpha_cpu,
                window_M=window_M,
                stride=stride,
                eps=eps,
            )
        else:
            scale = 1.0
        traj["scale"] = float(scale)


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
    per_traj_norm_eps: float = 0.0,
    expect_scale: bool = False,
    expect_f0: bool = False,
) -> dict[str, float]:
    model.eval()
    loss_f_sum = 0.0
    loss_w_sum = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 4:
                x_win, v_win, f_meas, ur_win = batch
                scale = None
                f0 = None
            elif len(batch) == 5:
                x_win, v_win, f_meas, ur_win, extra = batch
                if expect_scale and not expect_f0:
                    scale = extra
                    f0 = None
                elif expect_f0 and not expect_scale:
                    scale = None
                    f0 = extra
                else:
                    raise ValueError("Unexpected batch format (missing scale or f0).")
            elif len(batch) == 6:
                x_win, v_win, f_meas, ur_win, scale, f0 = batch
            else:
                raise ValueError("Unexpected batch format from dataloader.")
            x_win = x_win.to(device, non_blocking=non_blocking)
            v_win = v_win.to(device, non_blocking=non_blocking)
            f_meas = f_meas.to(device, non_blocking=non_blocking)
            ur_win = ur_win.to(device, non_blocking=non_blocking)
            if scale is not None:
                scale = scale.to(device, non_blocking=non_blocking).view(-1)
            if f0 is not None:
                f0 = f0.to(device, non_blocking=non_blocking)

            M1_target = int(w.shape[-1])
            if int(x_win.shape[1]) < M1_target:
                raise ValueError(
                    f"Window length {int(x_win.shape[1])} is shorter than target length {M1_target}."
                )
            start_idx = int(x_win.shape[1]) - M1_target
            x_eval = x_win[:, start_idx:, :]
            v_eval = v_win[:, start_idx:, :]
            f_eval = f_meas[:, start_idx:, :]
            f0_eval = f0[:, start_idx:, :] if f0 is not None else None
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                f_pred = _vpinn_force_sequence(model, x_win, v_win, ur_win)
                f_pred_eval = f_pred[:, start_idx:, :]
                per_loss_f = torch.mean((f_pred_eval - f_eval) ** 2, dim=(1, 2))
                if scale is not None:
                    per_loss_f = per_loss_f / (scale * scale + float(per_traj_norm_eps))
                loss_f = torch.mean(per_loss_f)
                if use_weak_loss:
                    R = _weak_residual(
                        x=x_eval,
                        v=v_eval,
                        f_pred=f_pred_eval,
                        m=m,
                        c=c,
                        k=k,
                        dt=dt,
                        w=w,
                        wdot=wdot,
                        alpha=alpha,
                        f0=f0_eval,
                    )
                    per_loss_w = torch.mean(R.pow(2), dim=(1, 2))
                    if scale is not None:
                        per_loss_w = per_loss_w / (scale * scale + float(per_traj_norm_eps))
                    loss_w = torch.mean(per_loss_w)
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


def _per_ur_loss_map_vpinn(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    non_blocking: bool,
    dt: float,
    m: torch.Tensor,
    c: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    wdot: torch.Tensor,
    alpha: torch.Tensor,
    use_force_loss: bool,
    use_weak_loss: bool,
    rollout_force_steps: int,
    rollout_substeps: int,
    expect_scale: bool,
    expect_f0: bool,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    per_traj_norm_eps: float,
) -> dict[str, dict[float, float]]:
    model.eval()
    amp_enabled = bool(amp_enabled) and device.type == "cuda"
    buckets: dict[str, dict[float, list[float]]] = {
        "loss_data": {},
        "loss_physics": {},
    }
    if rollout_force_steps > 0:
        buckets["loss_rollout_force"] = {}

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 4:
                x_win, v_win, f_meas, ur_win = batch
                scale = None
                f0 = None
            elif len(batch) == 5:
                x_win, v_win, f_meas, ur_win, extra = batch
                if expect_scale and not expect_f0:
                    scale = extra
                    f0 = None
                elif expect_f0 and not expect_scale:
                    scale = None
                    f0 = extra
                else:
                    raise ValueError("Unexpected batch format (missing scale or f0).")
            elif len(batch) == 6:
                x_win, v_win, f_meas, ur_win, scale, f0 = batch
            else:
                raise ValueError("Unexpected batch format from dataloader.")

            x_win = x_win.to(device, non_blocking=non_blocking)
            v_win = v_win.to(device, non_blocking=non_blocking)
            f_meas = f_meas.to(device, non_blocking=non_blocking)
            ur_win = ur_win.to(device, non_blocking=non_blocking)
            if scale is not None:
                scale = scale.to(device, non_blocking=non_blocking).view(-1)
            if f0 is not None:
                f0 = f0.to(device, non_blocking=non_blocking)

            M1_target = int(w.shape[-1])
            if int(x_win.shape[1]) < M1_target:
                raise ValueError(
                    f"Window length {int(x_win.shape[1])} is shorter than target length {M1_target}."
                )
            start_idx = int(x_win.shape[1]) - M1_target
            x_eval = x_win[:, start_idx:, :]
            v_eval = v_win[:, start_idx:, :]
            f_eval = f_meas[:, start_idx:, :]
            ur_eval = ur_win[:, start_idx:, :]
            f0_eval = f0[:, start_idx:, :] if f0 is not None else None
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                f_pred = _vpinn_force_sequence(model, x_win, v_win, ur_win)
                f_pred_eval = f_pred[:, start_idx:, :]
                per_loss_f = torch.mean((f_pred_eval - f_eval) ** 2, dim=(1, 2))
                if scale is not None:
                    per_loss_f = per_loss_f / (scale * scale + float(per_traj_norm_eps))
                per_loss_w = per_loss_f.new_zeros(per_loss_f.shape)
                if use_weak_loss:
                    R = _weak_residual(
                        x=x_eval,
                        v=v_eval,
                        f_pred=f_pred_eval,
                        m=m,
                        c=c,
                        k=k,
                        dt=dt,
                        w=w,
                        wdot=wdot,
                        alpha=alpha,
                        f0=f0_eval,
                    )
                    per_loss_w = torch.mean(R.pow(2), dim=(1, 2))
                    if scale is not None:
                        per_loss_w = per_loss_w / (scale * scale + float(per_traj_norm_eps))

                per_roll = None
                if rollout_force_steps > 0:
                    steps_k = min(int(rollout_force_steps), int(M1_target) - 1)
                    if steps_k > 0:
                        f0_step = f0_eval[:, 0, :] if f0_eval is not None else None
                        _x_seq, _v_seq, f_seq = rollout_rk4(
                            model=model,
                            x0=x_eval[:, 0, :],
                            v0=v_eval[:, 0, :],
                            ur0=ur_eval[:, 0, :],
                            steps=steps_k,
                            dt=dt,
                            substeps=rollout_substeps,
                            m=m,
                            c=c,
                            k=k,
                            f0=f0_step,
                        )
                        f_roll = f_seq[:, : steps_k + 1, :]
                        f_true = f_eval[:, : steps_k + 1, :]
                        per_roll = torch.mean((f_roll - f_true) ** 2, dim=(1, 2))
                        if scale is not None:
                            per_roll = per_roll / (scale * scale + float(per_traj_norm_eps))

            ur_vals = ur_eval[:, 0, 0].detach().cpu().numpy()
            per_loss_f_vals = per_loss_f.detach().cpu().numpy()
            per_loss_w_vals = per_loss_w.detach().cpu().numpy()
            for i, u in enumerate(ur_vals):
                key = float(np.round(u, 6))
                if use_force_loss:
                    buckets["loss_data"].setdefault(key, []).append(float(per_loss_f_vals[i]))
                if use_weak_loss:
                    buckets["loss_physics"].setdefault(key, []).append(float(per_loss_w_vals[i]))
            if per_roll is not None:
                per_roll_vals = per_roll.detach().cpu().numpy()
                for i, u in enumerate(ur_vals):
                    key = float(np.round(u, 6))
                    buckets["loss_rollout_force"].setdefault(key, []).append(float(per_roll_vals[i]))

    out: dict[str, dict[float, float]] = {}
    for name, by_ur in buckets.items():
        if not by_ur:
            continue
        out[name] = {ur: float(np.mean(vals)) for ur, vals in by_ur.items()}
    return out


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
    rollout_substeps: int = 1,
    tag_prefix: str = "val/rollout",
    step: int | None = None,
    log_extra_metrics: bool = False,
    include_disp_nrmse: bool = True,
    include_force_nrmse: bool = True,
    log_metrics: bool = True,
    log_plots: bool = True,
    title_suffix: str = "",
) -> dict[str, float]:
    x_true_t = traj["x"].to(device)
    v_true_t = traj["v"].to(device)
    f_true_t = traj["f"].to(device)
    ur_true_t = traj["ur"].to(device)
    f0_true_t = traj.get("f0", None)
    if f0_true_t is not None:
        f0_true_t = f0_true_t.to(device)
    t_full = traj["t"].detach().cpu().numpy()
    val_start_idx = max(0, int(traj.get("val_start_idx", 0)))
    if x_true_t.ndim != 2:
        return {}
    d = int(x_true_t.shape[-1])
    if d < 1:
        return {}
    if d > 1:
        print("vpinn rollout validation: d>1 detected; logging only the first DOF.")
    if val_start_idx >= int(x_true_t.shape[0]) - 1:
        raise ValueError(
            f"Trajectory '{traj.get('name', '<unknown>')}' has too few samples after validation start "
            f"(val_start_idx={val_start_idx}, total={int(x_true_t.shape[0])})."
        )

    use_tcn = _is_tcn_force_model(model)
    context = _tcn_history_len(model) if use_tcn else 0
    if use_tcn and val_start_idx < context:
        raise ValueError(
            f"Trajectory '{traj.get('name', '<unknown>')}' validation starts at index {val_start_idx}, "
            f"but TCN history_len is {context}. Increase validation start time/cut."
        )

    x_eval_t = x_true_t[val_start_idx:, :]
    v_eval_t = v_true_t[val_start_idx:, :]
    f_eval_t = f_true_t[val_start_idx:, :]
    ur_eval_t = ur_true_t[val_start_idx:, :]
    f0_eval_t = f0_true_t[val_start_idx:, :] if f0_true_t is not None else None
    t_np = t_full[val_start_idx:]

    steps = int(x_eval_t.shape[0] - 1)
    if steps < 1:
        return {}
    if use_tcn and context > 0:
        hist_start = val_start_idx - context
        x_hist_init = x_true_t[hist_start : val_start_idx + 1, :].unsqueeze(0)
        v_hist_init = v_true_t[hist_start : val_start_idx + 1, :].unsqueeze(0)
        ur_hist_init = ur_true_t[hist_start : val_start_idx + 1, :].unsqueeze(0)
        x_seq, v_seq, f_seq = rollout_rk4(
            model=model,
            x0=x_eval_t[0:1, :],
            v0=v_eval_t[0:1, :],
            ur0=ur_eval_t[0:1, :],
            steps=steps,
            dt=dt,
            substeps=rollout_substeps,
            m=m,
            c=c,
            k=k,
            f0=(f0_eval_t[0:1, :] if f0_eval_t is not None else None),
            x_hist_init=x_hist_init,
            v_hist_init=v_hist_init,
            ur_hist_init=ur_hist_init,
        )
    else:
        x_seq, v_seq, f_seq = rollout_rk4(
            model=model,
            x0=x_eval_t[0:1, :],
            v0=v_eval_t[0:1, :],
            ur0=ur_eval_t[0:1, :],
            steps=steps,
            dt=dt,
            substeps=rollout_substeps,
            m=m,
            c=c,
            k=k,
            f0=(f0_eval_t[0:1, :] if f0_eval_t is not None else None),
        )
    x_pred = x_seq[0, :, 0].detach().cpu().numpy()
    v_pred = v_seq[0, :, 0].detach().cpu().numpy()
    f_pred = f_seq[0, :, 0].detach().cpu().numpy()
    x_true = x_eval_t[:, 0].detach().cpu().numpy()
    f_true = f_eval_t[:, 0].detach().cpu().numpy()

    disp_std = _finite_std_or_one(x_true)
    rel_rmse_y = _safe_rel_rmse(x_pred, x_true, disp_std)
    metrics: dict[str, float] = {}
    if include_disp_nrmse:
        metrics[DISP_ROLLOUT_NRMSE_KEY] = rel_rmse_y

    force_std = _finite_std_or_one(f_true)
    if f0_eval_t is not None:
        f0_np = np.asarray(f0_eval_t[:, 0].detach().cpu().numpy(), dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            f_true_force = np.asarray(f_true, dtype=np.float64) * f0_np
            f_pred_force = np.asarray(f_pred, dtype=np.float64) * f0_np
        force_std = _finite_std_or_one(f_true_force)
        rel_rmse_force = _safe_rel_rmse(f_pred_force, f_true_force, force_std)
        if include_force_nrmse:
            metrics[FORCE_ROLLOUT_NRMSE_KEY] = rel_rmse_force
    else:
        rel_rmse_force = _safe_rel_rmse(f_pred, f_true, force_std)
        if include_force_nrmse:
            metrics[FORCE_ROLLOUT_NRMSE_KEY] = rel_rmse_force

    with torch.no_grad():
        f_on_data_full = _vpinn_force_on_trajectory(model, x_true_t, v_true_t, ur_true_t)[:, 0].detach().cpu().numpy()
        f_on_data = f_on_data_full[val_start_idx:]
    if f0_eval_t is not None:
        f0_np = np.asarray(f0_eval_t[:, 0].detach().cpu().numpy(), dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            f_on_data_force = np.asarray(f_on_data, dtype=np.float64) * f0_np
            f_true_force = np.asarray(f_true, dtype=np.float64) * f0_np
        force_std = _finite_std_or_one(f_true_force)
        rel_rmse_force_on_data = _safe_rel_rmse(f_on_data_force, f_true_force, force_std)
        metrics[FORCE_MAPPING_NRMSE_KEY] = rel_rmse_force_on_data
    else:
        rel_rmse_force_on_data = _safe_rel_rmse(f_on_data, f_true, force_std)
        metrics[FORCE_MAPPING_NRMSE_KEY] = rel_rmse_force_on_data

    if log_extra_metrics:
        if np.all(np.isfinite(x_true)) and np.all(np.isfinite(x_pred)):
            freq_true = dominant_frequency(x_true, dt)
            freq_pred = dominant_frequency(x_pred, dt)
            freq_rel = abs(relative_error(freq_pred, freq_true))
            if np.isfinite(freq_rel):
                metrics[DOMINANT_FREQ_REL_ERROR_KEY] = float(freq_rel)

            amp_true = mean_displacement_amplitude(x_true)
            amp_pred = mean_displacement_amplitude(x_pred)
            amp_rel = abs(relative_error(amp_pred, amp_true))
            if np.isfinite(amp_rel):
                metrics[MEAN_DISP_AMP_REL_ERROR_KEY] = float(amp_rel)

            disp_spec_err = spectral_relative_error(x_true, x_pred, dt)
            if np.isfinite(disp_spec_err):
                metrics[DISP_SPECTRAL_SHAPE_ERROR_KEY] = float(disp_spec_err)

        if f0_eval_t is not None:
            force_true_for_spec = f_true_force
            force_pred_for_spec = f_pred_force
        else:
            force_true_for_spec = f_true
            force_pred_for_spec = f_pred
        if np.all(np.isfinite(force_true_for_spec)) and np.all(np.isfinite(force_pred_for_spec)):
            force_spec_err = spectral_relative_error(force_true_for_spec, force_pred_for_spec, dt)
            if np.isfinite(force_spec_err):
                metrics[FORCE_SPECTRAL_SHAPE_ERROR_KEY] = float(force_spec_err)

    if log_metrics:
        for metric_name, metric_value in metrics.items():
            writer.add_scalar(f"val/{metric_name}", float(metric_value), epoch)

    if log_plots:
        y_true_norm = x_true / float(D)
        y_pred_norm = x_pred / float(D)
        freq = float(torch.sqrt(k[0] / m[0]).detach().cpu())
        denom = float(freq * float(D)) if freq > 0 else 1.0
        p_pred_norm = v_pred / denom

        zoom_mask = create_zoom_mask(t_np)
        middle_mask = create_window_mask(t_np, middle_time_plot)
        middle_window = (float(middle_time_plot[0]), float(middle_time_plot[1]))
        ur_val = float(ur_eval_t[0, 0].detach().cpu().item())
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
            reduced_velocity=ur_val,
            tag_prefix=tag_prefix,
            step=step,
            title_suffix=title_suffix,
        )
        log_force_plots(
            writer,
            epoch,
            t_np,
            f_pred,
            f_true,
            zoom_mask,
            middle_mask,
            middle_window,
            reduced_velocity=ur_val,
            tag_prefix=tag_prefix,
            step=step,
            title_suffix=title_suffix,
        )
    return metrics


def _sample_one_traj_per_ur(
    trajs: Sequence[dict[str, Any]],
    *,
    seed: int,
) -> list[int]:
    by_ur: dict[float, list[int]] = {}
    for idx, traj in enumerate(trajs):
        ur_obj = traj.get("ur", None)
        if ur_obj is None:
            continue
        if torch.is_tensor(ur_obj):
            ur_arr = np.asarray(ur_obj.detach().cpu(), dtype=float).reshape(-1)
        else:
            ur_arr = np.asarray(ur_obj, dtype=float).reshape(-1)
        if ur_arr.size == 0 or not np.isfinite(ur_arr[0]):
            continue
        ur_key = float(np.round(float(ur_arr[0]), 6))
        by_ur.setdefault(ur_key, []).append(idx)
    if not by_ur:
        return []
    rng = np.random.default_rng(int(seed))
    selected: list[int] = []
    for ur_key in sorted(by_ur):
        candidates = np.asarray(by_ur[ur_key], dtype=int)
        selected.append(int(rng.choice(candidates)))
    return selected


def train(config: Config, config_name: str) -> None:
    vp = dict(config.vpinn or {})
    runtime_cfg = config.runtime
    precision_cfg = config.precision
    compile_cfg = config.compile
    training_cfg = config.training
    optim_cfg = config.optim
    monitoring_cfg = config.monitoring
    log_extra_validation_metrics = bool(getattr(monitoring_cfg, "log_extra_validation_metrics", False))
    rollout_include_disp_nrmse = bool(getattr(monitoring_cfg, "rollout_include_disp_nrmse", True))
    rollout_include_force_nrmse = bool(getattr(monitoring_cfg, "rollout_include_force_nrmse", True))

    window_M = int(vp.get("window_M", 50))
    stride = int(vp.get("stride", 1))
    val_fraction = float(vp.get("val_fraction", 0.1))
    split_seed = int(vp.get("split_seed", 0))
    wf = float(vp.get("wf", 1.0))
    ww = float(vp.get("ww", 1.0))
    use_force_loss = bool(vp.get("use_force_loss", True))
    use_weak_loss = bool(vp.get("use_weak_loss", True))
    num_poly = int(vp.get("num_poly_test", 2))
    num_sine = int(vp.get("num_sine_test", 0))
    rollout_force_weight = float(vp.get("rollout_force_weight", 0.0))
    rollout_force_steps = int(vp.get("rollout_force_steps", 0))
    rollout_force_every = int(vp.get("rollout_force_every", 1))
    rollout_force_batch_size = int(vp.get("rollout_force_batch_size", 0))
    rollout_train_substeps = int(vp.get("rollout_train_substeps", 1))
    rollout_val_substeps = int(vp.get("rollout_val_substeps", 1))
    if rollout_force_batch_size < 0:
        raise ValueError("vpinn.rollout_force_batch_size must be >= 0.")
    if rollout_train_substeps < 1:
        raise ValueError("vpinn.rollout_train_substeps must be >= 1.")
    if rollout_val_substeps < 1:
        raise ValueError("vpinn.rollout_val_substeps must be >= 1.")
    force_representation = str(vp.get("force_representation", "force")).strip().lower()
    if force_representation not in {"force", "coefficient"}:
        raise ValueError("vpinn.force_representation must be one of: force, coefficient.")
    use_force_coeff = force_representation == "coefficient"
    f0_lookup: Optional[dict[str, float]] = None
    if use_force_coeff:
        if bool(getattr(config.data, "use_generated_train_series", False)):
            meta_path = Path(config.data.train_series_dir) / "metadata.json"
        else:
            meta_path = Path(config.data.file).resolve().parent / "metadata.json"
        f0_lookup = _try_load_metadata_map(meta_path)
    per_traj_norm = str(vp.get("per_traj_norm", "none")).strip().lower()
    per_traj_norm_eps = float(vp.get("per_traj_norm_eps", 1e-8))
    if per_traj_norm not in {"none", "force_rms", "residual_rms"}:
        raise ValueError("vpinn.per_traj_norm must be one of: none, force_rms, residual_rms.")
    if not (use_force_loss or use_weak_loss):
        raise ValueError("vpinn must enable at least one of: use_force_loss, use_weak_loss.")

    use_gradnorm = bool(vp.get("use_gradnorm", False))
    gradnorm_alpha = float(vp.get("gradnorm_alpha", 0.9))
    gradnorm_eps = float(vp.get("gradnorm_eps", 1e-8))
    gradnorm_min_weight = float(vp.get("gradnorm_min_weight", 0.1))
    gradnorm_max_weight = float(vp.get("gradnorm_max_weight", 10.0))
    gradnorm_update_every_steps = int(vp.get("gradnorm_update_every_steps", 1))
    gradnorm_update_every_steps = max(1, gradnorm_update_every_steps)

    device = select_device(os.getenv("TRAIN_DEVICE", str(runtime_cfg.device)))
    print(f"Using device: {device}")
    configure_tf32(device, bool(precision_cfg.use_tf32))
    set_num_threads_from_slurm(default=1)
    non_blocking = device.type == "cuda"

    val_history_context = _configured_validation_history_len(config)
    train_trajs, val_trajs, dt = _prepare_trajectories(config)
    validation_only_data_file = bool(vp.get("validation_only_data_file", False))
    use_data_file_for_validation = bool(vp.get("use_data_file_for_validation", False))
    if val_trajs:
        validation_only_data_file = False
        use_data_file_for_validation = False
    if validation_only_data_file:
        val_trajs = []
    else:
        if not val_trajs:
            train_trajs, val_trajs = _split_by_trajectory(train_trajs, val_fraction=val_fraction, seed=split_seed)
    if not train_trajs:
        raise ValueError("Empty training split. Reduce vpinn.val_fraction or provide more trajectories.")
    if use_data_file_for_validation:
        data_path = Path(config.data.file)
        if not data_path.is_absolute():
            data_path = (Path.cwd() / data_path).resolve()
        val_reduce_time = bool(getattr(config.data, "reduce_time", False))
        val_reduction_factor = int(getattr(config.data, "reduction_factor", 1))
        cut_start_seconds = float(
            config.data.cut_start_seconds_val
            if getattr(config.data, "cut_start_seconds_val", None) is not None
            else getattr(config.data, "cut_start_seconds", 0.0)
        )
        val_traj, val_dt = _load_trajectory(
            path=data_path,
            dt_target=dt,
            velocity_source=str(vp.get("velocity_source", "compute")).strip().lower(),
            smoothing_cfg=config.smoothing,
            reduce_time=val_reduce_time,
            reduction_factor=val_reduction_factor,
            cut_start_seconds=cut_start_seconds,
            force_representation=force_representation,
            f0_lookup=f0_lookup,
            rho=float(getattr(config.model, "rho", 1000.0)),
            D=float(getattr(config.model, "D", 0.1)),
            m_eff=float(_m_eff_from_model_cfg(config.model)),
            k=float(getattr(config.model, "k", 1218.0)),
            preserve_prefix_for_history=(val_history_context > 0),
            min_history_context=val_history_context,
        )
        if val_dt != dt:
            raise ValueError(f"Validation data dt={val_dt} does not match training dt={dt}.")
        if val_trajs is None:
            val_trajs = []
        # Put `data.npz` first so rollout validation uses it by default.
        val_trajs = [val_traj] + list(val_trajs)

    batch_size = int(training_cfg.batch_size)
    epochs = int(training_cfg.epochs)
    max_grad_norm = float(training_cfg.max_grad_norm)

    d = int(train_trajs[0]["x"].shape[-1])
    m = _as_diag_param(vp.get("m", _m_eff_from_model_cfg(config.model)), d, device, "m")
    c = _as_diag_param(vp.get("c", getattr(config.model, "damping_c", 1e-4)), d, device, "c")
    k = _as_diag_param(vp.get("k", getattr(config.model, "k", 1218.0)), d, device, "k")

    input_dim = 2 * d + 1
    output_dim = d
    model = _build_force_model(config, input_dim=input_dim, output_dim=output_dim)
    model = model.to(device)

    use_input_scaling = bool(vp.get("use_input_scaling", False))
    if use_input_scaling:
        D_val = float(getattr(config.model, "D", 1.0))
        x_scale = D_val if np.isfinite(D_val) and D_val != 0.0 else 1.0
        # Typical velocity scale: omega * D, with omega = sqrt(k/m).
        omega = torch.sqrt(torch.clamp(k / m, min=1e-12))
        v_scale = omega * float(x_scale)
        # Reduce velocity is dimensionless; scale it to O(1) for the force network.
        ur_scale = float(vp.get("ur_scale", 10.0))
        # Typical force scale: k * D (unless output is force coefficient).
        f_scale = 1.0 if use_force_coeff else k * float(x_scale)
        model = ScaledForceWrapper(
            model,
            d=d,
            x_scale=x_scale,
            v_scale=v_scale,
            ur_scale=ur_scale,
            f_scale=f_scale,
        )
        print(
            f"VPINN scaling enabled: x/D (D={x_scale:g}), v/(sqrt(k/m)D), U_r/{ur_scale:g}, "
            f"f_scale=kD."
        )

    model = maybe_compile_model(model, bool(compile_cfg.use_compile), str(compile_cfg.compile_mode))

    gradnorm_balancer: Optional[GradNormBalancer] = None
    gradnorm_last_force = None
    gradnorm_last_weak = None
    gradnorm_last_rollout = None
    use_rollout_loss = rollout_force_weight > 0.0 and rollout_force_steps > 0
    if use_rollout_loss and rollout_force_batch_size > 0:
        print(
            "VPINN rollout loss uses sub-batches: "
            f"rollout_force_batch_size={rollout_force_batch_size} (0 means full batch)."
        )
    if use_gradnorm:
        gradnorm_names: list[str] = []
        if use_force_loss:
            gradnorm_names.append("force")
        if use_weak_loss:
            gradnorm_names.append("weak")
        if use_rollout_loss:
            gradnorm_names.append("rollout")
        if len(gradnorm_names) >= 2:
            gradnorm_balancer = GradNormBalancer(
                model,
                gradnorm_names,
                alpha=gradnorm_alpha,
                eps=gradnorm_eps,
                min_weight=gradnorm_min_weight,
                max_weight=gradnorm_max_weight,
            )
        else:
            print("vpinn.use_gradnorm is True but fewer than two losses are enabled; skipping GradNorm.")

    opt, lr_scheduler = setup_optimizer_and_scheduler(
        model,
        optim_cfg=optim_cfg,
        scheduler_cfg=optim_cfg.scheduler,
        epochs=epochs,
    )

    amp_enabled, amp_dtype, scaler = setup_amp(
        device, use_amp=bool(precision_cfg.use_amp), amp_dtype=str(precision_cfg.amp_dtype)
    )

    w, wdot, alpha = _test_functions(window_M, dt, num_poly=num_poly, num_sine=num_sine)
    w = w.to(device)
    wdot = wdot.to(device)
    alpha = alpha.to(device)

    if per_traj_norm != "none":
        _apply_per_traj_scale(
            train_trajs,
            mode=per_traj_norm,
            dt=dt,
            m=m,
            c=c,
            k=k,
            w=w,
            wdot=wdot,
            alpha=alpha,
            window_M=window_M,
            stride=stride,
            eps=per_traj_norm_eps,
        )
        if val_trajs:
            _apply_per_traj_scale(
                val_trajs,
                mode=per_traj_norm,
                dt=dt,
                m=m,
                c=c,
                k=k,
                w=w,
                wdot=wdot,
                alpha=alpha,
                window_M=window_M,
                stride=stride,
                eps=per_traj_norm_eps,
            )

    history_context = _tcn_history_len(model) if _is_tcn_force_model(model) else 0
    if history_context > 0:
        print(
            f"VPINN TCN context enabled: each loaded window has M+T+1 samples "
            f"(M={window_M}, T={history_context})."
        )
    return_scale = per_traj_norm != "none"
    train_dataset = WindowDataset(
        train_trajs,
        window_intervals=window_M,
        stride=stride,
        history_context=history_context,
        return_scale=return_scale,
    )
    val_dataset = (
        WindowDataset(
            val_trajs,
            window_intervals=window_M,
            stride=stride,
            history_context=history_context,
            return_scale=return_scale,
        )
        if val_trajs
        else None
    )
    if len(train_dataset) == 0:
        raise ValueError("No windows available for training. Reduce vpinn.window_M or check data lengths.")

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

    log_every = int(getattr(monitoring_cfg, "log_every_epochs", 1))
    print_every = int(getattr(monitoring_cfg, "print_every_epochs", 1))
    validate_every = int(getattr(monitoring_cfg, "validate_every_epochs", 1))
    rollout_every = int(getattr(monitoring_cfg, "rollout_every_epochs", 0))
    rollout_max_trajs = int(getattr(monitoring_cfg, "rollout_max_trajectories", 1))
    cycle_validation_rollout = bool(getattr(monitoring_cfg, "cycle_validation_rollout", False))
    final_rollout_all_validation = bool(getattr(monitoring_cfg, "final_rollout_all_validation", False))
    async_validation = bool(getattr(monitoring_cfg, "async_validation", False))
    async_device = str(getattr(monitoring_cfg, "async_validation_device", "cpu"))
    async_num_workers = int(getattr(monitoring_cfg, "async_validation_num_workers", 0))
    async_num_threads = int(getattr(monitoring_cfg, "async_validation_num_threads", 4))
    async_max_concurrent = int(getattr(monitoring_cfg, "async_validation_max_concurrent", 1))
    async_do_losses = bool(getattr(monitoring_cfg, "async_validation_do_losses", True))
    async_do_rollout = bool(getattr(monitoring_cfg, "async_validation_do_rollout", True))

    writer, run_name = setup_writer(
        config.logging.run_dir_root,
        config_name,
        run_name_override=getattr(config.logging, "run_name", None),
        append_timestamp=bool(getattr(config.logging, "append_timestamp", True)),
    )
    async_processes: list[subprocess.Popen] = []
    async_dir = Path(writer.log_dir) / "async_validation"
    if async_validation:
        async_dir.mkdir(parents=True, exist_ok=True)

    use_lr_scheduler = bool(optim_cfg.use_lr_scheduler)
    base_lr = float(optim_cfg.lr)
    middle_time_plot = resolve_middle_time_plot(config.data, vp, method_name="vpinn")
    D_val = float(getattr(config.model, "D", 1.0))

    for epoch in range(epochs):
        model.train()
        if use_lr_scheduler:
            for group in opt.param_groups:
                group["lr"] = lr_scheduler.get_lr(epoch)

        loss_f_sum = torch.zeros((), device=device)
        loss_w_sum = torch.zeros((), device=device)
        loss_roll_sum = torch.zeros((), device=device)
        loss_sum = torch.zeros((), device=device)
        roll_count = 0
        grad_norm_sum = torch.zeros((), device=device)
        gradnorm_force_w_sum = torch.zeros((), device=device)
        gradnorm_weak_w_sum = torch.zeros((), device=device)
        gradnorm_rollout_w_sum = torch.zeros((), device=device)
        gradnorm_count = 0
        batches = 0

        expect_scale = per_traj_norm != "none"
        expect_f0 = use_force_coeff
        for step, batch in enumerate(train_loader):
            if len(batch) == 4:
                x_win, v_win, f_meas, ur_win = batch
                scale = None
                f0 = None
            elif len(batch) == 5:
                x_win, v_win, f_meas, ur_win, extra = batch
                if expect_scale and not expect_f0:
                    scale = extra
                    f0 = None
                elif expect_f0 and not expect_scale:
                    scale = None
                    f0 = extra
                else:
                    raise ValueError("Unexpected batch format (missing scale or f0).")
            elif len(batch) == 6:
                x_win, v_win, f_meas, ur_win, scale, f0 = batch
            else:
                raise ValueError("Unexpected batch format from dataloader.")
            x_win = x_win.to(device, non_blocking=non_blocking)
            v_win = v_win.to(device, non_blocking=non_blocking)
            f_meas = f_meas.to(device, non_blocking=non_blocking)
            ur_win = ur_win.to(device, non_blocking=non_blocking)
            if scale is not None:
                scale = scale.to(device, non_blocking=non_blocking).view(-1)
            if f0 is not None:
                f0 = f0.to(device, non_blocking=non_blocking)

            M1_target = int(window_M) + 1
            if int(x_win.shape[1]) < M1_target:
                raise ValueError(
                    f"Window length {int(x_win.shape[1])} is shorter than target length {M1_target}."
                )
            start_idx = int(x_win.shape[1]) - M1_target
            x_eval = x_win[:, start_idx:, :]
            v_eval = v_win[:, start_idx:, :]
            f_eval = f_meas[:, start_idx:, :]
            ur_eval = ur_win[:, start_idx:, :]
            f0_eval = f0[:, start_idx:, :] if f0 is not None else None
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                f_pred = _vpinn_force_sequence(model, x_win, v_win, ur_win)
                f_pred_eval = f_pred[:, start_idx:, :]
                per_loss_f = torch.mean((f_pred_eval - f_eval) ** 2, dim=(1, 2))
                if scale is not None:
                    per_loss_f = per_loss_f / (scale * scale + float(per_traj_norm_eps))
                loss_f = torch.mean(per_loss_f)
                if use_weak_loss:
                    R = _weak_residual(
                        x=x_eval,
                        v=v_eval,
                        f_pred=f_pred_eval,
                        m=m,
                        c=c,
                        k=k,
                        dt=dt,
                        w=w,
                        wdot=wdot,
                        alpha=alpha,
                        f0=f0_eval,
                    )
                    per_loss_w = torch.mean(R.pow(2), dim=(1, 2))
                    if scale is not None:
                        per_loss_w = per_loss_w / (scale * scale + float(per_traj_norm_eps))
                    loss_w = torch.mean(per_loss_w)
                else:
                    loss_w = loss_f.new_tensor(0.0)

                loss_roll = loss_f.new_tensor(0.0)
                roll_computed = False
                if rollout_force_weight > 0.0 and rollout_force_steps > 0:
                    if (step % max(1, int(rollout_force_every))) == 0:
                        steps_k = min(int(rollout_force_steps), int(M1_target) - 1)
                        if steps_k > 0:
                            x_roll = x_eval
                            v_roll = v_eval
                            ur_roll = ur_eval
                            f_roll_true = f_eval
                            f0_roll = f0_eval
                            scale_roll = scale

                            if int(rollout_force_batch_size) > 0 and int(x_eval.shape[0]) > int(rollout_force_batch_size):
                                sel = torch.randperm(int(x_eval.shape[0]), device=x_eval.device)[
                                    : int(rollout_force_batch_size)
                                ]
                                x_roll = x_eval.index_select(0, sel)
                                v_roll = v_eval.index_select(0, sel)
                                ur_roll = ur_eval.index_select(0, sel)
                                f_roll_true = f_eval.index_select(0, sel)
                                if f0_eval is not None:
                                    f0_roll = f0_eval.index_select(0, sel)
                                if scale is not None:
                                    scale_roll = scale.index_select(0, sel)

                            f0_step = f0_roll[:, 0, :] if f0_roll is not None else None
                            _x_seq, _v_seq, f_seq = rollout_rk4(
                                model=model,
                                x0=x_roll[:, 0, :],
                                v0=v_roll[:, 0, :],
                                ur0=ur_roll[:, 0, :],
                                steps=steps_k,
                                dt=dt,
                                substeps=rollout_train_substeps,
                                m=m,
                                c=c,
                                k=k,
                                f0=f0_step,
                            )
                            f_roll = f_seq[:, : steps_k + 1, :]
                            f_true = f_roll_true[:, : steps_k + 1, :]
                            per_roll = torch.mean((f_roll - f_true) ** 2, dim=(1, 2))
                            if scale_roll is not None:
                                per_roll = per_roll / (scale_roll * scale_roll + float(per_traj_norm_eps))
                            loss_roll = torch.mean(per_roll)
                            roll_computed = True

                wf_eff = float(wf) if use_force_loss else 0.0
                ww_eff = float(ww) if use_weak_loss else 0.0
                if gradnorm_balancer is not None:
                    need_init = gradnorm_last_force is None and use_force_loss
                    need_init = need_init or (gradnorm_last_weak is None and use_weak_loss)
                    need_init = need_init or (gradnorm_last_rollout is None and use_rollout_loss)
                    do_update = (step % gradnorm_update_every_steps) == 0 or need_init
                    if use_rollout_loss and not roll_computed:
                        do_update = False
                    if do_update:
                        losses: dict[str, torch.Tensor] = {}
                        if use_force_loss:
                            losses["force"] = loss_f.float()
                        if use_weak_loss:
                            losses["weak"] = loss_w.float()
                        if use_rollout_loss:
                            losses["rollout"] = loss_roll.float()
                        weights = gradnorm_balancer.update(losses)
                        if "force" in weights:
                            gradnorm_last_force = weights["force"]
                        if "weak" in weights:
                            gradnorm_last_weak = weights["weak"]
                        if "rollout" in weights:
                            gradnorm_last_rollout = weights["rollout"]
                    w_force = gradnorm_last_force if use_force_loss else loss_f.new_tensor(1.0)
                    w_weak = gradnorm_last_weak if use_weak_loss else loss_f.new_tensor(1.0)
                    w_roll = gradnorm_last_rollout if use_rollout_loss else loss_f.new_tensor(1.0)
                    if use_force_loss and w_force is not None:
                        gradnorm_force_w_sum = gradnorm_force_w_sum + w_force.detach()
                    if use_weak_loss and w_weak is not None:
                        gradnorm_weak_w_sum = gradnorm_weak_w_sum + w_weak.detach()
                    if use_rollout_loss and w_roll is not None:
                        gradnorm_rollout_w_sum = gradnorm_rollout_w_sum + w_roll.detach()
                    gradnorm_count += 1
                    loss = (
                        wf_eff * w_force * loss_f
                        + ww_eff * w_weak * loss_w
                        + float(rollout_force_weight) * w_roll * loss_roll
                    )
                else:
                    loss = wf_eff * loss_f + ww_eff * loss_w + float(rollout_force_weight) * loss_roll

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
            if rollout_force_weight > 0.0 and rollout_force_steps > 0 and (step % max(1, int(rollout_force_every))) == 0:
                loss_roll_sum = loss_roll_sum + loss_roll.detach()
                roll_count += 1
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
        if roll_count > 0:
            metrics["loss_rollout_force"] = float((loss_roll_sum / float(roll_count)).detach().cpu())
        if gradnorm_count > 0:
            if use_force_loss:
                metrics["gradnorm_weight_data"] = float((gradnorm_force_w_sum / float(gradnorm_count)).detach().cpu())
            if use_weak_loss:
                metrics["gradnorm_weight_physics"] = float((gradnorm_weak_w_sum / float(gradnorm_count)).detach().cpu())
            if use_rollout_loss:
                metrics["gradnorm_weight_rollout"] = float(
                    (gradnorm_rollout_w_sum / float(gradnorm_count)).detach().cpu()
                )

        if (epoch % max(1, log_every)) == 0 or epoch == (epochs - 1):
            for k_name, v_value in metrics.items():
                writer.add_scalar(f"train/{k_name}", v_value, epoch)

        if (epoch % max(1, print_every)) == 0 or epoch == (epochs - 1):
            print(
                f"Epoch {epoch}: loss={metrics['loss_total']:.4e}, "
                f"Ldata={metrics['loss_data']:.4e}, Lphys={metrics['loss_physics']:.4e}, lr={metrics['lr']:.3e}"
            )

        should_validate = (
            val_loader is not None
            and validate_every > 0
            and ((epoch % validate_every) == 0 or epoch == (epochs - 1))
        )
        should_rollout = rollout_every > 0 and ((epoch % rollout_every) == 0 or epoch == (epochs - 1))
        validation_timer_start: float | None = None
        if (should_validate or should_rollout) and not async_validation:
            validation_timer_start = time.perf_counter()

        if async_validation and (should_validate or should_rollout) and (async_do_losses or async_do_rollout):
            async_processes = _prune_async_processes(async_processes)
            state_source: nn.Module = model
            if hasattr(model, "_orig_mod"):
                state_source = getattr(model, "_orig_mod")
            async_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = async_dir / f"epoch_{epoch + 1:06d}.pt"
            torch.save(
                {
                    "model_state": state_source.state_dict(),
                    "config": asdict(config),
                    "run_name": run_name,
                    "dt": dt,
                    "method": "vpinn",
                },
                ckpt_path,
            )
            async_processes = _launch_async_validation(
                processes=async_processes,
                max_concurrent=async_max_concurrent,
                checkpoint_path=ckpt_path,
                epoch=epoch,
                writer=writer,
                async_device=async_device,
                async_num_workers=async_num_workers,
                async_num_threads=async_num_threads,
                rollout_every_epochs=rollout_every,
                cycle_validation_rollout=cycle_validation_rollout,
                do_losses=async_do_losses and should_validate,
                do_rollout=async_do_rollout and should_rollout,
            )
        elif should_validate and not async_validation:
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
                per_traj_norm_eps=per_traj_norm_eps,
                expect_scale=return_scale,
                expect_f0=use_force_coeff,
            )
            for k_name, v_value in val_metrics.items():
                writer.add_scalar(f"val/{k_name}", v_value, epoch)
            force_map = _force_mapping_nrmse_over_trajs(model=model, val_trajs=val_trajs or [], device=device)
            if force_map is not None:
                for k_name, v_value in force_map.items():
                    writer.add_scalar(f"val/{k_name}", v_value, epoch)
            loss_by_ur = _per_ur_loss_map_vpinn(
                model=model,
                loader=val_loader,
                device=device,
                non_blocking=non_blocking,
                dt=dt,
                m=m,
                c=c,
                k=k,
                w=w,
                wdot=wdot,
                alpha=alpha,
                use_force_loss=use_force_loss,
                use_weak_loss=use_weak_loss,
                rollout_force_steps=rollout_force_steps if use_rollout_loss else 0,
                rollout_substeps=rollout_val_substeps,
                expect_scale=return_scale,
                expect_f0=use_force_coeff,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                per_traj_norm_eps=per_traj_norm_eps,
            )
            log_loss_vs_ur(
                writer,
                epoch,
                loss_by_ur,
                tag="val/loss_vs_ur",
                title="Validation loss vs U_r",
            )

        if should_rollout and not async_validation:
            candidates = val_trajs if val_trajs else train_trajs
            if not candidates:
                continue
            sampled_indices = _sample_one_traj_per_ur(candidates, seed=int(epoch) + 1)
            if not sampled_indices:
                continue
            metrics_sum: dict[str, float] = {}
            metrics_count: dict[str, int] = {}
            for idx in sampled_indices:
                metrics = _log_rollout_validation(
                    writer=writer,
                    epoch=epoch,
                    model=model,
                    traj=candidates[idx],
                    dt=dt,
                    m=m,
                    c=c,
                    k=k,
                    D=D_val,
                    middle_time_plot=middle_time_plot,
                    device=device,
                    rollout_substeps=rollout_val_substeps,
                    log_extra_metrics=log_extra_validation_metrics,
                    include_disp_nrmse=rollout_include_disp_nrmse,
                    include_force_nrmse=rollout_include_force_nrmse,
                    log_metrics=False,
                    log_plots=False,
                )
                for name, value in metrics.items():
                    if not np.isfinite(value):
                        continue
                    metrics_sum[name] = metrics_sum.get(name, 0.0) + float(value)
                    metrics_count[name] = metrics_count.get(name, 0) + 1
            for name, total in metrics_sum.items():
                denom = max(1, int(metrics_count.get(name, 0)))
                writer.add_scalar(f"val/{name}", total / float(denom), epoch)
            if cycle_validation_rollout:
                step = max(0, (epoch + 1) // max(1, int(rollout_every)) - 1)
                plot_idx = sampled_indices[step % len(sampled_indices)]
            else:
                plot_idx = sampled_indices[0]
            _log_rollout_validation(
                writer=writer,
                epoch=epoch,
                model=model,
                traj=candidates[plot_idx],
                dt=dt,
                m=m,
                c=c,
                k=k,
                D=D_val,
                middle_time_plot=middle_time_plot,
                device=device,
                rollout_substeps=rollout_val_substeps,
                log_extra_metrics=log_extra_validation_metrics,
                include_disp_nrmse=rollout_include_disp_nrmse,
                include_force_nrmse=rollout_include_force_nrmse,
                log_metrics=False,
                log_plots=True,
            )
        if validation_timer_start is not None:
            validation_elapsed = time.perf_counter() - validation_timer_start
            writer.add_scalar("val/validation_wall_time_s", float(validation_elapsed), epoch)
            print(f"Validation epoch {epoch}: total wall time {validation_elapsed:.2f}s")

    if final_rollout_all_validation and val_trajs:
        print("Final validation rollout (all trajectories) started.")
        final_start = time.perf_counter()
        metrics_sum: dict[str, float] = {}
        metrics_count: dict[str, int] = {}
        used = 0
        ur_values: list[float] = []
        metrics_list: list[dict[str, float]] = []
        selected_trajs: list[dict[str, Any]] = []
        seen_ur: set[float] = set()
        for traj in val_trajs:
            ur_val = float(traj["ur"][0, 0].detach().cpu().item())
            ur_key = round(ur_val, 6)
            if ur_key in seen_ur:
                continue
            seen_ur.add(ur_key)
            selected_trajs.append(traj)
        for idx, traj in enumerate(selected_trajs):
            metrics = _log_rollout_validation(
                writer=writer,
                epoch=max(0, epochs - 1),
                model=model,
                traj=traj,
                dt=dt,
                m=m,
                c=c,
                k=k,
                D=D_val,
                middle_time_plot=middle_time_plot,
                device=device,
                rollout_substeps=rollout_val_substeps,
                tag_prefix="final_val/rollout",
                step=idx,
                log_extra_metrics=log_extra_validation_metrics,
                include_disp_nrmse=rollout_include_disp_nrmse,
                include_force_nrmse=rollout_include_force_nrmse,
                log_metrics=False,
                title_suffix=f" [final {idx+1}/{len(selected_trajs)}]",
            )
            if metrics:
                used += 1
                ur_values.append(float(traj["ur"][0, 0].detach().cpu().item()))
                metrics_list.append(metrics)
            for name, value in metrics.items():
                if not np.isfinite(value):
                    continue
                metrics_sum[name] = metrics_sum.get(name, 0.0) + float(value)
                metrics_count[name] = metrics_count.get(name, 0) + 1
        avg_metrics = {
            name: metrics_sum[name] / float(metrics_count[name])
            for name in metrics_sum
            if metrics_count.get(name, 0) > 0
        }
        if avg_metrics and used > 0:
            summary_lines = [f"Final rollout over {used} validation trajectories (unique U_r):"]
            for name in sorted(avg_metrics):
                summary_lines.append(f"{name}: {avg_metrics[name]:.6f}")
                writer.add_scalar(f"final_val/avg/{name}", avg_metrics[name], epochs)
            writer.add_text("final_val/summary", "\n".join(summary_lines), epochs)
        if ur_values and metrics_list:
            log_final_rollout_errors_vs_ur(writer, ur_values, metrics_list, epochs)
        elapsed = time.perf_counter() - final_start
        print(f"Final validation rollout finished in {elapsed:.2f}s.")

    if val_loader is not None:
        final_loss_by_ur = _per_ur_loss_map_vpinn(
            model=model,
            loader=val_loader,
            device=device,
            non_blocking=non_blocking,
            dt=dt,
            m=m,
            c=c,
            k=k,
            w=w,
            wdot=wdot,
            alpha=alpha,
            use_force_loss=use_force_loss,
            use_weak_loss=use_weak_loss,
            rollout_force_steps=rollout_force_steps if use_rollout_loss else 0,
            rollout_substeps=rollout_val_substeps,
            expect_scale=return_scale,
            expect_f0=use_force_coeff,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            per_traj_norm_eps=per_traj_norm_eps,
        )
        if final_loss_by_ur:
            log_loss_vs_ur(
                writer,
                epochs,
                final_loss_by_ur,
                tag="final_val/loss_vs_ur",
                title="Final validation loss vs U_r",
            )
            writer.add_text(
                "final_val/loss_vs_ur_text",
                format_loss_vs_ur_text(final_loss_by_ur, title="Final validation loss vs U_r"),
                epochs,
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
