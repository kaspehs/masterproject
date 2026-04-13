from __future__ import annotations

import json
import math
import os
import shutil
import time
import subprocess
import sys
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils
from torch.utils.data import ConcatDataset, DataLoader, Dataset, TensorDataset

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
    FORCE_MAPPING_NRMSE_KEY,
    GradNormBalancer,
    MEAN_DISP_AMP_REL_ERROR_KEY,
    Residual,
    build_phase_plot_grid,
    compute_validation_metrics,
    create_window_mask,
    create_zoom_mask,
    dominant_frequency,
    format_loss_vs_ur_text,
    log_area_normalized_rollout_spectra,
    log_final_rollout_errors_vs_ur,
    log_loss_vs_ur,
    log_correction_on_data_plot,
    log_displacement_plots,
    log_force_plots,
    log_output_distribution_vs_ur,
    log_signed_phase_output_plot,
    lookup_ur_bin_state_scale_tensor,
    mean_displacement_amplitude,
    nearest_phase_series_values,
    resolve_middle_time_plot,
    relative_error,
    resample_uniform_series,
    load_td_correction_trajectories,
    resolve_td_correction_mode,
    resolve_td_correction_params,
    resolve_td_phase_input_source,
    sample_indices_per_ur,
    sample_one_index_per_ur,
    spectral_relative_error,
    structural_step_constant_force_torch,
    td_bounded_delta_fhat_torch,
    td_hidden_inputs_from_context_torch,
    td_baseline_step_torch,
    td_correction_mode_flags,
)
from architectures import FourierFeatures, ODEPirateNet


def _vpinn_ur_bin_id(value: float, ur_bin_size: float) -> int:
    return int(np.rint(float(value) / float(ur_bin_size)))


def _build_vpinn_ur_bin_state_scale_info(
    trajs: list[dict[str, np.ndarray]],
    *,
    ur_bin_size: float,
    eps: float = 1e-6,
) -> dict[str, Any]:
    stats_by_bin: dict[int, dict[str, Any]] = {}
    global_count = 0
    global_sum = np.zeros(2, dtype=np.float64)
    global_sumsq = np.zeros(2, dtype=np.float64)

    def _finalize(count: int, sum_vec: np.ndarray, sumsq_vec: np.ndarray) -> np.ndarray:
        denom = float(max(int(count), 1))
        mean = sum_vec / denom
        var = np.maximum(sumsq_vec / denom - mean * mean, 0.0)
        return np.sqrt(var)

    for traj in trajs:
        x = np.asarray(traj["y"], dtype=np.float64).reshape(-1)
        v = np.asarray(traj["dy"], dtype=np.float64).reshape(-1)
        ur = np.asarray(traj["ur"], dtype=np.float64).reshape(-1)
        n = min(int(x.shape[0]), int(v.shape[0]))
        if n < 1:
            continue
        if ur.size == 1:
            ur = np.repeat(ur, n)
        else:
            ur = ur[:n]
        states = np.stack([x[:n], v[:n]], axis=1)
        for ur_val, state_vec in zip(ur, states):
            key = _vpinn_ur_bin_id(float(ur_val), ur_bin_size)
            stat = stats_by_bin.setdefault(
                key,
                {"count": 0, "sum": np.zeros(2, dtype=np.float64), "sumsq": np.zeros(2, dtype=np.float64)},
            )
            vec = np.asarray(state_vec, dtype=np.float64)
            stat["count"] += 1
            stat["sum"] += vec
            stat["sumsq"] += vec * vec
            global_count += 1
            global_sum += vec
            global_sumsq += vec * vec

    global_scale = _finalize(global_count, global_sum, global_sumsq)
    global_scale = np.maximum(global_scale, float(eps))
    by_bin: dict[str, list[float]] = {}
    for key, stat in stats_by_bin.items():
        scale = _finalize(int(stat["count"]), stat["sum"], stat["sumsq"])
        if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
            scale = global_scale.copy()
        scale = np.maximum(scale, float(eps))
        by_bin[str(int(key))] = [float(scale[0]), float(scale[1])]
    return {
        "global": [float(global_scale[0]), float(global_scale[1])],
        "by_bin": by_bin,
    }


def _vpinn_lookup_state_scale(
    ur_values: torch.Tensor,
    *,
    scale_info: dict[str, Any] | None,
    ur_bin_size: float,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return lookup_ur_bin_state_scale_tensor(
        ur_values,
        scale_info=scale_info,
        ur_bin_size=ur_bin_size,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
    )


def _vpinn_scale_weak_residual(
    residual: torch.Tensor,
    *,
    ur_values: torch.Tensor,
    mass: torch.Tensor,
    scale_info: dict[str, Any] | None,
    ur_bin_size: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if scale_info is None:
        return residual, None
    batch_size = int(residual.shape[0])
    state_scale = _vpinn_lookup_state_scale(
        ur_values,
        scale_info=scale_info,
        ur_bin_size=ur_bin_size,
        batch_size=batch_size,
        device=residual.device,
        dtype=residual.dtype,
    )
    p_scale = torch.clamp(
        mass.to(device=residual.device, dtype=residual.dtype) * state_scale[..., 1:2],
        min=1e-12,
    )
    return residual / p_scale.unsqueeze(1), p_scale


def _vpinn_rollout_state_loss(
    *,
    x_pred: torch.Tensor,
    v_pred: torch.Tensor,
    x_true: torch.Tensor,
    v_true: torch.Tensor,
    ur_values: torch.Tensor,
    scale_info: dict[str, Any] | None,
    ur_bin_size: float,
) -> torch.Tensor:
    per = _vpinn_rollout_state_loss_per_sample(
        x_pred=x_pred,
        v_pred=v_pred,
        x_true=x_true,
        v_true=v_true,
        ur_values=ur_values,
        scale_info=scale_info,
        ur_bin_size=ur_bin_size,
    )
    return torch.mean(per)


def _vpinn_rollout_state_loss_per_sample(
    *,
    x_pred: torch.Tensor,
    v_pred: torch.Tensor,
    x_true: torch.Tensor,
    v_true: torch.Tensor,
    ur_values: torch.Tensor,
    scale_info: dict[str, Any] | None,
    ur_bin_size: float,
) -> torch.Tensor:
    if scale_info is None:
        return torch.mean((x_pred - x_true) ** 2 + (v_pred - v_true) ** 2, dim=(1, 2))
    batch_size = int(x_pred.shape[0])
    state_scale = _vpinn_lookup_state_scale(
        ur_values,
        scale_info=scale_info,
        ur_bin_size=ur_bin_size,
        batch_size=batch_size,
        device=x_pred.device,
        dtype=x_pred.dtype,
    )
    x_scale = torch.clamp(state_scale[..., 0:1], min=1e-12).unsqueeze(1)
    v_scale = torch.clamp(state_scale[..., 1:2], min=1e-12).unsqueeze(1)
    return torch.mean(((x_pred - x_true) / x_scale) ** 2 + ((v_pred - v_true) / v_scale) ** 2, dim=(1, 2))


def _scheduled_rollout_det_steps(
    *,
    epoch: int,
    base_steps: int,
    final_steps: int,
    warmup_epochs: int,
) -> int:
    target_steps = base_steps if final_steps <= 0 else final_steps
    if target_steps <= base_steps:
        return int(base_steps)
    if warmup_epochs <= 0:
        return int(target_steps)
    progress = min(1.0, float(max(0, epoch)) / float(max(1, warmup_epochs)))
    steps = int(np.floor(float(base_steps) + progress * float(target_steps - base_steps)))
    return max(int(base_steps), min(int(target_steps), steps))


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


class FourierForceWrapper(nn.Module):
    def __init__(self, base: nn.Module, *, input_dim: int, fourier_features: int, sigma: float) -> None:
        super().__init__()
        self.ff = FourierFeatures(int(input_dim), int(fourier_features), float(sigma), torch.float32)
        self.base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.base(self.ff(x))


class OddSymmetricForceWrapper(nn.Module):
    """
    Enforce odd symmetry in the force mean by construction:
        f(s) = 0.5 * (g(s) - g(T(s)))
    where T flips the state coordinates [x, v] -> [-x, -v] and keeps U_r fixed.

    If the wrapped model outputs extra non-force channels (e.g. probabilistic sigma
    parameters), those channels are averaged evenly instead of antisymmetrized.
    """

    def __init__(self, base: nn.Module, *, input_dim: int, state_dim: int, mean_output_dim: int) -> None:
        super().__init__()
        self.base = base
        self.input_dim = int(input_dim)
        self.state_dim = int(state_dim)
        self.mean_output_dim = int(mean_output_dim)
        if self.state_dim < 1:
            raise ValueError("OddSymmetricForceWrapper requires state_dim >= 1.")
        if self.input_dim != 2 * self.state_dim + 1:
            raise ValueError(
                "OddSymmetricForceWrapper expects inputs of the form [x, v, U_r] with no extra channels."
            )
        if self.mean_output_dim < 1:
            raise ValueError("OddSymmetricForceWrapper requires mean_output_dim >= 1.")

    def forward(self, s: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if int(s.shape[-1]) != self.input_dim:
            raise ValueError(f"Expected input dim {self.input_dim}, got {int(s.shape[-1])}.")
        d = self.state_dim
        s_flip = torch.cat([-(s[..., :d]), -(s[..., d : 2 * d]), s[..., 2 * d :]], dim=-1)
        out_pos = self.base(s)
        out_neg = self.base(s_flip)
        if int(out_pos.shape[-1]) < self.mean_output_dim:
            raise ValueError(
                f"Wrapped model output dim {int(out_pos.shape[-1])} is smaller than mean_output_dim={self.mean_output_dim}."
            )
        mean = 0.5 * (out_pos[..., : self.mean_output_dim] - out_neg[..., : self.mean_output_dim])
        if int(out_pos.shape[-1]) == self.mean_output_dim:
            return mean
        rest = 0.5 * (out_pos[..., self.mean_output_dim :] + out_neg[..., self.mean_output_dim :])
        return torch.cat([mean, rest], dim=-1)


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


def _last_linear(module: nn.Module) -> nn.Linear | None:
    last: nn.Linear | None = None
    for sub in module.modules():
        if isinstance(sub, nn.Linear):
            last = sub
    return last


def _softplus_inverse_scalar(value: float) -> float:
    val = float(value)
    if val <= 1.0e-12:
        return -20.0
    if val > 20.0:
        return val
    return float(math.log(math.expm1(val)))


def _resolve_corr_init_settings(vp_cfg: dict[str, Any], model_cfg: Any) -> tuple[str, float]:
    mode = str(vp_cfg.get("corr_init_mode", getattr(model_cfg, "corr_init_mode", "standard"))).strip().lower()
    if mode not in {"zero", "tiny", "standard"}:
        raise ValueError("corr_init_mode must be one of: zero, tiny, standard.")
    tiny_std = float(vp_cfg.get("corr_init_tiny_std", getattr(model_cfg, "corr_init_tiny_std", 1.0e-4)))
    if not np.isfinite(tiny_std) or tiny_std <= 0.0:
        raise ValueError("corr_init_tiny_std must be finite and > 0.")
    return mode, tiny_std


def _apply_corr_head_init(
    model: nn.Module,
    *,
    mode: str,
    tiny_std: float,
    probabilistic: bool,
    sigma_min: float,
) -> None:
    if mode == "standard":
        return
    last = _last_linear(model)
    if last is None:
        return
    out_dim = int(last.weight.shape[0])
    mean_dim = out_dim if not probabilistic else out_dim // 2
    sigma_dim = 0 if not probabilistic else out_dim - mean_dim
    if probabilistic and sigma_dim <= 0:
        raise ValueError("Probabilistic correction init expects an output layer with mean and sigma channels.")
    target_sigma = float(sigma_min) if mode == "zero" else max(float(sigma_min), float(tiny_std))
    target_excess = max(0.0, target_sigma - float(sigma_min))
    sigma_bias = _softplus_inverse_scalar(target_excess)
    with torch.no_grad():
        if mode == "zero":
            nn.init.zeros_(last.weight[:mean_dim])
        elif mode == "tiny":
            nn.init.normal_(last.weight[:mean_dim], mean=0.0, std=float(tiny_std))
        else:
            raise ValueError("corr_init_mode must be one of: zero, tiny, standard.")
        nn.init.zeros_(last.bias[:mean_dim])
        if probabilistic:
            if mode == "zero":
                nn.init.zeros_(last.weight[mean_dim:])
            else:
                nn.init.normal_(last.weight[mean_dim:], mean=0.0, std=float(tiny_std))
            nn.init.constant_(last.bias[mean_dim:], sigma_bias)


def _build_force_model(config: Config, *, input_dim: int, output_dim: int, mean_output_dim: int | None = None) -> nn.Module:
    if config.architecture is None:
        raise ValueError("VPINN requires a shared 'architecture:' block.")
    arch = config.architecture
    net_type = str(getattr(arch, "force_net_type", "residual")).strip().lower()
    hard_force_symmetry = bool(getattr(arch, "hard_force_symmetry", False))
    use_fourier_features = bool(getattr(arch, "use_fourier_features", False))
    fourier_features = int(getattr(arch, "fourier_features", 64))
    fourier_sigma = float(getattr(arch, "fourier_sigma", 1.0))
    resolved_mean_output_dim = int(output_dim if mean_output_dim is None else mean_output_dim)
    if hard_force_symmetry and ((int(input_dim) - 1) % 2 != 0):
        raise ValueError(
            "architecture.hard_force_symmetry only supports VPINN inputs of the form [x, v, U_r]. "
            "Disable use_td_force_input to use this constraint."
        )

    if net_type == "pirate":
        pirate_kwargs = {}
        pirate_kwargs.update(getattr(arch, "pirate_force_kwargs", {}) or {})
        pirate_kwargs.setdefault("depth", 2)
        pirate_kwargs.setdefault("fourier_features", 64)
        pirate_kwargs.setdefault("sigma", 1.0)
        pirate_kwargs.setdefault("activation", "tanh")
        model = ODEPirateNet(
            input_size=int(input_dim),
            output_size=int(output_dim),
            **pirate_kwargs,
        )
        if hard_force_symmetry:
            return OddSymmetricForceWrapper(
                model,
                input_dim=int(input_dim),
                state_dim=(int(input_dim) - 1) // 2,
                mean_output_dim=resolved_mean_output_dim,
            )
        return model

    if net_type == "residual":
        cfg = dict(getattr(arch, "residual_kwargs", {}) or {})
        hidden = int(cfg.get("hidden", 128))
        layers = int(cfg.get("layers", 2))
        activation = str(cfg.get("activation", "gelu"))
        if use_fourier_features and fourier_features < 1:
            raise ValueError("architecture.fourier_features must be >= 1 when use_fourier_features is True.")
        model_input_dim = 2 * fourier_features if use_fourier_features else int(input_dim)
        layers_list: list[nn.Module] = [nn.Linear(model_input_dim, hidden)]
        for _ in range(max(1, layers)):
            layers_list.append(Residual(hidden, activation=activation))
        layers_list.append(nn.Linear(hidden, int(output_dim)))
        base = nn.Sequential(*layers_list)
        if use_fourier_features:
            model = FourierForceWrapper(
                base,
                input_dim=int(input_dim),
                fourier_features=fourier_features,
                sigma=fourier_sigma,
            )
        else:
            model = base
        if hard_force_symmetry:
            return OddSymmetricForceWrapper(
                model,
                input_dim=int(input_dim),
                state_dim=(int(input_dim) - 1) // 2,
                mean_output_dim=resolved_mean_output_dim,
            )
        return model

    if net_type == "mlp":
        cfg = dict(getattr(arch, "mlp_kwargs", {}) or {})
        hidden = int(cfg.get("hidden", 128))
        layers = int(cfg.get("layers", 2))
        activation = _activation_from_string(str(cfg.get("activation", "gelu")))
        modules: list[nn.Module] = []
        if use_fourier_features and fourier_features < 1:
            raise ValueError("architecture.fourier_features must be >= 1 when use_fourier_features is True.")
        in_features = 2 * fourier_features if use_fourier_features else int(input_dim)
        for _ in range(max(1, layers)):
            modules.append(nn.Linear(in_features, hidden))
            modules.append(activation)
            in_features = hidden
        modules.append(nn.Linear(in_features, int(output_dim)))
        base = nn.Sequential(*modules)
        if use_fourier_features:
            model = FourierForceWrapper(
                base,
                input_dim=int(input_dim),
                fourier_features=fourier_features,
                sigma=fourier_sigma,
            )
        else:
            model = base
        if hard_force_symmetry:
            return OddSymmetricForceWrapper(
                model,
                input_dim=int(input_dim),
                state_dim=(int(input_dim) - 1) // 2,
                mean_output_dim=resolved_mean_output_dim,
            )
        return model

    raise ValueError("architecture.force_net_type must be one of: residual, mlp, pirate")


def _vpinn_model_uses_td_force_input(model: nn.Module) -> bool:
    base = getattr(model, "_orig_mod", model)
    return bool(getattr(base, "use_td_force_input", False))


def _vpinn_model_uses_phi_input(model: nn.Module) -> bool:
    base = getattr(model, "_orig_mod", model)
    return bool(getattr(base, "use_phi_input", False))


def _vpinn_model_uses_acceleration_input(model: nn.Module) -> bool:
    base = getattr(model, "_orig_mod", model)
    return bool(getattr(base, "use_acceleration_input", False))


def _vpinn_model_phase_input_source(model: nn.Module) -> str:
    base = getattr(model, "_orig_mod", model)
    raw_value = getattr(
        base,
        "phi_input_source",
        (True if bool(getattr(base, "use_phi_input", False)) else False),
    )
    return resolve_td_phase_input_source(raw_value)


def _vpinn_model_uses_sigma_inputs(model: nn.Module) -> bool:
    base = getattr(model, "_orig_mod", model)
    return bool(getattr(base, "use_sigma_inputs", False))


def _vpinn_input_dim(
    *,
    d: int,
    use_td_force_input: bool,
    use_acceleration_input: bool,
    use_phi_input: bool,
    use_sigma_inputs: bool,
) -> int:
    return int(
        2 * d
        + 1
        + (1 if use_td_force_input else 0)
        + (1 if use_acceleration_input else 0)
        + (2 if use_phi_input else 0)
        + (2 if use_sigma_inputs else 0)
    )


def _vpinn_output_dim(
    *,
    mean_active: bool,
    sigma_active: bool,
    fhat_active: bool,
    d: int,
) -> int:
    return int((d if mean_active else 0) + (d if sigma_active else 0) + (d if fhat_active else 0))


def _vpinn_optional_hidden_inputs_from_context(
    model: nn.Module,
    *,
    td_context: torch.Tensor,
    velocity: torch.Tensor,
    structural_mass: torch.Tensor,
    stiffness: torch.Tensor,
    diameter: float,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if not (
        _vpinn_model_uses_phi_input(model)
        or _vpinn_model_uses_sigma_inputs(model)
        or _vpinn_model_uses_acceleration_input(model)
    ):
        return None, None, None
    phi_input, sigma_inputs, acceleration_input = td_hidden_inputs_from_context_torch(
        td_context=td_context,
        structural_mass=structural_mass,
        stiffness=stiffness,
        diameter=diameter,
        velocity=velocity,
        phase_input_source=_vpinn_model_phase_input_source(model),
    )
    return (
        phi_input if _vpinn_model_uses_phi_input(model) else None,
        sigma_inputs if _vpinn_model_uses_sigma_inputs(model) else None,
        acceleration_input if _vpinn_model_uses_acceleration_input(model) else None,
    )


def _vpinn_force(
    model: nn.Module,
    x: torch.Tensor,
    v: torch.Tensor,
    ur: torch.Tensor,
    td_force_input: torch.Tensor | None = None,
    acceleration_input: torch.Tensor | None = None,
    phi_input: torch.Tensor | None = None,
    sigma_inputs: torch.Tensor | None = None,
) -> torch.Tensor:
    parts = [x, v, ur]
    if _vpinn_model_uses_td_force_input(model):
        if td_force_input is None:
            raise ValueError("td_force_input is required when vpinn.use_td_force_input is enabled.")
        parts.append(td_force_input)
    if _vpinn_model_uses_acceleration_input(model):
        if acceleration_input is None:
            raise ValueError("acceleration_input is required when vpinn.use_acceleration_input is enabled.")
        parts.append(acceleration_input)
    if _vpinn_model_uses_phi_input(model):
        if phi_input is None:
            raise ValueError("phi_input is required when vpinn.use_phi_input is enabled.")
        parts.append(phi_input)
    if _vpinn_model_uses_sigma_inputs(model):
        if sigma_inputs is None:
            raise ValueError("sigma_inputs is required when vpinn.use_sigma_inputs is enabled.")
        parts.append(sigma_inputs)
    return model(torch.cat(parts, dim=-1))


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
            if x_true.ndim != 2:
                continue
            f_pred = _vpinn_force(model, x_true, v_true, ur_true)
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
    m: torch.Tensor,
    c: torch.Tensor,
    k: torch.Tensor,
    f0: Optional[torch.Tensor] = None,
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
    fs = [_vpinn_force(model, x, v, ur0)]

    dt_t = x0.new_tensor(float(dt))
    half = x0.new_tensor(0.5)
    sixth = x0.new_tensor(1.0 / 6.0)

    def accel(xi: torch.Tensor, vi: torch.Tensor) -> torch.Tensor:
        ci = _vpinn_force(model, xi, vi, ur0)
        fi = ci if f0 is None else ci * f0
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
        fs.append(_vpinn_force(model, x, v, ur0))

    return torch.stack(xs, dim=1), torch.stack(vs, dim=1), torch.stack(fs, dim=1)


def rollout_rk4_with_progress(
    *,
    model: nn.Module,
    x0: torch.Tensor,
    v0: torch.Tensor,
    ur0: torch.Tensor,
    steps: int,
    dt: float,
    m: torch.Tensor,
    c: torch.Tensor,
    k: torch.Tensor,
    f0: Optional[torch.Tensor] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    callback_every: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Same as rollout_rk4, but optionally emits progress callbacks.

    progress_callback(completed, total) is called every `callback_every` steps and
    on the final step.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    B, d = x0.shape
    m = m.view(1, d)
    c = c.view(1, d)
    k = k.view(1, d)
    every = max(1, int(callback_every))

    x = x0
    v = v0
    xs = [x]
    vs = [v]
    fs = [_vpinn_force(model, x, v, ur0)]

    dt_t = x0.new_tensor(float(dt))
    half = x0.new_tensor(0.5)
    sixth = x0.new_tensor(1.0 / 6.0)

    def accel(xi: torch.Tensor, vi: torch.Tensor) -> torch.Tensor:
        ci = _vpinn_force(model, xi, vi, ur0)
        fi = ci if f0 is None else ci * f0
        return (fi - c * vi - k * xi) / m

    total_steps = int(steps)
    for step_idx in range(total_steps):
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
        fs.append(_vpinn_force(model, x, v, ur0))

        completed = step_idx + 1
        if progress_callback is not None and (completed % every == 0 or completed == total_steps):
            progress_callback(completed, total_steps)

    return torch.stack(xs, dim=1), torch.stack(vs, dim=1), torch.stack(fs, dim=1)


def _reap_async_processes(
    processes: list[dict[str, Any]],
    *,
    best_state: dict[str, Any] | None = None,
    wait: bool = False,
) -> list[dict[str, Any]]:
    active: list[dict[str, Any]] = []
    for job in processes:
        proc: subprocess.Popen = job["proc"]
        if wait:
            return_code = proc.wait()
        else:
            return_code = proc.poll()
            if return_code is None:
                active.append(job)
                continue

        start_time = float(job.get("start_time", time.perf_counter()))
        elapsed = time.perf_counter() - start_time
        epoch = int(job.get("epoch", -1))
        ckpt_path = Path(job.get("checkpoint_path", ""))
        summary_path = Path(job.get("summary_path", ""))
        run_name = str(job.get("run_name", "")).strip()

        if return_code == 0:
            print(f"[async-val] epoch {epoch}: completed successfully in {elapsed:.2f}s")
            if writer is not None:
                writer.add_scalar("val_unseen/validation_wall_time_s", float(elapsed), int(epoch) + 1)
                writer.flush()
        else:
            print(
                f"[async-val] epoch {epoch}: FAILED with exit code {return_code} "
                f"after {elapsed:.2f}s"
            )

        if ckpt_path.exists() and return_code == 0:
            if best_state is not None and summary_path.exists() and run_name:
                try:
                    payload = json.loads(summary_path.read_text(encoding="utf-8"))
                    loss_total = payload.get("loss_total", None)
                    if loss_total is not None and np.isfinite(float(loss_total)):
                        loss_total_f = float(loss_total)
                        prev_best = float(best_state.get("loss_total", float("inf")))
                        if loss_total_f < prev_best:
                            models_dir = Path("models")
                            models_dir.mkdir(parents=True, exist_ok=True)
                            best_model_path = models_dir / f"{run_name}_best_val.pt"
                            best_meta_path = models_dir / f"{run_name}_best_val.json"
                            shutil.copy2(ckpt_path, best_model_path)
                            best_meta = {
                                "epoch": epoch,
                                "loss_total": loss_total_f,
                                "run_name": run_name,
                                "source_checkpoint": str(ckpt_path),
                                "summary_path": str(summary_path),
                            }
                            best_meta_path.write_text(json.dumps(best_meta, indent=2, sort_keys=True), encoding="utf-8")
                            best_state.update(
                                {
                                    "epoch": epoch,
                                    "loss_total": loss_total_f,
                                    "best_model_path": str(best_model_path),
                                }
                            )
                            print(
                                f"[async-val] epoch {epoch}: new best val_unseen/loss_total={loss_total_f:.6e}; "
                                f"kept {best_model_path}"
                            )
                except Exception as exc:
                    print(f"[async-val] epoch {epoch}: failed to promote best checkpoint ({exc})")
            try:
                ckpt_path.unlink()
            except OSError:
                pass

    return active


def _launch_async_validation(
    *,
    processes: list[dict[str, Any]],
    max_concurrent: int,
    checkpoint_path: Path,
    epoch: int,
    run_name: str,
    writer: SummaryWriter,
    async_device: str,
    async_num_workers: int,
    async_num_threads: int,
    do_losses: bool,
    do_rollout: bool,
    best_state: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    processes = _reap_async_processes(processes, best_state=best_state, wait=False)
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
        "--do-losses",
        "1" if do_losses else "0",
        "--do-rollout",
        "1" if do_rollout else "0",
    ]
    epoch_num = int(epoch + 1)
    summary_path = Path(writer.log_dir) / "async_validation" / "results" / f"epoch_{epoch_num:06d}.json"
    proc = subprocess.Popen(args, env=env)
    processes.append(
        {
            "proc": proc,
            "epoch": epoch_num,
            "start_time": time.perf_counter(),
            "checkpoint_path": str(checkpoint_path),
            "summary_path": str(summary_path),
            "run_name": run_name,
        }
    )
    print(
        f"[async-val] epoch {epoch_num}: launched "
        f"(losses={int(bool(do_losses))}, rollout={int(bool(do_rollout))})"
    )
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
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
        ur_series = np.full((t.size,), float(ur_arr), dtype=float)
    else:
        ur_flat = ur_arr.reshape(-1)
        if ur_flat.size == 1:
            ur_series = np.full((t.size,), float(ur_flat[0]), dtype=float)
        elif ur_flat.shape[0] == t.size:
            ur_series = ur_flat.astype(float, copy=False)
        else:
            raise ValueError(
                f"{path} reduced velocity must be scalar or length-matched to time "
                f"(got len={ur_flat.shape[0]}, expected {t.size})."
            )
    if not np.all(np.isfinite(ur_series)):
        idx = np.arange(ur_series.size, dtype=float)
        finite = np.isfinite(ur_series)
        if not np.any(finite):
            raise ValueError(f"{path} reduced velocity contains no finite values.")
        ur_series = np.interp(idx, idx[finite], ur_series[finite])
    return t, x, f, v, ur_series


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
    try:
        return _load_metadata_map(meta_path)
    except FileNotFoundError:
        warnings.warn(
            f"Metadata file '{meta_path}' not found. Falling back to U_r-based F0 conversion "
            "for vpinn.force_representation='coefficient'."
        )
        return None


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

    series_dir = Path(getattr(data_cfg, "train_series_dir", ""))
    if not series_dir:
        return None
    if not series_dir.is_absolute():
        series_dir = (Path.cwd() / series_dir).resolve()
    if not series_dir.exists():
        return None
    train_dir = series_dir / "train"
    val_dir = series_dir / "val"
    candidates: list[Path] = []
    if train_dir.exists():
        candidates.extend(sorted(train_dir.glob("*.npz")))
    if val_dir.exists():
        candidates.extend(sorted(val_dir.glob("*.npz")))
    for path in candidates:
        dt_val = _dt_from_npz(path)
        if dt_val is not None:
            if bool(getattr(data_cfg, "reduce_time", False)):
                rf = max(1, int(getattr(data_cfg, "reduction_factor", 1)))
                dt_val = float(dt_val * rf)
            return dt_val
    return None

def _maybe_reduce_time(
    t: np.ndarray,
    x: np.ndarray,
    f: np.ndarray,
    v: Optional[np.ndarray],
    ur: np.ndarray,
    *,
    enabled: bool,
    reduction_factor: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    if not enabled:
        return t, x, f, v, ur
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
    ur2 = ur[::rf]
    if t2.size < 2:
        raise ValueError("reduce_time produced too few samples")
    return t2, x2, f2, v2, ur2


def _load_trajectory(
    *,
    path: Path,
    dt_target: Optional[float],
    reduce_time: bool,
    reduction_factor: int,
    cut_start_seconds: float,
    force_representation: str,
    rho: float,
    D: float,
    k: float,
    m_eff: float,
) -> tuple[dict[str, Any], float]:
    t, x, f_meas, v_file, ur_series = _read_timeseries_npz(path)
    t, x, f_meas, v_file, ur_series = _maybe_reduce_time(
        t,
        x,
        f_meas,
        v_file,
        ur_series,
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
        ur_series = np.interp(t, t_in, ur_series)
        dt = float(dt_target)

    cut_start_seconds = max(0.0, float(cut_start_seconds))
    if cut_start_seconds > 0.0:
        t0 = float(t[0])
        mask = t >= (t0 + cut_start_seconds)
        t = t[mask]
        x = x[mask]
        f_meas = f_meas[mask]
        ur_series = ur_series[mask]
        if v_file is not None:
            v_file = v_file[mask]
        if t.size < 2:
            raise ValueError(f"{path.name}: too few samples remain after cut_start_seconds={cut_start_seconds}.")

    if v_file is None:
        raise ValueError(f"{path.name} has no 'dy'; in-file velocity is required.")
    v = v_file

    force_representation = str(force_representation).strip().lower()
    if force_representation not in {"force", "coefficient"}:
        raise ValueError("vpinn.force_representation must be one of: force, coefficient.")
    f0_series = None
    if force_representation == "coefficient":
        if not np.isfinite(float(k)) or not np.isfinite(float(m_eff)) or float(m_eff) <= 0.0:
            raise ValueError(f"Invalid (k, m_eff)=({k}, {m_eff}) for coefficient conversion.")
        omega_n = math.sqrt(float(k) / float(m_eff))
        f_n = omega_n / (2.0 * math.pi)
        ur_series_f = np.asarray(ur_series, dtype=float).reshape(-1)
        u_series = ur_series_f * f_n * float(D)
        v_arr = np.asarray(v, dtype=float)
        v_sq = np.sum(v_arr * v_arr, axis=1)
        speed_sq = u_series * u_series + v_sq
        f0_series = 0.5 * float(rho) * float(D) * speed_sq
        if not np.all(np.isfinite(f0_series)):
            raise ValueError(f"Invalid non-finite F0 values for '{path.name}'.")
        f0_series = np.clip(f0_series, 1e-12, None)
        f_meas = f_meas / f0_series.reshape(-1, 1)

    if ur_series.shape[0] != t.shape[0]:
        raise ValueError(f"{path.name}: U_r length {ur_series.shape[0]} does not match time {t.shape[0]}.")
    ur_series = np.asarray(ur_series, dtype=np.float32).reshape(-1, 1)
    traj = {
        "name": path.name,
        "t": torch.from_numpy(t.astype(np.float32)),
        "x": torch.from_numpy(x.astype(np.float32)),
        "v": torch.from_numpy(np.asarray(v, dtype=np.float32)),
        "f": torch.from_numpy(f_meas.astype(np.float32)),
        "ur": torch.from_numpy(ur_series),
    }
    if f0_series is not None:
        traj["f0"] = torch.from_numpy(np.asarray(f0_series, dtype=np.float32).reshape(-1, 1))
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


class WindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
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
        self._return_f0 = any("f0" in traj for traj in self.trajectories)

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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:  # type: ignore[override]
        traj = self.trajectories[int(self._traj_ids[idx])]
        start = int(self._starts[idx])
        end = start + self.M1
        x = traj["x"][start:end]
        v = traj["v"][start:end]
        f = traj["f"][start:end]
        ur = traj["ur"][start:end]
        items: list[torch.Tensor] = [x, v, f, ur]
        if self._return_f0:
            f0 = traj["f0"][start:end]
            items.append(f0)
        return tuple(items)  # type: ignore[return-value]


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
    def _reshape_param(param: torch.Tensor) -> torch.Tensor:
        if param.ndim == 1:
            return param.view(1, 1, -1)
        if param.ndim == 2:
            return param.unsqueeze(1)
        raise ValueError("m, c, and k must have shape (d,) or (B, d).")

    m = _reshape_param(m)
    c = _reshape_param(c)
    k = _reshape_param(k)
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
                f0 = None
            elif len(batch) == 5:
                x_win, v_win, f_meas, ur_win, f0 = batch
            else:
                raise ValueError("Unexpected batch format from dataloader.")
            x_win = x_win.to(device, non_blocking=non_blocking)
            v_win = v_win.to(device, non_blocking=non_blocking)
            f_meas = f_meas.to(device, non_blocking=non_blocking)
            ur_win = ur_win.to(device, non_blocking=non_blocking)
            if f0 is not None:
                f0 = f0.to(device, non_blocking=non_blocking)

            B, M1, d = x_win.shape
            inp = torch.cat([x_win, v_win, ur_win], dim=-1)

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                flat = inp.reshape(B * M1, -1)
                f_pred = model(flat).reshape(B, M1, d)
                per_loss_f = torch.mean((f_pred - f_meas) ** 2, dim=(1, 2))
                loss_f = torch.mean(per_loss_f)
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
                        f0=f0,
                    )
                    per_loss_w = torch.mean(R.pow(2), dim=(1, 2))
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
    expect_f0: bool,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
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
                f0 = None
            elif len(batch) == 5:
                x_win, v_win, f_meas, ur_win, f0 = batch
            else:
                raise ValueError("Unexpected batch format from dataloader.")

            x_win = x_win.to(device, non_blocking=non_blocking)
            v_win = v_win.to(device, non_blocking=non_blocking)
            f_meas = f_meas.to(device, non_blocking=non_blocking)
            ur_win = ur_win.to(device, non_blocking=non_blocking)
            if f0 is not None:
                f0 = f0.to(device, non_blocking=non_blocking)

            B, M1, d = x_win.shape
            inp = torch.cat([x_win, v_win, ur_win], dim=-1)

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                flat = inp.reshape(B * M1, -1)
                f_pred = model(flat).reshape(B, M1, d)
                per_loss_f = torch.mean((f_pred - f_meas) ** 2, dim=(1, 2))
                per_loss_w = per_loss_f.new_zeros(per_loss_f.shape)
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
                        f0=f0,
                    )
                    per_loss_w = torch.mean(R.pow(2), dim=(1, 2))

                per_roll = None
                if rollout_force_steps > 0:
                    steps_k = min(int(rollout_force_steps), int(M1) - 1)
                    if steps_k > 0:
                        f0_step = f0[:, 0, :] if f0 is not None else None
                        _x_seq, _v_seq, f_seq = rollout_rk4(
                            model=model,
                            x0=x_win[:, 0, :],
                            v0=v_win[:, 0, :],
                            ur0=ur_win[:, 0, :],
                            steps=steps_k,
                            dt=dt,
                            m=m,
                            c=c,
                            k=k,
                            f0=f0_step,
                        )
                        f_roll = f_seq[:, : steps_k + 1, :]
                        f_true = f_meas[:, : steps_k + 1, :]
                        per_roll = torch.mean((f_roll - f_true) ** 2, dim=(1, 2))

            ur_vals = ur_win[:, 0, 0].detach().cpu().numpy()
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
    device: torch.device,
    tag_prefix: str = "val/rollout",
    step: int | None = None,
    log_metrics: bool = True,
    log_plots: bool = True,
    title_suffix: str = "",
    log_spectra: bool = False,
) -> dict[str, float]:
    x_true_t = traj["x"].to(device)
    v_true_t = traj["v"].to(device)
    f_true_t = traj["f"].to(device)
    ur_true_t = traj["ur"].to(device)
    f0_true_t = traj.get("f0", None)
    if f0_true_t is not None:
        f0_true_t = f0_true_t.to(device)
    t_np = traj["t"].detach().cpu().numpy()
    if x_true_t.ndim != 2:
        return {}
    d = int(x_true_t.shape[-1])
    if d < 1:
        return {}
    if d > 1:
        print("vpinn rollout validation: d>1 detected; logging only the first DOF.")

    steps = int(x_true_t.shape[0] - 1)
    if steps < 1:
        return {}

    x_seq, v_seq, f_seq = rollout_rk4(
        model=model,
        x0=x_true_t[0:1, :],
        v0=v_true_t[0:1, :],
        ur0=ur_true_t[0:1, :],
        steps=steps,
        dt=dt,
        m=m,
        c=c,
        k=k,
        f0=(f0_true_t[0:1, :] if f0_true_t is not None else None),
    )
    x_pred = x_seq[0, :, 0].detach().cpu().numpy()
    v_pred = v_seq[0, :, 0].detach().cpu().numpy()
    f_pred = f_seq[0, :, 0].detach().cpu().numpy()
    x_true = x_true_t[:, 0].detach().cpu().numpy()
    f_true = f_true_t[:, 0].detach().cpu().numpy()

    metrics: dict[str, float] = {}

    with torch.no_grad():
        f_on_data = _vpinn_force(model, x_true_t, v_true_t, ur_true_t)[:, 0].detach().cpu().numpy()
    if f0_true_t is not None:
        f0_np = f0_true_t[:, 0].detach().cpu().numpy()
        f_on_data_force = f_on_data * f0_np
        f_true_force = f_true * f0_np
        force_std = float(np.std(f_true_force))
        if force_std <= 0.0:
            force_std = 1.0
        rel_rmse_force_on_data = float(np.sqrt(np.mean((f_on_data_force - f_true_force) ** 2))) / force_std
        metrics[FORCE_MAPPING_NRMSE_KEY] = rel_rmse_force_on_data
    else:
        force_std = float(np.std(f_true))
        if force_std <= 0.0:
            force_std = 1.0
        rel_rmse_force_on_data = float(np.sqrt(np.mean((f_on_data - f_true) ** 2))) / force_std
        metrics[FORCE_MAPPING_NRMSE_KEY] = rel_rmse_force_on_data

    if log_plots:
        y_true_norm = x_true / float(D)
        y_pred_norm = x_pred / float(D)
        freq = float(torch.sqrt(k[0] / m[0]).detach().cpu())
        denom = float(freq * float(D)) if freq > 0 else 1.0
        p_pred_norm = v_pred / denom

        zoom_mask = create_zoom_mask(t_np)
        ur_val = float(ur_true_t[0, 0].detach().cpu().item())
        log_displacement_plots(
            writer,
            epoch,
            t_np,
            y_true_norm,
            y_pred_norm,
            p_pred_norm,
            zoom_mask,
            reduced_velocity=ur_val,
            tag_prefix=tag_prefix,
            step=step,
            title_suffix=title_suffix,
            log_spectra=log_spectra,
        )
        log_force_plots(
            writer,
            epoch,
            t_np,
            f_pred,
            f_true,
            zoom_mask,
            reduced_velocity=ur_val,
            tag_prefix=tag_prefix,
            step=step,
            title_suffix=title_suffix,
            log_spectra=log_spectra,
        )
        if log_spectra:
            log_area_normalized_rollout_spectra(
                writer,
                epoch,
                disp_t=t_np,
                disp_true=y_true_norm,
                disp_pred=y_pred_norm,
                force_t=t_np,
                force_true=f_true,
                force_pred=f_pred,
                reduced_velocity=ur_val,
                tag=f"{tag_prefix}_spectra",
                step=step,
                title_suffix=title_suffix,
            )
    return metrics


def _td_rollout_traj_to_tensors(traj: dict[str, Any]) -> dict[str, Any]:
    if "x" in traj and "v" in traj and "f" in traj:
        return {
            "name": str(traj.get("name", "")),
            "x": traj["x"] if torch.is_tensor(traj["x"]) else torch.from_numpy(np.ascontiguousarray(traj["x"])).float(),
            "v": traj["v"] if torch.is_tensor(traj["v"]) else torch.from_numpy(np.ascontiguousarray(traj["v"])).float(),
            "f": traj["f"] if torch.is_tensor(traj["f"]) else torch.from_numpy(np.ascontiguousarray(traj["f"])).float(),
            "td_force": (
                traj["td_force"]
                if torch.is_tensor(traj["td_force"])
                else torch.from_numpy(np.ascontiguousarray(traj["td_force"])).float()
            ),
            "ur": traj["ur"] if torch.is_tensor(traj["ur"]) else torch.from_numpy(np.ascontiguousarray(traj["ur"])).float(),
            "td_context": (
                traj["td_context"]
                if torch.is_tensor(traj["td_context"])
                else torch.from_numpy(np.ascontiguousarray(traj["td_context"])).float()
            ),
            "t": traj["t"] if torch.is_tensor(traj["t"]) else torch.from_numpy(np.ascontiguousarray(traj["t"])).float(),
            "dry_mass_kg": np.asarray(traj["dry_mass_kg"]),
            "effective_mass_kg": np.asarray(traj["effective_mass_kg"]),
            "damping_c": np.asarray(traj["damping_c"]),
            "stiffness_n_m": np.asarray(traj["stiffness_n_m"]),
        }

    required = {"y", "dy", "force_total", "force_td", "ur", "td_context", "t"}
    missing = sorted(required.difference(traj))
    if missing:
        raise KeyError(f"TD VPINN rollout trajectory is missing required keys: {missing}")

    return {
        "name": str(traj.get("name", "")),
        "x": torch.from_numpy(np.ascontiguousarray(traj["y"])).float().unsqueeze(1),
        "v": torch.from_numpy(np.ascontiguousarray(traj["dy"])).float().unsqueeze(1),
        "f": torch.from_numpy(np.ascontiguousarray(traj["force_per_m"])).float().unsqueeze(1),
        "td_force": torch.from_numpy(np.ascontiguousarray(traj["force_td_per_m"])).float().unsqueeze(1),
        "ur": torch.from_numpy(np.ascontiguousarray(traj["ur"])).float().unsqueeze(1),
        "td_context": torch.from_numpy(np.ascontiguousarray(traj["td_context"])).float(),
        "t": torch.from_numpy(np.ascontiguousarray(traj["t"])).float(),
        "dry_mass_kg": np.asarray(traj["dry_mass_kg"]),
        "effective_mass_kg": np.asarray(traj["effective_mass_kg"]),
        "damping_c": np.asarray(traj["damping_c"]),
        "stiffness_n_m": np.asarray(traj["stiffness_n_m"]),
    }


def _log_td_correction_rollout_validation(
    *,
    writer: Any,
    epoch: int,
    model: nn.Module,
    traj: dict[str, Any],
    dt: float,
    td_mass_source: str,
    rho: float,
    diameter: float,
    td_params: dict[str, float],
    device: torch.device,
    sigma_min: float,
    mean_active: bool,
    probabilistic: bool,
    fhat_active: bool,
    use_td_force_input: bool,
    fhat_bound_multiplier: float,
    force_zero_output: bool = False,
    rollout_stochastic: bool = False,
    rollout_noise_scale: float = 1.0,
    rollout_seed: int | None = None,
    tag_prefix: str = "val/rollout",
    step: int | None = None,
    log_metrics: bool = True,
    log_plots: bool = True,
    log_phase_map: bool = False,
    log_correction_on_data: bool = False,
    title_suffix: str = "",
    log_spectra: bool = False,
) -> dict[str, float]:
    traj_t = _td_rollout_traj_to_tensors(traj)
    x_true_t = traj_t["x"].to(device)
    v_true_t = traj_t["v"].to(device)
    f_true_t = traj_t["f"].to(device)
    ur_true_t = traj_t["ur"].to(device)
    td_true_t = traj_t["td_force"].to(device)
    td_context_t = traj_t["td_context"].to(device)
    t_np = traj_t["t"].detach().cpu().numpy()
    if x_true_t.ndim != 2:
        return {}
    d = int(x_true_t.shape[-1])
    if d < 1:
        return {}
    if d > 1:
        print("vpinn td-correction rollout validation: d>1 detected; logging only the first DOF.")

    steps = int(x_true_t.shape[0] - 1)
    if steps < 1:
        return {}

    mass_key = "dry_mass_kg" if str(td_mass_source).strip().lower() == "dry" else "effective_mass_kg"
    mass_value = float(np.asarray(traj_t[mass_key]).reshape(()))
    damping_value = float(np.asarray(traj_t["damping_c"]).reshape(()))
    stiffness_value = float(np.asarray(traj_t["stiffness_n_m"]).reshape(()))
    m = torch.full((1, d), mass_value, dtype=x_true_t.dtype, device=device)
    c = torch.full((1, d), damping_value, dtype=x_true_t.dtype, device=device)
    k = torch.full((1, d), stiffness_value, dtype=x_true_t.dtype, device=device)

    x_seq, v_seq, f_seq, td_roll_seq, _corr_roll_seq, delta_fhat_roll_seq = _vpinn_td_rollout(
        model=model,
        x0=x_true_t[0:1, :],
        v0=v_true_t[0:1, :],
        ur0=ur_true_t[0:1, :],
        td_context0=td_context_t[0:1, :],
        steps=steps,
        dt=dt,
        m=m,
        c=c,
        k=k,
        rho=rho,
        diameter=diameter,
        td_params=td_params,
        mean_active=mean_active,
        sigma_active=probabilistic,
        fhat_active=fhat_active,
        use_td_force_input=use_td_force_input,
        fhat_bound_multiplier=fhat_bound_multiplier,
        sigma_min=sigma_min,
        force_zero_output=force_zero_output,
        rollout_stochastic=rollout_stochastic,
        rollout_noise_scale=rollout_noise_scale,
        rollout_seed=rollout_seed,
    )
    x_pred = x_seq[0, :, 0].detach().cpu().numpy()
    v_pred = v_seq[0, :, 0].detach().cpu().numpy()
    f_roll = f_seq[0, :, 0].detach().cpu().numpy()
    td_roll = td_roll_seq[0, :, 0].detach().cpu().numpy()
    delta_fhat_roll = delta_fhat_roll_seq[0, :, 0].detach().cpu().numpy()

    with torch.no_grad():
        step_on_data = _vpinn_step_with_corrections(
            model=model,
            x=x_true_t[:-1],
            v=v_true_t[:-1],
            ur=ur_true_t[:-1],
            td_context=td_context_t[:-1],
            dt=dt,
            m=torch.full((x_true_t.shape[0] - 1, d), mass_value, dtype=x_true_t.dtype, device=device),
            c=torch.full((x_true_t.shape[0] - 1, d), damping_value, dtype=x_true_t.dtype, device=device),
            k=torch.full((x_true_t.shape[0] - 1, d), stiffness_value, dtype=x_true_t.dtype, device=device),
            rho=rho,
            diameter=diameter,
            td_params=td_params,
            mean_active=mean_active,
            sigma_active=probabilistic,
            fhat_active=fhat_active,
            use_td_force_input=use_td_force_input,
            fhat_bound_multiplier=fhat_bound_multiplier,
            sigma_min=sigma_min,
            force_zero_output=force_zero_output,
        )
        corr_on_data = step_on_data["corr_mu"]
        sigma_on_data = step_on_data["corr_sigma"]
        td_on_data = step_on_data["td_force_next"]
        f_on_data = step_on_data["total_force_next"][:, 0].detach().cpu().numpy()
        delta_fhat_on_data = step_on_data["delta_fhat"][:, 0].detach().cpu().numpy()

    if f_roll.size > 0:
        force_total_full = np.concatenate([f_on_data[:1], f_roll], axis=0)
        force_td_full = np.concatenate([td_on_data[:1, 0].detach().cpu().numpy(), td_roll], axis=0)
    else:
        force_total_full = f_on_data
        force_td_full = td_on_data[:, 0].detach().cpu().numpy()

    metrics = compute_validation_metrics(
        model=model,  # ignored when rollout is provided
        y_data_t=x_true_t[:, 0],
        val_vel=v_true_t[:, 0],
        reduced_velocity=ur_true_t[:, 0],
        m_eff=mass_value,
        dt=dt,
        t=t_np,
        y_data_raw=x_true_t[:, 0].detach().cpu().numpy(),
        force_data=f_true_t[:, 0].detach().cpu().numpy(),
        D=diameter,
        k=stiffness_value,
        device=device,
        rollout={
            "y_norm": x_pred / float(diameter),
            "p_norm": v_pred / (float(np.sqrt(stiffness_value / mass_value)) * float(diameter)),
            "force_total": force_total_full,
        },
    )

    force_true = f_true_t[1:, 0].detach().cpu().numpy()
    force_std = float(np.std(force_true))
    if force_std <= 0.0:
        force_std = 1.0
    metrics[FORCE_MAPPING_NRMSE_KEY] = float(np.sqrt(np.mean((f_on_data - force_true) ** 2))) / force_std
    if fhat_active:
        metrics["Delta fhat mean abs"] = float(np.mean(np.abs(delta_fhat_on_data)))
        metrics["Delta fhat mean"] = float(np.mean(delta_fhat_on_data))

    if log_metrics:
        for name, value in metrics.items():
            if np.isfinite(float(value)):
                writer.add_scalar(f"val/{name}", float(value), epoch)

    if log_plots:
        y_true = x_true_t[:, 0].detach().cpu().numpy()
        zoom_mask = create_zoom_mask(t_np)
        ur_val = float(ur_true_t[0, 0].detach().cpu().item())
        omega = float(np.sqrt(stiffness_value / mass_value))
        q_true_norm = y_true / float(diameter)
        p_true_norm = v_true_t[:, 0].detach().cpu().numpy() / (omega * float(diameter))
        q_pred_norm = x_pred / float(diameter)
        p_pred_norm = v_pred / (omega * float(diameter))
        log_displacement_plots(
            writer,
            epoch,
            t_np,
            q_true_norm,
            q_pred_norm,
            p_pred_norm,
            zoom_mask,
            reduced_velocity=ur_val,
            tag_prefix=tag_prefix,
            step=step,
            title_suffix=title_suffix,
            log_spectra=log_spectra,
        )
        n_force = min(len(t_np), len(force_total_full), len(force_true))
        force_t = t_np[:n_force]
        force_true_plot = f_true_t[:, 0].detach().cpu().numpy()[:n_force]
        log_force_plots(
            writer,
            epoch,
            force_t,
            force_total_full[:n_force],
            force_true_plot,
            create_zoom_mask(force_t),
            reduced_velocity=ur_val,
            force_coeff_baseline=force_td_full[:n_force],
            baseline_label="C_F (Vivana-TD)",
            tag_prefix=tag_prefix,
            step=step,
            title_suffix=title_suffix,
            log_spectra=log_spectra,
        )
        if log_spectra:
            log_area_normalized_rollout_spectra(
                writer,
                epoch,
                disp_t=t_np,
                disp_true=q_true_norm,
                disp_pred=q_pred_norm,
                force_t=force_t,
                force_true=force_true_plot,
                force_pred=force_total_full[:n_force],
                reduced_velocity=ur_val,
                force_baseline=force_td_full[:n_force],
                force_baseline_label="C_F (Vivana-TD)",
                tag=f"{tag_prefix}_spectra",
                step=step,
                title_suffix=title_suffix,
            )
        if log_correction_on_data:
            output_label = (
                "Correction coefficient"
                if str(getattr(model, "force_representation", "force")).strip().lower() == "coefficient"
                else "Correction force"
            )
            log_correction_on_data_plot(
                writer,
                epoch,
                t=t_np[1:],
                corr_true=(f_true_t[1:, 0] - td_on_data[:, 0]).detach().cpu().numpy(),
                corr_pred=corr_on_data[:, 0].detach().cpu().numpy(),
                sigma=(
                    sigma_on_data[:, 0].detach().cpu().numpy()
                    if probabilistic
                    else None
                ),
                reduced_velocity=ur_val,
                value_label=output_label,
                sigma_label=(
                    "Sigma coefficient"
                    if str(getattr(model, "force_representation", "force")).strip().lower() == "coefficient"
                    else "Sigma force"
                ),
                tag="final_val/correction_on_data",
                step=step,
                title_suffix=title_suffix,
            )
            if fhat_active:
                log_correction_on_data_plot(
                    writer,
                    epoch,
                    t=t_np[1:],
                    corr_true=np.zeros_like(delta_fhat_on_data),
                    corr_pred=delta_fhat_on_data,
                    sigma=None,
                    reduced_velocity=ur_val,
                    value_label="Delta fhat",
                    sigma_label="",
                    tag="final_val/delta_fhat_on_data",
                    step=step,
                    title_suffix=title_suffix,
                )
        if log_phase_map:
            q_extent = np.concatenate([np.asarray(q_true_norm, dtype=float), np.asarray(q_pred_norm, dtype=float)])
            p_extent = np.concatenate([np.asarray(p_true_norm, dtype=float), np.asarray(p_pred_norm, dtype=float)])
            q_grid, p_grid = build_phase_plot_grid(q_extent, p_extent, bins=96, extent_scale=1.2)
            x_grid = torch.as_tensor(
                (q_grid.reshape(-1) * float(diameter)).reshape(-1, 1),
                dtype=x_true_t.dtype,
                device=device,
            )
            v_grid = torch.as_tensor(
                (p_grid.reshape(-1) * (omega * float(diameter))).reshape(-1, 1),
                dtype=x_true_t.dtype,
                device=device,
            )
            ur_grid = torch.full((x_grid.shape[0], 1), ur_val, dtype=x_true_t.dtype, device=device)
            td_force_grid = None
            if _vpinn_model_uses_td_force_input(model):
                td_force_grid_np = nearest_phase_series_values(
                    q_grid,
                    p_grid,
                    q_true_norm,
                    p_true_norm,
                    td_true_t[:, 0].detach().cpu().numpy(),
                )
                td_force_grid = torch.as_tensor(
                    td_force_grid_np.reshape(-1, 1),
                    dtype=x_true_t.dtype,
                    device=device,
                )
            phi_grid = None
            sigma_grid_inputs = None
            acceleration_grid = None
            if (
                _vpinn_model_uses_phi_input(model)
                or _vpinn_model_uses_sigma_inputs(model)
                or _vpinn_model_uses_acceleration_input(model)
            ):
                phi_series, sigma_series, acceleration_series = td_hidden_inputs_from_context_torch(
                    td_context=td_context_t,
                    structural_mass=torch.full_like(x_true_t, mass_value),
                    stiffness=torch.full_like(x_true_t, stiffness_value),
                    diameter=diameter,
                    velocity=v_true_t,
                    phase_input_source=_vpinn_model_phase_input_source(model),
                )
                phi_series_np = phi_series.detach().cpu().numpy()
                sigma_series_np = sigma_series.detach().cpu().numpy()
                if _vpinn_model_uses_phi_input(model):
                    phi_grid = torch.as_tensor(
                        np.stack(
                            [
                                nearest_phase_series_values(q_grid, p_grid, q_true_norm, p_true_norm, phi_series_np[:, 0]),
                                nearest_phase_series_values(q_grid, p_grid, q_true_norm, p_true_norm, phi_series_np[:, 1]),
                            ],
                            axis=-1,
                        ).reshape(-1, 2),
                        dtype=x_true_t.dtype,
                        device=device,
                    )
                if _vpinn_model_uses_sigma_inputs(model):
                    sigma_grid_inputs = torch.as_tensor(
                        np.stack(
                            [
                                nearest_phase_series_values(q_grid, p_grid, q_true_norm, p_true_norm, sigma_series_np[:, 0]),
                                nearest_phase_series_values(q_grid, p_grid, q_true_norm, p_true_norm, sigma_series_np[:, 1]),
                            ],
                            axis=-1,
                        ).reshape(-1, 2),
                        dtype=x_true_t.dtype,
                        device=device,
                    )
                if _vpinn_model_uses_acceleration_input(model):
                    acceleration_series_np = acceleration_series.detach().cpu().numpy()
                    acceleration_grid = torch.as_tensor(
                        nearest_phase_series_values(q_grid, p_grid, q_true_norm, p_true_norm, acceleration_series_np[:, 0]).reshape(-1, 1),
                        dtype=x_true_t.dtype,
                        device=device,
                    )
            with torch.no_grad():
                corr_grid, sigma_grid, raw_delta_fhat_grid = _vpinn_predict_outputs(
                    model,
                    x_grid,
                    v_grid,
                    ur_grid,
                    (td_force_grid if use_td_force_input else None),
                    acceleration_grid,
                    phi_grid,
                    sigma_grid_inputs,
                    mean_active=mean_active,
                    sigma_active=probabilistic,
                    fhat_active=fhat_active,
                    sigma_min=sigma_min,
                    force_zero_output=force_zero_output,
                )
            if mean_active:
                output_label = (
                    "Correction coefficient"
                    if str(getattr(model, "force_representation", "force")).strip().lower() == "coefficient"
                    else "Correction force"
                )
                log_signed_phase_output_plot(
                    writer,
                    epoch,
                    q_grid=q_grid,
                    p_grid=p_grid,
                    values=corr_grid[:, 0].detach().cpu().numpy().reshape(q_grid.shape),
                    q_true=q_true_norm,
                    p_true=p_true_norm,
                    q_pred=q_pred_norm,
                    p_pred=p_pred_norm,
                    reduced_velocity=ur_val,
                    output_label=output_label,
                    sigma_values=(
                        sigma_grid[:, 0].detach().cpu().numpy().reshape(q_grid.shape)
                        if probabilistic
                        else None
                    ),
                    sigma_label=(
                        "Sigma coefficient"
                        if str(getattr(model, "force_representation", "force")).strip().lower() == "coefficient"
                        else "Sigma force"
                    ),
                    tag="final_val/phase_output",
                    step=step,
                    title_suffix=title_suffix,
                )
            if fhat_active:
                delta_fhat_grid, _ = td_bounded_delta_fhat_torch(
                    raw_delta_fhat_grid,
                    fhat_td=torch.as_tensor(
                        nearest_phase_series_values(q_grid, p_grid, q_true_norm, p_true_norm, np.asarray(traj["fhat_td"], dtype=float)).reshape(-1, 1),
                        dtype=x_true_t.dtype,
                        device=device,
                    ),
                    fhat_min=float(td_params["fhat_min"]),
                    fhat_max=float(td_params["fhat_max"]),
                    fhat_bound_multiplier=float(fhat_bound_multiplier),
                )
                log_signed_phase_output_plot(
                    writer,
                    epoch,
                    q_grid=q_grid,
                    p_grid=p_grid,
                    values=delta_fhat_grid[:, 0].detach().cpu().numpy().reshape(q_grid.shape),
                    q_true=q_true_norm,
                    p_true=p_true_norm,
                    q_pred=q_pred_norm,
                    p_pred=p_pred_norm,
                    reduced_velocity=ur_val,
                    output_label="Delta fhat",
                    sigma_values=None,
                    sigma_label="",
                    tag="final_val/phase_output_delta_fhat",
                    step=step,
                    title_suffix=title_suffix,
                )
    return metrics


def _vpinn_predict_outputs(
    model: nn.Module,
    x: torch.Tensor,
    v: torch.Tensor,
    ur: torch.Tensor,
    td_force_input: torch.Tensor | None = None,
    acceleration_input: torch.Tensor | None = None,
    phi_input: torch.Tensor | None = None,
    sigma_inputs: torch.Tensor | None = None,
    *,
    mean_active: bool,
    sigma_active: bool,
    fhat_active: bool,
    sigma_min: float,
    force_zero_output: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    d = int(x.shape[-1])
    if force_zero_output:
        mu = x.new_zeros(x.shape[:-1] + (d,))
        if sigma_active:
            sigma = x.new_full(x.shape[:-1] + (d,), float(sigma_min))
        else:
            sigma = torch.zeros_like(mu)
        raw_delta_fhat = torch.zeros_like(mu)
        return mu, sigma, raw_delta_fhat
    out = _vpinn_force(
        model,
        x,
        v,
        ur,
        td_force_input,
        acceleration_input,
        phi_input,
        sigma_inputs,
    )
    expected_dim = _vpinn_output_dim(mean_active=mean_active, sigma_active=sigma_active, fhat_active=fhat_active, d=d)
    if int(out.shape[-1]) != expected_dim:
        raise ValueError(f"Unexpected VPINN TD correction output dimension {int(out.shape[-1])}; expected {expected_dim}.")
    cursor = 0
    if mean_active:
        mu = out[..., cursor:cursor + d]
        cursor += d
    else:
        mu = x.new_zeros(x.shape[:-1] + (d,))
    if sigma_active:
        raw_sigma = out[..., cursor:cursor + d]
        sigma = float(sigma_min) + F.softplus(raw_sigma)
        cursor += d
    else:
        sigma = x.new_zeros(x.shape[:-1] + (d,))
    if fhat_active:
        raw_delta_fhat = out[..., cursor:cursor + d]
    else:
        raw_delta_fhat = x.new_zeros(x.shape[:-1] + (d,))
    return mu, sigma, raw_delta_fhat


def _vpinn_step_with_corrections(
    *,
    model: nn.Module,
    x: torch.Tensor,
    v: torch.Tensor,
    ur: torch.Tensor,
    td_context: torch.Tensor,
    dt: float | torch.Tensor,
    m: torch.Tensor,
    c: torch.Tensor,
    k: torch.Tensor,
    rho: float,
    diameter: float,
    td_params: dict[str, float],
    mean_active: bool,
    sigma_active: bool,
    fhat_active: bool,
    use_td_force_input: bool,
    fhat_bound_multiplier: float,
    sigma_min: float,
    force_zero_output: bool = False,
    rollout_stochastic: bool = False,
    rollout_noise_scale: float = 1.0,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    baseline_force_next, baseline_context_next, baseline_diag = td_baseline_step_torch(
        velocity=v,
        acceleration=td_context[..., 0:1],
        td_context=td_context,
        dt=dt,
        rho=rho,
        diameter=diameter,
        params=td_params,
        return_diagnostics=True,
    )
    phi_input, sigma_inputs, acceleration_input = _vpinn_optional_hidden_inputs_from_context(
        model,
        td_context=td_context,
        velocity=v,
        structural_mass=m,
        stiffness=k,
        diameter=diameter,
    )
    corr_mu, corr_sigma, raw_delta_fhat = _vpinn_predict_outputs(
        model,
        x,
        v,
        ur,
        (baseline_force_next if use_td_force_input else None),
        acceleration_input,
        phi_input,
        sigma_inputs,
        mean_active=mean_active,
        sigma_active=sigma_active,
        fhat_active=fhat_active,
        sigma_min=sigma_min,
        force_zero_output=force_zero_output,
    )
    if fhat_active:
        td_force_next, td_context_next, td_diag = td_baseline_step_torch(
            velocity=v,
            acceleration=td_context[..., 0:1],
            td_context=td_context,
            dt=dt,
            rho=rho,
            diameter=diameter,
            params=td_params,
            raw_delta_fhat=raw_delta_fhat,
            fhat_bound_multiplier=float(fhat_bound_multiplier),
            return_diagnostics=True,
        )
    else:
        td_force_next = baseline_force_next
        td_context_next = baseline_context_next
        td_diag = dict(baseline_diag)
        td_diag["delta_fhat"] = raw_delta_fhat.new_zeros(raw_delta_fhat.shape)
        td_diag["fhat_corr"] = td_diag["fhat_td"]
        td_diag["omega_vy_corr"] = td_diag["omega_vy_td"]
    corr_force = corr_mu
    if rollout_stochastic and sigma_active:
        noise = torch.randn(
            corr_mu.shape,
            device=corr_mu.device,
            dtype=corr_mu.dtype,
            generator=generator,
        )
        corr_force = corr_mu + float(rollout_noise_scale) * corr_sigma * noise
    total_force = td_force_next + corr_force
    x_next, v_next, a_next = structural_step_constant_force_torch(
        y=x,
        velocity=v,
        force=total_force,
        dt=dt,
        mass=m,
        damping_c=c,
        stiffness=k,
    )
    td_context_next = td_context_next.clone()
    td_context_next[..., 0:1] = a_next
    return {
        "baseline_force_next": baseline_force_next,
        "td_force_next": td_force_next,
        "total_force_next": total_force,
        "corr_mu": corr_mu,
        "corr_sigma": corr_sigma,
        "corr_force": corr_force,
        "raw_delta_fhat": raw_delta_fhat,
        "delta_fhat": td_diag["delta_fhat"],
        "fhat_td": td_diag["fhat_td"],
        "fhat_corr": td_diag["fhat_corr"],
        "x_next": x_next,
        "v_next": v_next,
        "a_next": a_next,
        "td_context_next": td_context_next,
    }


def _vpinn_td_rollout(
    *,
    model: nn.Module,
    x0: torch.Tensor,
    v0: torch.Tensor,
    ur0: torch.Tensor,
    td_context0: torch.Tensor,
    steps: int,
    dt: float,
    m: torch.Tensor,
    c: torch.Tensor,
    k: torch.Tensor,
    rho: float,
    diameter: float,
    td_params: dict[str, float],
    mean_active: bool,
    sigma_active: bool,
    fhat_active: bool,
    use_td_force_input: bool,
    fhat_bound_multiplier: float,
    sigma_min: float,
    force_zero_output: bool = False,
    rollout_stochastic: bool = False,
    rollout_noise_scale: float = 1.0,
    rollout_seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = x0
    v = v0
    td_context = td_context0
    xs = [x]
    vs = [v]
    total_fs: list[torch.Tensor] = []
    td_fs: list[torch.Tensor] = []
    corr_fs: list[torch.Tensor] = []
    delta_fhats: list[torch.Tensor] = []
    generator: torch.Generator | None = None
    if rollout_seed is not None:
        generator = torch.Generator(device=x0.device)
        generator.manual_seed(int(rollout_seed))
    for _ in range(int(steps)):
        step = _vpinn_step_with_corrections(
            model=model,
            x=x,
            v=v,
            ur=ur0,
            td_context=td_context,
            dt=dt,
            m=m,
            c=c,
            k=k,
            rho=rho,
            diameter=diameter,
            td_params=td_params,
            mean_active=mean_active,
            sigma_active=sigma_active,
            fhat_active=fhat_active,
            use_td_force_input=use_td_force_input,
            fhat_bound_multiplier=fhat_bound_multiplier,
            sigma_min=sigma_min,
            force_zero_output=force_zero_output,
            rollout_stochastic=rollout_stochastic,
            rollout_noise_scale=rollout_noise_scale,
            generator=generator,
        )
        x = step["x_next"]
        v = step["v_next"]
        td_context = step["td_context_next"]
        xs.append(x)
        vs.append(v)
        total_fs.append(step["total_force_next"])
        td_fs.append(step["td_force_next"])
        corr_fs.append(step["corr_force"])
        delta_fhats.append(step["delta_fhat"])
    empty_force = x0.new_zeros((x0.shape[0], 0, x0.shape[1]))
    return (
        torch.stack(xs, dim=1),
        torch.stack(vs, dim=1),
        torch.stack(total_fs, dim=1) if total_fs else empty_force,
        torch.stack(td_fs, dim=1) if td_fs else empty_force,
        torch.stack(corr_fs, dim=1) if corr_fs else empty_force,
        torch.stack(delta_fhats, dim=1) if delta_fhats else empty_force,
    )


def _vpinn_td_rollout_loss_from_batch(
    *,
    model: nn.Module,
    batch: Any,
    device: torch.device,
    non_blocking: bool,
    dt: float,
    rho: float,
    diameter: float,
    td_params: dict[str, float],
    mean_active: bool,
    sigma_active: bool,
    fhat_active: bool,
    use_td_force_input: bool,
    fhat_bound_multiplier: float,
    sigma_min: float,
    force_zero_output: bool,
    rollout_loss_mode: str,
    rollout_stochastic_samples: int,
    rollout_noise_scale: float,
    ur_bin_state_scale_info: dict[str, Any] | None,
    ur_bin_size: float,
) -> torch.Tensor:
    x0, v0, ur0, td0, x_true_seq, v_true_seq, m0, c0, k0 = [item.to(device, non_blocking=non_blocking) for item in batch]
    mode_key = str(rollout_loss_mode).strip().lower()
    if mode_key == "stochastic":
        mode_key = "stochastic_nll"
    if mode_key not in {"deterministic", "stochastic_nll", "stochastic_mse"}:
        raise ValueError("loss.rollout_loss_mode must be one of: deterministic, stochastic_nll, stochastic_mse.")
    samples = max(1, int(rollout_stochastic_samples))
    steps = int(x_true_seq.shape[1] - 1)
    if mode_key in {"stochastic_nll", "stochastic_mse"} and samples > 1:
        batch_size = int(x0.shape[0])
        x0_in = x0.unsqueeze(0).expand(samples, *x0.shape).reshape(samples * batch_size, *x0.shape[1:])
        v0_in = v0.unsqueeze(0).expand(samples, *v0.shape).reshape(samples * batch_size, *v0.shape[1:])
        ur0_in = ur0.unsqueeze(0).expand(samples, *ur0.shape).reshape(samples * batch_size, *ur0.shape[1:])
        td0_in = td0.unsqueeze(0).expand(samples, *td0.shape).reshape(samples * batch_size, *td0.shape[1:])
        m0_in = m0.unsqueeze(0).expand(samples, *m0.shape).reshape(samples * batch_size, *m0.shape[1:])
        c0_in = c0.unsqueeze(0).expand(samples, *c0.shape).reshape(samples * batch_size, *c0.shape[1:])
        k0_in = k0.unsqueeze(0).expand(samples, *k0.shape).reshape(samples * batch_size, *k0.shape[1:])
        x_pred, v_pred, _, _, _, _ = _vpinn_td_rollout(
            model=model,
            x0=x0_in,
            v0=v0_in,
            ur0=ur0_in,
            td_context0=td0_in,
            steps=steps,
            dt=dt,
            m=m0_in,
            c=c0_in,
            k=k0_in,
            rho=rho,
            diameter=diameter,
            td_params=td_params,
            mean_active=mean_active,
            sigma_active=sigma_active,
            fhat_active=fhat_active,
            use_td_force_input=use_td_force_input,
            fhat_bound_multiplier=fhat_bound_multiplier,
            sigma_min=sigma_min,
            force_zero_output=force_zero_output,
            rollout_stochastic=True,
            rollout_noise_scale=rollout_noise_scale,
            rollout_seed=None,
        )
        x_pred = x_pred.reshape(samples, batch_size, *x_pred.shape[1:])
        v_pred = v_pred.reshape(samples, batch_size, *v_pred.shape[1:])
        state_scale = None
        if ur_bin_state_scale_info is not None:
            state_scale = _vpinn_lookup_state_scale(
                ur0,
                scale_info=ur_bin_state_scale_info,
                ur_bin_size=ur_bin_size,
                batch_size=batch_size,
                device=x_pred.device,
                dtype=x_pred.dtype,
            )
        if state_scale is None:
            x_scale = x_pred.new_ones((1, batch_size, 1, 1))
            v_scale = x_pred.new_ones((1, batch_size, 1, 1))
        else:
            x_scale = torch.clamp(state_scale[..., 0:1], min=1e-12).view(1, batch_size, 1, 1)
            v_scale = torch.clamp(state_scale[..., 1:2], min=1e-12).view(1, batch_size, 1, 1)
        x_true_ref = x_true_seq.unsqueeze(0)
        v_true_ref = v_true_seq.unsqueeze(0)
        if mode_key == "stochastic_nll":
            x_pred_scaled = x_pred / x_scale
            v_pred_scaled = v_pred / v_scale
            x_true_scaled = x_true_ref / x_scale
            v_true_scaled = v_true_ref / v_scale
            mu_x = torch.mean(x_pred_scaled, dim=0)
            mu_v = torch.mean(v_pred_scaled, dim=0)
            var_x = torch.mean((x_pred_scaled - mu_x.unsqueeze(0)) ** 2, dim=0)
            var_v = torch.mean((v_pred_scaled - mu_v.unsqueeze(0)) ** 2, dim=0)
            var_x = torch.clamp(var_x, min=1e-6)
            var_v = torch.clamp(var_v, min=1e-6)
            nll_x = 0.5 * (((x_true_scaled - mu_x) ** 2) / var_x + torch.log(var_x))
            nll_v = 0.5 * (((v_true_scaled - mu_v) ** 2) / var_v + torch.log(var_v))
            per = torch.mean(nll_x, dim=(1, 2)) + torch.mean(nll_v, dim=(1, 2))
            return torch.mean(per)
        err_x = (x_pred - x_true_ref) / x_scale
        err_v = (v_pred - v_true_ref) / v_scale
        per_samples = torch.mean(err_x * err_x, dim=(2, 3)) + torch.mean(err_v * err_v, dim=(2, 3))
        return torch.mean(torch.mean(per_samples, dim=0))

    x_pred, v_pred, _, _, _, _ = _vpinn_td_rollout(
        model=model,
        x0=x0,
        v0=v0,
        ur0=ur0,
        td_context0=td0,
        steps=steps,
        dt=dt,
        m=m0,
        c=c0,
        k=k0,
        rho=rho,
        diameter=diameter,
        td_params=td_params,
        mean_active=mean_active,
        sigma_active=sigma_active,
        fhat_active=fhat_active,
        use_td_force_input=use_td_force_input,
        fhat_bound_multiplier=fhat_bound_multiplier,
        sigma_min=sigma_min,
        force_zero_output=force_zero_output,
        rollout_stochastic=False,
        rollout_noise_scale=rollout_noise_scale,
        rollout_seed=None,
    )
    return _vpinn_rollout_state_loss(
        x_pred=x_pred,
        v_pred=v_pred,
        x_true=x_true_seq,
        v_true=v_true_seq,
        ur_values=ur0,
        scale_info=ur_bin_state_scale_info,
        ur_bin_size=ur_bin_size,
    )


def _vpinn_pure_baseline_rollout(
    *,
    x0: torch.Tensor,
    v0: torch.Tensor,
    td_context0: torch.Tensor,
    steps: int,
    dt: float,
    m: torch.Tensor,
    c: torch.Tensor,
    k: torch.Tensor,
    rho: float,
    diameter: float,
    td_params: dict[str, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = x0
    v = v0
    td_context = td_context0
    xs = [x]
    vs = [v]
    total_fs: list[torch.Tensor] = []
    for _ in range(int(steps)):
        td_force_next, td_context_next = td_baseline_step_torch(
            velocity=v,
            acceleration=td_context[:, 0:1],
            td_context=td_context,
            dt=dt,
            rho=rho,
            diameter=diameter,
            params=td_params,
        )
        x, v, a_next = structural_step_constant_force_torch(
            y=x,
            velocity=v,
            force=td_force_next,
            dt=dt,
            mass=m,
            damping_c=c,
            stiffness=k,
        )
        td_context = td_context_next.clone()
        td_context[:, 0:1] = a_next
        xs.append(x)
        vs.append(v)
        total_fs.append(td_force_next)
    empty_force = x0.new_zeros((x0.shape[0], 0, x0.shape[1]))
    return (
        torch.stack(xs, dim=1),
        torch.stack(vs, dim=1),
        torch.stack(total_fs, dim=1) if total_fs else empty_force,
    )


def _build_td_correction_vpinn_datasets(
    *,
    trajs: list[dict[str, np.ndarray]],
    rollout_steps: int,
    window_M: int,
    stride: int,
    td_mass_source: str,
) -> tuple[Dataset, Dataset | None]:
    train_items: list[TensorDataset] = []
    roll_items: list[TensorDataset] = []
    window = int(window_M) + 1
    roll_window = int(rollout_steps) + 1
    for traj in trajs:
        x = torch.from_numpy(np.ascontiguousarray(traj["y"])).float().unsqueeze(1)
        v = torch.from_numpy(np.ascontiguousarray(traj["dy"])).float().unsqueeze(1)
        f_true = torch.from_numpy(np.ascontiguousarray(traj["force_per_m"])).float().unsqueeze(1)
        ur = torch.from_numpy(np.ascontiguousarray(traj["ur"])).float().unsqueeze(1)
        td_context = torch.from_numpy(np.ascontiguousarray(traj["td_context"])).float()
        mass_value = float(np.asarray(traj["dry_mass_kg" if td_mass_source == "dry" else "effective_mass_kg"]).reshape(()))
        damping_value = float(np.asarray(traj["damping_c"]).reshape(()))
        stiffness_value = float(np.asarray(traj["stiffness_n_m"]).reshape(()))
        mass = torch.full((x.shape[0], 1), mass_value, dtype=torch.float32)
        damping = torch.full((x.shape[0], 1), damping_value, dtype=torch.float32)
        stiffness = torch.full((x.shape[0], 1), stiffness_value, dtype=torch.float32)
        if x.shape[0] >= window:
            xw = []
            vw = []
            fw = []
            urw = []
            tdw = []
            mw = []
            dw = []
            kw = []
            for start in range(0, x.shape[0] - window + 1, int(stride)):
                end = start + window
                xw.append(x[start:end])
                vw.append(v[start:end])
                fw.append(f_true[start:end])
                urw.append(ur[start:end])
                tdw.append(td_context[start:end])
                mw.append(mass[start:end])
                dw.append(damping[start:end])
                kw.append(stiffness[start:end])
            train_items.append(
                TensorDataset(
                    torch.stack(xw, dim=0),
                    torch.stack(vw, dim=0),
                    torch.stack(fw, dim=0),
                    torch.stack(urw, dim=0),
                    torch.stack(tdw, dim=0),
                    torch.stack(mw, dim=0),
                    torch.stack(dw, dim=0),
                    torch.stack(kw, dim=0),
                )
            )
        if rollout_steps > 0 and x.shape[0] >= roll_window:
            x0s = []
            v0s = []
            urs = []
            td0s = []
            xtr = []
            vtr = []
            m0s = []
            d0s = []
            k0s = []
            for start in range(0, x.shape[0] - roll_window + 1, int(stride)):
                end = start + roll_window
                x0s.append(x[start])
                v0s.append(v[start])
                urs.append(ur[start])
                td0s.append(td_context[start])
                xtr.append(x[start:end])
                vtr.append(v[start:end])
                m0s.append(mass[start])
                d0s.append(damping[start])
                k0s.append(stiffness[start])
            roll_items.append(
                TensorDataset(
                    torch.stack(x0s, dim=0),
                    torch.stack(v0s, dim=0),
                    torch.stack(urs, dim=0),
                    torch.stack(td0s, dim=0),
                    torch.stack(xtr, dim=0),
                    torch.stack(vtr, dim=0),
                    torch.stack(m0s, dim=0),
                    torch.stack(d0s, dim=0),
                    torch.stack(k0s, dim=0),
                )
            )
    train_ds = train_items[0] if len(train_items) == 1 else ConcatDataset(train_items)
    roll_ds = None
    if roll_items:
        roll_ds = roll_items[0] if len(roll_items) == 1 else ConcatDataset(roll_items)
    return train_ds, roll_ds


def _train_td_correction_vpinn(config: Config, config_name: str) -> None:
    vp = dict(config.vpinn or {})
    data_cfg = config.data
    model_cfg = config.model
    training_cfg = config.training
    optim_cfg = config.optim
    loss_cfg = config.loss
    runtime_cfg = config.runtime
    precision_cfg = config.precision
    compile_cfg = config.compile
    monitoring_cfg = config.monitoring

    correction_mode = resolve_td_correction_mode(vp)
    mode_flags = td_correction_mode_flags(correction_mode)
    mean_active = bool(mode_flags["mean_active"])
    probabilistic = bool(mode_flags["sigma_active"])
    fhat_active = bool(mode_flags["fhat_active"])
    force_zero_output = bool(vp.get("force_zero_output", False))
    sigma_min = float(vp.get("sigma_min", 1e-6))
    rollout_stochastic = bool(vp.get("rollout_stochastic", False))
    rollout_noise_scale = float(vp.get("rollout_noise_scale", 1.0))
    if not np.isfinite(rollout_noise_scale) or rollout_noise_scale < 0.0:
        raise ValueError("vpinn.rollout_noise_scale must be finite and non-negative.")
    rollout_seed_raw = vp.get("rollout_seed", None)
    rollout_seed = None if rollout_seed_raw is None else int(rollout_seed_raw)
    corr_init_mode, corr_init_tiny_std = _resolve_corr_init_settings(vp, model_cfg)
    if sigma_min < 0.0:
        raise ValueError("vpinn.sigma_min must be non-negative.")
    device = select_device(os.getenv("TRAIN_DEVICE", str(runtime_cfg.device)))
    print(f"Using device: {device}")
    configure_tf32(device, bool(precision_cfg.use_tf32))
    set_num_threads_from_slurm(default=1)
    non_blocking = device.type == "cuda"

    train_series_root = Path(data_cfg.train_series_dir)
    train_dir = train_series_root / "train"
    val_unseen_dir = train_series_root / "val_unseen"
    legacy_val_dir = train_series_root / "val"
    if not val_unseen_dir.exists():
        val_unseen_dir = legacy_val_dir
    val_seen_dir = train_series_root / "val_seen"
    train_paths = sorted(train_dir.glob("*.npz"))
    val_paths = sorted(val_unseen_dir.glob("*.npz"))
    val_seen_paths = sorted(val_seen_dir.glob("*.npz")) if val_seen_dir.exists() else []
    if not train_paths:
        raise FileNotFoundError("No TD correction VPINN training trajectories were found.")
    td_mass_source = str(vp.get("td_mass_source", "dry")).strip().lower()
    if td_mass_source not in {"dry", "effective"}:
        raise ValueError("vpinn.td_mass_source must be one of: dry, effective.")
    train_cut = float(getattr(data_cfg, "cut_start_seconds_train", 0.0) or 0.0)
    val_cut = float(getattr(data_cfg, "cut_start_seconds_val", 0.0) or 0.0)
    reduce_time_enabled = bool(getattr(data_cfg, "reduce_time", False))
    reduction_factor = int(getattr(data_cfg, "reduction_factor", 1))
    stagger_train_reduce = bool(
        getattr(
            data_cfg,
            "stagger_reduced_time_train",
            reduce_time_enabled and max(1, reduction_factor) > 1,
        )
    )
    stagger_val_reduce = bool(getattr(data_cfg, "stagger_reduced_time_val", False))
    train_trajs = load_td_correction_trajectories(
        paths=train_paths,
        cut_start_seconds=train_cut,
        reduce_time=reduce_time_enabled,
        reduction_factor=reduction_factor,
        stagger_reduced_time=stagger_train_reduce,
        ur_source=td_mass_source,
    )
    val_trajs = (
        load_td_correction_trajectories(
            paths=val_paths,
            cut_start_seconds=val_cut,
            reduce_time=reduce_time_enabled,
            reduction_factor=reduction_factor,
            stagger_reduced_time=stagger_val_reduce,
            ur_source=td_mass_source,
        )
        if val_paths
        else []
    )
    val_seen_trajs = (
        load_td_correction_trajectories(
            paths=val_seen_paths,
            cut_start_seconds=val_cut,
            reduce_time=reduce_time_enabled,
            reduction_factor=reduction_factor,
            stagger_reduced_time=stagger_val_reduce,
            ur_source=td_mass_source,
        )
        if val_seen_paths
        else []
    )

    dt = float(train_trajs[0]["t"][1] - train_trajs[0]["t"][0])
    rho = float(getattr(model_cfg, "rho", 1000.0))
    diameter = float(getattr(model_cfg, "D", 0.1))
    td_params = resolve_td_correction_params(vp)
    use_td_force_input = bool(vp.get("use_td_force_input", False))
    use_acceleration_input = bool(vp.get("use_acceleration_input", False))
    phase_input_source = resolve_td_phase_input_source(
        vp.get("phi_input_source", vp.get("use_phi_input", False))
    )
    use_phi_input = phase_input_source != "none"
    use_sigma_inputs = bool(vp.get("use_sigma_inputs", False))
    fhat_bound_multiplier = float(vp.get("fhat_bound_multiplier", 1.5))
    if not np.isfinite(fhat_bound_multiplier) or fhat_bound_multiplier <= 0.0:
        raise ValueError("vpinn.fhat_bound_multiplier must be finite and positive.")
    if fhat_active and use_td_force_input:
        raise ValueError("vpinn.use_td_force_input=true is invalid for correction_mode values that include fhat correction.")
    hard_force_symmetry = bool(getattr(config.architecture, "hard_force_symmetry", False))
    if hard_force_symmetry and (use_td_force_input or use_acceleration_input or use_phi_input or use_sigma_inputs):
        raise ValueError(
            "architecture.hard_force_symmetry requires vpinn.use_td_force_input=false, "
            "vpinn.use_acceleration_input=false, vpinn.use_phi_input=false, and vpinn.use_sigma_inputs=false because those "
            "auxiliary inputs do not have a defined sign-flip symmetry."
        )

    d = int(np.asarray(train_trajs[0]["y"]).ndim == 1)
    input_dim = _vpinn_input_dim(
        d=d,
        use_td_force_input=use_td_force_input,
        use_acceleration_input=use_acceleration_input,
        use_phi_input=use_phi_input,
        use_sigma_inputs=use_sigma_inputs,
    )
    output_dim = _vpinn_output_dim(mean_active=mean_active, sigma_active=probabilistic, fhat_active=fhat_active, d=d)
    model = _build_force_model(
        config,
        input_dim=input_dim,
        output_dim=output_dim,
        mean_output_dim=1,
    ).to(device)
    setattr(model, "use_td_force_input", use_td_force_input)
    setattr(model, "use_acceleration_input", use_acceleration_input)
    setattr(model, "use_phi_input", use_phi_input)
    setattr(model, "phi_input_source", None if not use_phi_input else phase_input_source)
    setattr(model, "use_sigma_inputs", use_sigma_inputs)
    setattr(model, "correction_mode", correction_mode)
    setattr(model, "fhat_bound_multiplier", float(fhat_bound_multiplier))
    setattr(model, "force_zero_output", force_zero_output)
    setattr(model, "force_representation", str(vp.get("force_representation", "force")).strip().lower())
    setattr(model, "hard_force_symmetry", hard_force_symmetry)
    _apply_corr_head_init(
        model,
        mode=corr_init_mode,
        tiny_std=corr_init_tiny_std,
        probabilistic=probabilistic,
        sigma_min=sigma_min,
    )
    model = maybe_compile_model(model, bool(compile_cfg.use_compile), str(compile_cfg.compile_mode))

    rollout_weight = float(vp.get("rollout_force_weight", float(getattr(loss_cfg, "rollout_det_weight", 0.0))))
    rollout_steps = int(vp.get("rollout_force_steps", int(getattr(loss_cfg, "rollout_det_steps", 0))))
    rollout_loss_mode = str(getattr(loss_cfg, "rollout_loss_mode", "deterministic")).strip().lower()
    rollout_stochastic_samples = int(getattr(loss_cfg, "rollout_stochastic_samples", 1))
    rollout_steps_final_raw = int(getattr(loss_cfg, "rollout_det_steps_final", 0))
    rollout_steps_warmup_epochs = int(getattr(loss_cfg, "rollout_det_steps_warmup_epochs", 0))
    rollout_steps_final = rollout_steps if rollout_steps_final_raw <= 0 else rollout_steps_final_raw
    rollout_batch_size_raw = int(vp.get("rollout_force_batch_size", int(getattr(loss_cfg, "rollout_det_batch_size", 0))))
    rollout_batch_size = int(training_cfg.batch_size) if rollout_batch_size_raw <= 0 else rollout_batch_size_raw
    if rollout_loss_mode == "stochastic":
        rollout_loss_mode = "stochastic_nll"
    if rollout_loss_mode not in {"deterministic", "stochastic_nll", "stochastic_mse"}:
        raise ValueError("loss.rollout_loss_mode must be one of: deterministic, stochastic_nll, stochastic_mse.")
    if rollout_stochastic_samples < 1:
        raise ValueError("loss.rollout_stochastic_samples must be >= 1.")
    if rollout_steps < 0:
        raise ValueError("loss.rollout_det_steps must be non-negative.")
    if rollout_weight < 0.0:
        raise ValueError("loss.rollout_det_weight must be non-negative.")
    if rollout_steps_final < 0:
        raise ValueError("loss.rollout_det_steps_final must be non-negative.")
    if rollout_steps_warmup_epochs < 0:
        raise ValueError("loss.rollout_det_steps_warmup_epochs must be non-negative.")
    if rollout_loss_mode in {"stochastic_nll", "stochastic_mse"} and rollout_weight > 0.0 and rollout_stochastic_samples < 2:
        raise ValueError(
            "loss.rollout_stochastic_samples must be >= 2 when loss.rollout_loss_mode is stochastic_nll or stochastic_mse."
        )
    if rollout_loss_mode in {"stochastic_nll", "stochastic_mse"} and rollout_weight > 0.0 and not probabilistic:
        raise ValueError(
            "VPINN rollout stochastic loss modes require a correction_mode that includes the sigma head."
        )
    if rollout_weight > 0.0 and rollout_steps < 1 and rollout_steps_final < 1:
        raise ValueError(
            "loss.rollout_det_steps or loss.rollout_det_steps_final must be >= 1 when loss.rollout_det_weight > 0."
        )
    window_M = int(vp.get("window_M", getattr(loss_cfg, "window_M", 50)))
    stride = max(1, int(vp.get("stride", getattr(loss_cfg, "stride", 1))))
    train_dataset, _ = _build_td_correction_vpinn_datasets(
        trajs=train_trajs,
        rollout_steps=0,
        window_M=window_M,
        stride=stride,
        td_mass_source=td_mass_source,
    )
    val_dataset = None
    if val_trajs:
        val_dataset, _ = _build_td_correction_vpinn_datasets(
            trajs=val_trajs,
            rollout_steps=0,
            window_M=window_M,
            stride=stride,
            td_mass_source=td_mass_source,
        )
    val_seen_dataset = None
    if val_seen_trajs:
        val_seen_dataset, _ = _build_td_correction_vpinn_datasets(
            trajs=val_seen_trajs,
            rollout_steps=0,
            window_M=window_M,
            stride=stride,
            td_mass_source=td_mass_source,
        )

    # TD-correction datasets are prebuilt in memory, so worker processes mainly add
    # fork/CUDA risk on clusters without meaningful throughput benefit.
    td_loader_num_workers = 0
    train_loader = DataLoader(train_dataset, batch_size=int(training_cfg.batch_size), shuffle=True, num_workers=td_loader_num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_dataset, batch_size=int(training_cfg.batch_size), shuffle=False, num_workers=td_loader_num_workers, pin_memory=(device.type == "cuda")) if val_dataset is not None else None
    val_seen_loader = DataLoader(val_seen_dataset, batch_size=int(training_cfg.batch_size), shuffle=False, num_workers=td_loader_num_workers, pin_memory=(device.type == "cuda")) if val_seen_dataset is not None else None
    current_rollout_steps = _scheduled_rollout_det_steps(
        epoch=0,
        base_steps=rollout_steps,
        final_steps=rollout_steps_final,
        warmup_epochs=rollout_steps_warmup_epochs,
    )

    def _rebuild_rollout_loaders(steps: int) -> tuple[Dataset | None, DataLoader | None, Dataset | None, DataLoader | None, Dataset | None, DataLoader | None]:
        train_roll_ds = None
        train_roll_loader = None
        val_roll_ds = None
        val_roll_loader = None
        val_seen_roll_ds = None
        val_seen_roll_loader = None
        if rollout_weight > 0.0 and steps > 0:
            train_roll_ds = _build_td_correction_vpinn_datasets(
                trajs=train_trajs,
                rollout_steps=steps,
                window_M=window_M,
                stride=stride,
                td_mass_source=td_mass_source,
            )[1]
            if train_roll_ds is not None:
                train_roll_loader = DataLoader(
                    train_roll_ds,
                    batch_size=rollout_batch_size,
                    shuffle=True,
                    num_workers=td_loader_num_workers,
                    pin_memory=(device.type == "cuda"),
                )
            if val_trajs:
                val_roll_ds = _build_td_correction_vpinn_datasets(
                    trajs=val_trajs,
                    rollout_steps=steps,
                    window_M=window_M,
                    stride=stride,
                    td_mass_source=td_mass_source,
                )[1]
                if val_roll_ds is not None:
                    val_roll_loader = DataLoader(
                        val_roll_ds,
                        batch_size=rollout_batch_size,
                        shuffle=False,
                        num_workers=td_loader_num_workers,
                        pin_memory=(device.type == "cuda"),
                    )
            if val_seen_trajs:
                val_seen_roll_ds = _build_td_correction_vpinn_datasets(
                    trajs=val_seen_trajs,
                    rollout_steps=steps,
                    window_M=window_M,
                    stride=stride,
                    td_mass_source=td_mass_source,
                )[1]
                if val_seen_roll_ds is not None:
                    val_seen_roll_loader = DataLoader(
                        val_seen_roll_ds,
                        batch_size=rollout_batch_size,
                        shuffle=False,
                        num_workers=td_loader_num_workers,
                        pin_memory=(device.type == "cuda"),
                    )
        return train_roll_ds, train_roll_loader, val_roll_ds, val_roll_loader, val_seen_roll_ds, val_seen_roll_loader

    train_rollout_dataset, rollout_loader, val_rollout_dataset, val_rollout_loader, val_seen_rollout_dataset, val_seen_rollout_loader = _rebuild_rollout_loaders(current_rollout_steps)

    opt, lr_scheduler = setup_optimizer_and_scheduler(model, optim_cfg=optim_cfg, scheduler_cfg=optim_cfg.scheduler, epochs=int(training_cfg.epochs))
    amp_enabled, amp_dtype, scaler = setup_amp(device, use_amp=bool(precision_cfg.use_amp), amp_dtype=str(precision_cfg.amp_dtype))
    writer, run_name = setup_writer(
        config.logging.run_dir_root,
        config_name,
        run_name_override=getattr(config.logging, "run_name", None),
        append_timestamp=bool(getattr(config.logging, "append_timestamp", True)),
    )
    writer.add_text("vpinn/td_correction_config", json.dumps(vp, indent=2, sort_keys=True), 0)
    writer.flush()
    run_models_dir = Path("models") / run_name
    run_models_dir.mkdir(parents=True, exist_ok=True)
    validation_models_dir = run_models_dir / "async_validation"
    validation_models_dir.mkdir(parents=True, exist_ok=True)

    def _save_td_validation_checkpoint(epoch_idx: int) -> Path:
        ckpt_path = validation_models_dir / f"model_epoch_{epoch_idx + 1:06d}.pt"
        latest_path = validation_models_dir / "model.pt"
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
                "td_correction": True,
                "ur_bin_state_scale_info": ur_bin_state_scale_info,
                "correction_mode": correction_mode,
                "predict_sigma": probabilistic,
                "mean_active": mean_active,
                "fhat_active": fhat_active,
                "use_td_force_input": use_td_force_input,
                "use_phi_input": use_phi_input,
                "phi_input_source": (None if not use_phi_input else phase_input_source),
                "use_sigma_inputs": use_sigma_inputs,
                "fhat_bound_multiplier": float(fhat_bound_multiplier),
                "fhat_reg": float(fhat_reg),
                "fhat_reg_norm": str(fhat_reg_norm),
            },
            ckpt_path,
        )
        shutil.copyfile(ckpt_path, latest_path)
        return ckpt_path

    mean_reg = float(getattr(loss_cfg, "mean_reg", 0.0))
    sigma_reg = float(getattr(loss_cfg, "sigma_reg", 0.0))
    mean_reg_norm = str(getattr(loss_cfg, "mean_reg_norm", "l1")).strip().lower()
    sigma_reg_norm = str(getattr(loss_cfg, "sigma_reg_norm", "l2")).strip().lower()
    ur_bin_size = float(getattr(loss_cfg, "ur_bin_size", 1.0e-6))
    normalize_by_ur_bin_std = bool(getattr(loss_cfg, "normalize_by_ur_bin_std", False))
    ur_bin_scale_eps = float(getattr(loss_cfg, "ur_bin_scale_eps", 1.0e-6))
    ur_bin_state_scale_info: dict[str, Any] | None = None
    if normalize_by_ur_bin_std:
        ur_bin_state_scale_info = _build_vpinn_ur_bin_state_scale_info(
            train_trajs,
            ur_bin_size=ur_bin_size,
            eps=ur_bin_scale_eps,
        )
    use_force_loss = bool(vp.get("use_force_loss", getattr(loss_cfg, "use_force_loss", True)))
    use_weak_loss = bool(vp.get("use_weak_loss", getattr(loss_cfg, "use_weak_loss", True)))
    if not (use_force_loss or use_weak_loss):
        raise ValueError("VPINN TD-correction training requires at least one of use_force_loss/use_weak_loss.")
    w, wdot, alpha = _test_functions(
        window_M,
        dt,
        num_poly=int(vp.get("num_poly_test", getattr(loss_cfg, "num_poly_test", 2))),
        num_sine=int(vp.get("num_sine_test", getattr(loss_cfg, "num_sine_test", 0))),
    )
    w = w.to(device)
    wdot = wdot.to(device)
    alpha = alpha.to(device)

    def _reg(val: torch.Tensor, norm: str) -> torch.Tensor:
        return torch.mean(torch.abs(val)) if str(norm).strip().lower() == "l1" else torch.mean(val * val)

    epochs = int(training_cfg.epochs)
    validate_every = max(1, int(getattr(monitoring_cfg, "validate_every_epochs", 1)))
    log_every = max(1, int(getattr(monitoring_cfg, "log_every_epochs", 1)))
    print_every = max(1, int(getattr(monitoring_cfg, "print_every_epochs", 1)))
    rollout_every = validate_every
    validation_samples_per_ur = max(1, int(getattr(monitoring_cfg, "validation_samples_per_ur", 1)))
    async_validation = bool(getattr(monitoring_cfg, "async_validation", False))
    async_device = str(getattr(monitoring_cfg, "async_validation_device", "cpu"))
    async_num_workers = int(getattr(monitoring_cfg, "async_validation_num_workers", 0))
    async_num_threads = int(getattr(monitoring_cfg, "async_validation_num_threads", 4))
    async_max_concurrent = int(getattr(monitoring_cfg, "async_validation_max_concurrent", 1))
    train_instances = len(train_dataset)
    train_steps_per_epoch = len(train_loader)
    val_instances = len(val_dataset) if val_dataset is not None else 0
    val_steps_per_epoch = len(val_loader) if val_loader is not None else 0
    val_seen_instances = len(val_seen_dataset) if val_seen_dataset is not None else 0
    val_seen_steps_per_epoch = len(val_seen_loader) if val_seen_loader is not None else 0
    train_rollout_instances = len(train_rollout_dataset) if train_rollout_dataset is not None else 0
    train_rollout_steps_per_epoch = len(rollout_loader) if rollout_loader is not None else 0
    val_rollout_instances = len(val_rollout_dataset) if val_rollout_dataset is not None else 0
    val_rollout_steps_per_epoch = len(val_rollout_loader) if val_rollout_loader is not None else 0
    final_rollout_all_validation = bool(getattr(monitoring_cfg, "final_rollout_all_validation", False))

    startup_lines = [
        f"Run name: {run_name}",
        (
            f"VPINN TD-correction setup: epochs={epochs}, batch_size={int(training_cfg.batch_size)}, "
            f"steps_per_epoch={train_steps_per_epoch}, train_instances={train_instances}, "
            f"train_trajectories={len(train_trajs)}, correction_mode={correction_mode}"
        ),
        (
            f"Validation setup: unseen_steps={val_steps_per_epoch}, unseen_instances={val_instances}, "
            f"val_unseen_trajectories={len(val_trajs)}, seen_steps={val_seen_steps_per_epoch}, "
            f"seen_instances={val_seen_instances}, val_seen_trajectories={len(val_seen_trajs)}"
        ),
        (
            f"Rollout setup: weight={rollout_weight:g}, steps={rollout_steps}, "
            f"train_rollout_windows={train_rollout_instances}, train_rollout_steps={train_rollout_steps_per_epoch}, "
            f"val_rollout_windows={val_rollout_instances}, val_rollout_steps={val_rollout_steps_per_epoch}"
        ),
        (
            f"Monitoring: validate_every={validate_every}, rollout_every={rollout_every}, "
            f"print_every={print_every}, async_validation={async_validation}, "
            f"val_samples_per_ur={validation_samples_per_ur}"
        ),
        (
            f"Runtime: device={device}, num_workers={int(runtime_cfg.num_workers)} "
            f"(td_loader_workers=0), amp={amp_enabled}, "
            f"compile={bool(compile_cfg.use_compile)}, lr={float(optim_cfg.lr):g}"
        ),
        (
            f"Rollout sampling: stochastic={rollout_stochastic}, "
            f"noise_scale={rollout_noise_scale:g}, seed={rollout_seed}"
        ),
    ]
    if rollout_weight > 0.0:
        rollout_mode_msg = rollout_loss_mode
        if rollout_loss_mode in {"stochastic_nll", "stochastic_mse"}:
            rollout_mode_msg = f"{rollout_loss_mode} (K={rollout_stochastic_samples})"
        startup_lines.append(
            "Rollout loss mode: "
            f"{rollout_mode_msg}, batch_size={rollout_batch_size}, "
            f"scheduled_steps={current_rollout_steps}"
        )
        if rollout_steps_final > rollout_steps and rollout_steps_warmup_epochs > 0:
            startup_lines.append(
                "Rollout schedule: "
                f"{rollout_steps}->{rollout_steps_final} over {rollout_steps_warmup_epochs} epoch(s)"
            )
    if normalize_by_ur_bin_std:
        startup_lines.append(
            "U_r loss scaling: "
            f"enabled=true, bin_size={ur_bin_size:g}, eps={ur_bin_scale_eps:g}"
        )
    print("\n".join(startup_lines))
    async_dir = Path(writer.log_dir) / "async_validation"
    if async_validation:
        async_dir.mkdir(parents=True, exist_ok=True)
    async_processes: list[dict[str, Any]] = []
    async_best_state: dict[str, Any] = {"loss_total": float("inf")}

    def _run_sync_validation_for_split(
        *,
        epoch_idx: int,
        split_tag: str,
        split_name: str,
        split_loader: DataLoader | None,
        split_rollout_loader: DataLoader | None,
        split_trajs: list[dict[str, Any]],
        log_rollout_plots: bool,
    ) -> None:
        if split_loader is None:
            return
        split_start = time.perf_counter()
        model.eval()
        val_sums = {name: torch.zeros((), device=device) for name in ["loss_total", "loss_data", "loss_physics", "loss_reg_mean", "loss_reg_sigma", "loss_reg_fhat"]}
        val_batches = 0
        with torch.no_grad():
            for batch in split_loader:
                x_win, v_win, force_true, ur_win, td_win, m_win, c_win, k_win = [item.to(device, non_blocking=non_blocking) for item in batch]
                with torch.amp.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                    step = _vpinn_step_with_corrections(
                        model=model,
                        x=x_win,
                        v=v_win,
                        ur=ur_win,
                        td_context=td_win,
                        dt=dt,
                        m=m_win,
                        c=c_win,
                        k=k_win,
                        rho=rho,
                        diameter=diameter,
                        td_params=td_params,
                        mean_active=mean_active,
                        sigma_active=probabilistic,
                        fhat_active=fhat_active,
                        use_td_force_input=use_td_force_input,
                        fhat_bound_multiplier=fhat_bound_multiplier,
                        sigma_min=sigma_min,
                        force_zero_output=force_zero_output,
                    )
                    mu_corr = step["corr_mu"]
                    sigma_corr = step["corr_sigma"]
                    total_force_mu = step["total_force_next"]
                    if use_force_loss:
                        if probabilistic:
                            var = torch.clamp(sigma_corr * sigma_corr, min=1e-9)
                            loss_data = torch.mean(0.5 * (((force_true - total_force_mu) ** 2) / var + torch.log(var)))
                        else:
                            loss_data = torch.mean((force_true - total_force_mu) ** 2)
                    else:
                        loss_data = mu_corr.new_tensor(0.0)
                    if use_weak_loss:
                        R_mean = _weak_residual(
                            x=x_win,
                            v=v_win,
                            f_pred=total_force_mu,
                            m=m_win[:, 0, :],
                            c=c_win[:, 0, :],
                            k=k_win[:, 0, :],
                            dt=dt,
                            w=w,
                            wdot=wdot,
                            alpha=alpha,
                            f0=None,
                        )
                        p_scale = None
                        if normalize_by_ur_bin_std:
                            R_mean, p_scale = _vpinn_scale_weak_residual(
                                R_mean,
                                ur_values=ur_win[:, 0, :],
                                mass=m_win[:, 0, :],
                                scale_info=ur_bin_state_scale_info,
                                ur_bin_size=ur_bin_size,
                            )
                        if probabilistic:
                            coeff = (float(dt) * alpha.view(1, 1, -1, 1) * w.view(1, w.shape[0], w.shape[1], 1)) ** 2
                            weak_var = torch.sum(coeff * (sigma_corr.unsqueeze(1) ** 2), dim=2)
                            if p_scale is not None:
                                weak_var = weak_var / torch.clamp(p_scale.unsqueeze(1) ** 2, min=1e-12)
                            weak_var = torch.clamp(weak_var, min=1e-9)
                            loss_physics = torch.mean(0.5 * ((R_mean * R_mean) / weak_var + torch.log(weak_var)))
                        else:
                            loss_physics = torch.mean(R_mean * R_mean)
                    else:
                        loss_physics = mu_corr.new_tensor(0.0)
                    mean_reg_loss = _reg(mu_corr, mean_reg_norm)
                    sigma_reg_loss = _reg(sigma_corr, sigma_reg_norm) if probabilistic else mu_corr.new_tensor(0.0)
                    fhat_reg_loss = _reg(step["delta_fhat"], str(getattr(loss_cfg, "fhat_reg_norm", "l2"))) if fhat_active else mu_corr.new_tensor(0.0)
                    total_loss = (
                        loss_data
                        + loss_physics
                        + float(mean_reg) * mean_reg_loss
                        + float(sigma_reg) * sigma_reg_loss
                        + float(getattr(loss_cfg, "fhat_reg", 0.0)) * fhat_reg_loss
                    )
                val_sums["loss_total"] += total_loss.detach()
                val_sums["loss_data"] += loss_data.detach()
                val_sums["loss_physics"] += loss_physics.detach()
                val_sums["loss_reg_mean"] += mean_reg_loss.detach()
                val_sums["loss_reg_sigma"] += sigma_reg_loss.detach()
                val_sums["loss_reg_fhat"] += fhat_reg_loss.detach()
                val_batches += 1
        val_denom = float(max(1, val_batches))
        val_metrics = {
            name: float((value / val_denom).detach().cpu()) for name, value in val_sums.items()
        }
        rollout_loss_avg = 0.0
        if split_rollout_loader is not None and rollout_weight > 0.0 and current_rollout_steps > 0:
            with torch.no_grad():
                roll_sum = torch.zeros((), device=device)
                roll_batches = 0
                for rb in split_rollout_loader:
                    roll_sum += _vpinn_td_rollout_loss_from_batch(
                        model=model,
                        batch=rb,
                        device=device,
                        non_blocking=non_blocking,
                        dt=dt,
                        rho=rho,
                        diameter=diameter,
                        td_params=td_params,
                        mean_active=mean_active,
                        sigma_active=probabilistic,
                        fhat_active=fhat_active,
                        use_td_force_input=use_td_force_input,
                        fhat_bound_multiplier=fhat_bound_multiplier,
                        sigma_min=sigma_min,
                        force_zero_output=force_zero_output,
                        rollout_loss_mode="deterministic",
                        rollout_stochastic_samples=1,
                        rollout_noise_scale=rollout_noise_scale,
                        ur_bin_state_scale_info=(ur_bin_state_scale_info if normalize_by_ur_bin_std else None),
                        ur_bin_size=ur_bin_size,
                    ).detach()
                    roll_batches += 1
                rollout_loss_avg = float((roll_sum / float(max(1, roll_batches))).detach().cpu())
                writer.add_scalar(f"{split_tag}/loss_rollout_det", rollout_loss_avg, epoch_idx + 1)
        val_metrics["loss_total"] = (
            val_metrics["loss_data"]
            + val_metrics["loss_physics"]
            + float(mean_reg) * val_metrics["loss_reg_mean"]
            + float(sigma_reg) * val_metrics["loss_reg_sigma"]
            + float(getattr(loss_cfg, "fhat_reg", 0.0)) * val_metrics["loss_reg_fhat"]
            + float(rollout_weight) * rollout_loss_avg
        )
        for name, value in val_metrics.items():
            writer.add_scalar(f"{split_tag}/{name}", value, epoch_idx + 1)

        if should_rollout and split_trajs:
            ur_all = [float(traj["ur"][0]) for traj in split_trajs]
            sampled_metric_indices = sample_indices_per_ur(
                ur_all,
                samples_per_ur=validation_samples_per_ur,
                seed=1,
            )
            sampled_names = [str(split_trajs[idx].get("name", f"traj_{idx}")) for idx in sampled_metric_indices]
            print(
                f"[td-{split_name}][vpinn] epoch {epoch_idx + 1}: sampled metric trajectories={sampled_names} "
                f"(force_zero_output={force_zero_output}, mass_source={td_mass_source})"
            )
            metrics_sum: dict[str, float] = {}
            metrics_count: dict[str, int] = {}
            for sidx in sampled_metric_indices:
                metrics_roll = _log_td_correction_rollout_validation(
                    writer=writer,
                    epoch=epoch_idx + 1,
                    model=model,
                    traj=split_trajs[sidx],
                    dt=dt,
                    td_mass_source=td_mass_source,
                    rho=rho,
                    diameter=diameter,
                    td_params=td_params,
                    device=device,
                    sigma_min=sigma_min,
                    mean_active=mean_active,
                    probabilistic=probabilistic,
                    fhat_active=fhat_active,
                    use_td_force_input=use_td_force_input,
                    fhat_bound_multiplier=fhat_bound_multiplier,
                    force_zero_output=force_zero_output,
                    rollout_stochastic=rollout_stochastic,
                    rollout_noise_scale=rollout_noise_scale,
                    rollout_seed=rollout_seed,
                    tag_prefix=f"{split_tag}/rollout",
                    log_metrics=False,
                    log_plots=False,
                )
                for name, value in metrics_roll.items():
                    if not np.isfinite(float(value)):
                        continue
                    metrics_sum[name] = metrics_sum.get(name, 0.0) + float(value)
                    metrics_count[name] = metrics_count.get(name, 0) + 1
            for name, total in metrics_sum.items():
                denom_roll = float(max(1, metrics_count.get(name, 0)))
                writer.add_scalar(f"{split_tag}/{name}", total / denom_roll, epoch_idx + 1)

            if log_rollout_plots:
                selected_indices = sample_one_index_per_ur(ur_all, seed=0)
                if not selected_indices:
                    selected_indices = list(range(len(split_trajs)))
                rollout_idx = selected_indices[0]
                rollout_traj = split_trajs[rollout_idx]
                print(
                    f"[td-{split_name}][vpinn] epoch {epoch_idx + 1}: plot trajectory={rollout_traj.get('name', f'traj_{rollout_idx}')} "
                    f"U_r={float(np.asarray(rollout_traj['ur']).reshape(-1)[0]):.6g} "
                    f"dt={float(dt):.6g} rho={float(rho):.6g} D={float(diameter):.6g} "
                    f"m={float(np.asarray(rollout_traj['dry_mass_kg' if td_mass_source == 'dry' else 'effective_mass_kg']).reshape(())):.6g} "
                    f"c={float(np.asarray(rollout_traj['damping_c']).reshape(())):.6g} "
                    f"k={float(np.asarray(rollout_traj['stiffness_n_m']).reshape(())):.6g}"
                )
                _log_td_correction_rollout_validation(
                    writer=writer,
                    epoch=epoch_idx + 1,
                    model=model,
                    traj=rollout_traj,
                    dt=dt,
                    td_mass_source=td_mass_source,
                    rho=rho,
                    diameter=diameter,
                    td_params=td_params,
                    device=device,
                    sigma_min=sigma_min,
                    mean_active=mean_active,
                    probabilistic=probabilistic,
                    fhat_active=fhat_active,
                    use_td_force_input=use_td_force_input,
                    fhat_bound_multiplier=fhat_bound_multiplier,
                    force_zero_output=force_zero_output,
                    rollout_stochastic=rollout_stochastic,
                    rollout_noise_scale=rollout_noise_scale,
                    rollout_seed=rollout_seed,
                    tag_prefix=f"{split_tag}/rollout",
                    log_metrics=False,
                    log_plots=True,
                    log_spectra=True,
                )
        elapsed = float(time.perf_counter() - split_start)
        writer.add_scalar(f"{split_tag}/validation_wall_time_s", elapsed, epoch_idx + 1)
        writer.flush()

    for epoch in range(epochs):
        model.train()
        if bool(optim_cfg.use_lr_scheduler):
            for group in opt.param_groups:
                group["lr"] = lr_scheduler.get_lr(epoch)
        scheduled_rollout_steps = _scheduled_rollout_det_steps(
            epoch=epoch,
            base_steps=rollout_steps,
            final_steps=rollout_steps_final,
            warmup_epochs=rollout_steps_warmup_epochs,
        )
        if scheduled_rollout_steps != current_rollout_steps:
            current_rollout_steps = scheduled_rollout_steps
            train_rollout_dataset, rollout_loader, val_rollout_dataset, val_rollout_loader, val_seen_rollout_dataset, val_seen_rollout_loader = _rebuild_rollout_loaders(current_rollout_steps)
            print(
                f"Epoch {epoch}: updated rollout loss horizon to {current_rollout_steps} step(s)."
            )
        sums = {name: torch.zeros((), device=device) for name in ["loss_total", "loss_data", "loss_physics", "loss_reg_mean", "loss_reg_sigma", "loss_reg_fhat", "loss_rollout_det", "grad_norm"]}
        batches = 0
        rollout_iter = iter(rollout_loader) if rollout_loader is not None else None
        for batch in train_loader:
            x_win, v_win, force_true, ur_win, td_win, m_win, c_win, k_win = batch
            x_win = x_win.to(device, non_blocking=non_blocking)
            v_win = v_win.to(device, non_blocking=non_blocking)
            force_true = force_true.to(device, non_blocking=non_blocking)
            ur_win = ur_win.to(device, non_blocking=non_blocking)
            td_win = td_win.to(device, non_blocking=non_blocking)
            m_win = m_win.to(device, non_blocking=non_blocking)
            c_win = c_win.to(device, non_blocking=non_blocking)
            k_win = k_win.to(device, non_blocking=non_blocking)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                step = _vpinn_step_with_corrections(
                    model=model,
                    x=x_win,
                    v=v_win,
                    ur=ur_win,
                    td_context=td_win,
                    dt=dt,
                    m=m_win,
                    c=c_win,
                    k=k_win,
                    rho=rho,
                    diameter=diameter,
                    td_params=td_params,
                    mean_active=mean_active,
                    sigma_active=probabilistic,
                    fhat_active=fhat_active,
                    use_td_force_input=use_td_force_input,
                    fhat_bound_multiplier=fhat_bound_multiplier,
                    sigma_min=sigma_min,
                    force_zero_output=force_zero_output,
                )
                mu_corr = step["corr_mu"]
                sigma_corr = step["corr_sigma"]
                total_force_mu = step["total_force_next"]
                if use_force_loss:
                    if probabilistic:
                        var = torch.clamp(sigma_corr * sigma_corr, min=1e-9)
                        loss_data = torch.mean(0.5 * (((force_true - total_force_mu) ** 2) / var + torch.log(var)))
                    else:
                        loss_data = torch.mean((force_true - total_force_mu) ** 2)
                else:
                    loss_data = mu_corr.new_tensor(0.0)
                if use_weak_loss:
                    R_mean = _weak_residual(
                        x=x_win,
                        v=v_win,
                        f_pred=total_force_mu,
                        m=m_win[:, 0, :],
                        c=c_win[:, 0, :],
                        k=k_win[:, 0, :],
                        dt=dt,
                        w=w,
                        wdot=wdot,
                        alpha=alpha,
                        f0=None,
                    )
                    p_scale = None
                    if normalize_by_ur_bin_std:
                        R_mean, p_scale = _vpinn_scale_weak_residual(
                            R_mean,
                            ur_values=ur_win[:, 0, :],
                            mass=m_win[:, 0, :],
                            scale_info=ur_bin_state_scale_info,
                            ur_bin_size=ur_bin_size,
                        )
                    if probabilistic:
                        coeff = (float(dt) * alpha.view(1, 1, -1, 1) * w.view(1, w.shape[0], w.shape[1], 1)) ** 2
                        weak_var = torch.sum(coeff * (sigma_corr.unsqueeze(1) ** 2), dim=2)
                        if p_scale is not None:
                            weak_var = weak_var / torch.clamp(p_scale.unsqueeze(1) ** 2, min=1e-12)
                        weak_var = torch.clamp(weak_var, min=1e-9)
                        loss_physics = torch.mean(0.5 * ((R_mean * R_mean) / weak_var + torch.log(weak_var)))
                    else:
                        loss_physics = torch.mean(R_mean * R_mean)
                else:
                    loss_physics = mu_corr.new_tensor(0.0)
                mean_reg_loss = _reg(mu_corr, mean_reg_norm)
                sigma_reg_loss = _reg(sigma_corr, sigma_reg_norm) if probabilistic else mu_corr.new_tensor(0.0)
                fhat_reg_loss = _reg(step["delta_fhat"], str(getattr(loss_cfg, "fhat_reg_norm", "l2"))) if fhat_active else mu_corr.new_tensor(0.0)
                rollout_det_loss = mu_corr.new_tensor(0.0)
                if rollout_iter is not None and rollout_weight > 0.0 and current_rollout_steps > 0:
                    try:
                        rb = next(rollout_iter)
                    except StopIteration:
                        rollout_iter = iter(rollout_loader)
                        rb = next(rollout_iter)
                    rollout_det_loss = _vpinn_td_rollout_loss_from_batch(
                        model=model,
                        batch=rb,
                        device=device,
                        non_blocking=non_blocking,
                        dt=dt,
                        rho=rho,
                        diameter=diameter,
                        td_params=td_params,
                        mean_active=mean_active,
                        sigma_active=probabilistic,
                        fhat_active=fhat_active,
                        use_td_force_input=use_td_force_input,
                        fhat_bound_multiplier=fhat_bound_multiplier,
                        sigma_min=sigma_min,
                        force_zero_output=force_zero_output,
                        rollout_loss_mode=rollout_loss_mode,
                        rollout_stochastic_samples=rollout_stochastic_samples,
                        rollout_noise_scale=rollout_noise_scale,
                        ur_bin_state_scale_info=(ur_bin_state_scale_info if normalize_by_ur_bin_std else None),
                        ur_bin_size=ur_bin_size,
                    )
                total_loss = (
                    loss_data
                    + loss_physics
                    + float(mean_reg) * mean_reg_loss
                    + float(sigma_reg) * sigma_reg_loss
                    + float(getattr(loss_cfg, "fhat_reg", 0.0)) * fhat_reg_loss
                    + float(rollout_weight) * rollout_det_loss
                )
            if total_loss.requires_grad:
                if scaler.is_enabled():
                    scaler.scale(total_loss).backward()
                    scaler.unscale_(opt)
                else:
                    total_loss.backward()
                grad_norm = nn_utils.clip_grad_norm_(model.parameters(), max_norm=float(training_cfg.max_grad_norm))
                if scaler.is_enabled():
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
            else:
                grad_norm = torch.zeros((), device=device)
            batches += 1
            sums["loss_total"] += total_loss.detach()
            sums["loss_data"] += loss_data.detach()
            sums["loss_physics"] += loss_physics.detach()
            sums["loss_reg_mean"] += mean_reg_loss.detach()
            sums["loss_reg_sigma"] += sigma_reg_loss.detach()
            sums["loss_reg_fhat"] += fhat_reg_loss.detach()
            sums["loss_rollout_det"] += rollout_det_loss.detach()
            sums["grad_norm"] += grad_norm.detach() if isinstance(grad_norm, torch.Tensor) else torch.tensor(float(grad_norm), device=device)
        denom = float(max(1, batches))
        metrics = {name: float((value / denom).detach().cpu()) for name, value in sums.items()}
        metrics["lr"] = float(opt.param_groups[0]["lr"]) if opt.param_groups else float(optim_cfg.lr)
        if epoch % log_every == 0 or epoch == epochs - 1:
            for name, value in metrics.items():
                writer.add_scalar(f"train/{name}", value, epoch + 1)
        if epoch % print_every == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch}: loss={metrics['loss_total']:.4e}, "
                f"Ldata={metrics['loss_data']:.4e}, Lphys={metrics['loss_physics']:.4e}, "
                f"Lroll={metrics['loss_rollout_det']:.4e}, lr={metrics['lr']:.3e}"
            )
        should_validate = (
            val_loader is not None
            and validate_every > 0
            and ((epoch % validate_every) == 0 or epoch == epochs - 1)
        )
        should_rollout = should_validate
        sync_validation_start: float | None = None
        if not async_validation and (should_validate or should_rollout):
            sync_validation_start = time.perf_counter()
        if async_validation and (should_validate or should_rollout):
            async_processes = _reap_async_processes(async_processes, best_state=async_best_state, wait=False)
            state_source: nn.Module = model
            if hasattr(model, "_orig_mod"):
                state_source = getattr(model, "_orig_mod")
            ckpt_path = async_dir / f"epoch_{epoch + 1:06d}.pt"
            torch.save(
                {
                    "model_state": state_source.state_dict(),
                    "config": asdict(config),
                    "run_name": run_name,
                    "dt": dt,
                    "method": "vpinn",
                    "td_correction": True,
                    "ur_bin_state_scale_info": ur_bin_state_scale_info,
                    "correction_mode": correction_mode,
                    "predict_sigma": probabilistic,
                    "mean_active": mean_active,
                    "fhat_active": fhat_active,
                    "use_td_force_input": use_td_force_input,
                    "use_acceleration_input": use_acceleration_input,
                    "use_phi_input": use_phi_input,
                    "phi_input_source": (None if not use_phi_input else phase_input_source),
                    "use_sigma_inputs": use_sigma_inputs,
                    "fhat_bound_multiplier": float(fhat_bound_multiplier),
                    "fhat_reg": float(fhat_reg),
                    "fhat_reg_norm": str(fhat_reg_norm),
                },
                ckpt_path,
            )
            snapshot_path = _save_td_validation_checkpoint(epoch)
            print(f"Saved validation checkpoint to {snapshot_path}")
            async_processes = _launch_async_validation(
                processes=async_processes,
                max_concurrent=async_max_concurrent,
                checkpoint_path=ckpt_path,
                epoch=epoch,
                run_name=run_name,
                writer=writer,
                async_device=async_device,
                async_num_workers=async_num_workers,
                async_num_threads=async_num_threads,
                do_losses=should_validate,
                do_rollout=should_rollout,
                best_state=async_best_state,
            )
            if val_seen_loader is not None and (should_validate or should_rollout):
                _run_sync_validation_for_split(
                    epoch_idx=epoch,
                    split_tag="val_seen",
                    split_name="val_seen",
                    split_loader=val_seen_loader,
                    split_rollout_loader=val_seen_rollout_loader,
                    split_trajs=val_seen_trajs,
                    log_rollout_plots=False,
                )
        elif should_validate:
            _run_sync_validation_for_split(
                epoch_idx=epoch,
                split_tag="val_unseen",
                split_name="val_unseen",
                split_loader=val_loader,
                split_rollout_loader=val_rollout_loader,
                split_trajs=val_trajs,
                log_rollout_plots=True,
            )
            if val_seen_loader is not None:
                _run_sync_validation_for_split(
                    epoch_idx=epoch,
                    split_tag="val_seen",
                    split_name="val_seen",
                    split_loader=val_seen_loader,
                    split_rollout_loader=val_seen_rollout_loader,
                    split_trajs=val_seen_trajs,
                    log_rollout_plots=False,
                )
            snapshot_path = _save_td_validation_checkpoint(epoch)
            print(f"Saved validation checkpoint to {snapshot_path}")

    if async_validation and async_processes:
        print(f"Waiting for {len(async_processes)} async validation job(s) to finish...")
        async_processes = _reap_async_processes(async_processes, best_state=async_best_state, wait=True)

    if final_rollout_all_validation and val_trajs:
        print("Final validation rollout (all trajectories) started.")
        metrics_sum: dict[str, float] = {}
        metrics_count: dict[str, int] = {}
        ur_values: list[float] = []
        metrics_list: list[dict[str, float]] = []
        plot_ur_values: list[float] = []
        plot_metrics_list: list[dict[str, float]] = []
        output_ur_values: list[float] = []
        corr_series_list: list[np.ndarray] = []
        sigma_series_list: list[np.ndarray] = []
        delta_fhat_series_list: list[np.ndarray] = []
        metric_trajs: list[dict[str, Any]] = []
        seen_metric_ur: set[float] = set()
        for traj in val_trajs:
            ur_val = float(np.asarray(traj["ur"]).reshape(-1)[0])
            ur_key = round(ur_val, 6)
            if ur_key in seen_metric_ur:
                continue
            seen_metric_ur.add(ur_key)
            metric_trajs.append(traj)
        plot_trajs = list(metric_trajs)
        seen_plot_ur = set(seen_metric_ur)
        for traj in train_trajs:
            ur_val = float(np.asarray(traj["ur"]).reshape(-1)[0])
            ur_key = round(ur_val, 6)
            if ur_key in seen_plot_ur:
                continue
            seen_plot_ur.add(ur_key)
            plot_trajs.append(traj)
        plot_trajs.sort(key=lambda traj: round(float(np.asarray(traj["ur"]).reshape(-1)[0]), 6))
        for traj in metric_trajs:
            metrics = _log_td_correction_rollout_validation(
                writer=writer,
                epoch=max(0, epochs - 1),
                model=model,
                traj=traj,
                dt=dt,
                td_mass_source=td_mass_source,
                rho=rho,
                diameter=diameter,
                td_params=td_params,
                device=device,
                sigma_min=sigma_min,
                mean_active=mean_active,
                probabilistic=probabilistic,
                fhat_active=fhat_active,
                use_td_force_input=use_td_force_input,
                fhat_bound_multiplier=fhat_bound_multiplier,
                force_zero_output=force_zero_output,
                rollout_stochastic=rollout_stochastic,
                rollout_noise_scale=rollout_noise_scale,
                rollout_seed=rollout_seed,
                tag_prefix="final_val/rollout",
                log_metrics=False,
                log_plots=False,
                log_correction_on_data=False,
                log_phase_map=False,
            )
            filtered = {name: float(value) for name, value in metrics.items() if np.isfinite(float(value))}
            if filtered:
                ur_values.append(float(np.asarray(traj["ur"]).reshape(-1)[0]))
                metrics_list.append(filtered)
            for name, value in filtered.items():
                metrics_sum[name] = metrics_sum.get(name, 0.0) + float(value)
                metrics_count[name] = metrics_count.get(name, 0) + 1
        for idx, traj in enumerate(plot_trajs, start=1):
            ur_val = float(np.asarray(traj["ur"]).reshape(-1)[0])
            traj_t = _td_rollout_traj_to_tensors(traj)
            x_true_t = traj_t["x"].to(device)
            v_true_t = traj_t["v"].to(device)
            td_true_t = traj_t["td_force"].to(device)
            ur_true_t = traj_t["ur"].to(device)
            mass_value = float(np.asarray(traj_t["dry_mass_kg" if td_mass_source == "dry" else "effective_mass_kg"]).reshape(()))
            stiffness_value = float(np.asarray(traj_t["stiffness_n_m"]).reshape(()))
            with torch.no_grad():
                step_on_data = _vpinn_step_with_corrections(
                    model=model,
                    x=x_true_t[:-1],
                    v=v_true_t[:-1],
                    ur=ur_true_t[:-1],
                    td_context=traj_t["td_context"].to(device)[:-1],
                    dt=dt,
                    m=torch.full((x_true_t.shape[0] - 1, x_true_t.shape[1]), mass_value, dtype=x_true_t.dtype, device=device),
                    c=torch.full((x_true_t.shape[0] - 1, x_true_t.shape[1]), float(np.asarray(traj_t["damping_c"]).reshape(())), dtype=x_true_t.dtype, device=device),
                    k=torch.full((x_true_t.shape[0] - 1, x_true_t.shape[1]), stiffness_value, dtype=x_true_t.dtype, device=device),
                    rho=rho,
                    diameter=diameter,
                    td_params=td_params,
                    mean_active=mean_active,
                    sigma_active=probabilistic,
                    fhat_active=fhat_active,
                    use_td_force_input=use_td_force_input,
                    fhat_bound_multiplier=fhat_bound_multiplier,
                    sigma_min=sigma_min,
                    force_zero_output=force_zero_output,
                )
                corr_on_data = step_on_data["corr_mu"]
                sigma_on_data = step_on_data["corr_sigma"]
                delta_fhat_on_data = step_on_data["delta_fhat"]
            output_ur_values.append(ur_val)
            corr_series_list.append(corr_on_data[:, 0].detach().cpu().numpy())
            if probabilistic:
                sigma_series_list.append(sigma_on_data[:, 0].detach().cpu().numpy())
            if fhat_active:
                delta_fhat_series_list.append(delta_fhat_on_data[:, 0].detach().cpu().numpy())

            plot_metrics = _log_td_correction_rollout_validation(
                writer=writer,
                epoch=max(0, epochs - 1),
                model=model,
                traj=traj,
                dt=dt,
                td_mass_source=td_mass_source,
                rho=rho,
                diameter=diameter,
                td_params=td_params,
                device=device,
                sigma_min=sigma_min,
                mean_active=mean_active,
                probabilistic=probabilistic,
                fhat_active=fhat_active,
                use_td_force_input=use_td_force_input,
                fhat_bound_multiplier=fhat_bound_multiplier,
                force_zero_output=force_zero_output,
                rollout_stochastic=rollout_stochastic,
                rollout_noise_scale=rollout_noise_scale,
                rollout_seed=rollout_seed,
                tag_prefix="final_val/rollout",
                step=idx,
                log_metrics=False,
                log_plots=True,
                log_correction_on_data=True,
                log_phase_map=True,
                title_suffix=f" [final {idx}/{len(plot_trajs)}]",
                log_spectra=True,
            )
            filtered_plot_metrics = {name: float(value) for name, value in plot_metrics.items() if np.isfinite(float(value))}
            if filtered_plot_metrics:
                plot_ur_values.append(ur_val)
                plot_metrics_list.append(filtered_plot_metrics)
        avg_metrics = {
            name: metrics_sum[name] / float(metrics_count[name])
            for name in metrics_sum
            if metrics_count.get(name, 0) > 0
        }
        if avg_metrics:
            summary_lines = [f"Final rollout over {len(metric_trajs)} validation trajectories (unique U_r):"]
            for name in sorted(avg_metrics):
                summary_lines.append(f"{name}: {avg_metrics[name]:.6f}")
                writer.add_scalar(f"final_val/avg/{name}", avg_metrics[name], epochs)
            writer.add_text("final_val/summary", "\n".join(summary_lines), epochs)
        if plot_ur_values and plot_metrics_list:
            reference_ur_values = [
                float(np.asarray(traj["ur"]).reshape(-1)[0])
                for traj in [*val_trajs, *train_trajs]
                if np.asarray(traj["ur"]).reshape(-1).size > 0
            ]
            log_final_rollout_errors_vs_ur(
                writer,
                plot_ur_values,
                plot_metrics_list,
                epochs,
                reference_ur_values=reference_ur_values,
            )
        if output_ur_values and mean_active and corr_series_list:
            force_mode = str(getattr(model, "force_representation", "force")).strip().lower()
            log_output_distribution_vs_ur(
                writer,
                epochs,
                ur_values=output_ur_values,
                mean_series=corr_series_list,
                sigma_series=(sigma_series_list if probabilistic else None),
                mean_label=("Correction coefficient" if force_mode == "coefficient" else "Correction force"),
                sigma_label=("Sigma coefficient" if force_mode == "coefficient" else "Sigma force"),
                tag="final_val/output_distribution_vs_ur",
            )
        if output_ur_values and fhat_active and delta_fhat_series_list:
            log_output_distribution_vs_ur(
                writer,
                epochs,
                ur_values=output_ur_values,
                mean_series=delta_fhat_series_list,
                sigma_series=None,
                mean_label="Delta fhat",
                sigma_label=None,
                tag="final_val/delta_fhat_distribution_vs_ur",
            )

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
            "td_correction": True,
            "ur_bin_state_scale_info": ur_bin_state_scale_info,
            "correction_mode": correction_mode,
            "predict_sigma": probabilistic,
            "mean_active": mean_active,
            "fhat_active": fhat_active,
            "use_td_force_input": use_td_force_input,
            "use_acceleration_input": use_acceleration_input,
            "use_phi_input": use_phi_input,
            "phi_input_source": (None if not use_phi_input else phase_input_source),
            "use_sigma_inputs": use_sigma_inputs,
            "fhat_bound_multiplier": float(fhat_bound_multiplier),
            "fhat_reg": float(fhat_reg),
            "fhat_reg_norm": str(fhat_reg_norm),
        },
        model_path,
    )
    print(f"Saved final model to {model_path}")
    writer.flush()
    writer.close()


def train(config: Config, config_name: str) -> None:
    vp = dict(config.vpinn or {})
    if "use_td_correction" in vp and not bool(vp.get("use_td_correction", True)):
        raise ValueError("VPINN now only supports TD-correction training. Remove vpinn.use_td_correction or set it to true.")
    _train_td_correction_vpinn(config, config_name)
    return
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
    num_poly = int(vp.get("num_poly_test", 2))
    num_sine = int(vp.get("num_sine_test", 0))
    rollout_force_weight = float(vp.get("rollout_force_weight", 0.0))
    rollout_force_steps = int(vp.get("rollout_force_steps", 0))
    rollout_force_every = int(vp.get("rollout_force_every", 1))
    rollout_force_batch_size = int(vp.get("rollout_force_batch_size", 0))
    force_representation = str(vp.get("force_representation", "force")).strip().lower()
    if force_representation not in {"force", "coefficient"}:
        raise ValueError("vpinn.force_representation must be one of: force, coefficient.")
    use_force_coeff = force_representation == "coefficient"
    coeff_k = float(getattr(config.model, "k", 1218.0))
    coeff_m_eff = float(_m_eff_from_model_cfg(config.model))
    if not (use_force_loss or use_weak_loss):
        raise ValueError("vpinn must enable at least one of: use_force_loss, use_weak_loss.")

    use_gradnorm = bool(vp.get("use_gradnorm", getattr(loss_cfg, "use_gradnorm", False)))
    gradnorm_alpha = float(vp.get("gradnorm_alpha", getattr(loss_cfg, "gradnorm_alpha", 0.9)))
    gradnorm_eps = float(vp.get("gradnorm_eps", getattr(loss_cfg, "gradnorm_eps", 1e-8)))
    gradnorm_min_weight = float(vp.get("gradnorm_min_weight", getattr(loss_cfg, "gradnorm_min_weight", 0.1)))
    gradnorm_max_weight = float(vp.get("gradnorm_max_weight", getattr(loss_cfg, "gradnorm_max_weight", 10.0)))
    gradnorm_update_every_steps = int(vp.get("gradnorm_update_every_steps", getattr(loss_cfg, "gradnorm_update_every_steps", 1)))
    gradnorm_update_every_steps = max(1, gradnorm_update_every_steps)

    device = select_device(os.getenv("TRAIN_DEVICE", str(runtime_cfg.device)))
    print(f"Using device: {device}")
    configure_tf32(device, bool(precision_cfg.use_tf32))
    set_num_threads_from_slurm(default=1)
    non_blocking = device.type == "cuda"

    train_trajs, val_trajs, dt = _prepare_trajectories(config)
    if not val_trajs:
        train_trajs, val_trajs = _split_by_trajectory(train_trajs, val_fraction=val_fraction, seed=split_seed)
    if not train_trajs:
        raise ValueError("Empty training split. Reduce vpinn.val_fraction or provide more trajectories.")

    batch_size = int(training_cfg.batch_size)
    epochs = int(training_cfg.epochs)
    max_grad_norm = float(training_cfg.max_grad_norm)

    d = int(train_trajs[0]["x"].shape[-1])
    m = _as_diag_param(vp.get("m", _m_eff_from_model_cfg(config.model)), d, device, "m")
    c = _as_diag_param(vp.get("c", getattr(config.model, "damping_c", 1e-4)), d, device, "c")
    k = _as_diag_param(vp.get("k", getattr(config.model, "k", 1218.0)), d, device, "k")

    input_dim = 2 * d + 1
    output_dim = d
    model = _build_force_model(
        config,
        input_dim=input_dim,
        output_dim=output_dim,
        mean_output_dim=d,
    )
    model = model.to(device)
    setattr(model, "hard_force_symmetry", bool(getattr(config.architecture, "hard_force_symmetry", False)))

    use_input_scaling = bool(vp.get("use_input_scaling", False))
    if use_input_scaling:
        D_val = float(getattr(config.model, "D", 1.0))
        x_scale = D_val if np.isfinite(D_val) and D_val != 0.0 else 1.0
        # Typical velocity scale: omega * D, with omega = sqrt(k/m).
        omega = torch.sqrt(torch.clamp(k / m, min=1e-12))
        v_scale = omega * float(x_scale)
        # Reduce velocity is dimensionless; scale it to O(1) for the force network.
        ur_scale = float(vp.get("ur_scale", 10.0))
        # In force mode, use a typical force scale k*D. In coefficient mode,
        # the network outputs C_F directly, so keep the wrapper output scale at 1.
        f_scale = 1.0 if use_force_coeff else k * float(x_scale)
        model = ScaledForceWrapper(
            model,
            d=d,
            x_scale=x_scale,
            v_scale=v_scale,
            ur_scale=ur_scale,
            f_scale=f_scale,
        )
        force_scale_msg = "f_scale=1 (coefficient output)" if use_force_coeff else "f_scale=kD"
        print(
            f"VPINN scaling enabled: x/D (D={x_scale:g}), v/(sqrt(k/m)D), U_r/{ur_scale:g}, "
            f"{force_scale_msg}."
        )

    model = maybe_compile_model(model, bool(compile_cfg.use_compile), str(compile_cfg.compile_mode))

    gradnorm_balancer: Optional[GradNormBalancer] = None
    gradnorm_last_force = None
    gradnorm_last_weak = None
    gradnorm_last_rollout = None
    use_rollout_loss = rollout_force_weight > 0.0 and rollout_force_steps > 0
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

    train_dataset = WindowDataset(
        train_trajs,
        window_intervals=window_M,
        stride=stride,
    )
    val_dataset = (
        WindowDataset(val_trajs, window_intervals=window_M, stride=stride)
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
    rollout_every = validate_every
    final_rollout_all_validation = bool(getattr(monitoring_cfg, "final_rollout_all_validation", False))
    async_validation = bool(getattr(monitoring_cfg, "async_validation", False))
    async_device = str(getattr(monitoring_cfg, "async_validation_device", "cpu"))
    async_num_workers = int(getattr(monitoring_cfg, "async_validation_num_workers", 0))
    async_num_threads = int(getattr(monitoring_cfg, "async_validation_num_threads", 4))
    async_max_concurrent = int(getattr(monitoring_cfg, "async_validation_max_concurrent", 1))

    writer, run_name = setup_writer(
        config.logging.run_dir_root,
        config_name,
        run_name_override=getattr(config.logging, "run_name", None),
        append_timestamp=bool(getattr(config.logging, "append_timestamp", True)),
    )
    async_processes: list[dict[str, Any]] = []
    async_best_state: dict[str, Any] = {"loss_total": float("inf")}
    async_dir = Path(writer.log_dir) / "async_validation"
    if async_validation:
        async_dir.mkdir(parents=True, exist_ok=True)
    run_models_dir = Path("models") / run_name
    run_models_dir.mkdir(parents=True, exist_ok=True)
    validation_models_dir = run_models_dir / "async_validation"
    validation_models_dir.mkdir(parents=True, exist_ok=True)

    def _save_validation_snapshot(epoch_idx: int) -> Path:
        ckpt_path = validation_models_dir / f"model_epoch_{epoch_idx + 1:06d}.pt"
        latest_path = validation_models_dir / "model.pt"
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
            ckpt_path,
        )
        shutil.copyfile(ckpt_path, latest_path)
        return ckpt_path

    use_lr_scheduler = bool(optim_cfg.use_lr_scheduler)
    base_lr = float(optim_cfg.lr)
    middle_time_plot = resolve_middle_time_plot(config.data, vp, method_name="vpinn")
    D_val = float(getattr(config.model, "D", 1.0))

    train_windows = len(train_dataset)
    train_steps_per_epoch = len(train_loader)
    val_windows = len(val_dataset) if val_dataset is not None else 0
    val_steps_per_epoch = len(val_loader) if val_loader is not None else 0
    startup_lines = [
        f"Run name: {run_name}",
        (
            f"VPINN training setup: epochs={epochs}, batch_size={batch_size}, "
            f"steps_per_epoch={train_steps_per_epoch}, train_windows={train_windows}, "
            f"train_trajectories={len(train_trajs)}"
        ),
        (
            f"Validation setup: steps={val_steps_per_epoch}, val_windows={val_windows}, "
            f"val_trajectories={len(val_trajs) if val_trajs is not None else 0}"
        ),
        (
            f"Windowing: window_M={window_M}, stride={stride}, dt={dt:g}, "
            f"use_force_loss={use_force_loss}, use_weak_loss={use_weak_loss}"
        ),
        (
            f"Runtime: device={device}, num_workers={int(runtime_cfg.num_workers)}, amp={amp_enabled}, "
            f"compile={bool(compile_cfg.use_compile)}, lr={base_lr:g}, scheduler={use_lr_scheduler}"
        ),
        (
            f"Monitoring: validate_every={validate_every}, rollout_every={rollout_every}, "
            f"print_every={print_every}, log_every={log_every}, async_validation={async_validation}"
        ),
    ]
    if use_rollout_loss:
        startup_lines.append(
            f"Rollout loss: weight={rollout_force_weight:g}, steps={rollout_force_steps}, "
            f"batch_size={'full' if rollout_force_batch_size <= 0 else rollout_force_batch_size}"
        )
    print("\n".join(startup_lines))

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

        expect_f0 = use_force_coeff
        for step, batch in enumerate(train_loader):
            if len(batch) == 4:
                x_win, v_win, f_meas, ur_win = batch
                f0 = None
            elif len(batch) == 5:
                x_win, v_win, f_meas, ur_win, f0 = batch
            else:
                raise ValueError("Unexpected batch format from dataloader.")
            x_win = x_win.to(device, non_blocking=non_blocking)
            v_win = v_win.to(device, non_blocking=non_blocking)
            f_meas = f_meas.to(device, non_blocking=non_blocking)
            ur_win = ur_win.to(device, non_blocking=non_blocking)
            if f0 is not None:
                f0 = f0.to(device, non_blocking=non_blocking)

            B, M1, d = x_win.shape
            inp = torch.cat([x_win, v_win, ur_win], dim=-1)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                flat = inp.reshape(B * M1, -1)
                f_pred = model(flat).reshape(B, M1, d)
                per_loss_f = torch.mean((f_pred - f_meas) ** 2, dim=(1, 2))
                loss_f = torch.mean(per_loss_f)
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
                        f0=f0,
                    )
                    per_loss_w = torch.mean(R.pow(2), dim=(1, 2))
                    loss_w = torch.mean(per_loss_w)
                else:
                    loss_w = loss_f.new_tensor(0.0)

                loss_roll = loss_f.new_tensor(0.0)
                roll_computed = False
                if rollout_force_weight > 0.0 and rollout_force_steps > 0:
                    steps_k = min(int(rollout_force_steps), int(M1) - 1)
                    rollout_batch_n = int(B) if rollout_force_batch_size <= 0 else min(int(B), int(rollout_force_batch_size))
                    if steps_k > 0 and rollout_batch_n > 0:
                        x_roll = x_win[:rollout_batch_n]
                        v_roll = v_win[:rollout_batch_n]
                        ur_roll = ur_win[:rollout_batch_n]
                        f_true_roll = f_meas[:rollout_batch_n]
                        f0_step = f0[:rollout_batch_n, 0, :] if f0 is not None else None
                        _x_seq, _v_seq, f_seq = rollout_rk4(
                            model=model,
                            x0=x_roll[:, 0, :],
                            v0=v_roll[:, 0, :],
                            ur0=ur_roll[:, 0, :],
                            steps=steps_k,
                            dt=dt,
                            m=m,
                            c=c,
                            k=k,
                            f0=f0_step,
                        )
                        f_roll = f_seq[:, : steps_k + 1, :]
                        f_true = f_true_roll[:, : steps_k + 1, :]
                        per_roll = torch.mean((f_roll - f_true) ** 2, dim=(1, 2))
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
            if rollout_force_weight > 0.0 and rollout_force_steps > 0 and roll_computed:
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
        sync_validation_start: float | None = None
        if not async_validation and (should_validate or should_rollout):
            sync_validation_start = time.perf_counter()

        if async_validation and (should_validate or should_rollout):
            async_processes = _reap_async_processes(async_processes, best_state=async_best_state, wait=False)
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
            snapshot_path = _save_validation_snapshot(epoch)
            print(f"Saved validation checkpoint to {snapshot_path}")
            async_processes = _launch_async_validation(
                processes=async_processes,
                max_concurrent=async_max_concurrent,
                checkpoint_path=ckpt_path,
                epoch=epoch,
                run_name=run_name,
                writer=writer,
                async_device=async_device,
                async_num_workers=async_num_workers,
                async_num_threads=async_num_threads,
                do_losses=should_validate,
                do_rollout=should_rollout,
                best_state=async_best_state,
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
                expect_f0=use_force_coeff,
            )
            for k_name, v_value in val_metrics.items():
                writer.add_scalar(f"val/{k_name}", v_value, epoch)
            force_map = _force_mapping_nrmse_over_trajs(model=model, val_trajs=val_trajs or [], device=device)
            if force_map is not None:
                for k_name, v_value in force_map.items():
                    writer.add_scalar(f"val/{k_name}", v_value, epoch)
        if should_rollout and not async_validation:
            candidates = val_trajs if val_trajs else train_trajs
            if not candidates:
                continue
            ur_all = [float(traj["ur"][0, 0].detach().cpu().item()) for traj in candidates]
            sampled_metric_indices = sample_one_index_per_ur(ur_all, seed=1)
            metrics_sum: dict[str, float] = {}
            metrics_count: dict[str, int] = {}
            for sidx in sampled_metric_indices:
                metrics = _log_rollout_validation(
                    writer=writer,
                    epoch=epoch,
                    model=model,
                    traj=candidates[sidx],
                    dt=dt,
                    m=m,
                    c=c,
                    k=k,
                    D=D_val,
                    middle_time_plot=middle_time_plot,
                    device=device,
                    log_metrics=False,
                    log_plots=False,
                )
                for name, value in metrics.items():
                    if not np.isfinite(float(value)):
                        continue
                    metrics_sum[name] = metrics_sum.get(name, 0.0) + float(value)
                    metrics_count[name] = metrics_count.get(name, 0) + 1
            for name, total in metrics_sum.items():
                denom = float(max(1, metrics_count.get(name, 0)))
                writer.add_scalar(f"val/{name}", total / denom, epoch)

            selected_indices = list(range(len(candidates)))
            rollout_idx = selected_indices[0]
            for traj in candidates[rollout_idx : rollout_idx + 1]:
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
                    log_metrics=False,
                    log_plots=True,
                    log_spectra=True,
                )
        if not async_validation and (should_validate or should_rollout):
            snapshot_path = _save_validation_snapshot(epoch)
            print(f"Saved validation checkpoint to {snapshot_path}")
        if sync_validation_start is not None:
            elapsed = float(time.perf_counter() - sync_validation_start)
            writer.add_scalar("val/validation_wall_time_s", elapsed, epoch)

    if async_validation and async_processes:
        print(f"Waiting for {len(async_processes)} async validation job(s) to finish...")
        async_processes = _reap_async_processes(async_processes, best_state=async_best_state, wait=True)

    if final_rollout_all_validation and val_trajs:
        print("Final validation rollout (all trajectories) started.")
        final_start = time.perf_counter()
        metrics_sum: dict[str, float] = {}
        metrics_count: dict[str, int] = {}
        used = 0
        ur_values: list[float] = []
        metrics_list: list[dict[str, float]] = []
        plot_ur_values: list[float] = []
        plot_metrics_list: list[dict[str, float]] = []
        metric_trajs: list[dict[str, Any]] = []
        seen_metric_ur: set[float] = set()
        for traj in val_trajs:
            ur_val = float(traj["ur"][0, 0].detach().cpu().item())
            ur_key = round(ur_val, 6)
            if ur_key in seen_metric_ur:
                continue
            seen_metric_ur.add(ur_key)
            metric_trajs.append(traj)
        plot_trajs = list(metric_trajs)
        seen_plot_ur = set(seen_metric_ur)
        for traj in train_trajs:
            ur_val = float(traj["ur"][0, 0].detach().cpu().item())
            ur_key = round(ur_val, 6)
            if ur_key in seen_plot_ur:
                continue
            seen_plot_ur.add(ur_key)
            plot_trajs.append(traj)
        plot_trajs.sort(key=lambda traj: round(float(traj["ur"][0, 0].detach().cpu().item()), 6))
        for traj in metric_trajs:
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
                tag_prefix="final_val/rollout",
                log_metrics=False,
                log_plots=False,
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
        for idx, traj in enumerate(plot_trajs, start=1):
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
                tag_prefix="final_val/rollout",
                step=idx,
                log_metrics=False,
                title_suffix=f" [final {idx}/{len(plot_trajs)}]",
                log_spectra=True,
            )
            filtered_plot_metrics = {name: float(value) for name, value in metrics.items() if np.isfinite(float(value))}
            if filtered_plot_metrics:
                plot_ur_values.append(float(traj["ur"][0, 0].detach().cpu().item()))
                plot_metrics_list.append(filtered_plot_metrics)
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
        if plot_ur_values and plot_metrics_list:
            reference_ur_values = [
                float(traj["ur"][0, 0].detach().cpu().item())
                for traj in [*val_trajs, *train_trajs]
            ]
            log_final_rollout_errors_vs_ur(
                writer,
                plot_ur_values,
                plot_metrics_list,
                epochs,
                reference_ur_values=reference_ur_values,
            )
        elapsed = time.perf_counter() - final_start
        print(f"Final validation rollout finished in {elapsed:.2f}s.")

    if final_rollout_all_validation and val_loader is not None:
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
            expect_f0=use_force_coeff,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
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
