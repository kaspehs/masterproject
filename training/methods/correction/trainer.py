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
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils as nn_utils
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from core.logging import setup_writer
from core.optim import setup_optimizer_and_scheduler
from core.runtime import (
    configure_tf32,
    maybe_compile_model,
    select_device,
    set_num_threads_from_slurm,
    setup_amp,
)
from training.training_utils import (
    AGGREGATE_FORCE_VALIDATION_ERROR_KEY,
    AGGREGATE_VALIDATION_ERROR_KEY,
    Config,
    FORCE_MAPPING_NRMSE_KEY,
    GradNormBalancer,
    PHVIV,
    ROLLOUT_DIVERGED_COUNT_KEY,
    ROLLOUT_DIVERGED_KEY,
    build_phase_plot_grid,
    build_dataloader_from_series,
    build_rollout_dataloader_from_series,
    compute_validation_metrics,
    compute_model_grad_norm,
    load_training_series,
    format_loss_vs_ur_text,
    create_window_mask,
    create_zoom_mask,
    log_area_normalized_rollout_spectra,
    log_loss_vs_ur,
    log_displacement_plots,
    log_final_rollout_errors_vs_ur,
    log_force_plots,
    log_correction_on_data_plot,
    log_output_distribution_vs_ur,
    log_phase_component_plots,
    log_rollout_phase_trajectory_plot,
    log_signed_phase_output_plot,
    log_training_metrics,
    log_validation_epoch,
    nearest_phase_series_values,
    preprocess_timeseries,
    load_td_correction_trajectories,
    resolve_middle_time_plot,
    resolve_td_correction_params,
    resolve_td_correction_mode,
    resolve_td_memory_config,
    resolve_td_n_memory_torch,
    resolve_td_input_configs,
    resolve_td_fhat_correction_bounds,
    resolve_phnn_input_scaling_mode,
    rollout_model,
    resolve_td_phase_input_source,
    structural_step_constant_force_torch,
    scaled_residual_loss_per_sample,
    td_bounded_delta_fhat_torch,
    td_baseline_step_torch,
    td_correction_mode_flags,
    td_fhat_active_from_mode,
    td_mean_active_from_mode,
    td_predict_sigma_from_mode,
    resolve_cut_start_seconds,
    sample_one_index_per_ur,
    _broadcast_td_hidden_param_torch,
    td_hidden_inputs_from_context_torch,
)


def _last_linear(module: nn.Module | None) -> nn.Linear | None:
    if module is None:
        return None
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


def _init_mean_head(module: nn.Module | None, *, mode: str, tiny_std: float) -> None:
    last = _last_linear(module)
    if last is None or mode == "standard":
        return
    with torch.no_grad():
        if mode == "zero":
            nn.init.zeros_(last.weight)
        elif mode == "tiny":
            nn.init.normal_(last.weight, mean=0.0, std=float(tiny_std))
        else:
            raise ValueError("corr_init_mode must be one of: zero, tiny, standard.")
        nn.init.zeros_(last.bias)


def _init_sigma_head(module: nn.Module | None, *, mode: str, tiny_std: float, sigma_min: float) -> None:
    last = _last_linear(module)
    if last is None or mode == "standard":
        return
    target_sigma = float(sigma_min) if mode == "zero" else max(float(sigma_min), float(tiny_std))
    target_excess = max(0.0, target_sigma - float(sigma_min))
    bias_value = _softplus_inverse_scalar(target_excess)
    with torch.no_grad():
        if mode == "zero":
            nn.init.zeros_(last.weight)
        elif mode == "tiny":
            nn.init.normal_(last.weight, mean=0.0, std=float(tiny_std))
        else:
            raise ValueError("corr_init_mode must be one of: zero, tiny, standard.")
        nn.init.constant_(last.bias, bias_value)


def _init_fhat_head(module: nn.Module | None, *, mode: str, tiny_std: float) -> None:
    last = _last_linear(module)
    if last is None or mode == "standard":
        return
    with torch.no_grad():
        if mode == "zero":
            nn.init.zeros_(last.weight)
        elif mode == "tiny":
            nn.init.normal_(last.weight, mean=0.0, std=float(tiny_std))
        else:
            raise ValueError("corr_init_mode must be one of: zero, tiny, standard.")
        nn.init.zeros_(last.bias)


def _resolve_td_correction_init_settings(method_cfg: dict[str, Any], model_cfg: Any) -> tuple[str, float]:
    mode = str(method_cfg.get("corr_init_mode", "standard")).strip().lower()
    if mode not in {"zero", "tiny", "standard"}:
        raise ValueError("training.corr_init_mode must be one of: zero, tiny, standard.")
    tiny_std = float(method_cfg.get("corr_init_tiny_std", 1.0e-4))
    if not np.isfinite(tiny_std) or tiny_std <= 0.0:
        raise ValueError("training.corr_init_tiny_std must be finite and > 0.")
    return mode, tiny_std


def _model_phase_input_source(model: PHVIV) -> str:
    base = getattr(model, "_orig_mod", model)
    raw_value = getattr(
        base,
        "phi_input_source",
        (True if bool(getattr(base, "use_phi_input", False)) else False),
    )
    return resolve_td_phase_input_source(raw_value)


def _apply_td_correction_head_init(
    model: PHVIV,
    *,
    mode: str,
    tiny_std: float,
    predict_sigma: bool,
    predict_fhat: bool,
) -> None:
    _init_mean_head(model.u_base_net, mode=mode, tiny_std=tiny_std)
    if predict_sigma:
        sigma_min = float(model.sigma_min.detach().cpu())
        _init_sigma_head(model.sigma_net, mode=mode, tiny_std=tiny_std, sigma_min=sigma_min)
    if predict_fhat:
        _init_fhat_head(model.fhat_net, mode=mode, tiny_std=tiny_std)


def _parse_hnn_batch(batch: Any) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    if len(batch) < 5:
        raise ValueError("Unexpected batch format from dataloader.")
    z_i, t_i, z_next, t_next, ur_i = batch[:5]
    f_i = None
    f_next = None
    scale = None
    remaining = len(batch) - 5
    if remaining == 0:
        pass
    elif remaining == 1:
        scale = batch[5]
    elif remaining == 2:
        f_i, f_next = batch[5], batch[6]
    elif remaining == 3:
        f_i, f_next, scale = batch[5], batch[6], batch[7]
    else:
        raise ValueError("Unexpected batch format from dataloader.")
    return z_i, t_i, z_next, t_next, ur_i, None, f_i, f_next, scale


def _parse_rollout_batch(batch: Any) -> tuple[Any, Any, Any, Any, Any, Any]:
    if len(batch) < 4:
        raise ValueError("Unexpected rollout batch format.")
    z0, t_seq, z_traj, ur0 = batch[:4]
    scale = None
    remaining = len(batch) - 4
    if remaining == 0:
        pass
    elif remaining == 1:
        scale = batch[4]
    else:
        raise ValueError("Unexpected rollout batch format.")
    return z0, t_seq, z_traj, ur0, None, scale


def _td_flow_feature_from_traj(
    traj: dict[str, np.ndarray],
    *,
    input_scaling_mode: str,
    diameter: float,
) -> np.ndarray:
    mode = resolve_phnn_input_scaling_mode(input_scaling_mode)
    if mode == "current":
        return np.asarray(traj["ur"], dtype=np.float32).reshape(-1)
    diameter_value = float(diameter)
    if not np.isfinite(diameter_value) or abs(diameter_value) <= 0.0:
        raise ValueError(f"diameter must be finite and non-zero for convective PHNN scaling, got {diameter!r}.")
    flow_speed = np.asarray(traj["flow_speed"], dtype=np.float32).reshape(-1)
    return flow_speed / np.float32(diameter_value)


def _td_output_scale_tensor(
    model: PHVIV,
    *,
    reduced_velocity: torch.Tensor,
    structural_mass: torch.Tensor | float,
    stiffness: torch.Tensor | float,
    like: torch.Tensor,
) -> torch.Tensor:
    mass_t = _broadcast_td_hidden_param_torch(structural_mass, like=like, name="structural_mass")
    stiffness_t = _broadcast_td_hidden_param_torch(stiffness, like=like, name="stiffness")
    if getattr(model, "force_output", "force") == "coefficient":
        rv_raw = model._prepare_reduced_velocity_raw(reduced_velocity, like=like)
        if rv_raw is None:
            raise ValueError("reduced_velocity is required for PHNN coefficient-force scaling.")
        if getattr(model, "input_scaling_mode", "current") == "convective":
            u_flow = rv_raw * float(model.D)
        else:
            omega_n = torch.sqrt(torch.clamp(stiffness_t / mass_t, min=1e-12))
            f_n = omega_n / (2.0 * np.pi)
            u_flow = rv_raw * f_n * float(model.D)
        f0 = 0.5 * float(model.rho) * float(model.D) * (u_flow**2)
        return torch.clamp(f0, min=1e-12)
    return stiffness_t * float(model.D)


def _td_p_scale_tensor(
    model: PHVIV,
    *,
    reduced_velocity: torch.Tensor,
    structural_mass: torch.Tensor | float,
    stiffness: torch.Tensor | float,
    like: torch.Tensor,
) -> torch.Tensor:
    mass_t = _broadcast_td_hidden_param_torch(structural_mass, like=like, name="structural_mass")
    stiffness_t = _broadcast_td_hidden_param_torch(stiffness, like=like, name="stiffness")
    if getattr(model, "input_scaling_mode", "current") == "convective":
        rv_raw = model._prepare_reduced_velocity_raw(reduced_velocity, like=like)
        if rv_raw is None:
            raise ValueError("reduced_velocity is required for convective PHNN momentum scaling.")
        u_flow = torch.clamp(torch.abs(rv_raw * float(model.D)), min=1e-12)
        return torch.clamp(mass_t * u_flow, min=1e-12)
    return torch.sqrt(torch.clamp(mass_t * stiffness_t, min=1e-12)) * float(model.D)


def _td_state_for_model_scaling(
    model: PHVIV,
    *,
    z: torch.Tensor,
    reduced_velocity: torch.Tensor,
    structural_mass: torch.Tensor | float,
    stiffness: torch.Tensor | float,
) -> torch.Tensor:
    p_scale_actual = _td_p_scale_tensor(
        model,
        reduced_velocity=reduced_velocity,
        structural_mass=structural_mass,
        stiffness=stiffness,
        like=z[..., :1],
    )
    p_scale_model = torch.as_tensor(float(model.nn_p_scale), device=z.device, dtype=z.dtype)
    p_model = z[..., 1:2] * (p_scale_model / p_scale_actual)
    return torch.cat([z[..., 0:1], p_model], dim=-1)


def _td_rollout_state_scale(
    model: PHVIV,
    *,
    z_like: torch.Tensor,
    reduced_velocity: torch.Tensor,
    structural_mass: torch.Tensor | float,
    stiffness: torch.Tensor | float,
) -> torch.Tensor:
    q_scale = torch.full_like(z_like[..., 0:1], float(model.D))
    p_scale = _td_p_scale_tensor(
        model,
        reduced_velocity=reduced_velocity,
        structural_mass=structural_mass,
        stiffness=stiffness,
        like=z_like[..., :1],
    )
    return torch.cat([q_scale, p_scale], dim=-1)


def _td_optional_hidden_inputs_for_model(
    model: PHVIV,
    *,
    td_context: torch.Tensor,
    velocity: torch.Tensor,
    structural_mass: torch.Tensor | float,
    stiffness: torch.Tensor | float,
) -> dict[str, dict[str, torch.Tensor | None]]:
    input_configs = getattr(model, "td_input_configs", None)
    if not isinstance(input_configs, dict):
        phase_input_source = _model_phase_input_source(model)
        input_configs = {
            "mean": {
                "use_phi_input": bool(getattr(model, "use_phi_input", False)),
                "phase_input_source": phase_input_source,
                "use_sigma_inputs": bool(getattr(model, "use_sigma_inputs", False)),
                "use_acceleration_input": bool(getattr(model, "use_acceleration_input", False)),
            }
        }
    hidden_by_head: dict[str, dict[str, torch.Tensor | None]] = {}
    cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for head in ("mean", "fhat", "sigma"):
        cfg = input_configs.get(head, input_configs.get("mean", {}))
        uses_phi = bool(cfg.get("use_phi_input", False))
        uses_sigma = bool(cfg.get("use_sigma_inputs", False))
        uses_accel = bool(cfg.get("use_acceleration_input", False))
        if not (uses_phi or uses_sigma or uses_accel):
            hidden_by_head[head] = {
                "phi_input": None,
                "sigma_inputs": None,
                "acceleration_input": None,
            }
            continue
        phase_input_source = resolve_td_phase_input_source(
            cfg.get("phase_input_source", cfg.get("phi_input_source", False))
            if uses_phi
            else "phi_vy"
        )
        if phase_input_source not in cache:
            cache[phase_input_source] = td_hidden_inputs_from_context_torch(
                td_context=td_context,
                structural_mass=structural_mass,
                stiffness=stiffness,
                diameter=float(model.D),
                velocity=velocity,
                phase_input_source=phase_input_source,
                input_scaling_mode=getattr(model, "input_scaling_mode", "current"),
            )
        phi_input, sigma_inputs, acceleration_input = cache[phase_input_source]
        hidden_by_head[head] = {
            "phi_input": phi_input if uses_phi else None,
            "sigma_inputs": sigma_inputs if uses_sigma else None,
            "acceleration_input": acceleration_input if uses_accel else None,
        }
    return hidden_by_head


def _td_predict_outputs(
    model: PHVIV,
    *,
    z: torch.Tensor,
    reduced_velocity: torch.Tensor,
    head_inputs: dict[str, dict[str, torch.Tensor | None]],
    structural_mass: torch.Tensor | float,
    stiffness: torch.Tensor | float,
    mean_active: bool,
    sigma_active: bool,
    fhat_active: bool,
    force_zero_output: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    z_model = _td_state_for_model_scaling(
        model,
        z=z,
        reduced_velocity=reduced_velocity,
        structural_mass=structural_mass,
        stiffness=stiffness,
    )
    output_scale = _td_output_scale_tensor(
        model,
        reduced_velocity=reduced_velocity,
        structural_mass=structural_mass,
        stiffness=stiffness,
        like=z_model[..., :1],
    )
    if force_zero_output or not mean_active:
        raw_force = z_model[..., :1].new_zeros(z_model.shape[:-1] + (1,))
    else:
        raw_force = model._force_net_raw(
            z_model,
            reduced_velocity=reduced_velocity,
            td_force_input=head_inputs["mean"].get("td_force_input"),
            td_fhat_input=head_inputs["mean"].get("td_fhat_input"),
            acceleration_input=head_inputs["mean"].get("acceleration_input"),
            phi_input=head_inputs["mean"].get("phi_input"),
            sigma_inputs=head_inputs["mean"].get("sigma_inputs"),
            td_force_scale=output_scale,
        )
    raw_force = model._apply_coefficient_output_bound(raw_force)
    corr_mu = raw_force * output_scale
    if sigma_active:
        if force_zero_output:
            sigma = model.sigma_min.to(device=raw_force.device, dtype=raw_force.dtype) * output_scale
        else:
            raw_sigma = model._sigma_net_raw(
                z_model,
                reduced_velocity=reduced_velocity,
                td_force_input=head_inputs["sigma"].get("td_force_input"),
                td_fhat_input=head_inputs["sigma"].get("td_fhat_input"),
                acceleration_input=head_inputs["sigma"].get("acceleration_input"),
                phi_input=head_inputs["sigma"].get("phi_input"),
                sigma_inputs=head_inputs["sigma"].get("sigma_inputs"),
                td_force_scale=output_scale,
            )
            sigma = model.sigma_min.to(device=raw_sigma.device, dtype=raw_sigma.dtype) + F.softplus(raw_sigma)
            sigma = sigma * output_scale
    else:
        sigma = corr_mu.new_zeros(corr_mu.shape)
    if force_zero_output or not fhat_active:
        raw_delta_fhat = corr_mu.new_zeros(corr_mu.shape)
    else:
        raw_delta_fhat = model._fhat_net_raw(
            z_model,
            reduced_velocity=reduced_velocity,
            td_force_input=head_inputs["fhat"].get("td_force_input"),
            td_fhat_input=head_inputs["fhat"].get("td_fhat_input"),
            acceleration_input=head_inputs["fhat"].get("acceleration_input"),
            phi_input=head_inputs["fhat"].get("phi_input"),
            sigma_inputs=head_inputs["fhat"].get("sigma_inputs"),
            td_force_scale=output_scale,
        )
    return corr_mu, sigma, raw_delta_fhat, raw_force


def _td_force_input_tensor_from_source(
    *,
    source: str,
    baseline_force_next: torch.Tensor,
    baseline_diag: dict[str, torch.Tensor],
) -> torch.Tensor | None:
    key = str(source).strip().lower()
    if key == "none":
        return None
    if key == "total":
        return baseline_force_next
    if key == "fcv":
        return baseline_diag["force_cv"]
    raise ValueError(f"Unsupported TD force input source {source!r}.")


def _td_step_with_corrections(
    *,
    model: PHVIV,
    z: torch.Tensor,
    reduced_velocity: torch.Tensor,
    td_context: torch.Tensor,
    dt: float | torch.Tensor,
    structural_mass: torch.Tensor | float,
    damping_c: torch.Tensor | float,
    stiffness: torch.Tensor | float,
    td_params: dict[str, float],
    td_memory_cfg: dict[str, Any],
    mean_active: bool,
    sigma_active: bool,
    fhat_active: bool,
    td_force_input_source: str,
    fhat_bound_multiplier: float,
    force_zero_output: bool = False,
    force_phase_convention: str = "next",
) -> dict[str, torch.Tensor]:
    velocity = z[:, 1:2] / structural_mass
    step_params = dict(td_params)
    step_params["n_memory"] = resolve_td_n_memory_torch(
        td_params,
        dt=dt,
        flow_speed=td_context[:, 4:5],
        diameter=float(model.D),
        memory_cfg=td_memory_cfg,
    )
    baseline_force_next, baseline_context_next, baseline_diag = td_baseline_step_torch(
        velocity=velocity,
        acceleration=td_context[:, 0:1],
        td_context=td_context,
        dt=dt,
        rho=float(model.rho),
        diameter=float(model.D),
        params=step_params,
        force_phase_convention=force_phase_convention,
        return_diagnostics=True,
    )
    hidden_inputs_by_head = _td_optional_hidden_inputs_for_model(
        model,
        td_context=td_context,
        velocity=velocity,
        structural_mass=structural_mass,
        stiffness=stiffness,
    )
    input_configs = getattr(model, "td_input_configs", None)
    if not isinstance(input_configs, dict):
        input_configs = {
            "mean": {
                "use_td_force_input": td_force_input_source != "none",
                "td_force_input_source": td_force_input_source,
                "use_td_fhat_input": bool(getattr(model, "use_td_fhat_input", False)),
            }
        }
        input_configs["fhat"] = dict(input_configs["mean"])
        input_configs["sigma"] = dict(input_configs["mean"])
    head_inputs: dict[str, dict[str, torch.Tensor | None]] = {}
    for head in ("mean", "fhat", "sigma"):
        cfg = input_configs.get(head, input_configs.get("mean", {}))
        hidden = hidden_inputs_by_head.get(head, hidden_inputs_by_head.get("mean", {}))
        force_source = str(cfg.get("td_force_input_source", "none"))
        head_inputs[head] = {
            "td_force_input": (
                _td_force_input_tensor_from_source(
                    source=force_source,
                    baseline_force_next=baseline_force_next,
                    baseline_diag=baseline_diag,
                )
                if bool(cfg.get("use_td_force_input", False))
                else None
            ),
            "td_fhat_input": baseline_diag["fhat_td"] if bool(cfg.get("use_td_fhat_input", False)) else None,
            "acceleration_input": hidden.get("acceleration_input"),
            "phi_input": hidden.get("phi_input"),
            "sigma_inputs": hidden.get("sigma_inputs"),
        }
    corr_mu, sigma_corr, raw_delta_fhat, raw_corr_mu = _td_predict_outputs(
        model,
        z=z,
        reduced_velocity=reduced_velocity,
        head_inputs=head_inputs,
        structural_mass=structural_mass,
        stiffness=stiffness,
        mean_active=mean_active,
        sigma_active=sigma_active,
        fhat_active=fhat_active,
        force_zero_output=force_zero_output,
    )
    if fhat_active:
        fhat_correction_bounds = getattr(model, "fhat_correction_bounds", None)
        if fhat_correction_bounds is None:
            fhat_bound_min = None
            fhat_bound_max = None
        else:
            fhat_bound_min, fhat_bound_max = fhat_correction_bounds
        td_force_next, td_context_next, td_diag = td_baseline_step_torch(
            velocity=velocity,
            acceleration=td_context[:, 0:1],
            td_context=td_context,
            dt=dt,
            rho=float(model.rho),
            diameter=float(model.D),
            params=step_params,
            raw_delta_fhat=raw_delta_fhat,
            fhat_bound_multiplier=float(fhat_bound_multiplier),
            fhat_bound_min=fhat_bound_min,
            fhat_bound_max=fhat_bound_max,
            force_phase_convention=force_phase_convention,
            return_diagnostics=True,
        )
    else:
        td_force_next = baseline_force_next
        td_context_next = baseline_context_next
        td_diag = baseline_diag
        td_diag = dict(td_diag)
        td_diag["delta_fhat"] = raw_delta_fhat.new_zeros(raw_delta_fhat.shape)
        td_diag["fhat_corr"] = td_diag["fhat_td"]
        td_diag["omega_vy_corr"] = td_diag["omega_vy_td"]
    corr_force = corr_mu
    total_force_next = td_force_next + corr_force
    y_next, v_next, a_next = structural_step_constant_force_torch(
        y=z[:, 0:1],
        velocity=velocity,
        force=total_force_next,
        dt=dt,
        mass=structural_mass,
        damping_c=damping_c,
        stiffness=stiffness,
    )
    z_next_mean = torch.cat([y_next, v_next * structural_mass], dim=1)
    td_context_next = td_context_next.clone()
    td_context_next[:, 0:1] = a_next
    return {
        "baseline_force_next": baseline_force_next,
        "td_force_next": td_force_next,
        "total_force_next": total_force_next,
        "corr_mu": corr_mu,
        "raw_corr_mu": raw_corr_mu,
        "corr_force": corr_force,
        "sigma_corr": sigma_corr,
        "raw_delta_fhat": raw_delta_fhat,
        "delta_fhat": td_diag["delta_fhat"],
        "fhat_td": td_diag["fhat_td"],
        "fhat_corr": td_diag["fhat_corr"],
        "omega_vy_td": td_diag["omega_vy_td"],
        "omega_vy_corr": td_diag["omega_vy_corr"],
        "theta_td": td_diag["theta_td"],
        "z_next_mean": z_next_mean,
        "td_context_next": td_context_next,
        "a_next": a_next,
    }


def _td_state_mse_loss(
    *,
    z_i: torch.Tensor,
    dt_i: torch.Tensor,
    z_next: torch.Tensor,
    total_force_next: torch.Tensor,
    mass_i: torch.Tensor,
    damping_i: torch.Tensor,
    stiffness_i: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    velocity_i = z_i[:, 1:2] / mass_i
    y_next_mean, v_next_mean, _a_next = structural_step_constant_force_torch(
        y=z_i[:, 0:1],
        velocity=velocity_i,
        force=total_force_next,
        dt=dt_i,
        mass=mass_i,
        damping_c=damping_i,
        stiffness=stiffness_i,
    )
    z_next_mean = torch.cat([y_next_mean, v_next_mean * mass_i], dim=1)
    state_loss = torch.mean(torch.sum((z_next_mean - z_next) ** 2, dim=1))
    return state_loss, z_next_mean


def _td_state_propagated_nll_loss(
    *,
    z_i: torch.Tensor,
    dt_i: torch.Tensor,
    z_next: torch.Tensor,
    total_force_next: torch.Tensor,
    sigma_corr: torch.Tensor,
    mass_i: torch.Tensor,
    damping_i: torch.Tensor,
    stiffness_i: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    state_mse_loss, z_next_mean = _td_state_mse_loss(
        z_i=z_i,
        dt_i=dt_i,
        z_next=z_next,
        total_force_next=total_force_next,
        mass_i=mass_i,
        damping_i=damping_i,
        stiffness_i=stiffness_i,
    )
    del state_mse_loss
    var_p = torch.clamp((dt_i * sigma_corr) ** 2, min=1e-9)
    var_y = torch.clamp(((0.5 * (dt_i ** 2) / mass_i) * sigma_corr) ** 2, min=1e-9)
    nll_y = 0.5 * (((z_next[:, 0:1] - z_next_mean[:, 0:1]) ** 2) / var_y + torch.log(var_y))
    nll_p = 0.5 * (((z_next[:, 1:2] - z_next_mean[:, 1:2]) ** 2) / var_p + torch.log(var_p))
    return torch.mean(nll_y + nll_p), z_next_mean


def _normalize_psd_dt_values(dt: float | torch.Tensor, *, batch_size: int) -> torch.Tensor:
    if isinstance(dt, torch.Tensor):
        flat = dt.detach().reshape(-1).to(dtype=torch.float64)
        if flat.numel() == 0:
            raise ValueError("PSD loss received an empty dt tensor.")
        if flat.numel() == 1:
            flat = flat.expand(batch_size)
        elif flat.numel() != batch_size:
            raise ValueError(
                f"PSD loss expected 1 or {batch_size} dt values, got {flat.numel()}."
            )
    else:
        flat = torch.full((batch_size,), float(dt), dtype=torch.float64)
    if not torch.isfinite(flat).all():
        raise ValueError("PSD loss requires finite dt values.")
    if torch.any(flat <= 0.0):
        raise ValueError("PSD loss requires positive dt values.")
    return flat


def _dt_values_effectively_constant(dt_values: torch.Tensor, *, rel_tol: float = 1.0e-6) -> bool:
    if dt_values.numel() <= 1:
        return True
    mean_dt = float(torch.mean(dt_values).item())
    max_dev = float(torch.max(torch.abs(dt_values - mean_dt)).item())
    rel_dev = max_dev / max(abs(mean_dt), 1.0e-12)
    return rel_dev <= float(rel_tol)


def _group_sample_indices_by_dt(
    dt_values: torch.Tensor,
    *,
    rounding_decimals: int = 7,
) -> list[tuple[float, torch.Tensor]]:
    if dt_values.ndim != 1:
        raise ValueError("dt grouping expects a 1D tensor of per-sample dt values.")
    scale = float(10 ** int(rounding_decimals))
    groups: dict[int, list[int]] = {}
    dt_cpu = dt_values.detach().cpu()
    for idx, dt_val in enumerate(dt_cpu.tolist()):
        key = int(round(float(dt_val) * scale))
        groups.setdefault(key, []).append(int(idx))
    grouped: list[tuple[float, torch.Tensor]] = []
    for key in sorted(groups):
        indices = torch.tensor(groups[key], device=dt_values.device, dtype=torch.long)
        group_dt = float(torch.mean(dt_values.index_select(0, indices)).item())
        grouped.append((group_dt, indices))
    return grouped


def _rollout_disp_spectra_common_dt_torch(
    *,
    true_signal: torch.Tensor,
    pred_signal: torch.Tensor,
    dt: float,
    use_hann_window: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if true_signal.ndim != 2 or pred_signal.ndim != 2:
        raise ValueError("Spectral rollout loss expects [batch, time] tensors.")
    if true_signal.shape != pred_signal.shape:
        raise ValueError("Spectral rollout loss requires true and predicted signals with matching shapes.")
    compute_dtype = torch.float32 if true_signal.dtype in {torch.float16, torch.bfloat16} else true_signal.dtype
    true_signal = true_signal.to(dtype=compute_dtype)
    pred_signal = pred_signal.to(dtype=compute_dtype)
    batch_size, length = true_signal.shape
    if batch_size < 1 or length < 4:
        return None

    true_centered = true_signal - torch.mean(true_signal, dim=1, keepdim=True)
    pred_centered = pred_signal - torch.mean(pred_signal, dim=1, keepdim=True)
    if bool(use_hann_window):
        window = torch.hann_window(length, periodic=False, device=true_signal.device, dtype=true_signal.dtype).view(1, -1)
    else:
        window = torch.ones((1, length), device=true_signal.device, dtype=true_signal.dtype)
    true_fft = torch.fft.rfft(true_centered * window, dim=1)
    pred_fft = torch.fft.rfft(pred_centered * window, dim=1)
    true_psd = torch.abs(true_fft) ** 2
    pred_psd = torch.abs(pred_fft) ** 2
    freqs = torch.fft.rfftfreq(length, d=float(dt), device=true_signal.device)
    base_mask = torch.isfinite(freqs) & (freqs > 0.0)
    if int(base_mask.sum().item()) < 2:
        return None
    return true_psd, pred_psd, freqs, base_mask


def _spectral_band_mask_from_true_peak_torch(
    *,
    true_psd: torch.Tensor,
    freqs: torch.Tensor,
    base_mask: torch.Tensor,
    peak_rel_bandwidth: float,
) -> torch.Tensor:
    batch_size = int(true_psd.shape[0])
    band_mask = base_mask.view(1, -1).expand(batch_size, -1)
    rel_bw = float(peak_rel_bandwidth)
    if not (np.isfinite(rel_bw) and rel_bw > 0.0):
        return band_mask

    pos_freqs = freqs[base_mask]
    peak_idx = torch.argmax(true_psd[:, base_mask], dim=1)
    peak_freq = pos_freqs[peak_idx]
    freq_res = float(freqs[1].item() - freqs[0].item()) if freqs.numel() > 1 else float("nan")
    min_half_width = max(0.5 * freq_res if np.isfinite(freq_res) and freq_res > 0.0 else 0.0, 1.0e-12)
    half_width = torch.clamp(peak_freq * rel_bw, min=min_half_width)
    band_mask = base_mask.view(1, -1) & (
        torch.abs(freqs.view(1, -1) - peak_freq.view(-1, 1)) <= half_width.view(-1, 1)
    )
    fallback = band_mask.sum(dim=1) < 2
    if torch.any(fallback):
        band_mask = torch.where(fallback.view(-1, 1), base_mask.view(1, -1), band_mask)
    return band_mask


def _psd_error_common_dt_torch(
    *,
    true_signal: torch.Tensor,
    pred_signal: torch.Tensor,
    dt: float,
    peak_rel_bandwidth: float = 0.0,
    use_hann_window: bool = True,
    relative: bool = False,
    eps: float = 1e-12,
) -> torch.Tensor:
    spec = _rollout_disp_spectra_common_dt_torch(
        true_signal=true_signal,
        pred_signal=pred_signal,
        dt=dt,
        use_hann_window=use_hann_window,
    )
    if spec is None:
        return true_signal.new_zeros(())
    true_psd, pred_psd, freqs, base_mask = spec
    band_mask = _spectral_band_mask_from_true_peak_torch(
        true_psd=true_psd,
        freqs=freqs,
        base_mask=base_mask,
        peak_rel_bandwidth=peak_rel_bandwidth,
    )
    mask_f = band_mask.to(dtype=true_psd.dtype)
    amp_true = torch.sqrt(torch.clamp(true_psd, min=0.0) + float(eps)) * mask_f
    amp_pred = torch.sqrt(torch.clamp(pred_psd, min=0.0) + float(eps)) * mask_f
    valid = band_mask.sum(dim=1) >= 1
    if not torch.any(valid):
        return true_signal.new_zeros(())
    num_bins = torch.clamp(mask_f.sum(dim=1), min=1.0)
    loss = torch.sum(((amp_pred - amp_true) ** 2) * mask_f, dim=1) / num_bins
    if bool(relative):
        denom = torch.sum((amp_true * amp_true) * mask_f, dim=1) / num_bins + float(eps)
        loss = loss / denom
    loss = torch.where(valid, loss, torch.zeros_like(loss))
    return torch.sum(loss) / torch.clamp(valid.to(dtype=loss.dtype).sum(), min=1.0)


def _psd_error_torch(
    *,
    true_signal: torch.Tensor,
    pred_signal: torch.Tensor,
    dt: float | torch.Tensor,
    peak_rel_bandwidth: float = 0.0,
    use_hann_window: bool = True,
    relative: bool = False,
    eps: float = 1e-12,
) -> torch.Tensor:
    if true_signal.ndim != 2 or pred_signal.ndim != 2:
        raise ValueError("PSD loss expects [batch, time] tensors.")
    if true_signal.shape != pred_signal.shape:
        raise ValueError("PSD loss requires true and predicted signals with matching shapes.")
    batch_size = int(true_signal.shape[0])
    if batch_size < 1:
        return true_signal.new_zeros(())
    dt_values = _normalize_psd_dt_values(dt, batch_size=batch_size)
    if _dt_values_effectively_constant(dt_values):
        return _psd_error_common_dt_torch(
            true_signal=true_signal,
            pred_signal=pred_signal,
            dt=float(torch.mean(dt_values).item()),
            peak_rel_bandwidth=peak_rel_bandwidth,
            use_hann_window=use_hann_window,
            relative=relative,
            eps=eps,
        )

    grouped = _group_sample_indices_by_dt(dt_values)
    losses: list[torch.Tensor] = []
    weights: list[float] = []
    for group_dt, group_indices in grouped:
        losses.append(
            _psd_error_common_dt_torch(
                true_signal=true_signal.index_select(0, group_indices),
                pred_signal=pred_signal.index_select(0, group_indices),
                dt=group_dt,
                peak_rel_bandwidth=peak_rel_bandwidth,
                use_hann_window=use_hann_window,
                relative=relative,
                eps=eps,
            )
        )
        weights.append(float(group_indices.numel()))
    if not losses:
        return true_signal.new_zeros(())
    weight_tensor = torch.tensor(weights, device=true_signal.device, dtype=losses[0].dtype)
    loss_tensor = torch.stack(losses)
    return torch.sum(loss_tensor * weight_tensor) / torch.clamp(weight_tensor.sum(), min=1.0)


def _soft_dominant_frequency_from_psd_torch(
    *,
    psd: torch.Tensor,
    freqs: torch.Tensor,
    band_mask: torch.Tensor,
    alpha: float = 12.0,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask_f = band_mask.to(dtype=psd.dtype)
    p_masked = torch.clamp(psd, min=0.0) * mask_f
    peak_power = torch.amax(p_masked, dim=1, keepdim=True)
    valid = (peak_power[:, 0] > float(eps)) & (band_mask.sum(dim=1) >= 1)
    scaled = p_masked / peak_power.clamp_min(float(eps))
    weights = torch.pow(torch.clamp(scaled, min=0.0), float(alpha)) * mask_f
    weight_sum = torch.sum(weights, dim=1, keepdim=True)
    freq_grid = freqs.view(1, -1).to(dtype=psd.dtype)
    dominant = torch.sum(weights * freq_grid, dim=1) / weight_sum.clamp_min(float(eps)).view(-1)
    dominant = torch.where(valid, dominant, torch.zeros_like(dominant))
    return dominant, valid


def _dominant_frequency_error_common_dt_torch(
    *,
    true_signal: torch.Tensor,
    pred_signal: torch.Tensor,
    dt: float,
    peak_rel_bandwidth: float = 0.0,
    use_hann_window: bool = True,
    relative: bool = True,
    power: float = 1.0,
    alpha: float = 12.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    spec = _rollout_disp_spectra_common_dt_torch(
        true_signal=true_signal,
        pred_signal=pred_signal,
        dt=dt,
        use_hann_window=use_hann_window,
    )
    if spec is None:
        return true_signal.new_zeros(())
    true_psd, pred_psd, freqs, base_mask = spec
    band_mask = _spectral_band_mask_from_true_peak_torch(
        true_psd=true_psd,
        freqs=freqs,
        base_mask=base_mask,
        peak_rel_bandwidth=peak_rel_bandwidth,
    )
    true_dom, true_valid = _soft_dominant_frequency_from_psd_torch(
        psd=true_psd,
        freqs=freqs,
        band_mask=band_mask,
        alpha=alpha,
        eps=eps,
    )
    pred_dom, pred_valid = _soft_dominant_frequency_from_psd_torch(
        psd=pred_psd,
        freqs=freqs,
        band_mask=band_mask,
        alpha=alpha,
        eps=eps,
    )
    valid = true_valid & pred_valid
    if not torch.any(valid):
        return true_signal.new_zeros(())
    loss = torch.pow(torch.abs(pred_dom - true_dom), float(power))
    if bool(relative):
        denom = torch.pow(torch.abs(true_dom), float(power)) + float(eps)
        loss = loss / denom
    loss = torch.where(valid, loss, torch.zeros_like(loss))
    return torch.sum(loss) / torch.clamp(valid.to(dtype=loss.dtype).sum(), min=1.0)


def _dominant_frequency_error_torch(
    *,
    true_signal: torch.Tensor,
    pred_signal: torch.Tensor,
    dt: float | torch.Tensor,
    peak_rel_bandwidth: float = 0.0,
    use_hann_window: bool = True,
    relative: bool = True,
    power: float = 1.0,
    alpha: float = 12.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    if true_signal.ndim != 2 or pred_signal.ndim != 2:
        raise ValueError("Dominant-frequency loss expects [batch, time] tensors.")
    if true_signal.shape != pred_signal.shape:
        raise ValueError("Dominant-frequency loss requires true and predicted signals with matching shapes.")
    batch_size = int(true_signal.shape[0])
    if batch_size < 1:
        return true_signal.new_zeros(())
    dt_values = _normalize_psd_dt_values(dt, batch_size=batch_size)
    if _dt_values_effectively_constant(dt_values):
        return _dominant_frequency_error_common_dt_torch(
            true_signal=true_signal,
            pred_signal=pred_signal,
            dt=float(torch.mean(dt_values).item()),
            peak_rel_bandwidth=peak_rel_bandwidth,
            use_hann_window=use_hann_window,
            relative=relative,
            power=power,
            alpha=alpha,
            eps=eps,
        )

    grouped = _group_sample_indices_by_dt(dt_values)
    losses: list[torch.Tensor] = []
    weights: list[float] = []
    for group_dt, group_indices in grouped:
        losses.append(
            _dominant_frequency_error_common_dt_torch(
                true_signal=true_signal.index_select(0, group_indices),
                pred_signal=pred_signal.index_select(0, group_indices),
                dt=group_dt,
                peak_rel_bandwidth=peak_rel_bandwidth,
                use_hann_window=use_hann_window,
                relative=relative,
                power=power,
                alpha=alpha,
                eps=eps,
            )
        )
        weights.append(float(group_indices.numel()))
    if not losses:
        return true_signal.new_zeros(())
    weight_tensor = torch.tensor(weights, device=true_signal.device, dtype=losses[0].dtype)
    loss_tensor = torch.stack(losses)
    return torch.sum(loss_tensor * weight_tensor) / torch.clamp(weight_tensor.sum(), min=1.0)


def _target_centered_rms_scale_torch(
    target: torch.Tensor,
    *,
    time_dim: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    centered = target - torch.mean(target, dim=time_dim, keepdim=True)
    scale = torch.sqrt(torch.mean(centered * centered, dim=time_dim, keepdim=True))
    return torch.clamp(scale, min=float(eps))


def _displacement_std_error_torch(
    *,
    true_signal: torch.Tensor,
    pred_signal: torch.Tensor,
    relative: bool = False,
    power: float = 2.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    if true_signal.ndim != 2 or pred_signal.ndim != 2:
        raise ValueError("Std rollout loss expects [batch, time] tensors.")
    if true_signal.shape != pred_signal.shape:
        raise ValueError("Std rollout loss requires true and predicted signals with matching shapes.")
    compute_dtype = torch.float32 if true_signal.dtype in {torch.float16, torch.bfloat16} else true_signal.dtype
    true_signal = true_signal.to(dtype=compute_dtype)
    pred_signal = pred_signal.to(dtype=compute_dtype)
    if int(true_signal.shape[0]) < 1 or int(true_signal.shape[1]) < 2:
        return true_signal.new_zeros(())

    true_centered = true_signal - torch.mean(true_signal, dim=1, keepdim=True)
    pred_centered = pred_signal - torch.mean(pred_signal, dim=1, keepdim=True)
    true_std = torch.sqrt(torch.mean(true_centered * true_centered, dim=1))
    pred_std = torch.sqrt(torch.mean(pred_centered * pred_centered, dim=1))
    loss = torch.pow(torch.abs(pred_std - true_std), float(power))
    if bool(relative):
        denom = torch.pow(torch.abs(true_std) + float(eps), float(power))
        loss = loss / denom
    return torch.mean(loss)


def _displacement_mean_error_torch(
    *,
    true_signal: torch.Tensor,
    pred_signal: torch.Tensor,
    relative: bool = False,
    power: float = 2.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    if true_signal.ndim != 2 or pred_signal.ndim != 2:
        raise ValueError("Mean rollout loss expects [batch, time] tensors.")
    if true_signal.shape != pred_signal.shape:
        raise ValueError("Mean rollout loss requires true and predicted signals with matching shapes.")
    compute_dtype = torch.float32 if true_signal.dtype in {torch.float16, torch.bfloat16} else true_signal.dtype
    true_signal = true_signal.to(dtype=compute_dtype)
    pred_signal = pred_signal.to(dtype=compute_dtype)
    if int(true_signal.shape[0]) < 1 or int(true_signal.shape[1]) < 1:
        return true_signal.new_zeros(())

    true_mean = torch.mean(true_signal, dim=1)
    pred_mean = torch.mean(pred_signal, dim=1)
    loss = torch.pow(torch.abs(pred_mean - true_mean), float(power))
    if bool(relative):
        true_centered = true_signal - true_mean.view(-1, 1)
        true_std = torch.sqrt(torch.mean(true_centered * true_centered, dim=1))
        denom = torch.pow(torch.abs(true_std) + float(eps), float(power))
        loss = loss / denom
    return torch.mean(loss)


def _trajectory_rollout_mse_torch(
    *,
    true_traj: torch.Tensor,
    pred_traj: torch.Tensor,
    z_scale: torch.Tensor,
    relative: bool = False,
    eps: float = 1e-12,
) -> torch.Tensor:
    if true_traj.ndim != 3 or pred_traj.ndim != 3 or z_scale.ndim != 3:
        raise ValueError("Rollout trajectory MSE expects [batch, time, state] tensors.")
    if true_traj.shape != pred_traj.shape or true_traj.shape != z_scale.shape:
        raise ValueError("Rollout trajectory MSE requires matching true, predicted, and scale shapes.")
    if int(true_traj.shape[0]) < 1 or int(true_traj.shape[1]) < 1:
        return true_traj.new_zeros(())
    err = (pred_traj - true_traj) / z_scale
    loss = torch.mean(torch.sum(err * err, dim=2), dim=1)
    if bool(relative):
        true_norm = true_traj / z_scale
        denom = torch.mean(torch.sum(true_norm * true_norm, dim=2), dim=1).clamp_min(float(eps))
        loss = loss / denom
    return torch.mean(loss)


def _resolve_td_rollout_loss_settings(loss_cfg: Any) -> dict[str, Any]:
    disp_std_p = float(getattr(loss_cfg, "rollout_disp_std_p", 2.0))
    disp_freq_p = float(getattr(loss_cfg, "rollout_disp_freq_p", 1.0))
    disp_freq_alpha = float(getattr(loss_cfg, "rollout_disp_freq_alpha", 12.0))
    if not np.isfinite(disp_std_p) or disp_std_p < 1.0 or disp_std_p > 2.0:
        raise ValueError("loss.rollout_disp_std_p must be finite and in [1, 2].")
    if not np.isfinite(disp_freq_p) or disp_freq_p < 1.0 or disp_freq_p > 2.0:
        raise ValueError("loss.rollout_disp_freq_p must be finite and in [1, 2].")
    if not np.isfinite(disp_freq_alpha) or disp_freq_alpha < 1.0:
        raise ValueError("loss.rollout_disp_freq_alpha must be finite and >= 1.")
    return {
        "disp_std_p": disp_std_p,
        "disp_freq_p": disp_freq_p,
        "disp_freq_alpha": disp_freq_alpha,
    }


def _normalize_rollout_disp_spectral_loss_mode(raw_mode: Any) -> str:
    key = str(raw_mode).strip().lower()
    if key in {"", "psd", "psd_shape", "area_normalized_psd", "area_normalized", "spectral_amplitude", "spectral_amp"}:
        return "psd"
    if key in {"dominant_frequency", "dominant_freq", "dom_freq", "peak_frequency"}:
        return "dominant_frequency"
    raise ValueError(
        "loss.rollout_disp_spectral_loss must be one of: psd, spectral_amplitude, dominant_frequency."
    )


def _td_correction_rollout_losses_from_batch(
    *,
    model: PHVIV,
    batch: Any,
    device: torch.device,
    non_blocking: bool,
    td_params: dict[str, float],
    td_memory_cfg: dict[str, Any],
    mean_active: bool,
    sigma_active: bool,
    fhat_active: bool,
    td_force_input_source: str,
    fhat_bound_multiplier: float,
    force_zero_output: bool,
    compute_disp_std_loss: bool = False,
    compute_disp_mean_loss: bool = False,
    disp_std_power: float = 2.0,
    compute_disp_spectral_loss: bool = False,
    disp_spectral_loss_mode: str = "psd",
    disp_freq_power: float = 1.0,
    disp_freq_alpha: float = 12.0,
    disp_psd_peak_rel_bandwidth: float = 0.0,
    disp_psd_use_hann_window: bool = True,
) -> dict[str, torch.Tensor]:
    if len(batch) != 8:
        raise ValueError("Unexpected TD correction rollout batch format.")
    z0, t_seq, z_traj, ur0, td_context0, mass0, damping0, stiffness0 = batch
    z0 = z0.to(device, non_blocking=non_blocking)
    t_seq = t_seq.to(device, non_blocking=non_blocking)
    z_traj = z_traj.to(device, non_blocking=non_blocking)
    ur0 = ur0.to(device, non_blocking=non_blocking)
    td_context0 = td_context0.to(device, non_blocking=non_blocking)
    mass0 = mass0.to(device, non_blocking=non_blocking)
    damping0 = damping0.to(device, non_blocking=non_blocking)
    stiffness0 = stiffness0.to(device, non_blocking=non_blocking)

    dt_roll = torch.clamp((t_seq[:, 1] - t_seq[:, 0]).unsqueeze(1), min=1.0e-12)
    dt_values = dt_roll[:, 0]
    disp_std_loss = z_traj.new_zeros(())
    disp_mean_loss = z_traj.new_zeros(())
    disp_spectral_loss = z_traj.new_zeros(())
    z_scale = _td_rollout_state_scale(
        model,
        z_like=z_traj,
        reduced_velocity=ur0,
        structural_mass=mass0,
        stiffness=stiffness0,
    )
    z_traj_horizon = z_traj[:, 1:, :]
    z_scale_horizon = z_scale[:, 1:, :]
    z_pred, _force_seq, _corr_seq, _sigma_seq, _delta_fhat_seq = _td_correction_state_rollout(
        model=model,
        z0=z0,
        ur0=ur0,
        td_context0=td_context0,
        steps=int(z_traj.shape[1] - 1),
        dt=dt_roll,
        structural_mass=mass0,
        damping_c=damping0,
        stiffness=stiffness0,
        td_params=td_params,
        td_memory_cfg=td_memory_cfg,
        mean_active=mean_active,
        sigma_active=False,
        fhat_active=fhat_active,
        td_force_input_source=td_force_input_source,
        fhat_bound_multiplier=fhat_bound_multiplier,
        force_zero_output=force_zero_output,
    )
    z_pred_horizon = z_pred[:, 1:, :]
    trajectory_loss = _trajectory_rollout_mse_torch(
        true_traj=z_traj_horizon,
        pred_traj=z_pred_horizon,
        z_scale=z_scale_horizon,
        relative=False,
    )
    if compute_disp_std_loss:
        disp_std_loss = _displacement_std_error_torch(
            true_signal=z_traj_horizon[:, :, 0],
            pred_signal=z_pred_horizon[:, :, 0],
            relative=False,
            power=disp_std_power,
        )
    if compute_disp_mean_loss:
        disp_mean_loss = _displacement_mean_error_torch(
            true_signal=z_traj_horizon[:, :, 0],
            pred_signal=z_pred_horizon[:, :, 0],
            relative=False,
            power=disp_std_power,
        )
    if compute_disp_spectral_loss:
        if disp_spectral_loss_mode == "dominant_frequency":
            disp_spectral_loss = _dominant_frequency_error_torch(
                true_signal=z_traj_horizon[:, :, 0],
                pred_signal=z_pred_horizon[:, :, 0],
                dt=dt_values,
                peak_rel_bandwidth=disp_psd_peak_rel_bandwidth,
                use_hann_window=disp_psd_use_hann_window,
                relative=False,
                power=disp_freq_power,
                alpha=disp_freq_alpha,
            )
        else:
            disp_spectral_loss = _psd_error_torch(
                true_signal=z_traj_horizon[:, :, 0],
                pred_signal=z_pred_horizon[:, :, 0],
                dt=dt_values,
                peak_rel_bandwidth=disp_psd_peak_rel_bandwidth,
                use_hann_window=disp_psd_use_hann_window,
                relative=False,
            )
    return {
        "trajectory_loss": trajectory_loss,
        "disp_std_loss": disp_std_loss,
        "disp_mean_loss": disp_mean_loss,
        "disp_spectral_loss": disp_spectral_loss,
    }


def _td_correction_rollout_loss_from_batch(
    *,
    model: PHVIV,
    batch: Any,
    device: torch.device,
    non_blocking: bool,
    td_params: dict[str, float],
    td_memory_cfg: dict[str, Any],
    mean_active: bool,
    sigma_active: bool,
    fhat_active: bool,
    td_force_input_source: str,
    fhat_bound_multiplier: float,
    force_zero_output: bool,
) -> torch.Tensor:
    return _td_correction_rollout_losses_from_batch(
        model=model,
        batch=batch,
        device=device,
        non_blocking=non_blocking,
        td_params=td_params,
        td_memory_cfg=td_memory_cfg,
        mean_active=mean_active,
        sigma_active=sigma_active,
        fhat_active=fhat_active,
        td_force_input_source=td_force_input_source,
        fhat_bound_multiplier=fhat_bound_multiplier,
        force_zero_output=force_zero_output,
        compute_disp_std_loss=False,
        compute_disp_mean_loss=False,
        disp_std_power=2.0,
        compute_disp_spectral_loss=False,
        disp_spectral_loss_mode="psd",
        disp_freq_power=1.0,
        disp_freq_alpha=12.0,
        disp_psd_peak_rel_bandwidth=0.0,
    )["trajectory_loss"]


def _td_correction_state_rollout(
    *,
    model: PHVIV,
    z0: torch.Tensor,
    ur0: torch.Tensor,
    td_context0: torch.Tensor,
    steps: int,
    dt: float | torch.Tensor,
    structural_mass: torch.Tensor,
    damping_c: torch.Tensor,
    stiffness: torch.Tensor,
    td_params: dict[str, float],
    td_memory_cfg: dict[str, Any],
    mean_active: bool,
    sigma_active: bool,
    fhat_active: bool,
    td_force_input_source: str,
    fhat_bound_multiplier: float,
    force_zero_output: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    z = z0
    td_context = td_context0
    z_hist = [z0]
    total_force_hist: list[torch.Tensor] = []
    corr_mu_hist: list[torch.Tensor] = []
    sigma_hist: list[torch.Tensor] = []
    delta_fhat_hist: list[torch.Tensor] = []
    for _ in range(int(steps)):
        step = _td_step_with_corrections(
            model=model,
            z=z,
            reduced_velocity=ur0,
            td_context=td_context,
            dt=dt,
            structural_mass=structural_mass,
            damping_c=damping_c,
            stiffness=stiffness,
            td_params=td_params,
            td_memory_cfg=td_memory_cfg,
            mean_active=mean_active,
            sigma_active=sigma_active,
            fhat_active=fhat_active,
            td_force_input_source=td_force_input_source,
            fhat_bound_multiplier=fhat_bound_multiplier,
            force_zero_output=force_zero_output,
        )
        z = step["z_next_mean"]
        td_context = step["td_context_next"]
        z_hist.append(z)
        total_force_hist.append(step["total_force_next"])
        corr_mu_hist.append(step["corr_force"])
        sigma_hist.append(step["sigma_corr"])
        delta_fhat_hist.append(step["delta_fhat"])
    z_seq = torch.stack(z_hist, dim=1)
    total_force_seq = torch.stack(total_force_hist, dim=1) if total_force_hist else z0.new_zeros((z0.shape[0], 0, 1))
    corr_mu_seq = torch.stack(corr_mu_hist, dim=1) if corr_mu_hist else z0.new_zeros((z0.shape[0], 0, 1))
    sigma_seq = torch.stack(sigma_hist, dim=1) if sigma_hist else z0.new_zeros((z0.shape[0], 0, 1))
    delta_fhat_seq = torch.stack(delta_fhat_hist, dim=1) if delta_fhat_hist else z0.new_zeros((z0.shape[0], 0, 1))
    return z_seq, total_force_seq, corr_mu_seq, sigma_seq, delta_fhat_seq


def _td_pure_baseline_state_rollout(
    *,
    z0: torch.Tensor,
    td_context0: torch.Tensor,
    steps: int,
    dt: float | torch.Tensor,
    structural_mass: torch.Tensor,
    damping_c: torch.Tensor,
    stiffness: torch.Tensor,
    rho: float,
    diameter: float,
    td_params: dict[str, float],
    td_memory_cfg: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    z = z0
    td_context = td_context0
    z_hist = [z0]
    total_force_hist: list[torch.Tensor] = []
    for _ in range(int(steps)):
        velocity = z[:, 1:2] / structural_mass
        step_params = dict(td_params)
        step_params["n_memory"] = resolve_td_n_memory_torch(
            td_params,
            dt=dt,
            flow_speed=td_context[:, 4:5],
            diameter=float(diameter),
            memory_cfg=td_memory_cfg,
        )
        td_force_next, td_context_next = td_baseline_step_torch(
            velocity=velocity,
            acceleration=td_context[:, 0:1],
            td_context=td_context,
            dt=dt,
            rho=float(rho),
            diameter=float(diameter),
            params=step_params,
        )
        y_next, v_next, a_next = structural_step_constant_force_torch(
            y=z[:, 0:1],
            velocity=velocity,
            force=td_force_next,
            dt=dt,
            mass=structural_mass,
            damping_c=damping_c,
            stiffness=stiffness,
        )
        z = torch.cat([y_next, v_next * structural_mass], dim=1)
        td_context = td_context_next.clone()
        td_context[:, 0:1] = a_next
        z_hist.append(z)
        total_force_hist.append(td_force_next)
    z_seq = torch.stack(z_hist, dim=1)
    total_force_seq = torch.stack(total_force_hist, dim=1) if total_force_hist else z0.new_zeros((z0.shape[0], 0, 1))
    return z_seq, total_force_seq


def _build_td_correction_hnn_loaders(
    *,
    train_trajs: list[dict[str, np.ndarray]],
    val_trajs: list[dict[str, np.ndarray]],
    input_scaling_mode: str,
    diameter: float,
    batch_size: int,
    rollout_batch_size: int,
    rollout_steps: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader | None, DataLoader | None, DataLoader | None]:
    # These datasets are fully materialized in CPU tensors, so worker processes add
    # little value and can crash on CUDA clusters due to forked worker initialization.
    loader_num_workers = 0
    mass_key = "dry_mass_kg"

    def _one_step_dataset(trajs: list[dict[str, np.ndarray]]) -> TensorDataset | None:
        tensors: list[TensorDataset] = []
        for traj in trajs:
            y = torch.from_numpy(np.ascontiguousarray(traj["y"])).float()
            dy = torch.from_numpy(np.ascontiguousarray(traj["dy"])).float()
            t = torch.from_numpy(np.ascontiguousarray(traj["t"])).float()
            flow_feature = torch.from_numpy(
                np.ascontiguousarray(
                    _td_flow_feature_from_traj(
                        traj,
                        input_scaling_mode=input_scaling_mode,
                        diameter=diameter,
                    )
                )
            ).float().unsqueeze(1)
            force_true = torch.from_numpy(np.ascontiguousarray(traj["force_per_m"])).float().unsqueeze(1)
            td_context = torch.from_numpy(np.ascontiguousarray(traj["td_context"])).float()
            mass = torch.full((y.shape[0], 1), float(np.asarray(traj[mass_key]).reshape(())), dtype=torch.float32)
            damping = torch.full((y.shape[0], 1), float(np.asarray(traj["damping_c"]).reshape(())), dtype=torch.float32)
            stiffness = torch.full((y.shape[0], 1), float(np.asarray(traj["stiffness_n_m"]).reshape(())), dtype=torch.float32)
            z = torch.cat([y.unsqueeze(1), dy.unsqueeze(1) * mass], dim=1)
            items: list[torch.Tensor] = [
                z[:-1],
                t[:-1].unsqueeze(1),
                z[1:],
                t[1:].unsqueeze(1),
                flow_feature[:-1],
                force_true[1:],
                td_context[:-1],
                mass[:-1],
                damping[:-1],
                stiffness[:-1],
            ]
            tensors.append(TensorDataset(*items))
        if not tensors:
            return None
        return tensors[0] if len(tensors) == 1 else ConcatDataset(tensors)

    def _rollout_dataset(trajs: list[dict[str, np.ndarray]]) -> TensorDataset | None:
        if rollout_steps < 1:
            return None
        tensors: list[TensorDataset] = []
        window = int(rollout_steps) + 1
        for traj in trajs:
            y = torch.from_numpy(np.ascontiguousarray(traj["y"])).float()
            dy = torch.from_numpy(np.ascontiguousarray(traj["dy"])).float()
            t = torch.from_numpy(np.ascontiguousarray(traj["t"])).float()
            flow_feature = torch.from_numpy(
                np.ascontiguousarray(
                    _td_flow_feature_from_traj(
                        traj,
                        input_scaling_mode=input_scaling_mode,
                        diameter=diameter,
                    )
                )
            ).float().unsqueeze(1)
            td_context = torch.from_numpy(np.ascontiguousarray(traj["td_context"])).float()
            mass = torch.full((y.shape[0], 1), float(np.asarray(traj[mass_key]).reshape(())), dtype=torch.float32)
            damping = torch.full((y.shape[0], 1), float(np.asarray(traj["damping_c"]).reshape(())), dtype=torch.float32)
            stiffness = torch.full((y.shape[0], 1), float(np.asarray(traj["stiffness_n_m"]).reshape(())), dtype=torch.float32)
            z = torch.cat([y.unsqueeze(1), dy.unsqueeze(1) * mass], dim=1)
            if z.shape[0] < window:
                continue
            z0_list = []
            t_list = []
            ztraj_list = []
            ur0_list = []
            td0_list = []
            mass0_list = []
            damping0_list = []
            stiffness0_list = []
            for start in range(z.shape[0] - window + 1):
                end = start + window
                z0_list.append(z[start])
                t_list.append(t[start:end])
                ztraj_list.append(z[start:end])
                ur0_list.append(flow_feature[start])
                td0_list.append(td_context[start])
                mass0_list.append(mass[start])
                damping0_list.append(damping[start])
                stiffness0_list.append(stiffness[start])
            items: list[torch.Tensor] = [
                torch.stack(z0_list, dim=0),
                torch.stack(t_list, dim=0),
                torch.stack(ztraj_list, dim=0),
                torch.stack(ur0_list, dim=0),
                torch.stack(td0_list, dim=0),
                torch.stack(mass0_list, dim=0),
                torch.stack(damping0_list, dim=0),
                torch.stack(stiffness0_list, dim=0),
            ]
            tensors.append(TensorDataset(*items))
        if not tensors:
            return None
        return tensors[0] if len(tensors) == 1 else ConcatDataset(tensors)

    train_dataset = _one_step_dataset(train_trajs)
    if train_dataset is None:
        raise ValueError("No TD correction training samples were built.")
    val_dataset = _one_step_dataset(val_trajs)
    train_rollout_dataset = _rollout_dataset(train_trajs)
    val_rollout_dataset = _rollout_dataset(val_trajs)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=loader_num_workers,
        pin_memory=pin_memory,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=loader_num_workers,
            pin_memory=pin_memory,
        )
    train_rollout_loader = None
    if train_rollout_dataset is not None:
        train_rollout_loader = DataLoader(
            train_rollout_dataset,
            batch_size=rollout_batch_size,
            shuffle=False,
            num_workers=loader_num_workers,
            pin_memory=pin_memory,
        )
    val_rollout_loader = None
    if val_rollout_dataset is not None:
        val_rollout_loader = DataLoader(
            val_rollout_dataset,
            batch_size=rollout_batch_size,
            shuffle=False,
            num_workers=loader_num_workers,
            pin_memory=pin_memory,
        )
    return train_loader, val_loader, train_rollout_loader, val_rollout_loader


def _log_td_correction_rollout_validation(
    *,
    writer: SummaryWriter,
    epoch: int,
    model: PHVIV,
    traj: dict[str, np.ndarray],
    dt: float,
    td_params: dict[str, float],
    td_memory_cfg: dict[str, Any],
    device: torch.device,
    mean_active: bool,
    predict_sigma: bool,
    fhat_active: bool,
    td_force_input_source: str,
    fhat_bound_multiplier: float,
    force_zero_output: bool = False,
    tag_prefix: str = "val/rollout",
    step: int | None = None,
    log_metrics: bool = True,
    log_plots: bool = True,
    log_phase_map: bool = False,
    log_correction_on_data: bool = False,
    title_suffix: str = "",
    log_spectra: bool = False,
    log_only_spectra: bool = False,
) -> dict[str, float]:
    def _run_rollout_validation_case() -> dict[str, Any]:
        mass_key = "dry_mass_kg"
        mass_value = float(np.asarray(traj[mass_key]).reshape(()))
        damping_value = float(np.asarray(traj["damping_c"]).reshape(()))
        stiffness_value = float(np.asarray(traj["stiffness_n_m"]).reshape(()))
        y_true_t = torch.from_numpy(np.ascontiguousarray(traj["y"])).float().unsqueeze(1).to(device)
        v_true_t = torch.from_numpy(np.ascontiguousarray(traj["dy"])).float().unsqueeze(1).to(device)
        z_true_t = torch.cat([y_true_t, v_true_t * mass_value], dim=1)
        f_true_t = torch.from_numpy(np.ascontiguousarray(traj["force_per_m"])).float().unsqueeze(1).to(device)
        ur_t = torch.from_numpy(
            np.ascontiguousarray(
                _td_flow_feature_from_traj(
                    traj,
                    input_scaling_mode=str(getattr(model, "input_scaling_mode", "current")),
                    diameter=float(model.D),
                )
            )
        ).float().unsqueeze(1).to(device)
        td_context_np = np.asarray(traj["td_context"], dtype=float)
        td_context_t = torch.from_numpy(np.ascontiguousarray(td_context_np)).float().to(device)
        t_np = np.asarray(traj["t"], dtype=float).reshape(-1)
        if z_true_t.shape[0] < 2:
            return {}
        traj_dt = float(t_np[1] - t_np[0])

        mass_t = torch.full((1, 1), mass_value, dtype=z_true_t.dtype, device=device)
        damping_t = torch.full((1, 1), damping_value, dtype=z_true_t.dtype, device=device)
        stiffness_t = torch.full((1, 1), stiffness_value, dtype=z_true_t.dtype, device=device)
        z_pred, total_force_seq, corr_mu_seq, sigma_roll_seq, delta_fhat_seq = _td_correction_state_rollout(
            model=model,
            z0=z_true_t[0:1],
            ur0=ur_t[0:1],
            td_context0=td_context_t[0:1],
            steps=int(z_true_t.shape[0] - 1),
            dt=traj_dt,
            structural_mass=mass_t,
            damping_c=damping_t,
            stiffness=stiffness_t,
            td_params=td_params,
            td_memory_cfg=td_memory_cfg,
            mean_active=mean_active,
            sigma_active=predict_sigma,
            fhat_active=fhat_active,
            td_force_input_source=td_force_input_source,
            fhat_bound_multiplier=fhat_bound_multiplier,
            force_zero_output=force_zero_output,
        )
        y_pred = z_pred[0, :, 0].detach().cpu().numpy()
        v_pred = (z_pred[0, :, 1] / mass_value).detach().cpu().numpy()
        force_roll = total_force_seq[0, :, 0].detach().cpu().numpy()
        corr_roll = corr_mu_seq[0, :, 0].detach().cpu().numpy()
        sigma_roll = sigma_roll_seq[0, :, 0].detach().cpu().numpy()
        delta_fhat_roll = delta_fhat_seq[0, :, 0].detach().cpu().numpy()
        td_roll = force_roll - corr_roll

        with torch.no_grad():
            step_on_data = _td_step_with_corrections(
                model=model,
                z=z_true_t[:-1],
                reduced_velocity=ur_t[:-1],
                td_context=td_context_t[:-1],
                dt=traj_dt,
                structural_mass=torch.full((z_true_t.shape[0] - 1, 1), mass_value, dtype=z_true_t.dtype, device=device),
                damping_c=torch.full((z_true_t.shape[0] - 1, 1), damping_value, dtype=z_true_t.dtype, device=device),
                stiffness=torch.full((z_true_t.shape[0] - 1, 1), stiffness_value, dtype=z_true_t.dtype, device=device),
                td_params=td_params,
                td_memory_cfg=td_memory_cfg,
                mean_active=mean_active,
                sigma_active=predict_sigma,
                fhat_active=fhat_active,
                td_force_input_source=td_force_input_source,
                fhat_bound_multiplier=fhat_bound_multiplier,
                force_zero_output=force_zero_output,
            )
            corr_on_data = step_on_data["corr_mu"]
            sigma_on_data = step_on_data["sigma_corr"]
            td_force_on_data = step_on_data["td_force_next"]
            total_force_on_data = step_on_data["total_force_next"]
            delta_fhat_on_data = step_on_data["delta_fhat"]
        force_total_full = np.concatenate(
            [np.asarray([float(total_force_on_data[0, 0].detach().cpu())]), force_roll],
            axis=0,
        )
        force_td_full = np.concatenate([td_force_on_data[:1, 0].detach().cpu().numpy(), td_roll], axis=0)

        metrics = compute_validation_metrics(
            model=model,
            y_data_t=y_true_t[:, 0],
            val_vel=v_true_t[:, 0],
            reduced_velocity=ur_t[:, 0],
            m_eff=mass_value,
            dt=traj_dt,
            t=t_np,
            y_data_raw=y_true_t[:, 0].detach().cpu().numpy(),
            force_data=f_true_t[:, 0].detach().cpu().numpy(),
            D=float(model.D),
            k=stiffness_value,
            device=device,
            rollout={
                "y_norm": y_pred / float(model.D),
                "p_norm": v_pred / (float(np.sqrt(stiffness_value / mass_value)) * float(model.D)),
                "force_total": force_total_full,
            },
        )
        force_true = f_true_t[1:, 0].detach().cpu().numpy()
        force_model_on_data = total_force_on_data[:, 0].detach().cpu().numpy()
        force_std = float(np.std(force_true))
        if force_std <= 0.0:
            force_std = 1.0
        metrics[FORCE_MAPPING_NRMSE_KEY] = float(np.sqrt(np.mean((force_model_on_data - force_true) ** 2))) / force_std
        if fhat_active:
            metrics["Delta fhat mean abs"] = float(torch.mean(torch.abs(delta_fhat_on_data)).detach().cpu())
            metrics["Delta fhat mean"] = float(torch.mean(delta_fhat_on_data).detach().cpu())

        omega = float(np.sqrt(stiffness_value / mass_value))
        q_true_norm = y_true_t[:, 0].detach().cpu().numpy() / float(model.D)
        p_true_norm = v_true_t[:, 0].detach().cpu().numpy() / (omega * float(model.D))
        q_pred_norm = y_pred / float(model.D)
        p_pred_norm = v_pred / (omega * float(model.D))
        return {
            "metrics": metrics,
            "t_np": t_np,
            "traj_dt": traj_dt,
            "ur_val": float(ur_t[0, 0].detach().cpu().item()),
            "q_true_norm": q_true_norm,
            "p_true_norm": p_true_norm,
            "q_pred_norm": q_pred_norm,
            "p_pred_norm": p_pred_norm,
            "force_true_full": f_true_t[:, 0].detach().cpu().numpy(),
            "force_total_full": force_total_full,
            "force_td_full": force_td_full,
            "corr_roll": corr_roll,
            "sigma_roll": sigma_roll,
            "delta_fhat_roll": delta_fhat_roll,
            "td_force_on_data": td_force_on_data[:, 0].detach().cpu().numpy(),
            "corr_on_data": corr_on_data[:, 0].detach().cpu().numpy(),
            "sigma_on_data": sigma_on_data[:, 0].detach().cpu().numpy(),
            "delta_fhat_on_data": delta_fhat_on_data[:, 0].detach().cpu().numpy(),
        }

    base_result = _run_rollout_validation_case()
    if not base_result:
        return {}
    metrics = dict(base_result["metrics"])

    if log_metrics:
        for name, value in metrics.items():
            if np.isfinite(float(value)):
                writer.add_scalar(f"val/{name}", float(value), epoch)

    t_np = np.asarray(base_result["t_np"], dtype=float)
    ur_val = float(base_result["ur_val"])
    q_true_norm = np.asarray(base_result["q_true_norm"], dtype=float)
    p_true_norm = np.asarray(base_result["p_true_norm"], dtype=float)
    q_pred_norm = np.asarray(base_result["q_pred_norm"], dtype=float)
    p_pred_norm = np.asarray(base_result["p_pred_norm"], dtype=float)
    force_true_full = np.asarray(base_result["force_true_full"], dtype=float)
    force_total_full = np.asarray(base_result["force_total_full"], dtype=float)
    force_td_full = np.asarray(base_result["force_td_full"], dtype=float)
    corr_roll = np.asarray(base_result["corr_roll"], dtype=float)
    sigma_roll = np.asarray(base_result["sigma_roll"], dtype=float)
    delta_fhat_roll = np.asarray(base_result["delta_fhat_roll"], dtype=float)
    td_force_on_data = np.asarray(base_result["td_force_on_data"], dtype=float)
    corr_on_data = np.asarray(base_result["corr_on_data"], dtype=float)
    sigma_on_data = np.asarray(base_result["sigma_on_data"], dtype=float)
    delta_fhat_on_data = np.asarray(base_result["delta_fhat_on_data"], dtype=float)

    if log_plots or (log_only_spectra and log_spectra):
        zoom_mask = create_zoom_mask(t_np)
        n_force = min(len(t_np), len(force_total_full), len(force_true_full), len(force_td_full))
        force_t = t_np[:n_force]
        if log_plots:
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
            log_force_plots(
                writer,
                epoch,
                force_t,
                force_total_full[:n_force],
                force_true_full[:n_force],
                create_zoom_mask(force_t),
                reduced_velocity=ur_val,
                force_coeff_baseline=force_td_full[:n_force],
                force_coeff_delta_pred=(corr_roll if mean_active else None),
                force_coeff_sigma_pred=(sigma_roll if predict_sigma else None),
                delta_fhat_pred=(delta_fhat_roll if fhat_active else None),
                delta_fhat_t=(t_np[1 : 1 + delta_fhat_roll.shape[0]] if fhat_active else None),
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
                force_true=force_true_full[:n_force],
                force_pred=force_total_full[:n_force],
                reduced_velocity=ur_val,
                force_baseline=force_td_full[:n_force],
                force_baseline_label="C_F (Vivana-TD)",
                tag=f"{tag_prefix}_spectra",
                step=step,
                title_suffix=title_suffix,
            )
        if log_plots and log_correction_on_data:
            output_label = "Correction coefficient" if str(getattr(model, "force_output", "force")) == "coefficient" else "Correction force"
            log_correction_on_data_plot(
                writer,
                epoch,
                t=t_np[1:],
                corr_true=force_true_full[1:] - td_force_on_data,
                corr_pred=corr_on_data,
                sigma=(
                    sigma_on_data
                    if predict_sigma
                    else None
                ),
                reduced_velocity=ur_val,
                value_label=output_label,
                sigma_label=("Sigma coefficient" if str(getattr(model, "force_output", "force")) == "coefficient" else "Sigma force"),
                tag="final_val/correction_on_data",
                step=step,
                title_suffix=title_suffix,
            )
            if fhat_active:
                log_correction_on_data_plot(
                    writer,
                    epoch,
                    t=t_np[1:],
                    corr_true=np.zeros_like(delta_fhat_roll),
                    corr_pred=delta_fhat_roll,
                    sigma=None,
                    reduced_velocity=ur_val,
                    value_label="Delta fhat",
                    sigma_label="",
                    tag="final_val/delta_fhat_on_rollout",
                    step=step,
                    title_suffix=title_suffix,
                    trajectory_label="rollout",
                )
        if log_phase_map:
            phase_specs: list[tuple[str, np.ndarray, bool]] = []
            if mean_active:
                output_label = "Correction coefficient" if str(getattr(model, "force_output", "force")) == "coefficient" else "Correction force"
                phase_specs.append((output_label, corr_roll, True))
            if predict_sigma:
                sigma_label = "Sigma coefficient" if str(getattr(model, "force_output", "force")) == "coefficient" else "Sigma force"
                phase_specs.append((sigma_label, sigma_roll, False))
            if fhat_active:
                phase_specs.append(("Delta fhat", delta_fhat_roll, True))
            log_rollout_phase_trajectory_plot(
                writer,
                epoch,
                q_true=q_true_norm,
                p_true=p_true_norm,
                q_pred=q_pred_norm,
                p_pred=p_pred_norm,
                value_specs=phase_specs,
                reduced_velocity=ur_val,
                tag="final_val/phase_rollout_outputs",
                step=step,
                title_suffix=title_suffix,
            )
    return metrics


def _rollout_loss_from_batch(
    *,
    model: PHVIV,
    batch: Any,
    device: torch.device,
    non_blocking: bool,
    return_per_sample: bool = False,
) -> torch.Tensor:
    z0, t_seq, z_traj, ur0, _history0, _scale = _parse_rollout_batch(batch)
    z0 = z0.to(device, non_blocking=non_blocking)
    t_seq = t_seq.to(device, non_blocking=non_blocking)
    z_traj = z_traj.to(device, non_blocking=non_blocking)
    ur0 = ur0.to(device, non_blocking=non_blocking)
    z_pred, _ = model.rollout(
        z0,
        t_seq,
        float(model.dt),
        reduced_velocity=ur0,
        stochastic=False,
    )
    z_scale = model.res_scale.to(device=z_pred.device, dtype=z_pred.dtype).view(1, 1, -1)
    err = (z_pred - z_traj) / z_scale
    per = torch.mean(err[..., 0] * err[..., 0], dim=1) + torch.mean(err[..., 1] * err[..., 1], dim=1)
    if return_per_sample:
        return per
    return torch.mean(per)


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


def _train_one_epoch(
    *,
    model: torch.nn.Module,
    opt: optim.Optimizer,
    train_loader: Any,
    train_rollout_loader: Any | None,
    device: torch.device,
    non_blocking: bool,
    max_grad_norm: float,
    mean_reg: float,
    mean_reg_norm: str,
    sigma_reg: float,
    sigma_reg_norm: str,
    rollout_det_weight: float,
    gradnorm_balancer: Optional[GradNormBalancer],
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    scaler: torch.amp.GradScaler,
    log_component_grad_norms: bool,
) -> dict[str, float]:
    batch_count = 0
    loss_sum = torch.zeros((), device=device)
    res_loss_sum = torch.zeros((), device=device)
    sigma_reg_sum = torch.zeros((), device=device)
    mean_reg_sum = torch.zeros((), device=device)
    rollout_det_loss_sum = torch.zeros((), device=device)
    grad_norm_sum = torch.zeros((), device=device)
    avg_sigma_reg_sum = torch.zeros((), device=device)
    avg_mean_reg_sum = torch.zeros((), device=device)
    res_grad_component_sum = torch.zeros((), device=device)
    sigma_grad_component_sum = torch.zeros((), device=device)
    mean_grad_component_sum = torch.zeros((), device=device)
    gradnorm_res_weight_sum = torch.zeros((), device=device)
    gradnorm_weight_count = 0

    rollout_iter = iter(train_rollout_loader) if (train_rollout_loader is not None and float(rollout_det_weight) > 0.0) else None
    for batch in train_loader:
        z_i, t_i, z_next, t_next, ur_i, _history_i, _f_i, _f_next, _scale = _parse_hnn_batch(batch)
        z_i = z_i.to(device, non_blocking=non_blocking)
        t_i = t_i.to(device, non_blocking=non_blocking)
        z_next = z_next.to(device, non_blocking=non_blocking)
        t_next = t_next.to(device, non_blocking=non_blocking)
        ur_i = ur_i.to(device, non_blocking=non_blocking)
        opt.zero_grad()

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
            res_loss = model.res_loss(
                z_i,
                t_i,
                z_next,
                t_next,
                reduced_velocity=ur_i,
            )
            sigma_reg_loss = model.avg_sigma_reg_SRK4(
                z_i,
                t_i,
                z_next,
                t_next,
                reduced_velocity=ur_i,
                norm=sigma_reg_norm,
            )
            mean_reg_loss = model.avg_mean_reg_SRK4(
                z_i,
                t_i,
                z_next,
                t_next,
                reduced_velocity=ur_i,
                norm=mean_reg_norm,
            )
            base_sigma_reg_loss = sigma_reg_loss
            base_mean_reg_loss = mean_reg_loss

            if gradnorm_balancer is not None:
                loss_inputs: dict[str, torch.Tensor] = {
                    "residual": res_loss.float(),
                }
                weights = gradnorm_balancer.update(loss_inputs)
                res_weight = weights["residual"]
                sigma_weight = res_loss.new_tensor(1.0)
                mean_weight = res_loss.new_tensor(1.0)
                gradnorm_res_weight_sum = gradnorm_res_weight_sum + res_weight
                gradnorm_weight_count += 1
            else:
                res_weight = res_loss.new_tensor(1.0)
                sigma_weight = res_loss.new_tensor(1.0)
                mean_weight = res_loss.new_tensor(1.0)

            # GradNorm balances raw branch losses first; user multipliers are applied after.
            gradnorm_weighted_res = res_weight * res_loss
            gradnorm_weighted_sigma = sigma_weight * base_sigma_reg_loss
            gradnorm_weighted_mean = mean_weight * base_mean_reg_loss

            weighted_sigma = float(sigma_reg) * gradnorm_weighted_sigma
            weighted_mean = float(mean_reg) * gradnorm_weighted_mean
            if rollout_iter is not None:
                try:
                    rollout_batch = next(rollout_iter)
                except StopIteration:
                    rollout_iter = iter(train_rollout_loader)
                    rollout_batch = next(rollout_iter)
                rollout_det_loss = _rollout_loss_from_batch(
                    model=model,
                    batch=rollout_batch,
                    device=device,
                    non_blocking=non_blocking,
                )
            else:
                rollout_det_loss = res_loss.new_tensor(0.0)
            weighted_rollout_det = float(rollout_det_weight) * rollout_det_loss
            loss = (
                gradnorm_weighted_res
                + weighted_sigma
                + weighted_mean
                + weighted_rollout_det
            ).float()

        if log_component_grad_norms and scaler.is_enabled():
            raise ValueError(
                "monitoring.log_component_grad_norms is not supported with AMP fp16 (GradScaler enabled)."
            )
        if log_component_grad_norms:
            gradnorm_weighted_res.backward(retain_graph=True)
            res_grad_component_sum = res_grad_component_sum + torch.as_tensor(
                compute_model_grad_norm(model), device=device
            )
            model.zero_grad(set_to_none=True)
            weighted_sigma.backward(retain_graph=True)
            sigma_grad_component_sum = sigma_grad_component_sum + torch.as_tensor(
                compute_model_grad_norm(model), device=device
            )
            model.zero_grad(set_to_none=True)
            weighted_mean.backward(retain_graph=True)
            mean_grad_component_sum = mean_grad_component_sum + torch.as_tensor(
                compute_model_grad_norm(model), device=device
            )
            model.zero_grad(set_to_none=True)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
        else:
            loss.backward()

        grad_norm = nn_utils.clip_grad_norm_(model.parameters(), max_norm=float(max_grad_norm))
        if isinstance(grad_norm, torch.Tensor):
            grad_norm_sum = grad_norm_sum + grad_norm.detach()
        else:
            grad_norm_sum = grad_norm_sum + torch.tensor(float(grad_norm), device=device)

        if scaler.is_enabled():
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()

        batch_count += 1
        loss_sum = loss_sum + loss.detach()
        res_loss_sum = res_loss_sum + res_loss.detach().float()
        # Log loss_reg as the raw regularizer magnitude (before sigma_reg scaling).
        sigma_reg_sum = sigma_reg_sum + base_sigma_reg_loss.detach().float()
        mean_reg_sum = mean_reg_sum + base_mean_reg_loss.detach().float()
        rollout_det_loss_sum = rollout_det_loss_sum + rollout_det_loss.detach().float()
        avg_sigma_reg_sum = avg_sigma_reg_sum + sigma_reg_loss.detach().float()
        avg_mean_reg_sum = avg_mean_reg_sum + mean_reg_loss.detach().float()

    denom = float(max(batch_count, 1))
    metrics: dict[str, float] = {
        "mean_loss": float((loss_sum / denom).detach().cpu()),
        "mean_res_loss": float((res_loss_sum / denom).detach().cpu()),
        "mean_force_loss": float((sigma_reg_sum / denom).detach().cpu()),
        "mean_sigma_reg_loss": float((sigma_reg_sum / denom).detach().cpu()),
        "mean_mean_reg_loss": float((mean_reg_sum / denom).detach().cpu()),
        "mean_rollout_det_loss": float((rollout_det_loss_sum / denom).detach().cpu()),
        "mean_grad_norm": float((grad_norm_sum / denom).detach().cpu()),
        "mean_force": float((avg_sigma_reg_sum / denom).detach().cpu()),
        "mean_sigma_reg": float((avg_sigma_reg_sum / denom).detach().cpu()),
        "mean_mean_reg": float((avg_mean_reg_sum / denom).detach().cpu()),
        "mean_res_grad_component": float((res_grad_component_sum / denom).detach().cpu()),
        "mean_force_grad_component": float((sigma_grad_component_sum / denom).detach().cpu()),
        "mean_sigma_grad_component": float((sigma_grad_component_sum / denom).detach().cpu()),
        "mean_mean_grad_component": float((mean_grad_component_sum / denom).detach().cpu()),
    }
    if gradnorm_weight_count > 0:
        metrics["mean_gradnorm_weight_residual"] = float(
            (gradnorm_res_weight_sum / float(gradnorm_weight_count)).detach().cpu()
        )
    return metrics


def _validate_if_needed(
    *,
    writer: SummaryWriter,
    epoch: int,
    validate_now: bool,
    rollout_now: bool,
    model: PHVIV,
    y_data_t: torch.Tensor,
    val_vel: torch.Tensor,
    reduced_velocity: torch.Tensor,
    val_series_raw: list[tuple[np.ndarray, np.ndarray, float, np.ndarray | None, np.ndarray | None, np.ndarray]] | None,
    val_sequences: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] | None,
    val_loader: Any | None,
    val_rollout_loader: Any | None,
    m_eff: float,
    dt: float,
    t: np.ndarray,
    y_true_norm: np.ndarray,
    y_data: np.ndarray,
    force_data: np.ndarray | None,
    D: float,
    k: float,
    device: torch.device,
    middle_time_plot,
    hamiltonian_data,
    log_extra_validation_metrics: bool,
    mean_reg: float,
    mean_reg_norm: str,
    sigma_reg: float,
    sigma_reg_norm: str,
    rollout_det_weight: float,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> None:
    if not validate_now and not rollout_now:
        return
    validation_start = time.perf_counter()
    def _log_validation_timing() -> None:
        elapsed = float(time.perf_counter() - validation_start)
        writer.add_scalar("val/validation_wall_time_s", elapsed, epoch + 1)
    if validate_now and val_loader is not None:
        val_loss_metrics = _evaluate_val_losses(
            model=model,
            loader=val_loader,
            device=device,
            non_blocking=(device.type == "cuda"),
            rollout_loader=val_rollout_loader,
            mean_reg=mean_reg,
            mean_reg_norm=mean_reg_norm,
            sigma_reg=sigma_reg,
            sigma_reg_norm=sigma_reg_norm,
            rollout_det_weight=rollout_det_weight,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        for name, value in val_loss_metrics.items():
            writer.add_scalar(f"val/{name}", value, epoch + 1)
    if not rollout_now:
        _log_validation_timing()
        return

    if val_series_raw is not None and val_sequences is not None:
        metrics_sum: dict[str, float] = {}
        count = 0
        diverged_count = 0
        total = min(len(val_series_raw), len(val_sequences))
        sampled_indices = list(range(total))
        for idx in sampled_indices:
            series_raw = val_series_raw[idx]
            sequence = val_sequences[idx]
            y_np, t_np, dt_value, _vel_np, force_np, _ur_np = series_raw
            y_tensor, vel_tensor, _t_tensor, ur_tensor = sequence
            if force_np is None:
                continue
            metrics = compute_validation_metrics(
                model=model,
                y_data_t=y_tensor,
                val_vel=vel_tensor,
                reduced_velocity=ur_tensor,
                m_eff=m_eff,
                dt=dt_value,
                t=t_np,
                y_data_raw=y_np,
                force_data=force_np,
                D=D,
                k=k,
                device=device,
                log_extra_metrics=log_extra_validation_metrics,
            )
            diverged_flag = float(metrics.get(ROLLOUT_DIVERGED_KEY, 0.0))
            if np.isfinite(diverged_flag) and diverged_flag > 0.5:
                diverged_count += 1
            for name, value in metrics.items():
                if name == ROLLOUT_DIVERGED_KEY:
                    continue
                metrics_sum[name] = metrics_sum.get(name, 0.0) + float(value)
            count += 1
        if count > 0:
            for name, total in metrics_sum.items():
                writer.add_scalar(f"val/{name}", total / float(count), epoch + 1)
            writer.add_scalar(f"val/{ROLLOUT_DIVERGED_COUNT_KEY}", float(diverged_count), epoch + 1)
        ur_for_rollout = [
            float(np.asarray(series_raw[5]).reshape(-1)[0])
            for series_raw in val_series_raw
            if np.asarray(series_raw[5]).reshape(-1).size > 0
        ]
        selected_indices = sample_one_index_per_ur(ur_for_rollout, seed=0)
        if not selected_indices:
            selected_indices = list(range(total))
        rollout_idx = selected_indices[0]
        series_raw = val_series_raw[rollout_idx]
        sequence = val_sequences[rollout_idx]
        y_np, t_np, dt_value, _vel_np, force_np, _ur_np = series_raw
        y_tensor, vel_tensor, _t_tensor, ur_tensor = sequence
        log_validation_epoch(
            writer,
            epoch + 1,
            model,
            y_tensor,
            vel_tensor,
            ur_tensor,
            m_eff,
            dt_value,
            t_np,
            y_np / D,
            y_np,
            force_np,
            D,
            k,
            device,
            middle_time_plot,
            hamiltonian_data,
            log_extra_metrics=log_extra_validation_metrics,
            log_metrics=False,
            log_spectra=False,
        )
        _log_validation_timing()
        return
    log_validation_epoch(
        writer,
        epoch + 1,
        model,
        y_data_t,
        val_vel,
        reduced_velocity,
        m_eff,
        dt,
        t,
        y_true_norm,
        y_data,
        force_data,
        D,
        k,
        device,
        middle_time_plot,
        hamiltonian_data,
        log_extra_metrics=log_extra_validation_metrics,
        log_spectra=False,
    )
    _log_validation_timing()


def _log_final_rollouts_all(
    *,
    writer: SummaryWriter,
    epoch: int,
    model: PHVIV,
    val_series_raw: list[tuple[np.ndarray, np.ndarray, float, np.ndarray | None, np.ndarray | None, np.ndarray]],
    val_sequences: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    m_eff: float,
    D: float,
    k: float,
    device: torch.device,
    middle_time_plot,
    log_extra_validation_metrics: bool,
    extra_series_raw: list[tuple[np.ndarray, np.ndarray, float, np.ndarray | None, np.ndarray | None, np.ndarray]] | None = None,
    extra_sequences: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] | None = None,
) -> tuple[dict[str, float], int, list[float], list[dict[str, float]]]:
    total = min(len(val_series_raw), len(val_sequences))
    if total <= 0:
        return {}, 0, [], []
    metric_pairs: list[tuple[tuple[np.ndarray, np.ndarray, float, np.ndarray | None, np.ndarray | None, np.ndarray], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]] = []
    seen_metric_ur: set[float] = set()
    for idx in range(total):
        ur_arr = np.asarray(val_series_raw[idx][5]).reshape(-1)
        if ur_arr.size == 0:
            continue
        ur_key = round(float(ur_arr[0]), 6)
        if ur_key in seen_metric_ur:
            continue
        seen_metric_ur.add(ur_key)
        metric_pairs.append((val_series_raw[idx], val_sequences[idx]))
    if not metric_pairs:
        return {}, 0, [], []
    plot_pairs_by_ur: dict[float, tuple[tuple[np.ndarray, np.ndarray, float, np.ndarray | None, np.ndarray | None, np.ndarray], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]] = {}
    for series_raw, sequence in metric_pairs:
        ur_arr = np.asarray(series_raw[5]).reshape(-1)
        if ur_arr.size == 0:
            continue
        plot_pairs_by_ur[round(float(ur_arr[0]), 6)] = (series_raw, sequence)
    if extra_series_raw is not None and extra_sequences is not None:
        extra_total = min(len(extra_series_raw), len(extra_sequences))
        for idx in range(extra_total):
            ur_arr = np.asarray(extra_series_raw[idx][5]).reshape(-1)
            if ur_arr.size == 0:
                continue
            ur_key = round(float(ur_arr[0]), 6)
            if ur_key not in plot_pairs_by_ur:
                plot_pairs_by_ur[ur_key] = (extra_series_raw[idx], extra_sequences[idx])
    plot_pairs = [plot_pairs_by_ur[key] for key in sorted(plot_pairs_by_ur)]
    metrics_sum: dict[str, float] = {}
    metrics_count: dict[str, int] = {}
    used = 0
    plot_ur_values: list[float] = []
    plot_metrics_list: list[dict[str, float]] = []
    for series_raw, sequence in metric_pairs:
        y_np, t_np, dt_value, _vel_np, force_np, _ur_np = series_raw
        y_tensor, vel_tensor, _t_tensor, ur_tensor = sequence
        ur_val = float(np.asarray(ur_tensor.detach().cpu()).reshape(-1)[0])
        metrics = log_validation_epoch(
            writer,
            epoch,
            model,
            y_tensor,
            vel_tensor,
            ur_tensor,
            m_eff,
            dt_value,
            t_np,
            y_np / D,
            y_np,
            force_np,
            D,
            k,
            device,
            middle_time_plot,
            None,
            log_extra_metrics=log_extra_validation_metrics,
            log_metrics=False,
            log_plots=False,
        )
        filtered_metrics = {
            name: float(value)
            for name, value in metrics.items()
            if name != ROLLOUT_DIVERGED_KEY and np.isfinite(float(value))
        }
        for name, value in filtered_metrics.items():
            metrics_sum[name] = metrics_sum.get(name, 0.0) + float(value)
            metrics_count[name] = metrics_count.get(name, 0) + 1
        used += 1
    for step_idx, (series_raw, sequence) in enumerate(plot_pairs, start=1):
        y_np, t_np, dt_value, _vel_np, force_np, _ur_np = series_raw
        y_tensor, vel_tensor, _t_tensor, ur_tensor = sequence
        ur_val = float(np.asarray(ur_tensor.detach().cpu()).reshape(-1)[0])
        plot_metrics = log_validation_epoch(
            writer,
            epoch,
            model,
            y_tensor,
            vel_tensor,
            ur_tensor,
            m_eff,
            dt_value,
            t_np,
            y_np / D,
            y_np,
            force_np,
            D,
            k,
            device,
            middle_time_plot,
            None,
            log_extra_metrics=log_extra_validation_metrics,
            log_metrics=False,
            log_plots=True,
            tag_prefix="final_val/rollout",
            step=step_idx,
            title_suffix=f" [final {step_idx}/{len(plot_pairs)}]",
            log_spectra=False,
        )
        filtered_plot_metrics = {
            name: float(value)
            for name, value in plot_metrics.items()
            if name != ROLLOUT_DIVERGED_KEY and np.isfinite(float(value))
        }
        if filtered_plot_metrics:
            plot_ur_values.append(ur_val)
            plot_metrics_list.append(filtered_plot_metrics)
        rollout = rollout_model(
            model,
            y_tensor,
            vel_tensor,
            ur_tensor,
            m_eff,
            dt_value,
            t_np,
            D,
            k,
            device,
        )
        log_phase_component_plots(
            writer,
            epoch,
            model,
            y_np / D,
            np.asarray(vel_tensor.detach().cpu().numpy(), dtype=float).reshape(-1) / (np.sqrt(k / m_eff) * D),
            rollout["y_norm"],
            rollout["p_norm"],
            reduced_velocity=ur_val,
            D=D,
            k=k,
            m_eff=m_eff,
            device=device,
            tag_prefix="final_val/phase",
            step=step_idx,
            title_suffix=f" [final {step_idx}/{len(plot_pairs)}]",
        )
    averaged = {
        name: metrics_sum[name] / float(metrics_count[name])
        for name in metrics_sum
        if metrics_count.get(name, 0) > 0
    }
    return averaged, used, plot_ur_values, plot_metrics_list


def _reap_async_processes(
    processes: list[dict[str, Any]],
    *,
    writer: SummaryWriter | None = None,
    best_state: dict[str, Any] | None = None,
    wait: bool = False,
) -> list[dict[str, Any]]:
    def _mirror_async_summary_scalars(payload: dict[str, Any], step: int) -> None:
        if writer is None:
            return
        split_results = payload.get("split_results", {})
        if not isinstance(split_results, dict):
            return
        for split_tag, split_payload in split_results.items():
            if not isinstance(split_payload, dict):
                continue
            metrics = split_payload.get("val_metrics", {})
            if not isinstance(metrics, dict):
                continue
            for name, value in metrics.items():
                if value is None:
                    continue
                try:
                    value_f = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(value_f):
                    writer.add_scalar(f"{split_tag}/{name}", value_f, step)
        writer.flush()

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
            print(
                f"[async-val] epoch {epoch}: completed successfully in {elapsed:.2f}s"
            )
        else:
            print(
                f"[async-val] epoch {epoch}: FAILED with exit code {return_code} "
                f"after {elapsed:.2f}s"
            )
            if summary_path.exists():
                try:
                    payload = json.loads(summary_path.read_text(encoding="utf-8"))
                    error_msg = str(payload.get("error", "")).strip()
                    if error_msg:
                        print(f"[async-val] epoch {epoch}: {error_msg}")
                except Exception:
                    pass

        if ckpt_path.exists() and return_code == 0:
            if summary_path.exists():
                try:
                    payload = json.loads(summary_path.read_text(encoding="utf-8"))
                    _mirror_async_summary_scalars(payload, epoch)
                    if best_state is None or not run_name:
                        raise StopIteration
                    val_metrics = payload.get("val_metrics", {})
                    if not isinstance(val_metrics, dict):
                        val_metrics = {}
                    best_metric_name = str(payload.get("best_metric_name", AGGREGATE_FORCE_VALIDATION_ERROR_KEY))
                    best_metric_value = val_metrics.get(best_metric_name, payload.get(best_metric_name, None))
                    if best_metric_value is None or not np.isfinite(float(best_metric_value)):
                        best_metric_name = "loss_total"
                        best_metric_value = payload.get("loss_total", None)
                    if best_metric_value is not None and np.isfinite(float(best_metric_value)):
                        best_metric_f = float(best_metric_value)
                        loss_total = payload.get("loss_total", None)
                        loss_total_f = float(loss_total) if loss_total is not None and np.isfinite(float(loss_total)) else None
                        prev_metric_name = str(best_state.get("best_metric_name", best_metric_name))
                        prev_best = (
                            float("inf")
                            if prev_metric_name != best_metric_name
                            else float(best_state.get("best_metric_value", float("inf")))
                        )
                        if best_metric_f < prev_best:
                            models_dir = Path("models")
                            models_dir.mkdir(parents=True, exist_ok=True)
                            best_model_path = models_dir / f"{run_name}_best_val.pt"
                            best_meta_path = models_dir / f"{run_name}_best_val.json"
                            shutil.copy2(ckpt_path, best_model_path)
                            best_meta = {
                                "epoch": epoch,
                                "best_metric_name": best_metric_name,
                                "best_metric_value": best_metric_f,
                                "loss_total": loss_total_f,
                                "run_name": run_name,
                                "source_checkpoint": str(ckpt_path),
                                "summary_path": str(summary_path),
                            }
                            best_meta_path.write_text(json.dumps(best_meta, indent=2, sort_keys=True), encoding="utf-8")
                            best_state.update(
                                {
                                    "epoch": epoch,
                                    "best_metric_name": best_metric_name,
                                    "best_metric_value": best_metric_f,
                                    "loss_total": loss_total_f,
                                    "best_model_path": str(best_model_path),
                                }
                            )
                            print(
                                f"[async-val] epoch {epoch}: new best {best_metric_name}={best_metric_f:.6e}; "
                                f"kept {best_model_path}"
                            )
                except StopIteration:
                    pass
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
    processes = _reap_async_processes(processes, writer=writer, best_state=best_state, wait=False)
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


def _evaluate_val_losses(
    *,
    model: PHVIV,
    loader: Any,
    rollout_loader: Any | None,
    device: torch.device,
    non_blocking: bool,
    mean_reg: float,
    mean_reg_norm: str,
    sigma_reg: float,
    sigma_reg_norm: str,
    rollout_det_weight: float,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    loss_sum = torch.zeros((), device=device)
    res_sum = torch.zeros((), device=device)
    sigma_sum = torch.zeros((), device=device)
    mean_reg_sum = torch.zeros((), device=device)
    rollout_det_sum = torch.zeros((), device=device)
    batches = 0
    rollout_iter = iter(rollout_loader) if (rollout_loader is not None and float(rollout_det_weight) > 0.0) else None
    with torch.no_grad():
        for batch in loader:
            z_i, t_i, z_next, t_next, ur_i, _history_i, _f_i, _f_next, _scale = _parse_hnn_batch(batch)
            z_i = z_i.to(device, non_blocking=non_blocking)
            t_i = t_i.to(device, non_blocking=non_blocking)
            z_next = z_next.to(device, non_blocking=non_blocking)
            t_next = t_next.to(device, non_blocking=non_blocking)
            ur_i = ur_i.to(device, non_blocking=non_blocking)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                res_loss = model.res_loss(
                    z_i,
                    t_i,
                    z_next,
                    t_next,
                    reduced_velocity=ur_i,
                )
                sigma_reg_loss = model.avg_sigma_reg_SRK4(
                    z_i,
                    t_i,
                    z_next,
                    t_next,
                    reduced_velocity=ur_i,
                    norm=sigma_reg_norm,
                )
                mean_reg_loss = model.avg_mean_reg_SRK4(
                    z_i,
                    t_i,
                    z_next,
                    t_next,
                    reduced_velocity=ur_i,
                    norm=mean_reg_norm,
                )
                sigma_loss = float(sigma_reg) * sigma_reg_loss
                mean_loss_reg = float(mean_reg) * mean_reg_loss
                total = res_loss + sigma_loss + mean_loss_reg
                if rollout_iter is not None:
                    try:
                        rollout_batch = next(rollout_iter)
                    except StopIteration:
                        rollout_iter = iter(rollout_loader)
                        rollout_batch = next(rollout_iter)
                    rollout_det_loss = _rollout_loss_from_batch(
                        model=model,
                        batch=rollout_batch,
                        device=device,
                        non_blocking=non_blocking,
                    )
                else:
                    rollout_det_loss = res_loss.new_tensor(0.0)
                total = total + float(rollout_det_weight) * rollout_det_loss

            loss_sum = loss_sum + total.detach().float()
            res_sum = res_sum + res_loss.detach().float()
            # Log loss_reg as the raw regularizer magnitude (before sigma_reg scaling).
            sigma_sum = sigma_sum + sigma_reg_loss.detach().float()
            mean_reg_sum = mean_reg_sum + mean_reg_loss.detach().float()
            rollout_det_sum = rollout_det_sum + rollout_det_loss.detach().float()
            batches += 1

    if was_training:
        model.train()
    denom = float(max(batches, 1))
    return {
        "loss_total": float((loss_sum / denom).detach().cpu()),
        "loss_physics": float((res_sum / denom).detach().cpu()),
        "loss_reg": float((sigma_sum / denom).detach().cpu()),
        "loss_reg_mean": float((mean_reg_sum / denom).detach().cpu()),
        "loss_rollout_det": float((rollout_det_sum / denom).detach().cpu()),
    }


def _per_ur_loss_map_hnn(
    *,
    model: PHVIV,
    loader: Any,
    rollout_loader: Any | None,
    device: torch.device,
    non_blocking: bool,
    mean_reg: float,
    mean_reg_norm: str,
    sigma_reg: float,
    sigma_reg_norm: str,
    rollout_det_weight: float,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> dict[str, dict[float, float]]:
    model.eval()
    amp_enabled = bool(amp_enabled) and device.type == "cuda"
    buckets: dict[str, dict[float, list[float]]] = {
        "loss_physics": {},
        "loss_reg": {},
        "loss_reg_mean": {},
        "loss_rollout_det": {},
    }
    with torch.no_grad():
        for batch in loader:
            z_i, t_i, z_next, t_next, ur_i, _history_i, _f_i, _f_next, _scale = _parse_hnn_batch(batch)
            z_i = z_i.to(device, non_blocking=non_blocking)
            t_i = t_i.to(device, non_blocking=non_blocking)
            z_next = z_next.to(device, non_blocking=non_blocking)
            t_next = t_next.to(device, non_blocking=non_blocking)
            ur_i = ur_i.to(device, non_blocking=non_blocking)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                per_res = model.res_loss_per_sample(
                    z_i,
                    t_i,
                    z_next,
                    t_next,
                    reduced_velocity=ur_i,
                )
                per_sigma_reg = model.avg_sigma_reg_SRK4_per_sample(
                    z_i,
                    t_i,
                    z_next,
                    t_next,
                    reduced_velocity=ur_i,
                    norm=sigma_reg_norm,
                )
                per_mean_reg = model.avg_mean_reg_SRK4_per_sample(
                    z_i,
                    t_i,
                    z_next,
                    t_next,
                    reduced_velocity=ur_i,
                    norm=mean_reg_norm,
                )
                per_sigma = float(sigma_reg) * per_sigma_reg
                per_mean = float(mean_reg) * per_mean_reg

            ur_vals = ur_i.detach().cpu().view(-1).numpy()
            per_res_vals = per_res.detach().cpu().view(-1).numpy()
            per_reg_vals = per_sigma.detach().cpu().view(-1).numpy()
            per_mean_vals = per_mean.detach().cpu().view(-1).numpy()
            for u, res_v, reg_v, mean_v in zip(ur_vals, per_res_vals, per_reg_vals, per_mean_vals):
                key = float(np.round(u, 6))
                buckets["loss_physics"].setdefault(key, []).append(float(res_v))
                buckets["loss_reg"].setdefault(key, []).append(float(reg_v))
                buckets["loss_reg_mean"].setdefault(key, []).append(float(mean_v))

    if rollout_loader is not None and float(rollout_det_weight) > 0.0:
        with torch.no_grad():
            for batch in rollout_loader:
                _z0, _t_seq, _z_traj, ur0, _history0, _scale = _parse_rollout_batch(batch)
                per_rollout = _rollout_loss_from_batch(
                    model=model,
                    batch=batch,
                    device=device,
                    non_blocking=non_blocking,
                    return_per_sample=True,
                )
                ur_vals = ur0.detach().cpu().view(-1).numpy()
                per_vals = per_rollout.detach().cpu().view(-1).numpy()
                for u, per_v in zip(ur_vals, per_vals):
                    key = float(np.round(float(u), 6))
                    buckets["loss_rollout_det"].setdefault(key, []).append(float(rollout_det_weight) * float(per_v))

    out: dict[str, dict[float, float]] = {}
    for name, by_ur in buckets.items():
        out[name] = {ur: float(np.mean(vals)) for ur, vals in by_ur.items()}
    return out


def _train_td_correction(config: Config, config_name: str) -> None:
    data_cfg = config.data
    model_cfg = config.model
    training_cfg = config.training
    optim_cfg = config.optim
    loss_cfg = config.loss
    constraints_cfg = config.constraints
    vivana_td_cfg = asdict(config.vivana_td)
    runtime_cfg = config.runtime
    precision_cfg = config.precision
    compile_cfg = config.compile
    monitoring_cfg = config.monitoring
    hnn_cfg = dict(config.correction or {})

    device = select_device(os.getenv("TRAIN_DEVICE", str(runtime_cfg.device)))
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA available: {torch.cuda.is_available()}, gpu0: {torch.cuda.get_device_name(0)}")
    configure_tf32(device, bool(precision_cfg.use_tf32))
    set_num_threads_from_slurm(default=1)
    non_blocking = device.type == "cuda"

    train_series_root = Path(data_cfg.train_series_dir)
    train_dir = train_series_root / "train"
    val_seen_dir = train_series_root / "val_seen"
    legacy_val_dir = train_series_root / "val"
    if not val_seen_dir.exists() and legacy_val_dir.exists():
        val_seen_dir = legacy_val_dir
    if not train_dir.exists():
        raise FileNotFoundError("TD correction mode expects train/ under data.train_series_dir.")
    train_paths = sorted(train_dir.glob("*.npz"))
    val_seen_paths = sorted(val_seen_dir.glob("*.npz")) if val_seen_dir.exists() else []
    if not train_paths:
        raise FileNotFoundError("No TD correction training trajectories were found.")

    td_params = resolve_td_correction_params(vivana_td_cfg)
    td_memory_cfg = resolve_td_memory_config(vivana_td_cfg)
    train_cut = resolve_cut_start_seconds(data_cfg, "train")
    val_cut = resolve_cut_start_seconds(data_cfg, "val")
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
        ur_source="dry",
        td_params=td_params,
        td_memory_cfg=td_memory_cfg,
    )
    val_trajs: list[dict[str, np.ndarray]] = []
    val_seen_trajs = (
        load_td_correction_trajectories(
            paths=val_seen_paths,
            cut_start_seconds=val_cut,
            reduce_time=reduce_time_enabled,
            reduction_factor=reduction_factor,
            stagger_reduced_time=stagger_val_reduce,
            ur_source="dry",
            td_params=td_params,
            td_memory_cfg=td_memory_cfg,
        )
        if val_seen_paths
        else []
    )
    dt = float(train_trajs[0]["t"][1] - train_trajs[0]["t"][0])
    correction_mode = resolve_td_correction_mode(asdict(model_cfg))
    mode_flags = td_correction_mode_flags(correction_mode)
    mean_active = bool(mode_flags["mean_active"])
    predict_sigma = False
    fhat_active = bool(mode_flags["fhat_active"])
    arch_dict = asdict(config.architecture)
    shared_td_correction_trunk_cfg = bool(arch_dict.get("shared_td_correction_trunk", False))
    model_input_cfg = asdict(model_cfg)
    model_input_cfg["correction_mode"] = correction_mode
    td_input_configs = resolve_td_input_configs(
        model_input_cfg,
        shared_td_correction_trunk=shared_td_correction_trunk_cfg,
    )
    mean_input_cfg = td_input_configs.get("mean", {})
    td_force_input_source = str(mean_input_cfg.get("td_force_input_source", "none"))
    use_td_force_input = any(bool(cfg.get("use_td_force_input", False)) for cfg in td_input_configs.values())
    use_td_fhat_input = any(bool(cfg.get("use_td_fhat_input", False)) for cfg in td_input_configs.values())
    phase_input_source = str(mean_input_cfg.get("phase_input_source", "none"))
    use_phi_input = any(bool(cfg.get("use_phi_input", False)) for cfg in td_input_configs.values())
    if bool(hnn_cfg.get("predict_sigma", False)):
        raise ValueError("correction.predict_sigma is no longer supported; sigma prediction has been removed.")
    use_sigma_inputs = any(bool(cfg.get("use_sigma_inputs", False)) for cfg in td_input_configs.values())
    use_acceleration_input = any(bool(cfg.get("use_acceleration_input", False)) for cfg in td_input_configs.values())
    constraints_dict = asdict(constraints_cfg)
    fhat_bound_multiplier = float(getattr(constraints_cfg, "fhat_bound_multiplier", 1.5))
    fhat_correction_bounds = resolve_td_fhat_correction_bounds(constraints_dict)
    if not np.isfinite(fhat_bound_multiplier) or fhat_bound_multiplier <= 0.0:
        raise ValueError("constraints.fhat_bound_multiplier must be finite and positive.")
    state_loss_mode = str(hnn_cfg.get("state_loss_mode", "mse")).strip().lower()
    if state_loss_mode != "mse":
        raise ValueError("correction.state_loss_mode now only supports mse.")
    force_zero_output = bool(getattr(constraints_cfg, "force_zero_output", False))
    corr_init_mode, corr_init_tiny_std = _resolve_td_correction_init_settings(asdict(training_cfg), model_cfg)
    rollout_det_weight = float(getattr(loss_cfg, "rollout_det_weight", 0.0))
    rollout_disp_std_weight = float(getattr(loss_cfg, "rollout_disp_std_weight", 0.0))
    rollout_disp_mean_in_std_loss = bool(getattr(loss_cfg, "rollout_disp_mean_in_std_loss", True))
    rollout_disp_spectral_weight_raw = getattr(loss_cfg, "rollout_disp_spectral_weight", None)
    if rollout_disp_spectral_weight_raw is None:
        rollout_disp_spectral_weight = float(getattr(loss_cfg, "rollout_disp_psd_weight", 0.0))
    else:
        rollout_disp_spectral_weight = float(rollout_disp_spectral_weight_raw)
    rollout_disp_spectral_loss = _normalize_rollout_disp_spectral_loss_mode(
        getattr(loss_cfg, "rollout_disp_spectral_loss", "psd")
    )
    rollout_disp_psd_peak_rel_bandwidth = float(getattr(loss_cfg, "rollout_disp_psd_peak_rel_bandwidth", 0.0))
    rollout_disp_psd_use_hann_window = bool(getattr(loss_cfg, "rollout_disp_psd_use_hann_window", True))
    rollout_loss_settings = _resolve_td_rollout_loss_settings(loss_cfg)
    rollout_disp_std_p = rollout_loss_settings["disp_std_p"]
    rollout_disp_freq_p = rollout_loss_settings["disp_freq_p"]
    rollout_disp_freq_alpha = rollout_loss_settings["disp_freq_alpha"]
    rollout_det_steps = int(getattr(loss_cfg, "rollout_det_steps", 0))
    rollout_batch_size_raw = int(getattr(loss_cfg, "rollout_det_batch_size", 0))
    rollout_batch_size = int(training_cfg.batch_size) if rollout_batch_size_raw <= 0 else rollout_batch_size_raw
    state_loss_weight = float(getattr(loss_cfg, "state_weight", 1.0))
    mean_reg = float(getattr(constraints_cfg, "mean_reg", 0.0))
    sigma_reg = 0.0
    fhat_reg = float(getattr(constraints_cfg, "fhat_reg", 0.0))
    mean_reg_norm = str(getattr(constraints_cfg, "mean_reg_norm", "l1")).strip().lower()
    sigma_reg_norm = "l2"
    fhat_reg_norm = str(getattr(constraints_cfg, "fhat_reg_norm", "l2")).strip().lower()
    rollout_det_steps_final_raw = int(getattr(loss_cfg, "rollout_det_steps_final", 0))
    rollout_det_steps_warmup_epochs = int(getattr(loss_cfg, "rollout_det_steps_warmup_epochs", 0))
    rollout_det_steps_final = rollout_det_steps if rollout_det_steps_final_raw <= 0 else rollout_det_steps_final_raw
    if rollout_det_steps_final < 0:
        raise ValueError("loss.rollout_det_steps_final must be non-negative.")
    if rollout_det_steps_warmup_epochs < 0:
        raise ValueError("loss.rollout_det_steps_warmup_epochs must be non-negative.")
    if state_loss_weight < 0.0:
        raise ValueError("loss.state_weight must be non-negative.")
    if rollout_det_weight < 0.0:
        raise ValueError("loss.rollout_det_weight must be non-negative.")
    if rollout_disp_std_weight < 0.0:
        raise ValueError("loss.rollout_disp_std_weight must be non-negative.")
    if rollout_disp_spectral_weight < 0.0:
        raise ValueError("loss.rollout_disp_spectral_weight must be non-negative.")
    if not np.isfinite(rollout_disp_psd_peak_rel_bandwidth) or rollout_disp_psd_peak_rel_bandwidth < 0.0:
        raise ValueError("loss.rollout_disp_psd_peak_rel_bandwidth must be finite and non-negative.")
    rollout_loss_active = (
        (rollout_det_weight > 0.0)
        or (rollout_disp_std_weight > 0.0)
        or (rollout_disp_spectral_weight > 0.0)
    )
    if rollout_loss_active and rollout_det_steps < 1 and rollout_det_steps_final < 1:
        raise ValueError(
            "loss.rollout_det_steps or loss.rollout_det_steps_final must be >= 1 when "
            "a rollout loss weight is active."
        )

    model_dict = asdict(model_cfg)
    model_dict["coefficient_output_bound"] = getattr(constraints_cfg, "coefficient_output_bound", None)
    first_train_traj = train_trajs[0]
    model_dict["structural_mass"] = float(np.asarray(first_train_traj["dry_mass_kg"]).reshape(()))
    model_dict["k"] = float(np.asarray(first_train_traj["stiffness_n_m"]).reshape(()))
    model_dict["damping_c"] = float(np.asarray(first_train_traj["damping_c"]).reshape(()))
    model_dict["Ca"] = 0.0
    model_dict["use_stochastic_process_noise"] = predict_sigma
    model_dict["use_td_force_input"] = use_td_force_input
    model_dict["use_td_fhat_input"] = use_td_fhat_input
    model_dict["use_acceleration_input"] = use_acceleration_input
    model_dict["use_phi_input"] = use_phi_input
    model_dict["phi_input_source"] = None if not use_phi_input else phase_input_source
    model_dict["use_sigma_inputs"] = use_sigma_inputs
    model_dict["correction_mode"] = correction_mode
    model_dict["input_configs"] = td_input_configs
    model, _derived = PHVIV.from_config(dt=dt, cfg=model_dict, arch_cfg=arch_dict, device=device)
    setattr(model, "correction_mode", correction_mode)
    setattr(model, "td_force_input_source", td_force_input_source)
    setattr(model, "fhat_bound_multiplier", float(fhat_bound_multiplier))
    setattr(model, "fhat_correction_bounds", fhat_correction_bounds)
    setattr(model, "force_zero_output", force_zero_output)

    def _checkpoint_config(*, rollout_steps: int | None = None) -> dict[str, Any]:
        checkpoint_config = asdict(config)
        checkpoint_config["model"].update(
            {
                "structural_mass": float(model_dict["structural_mass"]),
                "Ca": float(model_dict["Ca"]),
                "k": float(model_dict["k"]),
                "damping_c": float(model_dict["damping_c"]),
            }
        )
        if rollout_steps is not None:
            checkpoint_config["loss"]["rollout_det_steps"] = int(rollout_steps)
        return checkpoint_config

    _apply_td_correction_head_init(
        model,
        mode=corr_init_mode,
        tiny_std=corr_init_tiny_std,
        predict_sigma=predict_sigma,
        predict_fhat=fhat_active,
    )
    model = maybe_compile_model(model, bool(compile_cfg.use_compile), str(compile_cfg.compile_mode))

    current_rollout_det_steps = _scheduled_rollout_det_steps(
        epoch=0,
        base_steps=rollout_det_steps,
        final_steps=rollout_det_steps_final,
        warmup_epochs=rollout_det_steps_warmup_epochs,
    )

    train_loader, val_loader, train_rollout_loader, val_rollout_loader = _build_td_correction_hnn_loaders(
        train_trajs=train_trajs,
        val_trajs=val_trajs,
        input_scaling_mode=str(getattr(model, "input_scaling_mode", "current")),
        diameter=float(model.D),
        batch_size=int(training_cfg.batch_size),
        rollout_batch_size=rollout_batch_size,
        rollout_steps=current_rollout_det_steps,
        num_workers=int(runtime_cfg.num_workers),
        pin_memory=(device.type == "cuda"),
    )
    _seen_train_loader_unused = None
    val_seen_loader = None
    val_seen_rollout_loader = None
    if val_seen_trajs:
        (
            _seen_train_loader_unused,
            val_seen_loader,
            _seen_train_rollout_loader_unused,
            val_seen_rollout_loader,
        ) = _build_td_correction_hnn_loaders(
            train_trajs=train_trajs,
            val_trajs=val_seen_trajs,
            input_scaling_mode=str(getattr(model, "input_scaling_mode", "current")),
            diameter=float(model.D),
            batch_size=int(training_cfg.batch_size),
            rollout_batch_size=rollout_batch_size,
            rollout_steps=current_rollout_det_steps,
            num_workers=int(runtime_cfg.num_workers),
            pin_memory=(device.type == "cuda"),
        )
        del _seen_train_rollout_loader_unused
        del _seen_train_loader_unused

    state_loss_active = state_loss_weight > 0.0
    gradnorm_balancer: Optional[GradNormBalancer] = None
    if bool(getattr(loss_cfg, "use_gradnorm", False)):
        gradnorm_loss_names: list[str] = []
        if state_loss_active:
            gradnorm_loss_names.append("state")
        if rollout_loss_active:
            gradnorm_loss_names.append("rollout")
        if len(gradnorm_loss_names) >= 2:
            gradnorm_balancer = GradNormBalancer(
                model,
                gradnorm_loss_names,
                alpha=float(getattr(loss_cfg, "gradnorm_alpha", 0.9)),
                eps=float(getattr(loss_cfg, "gradnorm_eps", 1e-8)),
                min_weight=float(getattr(loss_cfg, "gradnorm_min_weight", 0.1)),
                max_weight=float(getattr(loss_cfg, "gradnorm_max_weight", 10.0)),
            )

    opt, lr_scheduler = setup_optimizer_and_scheduler(
        model,
        optim_cfg=optim_cfg,
        scheduler_cfg=optim_cfg.scheduler,
        epochs=int(training_cfg.epochs),
    )
    amp_enabled, amp_dtype, scaler = setup_amp(
        device, use_amp=bool(precision_cfg.use_amp), amp_dtype=str(precision_cfg.amp_dtype)
    )
    async_validation = bool(getattr(monitoring_cfg, "async_validation", False))
    async_device = str(getattr(monitoring_cfg, "async_validation_device", "cpu"))
    async_num_workers = int(getattr(monitoring_cfg, "async_validation_num_workers", 0))
    async_num_threads = int(getattr(monitoring_cfg, "async_validation_num_threads", 4))
    async_max_concurrent = int(getattr(monitoring_cfg, "async_validation_max_concurrent", 1))
    if async_validation:
        surrogate_path = Path(data_cfg.train_series_dir) / "surrogate_validation_points.npz"
        if not surrogate_path.exists():
            raise FileNotFoundError(
                "Surrogate validation NPZ does not exist under data.train_series_dir: "
                f"{surrogate_path}. Generate/copy this NPZ before starting async surrogate validation."
            )
    writer, run_name = setup_writer(
        config.logging.run_dir_root,
        config_name,
        run_name_override=getattr(config.logging, "run_name", None),
        append_timestamp=bool(getattr(config.logging, "append_timestamp", True)),
    )
    writer.add_text("hnn/td_correction_config", json.dumps(hnn_cfg, indent=2, sort_keys=True), 0)
    writer.flush()
    async_processes: list[dict[str, Any]] = []
    async_best_state: dict[str, Any] = {
        "best_metric_name": AGGREGATE_VALIDATION_ERROR_KEY,
        "best_metric_value": float("inf"),
        "loss_total": float("inf"),
    }
    if async_validation:
        (Path(writer.log_dir) / "async_validation").mkdir(parents=True, exist_ok=True)
    run_models_dir = Path("models") / run_name
    run_models_dir.mkdir(parents=True, exist_ok=True)
    validation_models_dir = run_models_dir / "async_validation"
    validation_models_dir.mkdir(parents=True, exist_ok=True)

    def _save_td_validation_checkpoint(epoch_idx: int) -> Path:
        ckpt_path = validation_models_dir / f"model_epoch_{epoch_idx + 1:06d}.pt"
        latest_path = validation_models_dir / "model.pt"
        state_source: torch.nn.Module = model
        if hasattr(model, "_orig_mod"):
            state_source = getattr(model, "_orig_mod")
        checkpoint_config = _checkpoint_config(rollout_steps=current_rollout_det_steps)
        torch.save(
            {
                "model_state": state_source.state_dict(),
                "config": checkpoint_config,
                "run_name": run_name,
                "dt": dt,
                "method": "correction",
                "td_correction": True,
                "correction_mode": correction_mode,
                "predict_sigma": predict_sigma,
                "mean_active": mean_active,
                "fhat_active": fhat_active,
                "use_td_force_input": use_td_force_input,
                "td_force_input_source": td_force_input_source,
                "use_td_fhat_input": use_td_fhat_input,
                "use_phi_input": use_phi_input,
                "phi_input_source": (None if not use_phi_input else phase_input_source),
                "use_sigma_inputs": use_sigma_inputs,
                "shared_td_correction_trunk": shared_td_correction_trunk_cfg,
                "fhat_bound_multiplier": float(fhat_bound_multiplier),
                "fhat_correction_bounds": fhat_correction_bounds,
                "fhat_reg": float(fhat_reg),
                "fhat_reg_norm": str(fhat_reg_norm),
            },
            ckpt_path,
        )
        shutil.copyfile(ckpt_path, latest_path)
        return ckpt_path

    def _parse_td_train_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(batch) != 10:
            raise ValueError("Unexpected TD correction HNN batch format.")
        z_i, t_i, z_next, t_next, ur_i, force_true_next, td_context_i, mass_i, damping_i, stiffness_i = batch
        return z_i, t_i, z_next, t_next, ur_i, force_true_next, td_context_i, mass_i, damping_i, stiffness_i

    def _state_loss(
        *,
        z_i: torch.Tensor,
        dt_i: torch.Tensor,
        z_next: torch.Tensor,
        ur_i: torch.Tensor,
        td_context_i: torch.Tensor,
        mass_i: torch.Tensor,
        damping_i: torch.Tensor,
        stiffness_i: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        step = _td_step_with_corrections(
            model=model,
            z=z_i,
            reduced_velocity=ur_i,
            td_context=td_context_i,
            dt=dt_i,
            structural_mass=mass_i,
            damping_c=damping_i,
            stiffness=stiffness_i,
            td_params=td_params,
            td_memory_cfg=td_memory_cfg,
            mean_active=mean_active,
            sigma_active=predict_sigma,
            fhat_active=fhat_active,
            td_force_input_source=td_force_input_source,
            fhat_bound_multiplier=fhat_bound_multiplier,
            force_zero_output=force_zero_output,
        )
        if predict_sigma and state_loss_mode == "propagated_nll":
            state_loss, _z_next_mean = _td_state_propagated_nll_loss(
                z_i=z_i,
                dt_i=dt_i,
                z_next=z_next,
                total_force_next=step["total_force_next"],
                sigma_corr=step["sigma_corr"],
                mass_i=mass_i,
                damping_i=damping_i,
                stiffness_i=stiffness_i,
            )
        else:
            state_loss, _z_next_mean = _td_state_mse_loss(
                z_i=z_i,
                dt_i=dt_i,
                z_next=z_next,
                total_force_next=step["total_force_next"],
                mass_i=mass_i,
                damping_i=damping_i,
                stiffness_i=stiffness_i,
            )
        step["z_next_mean"] = _z_next_mean
        return state_loss, step

    def _regularizer(value: torch.Tensor, norm: str) -> torch.Tensor:
        key = str(norm).strip().lower()
        if key == "l1":
            return torch.mean(torch.abs(value))
        if key == "l2":
            return torch.mean(value * value)
        raise ValueError("Regularizer norm must be one of: l1, l2.")

    epochs = int(training_cfg.epochs)
    validate_every = max(1, int(getattr(monitoring_cfg, "validate_every_epochs", 1)))
    log_every = max(1, int(getattr(monitoring_cfg, "log_every_epochs", 1)))
    print_every = max(1, int(getattr(monitoring_cfg, "print_every_epochs", 1)))
    train_instances = len(train_loader.dataset)
    train_steps_per_epoch = len(train_loader)
    val_instances = len(val_loader.dataset) if val_loader is not None else 0
    val_steps_per_epoch = len(val_loader) if val_loader is not None else 0
    val_seen_instances = len(val_seen_loader.dataset) if val_seen_loader is not None else 0
    val_seen_steps_per_epoch = len(val_seen_loader) if val_seen_loader is not None else 0
    train_rollout_instances = len(train_rollout_loader.dataset) if train_rollout_loader is not None else 0
    train_rollout_steps_per_epoch = len(train_rollout_loader) if train_rollout_loader is not None else 0
    val_rollout_instances = len(val_rollout_loader.dataset) if val_rollout_loader is not None else 0
    val_rollout_steps_per_epoch = len(val_rollout_loader) if val_rollout_loader is not None else 0

    startup_lines = [
        f"Run name: {run_name}",
        (
            f"HNN TD-correction setup: epochs={epochs}, batch_size={int(training_cfg.batch_size)}, "
            f"steps_per_epoch={train_steps_per_epoch}, train_instances={train_instances}, "
            f"train_trajectories={len(train_trajs)}, correction_mode={correction_mode}"
        ),
        (
            f"Validation setup: seen_steps={val_seen_steps_per_epoch}, seen_instances={val_seen_instances}, "
            f"val_seen_trajectories={len(val_seen_trajs)}, validate_every={validate_every}, "
            f"validation_trajectories=all, async_validation={async_validation}"
        ),
        (
            f"Rollout setup: det_weight={rollout_det_weight:g}, std_weight={rollout_disp_std_weight:g}, "
            f"mean_in_std_loss={rollout_disp_mean_in_std_loss}, "
            f"spectral_weight={rollout_disp_spectral_weight:g}, "
            f"steps={current_rollout_det_steps}, "
            f"train_rollout_windows={train_rollout_instances}, train_rollout_steps={train_rollout_steps_per_epoch}, "
            f"val_rollout_windows={val_rollout_instances}, val_rollout_steps={val_rollout_steps_per_epoch}"
        ),
        (
            f"Runtime: device={device}, num_workers={int(runtime_cfg.num_workers)} "
            f"(td_loader_workers=0), amp={amp_enabled}, "
            f"compile={bool(compile_cfg.use_compile)}, lr={float(optim_cfg.lr):g}"
        ),
    ]
    startup_lines.append(
        f"Loss weights: state={state_loss_weight:g}, rollout_det={rollout_det_weight:g}, "
        f"rollout_disp_std={rollout_disp_std_weight:g}, "
        f"rollout_disp_mean=same_as_std({rollout_disp_mean_in_std_loss}), "
        f"rollout_disp_spectral={rollout_disp_spectral_weight:g}"
    )
    if fhat_correction_bounds is None:
        startup_lines.append("Fhat correction bounds: using Vivana-TD td_fhat_min/td_fhat_max as the base interval.")
    else:
        startup_lines.append(
            "Fhat correction bounds: "
            f"using explicit base interval [{fhat_correction_bounds[0]:g}, {fhat_correction_bounds[1]:g}]."
        )
    if getattr(model, "force_output", "force") == "coefficient" and getattr(model, "coefficient_output_bound", None) is not None:
        startup_lines.append(
            "Coefficient output bound: "
            f"tanh cap at +/-{float(getattr(model, 'coefficient_output_bound')):g}"
        )
    if rollout_loss_active:
        startup_lines.append(
            f"Rollout loss mode: deterministic, state_loss_mode={state_loss_mode}, "
            f"disp_mean_in_std_loss={rollout_disp_mean_in_std_loss}, "
            f"disp_std_p={rollout_disp_std_p:g}, "
            f"disp_freq_p={rollout_disp_freq_p:g}, "
            f"disp_freq_alpha={rollout_disp_freq_alpha:g}, "
            f"disp_spectral_loss={rollout_disp_spectral_loss}, "
            f"disp_psd_peak_rel_bandwidth={rollout_disp_psd_peak_rel_bandwidth:g}, "
            f"disp_psd_use_hann_window={rollout_disp_psd_use_hann_window}"
        )
    print("\n".join(startup_lines))

    def _rebuild_td_rollout_loader(steps: int, *, split_trajs: list[dict[str, np.ndarray]]) -> Any | None:
        if steps <= 0 or not rollout_loss_active:
            return None
        _train_loader_tmp, _val_loader_tmp, _train_rollout_loader_tmp, rollout_loader_tmp = _build_td_correction_hnn_loaders(
            train_trajs=train_trajs,
            val_trajs=split_trajs,
            input_scaling_mode=str(getattr(model, "input_scaling_mode", "current")),
            diameter=float(model.D),
            batch_size=int(training_cfg.batch_size),
            rollout_batch_size=rollout_batch_size,
            rollout_steps=steps,
            num_workers=int(runtime_cfg.num_workers),
            pin_memory=(device.type == "cuda"),
        )
        del _train_loader_tmp
        del _val_loader_tmp
        del _train_rollout_loader_tmp
        return rollout_loader_tmp

    def _run_td_validation_for_split(
        *,
        epoch_idx: int,
        split_tag: str,
        split_name: str,
        split_loader: Any | None,
        split_rollout_loader: Any | None,
        split_trajs: list[dict[str, np.ndarray]],
        log_rollout_plots: bool,
        log_all_rollout_spectra: bool = False,
    ) -> None:
        if split_loader is None:
            return
        split_start = time.perf_counter()
        model.eval()
        val_sums = {
            "loss_total": torch.zeros((), device=device),
            "loss_state": torch.zeros((), device=device),
            "loss_reg_mean": torch.zeros((), device=device),
            "loss_reg_sigma": torch.zeros((), device=device),
            "loss_reg_fhat": torch.zeros((), device=device),
            "loss_rollout_spectral": torch.zeros((), device=device),
        }
        val_count = 0
        with torch.no_grad():
            for batch in split_loader:
                z_i, t_i, z_next, t_next, ur_i, _force_true_next, td_context_i, mass_i, damping_i, stiffness_i = _parse_td_train_batch(batch)
                z_i = z_i.to(device, non_blocking=non_blocking)
                t_i = t_i.to(device, non_blocking=non_blocking)
                z_next = z_next.to(device, non_blocking=non_blocking)
                t_next = t_next.to(device, non_blocking=non_blocking)
                ur_i = ur_i.to(device, non_blocking=non_blocking)
                td_context_i = td_context_i.to(device, non_blocking=non_blocking)
                mass_i = mass_i.to(device, non_blocking=non_blocking)
                damping_i = damping_i.to(device, non_blocking=non_blocking)
                stiffness_i = stiffness_i.to(device, non_blocking=non_blocking)
                with torch.amp.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                    dt_i = torch.clamp(t_next - t_i, min=1.0e-12)
                    state_loss, step = _state_loss(
                        z_i=z_i,
                        dt_i=dt_i,
                        z_next=z_next,
                        ur_i=ur_i,
                        td_context_i=td_context_i,
                        mass_i=mass_i,
                        damping_i=damping_i,
                        stiffness_i=stiffness_i,
                    )
                    corr_mu = step["corr_mu"]
                    raw_corr_mu = step["raw_corr_mu"]
                    sigma_corr = step["sigma_corr"]
                    mean_reg_loss = _regularizer(raw_corr_mu, mean_reg_norm)
                    sigma_reg_loss = _regularizer(sigma_corr, sigma_reg_norm) if predict_sigma else state_loss.new_tensor(0.0)
                    fhat_reg_loss = _regularizer(step["delta_fhat"], fhat_reg_norm) if fhat_active else state_loss.new_tensor(0.0)
                    total_loss = (
                        float(state_loss_weight) * state_loss
                        + float(mean_reg) * mean_reg_loss
                        + float(sigma_reg) * sigma_reg_loss
                        + float(fhat_reg) * fhat_reg_loss
                    )
                val_sums["loss_total"] += total_loss.detach()
                val_sums["loss_state"] += state_loss.detach()
                val_sums["loss_reg_mean"] += mean_reg_loss.detach()
                val_sums["loss_reg_sigma"] += sigma_reg_loss.detach()
                val_sums["loss_reg_fhat"] += fhat_reg_loss.detach()
                val_count += 1
        val_denom = float(max(1, val_count))
        val_metrics = {
            name: float((value / val_denom).detach().cpu()) for name, value in val_sums.items()
        }
        rollout_loss_avg = 0.0
        rollout_std_loss_avg = 0.0
        rollout_mean_loss_avg = 0.0
        rollout_spectral_loss_avg = 0.0
        if split_rollout_loader is not None and rollout_loss_active:
            with torch.no_grad():
                rollout_loss_sum = torch.zeros((), device=device)
                rollout_std_loss_sum = torch.zeros((), device=device)
                rollout_mean_loss_sum = torch.zeros((), device=device)
                rollout_spectral_loss_sum = torch.zeros((), device=device)
                rollout_count = 0
                for rollout_batch in split_rollout_loader:
                    rollout_losses = _td_correction_rollout_losses_from_batch(
                        model=model,
                        batch=rollout_batch,
                        device=device,
                        non_blocking=non_blocking,
                        td_params=td_params,
                        td_memory_cfg=td_memory_cfg,
                        mean_active=mean_active,
                        sigma_active=predict_sigma,
                        fhat_active=fhat_active,
                        td_force_input_source=td_force_input_source,
                        fhat_bound_multiplier=fhat_bound_multiplier,
                        force_zero_output=force_zero_output,
                        compute_disp_std_loss=(rollout_disp_std_weight > 0.0),
                        compute_disp_mean_loss=(
                            rollout_disp_std_weight > 0.0 and rollout_disp_mean_in_std_loss
                        ),
                        disp_std_power=rollout_disp_std_p,
                        compute_disp_spectral_loss=(rollout_disp_spectral_weight > 0.0),
                        disp_spectral_loss_mode=rollout_disp_spectral_loss,
                        disp_freq_power=rollout_disp_freq_p,
                        disp_freq_alpha=rollout_disp_freq_alpha,
                        disp_psd_peak_rel_bandwidth=rollout_disp_psd_peak_rel_bandwidth,
                        disp_psd_use_hann_window=rollout_disp_psd_use_hann_window,
                    )
                    rollout_loss_sum += rollout_losses["trajectory_loss"].detach()
                    rollout_std_loss_sum += rollout_losses["disp_std_loss"].detach()
                    rollout_mean_loss_sum += rollout_losses["disp_mean_loss"].detach()
                    rollout_spectral_loss_sum += rollout_losses["disp_spectral_loss"].detach()
                    rollout_count += 1
                rollout_loss_avg = float((rollout_loss_sum / float(max(1, rollout_count))).detach().cpu())
                rollout_std_loss_avg = float((rollout_std_loss_sum / float(max(1, rollout_count))).detach().cpu())
                rollout_mean_loss_avg = float((rollout_mean_loss_sum / float(max(1, rollout_count))).detach().cpu())
                rollout_spectral_loss_avg = float(
                    (rollout_spectral_loss_sum / float(max(1, rollout_count))).detach().cpu()
                )
                writer.add_scalar(f"{split_tag}/loss_rollout_det", rollout_loss_avg, epoch_idx + 1)
                writer.add_scalar(f"{split_tag}/loss_rollout_disp_std", rollout_std_loss_avg, epoch_idx + 1)
                writer.add_scalar(f"{split_tag}/loss_rollout_disp_mean", rollout_mean_loss_avg, epoch_idx + 1)
                writer.add_scalar(f"{split_tag}/loss_rollout_spectral", rollout_spectral_loss_avg, epoch_idx + 1)
        val_metrics["loss_rollout_disp_std"] = rollout_std_loss_avg
        val_metrics["loss_rollout_disp_mean"] = rollout_mean_loss_avg
        val_metrics["loss_rollout_spectral"] = rollout_spectral_loss_avg
        val_metrics["loss_total"] = (
            float(state_loss_weight) * val_metrics["loss_state"]
            + float(mean_reg) * val_metrics["loss_reg_mean"]
            + float(sigma_reg) * val_metrics["loss_reg_sigma"]
            + float(fhat_reg) * val_metrics["loss_reg_fhat"]
            + float(rollout_det_weight) * rollout_loss_avg
            + float(rollout_disp_std_weight) * (rollout_std_loss_avg + rollout_mean_loss_avg)
            + float(rollout_disp_spectral_weight) * rollout_spectral_loss_avg
        )
        for name, value in val_metrics.items():
            writer.add_scalar(f"{split_tag}/{name}", value, epoch_idx + 1)

        if split_trajs and ((epoch_idx % validate_every) == 0 or epoch_idx == epochs - 1):
            sampled_metric_indices = list(range(len(split_trajs)))
            sampled_names = [str(split_trajs[idx].get("name", f"traj_{idx}")) for idx in sampled_metric_indices]
            print(
                f"[td-{split_name}][phnn] epoch {epoch_idx + 1}: metric trajectories={sampled_names} "
                f"(force_zero_output={force_zero_output}, mass=dry_mass_kg)"
            )
            metrics_sum: dict[str, float] = {}
            metrics_count: dict[str, int] = {}
            diverged_count = 0
            for sidx in sampled_metric_indices:
                metrics_roll = _log_td_correction_rollout_validation(
                    writer=writer,
                    epoch=epoch_idx + 1,
                    model=model,
                    traj=split_trajs[sidx],
                    dt=dt,
                    td_params=td_params,
                    td_memory_cfg=td_memory_cfg,
                    device=device,
                    mean_active=mean_active,
                    predict_sigma=predict_sigma,
                    fhat_active=fhat_active,
                    td_force_input_source=td_force_input_source,
                    fhat_bound_multiplier=fhat_bound_multiplier,
                    force_zero_output=force_zero_output,
                    tag_prefix=f"{split_tag}/rollout",
                    log_metrics=False,
                    log_plots=False,
                )
                diverged_flag = float(metrics_roll.get(ROLLOUT_DIVERGED_KEY, 0.0))
                if np.isfinite(diverged_flag) and diverged_flag > 0.5:
                    diverged_count += 1
                for name, value in metrics_roll.items():
                    if name == ROLLOUT_DIVERGED_KEY or not np.isfinite(float(value)):
                        continue
                    metrics_sum[name] = metrics_sum.get(name, 0.0) + float(value)
                    metrics_count[name] = metrics_count.get(name, 0) + 1
            for name, total in metrics_sum.items():
                writer.add_scalar(f"{split_tag}/{name}", total / float(max(1, metrics_count.get(name, 0))), epoch_idx + 1)
            writer.add_scalar(f"{split_tag}/{ROLLOUT_DIVERGED_COUNT_KEY}", float(diverged_count), epoch_idx + 1)

            if log_rollout_plots:
                selected_indices = sample_one_index_per_ur(ur_all, seed=0)
                if not selected_indices:
                    selected_indices = list(range(len(split_trajs)))
                rollout_idx = selected_indices[0]
                rollout_traj = split_trajs[rollout_idx]
                rollout_dt = float(np.asarray(rollout_traj["t"])[1] - np.asarray(rollout_traj["t"])[0])
                print(
                    f"[td-{split_name}][phnn] epoch {epoch_idx + 1}: plot trajectory={rollout_traj.get('name', f'traj_{rollout_idx}')} "
                    f"U_r={float(np.asarray(rollout_traj['ur']).reshape(-1)[0]):.6g} "
                    f"dt={rollout_dt:.6g} rho={float(model.rho):.6g} D={float(model.D):.6g} "
                    f"m={float(np.asarray(rollout_traj['dry_mass_kg']).reshape(())):.6g} "
                    f"c={float(np.asarray(rollout_traj['damping_c']).reshape(())):.6g} "
                    f"k={float(np.asarray(rollout_traj['stiffness_n_m']).reshape(())):.6g}"
                )
                _log_td_correction_rollout_validation(
                    writer=writer,
                    epoch=epoch_idx + 1,
                    model=model,
                    traj=rollout_traj,
                    dt=dt,
                    td_params=td_params,
                    td_memory_cfg=td_memory_cfg,
                    device=device,
                    mean_active=mean_active,
                    predict_sigma=predict_sigma,
                    fhat_active=fhat_active,
                    td_force_input_source=td_force_input_source,
                    fhat_bound_multiplier=fhat_bound_multiplier,
                    force_zero_output=force_zero_output,
                    tag_prefix=f"{split_tag}/rollout",
                    log_metrics=False,
                    log_plots=True,
                    log_spectra=True,
                )
            if log_all_rollout_spectra:
                def _safe_tag_component(raw: Any) -> str:
                    text = str(raw).strip()
                    if not text:
                        return "unnamed"
                    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in text)
                    return cleaned or "unnamed"

                for traj in split_trajs:
                    traj_name = str(traj.get("name", "unnamed"))
                    _log_td_correction_rollout_validation(
                        writer=writer,
                        epoch=epoch_idx,
                        model=model,
                        traj=traj,
                        dt=dt,
                        td_params=td_params,
                        td_memory_cfg=td_memory_cfg,
                        device=device,
                        mean_active=mean_active,
                        predict_sigma=predict_sigma,
                        fhat_active=fhat_active,
                        td_force_input_source=td_force_input_source,
                        fhat_bound_multiplier=fhat_bound_multiplier,
                        force_zero_output=force_zero_output,
                        tag_prefix=f"{split_tag}/rollout/{_safe_tag_component(traj_name)}",
                        log_metrics=False,
                        log_plots=True,
                        log_spectra=True,
                        log_only_spectra=True,
                        step=epoch_idx + 1,
                        title_suffix=f" [{traj_name}]",
                    )
        elapsed = float(time.perf_counter() - split_start)
        writer.add_scalar(f"{split_tag}/validation_wall_time_s", elapsed, epoch_idx + 1)
        writer.flush()

    for epoch in range(epochs):
        scheduled_rollout_det_steps = _scheduled_rollout_det_steps(
            epoch=epoch,
            base_steps=rollout_det_steps,
            final_steps=rollout_det_steps_final,
            warmup_epochs=rollout_det_steps_warmup_epochs,
        )
        if scheduled_rollout_det_steps != current_rollout_det_steps:
            current_rollout_det_steps = scheduled_rollout_det_steps
            train_rollout_loader = _rebuild_td_rollout_loader(current_rollout_det_steps, split_trajs=train_trajs)
            val_rollout_loader = _rebuild_td_rollout_loader(current_rollout_det_steps, split_trajs=val_trajs)
            if val_seen_trajs:
                val_seen_rollout_loader = _rebuild_td_rollout_loader(current_rollout_det_steps, split_trajs=val_seen_trajs)
        model.train()
        if bool(optim_cfg.use_lr_scheduler):
            for group in opt.param_groups:
                group["lr"] = lr_scheduler.get_lr(epoch)
        sums = {
            "loss_total": torch.zeros((), device=device),
            "loss_state": torch.zeros((), device=device),
            "loss_reg_mean": torch.zeros((), device=device),
            "loss_reg_sigma": torch.zeros((), device=device),
            "loss_reg_fhat": torch.zeros((), device=device),
            "loss_rollout_det": torch.zeros((), device=device),
            "loss_rollout_disp_std": torch.zeros((), device=device),
            "loss_rollout_disp_mean": torch.zeros((), device=device),
            "loss_rollout_spectral": torch.zeros((), device=device),
            "grad_norm": torch.zeros((), device=device),
        }
        gradnorm_state_w_sum = (
            torch.zeros((), device=device) if gradnorm_balancer is not None and state_loss_active else None
        )
        gradnorm_rollout_w_sum = (
            torch.zeros((), device=device) if gradnorm_balancer is not None and rollout_loss_active else None
        )
        gradnorm_count = 0
        batch_count = 0
        rollout_iter = iter(train_rollout_loader) if train_rollout_loader is not None and rollout_loss_active else None
        for batch in train_loader:
            z_i, t_i, z_next, t_next, ur_i, _force_true_next, td_context_i, mass_i, damping_i, stiffness_i = _parse_td_train_batch(batch)
            z_i = z_i.to(device, non_blocking=non_blocking)
            t_i = t_i.to(device, non_blocking=non_blocking)
            z_next = z_next.to(device, non_blocking=non_blocking)
            t_next = t_next.to(device, non_blocking=non_blocking)
            ur_i = ur_i.to(device, non_blocking=non_blocking)
            td_context_i = td_context_i.to(device, non_blocking=non_blocking)
            mass_i = mass_i.to(device, non_blocking=non_blocking)
            damping_i = damping_i.to(device, non_blocking=non_blocking)
            stiffness_i = stiffness_i.to(device, non_blocking=non_blocking)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                dt_i = torch.clamp(t_next - t_i, min=1.0e-12)
                state_loss, step = _state_loss(
                    z_i=z_i,
                    dt_i=dt_i,
                    z_next=z_next,
                    ur_i=ur_i,
                    td_context_i=td_context_i,
                    mass_i=mass_i,
                    damping_i=damping_i,
                    stiffness_i=stiffness_i,
                )
                corr_mu = step["corr_mu"]
                raw_corr_mu = step["raw_corr_mu"]
                sigma_corr = step["sigma_corr"]
                mean_reg_loss = _regularizer(raw_corr_mu, mean_reg_norm)
                sigma_reg_loss = _regularizer(sigma_corr, sigma_reg_norm) if predict_sigma else state_loss.new_tensor(0.0)
                fhat_reg_loss = _regularizer(step["delta_fhat"], fhat_reg_norm) if fhat_active else state_loss.new_tensor(0.0)
                rollout_det_loss = state_loss.new_tensor(0.0)
                rollout_std_loss = state_loss.new_tensor(0.0)
                rollout_mean_loss = state_loss.new_tensor(0.0)
                rollout_spectral_loss = state_loss.new_tensor(0.0)
                if rollout_iter is not None:
                    try:
                        rollout_batch = next(rollout_iter)
                    except StopIteration:
                        rollout_iter = iter(train_rollout_loader)
                        rollout_batch = next(rollout_iter)
                    rollout_losses = _td_correction_rollout_losses_from_batch(
                        model=model,
                        batch=rollout_batch,
                        device=device,
                        non_blocking=non_blocking,
                        td_params=td_params,
                        td_memory_cfg=td_memory_cfg,
                        mean_active=mean_active,
                        sigma_active=predict_sigma,
                        fhat_active=fhat_active,
                        td_force_input_source=td_force_input_source,
                        fhat_bound_multiplier=fhat_bound_multiplier,
                        force_zero_output=force_zero_output,
                        compute_disp_std_loss=(rollout_disp_std_weight > 0.0),
                        compute_disp_mean_loss=(
                            rollout_disp_std_weight > 0.0 and rollout_disp_mean_in_std_loss
                        ),
                        disp_std_power=rollout_disp_std_p,
                        compute_disp_spectral_loss=(rollout_disp_spectral_weight > 0.0),
                        disp_spectral_loss_mode=rollout_disp_spectral_loss,
                        disp_freq_power=rollout_disp_freq_p,
                        disp_freq_alpha=rollout_disp_freq_alpha,
                        disp_psd_peak_rel_bandwidth=rollout_disp_psd_peak_rel_bandwidth,
                        disp_psd_use_hann_window=rollout_disp_psd_use_hann_window,
                    )
                    rollout_det_loss = rollout_losses["trajectory_loss"]
                    rollout_std_loss = rollout_losses["disp_std_loss"]
                    rollout_mean_loss = rollout_losses["disp_mean_loss"]
                    rollout_spectral_loss = rollout_losses["disp_spectral_loss"]
                rollout_total_loss = (
                    float(rollout_det_weight) * rollout_det_loss
                    + float(rollout_disp_std_weight) * (rollout_std_loss + rollout_mean_loss)
                    + float(rollout_disp_spectral_weight) * rollout_spectral_loss
                )
                if gradnorm_balancer is not None:
                    loss_inputs: dict[str, torch.Tensor] = {}
                    if state_loss_active:
                        loss_inputs["state"] = state_loss.float()
                    if rollout_loss_active:
                        loss_inputs["rollout"] = rollout_total_loss.float()
                    weights = gradnorm_balancer.update(loss_inputs)
                    base_loss = state_loss.new_tensor(0.0)
                    if state_loss_active:
                        state_w = weights["state"]
                        base_loss = base_loss + float(state_loss_weight) * state_w * state_loss
                        if gradnorm_state_w_sum is not None:
                            gradnorm_state_w_sum = gradnorm_state_w_sum + state_w.detach()
                    if rollout_loss_active:
                        rollout_w = weights["rollout"]
                        if gradnorm_rollout_w_sum is not None:
                            gradnorm_rollout_w_sum = gradnorm_rollout_w_sum + rollout_w.detach()
                        weighted_rollout_total = rollout_w * rollout_total_loss
                    else:
                        weighted_rollout_total = rollout_total_loss
                    gradnorm_count += 1
                else:
                    base_loss = float(state_loss_weight) * state_loss
                    weighted_rollout_total = rollout_total_loss
                total_loss = (
                    base_loss
                    + float(mean_reg) * mean_reg_loss
                    + float(sigma_reg) * sigma_reg_loss
                    + float(fhat_reg) * fhat_reg_loss
                    + weighted_rollout_total
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
            batch_count += 1
            sums["loss_total"] += total_loss.detach()
            sums["loss_state"] += state_loss.detach()
            sums["loss_reg_mean"] += mean_reg_loss.detach()
            sums["loss_reg_sigma"] += sigma_reg_loss.detach()
            sums["loss_reg_fhat"] += fhat_reg_loss.detach()
            sums["loss_rollout_det"] += rollout_det_loss.detach()
            sums["loss_rollout_disp_std"] += rollout_std_loss.detach()
            sums["loss_rollout_disp_mean"] += rollout_mean_loss.detach()
            sums["loss_rollout_spectral"] += rollout_spectral_loss.detach()
            sums["grad_norm"] += grad_norm.detach() if isinstance(grad_norm, torch.Tensor) else torch.tensor(float(grad_norm), device=device)

        denom = float(max(1, batch_count))
        train_metrics = {name: float((value / denom).detach().cpu()) for name, value in sums.items()}
        train_metrics["lr"] = float(opt.param_groups[0]["lr"]) if opt.param_groups else float(optim_cfg.lr)
        if gradnorm_count > 0:
            if gradnorm_state_w_sum is not None:
                train_metrics["gradnorm_weight_physics"] = float(
                    (gradnorm_state_w_sum / float(gradnorm_count)).detach().cpu()
                )
            if gradnorm_rollout_w_sum is not None:
                train_metrics["gradnorm_weight_rollout"] = float(
                    (gradnorm_rollout_w_sum / float(gradnorm_count)).detach().cpu()
                )
        if epoch == 0 or (epoch + 1) % log_every == 0 or epoch == epochs - 1:
            for name, value in train_metrics.items():
                writer.add_scalar(f"train/{name}", value, epoch + 1)
            writer.flush()
        if epoch % print_every == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch}: loss={train_metrics['loss_total']:.4e}, "
                f"Lstate={train_metrics['loss_state']:.4e}, "
                f"Lroll={train_metrics['loss_rollout_det']:.4e}, "
                f"Lstd={train_metrics['loss_rollout_disp_std']:.4e}, "
                f"Lmean={train_metrics['loss_rollout_disp_mean']:.4e}, "
                f"Lspec={train_metrics['loss_rollout_spectral']:.4e}, lr={train_metrics['lr']:.3e}"
            )

        should_validate = (epoch % validate_every) == 0 or epoch == epochs - 1
        if should_validate:
            if async_validation:
                async_processes = _reap_async_processes(
                    async_processes,
                    writer=writer,
                    best_state=async_best_state,
                    wait=False,
                )
                ckpt_path = _save_td_validation_checkpoint(epoch)
                print(f"Saved async validation checkpoint to {ckpt_path}")
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
                    do_losses=True,
                    do_rollout=True,
                    best_state=async_best_state,
                )
            elif val_seen_loader is not None:
                _run_td_validation_for_split(
                    epoch_idx=epoch,
                    split_tag="val_seen",
                    split_name="val_seen",
                    split_loader=val_seen_loader,
                    split_rollout_loader=val_seen_rollout_loader,
                    split_trajs=val_seen_trajs,
                    log_rollout_plots=True,
                    log_all_rollout_spectra=True,
                )
                ckpt_path = _save_td_validation_checkpoint(epoch)
                print(f"Saved validation checkpoint to {ckpt_path}")

    if async_validation and async_processes:
        print(f"Waiting for {len(async_processes)} async validation job(s) to finish...")
        async_processes = _reap_async_processes(
            async_processes,
            writer=writer,
            best_state=async_best_state,
            wait=True,
        )

    final_val_trajs = [*val_seen_trajs]
    if final_val_trajs:
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
        metric_trajs = list(final_val_trajs)
        plot_trajs = list(final_val_trajs)
        plot_trajs.sort(key=lambda traj: round(float(np.asarray(traj["ur"]).reshape(-1)[0]), 6))
        for traj in metric_trajs:
            metrics = _log_td_correction_rollout_validation(
                writer=writer,
                epoch=max(0, epochs - 1),
                model=model,
                traj=traj,
                dt=dt,
                td_params=td_params,
                td_memory_cfg=td_memory_cfg,
                device=device,
                mean_active=mean_active,
                predict_sigma=predict_sigma,
                fhat_active=fhat_active,
                td_force_input_source=td_force_input_source,
                fhat_bound_multiplier=fhat_bound_multiplier,
                force_zero_output=force_zero_output,
                tag_prefix="final_val/rollout",
                log_metrics=False,
                log_plots=False,
                log_correction_on_data=False,
                log_phase_map=False,
            )
            filtered = {
                name: float(value)
                for name, value in metrics.items()
                if name != ROLLOUT_DIVERGED_KEY and np.isfinite(float(value))
            }
            if filtered:
                ur_values.append(float(np.asarray(traj["ur"]).reshape(-1)[0]))
                metrics_list.append(filtered)
            for name, value in filtered.items():
                metrics_sum[name] = metrics_sum.get(name, 0.0) + float(value)
                metrics_count[name] = metrics_count.get(name, 0) + 1
        for idx, traj in enumerate(plot_trajs, start=1):
            ur_val = float(np.asarray(traj["ur"]).reshape(-1)[0])
            mass_key = "dry_mass_kg"
            mass_value = float(np.asarray(traj[mass_key]).reshape(()))
            stiffness_value = float(np.asarray(traj["stiffness_n_m"]).reshape(()))
            y_true_t = torch.from_numpy(np.ascontiguousarray(traj["y"])).float().unsqueeze(1).to(device)
            v_true_t = torch.from_numpy(np.ascontiguousarray(traj["dy"])).float().unsqueeze(1).to(device)
            z_true_t = torch.cat([y_true_t, v_true_t * mass_value], dim=1)
            t_traj = torch.from_numpy(np.ascontiguousarray(traj["t"])).float().to(device)
            dt_traj = torch.clamp((t_traj[1:] - t_traj[:-1]).unsqueeze(1), min=1.0e-12)
            ur_t = torch.from_numpy(
                np.ascontiguousarray(
                    _td_flow_feature_from_traj(
                        traj,
                        input_scaling_mode=str(getattr(model, "input_scaling_mode", "current")),
                        diameter=float(model.D),
                    )
                )
            ).float().unsqueeze(1).to(device)
            with torch.no_grad():
                step_on_data = _td_step_with_corrections(
                    model=model,
                    z=z_true_t[:-1],
                    reduced_velocity=ur_t[:-1],
                    td_context=torch.from_numpy(np.ascontiguousarray(traj["td_context"][:-1])).float().to(device),
                    dt=dt_traj,
                    structural_mass=torch.full((z_true_t.shape[0] - 1, 1), mass_value, dtype=z_true_t.dtype, device=device),
                    damping_c=torch.full((z_true_t.shape[0] - 1, 1), float(np.asarray(traj["damping_c"]).reshape(())), dtype=z_true_t.dtype, device=device),
                    stiffness=torch.full((z_true_t.shape[0] - 1, 1), stiffness_value, dtype=z_true_t.dtype, device=device),
                    td_params=td_params,
                    td_memory_cfg=td_memory_cfg,
                    mean_active=mean_active,
                    sigma_active=predict_sigma,
                    fhat_active=fhat_active,
                    td_force_input_source=td_force_input_source,
                    fhat_bound_multiplier=fhat_bound_multiplier,
                    force_zero_output=force_zero_output,
                )
                corr_on_data = step_on_data["corr_mu"]
                sigma_on_data = step_on_data["sigma_corr"]
                delta_fhat_on_data = step_on_data["delta_fhat"]
            output_ur_values.append(ur_val)
            corr_series_list.append(corr_on_data[:, 0].detach().cpu().numpy())
            if predict_sigma:
                sigma_series_list.append(sigma_on_data[:, 0].detach().cpu().numpy())
            if fhat_active:
                delta_fhat_series_list.append(delta_fhat_on_data[:, 0].detach().cpu().numpy())

            plot_metrics = _log_td_correction_rollout_validation(
                writer=writer,
                epoch=max(0, epochs - 1),
                model=model,
                traj=traj,
                dt=dt,
                td_params=td_params,
                td_memory_cfg=td_memory_cfg,
                device=device,
                mean_active=mean_active,
                predict_sigma=predict_sigma,
                fhat_active=fhat_active,
                td_force_input_source=td_force_input_source,
                fhat_bound_multiplier=fhat_bound_multiplier,
                force_zero_output=force_zero_output,
                tag_prefix="final_val/rollout",
                step=idx,
                log_metrics=False,
                log_plots=True,
                log_correction_on_data=False,
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
            summary_lines = [
                "Final rollout over all validation trajectories:",
                f"val_seen={len(val_seen_trajs)}, total={len(metric_trajs)}",
            ]
            for name in sorted(avg_metrics):
                summary_lines.append(f"{name}: {avg_metrics[name]:.6f}")
                writer.add_scalar(f"final_val/avg/{name}", avg_metrics[name], epochs)
            writer.add_text("final_val/summary", "\n".join(summary_lines), epochs)
        if plot_ur_values and plot_metrics_list:
            reference_ur_values = [
                float(np.asarray(traj["ur"]).reshape(-1)[0])
                for traj in final_val_trajs
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
            force_mode = str(getattr(model, "force_output", "force"))
            log_output_distribution_vs_ur(
                writer,
                epochs,
                ur_values=output_ur_values,
                mean_series=corr_series_list,
                sigma_series=(sigma_series_list if predict_sigma else None),
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

    model_path = run_models_dir / "final.pt"
    state_source: torch.nn.Module = model
    if hasattr(model, "_orig_mod"):
        state_source = getattr(model, "_orig_mod")
    torch.save(
        {
            "model_state": state_source.state_dict(),
            "config": _checkpoint_config(),
            "run_name": run_name,
            "dt": dt,
            "method": "correction",
            "td_correction": True,
            "correction_mode": correction_mode,
            "predict_sigma": predict_sigma,
            "mean_active": mean_active,
            "fhat_active": fhat_active,
            "use_td_force_input": use_td_force_input,
            "td_force_input_source": td_force_input_source,
            "use_td_fhat_input": use_td_fhat_input,
            "use_acceleration_input": use_acceleration_input,
            "use_phi_input": use_phi_input,
            "phi_input_source": (None if not use_phi_input else phase_input_source),
            "use_sigma_inputs": use_sigma_inputs,
            "shared_td_correction_trunk": shared_td_correction_trunk_cfg,
            "fhat_bound_multiplier": float(fhat_bound_multiplier),
            "fhat_correction_bounds": fhat_correction_bounds,
            "fhat_reg": float(fhat_reg),
            "fhat_reg_norm": str(fhat_reg_norm),
        },
        model_path,
    )
    print(f"Saved final model to {model_path}")
    writer.flush()
    writer.close()


def train(config: Config, config_name: str) -> None:
    hnn_cfg = dict(config.correction or {})
    if "use_td_correction" in hnn_cfg and not bool(hnn_cfg.get("use_td_correction", True)):
        raise ValueError("Correction training only supports TD-correction mode. Remove correction.use_td_correction or set it to true.")
    _train_td_correction(config, config_name)
    return
