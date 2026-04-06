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
from HNN_helper import (
    Config,
    FORCE_MAPPING_NRMSE_KEY,
    GradNormBalancer,
    PHVIV,
    ROLLOUT_DIVERGED_COUNT_KEY,
    ROLLOUT_DIVERGED_KEY,
    build_ur_bin_state_scale_info_from_dataset,
    build_phase_plot_grid,
    build_dataloader_from_series,
    build_rollout_dataloader_from_series,
    compute_validation_metrics,
    compute_model_grad_norm,
    load_training_series,
    lookup_ur_bin_state_scale_tensor,
    format_loss_vs_ur_text,
    create_window_mask,
    create_zoom_mask,
    log_loss_vs_ur,
    log_displacement_plots,
    log_final_rollout_errors_vs_ur,
    log_force_plots,
    log_correction_on_data_plot,
    log_output_distribution_vs_ur,
    log_phase_component_plots,
    log_signed_phase_output_plot,
    log_training_metrics,
    log_validation_epoch,
    nearest_phase_series_values,
    preprocess_timeseries,
    load_td_correction_trajectories,
    resolve_middle_time_plot,
    resolve_td_correction_params,
    resolve_td_memory_config,
    resolve_td_n_memory_torch,
    rollout_model,
    structural_step_constant_force_torch,
    scaled_residual_loss_per_sample,
    td_baseline_step_torch,
    resolve_cut_start_seconds,
    sample_indices_per_ur,
    sample_one_index_per_ur,
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


def _resolve_td_correction_init_settings(method_cfg: dict[str, Any], model_cfg: Any) -> tuple[str, float]:
    mode = str(method_cfg.get("corr_init_mode", "standard")).strip().lower()
    if mode not in {"zero", "tiny", "standard"}:
        raise ValueError("corr_init_mode must be one of: zero, tiny, standard.")
    tiny_std = float(method_cfg.get("corr_init_tiny_std", 1.0e-4))
    if not np.isfinite(tiny_std) or tiny_std <= 0.0:
        raise ValueError("corr_init_tiny_std must be finite and > 0.")
    return mode, tiny_std


def _apply_td_correction_head_init(
    model: PHVIV,
    *,
    mode: str,
    tiny_std: float,
    predict_sigma: bool,
) -> None:
    _init_mean_head(model.u_base_net, mode=mode, tiny_std=tiny_std)
    if predict_sigma:
        sigma_min = float(model.sigma_min.detach().cpu())
        _init_sigma_head(model.sigma_net, mode=mode, tiny_std=tiny_std, sigma_min=sigma_min)


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


def _td_output_scale_tensor(
    model: PHVIV,
    *,
    reduced_velocity: torch.Tensor,
    structural_mass: torch.Tensor | float,
    stiffness: torch.Tensor | float,
    like: torch.Tensor,
) -> torch.Tensor:
    mass_t = torch.as_tensor(structural_mass, device=like.device, dtype=like.dtype)
    stiffness_t = torch.as_tensor(stiffness, device=like.device, dtype=like.dtype)
    if mass_t.ndim == 0:
        mass_t = mass_t.view(1, 1)
    elif mass_t.ndim == 1:
        mass_t = mass_t.view(-1, 1)
    if stiffness_t.ndim == 0:
        stiffness_t = stiffness_t.view(1, 1)
    elif stiffness_t.ndim == 1:
        stiffness_t = stiffness_t.view(-1, 1)
    shape = like.shape[:-1] + (1,)
    mass_t = mass_t.expand(shape)
    stiffness_t = stiffness_t.expand(shape)
    if getattr(model, "force_output", "force") == "coefficient":
        rv_raw = model._prepare_reduced_velocity_raw(reduced_velocity, like=like)
        if rv_raw is None:
            raise ValueError("reduced_velocity is required for PHNN coefficient-force scaling.")
        omega_n = torch.sqrt(torch.clamp(stiffness_t / mass_t, min=1e-12))
        f_n = omega_n / (2.0 * np.pi)
        u_flow = rv_raw * f_n * float(model.D)
        f0 = 0.5 * float(model.rho) * float(model.D) * (u_flow**2)
        return torch.clamp(f0, min=1e-12)
    return stiffness_t * float(model.D)


def _td_p_scale_tensor(
    model: PHVIV,
    *,
    structural_mass: torch.Tensor | float,
    stiffness: torch.Tensor | float,
    like: torch.Tensor,
) -> torch.Tensor:
    mass_t = torch.as_tensor(structural_mass, device=like.device, dtype=like.dtype)
    stiffness_t = torch.as_tensor(stiffness, device=like.device, dtype=like.dtype)
    if mass_t.ndim == 0:
        mass_t = mass_t.view(1, 1)
    elif mass_t.ndim == 1:
        mass_t = mass_t.view(-1, 1)
    if stiffness_t.ndim == 0:
        stiffness_t = stiffness_t.view(1, 1)
    elif stiffness_t.ndim == 1:
        stiffness_t = stiffness_t.view(-1, 1)
    shape = like.shape[:-1] + (1,)
    mass_t = mass_t.expand(shape)
    stiffness_t = stiffness_t.expand(shape)
    return torch.sqrt(torch.clamp(mass_t * stiffness_t, min=1e-12)) * float(model.D)


def _td_state_for_model_scaling(
    model: PHVIV,
    *,
    z: torch.Tensor,
    structural_mass: torch.Tensor | float,
    stiffness: torch.Tensor | float,
) -> torch.Tensor:
    p_scale_actual = _td_p_scale_tensor(
        model,
        structural_mass=structural_mass,
        stiffness=stiffness,
        like=z[..., :1],
    )
    p_scale_model = torch.as_tensor(float(model.nn_p_scale), device=z.device, dtype=z.dtype)
    p_model = z[..., 1:2] * (p_scale_model / p_scale_actual)
    return torch.cat([z[..., 0:1], p_model], dim=-1)


def _td_predict_correction(
    model: PHVIV,
    *,
    z: torch.Tensor,
    reduced_velocity: torch.Tensor,
    td_force_input: torch.Tensor | None,
    structural_mass: torch.Tensor | float,
    stiffness: torch.Tensor | float,
    predict_sigma: bool,
    force_zero_output: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    z_model = _td_state_for_model_scaling(
        model,
        z=z,
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
    if force_zero_output:
        raw_force = z_model[..., :1].new_zeros(z_model.shape[:-1] + (1,))
    else:
        raw_force = model._force_net_raw(
            z_model,
            reduced_velocity=reduced_velocity,
            td_force_input=td_force_input,
            td_force_scale=output_scale,
        )
    corr_mu = raw_force * output_scale
    if predict_sigma:
        if force_zero_output:
            sigma = model.sigma_min.to(device=raw_force.device, dtype=raw_force.dtype) * output_scale
        else:
            raw_sigma = model._sigma_net_raw(
                z_model,
                reduced_velocity=reduced_velocity,
                td_force_input=td_force_input,
                td_force_scale=output_scale,
            )
            sigma = model.sigma_min.to(device=raw_sigma.device, dtype=raw_sigma.dtype) + F.softplus(raw_sigma)
            sigma = sigma * output_scale
    else:
        sigma = corr_mu.new_zeros(corr_mu.shape)
    return corr_mu, sigma


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


def _td_correction_rollout_loss_from_batch(
    *,
    model: PHVIV,
    batch: Any,
    device: torch.device,
    non_blocking: bool,
    td_params: dict[str, float],
    td_memory_cfg: dict[str, Any],
    predict_sigma: bool,
    force_zero_output: bool,
    rollout_loss_mode: str,
    rollout_stochastic_samples: int,
    rollout_noise_scale: float,
) -> torch.Tensor:
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

    mode_key = str(rollout_loss_mode).strip().lower()
    if mode_key == "stochastic":
        mode_key = "stochastic_nll"
    if mode_key not in {"deterministic", "stochastic_nll", "stochastic_mse"}:
        raise ValueError(
            "loss.rollout_loss_mode must be one of: deterministic, stochastic_nll, stochastic_mse."
        )
    dt_roll = torch.clamp((t_seq[:, 1] - t_seq[:, 0]).unsqueeze(1), min=1.0e-12)
    if mode_key == "deterministic" or not predict_sigma:
        z_pred, _force_seq, _corr_seq = _td_correction_state_rollout(
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
            predict_sigma=False,
            force_zero_output=force_zero_output,
        )
        return torch.mean(torch.sum((z_pred - z_traj) ** 2, dim=2))

    samples = max(1, int(rollout_stochastic_samples))
    batch_size = int(z0.shape[0])
    z0_in = z0.unsqueeze(0).expand(samples, *z0.shape).reshape(samples * batch_size, *z0.shape[1:])
    t_seq_in = t_seq.unsqueeze(0).expand(samples, *t_seq.shape).reshape(samples * batch_size, *t_seq.shape[1:])
    z_traj_ref = z_traj.unsqueeze(0)
    ur0_in = ur0.unsqueeze(0).expand(samples, *ur0.shape).reshape(samples * batch_size, *ur0.shape[1:])
    td_context0_in = td_context0.unsqueeze(0).expand(samples, *td_context0.shape).reshape(samples * batch_size, *td_context0.shape[1:])
    mass0_in = mass0.unsqueeze(0).expand(samples, *mass0.shape).reshape(samples * batch_size, *mass0.shape[1:])
    damping0_in = damping0.unsqueeze(0).expand(samples, *damping0.shape).reshape(samples * batch_size, *damping0.shape[1:])
    stiffness0_in = stiffness0.unsqueeze(0).expand(samples, *stiffness0.shape).reshape(samples * batch_size, *stiffness0.shape[1:])
    dt_roll_in = dt_roll.unsqueeze(0).expand(samples, *dt_roll.shape).reshape(samples * batch_size, *dt_roll.shape[1:])

    z_pred, _force_seq, _corr_seq = _td_correction_state_rollout(
        model=model,
        z0=z0_in,
        ur0=ur0_in,
        td_context0=td_context0_in,
        steps=int(z_traj.shape[1] - 1),
        dt=dt_roll_in,
        structural_mass=mass0_in,
        damping_c=damping0_in,
        stiffness=stiffness0_in,
        td_params=td_params,
        td_memory_cfg=td_memory_cfg,
        predict_sigma=True,
        force_zero_output=force_zero_output,
        rollout_stochastic=True,
        rollout_noise_scale=rollout_noise_scale,
    )
    z_pred = z_pred.reshape(samples, batch_size, *z_pred.shape[1:])
    if mode_key == "stochastic_mse":
        err = z_pred - z_traj_ref
        per_samples = torch.mean(err[..., 0] * err[..., 0], dim=2) + torch.mean(err[..., 1] * err[..., 1], dim=2)
        return torch.mean(torch.mean(per_samples, dim=0))

    mu = torch.mean(z_pred, dim=0)
    var = torch.mean((z_pred - mu.unsqueeze(0)) ** 2, dim=0)
    var = torch.clamp(var, min=1e-6)
    nll = 0.5 * (((z_traj - mu) ** 2) / var + torch.log(var))
    per = torch.mean(nll[..., 0], dim=1) + torch.mean(nll[..., 1], dim=1)
    return torch.mean(per)


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
    predict_sigma: bool = False,
    force_zero_output: bool = False,
    rollout_stochastic: bool = False,
    rollout_noise_scale: float = 1.0,
    rollout_seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z = z0
    td_context = td_context0
    z_hist = [z0]
    total_force_hist: list[torch.Tensor] = []
    corr_mu_hist: list[torch.Tensor] = []
    generator: torch.Generator | None = None
    if rollout_seed is not None:
        generator = torch.Generator(device=z0.device)
        generator.manual_seed(int(rollout_seed))
    for _ in range(int(steps)):
        velocity = z[:, 1:2] / structural_mass
        step_params = dict(td_params)
        step_params["n_memory"] = resolve_td_n_memory_torch(
            td_params,
            dt=dt,
            flow_speed=td_context[:, 4:5],
            diameter=float(model.D),
            memory_cfg=td_memory_cfg,
        )
        td_force_next, td_context_next = td_baseline_step_torch(
            velocity=velocity,
            acceleration=td_context[:, 0:1],
            td_context=td_context,
            dt=dt,
            rho=float(model.rho),
            diameter=float(model.D),
            params=step_params,
        )
        corr_mu, sigma_corr = _td_predict_correction(
            model,
            z=z,
            reduced_velocity=ur0,
            td_force_input=td_force_next,
            structural_mass=structural_mass,
            stiffness=stiffness,
            predict_sigma=predict_sigma,
            force_zero_output=force_zero_output,
        )
        corr_force = corr_mu
        if rollout_stochastic and predict_sigma:
            noise = torch.randn(
                corr_mu.shape,
                device=corr_mu.device,
                dtype=corr_mu.dtype,
                generator=generator,
            )
            corr_force = corr_mu + float(rollout_noise_scale) * sigma_corr * noise
        total_force = td_force_next + corr_force
        y_next, v_next, a_next = structural_step_constant_force_torch(
            y=z[:, 0:1],
            velocity=velocity,
            force=total_force,
            dt=dt,
            mass=structural_mass,
            damping_c=damping_c,
            stiffness=stiffness,
        )
        z = torch.cat([y_next, v_next * structural_mass], dim=1)
        td_context = td_context_next.clone()
        td_context[:, 0:1] = a_next
        z_hist.append(z)
        total_force_hist.append(total_force)
        corr_mu_hist.append(corr_force)
    z_seq = torch.stack(z_hist, dim=1)
    total_force_seq = torch.stack(total_force_hist, dim=1) if total_force_hist else z0.new_zeros((z0.shape[0], 0, 1))
    corr_mu_seq = torch.stack(corr_mu_hist, dim=1) if corr_mu_hist else z0.new_zeros((z0.shape[0], 0, 1))
    return z_seq, total_force_seq, corr_mu_seq


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
    mass_source: str,
    batch_size: int,
    rollout_batch_size: int,
    rollout_steps: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader | None, DataLoader | None]:
    mass_key = {
        "dry": "dry_mass_kg",
        "effective": "effective_mass_kg",
    }.get(str(mass_source).strip().lower())
    if mass_key is None:
        raise ValueError("td_mass_source must be one of: dry, effective.")

    def _one_step_dataset(trajs: list[dict[str, np.ndarray]]) -> TensorDataset | None:
        tensors: list[TensorDataset] = []
        for traj in trajs:
            y = torch.from_numpy(np.ascontiguousarray(traj["y"])).float()
            dy = torch.from_numpy(np.ascontiguousarray(traj["dy"])).float()
            t = torch.from_numpy(np.ascontiguousarray(traj["t"])).float()
            ur = torch.from_numpy(np.ascontiguousarray(traj["ur"])).float().unsqueeze(1)
            corr = torch.from_numpy(np.ascontiguousarray(traj["force_corr_per_m"])).float().unsqueeze(1)
            td_force = torch.from_numpy(np.ascontiguousarray(traj["force_td_per_m"])).float().unsqueeze(1)
            mass = torch.full((y.shape[0], 1), float(np.asarray(traj[mass_key]).reshape(())), dtype=torch.float32)
            damping = torch.full((y.shape[0], 1), float(np.asarray(traj["damping_c"]).reshape(())), dtype=torch.float32)
            stiffness = torch.full((y.shape[0], 1), float(np.asarray(traj["stiffness_n_m"]).reshape(())), dtype=torch.float32)
            z = torch.cat([y.unsqueeze(1), dy.unsqueeze(1) * mass], dim=1)
            items: list[torch.Tensor] = [
                z[:-1],
                t[:-1].unsqueeze(1),
                z[1:],
                t[1:].unsqueeze(1),
                ur[:-1],
                corr[1:],
                td_force[1:],
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
            ur = torch.from_numpy(np.ascontiguousarray(traj["ur"])).float().unsqueeze(1)
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
                ur0_list.append(ur[start])
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
    rollout_dataset = _rollout_dataset(val_trajs if val_trajs else train_trajs)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    rollout_loader = None
    if rollout_dataset is not None:
        rollout_loader = DataLoader(
            rollout_dataset,
            batch_size=rollout_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    return train_loader, val_loader, rollout_loader


def _log_td_correction_rollout_validation(
    *,
    writer: SummaryWriter,
    epoch: int,
    model: PHVIV,
    traj: dict[str, np.ndarray],
    dt: float,
    td_mass_source: str,
    td_params: dict[str, float],
    td_memory_cfg: dict[str, Any],
    device: torch.device,
    predict_sigma: bool = False,
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
) -> dict[str, float]:
    mass_key = "dry_mass_kg" if str(td_mass_source).strip().lower() == "dry" else "effective_mass_kg"
    mass_value = float(np.asarray(traj[mass_key]).reshape(()))
    damping_value = float(np.asarray(traj["damping_c"]).reshape(()))
    stiffness_value = float(np.asarray(traj["stiffness_n_m"]).reshape(()))
    y_true_t = torch.from_numpy(np.ascontiguousarray(traj["y"])).float().unsqueeze(1).to(device)
    v_true_t = torch.from_numpy(np.ascontiguousarray(traj["dy"])).float().unsqueeze(1).to(device)
    z_true_t = torch.cat([y_true_t, v_true_t * mass_value], dim=1)
    f_true_t = torch.from_numpy(np.ascontiguousarray(traj["force_per_m"])).float().unsqueeze(1).to(device)
    td_force_t = torch.from_numpy(np.ascontiguousarray(traj["force_td_per_m"])).float().unsqueeze(1).to(device)
    ur_t = torch.from_numpy(np.ascontiguousarray(traj["ur"])).float().unsqueeze(1).to(device)
    td_context_t = torch.from_numpy(np.ascontiguousarray(traj["td_context"])).float().to(device)
    t_np = np.asarray(traj["t"], dtype=float).reshape(-1)
    if z_true_t.shape[0] < 2:
        return {}
    traj_dt = float(t_np[1] - t_np[0])

    mass_t = torch.full((1, 1), mass_value, dtype=z_true_t.dtype, device=device)
    damping_t = torch.full((1, 1), damping_value, dtype=z_true_t.dtype, device=device)
    stiffness_t = torch.full((1, 1), stiffness_value, dtype=z_true_t.dtype, device=device)
    z_pred, total_force_seq, corr_mu_seq = _td_correction_state_rollout(
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
        predict_sigma=predict_sigma,
        force_zero_output=force_zero_output,
        rollout_stochastic=rollout_stochastic,
        rollout_noise_scale=rollout_noise_scale,
        rollout_seed=rollout_seed,
    )
    y_pred = z_pred[0, :, 0].detach().cpu().numpy()
    v_pred = (z_pred[0, :, 1] / mass_value).detach().cpu().numpy()
    force_roll = total_force_seq[0, :, 0].detach().cpu().numpy()
    corr_roll = corr_mu_seq[0, :, 0].detach().cpu().numpy()
    td_roll = force_roll - corr_roll

    with torch.no_grad():
        corr_on_data, sigma_on_data = _td_predict_correction(
            model,
            z=z_true_t,
            reduced_velocity=ur_t,
            td_force_input=td_force_t,
            structural_mass=torch.full_like(y_true_t, mass_value),
            stiffness=torch.full_like(y_true_t, stiffness_value),
            predict_sigma=False,
            force_zero_output=force_zero_output,
        )
    force_total_full = np.concatenate(
        [np.asarray([float((td_force_t[0:1] + corr_on_data[0:1])[0, 0].detach().cpu())]), force_roll],
        axis=0,
    )
    force_td_full = np.concatenate([td_force_t[:1, 0].detach().cpu().numpy(), td_roll], axis=0)

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
    force_true = f_true_t[:, 0].detach().cpu().numpy()
    force_model_on_data = (td_force_t[:, 0] + corr_on_data[:, 0]).detach().cpu().numpy()
    force_std = float(np.std(force_true))
    if force_std <= 0.0:
        force_std = 1.0
    metrics[FORCE_MAPPING_NRMSE_KEY] = float(np.sqrt(np.mean((force_model_on_data - force_true) ** 2))) / force_std

    if log_metrics:
        for name, value in metrics.items():
            if np.isfinite(float(value)):
                writer.add_scalar(f"val/{name}", float(value), epoch)

    if log_plots:
        zoom_mask = create_zoom_mask(t_np)
        ur_val = float(ur_t[0, 0].detach().cpu().item())
        omega = float(np.sqrt(stiffness_value / mass_value))
        q_true_norm = y_true_t[:, 0].detach().cpu().numpy() / float(model.D)
        p_true_norm = v_true_t[:, 0].detach().cpu().numpy() / (omega * float(model.D))
        q_pred_norm = y_pred / float(model.D)
        p_pred_norm = v_pred / (omega * float(model.D))
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
        )
        n_force = min(len(t_np), len(force_total_full), len(force_true), len(force_td_full))
        force_t = t_np[:n_force]
        log_force_plots(
            writer,
            epoch,
            force_t,
            force_total_full[:n_force],
            force_true[:n_force],
            create_zoom_mask(force_t),
            reduced_velocity=ur_val,
            force_coeff_baseline=force_td_full[:n_force],
            baseline_label="C_F (Vivana-TD)",
            tag_prefix=tag_prefix,
            step=step,
            title_suffix=title_suffix,
        )
        if log_correction_on_data:
            output_label = "Correction coefficient" if str(getattr(model, "force_output", "force")) == "coefficient" else "Correction force"
            log_correction_on_data_plot(
                writer,
                epoch,
                t=t_np,
                corr_true=(f_true_t[:, 0] - td_force_t[:, 0]).detach().cpu().numpy(),
                corr_pred=corr_on_data[:, 0].detach().cpu().numpy(),
                sigma=(
                    sigma_on_data[:, 0].detach().cpu().numpy()
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
        if log_phase_map:
            q_extent = np.concatenate([np.asarray(q_true_norm, dtype=float), np.asarray(q_pred_norm, dtype=float)])
            p_extent = np.concatenate([np.asarray(p_true_norm, dtype=float), np.asarray(p_pred_norm, dtype=float)])
            q_grid, p_grid = build_phase_plot_grid(q_extent, p_extent, bins=96, extent_scale=1.2)
            y_grid = torch.as_tensor(
                (q_grid.reshape(-1) * float(model.D)).reshape(-1, 1),
                dtype=z_true_t.dtype,
                device=device,
            )
            v_grid = torch.as_tensor(
                (p_grid.reshape(-1) * (omega * float(model.D))).reshape(-1, 1),
                dtype=z_true_t.dtype,
                device=device,
            )
            z_grid = torch.cat([y_grid, v_grid * mass_value], dim=1)
            ur_grid = torch.full((z_grid.shape[0], 1), ur_val, dtype=z_true_t.dtype, device=device)
            td_force_grid = None
            if bool(getattr(model, "use_td_force_input", False)):
                td_force_grid_np = nearest_phase_series_values(
                    q_grid,
                    p_grid,
                    q_true_norm,
                    p_true_norm,
                    td_force_t[:, 0].detach().cpu().numpy(),
                )
                td_force_grid = torch.as_tensor(
                    td_force_grid_np.reshape(-1, 1),
                    dtype=z_true_t.dtype,
                    device=device,
                )
            with torch.no_grad():
                corr_grid, sigma_grid = _td_predict_correction(
                    model,
                    z=z_grid,
                    reduced_velocity=ur_grid,
                    td_force_input=td_force_grid,
                    structural_mass=torch.full((z_grid.shape[0], 1), mass_value, dtype=z_true_t.dtype, device=device),
                    stiffness=torch.full((z_grid.shape[0], 1), stiffness_value, dtype=z_true_t.dtype, device=device),
                    predict_sigma=False,
                    force_zero_output=force_zero_output,
                )
            output_label = "Correction coefficient" if str(getattr(model, "force_output", "force")) == "coefficient" else "Correction force"
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
                    if predict_sigma
                    else None
                ),
                sigma_label=(
                    "Sigma coefficient" if str(getattr(model, "force_output", "force")) == "coefficient" else "Sigma force"
                ),
                tag="final_val/phase_output",
                step=step,
                title_suffix=title_suffix,
            )
    return metrics


def _odd_symmetry_penalty_per_sample(
    *,
    model: torch.nn.Module,
    z: torch.Tensor,
    ur: torch.Tensor,
    norm: str,
) -> torch.Tensor:
    z_flip = -z
    if getattr(model, "force_output", "force") == "coefficient":
        f_pos = model.u_theta_coeff(z, reduced_velocity=ur)
        f_neg = model.u_theta_coeff(z_flip, reduced_velocity=ur)
    else:
        f_pos = model.u_theta(z, reduced_velocity=ur)
        f_neg = model.u_theta(z_flip, reduced_velocity=ur)
    sym_residual = f_pos + f_neg
    if sym_residual.ndim == 1:
        sym_residual = sym_residual.unsqueeze(-1)
    if norm == "l1":
        return torch.mean(torch.abs(sym_residual), dim=1)
    return torch.mean(sym_residual * sym_residual, dim=1)


def _rollout_loss_from_batch(
    *,
    model: PHVIV,
    batch: Any,
    device: torch.device,
    non_blocking: bool,
    rollout_loss_mode: str,
    rollout_stochastic_samples: int,
    rollout_noise_scale: float,
    ur_bin_state_scale_info: dict[str, Any] | None = None,
    ur_bin_size: float = 1e-6,
    return_per_sample: bool = False,
) -> torch.Tensor:
    z0, t_seq, z_traj, ur0, _history0, _scale = _parse_rollout_batch(batch)
    z0 = z0.to(device, non_blocking=non_blocking)
    t_seq = t_seq.to(device, non_blocking=non_blocking)
    z_traj = z_traj.to(device, non_blocking=non_blocking)
    ur0 = ur0.to(device, non_blocking=non_blocking)
    mode_key = str(rollout_loss_mode).strip().lower()
    if mode_key == "stochastic":
        mode_key = "stochastic_nll"
    if mode_key not in {"deterministic", "stochastic_nll", "stochastic_mse"}:
        raise ValueError(
            "loss.rollout_loss_mode must be one of: deterministic, stochastic_nll, stochastic_mse."
        )
    samples = max(1, int(rollout_stochastic_samples))
    batch_size = z0.shape[0]
    extra_state_scale = None
    if ur_bin_state_scale_info is not None:
        extra_state_scale = lookup_ur_bin_state_scale_tensor(
            ur0,
            scale_info=ur_bin_state_scale_info,
            ur_bin_size=ur_bin_size,
            batch_size=batch_size,
            device=z0.device,
            dtype=z0.dtype,
        )
    if mode_key in {"stochastic_nll", "stochastic_mse"} and samples > 1:
        z0_in = z0.unsqueeze(0).expand(samples, *z0.shape).reshape(samples * batch_size, *z0.shape[1:])
        t_seq_in = t_seq.unsqueeze(0).expand(samples, *t_seq.shape).reshape(samples * batch_size, *t_seq.shape[1:])
        z_traj_ref = z_traj.unsqueeze(0)
        ur0_in = ur0.unsqueeze(0).expand(samples, *ur0.shape).reshape(samples * batch_size, *ur0.shape[1:])
        z_pred, _ = model.rollout(
            z0_in,
            t_seq_in,
            float(model.dt),
            reduced_velocity=ur0_in,
            stochastic=True,
            noise_scale=rollout_noise_scale,
        )
        z_pred = z_pred.reshape(samples, batch_size, *z_pred.shape[1:])
        z_scale = model.res_scale.to(device=z_pred.device, dtype=z_pred.dtype).view(1, 1, 1, -1)
        if extra_state_scale is not None:
            z_scale = z_scale * extra_state_scale.view(1, batch_size, 1, -1).to(device=z_pred.device, dtype=z_pred.dtype)
        if mode_key == "stochastic_nll":
            z_pred_scaled = z_pred / z_scale
            z_true_scaled = z_traj_ref / z_scale
            mu = torch.mean(z_pred_scaled, dim=0)
            var = torch.mean((z_pred_scaled - mu.unsqueeze(0)) ** 2, dim=0)
            var = torch.clamp(var, min=1e-6)
            nll = 0.5 * (((z_true_scaled - mu) ** 2) / var + torch.log(var))
            per = torch.mean(nll[..., 0], dim=1) + torch.mean(nll[..., 1], dim=1)
        else:
            err = (z_pred - z_traj_ref) / z_scale
            per_samples = torch.mean(err[..., 0] * err[..., 0], dim=2) + torch.mean(err[..., 1] * err[..., 1], dim=2)
            per = torch.mean(per_samples, dim=0)
    else:
        z_pred, _ = model.rollout(
            z0,
            t_seq,
            float(model.dt),
            reduced_velocity=ur0,
            stochastic=(mode_key != "deterministic"),
            noise_scale=rollout_noise_scale,
        )
        z_scale = model.res_scale.to(device=z_pred.device, dtype=z_pred.dtype).view(1, 1, -1)
        if extra_state_scale is not None:
            z_scale = z_scale * extra_state_scale.view(batch_size, 1, -1).to(device=z_pred.device, dtype=z_pred.dtype)
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
    ur_bin_size: float,
    normalize_residual_by_ur_bin_std: bool,
    normalize_rollout_by_ur_bin_std: bool,
    ur_bin_state_scale_info: dict[str, Any] | None,
    rollout_det_weight: float,
    rollout_loss_mode: str,
    rollout_stochastic_samples: int,
    rollout_noise_scale: float,
    use_force_data_loss: bool,
    force_data_weight: float,
    gradnorm_balancer: Optional[GradNormBalancer],
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    scaler: torch.amp.GradScaler,
    log_component_grad_norms: bool,
    symmetry_weight: float,
    symmetry_norm: str,
) -> dict[str, float]:
    batch_count = 0
    loss_sum = torch.zeros((), device=device)
    res_loss_sum = torch.zeros((), device=device)
    sigma_reg_sum = torch.zeros((), device=device)
    mean_reg_sum = torch.zeros((), device=device)
    force_data_loss_sum = torch.zeros((), device=device)
    sym_loss_sum = torch.zeros((), device=device)
    rollout_det_loss_sum = torch.zeros((), device=device)
    grad_norm_sum = torch.zeros((), device=device)
    avg_sigma_reg_sum = torch.zeros((), device=device)
    avg_mean_reg_sum = torch.zeros((), device=device)
    res_grad_component_sum = torch.zeros((), device=device)
    sigma_grad_component_sum = torch.zeros((), device=device)
    mean_grad_component_sum = torch.zeros((), device=device)
    gradnorm_res_weight_sum = torch.zeros((), device=device)
    gradnorm_data_weight_sum = torch.zeros((), device=device) if use_force_data_loss else None
    gradnorm_sym_weight_sum = torch.zeros((), device=device) if float(symmetry_weight) > 0.0 else None
    gradnorm_weight_count = 0

    force_output_coeff = getattr(model, "force_output", "force") == "coefficient"
    rollout_iter = iter(train_rollout_loader) if (train_rollout_loader is not None and float(rollout_det_weight) > 0.0) else None
    for batch in train_loader:
        z_i, t_i, z_next, t_next, ur_i, _history_i, f_i, f_next, _scale = _parse_hnn_batch(batch)
        z_i = z_i.to(device, non_blocking=non_blocking)
        t_i = t_i.to(device, non_blocking=non_blocking)
        z_next = z_next.to(device, non_blocking=non_blocking)
        t_next = t_next.to(device, non_blocking=non_blocking)
        ur_i = ur_i.to(device, non_blocking=non_blocking)
        if f_i is not None:
            f_i = f_i.to(device, non_blocking=non_blocking)
        if f_next is not None:
            f_next = f_next.to(device, non_blocking=non_blocking)
        opt.zero_grad()

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
            if normalize_residual_by_ur_bin_std:
                per_res = scaled_residual_loss_per_sample(
                    model,
                    z_i,
                    z_next,
                    reduced_velocity=ur_i,
                    ur_bin_state_scale_info=(ur_bin_state_scale_info if normalize_residual_by_ur_bin_std else None),
                    ur_bin_size=ur_bin_size,
                )
            else:
                per_res = None
            res_loss = (
                torch.mean(per_res)
                if per_res is not None
                else model.res_loss(
                    z_i,
                    t_i,
                    z_next,
                    t_next,
                    reduced_velocity=ur_i,
                )
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
            if use_force_data_loss:
                if f_i is None or f_next is None:
                    raise ValueError(
                        "use_force_data_loss is True but the dataloader did not provide force labels."
                    )
                z_mid = 0.5 * (z_i + z_next)
                f_mid = 0.5 * (f_i + f_next)
                if force_output_coeff:
                    f0 = model._force_scale_from_reduced_velocity(ur_i, like=f_mid, state=z_mid)
                    f_pred = model.u_theta_coeff(z_mid, reduced_velocity=ur_i)
                    f_mid = f_mid / f0
                else:
                    f_pred = model.u_theta(z_mid, reduced_velocity=ur_i)
                per_data = torch.mean((f_pred - f_mid) ** 2, dim=1)
                data_force_loss = torch.mean(per_data)
            else:
                data_force_loss = res_loss.new_tensor(0.0)
                z_mid = 0.5 * (z_i + z_next)

            if float(symmetry_weight) > 0.0:
                per_sym = _odd_symmetry_penalty_per_sample(
                    model=model,
                    z=z_mid,
                    ur=ur_i,
                    norm=symmetry_norm,
                )
                sym_loss = torch.mean(per_sym)
            else:
                sym_loss = res_loss.new_tensor(0.0)

            if gradnorm_balancer is not None:
                loss_inputs: dict[str, torch.Tensor] = {
                    "residual": res_loss.float(),
                }
                if "data" in gradnorm_balancer.names:
                    loss_inputs["data"] = data_force_loss.float()
                if "symmetry" in gradnorm_balancer.names:
                    loss_inputs["symmetry"] = sym_loss.float()
                weights = gradnorm_balancer.update(loss_inputs)
                res_weight = weights["residual"]
                sigma_weight = res_loss.new_tensor(1.0)
                mean_weight = res_loss.new_tensor(1.0)
                data_weight = weights.get("data", res_loss.new_tensor(1.0))
                sym_weight = weights.get("symmetry", res_loss.new_tensor(1.0))
                gradnorm_res_weight_sum = gradnorm_res_weight_sum + res_weight
                if gradnorm_data_weight_sum is not None:
                    gradnorm_data_weight_sum = gradnorm_data_weight_sum + data_weight
                if gradnorm_sym_weight_sum is not None:
                    gradnorm_sym_weight_sum = gradnorm_sym_weight_sum + sym_weight
                gradnorm_weight_count += 1
            else:
                res_weight = res_loss.new_tensor(1.0)
                sigma_weight = res_loss.new_tensor(1.0)
                mean_weight = res_loss.new_tensor(1.0)
                data_weight = res_loss.new_tensor(1.0)
                sym_weight = res_loss.new_tensor(1.0)

            # GradNorm balances raw branch losses first; user multipliers are applied after.
            gradnorm_weighted_res = res_weight * res_loss
            gradnorm_weighted_sigma = sigma_weight * base_sigma_reg_loss
            gradnorm_weighted_mean = mean_weight * base_mean_reg_loss
            gradnorm_weighted_data = data_weight * data_force_loss

            weighted_sigma = float(sigma_reg) * gradnorm_weighted_sigma
            weighted_mean = float(mean_reg) * gradnorm_weighted_mean
            weighted_data = float(force_data_weight) * gradnorm_weighted_data
            gradnorm_weighted_sym = sym_weight * sym_loss
            weighted_sym = float(symmetry_weight) * gradnorm_weighted_sym
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
                    rollout_loss_mode=rollout_loss_mode,
                    rollout_stochastic_samples=rollout_stochastic_samples,
                    rollout_noise_scale=rollout_noise_scale,
                    ur_bin_state_scale_info=(ur_bin_state_scale_info if normalize_rollout_by_ur_bin_std else None),
                    ur_bin_size=ur_bin_size,
                )
            else:
                rollout_det_loss = res_loss.new_tensor(0.0)
            weighted_rollout_det = float(rollout_det_weight) * rollout_det_loss
            loss = (
                gradnorm_weighted_res
                + weighted_sigma
                + weighted_mean
                + weighted_data
                + weighted_sym
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
        force_data_loss_sum = force_data_loss_sum + data_force_loss.detach().float()
        sym_loss_sum = sym_loss_sum + sym_loss.detach().float()
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
        "mean_force_data_loss": float((force_data_loss_sum / denom).detach().cpu()),
        "mean_sym_loss": float((sym_loss_sum / denom).detach().cpu()),
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
        if gradnorm_data_weight_sum is not None:
            metrics["mean_gradnorm_weight_data"] = float(
                (gradnorm_data_weight_sum / float(gradnorm_weight_count)).detach().cpu()
            )
        if gradnorm_sym_weight_sum is not None:
            metrics["mean_gradnorm_weight_symmetry"] = float(
                (gradnorm_sym_weight_sum / float(gradnorm_weight_count)).detach().cpu()
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
    validation_samples_per_ur: int,
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
    rollout_stochastic: bool,
    rollout_noise_scale: float,
    rollout_seed: int | None,
    mean_reg: float,
    mean_reg_norm: str,
    sigma_reg: float,
    sigma_reg_norm: str,
    ur_bin_size: float,
    normalize_residual_by_ur_bin_std: bool,
    normalize_rollout_by_ur_bin_std: bool,
    ur_bin_state_scale_info: dict[str, Any] | None,
    rollout_det_weight: float,
    rollout_loss_mode: str,
    rollout_stochastic_samples: int,
    use_force_data_loss: bool,
    force_data_weight: float,
    symmetry_weight: float,
    symmetry_norm: str,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> None:
    if not validate_now and not rollout_now:
        return
    validation_start = time.perf_counter()
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
            normalize_residual_by_ur_bin_std=normalize_residual_by_ur_bin_std,
            normalize_rollout_by_ur_bin_std=normalize_rollout_by_ur_bin_std,
            ur_bin_state_scale_info=ur_bin_state_scale_info,
            ur_bin_size=ur_bin_size,
            rollout_det_weight=rollout_det_weight,
            rollout_loss_mode=rollout_loss_mode,
            rollout_stochastic_samples=rollout_stochastic_samples,
            rollout_noise_scale=rollout_noise_scale,
            use_force_data_loss=use_force_data_loss,
            force_data_weight=force_data_weight,
            symmetry_weight=symmetry_weight,
            symmetry_norm=symmetry_norm,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        for name, value in val_loss_metrics.items():
            writer.add_scalar(f"val/{name}", value, epoch + 1)
    if not rollout_now:
        writer.add_scalar("val/validation_wall_time_s", time.perf_counter() - validation_start, epoch + 1)
        return

    if val_series_raw is not None and val_sequences is not None:
        metrics_sum: dict[str, float] = {}
        count = 0
        diverged_count = 0
        total = min(len(val_series_raw), len(val_sequences))
        ur_for_sampling: list[float] = []
        for idx in range(total):
            ur_arr = np.asarray(val_series_raw[idx][5]).reshape(-1)
            ur_for_sampling.append(float(ur_arr[0]) if ur_arr.size > 0 else float("nan"))
        sampled_indices = sample_indices_per_ur(
            ur_for_sampling,
            samples_per_ur=validation_samples_per_ur,
            seed=1,
        )
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
                rollout_stochastic=rollout_stochastic,
                rollout_noise_scale=rollout_noise_scale,
                rollout_seed=rollout_seed,
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
            rollout_stochastic=rollout_stochastic,
            rollout_noise_scale=rollout_noise_scale,
            rollout_seed=rollout_seed,
        )
        writer.add_scalar("val/validation_wall_time_s", time.perf_counter() - validation_start, epoch + 1)
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
        rollout_stochastic=rollout_stochastic,
        rollout_noise_scale=rollout_noise_scale,
        rollout_seed=rollout_seed,
    )
    writer.add_scalar("val/validation_wall_time_s", time.perf_counter() - validation_start, epoch + 1)


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
    rollout_stochastic: bool,
    rollout_noise_scale: float,
    rollout_seed: int | None,
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
            rollout_stochastic=rollout_stochastic,
            rollout_noise_scale=rollout_noise_scale,
            rollout_seed=rollout_seed,
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
            rollout_stochastic=rollout_stochastic,
            rollout_noise_scale=rollout_noise_scale,
            rollout_seed=rollout_seed,
            tag_prefix="final_val/rollout",
            step=step_idx,
            title_suffix=f" [final {step_idx}/{len(plot_pairs)}]",
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
            stochastic=rollout_stochastic,
            rollout_seed=rollout_seed,
            noise_scale=rollout_noise_scale,
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
            if writer is not None and epoch > 0:
                writer.add_scalar("val/validation_wall_time_s", float(elapsed), int(epoch))
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
                                f"[async-val] epoch {epoch}: new best val/loss_total={loss_total_f:.6e}; "
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
    ur_bin_size: float,
    normalize_residual_by_ur_bin_std: bool,
    normalize_rollout_by_ur_bin_std: bool,
    ur_bin_state_scale_info: dict[str, Any] | None,
    rollout_det_weight: float,
    rollout_loss_mode: str,
    rollout_stochastic_samples: int,
    rollout_noise_scale: float,
    use_force_data_loss: bool,
    force_data_weight: float,
    symmetry_weight: float,
    symmetry_norm: str,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    # Keep validation rollout-loss estimation deterministic to control cost and variance.
    val_rollout_loss_mode = "deterministic"
    val_rollout_stochastic_samples = 1
    force_output_coeff = getattr(model, "force_output", "force") == "coefficient"
    loss_sum = torch.zeros((), device=device)
    res_sum = torch.zeros((), device=device)
    sigma_sum = torch.zeros((), device=device)
    mean_reg_sum = torch.zeros((), device=device)
    data_sum = torch.zeros((), device=device)
    sym_sum = torch.zeros((), device=device)
    rollout_det_sum = torch.zeros((), device=device)
    batches = 0
    rollout_iter = iter(rollout_loader) if (rollout_loader is not None and float(rollout_det_weight) > 0.0) else None
    with torch.no_grad():
        for batch in loader:
            z_i, t_i, z_next, t_next, ur_i, _history_i, f_i, f_next, _scale = _parse_hnn_batch(batch)
            z_i = z_i.to(device, non_blocking=non_blocking)
            t_i = t_i.to(device, non_blocking=non_blocking)
            z_next = z_next.to(device, non_blocking=non_blocking)
            t_next = t_next.to(device, non_blocking=non_blocking)
            ur_i = ur_i.to(device, non_blocking=non_blocking)
            if f_i is not None:
                f_i = f_i.to(device, non_blocking=non_blocking)
            if f_next is not None:
                f_next = f_next.to(device, non_blocking=non_blocking)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                if normalize_residual_by_ur_bin_std:
                    per_res = scaled_residual_loss_per_sample(
                        model,
                        z_i,
                        z_next,
                        reduced_velocity=ur_i,
                        ur_bin_state_scale_info=(ur_bin_state_scale_info if normalize_residual_by_ur_bin_std else None),
                        ur_bin_size=ur_bin_size,
                    )
                else:
                    per_res = None
                res_loss = (
                    torch.mean(per_res)
                    if per_res is not None
                    else model.res_loss(
                        z_i,
                        t_i,
                        z_next,
                        t_next,
                        reduced_velocity=ur_i,
                    )
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
                if use_force_data_loss:
                    if f_i is None or f_next is None:
                        raise ValueError(
                            "use_force_data_loss is True but the dataloader did not provide force labels."
                        )
                    z_mid = 0.5 * (z_i + z_next)
                    f_mid = 0.5 * (f_i + f_next)
                    if force_output_coeff:
                        f0 = model._force_scale_from_reduced_velocity(ur_i, like=f_mid, state=z_mid)
                        f_pred = model.u_theta_coeff(z_mid, reduced_velocity=ur_i)
                        f_mid = f_mid / f0
                    else:
                        f_pred = model.u_theta(z_mid, reduced_velocity=ur_i)
                    per_data = torch.mean((f_pred - f_mid) ** 2, dim=1)
                    data_force_loss = torch.mean(per_data)
                else:
                    data_force_loss = res_loss.new_tensor(0.0)
                    z_mid = 0.5 * (z_i + z_next)
                if float(symmetry_weight) > 0.0:
                    per_sym = _odd_symmetry_penalty_per_sample(
                        model=model,
                        z=z_mid,
                        ur=ur_i,
                        norm=symmetry_norm,
                    )
                    sym_loss = torch.mean(per_sym)
                else:
                    sym_loss = res_loss.new_tensor(0.0)
                total = (
                    res_loss
                    + sigma_loss
                    + mean_loss_reg
                    + float(force_data_weight) * data_force_loss
                    + float(symmetry_weight) * sym_loss
                )
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
                        rollout_loss_mode=val_rollout_loss_mode,
                        rollout_stochastic_samples=val_rollout_stochastic_samples,
                        rollout_noise_scale=rollout_noise_scale,
                        ur_bin_state_scale_info=(ur_bin_state_scale_info if normalize_rollout_by_ur_bin_std else None),
                        ur_bin_size=ur_bin_size,
                    )
                else:
                    rollout_det_loss = res_loss.new_tensor(0.0)
                total = total + float(rollout_det_weight) * rollout_det_loss

            loss_sum = loss_sum + total.detach().float()
            res_sum = res_sum + res_loss.detach().float()
            # Log loss_reg as the raw regularizer magnitude (before sigma_reg scaling).
            sigma_sum = sigma_sum + sigma_reg_loss.detach().float()
            mean_reg_sum = mean_reg_sum + mean_reg_loss.detach().float()
            data_sum = data_sum + data_force_loss.detach().float()
            sym_sum = sym_sum + sym_loss.detach().float()
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
        "loss_data": float((data_sum / denom).detach().cpu()),
        "loss_sym": float((sym_sum / denom).detach().cpu()),
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
    normalize_residual_by_ur_bin_std: bool,
    normalize_rollout_by_ur_bin_std: bool,
    ur_bin_state_scale_info: dict[str, Any] | None,
    ur_bin_size: float,
    rollout_det_weight: float,
    rollout_loss_mode: str,
    rollout_stochastic_samples: int,
    rollout_noise_scale: float,
    use_force_data_loss: bool,
    force_data_weight: float,
    symmetry_weight: float,
    symmetry_norm: str,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> dict[str, dict[float, float]]:
    model.eval()
    # Keep validation rollout-loss estimation deterministic to control cost and variance.
    val_rollout_loss_mode = "deterministic"
    val_rollout_stochastic_samples = 1
    amp_enabled = bool(amp_enabled) and device.type == "cuda"
    force_output_coeff = getattr(model, "force_output", "force") == "coefficient"
    buckets: dict[str, dict[float, list[float]]] = {
        "loss_physics": {},
        "loss_reg": {},
        "loss_reg_mean": {},
        "loss_data": {},
        "loss_sym": {},
        "loss_rollout_det": {},
    }
    with torch.no_grad():
        for batch in loader:
            z_i, t_i, z_next, t_next, ur_i, _history_i, f_i, f_next, _scale = _parse_hnn_batch(batch)
            z_i = z_i.to(device, non_blocking=non_blocking)
            t_i = t_i.to(device, non_blocking=non_blocking)
            z_next = z_next.to(device, non_blocking=non_blocking)
            t_next = t_next.to(device, non_blocking=non_blocking)
            ur_i = ur_i.to(device, non_blocking=non_blocking)
            if f_i is not None:
                f_i = f_i.to(device, non_blocking=non_blocking)
            if f_next is not None:
                f_next = f_next.to(device, non_blocking=non_blocking)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                per_res = (
                    scaled_residual_loss_per_sample(
                        model,
                        z_i,
                        z_next,
                        reduced_velocity=ur_i,
                        ur_bin_state_scale_info=(ur_bin_state_scale_info if normalize_residual_by_ur_bin_std else None),
                        ur_bin_size=ur_bin_size,
                    )
                    if normalize_residual_by_ur_bin_std
                    else model.res_loss_per_sample(
                        z_i,
                        t_i,
                        z_next,
                        t_next,
                        reduced_velocity=ur_i,
                    )
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
                if use_force_data_loss and f_i is not None and f_next is not None:
                    z_mid = 0.5 * (z_i + z_next)
                    f_mid = 0.5 * (f_i + f_next)
                    if force_output_coeff:
                        f0 = model._force_scale_from_reduced_velocity(ur_i, like=f_mid, state=z_mid)
                        f_pred = model.u_theta_coeff(z_mid, reduced_velocity=ur_i)
                        f_mid = f_mid / f0
                    else:
                        f_pred = model.u_theta(z_mid, reduced_velocity=ur_i)
                    per_data = torch.mean((f_pred - f_mid) ** 2, dim=1)
                    per_data = float(force_data_weight) * per_data
                else:
                    per_data = per_res.new_zeros(per_res.shape)
                    z_mid = 0.5 * (z_i + z_next)
                if float(symmetry_weight) > 0.0:
                    per_sym = _odd_symmetry_penalty_per_sample(
                        model=model,
                        z=z_mid,
                        ur=ur_i,
                        norm=symmetry_norm,
                    )
                else:
                    per_sym = per_res.new_zeros(per_res.shape)

            ur_vals = ur_i.detach().cpu().view(-1).numpy()
            per_res_vals = per_res.detach().cpu().view(-1).numpy()
            per_reg_vals = per_sigma.detach().cpu().view(-1).numpy()
            per_mean_vals = per_mean.detach().cpu().view(-1).numpy()
            per_data_vals = per_data.detach().cpu().view(-1).numpy()
            per_sym_vals = per_sym.detach().cpu().view(-1).numpy()
            for u, res_v, reg_v, mean_v, data_v, sym_v in zip(
                ur_vals, per_res_vals, per_reg_vals, per_mean_vals, per_data_vals, per_sym_vals
            ):
                key = float(np.round(u, 6))
                buckets["loss_physics"].setdefault(key, []).append(float(res_v))
                buckets["loss_reg"].setdefault(key, []).append(float(reg_v))
                buckets["loss_reg_mean"].setdefault(key, []).append(float(mean_v))
                buckets["loss_data"].setdefault(key, []).append(float(data_v))
                buckets["loss_sym"].setdefault(key, []).append(float(sym_v))

    if rollout_loader is not None and float(rollout_det_weight) > 0.0:
        with torch.no_grad():
            for batch in rollout_loader:
                _z0, _t_seq, _z_traj, ur0, _history0, _scale = _parse_rollout_batch(batch)
                per_rollout = _rollout_loss_from_batch(
                    model=model,
                    batch=batch,
                    device=device,
                    non_blocking=non_blocking,
                    rollout_loss_mode=val_rollout_loss_mode,
                    rollout_stochastic_samples=val_rollout_stochastic_samples,
                    rollout_noise_scale=rollout_noise_scale,
                    ur_bin_state_scale_info=(ur_bin_state_scale_info if normalize_rollout_by_ur_bin_std else None),
                    ur_bin_size=ur_bin_size,
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


def _as_float_list(values: Any, *, key: str) -> list[float] | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return []
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{key} must contain only finite numeric values.")
    return [float(v) for v in arr.tolist()]


def _train_td_correction(config: Config, config_name: str) -> None:
    data_cfg = config.data
    model_cfg = config.model
    training_cfg = config.training
    optim_cfg = config.optim
    loss_cfg = config.loss
    runtime_cfg = config.runtime
    precision_cfg = config.precision
    compile_cfg = config.compile
    monitoring_cfg = config.monitoring
    hnn_cfg = dict(config.hnn or {})

    device = select_device(os.getenv("TRAIN_DEVICE", str(runtime_cfg.device)))
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA available: {torch.cuda.is_available()}, gpu0: {torch.cuda.get_device_name(0)}")
    configure_tf32(device, bool(precision_cfg.use_tf32))
    set_num_threads_from_slurm(default=1)
    non_blocking = device.type == "cuda"

    train_series_root = Path(data_cfg.train_series_dir)
    train_dir = train_series_root / "train"
    val_dir = train_series_root / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError("TD correction mode expects train/ and val/ directories under data.train_series_dir.")
    train_paths = sorted(train_dir.glob("*.npz"))
    val_paths = sorted(val_dir.glob("*.npz"))
    if not train_paths:
        raise FileNotFoundError("No TD correction training trajectories were found.")

    td_mass_source = str(hnn_cfg.get("td_mass_source", "dry")).strip().lower()
    if td_mass_source not in {"dry", "effective"}:
        raise ValueError("hnn.td_mass_source must be one of: dry, effective.")
    td_params = resolve_td_correction_params(hnn_cfg)
    td_memory_cfg = resolve_td_memory_config(hnn_cfg)
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
        ur_source=td_mass_source,
        td_params=td_params,
        td_memory_cfg=td_memory_cfg,
    )
    val_trajs = (
        load_td_correction_trajectories(
            paths=val_paths,
            cut_start_seconds=val_cut,
            reduce_time=reduce_time_enabled,
            reduction_factor=reduction_factor,
            stagger_reduced_time=stagger_val_reduce,
            ur_source=td_mass_source,
            td_params=td_params,
            td_memory_cfg=td_memory_cfg,
        )
        if val_paths
        else []
    )
    dt = float(train_trajs[0]["t"][1] - train_trajs[0]["t"][0])
    predict_sigma = bool(hnn_cfg.get("predict_sigma", False))
    use_td_force_input = bool(hnn_cfg.get("use_td_force_input", False))
    state_loss_mode = str(hnn_cfg.get("state_loss_mode", "mse")).strip().lower()
    if state_loss_mode not in {"mse", "propagated_nll"}:
        raise ValueError("hnn.state_loss_mode must be one of: mse, propagated_nll.")
    force_zero_output = bool(hnn_cfg.get("force_zero_output", False))
    corr_init_mode, corr_init_tiny_std = _resolve_td_correction_init_settings(hnn_cfg, model_cfg)
    rollout_det_weight = float(getattr(loss_cfg, "rollout_det_weight", 0.0))
    rollout_det_steps = int(getattr(loss_cfg, "rollout_det_steps", 0))
    rollout_loss_mode = str(getattr(loss_cfg, "rollout_loss_mode", "deterministic")).strip().lower()
    rollout_stochastic_samples = int(getattr(loss_cfg, "rollout_stochastic_samples", 1))
    rollout_batch_size_raw = int(getattr(loss_cfg, "rollout_det_batch_size", 0))
    rollout_batch_size = int(training_cfg.batch_size) if rollout_batch_size_raw <= 0 else rollout_batch_size_raw
    mean_reg = float(getattr(loss_cfg, "mean_reg", 0.0))
    sigma_reg = float(getattr(loss_cfg, "sigma_reg", 0.0))
    mean_reg_norm = str(getattr(loss_cfg, "mean_reg_norm", "l1")).strip().lower()
    sigma_reg_norm = str(getattr(loss_cfg, "sigma_reg_norm", "l2")).strip().lower()
    use_force_data_loss = bool(getattr(loss_cfg, "use_force_data_loss", True))
    force_data_weight = float(getattr(loss_cfg, "force_data_weight", 1.0))
    rollout_det_steps_final_raw = int(getattr(loss_cfg, "rollout_det_steps_final", 0))
    rollout_det_steps_warmup_epochs = int(getattr(loss_cfg, "rollout_det_steps_warmup_epochs", 0))
    rollout_det_steps_final = rollout_det_steps if rollout_det_steps_final_raw <= 0 else rollout_det_steps_final_raw
    rollout_stochastic = bool(hnn_cfg.get("rollout_stochastic", False))
    rollout_noise_scale = float(hnn_cfg.get("rollout_noise_scale", 1.0))
    if not np.isfinite(rollout_noise_scale) or rollout_noise_scale < 0.0:
        raise ValueError("hnn.rollout_noise_scale must be finite and non-negative.")
    rollout_seed_raw = hnn_cfg.get("rollout_seed", None)
    rollout_seed = None if rollout_seed_raw is None else int(rollout_seed_raw)

    if rollout_loss_mode == "stochastic":
        rollout_loss_mode = "stochastic_nll"
    if rollout_loss_mode not in {"deterministic", "stochastic_nll", "stochastic_mse"}:
        raise ValueError(
            "loss.rollout_loss_mode must be one of: deterministic, stochastic_nll, stochastic_mse."
        )
    if rollout_stochastic_samples < 1:
        raise ValueError("loss.rollout_stochastic_samples must be >= 1.")
    if rollout_loss_mode in {"stochastic_nll", "stochastic_mse"} and rollout_det_weight > 0.0:
        if not predict_sigma:
            raise ValueError("PHNN stochastic rollout loss modes require hnn.predict_sigma=true.")
        if rollout_stochastic_samples < 2:
            raise ValueError(
                "loss.rollout_stochastic_samples must be >= 2 when "
                "loss.rollout_loss_mode is stochastic_nll or stochastic_mse."
            )
    if rollout_det_steps_final < 0:
        raise ValueError("loss.rollout_det_steps_final must be non-negative.")
    if rollout_det_steps_warmup_epochs < 0:
        raise ValueError("loss.rollout_det_steps_warmup_epochs must be non-negative.")
    if rollout_det_weight > 0.0 and rollout_det_steps < 1 and rollout_det_steps_final < 1:
        raise ValueError(
            "loss.rollout_det_steps or loss.rollout_det_steps_final must be >= 1 when "
            "loss.rollout_det_weight > 0."
        )

    model_dict = asdict(model_cfg)
    first_train_traj = train_trajs[0]
    model_dict["structural_mass"] = float(np.asarray(first_train_traj["dry_mass_kg" if td_mass_source == "dry" else "effective_mass_kg"]).reshape(()))
    model_dict["k"] = float(np.asarray(first_train_traj["stiffness_n_m"]).reshape(()))
    model_dict["damping_c"] = float(np.asarray(first_train_traj["damping_c"]).reshape(()))
    model_dict["Ca"] = 0.0
    model_dict["use_stochastic_process_noise"] = predict_sigma
    model_dict["use_td_force_input"] = use_td_force_input
    arch_dict = asdict(config.architecture)
    model, _derived = PHVIV.from_config(dt=dt, cfg=model_dict, arch_cfg=arch_dict, device=device)
    _apply_td_correction_head_init(
        model,
        mode=corr_init_mode,
        tiny_std=corr_init_tiny_std,
        predict_sigma=predict_sigma,
    )
    model = maybe_compile_model(model, bool(compile_cfg.use_compile), str(compile_cfg.compile_mode))

    current_rollout_det_steps = _scheduled_rollout_det_steps(
        epoch=0,
        base_steps=rollout_det_steps,
        final_steps=rollout_det_steps_final,
        warmup_epochs=rollout_det_steps_warmup_epochs,
    )

    train_loader, val_loader, rollout_loader = _build_td_correction_hnn_loaders(
        train_trajs=train_trajs,
        val_trajs=val_trajs,
        mass_source=td_mass_source,
        batch_size=int(training_cfg.batch_size),
        rollout_batch_size=rollout_batch_size,
        rollout_steps=current_rollout_det_steps,
        num_workers=int(runtime_cfg.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    gradnorm_balancer: Optional[GradNormBalancer] = None
    if bool(getattr(loss_cfg, "use_gradnorm", False)) and (use_force_data_loss or rollout_det_weight > 0.0):
        gradnorm_loss_names = ["state"]
        if use_force_data_loss:
            gradnorm_loss_names.append("data")
        if rollout_det_weight > 0.0:
            gradnorm_loss_names.append("rollout")
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
    writer, run_name = setup_writer(
        config.logging.run_dir_root,
        config_name,
        run_name_override=getattr(config.logging, "run_name", None),
        append_timestamp=bool(getattr(config.logging, "append_timestamp", True)),
    )
    writer.add_text("hnn/td_correction_config", json.dumps(hnn_cfg, indent=2, sort_keys=True), 0)
    writer.flush()

    def _parse_td_train_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(batch) != 10:
            raise ValueError("Unexpected TD correction HNN batch format.")
        z_i, t_i, z_next, t_next, ur_i, corr_next, td_force_next, mass_i, damping_i, stiffness_i = batch
        return z_i, t_i, z_next, t_next, ur_i, corr_next, td_force_next, mass_i, damping_i, stiffness_i

    def _state_loss(
        *,
        z_i: torch.Tensor,
        dt_i: torch.Tensor,
        z_next: torch.Tensor,
        ur_i: torch.Tensor,
        td_force_next: torch.Tensor,
        mass_i: torch.Tensor,
        damping_i: torch.Tensor,
        stiffness_i: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        corr_mu, sigma_corr = _td_predict_correction(
            model,
            z=z_i,
            reduced_velocity=ur_i,
            td_force_input=td_force_next,
            structural_mass=mass_i,
            stiffness=stiffness_i,
            predict_sigma=predict_sigma,
            force_zero_output=force_zero_output,
        )
        total_force_next = td_force_next + corr_mu
        if predict_sigma and state_loss_mode == "propagated_nll":
            state_loss, _z_next_mean = _td_state_propagated_nll_loss(
                z_i=z_i,
                dt_i=dt_i,
                z_next=z_next,
                total_force_next=total_force_next,
                sigma_corr=sigma_corr,
                mass_i=mass_i,
                damping_i=damping_i,
                stiffness_i=stiffness_i,
            )
        else:
            state_loss, _z_next_mean = _td_state_mse_loss(
                z_i=z_i,
                dt_i=dt_i,
                z_next=z_next,
                total_force_next=total_force_next,
                mass_i=mass_i,
                damping_i=damping_i,
                stiffness_i=stiffness_i,
            )
        return state_loss, corr_mu, sigma_corr

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
    validation_samples_per_ur = max(1, int(getattr(monitoring_cfg, "validation_samples_per_ur", 1)))
    final_rollout_all_validation = bool(getattr(monitoring_cfg, "final_rollout_all_validation", False))
    train_instances = len(train_loader.dataset)
    train_steps_per_epoch = len(train_loader)
    val_instances = len(val_loader.dataset) if val_loader is not None else 0
    val_steps_per_epoch = len(val_loader) if val_loader is not None else 0
    train_rollout_instances = len(rollout_loader.dataset) if rollout_loader is not None else 0
    train_rollout_steps_per_epoch = len(rollout_loader) if rollout_loader is not None else 0

    startup_lines = [
        f"Run name: {run_name}",
        (
            f"HNN TD-correction setup: epochs={epochs}, batch_size={int(training_cfg.batch_size)}, "
            f"steps_per_epoch={train_steps_per_epoch}, train_instances={train_instances}, "
            f"train_trajectories={len(train_trajs)}"
        ),
        (
            f"Validation setup: steps={val_steps_per_epoch}, val_instances={val_instances}, "
            f"val_trajectories={len(val_trajs)}, validate_every={validate_every}, "
            f"val_samples_per_ur={validation_samples_per_ur}"
        ),
        (
            f"Rollout setup: weight={rollout_det_weight:g}, steps={current_rollout_det_steps}, "
            f"train_rollout_windows={train_rollout_instances}, train_rollout_steps={train_rollout_steps_per_epoch}"
        ),
        (
            f"Runtime: device={device}, num_workers={int(runtime_cfg.num_workers)}, amp={amp_enabled}, "
            f"compile={bool(compile_cfg.use_compile)}, lr={float(optim_cfg.lr):g}"
        ),
    ]
    sigma_identified_by_rollout = (
        rollout_det_weight > 0.0 and rollout_loss_mode in {"stochastic_nll", "stochastic_mse"}
    )
    if predict_sigma and not use_force_data_loss and not sigma_identified_by_rollout:
        startup_lines.append(
            "Warning: hnn.predict_sigma=true with loss.use_force_data_loss=false. "
            "Sigma is treated as correction-force uncertainty, and no stochastic rollout loss is enabled."
        )
    if rollout_det_weight > 0.0:
        rollout_mode_msg = rollout_loss_mode
        if rollout_loss_mode in {"stochastic_nll", "stochastic_mse"}:
            rollout_mode_msg = f"{rollout_loss_mode} (K={rollout_stochastic_samples})"
        startup_lines.append(
            f"Rollout loss mode: {rollout_mode_msg}, state_loss_mode={state_loss_mode}"
        )
    print("\n".join(startup_lines))

    def _rebuild_td_rollout_loader(steps: int) -> Any | None:
        if steps <= 0 or rollout_det_weight <= 0.0:
            return None
        _train_loader_tmp, _val_loader_tmp, rollout_loader_tmp = _build_td_correction_hnn_loaders(
            train_trajs=train_trajs,
            val_trajs=val_trajs,
            mass_source=td_mass_source,
            batch_size=int(training_cfg.batch_size),
            rollout_batch_size=rollout_batch_size,
            rollout_steps=steps,
            num_workers=int(runtime_cfg.num_workers),
            pin_memory=(device.type == "cuda"),
        )
        del _train_loader_tmp
        del _val_loader_tmp
        return rollout_loader_tmp

    for epoch in range(epochs):
        scheduled_rollout_det_steps = _scheduled_rollout_det_steps(
            epoch=epoch,
            base_steps=rollout_det_steps,
            final_steps=rollout_det_steps_final,
            warmup_epochs=rollout_det_steps_warmup_epochs,
        )
        if scheduled_rollout_det_steps != current_rollout_det_steps:
            current_rollout_det_steps = scheduled_rollout_det_steps
            rollout_loader = _rebuild_td_rollout_loader(current_rollout_det_steps)
        model.train()
        if bool(optim_cfg.use_lr_scheduler):
            for group in opt.param_groups:
                group["lr"] = lr_scheduler.get_lr(epoch)
        sums = {
            "loss_total": torch.zeros((), device=device),
            "loss_state": torch.zeros((), device=device),
            "loss_data": torch.zeros((), device=device),
            "loss_reg_mean": torch.zeros((), device=device),
            "loss_reg_sigma": torch.zeros((), device=device),
            "loss_rollout_det": torch.zeros((), device=device),
            "grad_norm": torch.zeros((), device=device),
        }
        gradnorm_state_w_sum = torch.zeros((), device=device) if gradnorm_balancer is not None else None
        gradnorm_data_w_sum = torch.zeros((), device=device) if gradnorm_balancer is not None and use_force_data_loss else None
        gradnorm_rollout_w_sum = (
            torch.zeros((), device=device) if gradnorm_balancer is not None and rollout_det_weight > 0.0 else None
        )
        gradnorm_count = 0
        batch_count = 0
        rollout_iter = iter(rollout_loader) if rollout_loader is not None and rollout_det_weight > 0.0 else None
        for batch in train_loader:
            z_i, t_i, z_next, t_next, ur_i, corr_next, td_force_next, mass_i, damping_i, stiffness_i = _parse_td_train_batch(batch)
            z_i = z_i.to(device, non_blocking=non_blocking)
            t_i = t_i.to(device, non_blocking=non_blocking)
            z_next = z_next.to(device, non_blocking=non_blocking)
            t_next = t_next.to(device, non_blocking=non_blocking)
            ur_i = ur_i.to(device, non_blocking=non_blocking)
            corr_next = corr_next.to(device, non_blocking=non_blocking)
            td_force_next = td_force_next.to(device, non_blocking=non_blocking)
            mass_i = mass_i.to(device, non_blocking=non_blocking)
            damping_i = damping_i.to(device, non_blocking=non_blocking)
            stiffness_i = stiffness_i.to(device, non_blocking=non_blocking)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                dt_i = torch.clamp(t_next - t_i, min=1.0e-12)
                state_loss, corr_mu, sigma_corr = _state_loss(
                    z_i=z_i,
                    dt_i=dt_i,
                    z_next=z_next,
                    ur_i=ur_i,
                    td_force_next=td_force_next,
                    mass_i=mass_i,
                    damping_i=damping_i,
                    stiffness_i=stiffness_i,
                )
                if use_force_data_loss:
                    if predict_sigma:
                        var = torch.clamp(sigma_corr * sigma_corr, min=1e-9)
                        data_loss = torch.mean(0.5 * (((corr_next - corr_mu) ** 2) / var + torch.log(var)))
                    else:
                        data_loss = torch.mean((corr_next - corr_mu) ** 2)
                else:
                    data_loss = state_loss.new_tensor(0.0)
                mean_reg_loss = _regularizer(corr_mu, mean_reg_norm)
                sigma_reg_loss = _regularizer(sigma_corr, sigma_reg_norm) if predict_sigma else state_loss.new_tensor(0.0)
                rollout_det_loss = state_loss.new_tensor(0.0)
                if rollout_iter is not None:
                    try:
                        rollout_batch = next(rollout_iter)
                    except StopIteration:
                        rollout_iter = iter(rollout_loader)
                        rollout_batch = next(rollout_iter)
                    rollout_det_loss = _td_correction_rollout_loss_from_batch(
                        model=model,
                        batch=rollout_batch,
                        device=device,
                        non_blocking=non_blocking,
                        td_params=td_params,
                        td_memory_cfg=td_memory_cfg,
                        predict_sigma=predict_sigma,
                        force_zero_output=force_zero_output,
                        rollout_loss_mode=rollout_loss_mode,
                        rollout_stochastic_samples=rollout_stochastic_samples,
                        rollout_noise_scale=rollout_noise_scale,
                    )
                if gradnorm_balancer is not None:
                    loss_inputs: dict[str, torch.Tensor] = {"state": state_loss.float()}
                    if use_force_data_loss:
                        loss_inputs["data"] = data_loss.float()
                    if rollout_det_weight > 0.0:
                        loss_inputs["rollout"] = rollout_det_loss.float()
                    weights = gradnorm_balancer.update(loss_inputs)
                    state_w = weights["state"]
                    base_loss = state_w * state_loss
                    if gradnorm_state_w_sum is not None:
                        gradnorm_state_w_sum = gradnorm_state_w_sum + state_w.detach()
                    if use_force_data_loss:
                        data_w = weights["data"]
                        base_loss = base_loss + float(force_data_weight) * data_w * data_loss
                        if gradnorm_data_w_sum is not None:
                            gradnorm_data_w_sum = gradnorm_data_w_sum + data_w.detach()
                    if rollout_det_weight > 0.0:
                        rollout_w = weights["rollout"]
                        if gradnorm_rollout_w_sum is not None:
                            gradnorm_rollout_w_sum = gradnorm_rollout_w_sum + rollout_w.detach()
                        weighted_rollout_det = float(rollout_det_weight) * rollout_w * rollout_det_loss
                    else:
                        weighted_rollout_det = float(rollout_det_weight) * rollout_det_loss
                    gradnorm_count += 1
                else:
                    base_loss = state_loss + float(force_data_weight) * data_loss
                    weighted_rollout_det = float(rollout_det_weight) * rollout_det_loss
                total_loss = base_loss + float(mean_reg) * mean_reg_loss + float(sigma_reg) * sigma_reg_loss + weighted_rollout_det
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
            sums["loss_data"] += data_loss.detach()
            sums["loss_reg_mean"] += mean_reg_loss.detach()
            sums["loss_reg_sigma"] += sigma_reg_loss.detach()
            sums["loss_rollout_det"] += rollout_det_loss.detach()
            sums["grad_norm"] += grad_norm.detach() if isinstance(grad_norm, torch.Tensor) else torch.tensor(float(grad_norm), device=device)

        denom = float(max(1, batch_count))
        train_metrics = {name: float((value / denom).detach().cpu()) for name, value in sums.items()}
        train_metrics["lr"] = float(opt.param_groups[0]["lr"]) if opt.param_groups else float(optim_cfg.lr)
        if gradnorm_count > 0:
            if gradnorm_state_w_sum is not None:
                train_metrics["gradnorm_weight_physics"] = float(
                    (gradnorm_state_w_sum / float(gradnorm_count)).detach().cpu()
                )
            if gradnorm_data_w_sum is not None:
                train_metrics["gradnorm_weight_data"] = float(
                    (gradnorm_data_w_sum / float(gradnorm_count)).detach().cpu()
                )
            if gradnorm_rollout_w_sum is not None:
                train_metrics["gradnorm_weight_rollout"] = float(
                    (gradnorm_rollout_w_sum / float(gradnorm_count)).detach().cpu()
                )
        if epoch % log_every == 0 or epoch == epochs - 1:
            for name, value in train_metrics.items():
                writer.add_scalar(f"train/{name}", value, epoch + 1)
        if epoch % print_every == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch}: loss={train_metrics['loss_total']:.4e}, "
                f"Lstate={train_metrics['loss_state']:.4e}, Ldata={train_metrics['loss_data']:.4e}, "
                f"Lroll={train_metrics['loss_rollout_det']:.4e}, lr={train_metrics['lr']:.3e}"
            )

        if val_loader is not None and ((epoch % validate_every) == 0 or epoch == epochs - 1):
            model.eval()
            val_sums = {
                "loss_total": torch.zeros((), device=device),
                "loss_state": torch.zeros((), device=device),
                "loss_data": torch.zeros((), device=device),
                "loss_reg_mean": torch.zeros((), device=device),
                "loss_reg_sigma": torch.zeros((), device=device),
            }
            val_count = 0
            with torch.no_grad():
                for batch in val_loader:
                    z_i, t_i, z_next, t_next, ur_i, corr_next, td_force_next, mass_i, damping_i, stiffness_i = _parse_td_train_batch(batch)
                    z_i = z_i.to(device, non_blocking=non_blocking)
                    t_i = t_i.to(device, non_blocking=non_blocking)
                    z_next = z_next.to(device, non_blocking=non_blocking)
                    t_next = t_next.to(device, non_blocking=non_blocking)
                    ur_i = ur_i.to(device, non_blocking=non_blocking)
                    corr_next = corr_next.to(device, non_blocking=non_blocking)
                    td_force_next = td_force_next.to(device, non_blocking=non_blocking)
                    mass_i = mass_i.to(device, non_blocking=non_blocking)
                    damping_i = damping_i.to(device, non_blocking=non_blocking)
                    stiffness_i = stiffness_i.to(device, non_blocking=non_blocking)
                    with torch.amp.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                        dt_i = torch.clamp(t_next - t_i, min=1.0e-12)
                        state_loss, corr_mu, sigma_corr = _state_loss(
                            z_i=z_i,
                            dt_i=dt_i,
                            z_next=z_next,
                            ur_i=ur_i,
                            td_force_next=td_force_next,
                            mass_i=mass_i,
                            damping_i=damping_i,
                            stiffness_i=stiffness_i,
                        )
                        if use_force_data_loss:
                            if predict_sigma:
                                var = torch.clamp(sigma_corr * sigma_corr, min=1e-9)
                                data_loss = torch.mean(0.5 * (((corr_next - corr_mu) ** 2) / var + torch.log(var)))
                            else:
                                data_loss = torch.mean((corr_next - corr_mu) ** 2)
                        else:
                            data_loss = state_loss.new_tensor(0.0)
                        mean_reg_loss = _regularizer(corr_mu, mean_reg_norm)
                        sigma_reg_loss = _regularizer(sigma_corr, sigma_reg_norm) if predict_sigma else state_loss.new_tensor(0.0)
                        total_loss = state_loss + float(force_data_weight) * data_loss + float(mean_reg) * mean_reg_loss + float(sigma_reg) * sigma_reg_loss
                    val_sums["loss_total"] += total_loss.detach()
                    val_sums["loss_state"] += state_loss.detach()
                    val_sums["loss_data"] += data_loss.detach()
                    val_sums["loss_reg_mean"] += mean_reg_loss.detach()
                    val_sums["loss_reg_sigma"] += sigma_reg_loss.detach()
                    val_count += 1
            val_denom = float(max(1, val_count))
            val_metrics = {
                name: float((value / val_denom).detach().cpu()) for name, value in val_sums.items()
            }
            rollout_loss_avg = 0.0
            if rollout_loader is not None and rollout_det_weight > 0.0:
                with torch.no_grad():
                    rollout_loss_sum = torch.zeros((), device=device)
                    rollout_count = 0
                    for rollout_batch in rollout_loader:
                        rollout_loss_sum += _td_correction_rollout_loss_from_batch(
                            model=model,
                            batch=rollout_batch,
                            device=device,
                            non_blocking=non_blocking,
                            td_params=td_params,
                            td_memory_cfg=td_memory_cfg,
                            predict_sigma=predict_sigma,
                            force_zero_output=force_zero_output,
                            rollout_loss_mode=rollout_loss_mode,
                            rollout_stochastic_samples=rollout_stochastic_samples,
                            rollout_noise_scale=rollout_noise_scale,
                        ).detach()
                        rollout_count += 1
                    rollout_loss_avg = float((rollout_loss_sum / float(max(1, rollout_count))).detach().cpu())
                    writer.add_scalar("val/loss_rollout_det", rollout_loss_avg, epoch + 1)
            val_metrics["loss_total"] = (
                val_metrics["loss_state"]
                + float(force_data_weight) * val_metrics["loss_data"]
                + float(mean_reg) * val_metrics["loss_reg_mean"]
                + float(sigma_reg) * val_metrics["loss_reg_sigma"]
                + float(rollout_det_weight) * rollout_loss_avg
            )
            for name, value in val_metrics.items():
                writer.add_scalar(f"val/{name}", value, epoch + 1)

            if val_trajs and ((epoch % validate_every) == 0 or epoch == epochs - 1):
                ur_all = [float(np.asarray(traj["ur"]).reshape(-1)[0]) for traj in val_trajs]
                sampled_metric_indices = sample_indices_per_ur(
                    ur_all,
                    samples_per_ur=validation_samples_per_ur,
                    seed=1,
                )
                sampled_names = [str(val_trajs[idx].get("name", f"traj_{idx}")) for idx in sampled_metric_indices]
                print(
                    f"[td-val][phnn] epoch {epoch + 1}: sampled metric trajectories={sampled_names} "
                    f"(force_zero_output={force_zero_output}, mass_source={td_mass_source})"
                )
                metrics_sum: dict[str, float] = {}
                metrics_count: dict[str, int] = {}
                diverged_count = 0
                for sidx in sampled_metric_indices:
                    metrics_roll = _log_td_correction_rollout_validation(
                        writer=writer,
                        epoch=epoch + 1,
                        model=model,
                        traj=val_trajs[sidx],
                        dt=dt,
                        td_mass_source=td_mass_source,
                        td_params=td_params,
                        td_memory_cfg=td_memory_cfg,
                        device=device,
                        predict_sigma=predict_sigma,
                        force_zero_output=force_zero_output,
                        rollout_stochastic=rollout_stochastic,
                        rollout_noise_scale=rollout_noise_scale,
                        rollout_seed=rollout_seed,
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
                    writer.add_scalar(f"val/{name}", total / float(max(1, metrics_count.get(name, 0))), epoch + 1)
                writer.add_scalar(f"val/{ROLLOUT_DIVERGED_COUNT_KEY}", float(diverged_count), epoch + 1)

                selected_indices = sample_one_index_per_ur(ur_all, seed=0)
                if not selected_indices:
                    selected_indices = list(range(len(val_trajs)))
                rollout_idx = selected_indices[0]
                rollout_traj = val_trajs[rollout_idx]
                rollout_dt = float(np.asarray(rollout_traj["t"])[1] - np.asarray(rollout_traj["t"])[0])
                print(
                    f"[td-val][phnn] epoch {epoch + 1}: plot trajectory={rollout_traj.get('name', f'traj_{rollout_idx}')} "
                    f"U_r={float(np.asarray(rollout_traj['ur']).reshape(-1)[0]):.6g} "
                    f"dt={rollout_dt:.6g} rho={float(model.rho):.6g} D={float(model.D):.6g} "
                    f"m={float(np.asarray(rollout_traj['dry_mass_kg' if td_mass_source == 'dry' else 'effective_mass_kg']).reshape(())):.6g} "
                    f"c={float(np.asarray(rollout_traj['damping_c']).reshape(())):.6g} "
                    f"k={float(np.asarray(rollout_traj['stiffness_n_m']).reshape(())):.6g}"
                )
                _log_td_correction_rollout_validation(
                    writer=writer,
                    epoch=epoch + 1,
                    model=model,
                    traj=rollout_traj,
                    dt=dt,
                    td_mass_source=td_mass_source,
                    td_params=td_params,
                    td_memory_cfg=td_memory_cfg,
                    device=device,
                    predict_sigma=predict_sigma,
                    force_zero_output=force_zero_output,
                    rollout_stochastic=rollout_stochastic,
                    rollout_noise_scale=rollout_noise_scale,
                    rollout_seed=rollout_seed,
                    log_metrics=False,
                    log_plots=True,
                )

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
                td_params=td_params,
                td_memory_cfg=td_memory_cfg,
                device=device,
                predict_sigma=predict_sigma,
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
            mass_key = "dry_mass_kg" if str(td_mass_source).strip().lower() == "dry" else "effective_mass_kg"
            mass_value = float(np.asarray(traj[mass_key]).reshape(()))
            stiffness_value = float(np.asarray(traj["stiffness_n_m"]).reshape(()))
            y_true_t = torch.from_numpy(np.ascontiguousarray(traj["y"])).float().unsqueeze(1).to(device)
            v_true_t = torch.from_numpy(np.ascontiguousarray(traj["dy"])).float().unsqueeze(1).to(device)
            z_true_t = torch.cat([y_true_t, v_true_t * mass_value], dim=1)
            td_force_t = torch.from_numpy(np.ascontiguousarray(traj["force_td_per_m"])).float().unsqueeze(1).to(device)
            ur_t = torch.from_numpy(np.ascontiguousarray(traj["ur"])).float().unsqueeze(1).to(device)
            with torch.no_grad():
                corr_on_data, sigma_on_data = _td_predict_correction(
                    model,
                    z=z_true_t,
                    reduced_velocity=ur_t,
                    td_force_input=td_force_t,
                    structural_mass=torch.full_like(y_true_t, mass_value),
                    stiffness=torch.full_like(y_true_t, stiffness_value),
                    predict_sigma=predict_sigma,
                    force_zero_output=force_zero_output,
                )
            output_ur_values.append(ur_val)
            corr_series_list.append(corr_on_data[:, 0].detach().cpu().numpy())
            if predict_sigma:
                sigma_series_list.append(sigma_on_data[:, 0].detach().cpu().numpy())

            plot_metrics = _log_td_correction_rollout_validation(
                writer=writer,
                epoch=max(0, epochs - 1),
                model=model,
                traj=traj,
                dt=dt,
                td_mass_source=td_mass_source,
                td_params=td_params,
                td_memory_cfg=td_memory_cfg,
                device=device,
                predict_sigma=predict_sigma,
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
        if output_ur_values and corr_series_list:
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

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{run_name}.pt"
    state_source: torch.nn.Module = model
    if hasattr(model, "_orig_mod"):
        state_source = getattr(model, "_orig_mod")
    torch.save(
        {
            "model_state": state_source.state_dict(),
            "config": asdict(config),
            "run_name": run_name,
            "dt": dt,
            "method": "phnn",
            "td_correction": True,
        },
        model_path,
    )
    print(f"Saved final model to {model_path}")
    writer.flush()
    writer.close()


def train(config: Config, config_name: str) -> None:
    hnn_cfg = dict(config.hnn or {})
    if "use_td_correction" in hnn_cfg and not bool(hnn_cfg.get("use_td_correction", True)):
        raise ValueError("PHNN now only supports TD-correction training. Remove hnn.use_td_correction or set it to true.")
    _train_td_correction(config, config_name)
    return

    data_cfg = config.data
    middle_time_plot = data_cfg.middle_time_plot
    train_series_root = Path(data_cfg.train_series_dir)
    train_dir = train_series_root / "train"
    val_dir = train_series_root / "val"
    if not val_dir.exists():
        raise FileNotFoundError(
            f"Expected validation data in '{val_dir}'. data.npz is no longer used for validation."
        )
    if not train_dir.exists():
        raise FileNotFoundError(f"Expected training data in '{train_dir}'.")
    val_files = sorted(val_dir.glob("*.npz"))
    if not val_files:
        raise FileNotFoundError(f"No '.npz' files found in validation directory '{val_dir}'.")
    data_path = val_files[0]
    train_series_dir = train_dir

    data = np.load(data_path)
    t = data["a"]
    y_data = data["b"]
    has_force_data = "c" in data
    F_data = data["c"] if has_force_data else np.zeros_like(y_data)
    H_data = data["d"] if "d" in data else None
    if "U_r" not in data:
        raise KeyError(f"{data_path} is missing reduced velocity 'U_r'.")
    reduced_velocity = data["U_r"]
    vel_data = None
    for key in ("e", "dy", "v"):
        if key in data:
            vel_data = data[key]
            break

    train_cut = resolve_cut_start_seconds(data_cfg, "train")
    t, y_data, F_data, hamiltonian_data, vel_data, dt = preprocess_timeseries(
        t,
        y_data,
        F_data,
        H_data,
        data_cfg,
        velocity=vel_data,
        cut_start_seconds=train_cut,
    )

    model_cfg = config.model
    hnn_cfg = dict(config.hnn or {})
    rollout_stochastic = bool(hnn_cfg.get("rollout_stochastic", False))
    rollout_noise_scale = float(hnn_cfg.get("rollout_noise_scale", 1.0))
    if not np.isfinite(rollout_noise_scale) or rollout_noise_scale < 0.0:
        raise ValueError("hnn.rollout_noise_scale must be finite and non-negative.")
    rollout_seed_raw = hnn_cfg.get("rollout_seed", None)
    rollout_seed = None if rollout_seed_raw is None else int(rollout_seed_raw)
    train_include_ur = _as_float_list(hnn_cfg.get("train_include_ur"), key="hnn.train_include_ur")
    train_exclude_ur = _as_float_list(hnn_cfg.get("train_exclude_ur"), key="hnn.train_exclude_ur")
    train_ur_filter_tol = float(hnn_cfg.get("train_ur_filter_tol", 1e-6))
    if train_ur_filter_tol < 0.0:
        raise ValueError("hnn.train_ur_filter_tol must be non-negative.")
    if train_include_ur is not None or train_exclude_ur is not None:
        print(
            "Applying training U_r filter: "
            f"include={train_include_ur}, exclude={train_exclude_ur}, tol={train_ur_filter_tol:g}"
        )

    training_cfg = config.training
    optim_cfg = config.optim
    loss_cfg = config.loss
    runtime_cfg = config.runtime
    precision_cfg = config.precision
    compile_cfg = config.compile
    monitoring_cfg = config.monitoring

    batch_size = int(training_cfg.batch_size)
    max_grad_norm = float(training_cfg.max_grad_norm)
    epochs = int(training_cfg.epochs)

    lr = float(optim_cfg.lr)
    use_lr_scheduler = bool(optim_cfg.use_lr_scheduler)
    scheduler_cfg = optim_cfg.scheduler

    sigma_reg = float(loss_cfg.sigma_reg)
    mean_reg = float(getattr(loss_cfg, "mean_reg", 0.0))
    mean_reg_norm = str(getattr(loss_cfg, "mean_reg_norm", "l1")).strip().lower()
    sigma_reg_norm = str(getattr(loss_cfg, "sigma_reg_norm", "l2")).strip().lower()
    rollout_det_weight = float(getattr(loss_cfg, "rollout_det_weight", 0.0))
    rollout_det_steps = int(getattr(loss_cfg, "rollout_det_steps", 0))
    rollout_loss_mode = str(getattr(loss_cfg, "rollout_loss_mode", "deterministic")).strip().lower()
    rollout_stochastic_samples = int(getattr(loss_cfg, "rollout_stochastic_samples", 1))
    ur_bin_size = float(getattr(loss_cfg, "ur_bin_size", 1e-6))
    normalize_by_ur_bin_std = bool(getattr(loss_cfg, "normalize_by_ur_bin_std", False))
    ur_bin_scale_eps = float(getattr(loss_cfg, "ur_bin_scale_eps", 1e-6))
    rollout_det_steps_final_raw = int(getattr(loss_cfg, "rollout_det_steps_final", 0))
    rollout_det_steps_warmup_epochs = int(getattr(loss_cfg, "rollout_det_steps_warmup_epochs", 0))
    rollout_det_steps_final = rollout_det_steps if rollout_det_steps_final_raw <= 0 else rollout_det_steps_final_raw
    rollout_det_batch_size_raw = int(getattr(loss_cfg, "rollout_det_batch_size", 0))
    rollout_det_batch_size = batch_size if rollout_det_batch_size_raw <= 0 else rollout_det_batch_size_raw
    use_force_data_loss = bool(getattr(loss_cfg, "use_force_data_loss", False))
    force_data_weight = float(getattr(loss_cfg, "force_data_weight", 1.0))
    symmetry_weight = float(getattr(loss_cfg, "symmetry_weight", 0.0))
    symmetry_norm = str(getattr(loss_cfg, "symmetry_norm", "l2")).strip().lower()
    if symmetry_norm not in {"l1", "l2"}:
        raise ValueError("loss.symmetry_norm must be one of: l1, l2.")
    if mean_reg_norm not in {"l1", "l2"}:
        raise ValueError("loss.mean_reg_norm must be one of: l1, l2.")
    if sigma_reg_norm not in {"l1", "l2"}:
        raise ValueError("loss.sigma_reg_norm must be one of: l1, l2.")
    if rollout_det_weight < 0.0:
        raise ValueError("loss.rollout_det_weight must be non-negative.")
    if rollout_det_steps < 0:
        raise ValueError("loss.rollout_det_steps must be non-negative.")
    if rollout_loss_mode == "stochastic":
        rollout_loss_mode = "stochastic_nll"
    if rollout_loss_mode not in {"deterministic", "stochastic_nll", "stochastic_mse"}:
        raise ValueError(
            "loss.rollout_loss_mode must be one of: deterministic, stochastic_nll, stochastic_mse."
        )
    if rollout_stochastic_samples < 1:
        raise ValueError("loss.rollout_stochastic_samples must be >= 1.")
    if not np.isfinite(ur_bin_size) or ur_bin_size <= 0.0:
        raise ValueError("loss.ur_bin_size must be finite and > 0.")
    if not np.isfinite(ur_bin_scale_eps) or ur_bin_scale_eps <= 0.0:
        raise ValueError("loss.ur_bin_scale_eps must be finite and > 0.")
    if rollout_loss_mode in {"stochastic_nll", "stochastic_mse"} and rollout_det_weight > 0.0 and rollout_stochastic_samples < 2:
        raise ValueError(
            "loss.rollout_stochastic_samples must be >= 2 when "
            "loss.rollout_loss_mode is stochastic_nll or stochastic_mse."
        )
    if rollout_det_steps_final < 0:
        raise ValueError("loss.rollout_det_steps_final must be non-negative.")
    if rollout_det_steps_warmup_epochs < 0:
        raise ValueError("loss.rollout_det_steps_warmup_epochs must be non-negative.")
    if rollout_det_weight > 0.0 and rollout_det_steps < 1:
        if rollout_det_steps_final < 1:
            raise ValueError(
                "loss.rollout_det_steps or loss.rollout_det_steps_final must be >= 1 when "
                "loss.rollout_det_weight > 0."
            )
    if rollout_det_batch_size < 1:
        raise ValueError("loss.rollout_det_batch_size must be >= 1 after fallback resolution.")

    validate_every_epochs = max(1, int(getattr(monitoring_cfg, "validate_every_epochs", 10)))
    validation_samples_per_ur = max(1, int(getattr(monitoring_cfg, "validation_samples_per_ur", 1)))
    log_every_epochs = max(1, int(monitoring_cfg.log_every_epochs))
    print_every_epochs = max(1, int(monitoring_cfg.print_every_epochs))
    log_component_grad_norms = bool(monitoring_cfg.log_component_grad_norms)
    log_extra_validation_metrics = bool(getattr(monitoring_cfg, "log_extra_validation_metrics", False))
    final_rollout_all_validation = bool(getattr(monitoring_cfg, "final_rollout_all_validation", False))
    async_validation = bool(getattr(monitoring_cfg, "async_validation", False))
    async_device = str(getattr(monitoring_cfg, "async_validation_device", "cpu"))
    async_num_workers = int(getattr(monitoring_cfg, "async_validation_num_workers", 0))
    async_num_threads = int(getattr(monitoring_cfg, "async_validation_num_threads", 4))
    async_max_concurrent = int(getattr(monitoring_cfg, "async_validation_max_concurrent", 1))

    device = select_device(os.getenv("TRAIN_DEVICE", str(runtime_cfg.device)))
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA available: {torch.cuda.is_available()}, gpu0: {torch.cuda.get_device_name(0)}")
    configure_tf32(device, bool(precision_cfg.use_tf32))
    set_num_threads_from_slurm(default=1)
    non_blocking = device.type == "cuda"

    model_dict = asdict(model_cfg)
    arch_dict = asdict(config.architecture)
    model, derived_params = PHVIV.from_config(dt=dt, cfg=model_dict, arch_cfg=arch_dict, device=device)
    model = maybe_compile_model(model, bool(compile_cfg.use_compile), str(compile_cfg.compile_mode))
    D = derived_params["D"]
    k = derived_params["k"]
    m_eff = derived_params["m_eff"]

    train_series_raw, eval_tensors = load_training_series(
        y_data,
        t,
        dt,
        train_series_dir,
        m_eff,
        device,
        eval_velocity=vel_data,
        eval_reduced_velocity=reduced_velocity,
        require_force=use_force_data_loss,
        eval_force=(F_data if has_force_data else None),
        cut_start_seconds=train_cut,
        include_reduced_velocity=train_include_ur,
        exclude_reduced_velocity=train_exclude_ur,
        ur_filter_tol=train_ur_filter_tol,
    )
    eval_y_tensor, eval_vel_tensor, eval_t_tensor, eval_ur_tensor = eval_tensors

    pin_memory = device.type == "cuda"
    num_workers = int(runtime_cfg.num_workers)
    train_loader, train_sequences, _ = build_dataloader_from_series(
        train_series_raw,
        m_eff=m_eff,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    ur_bin_state_scale_info: dict[str, Any] | None = None
    if normalize_by_ur_bin_std:
        ur_bin_state_scale_info = build_ur_bin_state_scale_info_from_dataset(
            train_loader.dataset,
            ur_tensor_index=4,
            state_tensor_indices=(0, 2),
            ur_bin_size=ur_bin_size,
            eps=ur_bin_scale_eps,
        )

    val_series_raw: list[tuple[np.ndarray, np.ndarray, float, np.ndarray | None, np.ndarray | None, np.ndarray]] | None = None
    val_sequences: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] | None = None
    val_loader: Any | None = None
    if val_dir.exists():
        val_cut = resolve_cut_start_seconds(data_cfg, "val")
        val_require_force = bool(use_force_data_loss or has_force_data)
        val_series_raw, _ = load_training_series(
            y_data,
            t,
            dt,
            val_dir,
            m_eff,
            device,
            eval_velocity=vel_data,
            eval_reduced_velocity=reduced_velocity,
            require_force=val_require_force,
            eval_force=(F_data if has_force_data else None),
            cut_start_seconds=val_cut,
        )
        val_loader, val_sequences, _ = build_dataloader_from_series(
            val_series_raw,
            m_eff=m_eff,
            batch_size=batch_size,
            device=device,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    train_rollout_loader: Any | None = None
    val_rollout_loader: Any | None = None
    current_rollout_det_steps = _scheduled_rollout_det_steps(
        epoch=0,
        base_steps=rollout_det_steps,
        final_steps=rollout_det_steps_final,
        warmup_epochs=rollout_det_steps_warmup_epochs,
    )

    def _rebuild_rollout_loaders(steps: int) -> tuple[Any | None, Any | None, int]:
        train_loader_out: Any | None = None
        val_loader_out: Any | None = None
        train_windows = 0
        if rollout_det_weight > 0.0 and steps > 0:
            train_loader_out, train_windows = build_rollout_dataloader_from_series(
                train_series_raw,
                m_eff=m_eff,
                batch_size=rollout_det_batch_size,
                device=device,
                rollout_steps=steps,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            if val_series_raw is not None:
                val_loader_out, _ = build_rollout_dataloader_from_series(
                    val_series_raw,
                    m_eff=m_eff,
                    batch_size=rollout_det_batch_size,
                    device=device,
                    rollout_steps=steps,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                )
        return train_loader_out, val_loader_out, int(train_windows)

    if rollout_det_weight > 0.0:
        train_rollout_loader, val_rollout_loader, train_rollout_windows = _rebuild_rollout_loaders(current_rollout_det_steps)
        rollout_mode_msg = rollout_loss_mode
        if rollout_loss_mode in {"stochastic_nll", "stochastic_mse"}:
            rollout_mode_msg = f"{rollout_loss_mode} (K={rollout_stochastic_samples})"
        if rollout_det_steps_final > rollout_det_steps and rollout_det_steps_warmup_epochs > 0:
            print(
                "Enabled rollout loss schedule: "
                f"mode={rollout_mode_msg}, "
                f"steps={rollout_det_steps}->{rollout_det_steps_final} over "
                f"{rollout_det_steps_warmup_epochs} epoch(s), "
                f"current_steps={current_rollout_det_steps}, "
                f"windows={train_rollout_windows}, rollout_batch_size={rollout_det_batch_size}"
            )
        else:
            print(
                f"Enabled rollout loss: mode={rollout_mode_msg}, steps={current_rollout_det_steps}, "
                f"weight={rollout_det_weight:g}, windows={train_rollout_windows}, "
                f"rollout_batch_size={rollout_det_batch_size}"
            )

    y_data_t, val_vel, _t_tensor, val_ur = eval_y_tensor, eval_vel_tensor, eval_t_tensor, eval_ur_tensor

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

    y_true_norm = y_data / D
    force_data = F_data if has_force_data else None

    opt, lr_scheduler = setup_optimizer_and_scheduler(
        model,
        optim_cfg=optim_cfg,
        scheduler_cfg=scheduler_cfg,
        epochs=epochs,
    )

    gradnorm_balancer: Optional[GradNormBalancer] = None
    if bool(loss_cfg.use_gradnorm):
        names = ["residual"]
        if use_force_data_loss and force_data_weight > 0.0:
            names.append("data")
        if symmetry_weight > 0.0:
            names.append("symmetry")
        if len(names) >= 2:
            gradnorm_balancer = GradNormBalancer(
                model,
                names,
                alpha=float(loss_cfg.gradnorm_alpha),
                eps=float(loss_cfg.gradnorm_eps),
                min_weight=float(loss_cfg.gradnorm_min_weight),
                max_weight=float(loss_cfg.gradnorm_max_weight),
            )
        else:
            print("loss.use_gradnorm is True but fewer than two differentiable HNN losses are active; skipping GradNorm.")

    amp_enabled, amp_dtype, scaler = setup_amp(
        device, use_amp=bool(precision_cfg.use_amp), amp_dtype=str(precision_cfg.amp_dtype)
    )

    train_instances = len(train_loader.dataset)
    train_steps_per_epoch = len(train_loader)
    val_instances = len(val_loader.dataset) if val_loader is not None else 0
    val_steps_per_epoch = len(val_loader) if val_loader is not None else 0
    train_rollout_instances = len(train_rollout_loader.dataset) if train_rollout_loader is not None else 0
    train_rollout_steps_per_epoch = len(train_rollout_loader) if train_rollout_loader is not None else 0
    val_rollout_instances = len(val_rollout_loader.dataset) if val_rollout_loader is not None else 0
    val_rollout_steps_per_epoch = len(val_rollout_loader) if val_rollout_loader is not None else 0
    rollout_mode_msg = rollout_loss_mode
    if rollout_loss_mode in {"stochastic_nll", "stochastic_mse"}:
        rollout_mode_msg = f"{rollout_loss_mode} (K={rollout_stochastic_samples})"
    startup_lines = [
        f"Run name: {run_name}",
        (
            f"HNN training setup: epochs={epochs}, batch_size={batch_size}, "
            f"steps_per_epoch={train_steps_per_epoch}, train_instances={train_instances}, "
            f"train_trajectories={len(train_series_raw)}"
        ),
        (
            f"Validation setup: steps={val_steps_per_epoch}, val_instances={val_instances}, "
            f"val_trajectories={0 if val_series_raw is None else len(val_series_raw)}"
        ),
        (
            f"Runtime: device={device}, num_workers={num_workers}, amp={amp_enabled}, "
            f"compile={bool(compile_cfg.use_compile)}, lr={lr:g}, scheduler={use_lr_scheduler}"
        ),
        (
            f"Monitoring: validate_every={validate_every_epochs}, "
            f"print_every={print_every_epochs}, log_every={log_every_epochs}, async_validation={async_validation}, "
            f"val_samples_per_ur={validation_samples_per_ur}"
        ),
    ]
    if rollout_det_weight > 0.0 and current_rollout_det_steps > 0:
        startup_lines.append(
            "Rollout loss: "
            f"mode={rollout_mode_msg}, weight={rollout_det_weight:g}, "
            f"steps={current_rollout_det_steps}, rollout_batch_size={rollout_det_batch_size}, "
            f"train_windows={train_rollout_instances}, train_rollout_steps={train_rollout_steps_per_epoch}, "
            f"val_windows={val_rollout_instances}, val_rollout_steps={val_rollout_steps_per_epoch}"
        )
    if normalize_by_ur_bin_std:
        startup_lines.append(
            "U_r loss scaling: "
            "enabled=true, "
            f"bin_size={ur_bin_size:g}, "
            f"eps={ur_bin_scale_eps:g}"
        )
    print("\n".join(startup_lines))

    for epoch in range(epochs):
        scheduled_rollout_det_steps = _scheduled_rollout_det_steps(
            epoch=epoch,
            base_steps=rollout_det_steps,
            final_steps=rollout_det_steps_final,
            warmup_epochs=rollout_det_steps_warmup_epochs,
        )
        if scheduled_rollout_det_steps != current_rollout_det_steps:
            current_rollout_det_steps = scheduled_rollout_det_steps
            train_rollout_loader, val_rollout_loader, train_rollout_windows = _rebuild_rollout_loaders(current_rollout_det_steps)
            print(
                f"Epoch {epoch}: updated rollout loss horizon to "
                f"{current_rollout_det_steps} step(s) with {train_rollout_windows} training windows."
            )
        if use_lr_scheduler:
            for group in opt.param_groups:
                group["lr"] = lr_scheduler.get_lr(epoch)

        epoch_metrics = _train_one_epoch(
            model=model,
            opt=opt,
            train_loader=train_loader,
            train_rollout_loader=train_rollout_loader,
            device=device,
            non_blocking=non_blocking,
            max_grad_norm=max_grad_norm,
            mean_reg=mean_reg,
            mean_reg_norm=mean_reg_norm,
            sigma_reg=sigma_reg,
            sigma_reg_norm=sigma_reg_norm,
            ur_bin_size=ur_bin_size,
            normalize_residual_by_ur_bin_std=normalize_by_ur_bin_std,
            normalize_rollout_by_ur_bin_std=normalize_by_ur_bin_std,
            ur_bin_state_scale_info=ur_bin_state_scale_info,
            rollout_det_weight=rollout_det_weight,
            rollout_loss_mode=rollout_loss_mode,
            rollout_stochastic_samples=rollout_stochastic_samples,
            rollout_noise_scale=rollout_noise_scale,
            use_force_data_loss=use_force_data_loss,
            force_data_weight=force_data_weight,
            gradnorm_balancer=gradnorm_balancer,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            scaler=scaler,
            log_component_grad_norms=log_component_grad_norms,
            symmetry_weight=symmetry_weight,
            symmetry_norm=symmetry_norm,
        )

        mean_loss = epoch_metrics["mean_loss"]
        mean_res_loss = epoch_metrics["mean_res_loss"]
        mean_sigma_reg_loss = epoch_metrics["mean_sigma_reg_loss"]
        mean_mean_reg_loss = epoch_metrics["mean_mean_reg_loss"]
        mean_force_data_loss = epoch_metrics["mean_force_data_loss"]
        mean_sym_loss = epoch_metrics["mean_sym_loss"]
        mean_rollout_det_loss = epoch_metrics["mean_rollout_det_loss"]
        mean_grad_norm = epoch_metrics["mean_grad_norm"]
        mean_sigma_reg = epoch_metrics["mean_sigma_reg"]
        mean_mean_reg = epoch_metrics["mean_mean_reg"]
        mean_res_grad_component = epoch_metrics["mean_res_grad_component"]
        mean_sigma_grad_component = epoch_metrics["mean_sigma_grad_component"]
        mean_mean_grad_component = epoch_metrics["mean_mean_grad_component"]

        current_lr = float(opt.param_groups[0]["lr"]) if opt.param_groups else lr

        train_metrics: dict[str, float] = {
            "loss_total": mean_loss,
            "loss_physics": mean_res_loss,
            "loss_reg": mean_sigma_reg_loss,
            "loss_reg_mean": mean_mean_reg_loss,
            "loss_data": mean_force_data_loss,
            "loss_sym": mean_sym_loss,
            "loss_rollout_det": mean_rollout_det_loss,
            "rollout_det_steps": float(current_rollout_det_steps),
            "lr": current_lr,
            "grad_norm": mean_grad_norm,
            "avg_sigma_reg": mean_sigma_reg,
            "avg_mean_reg": mean_mean_reg,
        }
        if log_component_grad_norms:
            train_metrics["grad_norm_residual_comp"] = mean_res_grad_component
            train_metrics["grad_norm_sigma_comp"] = mean_sigma_grad_component
            train_metrics["grad_norm_mean_comp"] = mean_mean_grad_component
        if "mean_gradnorm_weight_residual" in epoch_metrics:
            train_metrics["gradnorm_weight_physics"] = float(epoch_metrics["mean_gradnorm_weight_residual"])
        if "mean_gradnorm_weight_data" in epoch_metrics:
            train_metrics["gradnorm_weight_data"] = float(epoch_metrics["mean_gradnorm_weight_data"])

        log_this_epoch = (epoch % log_every_epochs) == 0 or epoch == (epochs - 1)
        if log_this_epoch:
            log_training_metrics(writer, epoch, train_metrics)
        print_this_epoch = (epoch % print_every_epochs) == 0 or epoch == (epochs - 1)
        if print_this_epoch:
            if use_force_data_loss:
                print(
                    f"Epoch {epoch}: loss={mean_loss:.4e}, res={mean_res_loss:.4e}, "
                    f"sigma_reg={mean_sigma_reg_loss:.4e}, mean_reg={mean_mean_reg_loss:.4e}, "
                    f"data={mean_force_data_loss:.4e}, sym={mean_sym_loss:.4e}, rollout_det={mean_rollout_det_loss:.4e}"
                )
            else:
                print(
                    f"Epoch {epoch}: loss={mean_loss:.4e}, res={mean_res_loss:.4e}, "
                    f"sigma_reg={mean_sigma_reg_loss:.4e}, mean_reg={mean_mean_reg_loss:.4e}, "
                    f"sym={mean_sym_loss:.4e}, rollout_det={mean_rollout_det_loss:.4e}"
                )

        should_validate_losses = validate_every_epochs > 0 and (
            (epoch + 1) % int(validate_every_epochs) == 0 or epoch == (epochs - 1)
        )
        should_validate_rollout = should_validate_losses
        if async_validation and (should_validate_losses or should_validate_rollout):
            async_processes = _reap_async_processes(async_processes, writer=writer, best_state=async_best_state, wait=False)
            state_source = model
            if hasattr(model, "_orig_mod"):
                state_source = getattr(model, "_orig_mod")
            async_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = async_dir / f"epoch_{epoch + 1:06d}.pt"
            checkpoint_config = asdict(config)
            checkpoint_config["loss"]["rollout_det_steps"] = int(current_rollout_det_steps)
            torch.save(
                {
                    "model_state": state_source.state_dict(),
                    "config": checkpoint_config,
                    "ur_bin_state_scale_info": ur_bin_state_scale_info,
                    "run_name": run_name,
                    "dt": dt,
                    "method": str(config.method),
                    "td_correction": True,
                },
                ckpt_path,
            )
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
                do_losses=should_validate_losses,
                do_rollout=should_validate_rollout,
                best_state=async_best_state,
            )
        elif not async_validation:
            _validate_if_needed(
                writer=writer,
                epoch=epoch,
                validate_now=should_validate_losses,
                rollout_now=should_validate_rollout,
                model=model,
                y_data_t=y_data_t,
                val_vel=val_vel,
                reduced_velocity=val_ur,
                val_series_raw=val_series_raw,
                val_sequences=val_sequences,
                val_loader=val_loader,
                val_rollout_loader=val_rollout_loader,
                validation_samples_per_ur=validation_samples_per_ur,
                m_eff=m_eff,
                dt=dt,
                t=t,
                y_true_norm=y_true_norm,
                y_data=y_data,
                force_data=force_data,
                D=D,
                k=k,
                device=device,
                middle_time_plot=middle_time_plot,
                hamiltonian_data=hamiltonian_data,
                log_extra_validation_metrics=log_extra_validation_metrics,
                rollout_stochastic=rollout_stochastic,
                rollout_noise_scale=rollout_noise_scale,
                rollout_seed=rollout_seed,
                mean_reg=mean_reg,
                mean_reg_norm=mean_reg_norm,
                sigma_reg=sigma_reg,
                sigma_reg_norm=sigma_reg_norm,
                ur_bin_size=ur_bin_size,
                normalize_residual_by_ur_bin_std=normalize_by_ur_bin_std,
                normalize_rollout_by_ur_bin_std=normalize_by_ur_bin_std,
                ur_bin_state_scale_info=ur_bin_state_scale_info,
                rollout_det_weight=rollout_det_weight,
                rollout_loss_mode=rollout_loss_mode,
                rollout_stochastic_samples=rollout_stochastic_samples,
                use_force_data_loss=use_force_data_loss,
                force_data_weight=force_data_weight,
                symmetry_weight=symmetry_weight,
                symmetry_norm=symmetry_norm,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )

    if async_validation and async_processes:
        print(f"Waiting for {len(async_processes)} async validation job(s) to finish...")
        async_processes = _reap_async_processes(async_processes, writer=writer, best_state=async_best_state, wait=True)

    writer.add_text("phnn/config_hnn", json.dumps(hnn_cfg, indent=2, sort_keys=True), 0)

    if final_rollout_all_validation and val_series_raw is not None and val_sequences is not None:
        print("Final validation rollout (all trajectories) started.")
        final_start = time.perf_counter()
        avg_metrics, used, ur_values, metrics_list = _log_final_rollouts_all(
            writer=writer,
            epoch=max(0, epochs - 1),
            model=model,
            val_series_raw=val_series_raw,
            val_sequences=val_sequences,
            m_eff=m_eff,
            D=D,
            k=k,
            device=device,
            middle_time_plot=middle_time_plot,
            log_extra_validation_metrics=log_extra_validation_metrics,
            rollout_stochastic=rollout_stochastic,
            rollout_noise_scale=rollout_noise_scale,
            rollout_seed=rollout_seed,
            extra_series_raw=train_series_raw,
            extra_sequences=train_sequences,
        )
        if used > 0 and avg_metrics:
            summary_lines = [f"Final rollout over {used} validation trajectories (unique U_r):"]
            for name in sorted(avg_metrics):
                summary_lines.append(f"{name}: {avg_metrics[name]:.6f}")
                writer.add_scalar(f"final_val/avg/{name}", avg_metrics[name], epochs)
            writer.add_text("final_val/summary", "\n".join(summary_lines), epochs)
        if ur_values and metrics_list:
            reference_ur_values = [
                float(np.asarray(series_raw[5]).reshape(-1)[0])
                for series_raw in [*val_series_raw, *train_series_raw]
                if np.asarray(series_raw[5]).reshape(-1).size > 0
            ]
            log_final_rollout_errors_vs_ur(
                writer,
                ur_values,
                metrics_list,
                epochs,
                reference_ur_values=reference_ur_values,
            )
        elapsed = time.perf_counter() - final_start
        print(f"Final validation rollout finished in {elapsed:.2f}s.")

    if final_rollout_all_validation and val_loader is not None:
        final_loss_by_ur = _per_ur_loss_map_hnn(
            model=model,
            loader=val_loader,
            rollout_loader=val_rollout_loader,
            device=device,
            non_blocking=(device.type == "cuda"),
            mean_reg=mean_reg,
            mean_reg_norm=mean_reg_norm,
            sigma_reg=sigma_reg,
            sigma_reg_norm=sigma_reg_norm,
            ur_bin_size=ur_bin_size,
            normalize_residual_by_ur_bin_std=normalize_by_ur_bin_std,
            normalize_rollout_by_ur_bin_std=normalize_by_ur_bin_std,
            ur_bin_state_scale_info=ur_bin_state_scale_info,
            rollout_det_weight=rollout_det_weight,
            rollout_loss_mode=rollout_loss_mode,
            rollout_stochastic_samples=rollout_stochastic_samples,
            rollout_noise_scale=rollout_noise_scale,
            use_force_data_loss=use_force_data_loss,
            force_data_weight=force_data_weight,
            symmetry_weight=symmetry_weight,
            symmetry_norm=symmetry_norm,
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

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{run_name}.pt"
    state_source = model
    if hasattr(model, "_orig_mod"):
        state_source = getattr(model, "_orig_mod")
    final_config = asdict(config)
    final_config["loss"]["rollout_det_steps"] = int(current_rollout_det_steps)
    torch.save(
        {
            "model_state": state_source.state_dict(),
            "config": final_config,
            "ur_bin_state_scale_info": ur_bin_state_scale_info,
            "run_name": run_name,
        },
        model_path,
    )
    print(f"Saved final model to {model_path}")

    writer.flush()
    writer.close()
