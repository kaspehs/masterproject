from __future__ import annotations

import json
import os
import time
import subprocess
import sys
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.optim as optim
import torch.nn.utils as nn_utils
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
    GradNormBalancer,
    PHVIV,
    ROLLOUT_DIVERGED_COUNT_KEY,
    ROLLOUT_DIVERGED_KEY,
    build_dataloader_from_series,
    build_rollout_dataloader_from_series,
    compute_validation_metrics,
    compute_model_grad_norm,
    load_training_series,
    format_loss_vs_ur_text,
    log_loss_vs_ur,
    log_final_rollout_errors_vs_ur,
    log_training_metrics,
    log_validation_epoch,
    preprocess_timeseries,
    resolve_cut_start_seconds,
    sample_one_index_per_ur,
)


def _parse_hnn_batch(batch: Any) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    if len(batch) < 5:
        raise ValueError("Unexpected batch format from dataloader.")
    z_i, t_i, z_next, t_next, ur_i = batch[:5]
    idx = 5
    history = None
    f_i = None
    f_next = None
    scale = None
    if len(batch) > idx:
        candidate = batch[idx]
        if torch.is_tensor(candidate) and candidate.ndim == 3 and candidate.shape[-1] == 3:
            history = candidate
            idx += 1
    remaining = len(batch) - idx
    if remaining == 0:
        pass
    elif remaining == 1:
        scale = batch[idx]
    elif remaining == 2:
        f_i, f_next = batch[idx], batch[idx + 1]
    elif remaining == 3:
        f_i, f_next, scale = batch[idx], batch[idx + 1], batch[idx + 2]
    else:
        raise ValueError("Unexpected batch format from dataloader.")
    return z_i, t_i, z_next, t_next, ur_i, history, f_i, f_next, scale


def _parse_rollout_batch(batch: Any) -> tuple[Any, Any, Any, Any, Any, Any]:
    if len(batch) < 4:
        raise ValueError("Unexpected rollout batch format.")
    z0, t_seq, z_traj, ur0 = batch[:4]
    idx = 4
    history = None
    scale = None
    if len(batch) > idx:
        candidate = batch[idx]
        if torch.is_tensor(candidate) and candidate.ndim == 3 and candidate.shape[-1] == 3:
            history = candidate
            idx += 1
    remaining = len(batch) - idx
    if remaining == 0:
        pass
    elif remaining == 1:
        scale = batch[idx]
    else:
        raise ValueError("Unexpected rollout batch format.")
    return z0, t_seq, z_traj, ur0, history, scale


def _odd_symmetry_penalty_per_sample(
    *,
    model: torch.nn.Module,
    z: torch.Tensor,
    ur: torch.Tensor,
    norm: str,
    history_context: torch.Tensor | None = None,
) -> torch.Tensor:
    z_flip = -z
    if getattr(model, "force_output", "force") == "coefficient":
        f_pos = model.u_theta_coeff(z, reduced_velocity=ur, history_context=history_context)
        f_neg = model.u_theta_coeff(z_flip, reduced_velocity=ur, history_context=history_context)
    else:
        f_pos = model.u_theta(z, reduced_velocity=ur, history_context=history_context)
        f_neg = model.u_theta(z_flip, reduced_velocity=ur, history_context=history_context)
    sym_residual = f_pos + f_neg
    if sym_residual.ndim == 1:
        sym_residual = sym_residual.unsqueeze(-1)
    if norm == "l1":
        return torch.mean(torch.abs(sym_residual), dim=1)
    return torch.mean(sym_residual * sym_residual, dim=1)


def _deterministic_rollout_loss_from_batch(
    *,
    model: PHVIV,
    batch: Any,
    device: torch.device,
    non_blocking: bool,
    per_traj_norm_eps: float,
    return_per_sample: bool = False,
) -> torch.Tensor:
    z0, t_seq, z_traj, ur0, history0, scale = _parse_rollout_batch(batch)
    z0 = z0.to(device, non_blocking=non_blocking)
    t_seq = t_seq.to(device, non_blocking=non_blocking)
    z_traj = z_traj.to(device, non_blocking=non_blocking)
    ur0 = ur0.to(device, non_blocking=non_blocking)
    if history0 is not None:
        history0 = history0.to(device, non_blocking=non_blocking)
    if scale is not None:
        scale = scale.to(device, non_blocking=non_blocking).view(-1)

    z_pred, _ = model.rollout(
        z0,
        t_seq,
        float(model.dt),
        reduced_velocity=ur0,
        history_init=history0,
    )
    z_scale = model.res_scale.to(device=z_pred.device, dtype=z_pred.dtype).view(1, 1, -1)
    err = (z_pred - z_traj) / z_scale
    per = torch.mean(err[..., 0] * err[..., 0], dim=1) + torch.mean(err[..., 1] * err[..., 1], dim=1)
    if scale is not None:
        per = per / (scale * scale + float(per_traj_norm_eps))
    if return_per_sample:
        return per
    return torch.mean(per)


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
    force_reg: float,
    sigma_reg_norm: str,
    rollout_det_weight: float,
    use_force_data_loss: bool,
    force_data_weight: float,
    gradnorm_balancer: Optional[GradNormBalancer],
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    scaler: torch.amp.GradScaler,
    log_component_grad_norms: bool,
    per_traj_norm_eps: float,
    force_reg_on_coeff: bool,
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
    gradnorm_sigma_weight_sum = torch.zeros((), device=device)
    gradnorm_mean_weight_sum = torch.zeros((), device=device) if float(mean_reg) > 0.0 else None
    gradnorm_data_weight_sum = torch.zeros((), device=device) if use_force_data_loss else None
    gradnorm_weight_count = 0

    force_output_coeff = getattr(model, "force_output", "force") == "coefficient"
    rollout_iter = iter(train_rollout_loader) if (train_rollout_loader is not None and float(rollout_det_weight) > 0.0) else None
    for batch in train_loader:
        z_i, t_i, z_next, t_next, ur_i, history_i, f_i, f_next, scale = _parse_hnn_batch(batch)
        z_i = z_i.to(device, non_blocking=non_blocking)
        t_i = t_i.to(device, non_blocking=non_blocking)
        z_next = z_next.to(device, non_blocking=non_blocking)
        t_next = t_next.to(device, non_blocking=non_blocking)
        ur_i = ur_i.to(device, non_blocking=non_blocking)
        if history_i is not None:
            history_i = history_i.to(device, non_blocking=non_blocking)
        if f_i is not None:
            f_i = f_i.to(device, non_blocking=non_blocking)
        if f_next is not None:
            f_next = f_next.to(device, non_blocking=non_blocking)
        if scale is not None:
            scale = scale.to(device, non_blocking=non_blocking).view(-1)

        opt.zero_grad()

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
            history_context = model.history_context(
                z_i,
                reduced_velocity=ur_i,
                history_window=history_i,
            )
            if scale is None:
                res_loss = model.res_loss(
                    z_i,
                    t_i,
                    z_next,
                    t_next,
                    reduced_velocity=ur_i,
                    history_context=history_context,
                )
                sigma_reg_loss = model.avg_sigma_reg_SRK4(
                    z_i,
                    t_i,
                    z_next,
                    t_next,
                    reduced_velocity=ur_i,
                    norm=sigma_reg_norm,
                    history_context=history_context,
                )
                mean_reg_loss = model.avg_mean_reg_SRK4(
                    z_i,
                    t_i,
                    z_next,
                    t_next,
                    reduced_velocity=ur_i,
                    norm=mean_reg_norm,
                    on_coeff=force_reg_on_coeff,
                    history_context=history_context,
                )
            else:
                per_res = model.res_loss_per_sample(
                    z_i,
                    t_i,
                    z_next,
                    t_next,
                    reduced_velocity=ur_i,
                    history_context=history_context,
                )
                per_sigma_reg = model.avg_sigma_reg_SRK4_per_sample(
                    z_i,
                    t_i,
                    z_next,
                    t_next,
                    reduced_velocity=ur_i,
                    norm=sigma_reg_norm,
                    history_context=history_context,
                )
                per_mean_reg = model.avg_mean_reg_SRK4_per_sample(
                    z_i,
                    t_i,
                    z_next,
                    t_next,
                    reduced_velocity=ur_i,
                    norm=mean_reg_norm,
                    on_coeff=force_reg_on_coeff,
                    history_context=history_context,
                )
                denom = scale * scale + float(per_traj_norm_eps)
                res_loss = torch.mean(per_res / denom)
                sigma_reg_loss = torch.mean(per_sigma_reg / denom)
                mean_reg_loss = torch.mean(per_mean_reg / denom)
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
                    f_pred = model.u_theta_coeff(z_mid, reduced_velocity=ur_i, history_context=history_context)
                    f_mid = f_mid / f0
                else:
                    f_pred = model.u_theta(z_mid, reduced_velocity=ur_i, history_context=history_context)
                per_data = torch.mean((f_pred - f_mid) ** 2, dim=1)
                if scale is not None:
                    per_data = per_data / (scale * scale + float(per_traj_norm_eps))
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
                    history_context=history_context,
                )
                if scale is not None:
                    if symmetry_norm == "l1":
                        per_sym = per_sym / (scale + float(per_traj_norm_eps))
                    else:
                        per_sym = per_sym / (scale * scale + float(per_traj_norm_eps))
                sym_loss = torch.mean(per_sym)
            else:
                sym_loss = res_loss.new_tensor(0.0)

            if gradnorm_balancer is not None:
                loss_inputs: dict[str, torch.Tensor] = {
                    "residual": res_loss.float(),
                    "sigma": base_sigma_reg_loss.float(),
                    "data": data_force_loss.float() if use_force_data_loss else res_loss.float(),
                }
                if float(mean_reg) > 0.0:
                    loss_inputs["mean"] = base_mean_reg_loss.float()
                weights = gradnorm_balancer.update(loss_inputs)
                res_weight = weights["residual"]
                sigma_weight = weights["sigma"]
                mean_weight = weights.get("mean", res_loss.new_tensor(1.0))
                data_weight = weights.get("data", res_loss.new_tensor(1.0))
                gradnorm_res_weight_sum = gradnorm_res_weight_sum + res_weight
                gradnorm_sigma_weight_sum = gradnorm_sigma_weight_sum + sigma_weight
                if gradnorm_mean_weight_sum is not None:
                    gradnorm_mean_weight_sum = gradnorm_mean_weight_sum + mean_weight
                if gradnorm_data_weight_sum is not None:
                    gradnorm_data_weight_sum = gradnorm_data_weight_sum + data_weight
                gradnorm_weight_count += 1
            else:
                res_weight = res_loss.new_tensor(1.0)
                sigma_weight = res_loss.new_tensor(1.0)
                mean_weight = res_loss.new_tensor(1.0)
                data_weight = res_loss.new_tensor(1.0)

            # GradNorm balances raw branch losses first; user multipliers are applied after.
            gradnorm_weighted_res = res_weight * res_loss
            gradnorm_weighted_sigma = sigma_weight * base_sigma_reg_loss
            gradnorm_weighted_mean = mean_weight * base_mean_reg_loss
            gradnorm_weighted_data = data_weight * data_force_loss

            weighted_sigma = float(force_reg) * gradnorm_weighted_sigma
            weighted_mean = float(mean_reg) * gradnorm_weighted_mean
            weighted_data = float(force_data_weight) * gradnorm_weighted_data
            weighted_sym = float(symmetry_weight) * sym_loss
            if rollout_iter is not None:
                try:
                    rollout_batch = next(rollout_iter)
                except StopIteration:
                    rollout_iter = iter(train_rollout_loader)
                    rollout_batch = next(rollout_iter)
                rollout_det_loss = _deterministic_rollout_loss_from_batch(
                    model=model,
                    batch=rollout_batch,
                    device=device,
                    non_blocking=non_blocking,
                    per_traj_norm_eps=per_traj_norm_eps,
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
        # Log loss_reg as the raw regularizer magnitude (before force_reg scaling).
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
        metrics["mean_gradnorm_weight_force"] = float(
            (gradnorm_sigma_weight_sum / float(gradnorm_weight_count)).detach().cpu()
        )
        metrics["mean_gradnorm_weight_sigma"] = float(
            (gradnorm_sigma_weight_sum / float(gradnorm_weight_count)).detach().cpu()
        )
        if gradnorm_mean_weight_sum is not None:
            metrics["mean_gradnorm_weight_mean"] = float(
                (gradnorm_mean_weight_sum / float(gradnorm_weight_count)).detach().cpu()
            )
        if gradnorm_data_weight_sum is not None:
            metrics["mean_gradnorm_weight_data"] = float(
                (gradnorm_data_weight_sum / float(gradnorm_weight_count)).detach().cpu()
            )
    return metrics


def _validate_if_needed(
    *,
    writer: SummaryWriter,
    epoch: int,
    rollout_every_epochs: int,
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
    cycle_validation_rollout: bool,
    fixed_validation_sampling: bool,
    validation_sampling_seed: int,
    rollout_target_ur: float | None,
    rollout_target_ur_tol: float,
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
    force_reg: float,
    sigma_reg_norm: str,
    rollout_det_weight: float,
    use_force_data_loss: bool,
    force_data_weight: float,
    symmetry_weight: float,
    symmetry_norm: str,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    per_traj_norm_eps: float,
    force_reg_on_coeff: bool,
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
            force_reg=force_reg,
            sigma_reg_norm=sigma_reg_norm,
            rollout_det_weight=rollout_det_weight,
            use_force_data_loss=use_force_data_loss,
            force_data_weight=force_data_weight,
            symmetry_weight=symmetry_weight,
            symmetry_norm=symmetry_norm,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            per_traj_norm_eps=per_traj_norm_eps,
            force_reg_on_coeff=force_reg_on_coeff,
        )
        for name, value in val_loss_metrics.items():
            writer.add_scalar(f"val/{name}", value, epoch + 1)
        loss_by_ur = _per_ur_loss_map_hnn(
            model=model,
            loader=val_loader,
            device=device,
            non_blocking=(device.type == "cuda"),
            rollout_loader=val_rollout_loader,
            mean_reg=mean_reg,
            mean_reg_norm=mean_reg_norm,
            force_reg=force_reg,
            sigma_reg_norm=sigma_reg_norm,
            rollout_det_weight=rollout_det_weight,
            use_force_data_loss=use_force_data_loss,
            force_data_weight=force_data_weight,
            symmetry_weight=symmetry_weight,
            symmetry_norm=symmetry_norm,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            per_traj_norm_eps=per_traj_norm_eps,
            force_reg_on_coeff=force_reg_on_coeff,
        )
        log_loss_vs_ur(
            writer,
            epoch + 1,
            loss_by_ur,
            tag="val/loss_vs_ur",
            title="Validation loss vs U_r",
        )

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
        sample_seed = int(validation_sampling_seed) if fixed_validation_sampling else (int(epoch) + 1)
        sampled_indices = sample_one_index_per_ur(ur_for_sampling, seed=sample_seed)
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
        selected_indices: list[int] | None = None
        if rollout_target_ur is not None:
            matches: list[int] = []
            for idx, series_raw in enumerate(val_series_raw):
                ur_arr = np.asarray(series_raw[5]).reshape(-1)
                if ur_arr.size == 0:
                    continue
                ur_val = float(ur_arr[0])
                if np.isclose(
                    ur_val,
                    float(rollout_target_ur),
                    rtol=0.0,
                    atol=float(rollout_target_ur_tol),
                ):
                    matches.append(idx)
            if matches:
                selected_indices = matches
            else:
                warnings.warn(
                    "monitoring.rollout_use_excluded_ur is enabled but no validation rollout "
                    f"trajectory matched U_r={float(rollout_target_ur):.6g} "
                    f"(tol={float(rollout_target_ur_tol):.3g}); falling back to default rollout selection."
                )
        if selected_indices is None:
            selected_indices = list(range(total))
        if cycle_validation_rollout:
            step = max(0, (epoch + 1) // max(1, int(rollout_every_epochs)) - 1)
            rollout_idx = selected_indices[step % len(selected_indices)]
        else:
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
) -> tuple[dict[str, float], int, list[float], list[dict[str, float]]]:
    total = min(len(val_series_raw), len(val_sequences))
    if total <= 0:
        return {}, 0, [], []
    selected_indices: list[int] = []
    seen_ur: set[float] = set()
    for idx in range(total):
        ur_np = val_series_raw[idx][5]
        ur_val = float(np.asarray(ur_np).reshape(-1)[0])
        ur_key = round(ur_val, 6)
        if ur_key in seen_ur:
            continue
        seen_ur.add(ur_key)
        selected_indices.append(idx)
    if not selected_indices:
        return {}, 0, [], []
    metrics_sum: dict[str, float] = {}
    metrics_count: dict[str, int] = {}
    used = 0
    ur_values: list[float] = []
    metrics_list: list[dict[str, float]] = []
    for step_idx, idx in enumerate(selected_indices):
        y_np, t_np, dt_value, _vel_np, force_np, _ur_np = val_series_raw[idx]
        y_tensor, vel_tensor, _t_tensor, ur_tensor = val_sequences[idx]
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
            rollout_stochastic=rollout_stochastic,
            rollout_noise_scale=rollout_noise_scale,
            rollout_seed=rollout_seed,
            tag_prefix="final_val/rollout",
            step=step_idx,
            title_suffix=f" [final {step_idx+1}/{len(selected_indices)}]",
        )
        filtered_metrics = {
            name: float(value)
            for name, value in metrics.items()
            if name != ROLLOUT_DIVERGED_KEY and np.isfinite(float(value))
        }
        if filtered_metrics:
            ur_values.append(ur_val)
            metrics_list.append(filtered_metrics)
        for name, value in filtered_metrics.items():
            metrics_sum[name] = metrics_sum.get(name, 0.0) + float(value)
            metrics_count[name] = metrics_count.get(name, 0) + 1
        used += 1
    averaged = {
        name: metrics_sum[name] / float(metrics_count[name])
        for name in metrics_sum
        if metrics_count.get(name, 0) > 0
    }
    return averaged, used, ur_values, metrics_list


def _reap_async_processes(
    processes: list[dict[str, Any]],
    *,
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

        if return_code == 0:
            print(
                f"[async-val] epoch {epoch}: completed successfully in {elapsed:.2f}s"
            )
        else:
            print(
                f"[async-val] epoch {epoch}: FAILED with exit code {return_code} "
                f"after {elapsed:.2f}s"
            )

        if ckpt_path.exists() and return_code == 0:
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
    writer: SummaryWriter,
    async_device: str,
    async_num_workers: int,
    async_num_threads: int,
    rollout_every_epochs: int,
    cycle_validation_rollout: bool,
    rollout_target_ur: float | None,
    rollout_target_ur_tol: float,
    do_losses: bool,
    do_rollout: bool,
) -> list[dict[str, Any]]:
    processes = _reap_async_processes(processes, wait=False)
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
        "--rollout-target-ur-tol",
        str(float(rollout_target_ur_tol)),
        "--do-losses",
        "1" if do_losses else "0",
        "--do-rollout",
        "1" if do_rollout else "0",
    ]
    if rollout_target_ur is not None:
        args.extend(["--rollout-target-ur", str(float(rollout_target_ur))])
    epoch_num = int(epoch + 1)
    proc = subprocess.Popen(args, env=env)
    processes.append(
        {
            "proc": proc,
            "epoch": epoch_num,
            "start_time": time.perf_counter(),
            "checkpoint_path": str(checkpoint_path),
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
    force_reg: float,
    sigma_reg_norm: str,
    rollout_det_weight: float,
    use_force_data_loss: bool,
    force_data_weight: float,
    symmetry_weight: float,
    symmetry_norm: str,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    per_traj_norm_eps: float,
    force_reg_on_coeff: bool,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
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
            z_i, t_i, z_next, t_next, ur_i, history_i, f_i, f_next, scale = _parse_hnn_batch(batch)
            z_i = z_i.to(device, non_blocking=non_blocking)
            t_i = t_i.to(device, non_blocking=non_blocking)
            z_next = z_next.to(device, non_blocking=non_blocking)
            t_next = t_next.to(device, non_blocking=non_blocking)
            ur_i = ur_i.to(device, non_blocking=non_blocking)
            if history_i is not None:
                history_i = history_i.to(device, non_blocking=non_blocking)
            if f_i is not None:
                f_i = f_i.to(device, non_blocking=non_blocking)
            if f_next is not None:
                f_next = f_next.to(device, non_blocking=non_blocking)
            if scale is not None:
                scale = scale.to(device, non_blocking=non_blocking).view(-1)

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                history_context = model.history_context(
                    z_i,
                    reduced_velocity=ur_i,
                    history_window=history_i,
                )
                if scale is None:
                    res_loss = model.res_loss(
                        z_i,
                        t_i,
                        z_next,
                        t_next,
                        reduced_velocity=ur_i,
                        history_context=history_context,
                    )
                    sigma_reg_loss = model.avg_sigma_reg_SRK4(
                        z_i,
                        t_i,
                        z_next,
                        t_next,
                        reduced_velocity=ur_i,
                        norm=sigma_reg_norm,
                        history_context=history_context,
                    )
                    mean_reg_loss = model.avg_mean_reg_SRK4(
                        z_i,
                        t_i,
                        z_next,
                        t_next,
                        reduced_velocity=ur_i,
                        norm=mean_reg_norm,
                        on_coeff=force_reg_on_coeff,
                        history_context=history_context,
                    )
                else:
                    per_res = model.res_loss_per_sample(
                        z_i,
                        t_i,
                        z_next,
                        t_next,
                        reduced_velocity=ur_i,
                        history_context=history_context,
                    )
                    per_sigma_reg = model.avg_sigma_reg_SRK4_per_sample(
                        z_i,
                        t_i,
                        z_next,
                        t_next,
                        reduced_velocity=ur_i,
                        norm=sigma_reg_norm,
                        history_context=history_context,
                    )
                    per_mean_reg = model.avg_mean_reg_SRK4_per_sample(
                        z_i,
                        t_i,
                        z_next,
                        t_next,
                        reduced_velocity=ur_i,
                        norm=mean_reg_norm,
                        on_coeff=force_reg_on_coeff,
                        history_context=history_context,
                    )
                    denom = scale * scale + float(per_traj_norm_eps)
                    res_loss = torch.mean(per_res / denom)
                    sigma_reg_loss = torch.mean(per_sigma_reg / denom)
                    mean_reg_loss = torch.mean(per_mean_reg / denom)
                sigma_loss = float(force_reg) * sigma_reg_loss
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
                        f_pred = model.u_theta_coeff(z_mid, reduced_velocity=ur_i, history_context=history_context)
                        f_mid = f_mid / f0
                    else:
                        f_pred = model.u_theta(z_mid, reduced_velocity=ur_i, history_context=history_context)
                    per_data = torch.mean((f_pred - f_mid) ** 2, dim=1)
                    if scale is not None:
                        per_data = per_data / (scale * scale + float(per_traj_norm_eps))
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
                        history_context=history_context,
                    )
                    if scale is not None:
                        if symmetry_norm == "l1":
                            per_sym = per_sym / (scale + float(per_traj_norm_eps))
                        else:
                            per_sym = per_sym / (scale * scale + float(per_traj_norm_eps))
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
                    rollout_det_loss = _deterministic_rollout_loss_from_batch(
                        model=model,
                        batch=rollout_batch,
                        device=device,
                        non_blocking=non_blocking,
                        per_traj_norm_eps=per_traj_norm_eps,
                    )
                else:
                    rollout_det_loss = res_loss.new_tensor(0.0)
                total = total + float(rollout_det_weight) * rollout_det_loss

            loss_sum = loss_sum + total.detach().float()
            res_sum = res_sum + res_loss.detach().float()
            # Log loss_reg as the raw regularizer magnitude (before force_reg scaling).
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
    force_reg: float,
    sigma_reg_norm: str,
    rollout_det_weight: float,
    use_force_data_loss: bool,
    force_data_weight: float,
    symmetry_weight: float,
    symmetry_norm: str,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    per_traj_norm_eps: float,
    force_reg_on_coeff: bool,
) -> dict[str, dict[float, float]]:
    model.eval()
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
            z_i, t_i, z_next, t_next, ur_i, history_i, f_i, f_next, scale = _parse_hnn_batch(batch)
            z_i = z_i.to(device, non_blocking=non_blocking)
            t_i = t_i.to(device, non_blocking=non_blocking)
            z_next = z_next.to(device, non_blocking=non_blocking)
            t_next = t_next.to(device, non_blocking=non_blocking)
            ur_i = ur_i.to(device, non_blocking=non_blocking)
            if history_i is not None:
                history_i = history_i.to(device, non_blocking=non_blocking)
            if f_i is not None:
                f_i = f_i.to(device, non_blocking=non_blocking)
            if f_next is not None:
                f_next = f_next.to(device, non_blocking=non_blocking)
            if scale is not None:
                scale = scale.to(device, non_blocking=non_blocking).view(-1)

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                history_context = model.history_context(
                    z_i,
                    reduced_velocity=ur_i,
                    history_window=history_i,
                )
                per_res = model.res_loss_per_sample(
                    z_i,
                    t_i,
                    z_next,
                    t_next,
                    reduced_velocity=ur_i,
                    history_context=history_context,
                )
                per_sigma_reg = model.avg_sigma_reg_SRK4_per_sample(
                    z_i,
                    t_i,
                    z_next,
                    t_next,
                    reduced_velocity=ur_i,
                    norm=sigma_reg_norm,
                    history_context=history_context,
                )
                per_mean_reg = model.avg_mean_reg_SRK4_per_sample(
                    z_i,
                    t_i,
                    z_next,
                    t_next,
                    reduced_velocity=ur_i,
                    norm=mean_reg_norm,
                    on_coeff=force_reg_on_coeff,
                    history_context=history_context,
                )
                if scale is not None:
                    denom = scale * scale + float(per_traj_norm_eps)
                    per_res = per_res / denom
                    per_sigma_reg = per_sigma_reg / denom
                    per_mean_reg = per_mean_reg / denom
                per_sigma = float(force_reg) * per_sigma_reg
                per_mean = float(mean_reg) * per_mean_reg
                if use_force_data_loss and f_i is not None and f_next is not None:
                    z_mid = 0.5 * (z_i + z_next)
                    f_mid = 0.5 * (f_i + f_next)
                    if force_output_coeff:
                        f0 = model._force_scale_from_reduced_velocity(ur_i, like=f_mid, state=z_mid)
                        f_pred = model.u_theta_coeff(z_mid, reduced_velocity=ur_i, history_context=history_context)
                        f_mid = f_mid / f0
                    else:
                        f_pred = model.u_theta(z_mid, reduced_velocity=ur_i, history_context=history_context)
                    per_data = torch.mean((f_pred - f_mid) ** 2, dim=1)
                    if scale is not None:
                        per_data = per_data / (scale * scale + float(per_traj_norm_eps))
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
                        history_context=history_context,
                    )
                    if scale is not None:
                        if symmetry_norm == "l1":
                            per_sym = per_sym / (scale + float(per_traj_norm_eps))
                        else:
                            per_sym = per_sym / (scale * scale + float(per_traj_norm_eps))
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
                per_rollout = _deterministic_rollout_loss_from_batch(
                    model=model,
                    batch=batch,
                    device=device,
                    non_blocking=non_blocking,
                    per_traj_norm_eps=per_traj_norm_eps,
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


def train(config: Config, config_name: str) -> None:
    data_cfg = config.data
    middle_time_plot = data_cfg.middle_time_plot
    use_generated_train_series = data_cfg.use_generated_train_series
    train_series_root = Path(data_cfg.train_series_dir)
    train_series_dir = train_series_root

    if use_generated_train_series:
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
    else:
        data_path = Path(data_cfg.file)

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
    smoothing_cfg = config.smoothing
    hnn_cfg = dict(config.hnn or {})
    per_traj_norm = str(hnn_cfg.get("per_traj_norm", "none")).strip().lower()
    per_traj_norm_eps = float(hnn_cfg.get("per_traj_norm_eps", 1e-8))
    if per_traj_norm not in {"none", "force_rms"}:
        raise ValueError("hnn.per_traj_norm must be one of: none, force_rms.")
    velocity_source = str(hnn_cfg.get("velocity_source", "compute")).strip().lower()
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
    if use_generated_train_series and (train_include_ur is not None or train_exclude_ur is not None):
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

    force_reg = float(loss_cfg.force_reg)
    mean_reg = float(getattr(loss_cfg, "mean_reg", 0.0))
    mean_reg_norm = str(getattr(loss_cfg, "mean_reg_norm", "l1")).strip().lower()
    sigma_reg_norm = str(getattr(loss_cfg, "sigma_reg_norm", "l2")).strip().lower()
    rollout_det_weight = float(getattr(loss_cfg, "rollout_det_weight", 0.0))
    rollout_det_steps = int(getattr(loss_cfg, "rollout_det_steps", 0))
    rollout_det_batch_size_raw = int(getattr(loss_cfg, "rollout_det_batch_size", 0))
    rollout_det_batch_size = batch_size if rollout_det_batch_size_raw <= 0 else rollout_det_batch_size_raw
    force_reg_on_coeff = bool(getattr(loss_cfg, "force_reg_on_coeff", False))
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
    if rollout_det_weight > 0.0 and rollout_det_steps < 1:
        raise ValueError("loss.rollout_det_steps must be >= 1 when loss.rollout_det_weight > 0.")
    if rollout_det_batch_size < 1:
        raise ValueError("loss.rollout_det_batch_size must be >= 1 after fallback resolution.")

    rollout_every_epochs = int(monitoring_cfg.rollout_every_epochs)
    validate_every_epochs = int(getattr(monitoring_cfg, "validate_every_epochs", rollout_every_epochs))
    cycle_validation_rollout = bool(getattr(monitoring_cfg, "cycle_validation_rollout", False))
    fixed_validation_sampling = bool(getattr(monitoring_cfg, "fixed_validation_sampling", False))
    validation_sampling_seed = int(getattr(monitoring_cfg, "validation_sampling_seed", 1))
    rollout_use_excluded_ur = bool(getattr(monitoring_cfg, "rollout_use_excluded_ur", False))
    rollout_target_ur_tol = float(getattr(monitoring_cfg, "rollout_target_ur_tol", 1e-6))
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
    async_do_losses = bool(getattr(monitoring_cfg, "async_validation_do_losses", True))
    async_do_rollout = bool(getattr(monitoring_cfg, "async_validation_do_rollout", True))
    rollout_target_ur: float | None = None
    if rollout_use_excluded_ur:
        if train_exclude_ur is not None and len(train_exclude_ur) == 1:
            rollout_target_ur = float(train_exclude_ur[0])
        else:
            warnings.warn(
                "monitoring.rollout_use_excluded_ur is enabled, but hnn.train_exclude_ur "
                "must contain exactly one value. Falling back to default rollout selection."
            )

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
    history_window = int(getattr(model_cfg, "history_window", 32)) if bool(getattr(model_cfg, "use_history_tcn", False)) else None

    train_series_raw, eval_tensors = load_training_series(
        y_data,
        t,
        dt,
        use_generated_train_series,
        train_series_dir,
        m_eff,
        device,
        smoothing_cfg=smoothing_cfg,
        velocity_source=velocity_source,
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
        smoothing_cfg=smoothing_cfg,
        num_workers=num_workers,
        pin_memory=pin_memory,
        per_traj_norm=per_traj_norm,
        per_traj_norm_eps=per_traj_norm_eps,
        history_window=history_window,
    )
    train_rollout_loader: Any | None = None
    if rollout_det_weight > 0.0 and rollout_det_steps > 0:
        train_rollout_loader, train_rollout_windows = build_rollout_dataloader_from_series(
            train_series_raw,
            m_eff=m_eff,
            batch_size=rollout_det_batch_size,
            device=device,
            smoothing_cfg=smoothing_cfg,
            rollout_steps=rollout_det_steps,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            per_traj_norm=per_traj_norm,
            history_window=history_window,
        )
        print(
            f"Enabled deterministic rollout loss: steps={rollout_det_steps}, "
            f"weight={rollout_det_weight:g}, windows={train_rollout_windows}, "
            f"rollout_batch_size={rollout_det_batch_size}"
        )

    val_series_raw: list[tuple[np.ndarray, np.ndarray, float, np.ndarray | None, np.ndarray | None, np.ndarray]] | None = None
    val_sequences: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] | None = None
    val_loader: Any | None = None
    val_rollout_loader: Any | None = None
    if use_generated_train_series:
        val_dir = train_series_root / "val"
        if val_dir.exists():
            val_cut = resolve_cut_start_seconds(data_cfg, "val")
            val_require_force = bool(use_force_data_loss or has_force_data)
            val_series_raw, _ = load_training_series(
                y_data,
                t,
                dt,
                True,
                val_dir,
                m_eff,
                device,
                smoothing_cfg=smoothing_cfg,
                velocity_source=velocity_source,
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
                smoothing_cfg=smoothing_cfg,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                per_traj_norm=per_traj_norm,
                per_traj_norm_eps=per_traj_norm_eps,
                history_window=history_window,
            )
            if rollout_det_weight > 0.0 and rollout_det_steps > 0:
                val_rollout_loader, _ = build_rollout_dataloader_from_series(
                    val_series_raw,
                    m_eff=m_eff,
                    batch_size=rollout_det_batch_size,
                    device=device,
                    smoothing_cfg=smoothing_cfg,
                    rollout_steps=rollout_det_steps,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    per_traj_norm=per_traj_norm,
                    history_window=history_window,
                )

    if use_generated_train_series:
        y_data_t, val_vel, _t_tensor, val_ur = eval_y_tensor, eval_vel_tensor, eval_t_tensor, eval_ur_tensor
    else:
        y_data_t, val_vel, _t_tensor, val_ur = train_sequences[0]

    writer, run_name = setup_writer(
        config.logging.run_dir_root,
        config_name,
        run_name_override=getattr(config.logging, "run_name", None),
        append_timestamp=bool(getattr(config.logging, "append_timestamp", True)),
    )
    async_processes: list[dict[str, Any]] = []
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
        names = ["residual", "sigma"]
        if mean_reg > 0.0:
            names.append("mean")
        if use_force_data_loss:
            names.append("data")
        gradnorm_balancer = GradNormBalancer(
            model,
            names,
            alpha=float(loss_cfg.gradnorm_alpha),
            eps=float(loss_cfg.gradnorm_eps),
            min_weight=float(loss_cfg.gradnorm_min_weight),
            max_weight=float(loss_cfg.gradnorm_max_weight),
        )

    amp_enabled, amp_dtype, scaler = setup_amp(
        device, use_amp=bool(precision_cfg.use_amp), amp_dtype=str(precision_cfg.amp_dtype)
    )

    for epoch in range(epochs):
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
            force_reg=force_reg,
            sigma_reg_norm=sigma_reg_norm,
            rollout_det_weight=rollout_det_weight,
            use_force_data_loss=use_force_data_loss,
            force_data_weight=force_data_weight,
            gradnorm_balancer=gradnorm_balancer,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            scaler=scaler,
            log_component_grad_norms=log_component_grad_norms,
            per_traj_norm_eps=per_traj_norm_eps,
            force_reg_on_coeff=force_reg_on_coeff,
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
        if "mean_gradnorm_weight_force" in epoch_metrics:
            train_metrics["gradnorm_weight_reg"] = float(epoch_metrics["mean_gradnorm_weight_force"])
        if "mean_gradnorm_weight_mean" in epoch_metrics:
            train_metrics["gradnorm_weight_mean_reg"] = float(epoch_metrics["mean_gradnorm_weight_mean"])
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
        should_validate_rollout = rollout_every_epochs > 0 and (
            (epoch + 1) % int(rollout_every_epochs) == 0 or epoch == (epochs - 1)
        )
        if async_validation and (should_validate_losses or should_validate_rollout) and (async_do_losses or async_do_rollout):
            async_processes = _reap_async_processes(async_processes, wait=False)
            state_source = model
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
                    "method": str(config.method),
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
                rollout_every_epochs=rollout_every_epochs,
                cycle_validation_rollout=cycle_validation_rollout,
                rollout_target_ur=rollout_target_ur,
                rollout_target_ur_tol=rollout_target_ur_tol,
                do_losses=async_do_losses and should_validate_losses,
                do_rollout=async_do_rollout and should_validate_rollout,
            )
        elif not async_validation:
            _validate_if_needed(
                writer=writer,
                epoch=epoch,
                rollout_every_epochs=rollout_every_epochs,
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
                cycle_validation_rollout=cycle_validation_rollout,
                fixed_validation_sampling=fixed_validation_sampling,
                validation_sampling_seed=validation_sampling_seed,
                rollout_target_ur=rollout_target_ur,
                rollout_target_ur_tol=rollout_target_ur_tol,
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
                force_reg=force_reg,
                sigma_reg_norm=sigma_reg_norm,
                rollout_det_weight=rollout_det_weight,
                use_force_data_loss=use_force_data_loss,
                force_data_weight=force_data_weight,
                symmetry_weight=symmetry_weight,
                symmetry_norm=symmetry_norm,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                per_traj_norm_eps=per_traj_norm_eps,
                force_reg_on_coeff=force_reg_on_coeff,
            )

    if async_validation and async_processes:
        print(f"Waiting for {len(async_processes)} async validation job(s) to finish...")
        async_processes = _reap_async_processes(async_processes, wait=True)

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
        )
        if used > 0 and avg_metrics:
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
        final_loss_by_ur = _per_ur_loss_map_hnn(
            model=model,
            loader=val_loader,
            rollout_loader=val_rollout_loader,
            device=device,
            non_blocking=(device.type == "cuda"),
            mean_reg=mean_reg,
            mean_reg_norm=mean_reg_norm,
            force_reg=force_reg,
            sigma_reg_norm=sigma_reg_norm,
            rollout_det_weight=rollout_det_weight,
            use_force_data_loss=use_force_data_loss,
            force_data_weight=force_data_weight,
            symmetry_weight=symmetry_weight,
            symmetry_norm=symmetry_norm,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            per_traj_norm_eps=per_traj_norm_eps,
            force_reg_on_coeff=force_reg_on_coeff,
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
    torch.save(
        {
            "model_state": state_source.state_dict(),
            "config": asdict(config),
            "run_name": run_name,
        },
        model_path,
    )
    print(f"Saved final model to {model_path}")

    writer.flush()
    writer.close()
