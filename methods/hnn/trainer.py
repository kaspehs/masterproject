from __future__ import annotations

import json
import os
import time
import subprocess
import sys
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
    build_dataloader_from_series,
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
)


def _train_one_epoch(
    *,
    model: torch.nn.Module,
    opt: optim.Optimizer,
    train_loader: Any,
    device: torch.device,
    non_blocking: bool,
    max_grad_norm: float,
    force_reg: float,
    use_force_data_loss: bool,
    force_data_weight: float,
    gradnorm_balancer: Optional[GradNormBalancer],
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    scaler: torch.amp.GradScaler,
    log_component_grad_norms: bool,
    per_traj_norm_eps: float,
    force_reg_on_coeff: bool,
) -> dict[str, float]:
    batch_count = 0
    loss_sum = torch.zeros((), device=device)
    res_loss_sum = torch.zeros((), device=device)
    force_loss_sum = torch.zeros((), device=device)
    force_data_loss_sum = torch.zeros((), device=device)
    grad_norm_sum = torch.zeros((), device=device)
    avg_force_sum = torch.zeros((), device=device)
    res_grad_component_sum = torch.zeros((), device=device)
    force_grad_component_sum = torch.zeros((), device=device)
    gradnorm_res_weight_sum = torch.zeros((), device=device)
    gradnorm_force_weight_sum = torch.zeros((), device=device)
    gradnorm_data_weight_sum = torch.zeros((), device=device) if use_force_data_loss else None
    gradnorm_weight_count = 0

    force_output_coeff = getattr(model, "force_output", "force") == "coefficient"
    use_tcn_force = bool(getattr(model, "is_tcn_force_model", False))
    for batch in train_loader:
        z_hist = None
        ur_hist = None
        if not use_tcn_force:
            if len(batch) == 5:
                z_i, t_i, z_next, t_next, ur_i = batch
                f_i = None
                f_next = None
                scale = None
            elif len(batch) == 6:
                z_i, t_i, z_next, t_next, ur_i, scale = batch
                f_i = None
                f_next = None
            elif len(batch) == 7:
                z_i, t_i, z_next, t_next, ur_i, f_i, f_next = batch
                scale = None
            elif len(batch) == 8:
                z_i, t_i, z_next, t_next, ur_i, f_i, f_next, scale = batch
            else:
                raise ValueError("Unexpected batch format from dataloader.")
        else:
            if len(batch) == 7:
                z_i, t_i, z_next, t_next, ur_i, z_hist, ur_hist = batch
                f_i = None
                f_next = None
                scale = None
            elif len(batch) == 8:
                z_i, t_i, z_next, t_next, ur_i, z_hist, ur_hist, scale = batch
                f_i = None
                f_next = None
            elif len(batch) == 9:
                z_i, t_i, z_next, t_next, ur_i, z_hist, ur_hist, f_i, f_next = batch
                scale = None
            elif len(batch) == 10:
                z_i, t_i, z_next, t_next, ur_i, z_hist, ur_hist, f_i, f_next, scale = batch
            else:
                raise ValueError("Unexpected TCN batch format from dataloader.")
        z_i = z_i.to(device, non_blocking=non_blocking)
        t_i = t_i.to(device, non_blocking=non_blocking)
        z_next = z_next.to(device, non_blocking=non_blocking)
        t_next = t_next.to(device, non_blocking=non_blocking)
        ur_i = ur_i.to(device, non_blocking=non_blocking)
        if z_hist is not None:
            z_hist = z_hist.to(device, non_blocking=non_blocking)
        if ur_hist is not None:
            ur_hist = ur_hist.to(device, non_blocking=non_blocking)
        if f_i is not None:
            f_i = f_i.to(device, non_blocking=non_blocking)
        if f_next is not None:
            f_next = f_next.to(device, non_blocking=non_blocking)
        if scale is not None:
            scale = scale.to(device, non_blocking=non_blocking).view(-1)

        opt.zero_grad()

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
            if scale is None:
                res_loss = model.res_loss(
                    z_i,
                    t_i,
                    z_next,
                    t_next,
                    reduced_velocity=ur_i,
                    z_hist=z_hist,
                    ur_hist=ur_hist,
                )
                if force_reg_on_coeff:
                    avg_force = model.avg_force_coeff(
                        z_i,
                        t_i,
                        z_next,
                        t_next,
                        reduced_velocity=ur_i,
                        z_hist=z_hist,
                        ur_hist=ur_hist,
                    )
                else:
                    avg_force = model.avg_force(
                        z_i,
                        t_i,
                        z_next,
                        t_next,
                        reduced_velocity=ur_i,
                        z_hist=z_hist,
                        ur_hist=ur_hist,
                    )
            else:
                per_res = model.res_loss_per_sample(
                    z_i,
                    t_i,
                    z_next,
                    t_next,
                    reduced_velocity=ur_i,
                    z_hist=z_hist,
                    ur_hist=ur_hist,
                )
                if force_reg_on_coeff:
                    per_force = model.avg_force_coeff_per_sample(
                        z_i,
                        t_i,
                        z_next,
                        t_next,
                        reduced_velocity=ur_i,
                        z_hist=z_hist,
                        ur_hist=ur_hist,
                    )
                else:
                    per_force = model.avg_force_per_sample(
                        z_i,
                        t_i,
                        z_next,
                        t_next,
                        reduced_velocity=ur_i,
                        z_hist=z_hist,
                        ur_hist=ur_hist,
                    )
                denom = scale * scale + float(per_traj_norm_eps)
                res_loss = torch.mean(per_res / denom)
                avg_force = torch.mean(per_force / denom)
            base_force_loss = avg_force
            if use_force_data_loss:
                if f_i is None or f_next is None:
                    raise ValueError(
                        "use_force_data_loss is True but the dataloader did not provide force labels."
                    )
                z_mid = 0.5 * (z_i + z_next)
                f_mid = 0.5 * (f_i + f_next)
                if force_output_coeff:
                    f0 = model._force_scale_from_reduced_velocity(ur_i, like=f_mid, state=z_mid)
                    f_pred = model.u_theta_coeff(
                        z_mid,
                        reduced_velocity=ur_i,
                        z_hist=z_hist,
                        ur_hist=ur_hist,
                    )
                    f_mid = f_mid / f0
                else:
                    f_pred = model.u_theta(
                        z_mid,
                        reduced_velocity=ur_i,
                        z_hist=z_hist,
                        ur_hist=ur_hist,
                    )
                per_data = torch.mean((f_pred - f_mid) ** 2, dim=1)
                if scale is not None:
                    per_data = per_data / (scale * scale + float(per_traj_norm_eps))
                data_force_loss = torch.mean(per_data)
            else:
                data_force_loss = res_loss.new_tensor(0.0)

            if gradnorm_balancer is not None:
                weights = gradnorm_balancer.update(
                    {
                        "residual": res_loss.float(),
                        "force": base_force_loss.float(),
                        "data": data_force_loss.float() if use_force_data_loss else res_loss.float(),
                    }
                )
                res_weight = weights["residual"]
                force_weight = weights["force"]
                data_weight = weights.get("data", res_loss.new_tensor(1.0))
                gradnorm_res_weight_sum = gradnorm_res_weight_sum + res_weight
                gradnorm_force_weight_sum = gradnorm_force_weight_sum + force_weight
                if gradnorm_data_weight_sum is not None:
                    gradnorm_data_weight_sum = gradnorm_data_weight_sum + data_weight
                gradnorm_weight_count += 1
            else:
                res_weight = res_loss.new_tensor(1.0)
                force_weight = res_loss.new_tensor(1.0)
                data_weight = res_loss.new_tensor(1.0)

            # GradNorm balances raw branch losses first; user multipliers are applied after.
            # This makes force_reg behave as a post-GradNorm scaling of the force branch.
            gradnorm_weighted_res = res_weight * res_loss
            gradnorm_weighted_force = force_weight * base_force_loss
            gradnorm_weighted_data = data_weight * data_force_loss

            force_loss = float(force_reg) * base_force_loss
            weighted_force = float(force_reg) * gradnorm_weighted_force
            weighted_data = float(force_data_weight) * gradnorm_weighted_data
            loss = (gradnorm_weighted_res + weighted_force + weighted_data).float()

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
            weighted_force.backward(retain_graph=True)
            force_grad_component_sum = force_grad_component_sum + torch.as_tensor(
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
        force_loss_sum = force_loss_sum + base_force_loss.detach().float()
        force_data_loss_sum = force_data_loss_sum + data_force_loss.detach().float()
        avg_force_sum = avg_force_sum + avg_force.detach().float()

    denom = float(max(batch_count, 1))
    metrics: dict[str, float] = {
        "mean_loss": float((loss_sum / denom).detach().cpu()),
        "mean_res_loss": float((res_loss_sum / denom).detach().cpu()),
        "mean_force_loss": float((force_loss_sum / denom).detach().cpu()),
        "mean_force_data_loss": float((force_data_loss_sum / denom).detach().cpu()),
        "mean_grad_norm": float((grad_norm_sum / denom).detach().cpu()),
        "mean_force": float((avg_force_sum / denom).detach().cpu()),
        "mean_res_grad_component": float((res_grad_component_sum / denom).detach().cpu()),
        "mean_force_grad_component": float((force_grad_component_sum / denom).detach().cpu()),
    }
    if gradnorm_weight_count > 0:
        metrics["mean_gradnorm_weight_residual"] = float(
            (gradnorm_res_weight_sum / float(gradnorm_weight_count)).detach().cpu()
        )
        metrics["mean_gradnorm_weight_force"] = float(
            (gradnorm_force_weight_sum / float(gradnorm_weight_count)).detach().cpu()
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
    model: PHVIV,
    y_data_t: torch.Tensor,
    val_vel: torch.Tensor,
    reduced_velocity: torch.Tensor,
    val_series_raw: list[tuple[np.ndarray, np.ndarray, float, np.ndarray | None, np.ndarray | None, np.ndarray]] | None,
    val_sequences: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] | None,
    val_loader: Any | None,
    cycle_validation_rollout: bool,
    m_eff: float,
    dt: float,
    t: np.ndarray,
    y_true_norm: np.ndarray,
    y_data: np.ndarray,
    force_data: np.ndarray,
    D: float,
    k: float,
    device: torch.device,
    middle_time_plot,
    hamiltonian_data,
    log_extra_validation_metrics: bool,
    log_loss_vs_ur_map: bool,
    rollout_include_disp_nrmse: bool,
    rollout_include_force_nrmse: bool,
    rollout_include_force_mapping_nrmse: bool,
    validation_start_seconds: float,
    history_context: int,
    force_reg: float,
    use_force_data_loss: bool,
    force_data_weight: float,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    per_traj_norm_eps: float,
    force_reg_on_coeff: bool,
) -> None:
    def _validation_start_index_for_series(series_t: np.ndarray, series_dt: float, *, label: str) -> int:
        t_arr = np.asarray(series_t, dtype=float).reshape(-1)
        if t_arr.size < 2:
            raise ValueError(f"{label}: not enough samples to validate.")
        t0 = float(t_arr[0])
        start_s = max(0.0, float(validation_start_seconds))
        start_idx = int(np.searchsorted(t_arr, t0 + start_s, side="left"))
        if history_context > 0 and start_idx < int(history_context):
            available_s = float(start_idx * series_dt)
            needed_s = float(int(history_context) * series_dt)
            raise ValueError(
                f"{label}: validation start at t={t0 + start_s:.6g}s (index={start_idx}) is too early for "
                f"TCN history_len={int(history_context)}. Need at least {needed_s:.6g}s "
                f"({int(history_context)} samples) before validation start, but only {available_s:.6g}s are available."
            )
        if (int(t_arr.size) - start_idx) < 2:
            raise ValueError(f"{label}: too few validation samples after validation start.")
        return start_idx

    if rollout_every_epochs <= 0:
        return
    if (epoch + 1) % int(rollout_every_epochs) != 0:
        return
    if val_loader is not None:
        val_loss_metrics = _evaluate_val_losses(
            model=model,
            loader=val_loader,
            device=device,
            non_blocking=(device.type == "cuda"),
            force_reg=force_reg,
            use_force_data_loss=use_force_data_loss,
            force_data_weight=force_data_weight,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            per_traj_norm_eps=per_traj_norm_eps,
            force_reg_on_coeff=force_reg_on_coeff,
        )
        for name, value in val_loss_metrics.items():
            writer.add_scalar(f"val/{name}", value, epoch + 1)
        if log_loss_vs_ur_map:
            loss_by_ur = _per_ur_loss_map_hnn(
                model=model,
                loader=val_loader,
                device=device,
                non_blocking=(device.type == "cuda"),
                force_reg=force_reg,
                use_force_data_loss=use_force_data_loss,
                force_data_weight=force_data_weight,
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

    if val_series_raw is not None and val_sequences is not None:
        total = min(len(val_series_raw), len(val_sequences))
        by_ur: dict[float, list[int]] = {}
        for idx in range(total):
            ur_np = np.asarray(val_series_raw[idx][5], dtype=float).reshape(-1)
            if ur_np.size == 0 or not np.isfinite(ur_np[0]):
                continue
            ur_key = float(np.round(float(ur_np[0]), 6))
            by_ur.setdefault(ur_key, []).append(idx)
        if not by_ur:
            return
        rng = np.random.default_rng(int(epoch) + 1)
        selected_indices: list[int] = []
        for ur_key in sorted(by_ur):
            candidates = np.asarray(by_ur[ur_key], dtype=int)
            selected_indices.append(int(rng.choice(candidates)))

        metrics_sum: dict[str, float] = {}
        count = 0
        for idx in selected_indices:
            series_raw = val_series_raw[idx]
            sequence = val_sequences[idx]
            y_np, t_np, dt_value, _vel_np, force_np, _ur_np = series_raw
            y_tensor, vel_tensor, _t_tensor, ur_tensor = sequence
            if force_np is None:
                continue
            val_start_idx = _validation_start_index_for_series(
                np.asarray(t_np),
                float(dt_value),
                label=f"validation series #{idx}",
            )
            t_eval = np.asarray(t_np)[val_start_idx:]
            y_eval = np.asarray(y_np)[val_start_idx:]
            force_eval = np.asarray(force_np)[val_start_idx:]
            metrics = compute_validation_metrics(
                model=model,
                y_data_t=y_tensor,
                val_vel=vel_tensor,
                reduced_velocity=ur_tensor,
                m_eff=m_eff,
                dt=dt_value,
                t=t_eval,
                y_data_raw=y_eval,
                force_data=force_eval,
                D=D,
                k=k,
                device=device,
                log_extra_metrics=log_extra_validation_metrics,
                include_disp_nrmse=rollout_include_disp_nrmse,
                include_force_nrmse=rollout_include_force_nrmse,
                include_force_mapping_nrmse=rollout_include_force_mapping_nrmse,
                validation_start_idx=val_start_idx,
            )
            for name, value in metrics.items():
                metrics_sum[name] = metrics_sum.get(name, 0.0) + float(value)
            count += 1
        if count > 0:
            for name, total in metrics_sum.items():
                writer.add_scalar(f"val/{name}", total / float(count), epoch + 1)
        if not selected_indices:
            return
        if cycle_validation_rollout:
            step = max(0, (epoch + 1) // max(1, int(rollout_every_epochs)) - 1)
            rollout_idx = selected_indices[step % len(selected_indices)]
        else:
            rollout_idx = selected_indices[0]
        series_raw = val_series_raw[rollout_idx]
        sequence = val_sequences[rollout_idx]
        y_np, t_np, dt_value, _vel_np, force_np, _ur_np = series_raw
        y_tensor, vel_tensor, _t_tensor, ur_tensor = sequence
        if force_np is None:
            return
        val_start_idx = _validation_start_index_for_series(
            np.asarray(t_np),
            float(dt_value),
            label=f"validation series #{rollout_idx}",
        )
        t_eval = np.asarray(t_np)[val_start_idx:]
        y_eval = np.asarray(y_np)[val_start_idx:]
        force_eval = np.asarray(force_np)[val_start_idx:]
        log_validation_epoch(
            writer,
            epoch + 1,
            model,
            y_tensor,
            vel_tensor,
            ur_tensor,
            m_eff,
            dt_value,
            t_eval,
            y_eval / D,
            y_eval,
            force_eval,
            D,
            k,
            device,
            middle_time_plot,
            hamiltonian_data,
            log_extra_metrics=log_extra_validation_metrics,
            include_disp_nrmse=rollout_include_disp_nrmse,
            include_force_nrmse=rollout_include_force_nrmse,
            include_force_mapping_nrmse=rollout_include_force_mapping_nrmse,
            log_metrics=False,
            validation_start_idx=val_start_idx,
        )
        return
    val_start_idx = 0
    t_eval = np.asarray(t)
    y_eval = np.asarray(y_data)
    force_eval = np.asarray(force_data)
    if history_context > 0:
        t_arr = np.asarray(t, dtype=float).reshape(-1)
        if t_arr.size < 2:
            raise ValueError("validation series: not enough samples to validate.")
        t0 = float(t_arr[0])
        start_s = max(0.0, float(validation_start_seconds))
        val_start_idx = int(np.searchsorted(t_arr, t0 + start_s, side="left"))
        if val_start_idx < int(history_context):
            available_s = float(val_start_idx * dt)
            needed_s = float(int(history_context) * dt)
            raise ValueError(
                f"validation series: validation start at t={t0 + start_s:.6g}s (index={val_start_idx}) is too early "
                f"for TCN history_len={int(history_context)}. Need at least {needed_s:.6g}s "
                f"({int(history_context)} samples) before validation start, but only {available_s:.6g}s are available."
            )
        if (int(t_arr.size) - val_start_idx) < 2:
            raise ValueError("validation series: too few validation samples after validation start.")
        t_eval = t_arr[val_start_idx:]
        y_eval = np.asarray(y_data)[val_start_idx:]
        force_eval = np.asarray(force_data)[val_start_idx:]

    log_validation_epoch(
        writer,
        epoch + 1,
        model,
        y_data_t,
        val_vel,
        reduced_velocity,
        m_eff,
        dt,
        t_eval,
        y_eval / D,
        y_eval,
        force_eval,
        D,
        k,
        device,
        middle_time_plot,
        hamiltonian_data,
        log_extra_metrics=log_extra_validation_metrics,
        include_disp_nrmse=rollout_include_disp_nrmse,
        include_force_nrmse=rollout_include_force_nrmse,
        include_force_mapping_nrmse=rollout_include_force_mapping_nrmse,
        validation_start_idx=val_start_idx,
    )


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
    rollout_include_disp_nrmse: bool,
    rollout_include_force_nrmse: bool,
    rollout_include_force_mapping_nrmse: bool,
    validation_start_seconds: float,
    history_context: int,
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
        if force_np is None:
            continue
        y_tensor, vel_tensor, _t_tensor, ur_tensor = val_sequences[idx]
        t_arr = np.asarray(t_np, dtype=float).reshape(-1)
        t0 = float(t_arr[0])
        start_s = max(0.0, float(validation_start_seconds))
        val_start_idx = int(np.searchsorted(t_arr, t0 + start_s, side="left"))
        if history_context > 0 and val_start_idx < int(history_context):
            available_s = float(val_start_idx * float(dt_value))
            needed_s = float(int(history_context) * float(dt_value))
            raise ValueError(
                f"validation series #{idx}: validation start at t={t0 + start_s:.6g}s (index={val_start_idx}) "
                f"is too early for TCN history_len={int(history_context)}. Need at least {needed_s:.6g}s "
                f"({int(history_context)} samples) before validation start, but only {available_s:.6g}s are available."
            )
        if (int(t_arr.size) - val_start_idx) < 2:
            raise ValueError(f"validation series #{idx}: too few validation samples after validation start.")
        t_eval = t_arr[val_start_idx:]
        y_eval = np.asarray(y_np)[val_start_idx:]
        force_eval = np.asarray(force_np)[val_start_idx:]
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
            t_eval,
            y_eval / D,
            y_eval,
            force_eval,
            D,
            k,
            device,
            middle_time_plot,
            None,
            log_extra_metrics=log_extra_validation_metrics,
            include_disp_nrmse=rollout_include_disp_nrmse,
            include_force_nrmse=rollout_include_force_nrmse,
            include_force_mapping_nrmse=rollout_include_force_mapping_nrmse,
            log_metrics=False,
            tag_prefix="final_val/rollout",
            step=step_idx,
            title_suffix=f" [final {step_idx+1}/{len(selected_indices)}]",
            validation_start_idx=val_start_idx,
        )
        if metrics:
            ur_values.append(ur_val)
            metrics_list.append(metrics)
        for name, value in metrics.items():
            if not np.isfinite(value):
                continue
            metrics_sum[name] = metrics_sum.get(name, 0.0) + float(value)
            metrics_count[name] = metrics_count.get(name, 0) + 1
        used += 1
    averaged = {
        name: metrics_sum[name] / float(metrics_count[name])
        for name in metrics_sum
        if metrics_count.get(name, 0) > 0
    }
    return averaged, used, ur_values, metrics_list


def _prune_async_processes(processes: list[subprocess.Popen]) -> list[subprocess.Popen]:
    alive: list[subprocess.Popen] = []
    for proc in processes:
        rc = proc.poll()
        if rc is None:
            alive.append(proc)
            continue
        if rc != 0:
            print(f"[ASYNC-VALIDATION][ERROR] Worker pid={proc.pid} exited with code {rc}.")
        else:
            print(f"[ASYNC-VALIDATION][OK] Worker pid={proc.pid} finished successfully.")
    return alive


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


def _evaluate_val_losses(
    *,
    model: PHVIV,
    loader: Any,
    device: torch.device,
    non_blocking: bool,
    force_reg: float,
    use_force_data_loss: bool,
    force_data_weight: float,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    per_traj_norm_eps: float,
    force_reg_on_coeff: bool,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    force_output_coeff = getattr(model, "force_output", "force") == "coefficient"
    use_tcn_force = bool(getattr(model, "is_tcn_force_model", False))
    loss_sum = torch.zeros((), device=device)
    res_sum = torch.zeros((), device=device)
    force_sum = torch.zeros((), device=device)
    data_sum = torch.zeros((), device=device)
    batches = 0
    with torch.no_grad():
        for batch in loader:
            z_hist = None
            ur_hist = None
            if not use_tcn_force:
                if len(batch) == 5:
                    z_i, t_i, z_next, t_next, ur_i = batch
                    f_i = None
                    f_next = None
                    scale = None
                elif len(batch) == 6:
                    z_i, t_i, z_next, t_next, ur_i, scale = batch
                    f_i = None
                    f_next = None
                elif len(batch) == 7:
                    z_i, t_i, z_next, t_next, ur_i, f_i, f_next = batch
                    scale = None
                elif len(batch) == 8:
                    z_i, t_i, z_next, t_next, ur_i, f_i, f_next, scale = batch
                else:
                    raise ValueError("Unexpected batch format from dataloader.")
            else:
                if len(batch) == 7:
                    z_i, t_i, z_next, t_next, ur_i, z_hist, ur_hist = batch
                    f_i = None
                    f_next = None
                    scale = None
                elif len(batch) == 8:
                    z_i, t_i, z_next, t_next, ur_i, z_hist, ur_hist, scale = batch
                    f_i = None
                    f_next = None
                elif len(batch) == 9:
                    z_i, t_i, z_next, t_next, ur_i, z_hist, ur_hist, f_i, f_next = batch
                    scale = None
                elif len(batch) == 10:
                    z_i, t_i, z_next, t_next, ur_i, z_hist, ur_hist, f_i, f_next, scale = batch
                else:
                    raise ValueError("Unexpected TCN batch format from dataloader.")
            z_i = z_i.to(device, non_blocking=non_blocking)
            t_i = t_i.to(device, non_blocking=non_blocking)
            z_next = z_next.to(device, non_blocking=non_blocking)
            t_next = t_next.to(device, non_blocking=non_blocking)
            ur_i = ur_i.to(device, non_blocking=non_blocking)
            if z_hist is not None:
                z_hist = z_hist.to(device, non_blocking=non_blocking)
            if ur_hist is not None:
                ur_hist = ur_hist.to(device, non_blocking=non_blocking)
            if f_i is not None:
                f_i = f_i.to(device, non_blocking=non_blocking)
            if f_next is not None:
                f_next = f_next.to(device, non_blocking=non_blocking)
            if scale is not None:
                scale = scale.to(device, non_blocking=non_blocking).view(-1)

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                if scale is None:
                    res_loss = model.res_loss(
                        z_i,
                        t_i,
                        z_next,
                        t_next,
                        reduced_velocity=ur_i,
                        z_hist=z_hist,
                        ur_hist=ur_hist,
                    )
                    if force_reg_on_coeff:
                        avg_force = model.avg_force_coeff(
                            z_i,
                            t_i,
                            z_next,
                            t_next,
                            reduced_velocity=ur_i,
                            z_hist=z_hist,
                            ur_hist=ur_hist,
                        )
                    else:
                        avg_force = model.avg_force(
                            z_i,
                            t_i,
                            z_next,
                            t_next,
                            reduced_velocity=ur_i,
                            z_hist=z_hist,
                            ur_hist=ur_hist,
                        )
                else:
                    per_res = model.res_loss_per_sample(
                        z_i,
                        t_i,
                        z_next,
                        t_next,
                        reduced_velocity=ur_i,
                        z_hist=z_hist,
                        ur_hist=ur_hist,
                    )
                    if force_reg_on_coeff:
                        per_force = model.avg_force_coeff_per_sample(
                            z_i,
                            t_i,
                            z_next,
                            t_next,
                            reduced_velocity=ur_i,
                            z_hist=z_hist,
                            ur_hist=ur_hist,
                        )
                    else:
                        per_force = model.avg_force_per_sample(
                            z_i,
                            t_i,
                            z_next,
                            t_next,
                            reduced_velocity=ur_i,
                            z_hist=z_hist,
                            ur_hist=ur_hist,
                        )
                    denom = scale * scale + float(per_traj_norm_eps)
                    res_loss = torch.mean(per_res / denom)
                    avg_force = torch.mean(per_force / denom)
                force_loss = float(force_reg) * avg_force
                if use_force_data_loss:
                    if f_i is None or f_next is None:
                        raise ValueError(
                            "use_force_data_loss is True but the dataloader did not provide force labels."
                        )
                    z_mid = 0.5 * (z_i + z_next)
                    f_mid = 0.5 * (f_i + f_next)
                    if force_output_coeff:
                        f0 = model._force_scale_from_reduced_velocity(ur_i, like=f_mid, state=z_mid)
                        f_pred = model.u_theta_coeff(
                            z_mid,
                            reduced_velocity=ur_i,
                            z_hist=z_hist,
                            ur_hist=ur_hist,
                        )
                        f_mid = f_mid / f0
                    else:
                        f_pred = model.u_theta(
                            z_mid,
                            reduced_velocity=ur_i,
                            z_hist=z_hist,
                            ur_hist=ur_hist,
                        )
                    per_data = torch.mean((f_pred - f_mid) ** 2, dim=1)
                    if scale is not None:
                        per_data = per_data / (scale * scale + float(per_traj_norm_eps))
                    data_force_loss = torch.mean(per_data)
                else:
                    data_force_loss = res_loss.new_tensor(0.0)
                total = res_loss + force_loss + float(force_data_weight) * data_force_loss

            loss_sum = loss_sum + total.detach().float()
            res_sum = res_sum + res_loss.detach().float()
            # Log loss_reg as the raw regularizer magnitude (before force_reg scaling).
            force_sum = force_sum + avg_force.detach().float()
            data_sum = data_sum + data_force_loss.detach().float()
            batches += 1

    if was_training:
        model.train()
    denom = float(max(batches, 1))
    return {
        "loss_total": float((loss_sum / denom).detach().cpu()),
        "loss_physics": float((res_sum / denom).detach().cpu()),
        "loss_reg": float((force_sum / denom).detach().cpu()),
        "loss_data": float((data_sum / denom).detach().cpu()),
    }


def _per_ur_loss_map_hnn(
    *,
    model: PHVIV,
    loader: Any,
    device: torch.device,
    non_blocking: bool,
    force_reg: float,
    use_force_data_loss: bool,
    force_data_weight: float,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    per_traj_norm_eps: float,
    force_reg_on_coeff: bool,
) -> dict[str, dict[float, float]]:
    model.eval()
    amp_enabled = bool(amp_enabled) and device.type == "cuda"
    force_output_coeff = getattr(model, "force_output", "force") == "coefficient"
    use_tcn_force = bool(getattr(model, "is_tcn_force_model", False))
    buckets: dict[str, dict[float, list[float]]] = {
        "loss_physics": {},
        "loss_reg": {},
        "loss_data": {},
    }
    with torch.no_grad():
        for batch in loader:
            z_hist = None
            ur_hist = None
            if not use_tcn_force:
                if len(batch) == 5:
                    z_i, t_i, z_next, t_next, ur_i = batch
                    f_i = None
                    f_next = None
                    scale = None
                elif len(batch) == 6:
                    z_i, t_i, z_next, t_next, ur_i, scale = batch
                    f_i = None
                    f_next = None
                elif len(batch) == 7:
                    z_i, t_i, z_next, t_next, ur_i, f_i, f_next = batch
                    scale = None
                elif len(batch) == 8:
                    z_i, t_i, z_next, t_next, ur_i, f_i, f_next, scale = batch
                else:
                    raise ValueError("Unexpected batch format from dataloader.")
            else:
                if len(batch) == 7:
                    z_i, t_i, z_next, t_next, ur_i, z_hist, ur_hist = batch
                    f_i = None
                    f_next = None
                    scale = None
                elif len(batch) == 8:
                    z_i, t_i, z_next, t_next, ur_i, z_hist, ur_hist, scale = batch
                    f_i = None
                    f_next = None
                elif len(batch) == 9:
                    z_i, t_i, z_next, t_next, ur_i, z_hist, ur_hist, f_i, f_next = batch
                    scale = None
                elif len(batch) == 10:
                    z_i, t_i, z_next, t_next, ur_i, z_hist, ur_hist, f_i, f_next, scale = batch
                else:
                    raise ValueError("Unexpected TCN batch format from dataloader.")
            z_i = z_i.to(device, non_blocking=non_blocking)
            t_i = t_i.to(device, non_blocking=non_blocking)
            z_next = z_next.to(device, non_blocking=non_blocking)
            t_next = t_next.to(device, non_blocking=non_blocking)
            ur_i = ur_i.to(device, non_blocking=non_blocking)
            if z_hist is not None:
                z_hist = z_hist.to(device, non_blocking=non_blocking)
            if ur_hist is not None:
                ur_hist = ur_hist.to(device, non_blocking=non_blocking)
            if f_i is not None:
                f_i = f_i.to(device, non_blocking=non_blocking)
            if f_next is not None:
                f_next = f_next.to(device, non_blocking=non_blocking)
            if scale is not None:
                scale = scale.to(device, non_blocking=non_blocking).view(-1)

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                per_res = model.res_loss_per_sample(
                    z_i,
                    t_i,
                    z_next,
                    t_next,
                    reduced_velocity=ur_i,
                    z_hist=z_hist,
                    ur_hist=ur_hist,
                )
                if force_reg_on_coeff:
                    per_force = model.avg_force_coeff_per_sample(
                        z_i,
                        t_i,
                        z_next,
                        t_next,
                        reduced_velocity=ur_i,
                        z_hist=z_hist,
                        ur_hist=ur_hist,
                    )
                else:
                    per_force = model.avg_force_per_sample(
                        z_i,
                        t_i,
                        z_next,
                        t_next,
                        reduced_velocity=ur_i,
                        z_hist=z_hist,
                        ur_hist=ur_hist,
                    )
                if scale is not None:
                    denom = scale * scale + float(per_traj_norm_eps)
                    per_res = per_res / denom
                    per_force = per_force / denom
                if use_force_data_loss and f_i is not None and f_next is not None:
                    z_mid = 0.5 * (z_i + z_next)
                    f_mid = 0.5 * (f_i + f_next)
                    if force_output_coeff:
                        f0 = model._force_scale_from_reduced_velocity(ur_i, like=f_mid, state=z_mid)
                        f_pred = model.u_theta_coeff(
                            z_mid,
                            reduced_velocity=ur_i,
                            z_hist=z_hist,
                            ur_hist=ur_hist,
                        )
                        f_mid = f_mid / f0
                    else:
                        f_pred = model.u_theta(
                            z_mid,
                            reduced_velocity=ur_i,
                            z_hist=z_hist,
                            ur_hist=ur_hist,
                        )
                    per_data = torch.mean((f_pred - f_mid) ** 2, dim=1)
                    if scale is not None:
                        per_data = per_data / (scale * scale + float(per_traj_norm_eps))
                    per_data = float(force_data_weight) * per_data
                else:
                    per_data = per_res.new_zeros(per_res.shape)

            ur_vals = ur_i.detach().cpu().view(-1).numpy()
            per_res_vals = per_res.detach().cpu().view(-1).numpy()
            per_reg_vals = per_force.detach().cpu().view(-1).numpy()
            per_data_vals = per_data.detach().cpu().view(-1).numpy()
            for u, res_v, reg_v, data_v in zip(ur_vals, per_res_vals, per_reg_vals, per_data_vals):
                key = float(np.round(u, 6))
                buckets["loss_physics"].setdefault(key, []).append(float(res_v))
                buckets["loss_reg"].setdefault(key, []).append(float(reg_v))
                buckets["loss_data"].setdefault(key, []).append(float(data_v))

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
    F_data = data["c"]
    H_data = data["d"] if "d" in data else None
    if H_data is None:
        print(f"{data_path}: no Hamiltonian channel 'd' found. Continuing without Hamiltonian data.")
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
    physics_loss_discretization = str(getattr(loss_cfg, "physics_loss_discretization", "srk4"))
    force_reg_on_coeff = bool(getattr(loss_cfg, "force_reg_on_coeff", False))
    use_force_data_loss = bool(getattr(loss_cfg, "use_force_data_loss", False))
    force_data_weight = float(getattr(loss_cfg, "force_data_weight", 1.0))

    rollout_every_epochs = int(monitoring_cfg.rollout_every_epochs)
    cycle_validation_rollout = bool(getattr(monitoring_cfg, "cycle_validation_rollout", False))
    log_every_epochs = max(1, int(monitoring_cfg.log_every_epochs))
    print_every_epochs = max(1, int(monitoring_cfg.print_every_epochs))
    log_component_grad_norms = bool(monitoring_cfg.log_component_grad_norms)
    log_loss_vs_ur_map = bool(getattr(monitoring_cfg, "log_loss_vs_ur_map", True))
    log_extra_validation_metrics = bool(getattr(monitoring_cfg, "log_extra_validation_metrics", False))
    rollout_include_disp_nrmse = bool(getattr(monitoring_cfg, "rollout_include_disp_nrmse", True))
    rollout_include_force_nrmse = bool(getattr(monitoring_cfg, "rollout_include_force_nrmse", True))
    rollout_include_force_mapping_nrmse = bool(
        getattr(monitoring_cfg, "rollout_include_force_mapping_nrmse", True)
    )
    final_rollout_all_validation = bool(getattr(monitoring_cfg, "final_rollout_all_validation", False))
    async_validation = bool(getattr(monitoring_cfg, "async_validation", False))
    async_device = str(getattr(monitoring_cfg, "async_validation_device", "cpu"))
    async_num_workers = int(getattr(monitoring_cfg, "async_validation_num_workers", 0))
    async_num_threads = int(getattr(monitoring_cfg, "async_validation_num_threads", 4))
    async_max_concurrent = int(getattr(monitoring_cfg, "async_validation_max_concurrent", 1))
    async_do_losses = bool(getattr(monitoring_cfg, "async_validation_do_losses", True))
    async_do_rollout = bool(getattr(monitoring_cfg, "async_validation_do_rollout", True))

    device = select_device(os.getenv("TRAIN_DEVICE", str(runtime_cfg.device)))
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA available: {torch.cuda.is_available()}, gpu0: {torch.cuda.get_device_name(0)}")
    configure_tf32(device, bool(precision_cfg.use_tf32))
    set_num_threads_from_slurm(default=1)
    non_blocking = device.type == "cuda"

    model_dict = asdict(model_cfg)
    model_dict["physics_loss_discretization"] = physics_loss_discretization
    arch_dict = asdict(config.architecture)
    model, derived_params = PHVIV.from_config(dt=dt, cfg=model_dict, arch_cfg=arch_dict, device=device)
    model.set_loss_discretization(physics_loss_discretization)
    history_context = int(getattr(model, "history_len", 0)) if bool(getattr(model, "is_tcn_force_model", False)) else 0
    if history_context > 0:
        print(f"PHNN TCN context enabled: history_len={history_context}")
    print(f"PHNN physics loss discretization: {model.loss_discretization}")
    model = maybe_compile_model(model, bool(compile_cfg.use_compile), str(compile_cfg.compile_mode))
    try:
        setattr(model, "is_tcn_force_model", history_context > 0)
        setattr(model, "history_len", history_context)
        if hasattr(model, "set_loss_discretization"):
            model.set_loss_discretization(physics_loss_discretization)
        else:
            setattr(model, "loss_discretization", physics_loss_discretization)
    except Exception:
        pass
    D = derived_params["D"]
    k = derived_params["k"]
    m_eff = derived_params["m_eff"]

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
        eval_force=F_data,
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
        history_len=history_context,
    )

    val_series_raw: list[tuple[np.ndarray, np.ndarray, float, np.ndarray | None, np.ndarray | None, np.ndarray]] | None = None
    val_sequences: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] | None = None
    val_loader: Any | None = None
    if use_generated_train_series:
        val_dir = train_series_root / "val"
        if val_dir.exists():
            val_cut = resolve_cut_start_seconds(data_cfg, "val")
            val_cut_for_load = 0.0 if history_context > 0 else val_cut
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
                require_force=True,
                eval_force=F_data,
                cut_start_seconds=val_cut_for_load,
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
                history_len=history_context,
            )

    train_instances = int(len(train_loader.dataset))
    train_steps_per_epoch = (train_instances + batch_size - 1) // batch_size
    val_instances = int(len(val_loader.dataset)) if val_loader is not None else 0
    train_traj_count = int(len(train_series_raw))
    val_traj_count = int(len(val_series_raw)) if val_series_raw is not None else 0
    print(
        f"PHNN data summary: train trajectories={train_traj_count}, train instances={train_instances}, "
        f"val trajectories={val_traj_count}, val instances={val_instances}."
    )
    print(f"PHNN optimization summary: batch_size={batch_size}, steps_per_epoch={train_steps_per_epoch}.")

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
    async_processes: list[subprocess.Popen] = []
    async_dir = Path(writer.log_dir) / "async_validation"
    if async_validation:
        async_dir.mkdir(parents=True, exist_ok=True)

    y_true_norm = y_data / D
    force_data = F_data

    opt, lr_scheduler = setup_optimizer_and_scheduler(
        model,
        optim_cfg=optim_cfg,
        scheduler_cfg=scheduler_cfg,
        epochs=epochs,
    )

    gradnorm_balancer: Optional[GradNormBalancer] = None
    if bool(loss_cfg.use_gradnorm):
        names = ["residual", "force"]
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
            device=device,
            non_blocking=non_blocking,
            max_grad_norm=max_grad_norm,
            force_reg=force_reg,
            use_force_data_loss=use_force_data_loss,
            force_data_weight=force_data_weight,
            gradnorm_balancer=gradnorm_balancer,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            scaler=scaler,
            log_component_grad_norms=log_component_grad_norms,
            per_traj_norm_eps=per_traj_norm_eps,
            force_reg_on_coeff=force_reg_on_coeff,
        )

        mean_loss = epoch_metrics["mean_loss"]
        mean_res_loss = epoch_metrics["mean_res_loss"]
        mean_force_loss = epoch_metrics["mean_force_loss"]
        mean_force_data_loss = epoch_metrics["mean_force_data_loss"]
        mean_grad_norm = epoch_metrics["mean_grad_norm"]
        mean_force = epoch_metrics["mean_force"]
        mean_res_grad_component = epoch_metrics["mean_res_grad_component"]
        mean_force_grad_component = epoch_metrics["mean_force_grad_component"]

        current_lr = float(opt.param_groups[0]["lr"]) if opt.param_groups else lr

        train_metrics: dict[str, float] = {
            "loss_total": mean_loss,
            "loss_physics": mean_res_loss,
            "loss_reg": mean_force_loss,
            "loss_data": mean_force_data_loss,
            "lr": current_lr,
            "grad_norm": mean_grad_norm,
            "avg_force": mean_force,
        }
        if log_component_grad_norms:
            train_metrics["grad_norm_residual_comp"] = mean_res_grad_component
            train_metrics["grad_norm_force_comp"] = mean_force_grad_component
        if "mean_gradnorm_weight_residual" in epoch_metrics:
            train_metrics["gradnorm_weight_physics"] = float(epoch_metrics["mean_gradnorm_weight_residual"])
        if "mean_gradnorm_weight_force" in epoch_metrics:
            train_metrics["gradnorm_weight_reg"] = float(epoch_metrics["mean_gradnorm_weight_force"])
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
                    f"force={mean_force_loss:.4e}, data={mean_force_data_loss:.4e}"
                )
            else:
                print(f"Epoch {epoch}: loss={mean_loss:.4e}, res={mean_res_loss:.4e}, force={mean_force_loss:.4e}")

        should_validate = rollout_every_epochs > 0 and (
            (epoch + 1) % int(rollout_every_epochs) == 0 or epoch == (epochs - 1)
        )
        validation_timer_start: float | None = None
        if should_validate and not async_validation:
            validation_timer_start = time.perf_counter()
        if async_validation and should_validate and (async_do_losses or async_do_rollout):
            async_processes = _prune_async_processes(async_processes)
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
                do_losses=async_do_losses,
                do_rollout=async_do_rollout,
            )
        elif not async_validation:
            _validate_if_needed(
                writer=writer,
                epoch=epoch,
                rollout_every_epochs=rollout_every_epochs,
                model=model,
                y_data_t=y_data_t,
                val_vel=val_vel,
                reduced_velocity=val_ur,
                val_series_raw=val_series_raw,
                val_sequences=val_sequences,
                val_loader=val_loader,
                cycle_validation_rollout=cycle_validation_rollout,
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
                log_loss_vs_ur_map=log_loss_vs_ur_map,
                rollout_include_disp_nrmse=rollout_include_disp_nrmse,
                rollout_include_force_nrmse=rollout_include_force_nrmse,
                rollout_include_force_mapping_nrmse=rollout_include_force_mapping_nrmse,
                validation_start_seconds=resolve_cut_start_seconds(data_cfg, "val"),
                history_context=history_context,
                force_reg=force_reg,
                use_force_data_loss=use_force_data_loss,
                force_data_weight=force_data_weight,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                per_traj_norm_eps=per_traj_norm_eps,
                force_reg_on_coeff=force_reg_on_coeff,
            )
            if validation_timer_start is not None:
                validation_elapsed = time.perf_counter() - validation_timer_start
                writer.add_scalar("val/validation_wall_time_s", float(validation_elapsed), epoch + 1)
                print(f"Validation epoch {epoch + 1}: total wall time {validation_elapsed:.2f}s")

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
            rollout_include_disp_nrmse=rollout_include_disp_nrmse,
            rollout_include_force_nrmse=rollout_include_force_nrmse,
            rollout_include_force_mapping_nrmse=rollout_include_force_mapping_nrmse,
            validation_start_seconds=resolve_cut_start_seconds(data_cfg, "val"),
            history_context=history_context,
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

    if val_loader is not None and log_loss_vs_ur_map:
        final_loss_by_ur = _per_ur_loss_map_hnn(
            model=model,
            loader=val_loader,
            device=device,
            non_blocking=(device.type == "cuda"),
            force_reg=force_reg,
            use_force_data_loss=use_force_data_loss,
            force_data_weight=force_data_weight,
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
    if async_validation:
        async_processes = _prune_async_processes(async_processes)
        if async_processes:
            print(
                f"[ASYNC-VALIDATION][INFO] {len(async_processes)} worker(s) still running at shutdown."
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
