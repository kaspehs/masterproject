"""
Asynchronous validation runner.

Loads a checkpoint saved during training and runs validation on the unseen val split.
Intended to be spawned as a child process so training can continue.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Optional
import os
import sys

import numpy as np
import torch
from torch.utils.data import ConcatDataset, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from dataclasses import asdict

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from HNN_helper import (
    FORCE_MAPPING_NRMSE_KEY,
    PHVIV,
    ROLLOUT_DIVERGED_COUNT_KEY,
    ROLLOUT_DIVERGED_KEY,
    create_window_mask,
    create_zoom_mask,
    load_td_correction_trajectories,
    lookup_ur_bin_state_scale_tensor,
    build_dataloader_from_series,
    build_rollout_dataloader_from_series,
    compute_validation_metrics,
    load_training_series,
    log_displacement_plots,
    log_force_plots,
    log_loss_vs_ur,
    log_validation_epoch,
    parse_config,
    preprocess_timeseries,
    resolve_phnn_input_scaling_mode,
    resolve_cut_start_seconds,
    sample_indices_per_ur,
    scaled_residual_loss_per_sample,
    sample_one_index_per_ur,
    resolve_td_correction_params,
    resolve_td_correction_mode,
    resolve_td_force_input_source,
    resolve_td_phase_input_source,
    resolve_td_memory_config,
    td_correction_mode_flags,
)
from methods.hnn.trainer import (
    _build_td_correction_hnn_loaders,
    _log_td_correction_rollout_validation as _hnn_td_rollout_validation,
    _normalize_rollout_disp_spectral_loss_mode,
    _resolve_td_rollout_loss_settings,
    _td_correction_rollout_losses_from_batch,
    _td_step_with_corrections,
    _td_state_mse_loss,
    _td_state_propagated_nll_loss,
)
from methods.vpinn.trainer import (
    _force_mapping_nrmse_over_trajs,
    ScaledForceWrapper,
    WindowDataset,
    _as_diag_param,
    _build_force_model,
    _evaluate_epoch,
    _infer_dt_target_from_data_cfg,
    _load_trajectory,
    _m_eff_from_model_cfg,
    _test_functions,
    _log_rollout_validation,
    _log_td_correction_rollout_validation,
    _td_rollout_traj_to_tensors,
    _build_td_correction_vpinn_datasets,
    _vpinn_input_dim,
    _vpinn_output_dim,
    _vpinn_optional_hidden_inputs_from_context,
    _vpinn_step_with_corrections,
    _vpinn_rollout_state_loss,
    _vpinn_scale_weak_residual,
    _vpinn_td_rollout,
    rollout_rk4,
    _weak_residual,
)

ASYNC_VAL_SPLIT_TAG = "val_unseen"


def _set_threading(num_threads: int) -> None:
    num_threads = max(1, int(num_threads))
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(max(1, min(4, num_threads)))


def _rollout_index(
    num_series: int,
    *,
    ur_values: list[float] | None = None,
) -> int:
    if num_series <= 0:
        return 0
    selected = list(range(num_series))
    if ur_values is not None and len(ur_values) == num_series:
        sampled = sample_one_index_per_ur(ur_values, seed=0)
        if sampled:
            selected = sampled
    return int(selected[0])


def _async_summary_path(log_dir: Path, epoch: int) -> Path:
    return Path(log_dir) / "async_validation" / "results" / f"epoch_{int(epoch):06d}.json"


def _resolve_val_unseen_dir(train_series_root: Path) -> Path:
    val_unseen_dir = train_series_root / ASYNC_VAL_SPLIT_TAG
    if val_unseen_dir.exists():
        return val_unseen_dir
    legacy_val_dir = train_series_root / "val"
    if legacy_val_dir.exists():
        return legacy_val_dir
    raise FileNotFoundError(
        f"Validation directory '{val_unseen_dir}' not found and legacy fallback '{legacy_val_dir}' is also missing."
    )


def _resolve_optional_val_split_dir(train_series_root: Path, split_tag: str) -> Path | None:
    split_dir = train_series_root / str(split_tag).strip()
    return split_dir if split_dir.exists() else None


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


def _load_checkpoint(path: Path) -> tuple[dict[str, Any], Any, str]:
    ckpt = torch.load(path, map_location="cpu")
    cfg_raw = ckpt.get("config", {})
    cfg = parse_config(cfg_raw)
    method = str(ckpt.get("method", cfg.method)).strip().lower()
    return ckpt, cfg, method


def _load_state(model: torch.nn.Module, state: dict[str, Any]) -> None:
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    if any(k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)


def _td_hnn_traj_to_tensors(
    traj: dict[str, Any],
    *,
    mass_source: str,
) -> dict[str, Any]:
    mass_key = "dry_mass_kg" if str(mass_source).strip().lower() == "dry" else "effective_mass_kg"
    mass_value = float(np.asarray(traj[mass_key]).reshape(()))
    y_t = torch.from_numpy(np.ascontiguousarray(traj["y"])).float().unsqueeze(1)
    v_t = torch.from_numpy(np.ascontiguousarray(traj["dy"])).float().unsqueeze(1)
    z_t = torch.cat([y_t, v_t * mass_value], dim=1)
    return {
        "name": str(traj.get("name", "")),
        "y": y_t,
        "v": v_t,
        "z": z_t,
        "f": torch.from_numpy(np.ascontiguousarray(traj["force_per_m"])).float().unsqueeze(1),
        "td_force": torch.from_numpy(np.ascontiguousarray(traj["force_td_per_m"])).float().unsqueeze(1),
        "ur": torch.from_numpy(np.ascontiguousarray(traj["ur"])).float().unsqueeze(1),
        "td_context": torch.from_numpy(np.ascontiguousarray(traj["td_context"])).float(),
        "t": torch.from_numpy(np.ascontiguousarray(traj["t"])).float(),
        "mass_value": mass_value,
        "damping_value": float(np.asarray(traj["damping_c"]).reshape(())),
        "stiffness_value": float(np.asarray(traj["stiffness_n_m"]).reshape(())),
    }


def _log_td_correction_hnn_rollout_validation(
    *,
    writer: SummaryWriter,
    epoch: int,
    model: PHVIV,
    traj: dict[str, Any],
    dt: float,
    td_mass_source: str,
    td_params: dict[str, float],
    td_memory_cfg: dict[str, Any],
    device: torch.device,
    tag_prefix: str = f"{ASYNC_VAL_SPLIT_TAG}/rollout",
    step: int | None = None,
    log_metrics: bool = True,
    log_plots: bool = True,
    title_suffix: str = "",
) -> dict[str, float]:
    correction_mode = str(
        getattr(
            model,
            "correction_mode",
            ("mean_sigma_only" if bool(getattr(model, "use_stochastic_process_noise", False)) else "mean_only"),
        )
    )
    mode_flags = td_correction_mode_flags(correction_mode)
    return _hnn_td_rollout_validation(
        writer=writer,
        epoch=epoch,
        model=model,
        traj=traj,
        dt=dt,
        td_mass_source=td_mass_source,
        td_params=td_params,
        td_memory_cfg=td_memory_cfg,
        device=device,
        mean_active=bool(mode_flags["mean_active"]),
        predict_sigma=bool(mode_flags["sigma_active"]),
        fhat_active=bool(mode_flags["fhat_active"]),
        td_force_input_source=str(getattr(model, "td_force_input_source", "none")),
        fhat_bound_multiplier=float(getattr(model, "fhat_bound_multiplier", 1.5)),
        force_zero_output=bool(getattr(model, "force_zero_output", False)),
        tag_prefix=tag_prefix,
        step=step,
        log_metrics=log_metrics,
        log_plots=log_plots,
        title_suffix=title_suffix,
    )


def _run_hnn_validation(
    *,
    ckpt: dict[str, Any],
    cfg: Any,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    do_losses: bool,
    do_rollout: bool,
    num_workers: int,
) -> None:
    tb_step = int(epoch) + 1
    data_cfg = cfg.data
    monitoring_cfg = cfg.monitoring
    hnn_cfg = dict(cfg.hnn or {})
    rollout_stochastic = bool(hnn_cfg.get("rollout_stochastic", False))
    rollout_noise_scale = float(hnn_cfg.get("rollout_noise_scale", 1.0))
    if not np.isfinite(rollout_noise_scale) or rollout_noise_scale < 0.0:
        raise ValueError("hnn.rollout_noise_scale must be finite and non-negative.")
    rollout_seed_raw = hnn_cfg.get("rollout_seed", None)
    rollout_seed = None if rollout_seed_raw is None else int(rollout_seed_raw)
    loss_cfg = cfg.loss
    validation_samples_per_ur = max(1, int(getattr(monitoring_cfg, "validation_samples_per_ur", 1)))
    rollout_det_weight = float(getattr(loss_cfg, "rollout_det_weight", 0.0))
    rollout_det_steps = int(getattr(loss_cfg, "rollout_det_steps", 0))
    rollout_loss_mode = str(getattr(loss_cfg, "rollout_loss_mode", "deterministic")).strip().lower()
    rollout_stochastic_samples = int(getattr(loss_cfg, "rollout_stochastic_samples", 1))
    ur_bin_size = float(getattr(loss_cfg, "ur_bin_size", 1e-6))
    normalize_by_ur_bin_std = bool(getattr(loss_cfg, "normalize_by_ur_bin_std", False))
    ur_bin_scale_eps = float(getattr(loss_cfg, "ur_bin_scale_eps", 1e-6))
    rollout_det_batch_size_raw = int(getattr(loss_cfg, "rollout_det_batch_size", 0))
    rollout_det_batch_size = int(cfg.training.batch_size) if rollout_det_batch_size_raw <= 0 else rollout_det_batch_size_raw
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
    if rollout_det_weight > 0.0 and rollout_det_steps < 1:
        raise ValueError("loss.rollout_det_steps must be >= 1 when loss.rollout_det_weight > 0.")
    if rollout_det_batch_size < 1:
        raise ValueError("loss.rollout_det_batch_size must be >= 1 after fallback resolution.")

    val_dir = _resolve_val_unseen_dir(Path(data_cfg.train_series_dir))
    val_files = sorted(val_dir.glob("*.npz"))
    if not val_files:
        raise FileNotFoundError(f"No '.npz' files found in '{val_dir}'.")
    data_path = val_files[0]

    with np.load(data_path) as data:
        t = np.asarray(data["a"])
        y_data = np.asarray(data["b"])
        has_force_data = "c" in data
        F_data = np.asarray(data["c"]) if has_force_data else np.zeros_like(y_data)
        H_data = np.asarray(data["d"]) if "d" in data else np.zeros_like(y_data)
        if "U_r" not in data:
            raise KeyError(f"{data_path} is missing reduced velocity 'U_r'.")
        reduced_velocity = np.asarray(data["U_r"])
        vel_data = None
        for key in ("e", "dy", "v"):
            if key in data:
                vel_data = np.asarray(data[key])
                break

    val_cut = resolve_cut_start_seconds(data_cfg, "val")
    t, y_data, F_data, hamiltonian_data, vel_data, dt = preprocess_timeseries(
        t,
        y_data,
        F_data,
        H_data,
        data_cfg,
        velocity=vel_data,
        cut_start_seconds=val_cut,
    )

    model_dict = asdict(cfg.model)
    arch_dict = asdict(cfg.architecture)
    model, derived = PHVIV.from_config(dt=float(dt), cfg=model_dict, arch_cfg=arch_dict, device=device)
    _load_state(model, ckpt["model_state"])
    model.eval()
    ur_bin_state_scale_info = ckpt.get("ur_bin_state_scale_info", None)

    m_eff = float(derived["m_eff"])
    D = float(derived["D"])
    k = float(derived["k"])

    series_dir = _resolve_val_unseen_dir(Path(data_cfg.train_series_dir))
    val_require_force = bool(getattr(loss_cfg, "use_force_data_loss", False) or has_force_data)
    val_series_raw, _ = load_training_series(
        y_data,
        t,
        dt,
        series_dir,
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
        batch_size=int(cfg.training.batch_size),
        device=device,
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=(device.type == "cuda"),
    )
    val_rollout_loader: Any | None = None
    if rollout_det_weight > 0.0 and rollout_det_steps > 0:
        val_rollout_loader, _ = build_rollout_dataloader_from_series(
            val_series_raw,
            m_eff=m_eff,
            batch_size=rollout_det_batch_size,
            device=device,
            rollout_steps=rollout_det_steps,
            shuffle=False,
            num_workers=int(num_workers),
            pin_memory=(device.type == "cuda"),
        )

    num_loss_scalars_written = 0
    num_rollout_scalars_written = 0

    if do_losses:
        amp_enabled = bool(cfg.precision.use_amp) and device.type == "cuda"
        mean_reg = float(getattr(loss_cfg, "mean_reg", 0.0))
        mean_reg_norm = str(getattr(loss_cfg, "mean_reg_norm", "l1")).strip().lower()
        sigma_reg_norm = str(getattr(loss_cfg, "sigma_reg_norm", "l2")).strip().lower()
        symmetry_weight = float(getattr(loss_cfg, "symmetry_weight", 0.0))
        symmetry_norm = str(getattr(loss_cfg, "symmetry_norm", "l2")).strip().lower()
        if symmetry_norm not in {"l1", "l2"}:
            raise ValueError("loss.symmetry_norm must be one of: l1, l2.")
        if mean_reg_norm not in {"l1", "l2"}:
            raise ValueError("loss.mean_reg_norm must be one of: l1, l2.")
        if sigma_reg_norm not in {"l1", "l2"}:
            raise ValueError("loss.sigma_reg_norm must be one of: l1, l2.")
        loss_metrics = _evaluate_val_losses(
            model=model,
            loader=val_loader,
            rollout_loader=val_rollout_loader,
            device=device,
            non_blocking=(device.type == "cuda"),
            mean_reg=mean_reg,
            mean_reg_norm=mean_reg_norm,
            sigma_reg=float(loss_cfg.sigma_reg),
            sigma_reg_norm=sigma_reg_norm,
            ur_bin_size=ur_bin_size,
            normalize_residual_by_ur_bin_std=normalize_by_ur_bin_std,
            normalize_rollout_by_ur_bin_std=normalize_by_ur_bin_std,
            ur_bin_state_scale_info=ur_bin_state_scale_info,
            rollout_det_weight=rollout_det_weight,
            rollout_loss_mode=rollout_loss_mode,
            rollout_stochastic_samples=rollout_stochastic_samples,
            rollout_noise_scale=rollout_noise_scale,
            use_force_data_loss=bool(getattr(loss_cfg, "use_force_data_loss", False)),
            force_data_weight=float(getattr(loss_cfg, "force_data_weight", 1.0)),
            symmetry_weight=symmetry_weight,
            symmetry_norm=symmetry_norm,
            amp_enabled=amp_enabled,
            amp_dtype=_amp_dtype(cfg.precision.amp_dtype),
        )
        for name, value in loss_metrics.items():
            value_f = float(value)
            if not np.isfinite(value_f):
                print(f"[async-val] epoch {epoch}: skipping non-finite loss metric '{name}'={value_f}")
                continue
            writer.add_scalar(f"{ASYNC_VAL_SPLIT_TAG}/{name}", value_f, tb_step)
            num_loss_scalars_written += 1
    if do_rollout:
        metrics_sum: dict[str, float] = {}
        count = 0
        diverged_count = 0
        total = min(len(val_series_raw), len(val_sequences))
        ur_for_sampling: list[float] = []
        for idx in range(total):
            ur_arr = np.asarray(val_series_raw[idx][5]).reshape(-1)
            ur_for_sampling.append(float(ur_arr[0]) if ur_arr.size > 0 else float("nan"))
        sample_seed = 1
        sampled_indices = sample_indices_per_ur(
            ur_for_sampling,
            samples_per_ur=validation_samples_per_ur,
            seed=sample_seed,
        )
        for idx in sampled_indices:
            series_raw = val_series_raw[idx]
            sequence = val_sequences[idx]
            y_np, t_np, dt_value, _vel_np, force_np, _ur_np = series_raw
            y_tensor, vel_tensor, _t_tensor, ur_tensor = sequence
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
                log_extra_metrics=bool(getattr(cfg.monitoring, "log_extra_validation_metrics", False)),
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
                value_f = float(total / float(count))
                if not np.isfinite(value_f):
                    print(f"[async-val] epoch {epoch}: skipping non-finite rollout metric '{name}'={value_f}")
                    continue
                writer.add_scalar(f"{ASYNC_VAL_SPLIT_TAG}/{name}", value_f, tb_step)
                num_rollout_scalars_written += 1
            writer.add_scalar(f"{ASYNC_VAL_SPLIT_TAG}/{ROLLOUT_DIVERGED_COUNT_KEY}", float(diverged_count), tb_step)
            num_rollout_scalars_written += 1

        ur_values = [float(np.asarray(series_raw[5]).reshape(-1)[0]) for series_raw in val_series_raw]
        rollout_idx = _rollout_index(
            len(val_series_raw),
            ur_values=ur_values,
        )
        y_np, t_np, dt_value, _vel_np, force_np, _ur_np = val_series_raw[rollout_idx]
        y_tensor, vel_tensor, _t_tensor, ur_tensor = val_sequences[rollout_idx]
        log_validation_epoch(
            writer,
            tb_step,
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
            getattr(data_cfg, "middle_time_plot", [0.0, 1.0]),
            hamiltonian_data,
            log_extra_metrics=bool(getattr(cfg.monitoring, "log_extra_validation_metrics", False)),
            log_metrics=False,
            rollout_stochastic=rollout_stochastic,
            rollout_noise_scale=rollout_noise_scale,
            rollout_seed=rollout_seed,
            log_spectra=True,
            tag_prefix=f"{ASYNC_VAL_SPLIT_TAG}/rollout",
        )

    print(
        f"[async-val] epoch {epoch}: HNN scalar writes "
        f"(loss={num_loss_scalars_written}, rollout={num_rollout_scalars_written})"
    )


def _run_hnn_td_correction_validation(
    *,
    ckpt: dict[str, Any],
    cfg: Any,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    do_losses: bool,
    do_rollout: bool,
    num_workers: int,
) -> dict[str, Any]:
    tb_step = int(epoch) + 1
    data_cfg = cfg.data
    monitoring_cfg = cfg.monitoring
    hnn_cfg = dict(cfg.hnn or {})
    loss_cfg = cfg.loss
    td_mass_source = str(hnn_cfg.get("td_mass_source", "dry")).strip().lower()
    if td_mass_source not in {"dry", "effective"}:
        raise ValueError("hnn.td_mass_source must be one of: dry, effective.")
    rollout_stochastic = bool(hnn_cfg.get("rollout_stochastic", False))
    rollout_noise_scale = float(hnn_cfg.get("rollout_noise_scale", 1.0))
    if not np.isfinite(rollout_noise_scale) or rollout_noise_scale < 0.0:
        raise ValueError("hnn.rollout_noise_scale must be finite and non-negative.")
    rollout_seed_raw = hnn_cfg.get("rollout_seed", None)
    rollout_seed = None if rollout_seed_raw is None else int(rollout_seed_raw)
    rollout_loss_mode = str(getattr(loss_cfg, "rollout_loss_mode", "deterministic")).strip().lower()
    rollout_stochastic_samples = int(getattr(loss_cfg, "rollout_stochastic_samples", 1))
    if rollout_loss_mode == "stochastic":
        rollout_loss_mode = "stochastic_nll"
    if rollout_loss_mode not in {"deterministic", "stochastic_nll", "stochastic_mse"}:
        raise ValueError(
            "loss.rollout_loss_mode must be one of: deterministic, stochastic_nll, stochastic_mse."
        )
    if rollout_stochastic_samples < 1:
        raise ValueError("loss.rollout_stochastic_samples must be >= 1.")

    train_series_root = Path(data_cfg.train_series_dir)
    td_params = resolve_td_correction_params(hnn_cfg)
    td_memory_cfg = resolve_td_memory_config(hnn_cfg)
    correction_mode = str(ckpt.get("correction_mode", resolve_td_correction_mode(hnn_cfg))).strip().lower()
    mode_flags = td_correction_mode_flags(correction_mode)
    mean_active = bool(mode_flags["mean_active"])
    predict_sigma = bool(mode_flags["sigma_active"])
    fhat_active = bool(mode_flags["fhat_active"])
    td_force_input_source = resolve_td_force_input_source(
        ckpt.get("td_force_input_source", hnn_cfg.get("use_td_force_input", False))
    )
    use_td_force_input = td_force_input_source != "none"
    use_td_fhat_input = bool(ckpt.get("use_td_fhat_input", hnn_cfg.get("use_td_fhat_input", False)))
    use_acceleration_input = bool(hnn_cfg.get("use_acceleration_input", False))
    phase_input_source = resolve_td_phase_input_source(
        hnn_cfg.get("phi_input_source", hnn_cfg.get("use_phi_input", False))
    )
    use_phi_input = phase_input_source != "none"
    use_sigma_inputs = bool(hnn_cfg.get("use_sigma_inputs", False))
    input_scaling_mode = resolve_phnn_input_scaling_mode(getattr(cfg.model, "input_scaling_mode", "current"))
    fhat_bound_multiplier = float(ckpt.get("fhat_bound_multiplier", hnn_cfg.get("fhat_bound_multiplier", 1.5)))
    fhat_reg = float(getattr(loss_cfg, "fhat_reg", 0.0))
    fhat_reg_norm = str(getattr(loss_cfg, "fhat_reg_norm", "l2")).strip().lower()
    state_loss_mode = str(hnn_cfg.get("state_loss_mode", "mse")).strip().lower()
    if state_loss_mode not in {"mse", "propagated_nll"}:
        raise ValueError("hnn.state_loss_mode must be one of: mse, propagated_nll.")
    force_zero_output = bool(hnn_cfg.get("force_zero_output", False))
    rollout_det_weight = float(getattr(loss_cfg, "rollout_det_weight", 0.0))
    rollout_disp_std_weight = float(getattr(loss_cfg, "rollout_disp_std_weight", 0.0))
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
    rollout_det_relative = rollout_loss_settings["trajectory_relative"]
    rollout_disp_std_relative = rollout_loss_settings["disp_std_relative"]
    rollout_disp_psd_relative = rollout_loss_settings["disp_psd_relative"]
    rollout_disp_freq_relative = rollout_loss_settings["disp_freq_relative"]
    rollout_disp_std_p = rollout_loss_settings["disp_std_p"]
    rollout_disp_freq_p = rollout_loss_settings["disp_freq_p"]
    rollout_disp_freq_alpha = rollout_loss_settings["disp_freq_alpha"]
    rollout_det_steps = int(getattr(loss_cfg, "rollout_det_steps", 0))
    rollout_batch_size_raw = int(getattr(loss_cfg, "rollout_det_batch_size", 0))
    rollout_batch_size = int(cfg.training.batch_size) if rollout_batch_size_raw <= 0 else rollout_batch_size_raw
    mean_reg = float(getattr(loss_cfg, "mean_reg", 0.0))
    sigma_reg = float(getattr(loss_cfg, "sigma_reg", 0.0))
    mean_reg_norm = str(getattr(loss_cfg, "mean_reg_norm", "l1")).strip().lower()
    sigma_reg_norm = str(getattr(loss_cfg, "sigma_reg_norm", "l2")).strip().lower()
    force_data_weight = float(getattr(loss_cfg, "force_data_weight", 1.0))
    use_force_data_loss = bool(getattr(loss_cfg, "use_force_data_loss", True))
    validation_samples_per_ur = max(1, int(getattr(monitoring_cfg, "validation_samples_per_ur", 1)))

    split_dirs: dict[str, Path] = {ASYNC_VAL_SPLIT_TAG: _resolve_val_unseen_dir(train_series_root)}
    val_seen_dir = _resolve_optional_val_split_dir(train_series_root, "val_seen")
    if val_seen_dir is not None:
        split_dirs["val_seen"] = val_seen_dir
    cut_start_seconds = resolve_cut_start_seconds(data_cfg, "val")
    split_trajs_map: dict[str, list[dict[str, Any]]] = {}
    for split_tag, split_dir in split_dirs.items():
        split_paths = sorted(split_dir.glob("*.npz"))
        if not split_paths:
            if split_tag == ASYNC_VAL_SPLIT_TAG:
                raise FileNotFoundError(f"No '.npz' files found in '{split_dir}'.")
            continue
        split_trajs_map[split_tag] = load_td_correction_trajectories(
            paths=split_paths,
            cut_start_seconds=cut_start_seconds,
            reduce_time=bool(getattr(data_cfg, "reduce_time", False)),
            reduction_factor=int(getattr(data_cfg, "reduction_factor", 1)),
            ur_source=td_mass_source,
            td_params=td_params,
            td_memory_cfg=td_memory_cfg,
        )

    val_trajs_np = split_trajs_map.get(ASYNC_VAL_SPLIT_TAG, [])
    if not val_trajs_np:
        raise FileNotFoundError("Async PHNN validation requires a non-empty val_unseen split.")
    dt = float(val_trajs_np[0]["t"][1] - val_trajs_np[0]["t"][0])

    model_dict = asdict(cfg.model)
    first_val_traj = val_trajs_np[0]
    mass_key = "dry_mass_kg" if td_mass_source == "dry" else "effective_mass_kg"
    model_dict["structural_mass"] = float(np.asarray(first_val_traj[mass_key]).reshape(()))
    model_dict["k"] = float(np.asarray(first_val_traj["stiffness_n_m"]).reshape(()))
    model_dict["damping_c"] = float(np.asarray(first_val_traj["damping_c"]).reshape(()))
    model_dict["Ca"] = 0.0
    model_dict["use_stochastic_process_noise"] = predict_sigma
    model_dict["use_td_force_input"] = use_td_force_input
    model_dict["use_td_fhat_input"] = use_td_fhat_input
    model_dict["use_acceleration_input"] = use_acceleration_input
    model_dict["use_phi_input"] = use_phi_input
    model_dict["phi_input_source"] = None if not use_phi_input else phase_input_source
    model_dict["use_sigma_inputs"] = use_sigma_inputs
    model_dict["correction_mode"] = correction_mode
    arch_dict = asdict(cfg.architecture)
    model, _derived = PHVIV.from_config(dt=dt, cfg=model_dict, arch_cfg=arch_dict, device=device)
    setattr(model, "correction_mode", correction_mode)
    setattr(model, "td_force_input_source", td_force_input_source)
    setattr(model, "fhat_bound_multiplier", float(fhat_bound_multiplier))
    setattr(model, "force_zero_output", force_zero_output)
    _load_state(model, ckpt["model_state"])
    model.eval()

    def _reg(value: torch.Tensor, norm: str) -> torch.Tensor:
        return torch.mean(torch.abs(value)) if str(norm).strip().lower() == "l1" else torch.mean(value * value)

    def _run_split(split_tag: str, split_trajs_np: list[dict[str, Any]]) -> dict[str, Any]:
        split_start = time.perf_counter()
        _train_loader, val_loader, rollout_loader = _build_td_correction_hnn_loaders(
            train_trajs=split_trajs_np,
            val_trajs=split_trajs_np,
            mass_source=td_mass_source,
            input_scaling_mode=input_scaling_mode,
            diameter=float(model.D),
            batch_size=int(cfg.training.batch_size),
            rollout_batch_size=rollout_batch_size,
            rollout_steps=rollout_det_steps,
            num_workers=int(num_workers),
            pin_memory=(device.type == "cuda"),
        )

        num_loss_scalars_written = 0
        num_rollout_scalars_written = 0
        val_metrics: dict[str, float] = {}

        if do_losses and val_loader is not None:
            val_sums = {
                name: torch.zeros((), device=device)
                for name in ["loss_total", "loss_state", "loss_data", "loss_reg_mean", "loss_reg_sigma", "loss_reg_fhat"]
            }
            val_count = 0
            with torch.no_grad():
                for batch in val_loader:
                    if len(batch) != 10:
                        raise ValueError("Unexpected TD correction HNN batch format.")
                    z_i, t_i, z_next, t_next, ur_i, force_true_next, td_context_i, mass_i, damping_i, stiffness_i = batch
                    z_i = z_i.to(device, non_blocking=(device.type == "cuda"))
                    t_i = t_i.to(device, non_blocking=(device.type == "cuda"))
                    z_next = z_next.to(device, non_blocking=(device.type == "cuda"))
                    t_next = t_next.to(device, non_blocking=(device.type == "cuda"))
                    ur_i = ur_i.to(device, non_blocking=(device.type == "cuda"))
                    force_true_next = force_true_next.to(device, non_blocking=(device.type == "cuda"))
                    td_context_i = td_context_i.to(device, non_blocking=(device.type == "cuda"))
                    mass_i = mass_i.to(device, non_blocking=(device.type == "cuda"))
                    damping_i = damping_i.to(device, non_blocking=(device.type == "cuda"))
                    stiffness_i = stiffness_i.to(device, non_blocking=(device.type == "cuda"))
                    dt_i = torch.clamp(t_next - t_i, min=1.0e-12)
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
                    corr_mu = step["corr_mu"]
                    sigma_corr = step["sigma_corr"]
                    total_force_next = step["total_force_next"]
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
                    if use_force_data_loss:
                        if predict_sigma:
                            var = torch.clamp(sigma_corr * sigma_corr, min=1e-9)
                            data_loss = torch.mean(
                                0.5 * (((force_true_next - total_force_next) ** 2) / var + torch.log(var))
                            )
                        else:
                            data_loss = torch.mean((force_true_next - total_force_next) ** 2)
                    else:
                        data_loss = state_loss.new_tensor(0.0)
                    mean_reg_loss = _reg(corr_mu, mean_reg_norm)
                    sigma_reg_loss = _reg(sigma_corr, sigma_reg_norm) if predict_sigma else state_loss.new_tensor(0.0)
                    fhat_reg_loss = _reg(step["delta_fhat"], fhat_reg_norm) if fhat_active else state_loss.new_tensor(0.0)
                    total_loss = (
                        state_loss
                        + float(force_data_weight) * data_loss
                        + float(mean_reg) * mean_reg_loss
                        + float(sigma_reg) * sigma_reg_loss
                        + float(fhat_reg) * fhat_reg_loss
                    )
                    val_sums["loss_total"] += total_loss.detach()
                    val_sums["loss_state"] += state_loss.detach()
                    val_sums["loss_data"] += data_loss.detach()
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
            rollout_spectral_loss_avg = 0.0

            if rollout_loader is not None and (
                rollout_det_weight > 0.0
                or rollout_disp_std_weight > 0.0
                or rollout_disp_spectral_weight > 0.0
            ):
                rollout_loss_sum = torch.zeros((), device=device)
                rollout_std_loss_sum = torch.zeros((), device=device)
                rollout_spectral_loss_sum = torch.zeros((), device=device)
                rollout_count = 0
                with torch.no_grad():
                    for rollout_batch in rollout_loader:
                        rollout_losses = _td_correction_rollout_losses_from_batch(
                            model=model,
                            batch=rollout_batch,
                            device=device,
                            non_blocking=(device.type == "cuda"),
                            td_params=td_params,
                            td_memory_cfg=td_memory_cfg,
                            mean_active=mean_active,
                            sigma_active=predict_sigma,
                            fhat_active=fhat_active,
                            td_force_input_source=td_force_input_source,
                            fhat_bound_multiplier=fhat_bound_multiplier,
                            force_zero_output=force_zero_output,
                            rollout_loss_mode=rollout_loss_mode,
                            rollout_stochastic_samples=rollout_stochastic_samples,
                            rollout_noise_scale=rollout_noise_scale,
                            trajectory_relative=rollout_det_relative,
                            compute_disp_std_loss=(rollout_disp_std_weight > 0.0),
                            disp_std_relative=rollout_disp_std_relative,
                            disp_std_power=rollout_disp_std_p,
                            compute_disp_spectral_loss=(rollout_disp_spectral_weight > 0.0),
                            disp_spectral_loss_mode=rollout_disp_spectral_loss,
                            disp_freq_relative=rollout_disp_freq_relative,
                            disp_freq_power=rollout_disp_freq_p,
                            disp_freq_alpha=rollout_disp_freq_alpha,
                            disp_psd_relative=rollout_disp_psd_relative,
                            disp_psd_peak_rel_bandwidth=rollout_disp_psd_peak_rel_bandwidth,
                            disp_psd_use_hann_window=rollout_disp_psd_use_hann_window,
                        )
                        rollout_loss_sum += rollout_losses["trajectory_loss"].detach()
                        rollout_std_loss_sum += rollout_losses["disp_std_loss"].detach()
                        rollout_spectral_loss_sum += rollout_losses["disp_spectral_loss"].detach()
                        rollout_count += 1
                rollout_loss_avg = float((rollout_loss_sum / float(max(1, rollout_count))).detach().cpu())
                rollout_std_loss_avg = float((rollout_std_loss_sum / float(max(1, rollout_count))).detach().cpu())
                rollout_spectral_loss_avg = float(
                    (rollout_spectral_loss_sum / float(max(1, rollout_count))).detach().cpu()
                )
                writer.add_scalar(f"{split_tag}/loss_rollout_det", rollout_loss_avg, tb_step)
                writer.add_scalar(f"{split_tag}/loss_rollout_disp_std", rollout_std_loss_avg, tb_step)
                writer.add_scalar(f"{split_tag}/loss_rollout_spectral", rollout_spectral_loss_avg, tb_step)
                num_loss_scalars_written += 3
            val_metrics["loss_rollout_disp_std"] = rollout_std_loss_avg
            val_metrics["loss_rollout_spectral"] = rollout_spectral_loss_avg
            val_metrics["loss_total"] = (
                val_metrics["loss_state"]
                + float(force_data_weight) * val_metrics["loss_data"]
                + float(mean_reg) * val_metrics["loss_reg_mean"]
                + float(sigma_reg) * val_metrics["loss_reg_sigma"]
                + float(fhat_reg) * val_metrics["loss_reg_fhat"]
                + float(rollout_det_weight) * rollout_loss_avg
                + float(rollout_disp_std_weight) * rollout_std_loss_avg
                + float(rollout_disp_spectral_weight) * rollout_spectral_loss_avg
            )
            for name, value in val_metrics.items():
                writer.add_scalar(f"{split_tag}/{name}", value, tb_step)
                num_loss_scalars_written += 1

        if do_rollout:
            ur_values_all = [float(np.asarray(traj["ur"]).reshape(-1)[0]) for traj in split_trajs_np]
            sampled_metric_indices = sample_indices_per_ur(
                ur_values_all,
                samples_per_ur=validation_samples_per_ur,
                seed=1,
            )
            sampled_names = [str(split_trajs_np[idx].get("name", f"traj_{idx}")) for idx in sampled_metric_indices]
            print(
                f"[async-val][phnn][{split_tag}] epoch {epoch + 1}: sampled metric trajectories={sampled_names} "
                f"(force_zero_output={force_zero_output}, mass_source={td_mass_source})"
            )
            metrics_sum: dict[str, float] = {}
            metrics_count: dict[str, int] = {}
            diverged_count = 0
            for sidx in sampled_metric_indices:
                metrics = _hnn_td_rollout_validation(
                    writer=writer,
                    epoch=tb_step,
                    model=model,
                    traj=split_trajs_np[sidx],
                    dt=dt,
                    td_mass_source=td_mass_source,
                    td_params=td_params,
                    td_memory_cfg=td_memory_cfg,
                    device=device,
                    mean_active=mean_active,
                    predict_sigma=predict_sigma,
                    fhat_active=fhat_active,
                    td_force_input_source=td_force_input_source,
                    fhat_bound_multiplier=fhat_bound_multiplier,
                    force_zero_output=force_zero_output,
                    rollout_stochastic=rollout_stochastic,
                    rollout_noise_scale=rollout_noise_scale,
                    rollout_seed=rollout_seed,
                    log_metrics=False,
                    log_plots=False,
                )
                diverged_flag = float(metrics.get(ROLLOUT_DIVERGED_KEY, 0.0))
                if np.isfinite(diverged_flag) and diverged_flag > 0.5:
                    diverged_count += 1
                for name, value in metrics.items():
                    if name == ROLLOUT_DIVERGED_KEY or not np.isfinite(float(value)):
                        continue
                    metrics_sum[name] = metrics_sum.get(name, 0.0) + float(value)
                    metrics_count[name] = metrics_count.get(name, 0) + 1
            for name, total in metrics_sum.items():
                writer.add_scalar(f"{split_tag}/{name}", total / float(max(1, metrics_count.get(name, 0))), tb_step)
                num_rollout_scalars_written += 1
            writer.add_scalar(f"{split_tag}/{ROLLOUT_DIVERGED_COUNT_KEY}", float(diverged_count), tb_step)
            num_rollout_scalars_written += 1

            rollout_idx = _rollout_index(
                len(split_trajs_np),
                ur_values=ur_values_all,
            )
            rollout_traj = split_trajs_np[rollout_idx]
            rollout_dt = float(np.asarray(rollout_traj["t"])[1] - np.asarray(rollout_traj["t"])[0])
            print(
                f"[async-val][phnn][{split_tag}] epoch {epoch + 1}: plot trajectory={rollout_traj.get('name', f'traj_{rollout_idx}')} "
                f"U_r={float(np.asarray(rollout_traj['ur']).reshape(-1)[0]):.6g} "
                f"dt={rollout_dt:.6g} rho={float(model.rho):.6g} D={float(model.D):.6g} "
                f"m={float(np.asarray(rollout_traj['dry_mass_kg' if td_mass_source == 'dry' else 'effective_mass_kg']).reshape(())):.6g} "
                f"c={float(np.asarray(rollout_traj['damping_c']).reshape(())):.6g} "
                f"k={float(np.asarray(rollout_traj['stiffness_n_m']).reshape(())):.6g}"
            )
            _hnn_td_rollout_validation(
                writer=writer,
                epoch=tb_step,
                model=model,
                traj=rollout_traj,
                dt=dt,
                td_mass_source=td_mass_source,
                td_params=td_params,
                td_memory_cfg=td_memory_cfg,
                device=device,
                mean_active=mean_active,
                predict_sigma=predict_sigma,
                fhat_active=fhat_active,
                td_force_input_source=td_force_input_source,
                fhat_bound_multiplier=fhat_bound_multiplier,
                force_zero_output=force_zero_output,
                rollout_stochastic=rollout_stochastic,
                rollout_noise_scale=rollout_noise_scale,
                rollout_seed=rollout_seed,
                log_metrics=False,
                log_plots=True,
                log_spectra=True,
                tag_prefix=f"{split_tag}/rollout",
            )

        split_elapsed = time.perf_counter() - split_start
        writer.add_scalar(f"{split_tag}/validation_wall_time_s", float(split_elapsed), tb_step)
        print(
            f"[async-val] epoch {epoch}: HNN TD split='{split_tag}' scalar writes "
            f"(loss={num_loss_scalars_written}, rollout={num_rollout_scalars_written})"
        )
        return {
            "loss_total": (float(val_metrics["loss_total"]) if "loss_total" in val_metrics else None),
            "val_metrics": val_metrics,
            "validation_wall_time_s": float(split_elapsed),
        }

    split_results = {
        split_tag: _run_split(split_tag, split_trajs_np)
        for split_tag, split_trajs_np in split_trajs_map.items()
        if split_trajs_np
    }
    unseen_result = split_results.get(ASYNC_VAL_SPLIT_TAG, {})
    summary: dict[str, Any] = {
        "loss_total": unseen_result.get("loss_total"),
        "val_metrics": unseen_result.get("val_metrics", {}),
        "split_results": split_results,
    }
    if "val_seen" in split_results:
        summary["val_seen_loss_total"] = split_results["val_seen"].get("loss_total")
    return summary


def _run_vpinn_validation(
    *,
    ckpt: dict[str, Any],
    cfg: Any,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    do_losses: bool,
    do_rollout: bool,
    num_workers: int,
) -> None:
    tb_step = int(epoch) + 1
    data_cfg = cfg.data
    monitoring_cfg = cfg.monitoring
    vp = dict(cfg.vpinn or {})
    validation_samples_per_ur = max(1, int(getattr(monitoring_cfg, "validation_samples_per_ur", 1)))
    force_representation = str(vp.get("force_representation", "force")).strip().lower()
    if force_representation not in {"force", "coefficient"}:
        raise ValueError("vpinn.force_representation must be one of: force, coefficient.")
    num_poly = int(vp.get("num_poly_test", 2))
    num_sine = int(vp.get("num_sine_test", 0))

    val_dir = _resolve_val_unseen_dir(Path(data_cfg.train_series_dir))
    val_files = sorted(val_dir.glob("*.npz"))
    if not val_files:
        raise FileNotFoundError(f"No '.npz' files found in '{val_dir}'.")
    sources = val_files

    val_trajs: list[dict[str, Any]] = []
    dt_ref: Optional[float] = None
    cut_start_seconds = float(
        data_cfg.cut_start_seconds_val
        if getattr(data_cfg, "cut_start_seconds_val", None) is not None
        else 0.0
    )
    dt_target = vp.get("dt_target", None)
    if dt_target is None:
        dt_target = _infer_dt_target_from_data_cfg(data_cfg)
    dt_target = None if dt_target is None else float(dt_target)
    coeff_k = float(getattr(cfg.model, "k", 1218.0))
    coeff_m_eff = float(_m_eff_from_model_cfg(cfg.model))

    for path in sources:
        traj, dt = _load_trajectory(
            path=path,
            dt_target=dt_target,
            reduce_time=bool(getattr(data_cfg, "reduce_time", False)),
            reduction_factor=int(getattr(data_cfg, "reduction_factor", 1)),
            cut_start_seconds=cut_start_seconds,
            force_representation=force_representation,
            rho=float(getattr(cfg.model, "rho", 1000.0)),
            D=float(getattr(cfg.model, "D", 0.1)),
            k=coeff_k,
            m_eff=coeff_m_eff,
        )
        if dt_ref is None:
            dt_ref = dt
        elif not np.isclose(dt, float(dt_ref), rtol=1e-9, atol=1e-12):
            raise ValueError(f"{path} has dt={dt} but expected dt={dt_ref}.")
        val_trajs.append(traj)
    if dt_ref is None:
        raise ValueError("No validation trajectories loaded.")
    dt = float(dt_ref)

    d = int(val_trajs[0]["x"].shape[-1])
    m = _as_diag_param(vp.get("m", _m_eff_from_model_cfg(cfg.model)), d, device, "m")
    c = _as_diag_param(vp.get("c", getattr(cfg.model, "damping_c", 1e-4)), d, device, "c")
    k = _as_diag_param(vp.get("k", getattr(cfg.model, "k", 1218.0)), d, device, "k")

    input_dim = 2 * d + 1
    output_dim = d
    base_model = _build_force_model(cfg, input_dim=input_dim, output_dim=output_dim).to(device)
    # Keep validation backward-compatible with older checkpoints/configs:
    # scaling must be explicitly enabled, otherwise state_dict keys won't match.
    use_input_scaling = bool(vp.get("use_input_scaling", False))
    if use_input_scaling:
        D_val = float(getattr(cfg.model, "D", 1.0))
        x_scale = D_val if np.isfinite(D_val) and D_val != 0.0 else 1.0
        omega = torch.sqrt(torch.clamp(k / m, min=1e-12))
        v_scale = omega * float(x_scale)
        ur_scale = float(vp.get("ur_scale", 10.0))
        f_scale = 1.0 if force_representation == "coefficient" else k * float(x_scale)
        model = ScaledForceWrapper(
            base_model,
            d=d,
            x_scale=x_scale,
            v_scale=v_scale,
            ur_scale=ur_scale,
            f_scale=f_scale,
        )
    else:
        model = base_model
    _load_state(model, ckpt["model_state"])
    model.eval()

    w, wdot, alpha = _test_functions(int(vp.get("window_M", 50)), dt, num_poly=num_poly, num_sine=num_sine)
    w = w.to(device)
    wdot = wdot.to(device)
    alpha = alpha.to(device)

    val_dataset = WindowDataset(
        val_trajs,
        window_intervals=int(vp.get("window_M", 50)),
        stride=int(vp.get("stride", 1)),
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    if do_losses:
        amp_enabled = bool(cfg.precision.use_amp) and device.type == "cuda"
        val_metrics = _evaluate_epoch(
            model=model,
            loader=val_loader,
            device=device,
            non_blocking=(device.type == "cuda"),
            dt=dt,
            m=m,
            c=c,
            k=k,
            wf=float(vp.get("wf", 1.0)),
            ww=float(vp.get("ww", 1.0)),
            use_force_loss=bool(vp.get("use_force_loss", True)),
            use_weak_loss=bool(vp.get("use_weak_loss", True)),
            w=w,
            wdot=wdot,
            alpha=alpha,
            amp_enabled=amp_enabled,
            amp_dtype=_amp_dtype(cfg.precision.amp_dtype),
            expect_f0=(force_representation == "coefficient"),
        )
        for name, value in val_metrics.items():
            writer.add_scalar(f"{ASYNC_VAL_SPLIT_TAG}/{name}", value, tb_step)
        force_map = _force_mapping_nrmse_over_trajs(model=model, val_trajs=val_trajs, device=device)
        if force_map is not None:
            for k_name, v_value in force_map.items():
                writer.add_scalar(f"{ASYNC_VAL_SPLIT_TAG}/{k_name}", v_value, tb_step)
    if do_rollout:
        ur_values_all = [float(traj["ur"][0, 0].detach().cpu().item()) for traj in val_trajs]
        sample_seed = 1
        sampled_metric_indices = sample_indices_per_ur(
            ur_values_all,
            samples_per_ur=validation_samples_per_ur,
            seed=sample_seed,
        )
        metrics_sum: dict[str, float] = {}
        metrics_count: dict[str, int] = {}
        for sidx in sampled_metric_indices:
            metrics = _log_rollout_validation(
                writer=writer,
                epoch=tb_step,
                model=model,
                traj=val_trajs[sidx],
                dt=dt,
                m=m,
                c=c,
                k=k,
                D=float(getattr(cfg.model, "D", 1.0)),
                middle_time_plot=resolve_middle_time_plot(data_cfg, vp, method_name="vpinn"),
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
            writer.add_scalar(f"{ASYNC_VAL_SPLIT_TAG}/{name}", total / denom, tb_step)

        ur_values = ur_values_all
        rollout_idx = _rollout_index(
            len(val_trajs),
            ur_values=ur_values,
        )
        _log_rollout_validation(
            writer=writer,
            epoch=tb_step,
            model=model,
            traj=val_trajs[rollout_idx],
            dt=dt,
            m=m,
            c=c,
            k=k,
            D=float(getattr(cfg.model, "D", 1.0)),
            middle_time_plot=resolve_middle_time_plot(data_cfg, vp, method_name="vpinn"),
            device=device,
            log_metrics=False,
            log_plots=True,
            log_spectra=True,
            tag_prefix=f"{ASYNC_VAL_SPLIT_TAG}/rollout",
        )


def _run_vpinn_td_correction_validation(
    *,
    ckpt: dict[str, Any],
    cfg: Any,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    do_losses: bool,
    do_rollout: bool,
    num_workers: int,
) -> dict[str, Any]:
    tb_step = int(epoch) + 1
    td_loader_num_workers = 0
    data_cfg = cfg.data
    monitoring_cfg = cfg.monitoring
    vp = dict(cfg.vpinn or {})
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
    td_mass_source = str(vp.get("td_mass_source", "dry")).strip().lower()
    if td_mass_source not in {"dry", "effective"}:
        raise ValueError("vpinn.td_mass_source must be one of: dry, effective.")

    train_series_root = Path(data_cfg.train_series_dir)
    val_dir = _resolve_val_unseen_dir(train_series_root)
    val_paths = sorted(val_dir.glob("*.npz"))
    if not val_paths:
        raise FileNotFoundError(f"No '.npz' files found in '{val_dir}'.")

    val_trajs_np = load_td_correction_trajectories(
        paths=val_paths,
        cut_start_seconds=resolve_cut_start_seconds(data_cfg, "val"),
        reduce_time=bool(getattr(data_cfg, "reduce_time", False)),
        reduction_factor=int(getattr(data_cfg, "reduction_factor", 1)),
        ur_source=td_mass_source,
    )
    dt = float(val_trajs_np[0]["t"][1] - val_trajs_np[0]["t"][0])
    rho = float(getattr(cfg.model, "rho", 1000.0))
    diameter = float(getattr(cfg.model, "D", 0.1))
    td_params = resolve_td_correction_params(vp)
    use_td_force_input = bool(vp.get("use_td_force_input", False))
    use_acceleration_input = bool(vp.get("use_acceleration_input", False))
    phase_input_source = resolve_td_phase_input_source(
        vp.get("phi_input_source", vp.get("use_phi_input", False))
    )
    use_phi_input = phase_input_source != "none"
    use_sigma_inputs = bool(vp.get("use_sigma_inputs", False))
    fhat_bound_multiplier = float(vp.get("fhat_bound_multiplier", 1.5))

    model = _build_force_model(
        cfg,
        input_dim=_vpinn_input_dim(
            d=1,
            use_td_force_input=use_td_force_input,
            use_acceleration_input=use_acceleration_input,
            use_phi_input=use_phi_input,
            phase_input_source=phase_input_source,
            use_sigma_inputs=use_sigma_inputs,
        ),
        output_dim=_vpinn_output_dim(mean_active=mean_active, sigma_active=probabilistic, fhat_active=fhat_active, d=1),
    ).to(device)
    setattr(model, "use_td_force_input", use_td_force_input)
    setattr(model, "use_acceleration_input", use_acceleration_input)
    setattr(model, "use_phi_input", use_phi_input)
    setattr(model, "phi_input_source", None if not use_phi_input else phase_input_source)
    setattr(model, "use_sigma_inputs", use_sigma_inputs)
    setattr(model, "correction_mode", correction_mode)
    setattr(model, "fhat_bound_multiplier", float(fhat_bound_multiplier))
    setattr(model, "force_zero_output", force_zero_output)
    _load_state(model, ckpt["model_state"])
    model.eval()

    loss_cfg = cfg.loss
    window_M = int(vp.get("window_M", getattr(loss_cfg, "window_M", 50)))
    stride = max(1, int(vp.get("stride", getattr(loss_cfg, "stride", 1))))
    rollout_steps = int(vp.get("rollout_force_steps", int(getattr(cfg.loss, "rollout_det_steps", 0))))
    val_dataset, val_rollout_dataset = _build_td_correction_vpinn_datasets(
        trajs=val_trajs_np,
        rollout_steps=rollout_steps,
        window_M=window_M,
        stride=stride,
        td_mass_source=td_mass_source,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=td_loader_num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_rollout_loader = (
        torch.utils.data.DataLoader(
            val_rollout_dataset,
            batch_size=int(cfg.training.batch_size),
            shuffle=False,
            num_workers=td_loader_num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )
        if val_rollout_dataset is not None
        else None
    )

    mean_reg = float(getattr(loss_cfg, "mean_reg", 0.0))
    sigma_reg = float(getattr(loss_cfg, "sigma_reg", 0.0))
    fhat_reg = float(getattr(loss_cfg, "fhat_reg", 0.0))
    mean_reg_norm = str(getattr(loss_cfg, "mean_reg_norm", "l1")).strip().lower()
    sigma_reg_norm = str(getattr(loss_cfg, "sigma_reg_norm", "l2")).strip().lower()
    fhat_reg_norm = str(getattr(loss_cfg, "fhat_reg_norm", "l2")).strip().lower()
    ur_bin_size = float(getattr(cfg.loss, "ur_bin_size", 1.0e-6))
    normalize_by_ur_bin_std = bool(getattr(loss_cfg, "normalize_by_ur_bin_std", False))
    ur_bin_state_scale_info = ckpt.get("ur_bin_state_scale_info", None)
    use_force_loss = bool(vp.get("use_force_loss", getattr(loss_cfg, "use_force_loss", True)))
    use_weak_loss = bool(vp.get("use_weak_loss", getattr(loss_cfg, "use_weak_loss", True)))
    w, wdot, alpha = _test_functions(
        window_M,
        dt,
        num_poly=int(vp.get("num_poly_test", getattr(loss_cfg, "num_poly_test", 2))),
        num_sine=int(vp.get("num_sine_test", getattr(loss_cfg, "num_sine_test", 0))),
    )
    w = w.to(device)
    wdot = wdot.to(device)
    alpha = alpha.to(device)

    val_metrics: dict[str, float] = {}

    if do_losses:
        val_sums = {
            name: torch.zeros((), device=device)
            for name in ["loss_total", "loss_data", "loss_physics", "loss_reg_mean", "loss_reg_sigma", "loss_reg_fhat"]
        }
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                x_win, v_win, force_true, ur_win, td_win, m_win, c_win, k_win = [
                    item.to(device, non_blocking=(device.type == "cuda")) for item in batch
                ]
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
                mean_reg_loss = torch.mean(torch.abs(mu_corr)) if mean_reg_norm == "l1" else torch.mean(mu_corr * mu_corr)
                sigma_reg_loss = (
                    (torch.mean(torch.abs(sigma_corr)) if sigma_reg_norm == "l1" else torch.mean(sigma_corr * sigma_corr))
                    if probabilistic
                    else mu_corr.new_tensor(0.0)
                )
                fhat_reg_loss = (
                    (torch.mean(torch.abs(step["delta_fhat"])) if fhat_reg_norm == "l1" else torch.mean(step["delta_fhat"] * step["delta_fhat"]))
                    if fhat_active
                    else mu_corr.new_tensor(0.0)
                )
                total_loss = (
                    loss_data
                    + loss_physics
                    + float(mean_reg) * mean_reg_loss
                    + float(sigma_reg) * sigma_reg_loss
                    + float(fhat_reg) * fhat_reg_loss
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

        if val_rollout_loader is not None and float(vp.get("rollout_force_weight", float(getattr(loss_cfg, "rollout_det_weight", 0.0)))) > 0.0 and rollout_steps > 0:
            roll_sum = torch.zeros((), device=device)
            roll_batches = 0
            with torch.no_grad():
                for rb in val_rollout_loader:
                    x0, v0, ur0, td0, x_true_seq, v_true_seq, m0, c0, k0 = [
                        item.to(device, non_blocking=(device.type == "cuda")) for item in rb
                    ]
                    x_seq, v_seq, _, _, _, _ = _vpinn_td_rollout(
                        model=model,
                        x0=x0,
                        v0=v0,
                        ur0=ur0,
                        td_context0=td0,
                        steps=int(x_true_seq.shape[1] - 1),
                        dt=dt,
                        m=m0,
                        c=c0,
                        k=k0,
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
                    roll_sum += _vpinn_rollout_state_loss(
                        x_pred=x_seq,
                        v_pred=v_seq,
                        x_true=x_true_seq,
                        v_true=v_true_seq,
                        ur_values=ur0,
                        scale_info=(ur_bin_state_scale_info if normalize_by_ur_bin_std else None),
                        ur_bin_size=ur_bin_size,
                    ).detach()
                    roll_batches += 1
            rollout_loss_avg = float((roll_sum / float(max(1, roll_batches))).detach().cpu())
            writer.add_scalar(f"{ASYNC_VAL_SPLIT_TAG}/loss_rollout_det", rollout_loss_avg, tb_step)
        rollout_weight = float(vp.get("rollout_force_weight", float(getattr(loss_cfg, "rollout_det_weight", 0.0))))
        val_metrics["loss_total"] = (
            val_metrics["loss_data"]
            + val_metrics["loss_physics"]
            + float(mean_reg) * val_metrics["loss_reg_mean"]
            + float(sigma_reg) * val_metrics["loss_reg_sigma"]
            + float(fhat_reg) * val_metrics["loss_reg_fhat"]
            + rollout_weight * rollout_loss_avg
        )
        for name, value in val_metrics.items():
            writer.add_scalar(f"{ASYNC_VAL_SPLIT_TAG}/{name}", value, tb_step)

    if do_rollout:
        validation_samples_per_ur = max(1, int(getattr(monitoring_cfg, "validation_samples_per_ur", 1)))
        ur_values_all = [float(np.asarray(traj["ur"]).reshape(-1)[0]) for traj in val_trajs_np]
        sample_seed = 1
        sampled_metric_indices = sample_indices_per_ur(
            ur_values_all,
            samples_per_ur=validation_samples_per_ur,
            seed=sample_seed,
        )
        sampled_names = [str(val_trajs_np[idx].get("name", f"traj_{idx}")) for idx in sampled_metric_indices]
        print(
            f"[async-val][vpinn] epoch {epoch + 1}: sampled metric trajectories={sampled_names} "
            f"(force_zero_output={force_zero_output}, mass_source={td_mass_source})"
        )
        metrics_sum: dict[str, float] = {}
        metrics_count: dict[str, int] = {}
        diverged_count = 0
        val_trajs_plot = [_td_rollout_traj_to_tensors(traj) for traj in val_trajs_np]
        for sidx in sampled_metric_indices:
            metrics = _log_td_correction_rollout_validation(
                writer=writer,
                epoch=tb_step,
                model=model,
                traj=val_trajs_plot[sidx],
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
                log_metrics=False,
                log_plots=False,
            )
            diverged_flag = float(metrics.get(ROLLOUT_DIVERGED_KEY, 0.0))
            if np.isfinite(diverged_flag) and diverged_flag > 0.5:
                diverged_count += 1
            for name, value in metrics.items():
                if name == ROLLOUT_DIVERGED_KEY or not np.isfinite(float(value)):
                    continue
                metrics_sum[name] = metrics_sum.get(name, 0.0) + float(value)
                metrics_count[name] = metrics_count.get(name, 0) + 1
        for name, total in metrics_sum.items():
            writer.add_scalar(f"{ASYNC_VAL_SPLIT_TAG}/{name}", total / float(max(1, metrics_count.get(name, 0))), tb_step)
        writer.add_scalar(f"{ASYNC_VAL_SPLIT_TAG}/{ROLLOUT_DIVERGED_COUNT_KEY}", float(diverged_count), tb_step)

        rollout_idx = _rollout_index(
            len(val_trajs_plot),
            ur_values=ur_values_all,
        )
        rollout_traj = val_trajs_plot[rollout_idx]
        print(
            f"[async-val][vpinn] epoch {epoch + 1}: plot trajectory={rollout_traj.get('name', f'traj_{rollout_idx}')} "
            f"U_r={float(np.asarray(rollout_traj['ur']).reshape(-1)[0]):.6g} "
            f"dt={float(dt):.6g} rho={float(rho):.6g} D={float(diameter):.6g} "
            f"m={float(np.asarray(rollout_traj['dry_mass_kg' if td_mass_source == 'dry' else 'effective_mass_kg']).reshape(())):.6g} "
            f"c={float(np.asarray(rollout_traj['damping_c']).reshape(())):.6g} "
            f"k={float(np.asarray(rollout_traj['stiffness_n_m']).reshape(())):.6g}"
        )
        _log_td_correction_rollout_validation(
            writer=writer,
            epoch=tb_step,
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
            log_metrics=False,
            log_plots=True,
            log_spectra=True,
            tag_prefix=f"{ASYNC_VAL_SPLIT_TAG}/rollout",
        )
    return {
        "loss_total": (float(val_metrics["loss_total"]) if "loss_total" in val_metrics else None),
        "val_metrics": val_metrics,
    }


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
    model.eval()
    # Keep validation rollout-loss estimation deterministic to control cost and variance.
    val_rollout_loss_mode = "deterministic"
    val_rollout_stochastic_samples = 1
    amp_enabled = bool(amp_enabled) and device.type == "cuda"
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
                    if getattr(model, "force_output", "force") == "coefficient":
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
                    z_flip = -z_mid
                    if getattr(model, "force_output", "force") == "coefficient":
                        f_pos = model.u_theta_coeff(z_mid, reduced_velocity=ur_i)
                        f_neg = model.u_theta_coeff(z_flip, reduced_velocity=ur_i)
                    else:
                        f_pos = model.u_theta(z_mid, reduced_velocity=ur_i)
                        f_neg = model.u_theta(z_flip, reduced_velocity=ur_i)
                    sym_res = f_pos + f_neg
                    if sym_res.ndim == 1:
                        sym_res = sym_res.unsqueeze(-1)
                    if symmetry_norm == "l1":
                        per_sym = torch.mean(torch.abs(sym_res), dim=1)
                    else:
                        per_sym = torch.mean(sym_res * sym_res, dim=1)
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
            sigma_sum = sigma_sum + sigma_reg_loss.detach().float()
            mean_reg_sum = mean_reg_sum + mean_reg_loss.detach().float()
            data_sum = data_sum + data_force_loss.detach().float()
            sym_sum = sym_sum + sym_loss.detach().float()
            rollout_det_sum = rollout_det_sum + rollout_det_loss.detach().float()
            batches += 1

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
                per_reg = float(sigma_reg) * per_sigma_reg
                per_reg_mean = float(mean_reg) * per_mean_reg

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
                    z_flip = -z_mid
                    if getattr(model, "force_output", "force") == "coefficient":
                        f_pos = model.u_theta_coeff(z_mid, reduced_velocity=ur_i)
                        f_neg = model.u_theta_coeff(z_flip, reduced_velocity=ur_i)
                    else:
                        f_pos = model.u_theta(z_mid, reduced_velocity=ur_i)
                        f_neg = model.u_theta(z_flip, reduced_velocity=ur_i)
                    sym_res = f_pos + f_neg
                    if sym_res.ndim == 1:
                        sym_res = sym_res.unsqueeze(-1)
                    if symmetry_norm == "l1":
                        per_sym = torch.mean(torch.abs(sym_res), dim=1)
                    else:
                        per_sym = torch.mean(sym_res * sym_res, dim=1)
                else:
                    per_sym = per_res.new_zeros(per_res.shape)

            ur_vals = ur_i.detach().cpu().view(-1).numpy()
            per_res_vals = per_res.detach().cpu().view(-1).numpy()
            per_reg_vals = per_reg.detach().cpu().view(-1).numpy()
            per_reg_mean_vals = per_reg_mean.detach().cpu().view(-1).numpy()
            per_data_vals = per_data.detach().cpu().view(-1).numpy()
            per_sym_vals = per_sym.detach().cpu().view(-1).numpy()
            for u, res_v, reg_v, mean_v, data_v, sym_v in zip(
                ur_vals, per_res_vals, per_reg_vals, per_reg_mean_vals, per_data_vals, per_sym_vals
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


def _amp_dtype(name: str) -> torch.dtype:
    key = str(name).lower()
    if key == "fp16":
        return torch.float16
    if key == "bf16":
        return torch.bfloat16
    return torch.bfloat16


def _per_ur_loss_map_vpinn(
    *,
    model: torch.nn.Module,
    loader: Any,
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
                if not expect_f0:
                    raise ValueError("Unexpected batch format (received f0 when coefficient mode is disabled).")
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Async validation runner.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--do-losses", type=int, default=1)
    parser.add_argument("--do-rollout", type=int, default=1)
    args = parser.parse_args()

    _set_threading(int(args.num_threads))
    device = torch.device(str(args.device))

    ckpt, cfg, method = _load_checkpoint(args.checkpoint)

    writer = SummaryWriter(log_dir=str(args.log_dir))
    summary: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "epoch": int(args.epoch),
        "loss_total": None,
        "method": method,
        "run_name": ckpt.get("run_name"),
        "status": "started",
    }
    try:
        validation_start = time.perf_counter()
        if method in {"hnn", "phnn"}:
            if not bool(ckpt.get("td_correction", False)):
                raise ValueError("PHNN async validation now only supports TD-correction checkpoints.")
            summary.update(_run_hnn_td_correction_validation(
                ckpt=ckpt,
                cfg=cfg,
                device=device,
                writer=writer,
                epoch=int(args.epoch),
                do_losses=bool(int(args.do_losses)),
                do_rollout=bool(int(args.do_rollout)),
                num_workers=int(args.num_workers),
            ))
        elif method == "vpinn":
            if not bool(ckpt.get("td_correction", False)):
                raise ValueError("VPINN async validation now only supports TD-correction checkpoints.")
            summary.update(_run_vpinn_td_correction_validation(
                ckpt=ckpt,
                cfg=cfg,
                device=device,
                writer=writer,
                epoch=int(args.epoch),
                do_losses=bool(int(args.do_losses)),
                do_rollout=bool(int(args.do_rollout)),
                num_workers=int(args.num_workers),
            ))
        else:
            raise ValueError(f"Unsupported method '{method}'.")
        elapsed = time.perf_counter() - validation_start
        summary["status"] = "completed"
        summary["validation_wall_time_s"] = float(elapsed)
        tb_step = int(args.epoch) + 1
        writer.add_scalar(f"{ASYNC_VAL_SPLIT_TAG}/validation_wall_time_s", float(elapsed), tb_step)
        writer.flush()
        summary_path = _async_summary_path(args.log_dir, int(args.epoch))
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    finally:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()
