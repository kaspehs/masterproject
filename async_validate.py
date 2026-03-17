"""
Asynchronous validation runner.

Loads a checkpoint saved during training and runs validation on the val split.
Intended to be spawned as a child process so training can continue.
"""

from __future__ import annotations

import argparse
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
    resolve_cut_start_seconds,
    resolve_middle_time_plot,
    sample_indices_per_ur,
    scaled_residual_loss_per_sample,
    sample_one_index_per_ur,
    resolve_td_correction_params,
)
from methods.hnn.trainer import (
    _build_td_correction_hnn_loaders,
    _log_td_correction_rollout_validation as _hnn_td_rollout_validation,
    _td_correction_state_rollout,
    _td_predict_correction,
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
    _vpinn_predict_correction,
    _vpinn_td_rollout,
    rollout_rk4,
    _weak_residual,
)


def _set_threading(num_threads: int) -> None:
    num_threads = max(1, int(num_threads))
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(max(1, min(4, num_threads)))


def _rollout_index(
    epoch: int,
    rollout_every: int,
    num_series: int,
    cycle: bool,
    *,
    ur_values: list[float] | None = None,
    target_ur: float | None = None,
    target_ur_tol: float = 1e-6,
) -> int:
    if num_series <= 0:
        return 0
    selected = list(range(num_series))
    if ur_values is not None and len(ur_values) == num_series:
        if target_ur is not None:
            matched = [
                idx
                for idx, ur_val in enumerate(ur_values)
                if np.isclose(float(ur_val), float(target_ur), rtol=0.0, atol=float(target_ur_tol))
            ]
            if matched:
                selected = [matched[0]]
        else:
            sampled = sample_one_index_per_ur(ur_values, seed=0)
            if sampled:
                selected = sampled
    if not cycle:
        return int(selected[0])
    step = max(0, (epoch + 1) // max(1, int(rollout_every)) - 1)
    return int(selected[step % len(selected)])


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


def _ur_bin_id(value: float, ur_bin_size: float) -> int:
    return int(np.rint(float(value) / float(ur_bin_size)))


def _collect_ur_bin_counts_from_dataset(
    dataset: Any,
    *,
    ur_tensor_index: int,
    ur_bin_size: float,
) -> dict[int, int]:
    cache_key = f"ur_bin_counts:{int(ur_tensor_index)}:{float(ur_bin_size):.12g}"
    cache = getattr(dataset, "_codex_cache", None)
    if isinstance(cache, dict) and cache_key in cache:
        return dict(cache[cache_key])

    counts: dict[int, int] = {}
    if isinstance(dataset, TensorDataset):
        ur_tensor = dataset.tensors[int(ur_tensor_index)]
        ur_vals = ur_tensor.reshape(ur_tensor.shape[0], -1)[:, 0].detach().cpu().numpy()
        for ur_val in ur_vals:
            key = _ur_bin_id(float(ur_val), ur_bin_size)
            counts[key] = counts.get(key, 0) + 1
    elif isinstance(dataset, ConcatDataset):
        for subdataset in dataset.datasets:
            sub_counts = _collect_ur_bin_counts_from_dataset(
                subdataset,
                ur_tensor_index=ur_tensor_index,
                ur_bin_size=ur_bin_size,
            )
            for key, value in sub_counts.items():
                counts[key] = counts.get(key, 0) + int(value)
    else:
        raise TypeError(f"Unsupported dataset type for U_r bin counting: {type(dataset)!r}")

    if cache is None or not isinstance(cache, dict):
        cache = {}
        setattr(dataset, "_codex_cache", cache)
    cache[cache_key] = dict(counts)
    return counts


def _weighted_mean_by_ur_bins(
    per_sample: torch.Tensor,
    ur_values: torch.Tensor,
    *,
    ur_bin_counts: dict[int, int] | None,
    ur_bin_size: float,
) -> torch.Tensor:
    if not ur_bin_counts:
        return torch.mean(per_sample)
    ur_flat = ur_values.reshape(-1).detach().cpu().numpy()
    weights = [1.0 / float(max(1, ur_bin_counts.get(_ur_bin_id(float(ur), ur_bin_size), 1))) for ur in ur_flat]
    weight_t = torch.as_tensor(weights, device=per_sample.device, dtype=per_sample.dtype)
    denom = torch.clamp(torch.sum(weight_t), min=torch.finfo(weight_t.dtype).eps)
    return torch.sum(weight_t * per_sample) / denom


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
        "f": torch.from_numpy(np.ascontiguousarray(traj["force_total"])).float().unsqueeze(1),
        "td_force": torch.from_numpy(np.ascontiguousarray(traj["force_td"])).float().unsqueeze(1),
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
    middle_time_plot: list[float] | tuple[float, float],
    device: torch.device,
    tag_prefix: str = "val/rollout",
    step: int | None = None,
    log_metrics: bool = True,
    log_plots: bool = True,
    title_suffix: str = "",
) -> dict[str, float]:
    traj_t = _td_hnn_traj_to_tensors(traj, mass_source=td_mass_source)
    y_true_t = traj_t["y"].to(device)
    v_true_t = traj_t["v"].to(device)
    z_true_t = traj_t["z"].to(device)
    f_true_t = traj_t["f"].to(device)
    td_force_t = traj_t["td_force"].to(device)
    ur_t = traj_t["ur"].to(device)
    td_context_t = traj_t["td_context"].to(device)
    t_np = traj_t["t"].detach().cpu().numpy()
    mass_value = float(traj_t["mass_value"])
    damping_value = float(traj_t["damping_value"])
    stiffness_value = float(traj_t["stiffness_value"])
    if z_true_t.shape[0] < 2:
        return {}

    mass_t = torch.full((1, 1), mass_value, dtype=z_true_t.dtype, device=device)
    damping_t = torch.full((1, 1), damping_value, dtype=z_true_t.dtype, device=device)
    stiffness_t = torch.full((1, 1), stiffness_value, dtype=z_true_t.dtype, device=device)

    z_pred, total_force_seq, corr_mu_seq = _td_correction_state_rollout(
        model=model,
        z0=z_true_t[0:1],
        ur0=ur_t[0:1],
        td_context0=td_context_t[0:1],
        steps=int(z_true_t.shape[0] - 1),
        dt=dt,
        structural_mass=mass_t,
        damping_c=damping_t,
        stiffness=stiffness_t,
        td_params=td_params,
    )
    y_pred = z_pred[0, :, 0].detach().cpu().numpy()
    v_pred = (z_pred[0, :, 1] / mass_value).detach().cpu().numpy()
    force_roll = total_force_seq[0, :, 0].detach().cpu().numpy()
    corr_roll = corr_mu_seq[0, :, 0].detach().cpu().numpy()
    td_roll = force_roll - corr_roll

    with torch.no_grad():
        corr0 = model.learned_force(z_true_t[0:1], reduced_velocity=ur_t[0:1])[0, 0]
        corr_on_data_t = model.learned_force(z_true_t, reduced_velocity=ur_t)

    force_total_full = np.concatenate(
        [np.asarray([float(td_force_t[0, 0].detach().cpu() + corr0.detach().cpu())]), force_roll],
        axis=0,
    )
    force_td_full = np.concatenate(
        [td_force_t[:1, 0].detach().cpu().numpy(), td_roll],
        axis=0,
    )

    metrics = compute_validation_metrics(
        model=model,
        y_data_t=y_true_t[:, 0],
        val_vel=v_true_t[:, 0],
        reduced_velocity=ur_t[:, 0],
        m_eff=mass_value,
        dt=dt,
        t=t_np,
        y_data_raw=y_true_t[:, 0].detach().cpu().numpy(),
        force_data=f_true_t[:, 0].detach().cpu().numpy(),
        D=float(model.D),
        k=stiffness_value,
        device=device,
        rollout={
            "y_norm": y_pred / float(model.D),
            "p_norm": (v_pred / (float(np.sqrt(stiffness_value / mass_value)) * float(model.D))),
            "force_total": force_total_full,
        },
    )

    force_true = f_true_t[:, 0].detach().cpu().numpy()
    force_model_on_data = (td_force_t + corr_on_data_t.to(device))[:, 0].detach().cpu().numpy()
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
        middle_mask = create_window_mask(t_np, middle_time_plot)
        middle_window = (float(middle_time_plot[0]), float(middle_time_plot[1]))
        ur_val = float(ur_t[0, 0].detach().cpu().item())
        log_displacement_plots(
            writer,
            epoch,
            t_np,
            y_true_t[:, 0].detach().cpu().numpy() / float(model.D),
            y_pred / float(model.D),
            v_pred / (float(np.sqrt(stiffness_value / mass_value)) * float(model.D)),
            zoom_mask,
            middle_mask,
            middle_window,
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
            create_window_mask(force_t, middle_time_plot),
            middle_window,
            reduced_velocity=ur_val,
            force_coeff_baseline=force_td_full[:n_force],
            baseline_label="C_F (Vivana-TD)",
            tag_prefix=tag_prefix,
            step=step,
            title_suffix=title_suffix,
        )
    return metrics


def _run_hnn_validation(
    *,
    ckpt: dict[str, Any],
    cfg: Any,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    rollout_every: int,
    cycle_rollout: bool,
    rollout_target_ur: float | None,
    rollout_target_ur_tol: float,
    do_losses: bool,
    do_rollout: bool,
    num_workers: int,
) -> None:
    data_cfg = cfg.data
    monitoring_cfg = cfg.monitoring
    hnn_cfg = dict(cfg.hnn or {})
    velocity_source = str(hnn_cfg.get("velocity_source", "compute")).strip().lower()
    rollout_stochastic = bool(hnn_cfg.get("rollout_stochastic", False))
    rollout_noise_scale = float(hnn_cfg.get("rollout_noise_scale", 1.0))
    if not np.isfinite(rollout_noise_scale) or rollout_noise_scale < 0.0:
        raise ValueError("hnn.rollout_noise_scale must be finite and non-negative.")
    rollout_seed_raw = hnn_cfg.get("rollout_seed", None)
    rollout_seed = None if rollout_seed_raw is None else int(rollout_seed_raw)
    loss_cfg = cfg.loss
    fixed_validation_sampling = bool(getattr(monitoring_cfg, "fixed_validation_sampling", False))
    validation_sampling_seed = int(getattr(monitoring_cfg, "validation_sampling_seed", 1))
    validation_samples_per_ur = max(1, int(getattr(monitoring_cfg, "validation_samples_per_ur", 1)))
    rollout_det_weight = float(getattr(loss_cfg, "rollout_det_weight", 0.0))
    rollout_det_steps = int(getattr(loss_cfg, "rollout_det_steps", 0))
    rollout_loss_mode = str(getattr(loss_cfg, "rollout_loss_mode", "deterministic")).strip().lower()
    rollout_stochastic_samples = int(getattr(loss_cfg, "rollout_stochastic_samples", 1))
    equalize_residual_over_ur_bins = bool(getattr(loss_cfg, "equalize_residual_over_ur_bins", False))
    equalize_rollout_over_ur_bins = bool(getattr(loss_cfg, "equalize_rollout_over_ur_bins", False))
    ur_bin_size = float(getattr(loss_cfg, "ur_bin_size", 1e-6))
    normalize_residual_by_ur_bin_std = bool(getattr(loss_cfg, "normalize_residual_by_ur_bin_std", False))
    normalize_rollout_by_ur_bin_std = bool(getattr(loss_cfg, "normalize_rollout_by_ur_bin_std", False))
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

    if bool(getattr(data_cfg, "use_generated_train_series", False)):
        val_dir = Path(data_cfg.train_series_dir) / "val"
        if not val_dir.exists():
            raise FileNotFoundError(f"Validation directory '{val_dir}' not found.")
        val_files = sorted(val_dir.glob("*.npz"))
        if not val_files:
            raise FileNotFoundError(f"No '.npz' files found in '{val_dir}'.")
        data_path = val_files[0]
    else:
        data_path = Path(data_cfg.file)

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

    if bool(getattr(data_cfg, "use_generated_train_series", False)):
        series_dir = Path(data_cfg.train_series_dir) / "val"
        val_require_force = bool(getattr(loss_cfg, "use_force_data_loss", False) or has_force_data)
        val_series_raw, _ = load_training_series(
            y_data,
            t,
            dt,
            True,
            series_dir,
            m_eff,
            device,
            smoothing_cfg=cfg.smoothing,
            velocity_source=velocity_source,
            eval_velocity=vel_data,
            eval_reduced_velocity=reduced_velocity,
            require_force=val_require_force,
            eval_force=(F_data if has_force_data else None),
            cut_start_seconds=val_cut,
        )
    else:
        val_require_force = bool(getattr(loss_cfg, "use_force_data_loss", False) or has_force_data)
        val_series_raw, _ = load_training_series(
            y_data,
            t,
            dt,
            False,
            Path("."),
            m_eff,
            device,
            smoothing_cfg=cfg.smoothing,
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
        batch_size=int(cfg.training.batch_size),
        device=device,
        smoothing_cfg=cfg.smoothing,
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
            smoothing_cfg=cfg.smoothing,
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
            equalize_residual_over_ur_bins=equalize_residual_over_ur_bins,
            equalize_rollout_over_ur_bins=equalize_rollout_over_ur_bins,
            ur_bin_size=ur_bin_size,
            normalize_residual_by_ur_bin_std=normalize_residual_by_ur_bin_std,
            normalize_rollout_by_ur_bin_std=normalize_rollout_by_ur_bin_std,
            ur_bin_state_scale_info=ur_bin_state_scale_info,
            rollout_det_weight=rollout_det_weight,
            rollout_loss_mode=rollout_loss_mode,
            rollout_stochastic_samples=rollout_stochastic_samples,
            rollout_noise_scale=rollout_noise_scale,
            force_reg_on_coeff=bool(getattr(loss_cfg, "force_reg_on_coeff", False)),
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
            writer.add_scalar(f"val/{name}", value_f, epoch)
            num_loss_scalars_written += 1
        loss_by_ur = _per_ur_loss_map_hnn(
            model=model,
            loader=val_loader,
            rollout_loader=val_rollout_loader,
            device=device,
            non_blocking=(device.type == "cuda"),
            mean_reg=mean_reg,
            mean_reg_norm=mean_reg_norm,
            sigma_reg=float(loss_cfg.sigma_reg),
            sigma_reg_norm=sigma_reg_norm,
            normalize_residual_by_ur_bin_std=normalize_residual_by_ur_bin_std,
            normalize_rollout_by_ur_bin_std=normalize_rollout_by_ur_bin_std,
            ur_bin_state_scale_info=ur_bin_state_scale_info,
            ur_bin_size=ur_bin_size,
            rollout_det_weight=rollout_det_weight,
            rollout_loss_mode=rollout_loss_mode,
            rollout_stochastic_samples=rollout_stochastic_samples,
            rollout_noise_scale=rollout_noise_scale,
            force_reg_on_coeff=bool(getattr(loss_cfg, "force_reg_on_coeff", False)),
            use_force_data_loss=bool(getattr(loss_cfg, "use_force_data_loss", False)),
            force_data_weight=float(getattr(loss_cfg, "force_data_weight", 1.0)),
            symmetry_weight=symmetry_weight,
            symmetry_norm=symmetry_norm,
            amp_enabled=amp_enabled,
            amp_dtype=_amp_dtype(cfg.precision.amp_dtype),
        )
        log_loss_vs_ur(
            writer,
            epoch,
            loss_by_ur,
            tag="val/loss_vs_ur",
            title="Validation loss vs U_r",
        )

    if do_rollout:
        metrics_sum: dict[str, float] = {}
        count = 0
        diverged_count = 0
        total = min(len(val_series_raw), len(val_sequences))
        ur_for_sampling: list[float] = []
        for idx in range(total):
            ur_arr = np.asarray(val_series_raw[idx][5]).reshape(-1)
            ur_for_sampling.append(float(ur_arr[0]) if ur_arr.size > 0 else float("nan"))
        sample_seed = int(validation_sampling_seed) if fixed_validation_sampling else (int(epoch) + 1)
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
                writer.add_scalar(f"val/{name}", value_f, epoch)
                num_rollout_scalars_written += 1
            writer.add_scalar(f"val/{ROLLOUT_DIVERGED_COUNT_KEY}", float(diverged_count), epoch)
            num_rollout_scalars_written += 1

        ur_values = [float(np.asarray(series_raw[5]).reshape(-1)[0]) for series_raw in val_series_raw]
        rollout_idx = _rollout_index(
            epoch,
            rollout_every,
            len(val_series_raw),
            cycle_rollout,
            ur_values=ur_values,
            target_ur=rollout_target_ur,
            target_ur_tol=rollout_target_ur_tol,
        )
        y_np, t_np, dt_value, _vel_np, force_np, _ur_np = val_series_raw[rollout_idx]
        y_tensor, vel_tensor, _t_tensor, ur_tensor = val_sequences[rollout_idx]
        log_validation_epoch(
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
            getattr(data_cfg, "middle_time_plot", [0.0, 1.0]),
            hamiltonian_data,
            log_extra_metrics=bool(getattr(cfg.monitoring, "log_extra_validation_metrics", False)),
            log_metrics=False,
            rollout_stochastic=rollout_stochastic,
            rollout_noise_scale=rollout_noise_scale,
            rollout_seed=rollout_seed,
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
    rollout_every: int,
    cycle_rollout: bool,
    rollout_target_ur: float | None,
    rollout_target_ur_tol: float,
    do_losses: bool,
    do_rollout: bool,
    num_workers: int,
) -> None:
    data_cfg = cfg.data
    monitoring_cfg = cfg.monitoring
    hnn_cfg = dict(cfg.hnn or {})
    loss_cfg = cfg.loss
    td_mass_source = str(hnn_cfg.get("td_mass_source", "dry")).strip().lower()
    if td_mass_source not in {"dry", "effective"}:
        raise ValueError("hnn.td_mass_source must be one of: dry, effective.")

    train_series_root = Path(data_cfg.train_series_dir)
    val_dir = train_series_root / "val"
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation directory '{val_dir}' not found.")
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
    td_params = resolve_td_correction_params(hnn_cfg)
    predict_sigma = bool(hnn_cfg.get("predict_sigma", False))
    force_zero_output = bool(hnn_cfg.get("force_zero_output", False))
    rollout_det_weight = float(getattr(loss_cfg, "rollout_det_weight", 0.0))
    rollout_det_steps = int(getattr(loss_cfg, "rollout_det_steps", 0))
    rollout_batch_size_raw = int(getattr(loss_cfg, "rollout_det_batch_size", 0))
    rollout_batch_size = int(cfg.training.batch_size) if rollout_batch_size_raw <= 0 else rollout_batch_size_raw
    mean_reg = float(getattr(loss_cfg, "mean_reg", 0.0))
    sigma_reg = float(getattr(loss_cfg, "sigma_reg", 0.0))
    mean_reg_norm = str(getattr(loss_cfg, "mean_reg_norm", "l1")).strip().lower()
    sigma_reg_norm = str(getattr(loss_cfg, "sigma_reg_norm", "l2")).strip().lower()
    force_data_weight = float(getattr(loss_cfg, "force_data_weight", 1.0))
    use_force_data_loss = bool(getattr(loss_cfg, "use_force_data_loss", True))
    fixed_validation_sampling = bool(getattr(monitoring_cfg, "fixed_validation_sampling", False))
    validation_sampling_seed = int(getattr(monitoring_cfg, "validation_sampling_seed", 1))
    validation_samples_per_ur = max(1, int(getattr(monitoring_cfg, "validation_samples_per_ur", 1)))

    model_dict = asdict(cfg.model)
    first_val_traj = val_trajs_np[0]
    mass_key = "dry_mass_kg" if td_mass_source == "dry" else "effective_mass_kg"
    model_dict["structural_mass"] = float(np.asarray(first_val_traj[mass_key]).reshape(()))
    model_dict["k"] = float(np.asarray(first_val_traj["stiffness_n_m"]).reshape(()))
    model_dict["damping_c"] = float(np.asarray(first_val_traj["damping_c"]).reshape(()))
    model_dict["Ca"] = 0.0
    model_dict["include_physical_drag"] = False
    model_dict["use_stochastic_process_noise"] = predict_sigma
    arch_dict = asdict(cfg.architecture)
    model, _derived = PHVIV.from_config(dt=dt, cfg=model_dict, arch_cfg=arch_dict, device=device)
    _load_state(model, ckpt["model_state"])
    model.eval()

    _train_loader, val_loader, rollout_loader = _build_td_correction_hnn_loaders(
        train_trajs=val_trajs_np,
        val_trajs=val_trajs_np,
        mass_source=td_mass_source,
        batch_size=int(cfg.training.batch_size),
        rollout_batch_size=rollout_batch_size,
        rollout_steps=rollout_det_steps,
        num_workers=int(num_workers),
        pin_memory=(device.type == "cuda"),
    )

    def _reg(value: torch.Tensor, norm: str) -> torch.Tensor:
        return torch.mean(torch.abs(value)) if str(norm).strip().lower() == "l1" else torch.mean(value * value)

    num_loss_scalars_written = 0
    num_rollout_scalars_written = 0

    if do_losses and val_loader is not None:
        val_sums = {
            name: torch.zeros((), device=device)
            for name in ["loss_total", "loss_state", "loss_data", "loss_reg_mean", "loss_reg_sigma"]
        }
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) != 10:
                    raise ValueError("Unexpected TD correction HNN batch format.")
                z_i, t_i, z_next, t_next, ur_i, corr_next, td_force_next, mass_i, damping_i, stiffness_i = batch
                z_i = z_i.to(device, non_blocking=(device.type == "cuda"))
                t_i = t_i.to(device, non_blocking=(device.type == "cuda"))
                z_next = z_next.to(device, non_blocking=(device.type == "cuda"))
                t_next = t_next.to(device, non_blocking=(device.type == "cuda"))
                ur_i = ur_i.to(device, non_blocking=(device.type == "cuda"))
                corr_next = corr_next.to(device, non_blocking=(device.type == "cuda"))
                td_force_next = td_force_next.to(device, non_blocking=(device.type == "cuda"))
                mass_i = mass_i.to(device, non_blocking=(device.type == "cuda"))
                damping_i = damping_i.to(device, non_blocking=(device.type == "cuda"))
                stiffness_i = stiffness_i.to(device, non_blocking=(device.type == "cuda"))

                corr_mu, sigma_corr = _td_predict_correction(
                    model,
                    z=z_i,
                    reduced_velocity=ur_i,
                    structural_mass=mass_i,
                    stiffness=stiffness_i,
                    predict_sigma=predict_sigma,
                    force_zero_output=force_zero_output,
                )
                dt_i = torch.clamp(t_next - t_i, min=1.0e-12)
                total_force_next = td_force_next + corr_mu
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
                if predict_sigma:
                    var_p = torch.clamp((dt_i * sigma_corr) ** 2, min=1e-9)
                    var_y = torch.clamp(((0.5 * (dt_i ** 2) / mass_i) * sigma_corr) ** 2, min=1e-9)
                    nll_y = 0.5 * (((z_next[:, 0:1] - z_next_mean[:, 0:1]) ** 2) / var_y + torch.log(var_y))
                    nll_p = 0.5 * (((z_next[:, 1:2] - z_next_mean[:, 1:2]) ** 2) / var_p + torch.log(var_p))
                    state_loss = torch.mean(nll_y + nll_p)
                else:
                    state_loss = torch.mean(torch.sum((z_next_mean - z_next) ** 2, dim=1))
                if use_force_data_loss:
                    if predict_sigma:
                        var = torch.clamp(sigma_corr * sigma_corr, min=1e-9)
                        data_loss = torch.mean(0.5 * (((corr_next - corr_mu) ** 2) / var + torch.log(var)))
                    else:
                        data_loss = torch.mean((corr_next - corr_mu) ** 2)
                else:
                    data_loss = state_loss.new_tensor(0.0)
                mean_reg_loss = _reg(corr_mu, mean_reg_norm)
                sigma_reg_loss = _reg(sigma_corr, sigma_reg_norm) if predict_sigma else state_loss.new_tensor(0.0)
                total_loss = state_loss + float(force_data_weight) * data_loss + float(mean_reg) * mean_reg_loss + float(sigma_reg) * sigma_reg_loss
                val_sums["loss_total"] += total_loss.detach()
                val_sums["loss_state"] += state_loss.detach()
                val_sums["loss_data"] += data_loss.detach()
                val_sums["loss_reg_mean"] += mean_reg_loss.detach()
                val_sums["loss_reg_sigma"] += sigma_reg_loss.detach()
                val_count += 1
        val_denom = float(max(1, val_count))
        for name, value in val_sums.items():
            writer.add_scalar(f"val/{name}", float((value / val_denom).detach().cpu()), epoch)
            num_loss_scalars_written += 1

        if rollout_loader is not None and rollout_det_weight > 0.0:
            rollout_loss_sum = torch.zeros((), device=device)
            rollout_count = 0
            with torch.no_grad():
                for rollout_batch in rollout_loader:
                    if len(rollout_batch) != 8:
                        raise ValueError("Unexpected TD correction rollout batch format.")
                    z0, t_seq, z_traj, ur0, td_context0, mass0, damping0, stiffness0 = rollout_batch
                    z0 = z0.to(device, non_blocking=(device.type == "cuda"))
                    t_seq = t_seq.to(device, non_blocking=(device.type == "cuda"))
                    z_traj = z_traj.to(device, non_blocking=(device.type == "cuda"))
                    ur0 = ur0.to(device, non_blocking=(device.type == "cuda"))
                    td_context0 = td_context0.to(device, non_blocking=(device.type == "cuda"))
                    mass0 = mass0.to(device, non_blocking=(device.type == "cuda"))
                    damping0 = damping0.to(device, non_blocking=(device.type == "cuda"))
                    stiffness0 = stiffness0.to(device, non_blocking=(device.type == "cuda"))
                    dt_roll = torch.clamp((t_seq[:, 1] - t_seq[:, 0]).unsqueeze(1), min=1.0e-12)
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
                        force_zero_output=force_zero_output,
                    )
                    rollout_loss_sum += torch.mean(torch.sum((z_pred - z_traj) ** 2, dim=2)).detach()
                    rollout_count += 1
            writer.add_scalar("val/loss_rollout_det", float((rollout_loss_sum / float(max(1, rollout_count))).detach().cpu()), epoch)
            num_loss_scalars_written += 1

    if do_rollout:
        ur_values_all = [float(np.asarray(traj["ur"]).reshape(-1)[0]) for traj in val_trajs_np]
        sample_seed = int(validation_sampling_seed) if fixed_validation_sampling else (int(epoch) + 1)
        sampled_metric_indices = sample_indices_per_ur(
            ur_values_all,
            samples_per_ur=validation_samples_per_ur,
            seed=sample_seed,
        )
        sampled_names = [str(val_trajs_np[idx].get("name", f"traj_{idx}")) for idx in sampled_metric_indices]
        print(
            f"[async-val][phnn] epoch {epoch + 1}: sampled metric trajectories={sampled_names} "
            f"(force_zero_output={force_zero_output}, mass_source={td_mass_source})"
        )
        metrics_sum: dict[str, float] = {}
        metrics_count: dict[str, int] = {}
        diverged_count = 0
        middle_time_plot = resolve_middle_time_plot(data_cfg, hnn_cfg, method_name="hnn")
        for sidx in sampled_metric_indices:
            metrics = _hnn_td_rollout_validation(
                writer=writer,
                epoch=epoch,
                model=model,
                traj=val_trajs_np[sidx],
                dt=dt,
                td_mass_source=td_mass_source,
                td_params=td_params,
                middle_time_plot=middle_time_plot,
                device=device,
                force_zero_output=force_zero_output,
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
            writer.add_scalar(f"val/{name}", total / float(max(1, metrics_count.get(name, 0))), epoch)
            num_rollout_scalars_written += 1
        writer.add_scalar(f"val/{ROLLOUT_DIVERGED_COUNT_KEY}", float(diverged_count), epoch)
        num_rollout_scalars_written += 1

        rollout_idx = _rollout_index(
            epoch,
            rollout_every,
            len(val_trajs_np),
            cycle_rollout,
            ur_values=ur_values_all,
            target_ur=rollout_target_ur,
            target_ur_tol=rollout_target_ur_tol,
        )
        rollout_traj = val_trajs_np[rollout_idx]
        rollout_dt = float(np.asarray(rollout_traj["t"])[1] - np.asarray(rollout_traj["t"])[0])
        print(
            f"[async-val][phnn] epoch {epoch + 1}: plot trajectory={rollout_traj.get('name', f'traj_{rollout_idx}')} "
            f"U_r={float(np.asarray(rollout_traj['ur']).reshape(-1)[0]):.6g} "
            f"dt={rollout_dt:.6g} rho={float(model.rho):.6g} D={float(model.D):.6g} "
            f"m={float(np.asarray(rollout_traj['dry_mass_kg' if td_mass_source == 'dry' else 'effective_mass_kg']).reshape(())):.6g} "
            f"c={float(np.asarray(rollout_traj['damping_c']).reshape(())):.6g} "
            f"k={float(np.asarray(rollout_traj['stiffness_n_m']).reshape(())):.6g}"
        )
        _hnn_td_rollout_validation(
            writer=writer,
            epoch=epoch + 1,
            model=model,
            traj=rollout_traj,
            dt=dt,
            td_mass_source=td_mass_source,
            td_params=td_params,
            middle_time_plot=middle_time_plot,
            device=device,
            force_zero_output=force_zero_output,
            log_metrics=False,
            log_plots=True,
        )

    print(
        f"[async-val] epoch {epoch}: HNN TD scalar writes "
        f"(loss={num_loss_scalars_written}, rollout={num_rollout_scalars_written})"
    )


def _run_vpinn_validation(
    *,
    ckpt: dict[str, Any],
    cfg: Any,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    rollout_every: int,
    cycle_rollout: bool,
    rollout_target_ur: float | None,
    rollout_target_ur_tol: float,
    do_losses: bool,
    do_rollout: bool,
    num_workers: int,
) -> None:
    data_cfg = cfg.data
    monitoring_cfg = cfg.monitoring
    vp = dict(cfg.vpinn or {})
    velocity_source = str(vp.get("velocity_source", "compute")).strip().lower()
    fixed_validation_sampling = bool(getattr(monitoring_cfg, "fixed_validation_sampling", False))
    validation_sampling_seed = int(getattr(monitoring_cfg, "validation_sampling_seed", 1))
    validation_samples_per_ur = max(1, int(getattr(monitoring_cfg, "validation_samples_per_ur", 1)))
    force_representation = str(vp.get("force_representation", "force")).strip().lower()
    if force_representation not in {"force", "coefficient"}:
        raise ValueError("vpinn.force_representation must be one of: force, coefficient.")
    num_poly = int(vp.get("num_poly_test", 2))
    num_sine = int(vp.get("num_sine_test", 0))

    if bool(getattr(data_cfg, "use_generated_train_series", False)):
        val_dir = Path(data_cfg.train_series_dir) / "val"
        if not val_dir.exists():
            raise FileNotFoundError(f"Validation directory '{val_dir}' not found.")
        val_files = sorted(val_dir.glob("*.npz"))
        if not val_files:
            raise FileNotFoundError(f"No '.npz' files found in '{val_dir}'.")
        sources = val_files
    else:
        sources = [Path(data_cfg.file)]

    val_trajs: list[dict[str, Any]] = []
    dt_ref: Optional[float] = None
    cut_start_seconds = float(
        data_cfg.cut_start_seconds_val
        if getattr(data_cfg, "cut_start_seconds_val", None) is not None
        else getattr(data_cfg, "cut_start_seconds", 0.0)
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
            velocity_source=velocity_source,
            smoothing_cfg=cfg.smoothing,
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
            writer.add_scalar(f"val/{name}", value, epoch)
        force_map = _force_mapping_nrmse_over_trajs(model=model, val_trajs=val_trajs, device=device)
        if force_map is not None:
            for k_name, v_value in force_map.items():
                writer.add_scalar(f"val/{k_name}", v_value, epoch)
        loss_by_ur = _per_ur_loss_map_vpinn(
            model=model,
            loader=val_loader,
            device=device,
            non_blocking=(device.type == "cuda"),
            dt=dt,
            m=m,
            c=c,
            k=k,
            w=w,
            wdot=wdot,
            alpha=alpha,
            use_force_loss=bool(vp.get("use_force_loss", True)),
            use_weak_loss=bool(vp.get("use_weak_loss", True)),
            rollout_force_steps=int(vp.get("rollout_force_steps", 0)),
            expect_f0=(force_representation == "coefficient"),
            amp_enabled=amp_enabled,
            amp_dtype=_amp_dtype(cfg.precision.amp_dtype),
        )
        log_loss_vs_ur(
            writer,
            epoch,
            loss_by_ur,
            tag="val/loss_vs_ur",
            title="Validation loss vs U_r",
        )

    if do_rollout:
        ur_values_all = [float(traj["ur"][0, 0].detach().cpu().item()) for traj in val_trajs]
        sample_seed = int(validation_sampling_seed) if fixed_validation_sampling else (int(epoch) + 1)
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
                epoch=epoch,
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
            writer.add_scalar(f"val/{name}", total / denom, epoch)

        ur_values = ur_values_all
        rollout_idx = _rollout_index(
            epoch,
            rollout_every,
            len(val_trajs),
            cycle_rollout,
            ur_values=ur_values,
            target_ur=rollout_target_ur,
            target_ur_tol=rollout_target_ur_tol,
        )
        _log_rollout_validation(
            writer=writer,
            epoch=epoch,
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
        )


def _run_vpinn_td_correction_validation(
    *,
    ckpt: dict[str, Any],
    cfg: Any,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    rollout_every: int,
    cycle_rollout: bool,
    rollout_target_ur: float | None,
    rollout_target_ur_tol: float,
    do_losses: bool,
    do_rollout: bool,
    num_workers: int,
) -> None:
    data_cfg = cfg.data
    monitoring_cfg = cfg.monitoring
    vp = dict(cfg.vpinn or {})
    probabilistic = bool(vp.get("predict_sigma", False))
    force_zero_output = bool(vp.get("force_zero_output", False))
    sigma_min = float(vp.get("sigma_min", 1e-6))
    td_mass_source = str(vp.get("td_mass_source", "dry")).strip().lower()
    if td_mass_source not in {"dry", "effective"}:
        raise ValueError("vpinn.td_mass_source must be one of: dry, effective.")

    train_series_root = Path(data_cfg.train_series_dir)
    val_dir = train_series_root / "val"
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation directory '{val_dir}' not found.")
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

    model = _build_force_model(cfg, input_dim=3, output_dim=(2 if probabilistic else 1)).to(device)
    _load_state(model, ckpt["model_state"])
    model.eval()

    window_M = int(vp.get("window_M", 50))
    stride = max(1, int(vp.get("stride", 1)))
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
        num_workers=int(num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_rollout_loader = (
        torch.utils.data.DataLoader(
            val_rollout_dataset,
            batch_size=int(cfg.training.batch_size),
            shuffle=False,
            num_workers=int(num_workers),
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )
        if val_rollout_dataset is not None
        else None
    )

    mean_reg = float(getattr(cfg.loss, "mean_reg", 0.0))
    sigma_reg = float(getattr(cfg.loss, "sigma_reg", 0.0))
    mean_reg_norm = str(getattr(cfg.loss, "mean_reg_norm", "l1")).strip().lower()
    sigma_reg_norm = str(getattr(cfg.loss, "sigma_reg_norm", "l2")).strip().lower()
    use_force_loss = bool(vp.get("use_force_loss", True))
    use_weak_loss = bool(vp.get("use_weak_loss", True))
    w, wdot, alpha = _test_functions(
        int(vp.get("window_M", 50)),
        dt,
        num_poly=int(vp.get("num_poly_test", 2)),
        num_sine=int(vp.get("num_sine_test", 0)),
    )
    w = w.to(device)
    wdot = wdot.to(device)
    alpha = alpha.to(device)

    if do_losses:
        val_sums = {
            name: torch.zeros((), device=device)
            for name in ["loss_total", "loss_data", "loss_physics", "loss_reg_mean", "loss_reg_sigma"]
        }
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                x_win, v_win, corr_true, td_force, ur_win, m_win, c_win, k_win = [
                    item.to(device, non_blocking=(device.type == "cuda")) for item in batch
                ]
                mu_corr, sigma_corr = _vpinn_predict_correction(
                    model,
                    x_win,
                    v_win,
                    ur_win,
                    probabilistic=probabilistic,
                    sigma_min=sigma_min,
                    force_zero_output=force_zero_output,
                )
                total_force_mu = td_force + mu_corr
                if use_force_loss:
                    if probabilistic:
                        var = torch.clamp(sigma_corr * sigma_corr, min=1e-9)
                        loss_data = torch.mean(0.5 * (((corr_true - mu_corr) ** 2) / var + torch.log(var)))
                    else:
                        loss_data = torch.mean((corr_true - mu_corr) ** 2)
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
                    if probabilistic:
                        coeff = (float(dt) * alpha.view(1, 1, -1, 1) * w.view(1, w.shape[0], w.shape[1], 1)) ** 2
                        weak_var = torch.sum(coeff * (sigma_corr.unsqueeze(1) ** 2), dim=2)
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
                total_loss = loss_data + loss_physics + float(mean_reg) * mean_reg_loss + float(sigma_reg) * sigma_reg_loss
                val_sums["loss_total"] += total_loss.detach()
                val_sums["loss_data"] += loss_data.detach()
                val_sums["loss_physics"] += loss_physics.detach()
                val_sums["loss_reg_mean"] += mean_reg_loss.detach()
                val_sums["loss_reg_sigma"] += sigma_reg_loss.detach()
                val_batches += 1
        val_denom = float(max(1, val_batches))
        for name, value in val_sums.items():
            writer.add_scalar(f"val/{name}", float((value / val_denom).detach().cpu()), epoch)

        if val_rollout_loader is not None and float(vp.get("rollout_force_weight", float(getattr(cfg.loss, "rollout_det_weight", 0.0)))) > 0.0 and rollout_steps > 0:
            roll_sum = torch.zeros((), device=device)
            roll_batches = 0
            with torch.no_grad():
                for rb in val_rollout_loader:
                    x0, v0, ur0, td0, x_true_seq, v_true_seq, m0, c0, k0 = [
                        item.to(device, non_blocking=(device.type == "cuda")) for item in rb
                    ]
                    x_seq, v_seq, _, _, _ = _vpinn_td_rollout(
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
                        probabilistic=probabilistic,
                        sigma_min=sigma_min,
                        force_zero_output=force_zero_output,
                    )
                    roll_sum += torch.mean((x_seq - x_true_seq) ** 2 + (v_seq - v_true_seq) ** 2).detach()
                    roll_batches += 1
            writer.add_scalar("val/loss_rollout_det", float((roll_sum / float(max(1, roll_batches))).detach().cpu()), epoch)

    if do_rollout:
        fixed_validation_sampling = bool(getattr(monitoring_cfg, "fixed_validation_sampling", False))
        validation_sampling_seed = int(getattr(monitoring_cfg, "validation_sampling_seed", 1))
        validation_samples_per_ur = max(1, int(getattr(monitoring_cfg, "validation_samples_per_ur", 1)))
        ur_values_all = [float(np.asarray(traj["ur"]).reshape(-1)[0]) for traj in val_trajs_np]
        sample_seed = int(validation_sampling_seed) if fixed_validation_sampling else (int(epoch) + 1)
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
        middle_time_plot = resolve_middle_time_plot(data_cfg, vp, method_name="vpinn")
        val_trajs_plot = [_td_rollout_traj_to_tensors(traj) for traj in val_trajs_np]
        for sidx in sampled_metric_indices:
            metrics = _log_td_correction_rollout_validation(
                writer=writer,
                epoch=epoch,
                model=model,
                traj=val_trajs_plot[sidx],
                dt=dt,
                td_mass_source=td_mass_source,
                rho=rho,
                diameter=diameter,
                td_params=td_params,
                middle_time_plot=middle_time_plot,
                device=device,
                sigma_min=sigma_min,
                probabilistic=probabilistic,
                force_zero_output=force_zero_output,
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
            writer.add_scalar(f"val/{name}", total / float(max(1, metrics_count.get(name, 0))), epoch)
        writer.add_scalar(f"val/{ROLLOUT_DIVERGED_COUNT_KEY}", float(diverged_count), epoch)

        rollout_idx = _rollout_index(
            epoch,
            rollout_every,
            len(val_trajs_plot),
            cycle_rollout,
            ur_values=ur_values_all,
            target_ur=rollout_target_ur,
            target_ur_tol=rollout_target_ur_tol,
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
            epoch=epoch,
            model=model,
            traj=rollout_traj,
            dt=dt,
            td_mass_source=td_mass_source,
            rho=rho,
            diameter=diameter,
            td_params=td_params,
            middle_time_plot=middle_time_plot,
            device=device,
            sigma_min=sigma_min,
            probabilistic=probabilistic,
            force_zero_output=force_zero_output,
            log_metrics=False,
            log_plots=True,
        )


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
    equalize_residual_over_ur_bins: bool,
    equalize_rollout_over_ur_bins: bool,
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
    force_reg_on_coeff: bool,
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
    residual_ur_bin_counts = (
        _collect_ur_bin_counts_from_dataset(
            loader.dataset,
            ur_tensor_index=4,
            ur_bin_size=ur_bin_size,
        )
        if equalize_residual_over_ur_bins
        else None
    )
    rollout_ur_bin_counts = (
        _collect_ur_bin_counts_from_dataset(
            rollout_loader.dataset,
            ur_tensor_index=3,
            ur_bin_size=ur_bin_size,
        )
        if equalize_rollout_over_ur_bins and rollout_loader is not None and float(rollout_det_weight) > 0.0
        else None
    )
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
                if equalize_residual_over_ur_bins or normalize_residual_by_ur_bin_std:
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
                if equalize_residual_over_ur_bins:
                    res_loss = _weighted_mean_by_ur_bins(
                        per_res if per_res is not None else model.res_loss_per_sample(
                            z_i,
                            t_i,
                            z_next,
                            t_next,
                            reduced_velocity=ur_i,
                        ),
                        ur_i,
                        ur_bin_counts=residual_ur_bin_counts,
                        ur_bin_size=ur_bin_size,
                    )
                else:
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
                    on_coeff=force_reg_on_coeff,
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
                    if equalize_rollout_over_ur_bins:
                        per_rollout = _rollout_loss_from_batch(
                            model=model,
                            batch=rollout_batch,
                            device=device,
                            non_blocking=non_blocking,
                            rollout_loss_mode=val_rollout_loss_mode,
                            rollout_stochastic_samples=val_rollout_stochastic_samples,
                            rollout_noise_scale=rollout_noise_scale,
                            ur_bin_state_scale_info=(ur_bin_state_scale_info if normalize_rollout_by_ur_bin_std else None),
                            ur_bin_size=ur_bin_size,
                            return_per_sample=True,
                        )
                        _z0, _t_seq, _z_traj, ur0, _history0, _scale = _parse_rollout_batch(rollout_batch)
                        rollout_det_loss = _weighted_mean_by_ur_bins(
                            per_rollout,
                            ur0,
                            ur_bin_counts=rollout_ur_bin_counts,
                            ur_bin_size=ur_bin_size,
                        )
                    else:
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
    equalize_residual_over_ur_bins: bool = False,
    equalize_rollout_over_ur_bins: bool = False,
    normalize_residual_by_ur_bin_std: bool,
    normalize_rollout_by_ur_bin_std: bool,
    ur_bin_state_scale_info: dict[str, Any] | None,
    ur_bin_size: float,
    rollout_det_weight: float,
    rollout_loss_mode: str,
    rollout_stochastic_samples: int,
    rollout_noise_scale: float,
    force_reg_on_coeff: bool,
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
                    on_coeff=force_reg_on_coeff,
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
    parser.add_argument("--rollout-every", type=int, default=1)
    parser.add_argument("--cycle-rollout", type=int, default=0)
    parser.add_argument("--rollout-target-ur", type=float, default=None)
    parser.add_argument("--rollout-target-ur-tol", type=float, default=1e-6)
    parser.add_argument("--do-losses", type=int, default=1)
    parser.add_argument("--do-rollout", type=int, default=1)
    args = parser.parse_args()

    _set_threading(int(args.num_threads))
    device = torch.device(str(args.device))

    ckpt, cfg, method = _load_checkpoint(args.checkpoint)

    writer = SummaryWriter(log_dir=str(args.log_dir))
    try:
        validation_start = time.perf_counter()
        if method in {"hnn", "phnn"}:
            if bool(ckpt.get("td_correction", False)):
                _run_hnn_td_correction_validation(
                    ckpt=ckpt,
                    cfg=cfg,
                    device=device,
                    writer=writer,
                    epoch=int(args.epoch),
                    rollout_every=int(args.rollout_every),
                    cycle_rollout=bool(int(args.cycle_rollout)),
                    rollout_target_ur=args.rollout_target_ur,
                    rollout_target_ur_tol=float(args.rollout_target_ur_tol),
                    do_losses=bool(int(args.do_losses)),
                    do_rollout=bool(int(args.do_rollout)),
                    num_workers=int(args.num_workers),
                )
            else:
                _run_hnn_validation(
                    ckpt=ckpt,
                    cfg=cfg,
                    device=device,
                    writer=writer,
                    epoch=int(args.epoch),
                    rollout_every=int(args.rollout_every),
                    cycle_rollout=bool(int(args.cycle_rollout)),
                    rollout_target_ur=args.rollout_target_ur,
                    rollout_target_ur_tol=float(args.rollout_target_ur_tol),
                    do_losses=bool(int(args.do_losses)),
                    do_rollout=bool(int(args.do_rollout)),
                    num_workers=int(args.num_workers),
                )
        elif method == "vpinn":
            if bool(ckpt.get("td_correction", False)):
                _run_vpinn_td_correction_validation(
                    ckpt=ckpt,
                    cfg=cfg,
                    device=device,
                    writer=writer,
                    epoch=int(args.epoch),
                    rollout_every=int(args.rollout_every),
                    cycle_rollout=bool(int(args.cycle_rollout)),
                    rollout_target_ur=args.rollout_target_ur,
                    rollout_target_ur_tol=float(args.rollout_target_ur_tol),
                    do_losses=bool(int(args.do_losses)),
                    do_rollout=bool(int(args.do_rollout)),
                    num_workers=int(args.num_workers),
                )
            else:
                _run_vpinn_validation(
                    ckpt=ckpt,
                    cfg=cfg,
                    device=device,
                    writer=writer,
                    epoch=int(args.epoch),
                    rollout_every=int(args.rollout_every),
                    cycle_rollout=bool(int(args.cycle_rollout)),
                    rollout_target_ur=args.rollout_target_ur,
                    rollout_target_ur_tol=float(args.rollout_target_ur_tol),
                    do_losses=bool(int(args.do_losses)),
                    do_rollout=bool(int(args.do_rollout)),
                    num_workers=int(args.num_workers),
                )
        else:
            raise ValueError(f"Unsupported method '{method}'.")
        elapsed = time.perf_counter() - validation_start
        writer.add_scalar("val/validation_wall_time_s", float(elapsed), int(args.epoch))
    finally:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()
