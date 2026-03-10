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
    PHVIV,
    ROLLOUT_DIVERGED_COUNT_KEY,
    ROLLOUT_DIVERGED_KEY,
    lookup_ur_bin_state_scale_tensor,
    build_dataloader_from_series,
    build_rollout_dataloader_from_series,
    compute_validation_metrics,
    load_training_series,
    log_loss_vs_ur,
    log_validation_epoch,
    parse_config,
    preprocess_timeseries,
    resolve_cut_start_seconds,
    resolve_middle_time_plot,
    scaled_residual_loss_per_sample,
    sample_one_index_per_ur,
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
    z0, t_seq, z_traj, ur0, history0, _scale = _parse_rollout_batch(batch)
    z0 = z0.to(device, non_blocking=non_blocking)
    t_seq = t_seq.to(device, non_blocking=non_blocking)
    z_traj = z_traj.to(device, non_blocking=non_blocking)
    ur0 = ur0.to(device, non_blocking=non_blocking)
    if history0 is not None:
        history0 = history0.to(device, non_blocking=non_blocking)
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
        history0_in = None
        if history0 is not None:
            history0_in = history0.unsqueeze(0).expand(samples, *history0.shape).reshape(
                samples * batch_size, *history0.shape[1:]
            )
        z_pred, _ = model.rollout(
            z0_in,
            t_seq_in,
            float(model.dt),
            reduced_velocity=ur0_in,
            history_init=history0_in,
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
            history_init=history0,
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
    history_window = int(getattr(cfg.model, "history_window", 32)) if bool(getattr(cfg.model, "use_history_tcn", False)) else None

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
        history_window=history_window,
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
            history_window=history_window,
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
            force_reg=float(loss_cfg.force_reg),
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
            force_reg=float(loss_cfg.force_reg),
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
        sampled_indices = sample_one_index_per_ur(ur_for_sampling, seed=sample_seed)
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
        sampled_metric_indices = sample_one_index_per_ur(ur_values_all, seed=sample_seed)
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
            z_i, t_i, z_next, t_next, ur_i, history_i, f_i, f_next, _scale = _parse_hnn_batch(batch)
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
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                history_context = model.history_context(
                    z_i,
                    reduced_velocity=ur_i,
                    history_window=history_i,
                )
                if equalize_residual_over_ur_bins or normalize_residual_by_ur_bin_std:
                    per_res = scaled_residual_loss_per_sample(
                        model,
                        z_i,
                        z_next,
                        reduced_velocity=ur_i,
                        history_context=history_context,
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
                            history_context=history_context,
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
                            history_context=history_context,
                        )
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
                sigma_loss = float(force_reg) * sigma_reg_loss
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
                        f_pred = model.u_theta_coeff(z_mid, reduced_velocity=ur_i, history_context=history_context)
                        f_mid = f_mid / f0
                    else:
                        f_pred = model.u_theta(z_mid, reduced_velocity=ur_i, history_context=history_context)
                    per_data = torch.mean((f_pred - f_mid) ** 2, dim=1)
                    data_force_loss = torch.mean(per_data)
                else:
                    data_force_loss = res_loss.new_tensor(0.0)
                    z_mid = 0.5 * (z_i + z_next)
                if float(symmetry_weight) > 0.0:
                    z_flip = -z_mid
                    if getattr(model, "force_output", "force") == "coefficient":
                        f_pos = model.u_theta_coeff(z_mid, reduced_velocity=ur_i, history_context=history_context)
                        f_neg = model.u_theta_coeff(z_flip, reduced_velocity=ur_i, history_context=history_context)
                    else:
                        f_pos = model.u_theta(z_mid, reduced_velocity=ur_i, history_context=history_context)
                        f_neg = model.u_theta(z_flip, reduced_velocity=ur_i, history_context=history_context)
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
    force_reg: float,
    sigma_reg_norm: str,
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
            z_i, t_i, z_next, t_next, ur_i, history_i, f_i, f_next, _scale = _parse_hnn_batch(batch)
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
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                history_context = model.history_context(
                    z_i,
                    reduced_velocity=ur_i,
                    history_window=history_i,
                )
                per_res = (
                    scaled_residual_loss_per_sample(
                        model,
                        z_i,
                        z_next,
                        reduced_velocity=ur_i,
                        history_context=history_context,
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
                        history_context=history_context,
                    )
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
                per_reg = float(force_reg) * per_sigma_reg
                per_reg_mean = float(mean_reg) * per_mean_reg

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
                    per_data = float(force_data_weight) * per_data
                else:
                    per_data = per_res.new_zeros(per_res.shape)
                    z_mid = 0.5 * (z_i + z_next)
                if float(symmetry_weight) > 0.0:
                    z_flip = -z_mid
                    if getattr(model, "force_output", "force") == "coefficient":
                        f_pos = model.u_theta_coeff(z_mid, reduced_velocity=ur_i, history_context=history_context)
                        f_neg = model.u_theta_coeff(z_flip, reduced_velocity=ur_i, history_context=history_context)
                    else:
                        f_pos = model.u_theta(z_mid, reduced_velocity=ur_i, history_context=history_context)
                        f_neg = model.u_theta(z_flip, reduced_velocity=ur_i, history_context=history_context)
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
