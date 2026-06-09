"""
Asynchronous validation runner.

Loads a checkpoint saved during training and runs validation on the unseen val split.
Intended to be spawned as a child process so training can continue.
"""

from __future__ import annotations

import argparse
import traceback
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.training_utils import (
    AGGREGATE_DISPLACEMENT_VALIDATION_ERROR_KEY,
    AGGREGATE_FORCE_VALIDATION_ERROR_KEY,
    AGGREGATE_VALIDATION_ERROR_KEY,
    DISP_STD_REL_ERROR_KEY,
    DOMINANT_FREQ_REL_ERROR_KEY,
    FORCE_DOMINANT_FREQ_REL_ERROR_KEY,
    FORCE_MAPPING_NRMSE_KEY,
    FORCE_STD_REL_ERROR_KEY,
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
    dominant_frequency,
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
    resolve_td_fhat_correction_bounds,
    resolve_td_input_configs,
    resolve_td_phase_input_source,
    resolve_td_memory_config,
    relative_error,
    td_correction_mode_flags,
)
from training.methods.hnn.trainer import (
    _build_td_correction_hnn_loaders,
    _log_td_correction_rollout_validation as _hnn_td_rollout_validation,
    _normalize_rollout_disp_spectral_loss_mode,
    _td_context_with_random_phi_torch,
    _resolve_td_rollout_loss_settings,
    _td_correction_rollout_losses_from_batch,
    _td_correction_state_rollout,
    _td_step_with_corrections,
    _td_state_mse_loss,
    _td_state_propagated_nll_loss,
)
from training.methods.latent_rnn.trainer import (
    LatentRNNForceModel,
    _build_latent_window_dataset,
    _build_surrogate_encoder_reference_groups,
    _latent_losses_from_batch,
    _latent_rollout_validation_case,
    _load_latent_rnn_trajectories,
    _load_surrogate_validation_rows as _load_latent_surrogate_validation_rows,
    _log_latent_rollout_validation,
    _maybe_reduce_surrogate_validation_rows as _maybe_reduce_latent_surrogate_validation_rows,
    _resolve_latent_time_scale,
    _run_latent_surrogate_validation,
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


_SURROGATE_TARGET_KEYS = (
    "disp_std",
    "force_std",
    "disp_dominant_frequency_hz",
    "force_dominant_frequency_hz",
)


def _as_1d_npz_array(data: np.lib.npyio.NpzFile, key: str, *, path: Path) -> np.ndarray:
    if key not in data:
        raise KeyError(f"Surrogate validation file '{path}' is missing required array '{key}'.")
    return np.asarray(data[key]).reshape(-1)


def _finite_scalar_from_npz(data: np.lib.npyio.NpzFile, key: str, idx: int, *, path: Path) -> float:
    arr = _as_1d_npz_array(data, key, path=path)
    if idx >= arr.size:
        raise ValueError(f"Surrogate validation array '{key}' in '{path}' is shorter than row {idx}.")
    value = float(arr[idx])
    if not np.isfinite(value):
        raise ValueError(f"Surrogate validation row {idx} has non-finite '{key}'={value!r}.")
    return value


def _load_surrogate_validation_rows(path: Path, *, td_mass_source: str) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Surrogate validation NPZ does not exist: {path}")
    rows: list[dict[str, Any]] = []
    with np.load(path, allow_pickle=True) as data:
        ur_key = "ur_effective" if "ur_effective" in data else "ur"
        ur = _as_1d_npz_array(data, ur_key, path=path).astype(float)
        n = int(ur.size)
        if n <= 0:
            raise ValueError(f"Surrogate validation file '{path}' contains no rows.")
        if "td_context0" not in data:
            raise KeyError(f"Surrogate validation file '{path}' is missing required array 'td_context0'.")
        td_context0 = np.asarray(data["td_context0"], dtype=float)
        if td_context0.shape != (n, 5):
            raise ValueError(f"Surrogate validation 'td_context0' must have shape ({n}, 5), got {td_context0.shape}.")
        mass_key = "ic_dry_mass_kg" if str(td_mass_source).strip().lower() == "dry" else "ic_effective_mass_kg"
        required_scalar_keys = (
            "ic_y0",
            "ic_dy0",
            "ic_dt",
            "ic_stiffness_n_m",
            "ic_effective_mass_kg",
            "ic_dry_mass_kg",
            "ic_damping_c",
            "rollout_steps",
            "eval_steps_after_discard",
            "rollout_discard_seconds",
            "ic_diameter_m",
            *_SURROGATE_TARGET_KEYS,
        )
        for idx in range(n):
            if not np.isfinite(float(ur[idx])):
                raise ValueError(f"Surrogate validation row {idx} has non-finite '{ur_key}'={ur[idx]!r}.")
            if not np.all(np.isfinite(td_context0[idx])):
                raise ValueError(f"Surrogate validation row {idx} has non-finite td_context0={td_context0[idx]!r}.")
            row = {
                "index": idx,
                "ur": float(ur[idx]),
                "ur_label": (
                    _finite_scalar_from_npz(data, "ur_label", idx, path=path)
                    if "ur_label" in data
                    else float("nan")
                ),
                "td_context0": td_context0[idx].astype(float),
                "mass_key": mass_key,
            }
            for key in required_scalar_keys:
                row[key] = _finite_scalar_from_npz(data, key, idx, path=path)
            row["mass"] = _finite_scalar_from_npz(data, mass_key, idx, path=path)
            rows.append(row)
    return rows


def _maybe_reduce_surrogate_validation_rows(
    rows: list[dict[str, Any]],
    *,
    reduce_time: bool,
    reduction_factor: int,
) -> list[dict[str, Any]]:
    rf = max(1, int(reduction_factor))
    if not bool(reduce_time) or rf <= 1:
        return rows
    reduced_rows: list[dict[str, Any]] = []
    for row in rows:
        old_dt = float(row["ic_dt"])
        if not np.isfinite(old_dt) or old_dt <= 0.0:
            raise ValueError(f"Surrogate row {row['index']} has invalid ic_dt={old_dt!r}.")
        old_eval_steps = int(round(float(row["eval_steps_after_discard"])))
        old_rollout_steps = int(round(float(row["rollout_steps"])))
        if old_eval_steps <= 1 or old_rollout_steps <= old_eval_steps:
            raise ValueError(
                f"Invalid surrogate rollout/eval steps for row {row['index']}: "
                f"rollout_steps={old_rollout_steps}, eval_steps_after_discard={old_eval_steps}"
            )
        discard_seconds = float(row["rollout_discard_seconds"])
        eval_seconds = float(old_eval_steps) * old_dt
        new_dt = old_dt * float(rf)
        new_discard_steps = int(np.ceil(discard_seconds / new_dt))
        new_eval_steps = int(np.ceil(eval_seconds / new_dt))
        new_rollout_steps = new_discard_steps + new_eval_steps
        if new_discard_steps < 1 or new_eval_steps <= 1 or new_rollout_steps <= new_eval_steps:
            raise ValueError(
                f"Time reduction produced invalid surrogate rollout/eval steps for row {row['index']}: "
                f"dt={new_dt}, rollout_steps={new_rollout_steps}, eval_steps_after_discard={new_eval_steps}"
            )
        reduced = dict(row)
        reduced["ic_dt"] = float(new_dt)
        reduced["rollout_steps"] = float(new_rollout_steps)
        reduced["eval_steps_after_discard"] = float(new_eval_steps)
        reduced_rows.append(reduced)
    return reduced_rows


def _surrogate_relative_abs(pred: float, target: float) -> float:
    value = relative_error(float(pred), float(target))
    return abs(float(value)) if np.isfinite(value) else float("nan")


def _surrogate_aggregate_metric(metrics: dict[str, float], keys: tuple[str, ...]) -> float:
    values = [float(metrics[key]) for key in keys if key in metrics and np.isfinite(float(metrics[key]))]
    return float(np.mean(values)) if len(values) == len(keys) else float("nan")


def _safe_scalar_stats(prefix: str, values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {}
    return {
        f"{prefix} mean": float(np.mean(arr)),
        f"{prefix} mean abs": float(np.mean(np.abs(arr))),
        f"{prefix} std": float(np.std(arr)),
    }


def _run_surrogate_td_validation(
    *,
    rows: list[dict[str, Any]],
    writer: SummaryWriter,
    tb_step: int,
    tag: str,
    model: PHVIV,
    device: torch.device,
    td_params: dict[str, float],
    td_memory_cfg: dict[str, Any],
    mean_active: bool,
    predict_sigma: bool,
    fhat_active: bool,
    td_force_input_source: str,
    fhat_bound_multiplier: float,
    force_zero_output: bool,
    rollout_stochastic: bool,
    rollout_noise_scale: float,
    rollout_seed: int | None,
) -> dict[str, Any]:
    metrics_sum: dict[str, float] = {}
    metrics_count: dict[str, int] = {}
    diverged_count = 0

    def _add_metric(name: str, value: float) -> None:
        value_f = float(value)
        if not np.isfinite(value_f):
            return
        metrics_sum[name] = metrics_sum.get(name, 0.0) + value_f
        metrics_count[name] = metrics_count.get(name, 0) + 1

    for row in rows:
        steps = int(round(float(row["rollout_steps"])))
        eval_steps = int(round(float(row["eval_steps_after_discard"])))
        if steps <= 1 or eval_steps <= 1 or eval_steps > steps:
            raise ValueError(
                f"Invalid surrogate rollout/eval steps for row {row['index']}: "
                f"rollout_steps={steps}, eval_steps_after_discard={eval_steps}"
            )
        dt = float(row["ic_dt"])
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"Surrogate row {row['index']} has invalid ic_dt={dt!r}.")
        discard_steps = int(round(float(row["rollout_discard_seconds"]) / dt))
        if discard_steps < 0 or discard_steps >= steps:
            raise ValueError(
                f"Invalid surrogate discard window for row {row['index']}: "
                f"rollout_discard_seconds={row['rollout_discard_seconds']}, dt={dt}, rollout_steps={steps}"
            )
        if steps - discard_steps != eval_steps:
            discard_steps = steps - eval_steps
        mass_value = float(row["mass"])
        stiffness_value = float(row["ic_stiffness_n_m"])
        damping_value = float(row["ic_damping_c"])
        ur_value = float(row["ur"])
        diameter_value = max(float(row.get("ic_diameter_m", float(model.D))), 1.0e-12)
        z0 = torch.tensor([[float(row["ic_y0"]), float(row["ic_dy0"]) * mass_value]], dtype=torch.float32, device=device)
        ur0 = torch.tensor([[ur_value]], dtype=torch.float32, device=device)
        td_context0 = torch.from_numpy(np.asarray(row["td_context0"], dtype=np.float32).reshape(1, 5)).to(device)
        mass_t = torch.full((1, 1), mass_value, dtype=torch.float32, device=device)
        damping_t = torch.full((1, 1), damping_value, dtype=torch.float32, device=device)
        stiffness_t = torch.full((1, 1), stiffness_value, dtype=torch.float32, device=device)
        z_pred, force_seq, corr_seq, sigma_seq, delta_fhat_seq = _td_correction_state_rollout(
            model=model,
            z0=z0,
            ur0=ur0,
            td_context0=td_context0,
            steps=steps,
            dt=dt,
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
            rollout_stochastic=rollout_stochastic,
            rollout_noise_scale=rollout_noise_scale,
            rollout_seed=rollout_seed,
        )
        y_pred = z_pred[0, :, 0].detach().cpu().numpy()
        force_pred = force_seq[0, :, 0].detach().cpu().numpy()
        corr_pred = corr_seq[0, :, 0].detach().cpu().numpy()
        sigma_pred = sigma_seq[0, :, 0].detach().cpu().numpy()
        delta_fhat_pred = delta_fhat_seq[0, :, 0].detach().cpu().numpy()
        if not np.all(np.isfinite(y_pred)) or not np.all(np.isfinite(force_pred)):
            diverged_count += 1
            continue
        y_eval = y_pred[discard_steps + 1 :]
        force_eval = force_pred[discard_steps:]
        corr_eval = corr_pred[discard_steps:]
        sigma_eval = sigma_pred[discard_steps:]
        delta_fhat_eval = delta_fhat_pred[discard_steps:]
        if y_eval.size < 4 or force_eval.size < 4:
            raise ValueError(f"Surrogate row {row['index']} evaluation window is too short after discard.")
        fn = float(np.sqrt(stiffness_value / float(row["ic_effective_mass_kg"])) / (2.0 * np.pi))
        if not np.isfinite(fn) or fn <= 0.0:
            raise ValueError(f"Surrogate row {row['index']} has invalid effective natural frequency.")
        with torch.no_grad():
            f0 = float(
                model._force_scale_from_reduced_velocity(
                    torch.tensor([[ur_value]], dtype=torch.float32, device=device),
                    like=torch.zeros((1, 1), dtype=torch.float32, device=device),
                ).detach().cpu().reshape(-1)[0]
            )
        f0 = max(float(f0), 1.0e-12)
        pred_disp_std = float(np.std(y_eval / diameter_value))
        pred_force_std = float(np.std(force_eval / f0))
        pred_disp_freq_ratio = float(dominant_frequency(y_eval, dt) / fn)
        pred_force_freq_ratio = float(dominant_frequency(force_eval, dt) / fn)
        target_disp_std = float(row["disp_std"])
        target_force_std = float(row["force_std"])
        target_disp_freq_ratio = float(row["disp_dominant_frequency_hz"] / fn)
        target_force_freq_ratio = float(row["force_dominant_frequency_hz"] / fn)
        row_metrics = {
            DOMINANT_FREQ_REL_ERROR_KEY: _surrogate_relative_abs(pred_disp_freq_ratio, target_disp_freq_ratio),
            DISP_STD_REL_ERROR_KEY: _surrogate_relative_abs(pred_disp_std, target_disp_std),
            FORCE_DOMINANT_FREQ_REL_ERROR_KEY: _surrogate_relative_abs(pred_force_freq_ratio, target_force_freq_ratio),
            FORCE_STD_REL_ERROR_KEY: _surrogate_relative_abs(pred_force_std, target_force_std),
            "Predicted std(y/D)": pred_disp_std,
            "Target std(y/D)": target_disp_std,
            "Predicted std(F/F0)": pred_force_std,
            "Target std(F/F0)": target_force_std,
            "Predicted f_y/f_n": pred_disp_freq_ratio,
            "Target f_y/f_n": target_disp_freq_ratio,
            "Predicted f_F/f_n": pred_force_freq_ratio,
            "Target f_F/f_n": target_force_freq_ratio,
        }
        row_metrics[AGGREGATE_DISPLACEMENT_VALIDATION_ERROR_KEY] = _surrogate_aggregate_metric(
            row_metrics,
            (DOMINANT_FREQ_REL_ERROR_KEY, DISP_STD_REL_ERROR_KEY),
        )
        row_metrics[AGGREGATE_FORCE_VALIDATION_ERROR_KEY] = _surrogate_aggregate_metric(
            row_metrics,
            (FORCE_DOMINANT_FREQ_REL_ERROR_KEY, FORCE_STD_REL_ERROR_KEY),
        )
        row_metrics[AGGREGATE_VALIDATION_ERROR_KEY] = _surrogate_aggregate_metric(
            row_metrics,
            (
                DOMINANT_FREQ_REL_ERROR_KEY,
                DISP_STD_REL_ERROR_KEY,
                FORCE_DOMINANT_FREQ_REL_ERROR_KEY,
                FORCE_STD_REL_ERROR_KEY,
            ),
        )
        for name, value in row_metrics.items():
            _add_metric(name, value)
        for name, value in _safe_scalar_stats("Correction", corr_eval / f0).items():
            _add_metric(name, value)
        if predict_sigma:
            for name, value in _safe_scalar_stats("Sigma correction", sigma_eval / f0).items():
                _add_metric(name, value)
        if fhat_active:
            for name, value in _safe_scalar_stats("Delta fhat", delta_fhat_eval).items():
                _add_metric(name, value)
    averaged = {
        name: total / float(max(1, metrics_count.get(name, 0)))
        for name, total in metrics_sum.items()
        if metrics_count.get(name, 0) > 0
    }
    averaged[ROLLOUT_DIVERGED_COUNT_KEY] = float(diverged_count)
    averaged["sample_count"] = float(len(rows))
    for name, value in averaged.items():
        writer.add_scalar(f"{tag}/{name}", float(value), tb_step)
    writer.flush()
    return {
        "loss_total": None,
        "val_metrics": averaged,
        "validation_wall_time_s": None,
    }


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
    shared_trunk_enabled = bool(getattr(model, "shared_td_correction_trunk", False))
    has_shared_trunk_weights = any(str(key).startswith("td_corr_shared_trunk.") for key in state)
    if shared_trunk_enabled and not has_shared_trunk_weights:
        raise ValueError(
            "Checkpoint state does not contain shared TD-correction trunk weights, "
            "but the validation model was built with architecture.shared_td_correction_trunk=true."
        )
    if not shared_trunk_enabled and has_shared_trunk_weights:
        raise ValueError(
            "Checkpoint state contains shared TD-correction trunk weights, "
            "but the validation model was built with architecture.shared_td_correction_trunk=false."
        )
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
    recompute_td_observables_from_phi = bool(hnn_cfg.get("recompute_td_observables_from_phi", False))
    correction_mode = str(ckpt.get("correction_mode", resolve_td_correction_mode(hnn_cfg))).strip().lower()
    mode_flags = td_correction_mode_flags(correction_mode)
    mean_active = bool(mode_flags["mean_active"])
    predict_sigma = bool(mode_flags["sigma_active"])
    fhat_active = bool(mode_flags["fhat_active"])
    arch_dict = asdict(cfg.architecture)
    shared_td_correction_trunk_cfg = bool(
        ckpt.get("shared_td_correction_trunk", arch_dict.get("shared_td_correction_trunk", False))
    )
    input_config_source = dict(hnn_cfg)
    if "input_configs" in ckpt:
        input_config_source["input_configs"] = ckpt["input_configs"]
    input_config_source["correction_mode"] = correction_mode
    td_input_configs = resolve_td_input_configs(
        input_config_source,
        shared_td_correction_trunk=shared_td_correction_trunk_cfg,
    )
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
    random_phase_training = bool(hnn_cfg.get("random_phase_training", False))
    effective_phase_sources = [
        config.get("phase_input_source", "none")
        for config in td_input_configs.values()
        if bool(config.get("use_phi_input", False))
    ]
    if random_phase_training and (
        not effective_phase_sources
        or any(resolve_td_phase_input_source(source) not in {"phi_vy", "both"} for source in effective_phase_sources)
    ):
        raise ValueError(
            "hnn.random_phase_training=true requires hnn.use_phi_input / hnn.phi_input_source "
            "to include phi_vy (use 'phi_vy' or 'both')."
        )
    use_sigma_inputs = bool(hnn_cfg.get("use_sigma_inputs", False))
    input_scaling_mode = resolve_phnn_input_scaling_mode(getattr(cfg.model, "input_scaling_mode", "current"))
    fhat_bound_multiplier = float(ckpt.get("fhat_bound_multiplier", hnn_cfg.get("fhat_bound_multiplier", 1.5)))
    if "fhat_correction_bounds" in ckpt:
        fhat_correction_bounds = resolve_td_fhat_correction_bounds(
            {"fhat_correction_bounds": ckpt.get("fhat_correction_bounds")}
        )
    else:
        fhat_correction_bounds = resolve_td_fhat_correction_bounds(hnn_cfg)
    fhat_reg = float(getattr(loss_cfg, "fhat_reg", 0.0))
    fhat_reg_norm = str(getattr(loss_cfg, "fhat_reg_norm", "l2")).strip().lower()
    state_loss_mode = str(hnn_cfg.get("state_loss_mode", "mse")).strip().lower()
    if state_loss_mode not in {"mse", "propagated_nll"}:
        raise ValueError("hnn.state_loss_mode must be one of: mse, propagated_nll.")
    state_loss_weight = float(getattr(loss_cfg, "state_weight", 1.0))
    if state_loss_weight < 0.0:
        raise ValueError("loss.state_weight must be non-negative.")
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
    surrogate_enabled = bool(getattr(monitoring_cfg, "surrogate_validation_enabled", True))
    surrogate_tag = str(getattr(monitoring_cfg, "surrogate_validation_tag", "val_surrogate")).strip() or "val_surrogate"
    combined_tag = str(getattr(monitoring_cfg, "combined_validation_tag", "val")).strip() or "val"
    surrogate_rows: list[dict[str, Any]] = []
    if surrogate_enabled:
        surrogate_rows = _load_surrogate_validation_rows(
            Path(getattr(monitoring_cfg, "surrogate_validation_npz", "vivana_cfd_data_pipeline/outputs/analysis/surrogate_validation_points.npz")),
            td_mass_source=td_mass_source,
        )
        surrogate_rows = _maybe_reduce_surrogate_validation_rows(
            surrogate_rows,
            reduce_time=bool(getattr(data_cfg, "reduce_time", False)),
            reduction_factor=int(getattr(data_cfg, "reduction_factor", 1)),
        )
        surrogate_step_counts = [int(round(float(row["rollout_steps"]))) for row in surrogate_rows]
        print(
            f"[async-val][phnn] loaded {len(surrogate_rows)} surrogate validation row(s) "
            f"from {getattr(monitoring_cfg, 'surrogate_validation_npz', 'vivana_cfd_data_pipeline/outputs/analysis/surrogate_validation_points.npz')} "
            f"(rollout_steps={surrogate_step_counts})"
        )

    split_dirs: dict[str, Path] = {}
    val_seen_dir = _resolve_optional_val_split_dir(train_series_root, "val_seen")
    if val_seen_dir is None:
        val_seen_dir = _resolve_optional_val_split_dir(train_series_root, "val")
    if val_seen_dir is not None:
        split_dirs["val_seen"] = val_seen_dir
    cut_start_seconds = resolve_cut_start_seconds(data_cfg, "val")
    split_trajs_map: dict[str, list[dict[str, Any]]] = {}
    for split_tag, split_dir in split_dirs.items():
        split_paths = sorted(split_dir.glob("*.npz"))
        if not split_paths:
            continue
        split_trajs_map[split_tag] = load_td_correction_trajectories(
            paths=split_paths,
            cut_start_seconds=cut_start_seconds,
            reduce_time=bool(getattr(data_cfg, "reduce_time", False)),
            reduction_factor=int(getattr(data_cfg, "reduction_factor", 1)),
            ur_source=td_mass_source,
            td_params=td_params,
            td_memory_cfg=td_memory_cfg,
            recompute_td_observables_from_phi=recompute_td_observables_from_phi,
        )

    first_split_trajs = next((trajs for trajs in split_trajs_map.values() if trajs), [])
    if not first_split_trajs and not surrogate_rows:
        raise FileNotFoundError("Async PHNN validation requires val_seen trajectories or surrogate validation rows.")
    if first_split_trajs:
        dt = float(first_split_trajs[0]["t"][1] - first_split_trajs[0]["t"][0])
    else:
        dt = float(surrogate_rows[0]["ic_dt"])

    model_dict = asdict(cfg.model)
    mass_key = "dry_mass_kg" if td_mass_source == "dry" else "effective_mass_kg"
    if first_split_trajs:
        first_val_traj = first_split_trajs[0]
        model_dict["structural_mass"] = float(np.asarray(first_val_traj[mass_key]).reshape(()))
        model_dict["k"] = float(np.asarray(first_val_traj["stiffness_n_m"]).reshape(()))
        model_dict["damping_c"] = float(np.asarray(first_val_traj["damping_c"]).reshape(()))
    else:
        first_surrogate = surrogate_rows[0]
        model_dict["structural_mass"] = float(first_surrogate["mass"])
        model_dict["k"] = float(first_surrogate["ic_stiffness_n_m"])
        model_dict["damping_c"] = float(first_surrogate["ic_damping_c"])
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
    setattr(model, "random_phase_training", random_phase_training)
    _load_state(model, ckpt["model_state"])
    model.eval()

    def _reg(value: torch.Tensor, norm: str) -> torch.Tensor:
        return torch.mean(torch.abs(value)) if str(norm).strip().lower() == "l1" else torch.mean(value * value)

    def _run_split(split_tag: str, split_trajs_np: list[dict[str, Any]]) -> dict[str, Any]:
        split_start = time.perf_counter()
        _train_loader, val_loader, _train_rollout_loader, rollout_loader = _build_td_correction_hnn_loaders(
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
        del _train_loader
        del _train_rollout_loader

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
                    if random_phase_training:
                        td_context_i = _td_context_with_random_phi_torch(td_context_i)
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
                    raw_corr_mu = step["raw_corr_mu"]
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
                    mean_reg_loss = _reg(raw_corr_mu, mean_reg_norm)
                    sigma_reg_loss = _reg(sigma_corr, sigma_reg_norm) if predict_sigma else state_loss.new_tensor(0.0)
                    fhat_reg_loss = _reg(step["delta_fhat"], fhat_reg_norm) if fhat_active else state_loss.new_tensor(0.0)
                    total_loss = (
                        float(state_loss_weight) * state_loss
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
                float(state_loss_weight) * val_metrics["loss_state"]
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
                value_f = total / float(max(1, metrics_count.get(name, 0)))
                writer.add_scalar(f"{split_tag}/{name}", value_f, tb_step)
                val_metrics[name] = float(value_f)
                num_rollout_scalars_written += 1
            writer.add_scalar(f"{split_tag}/{ROLLOUT_DIVERGED_COUNT_KEY}", float(diverged_count), tb_step)
            val_metrics[ROLLOUT_DIVERGED_COUNT_KEY] = float(diverged_count)
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
    if surrogate_enabled and surrogate_rows and do_rollout:
        print(f"[async-val][phnn] running surrogate validation split '{surrogate_tag}'")
        surrogate_start = time.perf_counter()
        surrogate_result = _run_surrogate_td_validation(
            rows=surrogate_rows,
            writer=writer,
            tb_step=tb_step,
            tag=surrogate_tag,
            model=model,
            device=device,
            td_params=td_params,
            td_memory_cfg=td_memory_cfg,
            mean_active=mean_active,
            predict_sigma=predict_sigma,
            fhat_active=fhat_active,
            td_force_input_source=td_force_input_source,
            fhat_bound_multiplier=fhat_bound_multiplier,
            force_zero_output=force_zero_output,
            rollout_stochastic=rollout_stochastic,
            rollout_noise_scale=rollout_noise_scale,
            rollout_seed=rollout_seed,
        )
        surrogate_result["validation_wall_time_s"] = float(time.perf_counter() - surrogate_start)
        writer.add_scalar(f"{surrogate_tag}/validation_wall_time_s", surrogate_result["validation_wall_time_s"], tb_step)
        split_results[surrogate_tag] = surrogate_result

    combined_metrics: dict[str, float] = {}
    seen_metrics = split_results.get("val_seen", {}).get("val_metrics", {})
    surrogate_metrics = split_results.get(surrogate_tag, {}).get("val_metrics", {})
    if seen_metrics and surrogate_metrics:
        shared_names = sorted(set(seen_metrics).intersection(surrogate_metrics))
        for name in shared_names:
            if str(name).startswith("loss_") or str(name) == "loss_total":
                continue
            seen_value = float(seen_metrics[name])
            surrogate_value = float(surrogate_metrics[name])
            if np.isfinite(seen_value) and np.isfinite(surrogate_value):
                combined_value = 0.5 * seen_value + 0.5 * surrogate_value
                combined_metrics[name] = float(combined_value)
                writer.add_scalar(f"{combined_tag}/{name}", combined_value, tb_step)
        if combined_metrics:
            split_results[combined_tag] = {
                "loss_total": None,
                "val_metrics": combined_metrics,
                "validation_wall_time_s": None,
            }
            writer.flush()

    summary_metrics = combined_metrics or surrogate_metrics or seen_metrics
    summary: dict[str, Any] = {
        "loss_total": None,
        "val_metrics": summary_metrics,
        "split_results": split_results,
    }
    summary["best_metric_name"] = AGGREGATE_VALIDATION_ERROR_KEY
    summary["best_metric_value"] = summary["val_metrics"].get(AGGREGATE_VALIDATION_ERROR_KEY)
    if "val_seen" in split_results:
        summary["val_seen_loss_total"] = split_results["val_seen"].get("loss_total")
    if surrogate_tag in split_results:
        summary["val_surrogate_aggregate"] = split_results[surrogate_tag].get("val_metrics", {}).get(
            AGGREGATE_VALIDATION_ERROR_KEY
        )
    return summary

def _run_latent_rnn_validation(
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
    tb_step = int(epoch)
    data_cfg = cfg.data
    model_cfg = cfg.model
    arch_cfg = cfg.architecture
    loss_cfg = cfg.loss
    monitoring_cfg = cfg.monitoring
    lrnn_cfg = dict(cfg.latent_rnn or {})
    non_blocking = device.type == "cuda"

    train_series_root = Path(data_cfg.train_series_dir)
    train_dir = train_series_root / "train"
    if not train_dir.exists():
        raise FileNotFoundError("latent_rnn async validation expects train/ under data.train_series_dir.")
    train_paths = sorted(train_dir.glob("*.npz"))
    if not train_paths:
        raise FileNotFoundError(f"No '.npz' files found in '{train_dir}'.")

    mass_source = str(lrnn_cfg.get("td_mass_source", "dry")).strip().lower()
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
    train_trajs = _load_latent_rnn_trajectories(
        train_paths,
        cut_start_seconds=resolve_cut_start_seconds(data_cfg, "train"),
        reduce_time=reduce_time_enabled,
        reduction_factor=reduction_factor,
        stagger_reduced_time=stagger_train_reduce,
        mass_source=mass_source,
    )
    if not train_trajs:
        raise ValueError("No latent_rnn training trajectories remained after async loading/reduction.")

    split_paths_map: dict[str, list[Path]] = {}
    val_unseen_dir = train_series_root / ASYNC_VAL_SPLIT_TAG
    legacy_val_dir = train_series_root / "val"
    if val_unseen_dir.exists():
        split_paths_map[ASYNC_VAL_SPLIT_TAG] = sorted(val_unseen_dir.glob("*.npz"))
    elif legacy_val_dir.exists():
        split_paths_map[ASYNC_VAL_SPLIT_TAG] = sorted(legacy_val_dir.glob("*.npz"))
    val_seen_dir = train_series_root / "val_seen"
    if val_seen_dir.exists():
        split_paths_map["val_seen"] = sorted(val_seen_dir.glob("*.npz"))

    split_trajs_map: dict[str, list[dict[str, Any]]] = {}
    for split_tag, paths in split_paths_map.items():
        if not paths:
            continue
        split_trajs_map[split_tag] = _load_latent_rnn_trajectories(
            paths,
            cut_start_seconds=resolve_cut_start_seconds(data_cfg, "val"),
            reduce_time=reduce_time_enabled,
            reduction_factor=reduction_factor,
            stagger_reduced_time=stagger_val_reduce,
            mass_source=mass_source,
        )

    latent_dim = int(lrnn_cfg.get("latent_dim", 3))
    encoder_length = int(lrnn_cfg.get("encoder_length", 50))
    include_acceleration = bool(lrnn_cfg.get("encoder_include_acceleration", True))
    input_scaling_mode = resolve_phnn_input_scaling_mode(getattr(model_cfg, "input_scaling_mode", "current"))
    ur_scale = 10.0 if getattr(model_cfg, "ur_scale", None) is None else float(model_cfg.ur_scale)
    latent_time_scale = float(
        ckpt.get(
            "latent_time_scale",
            _resolve_latent_time_scale(lrnn_cfg.get("latent_time_scale", "auto"), train_trajs=train_trajs),
        )
    )
    rollout_settings = _resolve_td_rollout_loss_settings(loss_cfg)
    rollout_det_weight = float(getattr(loss_cfg, "rollout_det_weight", 0.0))
    rollout_disp_std_weight = float(getattr(loss_cfg, "rollout_disp_std_weight", 0.0))
    rollout_disp_mean_in_std_loss = bool(getattr(loss_cfg, "rollout_disp_mean_in_std_loss", True))
    spectral_weight_raw = getattr(loss_cfg, "rollout_disp_spectral_weight", None)
    rollout_disp_spectral_weight = (
        float(getattr(loss_cfg, "rollout_disp_psd_weight", 0.0))
        if spectral_weight_raw is None
        else float(spectral_weight_raw)
    )
    rollout_active = (
        rollout_det_weight > 0.0
        or rollout_disp_std_weight > 0.0
        or rollout_disp_spectral_weight > 0.0
    )
    rollout_steps = int(getattr(loss_cfg, "rollout_det_steps", 1)) if rollout_active else 1
    rollout_batch_size_raw = int(getattr(loss_cfg, "rollout_det_batch_size", 0))
    rollout_batch_size = int(cfg.training.batch_size) if rollout_batch_size_raw <= 0 else rollout_batch_size_raw
    state_weight = float(getattr(loss_cfg, "state_weight", 1.0))
    mean_reg = float(getattr(loss_cfg, "mean_reg", 0.0))
    mean_reg_norm = str(getattr(loss_cfg, "mean_reg_norm", "l1")).strip().lower()
    force_data_weight = float(getattr(loss_cfg, "force_data_weight", 1.0))
    use_force_data_loss = bool(getattr(loss_cfg, "use_force_data_loss", False))
    spectral_mode = _normalize_rollout_disp_spectral_loss_mode(
        getattr(loss_cfg, "rollout_disp_spectral_loss", "psd")
    )
    psd_peak_rel_bw = float(getattr(loss_cfg, "rollout_disp_psd_peak_rel_bandwidth", 0.0))
    psd_use_hann = bool(getattr(loss_cfg, "rollout_disp_psd_use_hann_window", True))

    first_traj = train_trajs[0]
    model = LatentRNNForceModel(
        latent_dim=latent_dim,
        encoder_input_dim=3 + (1 if include_acceleration else 0),
        encoder_hidden=int(lrnn_cfg.get("encoder_hidden", 128)),
        encoder_layers=int(lrnn_cfg.get("encoder_layers", 1)),
        encoder_dropout=float(lrnn_cfg.get("encoder_dropout", 0.0)),
        backbone_input_dim=3 + latent_dim,
        architecture_cfg=arch_cfg,
        rho=float(np.asarray(first_traj["rho_kg_m3"]).reshape(())),
        diameter=float(model_cfg.D),
        force_output=str(model_cfg.force_output),
        coefficient_output_bound=(
            None
            if getattr(model_cfg, "coefficient_output_bound", None) is None
            else float(model_cfg.coefficient_output_bound)
        ),
        input_scaling_mode=input_scaling_mode,
        ur_scale=ur_scale,
        latent_time_scale=latent_time_scale,
        corr_init_mode=str(lrnn_cfg.get("corr_init_mode", getattr(model_cfg, "corr_init_mode", "zero"))),
    ).to(device)
    _load_state(model, ckpt["model_state"])
    model.eval()

    surrogate_enabled = bool(getattr(monitoring_cfg, "surrogate_validation_enabled", True))
    surrogate_tag = str(getattr(monitoring_cfg, "surrogate_validation_tag", "val_surrogate")).strip() or "val_surrogate"
    combined_tag = str(getattr(monitoring_cfg, "combined_validation_tag", "val")).strip() or "val"
    surrogate_rows: list[dict[str, Any]] = []
    if surrogate_enabled:
        surrogate_rows = _load_latent_surrogate_validation_rows(
            Path(getattr(monitoring_cfg, "surrogate_validation_npz", "vivana_cfd_data_pipeline/outputs/analysis/surrogate_validation_points.npz")),
            td_mass_source=mass_source,
        )
        surrogate_rows = _maybe_reduce_latent_surrogate_validation_rows(
            surrogate_rows,
            reduce_time=reduce_time_enabled,
            reduction_factor=reduction_factor,
        )
        surrogate_step_counts = [int(round(float(row["rollout_steps"]))) for row in surrogate_rows]
        print(
            f"[async-val][latent_rnn] loaded {len(surrogate_rows)} surrogate validation row(s) "
            f"from {getattr(monitoring_cfg, 'surrogate_validation_npz', 'vivana_cfd_data_pipeline/outputs/analysis/surrogate_validation_points.npz')} "
            f"(rollout_steps={surrogate_step_counts})"
        )
    if not any(split_trajs_map.values()) and not surrogate_rows:
        raise FileNotFoundError("Async latent_rnn validation requires validation trajectories or surrogate rows.")
    surrogate_encoder_references = _build_surrogate_encoder_reference_groups(
        train_trajs,
        encoder_length=encoder_length,
        mass_source=mass_source,
        input_scaling_mode=input_scaling_mode,
        diameter=float(model_cfg.D),
        ur_scale=ur_scale,
        include_acceleration=include_acceleration,
    )
    if surrogate_enabled and surrogate_rows and not surrogate_encoder_references:
        raise ValueError("Surrogate validation is enabled, but no latent encoder reference histories were available.")

    def _run_loss_loader(loader: Any) -> dict[str, float]:
        sums: dict[str, torch.Tensor] = {}
        count = 0
        with torch.no_grad():
            for batch in loader:
                losses = _latent_losses_from_batch(
                    model=model,
                    batch=batch,
                    device=device,
                    non_blocking=non_blocking,
                    trajectory_relative=bool(rollout_settings["trajectory_relative"]),
                    compute_disp_std_loss=rollout_disp_std_weight > 0.0,
                    compute_disp_mean_loss=(rollout_disp_std_weight > 0.0 and rollout_disp_mean_in_std_loss),
                    disp_std_relative=bool(rollout_settings["disp_std_relative"]),
                    disp_std_power=float(rollout_settings["disp_std_p"]),
                    compute_disp_spectral_loss=rollout_disp_spectral_weight > 0.0,
                    disp_spectral_loss_mode=spectral_mode,
                    disp_freq_relative=bool(rollout_settings["disp_freq_relative"]),
                    disp_freq_power=float(rollout_settings["disp_freq_p"]),
                    disp_freq_alpha=float(rollout_settings["disp_freq_alpha"]),
                    disp_psd_relative=bool(rollout_settings["disp_psd_relative"]),
                    disp_psd_peak_rel_bandwidth=psd_peak_rel_bw,
                    disp_psd_use_hann_window=psd_use_hann,
                    mean_reg_norm=mean_reg_norm,
                )
                rollout_total_loss = (
                    rollout_det_weight * losses["trajectory_loss"]
                    + rollout_disp_std_weight * (losses["disp_std_loss"] + losses["disp_mean_loss"])
                    + rollout_disp_spectral_weight * losses["disp_spectral_loss"]
                )
                total_loss = state_weight * losses["state_loss"] + rollout_total_loss + mean_reg * losses["mean_reg_loss"]
                if use_force_data_loss:
                    total_loss = total_loss + force_data_weight * losses["force_data_loss"]
                batch_metrics = {
                    "loss_total": total_loss,
                    "loss_state": losses["state_loss"],
                    "loss_rollout_det": losses["trajectory_loss"],
                    "loss_rollout_disp_std": losses["disp_std_loss"],
                    "loss_rollout_disp_mean": losses["disp_mean_loss"],
                    "loss_rollout_spectral": losses["disp_spectral_loss"],
                    "loss_reg_mean": losses["mean_reg_loss"],
                }
                batch_size = int(batch[0].shape[0])
                count += batch_size
                for name, value in batch_metrics.items():
                    sums[name] = sums.get(name, value.detach().new_zeros(())) + value.detach() * batch_size
        denom = float(max(1, count))
        return {name: float((value / denom).detach().cpu()) for name, value in sums.items()}

    def _run_split(split_tag: str, split_trajs: list[dict[str, Any]]) -> dict[str, Any]:
        split_start = time.perf_counter()
        val_metrics: dict[str, float] = {}
        dataset = _build_latent_window_dataset(
            split_trajs,
            encoder_length=encoder_length,
            rollout_steps=rollout_steps,
            mass_source=mass_source,
            input_scaling_mode=input_scaling_mode,
            diameter=float(model_cfg.D),
            ur_scale=ur_scale,
            include_acceleration=include_acceleration,
        )
        if do_losses and dataset is not None:
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=rollout_batch_size,
                shuffle=False,
                num_workers=int(num_workers),
                pin_memory=(device.type == "cuda"),
            )
            val_metrics.update(_run_loss_loader(loader))
            for name, value in val_metrics.items():
                writer.add_scalar(f"{split_tag}/{name}", value, tb_step)

        if do_rollout and split_trajs:
            validation_samples_per_ur = max(1, int(getattr(monitoring_cfg, "validation_samples_per_ur", 1)))
            ur_values = [
                float(np.asarray(traj["ur"], dtype=float).reshape(-1)[0])
                for traj in split_trajs
                if np.asarray(traj["ur"], dtype=float).reshape(-1).size > 0
            ]
            sampled_indices = sample_indices_per_ur(
                ur_values,
                samples_per_ur=validation_samples_per_ur,
                seed=1,
            )
            sampled_names = [str(split_trajs[idx].get("name", f"traj_{idx}")) for idx in sampled_indices if idx < len(split_trajs)]
            print(
                f"[async-val][latent_rnn][{split_tag}] epoch {epoch}: sampled metric trajectories={sampled_names} "
                f"(mass_source={mass_source})"
            )
            metrics_sum: dict[str, float] = {}
            metrics_count: dict[str, int] = {}
            diverged_count = 0
            for idx in sampled_indices:
                if idx >= len(split_trajs):
                    continue
                rollout_metrics = _latent_rollout_validation_case(
                    model=model,
                    traj=split_trajs[idx],
                    encoder_length=encoder_length,
                    include_acceleration=include_acceleration,
                    mass_source=mass_source,
                    input_scaling_mode=input_scaling_mode,
                    ur_scale=ur_scale,
                    device=device,
                ).get("metrics", {})
                diverged_flag = float(rollout_metrics.get(ROLLOUT_DIVERGED_KEY, 0.0))
                if np.isfinite(diverged_flag) and diverged_flag > 0.5:
                    diverged_count += 1
                for name, value in rollout_metrics.items():
                    if name == ROLLOUT_DIVERGED_KEY or not np.isfinite(float(value)):
                        continue
                    metrics_sum[name] = metrics_sum.get(name, 0.0) + float(value)
                    metrics_count[name] = metrics_count.get(name, 0) + 1
            for name, total in metrics_sum.items():
                value_f = total / float(max(1, metrics_count.get(name, 0)))
                writer.add_scalar(f"{split_tag}/{name}", value_f, tb_step)
                val_metrics[name] = float(value_f)
            writer.add_scalar(f"{split_tag}/{ROLLOUT_DIVERGED_COUNT_KEY}", float(diverged_count), tb_step)
            val_metrics[ROLLOUT_DIVERGED_COUNT_KEY] = float(diverged_count)

            plot_indices = sample_one_index_per_ur(ur_values, seed=0) or [0]
            plot_idx = int(plot_indices[0])
            if 0 <= plot_idx < len(split_trajs):
                _log_latent_rollout_validation(
                    writer=writer,
                    epoch=tb_step,
                    model=model,
                    traj=split_trajs[plot_idx],
                    encoder_length=encoder_length,
                    include_acceleration=include_acceleration,
                    mass_source=mass_source,
                    input_scaling_mode=input_scaling_mode,
                    ur_scale=ur_scale,
                    device=device,
                    tag_prefix=f"{split_tag}/rollout",
                    metric_prefix=split_tag,
                    step=tb_step,
                    log_metrics=False,
                    log_plots=True,
                    log_force_plot=(split_tag != "val_seen"),
                    log_spectra=False,
                )

        split_elapsed = time.perf_counter() - split_start
        writer.add_scalar(f"{split_tag}/validation_wall_time_s", float(split_elapsed), tb_step)
        return {
            "loss_total": (float(val_metrics["loss_total"]) if "loss_total" in val_metrics else None),
            "val_metrics": val_metrics,
            "validation_wall_time_s": float(split_elapsed),
        }

    split_results = {
        split_tag: _run_split(split_tag, split_trajs)
        for split_tag, split_trajs in split_trajs_map.items()
        if split_trajs
    }

    if surrogate_enabled and surrogate_rows and do_rollout:
        print(f"[async-val][latent_rnn] running surrogate validation split '{surrogate_tag}'")
        surrogate_start = time.perf_counter()
        surrogate_result = _run_latent_surrogate_validation(
            rows=surrogate_rows,
            writer=writer,
            tb_step=tb_step,
            tag=surrogate_tag,
            model=model,
            encoder_references=surrogate_encoder_references,
            device=device,
        )
        surrogate_result["validation_wall_time_s"] = float(time.perf_counter() - surrogate_start)
        writer.add_scalar(f"{surrogate_tag}/validation_wall_time_s", surrogate_result["validation_wall_time_s"], tb_step)
        split_results[surrogate_tag] = surrogate_result

    combined_metrics: dict[str, float] = {}
    seen_metrics = split_results.get("val_seen", {}).get("val_metrics", {})
    surrogate_metrics = split_results.get(surrogate_tag, {}).get("val_metrics", {})
    if seen_metrics and surrogate_metrics:
        shared_names = sorted(set(seen_metrics).intersection(surrogate_metrics))
        for name in shared_names:
            if str(name).startswith("loss_") or str(name) == "loss_total":
                continue
            seen_value = float(seen_metrics[name])
            surrogate_value = float(surrogate_metrics[name])
            if np.isfinite(seen_value) and np.isfinite(surrogate_value):
                combined_value = 0.5 * seen_value + 0.5 * surrogate_value
                combined_metrics[name] = float(combined_value)
                writer.add_scalar(f"{combined_tag}/{name}", combined_value, tb_step)
        if combined_metrics:
            split_results[combined_tag] = {
                "loss_total": None,
                "val_metrics": combined_metrics,
                "validation_wall_time_s": None,
            }
            writer.flush()

    unseen_metrics = split_results.get(ASYNC_VAL_SPLIT_TAG, {}).get("val_metrics", {})
    summary_metrics = combined_metrics or surrogate_metrics or seen_metrics or unseen_metrics
    loss_total = split_results.get(ASYNC_VAL_SPLIT_TAG, {}).get("loss_total")
    if loss_total is None:
        loss_total = split_results.get("val_seen", {}).get("loss_total")
    return {
        "loss_total": loss_total,
        "val_metrics": summary_metrics,
        "split_results": split_results,
        "best_metric_name": AGGREGATE_VALIDATION_ERROR_KEY,
        "best_metric_value": summary_metrics.get(AGGREGATE_VALIDATION_ERROR_KEY),
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
        elif method in {"latent_rnn", "scratch_latent_rnn"}:
            summary.update(_run_latent_rnn_validation(
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
        writer.flush()
        summary_path = _async_summary_path(args.log_dir, int(args.epoch))
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["traceback"] = traceback.format_exc()
        summary_path = _async_summary_path(args.log_dir, int(args.epoch))
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        raise
    finally:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()
