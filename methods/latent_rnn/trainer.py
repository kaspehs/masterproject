from __future__ import annotations

import json
import math
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
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
    AGGREGATE_DISPLACEMENT_VALIDATION_ERROR_KEY,
    AGGREGATE_FORCE_VALIDATION_ERROR_KEY,
    AGGREGATE_VALIDATION_ERROR_KEY,
    DISP_STD_REL_ERROR_KEY,
    DOMINANT_FREQ_REL_ERROR_KEY,
    FORCE_MAPPING_NRMSE_KEY,
    FORCE_DOMINANT_FREQ_REL_ERROR_KEY,
    FORCE_STD_REL_ERROR_KEY,
    Config,
    GradNormBalancer,
    ROLLOUT_DIVERGED_COUNT_KEY,
    ROLLOUT_DIVERGED_KEY,
    ODEPirateNet,
    Residual,
    _activation_factory,
    _default_mlp_kwargs,
    _default_residual_kwargs,
    compute_validation_metrics,
    create_zoom_mask,
    dominant_frequency,
    log_area_normalized_rollout_spectra,
    log_displacement_plots,
    log_final_rollout_errors_vs_ur,
    log_force_plots,
    relative_error,
    resolve_cut_start_seconds,
    resolve_phnn_input_scaling_mode,
    sample_indices_per_ur,
    sample_one_index_per_ur,
    structural_step_constant_force_torch,
)
from methods.hnn.trainer import (
    _displacement_mean_error_torch,
    _displacement_std_error_torch,
    _dominant_frequency_error_torch,
    _normalize_rollout_disp_spectral_loss_mode,
    _psd_error_torch,
    _resolve_td_rollout_loss_settings,
    _trajectory_rollout_mse_torch,
)


def _first_present(data: np.lib.npyio.NpzFile, keys: Sequence[str], *, path: Path) -> Any:
    for key in keys:
        if key in data.files:
            return data[key]
    raise KeyError(f"{path} is missing required keys. Tried: {', '.join(keys)}")


def _optional_present(data: np.lib.npyio.NpzFile, keys: Sequence[str]) -> Any | None:
    for key in keys:
        if key in data.files:
            return data[key]
    return None


def _scalar_from_npz(data: np.lib.npyio.NpzFile, keys: Sequence[str], *, path: Path) -> float:
    raw = _first_present(data, keys, path=path)
    value = float(np.asarray(raw, dtype=float).reshape(-1)[0])
    if not np.isfinite(value):
        raise ValueError(f"{path} has non-finite scalar for keys {keys!r}.")
    return value


def _series_from_raw(raw: Any, n: int, *, name: str, path: Path) -> np.ndarray:
    arr = np.asarray(raw, dtype=float)
    if arr.ndim == 0:
        return np.full((n,), float(arr), dtype=float)
    out = arr.reshape(-1)
    if out.shape[0] == 1:
        return np.full((n,), float(out[0]), dtype=float)
    if out.shape[0] != n:
        raise ValueError(f"{path} {name} must be scalar or length {n}, got {out.shape[0]}.")
    return out.astype(float, copy=False)


def _force_per_m_from_npz(data: np.lib.npyio.NpzFile, *, path: Path, n: int) -> np.ndarray:
    force_per_m_raw = _optional_present(data, ("y_force_per_m_dim", "force_per_m_dim"))
    if force_per_m_raw is not None:
        return _series_from_raw(force_per_m_raw, n, name="force_per_m", path=path)
    force_total_raw = _optional_present(data, ("y_force_dim", "c", "F_total", "force_total", "force"))
    if force_total_raw is None:
        raise KeyError(f"{path} is missing CFD force channels.")
    span_raw = _optional_present(data, ("physical_span_m", "raw_force_span_scale_applied", "span_m"))
    span = 1.0 if span_raw is None else float(np.asarray(span_raw, dtype=float).reshape(-1)[0])
    if not np.isfinite(span) or span <= 0.0:
        raise ValueError(f"{path} has invalid force span scale {span!r}.")
    return _series_from_raw(force_total_raw, n, name="force_total", path=path) / span


def _prepare_ur_series(
    *,
    data: np.lib.npyio.NpzFile,
    n: int,
    path: Path,
    mass_source: str,
    flow_speed: np.ndarray,
    dry_mass_kg: float,
    effective_mass_kg: float,
    stiffness_n_m: float,
    diameter_m: float,
) -> np.ndarray:
    raw = _optional_present(data, ("U_r_computed_series", "U_r", "computed_ur", "label_ur"))
    if raw is not None and str(mass_source).strip().lower() == "stored":
        return _series_from_raw(raw, n, name="U_r", path=path)
    if str(mass_source).strip().lower() == "label":
        label_raw = _optional_present(data, ("U_r_label_series", "U_r_label_scalar", "label_ur"))
        if label_raw is None:
            raise ValueError(f"{path} does not contain U_r label values.")
        return _series_from_raw(label_raw, n, name="U_r_label", path=path)
    if str(mass_source).strip().lower() not in {"dry", "effective"}:
        raise ValueError("latent_rnn.td_mass_source must be one of: dry, effective, stored, label.")
    mass = dry_mass_kg if str(mass_source).strip().lower() == "dry" else effective_mass_kg
    omega_n = math.sqrt(float(stiffness_n_m) / float(mass))
    f_n = omega_n / (2.0 * math.pi)
    return flow_speed / max(np.finfo(float).eps, f_n * float(diameter_m))


def _maybe_reduce_arrays(
    arrays: dict[str, np.ndarray],
    *,
    enabled: bool,
    factor: int,
    offset: int,
) -> dict[str, np.ndarray]:
    if not bool(enabled):
        return arrays
    rf = max(1, int(factor))
    return {key: np.asarray(value)[int(offset) :: rf] for key, value in arrays.items()}


def _load_latent_rnn_trajectories(
    paths: Sequence[Path],
    *,
    cut_start_seconds: float,
    reduce_time: bool,
    reduction_factor: int,
    stagger_reduced_time: bool,
    mass_source: str,
) -> list[dict[str, np.ndarray]]:
    trajectories: list[dict[str, np.ndarray]] = []
    offsets = list(range(max(1, int(reduction_factor)))) if bool(reduce_time and stagger_reduced_time) else [0]
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            t = np.asarray(_first_present(data, ("time_dim", "time", "t"), path=path), dtype=float).reshape(-1)
            y = np.asarray(_first_present(data, ("y_disp_dim", "y", "y_dim"), path=path), dtype=float).reshape(-1)
            dy = np.asarray(_first_present(data, ("y_vel_dim", "dy", "dy_dim"), path=path), dtype=float).reshape(-1)
            ddy = np.asarray(_first_present(data, ("y_acc_dim", "ddy", "ddy_dim"), path=path), dtype=float).reshape(-1)
            n = int(t.shape[0])
            if n < 3:
                raise ValueError(f"{path} is too short for latent RNN training.")
            force_per_m = _force_per_m_from_npz(data, path=path, n=n)
            stiffness = _scalar_from_npz(data, ("python_stiffness_n_m", "stiffness_n_m"), path=path)
            dry_mass = _scalar_from_npz(data, ("python_dry_mass_kg", "dry_mass_kg", "python_mass_kg"), path=path)
            effective_mass = _scalar_from_npz(data, ("python_effective_mass_kg", "effective_mass_kg"), path=path)
            damping_c = _scalar_from_npz(data, ("python_damping_c", "damping_c"), path=path)
            rho = _scalar_from_npz(data, ("python_rho_kg_m3", "rho_kg_m3"), path=path)
            diameter = _scalar_from_npz(data, ("python_diameter_m", "diameter_m"), path=path)
            flow_raw = _first_present(data, ("python_flow_speed_m_s", "flow_speed_m_s"), path=path)
            flow_speed = _series_from_raw(flow_raw, n, name="flow_speed_m_s", path=path)
            ur = _prepare_ur_series(
                data=data,
                n=n,
                path=path,
                mass_source=mass_source,
                flow_speed=flow_speed,
                dry_mass_kg=dry_mass,
                effective_mass_kg=effective_mass,
                stiffness_n_m=stiffness,
                diameter_m=diameter,
            )
        arrays = {
            "t": t,
            "y": y,
            "dy": dy,
            "ddy": ddy,
            "force_per_m": force_per_m,
            "flow_speed": flow_speed,
            "ur": ur,
        }
        for offset in offsets:
            reduced = _maybe_reduce_arrays(
                arrays,
                enabled=reduce_time,
                factor=reduction_factor,
                offset=offset,
            )
            if any(np.asarray(value).shape[0] < 3 for value in reduced.values()):
                continue
            t_reduced = np.asarray(reduced["t"], dtype=float)
            if float(cut_start_seconds) > 0.0:
                mask = t_reduced >= (float(t_reduced[0]) + float(cut_start_seconds))
                if int(np.count_nonzero(mask)) < 3:
                    continue
                reduced = {key: np.asarray(value)[mask] for key, value in reduced.items()}
                t_reduced = np.asarray(reduced["t"], dtype=float)
            if not np.all(np.isfinite(t_reduced)) or np.any(np.diff(t_reduced) <= 0.0):
                raise ValueError(f"{path} time vector must be strictly increasing after reduction.")
            dt_values = np.diff(t_reduced)
            if not np.allclose(dt_values, float(np.median(dt_values)), rtol=1e-5, atol=1e-9):
                raise ValueError(f"{path} time vector is not uniform after reduction.")
            suffix = ""
            if bool(reduce_time and stagger_reduced_time) and len(offsets) > 1:
                suffix = f"__rf{int(reduction_factor)}_offset{int(offset)}"
            trajectories.append(
                {
                    "name": f"{path.stem}{suffix}{path.suffix}",
                    "source_name": path.name,
                    "t": np.asarray(reduced["t"], dtype=np.float32),
                    "y": np.asarray(reduced["y"], dtype=np.float32),
                    "dy": np.asarray(reduced["dy"], dtype=np.float32),
                    "ddy": np.asarray(reduced["ddy"], dtype=np.float32),
                    "force_per_m": np.asarray(reduced["force_per_m"], dtype=np.float32),
                    "flow_speed": np.asarray(reduced["flow_speed"], dtype=np.float32),
                    "ur": np.asarray(reduced["ur"], dtype=np.float32),
                    "stiffness_n_m": np.asarray(stiffness, dtype=np.float32),
                    "dry_mass_kg": np.asarray(dry_mass, dtype=np.float32),
                    "effective_mass_kg": np.asarray(effective_mass, dtype=np.float32),
                    "damping_c": np.asarray(damping_c, dtype=np.float32),
                    "rho_kg_m3": np.asarray(rho, dtype=np.float32),
                    "diameter_m": np.asarray(diameter, dtype=np.float32),
                }
            )
    return trajectories


def _last_linear(module: nn.Module) -> nn.Linear | None:
    last: nn.Linear | None = None
    for sub in module.modules():
        if isinstance(sub, nn.Linear):
            last = sub
    return last


def _build_backbone(
    *,
    input_dim: int,
    output_dim: int,
    architecture_cfg: Any,
) -> nn.Module:
    net_type = str(getattr(architecture_cfg, "force_net_type", "residual")).strip().lower()
    if net_type == "pirate":
        kwargs = dict(getattr(architecture_cfg, "pirate_force_kwargs", {}) or {})
        return ODEPirateNet(
            input_size=int(input_dim),
            output_size=int(output_dim),
            depth=int(kwargs.pop("depth", kwargs.pop("pirate_layers", 2))),
            fourier_features=int(kwargs.pop("fourier_features", 64)),
            sigma=float(kwargs.pop("sigma", 1.0)),
            use_rwf=bool(kwargs.pop("use_rwf", False)),
            activation=str(kwargs.pop("activation", "gelu")),
            dtype=torch.float32,
            **kwargs,
        )
    if net_type == "residual":
        cfg = _default_residual_kwargs()
        cfg.update(getattr(architecture_cfg, "residual_kwargs", {}) or {})
        hidden = int(cfg.get("hidden", 128))
        layers = max(1, int(cfg.get("layers", 2)))
        modules: list[nn.Module] = [nn.Linear(int(input_dim), hidden)]
        for _ in range(layers):
            modules.append(Residual(hidden, activation=str(cfg.get("activation", "gelu"))))
        modules.append(nn.Linear(hidden, int(output_dim)))
        return nn.Sequential(*modules)
    if net_type == "mlp":
        cfg = _default_mlp_kwargs()
        cfg.update(getattr(architecture_cfg, "mlp_kwargs", {}) or {})
        hidden = int(cfg.get("hidden", 128))
        layers = max(1, int(cfg.get("layers", 2)))
        act_cls = _activation_factory(str(cfg.get("activation", "gelu")))
        modules = []
        in_features = int(input_dim)
        for _ in range(layers):
            modules.append(nn.Linear(in_features, hidden))
            modules.append(act_cls())
            in_features = hidden
        modules.append(nn.Linear(in_features, int(output_dim)))
        return nn.Sequential(*modules)
    raise ValueError("architecture.force_net_type must be one of: residual, mlp, pirate.")


class LatentRNNForceModel(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int,
        encoder_input_dim: int,
        encoder_hidden: int,
        encoder_layers: int,
        encoder_dropout: float,
        backbone_input_dim: int,
        architecture_cfg: Any,
        rho: float,
        diameter: float,
        force_output: str,
        coefficient_output_bound: float | None,
        input_scaling_mode: str,
        ur_scale: float,
        latent_time_scale: float,
        corr_init_mode: str = "zero",
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.encoder = nn.GRU(
            input_size=int(encoder_input_dim),
            hidden_size=int(encoder_hidden),
            num_layers=int(encoder_layers),
            batch_first=True,
            dropout=float(encoder_dropout) if int(encoder_layers) > 1 else 0.0,
        )
        self.encoder_out = nn.Linear(int(encoder_hidden), int(latent_dim))
        self.backbone = _build_backbone(
            input_dim=int(backbone_input_dim),
            output_dim=1 + int(latent_dim),
            architecture_cfg=architecture_cfg,
        )
        self.rho = float(rho)
        self.D = float(diameter)
        self.force_output = str(force_output).strip().lower()
        if self.force_output not in {"force", "coefficient"}:
            raise ValueError("model.force_output must be one of: force, coefficient.")
        self.coefficient_output_bound = (
            None if coefficient_output_bound is None else float(coefficient_output_bound)
        )
        self.input_scaling_mode = resolve_phnn_input_scaling_mode(input_scaling_mode)
        self.ur_scale = float(ur_scale)
        self.latent_time_scale = float(latent_time_scale)
        if not np.isfinite(self.latent_time_scale) or self.latent_time_scale <= 0.0:
            raise ValueError("latent_time_scale must resolve to a positive finite value.")
        self._apply_final_init(corr_init_mode)

    def _apply_final_init(self, mode: str) -> None:
        key = str(mode).strip().lower()
        if key == "standard":
            return
        last = _last_linear(self.backbone)
        if last is None:
            return
        with torch.no_grad():
            if key == "zero":
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)
            elif key == "tiny":
                nn.init.normal_(last.weight, mean=0.0, std=1.0e-4)
                nn.init.zeros_(last.bias)
            else:
                raise ValueError("latent_rnn.corr_init_mode must be one of: zero, tiny, standard.")

    def encode(self, history: torch.Tensor) -> torch.Tensor:
        _out, hidden = self.encoder(history)
        return self.encoder_out(hidden[-1])

    def _flow_speed_from_ur(
        self,
        ur: torch.Tensor,
        *,
        mass: torch.Tensor,
        stiffness: torch.Tensor,
    ) -> torch.Tensor:
        omega_n = torch.sqrt(torch.clamp(stiffness / torch.clamp(mass, min=1.0e-12), min=1.0e-12))
        f_n = omega_n / (2.0 * math.pi)
        return ur * f_n * float(self.D)

    def _state_features(
        self,
        z: torch.Tensor,
        *,
        ur: torch.Tensor,
        mass: torch.Tensor,
        stiffness: torch.Tensor,
        flow_speed: torch.Tensor,
    ) -> torch.Tensor:
        y = z[:, 0:1]
        p = z[:, 1:2]
        velocity = p / torch.clamp(mass, min=1.0e-12)
        y_feat = y / float(self.D)
        if self.input_scaling_mode == "convective":
            u = torch.clamp(torch.abs(flow_speed), min=1.0e-12)
            v_feat = velocity / u
            ur_feat = flow_speed / float(self.D)
        else:
            p_scale = torch.sqrt(torch.clamp(mass * stiffness, min=1.0e-12)) * float(self.D)
            v_feat = p / torch.clamp(p_scale, min=1.0e-12)
            ur_feat = ur / float(self.ur_scale)
        return torch.cat([y_feat, v_feat, ur_feat], dim=1)

    def _force_scale(
        self,
        *,
        raw_force: torch.Tensor,
        ur: torch.Tensor,
        mass: torch.Tensor,
        stiffness: torch.Tensor,
        flow_speed: torch.Tensor,
    ) -> torch.Tensor:
        if self.force_output == "coefficient":
            if self.input_scaling_mode == "convective":
                u = flow_speed
            else:
                u = self._flow_speed_from_ur(ur, mass=mass, stiffness=stiffness)
            return torch.clamp(0.5 * float(self.rho) * float(self.D) * (u * u), min=1.0e-12)
        return torch.clamp(stiffness * float(self.D), min=1.0e-12).expand_as(raw_force)

    def predict_step(
        self,
        *,
        z: torch.Tensor,
        h: torch.Tensor,
        ur: torch.Tensor,
        mass: torch.Tensor,
        damping_c: torch.Tensor,
        stiffness: torch.Tensor,
        flow_speed: torch.Tensor,
        dt: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        features = self._state_features(
            z,
            ur=ur,
            mass=mass,
            stiffness=stiffness,
            flow_speed=flow_speed,
        )
        raw = self.backbone(torch.cat([features, h], dim=1))
        raw_force = raw[:, 0:1]
        raw_dh = raw[:, 1:]
        if self.force_output == "coefficient" and self.coefficient_output_bound is not None:
            raw_force = float(self.coefficient_output_bound) * torch.tanh(raw_force / float(self.coefficient_output_bound))
        force = raw_force * self._force_scale(
            raw_force=raw_force,
            ur=ur,
            mass=mass,
            stiffness=stiffness,
            flow_speed=flow_speed,
        )
        h_next = h + (dt / float(self.latent_time_scale)) * raw_dh
        y_next, v_next, a_next = structural_step_constant_force_torch(
            y=z[:, 0:1],
            velocity=z[:, 1:2] / torch.clamp(mass, min=1.0e-12),
            force=force,
            dt=dt,
            mass=mass,
            damping_c=damping_c,
            stiffness=stiffness,
        )
        z_next = torch.cat([y_next, v_next * mass], dim=1)
        return {
            "z_next": z_next,
            "h_next": h_next,
            "force": force,
            "raw_force": raw_force,
            "raw_dh": raw_dh,
            "acceleration_next": a_next,
        }

    def rollout(
        self,
        *,
        history: torch.Tensor,
        z0: torch.Tensor,
        t_seq: torch.Tensor,
        ur: torch.Tensor,
        mass: torch.Tensor,
        damping_c: torch.Tensor,
        stiffness: torch.Tensor,
        flow_speed: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        h = self.encode(history)
        return self.rollout_from_latent(
            h=h,
            z0=z0,
            t_seq=t_seq,
            ur=ur,
            mass=mass,
            damping_c=damping_c,
            stiffness=stiffness,
            flow_speed=flow_speed,
        )

    def rollout_from_latent(
        self,
        *,
        h: torch.Tensor,
        z0: torch.Tensor,
        t_seq: torch.Tensor,
        ur: torch.Tensor,
        mass: torch.Tensor,
        damping_c: torch.Tensor,
        stiffness: torch.Tensor,
        flow_speed: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        z = z0
        z_hist = [z0]
        force_hist: list[torch.Tensor] = []
        raw_force_hist: list[torch.Tensor] = []
        h_hist = [h]
        for idx in range(int(t_seq.shape[1]) - 1):
            dt = torch.clamp((t_seq[:, idx + 1 : idx + 2] - t_seq[:, idx : idx + 1]), min=1.0e-12)
            step = self.predict_step(
                z=z,
                h=h,
                ur=ur,
                mass=mass,
                damping_c=damping_c,
                stiffness=stiffness,
                flow_speed=flow_speed,
                dt=dt,
            )
            z = step["z_next"]
            h = step["h_next"]
            z_hist.append(z)
            h_hist.append(h)
            force_hist.append(step["force"])
            raw_force_hist.append(step["raw_force"])
        return {
            "z": torch.stack(z_hist, dim=1),
            "h": torch.stack(h_hist, dim=1),
            "force": (
                torch.stack(force_hist, dim=1)
                if force_hist
                else z0.new_zeros((z0.shape[0], 0, 1))
            ),
            "raw_force": (
                torch.stack(raw_force_hist, dim=1)
                if raw_force_hist
                else z0.new_zeros((z0.shape[0], 0, 1))
            ),
        }


def _trajectory_state_scale(
    *,
    z_like: torch.Tensor,
    mass: torch.Tensor,
    stiffness: torch.Tensor,
    diameter: float,
    input_scaling_mode: str,
    flow_speed: torch.Tensor,
) -> torch.Tensor:
    q_scale = torch.full_like(z_like[..., 0:1], float(diameter))
    if resolve_phnn_input_scaling_mode(input_scaling_mode) == "convective":
        p_scale = mass.view(-1, 1, 1) * torch.clamp(torch.abs(flow_speed.view(-1, 1, 1)), min=1.0e-12)
    else:
        p_scale = torch.sqrt(torch.clamp(mass.view(-1, 1, 1) * stiffness.view(-1, 1, 1), min=1.0e-12)) * float(diameter)
    return torch.cat([q_scale, p_scale.expand_as(q_scale)], dim=2)


def _grad_norm_from_parameters(
    parameters: Any,
    *,
    like: torch.Tensor,
) -> torch.Tensor:
    total = like.detach().new_zeros(())
    found = False
    for param in parameters:
        if param.grad is None:
            continue
        grad = param.grad.detach()
        total = total + torch.sum(grad * grad)
        found = True
    if not found:
        return total
    return torch.sqrt(total)


def _regularizer(value: torch.Tensor, norm: str) -> torch.Tensor:
    key = str(norm).strip().lower()
    if key == "l1":
        return torch.mean(torch.abs(value))
    if key == "l2":
        return torch.mean(value * value)
    raise ValueError("Regularizer norm must be one of: l1, l2.")


def _encoder_features_for_traj(
    traj: dict[str, np.ndarray],
    *,
    mass_key: str,
    input_scaling_mode: str,
    diameter: float,
    ur_scale: float,
    include_acceleration: bool,
) -> torch.Tensor:
    y = torch.from_numpy(np.ascontiguousarray(traj["y"])).float().unsqueeze(1)
    dy = torch.from_numpy(np.ascontiguousarray(traj["dy"])).float().unsqueeze(1)
    ddy = torch.from_numpy(np.ascontiguousarray(traj["ddy"])).float().unsqueeze(1)
    ur = torch.from_numpy(np.ascontiguousarray(traj["ur"])).float().unsqueeze(1)
    flow_speed = torch.from_numpy(np.ascontiguousarray(traj["flow_speed"])).float().unsqueeze(1)
    mass = float(np.asarray(traj[mass_key]).reshape(()))
    stiffness = float(np.asarray(traj["stiffness_n_m"]).reshape(()))
    y_feat = y / float(diameter)
    mode = resolve_phnn_input_scaling_mode(input_scaling_mode)
    if mode == "convective":
        u = torch.clamp(torch.abs(flow_speed), min=1.0e-12)
        v_feat = dy / u
        ur_feat = flow_speed / float(diameter)
        acc_feat = ddy * float(diameter) / torch.clamp(u * u, min=1.0e-12)
    else:
        p_scale = math.sqrt(max(mass * stiffness, 1.0e-12)) * float(diameter)
        v_feat = (dy * mass) / max(p_scale, 1.0e-12)
        ur_feat = ur / float(ur_scale)
        omega_n = math.sqrt(max(stiffness / mass, 1.0e-12))
        acc_feat = ddy / max((omega_n * omega_n) * float(diameter), 1.0e-12)
    parts = [y_feat, v_feat, ur_feat]
    if include_acceleration:
        parts.append(acc_feat)
    return torch.cat(parts, dim=1)


def _build_latent_window_dataset(
    trajs: list[dict[str, np.ndarray]],
    *,
    encoder_length: int,
    rollout_steps: int,
    mass_source: str,
    input_scaling_mode: str,
    diameter: float,
    ur_scale: float,
    include_acceleration: bool,
) -> TensorDataset | ConcatDataset | None:
    mass_key = {
        "dry": "dry_mass_kg",
        "effective": "effective_mass_kg",
        "stored": "dry_mass_kg",
        "label": "dry_mass_kg",
    }.get(str(mass_source).strip().lower())
    if mass_key is None:
        raise ValueError("latent_rnn.td_mass_source must be one of: dry, effective, stored, label.")
    items: list[TensorDataset] = []
    enc_len = max(1, int(encoder_length))
    steps = max(1, int(rollout_steps))
    total = enc_len + steps
    for traj in trajs:
        t = torch.from_numpy(np.ascontiguousarray(traj["t"])).float()
        y = torch.from_numpy(np.ascontiguousarray(traj["y"])).float()
        dy = torch.from_numpy(np.ascontiguousarray(traj["dy"])).float()
        force = torch.from_numpy(np.ascontiguousarray(traj["force_per_m"])).float().unsqueeze(1)
        ur = torch.from_numpy(np.ascontiguousarray(traj["ur"])).float().unsqueeze(1)
        flow = torch.from_numpy(np.ascontiguousarray(traj["flow_speed"])).float().unsqueeze(1)
        mass_value = float(np.asarray(traj[mass_key]).reshape(()))
        damping_value = float(np.asarray(traj["damping_c"]).reshape(()))
        stiffness_value = float(np.asarray(traj["stiffness_n_m"]).reshape(()))
        z = torch.cat([y.unsqueeze(1), (dy * mass_value).unsqueeze(1)], dim=1)
        history_features = _encoder_features_for_traj(
            traj,
            mass_key=mass_key,
            input_scaling_mode=input_scaling_mode,
            diameter=diameter,
            ur_scale=ur_scale,
            include_acceleration=include_acceleration,
        )
        if int(z.shape[0]) < total:
            continue
        histories = []
        z0s = []
        tseqs = []
        ztrajs = []
        urs = []
        flows = []
        force_seqs = []
        masses = []
        dampings = []
        stiffnesses = []
        for start in range(int(z.shape[0]) - total + 1):
            context_end = start + enc_len - 1
            end = context_end + steps + 1
            histories.append(history_features[start : start + enc_len])
            z0s.append(z[context_end])
            tseqs.append(t[context_end:end])
            ztrajs.append(z[context_end:end])
            urs.append(ur[context_end])
            flows.append(flow[context_end])
            force_seqs.append(force[context_end + 1 : end])
            masses.append(torch.tensor([mass_value], dtype=torch.float32))
            dampings.append(torch.tensor([damping_value], dtype=torch.float32))
            stiffnesses.append(torch.tensor([stiffness_value], dtype=torch.float32))
        if histories:
            items.append(
                TensorDataset(
                    torch.stack(histories, dim=0),
                    torch.stack(z0s, dim=0),
                    torch.stack(tseqs, dim=0),
                    torch.stack(ztrajs, dim=0),
                    torch.stack(urs, dim=0),
                    torch.stack(flows, dim=0),
                    torch.stack(force_seqs, dim=0),
                    torch.stack(masses, dim=0),
                    torch.stack(dampings, dim=0),
                    torch.stack(stiffnesses, dim=0),
                )
            )
    if not items:
        return None
    return items[0] if len(items) == 1 else ConcatDataset(items)


def _unpack_batch(batch: Sequence[torch.Tensor], *, device: torch.device, non_blocking: bool) -> tuple[torch.Tensor, ...]:
    if len(batch) != 10:
        raise ValueError("Unexpected latent_rnn batch format.")
    return tuple(t.to(device, non_blocking=non_blocking) for t in batch)


def _latent_losses_from_batch(
    *,
    model: LatentRNNForceModel,
    batch: Sequence[torch.Tensor],
    device: torch.device,
    non_blocking: bool,
    trajectory_relative: bool,
    compute_disp_std_loss: bool,
    compute_disp_mean_loss: bool,
    disp_std_relative: bool,
    disp_std_power: float,
    compute_disp_spectral_loss: bool,
    disp_spectral_loss_mode: str,
    disp_freq_relative: bool,
    disp_freq_power: float,
    disp_freq_alpha: float,
    disp_psd_relative: bool,
    disp_psd_peak_rel_bandwidth: float,
    disp_psd_use_hann_window: bool,
    mean_reg_norm: str,
) -> dict[str, torch.Tensor]:
    history, z0, t_seq, z_traj, ur, flow, force_true, mass, damping, stiffness = _unpack_batch(
        batch,
        device=device,
        non_blocking=non_blocking,
    )
    rollout = model.rollout(
        history=history,
        z0=z0,
        t_seq=t_seq,
        ur=ur,
        mass=mass,
        damping_c=damping,
        stiffness=stiffness,
        flow_speed=flow,
    )
    z_pred = rollout["z"]
    z_scale = _trajectory_state_scale(
        z_like=z_traj,
        mass=mass,
        stiffness=stiffness,
        diameter=float(model.D),
        input_scaling_mode=str(model.input_scaling_mode),
        flow_speed=flow,
    )
    state_loss = _trajectory_rollout_mse_torch(
        true_traj=z_traj[:, 1:2, :],
        pred_traj=z_pred[:, 1:2, :],
        z_scale=z_scale[:, 1:2, :],
        relative=False,
    )
    trajectory_loss = _trajectory_rollout_mse_torch(
        true_traj=z_traj[:, 1:, :],
        pred_traj=z_pred[:, 1:, :],
        z_scale=z_scale[:, 1:, :],
        relative=trajectory_relative,
    )
    disp_std_loss = z_traj.new_zeros(())
    if compute_disp_std_loss:
        disp_std_loss = _displacement_std_error_torch(
            true_signal=z_traj[:, 1:, 0],
            pred_signal=z_pred[:, 1:, 0],
            relative=disp_std_relative,
            power=disp_std_power,
        )
    disp_mean_loss = z_traj.new_zeros(())
    if compute_disp_mean_loss:
        disp_mean_loss = _displacement_mean_error_torch(
            true_signal=z_traj[:, 1:, 0],
            pred_signal=z_pred[:, 1:, 0],
            relative=disp_std_relative,
            power=disp_std_power,
        )
    disp_spectral_loss = z_traj.new_zeros(())
    if compute_disp_spectral_loss:
        dt_values = torch.clamp(t_seq[:, 1] - t_seq[:, 0], min=1.0e-12)
        if disp_spectral_loss_mode == "dominant_frequency":
            disp_spectral_loss = _dominant_frequency_error_torch(
                true_signal=z_traj[:, 1:, 0],
                pred_signal=z_pred[:, 1:, 0],
                dt=dt_values,
                peak_rel_bandwidth=disp_psd_peak_rel_bandwidth,
                use_hann_window=disp_psd_use_hann_window,
                relative=disp_freq_relative,
                power=disp_freq_power,
                alpha=disp_freq_alpha,
            )
        else:
            disp_spectral_loss = _psd_error_torch(
                true_signal=z_traj[:, 1:, 0],
                pred_signal=z_pred[:, 1:, 0],
                dt=dt_values,
                peak_rel_bandwidth=disp_psd_peak_rel_bandwidth,
                use_hann_window=disp_psd_use_hann_window,
                relative=disp_psd_relative,
            )
    force_data_loss = torch.mean((rollout["force"] - force_true) ** 2)
    mean_reg_loss = _regularizer(rollout["raw_force"], mean_reg_norm)
    return {
        "state_loss": state_loss,
        "trajectory_loss": trajectory_loss,
        "disp_std_loss": disp_std_loss,
        "disp_mean_loss": disp_mean_loss,
        "disp_spectral_loss": disp_spectral_loss,
        "force_data_loss": force_data_loss,
        "mean_reg_loss": mean_reg_loss,
        "z_pred": z_pred,
        "force_pred": rollout["force"],
    }


def _trajectory_mass_key(mass_source: str) -> str:
    key = {
        "dry": "dry_mass_kg",
        "effective": "effective_mass_kg",
        "stored": "dry_mass_kg",
        "label": "dry_mass_kg",
    }.get(str(mass_source).strip().lower())
    if key is None:
        raise ValueError("latent_rnn.td_mass_source must be one of: dry, effective, stored, label.")
    return key


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
        mass_key = "ic_dry_mass_kg" if str(td_mass_source).strip().lower() == "dry" else "ic_effective_mass_kg"
        required_scalar_keys = (
            "ic_y0",
            "ic_dy0",
            "ic_ddy0",
            "ic_dt",
            "ic_flow_speed0",
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
            row = {
                "index": idx,
                "ur": float(ur[idx]),
                "ur_label": (
                    _finite_scalar_from_npz(data, "ur_label", idx, path=path)
                    if "ur_label" in data
                    else float("nan")
                ),
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


def _build_surrogate_encoder_reference_groups(
    trajs: list[dict[str, np.ndarray]],
    *,
    encoder_length: int,
    mass_source: str,
    input_scaling_mode: str,
    diameter: float,
    ur_scale: float,
    include_acceleration: bool,
) -> list[dict[str, Any]]:
    enc_len = max(1, int(encoder_length))
    mass_key = _trajectory_mass_key(mass_source)
    groups: dict[float, dict[str, Any]] = {}
    for traj in trajs:
        t = np.asarray(traj.get("t", []), dtype=float).reshape(-1)
        if t.size < enc_len:
            continue
        ur_arr = np.asarray(traj.get("ur", []), dtype=float).reshape(-1)
        if ur_arr.size == 0:
            continue
        ur_value = float(ur_arr[0])
        if not np.isfinite(ur_value):
            continue
        history = _encoder_features_for_traj(
            traj,
            mass_key=mass_key,
            input_scaling_mode=input_scaling_mode,
            diameter=diameter,
            ur_scale=ur_scale,
            include_acceleration=include_acceleration,
        )[:enc_len]
        group_key = round(ur_value, 10)
        group = groups.setdefault(group_key, {"ur_values": [], "histories": []})
        group["ur_values"].append(ur_value)
        group["histories"].append(history)
    refs: list[dict[str, Any]] = []
    for group in groups.values():
        histories = group["histories"]
        if not histories:
            continue
        refs.append(
            {
                "ur": float(np.mean(np.asarray(group["ur_values"], dtype=float))),
                "histories": torch.stack(histories, dim=0),
                "count": len(histories),
            }
        )
    refs.sort(key=lambda item: float(item["ur"]))
    return refs


def _latent_from_nearest_encoder_references(
    *,
    row: dict[str, Any],
    references: list[dict[str, Any]],
    model: LatentRNNForceModel,
    device: torch.device,
    max_groups: int = 2,
) -> tuple[torch.Tensor, list[float]]:
    if not references:
        raise ValueError("Surrogate latent validation requires at least one real encoder reference trajectory.")
    selected = sorted(references, key=lambda item: abs(float(item["ur"]) - float(row["ur"])))[: max(1, int(max_groups))]
    encoded_groups: list[torch.Tensor] = []
    selected_urs: list[float] = []
    for item in selected:
        histories = item["histories"].to(device=device, dtype=torch.float32)
        with torch.no_grad():
            encoded = model.encode(histories)
        encoded_groups.append(encoded.mean(dim=0, keepdim=True))
        selected_urs.append(float(item["ur"]))
    return torch.mean(torch.cat(encoded_groups, dim=0), dim=0, keepdim=True), selected_urs


def _surrogate_force_scale(
    *,
    row: dict[str, Any],
    model: LatentRNNForceModel,
) -> float:
    diameter = max(float(row.get("ic_diameter_m", float(model.D))), 1.0e-12)
    if str(model.input_scaling_mode) == "convective":
        flow_speed = float(row["ic_flow_speed0"])
    else:
        fn = math.sqrt(max(float(row["ic_stiffness_n_m"]) / max(float(row["ic_effective_mass_kg"]), 1.0e-12), 1.0e-12)) / (
            2.0 * math.pi
        )
        flow_speed = float(row["ur"]) * fn * diameter
    return max(0.5 * float(model.rho) * diameter * flow_speed * flow_speed, 1.0e-12)


def _run_latent_surrogate_validation(
    *,
    rows: list[dict[str, Any]],
    writer: SummaryWriter,
    tb_step: int,
    tag: str,
    model: LatentRNNForceModel,
    encoder_references: list[dict[str, Any]],
    device: torch.device,
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
        diameter_value = max(float(row.get("ic_diameter_m", float(model.D))), 1.0e-12)
        h0, selected_ref_urs = _latent_from_nearest_encoder_references(
            row=row,
            references=encoder_references,
            model=model,
            device=device,
            max_groups=2,
        )
        z0 = torch.tensor([[float(row["ic_y0"]), float(row["ic_dy0"]) * mass_value]], dtype=torch.float32, device=device)
        t_seq = (torch.arange(steps + 1, dtype=torch.float32, device=device) * float(dt)).unsqueeze(0)
        ur0 = torch.tensor([[float(row["ur"])]], dtype=torch.float32, device=device)
        flow0 = torch.tensor([[float(row["ic_flow_speed0"])]], dtype=torch.float32, device=device)
        mass_t = torch.full((1, 1), mass_value, dtype=torch.float32, device=device)
        damping_t = torch.full((1, 1), damping_value, dtype=torch.float32, device=device)
        stiffness_t = torch.full((1, 1), stiffness_value, dtype=torch.float32, device=device)
        with torch.no_grad():
            rollout = model.rollout_from_latent(
                h=h0,
                z0=z0,
                t_seq=t_seq,
                ur=ur0,
                mass=mass_t,
                damping_c=damping_t,
                stiffness=stiffness_t,
                flow_speed=flow0,
            )
        y_pred = rollout["z"][0, :, 0].detach().cpu().numpy()
        force_pred = rollout["force"][0, :, 0].detach().cpu().numpy()
        if not np.all(np.isfinite(y_pred)) or not np.all(np.isfinite(force_pred)):
            diverged_count += 1
            continue
        y_eval = y_pred[discard_steps + 1 :]
        force_eval = force_pred[discard_steps:]
        if y_eval.size < 4 or force_eval.size < 4:
            raise ValueError(f"Surrogate row {row['index']} evaluation window is too short after discard.")
        fn = float(np.sqrt(stiffness_value / float(row["ic_effective_mass_kg"])) / (2.0 * np.pi))
        if not np.isfinite(fn) or fn <= 0.0:
            raise ValueError(f"Surrogate row {row['index']} has invalid effective natural frequency.")
        f0 = _surrogate_force_scale(row=row, model=model)
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
            "Encoder reference count": float(len(selected_ref_urs)),
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


def _latent_force_scale_numpy(
    *,
    model: LatentRNNForceModel,
    traj: dict[str, np.ndarray],
    mass_value: float,
    stiffness_value: float,
    sl: slice,
) -> np.ndarray:
    n = int(np.asarray(traj["t"])[sl].reshape(-1).shape[0])
    if n <= 0:
        return np.asarray([], dtype=float)
    if str(model.force_output) == "coefficient":
        if str(model.input_scaling_mode) == "convective":
            u = np.asarray(traj["flow_speed"], dtype=float).reshape(-1)[sl]
        else:
            omega_n = math.sqrt(float(stiffness_value) / max(float(mass_value), np.finfo(float).eps))
            f_n = omega_n / (2.0 * math.pi)
            ur = np.asarray(traj["ur"], dtype=float).reshape(-1)[sl]
            u = ur * f_n * float(model.D)
        return np.clip(0.5 * float(model.rho) * float(model.D) * (u * u), 1.0e-12, None)
    return np.full((n,), max(1.0e-12, float(stiffness_value) * float(model.D)), dtype=float)


def _latent_rollout_validation_case(
    *,
    model: LatentRNNForceModel,
    traj: dict[str, np.ndarray],
    encoder_length: int,
    include_acceleration: bool,
    mass_source: str,
    input_scaling_mode: str,
    ur_scale: float,
    device: torch.device,
) -> dict[str, Any]:
    enc_len = int(encoder_length)
    t_np = np.asarray(traj["t"], dtype=float).reshape(-1)
    if t_np.shape[0] <= enc_len:
        return {}
    context_end = enc_len - 1
    steps = int(t_np.shape[0] - enc_len)
    if steps < 1:
        return {}

    mass_value = float(np.asarray(traj[_trajectory_mass_key(mass_source)]).reshape(()))
    damping_value = float(np.asarray(traj["damping_c"]).reshape(()))
    stiffness_value = float(np.asarray(traj["stiffness_n_m"]).reshape(()))
    y = torch.from_numpy(np.ascontiguousarray(traj["y"])).float().unsqueeze(1)
    dy = torch.from_numpy(np.ascontiguousarray(traj["dy"])).float().unsqueeze(1)
    z = torch.cat([y, dy * float(mass_value)], dim=1)
    history_features = _encoder_features_for_traj(
        traj,
        mass_key=_trajectory_mass_key(mass_source),
        input_scaling_mode=input_scaling_mode,
        diameter=float(model.D),
        ur_scale=ur_scale,
        include_acceleration=include_acceleration,
    )
    history = history_features[:enc_len].unsqueeze(0).to(device)
    z0 = z[context_end : context_end + 1].to(device)
    t_seq = torch.from_numpy(np.ascontiguousarray(t_np[context_end:])).float().unsqueeze(0).to(device)
    ur0 = torch.tensor([[float(np.asarray(traj["ur"], dtype=float).reshape(-1)[context_end])]], dtype=torch.float32, device=device)
    flow0 = torch.tensor(
        [[float(np.asarray(traj["flow_speed"], dtype=float).reshape(-1)[context_end])]],
        dtype=torch.float32,
        device=device,
    )
    mass_t = torch.tensor([[mass_value]], dtype=torch.float32, device=device)
    damping_t = torch.tensor([[damping_value]], dtype=torch.float32, device=device)
    stiffness_t = torch.tensor([[stiffness_value]], dtype=torch.float32, device=device)

    with torch.no_grad():
        rollout = model.rollout(
            history=history,
            z0=z0,
            t_seq=t_seq,
            ur=ur0,
            mass=mass_t,
            damping_c=damping_t,
            stiffness=stiffness_t,
            flow_speed=flow0,
        )
    z_pred = rollout["z"][0].detach().cpu()
    force_pred = rollout["force"][0, :, 0].detach().cpu().numpy()
    post_slice = slice(context_end + 1, None)
    t_post = t_np[post_slice]
    y_true_post = np.asarray(traj["y"], dtype=float).reshape(-1)[post_slice]
    dy_true_post = np.asarray(traj["dy"], dtype=float).reshape(-1)[post_slice]
    force_true_post = np.asarray(traj["force_per_m"], dtype=float).reshape(-1)[post_slice]
    y_pred_post = z_pred[1:, 0].numpy()
    v_pred_post = (z_pred[1:, 1] / float(mass_value)).numpy()
    omega = math.sqrt(float(stiffness_value) / max(float(mass_value), np.finfo(float).eps))
    rollout_for_metrics = {
        "y_norm": y_pred_post / float(model.D),
        "p_norm": v_pred_post / max(omega * float(model.D), 1.0e-12),
        "force_total": force_pred,
    }
    dt_value = float(np.median(np.diff(t_post))) if t_post.shape[0] > 1 else float(np.median(np.diff(t_np)))
    y_t = torch.from_numpy(np.ascontiguousarray(y_true_post)).float()
    dy_t = torch.from_numpy(np.ascontiguousarray(dy_true_post)).float()
    ur_post = torch.from_numpy(np.ascontiguousarray(np.asarray(traj["ur"], dtype=float).reshape(-1)[post_slice])).float()
    metrics = compute_validation_metrics(
        model=model,
        y_data_t=y_t,
        val_vel=dy_t,
        reduced_velocity=ur_post,
        m_eff=mass_value,
        dt=dt_value,
        t=t_post,
        y_data_raw=y_true_post,
        force_data=force_true_post,
        D=float(model.D),
        k=stiffness_value,
        device=device,
        rollout=rollout_for_metrics,
    )
    force_std = float(np.std(force_true_post))
    if force_pred.size > 1 and force_true_post.size > 1 and np.isfinite(force_std) and force_std > 0.0:
        n_force = min(int(force_pred.size), int(force_true_post.size))
        metrics[FORCE_MAPPING_NRMSE_KEY] = float(
            np.sqrt(np.mean((force_pred[:n_force] - force_true_post[:n_force]) ** 2)) / force_std
        )

    force_scale = _latent_force_scale_numpy(
        model=model,
        traj=traj,
        mass_value=mass_value,
        stiffness_value=stiffness_value,
        sl=post_slice,
    )
    plot_len = int(min(t_post.size, y_true_post.size, y_pred_post.size, force_true_post.size, force_pred.size, force_scale.size))
    return {
        "metrics": metrics,
        "t": t_post[:plot_len],
        "dt": dt_value,
        "ur": float(np.asarray(traj["ur"], dtype=float).reshape(-1)[context_end]),
        "y_true_norm": y_true_post[:plot_len] / float(model.D),
        "y_pred_norm": y_pred_post[:plot_len] / float(model.D),
        "p_pred_norm": rollout_for_metrics["p_norm"][:plot_len],
        "force_true_coeff": force_true_post[:plot_len] / force_scale[:plot_len],
        "force_pred_coeff": force_pred[:plot_len] / force_scale[:plot_len],
    }


def _log_latent_rollout_validation(
    *,
    writer: SummaryWriter,
    epoch: int,
    model: LatentRNNForceModel,
    traj: dict[str, np.ndarray],
    encoder_length: int,
    include_acceleration: bool,
    mass_source: str,
    input_scaling_mode: str,
    ur_scale: float,
    device: torch.device,
    tag_prefix: str = "val_unseen/rollout",
    metric_prefix: str = "val_unseen",
    step: int | None = None,
    log_metrics: bool = True,
    log_plots: bool = True,
    log_force_plot: bool = True,
    log_spectra: bool = False,
    title_suffix: str = "",
) -> dict[str, float]:
    result = _latent_rollout_validation_case(
        model=model,
        traj=traj,
        encoder_length=encoder_length,
        include_acceleration=include_acceleration,
        mass_source=mass_source,
        input_scaling_mode=input_scaling_mode,
        ur_scale=ur_scale,
        device=device,
    )
    if not result:
        return {}
    metrics = dict(result["metrics"])
    tensorboard_step = epoch + 1 if step is None else step
    if log_metrics:
        for name, value in metrics.items():
            if name == ROLLOUT_DIVERGED_KEY:
                continue
            value_f = float(value)
            if np.isfinite(value_f):
                writer.add_scalar(f"{metric_prefix}/{name}", value_f, tensorboard_step)
    if not log_plots:
        return metrics
    t_plot = np.asarray(result["t"], dtype=float).reshape(-1)
    if t_plot.size < 2:
        return metrics
    zoom_mask = create_zoom_mask(t_plot)
    log_displacement_plots(
        writer,
        epoch,
        t_plot,
        np.asarray(result["y_true_norm"], dtype=float),
        np.asarray(result["y_pred_norm"], dtype=float),
        np.asarray(result["p_pred_norm"], dtype=float),
        zoom_mask,
        reduced_velocity=float(result["ur"]),
        tag_prefix=tag_prefix,
        step=step,
        title_suffix=title_suffix,
        log_spectra=log_spectra,
    )
    if log_force_plot:
        log_force_plots(
            writer,
            epoch,
            t_plot,
            np.asarray(result["force_pred_coeff"], dtype=float),
            np.asarray(result["force_true_coeff"], dtype=float),
            zoom_mask,
            reduced_velocity=float(result["ur"]),
            tag_prefix=tag_prefix,
            step=step,
            title_suffix=title_suffix,
            log_spectra=log_spectra,
        )
    if log_spectra:
        log_area_normalized_rollout_spectra(
            writer,
            epoch,
            disp_t=t_plot,
            disp_true=np.asarray(result["y_true_norm"], dtype=float),
            disp_pred=np.asarray(result["y_pred_norm"], dtype=float),
            force_t=t_plot,
            force_true=np.asarray(result["force_true_coeff"], dtype=float),
            force_pred=np.asarray(result["force_pred_coeff"], dtype=float),
            reduced_velocity=float(result["ur"]),
            force_baseline=None,
            tag=f"{tag_prefix}_spectra",
            step=step,
            title_suffix=title_suffix,
        )
    return metrics


def _resolve_latent_time_scale(raw: Any, *, train_trajs: list[dict[str, np.ndarray]]) -> float:
    key = str(raw).strip().lower()
    if key in {"", "auto", "median_dt"}:
        dts: list[float] = []
        for traj in train_trajs:
            t = np.asarray(traj["t"], dtype=float).reshape(-1)
            if t.size > 1:
                dts.extend(np.diff(t).tolist())
        if not dts:
            raise ValueError("Cannot resolve latent_time_scale=auto without at least one dt.")
        return float(np.median(np.asarray(dts, dtype=float)))
    value = float(raw)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("latent_rnn.latent_time_scale must be auto or a positive finite value.")
    return value


def train(config: Config, config_name: str) -> None:
    data_cfg = config.data
    model_cfg = config.model
    arch_cfg = config.architecture
    training_cfg = config.training
    optim_cfg = config.optim
    loss_cfg = config.loss
    runtime_cfg = config.runtime
    precision_cfg = config.precision
    compile_cfg = config.compile
    monitoring_cfg = config.monitoring
    lrnn_cfg = dict(config.latent_rnn or {})

    device = select_device(os.getenv("TRAIN_DEVICE", str(runtime_cfg.device)))
    print(f"Using device: {device}")
    configure_tf32(device, bool(precision_cfg.use_tf32))
    set_num_threads_from_slurm(default=1)
    amp_enabled, amp_dtype, scaler = setup_amp(
        device,
        use_amp=bool(precision_cfg.use_amp),
        amp_dtype=str(precision_cfg.amp_dtype),
    )
    non_blocking = device.type == "cuda"

    train_series_root = Path(data_cfg.train_series_dir)
    train_dir = train_series_root / "train"
    val_unseen_dir = train_series_root / "val_unseen"
    if not val_unseen_dir.exists():
        val_unseen_dir = train_series_root / "val"
    val_seen_dir = train_series_root / "val_seen"
    if not train_dir.exists():
        raise FileNotFoundError("latent_rnn expects train/ under data.train_series_dir.")
    train_paths = sorted(train_dir.glob("*.npz"))
    val_paths = sorted(val_unseen_dir.glob("*.npz")) if val_unseen_dir.exists() else []
    val_seen_paths = sorted(val_seen_dir.glob("*.npz")) if val_seen_dir.exists() else []
    if not train_paths:
        raise FileNotFoundError("No latent_rnn training trajectories were found.")

    mass_source = str(lrnn_cfg.get("td_mass_source", "dry")).strip().lower()
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
    train_trajs = _load_latent_rnn_trajectories(
        train_paths,
        cut_start_seconds=train_cut,
        reduce_time=reduce_time_enabled,
        reduction_factor=reduction_factor,
        stagger_reduced_time=stagger_train_reduce,
        mass_source=mass_source,
    )
    val_trajs = _load_latent_rnn_trajectories(
        val_paths,
        cut_start_seconds=val_cut,
        reduce_time=reduce_time_enabled,
        reduction_factor=reduction_factor,
        stagger_reduced_time=stagger_val_reduce,
        mass_source=mass_source,
    )
    val_seen_trajs = (
        _load_latent_rnn_trajectories(
            val_seen_paths,
            cut_start_seconds=val_cut,
            reduce_time=reduce_time_enabled,
            reduction_factor=reduction_factor,
            stagger_reduced_time=stagger_val_reduce,
            mass_source=mass_source,
        )
        if val_seen_paths
        else []
    )
    if not train_trajs:
        raise ValueError("No latent_rnn training trajectories remained after loading/reduction.")

    latent_dim = int(lrnn_cfg.get("latent_dim", 3))
    encoder_length = int(lrnn_cfg.get("encoder_length", 50))
    if latent_dim < 1:
        raise ValueError("latent_rnn.latent_dim must be >= 1.")
    if encoder_length < 1:
        raise ValueError("latent_rnn.encoder_length must be >= 1.")
    encoder_type = str(lrnn_cfg.get("encoder_type", "gru")).strip().lower()
    if encoder_type != "gru":
        raise ValueError("latent_rnn.encoder_type currently supports only 'gru'.")
    latent_update = str(lrnn_cfg.get("latent_update", "dt_scaled")).strip().lower()
    if latent_update != "dt_scaled":
        raise ValueError("latent_rnn.latent_update currently supports only 'dt_scaled'.")
    include_acceleration = bool(lrnn_cfg.get("encoder_include_acceleration", True))
    input_scaling_mode = resolve_phnn_input_scaling_mode(getattr(model_cfg, "input_scaling_mode", "current"))
    ur_scale = 10.0 if getattr(model_cfg, "ur_scale", None) is None else float(model_cfg.ur_scale)
    latent_time_scale = _resolve_latent_time_scale(lrnn_cfg.get("latent_time_scale", "auto"), train_trajs=train_trajs)
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
    rollout_batch_size = int(training_cfg.batch_size) if rollout_batch_size_raw <= 0 else rollout_batch_size_raw
    state_weight = float(getattr(loss_cfg, "state_weight", 1.0))
    mean_reg = float(getattr(loss_cfg, "mean_reg", 0.0))
    mean_reg_norm = str(getattr(loss_cfg, "mean_reg_norm", "l1")).strip().lower()
    if mean_reg_norm not in {"l1", "l2"}:
        raise ValueError("loss.mean_reg_norm must be one of: l1, l2.")
    force_data_weight = float(getattr(loss_cfg, "force_data_weight", 1.0))
    use_force_data_loss = bool(getattr(loss_cfg, "use_force_data_loss", False))
    state_loss_active = state_weight > 0.0
    data_loss_active = use_force_data_loss and force_data_weight > 0.0
    spectral_mode = _normalize_rollout_disp_spectral_loss_mode(
        getattr(loss_cfg, "rollout_disp_spectral_loss", "psd")
    )
    psd_peak_rel_bw = float(getattr(loss_cfg, "rollout_disp_psd_peak_rel_bandwidth", 0.0))
    psd_use_hann = bool(getattr(loss_cfg, "rollout_disp_psd_use_hann_window", True))
    surrogate_enabled = bool(getattr(monitoring_cfg, "surrogate_validation_enabled", True))
    surrogate_tag = str(getattr(monitoring_cfg, "surrogate_validation_tag", "val_surrogate")).strip() or "val_surrogate"
    combined_tag = str(getattr(monitoring_cfg, "combined_validation_tag", "val")).strip() or "val"
    surrogate_rows: list[dict[str, Any]] = []
    if surrogate_enabled:
        surrogate_rows = _load_surrogate_validation_rows(
            Path(getattr(monitoring_cfg, "surrogate_validation_npz", "CFD_Data/analysis/surrogate_validation_points.npz")),
            td_mass_source=mass_source,
        )
        surrogate_rows = _maybe_reduce_surrogate_validation_rows(
            surrogate_rows,
            reduce_time=reduce_time_enabled,
            reduction_factor=reduction_factor,
        )
        surrogate_step_counts = [int(round(float(row["rollout_steps"]))) for row in surrogate_rows]
        print(
            f"[latent_rnn] loaded {len(surrogate_rows)} surrogate validation row(s) "
            f"from {getattr(monitoring_cfg, 'surrogate_validation_npz', 'CFD_Data/analysis/surrogate_validation_points.npz')} "
            f"(rollout_steps={surrogate_step_counts})"
        )
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
        raise ValueError("Surrogate validation is enabled, but no real latent encoder reference histories were available.")

    train_dataset = _build_latent_window_dataset(
        train_trajs,
        encoder_length=encoder_length,
        rollout_steps=rollout_steps,
        mass_source=mass_source,
        input_scaling_mode=input_scaling_mode,
        diameter=float(model_cfg.D),
        ur_scale=ur_scale,
        include_acceleration=include_acceleration,
    )
    if train_dataset is None:
        raise ValueError("No latent_rnn training windows were built.")
    val_dataset = _build_latent_window_dataset(
        val_trajs,
        encoder_length=encoder_length,
        rollout_steps=rollout_steps,
        mass_source=mass_source,
        input_scaling_mode=input_scaling_mode,
        diameter=float(model_cfg.D),
        ur_scale=ur_scale,
        include_acceleration=include_acceleration,
    )
    val_seen_dataset = _build_latent_window_dataset(
        val_seen_trajs,
        encoder_length=encoder_length,
        rollout_steps=rollout_steps,
        mass_source=mass_source,
        input_scaling_mode=input_scaling_mode,
        diameter=float(model_cfg.D),
        ur_scale=ur_scale,
        include_acceleration=include_acceleration,
    )
    loader_workers = 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=rollout_batch_size,
        shuffle=True,
        num_workers=loader_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=rollout_batch_size,
            shuffle=False,
            num_workers=loader_workers,
            pin_memory=(device.type == "cuda"),
        )
        if val_dataset is not None
        else None
    )
    val_seen_loader = (
        DataLoader(
            val_seen_dataset,
            batch_size=rollout_batch_size,
            shuffle=False,
            num_workers=loader_workers,
            pin_memory=(device.type == "cuda"),
        )
        if val_seen_dataset is not None
        else None
    )

    encoder_input_dim = 3 + (1 if include_acceleration else 0)
    backbone_input_dim = 3 + latent_dim
    first_traj = train_trajs[0]
    model = LatentRNNForceModel(
        latent_dim=latent_dim,
        encoder_input_dim=encoder_input_dim,
        encoder_hidden=int(lrnn_cfg.get("encoder_hidden", 128)),
        encoder_layers=int(lrnn_cfg.get("encoder_layers", 1)),
        encoder_dropout=float(lrnn_cfg.get("encoder_dropout", 0.0)),
        backbone_input_dim=backbone_input_dim,
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
    model = maybe_compile_model(model, bool(compile_cfg.use_compile), str(compile_cfg.compile_mode))
    gradnorm_balancer: GradNormBalancer | None = None
    if bool(getattr(loss_cfg, "use_gradnorm", False)):
        gradnorm_loss_names: list[str] = []
        if state_loss_active:
            gradnorm_loss_names.append("state")
        if data_loss_active:
            gradnorm_loss_names.append("data")
        if rollout_active:
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
        else:
            print("loss.use_gradnorm is True but fewer than two latent_rnn loss groups are active; skipping GradNorm.")

    optimizer, scheduler = setup_optimizer_and_scheduler(
        model,
        optim_cfg=optim_cfg,
        scheduler_cfg=optim_cfg.scheduler,
        epochs=int(training_cfg.epochs),
    )
    writer, run_name = setup_writer(
        config.logging.run_dir_root,
        config_name,
        run_name_override=getattr(config.logging, "run_name", None),
        append_timestamp=bool(getattr(config.logging, "append_timestamp", True)),
    )
    writer.add_text("latent_rnn/config", json.dumps(lrnn_cfg, indent=2, sort_keys=True), 0)
    writer.add_text(
        "latent_rnn/resolved",
        json.dumps(
            {
                "latent_time_scale": latent_time_scale,
                "rollout_steps": rollout_steps,
                "encoder_input_dim": encoder_input_dim,
                "backbone_input_dim": backbone_input_dim,
                "train_windows": len(train_loader.dataset),
                "val_unseen_windows": 0 if val_loader is None else len(val_loader.dataset),
                "val_seen_windows": 0 if val_seen_loader is None else len(val_seen_loader.dataset),
                "surrogate_validation_rows": len(surrogate_rows),
                "surrogate_encoder_reference_groups": len(surrogate_encoder_references),
            },
            indent=2,
            sort_keys=True,
        ),
        0,
    )

    print(
        "\n".join(
            [
                f"Run name: {run_name}",
                (
                    f"Latent RNN setup: epochs={int(training_cfg.epochs)}, batch_size={rollout_batch_size}, "
                    f"train_windows={len(train_loader.dataset)}, "
                    f"val_unseen_windows={0 if val_loader is None else len(val_loader.dataset)}, "
                    f"val_seen_windows={0 if val_seen_loader is None else len(val_seen_loader.dataset)}, "
                    f"surrogate_rows={len(surrogate_rows)}, "
                    f"surrogate_encoder_ref_groups={len(surrogate_encoder_references)}"
                ),
                (
                    f"Ablation: no Vivana-TD force/context/phase/sigma/fhat inputs are loaded into the model; "
                    f"latent_dim={latent_dim}, encoder_length={encoder_length}, latent_time_scale={latent_time_scale:g}"
                ),
                (
                    f"Loss weights: state={state_weight:g}, force_data="
                    f"{force_data_weight if use_force_data_loss else 0.0:g}, rollout_det={rollout_det_weight:g}, "
                    f"rollout_disp_std={rollout_disp_std_weight:g}, "
                    f"rollout_disp_mean=same_as_std({rollout_disp_mean_in_std_loss}), "
                    f"rollout_disp_spectral={rollout_disp_spectral_weight:g}, "
                    f"mean_reg={mean_reg:g}({mean_reg_norm})"
                ),
            ]
        )
    )

    def _run_epoch(loader: DataLoader, *, train_mode: bool) -> dict[str, float]:
        if train_mode:
            model.train()
        else:
            model.eval()
        sums: dict[str, torch.Tensor] = {}
        count = 0
        gradnorm_state_w_sum = (
            torch.zeros((), device=device) if train_mode and gradnorm_balancer is not None and state_loss_active else None
        )
        gradnorm_data_w_sum = (
            torch.zeros((), device=device) if train_mode and gradnorm_balancer is not None and data_loss_active else None
        )
        gradnorm_rollout_w_sum = (
            torch.zeros((), device=device) if train_mode and gradnorm_balancer is not None and rollout_active else None
        )
        gradnorm_count = 0
        for batch in loader:
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(train_mode):
                with torch.amp.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                    losses = _latent_losses_from_batch(
                        model=model,
                        batch=batch,
                        device=device,
                        non_blocking=non_blocking,
                        trajectory_relative=bool(rollout_settings["trajectory_relative"]),
                        compute_disp_std_loss=rollout_disp_std_weight > 0.0,
                        compute_disp_mean_loss=(
                            rollout_disp_std_weight > 0.0 and rollout_disp_mean_in_std_loss
                        ),
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
                    if train_mode and gradnorm_balancer is not None:
                        loss_inputs: dict[str, torch.Tensor] = {}
                        if state_loss_active:
                            loss_inputs["state"] = losses["state_loss"].float()
                        if data_loss_active:
                            loss_inputs["data"] = losses["force_data_loss"].float()
                        if rollout_active:
                            loss_inputs["rollout"] = rollout_total_loss.float()
                        weights = gradnorm_balancer.update(loss_inputs)
                        total_loss = losses["state_loss"].new_zeros(())
                        if state_loss_active:
                            state_w = weights["state"]
                            total_loss = total_loss + state_weight * state_w * losses["state_loss"]
                            if gradnorm_state_w_sum is not None:
                                gradnorm_state_w_sum = gradnorm_state_w_sum + state_w.detach()
                        if data_loss_active:
                            data_w = weights["data"]
                            total_loss = total_loss + force_data_weight * data_w * losses["force_data_loss"]
                            if gradnorm_data_w_sum is not None:
                                gradnorm_data_w_sum = gradnorm_data_w_sum + data_w.detach()
                        if rollout_active:
                            rollout_w = weights["rollout"]
                            total_loss = total_loss + rollout_w * rollout_total_loss
                            if gradnorm_rollout_w_sum is not None:
                                gradnorm_rollout_w_sum = gradnorm_rollout_w_sum + rollout_w.detach()
                        total_loss = total_loss + mean_reg * losses["mean_reg_loss"]
                        gradnorm_count += 1
                    else:
                        total_loss = state_weight * losses["state_loss"] + rollout_total_loss + mean_reg * losses["mean_reg_loss"]
                        if use_force_data_loss:
                            total_loss = total_loss + force_data_weight * losses["force_data_loss"]
                if train_mode:
                    scaler.scale(total_loss).backward()
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    grad_like = total_loss.detach()
                    grad_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                    grad_parameters = list(grad_model.parameters())
                    grad_norm_encoder = _grad_norm_from_parameters(
                        list(grad_model.encoder.parameters()) + list(grad_model.encoder_out.parameters()),
                        like=grad_like,
                    )
                    grad_norm_backbone = _grad_norm_from_parameters(grad_model.backbone.parameters(), like=grad_like)
                    if float(training_cfg.max_grad_norm) > 0.0:
                        grad_norm_raw = nn_utils.clip_grad_norm_(grad_parameters, float(training_cfg.max_grad_norm))
                        grad_norm_total = (
                            grad_norm_raw.detach()
                            if isinstance(grad_norm_raw, torch.Tensor)
                            else grad_like.new_tensor(float(grad_norm_raw))
                        )
                    else:
                        grad_norm_total = _grad_norm_from_parameters(grad_parameters, like=grad_like)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    grad_norm_total = total_loss.detach().new_zeros(())
                    grad_norm_encoder = total_loss.detach().new_zeros(())
                    grad_norm_backbone = total_loss.detach().new_zeros(())
            batch_size = int(batch[0].shape[0])
            count += batch_size
            batch_metrics = {
                "loss_total": total_loss,
                "loss_state": losses["state_loss"],
                "loss_rollout_det": losses["trajectory_loss"],
                "loss_rollout_disp_std": losses["disp_std_loss"],
                "loss_rollout_disp_mean": losses["disp_mean_loss"],
                "loss_rollout_spectral": losses["disp_spectral_loss"],
                "loss_reg_mean": losses["mean_reg_loss"],
            }
            if train_mode:
                batch_metrics.update(
                    {
                        "grad_norm": grad_norm_total,
                        "grad_norm_encoder": grad_norm_encoder,
                        "grad_norm_backbone": grad_norm_backbone,
                    }
                )
            for name, value in batch_metrics.items():
                sums[name] = sums.get(name, value.detach().new_zeros(())) + value.detach() * batch_size
        denom = float(max(1, count))
        metrics = {name: float((value / denom).detach().cpu()) for name, value in sums.items()}
        if gradnorm_count > 0:
            if gradnorm_state_w_sum is not None:
                metrics["gradnorm_weight_state"] = float((gradnorm_state_w_sum / float(gradnorm_count)).detach().cpu())
            if gradnorm_data_w_sum is not None:
                metrics["gradnorm_weight_data"] = float((gradnorm_data_w_sum / float(gradnorm_count)).detach().cpu())
            if gradnorm_rollout_w_sum is not None:
                metrics["gradnorm_weight_rollout"] = float((gradnorm_rollout_w_sum / float(gradnorm_count)).detach().cpu())
        return metrics

    run_models_dir = Path("models") / run_name
    run_models_dir.mkdir(parents=True, exist_ok=True)
    latest_path = run_models_dir / f"{run_name}_latest.pt"
    best_val = float("inf")
    epochs = int(training_cfg.epochs)
    validate_every = max(1, int(getattr(monitoring_cfg, "validate_every_epochs", 1)))
    print_every = max(1, int(getattr(monitoring_cfg, "print_every_epochs", 1)))
    log_every = max(1, int(getattr(monitoring_cfg, "log_every_epochs", 1)))
    validation_samples_per_ur = max(1, int(getattr(monitoring_cfg, "validation_samples_per_ur", 1)))
    final_rollout_all_validation = bool(getattr(monitoring_cfg, "final_rollout_all_validation", False))

    for epoch in range(epochs):
        if bool(optim_cfg.use_lr_scheduler):
            lr = scheduler.get_lr(epoch + 1)
            for group in optimizer.param_groups:
                group["lr"] = lr
        start = time.perf_counter()
        train_metrics = _run_epoch(train_loader, train_mode=True)
        elapsed = time.perf_counter() - start
        if (epoch + 1) % log_every == 0:
            for name, value in train_metrics.items():
                writer.add_scalar(f"train/{name}", value, epoch + 1)
            writer.add_scalar("train/lr", float(optimizer.param_groups[0]["lr"]), epoch + 1)
            writer.add_scalar("train/epoch_wall_time_s", elapsed, epoch + 1)
        if (epoch + 1) % print_every == 0 or epoch == 0:
            print(
                f"[latent_rnn] epoch {epoch + 1}/{epochs}: "
                f"loss={train_metrics.get('loss_total', float('nan')):.4e}, "
                f"state={train_metrics.get('loss_state', float('nan')):.4e}, "
                f"roll={train_metrics.get('loss_rollout_det', float('nan')):.4e}, "
                f"spectral={train_metrics.get('loss_rollout_spectral', float('nan')):.4e}"
            )
        validation_due = (epoch + 1) % validate_every == 0 or epoch == epochs - 1
        if validation_due and (val_loader is not None or val_seen_loader is not None or surrogate_rows):
            eval_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            eval_model.eval()

            def _validate_split(
                *,
                split_tag: str,
                split_loader: DataLoader | None,
                split_trajs: list[dict[str, np.ndarray]],
                log_rollout_plots: bool,
                log_all_rollout_plots: bool,
                log_force_plot: bool,
            ) -> tuple[dict[str, float], float]:
                if split_loader is None:
                    return {}, float("inf")
                validation_start = time.perf_counter()
                split_metrics = _run_epoch(split_loader, train_mode=False)
                for name, value in split_metrics.items():
                    writer.add_scalar(f"{split_tag}/{name}", value, epoch + 1)
                if split_trajs:
                    ur_for_sampling = [
                        float(np.asarray(traj["ur"], dtype=float).reshape(-1)[0])
                        for traj in split_trajs
                        if np.asarray(traj["ur"], dtype=float).reshape(-1).size > 0
                    ]
                    sampled_indices = sample_indices_per_ur(
                        ur_for_sampling,
                        samples_per_ur=validation_samples_per_ur,
                        seed=1,
                    )
                    metrics_sum: dict[str, float] = {}
                    metrics_count = 0
                    diverged_count = 0
                    for idx in sampled_indices:
                        if idx >= len(split_trajs):
                            continue
                        rollout_metrics = _latent_rollout_validation_case(
                            model=eval_model,
                            traj=split_trajs[idx],
                            encoder_length=encoder_length,
                            include_acceleration=include_acceleration,
                            mass_source=mass_source,
                            input_scaling_mode=input_scaling_mode,
                            ur_scale=ur_scale,
                            device=device,
                        ).get("metrics", {})
                        if not rollout_metrics:
                            continue
                        diverged_flag = float(rollout_metrics.get(ROLLOUT_DIVERGED_KEY, 0.0))
                        if np.isfinite(diverged_flag) and diverged_flag > 0.5:
                            diverged_count += 1
                        for name, value in rollout_metrics.items():
                            if name == ROLLOUT_DIVERGED_KEY:
                                continue
                            value_f = float(value)
                            if np.isfinite(value_f):
                                metrics_sum[name] = metrics_sum.get(name, 0.0) + value_f
                        metrics_count += 1
                    if metrics_count > 0:
                        for name, total in metrics_sum.items():
                            averaged_value = total / float(metrics_count)
                            split_metrics[name] = float(averaged_value)
                            writer.add_scalar(f"{split_tag}/{name}", averaged_value, epoch + 1)
                        split_metrics[ROLLOUT_DIVERGED_COUNT_KEY] = float(diverged_count)
                        writer.add_scalar(f"{split_tag}/{ROLLOUT_DIVERGED_COUNT_KEY}", float(diverged_count), epoch + 1)

                    if log_rollout_plots:
                        if log_all_rollout_plots:
                            plot_indices = list(range(len(split_trajs)))
                        else:
                            plot_indices = sample_one_index_per_ur(ur_for_sampling, seed=0)
                            if not plot_indices:
                                plot_indices = [0]
                            plot_indices = plot_indices[:1]
                        for rollout_idx_raw in plot_indices:
                            rollout_idx = int(rollout_idx_raw)
                            if not (0 <= rollout_idx < len(split_trajs)):
                                continue
                            tag_prefix = f"{split_tag}/rollout"
                            if log_all_rollout_plots:
                                tag_prefix = f"{split_tag}/rollout_{rollout_idx:03d}"
                            _log_latent_rollout_validation(
                                writer=writer,
                                epoch=epoch,
                                model=eval_model,
                                traj=split_trajs[rollout_idx],
                                encoder_length=encoder_length,
                                include_acceleration=include_acceleration,
                                mass_source=mass_source,
                                input_scaling_mode=input_scaling_mode,
                                ur_scale=ur_scale,
                                device=device,
                                tag_prefix=tag_prefix,
                                metric_prefix=split_tag,
                                step=epoch + 1,
                                log_metrics=False,
                                log_plots=True,
                                log_force_plot=log_force_plot,
                                log_spectra=False,
                            )
                writer.add_scalar(
                    f"{split_tag}/validation_wall_time_s",
                    float(time.perf_counter() - validation_start),
                    epoch + 1,
                )
                return split_metrics, split_metrics.get("loss_total", float("inf"))

            split_results: dict[str, dict[str, Any]] = {}
            val_metrics: dict[str, float] = {}
            val_loss = float("inf")
            if val_loader is not None:
                val_metrics, val_loss = _validate_split(
                    split_tag="val_unseen",
                    split_loader=val_loader,
                    split_trajs=val_trajs,
                    log_rollout_plots=True,
                    log_all_rollout_plots=False,
                    log_force_plot=True,
                )
                split_results["val_unseen"] = {"loss_total": val_loss, "val_metrics": val_metrics}
            if val_seen_loader is not None:
                val_seen_metrics, val_seen_loss = _validate_split(
                    split_tag="val_seen",
                    split_loader=val_seen_loader,
                    split_trajs=val_seen_trajs,
                    log_rollout_plots=True,
                    log_all_rollout_plots=True,
                    log_force_plot=False,
                )
                split_results["val_seen"] = {"loss_total": val_seen_loss, "val_metrics": val_seen_metrics}
            if surrogate_enabled and surrogate_rows:
                surrogate_start = time.perf_counter()
                surrogate_result = _run_latent_surrogate_validation(
                    rows=surrogate_rows,
                    writer=writer,
                    tb_step=epoch + 1,
                    tag=surrogate_tag,
                    model=eval_model,
                    encoder_references=surrogate_encoder_references,
                    device=device,
                )
                surrogate_result["validation_wall_time_s"] = float(time.perf_counter() - surrogate_start)
                writer.add_scalar(
                    f"{surrogate_tag}/validation_wall_time_s",
                    surrogate_result["validation_wall_time_s"],
                    epoch + 1,
                )
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
                        writer.add_scalar(f"{combined_tag}/{name}", combined_value, epoch + 1)
                if combined_metrics:
                    split_results[combined_tag] = {
                        "loss_total": None,
                        "val_metrics": combined_metrics,
                    }
                    writer.flush()
            summary_metrics = combined_metrics or surrogate_metrics or seen_metrics or val_metrics
            selection_value = float(summary_metrics.get(AGGREGATE_VALIDATION_ERROR_KEY, val_loss))
            ckpt_path = run_models_dir / f"{run_name}_epoch{epoch + 1:04d}.pt"
            state_source = model._orig_mod if hasattr(model, "_orig_mod") else model
            payload = {
                "model_state": state_source.state_dict(),
                "config": asdict(config),
                "run_name": run_name,
                "method": "latent_rnn",
                "latent_time_scale": latent_time_scale,
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "best_metric_name": AGGREGATE_VALIDATION_ERROR_KEY,
                "best_metric_value": selection_value if np.isfinite(selection_value) else None,
                "split_results": split_results,
            }
            torch.save(payload, ckpt_path)
            shutil.copyfile(ckpt_path, latest_path)
            if np.isfinite(selection_value) and selection_value < best_val:
                best_val = selection_value
                shutil.copyfile(ckpt_path, run_models_dir / f"{run_name}_best_val.pt")
            print(
                f"[latent_rnn][val_unseen] epoch {epoch + 1}: "
                f"loss={val_loss:.4e}, state={val_metrics.get('loss_state', float('nan')):.4e}, "
                f"selection={selection_value:.4e}"
            )

    final_val_trajs = [*val_trajs, *val_seen_trajs]
    if final_rollout_all_validation and final_val_trajs:
        print("Final latent_rnn validation rollout (all trajectories) started.")
        final_start = time.perf_counter()
        eval_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        eval_model.eval()
        metrics_sum: dict[str, float] = {}
        metrics_count: dict[str, int] = {}
        metrics_list: list[dict[str, float]] = []
        plot_ur_values: list[float] = []
        plot_metrics_list: list[dict[str, float]] = []
        metric_trajs = list(final_val_trajs)
        plot_trajs = list(final_val_trajs)
        plot_trajs.sort(key=lambda traj: round(float(np.asarray(traj["ur"]).reshape(-1)[0]), 6))
        for traj in metric_trajs:
            result = _latent_rollout_validation_case(
                model=eval_model,
                traj=traj,
                encoder_length=encoder_length,
                include_acceleration=include_acceleration,
                mass_source=mass_source,
                input_scaling_mode=input_scaling_mode,
                ur_scale=ur_scale,
                device=device,
            )
            filtered_metrics = {
                name: float(value)
                for name, value in dict(result.get("metrics", {})).items()
                if name != ROLLOUT_DIVERGED_KEY and np.isfinite(float(value))
            }
            if not filtered_metrics:
                continue
            metrics_list.append(filtered_metrics)
            for name, value in filtered_metrics.items():
                metrics_sum[name] = metrics_sum.get(name, 0.0) + float(value)
                metrics_count[name] = metrics_count.get(name, 0) + 1
        for idx, traj in enumerate(plot_trajs, start=1):
            plot_metrics = _log_latent_rollout_validation(
                writer=writer,
                epoch=max(0, epochs - 1),
                model=eval_model,
                traj=traj,
                encoder_length=encoder_length,
                include_acceleration=include_acceleration,
                mass_source=mass_source,
                input_scaling_mode=input_scaling_mode,
                ur_scale=ur_scale,
                device=device,
                tag_prefix="final_val/rollout",
                metric_prefix="final_val",
                step=idx,
                log_metrics=False,
                log_plots=True,
                log_force_plot=True,
                log_spectra=True,
                title_suffix=f" [final {idx}/{len(plot_trajs)}]",
            )
            filtered_plot_metrics = {
                name: float(value)
                for name, value in plot_metrics.items()
                if name != ROLLOUT_DIVERGED_KEY and np.isfinite(float(value))
            }
            if filtered_plot_metrics:
                plot_ur_values.append(float(np.asarray(traj["ur"], dtype=float).reshape(-1)[0]))
                plot_metrics_list.append(filtered_plot_metrics)
        if metrics_list:
            avg_metrics = {
                name: metrics_sum[name] / float(metrics_count[name])
                for name in metrics_sum
                if metrics_count.get(name, 0) > 0
            }
            summary_lines = [
                "Final rollout over all validation trajectories:",
                (
                    f"val_unseen={len(val_trajs)}, val_seen={len(val_seen_trajs)}, "
                    f"total={len(metric_trajs)}, metric_count={len(metrics_list)}, plot_count={len(plot_metrics_list)}"
                ),
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
        writer.add_scalar("final_val/rollout_wall_time_s", float(time.perf_counter() - final_start), epochs)
        print(f"Final latent_rnn validation rollout finished in {time.perf_counter() - final_start:.2f}s.")

    state_source = model._orig_mod if hasattr(model, "_orig_mod") else model
    final_path = Path("models") / f"{run_name}.pt"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": state_source.state_dict(),
            "config": asdict(config),
            "run_name": run_name,
            "method": "latent_rnn",
            "latent_time_scale": latent_time_scale,
        },
        final_path,
    )
    writer.close()
    print(f"Saved final latent_rnn model to {final_path}")
