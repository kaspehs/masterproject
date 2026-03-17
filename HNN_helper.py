import importlib
import math
import warnings
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

try:
    from scipy.signal import savgol_filter, welch
except ImportError:
    savgol_filter = None
    welch = None

from architectures import FourierFeatures, ODEPirateNet

FORCE_MAPPING_NRMSE_KEY = "Force mapping NRMSE"
FORCE_MAPPING_NRMSE_COEFF_KEY = "Force mapping NRMSE (coeff)"
DOMINANT_FREQ_REL_ERROR_KEY = "Dominant frequency relative error"
DISP_SPECTRAL_REL_ERROR_KEY = "Displacement spectral relative error"
DISP_STD_REL_ERROR_KEY = "Displacement std relative error"
# Backward-compat: VPINN still imports this legacy key name.
MEAN_DISP_AMP_REL_ERROR_KEY = DISP_STD_REL_ERROR_KEY
DISP_SPECTRAL_SHAPE_ERROR_KEY = "Disp spectral shape error"
FORCE_SPECTRAL_SHAPE_ERROR_KEY = "Force spectral shape error"
FORCE_SPECTRAL_REL_ERROR_KEY = "Force spectral relative error"
ROLLOUT_DIVERGED_KEY = "Rollout diverged"
ROLLOUT_DIVERGED_COUNT_KEY = "Rollout diverged count"

SPECTRAL_ERROR_FMIN_HZ = 0.1
SPECTRAL_ERROR_FMAX_HZ = 5.0
SPECTRAL_ERROR_NPERSEG = 1024
ROLLOUT_DIVERGENCE_ABS_Y_NORM_LIMIT = 1e3
ROLLOUT_DIVERGENCE_REL_Y_NORM_MULTIPLIER = 20.0

SIGNED_PHASE_CMAP = LinearSegmentedColormap.from_list(
    "signed_phase_map",
    [
        (0.00, "#2b6cb0"),
        (0.35, "#7fb3d5"),
        (0.50, "#b8b8b8"),
        (0.65, "#f4a3a3"),
        (1.00, "#b22222"),
    ],
)

@dataclass
class DataConfig:
    file: str = "data.npz"
    steadystate: bool = False
    steadystate_time_threshold: float = 10.0
    # Cut away the first N seconds from each time series (relative to the series start).
    # Applied during both training and validation loading.
    cut_start_seconds: float = 0.0
    # Optional overrides for training/validation splits (fallback to cut_start_seconds when unset).
    cut_start_seconds_train: float | None = None
    cut_start_seconds_val: float | None = None
    reduce_time: bool = False
    reduction_factor: int = 1
    middle_time_plot: list[float] = field(default_factory=lambda: [15.0, 17.0])
    use_generated_train_series: bool = False
    train_series_dir: str = "Data_Gen/generated_series"

@dataclass
class ModelConfig:
    rho: float = 1000.0
    D: float = 0.1
    structural_mass: float = 16.79
    Ca: float = 1.0
    k: float = 1218.0
    U: float = 0.65
    damping_c: float = 1e-4
    max_damping_ratio: float = 0.2
    include_physical_drag: bool = False
    force_output: str = "force"  # "force" or "coefficient"
    learn_hamiltonian: bool = False
    discover_damping: bool = False
    use_pirate_force: bool = False
    pirate_force_kwargs: dict[str, Any] = field(default_factory=dict)
    use_fourier_features: bool = False
    fourier_features: int = 64
    fourier_sigma: float = 1.0
    use_feature_engineering: bool = False
    use_reduced_velocity: bool = True
    use_stochastic_process_noise: bool = True
    sigma_min: float = 1e-6
    corr_init_mode: str = "zero"  # "zero" | "tiny" | "standard"
    corr_init_tiny_std: float = 1e-4
    q_scale: float | None = None
    p_scale: float | None = None
    ur_scale: float | None = None

def _default_residual_kwargs() -> dict[str, Any]:
    return {"hidden": 128, "layers": 2, "activation": "gelu"}


def _default_mlp_kwargs() -> dict[str, Any]:
    return {"hidden": 100, "layers": 2, "activation": "gelu"}


@dataclass
class ArchitectureConfig:
    force_net_type: str = "residual"
    residual_kwargs: dict[str, Any] = field(default_factory=_default_residual_kwargs)
    mlp_kwargs: dict[str, Any] = field(default_factory=_default_mlp_kwargs)
    pirate_force_kwargs: dict[str, Any] = field(default_factory=dict)

@dataclass
class SmoothingConfig:
    use_savgol_smoothing: bool = True
    window_length: int = 15
    polyorder: int = 4

@dataclass
class SchedulerConfig:
    max_lr: float = 5e-4
    decay_rate: float = 0.9
    warmup_steps: int = 1000
    # If set, overrides warmup_steps with a fraction of total epochs (e.g., 0.05 for 5%).
    warmup_fraction: float | None = None
    decay_steps: int = 1000
    min_lr: float = 1e-5
    scheduler_type: str = "cosine"  # or "exponential"

@dataclass
class TrainingConfig:
    max_grad_norm: float = 1e4
    batch_size: int = 32
    epochs: int = 2000

@dataclass
class OptimConfig:
    lr: float = 1e-3
    optimizer: str = "adam"
    weight_decay: float = 0.0
    use_lr_scheduler: bool = False
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

@dataclass
class LossConfig:
    mean_reg: float = 0.0
    mean_reg_norm: str = "l1"  # "l1" or "l2"
    sigma_reg: float = 1e-2
    sigma_reg_norm: str = "l2"  # "l1" or "l2"
    equalize_residual_over_ur_bins: bool = False
    equalize_rollout_over_ur_bins: bool = False
    ur_bin_size: float = 1e-6
    normalize_residual_by_ur_bin_std: bool = False
    normalize_rollout_by_ur_bin_std: bool = False
    ur_bin_scale_eps: float = 1e-6
    rollout_det_weight: float = 0.0
    rollout_det_steps: int = 0
    rollout_loss_mode: str = "deterministic"  # "deterministic" | "stochastic_nll" | "stochastic_mse"
    rollout_stochastic_samples: int = 1
    rollout_det_steps_final: int = 0  # <=0 keeps rollout_det_steps fixed
    rollout_det_steps_warmup_epochs: int = 0
    rollout_det_batch_size: int = 0  # <=0 -> fallback to training.batch_size
    force_reg_on_coeff: bool = False
    use_gradnorm: bool = False
    gradnorm_alpha: float = 0.9
    gradnorm_eps: float = 1e-8
    gradnorm_min_weight: float = 0.1
    gradnorm_max_weight: float = 10.0
    use_force_data_loss: bool = False
    force_data_weight: float = 1.0
    symmetry_weight: float = 0.0
    symmetry_norm: str = "l2"  # "l2" (default) or "l1"

@dataclass
class RuntimeConfig:
    device: str = "auto"
    num_workers: int = 0

@dataclass
class PrecisionConfig:
    use_tf32: bool = False
    use_amp: bool = False
    amp_dtype: str = "bf16"  # "bf16" (recommended on A100) or "fp16"

@dataclass
class CompileConfig:
    use_compile: bool = False
    compile_mode: str = "default"  # "default" | "reduce-overhead" | "max-autotune"

@dataclass
class MonitoringConfig:
    rollout_every_epochs: int = 50
    validate_every_epochs: int = 10
    rollout_max_trajectories: int = 1
    log_every_epochs: int = 1
    print_every_epochs: int = 1
    log_component_grad_norms: bool = False
    log_extra_validation_metrics: bool = False
    cycle_validation_rollout: bool = False
    fixed_validation_sampling: bool = False
    validation_sampling_seed: int = 1
    validation_samples_per_ur: int = 1
    rollout_use_excluded_ur: bool = False
    rollout_target_ur_tol: float = 1e-6
    final_rollout_all_validation: bool = False
    async_validation: bool = False
    async_validation_device: str = "cpu"
    async_validation_num_workers: int = 0
    async_validation_num_threads: int = 4
    async_validation_max_concurrent: int = 1
    async_validation_do_losses: bool = True
    async_validation_do_rollout: bool = True

@dataclass
class LoggingConfig:
    run_dir_root: str = "HNNruns"
    run_name: str | None = None
    append_timestamp: bool = True

@dataclass
class Config:
    method: str = "hnn"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    compile: CompileConfig = field(default_factory=CompileConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    hnn: dict[str, Any] = field(default_factory=dict)
    pinn: dict[str, Any] = field(default_factory=dict)
    vpinn: dict[str, Any] = field(default_factory=dict)


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data


_PHNN_HISTORY_MODEL_KEYS = (
    "use_history_tcn",
    "history_window",
    "history_tcn_channels",
    "history_tcn_layers",
    "history_tcn_kernel_size",
    "history_tcn_dropout",
    "use_history_correction",
    "sigma_from_history",
)


def parse_config(raw: dict[str, Any]) -> Config:
    method = raw.get("method", "hnn")
    data_cfg = raw.get("data", {}) or {}
    model_cfg = raw.get("model", {}) or {}
    architecture_cfg = dict(raw.get("architecture", {}) or {})
    smoothing_cfg = raw.get("smoothing", {}) or {}
    training_cfg = dict(raw.get("training", {}) or {})
    optim_cfg = dict(raw.get("optim", {}) or {})
    loss_cfg = dict(raw.get("loss", {}) or {})
    runtime_cfg = dict(raw.get("runtime", {}) or {})
    precision_cfg = dict(raw.get("precision", {}) or {})
    compile_cfg = dict(raw.get("compile", {}) or {})
    monitoring_cfg = dict(raw.get("monitoring", raw.get("train_logging", {})) or {})
    logging_cfg = raw.get("logging", {}) or {}
    hnn_block = dict(raw.get("hnn", {}) or {})
    pinn_block = dict(raw.get("pinn", {}) or {})
    vpinn_block = dict(raw.get("vpinn", {}) or {})

    method_key = str(method).strip().lower()
    if method_key in {"hnn", "phnn"}:
        forbidden = sorted(key for key in _PHNN_HISTORY_MODEL_KEYS if key in model_cfg)
        if forbidden:
            joined = ", ".join(forbidden)
            raise ValueError(
                "PHNN is now markovian-only. Remove these history-dependent model keys: "
                f"{joined}"
            )

    legacy_residual: dict[str, Any] = {}
    if "residual_hidden" in architecture_cfg:
        legacy_residual["hidden"] = architecture_cfg.pop("residual_hidden")
    if "residual_layers" in architecture_cfg:
        legacy_residual["layers"] = architecture_cfg.pop("residual_layers")
    if "residual_activation" in architecture_cfg:
        legacy_residual["activation"] = architecture_cfg.pop("residual_activation")
    if legacy_residual or "residual_kwargs" in architecture_cfg:
        residual_kwargs = dict(architecture_cfg.get("residual_kwargs", {}) or {})
        residual_kwargs.update(legacy_residual)
        architecture_cfg["residual_kwargs"] = residual_kwargs

    legacy_mlp: dict[str, Any] = {}
    if "mlp_hidden" in architecture_cfg:
        legacy_mlp["hidden"] = architecture_cfg.pop("mlp_hidden")
    if "mlp_layers" in architecture_cfg:
        legacy_mlp["layers"] = architecture_cfg.pop("mlp_layers")
    if "mlp_activation" in architecture_cfg:
        legacy_mlp["activation"] = architecture_cfg.pop("mlp_activation")
    if legacy_mlp or "mlp_kwargs" in architecture_cfg:
        mlp_kwargs = dict(architecture_cfg.get("mlp_kwargs", {}) or {})
        mlp_kwargs.update(legacy_mlp)
        architecture_cfg["mlp_kwargs"] = mlp_kwargs

    pirate_overrides: dict[str, Any] = {}
    if "pirate_activation" in architecture_cfg:
        pirate_overrides["activation"] = architecture_cfg.pop("pirate_activation")
    if "activation" in architecture_cfg:
        pirate_overrides["activation"] = architecture_cfg.pop("activation")
    if "pirate_rwf_mu" in architecture_cfg:
        pirate_overrides["rwf_mu"] = architecture_cfg.pop("pirate_rwf_mu")
    if "pirate_rwf_sigma" in architecture_cfg:
        pirate_overrides["rwf_sigma"] = architecture_cfg.pop("pirate_rwf_sigma")
    if "pirate_depth" in architecture_cfg:
        pirate_overrides["depth"] = architecture_cfg.pop("pirate_depth")
    if "pirate_layers" in architecture_cfg:
        pirate_overrides["depth"] = architecture_cfg.pop("pirate_layers")
    pirate_kwargs = dict(architecture_cfg.get("pirate_force_kwargs", {}) or {})
    pirate_kwargs.update(pirate_overrides)
    architecture_cfg["pirate_force_kwargs"] = pirate_kwargs

    # Backwards compatible mapping: allow legacy keys to live under training:
    legacy_training = dict(training_cfg)
    optim_keys = {"lr", "optimizer", "weight_decay", "use_lr_scheduler", "scheduler"}
    loss_keys = {
        "mean_reg",
        "mean_reg_norm",
        "sigma_reg",
        "force_reg",
        "sigma_reg_norm",
        "equalize_residual_over_ur_bins",
        "equalize_rollout_over_ur_bins",
        "ur_bin_size",
        "normalize_residual_by_ur_bin_std",
        "normalize_rollout_by_ur_bin_std",
        "ur_bin_scale_eps",
        "rollout_det_weight",
        "rollout_det_steps",
        "rollout_loss_mode",
        "rollout_stochastic_samples",
        "rollout_det_steps_final",
        "rollout_det_steps_warmup_epochs",
        "rollout_det_batch_size",
        "force_reg_on_coeff",
        "use_gradnorm",
        "gradnorm_alpha",
        "gradnorm_eps",
        "gradnorm_min_weight",
        "gradnorm_max_weight",
        "use_force_data_loss",
        "force_data_weight",
        "symmetry_weight",
        "symmetry_norm",
    }
    runtime_keys = {"device", "num_workers"}
    precision_keys = {"use_tf32", "use_amp", "amp_dtype"}
    compile_keys = {"use_compile", "compile_mode"}
    monitoring_keys = {
        "rollout_every_epoch",
        "rollout_every_epochs",
        "validate_every_epochs",
        "rollout_max_trajectories",
        "log_every_epochs",
        "print_every_epochs",
        "log_component_grad_norms",
        "log_extra_validation_metrics",
        "fixed_validation_sampling",
        "validation_sampling_seed",
        "validation_samples_per_ur",
        "rollout_use_excluded_ur",
        "rollout_target_ur_tol",
    }

    for key, value in legacy_training.items():
        if key in optim_keys and key not in optim_cfg:
            optim_cfg[key] = value
        elif key in loss_keys and key not in loss_cfg:
            loss_cfg[key] = value
        elif key in runtime_keys and key not in runtime_cfg:
            runtime_cfg[key] = value
        elif key in precision_keys and key not in precision_cfg:
            precision_cfg[key] = value
        elif key in compile_keys and key not in compile_cfg:
            compile_cfg[key] = value
        elif key in monitoring_keys and key not in monitoring_cfg:
            monitoring_cfg[key] = value

    training_cfg = {k: v for k, v in training_cfg.items() if k in {"batch_size", "max_grad_norm", "epochs"}}

    data = DataConfig(**data_cfg)
    model = ModelConfig(**model_cfg)
    architecture = ArchitectureConfig(**architecture_cfg)
    smoothing = SmoothingConfig(**smoothing_cfg)
    training = TrainingConfig(**training_cfg)

    scheduler_dict = optim_cfg.get("scheduler", {}) or {}
    scheduler = SchedulerConfig(**scheduler_dict)
    optim_fields = {k: v for k, v in optim_cfg.items() if k != "scheduler"}
    optim = OptimConfig(**optim_fields, scheduler=scheduler)
    if "sigma_reg" not in loss_cfg and "force_reg" in loss_cfg:
        loss_cfg["sigma_reg"] = loss_cfg["force_reg"]
    loss_cfg.pop("force_reg", None)
    loss = LossConfig(**loss_cfg)
    runtime = RuntimeConfig(**runtime_cfg)
    precision = PrecisionConfig(**precision_cfg)
    compile_cfg_obj = CompileConfig(**compile_cfg)
    # Back-compat key: `rollout_every_epoch` -> `rollout_every_epochs`
    if "rollout_every_epochs" not in monitoring_cfg and "rollout_every_epoch" in monitoring_cfg:
        monitoring_cfg["rollout_every_epochs"] = monitoring_cfg.pop("rollout_every_epoch")
    # Removed monitoring toggles: keep old configs loadable by ignoring stale keys.
    monitoring_cfg.pop("rollout_include_disp_nrmse", None)
    monitoring_cfg.pop("rollout_include_force_nrmse", None)
    monitoring = MonitoringConfig(**monitoring_cfg)
    logging = LoggingConfig(**logging_cfg)
    return Config(
        method=str(method),
        data=data,
        model=model,
        architecture=architecture,
        smoothing=smoothing,
        training=training,
        optim=optim,
        loss=loss,
        runtime=runtime,
        precision=precision,
        compile=compile_cfg_obj,
        monitoring=monitoring,
        logging=logging,
        hnn=hnn_block,
        pinn=pinn_block,
        vpinn=vpinn_block,
    )


def _ur_bin_id(value: float, ur_bin_size: float) -> int:
    return int(np.rint(float(value) / float(ur_bin_size)))


def build_ur_bin_state_scale_info_from_dataset(
    dataset: Any,
    *,
    ur_tensor_index: int,
    state_tensor_indices: Sequence[int],
    ur_bin_size: float,
    eps: float = 1e-6,
) -> dict[str, Any]:
    cache_key = (
        "ur_bin_state_scales:"
        f"{int(ur_tensor_index)}:"
        f"{','.join(str(int(idx)) for idx in state_tensor_indices)}:"
        f"{float(ur_bin_size):.12g}:"
        f"{float(eps):.12g}"
    )
    cache = getattr(dataset, "_codex_cache", None)
    if isinstance(cache, dict) and cache_key in cache:
        return dict(cache[cache_key])

    stats_by_bin: dict[int, dict[str, Any]] = {}
    global_count = 0
    global_sum = np.zeros(2, dtype=np.float64)
    global_sumsq = np.zeros(2, dtype=np.float64)

    def _accumulate(ds: Any) -> None:
        nonlocal global_count, global_sum, global_sumsq
        if isinstance(ds, ConcatDataset):
            for subdataset in ds.datasets:
                _accumulate(subdataset)
            return
        if not isinstance(ds, TensorDataset):
            raise TypeError(f"Unsupported dataset type for U_r bin scales: {type(ds)!r}")
        ur_tensor = ds.tensors[int(ur_tensor_index)]
        ur_vals = ur_tensor.reshape(ur_tensor.shape[0], -1)[:, 0].detach().cpu().numpy()
        state_arrays = [
            ds.tensors[int(idx)].reshape(ds.tensors[int(idx)].shape[0], -1)[:, :2].detach().cpu().numpy()
            for idx in state_tensor_indices
        ]
        repeated_ur = np.repeat(ur_vals, len(state_arrays))
        stacked_states = np.concatenate(state_arrays, axis=0)
        for ur_val, state_vec in zip(repeated_ur, stacked_states):
            key = _ur_bin_id(float(ur_val), ur_bin_size)
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

    def _finalize(count: int, sum_vec: np.ndarray, sumsq_vec: np.ndarray) -> np.ndarray:
        denom = float(max(int(count), 1))
        mean = sum_vec / denom
        var = np.maximum(sumsq_vec / denom - mean * mean, 0.0)
        return np.sqrt(var)

    _accumulate(dataset)
    global_scale = _finalize(global_count, global_sum, global_sumsq)
    global_scale = np.maximum(global_scale, float(eps))
    by_bin: dict[str, list[float]] = {}
    for key, stat in stats_by_bin.items():
        scale = _finalize(int(stat["count"]), stat["sum"], stat["sumsq"])
        if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
            scale = global_scale.copy()
        scale = np.maximum(scale, float(eps))
        by_bin[str(int(key))] = [float(scale[0]), float(scale[1])]
    out = {
        "global": [float(global_scale[0]), float(global_scale[1])],
        "by_bin": by_bin,
    }
    if cache is None or not isinstance(cache, dict):
        cache = {}
        setattr(dataset, "_codex_cache", cache)
    cache[cache_key] = dict(out)
    return out


def lookup_ur_bin_state_scale_tensor(
    ur_values: torch.Tensor | np.ndarray | float | None,
    *,
    scale_info: dict[str, Any] | None,
    ur_bin_size: float,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not scale_info:
        return torch.ones((int(batch_size), 2), device=device, dtype=dtype)
    global_scale = np.asarray(scale_info.get("global", [1.0, 1.0]), dtype=np.float64).reshape(2)
    by_bin = dict(scale_info.get("by_bin", {}) or {})
    if ur_values is None:
        scale_arr = np.repeat(global_scale.reshape(1, 2), int(batch_size), axis=0)
        return torch.as_tensor(scale_arr, device=device, dtype=dtype)
    if torch.is_tensor(ur_values):
        ur_flat = ur_values.reshape(-1).detach().cpu().numpy()
    else:
        ur_flat = np.asarray(ur_values, dtype=np.float64).reshape(-1)
    if ur_flat.size == 1 and int(batch_size) > 1:
        ur_flat = np.repeat(ur_flat, int(batch_size))
    scales: list[list[float]] = []
    for ur_val in ur_flat[: int(batch_size)]:
        key = str(_ur_bin_id(float(ur_val), ur_bin_size))
        scale = by_bin.get(key, global_scale.tolist())
        scales.append([float(scale[0]), float(scale[1])])
    if len(scales) < int(batch_size):
        scales.extend([global_scale.tolist()] * (int(batch_size) - len(scales)))
    return torch.as_tensor(scales, device=device, dtype=dtype)


def scaled_residual_loss_per_sample(
    model: "PHVIV",
    zi: torch.Tensor,
    zin: torch.Tensor,
    reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
    *,
    history_context: torch.Tensor | None = None,
    ur_bin_state_scale_info: dict[str, Any] | None = None,
    ur_bin_size: float = 1e-6,
) -> torch.Tensor:
    batch_size = int(zi.shape[0])
    if model.use_stochastic_process_noise:
        innovation, _sigma, var = model._momentum_transition_stats_srk4(
            zi,
            zin,
            reduced_velocity=reduced_velocity,
            history_context=history_context,
        )
        if ur_bin_state_scale_info is not None:
            state_scale = lookup_ur_bin_state_scale_tensor(
                reduced_velocity,
                scale_info=ur_bin_state_scale_info,
                ur_bin_size=ur_bin_size,
                batch_size=batch_size,
                device=innovation.device,
                dtype=innovation.dtype,
            )
            p_scale = torch.clamp(state_scale[..., 1:2], min=1e-12)
            innovation = innovation / p_scale
            var = var / torch.clamp(p_scale * p_scale, min=1e-12)
        nll = 0.5 * (innovation * innovation / var + torch.log(var))
        return nll.squeeze(-1)

    drift_rate, z_mid = model._srk4_drift_rate(
        zi,
        zin,
        reduced_velocity=reduced_velocity,
        history_context=history_context,
    )
    res = (zin - zi) - float(model.dt) * drift_rate
    res_scaled = res / model.res_scale.to(device=res.device, dtype=res.dtype)
    if model.force_output == "coefficient":
        f0 = model._force_scale_from_reduced_velocity(
            reduced_velocity,
            like=res_scaled[..., 1:2],
            state=z_mid,
        )
        res_scaled = res_scaled.clone()
        res_scaled[..., 1:2] = res_scaled[..., 1:2] / torch.clamp(f0, min=1e-12)
    if ur_bin_state_scale_info is not None:
        state_scale = lookup_ur_bin_state_scale_tensor(
            reduced_velocity,
            scale_info=ur_bin_state_scale_info,
            ur_bin_size=ur_bin_size,
            batch_size=batch_size,
            device=res_scaled.device,
            dtype=res_scaled.dtype,
        )
        res_scaled = res_scaled / torch.clamp(state_scale, min=1e-12)
    return torch.sum(res_scaled * res_scaled, dim=1)


def log_training_metrics(
    writer: SummaryWriter,
    epoch: int,
    metrics: dict[str, float],
) -> str:
    log_parts = [f"Epoch {epoch}"]
    for name, value in metrics.items():
        writer.add_scalar(f"train/{name}", value, epoch)
        log_parts.append(f"{name}={value:.4e}")
    return ", ".join(log_parts)


def log_validation_epoch(
    writer: SummaryWriter,
    epoch: int,
    model: "PHVIV",
    y_data_t: torch.Tensor,
    val_vel: torch.Tensor,
    reduced_velocity: torch.Tensor,
    m_eff: float,
    dt: float,
    t: np.ndarray,
    y_true_norm: np.ndarray,
    y_data_raw: np.ndarray,
    force_data: np.ndarray | None,
    D: float,
    k: float,
    device: torch.device,
    middle_time_plot: list[float] | tuple[float, float],
    hamiltonian_data: np.ndarray | None,
    *,
    log_extra_metrics: bool = False,
    log_metrics: bool = True,
    rollout_stochastic: bool = False,
    rollout_noise_scale: float = 1.0,
    rollout_seed: int | None = None,
    tag_prefix: str = "val/rollout",
    step: int | None = None,
    title_suffix: str = "",
) -> dict[str, float]:
    rollout = rollout_model(
        model,
        y_data_t,
        val_vel,
        reduced_velocity,
        m_eff,
        dt,
        t,
        D,
        k,
        device,
        stochastic=rollout_stochastic,
        rollout_seed=rollout_seed,
        noise_scale=rollout_noise_scale,
    )
    metrics = compute_validation_metrics(
        model=model,
        y_data_t=y_data_t,
        val_vel=val_vel,
        reduced_velocity=reduced_velocity,
        m_eff=m_eff,
        dt=dt,
        t=t,
        y_data_raw=y_data_raw,
        force_data=force_data,
        D=D,
        k=k,
        device=device,
        log_extra_metrics=log_extra_metrics,
        rollout=rollout,
        rollout_stochastic=rollout_stochastic,
        rollout_noise_scale=rollout_noise_scale,
        rollout_seed=rollout_seed,
    )
    if torch.is_tensor(reduced_velocity):
        reduced_velocity_scalar = float(reduced_velocity.reshape(-1)[0].detach().cpu())
    else:
        reduced_velocity_scalar = float(np.asarray(reduced_velocity).reshape(-1)[0])

    if log_metrics:
        for name, value in metrics.items():
            if name == ROLLOUT_DIVERGED_KEY:
                continue
            writer.add_scalar(f"val/{name}", value, epoch)
    zoom_mask = create_zoom_mask(t)
    middle_mask = create_window_mask(t, middle_time_plot)
    log_displacement_plots(
        writer,
        epoch,
        t,
        y_true_norm,
        rollout["y_norm"],
        rollout["p_norm"],
        zoom_mask,
        middle_mask,
        middle_time_plot,
        reduced_velocity=reduced_velocity_scalar,
        tag_prefix=tag_prefix,
        step=step,
        title_suffix=title_suffix,
    )
    force_coeff_pred = np.asarray(rollout["force_coeff_total"], dtype=float).reshape(-1)
    force_coeff_delta = np.asarray(rollout["force_coeff_delta"], dtype=float).reshape(-1)
    force_coeff_sigma = np.asarray(rollout["force_coeff_sigma"], dtype=float).reshape(-1)
    force_true = np.asarray(force_data, dtype=float).reshape(-1) if force_data is not None else np.asarray([], dtype=float)
    plot_len = int(min(force_coeff_pred.size, force_coeff_delta.size, force_coeff_sigma.size, force_true.size, len(t)))
    if plot_len <= 0:
        return metrics
    t_force_plot = np.asarray(t, dtype=float)[:plot_len]
    zoom_mask_force = create_zoom_mask(t_force_plot)
    with torch.no_grad():
        z_true_plot = torch.stack((y_data_t[:plot_len], val_vel[:plot_len] * float(m_eff)), dim=1).to(
            device=device, non_blocking=(device.type == "cuda")
        )
        rv_plot = reduced_velocity
        if not torch.is_tensor(rv_plot):
            rv_plot = torch.as_tensor(rv_plot, dtype=z_true_plot.dtype)
        rv_plot = rv_plot.to(device=z_true_plot.device, dtype=z_true_plot.dtype)
        f0_plot_t = model._force_scale_from_reduced_velocity(
            rv_plot[:plot_len] if rv_plot.ndim > 0 else rv_plot,
            like=z_true_plot[..., :1],
            state=z_true_plot,
        ).squeeze(-1).detach().cpu().numpy()
    f0_plot_t = np.asarray(f0_plot_t, dtype=float).reshape(-1)
    f0_plot_t = np.clip(np.nan_to_num(f0_plot_t, nan=1.0, posinf=1.0, neginf=1.0), 1e-12, None)
    force_coeff_true = force_true[:plot_len] / f0_plot_t[:plot_len]
    log_force_plots_with_components(
        writer,
        epoch,
        t_force_plot,
        force_coeff_pred[:plot_len],
        force_coeff_true,
        force_coeff_delta[:plot_len],
        force_coeff_sigma[:plot_len],
        zoom_mask_force,
        reduced_velocity=reduced_velocity_scalar,
        tag_prefix=tag_prefix,
        step=step,
        title_suffix=title_suffix,
    )
    return metrics


def compute_validation_metrics(
    *,
    model: "PHVIV",
    y_data_t: torch.Tensor,
    val_vel: torch.Tensor,
    reduced_velocity: torch.Tensor,
    m_eff: float,
    dt: float,
    t: np.ndarray,
    y_data_raw: np.ndarray,
    force_data: np.ndarray | None,
    D: float,
    k: float,
    device: torch.device,
    log_extra_metrics: bool = False,
    rollout: dict[str, np.ndarray] | None = None,
    rollout_stochastic: bool = False,
    rollout_noise_scale: float = 1.0,
    rollout_seed: int | None = None,
) -> dict[str, float]:
    if rollout is None:
        rollout = rollout_model(
            model,
            y_data_t,
            val_vel,
            reduced_velocity,
            m_eff,
            dt,
            t,
            D,
            k,
            device,
            stochastic=rollout_stochastic,
            rollout_seed=rollout_seed,
            noise_scale=rollout_noise_scale,
        )
    metrics: dict[str, float] = {}

    y_pred_norm = np.asarray(rollout["y_norm"], dtype=float).reshape(-1)
    p_pred_norm = np.asarray(rollout["p_norm"], dtype=float).reshape(-1)
    force_total_pred_full = np.asarray(rollout["force_total"], dtype=float).reshape(-1)
    y_true = np.asarray(y_data_raw, dtype=float).reshape(-1)
    y_true_norm = y_true / float(D)
    y_pred = y_pred_norm * float(D)

    diverged = False
    if not np.all(np.isfinite(y_pred_norm)):
        diverged = True
    if not np.all(np.isfinite(p_pred_norm)):
        diverged = True
    if not np.all(np.isfinite(force_total_pred_full)):
        diverged = True
    if not diverged and y_pred_norm.size > 0 and y_true_norm.size > 0:
        pred_max = float(np.max(np.abs(y_pred_norm)))
        true_max = float(np.max(np.abs(y_true_norm)))
        abs_limit = float(ROLLOUT_DIVERGENCE_ABS_Y_NORM_LIMIT)
        rel_limit = float(ROLLOUT_DIVERGENCE_REL_Y_NORM_MULTIPLIER) * max(1e-6, true_max)
        if pred_max > max(abs_limit, rel_limit):
            diverged = True
    metrics[ROLLOUT_DIVERGED_KEY] = 1.0 if diverged else 0.0

    min_len_y = min(y_pred.shape[0], y_true.shape[0])
    if min_len_y > 1:
        y_pred_aligned = y_pred[:min_len_y]
        y_true_aligned = y_true[:min_len_y]
        if y_true_aligned.size > 0 and y_pred_aligned.size > 0:
            disp_spectral_rel = spectral_l1_relative_error(y_true_aligned, y_pred_aligned, dt)
            if np.isfinite(disp_spectral_rel):
                metrics[DISP_SPECTRAL_REL_ERROR_KEY] = float(disp_spectral_rel)

        true_dom = dominant_frequency(y_true_aligned, dt)
        pred_dom = dominant_frequency(y_pred_aligned, dt)
        dom_rel = relative_error(pred_dom, true_dom)
        if np.isfinite(dom_rel):
            metrics[DOMINANT_FREQ_REL_ERROR_KEY] = abs(float(dom_rel))

        true_std = float(np.std(y_true_aligned))
        pred_std = float(np.std(y_pred_aligned))
        std_rel = relative_error(pred_std, true_std)
        if np.isfinite(std_rel):
            metrics[DISP_STD_REL_ERROR_KEY] = abs(float(std_rel))

    if force_data is not None:
        force_total_pred = force_total_pred_full
        force_target = np.asarray(force_data, dtype=float).reshape(-1)
        min_len_force = min(force_total_pred.shape[0], force_target.shape[0])
        if min_len_force > 1:
            force_pred_aligned = force_total_pred[:min_len_force]
            force_true_aligned = force_target[:min_len_force]
            if force_true_aligned.size > 0 and force_pred_aligned.size > 0:
                force_spectral_rel = spectral_l1_relative_error(force_true_aligned, force_pred_aligned, dt)
                if np.isfinite(force_spectral_rel):
                    metrics[FORCE_SPECTRAL_REL_ERROR_KEY] = float(force_spectral_rel)
    return metrics


def compute_model_grad_norm(model: "PHVIV") -> float:
    total = None
    for p in model.parameters():
        if p.grad is not None:
            grad_sq = torch.sum(p.grad.detach() ** 2)
            total = grad_sq if total is None else total + grad_sq
    if total is None:
        return 0.0
    return float(torch.sqrt(total).detach().cpu())


class GradNormBalancer:
    """Balance multiple loss terms by equalizing their gradient norms."""

    def __init__(
        self,
        model: nn.Module,
        names: Sequence[str],
        alpha: float = 0.9,
        eps: float = 1e-8,
        min_weight: float = 0.1,
        max_weight: float = 10.0,
    ) -> None:
        if not names:
            raise ValueError("GradNormBalancer requires at least one loss name")
        self.model = model
        self.names = tuple(names)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            raise ValueError("Model must have trainable parameters for GradNormBalancer")
        self.params = params
        self.device = params[0].device
        self.g_ema = {name: torch.tensor(1.0, device=self.device) for name in self.names}
        self.weights = {name: torch.tensor(1.0, device=self.device) for name in self.names}
        self.latest_grad_norms = {name: torch.tensor(0.0, device=self.device) for name in self.names}

    def _grad_norm(self, loss: torch.Tensor) -> torch.Tensor:
        if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
            return torch.tensor(0.0, device=self.device)
        grads = torch.autograd.grad(
            loss,
            self.params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        total = None
        for g in grads:
            if g is None:
                continue
            g = g.detach()
            if not torch.isfinite(g).all():
                g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
            val = torch.sum(g * g)
            total = val if total is None else total + val
        if total is None:
            return torch.tensor(0.0, device=self.device)
        return torch.sqrt(total + self.eps)

    def update(self, losses: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        missing = [name for name in self.names if name not in losses]
        if missing:
            raise KeyError(f"GradNormBalancer missing losses for: {missing}")
        grad_norms = {name: self._grad_norm(losses[name]) for name in self.names}
        with torch.no_grad():
            for name in self.names:
                self.latest_grad_norms[name] = grad_norms[name].detach()
                self.g_ema[name] = self.alpha * self.g_ema[name] + (1.0 - self.alpha) * torch.clamp(
                    grad_norms[name], min=self.eps
                )
            inv = {name: 1.0 / torch.clamp(self.g_ema[name], min=self.eps) for name in self.names}
            total_inv = sum(inv.values())
            count = float(len(self.names))
            for name in self.names:
                weight = inv[name] * (count / total_inv)
                self.weights[name] = torch.clamp(weight, self.min_weight, self.max_weight)
            return {name: self.weights[name].detach() for name in self.names}

def _activation_factory(name: str | None, default: str = "gelu") -> type[nn.Module]:
    mapping: dict[str, type[nn.Module]] = {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
        "silu": nn.SiLU,
        "swish": nn.SiLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "identity": nn.Identity,
        "none": nn.Identity,
    }
    key = str(name).lower() if name is not None else default
    key = "silu" if key == "swish" else key
    if key not in mapping:
        raise ValueError(
            f"Unsupported activation '{name}'. "
            f"Available options: {', '.join(sorted(set(mapping.keys())))}"
        )
    return mapping[key]


class Residual(nn.Module):
    def __init__(self, dim: int, activation: str | None = None):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        act_cls = _activation_factory(activation)
        self.activation = act_cls()

    def forward(self, x):
        out = self.activation(self.fc1(x))
        out = self.fc2(out)
        return out + x  


class _CausalConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        if kernel_size < 2:
            raise ValueError("history_tcn_kernel_size must be >= 2.")
        self.crop = (int(kernel_size) - 1) * int(dilation)
        self.conv = nn.Conv1d(
            int(in_ch),
            int(out_ch),
            kernel_size=int(kernel_size),
            dilation=int(dilation),
            padding=self.crop,
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(float(dropout))
        self.skip = nn.Identity() if int(in_ch) == int(out_ch) else nn.Conv1d(int(in_ch), int(out_ch), kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        if self.crop > 0:
            y = y[..., :-self.crop]
        y = self.activation(y)
        y = self.dropout(y)
        return y + self.skip(x)


class CausalTCNEncoder(nn.Module):
    """Simple causal TCN encoder returning the final time-step embedding."""

    def __init__(
        self,
        *,
        in_channels: int,
        channels: int,
        layers: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("history_tcn_layers must be >= 1.")
        ch = int(channels)
        blocks: list[nn.Module] = []
        cur_in = int(in_channels)
        for i in range(int(layers)):
            dilation = 2 ** i
            blocks.append(
                _CausalConvBlock(
                    in_ch=cur_in,
                    out_ch=ch,
                    kernel_size=int(kernel_size),
                    dilation=int(dilation),
                    dropout=float(dropout),
                )
            )
            cur_in = ch
        self.blocks = nn.ModuleList(blocks)
        self.out = nn.Linear(ch, ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        if x.ndim != 3:
            raise ValueError("TCN input must have shape (B, L, C).")
        y = x.transpose(1, 2)
        for block in self.blocks:
            y = block(y)
        last = y[..., -1]
        return self.out(last)

def dominant_frequency(signal: np.ndarray, dt: float) -> float:
    """Return dominant frequency (Hz) of the provided signal using FFT."""
    if dt <= 0.0:
        return float("nan")
    signal = np.asarray(signal)
    if signal.size < 2:
        return float("nan")
    centered = signal - np.mean(signal)
    if np.allclose(centered, 0.0):
        return float("nan")
    fft_vals = np.fft.rfft(centered)
    freqs = np.fft.rfftfreq(centered.size, d=dt)
    if freqs.size <= 1:
        return float("nan")
    magnitudes = np.abs(fft_vals)
    magnitudes[0] = 0.0  # ignore DC component
    dominant_idx = int(np.argmax(magnitudes))
    dominant_mag = magnitudes[dominant_idx]
    if dominant_mag <= 0.0:
        return float("nan")
    return float(freqs[dominant_idx])


def mean_displacement_amplitude(signal: np.ndarray) -> float:
    """Return mean absolute centered displacement as an amplitude proxy."""
    signal = np.asarray(signal, dtype=float).reshape(-1)
    if signal.size == 0:
        return float("nan")
    centered = signal - float(np.mean(signal))
    return float(np.mean(np.abs(centered)))


def relative_error(model_value: float, true_value: float, eps: float = 1e-12) -> float:
    """Compute signed (model - true)/|true| with small epsilon safeguard."""
    if not np.isfinite(true_value) or not np.isfinite(model_value):
        return float("nan")
    denom = abs(true_value)
    if denom <= eps:
        return float("nan")
    return float((model_value - true_value) / (denom + eps))


def sample_indices_per_ur(
    ur_values: Sequence[float],
    *,
    samples_per_ur: int = 1,
    seed: int | None = None,
    decimals: int = 6,
) -> list[int]:
    """
    Pick up to `samples_per_ur` random indices per reduced-velocity bucket.
    Buckets are formed by rounding U_r to `decimals`.
    """
    samples_per_ur = max(1, int(samples_per_ur))
    buckets: dict[float, list[int]] = {}
    for idx, ur in enumerate(ur_values):
        if not np.isfinite(float(ur)):
            continue
        key = float(np.round(float(ur), int(decimals)))
        buckets.setdefault(key, []).append(int(idx))
    if not buckets:
        return []
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for key in sorted(buckets):
        candidates = buckets[key]
        if len(candidates) <= samples_per_ur:
            selected.extend(int(idx) for idx in candidates)
            continue
        picks = rng.choice(len(candidates), size=samples_per_ur, replace=False)
        selected.extend(int(candidates[int(pick)]) for pick in np.sort(picks))
    return selected


def sample_one_index_per_ur(
    ur_values: Sequence[float],
    *,
    seed: int | None = None,
    decimals: int = 6,
) -> list[int]:
    return sample_indices_per_ur(
        ur_values,
        samples_per_ur=1,
        seed=seed,
        decimals=decimals,
    )


def spectral_relative_error(
    true_signal: np.ndarray,
    model_signal: np.ndarray,
    dt: float,
    fmin_hz: float = SPECTRAL_ERROR_FMIN_HZ,
    fmax_hz: float = SPECTRAL_ERROR_FMAX_HZ,
    nperseg: int = SPECTRAL_ERROR_NPERSEG,
    eps: float = 1e-12,
) -> float:
    """
    Compute a normalized PSD-shape error in [0, 1] using Welch spectra.
    Uses total-variation distance between band-limited, area-normalized PSDs.
    """
    if dt <= 0.0:
        return float("nan")
    true_signal = np.asarray(true_signal, dtype=float).reshape(-1)
    model_signal = np.asarray(model_signal, dtype=float).reshape(-1)
    length = min(true_signal.size, model_signal.size)
    if length < 8:
        return float("nan")
    true_trim = true_signal[-length:]
    model_trim = model_signal[-length:]
    true_proc = true_trim - np.mean(true_trim)
    model_proc = model_trim - np.mean(model_trim)

    if welch is not None:
        fs = 1.0 / float(dt)
        seg = int(max(8, min(int(nperseg), length)))
        ov = seg // 2
        freqs, true_psd = welch(
            true_proc,
            fs=fs,
            window="hann",
            nperseg=seg,
            noverlap=ov,
            detrend=False,
            scaling="density",
        )
        _, model_psd = welch(
            model_proc,
            fs=fs,
            window="hann",
            nperseg=seg,
            noverlap=ov,
            detrend=False,
            scaling="density",
        )
    else:
        # Fallback path when scipy is unavailable.
        freqs = np.fft.rfftfreq(length, d=float(dt))
        true_psd = np.abs(np.fft.rfft(true_proc)) ** 2
        model_psd = np.abs(np.fft.rfft(model_proc)) ** 2

    if freqs.size == 0:
        return float("nan")

    band = np.isfinite(freqs)
    band = band & (freqs >= float(fmin_hz))
    if np.isfinite(float(fmax_hz)) and float(fmax_hz) > 0.0:
        band = band & (freqs <= float(fmax_hz))
    if np.count_nonzero(band) < 2:
        return float("nan")

    f_band = freqs[band]
    p_true = np.clip(true_psd[band], a_min=0.0, a_max=None)
    p_model = np.clip(model_psd[band], a_min=0.0, a_max=None)

    area_true = float(np.trapz(p_true, f_band))
    area_model = float(np.trapz(p_model, f_band))
    if area_true <= eps or area_model <= eps:
        return float("nan")

    p_true_norm = p_true / (area_true + eps)
    p_model_norm = p_model / (area_model + eps)
    tv = 0.5 * float(np.trapz(np.abs(p_model_norm - p_true_norm), f_band))
    if not np.isfinite(tv):
        return float("nan")
    return float(np.clip(tv, 0.0, 1.0))


def spectral_l2_relative_error(
    true_signal: np.ndarray,
    model_signal: np.ndarray,
    dt: float,
    fmin_hz: float = SPECTRAL_ERROR_FMIN_HZ,
    fmax_hz: float = SPECTRAL_ERROR_FMAX_HZ,
    nperseg: int = SPECTRAL_ERROR_NPERSEG,
    eps: float = 1e-12,
) -> float:
    """Compute L2 relative error between band-limited PSDs."""
    if dt <= 0.0:
        return float("nan")
    true_signal = np.asarray(true_signal, dtype=float).reshape(-1)
    model_signal = np.asarray(model_signal, dtype=float).reshape(-1)
    length = min(true_signal.size, model_signal.size)
    if length < 8:
        return float("nan")
    true_trim = true_signal[-length:]
    model_trim = model_signal[-length:]
    true_proc = true_trim - np.mean(true_trim)
    model_proc = model_trim - np.mean(model_trim)

    if welch is not None:
        fs = 1.0 / float(dt)
        seg = int(max(8, min(int(nperseg), length)))
        ov = seg // 2
        freqs, true_psd = welch(
            true_proc,
            fs=fs,
            window="hann",
            nperseg=seg,
            noverlap=ov,
            detrend=False,
            scaling="density",
        )
        _, model_psd = welch(
            model_proc,
            fs=fs,
            window="hann",
            nperseg=seg,
            noverlap=ov,
            detrend=False,
            scaling="density",
        )
    else:
        freqs = np.fft.rfftfreq(length, d=float(dt))
        true_psd = np.abs(np.fft.rfft(true_proc)) ** 2
        model_psd = np.abs(np.fft.rfft(model_proc)) ** 2

    if freqs.size == 0:
        return float("nan")

    band = np.isfinite(freqs)
    band = band & (freqs >= float(fmin_hz))
    if np.isfinite(float(fmax_hz)) and float(fmax_hz) > 0.0:
        band = band & (freqs <= float(fmax_hz))
    if np.count_nonzero(band) < 2:
        return float("nan")

    p_true = np.clip(true_psd[band], a_min=0.0, a_max=None)
    p_model = np.clip(model_psd[band], a_min=0.0, a_max=None)
    denom = float(np.linalg.norm(p_true))
    if denom <= eps:
        return float("nan")
    rel = float(np.linalg.norm(p_model - p_true) / (denom + eps))
    if not np.isfinite(rel):
        return float("nan")
    return rel


def spectral_l1_relative_error(
    true_signal: np.ndarray,
    model_signal: np.ndarray,
    dt: float,
    fmin_hz: float = SPECTRAL_ERROR_FMIN_HZ,
    fmax_hz: float = SPECTRAL_ERROR_FMAX_HZ,
    nperseg: int = SPECTRAL_ERROR_NPERSEG,
    eps: float = 1e-12,
) -> float:
    """Compute L1 relative error between band-limited PSDs."""
    if dt <= 0.0:
        return float("nan")
    true_signal = np.asarray(true_signal, dtype=float).reshape(-1)
    model_signal = np.asarray(model_signal, dtype=float).reshape(-1)
    length = min(true_signal.size, model_signal.size)
    if length < 8:
        return float("nan")
    true_trim = true_signal[-length:]
    model_trim = model_signal[-length:]
    true_proc = true_trim - np.mean(true_trim)
    model_proc = model_trim - np.mean(model_trim)

    if welch is not None:
        fs = 1.0 / float(dt)
        seg = int(max(8, min(int(nperseg), length)))
        ov = seg // 2
        freqs, true_psd = welch(
            true_proc,
            fs=fs,
            window="hann",
            nperseg=seg,
            noverlap=ov,
            detrend=False,
            scaling="density",
        )
        _, model_psd = welch(
            model_proc,
            fs=fs,
            window="hann",
            nperseg=seg,
            noverlap=ov,
            detrend=False,
            scaling="density",
        )
    else:
        freqs = np.fft.rfftfreq(length, d=float(dt))
        true_psd = np.abs(np.fft.rfft(true_proc)) ** 2
        model_psd = np.abs(np.fft.rfft(model_proc)) ** 2

    if freqs.size == 0:
        return float("nan")

    band = np.isfinite(freqs)
    band = band & (freqs >= float(fmin_hz))
    if np.isfinite(float(fmax_hz)) and float(fmax_hz) > 0.0:
        band = band & (freqs <= float(fmax_hz))
    if np.count_nonzero(band) < 2:
        return float("nan")

    p_true = np.clip(true_psd[band], a_min=0.0, a_max=None)
    p_model = np.clip(model_psd[band], a_min=0.0, a_max=None)
    denom = float(np.sum(np.abs(p_true)))
    if denom <= eps:
        return float("nan")
    rel = float(np.sum(np.abs(p_model - p_true)) / (denom + eps))
    if not np.isfinite(rel):
        return float("nan")
    return rel


class PHVIV(nn.Module):
    """
    Pseudo-/port-Hamiltonian 1-DOF oscillator with NN force.
    State x = [y, v].
    dot x = (J - R(x)) ∇H(x) + G u_theta(x)
    """
    def __init__(
        self,
        dt,
        m=16.79,
        k=1218.0,
        U=0.65,
        rho=1000.0,
        D=0.1,
        q_scale=0.1,
        p_scale=10.0,
        max_damping_ratio=0.2,
        discover_damping: bool = False,
        damping_c: float | None = None,
        include_physical_drag: bool = True,
        force_output: str = "force",
        learn_hamiltonian: bool = False,
        use_pirate_force: bool = False,
        pirate_force_kwargs: dict | None = None,
        use_fourier_features: bool = False,
        fourier_features: int = 64,
        fourier_sigma: float = 1.0,
        use_feature_engineering: bool = False,
        use_reduced_velocity: bool = True,
        use_stochastic_process_noise: bool = True,
        sigma_min: float = 1e-6,
        ur_scale: float | None = None,
        force_net_type: str | None = None,
        residual_kwargs: dict[str, Any] | None = None,
        mlp_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.dt = dt
        self.m = m
        self.k = k
        self.U = U
        self.rho = rho
        self.D = D
        self.register_buffer("max_damping_ratio", torch.tensor(float(max_damping_ratio), dtype=torch.float32))
        self.q_scale = q_scale
        self.p_scale = p_scale
        self.discover_damping = bool(discover_damping)
        self.include_physical_drag = bool(include_physical_drag)
        force_output = str(force_output).strip().lower()
        if force_output not in {"force", "coefficient"}:
            raise ValueError("force_output must be one of: force, coefficient")
        self.force_output = force_output
        self.learn_hamiltonian = bool(learn_hamiltonian)
        self.use_feature_engineering = bool(use_feature_engineering)
        self.use_reduced_velocity = bool(use_reduced_velocity)
        self.use_stochastic_process_noise = bool(use_stochastic_process_noise)
        sigma_min_val = float(sigma_min)
        if not np.isfinite(sigma_min_val) or sigma_min_val < 0.0:
            raise ValueError(f"sigma_min must be finite and non-negative, got {sigma_min_val}")
        self.register_buffer("sigma_min", torch.tensor(sigma_min_val, dtype=torch.float32))
        ur_scale_val = 1.0 if ur_scale is None else float(ur_scale)
        if not np.isfinite(ur_scale_val) or ur_scale_val == 0.0:
            raise ValueError(f"ur_scale must be finite and non-zero, got {ur_scale_val}")
        self.register_buffer("ur_scale", torch.tensor(ur_scale_val, dtype=torch.float32))
        self.engineered_feature_dim = 7
        self.base_feature_dim = self.engineered_feature_dim if self.use_feature_engineering else 2
        self.force_input_dim = self.base_feature_dim + (1 if self.use_reduced_velocity else 0)

        residual_cfg = _default_residual_kwargs()
        if residual_kwargs:
            residual_cfg.update(residual_kwargs)
        mlp_cfg = _default_mlp_kwargs()
        if mlp_kwargs:
            mlp_cfg.update(mlp_kwargs)
        self.residual_hidden = int(residual_cfg.get("hidden", 128))
        self.residual_layers = max(1, int(residual_cfg.get("layers", 2)))
        self.residual_activation = residual_cfg.get("activation", "gelu")
        self.mlp_hidden = int(mlp_cfg.get("hidden", 100))
        self.mlp_layers = max(1, int(mlp_cfg.get("layers", 2)))
        self.mlp_activation = mlp_cfg.get("activation", "gelu")

        self.nn_q_scale = q_scale
        self.nn_p_scale = p_scale
        self.q_scale = 1.0
        self.p_scale = 1.0
        self.register_buffer(
            "res_scale",
            torch.tensor([float(self.q_scale), float(self.p_scale)], dtype=torch.float32),
        )
        self.register_buffer(
            "sqrt_km",
            torch.tensor((float(self.k) * float(self.m)) ** 0.5, dtype=torch.float32),
        )

        # NNs for instantaneous force u_theta(x) and diffusion sigma_theta(x).
        pirate_force_kwargs = pirate_force_kwargs or {}
        self.use_fourier_features = bool(use_fourier_features)
        self.fourier_features = int(fourier_features)
        self.fourier_sigma = float(fourier_sigma)
        self.force_embed = None
        base_force_dim = self.force_input_dim
        force_in_features = base_force_dim
        selected_net = force_net_type if force_net_type not in (None, "") else ("pirate" if use_pirate_force else "residual")
        net_type = str(selected_net).lower()
        valid_types = {"residual", "mlp", "pirate"}
        if net_type not in valid_types:
            raise ValueError(f"force_net_type must be one of {valid_types}, got '{force_net_type}'.")
        self.use_pirate_force = net_type == "pirate"
        self.residual_net = net_type == "residual"
        if self.use_fourier_features:
            if self.fourier_features < 1:
                raise ValueError("fourier_features must be >= 1 when use_fourier_features is True")
            if self.use_pirate_force:
                raise ValueError(
                    "Random Fourier features are already handled inside ODEPirateNet. "
                    "Disable use_fourier_features when use_pirate_force is True."
                )
            self.force_embed = FourierFeatures(
                in_dim=base_force_dim,
                out_features=self.fourier_features,
                sigma=self.fourier_sigma,
                dtype=torch.float32,
            )
            force_in_features = 2 * self.fourier_features

        pirate_cfg_base = dict(pirate_force_kwargs) if pirate_force_kwargs is not None else {}

        def _build_pirate_args(input_size: int) -> dict[str, Any]:
            pirate_cfg = dict(pirate_cfg_base)
            pirate_args = {
                "input_size": int(input_size),
                "output_size": 1,
                "depth": int(pirate_cfg.pop("depth", pirate_cfg.pop("pirate_layers", 2))),
                "fourier_features": int(pirate_cfg.pop("fourier_features", 64)),
                "sigma": float(pirate_cfg.pop("sigma", 1.0)),
                "use_rwf": bool(pirate_cfg.pop("use_rwf", True)),
                "activation": pirate_cfg.pop("activation", "tanh"),
            }
            pirate_args.update(pirate_cfg)
            return pirate_args

        def _build_scalar_net(input_features: int, *, use_selected_backbone: bool) -> nn.Module:
            if use_selected_backbone and self.use_pirate_force:
                return ODEPirateNet(**_build_pirate_args(input_features))
            if use_selected_backbone and self.residual_net:
                layers = [nn.Linear(int(input_features), self.residual_hidden)]
                for _ in range(self.residual_layers):
                    layers.append(Residual(self.residual_hidden, activation=self.residual_activation))
                layers.append(nn.Linear(self.residual_hidden, 1))
                return nn.Sequential(*layers)
            mlp_layers: list[nn.Module] = []
            in_features = int(input_features)
            mlp_act_cls = _activation_factory(self.mlp_activation)
            for _ in range(self.mlp_layers):
                mlp_layers.append(nn.Linear(in_features, self.mlp_hidden))
                mlp_layers.append(mlp_act_cls())
                in_features = self.mlp_hidden
            mlp_layers.append(nn.Linear(self.mlp_hidden, 1))
            return nn.Sequential(*mlp_layers)

        self.u_base_net = _build_scalar_net(force_in_features, use_selected_backbone=True)
        # Backward-compatible alias used throughout the codebase.
        self.u_net = self.u_base_net

        self.sigma_net = (
            _build_scalar_net(force_in_features, use_selected_backbone=True)
            if self.use_stochastic_process_noise
            else None
        )

        if self.learn_hamiltonian:
            h_in_features = self.base_feature_dim
            self.h_net = nn.Sequential(
                nn.Linear(h_in_features, 100),
                nn.GELU(),
                nn.Linear(100, 100),
                nn.GELU(),
                nn.Linear(100, 1),
            )
        else:
            self.h_net = None

        # damping handling
        if self.discover_damping:
            self.zeta0 = torch.tensor(0.01)
            self.zeta_raw = nn.Parameter(torch.logit(self.zeta0/self.max_damping_ratio))
            self.register_buffer("fixed_c", torch.tensor(0.0))
            self.fixed_damping_ratio = None
        else:
            if damping_c is None:
                raise ValueError("damping_c must be provided when discover_damping is False")
            damping_c = float(damping_c)
            self.register_buffer("fixed_c", torch.tensor(damping_c, dtype=torch.float32))
            self.fixed_damping_ratio = float(damping_c / (2.0 * (self.k * self.m) ** 0.5))
            self.zeta_raw = None
        #Learable drag coefficient
        self.log_Cd = nn.Parameter(torch.log(torch.tensor(1.2)))  # start at ~1.2


        self.register_buffer("J", torch.tensor([[0.0, 1.0],
                                                [-1.0, 0.0]]))
        self.register_buffer("G", torch.tensor([[0.0],
                                                [1.0]]))

    @classmethod
    def from_config(
        cls,
        dt: float,
        cfg: dict[str, object],
        arch_cfg: dict[str, object] | None = None,
        device: torch.device | None = None,
    ) -> tuple["PHVIV", dict[str, float]]:
        rho = float(cfg.get("rho", 1000.0))
        D = float(cfg.get("D", 0.1))
        Ca = float(cfg.get("Ca", 1.0))
        k = float(cfg.get("k", 1218.0))
        U = float(cfg.get("U", 0.65))
        damping_c = float(cfg.get("damping_c", 1e-4))
        structural_mass = float(cfg.get("structural_mass", 16.79))
        max_damping_ratio = float(cfg.get("max_damping_ratio", 0.2))
        discover_damping = bool(cfg.get("discover_damping", False))
        include_physical_drag = bool(cfg.get("include_physical_drag", False))
        force_output = str(cfg.get("force_output", "force")).strip().lower()
        learn_hamiltonian = bool(cfg.get("learn_hamiltonian", False))
        use_pirate_force = bool(cfg.get("use_pirate_force", False))
        pirate_force_kwargs = cfg.get("pirate_force_kwargs", {}) or {}
        use_fourier_features = bool(cfg.get("use_fourier_features", False))
        fourier_features = int(cfg.get("fourier_features", 64))
        fourier_sigma = float(cfg.get("fourier_sigma", 1.0))
        use_feature_engineering = bool(cfg.get("use_feature_engineering", False))
        use_reduced_velocity = bool(cfg.get("use_reduced_velocity", True))
        use_stochastic_process_noise = bool(cfg.get("use_stochastic_process_noise", True))
        sigma_min = float(cfg.get("sigma_min", 1e-6))
        ur_scale_val = cfg.get("ur_scale")
        ur_scale = None if ur_scale_val is None else float(ur_scale_val)
        arch_cfg = arch_cfg or {}
        force_net_type = arch_cfg.get("force_net_type")
        residual_kwargs = _default_residual_kwargs()
        residual_kwargs.update(arch_cfg.get("residual_kwargs", {}) or {})
        mlp_kwargs = _default_mlp_kwargs()
        mlp_kwargs.update(arch_cfg.get("mlp_kwargs", {}) or {})
        pirate_arch_kwargs = arch_cfg.get("pirate_force_kwargs", {}) or {}
        combined_pirate_kwargs = dict(pirate_force_kwargs)
        combined_pirate_kwargs.update(pirate_arch_kwargs)
        if "activation" not in combined_pirate_kwargs:
            combined_pirate_kwargs["activation"] = "tanh"
        if "rwf_mu" not in combined_pirate_kwargs:
            combined_pirate_kwargs["rwf_mu"] = 1.0
        if "rwf_sigma" not in combined_pirate_kwargs:
            combined_pirate_kwargs["rwf_sigma"] = 0.1
        q_scale_val = cfg.get("q_scale")
        q_scale = float(q_scale_val) if q_scale_val is not None else D
        m_a = 0.25 * np.pi * D**2 * rho * Ca
        m_eff = structural_mass + m_a
        default_p_scale = np.sqrt(k / m_eff) * m_eff * D
        p_scale_val = cfg.get("p_scale")
        p_scale = float(p_scale_val) if p_scale_val is not None else default_p_scale
        model = cls(
            dt=dt,
            m=m_eff,
            k=k,
            U=U,
            rho=rho,
            D=D,
            q_scale=q_scale,
            p_scale=p_scale,
            max_damping_ratio=max_damping_ratio,
            discover_damping=discover_damping,
            damping_c=damping_c,
            include_physical_drag=include_physical_drag,
            force_output=force_output,
            learn_hamiltonian=learn_hamiltonian,
            use_pirate_force=use_pirate_force,
            pirate_force_kwargs=combined_pirate_kwargs,
            use_fourier_features=use_fourier_features,
            fourier_features=fourier_features,
            fourier_sigma=fourier_sigma,
            use_feature_engineering=use_feature_engineering,
            use_reduced_velocity=use_reduced_velocity,
            use_stochastic_process_noise=use_stochastic_process_noise,
            sigma_min=sigma_min,
            ur_scale=ur_scale,
            force_net_type=force_net_type,
            residual_kwargs=residual_kwargs,
            mlp_kwargs=mlp_kwargs,
        )
        if device is not None:
            model = model.to(device)
        derived = {
            "m_eff": m_eff,
            "D": D,
            "k": k,
            "q_scale": q_scale,
            "p_scale": p_scale,
        }
        return model, derived

    def H(self, x):
        if not self.learn_hamiltonian:
            q = x[..., 0]
            p = x[..., 1]
            return 0.5 * self.k * q**2 + 0.5 * p**2 / self.m
        features = self._base_features(x)
        return self.h_net(features).squeeze(-1)

    def grad_H(self, x):
        if not self.learn_hamiltonian:
            q = x[..., 0]
            p = x[..., 1]
            return torch.stack((self.k * q, p / self.m), dim=-1)
        grad_enabled = torch.is_grad_enabled()
        with torch.enable_grad():
            x_req = x.detach().requires_grad_(True)
            H_val = self.H(x_req)
            grad = torch.autograd.grad(
                H_val.sum(),
                x_req,
                create_graph=grad_enabled,
                retain_graph=grad_enabled,
            )[0]
        return grad

    def R(self, x):
        R = torch.zeros(*x.shape[:-1], 2, 2, device=x.device, dtype=x.dtype)
        if self.discover_damping:
            zeta = torch.sigmoid(self.zeta_raw) * self.max_damping_ratio
            c_eff = 2.0 * zeta * self.sqrt_km
        else:
            c_eff = self.fixed_c
        R[..., 1, 1] = c_eff
        return R
    
    def drag_force(self, x):
        """
        Morison-like cross-flow drag: Fd = -0.5 * rho * D * Cd * |v| * v
        x: (..., 2)
        returns: (..., 1)
        """
        v = x[..., 1] / self.m
        U = torch.full_like(v, self.U)
        Cd = torch.exp(self.log_Cd)  # keep it positive
        rel_vel = torch.sqrt(v**2 + U**2)
        Fd = -0.5 * self.rho * self.D * Cd * torch.abs(rel_vel) * v
        return Fd.unsqueeze(-1)


    def _base_features(self, x):
        if self.use_feature_engineering:
            return self.feature_engineering(x)
        q_scaled = x[..., 0] / self.nn_q_scale
        p_scaled = x[..., 1] / self.nn_p_scale
        return torch.stack((q_scaled, p_scaled), dim=-1)

    def _prepare_reduced_velocity(self, reduced_velocity: torch.Tensor | np.ndarray | float | None, *, like: torch.Tensor) -> torch.Tensor | None:
        if reduced_velocity is None:
            if self.use_reduced_velocity:
                raise ValueError("reduced_velocity is required for this model.")
            return None
        if torch.is_tensor(reduced_velocity):
            rv = reduced_velocity.to(device=like.device, dtype=like.dtype)
        else:
            rv = torch.as_tensor(reduced_velocity, device=like.device, dtype=like.dtype)
        rv = rv / self.ur_scale.to(device=rv.device, dtype=rv.dtype)
        if rv.ndim == 0:
            rv = rv.view(1, 1)
        elif rv.ndim == like.ndim - 1:
            rv = rv.unsqueeze(-1)
        if rv.ndim != like.ndim or rv.shape[-1] != 1:
            raise ValueError("reduced_velocity must be a scalar or have shape (..., 1).")
        if rv.shape[:-1] != like.shape[:-1]:
            rv = rv.expand(like.shape[:-1] + (1,))
        return rv

    def _prepare_reduced_velocity_raw(
        self,
        reduced_velocity: torch.Tensor | np.ndarray | float | None,
        *,
        like: torch.Tensor,
    ) -> torch.Tensor | None:
        if reduced_velocity is None:
            if self.use_reduced_velocity:
                raise ValueError("reduced_velocity is required for force coefficient regularization.")
            return None
        if torch.is_tensor(reduced_velocity):
            rv = reduced_velocity.to(device=like.device, dtype=like.dtype)
        else:
            rv = torch.as_tensor(reduced_velocity, device=like.device, dtype=like.dtype)
        if rv.ndim == 0:
            rv = rv.view(1, 1)
        elif rv.ndim == like.ndim - 1:
            rv = rv.unsqueeze(-1)
        if rv.ndim != like.ndim or rv.shape[-1] != 1:
            raise ValueError("reduced_velocity must be a scalar or have shape (..., 1).")
        if rv.shape[:-1] != like.shape[:-1]:
            rv = rv.expand(like.shape[:-1] + (1,))
        return rv

    def _force_scale_from_reduced_velocity(
        self,
        reduced_velocity: torch.Tensor | np.ndarray | float | None,
        *,
        like: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # `state` is intentionally unused; kept for backward-compatible call sites.
        rv_raw = self._prepare_reduced_velocity_raw(reduced_velocity, like=like)
        if rv_raw is None:
            u_flow = like.new_full(like.shape[:-1] + (1,), float(self.U))
        else:
            omega_n = torch.sqrt(
                torch.as_tensor(float(self.k) / float(self.m), device=like.device, dtype=like.dtype)
            )
            f_n = omega_n / (2.0 * math.pi)
            u_flow = rv_raw * f_n * float(self.D)
        # Dynamic-pressure force scale (unit span): f0 = 0.5 * rho * D * U^2
        f0 = 0.5 * float(self.rho) * float(self.D) * (u_flow**2)
        return torch.clamp(f0, min=1e-12)

    def _force_features(self, x, reduced_velocity: torch.Tensor | np.ndarray | float | None = None):
        base_features = self._base_features(x)
        if self.use_reduced_velocity:
            rv = self._prepare_reduced_velocity(reduced_velocity, like=base_features)
            base_features = torch.cat([base_features, rv], dim=-1)
        return base_features

    def _force_net_raw(
        self,
        x,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
    ):
        base_features = self._force_features(x, reduced_velocity=reduced_velocity)
        features = self.force_embed(base_features) if self.force_embed is not None else base_features
        return self.u_base_net(features)

    def _sigma_net_raw(
        self,
        x,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
    ):
        if not self.use_stochastic_process_noise:
            like = x[..., :1]
            return torch.zeros_like(like)
        base_features = self._force_features(x, reduced_velocity=reduced_velocity)
        features = self.force_embed(base_features) if self.force_embed is not None else base_features
        if self.sigma_net is None:
            like = x[..., :1]
            return torch.zeros_like(like)
        return self.sigma_net(features)

    def sigma_theta(
        self,
        x,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
    ):
        if not self.use_stochastic_process_noise:
            return torch.zeros_like(x[..., :1])
        raw = self._sigma_net_raw(x, reduced_velocity=reduced_velocity)
        sigma_min = self.sigma_min.to(device=raw.device, dtype=raw.dtype)
        sigma = sigma_min + F.softplus(raw)
        if self.force_output == "coefficient":
            # In coefficient mode, interpret sigma_net output as coefficient-scale
            # diffusion and convert to force units using dynamic-pressure F0.
            f0 = self._force_scale_from_reduced_velocity(reduced_velocity, like=sigma, state=x)
            sigma = sigma * f0
        return sigma

    def learned_force_coeff(
        self,
        x,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
    ):
        raw = self._force_net_raw(x, reduced_velocity=reduced_velocity)
        if self.force_output == "coefficient":
            return raw
        f0 = self._force_scale_from_reduced_velocity(reduced_velocity, like=raw, state=x)
        return raw * self.k * self.D / f0

    def learned_force(
        self,
        x,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
    ):
        raw = self._force_net_raw(x, reduced_velocity=reduced_velocity)
        if self.force_output == "coefficient":
            f0 = self._force_scale_from_reduced_velocity(reduced_velocity, like=raw, state=x)
            return raw * f0
        return raw * self.k * self.D

    def drag_force_coeff(self, x, reduced_velocity: torch.Tensor | np.ndarray | float | None = None):
        drag = self.drag_force(x)
        f0 = self._force_scale_from_reduced_velocity(reduced_velocity, like=drag, state=x)
        return drag / f0

    def u_theta_coeff(
        self,
        x,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
    ):
        coeff = self.learned_force_coeff(x, reduced_velocity=reduced_velocity)
        if self.include_physical_drag:
            coeff = coeff + self.drag_force_coeff(x, reduced_velocity=reduced_velocity)
        return coeff

    def u_theta1(
        self,
        x,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
    ):
        return self.learned_force(x, reduced_velocity=reduced_velocity)
    
    def u_theta2(
        self,
        x,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
    ):
        return self.u_theta1(x, reduced_velocity=reduced_velocity) + self.drag_force(x)
    
    def u_theta(
        self,
        x,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
    ):
        if self.force_output == "coefficient":
            coeff = self.u_theta_coeff(x, reduced_velocity=reduced_velocity)
            f0 = self._force_scale_from_reduced_velocity(reduced_velocity, like=coeff, state=x)
            return coeff * f0
        return (
            self.u_theta2(x, reduced_velocity=reduced_velocity)
            if self.include_physical_drag
            else self.u_theta1(x, reduced_velocity=reduced_velocity)
        )
    
    def f(
        self,
        x,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
    ):
        u = self.u_theta(x, reduced_velocity=reduced_velocity)
        g_vec = self.G.squeeze(-1)
        return u * g_vec

    def g(
        self,
        x,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
    ):
        gH = self.grad_H(x)                         # (..., 2)
        JgH = torch.matmul(gH, self.J.T)
        if self.discover_damping:
            zeta = torch.sigmoid(self.zeta_raw) * self.max_damping_ratio
            c_eff = 2.0 * zeta * self.sqrt_km
        else:
            c_eff = self.fixed_c
        damping_term = torch.stack((torch.zeros_like(gH[..., 0]), c_eff * gH[..., 1]), dim=-1)
        return (JgH - damping_term) + self.f(x, reduced_velocity=reduced_velocity)

    def step_euler(
        self,
        x,
        dt,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
    ):
        return x + dt * self.g(x, reduced_velocity=reduced_velocity)

    def step_rk4(
        self,
        x,
        t,
        dt,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
    ):
        x_next, _ = self.rk4_step(x, t, dt, reduced_velocity=reduced_velocity)
        return x_next

    def step_rk4_stochastic(
        self,
        x,
        t,
        dt,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
        *,
        noise: torch.Tensor | None = None,
        noise_scale: float = 1.0,
    ):
        x_next, _ = self.rk4_step(x, t, dt, reduced_velocity=reduced_velocity)
        if not self.use_stochastic_process_noise:
            return x_next
        sigma = self.sigma_theta(x, reduced_velocity=reduced_velocity)
        if noise is None:
            noise = torch.randn_like(sigma)
        else:
            noise = noise.to(device=x_next.device, dtype=x_next.dtype)
            if noise.shape != sigma.shape:
                noise = noise.view(sigma.shape)
        dt_sqrt = math.sqrt(float(dt))
        x_next = x_next.clone()
        x_next[..., 1:2] = x_next[..., 1:2] + float(noise_scale) * sigma * dt_sqrt * noise
        return x_next

    def rk4_step(
        self,
        x,
        t,
        dt,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
    ):
        """
        Perform one Runge-Kutta 4 integration step and return both the next state
        and the averaged force over the step.
        """
        k1 = self.g(x, reduced_velocity=reduced_velocity)
        force1 = self.u_theta(x, reduced_velocity=reduced_velocity)

        x2 = x + 0.5 * dt * k1
        k2 = self.g(x2, reduced_velocity=reduced_velocity)
        force2 = self.u_theta(x2, reduced_velocity=reduced_velocity)

        x3 = x + 0.5 * dt * k2
        k3 = self.g(x3, reduced_velocity=reduced_velocity)
        force3 = self.u_theta(x3, reduced_velocity=reduced_velocity)

        x4 = x + dt * k3
        k4 = self.g(x4, reduced_velocity=reduced_velocity)
        force4 = self.u_theta(x4, reduced_velocity=reduced_velocity)

        x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        force_avg = (force1 + 2.0 * force2 + 2.0 * force3 + force4) / 6.0
        return x_next, force_avg

    def rollout(
        self,
        z0,
        t_seq,
        dt,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
        *,
        stochastic: bool = False,
        noise_scale: float = 1.0,
        generator: torch.Generator | None = None,
    ):
        """
        z0: (B, state_dim)    starting state from data
        t_seq: (B, K+1)       absolute times t0..tK
        returns:
        Z_pred: (B, K+1, state_dim)  predictions incl. z0
        F_hist: (B, K+1, 1)          optional, learned force per step
        """
        K = t_seq.shape[1] - 1

        Z_pred = [z0]
        F_hist = []

        z = z0
        for k in range(K):
            t = t_seq[:, k]
            z_det, Fk = self.rk4_step(z, t, dt, reduced_velocity=reduced_velocity)
            if stochastic and self.use_stochastic_process_noise:
                sigma = self.sigma_theta(z, reduced_velocity=reduced_velocity)
                noise = torch.randn(
                    sigma.shape,
                    device=z_det.device,
                    dtype=z_det.dtype,
                    generator=generator,
                )
                z = z_det.clone()
                z[..., 1:2] = z[..., 1:2] + float(noise_scale) * sigma * math.sqrt(float(dt)) * noise
            else:
                z = z_det
            Z_pred.append(z)
            F_hist.append(Fk)

        Z_pred = torch.stack(Z_pred, dim=1)            # (B,K+1,D)
        if F_hist:
            initial_force = self.u_theta(z0, reduced_velocity=reduced_velocity)
            F_hist = torch.stack([initial_force] + F_hist, dim=1)
        else:
            F_hist = None
        return Z_pred, F_hist

    @staticmethod
    def traj_loss(Z_pred, Z_data, w_state=(1.0, 1.0)):
        # For 1-DOF, assume z=[y,p]
        y_pred, p_pred = Z_pred[...,0], Z_pred[...,1]
        y_data, p_data = Z_data[...,0], Z_data[...,1]
        Ly = ((y_pred - y_data)**2).mean()
        Lp = ((p_pred - p_data)**2).mean()
        return w_state[0]*Ly + w_state[1]*Lp

    
    def res_loss(
        self,
        zi,
        ti,
        zin,
        tin,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
    ):
        return self.res_loss_SRK4(zi, ti, zin, tin, reduced_velocity=reduced_velocity)
    
    def avg_force(
        self,
        zi,
        ti,
        zin,
        tin,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
    ):
        return self.avg_force_SRK4(zi, ti, zin, tin, reduced_velocity=reduced_velocity)

    def res_loss_per_sample(
        self,
        zi,
        ti,
        zin,
        tin,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
    ) -> torch.Tensor:
        return self.res_loss_SRK4_per_sample(zi, ti, zin, tin, reduced_velocity=reduced_velocity)

    def avg_force_per_sample(
        self,
        zi,
        ti,
        zin,
        tin,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
    ) -> torch.Tensor:
        return self.avg_force_SRK4_per_sample(zi, ti, zin, tin, reduced_velocity=reduced_velocity)

    def avg_force_coeff(
        self,
        zi,
        ti,
        zin,
        tin,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
    ):
        return self.avg_force_coeff_SRK4(zi, ti, zin, tin, reduced_velocity=reduced_velocity)

    def avg_force_coeff_per_sample(
        self,
        zi,
        ti,
        zin,
        tin,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
    ) -> torch.Tensor:
        return self.avg_force_coeff_SRK4_per_sample(zi, ti, zin, tin, reduced_velocity=reduced_velocity)

    def _srk4_drift_rate(
        self,
        zi: torch.Tensor,
        zin: torch.Tensor,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dt = float(self.dt)
        b = math.sqrt(3.0) / 6.0
        z_mid = 0.5 * (zi + zin)
        z_a_plus = (0.5 + b) * zi + (0.5 - b) * zin
        z_a_minus = (0.5 - b) * zi + (0.5 + b) * zin
        g_a_plus = self.g(z_a_plus, reduced_velocity=reduced_velocity)
        g_a_minus = self.g(z_a_minus, reduced_velocity=reduced_velocity)
        z_corr_minus = z_mid - b * dt * g_a_plus
        z_corr_plus = z_mid + b * dt * g_a_minus
        g1 = self.g(z_corr_minus, reduced_velocity=reduced_velocity)
        g2 = self.g(z_corr_plus, reduced_velocity=reduced_velocity)
        return 0.5 * (g1 + g2), z_mid

    def _momentum_transition_stats_srk4(
        self,
        zi: torch.Tensor,
        zin: torch.Tensor,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        drift_rate, _ = self._srk4_drift_rate(zi, zin, reduced_velocity=reduced_velocity)
        dp_obs = zin[..., 1:2] - zi[..., 1:2]
        dp_drift = float(self.dt) * drift_rate[..., 1:2]
        innovation = dp_obs - dp_drift
        sigma = self.sigma_theta(zi, reduced_velocity=reduced_velocity)
        var = torch.clamp((sigma * sigma) * float(self.dt), min=1e-12)
        return innovation, sigma, var

    def _stochastic_nll_per_sample(
        self,
        zi: torch.Tensor,
        zin: torch.Tensor,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
    ) -> torch.Tensor:
        innovation, _sigma, var = self._momentum_transition_stats_srk4(zi, zin, reduced_velocity=reduced_velocity)
        nll = 0.5 * (innovation * innovation / var + torch.log(var))
        return nll.squeeze(-1)

    def _deterministic_residual_srk4_per_sample(
        self,
        zi: torch.Tensor,
        zin: torch.Tensor,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
    ) -> torch.Tensor:
        drift_rate, z_mid = self._srk4_drift_rate(zi, zin, reduced_velocity=reduced_velocity)
        res = (zin - zi) - float(self.dt) * drift_rate
        res_scaled = res / self.res_scale.to(device=res.device, dtype=res.dtype)
        if self.force_output == "coefficient":
            f0 = self._force_scale_from_reduced_velocity(
                reduced_velocity,
                like=res_scaled[..., 1:2],
                state=z_mid,
            )
            res_scaled = res_scaled.clone()
            res_scaled[..., 1:2] = res_scaled[..., 1:2] / torch.clamp(f0, min=1e-12)
        return torch.sum(res_scaled * res_scaled, dim=1)

    def avg_diffusion_SRK4(
        self,
        zi,
        ti,
        zin,
        tin,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
    ) -> torch.Tensor:
        sigma = self.sigma_theta(zi, reduced_velocity=reduced_velocity)
        return torch.mean(torch.square(sigma))

    def avg_diffusion_SRK4_per_sample(
        self,
        zi,
        ti,
        zin,
        tin,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
    ) -> torch.Tensor:
        sigma = self.sigma_theta(zi, reduced_velocity=reduced_velocity)
        return torch.square(sigma).squeeze(-1)

    def avg_sigma_reg_SRK4(
        self,
        zi,
        ti,
        zin,
        tin,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
        *,
        norm: str = "l2",
    ) -> torch.Tensor:
        per = self.avg_sigma_reg_SRK4_per_sample(zi, ti, zin, tin, reduced_velocity=reduced_velocity, norm=norm)
        return torch.mean(per)

    def avg_sigma_reg_SRK4_per_sample(
        self,
        zi,
        ti,
        zin,
        tin,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
        *,
        norm: str = "l2",
    ) -> torch.Tensor:
        sigma = self.sigma_theta(zi, reduced_velocity=reduced_velocity)
        norm_key = str(norm).strip().lower()
        if norm_key == "l1":
            return torch.abs(sigma).squeeze(-1)
        if norm_key == "l2":
            return torch.square(sigma).squeeze(-1)
        raise ValueError("sigma regularization norm must be one of: l1, l2.")

    def avg_mean_reg_SRK4(
        self,
        zi,
        ti,
        zin,
        tin,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
        *,
        norm: str = "l1",
        on_coeff: bool = False,
    ) -> torch.Tensor:
        per = self.avg_mean_reg_SRK4_per_sample(
            zi, ti, zin, tin, reduced_velocity=reduced_velocity, norm=norm, on_coeff=on_coeff
        )
        return torch.mean(per)

    def avg_mean_reg_SRK4_per_sample(
        self,
        zi,
        ti,
        zin,
        tin,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
        *,
        norm: str = "l1",
        on_coeff: bool = False,
    ) -> torch.Tensor:
        b = math.sqrt(3.0) / 6.0
        z_a_plus = (0.5 + b) * zi + (0.5 - b) * zin
        z_a_minus = (0.5 - b) * zi + (0.5 + b) * zin
        if on_coeff:
            f1 = self.u_theta_coeff(z_a_plus, reduced_velocity=reduced_velocity)
            f2 = self.u_theta_coeff(z_a_minus, reduced_velocity=reduced_velocity)
        else:
            f1 = self.u_theta(z_a_plus, reduced_velocity=reduced_velocity)
            f2 = self.u_theta(z_a_minus, reduced_velocity=reduced_velocity)
        norm_key = str(norm).strip().lower()
        if norm_key == "l1":
            r1 = torch.sum(torch.abs(f1), dim=1)
            r2 = torch.sum(torch.abs(f2), dim=1)
        elif norm_key == "l2":
            r1 = torch.sum(f1 * f1, dim=1)
            r2 = torch.sum(f2 * f2, dim=1)
        else:
            raise ValueError("mean regularization norm must be one of: l1, l2.")
        return 0.5 * (r1 + r2)
    
    def res_loss_Euler(
        self,
        zi,
        ti,
        zin,
        tin,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
    ):
        dz = (zin-zi)/self.dt
        z_mean = 0.5*(zin+zi)
        res = dz - self.g(z_mean, reduced_velocity=reduced_velocity)
        res_scaled = res / self.res_scale
        if self.force_output == "coefficient":
            f0 = self._force_scale_from_reduced_velocity(
                reduced_velocity,
                like=res_scaled[..., 1:2],
                state=z_mean,
            )
            res_scaled = res_scaled.clone()
            res_scaled[..., 1:2] = res_scaled[..., 1:2] / f0
        loss = torch.mean(torch.sum(res_scaled**2, dim=1))
        return loss

    def avg_force_Euler(
        self,
        zi,
        ti,
        zin,
        tin,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
    ):
        z_mean = 0.5*(zin+zi)
        forces = self.learned_force(z_mean, reduced_velocity=reduced_velocity)
        loss = torch.mean(torch.linalg.norm(forces, ord=1, dim=1))
        return loss
    

    def res_loss_SRK4(
        self,
        zi,
        ti,
        zin,
        tin,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
    ):
        if self.use_stochastic_process_noise:
            per_sample = self._stochastic_nll_per_sample(zi, zin, reduced_velocity=reduced_velocity)
        else:
            per_sample = self._deterministic_residual_srk4_per_sample(zi, zin, reduced_velocity=reduced_velocity)
        return torch.mean(per_sample)

    def res_loss_SRK4_per_sample(
        self,
        zi,
        ti,
        zin,
        tin,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
    ) -> torch.Tensor:
        if self.use_stochastic_process_noise:
            return self._stochastic_nll_per_sample(zi, zin, reduced_velocity=reduced_velocity)
        return self._deterministic_residual_srk4_per_sample(zi, zin, reduced_velocity=reduced_velocity)
    
    def avg_force_SRK4(
        self,
        zi,
        ti,
        zin,
        tin,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
    ):
        if self.use_stochastic_process_noise:
            return self.avg_diffusion_SRK4(zi, ti, zin, tin, reduced_velocity=reduced_velocity)
        b = math.sqrt(3.0) / 6.0

        # same stage points as in res_loss
        z_a_plus  = (0.5 + b) * zi + (0.5 - b) * zin
        z_a_minus = (0.5 - b) * zi + (0.5 + b) * zin

        # evaluate learned force at both stages
        f1 = self.f(z_a_plus, reduced_velocity=reduced_velocity)
        f2 = self.f(z_a_minus, reduced_velocity=reduced_velocity)

        loss = 0.5 * torch.mean(torch.sum(torch.abs(f1), dim=1)) \
            + 0.5 * torch.mean(torch.sum(torch.abs(f2), dim=1))
        return loss

    def avg_force_SRK4_per_sample(
        self,
        zi,
        ti,
        zin,
        tin,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
    ) -> torch.Tensor:
        if self.use_stochastic_process_noise:
            return self.avg_diffusion_SRK4_per_sample(zi, ti, zin, tin, reduced_velocity=reduced_velocity)
        b = math.sqrt(3.0) / 6.0
        z_a_plus = (0.5 + b) * zi + (0.5 - b) * zin
        z_a_minus = (0.5 - b) * zi + (0.5 + b) * zin
        f1 = self.f(z_a_plus, reduced_velocity=reduced_velocity)
        f2 = self.f(z_a_minus, reduced_velocity=reduced_velocity)
        return 0.5 * torch.sum(torch.abs(f1), dim=1) + 0.5 * torch.sum(torch.abs(f2), dim=1)

    def avg_force_coeff_SRK4(
        self,
        zi,
        ti,
        zin,
        tin,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
    ):
        if self.use_stochastic_process_noise:
            return self.avg_diffusion_SRK4(zi, ti, zin, tin, reduced_velocity=reduced_velocity)
        b = math.sqrt(3.0) / 6.0

        z_a_plus = (0.5 + b) * zi + (0.5 - b) * zin
        z_a_minus = (0.5 - b) * zi + (0.5 + b) * zin

        f1 = self.f(z_a_plus, reduced_velocity=reduced_velocity)
        f2 = self.f(z_a_minus, reduced_velocity=reduced_velocity)
        f0 = self._force_scale_from_reduced_velocity(reduced_velocity, like=f1, state=z_a_plus)

        f1c = f1 / f0
        f2c = f2 / f0
        loss = 0.5 * torch.mean(torch.sum(torch.abs(f1c), dim=1)) \
            + 0.5 * torch.mean(torch.sum(torch.abs(f2c), dim=1))
        return loss

    def avg_force_coeff_SRK4_per_sample(
        self,
        zi,
        ti,
        zin,
        tin,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
    ) -> torch.Tensor:
        if self.use_stochastic_process_noise:
            return self.avg_diffusion_SRK4_per_sample(zi, ti, zin, tin, reduced_velocity=reduced_velocity)
        b = math.sqrt(3.0) / 6.0
        z_a_plus = (0.5 + b) * zi + (0.5 - b) * zin
        z_a_minus = (0.5 - b) * zi + (0.5 + b) * zin
        f1 = self.f(z_a_plus, reduced_velocity=reduced_velocity)
        f2 = self.f(z_a_minus, reduced_velocity=reduced_velocity)
        f0 = self._force_scale_from_reduced_velocity(reduced_velocity, like=f1, state=z_a_plus)
        f1c = f1 / f0
        f2c = f2 / f0
        return 0.5 * torch.sum(torch.abs(f1c), dim=1) + 0.5 * torch.sum(torch.abs(f2c), dim=1)
    
    def feature_engineering(self, z):
        q_scaled = z[..., 0] / self.nn_q_scale
        p_scaled = z[..., 1] / self.nn_p_scale
        theta = torch.atan2(p_scaled, q_scaled)
        z_eng = torch.stack(
            (
                q_scaled,
                q_scaled**2,
                p_scaled,
                p_scaled**2,
                q_scaled * p_scaled,
                torch.cos(theta),
                torch.sin(theta),
            ),
            dim=-1,
        )
        return z_eng

def log_displacement_plots(
    writer,
    epoch,
    t,
    y_true_norm,
    y_pred_norm,
    p_pred_norm,
    zoom_mask,
    middle_mask,
    middle_window,
    reduced_velocity: float | None = None,
    *,
    tag_prefix: str = "val/rollout",
    step: int | None = None,
    title_suffix: str = "",
):
    fig, axes = plt.subplots(3, 1, figsize=(6, 9), sharex=False)
    ax_full, ax_diff, ax_zoom = axes
    ur_title = f" (U_r={float(reduced_velocity):.3f})" if reduced_velocity is not None else ""

    ax_full.plot(t, y_true_norm, label="y/D (true)")
    ax_full.plot(t, y_pred_norm, label="y/D (pred)")
    ax_full.set_xlabel("time")
    ax_full.set_ylabel("y/D")
    ax_full.grid(True, alpha=0.3)
    ax_full.set_title(f"Normalized rollout at epoch {epoch+1}{ur_title}{title_suffix}")
    ax_full.legend(loc="upper right")

    diff_y_norm = y_pred_norm - y_true_norm
    ax_diff.plot(t, diff_y_norm, label="Δ(y/D)", color="tab:orange")
    ax_diff.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax_diff.set_xlabel("time")
    ax_diff.set_ylabel("Δy/D")
    ax_diff.grid(True, alpha=0.3)
    ax_diff.set_title(f"Difference (pred - true) epoch {epoch+1}{ur_title}{title_suffix}")
    ax_diff.legend(loc="upper right")

    ax_zoom.plot(t[zoom_mask], y_true_norm[zoom_mask], label="y/D (true)")
    ax_zoom.plot(t[zoom_mask], y_pred_norm[zoom_mask], label="y/D (pred)")
    ax_zoom.set_xlabel("time")
    ax_zoom.set_ylabel("y/D")
    ax_zoom.grid(True, alpha=0.3)
    ax_zoom.set_title(f"Normalized rollout (first 1s) epoch {epoch+1}{ur_title}{title_suffix}")
    ax_zoom.legend(loc="upper right")

    plt.tight_layout()
    writer.add_figure(f"{tag_prefix}_displacement", fig, epoch + 1 if step is None else step)
    plt.close(fig)

def log_force_plots(
    writer,
    epoch,
    t,
    force_coeff_pred,
    force_coeff_true,
    zoom_mask,
    middle_mask,
    middle_window,
    reduced_velocity: float | None = None,
    *,
    force_coeff_baseline=None,
    baseline_label: str = "C_F (Vivana-TD)",
    tag_prefix: str = "val/rollout",
    step: int | None = None,
    title_suffix: str = "",
):
    fig, axes = plt.subplots(4, 1, figsize=(6, 12), sharex=False)
    ax_full, ax_diff, ax_zoom, ax_middle = axes
    ur_title = f" (U_r={float(reduced_velocity):.3f})" if reduced_velocity is not None else ""
    ax_full.plot(t, force_coeff_true, label="C_F (true)", color="tab:blue", alpha=0.7)
    ax_full.plot(t, force_coeff_pred, label="C_F (pred)", color="tab:purple")
    if force_coeff_baseline is not None:
        ax_full.plot(t, force_coeff_baseline, label=baseline_label, color="tab:green", alpha=0.85)
    ax_full.set_xlabel("time")
    ax_full.set_ylabel("C_F")
    ax_full.grid(True, alpha=0.3)
    ax_full.set_title(f"Force coefficient rollout at epoch {epoch+1}{ur_title}{title_suffix}")
    ax_full.legend(loc="upper right")

    diff_force = force_coeff_pred - force_coeff_true
    ax_diff.plot(t, diff_force, label="ΔC_F", color="tab:orange")
    if force_coeff_baseline is not None:
        diff_force_baseline = force_coeff_baseline - force_coeff_true
        ax_diff.plot(t, diff_force_baseline, label=f"Δ({baseline_label})", color="tab:green", alpha=0.85)
    ax_diff.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax_diff.set_xlabel("time")
    ax_diff.set_ylabel("ΔC_F")
    ax_diff.grid(True, alpha=0.3)
    ax_diff.set_title(f"Force coefficient difference (pred - true) epoch {epoch+1}{ur_title}{title_suffix}")
    ax_diff.legend(loc="upper right")

    ax_zoom.plot(t[zoom_mask], force_coeff_true[zoom_mask], label="C_F (true)", color="tab:blue", alpha=0.7)
    ax_zoom.plot(t[zoom_mask], force_coeff_pred[zoom_mask], label="C_F (pred)", color="tab:purple")
    if force_coeff_baseline is not None:
        ax_zoom.plot(t[zoom_mask], force_coeff_baseline[zoom_mask], label=baseline_label, color="tab:green", alpha=0.85)
    ax_zoom.set_xlabel("time")
    ax_zoom.set_ylabel("C_F")
    ax_zoom.grid(True, alpha=0.3)
    ax_zoom.set_title(f"Force coefficient rollout (first 1s) epoch {epoch+1}{ur_title}{title_suffix}")
    ax_zoom.legend(loc="upper right")

    mid_start, mid_end = middle_window
    ax_middle.plot(t[middle_mask], force_coeff_true[middle_mask], label="C_F (true)", color="tab:blue", alpha=0.7)
    ax_middle.plot(t[middle_mask], force_coeff_pred[middle_mask], label="C_F (pred)", color="tab:purple")
    if force_coeff_baseline is not None:
        ax_middle.plot(
            t[middle_mask],
            force_coeff_baseline[middle_mask],
            label=baseline_label,
            color="tab:green",
            alpha=0.85,
        )
    ax_middle.set_xlabel("time")
    ax_middle.set_ylabel("C_F")
    ax_middle.grid(True, alpha=0.3)
    ax_middle.set_title(f"Force coefficient rollout ({mid_start}-{mid_end}s) epoch {epoch+1}{ur_title}{title_suffix}")
    ax_middle.legend(loc="upper right")

    plt.tight_layout()
    writer.add_figure(f"{tag_prefix}_force", fig, epoch + 1 if step is None else step)
    plt.close(fig)


def log_force_component_plots(
    writer,
    epoch,
    t,
    force_coeff_pred,
    force_coeff_delta,
    sigma_coeff,
    zoom_mask,
    middle_mask,
    middle_window,
    reduced_velocity: float | None = None,
    *,
    tag_prefix: str = "val/rollout",
    step: int | None = None,
    title_suffix: str = "",
):
    fig, axes = plt.subplots(3, 1, figsize=(6, 9), sharex=False)
    ax_full, ax_zoom, ax_middle = axes
    ur_title = f" (U_r={float(reduced_velocity):.3f})" if reduced_velocity is not None else ""

    ax_full.plot(t, force_coeff_pred, label="a", color="tab:purple")
    ax_full.plot(t, force_coeff_delta, label="delta a", color="tab:orange")
    ax_full.plot(t, sigma_coeff, label="sigma", color="tab:green")
    ax_full.set_xlabel("time")
    ax_full.set_ylabel("coefficient")
    ax_full.grid(True, alpha=0.3)
    ax_full.set_title(f"Force coefficient components at epoch {epoch+1}{ur_title}{title_suffix}")
    ax_full.legend(loc="upper right")

    ax_zoom.plot(t[zoom_mask], force_coeff_pred[zoom_mask], label="a", color="tab:purple")
    ax_zoom.plot(t[zoom_mask], force_coeff_delta[zoom_mask], label="delta a", color="tab:orange")
    ax_zoom.plot(t[zoom_mask], sigma_coeff[zoom_mask], label="sigma", color="tab:green")
    ax_zoom.set_xlabel("time")
    ax_zoom.set_ylabel("coefficient")
    ax_zoom.grid(True, alpha=0.3)
    ax_zoom.set_title(f"Force coefficient components (first 1s) epoch {epoch+1}{ur_title}{title_suffix}")
    ax_zoom.legend(loc="upper right")

    mid_start, mid_end = middle_window
    ax_middle.plot(t[middle_mask], force_coeff_pred[middle_mask], label="a", color="tab:purple")
    ax_middle.plot(t[middle_mask], force_coeff_delta[middle_mask], label="delta a", color="tab:orange")
    ax_middle.plot(t[middle_mask], sigma_coeff[middle_mask], label="sigma", color="tab:green")
    ax_middle.set_xlabel("time")
    ax_middle.set_ylabel("coefficient")
    ax_middle.grid(True, alpha=0.3)
    ax_middle.set_title(f"Force coefficient components ({mid_start}-{mid_end}s) epoch {epoch+1}{ur_title}{title_suffix}")
    ax_middle.legend(loc="upper right")

    plt.tight_layout()
    writer.add_figure(f"{tag_prefix}_force", fig, epoch + 1 if step is None else step)
    plt.close(fig)


def log_force_plots_with_components(
    writer,
    epoch,
    t,
    force_coeff_pred,
    force_coeff_true,
    force_coeff_delta,
    sigma_coeff,
    zoom_mask,
    reduced_velocity: float | None = None,
    *,
    tag_prefix: str = "val/rollout",
    step: int | None = None,
    title_suffix: str = "",
):
    fig, axes = plt.subplots(4, 1, figsize=(6, 12), sharex=False)
    ax_full, ax_diff, ax_zoom, ax_components = axes
    ur_title = f" (U_r={float(reduced_velocity):.3f})" if reduced_velocity is not None else ""

    ax_full.plot(t, force_coeff_true, label="C_F (true)", color="tab:blue", alpha=0.7)
    ax_full.plot(t, force_coeff_pred, label="C_F (pred)", color="tab:purple")
    ax_full.set_xlabel("time")
    ax_full.set_ylabel("C_F")
    ax_full.grid(True, alpha=0.3)
    ax_full.set_title(f"Force coefficient rollout at epoch {epoch+1}{ur_title}{title_suffix}")
    ax_full.legend(loc="upper right")

    diff_force = force_coeff_pred - force_coeff_true
    ax_diff.plot(t, diff_force, label="ΔC_F", color="tab:orange")
    ax_diff.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax_diff.set_xlabel("time")
    ax_diff.set_ylabel("ΔC_F")
    ax_diff.grid(True, alpha=0.3)
    ax_diff.set_title(f"Force coefficient difference (pred - true) epoch {epoch+1}{ur_title}{title_suffix}")
    ax_diff.legend(loc="upper right")

    ax_zoom.plot(t[zoom_mask], force_coeff_true[zoom_mask], label="C_F (true)", color="tab:blue", alpha=0.7)
    ax_zoom.plot(t[zoom_mask], force_coeff_pred[zoom_mask], label="C_F (pred)", color="tab:purple")
    ax_zoom.set_xlabel("time")
    ax_zoom.set_ylabel("C_F")
    ax_zoom.grid(True, alpha=0.3)
    ax_zoom.set_title(f"Force coefficient rollout (first 1s) epoch {epoch+1}{ur_title}{title_suffix}")
    ax_zoom.legend(loc="upper right")

    ax_components.plot(t, force_coeff_pred, label="mean C_F", color="tab:purple")
    ax_components.plot(t, force_coeff_delta, label="corrected C_F", color="tab:orange")
    ax_components.plot(t, sigma_coeff, label="sigma", color="tab:green")
    ax_components.set_xlabel("time")
    ax_components.set_ylabel("coefficient")
    ax_components.grid(True, alpha=0.3)
    ax_components.set_title(f"Force coefficient components epoch {epoch+1}{ur_title}{title_suffix}")
    ax_components.legend(loc="upper right")

    plt.tight_layout()
    writer.add_figure(f"{tag_prefix}_force", fig, epoch + 1 if step is None else step)
    plt.close(fig)


def _phase_plot_extent(
    q_norm: np.ndarray,
    p_norm: np.ndarray,
    *,
    bins: int = 96,
    extent_scale: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(q_norm, dtype=float).reshape(-1)
    p = np.asarray(p_norm, dtype=float).reshape(-1)
    mask = np.isfinite(q) & np.isfinite(p)
    q = q[mask]
    p = p[mask]
    if q.size < 4:
        raise ValueError("Need at least four finite samples to build a phase-space extent.")

    q_center = 0.5 * (float(np.min(q)) + float(np.max(q)))
    p_center = 0.5 * (float(np.min(p)) + float(np.max(p)))
    q_radius = max(1e-6, float(np.max(np.abs(q - q_center))))
    p_radius = max(1e-6, float(np.max(np.abs(p - p_center))))
    q_half = float(extent_scale) * q_radius
    p_half = float(extent_scale) * p_radius
    q_vals = np.linspace(q_center - q_half, q_center + q_half, int(max(16, bins)))
    p_vals = np.linspace(p_center - p_half, p_center + p_half, int(max(16, bins)))
    return q_vals, p_vals


def _nearest_history_context_grid(
    q_grid: np.ndarray,
    p_grid: np.ndarray,
    q_ref: np.ndarray,
    p_ref: np.ndarray,
    history_context: np.ndarray | None,
) -> np.ndarray | None:
    if history_context is None:
        return None
    ctx = np.asarray(history_context, dtype=float)
    if ctx.ndim != 2 or ctx.shape[0] == 0:
        return None
    q_ref_arr = np.asarray(q_ref, dtype=float).reshape(-1)
    p_ref_arr = np.asarray(p_ref, dtype=float).reshape(-1)
    ref = np.stack([q_ref_arr, p_ref_arr], axis=1)
    grid = np.stack([q_grid.reshape(-1), p_grid.reshape(-1)], axis=1)
    dist2 = np.sum((grid[:, None, :] - ref[None, :, :]) ** 2, axis=2)
    nearest = np.argmin(dist2, axis=1)
    return ctx[nearest]


def log_phase_component_plots(
    writer,
    epoch,
    model: "PHVIV",
    q_norm_true,
    p_norm_true,
    q_norm_pred,
    p_norm_pred,
    reduced_velocity: float | None = None,
    *,
    D: float,
    k: float,
    m_eff: float,
    device: torch.device,
    history_context: np.ndarray | None = None,
    tag_prefix: str = "final_val/phase",
    step: int | None = None,
    title_suffix: str = "",
    bins: int = 96,
    extent_scale: float = 2.0,
):
    q_vals, p_vals = _phase_plot_extent(q_norm_true, p_norm_true, bins=bins, extent_scale=extent_scale)
    q_grid, p_grid = np.meshgrid(q_vals, p_vals, indexing="xy")
    omega = math.sqrt(float(k) / float(m_eff))
    q_phys = q_grid.reshape(-1) * float(D)
    v_phys = p_grid.reshape(-1) * (omega * float(D))
    p_phys = v_phys * float(m_eff)
    state = torch.stack(
        [
            torch.as_tensor(q_phys, dtype=torch.float32, device=device),
            torch.as_tensor(p_phys, dtype=torch.float32, device=device),
        ],
        dim=1,
    )
    rv_value = 0.0 if reduced_velocity is None else float(reduced_velocity)
    rv = torch.full((state.shape[0], 1), rv_value, dtype=state.dtype, device=device)
    ctx_grid_np = _nearest_history_context_grid(q_grid, p_grid, q_norm_pred, p_norm_pred, history_context)
    ctx_grid = None
    if ctx_grid_np is not None:
        ctx_grid = torch.as_tensor(ctx_grid_np, dtype=state.dtype, device=device)
    with torch.no_grad():
        coeff_total, coeff_delta, coeff_sigma = _force_component_coefficients(model, state, rv, ctx_grid)
    coeff_total_grid = coeff_total.detach().cpu().numpy().reshape(q_grid.shape)
    coeff_sigma_grid = coeff_sigma.detach().cpu().numpy().reshape(q_grid.shape)
    coeff_delta_grid = coeff_delta.detach().cpu().numpy().reshape(q_grid.shape)

    component_specs: list[tuple[str, np.ndarray, str]] = [
        ("a", coeff_total_grid, "Force coefficient a"),
        ("sigma", coeff_sigma_grid, "Sigma coefficient"),
    ]
    if np.any(np.abs(coeff_delta_grid) > 0.0):
        component_specs.append(("delta a", coeff_delta_grid, "Correction coefficient delta a"))

    fig, axes = plt.subplots(1, len(component_specs), figsize=(6 * len(component_specs), 5), sharex=True, sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    ur_title = f" (U_r={float(reduced_velocity):.3f})" if reduced_velocity is not None else ""

    q_true = np.asarray(q_norm_true, dtype=float).reshape(-1)
    p_true = np.asarray(p_norm_true, dtype=float).reshape(-1)
    q_pred = np.asarray(q_norm_pred, dtype=float).reshape(-1)
    p_pred = np.asarray(p_norm_pred, dtype=float).reshape(-1)

    for ax, (label, values, title) in zip(axes, component_specs):
        finite_vals = values[np.isfinite(values)]
        if label in {"a", "delta a"}:
            vmax = float(np.max(np.abs(finite_vals))) if finite_vals.size > 0 else 1.0
            vmax = max(vmax, 1e-12)
            levels = np.linspace(-vmax, vmax, 21)
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
            cmap = ax.contourf(q_grid, p_grid, values, levels=levels, cmap=SIGNED_PHASE_CMAP, norm=norm)
        else:
            vmin = float(np.min(finite_vals)) if finite_vals.size > 0 else 0.0
            vmax = float(np.max(finite_vals)) if finite_vals.size > 0 else 1.0
            if not np.isfinite(vmin):
                vmin = 0.0
            if not np.isfinite(vmax) or vmax <= vmin:
                vmax = vmin + 1e-12
            levels = np.linspace(vmin, vmax, 21)
            sigma_cmap = LinearSegmentedColormap.from_list(
                "sigma_phase_map",
                [(0.0, "#b8b8b8"), (1.0, "#2ca02c")],
            )
            norm = Normalize(vmin=vmin, vmax=vmax)
            cmap = ax.contourf(q_grid, p_grid, values, levels=levels, cmap=sigma_cmap, norm=norm)
        ax.plot(q_true, p_true, color="white", linewidth=1.4, alpha=0.9, linestyle="--", label="val traj")
        ax.plot(q_pred, p_pred, color="black", linewidth=1.0, alpha=0.9, label="rollout")
        ax.plot(q_pred[0], p_pred[0], marker="o", color="red", markersize=3)
        ax.set_xlabel("y/D")
        ax.set_ylabel("v/(omega D)")
        ax.set_title(f"{title}{ur_title}{title_suffix}")
        ax.grid(True, alpha=0.15)
        ax.legend(loc="upper right")
        cbar = fig.colorbar(cmap, ax=ax)
        cbar.set_label(label)

    plt.tight_layout()
    writer.add_figure(f"{tag_prefix}_phase_components", fig, epoch + 1 if step is None else step)
    plt.close(fig)

def log_hamiltonian_plots(
    writer,
    epoch,
    t,
    hamiltonian_model,
    zoom_mask,
    middle_mask,
    middle_window,
    hamiltonian_data: np.ndarray | None = None,
    reduced_velocity: float | None = None,
    *,
    tag_prefix: str = "val/rollout",
    step: int | None = None,
    title_suffix: str = "",
):
    fig, axes = plt.subplots(4, 1, figsize=(6, 12), sharex=False)
    ax_full, ax_diff, ax_zoom, ax_middle = axes
    ur_title = f" (U_r={float(reduced_velocity):.3f})" if reduced_velocity is not None else ""
    model_kwargs = {"color": "tab:orange", "label": "H_model"}
    data_kwargs = {"color": "tab:blue", "linestyle": "--", "alpha": 0.8, "label": "H_data"}

    h_model_rel = hamiltonian_model - (hamiltonian_model[0] if hamiltonian_model.size else 0.0)
    h_data_rel = None
    if hamiltonian_data is not None:
        h_data_rel = hamiltonian_data - (hamiltonian_data[0] if hamiltonian_data.size else 0.0)

    ax_full.plot(t, h_model_rel, **model_kwargs.copy())
    if hamiltonian_data is not None:
        ax_full.plot(t, h_data_rel, **data_kwargs.copy())
    ax_full.set_xlabel("time")
    ax_full.set_ylabel("Hamiltonian")
    ax_full.grid(True, alpha=0.3)
    ax_full.set_title(f"Hamiltonian rollout at epoch {epoch+1}{ur_title}{title_suffix}")
    ax_full.legend(loc="upper right")

    if hamiltonian_data is not None:
        diff_h = h_model_rel - h_data_rel
        ax_diff.plot(t, diff_h, label="ΔH", color="tab:purple")
    else:
        ax_diff.plot(t, np.zeros_like(t), label="ΔH (no data)", color="tab:gray", linestyle="--")
    ax_diff.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax_diff.set_xlabel("time")
    ax_diff.set_ylabel("ΔH")
    ax_diff.grid(True, alpha=0.3)
    ax_diff.set_title(f"Hamiltonian difference epoch {epoch+1}{ur_title}{title_suffix}")
    ax_diff.legend(loc="upper right")

    ax_zoom.plot(t[zoom_mask], h_model_rel[zoom_mask], **model_kwargs.copy())
    if hamiltonian_data is not None:
        ax_zoom.plot(t[zoom_mask], h_data_rel[zoom_mask], **data_kwargs.copy())
    ax_zoom.set_xlabel("time")
    ax_zoom.set_ylabel("Hamiltonian")
    ax_zoom.grid(True, alpha=0.3)
    ax_zoom.set_title(f"Hamiltonian (first 1s) epoch {epoch+1}{ur_title}{title_suffix}")
    ax_zoom.legend(loc="upper right")

    mid_start, mid_end = middle_window
    ax_middle.plot(t[middle_mask], h_model_rel[middle_mask], **model_kwargs.copy())
    if hamiltonian_data is not None:
        ax_middle.plot(t[middle_mask], h_data_rel[middle_mask], **data_kwargs.copy())
    ax_middle.set_xlabel("time")
    ax_middle.set_ylabel("Hamiltonian")
    ax_middle.grid(True, alpha=0.3)
    ax_middle.set_title(f"Hamiltonian ({mid_start}-{mid_end}s) epoch {epoch+1}{ur_title}{title_suffix}")
    ax_middle.legend(loc="upper right")

    plt.tight_layout()
    writer.add_figure(f"{tag_prefix}_hamiltonian", fig, epoch + 1 if step is None else step)
    plt.close(fig)


def log_loss_vs_ur(
    writer,
    epoch: int,
    losses_by_ur: dict[str, dict[float, float]],
    *,
    tag: str = "val/loss_vs_ur",
    title: str = "Loss vs reduced velocity",
):
    if not losses_by_ur:
        return
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for name, ur_map in losses_by_ur.items():
        if not ur_map:
            continue
        xs = sorted(ur_map.keys())
        ys = [ur_map[x] for x in xs]
        ax.plot(xs, ys, marker="o", label=name)
    ax.set_xlabel("Reduced velocity (U_r)")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    writer.add_figure(tag, fig, epoch)
    plt.close(fig)


def format_loss_vs_ur_text(
    losses_by_ur: dict[str, dict[float, float]],
    *,
    title: str = "Validation loss vs U_r",
    empty_message: str = "No per-U_r losses were available.",
) -> str:
    if not losses_by_ur:
        return f"{title}\n\n{empty_message}"
    names = [name for name, ur_map in losses_by_ur.items() if ur_map]
    if not names:
        return f"{title}\n\n{empty_message}"
    ur_values = sorted({float(ur) for name in names for ur in losses_by_ur[name].keys()})
    if not ur_values:
        return f"{title}\n\n{empty_message}"

    lines: list[str] = [title, "", "| U_r | " + " | ".join(names) + " |"]
    lines.append("|---|" + "|".join(["---"] * len(names)) + "|")
    for ur_val in ur_values:
        row = [f"{ur_val:.6g}"]
        for name in names:
            value = losses_by_ur[name].get(float(ur_val))
            if value is None or not np.isfinite(float(value)):
                row.append("nan")
            else:
                row.append(f"{float(value):.6e}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def log_final_rollout_errors_vs_ur(
    writer: SummaryWriter,
    ur_values: Sequence[float],
    metrics_list: Sequence[dict[str, float]],
    epoch: int,
    *,
    tag: str = "final_val/errors_vs_ur",
) -> None:
    pairs = []
    for ur_val, metrics in zip(ur_values, metrics_list):
        if not metrics:
            continue
        pairs.append((float(ur_val), metrics))
    if not pairs:
        return
    pairs.sort(key=lambda item: item[0])
    x_all = [p[0] for p in pairs]
    metrics_all = [p[1] for p in pairs]

    series = [
        (DISP_SPECTRAL_REL_ERROR_KEY, DISP_SPECTRAL_REL_ERROR_KEY),
        (DOMINANT_FREQ_REL_ERROR_KEY, DOMINANT_FREQ_REL_ERROR_KEY),
        (DISP_STD_REL_ERROR_KEY, DISP_STD_REL_ERROR_KEY),
        (FORCE_SPECTRAL_REL_ERROR_KEY, FORCE_SPECTRAL_REL_ERROR_KEY),
    ]
    grouped_errors: dict[str, dict[float, list[float]]] = {key: {} for key, _ in series}
    for ur_val, metrics in pairs:
        ur_key = float(np.round(ur_val, 6))
        for key, _label in series:
            if key not in metrics:
                continue
            value = float(metrics[key])
            if not np.isfinite(value):
                continue
            grouped_errors[key].setdefault(ur_key, []).append(value)
    errors_by_ur: dict[str, dict[float, float]] = {}
    for key, by_ur in grouped_errors.items():
        if not by_ur:
            continue
        errors_by_ur[key] = {ur: float(np.mean(vals)) for ur, vals in by_ur.items() if vals}
    for metric_name, by_ur in errors_by_ur.items():
        for ur_key, value in sorted(by_ur.items()):
            writer.add_scalar(f"final_val/by_ur/{metric_name}/U_r={ur_key:.6g}", float(value), epoch)
    writer.add_text(
        "final_val/errors_vs_ur_text",
        format_loss_vs_ur_text(
            errors_by_ur,
            title="Final rollout errors vs U_r",
            empty_message="No per-U_r errors were available.",
        ),
        epoch,
    )

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    plotted = False
    for key, label in series:
        xs: list[float] = []
        ys: list[float] = []
        for x_val, metrics in zip(x_all, metrics_all):
            if key not in metrics:
                continue
            y_val = float(metrics[key])
            if not np.isfinite(y_val) or y_val <= 0.0:
                continue
            xs.append(x_val)
            ys.append(y_val)
        if xs:
            ax.plot(xs, ys, marker="o", label=label)
            plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel("Reduced velocity (U_r)")
    ax.set_ylabel("Error")
    ax.set_yscale("log")
    ax.set_title("Final rollout errors vs U_r")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    writer.add_figure(tag, fig, epoch)
    plt.close(fig)

def preprocess_timeseries(
    t: np.ndarray,
    y: np.ndarray,
    force: np.ndarray,
    hamiltonian: np.ndarray | None,
    data_cfg: DataConfig,
    velocity: np.ndarray | None = None,
    *,
    cut_start_seconds: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, float]:
    """
    Apply optional trimming and uniform decimation to time-series arrays.
    """
    if t.size == 0:
        return t, y, force, hamiltonian, velocity, float("nan")
    if hamiltonian is not None and np.asarray(hamiltonian).shape[0] != t.shape[0]:
        raise ValueError("Hamiltonian array must have the same length as the time vector.")
    if velocity is not None and np.asarray(velocity).shape[0] != t.shape[0]:
        raise ValueError("Velocity array must have the same length as the time vector.")
    mask = np.ones_like(t, dtype=bool)
    if cut_start_seconds is None:
        cut_start_seconds = float(getattr(data_cfg, "cut_start_seconds", 0.0))
    cut_start_seconds = float(cut_start_seconds)
    if cut_start_seconds > 0.0:
        t0 = float(np.asarray(t)[0])
        mask &= t >= (t0 + cut_start_seconds)
    if data_cfg.steadystate:
        mask &= t > float(data_cfg.steadystate_time_threshold)
    t_proc = t[mask]
    y_proc = y[mask]
    f_proc = force[mask]
    h_proc = None if hamiltonian is None else np.asarray(hamiltonian)[mask]
    v_proc = None if velocity is None else np.asarray(velocity)[mask]
    step_requested = max(1, int(data_cfg.reduction_factor if data_cfg.reduce_time else 1))
    # Keep at least two samples after decimation whenever trimming left >=2 samples.
    # This prevents brittle failures for aggressive reduction on short trimmed series.
    step = min(step_requested, max(1, int(t_proc.size) - 1))
    if step < step_requested:
        warnings.warn(
            "Reduction factor was too large for the trimmed series; "
            f"using step={step} instead of {step_requested} to keep at least two samples."
        )
    if step > 1:
        t_proc = t_proc[::step]
        y_proc = y_proc[::step]
        f_proc = f_proc[::step]
        if h_proc is not None:
            h_proc = h_proc[::step]
        if v_proc is not None:
            v_proc = v_proc[::step]
    if t_proc.size < 2:
        raise ValueError(
            "After trimming/reduction, too few samples remain to infer dt. "
            f"(cut_start_seconds={cut_start_seconds}, steadystate={bool(data_cfg.steadystate)}, "
            f"reduce_time={bool(data_cfg.reduce_time)}, reduction_factor={int(data_cfg.reduction_factor)})"
        )
    dt_value = float(t_proc[1] - t_proc[0]) if t_proc.size > 1 else float("nan")
    return t_proc, y_proc, f_proc, h_proc, v_proc, dt_value


def resolve_cut_start_seconds(data_cfg: DataConfig, split: str) -> float:
    split = str(split).strip().lower()
    default_cut = float(getattr(data_cfg, "cut_start_seconds", 0.0))
    if split == "train":
        override = getattr(data_cfg, "cut_start_seconds_train", None)
    elif split == "val":
        override = getattr(data_cfg, "cut_start_seconds_val", None)
    else:
        override = None
    if override is None:
        return default_cut
    return float(override)


def resolve_middle_time_plot(
    data_cfg: DataConfig | Any,
    method_cfg: dict[str, Any] | None = None,
    *,
    method_name: str = "method",
    default_window: tuple[float, float] = (15.0, 17.0),
) -> list[float]:
    """Resolve rollout middle-window from method-specific config, then data config."""
    candidate = None
    if method_cfg is not None:
        candidate = method_cfg.get("middle_time_plot", None)
    if candidate is None:
        candidate = getattr(data_cfg, "middle_time_plot", None)
    if candidate is None:
        candidate = list(default_window)
    values = np.asarray(candidate, dtype=float).reshape(-1)
    if values.size != 2:
        raise ValueError(
            f"{method_name}.middle_time_plot (or data.middle_time_plot) must contain exactly 2 values."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"{method_name}.middle_time_plot (or data.middle_time_plot) must contain only finite values."
        )
    start = float(values[0])
    end = float(values[1])
    if end <= start:
        raise ValueError(
            f"{method_name}.middle_time_plot (or data.middle_time_plot) must satisfy end > start; got [{start}, {end}]."
        )
    return [start, end]


def compute_velocity_numpy(
    y_np: np.ndarray,
    dt: float,
    use_savgol: bool = True,
    savgol_window: int = 15,
    savgol_polyorder: int = 3,
) -> np.ndarray:
    signal = np.asarray(y_np, dtype=float)
    if signal.size < 2 or dt <= 0.0:
        return np.zeros_like(signal)
    if use_savgol and savgol_filter is not None and signal.size >= 3:
        window = min(int(savgol_window), signal.size)
        if window % 2 == 0:
            window -= 1
        if window >= 3:
            polyorder = min(int(savgol_polyorder), window - 1)
            try:
                vel = savgol_filter(
                    signal,
                    window_length=window,
                    polyorder=polyorder,
                    deriv=1,
                    delta=dt,
                    axis=0,
                    mode="interp",
                )
                return np.ascontiguousarray(vel)
            except ValueError:
                pass
    vel = np.zeros_like(signal)
    vel[0] = (signal[1] - signal[0]) / dt if signal.size >= 2 else 0.0
    vel[-1] = (signal[-1] - signal[-2]) / dt if signal.size >= 2 else 0.0
    if signal.size > 2:
        vel[1:-1] = (signal[2:] - signal[:-2]) / (2.0 * dt)
    return vel


def combine_datasets(datasets: list[TensorDataset | ConcatDataset]) -> TensorDataset | ConcatDataset:
    if not datasets:
        raise ValueError("No datasets provided for combination.")
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def _build_causal_feature_windows(feature_seq: torch.Tensor, window: int) -> torch.Tensor:
    if feature_seq.ndim != 2:
        raise ValueError("feature_seq must have shape (T, C).")
    if int(window) < 1:
        raise ValueError("window must be >= 1.")
    steps, channels = feature_seq.shape
    hist = feature_seq.new_empty((steps, int(window), channels))
    win = int(window)
    for idx in range(steps):
        start = max(0, idx - win + 1)
        seq = feature_seq[start : idx + 1]
        cur_len = int(seq.shape[0])
        if cur_len < win:
            pad_len = win - cur_len
            hist[idx, :pad_len, :] = seq[0:1, :].expand(pad_len, channels)
            hist[idx, pad_len:, :] = seq
        else:
            hist[idx] = seq
    return hist


def build_dataloader_from_series(
    series_data: list[tuple[np.ndarray, np.ndarray, float, np.ndarray | None, np.ndarray | None, np.ndarray]],
    m_eff: float,
    batch_size: int,
    device: torch.device,
    smoothing_cfg: SmoothingConfig | None = None,
    shuffle: bool = True,
    *,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
    history_window: int | None = None,
) -> tuple[DataLoader, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]], int]:
    if not series_data:
        raise ValueError("series_data must contain at least one (y, t, dt, vel, force, U_r) tuple.")
    sequence_tensors: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    datasets: list[TensorDataset | ConcatDataset] = []
    min_length: int | None = None
    for y_np, t_np, dt_value, vel_np, force_np, ur_np in series_data:
        y_tensor, vel_tensor, t_tensor = prepare_sequence_tensors(
            y_np,
            t_np,
            dt_value,
            m_eff,
            device,
            smoothing_cfg=smoothing_cfg,
            vel_np=vel_np,
        )
        ur_arr = np.asarray(ur_np, dtype=float)
        if ur_arr.ndim == 0:
            ur_arr = np.full((y_tensor.shape[0],), float(ur_arr), dtype=float)
        else:
            ur_arr = ur_arr.reshape(-1)
        if ur_arr.shape[0] != y_tensor.shape[0]:
            raise ValueError("Reduced velocity array must have the same length as displacement.")
        ur_tensor = torch.from_numpy(np.ascontiguousarray(ur_arr)).float()
        force_tensor: torch.Tensor | None = None
        if force_np is not None:
            force_arr = np.asarray(force_np, dtype=float)
            if force_arr.shape[0] != y_tensor.shape[0]:
                raise ValueError("Force array must have the same length as displacement.")
            force_tensor = torch.from_numpy(np.ascontiguousarray(force_arr)).float()
        sequence_tensors.append((y_tensor, vel_tensor, t_tensor, ur_tensor))
        datasets.append(
            build_dataset(
                y_tensor,
                vel_tensor,
                m_eff,
                t_tensor,
                reduced_velocity=ur_tensor,
                force_tensor=force_tensor,
                history_window=history_window,
            )
        )
        seq_len = y_tensor.shape[0]
        min_length = seq_len if min_length is None else min(min_length, seq_len)
    dataset = combine_datasets(datasets)
    loader_kwargs: dict[str, object] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
    }
    if int(num_workers) > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        loader_kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
    loader = DataLoader(**loader_kwargs)
    return loader, sequence_tensors, min_length if min_length is not None else 0


def prepare_sequence_tensors(
    y_np: np.ndarray,
    t_np: np.ndarray,
    dt: float,
    m_eff: float,
    device: torch.device,
    smoothing_cfg: SmoothingConfig | None = None,
    vel_np: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    y_arr = np.asarray(y_np, dtype=float)
    t_arr = np.asarray(t_np, dtype=float)
    if y_arr.shape[0] != t_arr.shape[0]:
        raise ValueError("Displacement and time arrays must have the same length.")
    if np.isfinite(dt):
        dt_value = float(dt)
    elif t_arr.size >= 2:
        dt_value = float(t_arr[1] - t_arr[0])
    else:
        dt_value = 1.0
    if smoothing_cfg is None:
        smoothing_cfg = SmoothingConfig()
    if vel_np is None:
        vel_arr = compute_velocity_numpy(
            y_arr,
            dt_value,
            use_savgol=smoothing_cfg.use_savgol_smoothing,
            savgol_window=smoothing_cfg.window_length,
            savgol_polyorder=smoothing_cfg.polyorder,
        )
    else:
        vel_arr = np.asarray(vel_np, dtype=float)
        if vel_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("Provided velocity array must have the same length as displacement.")
    # Keep dataset tensors on CPU and move per-batch in the training loop.
    # This avoids VRAM blow-ups and enables pinned-memory transfers on CUDA.
    y_tensor = torch.from_numpy(y_arr).float()
    vel_tensor = torch.from_numpy(np.ascontiguousarray(vel_arr)).float()
    t_tensor = torch.from_numpy(t_arr).float()
    return y_tensor, vel_tensor, t_tensor


def _prepare_reduced_velocity_series(
    ur_raw: np.ndarray | float,
    length: int,
    *,
    name: str,
) -> np.ndarray:
    if ur_raw is None:
        raise ValueError(f"{name} is missing reduced velocity 'U_r'.")
    ur_arr = np.asarray(ur_raw, dtype=float)
    if ur_arr.ndim == 0:
        return np.full((length,), float(ur_arr), dtype=float)
    ur_flat = ur_arr.reshape(-1)
    ur_val = float(ur_flat[0])
    if not np.allclose(ur_flat, ur_val, rtol=1e-6, atol=1e-9):
        raise ValueError(f"{name} reduced velocity must be constant within a series.")
    if ur_flat.shape[0] != length:
        return np.full((length,), ur_val, dtype=float)
    return np.full((length,), ur_val, dtype=float)


def resolve_td_correction_params(raw_cfg: dict[str, Any] | None) -> dict[str, float]:
    cfg = dict(raw_cfg or {})
    keys = {
        "Cv": "td_cv",
        "Cd": "td_cd",
        "Ca": "td_ca",
        "fhat0": "td_fhat0",
        "fhat_min": "td_fhat_min",
        "fhat_max": "td_fhat_max",
        "n_memory": "td_n_memory",
    }
    out: dict[str, float] = {}
    missing = [name for name, key in keys.items() if key not in cfg]
    if missing:
        try:
            td_hidden = importlib.import_module("Data_Gen.td_hidden_state")
        except ModuleNotFoundError:
            td_hidden = importlib.import_module("td_hidden_state")
        defaults = td_hidden.build_single_paramset_from_burnin_config()
        try:
            burnin = importlib.import_module("Data_Gen.analyze_vivana_td_burnin")
        except ModuleNotFoundError:
            burnin = importlib.import_module("analyze_vivana_td_burnin")
        for name in ("Cv", "Cd", "Ca", "fhat0", "fhat_min", "fhat_max"):
            out[name] = float(cfg.get(keys[name], defaults[name]))
        out["n_memory"] = float(cfg.get("td_n_memory", getattr(burnin, "N_MEMORY", 500)))
    else:
        for name, key in keys.items():
            out[name] = float(cfg[key])
    if not (out["fhat_min"] <= out["fhat0"] <= out["fhat_max"]):
        raise ValueError("Require td_fhat_min <= td_fhat0 <= td_fhat_max.")
    if out["n_memory"] < 1.0:
        raise ValueError("td_n_memory must be >= 1.")
    return out


def _extract_first_present(data: Any, keys: Sequence[str], *, path: Path, required: bool = True) -> np.ndarray | None:
    for key in keys:
        if key in data:
            return np.asarray(data[key])
    if required:
        raise KeyError(f"{path} is missing required keys {list(keys)}.")
    return None


def _extract_required_scalar(data: Any, keys: Sequence[str], *, path: Path) -> float:
    value = _extract_first_present(data, keys, path=path, required=True)
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size != 1 or not np.isfinite(arr[0]):
        raise ValueError(f"{path} key(s) {list(keys)} must resolve to one finite scalar.")
    return float(arr[0])


def _maybe_reduce_time_td(
    *,
    t: np.ndarray,
    arrays: dict[str, np.ndarray],
    enabled: bool,
    reduction_factor: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if not enabled:
        return t, arrays
    rf_requested = max(1, int(reduction_factor))
    rf = min(rf_requested, max(1, int(t.size) - 1))
    if rf < rf_requested:
        warnings.warn(
            "Reduction factor was too large for the current TD trajectory; "
            f"using step={rf} instead of {rf_requested} to keep at least two samples."
        )
    sl = slice(None, None, rf)
    t_out = t[sl]
    arrays_out = {name: np.asarray(value)[sl] for name, value in arrays.items()}
    if t_out.size < 2:
        raise ValueError("reduce_time produced too few TD-correction samples")
    return t_out, arrays_out


def load_td_correction_trajectories(
    *,
    paths: Sequence[Path],
    cut_start_seconds: float = 0.0,
    reduce_time: bool = False,
    reduction_factor: int = 1,
    ur_source: str = "stored",
) -> list[dict[str, np.ndarray]]:
    trajectories: list[dict[str, np.ndarray]] = []
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            t = np.asarray(
                _extract_first_present(data, ("time_dim", "a", "time"), path=path),
                dtype=float,
            ).reshape(-1)
            y = np.asarray(
                _extract_first_present(data, ("y_disp_dim", "b", "y"), path=path),
                dtype=float,
            ).reshape(-1)
            dy = np.asarray(
                _extract_first_present(data, ("y_vel_dim", "dy", "e", "v"), path=path),
                dtype=float,
            ).reshape(-1)
            ddy = np.asarray(
                _extract_first_present(data, ("y_acc_dim", "ddy"), path=path),
                dtype=float,
            ).reshape(-1)
            force_total = np.asarray(
                _extract_first_present(data, ("y_force_dim", "c", "F_total", "force_total", "force"), path=path),
                dtype=float,
            ).reshape(-1)
            force_td = np.asarray(
                _extract_first_present(data, ("F_total_td",), path=path),
                dtype=float,
            ).reshape(-1)
            phi_td = np.asarray(_extract_first_present(data, ("phi_vy_td",), path=path), dtype=float).reshape(-1)
            sig_dy_td = np.asarray(_extract_first_present(data, ("sig_dy_loc_td",), path=path), dtype=float).reshape(-1)
            sig_ddy_td = np.asarray(_extract_first_present(data, ("sig_ddy_loc_td",), path=path), dtype=float).reshape(-1)
            stiffness_n_m = _extract_required_scalar(data, ("stiffness_n_m",), path=path)
            effective_mass_kg = _extract_required_scalar(data, ("effective_mass_kg",), path=path)
            dry_mass_kg = _extract_required_scalar(data, ("dry_mass_kg",), path=path)
            damping_c = _extract_required_scalar(data, ("damping_c",), path=path)
            diameter_m = _extract_required_scalar(data, ("diameter_m",), path=path)
            flow_speed_raw = _extract_first_present(data, ("flow_speed_m_s",), path=path, required=False)
            ur_raw = _extract_first_present(data, ("U_r_computed_series", "U_r"), path=path)
            ur_label_raw = _extract_first_present(
                data,
                ("U_r_label_series", "U_r_label_scalar"),
                path=path,
                required=False,
            )
        arrays = [t, y, dy, ddy, force_total, force_td, phi_td, sig_dy_td, sig_ddy_td]
        n = t.shape[0]
        if any(arr.shape[0] != n for arr in arrays[1:]):
            raise ValueError(f"{path} has mismatched TD correction array lengths.")
        if n < 2:
            raise ValueError(f"{path} is too short for TD correction training.")
        dt = float(t[1] - t[0])
        if not np.allclose(np.diff(t), dt, rtol=1e-6, atol=1e-9):
            raise ValueError(f"{path} time vector is not uniform.")
        ur_stored = _prepare_reduced_velocity_series(ur_raw, n, name=str(path))
        if flow_speed_raw is None:
            flow_speed = np.full((n,), np.nan, dtype=float)
        else:
            flow_arr = np.asarray(flow_speed_raw, dtype=float)
            if flow_arr.ndim == 0:
                flow_speed = np.full((n,), float(flow_arr), dtype=float)
            else:
                flow_speed = flow_arr.reshape(-1)
                if flow_speed.shape[0] == 1:
                    flow_speed = np.full((n,), float(flow_speed[0]), dtype=float)
                elif flow_speed.shape[0] != n:
                    raise ValueError(f"{path} flow speed must be scalar or length-matched to time.")
        if not np.all(np.isfinite(flow_speed)):
            flow_speed = np.full((n,), float(np.nanmean(flow_speed)), dtype=float)
        ur_label = None
        if ur_label_raw is not None:
            ur_label = _prepare_reduced_velocity_series(ur_label_raw, n, name=f"{path} U_r_label")
        ur_key = str(ur_source).strip().lower()
        if ur_key == "stored":
            ur = ur_stored
        elif ur_key in {"dry", "effective"}:
            mass_value = float(dry_mass_kg) if ur_key == "dry" else float(effective_mass_kg)
            if not np.isfinite(float(diameter_m)) or float(diameter_m) <= 0.0:
                raise ValueError(f"{path} requires a positive diameter_m to derive U_r.")
            if not np.isfinite(mass_value) or mass_value <= 0.0:
                raise ValueError(f"{path} requires a positive {ur_key}_mass_kg to derive U_r.")
            omega_n = np.sqrt(float(stiffness_n_m) / mass_value)
            f_n = float(omega_n / (2.0 * np.pi))
            ur = np.asarray(flow_speed, dtype=float) / max(np.finfo(float).eps, f_n * float(diameter_m))
        elif ur_key == "label":
            if ur_label is None:
                raise ValueError(f"{path} does not contain U_r label values.")
            ur = ur_label
        else:
            raise ValueError("ur_source must be one of: stored, dry, effective, label.")
        t, reduced = _maybe_reduce_time_td(
            t=t,
            arrays={
                "y": y,
                "dy": dy,
                "ddy": ddy,
                "force_total": force_total,
                "force_td": force_td,
                "phi_td": phi_td,
                "sig_dy_td": sig_dy_td,
                "sig_ddy_td": sig_ddy_td,
                "ur": ur,
                "ur_stored": ur_stored,
                **({"ur_label": ur_label} if ur_label is not None else {}),
                "flow_speed": flow_speed,
            },
            enabled=bool(reduce_time),
            reduction_factor=int(reduction_factor),
        )
        y = reduced["y"]
        dy = reduced["dy"]
        ddy = reduced["ddy"]
        force_total = reduced["force_total"]
        force_td = reduced["force_td"]
        phi_td = reduced["phi_td"]
        sig_dy_td = reduced["sig_dy_td"]
        sig_ddy_td = reduced["sig_ddy_td"]
        ur = reduced["ur"]
        ur_stored = reduced["ur_stored"]
        if ur_label is not None:
            ur_label = reduced["ur_label"]
        flow_speed = reduced["flow_speed"]
        if cut_start_seconds > 0.0:
            t0 = float(t[0])
            mask = t >= (t0 + float(cut_start_seconds))
            if int(np.count_nonzero(mask)) < 2:
                raise ValueError(f"{path} became too short after cut_start_seconds={cut_start_seconds}.")
            t = t[mask]
            y = y[mask]
            dy = dy[mask]
            ddy = ddy[mask]
            force_total = force_total[mask]
            force_td = force_td[mask]
            phi_td = phi_td[mask]
            sig_dy_td = sig_dy_td[mask]
            sig_ddy_td = sig_ddy_td[mask]
            ur = ur[mask]
            ur_stored = ur_stored[mask]
            if ur_label is not None:
                ur_label = ur_label[mask]
            flow_speed = flow_speed[mask]
        td_context = np.stack([ddy, phi_td, sig_dy_td, sig_ddy_td, flow_speed], axis=1)
        trajectories.append(
            {
                "name": path.name,
                "t": np.asarray(t, dtype=np.float32),
                "y": np.asarray(y, dtype=np.float32),
                "dy": np.asarray(dy, dtype=np.float32),
                "ddy": np.asarray(ddy, dtype=np.float32),
                "force_total": np.asarray(force_total, dtype=np.float32),
                "force_td": np.asarray(force_td, dtype=np.float32),
                "force_corr": np.asarray(force_total - force_td, dtype=np.float32),
                "ur": np.asarray(ur, dtype=np.float32),
                "ur_stored": np.asarray(ur_stored, dtype=np.float32),
                "ur_label": (None if ur_label is None else np.asarray(ur_label, dtype=np.float32)),
                "flow_speed": np.asarray(flow_speed, dtype=np.float32),
                "td_context": np.asarray(td_context, dtype=np.float32),
                "stiffness_n_m": np.asarray(stiffness_n_m, dtype=np.float32),
                "effective_mass_kg": np.asarray(effective_mass_kg, dtype=np.float32),
                "dry_mass_kg": np.asarray(dry_mass_kg, dtype=np.float32),
                "damping_c": np.asarray(damping_c, dtype=np.float32),
            }
        )
    if not trajectories:
        raise ValueError("No TD correction trajectories were loaded.")
    return trajectories


def td_baseline_step_torch(
    *,
    velocity: torch.Tensor,
    acceleration: torch.Tensor,
    td_context: torch.Tensor,
    dt: float | torch.Tensor,
    rho: float,
    diameter: float,
    params: dict[str, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    if td_context.ndim != 2 or td_context.shape[1] < 5:
        raise ValueError("td_context must have shape (B, 5) with [ddy, phi, sig_dy, sig_ddy, flow_speed].")
    ddy = td_context[:, 0:1]
    phi_vy = td_context[:, 1:2]
    sig_dy = td_context[:, 2:3]
    sig_ddy = td_context[:, 3:4]
    flow_speed = td_context[:, 4:5]
    n_memory = max(1.0, float(params["n_memory"]))
    dt_t = torch.as_tensor(dt, device=velocity.device, dtype=velocity.dtype)
    if dt_t.ndim == 0:
        dt_t = dt_t.view(1, 1)
    elif dt_t.ndim == 1:
        dt_t = dt_t.view(-1, 1)
    dt_t = dt_t.expand_as(velocity)

    speed_mag = torch.sqrt(torch.clamp(flow_speed * flow_speed + velocity * velocity, min=1e-12))
    projection = flow_speed / speed_mag
    dy_r = velocity * projection
    ddy_r = ddy * projection

    sig_dy_next = torch.sqrt(torch.clamp(((n_memory - 1.0) / n_memory) * (sig_dy * sig_dy) + (dy_r * dy_r) / n_memory, min=1e-12))
    sig_ddy_next = torch.sqrt(torch.clamp(((n_memory - 1.0) / n_memory) * (sig_ddy * sig_ddy) + (ddy_r * ddy_r) / n_memory, min=1e-12))

    cos_phi_dy = dy_r / torch.clamp(sig_dy_next, min=1e-12)
    sin_phi_dy = -ddy_r / torch.clamp(sig_ddy_next, min=1e-12)
    phi_dy = torch.atan2(sin_phi_dy, cos_phi_dy)

    theta = phi_dy - phi_vy
    theta = torch.atan2(torch.sin(theta), torch.cos(theta))
    fhat = torch.where(
        theta <= 0.0,
        float(params["fhat0"]) + (float(params["fhat0"]) - float(params["fhat_min"])) * torch.sin(theta),
        float(params["fhat0"]) + (float(params["fhat_max"]) - float(params["fhat0"])) * torch.sin(theta),
    )
    omega_vy = 2.0 * math.pi * fhat * speed_mag / float(diameter)
    phi_vy_next = phi_vy + dt_t * omega_vy

    fdy = -0.5 * float(rho) * float(diameter) * float(params["Cd"]) * speed_mag * velocity
    fcv = 0.5 * float(rho) * float(diameter) * float(params["Cv"]) * speed_mag * flow_speed * torch.cos(phi_vy_next)
    fca = -0.25 * float(rho) * float(params["Ca"]) * math.pi * (float(diameter) ** 2) * acceleration
    force_total = fca + fcv + fdy

    next_context = torch.cat([acceleration, phi_vy_next, sig_dy_next, sig_ddy_next, flow_speed], dim=1)
    return force_total, next_context


def structural_step_constant_force_torch(
    *,
    y: torch.Tensor,
    velocity: torch.Tensor,
    force: torch.Tensor,
    dt: float | torch.Tensor,
    mass: torch.Tensor | float,
    damping_c: torch.Tensor | float,
    stiffness: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dt_t = torch.as_tensor(dt, device=y.device, dtype=y.dtype)
    if dt_t.ndim == 0:
        dt_t = dt_t.view(1, 1)
    elif dt_t.ndim == 1:
        dt_t = dt_t.view(-1, 1)
    half = y.new_tensor(0.5)
    sixth = y.new_tensor(1.0 / 6.0)
    mass_t = torch.as_tensor(mass, device=y.device, dtype=y.dtype)
    damping_t = torch.as_tensor(damping_c, device=y.device, dtype=y.dtype)
    stiffness_t = torch.as_tensor(stiffness, device=y.device, dtype=y.dtype)
    if mass_t.ndim == 0:
        mass_t = mass_t.view(1, 1)
    elif mass_t.ndim == 1:
        mass_t = mass_t.view(-1, 1)
    if damping_t.ndim == 0:
        damping_t = damping_t.view(1, 1)
    elif damping_t.ndim == 1:
        damping_t = damping_t.view(-1, 1)
    if stiffness_t.ndim == 0:
        stiffness_t = stiffness_t.view(1, 1)
    elif stiffness_t.ndim == 1:
        stiffness_t = stiffness_t.view(-1, 1)
    dt_t = dt_t.expand_as(y)
    mass_t = mass_t.expand_as(y)
    damping_t = damping_t.expand_as(y)
    stiffness_t = stiffness_t.expand_as(y)

    def accel(y_state: torch.Tensor, v_state: torch.Tensor) -> torch.Tensor:
        return (force - damping_t * v_state - stiffness_t * y_state) / mass_t

    k1_y = velocity
    k1_v = accel(y, velocity)

    y2 = y + half * dt_t * k1_y
    v2 = velocity + half * dt_t * k1_v
    k2_y = v2
    k2_v = accel(y2, v2)

    y3 = y + half * dt_t * k2_y
    v3 = velocity + half * dt_t * k2_v
    k3_y = v3
    k3_v = accel(y3, v3)

    y4 = y + dt_t * k3_y
    v4 = velocity + dt_t * k3_v
    k4_y = v4
    k4_v = accel(y4, v4)

    y_next = y + (dt_t * sixth) * (k1_y + 2.0 * k2_y + 2.0 * k3_y + k4_y)
    v_next = velocity + (dt_t * sixth) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
    a_next = accel(y_next, v_next)
    return y_next, v_next, a_next


def _normalize_ur_filter(
    values: Sequence[float] | np.ndarray | float | None,
    *,
    name: str,
) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return np.empty((0,), dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _ur_in_filter(
    ur_value: float,
    ur_filter: np.ndarray,
    *,
    tol: float,
) -> bool:
    if ur_filter.size == 0:
        return False
    return bool(np.any(np.isclose(float(ur_value), ur_filter, rtol=0.0, atol=float(tol))))


def load_training_series(
    y_eval: np.ndarray,
    t_eval: np.ndarray,
    dt_eval: float,
    use_generated: bool,
    series_dir: Path,
    m_eff: float,
    device: torch.device,
    smoothing_cfg: SmoothingConfig | None = None,
    *,
    velocity_source: str = "compute",
    eval_velocity: np.ndarray | None = None,
    eval_reduced_velocity: np.ndarray | float | None = None,
    velocity_key_candidates: Sequence[str] = ("e", "dy", "v"),
    require_force: bool = False,
    eval_force: np.ndarray | None = None,
    force_key_candidates: Sequence[str] = ("c", "F_total", "force_total", "force"),
    cut_start_seconds: float = 0.0,
    include_reduced_velocity: Sequence[float] | np.ndarray | float | None = None,
    exclude_reduced_velocity: Sequence[float] | np.ndarray | float | None = None,
    ur_filter_tol: float = 1e-6,
) -> tuple[
    list[tuple[np.ndarray, np.ndarray, float, np.ndarray | None, np.ndarray | None, np.ndarray]],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]:
    train_series_raw: list[tuple[np.ndarray, np.ndarray, float, np.ndarray | None, np.ndarray | None, np.ndarray]] = []
    vel_source = str(velocity_source).strip().lower()
    if vel_source not in {"compute", "file", "auto"}:
        raise ValueError("velocity_source must be one of: compute, file, auto")
    if vel_source == "file" and eval_velocity is None:
        raise ValueError("velocity_source is 'file' but no eval_velocity was provided.")
    if require_force and eval_force is None and not use_generated:
        raise ValueError("require_force is True but no eval_force was provided for the eval series.")
    if use_generated:
        if not series_dir.exists():
            raise FileNotFoundError(f"Training series directory '{series_dir}' does not exist.")
        series_files = sorted(series_dir.glob("*.npz"))
        if not series_files:
            raise FileNotFoundError(f"No '.npz' files found in training series directory '{series_dir}'.")
        include_ur = _normalize_ur_filter(include_reduced_velocity, name="include_reduced_velocity")
        exclude_ur = _normalize_ur_filter(exclude_reduced_velocity, name="exclude_reduced_velocity")
        tol = float(ur_filter_tol)
        if tol < 0.0:
            raise ValueError("ur_filter_tol must be non-negative.")
        cut_start_seconds = max(0.0, float(cut_start_seconds))
        for series_file in series_files:
            with np.load(series_file) as series_data:
                series_t = np.asarray(series_data["a"])
                series_y = np.asarray(series_data["b"])
                series_vel: np.ndarray | None = None
                if vel_source in {"file", "auto"}:
                    for key in velocity_key_candidates:
                        if key in series_data:
                            series_vel = np.asarray(series_data[key])
                            break
                    if series_vel is None and vel_source == "file":
                        raise KeyError(
                            f"Series '{series_file}' is missing velocity. Tried keys: {list(velocity_key_candidates)}"
                        )
                series_force: np.ndarray | None = None
                if require_force:
                    for key in force_key_candidates:
                        if key in series_data:
                            series_force = np.asarray(series_data[key])
                            break
                    if series_force is None:
                        raise KeyError(
                            f"Series '{series_file}' is missing force data. Tried keys: {list(force_key_candidates)}"
                        )
                if "U_r" not in series_data:
                    raise KeyError(f"Series '{series_file}' is missing reduced velocity 'U_r'.")
                series_ur = _prepare_reduced_velocity_series(series_data["U_r"], series_t.shape[0], name=str(series_file))
                ur_val = float(np.asarray(series_ur).reshape(-1)[0])
            if include_ur is not None and include_ur.size > 0 and not _ur_in_filter(ur_val, include_ur, tol=tol):
                continue
            if exclude_ur is not None and exclude_ur.size > 0 and _ur_in_filter(ur_val, exclude_ur, tol=tol):
                continue
            if series_t.ndim != 1 or series_y.ndim != 1:
                raise ValueError(f"Series '{series_file}' must contain 1D 'a' and 'b' arrays.")
            if series_t.shape[0] != series_y.shape[0]:
                raise ValueError(f"Series '{series_file}' has mismatched lengths.")
            if series_t.shape[0] < 2:
                raise ValueError(f"Series '{series_file}' is too short to build training samples.")
            if series_vel is not None and series_vel.shape[0] != series_t.shape[0]:
                raise ValueError(f"Series '{series_file}' has mismatched velocity length.")
            if series_force is not None and series_force.shape[0] != series_t.shape[0]:
                raise ValueError(f"Series '{series_file}' has mismatched force length.")
            series_dt = float(series_t[1] - series_t[0])
            if not np.allclose(np.diff(series_t), series_dt, rtol=1e-6, atol=1e-9):
                raise ValueError(f"Series '{series_file}' time vector is not uniform.")
            if not np.isclose(series_dt, dt_eval, rtol=1e-6, atol=1e-9):
                series_y, series_t_resampled = resample_uniform_series(series_t, series_y, dt_eval)
                if series_vel is not None:
                    series_vel = np.interp(series_t_resampled, series_t, series_vel)
                if series_force is not None:
                    series_force = np.interp(series_t_resampled, series_t, series_force)
                series_ur = np.full((series_t_resampled.shape[0],), ur_val, dtype=float)
                series_t = series_t_resampled
                series_dt = dt_eval
            if cut_start_seconds > 0.0:
                series_t0 = float(series_t[0])
                cut_mask = series_t >= (series_t0 + cut_start_seconds)
                series_t = series_t[cut_mask]
                series_y = series_y[cut_mask]
                if series_vel is not None:
                    series_vel = np.asarray(series_vel)[cut_mask]
                if series_force is not None:
                    series_force = np.asarray(series_force)[cut_mask]
                series_ur = np.full((series_t.shape[0],), ur_val, dtype=float)
                if series_t.shape[0] < 2:
                    raise ValueError(
                        f"Series '{series_file}' became too short after cut_start_seconds={cut_start_seconds}."
                    )
                series_dt = float(series_t[1] - series_t[0])
            if series_force is None:
                train_series_raw.append((series_y, series_t, series_dt, series_vel, None, series_ur))
            else:
                train_series_raw.append((series_y, series_t, series_dt, series_vel, series_force, series_ur))
        if not train_series_raw:
            raise ValueError(
                "No training series left after U_r filtering. "
                f"include_reduced_velocity={include_reduced_velocity}, "
                f"exclude_reduced_velocity={exclude_reduced_velocity}, "
                f"series_dir='{series_dir}'."
            )
    else:
        train_series_raw.append(
            (
                y_eval,
                t_eval,
                dt_eval,
                eval_velocity if vel_source in {"file", "auto"} else None,
                np.asarray(eval_force) if require_force else None,
                _prepare_reduced_velocity_series(eval_reduced_velocity, t_eval.shape[0], name="eval series"),
            )
        )

    eval_tensors = prepare_sequence_tensors(
        y_eval,
        t_eval,
        dt_eval,
        m_eff,
        device,
        smoothing_cfg=smoothing_cfg,
        vel_np=eval_velocity if vel_source in {"file", "auto"} else None,
    )
    eval_ur = _prepare_reduced_velocity_series(eval_reduced_velocity, t_eval.shape[0], name="eval series")
    eval_ur_tensor = torch.from_numpy(np.ascontiguousarray(eval_ur)).float()
    return train_series_raw, (*eval_tensors, eval_ur_tensor)


def build_dataset(
    y_data_t: torch.Tensor,
    vel: torch.Tensor,
    m_eff: float,
    t_tensor: torch.Tensor,
    *,
    reduced_velocity: torch.Tensor,
    force_tensor: torch.Tensor | None = None,
    history_window: int | None = None,
) -> TensorDataset:
    """Construct consecutive state/time pairs for training (optionally with force labels)."""
    z = torch.stack((y_data_t, vel * m_eff), dim=1)
    ur = reduced_velocity
    if ur.ndim == 1:
        ur = ur.unsqueeze(1)
    if ur.ndim != 2 or ur.shape[1] != 1:
        raise ValueError("Reduced velocity tensor must have shape (T,) or (T, 1).")
    if ur.shape[0] != z.shape[0]:
        raise ValueError("Reduced velocity tensor must match the sequence length.")
    history_tensor: torch.Tensor | None = None
    if history_window is not None and int(history_window) > 0:
        history_features = torch.cat([z, ur], dim=1)
        history_full = _build_causal_feature_windows(history_features, int(history_window))
        history_tensor = history_full[:-1]
    if force_tensor is None:
        if history_tensor is not None:
            return TensorDataset(
                z[:-1],
                t_tensor[:-1].unsqueeze(1),
                z[1:],
                t_tensor[1:].unsqueeze(1),
                ur[:-1],
                history_tensor,
            )
        return TensorDataset(
            z[:-1],
            t_tensor[:-1].unsqueeze(1),
            z[1:],
            t_tensor[1:].unsqueeze(1),
            ur[:-1],
        )
    if force_tensor.shape[0] != z.shape[0]:
        raise ValueError("force_tensor must match the sequence length.")
    if history_tensor is not None:
        return TensorDataset(
            z[:-1],
            t_tensor[:-1].unsqueeze(1),
            z[1:],
            t_tensor[1:].unsqueeze(1),
            ur[:-1],
            history_tensor,
            force_tensor[:-1].unsqueeze(1),
            force_tensor[1:].unsqueeze(1),
        )
    return TensorDataset(
        z[:-1],
        t_tensor[:-1].unsqueeze(1),
        z[1:],
        t_tensor[1:].unsqueeze(1),
        ur[:-1],
        force_tensor[:-1].unsqueeze(1),
        force_tensor[1:].unsqueeze(1),
    )

def build_rollout_dataset(
    y_data_t: torch.Tensor,
    vel: torch.Tensor,
    m_eff: float,
    t_tensor: torch.Tensor,
    rollout_steps: int,
    *,
    reduced_velocity: torch.Tensor,
    history_window: int | None = None,
) -> TensorDataset:
    """
    Build sliding-window sequences matching the inputs expected by `PHVIV.rollout`
    and the targets required by `traj_loss`.

    Each sample contains:
        - z0: initial state (y, p) at the window start
        - t_seq: absolute times for the window (length rollout_steps + 1)
        - z_traj: ground-truth state trajectory over the same window
    """
    if rollout_steps < 1:
        raise ValueError("rollout_steps must be at least 1")

    z = torch.stack((y_data_t, vel * m_eff), dim=1)  # (T, 2)
    window = rollout_steps + 1
    total_samples = z.shape[0]
    if total_samples < window:
        raise ValueError("Not enough samples to build rollout windows of the requested length")

    num_windows = total_samples - window + 1
    z0_list = []
    t_seq_list = []
    z_traj_list = []

    for start in range(num_windows):
        end = start + window
        z_window = z[start:end]                  # (window, 2)
        t_window = t_tensor[start:end]           # (window,)
        z0_list.append(z_window[0])
        t_seq_list.append(t_window)
        z_traj_list.append(z_window)

    z0_batch = torch.stack(z0_list, dim=0)                  # (B, 2)
    t_seq_batch = torch.stack(t_seq_list, dim=0)            # (B, window)
    z_traj_batch = torch.stack(z_traj_list, dim=0)          # (B, window, 2)

    ur = reduced_velocity
    if ur.ndim == 1:
        ur = ur.unsqueeze(1)
    if ur.ndim != 2 or ur.shape[1] != 1:
        raise ValueError("Reduced velocity tensor must have shape (T,) or (T, 1).")
    if ur.shape[0] != z.shape[0]:
        raise ValueError("Reduced velocity tensor must match the sequence length.")
    ur0_batch = ur[:num_windows]
    history0_batch: torch.Tensor | None = None
    if history_window is not None and int(history_window) > 0:
        history_features = torch.cat([z, ur], dim=1)
        history_full = _build_causal_feature_windows(history_features, int(history_window))
        history0_batch = history_full[:num_windows]
    if history0_batch is not None:
        return TensorDataset(z0_batch, t_seq_batch, z_traj_batch, ur0_batch, history0_batch)
    return TensorDataset(z0_batch, t_seq_batch, z_traj_batch, ur0_batch)


def build_rollout_dataloader_from_series(
    series_data: list[tuple[np.ndarray, np.ndarray, float, np.ndarray | None, np.ndarray | None, np.ndarray]],
    m_eff: float,
    batch_size: int,
    device: torch.device,
    smoothing_cfg: SmoothingConfig | None = None,
    *,
    rollout_steps: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
    history_window: int | None = None,
) -> tuple[DataLoader, int]:
    if rollout_steps < 1:
        raise ValueError("rollout_steps must be at least 1.")
    if not series_data:
        raise ValueError("series_data must contain at least one series.")
    datasets: list[TensorDataset | ConcatDataset] = []
    total_windows = 0
    for y_np, t_np, dt_value, vel_np, force_np, ur_np in series_data:
        y_tensor, vel_tensor, t_tensor = prepare_sequence_tensors(
            y_np,
            t_np,
            dt_value,
            m_eff,
            device,
            smoothing_cfg=smoothing_cfg,
            vel_np=vel_np,
        )
        ur_arr = np.asarray(ur_np, dtype=float)
        if ur_arr.ndim == 0:
            ur_arr = np.full((y_tensor.shape[0],), float(ur_arr), dtype=float)
        else:
            ur_arr = ur_arr.reshape(-1)
        if ur_arr.shape[0] != y_tensor.shape[0]:
            raise ValueError("Reduced velocity array must have the same length as displacement.")
        ur_tensor = torch.from_numpy(np.ascontiguousarray(ur_arr)).float()

        dataset = build_rollout_dataset(
            y_tensor,
            vel_tensor,
            m_eff,
            t_tensor,
            int(rollout_steps),
            reduced_velocity=ur_tensor,
            history_window=history_window,
        )
        total_windows += len(dataset)
        datasets.append(dataset)

    combined = combine_datasets(datasets)
    loader_kwargs: dict[str, object] = {
        "dataset": combined,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
    }
    if int(num_workers) > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        loader_kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
    loader = DataLoader(**loader_kwargs)
    return loader, int(total_windows)

def create_zoom_mask(t: np.ndarray, window: float = 1.0) -> np.ndarray | slice:
    mask = (t - t[0]) <= window
    return mask if np.count_nonzero(mask) > 1 else slice(None)

def create_window_mask(t: np.ndarray, time_window: tuple[float, float] | list[float]) -> np.ndarray | slice:
    start, end = time_window
    mask = (t >= start) & (t <= end)
    return mask if np.count_nonzero(mask) > 1 else slice(None)


def resample_uniform_series(
    series_t: np.ndarray,
    series_y: np.ndarray,
    target_dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a uniformly sampled series onto a new step using interpolation."""
    if series_t.ndim != 1 or series_y.ndim != 1:
        raise ValueError("Input series must be 1D arrays")
    if series_t.size != series_y.size:
        raise ValueError("Time and value arrays must have matching lengths")
    if series_t.size < 2:
        raise ValueError("Need at least two samples to resample a series")
    t_start = float(series_t[0])
    t_end = float(series_t[-1])
    if target_dt <= 0.0:
        raise ValueError("target_dt must be positive")
    duration = t_end - t_start
    if duration <= 0.0:
        raise ValueError("Time vector must span a positive duration")
    num_steps = int(np.floor(duration / target_dt))
    if num_steps < 1:
        raise ValueError("target_dt is larger than the available duration")
    resampled_t = t_start + np.arange(num_steps + 1) * target_dt
    # Ensure the resampled grid does not extend past the original support
    while resampled_t[-1] - t_end > 1e-9:
        resampled_t = resampled_t[:-1]
        if resampled_t.size < 2:
            raise ValueError("Resampled grid became too small")
    resampled_y = np.interp(resampled_t, series_t, series_y)
    return resampled_y, resampled_t


def _force_component_coefficients(
    model: "PHVIV",
    state: torch.Tensor,
    reduced_velocity: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    base_features = model._force_features(state, reduced_velocity=reduced_velocity)
    features = model.force_embed(base_features) if model.force_embed is not None else base_features
    base_raw = model.u_base_net(features)
    delta_raw = torch.zeros_like(base_raw)

    force_scale = model._force_scale_from_reduced_velocity(reduced_velocity, like=base_raw, state=state)
    if model.force_output == "coefficient":
        base_coeff = base_raw
        delta_coeff = delta_raw
    else:
        coeff_scale = (float(model.k) * float(model.D)) / torch.clamp(force_scale, min=1e-12)
        base_coeff = base_raw * coeff_scale
        delta_coeff = delta_raw * coeff_scale

    total_coeff = base_coeff + delta_coeff
    if model.include_physical_drag:
        total_coeff = total_coeff + model.drag_force_coeff(state, reduced_velocity=reduced_velocity)

    if model.use_stochastic_process_noise:
        sigma_force = model.sigma_theta(state, reduced_velocity=reduced_velocity)
        sigma_coeff = sigma_force / torch.clamp(force_scale, min=1e-12)
    else:
        sigma_coeff = torch.zeros_like(total_coeff)
    return total_coeff, delta_coeff, sigma_coeff

def rollout_model(
    model: PHVIV,
    y0: torch.Tensor,
    vel: torch.Tensor,
    reduced_velocity: torch.Tensor | np.ndarray | float,
    m_eff: float,
    dt: float,
    t: np.ndarray,
    D: float,
    k: float,
    device: torch.device,
    *,
    stochastic: bool = False,
    rollout_seed: int | None = None,
    noise_scale: float = 1.0,
) -> dict[str, np.ndarray]:
    """Roll the model forward over the full time grid and return normalised traces."""
    total_steps = int(min(len(t), int(y0.shape[0]), int(vel.shape[0])))
    if total_steps < 1:
        raise ValueError("rollout_model requires at least one time step.")
    y_series = y0[:total_steps].to(device=device)
    vel_series = vel[:total_steps].to(device=device)
    if torch.is_tensor(reduced_velocity):
        rv_series = reduced_velocity.to(device=device, dtype=y_series.dtype)
    else:
        rv_series = torch.as_tensor(reduced_velocity, device=device, dtype=y_series.dtype)
    if rv_series.ndim == 0:
        rv_series = rv_series.view(1, 1).expand(total_steps, 1)
    else:
        rv_series = rv_series.reshape(-1, 1)
        if rv_series.shape[0] == 1:
            rv_series = rv_series.expand(total_steps, 1)
        elif rv_series.shape[0] != total_steps:
            raise ValueError("Reduced velocity series must be scalar or match the rollout length.")

    observed_states = torch.stack((y_series, vel_series * float(m_eff)), dim=1)
    state = observed_states[0:1]
    y_samples: list[torch.Tensor] = []
    p_samples: list[torch.Tensor] = []
    force_total: list[torch.Tensor] = []
    force_coeff_total: list[torch.Tensor] = []
    force_coeff_delta: list[torch.Tensor] = []
    force_coeff_sigma: list[torch.Tensor] = []
    force_drag: list[torch.Tensor] = []
    force_model: list[torch.Tensor] = []
    hamiltonian_model_vals: list[torch.Tensor] = []
    generator: torch.Generator | None = None
    if rollout_seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(rollout_seed))
    with torch.no_grad():
        for step_idx in range(total_steps):
            rv_step = rv_series[step_idx : step_idx + 1]
            y_samples.append(state[0, 0].detach())
            p_samples.append(state[0, 1].detach())
            model_force = model.learned_force(state, reduced_velocity=rv_step).squeeze().detach()
            if model.include_physical_drag:
                drag_force = model.drag_force(state).squeeze().detach()
            else:
                drag_force = model_force.new_tensor(0.0)
            total_force = model.u_theta(state, reduced_velocity=rv_step).squeeze().detach()
            total_coeff, delta_coeff, sigma_coeff = _force_component_coefficients(model, state, rv_step)
            H_val = model.H(state).detach()
            force_model.append(model_force)
            force_drag.append(drag_force)
            force_total.append(total_force)
            force_coeff_total.append(total_coeff.squeeze().detach())
            force_coeff_delta.append(delta_coeff.squeeze().detach())
            force_coeff_sigma.append(sigma_coeff.squeeze().detach())
            hamiltonian_model_vals.append(H_val)
            if step_idx == total_steps - 1:
                break
            if stochastic and getattr(model, "use_stochastic_process_noise", False):
                noise = torch.randn(
                    state.shape[0],
                    1,
                    device=state.device,
                    dtype=state.dtype,
                    generator=generator,
                )
                state = model.step_rk4_stochastic(
                    state,
                    t,
                    dt,
                    reduced_velocity=rv_step,
                    noise=noise,
                    noise_scale=noise_scale,
                )
            else:
                state = model.step_rk4(state, t, dt, reduced_velocity=rv_step)

    y_samples_arr = torch.stack(y_samples).detach().cpu().numpy()
    p_samples_arr = torch.stack(p_samples).detach().cpu().numpy()
    y_pred_norm = y_samples_arr / D
    p_pred_norm = (p_samples_arr / m_eff) / (np.sqrt(k / m_eff) * D)
    force_total_arr = torch.stack(force_total).detach().cpu().numpy()
    force_coeff_total_arr = torch.stack(force_coeff_total).detach().cpu().numpy()
    force_coeff_delta_arr = torch.stack(force_coeff_delta).detach().cpu().numpy()
    force_coeff_sigma_arr = torch.stack(force_coeff_sigma).detach().cpu().numpy()
    force_drag_arr = torch.stack(force_drag).detach().cpu().numpy()
    force_model_arr = torch.stack(force_model).detach().cpu().numpy()
    hamiltonian_model_arr = torch.stack(hamiltonian_model_vals).detach().cpu().numpy()
    return {
        "y_norm": y_pred_norm,
        "p_norm": p_pred_norm,
        "force_total": force_total_arr,
        "force_coeff_total": force_coeff_total_arr,
        "force_coeff_delta": force_coeff_delta_arr,
        "force_coeff_sigma": force_coeff_sigma_arr,
        "force_drag": force_drag_arr,
        "force_model": force_model_arr,
        "hamiltonian_model": hamiltonian_model_arr,
        "history_context": None,
    }


def rollout_model_with_progress(
    model: PHVIV,
    y0: torch.Tensor,
    vel: torch.Tensor,
    reduced_velocity: torch.Tensor | np.ndarray | float,
    m_eff: float,
    dt: float,
    t: np.ndarray,
    D: float,
    k: float,
    device: torch.device,
    *,
    progress_callback: Callable[[int, int], None] | None = None,
    callback_every: int = 1,
) -> dict[str, np.ndarray]:
    """
    Same as rollout_model, but optionally emits progress callbacks.

    progress_callback(completed, total) is called every `callback_every` steps and
    on the final step.
    """
    every = max(1, int(callback_every))
    total_steps = int(min(len(t), int(y0.shape[0]), int(vel.shape[0])))
    if total_steps < 1:
        raise ValueError("rollout_model_with_progress requires at least one time step.")

    y_series = y0[:total_steps].to(device=device)
    vel_series = vel[:total_steps].to(device=device)
    if torch.is_tensor(reduced_velocity):
        rv_series = reduced_velocity.to(device=device, dtype=y_series.dtype)
    else:
        rv_series = torch.as_tensor(reduced_velocity, device=device, dtype=y_series.dtype)
    if rv_series.ndim == 0:
        rv_series = rv_series.view(1, 1).expand(total_steps, 1)
    else:
        rv_series = rv_series.reshape(-1, 1)
        if rv_series.shape[0] == 1:
            rv_series = rv_series.expand(total_steps, 1)
        elif rv_series.shape[0] != total_steps:
            raise ValueError("Reduced velocity series must be scalar or match the rollout length.")

    observed_states = torch.stack((y_series, vel_series * float(m_eff)), dim=1)
    state = observed_states[0:1]
    y_samples: list[torch.Tensor] = []
    p_samples: list[torch.Tensor] = []
    force_total: list[torch.Tensor] = []
    force_coeff_total: list[torch.Tensor] = []
    force_coeff_delta: list[torch.Tensor] = []
    force_coeff_sigma: list[torch.Tensor] = []
    force_drag: list[torch.Tensor] = []
    force_model: list[torch.Tensor] = []
    hamiltonian_model_vals: list[torch.Tensor] = []
    with torch.no_grad():
        for step_idx in range(total_steps):
            rv_step = rv_series[step_idx : step_idx + 1]
            y_samples.append(state[0, 0].detach())
            p_samples.append(state[0, 1].detach())
            model_force = model.learned_force(state, reduced_velocity=rv_step).squeeze().detach()
            if model.include_physical_drag:
                drag_force = model.drag_force(state).squeeze().detach()
            else:
                drag_force = model_force.new_tensor(0.0)
            total_force = model.u_theta(state, reduced_velocity=rv_step).squeeze().detach()
            total_coeff, delta_coeff, sigma_coeff = _force_component_coefficients(model, state, rv_step)
            H_val = model.H(state).detach()
            force_model.append(model_force)
            force_drag.append(drag_force)
            force_total.append(total_force)
            force_coeff_total.append(total_coeff.squeeze().detach())
            force_coeff_delta.append(delta_coeff.squeeze().detach())
            force_coeff_sigma.append(sigma_coeff.squeeze().detach())
            hamiltonian_model_vals.append(H_val)
            if step_idx < total_steps - 1:
                state = model.step_rk4(
                    state,
                    t,
                    dt,
                    reduced_velocity=rv_step,
                )

            completed = step_idx + 1
            if progress_callback is not None and (completed % every == 0 or completed == total_steps):
                progress_callback(completed, total_steps)

    y_samples_arr = torch.stack(y_samples).detach().cpu().numpy()
    p_samples_arr = torch.stack(p_samples).detach().cpu().numpy()
    y_pred_norm = y_samples_arr / D
    p_pred_norm = (p_samples_arr / m_eff) / (np.sqrt(k / m_eff) * D)
    force_total_arr = torch.stack(force_total).detach().cpu().numpy()
    force_coeff_total_arr = torch.stack(force_coeff_total).detach().cpu().numpy()
    force_coeff_delta_arr = torch.stack(force_coeff_delta).detach().cpu().numpy()
    force_coeff_sigma_arr = torch.stack(force_coeff_sigma).detach().cpu().numpy()
    force_drag_arr = torch.stack(force_drag).detach().cpu().numpy()
    force_model_arr = torch.stack(force_model).detach().cpu().numpy()
    hamiltonian_model_arr = torch.stack(hamiltonian_model_vals).detach().cpu().numpy()
    return {
        "y_norm": y_pred_norm,
        "p_norm": p_pred_norm,
        "force_total": force_total_arr,
        "force_coeff_total": force_coeff_total_arr,
        "force_coeff_delta": force_coeff_delta_arr,
        "force_coeff_sigma": force_coeff_sigma_arr,
        "force_drag": force_drag_arr,
        "force_model": force_model_arr,
        "hamiltonian_model": hamiltonian_model_arr,
        "history_context": None,
    }
