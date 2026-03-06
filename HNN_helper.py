import math
import warnings
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

try:
    from scipy.signal import savgol_filter
except ImportError:
    savgol_filter = None

from architectures import FourierFeatures, ODEPirateNet, TemporalConvForceNet

DISP_ROLLOUT_NRMSE_KEY = "Disp rollout NRMSE"
FORCE_ROLLOUT_NRMSE_KEY = "Force rollout NRMSE"
FORCE_MAPPING_NRMSE_KEY = "Force mapping NRMSE"
FORCE_ROLLOUT_NRMSE_COEFF_KEY = "Force rollout NRMSE (coeff)"
FORCE_MAPPING_NRMSE_COEFF_KEY = "Force mapping NRMSE (coeff)"
DOMINANT_FREQ_REL_ERROR_KEY = "Dominant frequency relative error"
MEAN_DISP_AMP_REL_ERROR_KEY = "Mean displacement amplitude relative error"
DISP_SPECTRAL_SHAPE_ERROR_KEY = "Disp spectral shape error"
FORCE_SPECTRAL_SHAPE_ERROR_KEY = "Force spectral shape error"

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
    force_output: str = "force"  # "force", "coefficient", or "two_head_vivana"
    learn_hamiltonian: bool = False
    discover_damping: bool = False
    use_pirate_force: bool = False
    pirate_force_kwargs: dict[str, Any] = field(default_factory=dict)
    use_fourier_features: bool = False
    fourier_features: int = 64
    fourier_sigma: float = 1.0
    use_feature_engineering: bool = False
    use_reduced_velocity: bool = True
    q_scale: float | None = None
    p_scale: float | None = None
    ur_scale: float | None = None
    # Optional coefficient bounding: C <- C_max * tanh(C / C_max)
    bound_force_coefficient: bool = False
    force_coefficient_c_max: float = 5.0

def _default_residual_kwargs() -> dict[str, Any]:
    return {"hidden": 128, "layers": 2, "activation": "gelu"}


def _default_mlp_kwargs() -> dict[str, Any]:
    return {"hidden": 100, "layers": 2, "activation": "gelu"}


def _default_tcn_kwargs() -> dict[str, Any]:
    return {
        "hidden": 128,
        "levels": 4,
        "dilation_start": 1,
        "kernel_size": 3,
        "dropout": 0.0,
        "activation": "gelu",
        "history_len": 64,
        "enabled": False,
        "use_as_backbone": False,
        "head_input_dim": None,
    }


@dataclass
class ArchitectureConfig:
    force_net_type: str = "residual"
    residual_kwargs: dict[str, Any] = field(default_factory=_default_residual_kwargs)
    mlp_kwargs: dict[str, Any] = field(default_factory=_default_mlp_kwargs)
    tcn_kwargs: dict[str, Any] = field(default_factory=_default_tcn_kwargs)
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
    force_reg: float = 1e-2
    # PHNN only: "srk4" (default) or "implicit_euler"
    physics_loss_discretization: str = "srk4"
    force_reg_on_coeff: bool = False
    # Optional PHNN trajectory rollout loss (RK4): disabled when weight<=0 or horizon<=0.
    rollout_loss_weight: float = 0.0
    rollout_horizon: int = 0
    rollout_every_steps: int = 1
    rollout_batch_size: int = 0
    use_gradnorm: bool = False
    gradnorm_alpha: float = 0.9
    gradnorm_eps: float = 1e-8
    gradnorm_min_weight: float = 0.1
    gradnorm_max_weight: float = 10.0
    use_force_data_loss: bool = False
    force_data_weight: float = 1.0

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
    log_loss_vs_ur_map: bool = True
    log_extra_validation_metrics: bool = False
    cycle_validation_rollout: bool = False
    final_rollout_all_validation: bool = False
    async_validation: bool = False
    async_validation_device: str = "cpu"
    async_validation_num_workers: int = 0
    async_validation_num_threads: int = 4
    async_validation_max_concurrent: int = 1
    async_validation_do_losses: bool = True
    async_validation_do_rollout: bool = True
    rollout_include_disp_nrmse: bool = True
    rollout_include_force_nrmse: bool = True
    rollout_include_force_mapping_nrmse: bool = True

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


def parse_config(raw: dict[str, Any]) -> Config:
    method = raw.get("method", "hnn")
    data_cfg = raw.get("data", {}) or {}
    model_cfg = raw.get("model", {}) or {}
    # Legacy model key no longer used.
    model_cfg.pop("include_physical_drag", None)
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

    legacy_tcn: dict[str, Any] = {}
    if "tcn_hidden" in architecture_cfg:
        legacy_tcn["hidden"] = architecture_cfg.pop("tcn_hidden")
    if "tcn_levels" in architecture_cfg:
        legacy_tcn["levels"] = architecture_cfg.pop("tcn_levels")
    if "tcn_dilation_start" in architecture_cfg:
        legacy_tcn["dilation_start"] = architecture_cfg.pop("tcn_dilation_start")
    if "tcn_kernel_size" in architecture_cfg:
        legacy_tcn["kernel_size"] = architecture_cfg.pop("tcn_kernel_size")
    if "tcn_dropout" in architecture_cfg:
        legacy_tcn["dropout"] = architecture_cfg.pop("tcn_dropout")
    if "tcn_activation" in architecture_cfg:
        legacy_tcn["activation"] = architecture_cfg.pop("tcn_activation")
    if "tcn_history_len" in architecture_cfg:
        legacy_tcn["history_len"] = architecture_cfg.pop("tcn_history_len")
    if "tcn_enabled" in architecture_cfg:
        legacy_tcn["enabled"] = architecture_cfg.pop("tcn_enabled")
    if "tcn_use_as_backbone" in architecture_cfg:
        legacy_tcn["use_as_backbone"] = architecture_cfg.pop("tcn_use_as_backbone")
    if "use_tcn_backbone" in architecture_cfg:
        legacy_tcn["enabled"] = architecture_cfg.pop("use_tcn_backbone")
    if "tcn_head_input_dim" in architecture_cfg:
        legacy_tcn["head_input_dim"] = architecture_cfg.pop("tcn_head_input_dim")
    if legacy_tcn or "tcn_kwargs" in architecture_cfg:
        tcn_kwargs = dict(architecture_cfg.get("tcn_kwargs", {}) or {})
        tcn_kwargs.update(legacy_tcn)
        architecture_cfg["tcn_kwargs"] = tcn_kwargs

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
        "force_reg",
        "physics_loss_discretization",
        "force_reg_on_coeff",
        "rollout_loss_weight",
        "rollout_horizon",
        "rollout_every_steps",
        "rollout_batch_size",
        "use_gradnorm",
        "gradnorm_alpha",
        "gradnorm_eps",
        "gradnorm_min_weight",
        "gradnorm_max_weight",
        "use_force_data_loss",
        "force_data_weight",
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
        "log_loss_vs_ur_map",
        "log_extra_validation_metrics",
        "rollout_include_disp_nrmse",
        "rollout_include_force_nrmse",
        "rollout_include_force_mapping_nrmse",
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
    loss = LossConfig(**loss_cfg)
    runtime = RuntimeConfig(**runtime_cfg)
    precision = PrecisionConfig(**precision_cfg)
    compile_cfg_obj = CompileConfig(**compile_cfg)
    # Back-compat key: `rollout_every_epoch` -> `rollout_every_epochs`
    if "rollout_every_epochs" not in monitoring_cfg and "rollout_every_epoch" in monitoring_cfg:
        monitoring_cfg["rollout_every_epochs"] = monitoring_cfg.pop("rollout_every_epoch")
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
    force_data: np.ndarray,
    D: float,
    k: float,
    device: torch.device,
    middle_time_plot: list[float] | tuple[float, float],
    hamiltonian_data: np.ndarray | None,
    *,
    log_extra_metrics: bool = False,
    include_disp_nrmse: bool = True,
    include_force_nrmse: bool = True,
    include_force_mapping_nrmse: bool = True,
    log_metrics: bool = True,
    tag_prefix: str = "val/rollout",
    step: int | None = None,
    title_suffix: str = "",
    validation_start_idx: int = 0,
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
        validation_start_idx=validation_start_idx,
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
        include_disp_nrmse=include_disp_nrmse,
        include_force_nrmse=include_force_nrmse,
        include_force_mapping_nrmse=include_force_mapping_nrmse,
        rollout=rollout,
        validation_start_idx=validation_start_idx,
    )
    if torch.is_tensor(reduced_velocity):
        reduced_velocity_scalar = float(reduced_velocity.reshape(-1)[0].detach().cpu())
    else:
        reduced_velocity_scalar = float(np.asarray(reduced_velocity).reshape(-1)[0])

    if log_metrics:
        for name, value in metrics.items():
            writer.add_scalar(f"val/{name}", value, epoch)
    t_plot = np.asarray(t, dtype=float).reshape(-1)
    y_true_plot = np.asarray(y_true_norm, dtype=float).reshape(-1)
    y_pred_plot = np.asarray(rollout["y_norm"], dtype=float).reshape(-1)
    p_pred_plot = np.asarray(rollout["p_norm"], dtype=float).reshape(-1)
    n_disp = int(min(t_plot.size, y_true_plot.size, y_pred_plot.size, p_pred_plot.size))
    if n_disp <= 1:
        raise ValueError("Validation plotting failed: not enough aligned displacement samples.")
    t_plot = t_plot[:n_disp]
    y_true_plot = y_true_plot[:n_disp]
    y_pred_plot = y_pred_plot[:n_disp]
    p_pred_plot = p_pred_plot[:n_disp]

    force_pred_plot = np.asarray(rollout["force_total"], dtype=float).reshape(-1)
    force_true_plot = np.asarray(force_data, dtype=float).reshape(-1)
    n_force = int(min(t_plot.size, force_pred_plot.size, force_true_plot.size))
    if n_force <= 1:
        raise ValueError("Validation plotting failed: not enough aligned force samples.")
    t_force = t_plot[:n_force]
    force_pred_plot = force_pred_plot[:n_force]
    force_true_plot = force_true_plot[:n_force]
    drag_coeff_pred_plot: np.ndarray | None = None
    vortex_coeff_pred_plot: np.ndarray | None = None
    if "drag_coeff_pred" in rollout and "vortex_coeff_pred" in rollout:
        drag_coeff_pred = np.asarray(rollout["drag_coeff_pred"], dtype=float).reshape(-1)
        vortex_coeff_pred = np.asarray(rollout["vortex_coeff_pred"], dtype=float).reshape(-1)
        if drag_coeff_pred.size >= n_force and vortex_coeff_pred.size >= n_force:
            drag_coeff_pred_plot = drag_coeff_pred[:n_force]
            vortex_coeff_pred_plot = vortex_coeff_pred[:n_force]

    zoom_mask = create_zoom_mask(t_plot)
    middle_mask = create_window_mask(t_plot, middle_time_plot)
    zoom_mask_force = create_zoom_mask(t_force)
    middle_mask_force = create_window_mask(t_force, middle_time_plot)
    log_displacement_plots(
        writer,
        epoch,
        t_plot,
        y_true_plot,
        y_pred_plot,
        p_pred_plot,
        zoom_mask,
        middle_mask,
        middle_time_plot,
        reduced_velocity=reduced_velocity_scalar,
        tag_prefix=tag_prefix,
        step=step,
        title_suffix=title_suffix,
    )
    with torch.no_grad():
        y_for_scale = y_data_t[:n_force]
        v_for_scale = val_vel[:n_force]
        if y_for_scale.ndim != 1:
            y_for_scale = y_for_scale.reshape(-1)
        if v_for_scale.ndim != 1:
            v_for_scale = v_for_scale.reshape(-1)
        z_for_scale = torch.stack((y_for_scale, v_for_scale * float(m_eff)), dim=1)
        rv_for_scale = reduced_velocity[:n_force]
        if rv_for_scale.ndim == 1:
            rv_for_scale = rv_for_scale.unsqueeze(-1)
        like = torch.ones((int(n_force), 1), device=z_for_scale.device, dtype=z_for_scale.dtype)
        force_scale_series = (
            model._force_scale_from_reduced_velocity(
                rv_for_scale,
                like=like,
                state=z_for_scale,
            )
            .reshape(-1)
            .detach()
            .cpu()
            .numpy()
        )
    if force_scale_series.shape[0] != n_force:
        raise ValueError("Validation plotting failed: invalid instantaneous force-scale length.")
    if not np.all(np.isfinite(force_scale_series)):
        raise ValueError("Validation plotting failed: non-finite instantaneous force scales.")
    force_scale_series = np.clip(force_scale_series, 1e-12, None)
    log_force_plots(
        writer,
        epoch,
        t_force,
        force_pred_plot / force_scale_series,
        force_true_plot / force_scale_series,
        zoom_mask_force,
        middle_mask_force,
        middle_time_plot,
        drag_coeff_pred=drag_coeff_pred_plot,
        vortex_coeff_pred=vortex_coeff_pred_plot,
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
    force_data: np.ndarray,
    D: float,
    k: float,
    device: torch.device,
    log_extra_metrics: bool = False,
    include_disp_nrmse: bool = True,
    include_force_nrmse: bool = True,
    include_force_mapping_nrmse: bool = True,
    rollout: dict[str, np.ndarray] | None = None,
    validation_start_idx: int = 0,
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
            validation_start_idx=validation_start_idx,
        )
    metrics: dict[str, float] = {}
    y_pred_raw = np.asarray(rollout["y_norm"], dtype=float).reshape(-1) * D
    y_true_raw = np.asarray(y_data_raw, dtype=float).reshape(-1)
    n_disp = int(min(y_pred_raw.size, y_true_raw.size))
    if n_disp <= 0:
        raise ValueError("Validation metrics failed: no aligned displacement samples.")
    y_pred_raw = y_pred_raw[:n_disp]
    y_true_raw = y_true_raw[:n_disp]
    disp_std_raw = float(np.std(y_true_raw))
    if disp_std_raw <= 0.0:
        disp_std_raw = 1.0
    rel_rmse_disp = float(np.sqrt(np.mean((y_pred_raw - y_true_raw) ** 2))) / disp_std_raw
    if include_disp_nrmse:
        metrics[DISP_ROLLOUT_NRMSE_KEY] = rel_rmse_disp
    force_total_pred = np.asarray(rollout["force_total"]).reshape(-1)
    force_target = np.asarray(force_data).reshape(-1)
    min_len = min(force_total_pred.shape[0], force_target.shape[0])
    if min_len > 0:
        rmse_force = float(
            np.sqrt(np.mean((force_total_pred[:min_len] - force_target[:min_len]) ** 2))
        )
        force_std = float(np.std(force_target[:min_len]))
        if force_std <= 0.0:
            force_std = 1.0
        rel_rmse_force_total = rmse_force / force_std
        if include_force_nrmse:
            metrics[FORCE_ROLLOUT_NRMSE_KEY] = rel_rmse_force_total
    if include_force_mapping_nrmse:
        with torch.no_grad():
            z_true = torch.stack((y_data_t, val_vel * m_eff), dim=1)
            z_true = z_true.to(device=device, non_blocking=(device.type == "cuda"))
            rv = reduced_velocity.to(device=device, non_blocking=(device.type == "cuda"))
            force_on_data = model.u_theta_on_trajectory(z_true, reduced_velocity=rv).squeeze(-1).detach().cpu().numpy()
        start_idx = max(0, int(validation_start_idx))
        if start_idx > 0:
            force_on_data = force_on_data[start_idx:]
        min_len_data = min(force_on_data.shape[0], force_target.shape[0])
        if min_len_data > 0:
            force_data_pred = force_on_data[:min_len_data]
            force_data_true = force_target[:min_len_data]
            rmse_force_data = float(np.sqrt(np.mean((force_data_pred - force_data_true) ** 2)))
            force_std_data = float(np.std(force_data_true))
            if force_std_data <= 0.0:
                force_std_data = 1.0
            rel_rmse_force_on_data = rmse_force_data / force_std_data
            metrics[FORCE_MAPPING_NRMSE_KEY] = rel_rmse_force_on_data
    if log_extra_metrics:
        freq_true = dominant_frequency(y_true_raw, dt)
        freq_pred = dominant_frequency(y_pred_raw, dt)
        freq_rel = abs(relative_error(freq_pred, freq_true))
        if np.isfinite(freq_rel):
            metrics[DOMINANT_FREQ_REL_ERROR_KEY] = float(freq_rel)

        amp_true = mean_displacement_amplitude(y_true_raw)
        amp_pred = mean_displacement_amplitude(y_pred_raw)
        amp_rel = abs(relative_error(amp_pred, amp_true))
        if np.isfinite(amp_rel):
            metrics[MEAN_DISP_AMP_REL_ERROR_KEY] = float(amp_rel)

        disp_spec_err = spectral_relative_error(y_true_raw, y_pred_raw, dt)
        if np.isfinite(disp_spec_err):
            metrics[DISP_SPECTRAL_SHAPE_ERROR_KEY] = float(disp_spec_err)

        force_spec_err = spectral_relative_error(force_target, force_total_pred, dt)
        if np.isfinite(force_spec_err):
            metrics[FORCE_SPECTRAL_SHAPE_ERROR_KEY] = float(force_spec_err)
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


class TemporalBackboneWithHead(nn.Module):
    """Causal temporal encoder followed by a pointwise head at each timestep."""

    def __init__(self, backbone: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.is_tcn_force_model = True
        self.history_len = int(getattr(backbone, "history_len", 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
            squeeze_time = True
        elif x.ndim == 3:
            squeeze_time = False
        else:
            raise ValueError("TemporalBackboneWithHead expects input shape (N,C) or (B,T,C).")

        h = self.backbone(x)
        if h.ndim != 3:
            raise ValueError("Temporal backbone must return shape (B,T,C_mid).")

        B, T, C = h.shape
        y = self.head(h.reshape(B * T, C)).reshape(B, T, -1)
        if squeeze_time:
            y = y.squeeze(1)
        return y


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


def relative_error(model_value: float, true_value: float, eps: float = 1e-12) -> float:
    """Compute signed (model - true)/|true| with small epsilon safeguard."""
    if not np.isfinite(true_value) or not np.isfinite(model_value):
        return float("nan")
    denom = abs(true_value)
    if denom <= eps:
        return float("nan")
    return float((model_value - true_value) / (denom + eps))


def mean_displacement_amplitude(signal: np.ndarray) -> float:
    """
    Mean oscillation amplitude estimate from a displacement signal.
    Uses half peak-to-peak around the sample range after finite filtering.
    """
    arr = np.asarray(signal, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("nan")
    return float(0.5 * (np.max(arr) - np.min(arr)))


def spectral_relative_error(
    true_signal: np.ndarray,
    model_signal: np.ndarray,
    dt: float,
    eps: float = 1e-12,
) -> float:
    """
    Compute relative L2 error between FFT magnitudes of true and model signals.
    Signals are centered and windowed with a Hann taper to reduce leakage.
    """
    if dt <= 0.0:
        return float("nan")
    true_signal = np.asarray(true_signal)
    model_signal = np.asarray(model_signal)
    length = min(true_signal.size, model_signal.size)
    if length < 2:
        return float("nan")
    true_trim = true_signal[-length:]
    model_trim = model_signal[-length:]
    window = np.hanning(length)
    true_proc = (true_trim - np.mean(true_trim)) * window
    model_proc = (model_trim - np.mean(model_trim)) * window
    true_fft = np.abs(np.fft.rfft(true_proc))
    model_fft = np.abs(np.fft.rfft(model_proc))
    if true_fft.size == 0:
        return float("nan")
    true_fft[0] = 0.0
    model_fft[0] = 0.0
    denom = np.linalg.norm(true_fft)
    if denom <= eps:
        return float("nan")
    return float(np.linalg.norm(model_fft - true_fft) / (denom + eps))


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
        force_output: str = "force",
        learn_hamiltonian: bool = False,
        use_pirate_force: bool = False,
        pirate_force_kwargs: dict | None = None,
        use_fourier_features: bool = False,
        fourier_features: int = 64,
        fourier_sigma: float = 1.0,
        use_feature_engineering: bool = False,
        use_reduced_velocity: bool = True,
        ur_scale: float | None = None,
        bound_force_coefficient: bool = False,
        force_coefficient_c_max: float = 5.0,
        force_net_type: str | None = None,
        residual_kwargs: dict[str, Any] | None = None,
        mlp_kwargs: dict[str, Any] | None = None,
        tcn_kwargs: dict[str, Any] | None = None,
        physics_loss_discretization: str = "srk4",
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
        force_output = str(force_output).strip().lower()
        if force_output not in {"force", "coefficient", "two_head_vivana"}:
            raise ValueError("force_output must be one of: force, coefficient, two_head_vivana")
        self.force_output = force_output
        self.use_two_head_vivana = self.force_output == "two_head_vivana"
        self.loss_discretization = "srk4"
        self.set_loss_discretization(physics_loss_discretization)
        self.learn_hamiltonian = bool(learn_hamiltonian)
        self.use_feature_engineering = bool(use_feature_engineering)
        self.use_reduced_velocity = bool(use_reduced_velocity)
        self.bound_force_coefficient = bool(bound_force_coefficient)
        cmax = float(force_coefficient_c_max)
        if not np.isfinite(cmax) or cmax <= 0.0:
            raise ValueError(f"force_coefficient_c_max must be finite and > 0, got {cmax}")
        self.register_buffer("force_coefficient_c_max", torch.tensor(cmax, dtype=torch.float32))
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
        tcn_cfg = _default_tcn_kwargs()
        if tcn_kwargs:
            tcn_cfg.update(tcn_kwargs)
        self.tcn_hidden = int(tcn_cfg.get("hidden", 128))
        self.tcn_levels = max(1, int(tcn_cfg.get("levels", 4)))
        self.tcn_dilation_start = max(1, int(tcn_cfg.get("dilation_start", 1)))
        self.tcn_kernel_size = max(1, int(tcn_cfg.get("kernel_size", 3)))
        self.tcn_dropout = float(tcn_cfg.get("dropout", 0.0))
        self.tcn_activation = str(tcn_cfg.get("activation", "gelu"))
        self.tcn_history_len = max(1, int(tcn_cfg.get("history_len", 64)))
        self.use_tcn_backbone = bool(
            tcn_cfg.get("enabled", tcn_cfg.get("use_as_backbone", False))
        )
        head_input_raw = tcn_cfg.get("head_input_dim", self.tcn_hidden)
        if head_input_raw is None:
            head_input_raw = self.tcn_hidden
        self.tcn_head_input_dim = int(head_input_raw)
        if self.tcn_head_input_dim < 1:
            raise ValueError("architecture.tcn_kwargs.head_input_dim must be >= 1.")
        self.is_tcn_force_model = False
        self.history_len = 0

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

        # NN for instantaneous force u(x)
        pirate_force_kwargs = pirate_force_kwargs or {}
        self.use_fourier_features = bool(use_fourier_features)
        self.fourier_features = int(fourier_features)
        self.fourier_sigma = float(fourier_sigma)
        self.force_embed = None
        base_force_dim = self.force_input_dim
        force_in_features = base_force_dim
        selected_net = force_net_type if force_net_type not in (None, "") else ("pirate" if use_pirate_force else "residual")
        net_type = str(selected_net).lower()
        valid_types = {"residual", "mlp", "pirate", "tcn"}
        if net_type not in valid_types:
            raise ValueError(f"force_net_type must be one of {valid_types}, got '{force_net_type}'.")
        self.use_pirate_force = net_type == "pirate"
        self.use_tcn_force = net_type == "tcn"
        self.residual_net = net_type == "residual"
        self.force_raw_dim = 2 if self.use_two_head_vivana else 1
        tcn_mode_requested = self.use_tcn_force or self.use_tcn_backbone
        if self.use_fourier_features:
            if self.fourier_features < 1:
                raise ValueError("fourier_features must be >= 1 when use_fourier_features is True")
            if self.use_pirate_force or tcn_mode_requested:
                raise ValueError(
                    "use_fourier_features is not supported together with pirate/tcn force networks."
                )
            self.force_embed = FourierFeatures(
                in_dim=base_force_dim,
                out_features=self.fourier_features,
                sigma=self.fourier_sigma,
                dtype=torch.float32,
            )
            force_in_features = 2 * self.fourier_features

        pirate_cfg = dict(pirate_force_kwargs) if pirate_force_kwargs is not None else {}
        def _build_pointwise_force_head(head_net_type: str, in_features: int) -> nn.Module:
            key = str(head_net_type).strip().lower()
            if key == "pirate":
                head_cfg = dict(pirate_cfg)
                pirate_args = {
                    "input_size": int(in_features),
                    "output_size": int(self.force_raw_dim),
                    "depth": int(head_cfg.pop("depth", head_cfg.pop("pirate_layers", 2))),
                    "fourier_features": int(head_cfg.pop("fourier_features", 64)),
                    "sigma": float(head_cfg.pop("sigma", 1.0)),
                    "use_rwf": bool(head_cfg.pop("use_rwf", True)),
                    "activation": head_cfg.pop("activation", "tanh"),
                }
                pirate_args.update(head_cfg)
                return ODEPirateNet(**pirate_args)
            if key == "residual":
                layers = [nn.Linear(int(in_features), self.residual_hidden)]
                for _ in range(self.residual_layers):
                    layers.append(Residual(self.residual_hidden, activation=self.residual_activation))
                layers.append(nn.Linear(self.residual_hidden, int(self.force_raw_dim)))
                return nn.Sequential(*layers)
            if key == "mlp":
                mlp_layers: list[nn.Module] = []
                head_in = int(in_features)
                mlp_act_cls = _activation_factory(self.mlp_activation)
                for _ in range(self.mlp_layers):
                    mlp_layers.append(nn.Linear(head_in, self.mlp_hidden))
                    mlp_layers.append(mlp_act_cls())
                    head_in = self.mlp_hidden
                mlp_layers.append(nn.Linear(self.mlp_hidden, int(self.force_raw_dim)))
                return nn.Sequential(*mlp_layers)
            raise ValueError(f"Unsupported PHNN force head type '{head_net_type}'.")

        if self.use_tcn_force:
            self.u_net = TemporalConvForceNet(
                input_size=force_in_features,
                output_size=int(self.force_raw_dim),
                hidden_channels=self.tcn_hidden,
                levels=self.tcn_levels,
                dilation_start=self.tcn_dilation_start,
                kernel_size=self.tcn_kernel_size,
                dropout=self.tcn_dropout,
                activation=self.tcn_activation,
                history_len=self.tcn_history_len,
            )
            self.is_tcn_force_model = True
            self.history_len = self.tcn_history_len
        elif self.use_tcn_backbone:
            backbone = TemporalConvForceNet(
                input_size=force_in_features,
                output_size=self.tcn_head_input_dim,
                hidden_channels=self.tcn_hidden,
                levels=self.tcn_levels,
                dilation_start=self.tcn_dilation_start,
                kernel_size=self.tcn_kernel_size,
                dropout=self.tcn_dropout,
                activation=self.tcn_activation,
                history_len=self.tcn_history_len,
            )
            head = _build_pointwise_force_head(net_type, self.tcn_head_input_dim)
            self.u_net = TemporalBackboneWithHead(backbone=backbone, head=head)
            self.is_tcn_force_model = True
            self.history_len = self.tcn_history_len
        else:
            self.u_net = _build_pointwise_force_head(net_type, force_in_features)

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
        force_output = str(cfg.get("force_output", "force")).strip().lower()
        learn_hamiltonian = bool(cfg.get("learn_hamiltonian", False))
        use_pirate_force = bool(cfg.get("use_pirate_force", False))
        pirate_force_kwargs = cfg.get("pirate_force_kwargs", {}) or {}
        use_fourier_features = bool(cfg.get("use_fourier_features", False))
        fourier_features = int(cfg.get("fourier_features", 64))
        fourier_sigma = float(cfg.get("fourier_sigma", 1.0))
        use_feature_engineering = bool(cfg.get("use_feature_engineering", False))
        use_reduced_velocity = bool(cfg.get("use_reduced_velocity", True))
        bound_force_coefficient = bool(cfg.get("bound_force_coefficient", False))
        force_coefficient_c_max = float(cfg.get("force_coefficient_c_max", 5.0))
        physics_loss_discretization = str(cfg.get("physics_loss_discretization", "srk4"))
        ur_scale_val = cfg.get("ur_scale")
        ur_scale = None if ur_scale_val is None else float(ur_scale_val)
        arch_cfg = arch_cfg or {}
        force_net_type = arch_cfg.get("force_net_type")
        residual_kwargs = _default_residual_kwargs()
        residual_kwargs.update(arch_cfg.get("residual_kwargs", {}) or {})
        mlp_kwargs = _default_mlp_kwargs()
        mlp_kwargs.update(arch_cfg.get("mlp_kwargs", {}) or {})
        tcn_kwargs = _default_tcn_kwargs()
        tcn_kwargs.update(arch_cfg.get("tcn_kwargs", {}) or {})
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
            force_output=force_output,
            learn_hamiltonian=learn_hamiltonian,
            use_pirate_force=use_pirate_force,
            pirate_force_kwargs=combined_pirate_kwargs,
            use_fourier_features=use_fourier_features,
            fourier_features=fourier_features,
            fourier_sigma=fourier_sigma,
            use_feature_engineering=use_feature_engineering,
            use_reduced_velocity=use_reduced_velocity,
            ur_scale=ur_scale,
            bound_force_coefficient=bound_force_coefficient,
            force_coefficient_c_max=force_coefficient_c_max,
            force_net_type=force_net_type,
            residual_kwargs=residual_kwargs,
            mlp_kwargs=mlp_kwargs,
            tcn_kwargs=tcn_kwargs,
            physics_loss_discretization=physics_loss_discretization,
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

    @staticmethod
    def _normalize_loss_discretization(name: str) -> str:
        key = str(name).strip().lower().replace("-", "_")
        aliases = {
            "srk4": "srk4",
            "rk4": "srk4",
            "s_rk4": "srk4",
            "implicit_euler": "implicit_euler",
            "euler": "implicit_euler",
            "midpoint_euler": "implicit_euler",
            "implicit_midpoint": "implicit_euler",
        }
        if key not in aliases:
            raise ValueError(
                "loss.physics_loss_discretization must be one of: srk4, implicit_euler."
            )
        return aliases[key]

    def set_loss_discretization(self, name: str) -> None:
        self.loss_discretization = self._normalize_loss_discretization(name)

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
        state: torch.Tensor | np.ndarray | None = None,
    ) -> torch.Tensor:
        rv_raw = self._prepare_reduced_velocity_raw(reduced_velocity, like=like)
        if rv_raw is None:
            U = like.new_full(like.shape[:-1] + (1,), float(self.U))
        else:
            omega_n = torch.sqrt(
                torch.as_tensor(float(self.k) / float(self.m), device=like.device, dtype=like.dtype)
            )
            f_n = omega_n / (2.0 * math.pi)
            U = rv_raw * f_n * float(self.D)

        speed_sq = U ** 2
        if state is not None:
            if torch.is_tensor(state):
                state_t = state.to(device=like.device, dtype=like.dtype)
            else:
                state_t = torch.as_tensor(state, device=like.device, dtype=like.dtype)
            if state_t.ndim != like.ndim or state_t.shape[-1] < 2:
                raise ValueError("state must have shape (..., >=2) and match 'like' dimensions.")
            if state_t.shape[:-1] != like.shape[:-1]:
                state_t = state_t.expand(like.shape[:-1] + (state_t.shape[-1],))
            v_inst = state_t[..., 1:2] / float(self.m)
            speed_sq = speed_sq + v_inst ** 2

        f0 = 0.5 * float(self.rho) * float(self.D) * speed_sq
        return torch.clamp(f0, min=1e-12)

    def _flow_speed_from_reduced_velocity(
        self,
        reduced_velocity: torch.Tensor | np.ndarray | float | None,
        *,
        like: torch.Tensor,
    ) -> torch.Tensor:
        rv_raw = self._prepare_reduced_velocity_raw(reduced_velocity, like=like)
        if rv_raw is None:
            return like.new_full(like.shape[:-1] + (1,), float(self.U))
        omega_n = torch.sqrt(
            torch.as_tensor(float(self.k) / float(self.m), device=like.device, dtype=like.dtype)
        )
        f_n = omega_n / (2.0 * math.pi)
        return rv_raw * f_n * float(self.D)

    def _force_features(self, x, reduced_velocity: torch.Tensor | np.ndarray | float | None = None):
        base_features = self._base_features(x)
        if self.use_reduced_velocity:
            rv = self._prepare_reduced_velocity(reduced_velocity, like=base_features)
            base_features = torch.cat([base_features, rv], dim=-1)
        return base_features

    def _force_features_sequence(
        self,
        x: torch.Tensor,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
    ) -> torch.Tensor:
        base_features = self._base_features(x)
        if self.use_reduced_velocity:
            rv = self._prepare_reduced_velocity(reduced_velocity, like=base_features)
            base_features = torch.cat([base_features, rv], dim=-1)
        return base_features

    def _force_net_raw_sequence(
        self,
        x: torch.Tensor,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("TCN sequence force evaluation expects x with shape (B, T, 2).")
        base_features = self._force_features_sequence(x, reduced_velocity=reduced_velocity)
        if self.force_embed is not None:
            B, T, C = base_features.shape
            embedded = self.force_embed(base_features.reshape(B * T, C))
            base_features = embedded.reshape(B, T, -1)
        if self.is_tcn_force_model:
            raw = self.u_net(base_features)
            if raw.ndim != 3:
                raise ValueError("TCN force network must return shape (B, T, 1).")
            return raw
        B, T, C = base_features.shape
        return self.u_net(base_features.reshape(B * T, C)).reshape(B, T, -1)

    def _force_net_raw_with_history(
        self,
        x: torch.Tensor,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
        z_hist: torch.Tensor | None = None,
        ur_hist: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.is_tcn_force_model:
            base_features = self._force_features(x, reduced_velocity=reduced_velocity)
            features = self.force_embed(base_features) if self.force_embed is not None else base_features
            return self.u_net(features)
        if x.ndim != 2 or x.shape[-1] != 2:
            raise ValueError("TCN force history evaluation expects x with shape (B, 2).")
        B = int(x.shape[0])
        hist_len = int(self.history_len) + 1
        rv_raw = None
        if self.use_reduced_velocity:
            rv_raw = self._prepare_reduced_velocity_raw(reduced_velocity, like=x[..., :1])
        if z_hist is None:
            z_hist = x.unsqueeze(1).repeat(1, hist_len, 1)
        if ur_hist is None and self.use_reduced_velocity:
            assert rv_raw is not None
            ur_hist = rv_raw.unsqueeze(1).repeat(1, hist_len, 1)
        if z_hist.ndim != 3 or z_hist.shape[0] != B or z_hist.shape[1] != hist_len or z_hist.shape[2] != 2:
            raise ValueError(f"z_hist must have shape (B, {hist_len}, 2) for TCN force evaluation.")
        z_in = z_hist.clone()
        z_in[:, -1, :] = x
        rv_in = None
        if self.use_reduced_velocity:
            if ur_hist is None:
                raise ValueError("ur_hist is required when use_reduced_velocity is True.")
            if ur_hist.ndim != 3 or ur_hist.shape[0] != B or ur_hist.shape[1] != hist_len or ur_hist.shape[2] != 1:
                raise ValueError(f"ur_hist must have shape (B, {hist_len}, 1) for TCN force evaluation.")
            rv_in = ur_hist.clone()
            assert rv_raw is not None
            rv_in[:, -1, :] = rv_raw
        raw_seq = self._force_net_raw_sequence(z_in, reduced_velocity=rv_in)
        return raw_seq[:, -1, :]

    def _force_net_raw(
        self,
        x,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
        z_hist: torch.Tensor | None = None,
        ur_hist: torch.Tensor | None = None,
    ):
        if self.is_tcn_force_model:
            return self._force_net_raw_with_history(
                x,
                reduced_velocity=reduced_velocity,
                z_hist=z_hist,
                ur_hist=ur_hist,
            )
        base_features = self._force_features(x, reduced_velocity=reduced_velocity)
        features = self.force_embed(base_features) if self.force_embed is not None else base_features
        return self.u_net(features)

    def _maybe_bound_force_coeff(self, coeff: torch.Tensor) -> torch.Tensor:
        if not self.bound_force_coefficient:
            return coeff
        cmax = self.force_coefficient_c_max.to(device=coeff.device, dtype=coeff.dtype)
        return cmax * torch.tanh(coeff / cmax)

    def _two_head_vivana_coeff_from_raw(self, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if raw.shape[-1] != 2:
            raise ValueError(
                f"two_head_vivana expects force head output with last dim=2, got shape {tuple(raw.shape)}."
            )
        # Coefficients per term: [drag_magnitude, vortex_like].
        # Constrain drag magnitude to [0, Cmax] and apply an explicit minus sign
        # in force_drag so the drag term is always dissipative.
        cmax = self.force_coefficient_c_max.to(device=raw.device, dtype=raw.dtype)
        c_drag = cmax * torch.sigmoid(raw[..., 0:1])
        c_vortex = self._maybe_bound_force_coeff(raw[..., 1:2])
        return c_drag, c_vortex

    def _two_head_vivana_force_from_raw(
        self,
        raw: torch.Tensor,
        *,
        x: torch.Tensor,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c_drag, c_vortex = self._two_head_vivana_coeff_from_raw(raw)

        # State x = [q, p], with velocity dy = p / m.
        dy = x[..., 1:2] / float(self.m)
        u_flow = self._flow_speed_from_reduced_velocity(reduced_velocity, like=dy)
        v_rel = torch.sqrt(torch.clamp(u_flow**2 + dy**2, min=1e-12))
        pref = 0.5 * float(self.rho) * float(self.D)

        # Requested scaling:
        #   F_drag_like   ~ -|V_rel| * dy  (always dissipative)
        #   F_vortex_like ~  |V_rel| * U
        force_drag = -pref * c_drag * v_rel * dy
        force_vortex = pref * c_vortex * v_rel * u_flow
        return force_drag + force_vortex, c_drag, c_vortex

    def learned_force_coeff(
        self,
        x,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
        z_hist: torch.Tensor | None = None,
        ur_hist: torch.Tensor | None = None,
    ):
        raw = self._force_net_raw(
            x,
            reduced_velocity=reduced_velocity,
            z_hist=z_hist,
            ur_hist=ur_hist,
        )
        if self.use_two_head_vivana:
            total_force, _, _ = self._two_head_vivana_force_from_raw(raw, x=x, reduced_velocity=reduced_velocity)
            f0 = self._force_scale_from_reduced_velocity(reduced_velocity, like=total_force, state=x)
            return total_force / f0
        if self.force_output == "coefficient":
            return self._maybe_bound_force_coeff(raw)
        f0 = self._force_scale_from_reduced_velocity(reduced_velocity, like=raw, state=x)
        coeff = raw * self.k * self.D / f0
        return self._maybe_bound_force_coeff(coeff)

    def learned_force(
        self,
        x,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
        z_hist: torch.Tensor | None = None,
        ur_hist: torch.Tensor | None = None,
    ):
        raw = self._force_net_raw(
            x,
            reduced_velocity=reduced_velocity,
            z_hist=z_hist,
            ur_hist=ur_hist,
        )
        if self.use_two_head_vivana:
            total_force, _, _ = self._two_head_vivana_force_from_raw(raw, x=x, reduced_velocity=reduced_velocity)
            return total_force
        if self.force_output == "coefficient":
            f0 = self._force_scale_from_reduced_velocity(reduced_velocity, like=raw, state=x)
            coeff = self._maybe_bound_force_coeff(raw)
            return coeff * f0
        return raw * self.k * self.D

    def learned_force_sequence(
        self,
        x: torch.Tensor,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
    ) -> torch.Tensor:
        raw = self._force_net_raw_sequence(x, reduced_velocity=reduced_velocity)
        if self.use_two_head_vivana:
            total_force, _, _ = self._two_head_vivana_force_from_raw(raw, x=x, reduced_velocity=reduced_velocity)
            return total_force
        if self.force_output == "coefficient":
            f0 = self._force_scale_from_reduced_velocity(reduced_velocity, like=raw, state=x)
            coeff = self._maybe_bound_force_coeff(raw)
            return coeff * f0
        return raw * self.k * self.D

    def learned_force_coeff_sequence(
        self,
        x: torch.Tensor,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
    ) -> torch.Tensor:
        raw = self._force_net_raw_sequence(x, reduced_velocity=reduced_velocity)
        if self.use_two_head_vivana:
            total_force, _, _ = self._two_head_vivana_force_from_raw(raw, x=x, reduced_velocity=reduced_velocity)
            f0 = self._force_scale_from_reduced_velocity(reduced_velocity, like=total_force, state=x)
            return total_force / f0
        if self.force_output == "coefficient":
            return self._maybe_bound_force_coeff(raw)
        f0 = self._force_scale_from_reduced_velocity(reduced_velocity, like=raw, state=x)
        coeff = raw * self.k * self.D / f0
        return self._maybe_bound_force_coeff(coeff)

    def u_theta_coeff(
        self,
        x,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
        z_hist: torch.Tensor | None = None,
        ur_hist: torch.Tensor | None = None,
    ):
        coeff = self.learned_force_coeff(
            x,
            reduced_velocity=reduced_velocity,
            z_hist=z_hist,
            ur_hist=ur_hist,
        )
        return coeff

    def u_theta_coeff_sequence(
        self,
        x: torch.Tensor,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
    ) -> torch.Tensor:
        return self.learned_force_coeff_sequence(x, reduced_velocity=reduced_velocity)

    def u_theta1(
        self,
        x,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
        z_hist: torch.Tensor | None = None,
        ur_hist: torch.Tensor | None = None,
    ):
        return self.learned_force(
            x,
            reduced_velocity=reduced_velocity,
            z_hist=z_hist,
            ur_hist=ur_hist,
        )

    def _force_with_coeff_parts(
        self,
        x,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
        z_hist: torch.Tensor | None = None,
        ur_hist: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        raw = self._force_net_raw(
            x,
            reduced_velocity=reduced_velocity,
            z_hist=z_hist,
            ur_hist=ur_hist,
        )
        if self.use_two_head_vivana:
            total_force, c_drag, c_vortex = self._two_head_vivana_force_from_raw(
                raw,
                x=x,
                reduced_velocity=reduced_velocity,
            )
            return total_force, c_drag, c_vortex
        if self.force_output == "coefficient":
            f0 = self._force_scale_from_reduced_velocity(reduced_velocity, like=raw, state=x)
            coeff = self._maybe_bound_force_coeff(raw)
            return coeff * f0, None, None
        return raw * self.k * self.D, None, None
    
    def u_theta(
        self,
        x,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
        z_hist: torch.Tensor | None = None,
        ur_hist: torch.Tensor | None = None,
    ):
        if self.force_output == "coefficient":
            coeff = self.u_theta_coeff(
                x,
                reduced_velocity=reduced_velocity,
                z_hist=z_hist,
                ur_hist=ur_hist,
            )
            f0 = self._force_scale_from_reduced_velocity(reduced_velocity, like=coeff, state=x)
            return coeff * f0
        return self.u_theta1(
            x,
            reduced_velocity=reduced_velocity,
            z_hist=z_hist,
            ur_hist=ur_hist,
        )

    def u_theta_sequence(
        self,
        x: torch.Tensor,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
    ) -> torch.Tensor:
        if self.force_output == "coefficient":
            coeff = self.u_theta_coeff_sequence(x, reduced_velocity=reduced_velocity)
            f0 = self._force_scale_from_reduced_velocity(reduced_velocity, like=coeff, state=x)
            return coeff * f0
        return self.learned_force_sequence(x, reduced_velocity=reduced_velocity)

    def u_theta_on_trajectory(
        self,
        x: torch.Tensor,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
    ) -> torch.Tensor:
        if x.ndim != 2 or x.shape[-1] != 2:
            raise ValueError("u_theta_on_trajectory expects x with shape (T, 2).")
        if not self.is_tcn_force_model:
            return self.u_theta(x, reduced_velocity=reduced_velocity)
        x_seq = x.unsqueeze(0)
        if self.use_reduced_velocity:
            rv_raw = self._prepare_reduced_velocity_raw(reduced_velocity, like=x[..., :1])
            assert rv_raw is not None
            rv_seq = rv_raw.unsqueeze(0)
        else:
            rv_seq = None
        context = int(self.history_len)
        if context > 0:
            x_seq = torch.cat([x_seq[:, 0:1, :].expand(-1, context, -1), x_seq], dim=1)
            if rv_seq is not None:
                rv_seq = torch.cat([rv_seq[:, 0:1, :].expand(-1, context, -1), rv_seq], dim=1)
        force_seq = self.u_theta_sequence(x_seq, reduced_velocity=rv_seq)
        if context > 0:
            force_seq = force_seq[:, context:, :]
        return force_seq.squeeze(0)
    
    def f(
        self,
        x,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
        z_hist: torch.Tensor | None = None,
        ur_hist: torch.Tensor | None = None,
    ):
        u = self.u_theta(
            x,
            reduced_velocity=reduced_velocity,
            z_hist=z_hist,
            ur_hist=ur_hist,
        )
        g_vec = self.G.squeeze(-1)
        return u * g_vec

    def _g_and_force(
        self,
        x,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
        z_hist: torch.Tensor | None = None,
        ur_hist: torch.Tensor | None = None,
        return_coeff_parts: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Compute state derivative and total force using a single force-network call.
        """
        force, c_drag, c_vortex = self._force_with_coeff_parts(
            x,
            reduced_velocity=reduced_velocity,
            z_hist=z_hist,
            ur_hist=ur_hist,
        )
        g_vec = self.G.squeeze(-1)
        forcing_term = force * g_vec

        gH = self.grad_H(x)  # (..., 2)
        JgH = torch.matmul(gH, self.J.T)
        if self.discover_damping:
            zeta = torch.sigmoid(self.zeta_raw) * self.max_damping_ratio
            c_eff = 2.0 * zeta * self.sqrt_km
        else:
            c_eff = self.fixed_c
        damping_term = torch.stack((torch.zeros_like(gH[..., 0]), c_eff * gH[..., 1]), dim=-1)
        dz = (JgH - damping_term) + forcing_term
        if return_coeff_parts:
            return dz, force, c_drag, c_vortex
        return dz, force

    def g(
        self,
        x,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
        z_hist: torch.Tensor | None = None,
        ur_hist: torch.Tensor | None = None,
    ):
        dz, _force = self._g_and_force(
            x,
            reduced_velocity=reduced_velocity,
            z_hist=z_hist,
            ur_hist=ur_hist,
        )
        return dz

    def step_euler(self, x, dt, reduced_velocity: torch.Tensor | np.ndarray | float | None = None):
        return x + dt * self.g(x, reduced_velocity=reduced_velocity)

    def step_rk4(
        self,
        x,
        t,
        dt,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
        z_hist: torch.Tensor | None = None,
        ur_hist: torch.Tensor | None = None,
    ):
        x_next, _ = self.rk4_step(
            x,
            t,
            dt,
            reduced_velocity=reduced_velocity,
            z_hist=z_hist,
            ur_hist=ur_hist,
        )
        return x_next

    def rk4_step(
        self,
        x,
        t,
        dt,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
        z_hist: torch.Tensor | None = None,
        ur_hist: torch.Tensor | None = None,
        return_coeff_parts: bool = False,
    ):
        """
        Perform one Runge-Kutta 4 integration step and return both the next state
        and the averaged force over the step.
        """
        if return_coeff_parts:
            k1, force1, c_drag1, c_vortex1 = self._g_and_force(
                x,
                reduced_velocity=reduced_velocity,
                z_hist=z_hist,
                ur_hist=ur_hist,
                return_coeff_parts=True,
            )
        else:
            k1, force1 = self._g_and_force(x, reduced_velocity=reduced_velocity, z_hist=z_hist, ur_hist=ur_hist)

        x2 = x + 0.5 * dt * k1
        if return_coeff_parts:
            k2, force2, c_drag2, c_vortex2 = self._g_and_force(
                x2,
                reduced_velocity=reduced_velocity,
                z_hist=z_hist,
                ur_hist=ur_hist,
                return_coeff_parts=True,
            )
        else:
            k2, force2 = self._g_and_force(x2, reduced_velocity=reduced_velocity, z_hist=z_hist, ur_hist=ur_hist)

        x3 = x + 0.5 * dt * k2
        if return_coeff_parts:
            k3, force3, c_drag3, c_vortex3 = self._g_and_force(
                x3,
                reduced_velocity=reduced_velocity,
                z_hist=z_hist,
                ur_hist=ur_hist,
                return_coeff_parts=True,
            )
        else:
            k3, force3 = self._g_and_force(x3, reduced_velocity=reduced_velocity, z_hist=z_hist, ur_hist=ur_hist)

        x4 = x + dt * k3
        if return_coeff_parts:
            k4, force4, c_drag4, c_vortex4 = self._g_and_force(
                x4,
                reduced_velocity=reduced_velocity,
                z_hist=z_hist,
                ur_hist=ur_hist,
                return_coeff_parts=True,
            )
        else:
            k4, force4 = self._g_and_force(x4, reduced_velocity=reduced_velocity, z_hist=z_hist, ur_hist=ur_hist)

        x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        force_avg = (force1 + 2.0 * force2 + 2.0 * force3 + force4) / 6.0
        if return_coeff_parts:
            if self.use_two_head_vivana:
                c_drag_avg = (c_drag1 + 2.0 * c_drag2 + 2.0 * c_drag3 + c_drag4) / 6.0
                c_vortex_avg = (c_vortex1 + 2.0 * c_vortex2 + 2.0 * c_vortex3 + c_vortex4) / 6.0
            else:
                c_drag_avg = None
                c_vortex_avg = None
            return x_next, force_avg, c_drag_avg, c_vortex_avg
        return x_next, force_avg

    def rollout(
        self,
        z0,
        t_seq,
        dt,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
        z_hist_init: torch.Tensor | None = None,
        ur_hist_init: torch.Tensor | None = None,
        record_coeff_parts: bool = False,
    ):
        """
        z0: (B, state_dim)    starting state from data
        t_seq: (B, K+1)       absolute times t0..tK
        returns:
        Z_pred: (B, K+1, state_dim)  predictions incl. z0
        F_hist: (B, K+1, 1)          optional, learned force per step
        C_drag_hist/C_vortex_hist: optional coefficient traces when record_coeff_parts=True
        """
        B = z0.shape[0]
        state_dim = z0.shape[-1]
        K = t_seq.shape[1] - 1

        Z_pred = [z0]
        F_hist = []
        C_drag_hist = []
        C_vortex_hist = []

        z = z0
        history_len = int(self.history_len) + 1 if self.is_tcn_force_model else 0
        z_hist = None
        ur_hist = None
        if self.is_tcn_force_model:
            provided_history = (z_hist_init is not None) or (ur_hist_init is not None)
            if provided_history and (z_hist_init is None or ur_hist_init is None):
                raise ValueError("z_hist_init and ur_hist_init must be provided together.")
            if provided_history:
                z_hist = z_hist_init.to(device=z.device, dtype=z.dtype)
                ur_hist = ur_hist_init.to(device=z.device, dtype=z.dtype)
                expected_z_shape = (B, history_len, state_dim)
                expected_ur_shape = (B, history_len, 1)
                if tuple(z_hist.shape) != expected_z_shape:
                    raise ValueError(
                        f"z_hist_init must have shape {expected_z_shape}, got {tuple(z_hist.shape)}."
                    )
                if tuple(ur_hist.shape) != expected_ur_shape:
                    raise ValueError(
                        f"ur_hist_init must have shape {expected_ur_shape}, got {tuple(ur_hist.shape)}."
                    )
            else:
                z_hist = z.unsqueeze(1).repeat(1, history_len, 1)
                if self.use_reduced_velocity:
                    rv_raw = self._prepare_reduced_velocity_raw(reduced_velocity, like=z[..., :1])
                    assert rv_raw is not None
                    ur_hist = rv_raw.unsqueeze(1).repeat(1, history_len, 1)
        initial_force, initial_c_drag, initial_c_vortex = self._force_with_coeff_parts(
            z,
            reduced_velocity=reduced_velocity,
            z_hist=z_hist,
            ur_hist=ur_hist,
        )
        for k in range(K):
            t = t_seq[:, k]
            if record_coeff_parts:
                z, Fk, c_drag_k, c_vortex_k = self.rk4_step(
                    z,
                    t,
                    dt,
                    reduced_velocity=reduced_velocity,
                    z_hist=z_hist,
                    ur_hist=ur_hist,
                    return_coeff_parts=True,
                )
            else:
                z, Fk = self.rk4_step(
                    z,
                    t,
                    dt,
                    reduced_velocity=reduced_velocity,
                    z_hist=z_hist,
                    ur_hist=ur_hist,
                )
            if self.is_tcn_force_model and z_hist is not None:
                z_hist = torch.cat([z_hist[:, 1:, :], z.unsqueeze(1)], dim=1)
                if ur_hist is not None:
                    rv_raw = self._prepare_reduced_velocity_raw(reduced_velocity, like=z[..., :1])
                    assert rv_raw is not None
                    ur_hist = torch.cat([ur_hist[:, 1:, :], rv_raw.unsqueeze(1)], dim=1)
            Z_pred.append(z)
            F_hist.append(Fk)
            if record_coeff_parts and self.use_two_head_vivana:
                assert c_drag_k is not None and c_vortex_k is not None
                C_drag_hist.append(c_drag_k)
                C_vortex_hist.append(c_vortex_k)

        Z_pred = torch.stack(Z_pred, dim=1)            # (B,K+1,D)
        if F_hist:
            F_hist = torch.stack([initial_force] + F_hist, dim=1)
        else:
            F_hist = None
        if record_coeff_parts:
            if self.use_two_head_vivana and C_drag_hist and C_vortex_hist:
                c_drag_hist = torch.stack([initial_c_drag] + C_drag_hist, dim=1)
                c_vortex_hist = torch.stack([initial_c_vortex] + C_vortex_hist, dim=1)
            else:
                c_drag_hist = None
                c_vortex_hist = None
            return Z_pred, F_hist, c_drag_hist, c_vortex_hist
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
        z_hist: torch.Tensor | None = None,
        ur_hist: torch.Tensor | None = None,
    ):
        if self.loss_discretization == "implicit_euler":
            return self.res_loss_Euler(
                zi,
                ti,
                zin,
                tin,
                reduced_velocity=reduced_velocity,
                z_hist=z_hist,
                ur_hist=ur_hist,
            )
        return self.res_loss_SRK4(
            zi,
            ti,
            zin,
            tin,
            reduced_velocity=reduced_velocity,
            z_hist=z_hist,
            ur_hist=ur_hist,
        )
    
    def avg_force(
        self,
        zi,
        ti,
        zin,
        tin,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
        z_hist: torch.Tensor | None = None,
        ur_hist: torch.Tensor | None = None,
    ):
        if self.loss_discretization == "implicit_euler":
            return self.avg_force_Euler(
                zi,
                ti,
                zin,
                tin,
                reduced_velocity=reduced_velocity,
                z_hist=z_hist,
                ur_hist=ur_hist,
            )
        return self.avg_force_SRK4(
            zi,
            ti,
            zin,
            tin,
            reduced_velocity=reduced_velocity,
            z_hist=z_hist,
            ur_hist=ur_hist,
        )

    def res_loss_per_sample(
        self,
        zi,
        ti,
        zin,
        tin,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
        z_hist: torch.Tensor | None = None,
        ur_hist: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.loss_discretization == "implicit_euler":
            return self.res_loss_Euler_per_sample(
                zi,
                ti,
                zin,
                tin,
                reduced_velocity=reduced_velocity,
                z_hist=z_hist,
                ur_hist=ur_hist,
            )
        return self.res_loss_SRK4_per_sample(
            zi,
            ti,
            zin,
            tin,
            reduced_velocity=reduced_velocity,
            z_hist=z_hist,
            ur_hist=ur_hist,
        )

    def avg_force_per_sample(
        self,
        zi,
        ti,
        zin,
        tin,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
        z_hist: torch.Tensor | None = None,
        ur_hist: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.loss_discretization == "implicit_euler":
            return self.avg_force_Euler_per_sample(
                zi,
                ti,
                zin,
                tin,
                reduced_velocity=reduced_velocity,
                z_hist=z_hist,
                ur_hist=ur_hist,
            )
        return self.avg_force_SRK4_per_sample(
            zi,
            ti,
            zin,
            tin,
            reduced_velocity=reduced_velocity,
            z_hist=z_hist,
            ur_hist=ur_hist,
        )

    def avg_force_coeff(
        self,
        zi,
        ti,
        zin,
        tin,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
        z_hist: torch.Tensor | None = None,
        ur_hist: torch.Tensor | None = None,
    ):
        if self.loss_discretization == "implicit_euler":
            return self.avg_force_coeff_Euler(
                zi,
                ti,
                zin,
                tin,
                reduced_velocity=reduced_velocity,
                z_hist=z_hist,
                ur_hist=ur_hist,
            )
        return self.avg_force_coeff_SRK4(
            zi,
            ti,
            zin,
            tin,
            reduced_velocity=reduced_velocity,
            z_hist=z_hist,
            ur_hist=ur_hist,
        )

    def avg_force_coeff_per_sample(
        self,
        zi,
        ti,
        zin,
        tin,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
        z_hist: torch.Tensor | None = None,
        ur_hist: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.loss_discretization == "implicit_euler":
            return self.avg_force_coeff_Euler_per_sample(
                zi,
                ti,
                zin,
                tin,
                reduced_velocity=reduced_velocity,
                z_hist=z_hist,
                ur_hist=ur_hist,
            )
        return self.avg_force_coeff_SRK4_per_sample(
            zi,
            ti,
            zin,
            tin,
            reduced_velocity=reduced_velocity,
            z_hist=z_hist,
            ur_hist=ur_hist,
        )

    def avg_drag_vortex_coeff(
        self,
        zi,
        ti,
        zin,
        tin,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
        z_hist: torch.Tensor | None = None,
        ur_hist: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_two_head_vivana:
            raise ValueError("avg_drag_vortex_coeff requires force_output='two_head_vivana'.")
        if self.loss_discretization == "implicit_euler":
            return self.avg_drag_vortex_coeff_Euler(
                zi,
                ti,
                zin,
                tin,
                reduced_velocity=reduced_velocity,
                z_hist=z_hist,
                ur_hist=ur_hist,
            )
        return self.avg_drag_vortex_coeff_SRK4(
            zi,
            ti,
            zin,
            tin,
            reduced_velocity=reduced_velocity,
            z_hist=z_hist,
            ur_hist=ur_hist,
        )

    def avg_drag_vortex_coeff_per_sample(
        self,
        zi,
        ti,
        zin,
        tin,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
        z_hist: torch.Tensor | None = None,
        ur_hist: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_two_head_vivana:
            raise ValueError("avg_drag_vortex_coeff_per_sample requires force_output='two_head_vivana'.")
        if self.loss_discretization == "implicit_euler":
            return self.avg_drag_vortex_coeff_Euler_per_sample(
                zi,
                ti,
                zin,
                tin,
                reduced_velocity=reduced_velocity,
                z_hist=z_hist,
                ur_hist=ur_hist,
            )
        return self.avg_drag_vortex_coeff_SRK4_per_sample(
            zi,
            ti,
            zin,
            tin,
            reduced_velocity=reduced_velocity,
            z_hist=z_hist,
            ur_hist=ur_hist,
        )

    def _pair_dt(self, ti: torch.Tensor | None, tin: torch.Tensor | None, like: torch.Tensor) -> torch.Tensor:
        if ti is None or tin is None:
            return like.new_full((like.shape[0], 1), float(self.dt))
        dt = tin - ti
        if dt.ndim == 1:
            dt = dt.unsqueeze(1)
        dt_safe = torch.where(dt.abs() > 0.0, dt, dt.new_full(dt.shape, float(self.dt)))
        return dt_safe

    def res_loss_Euler(
        self,
        zi,
        ti,
        zin,
        tin,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
        z_hist: torch.Tensor | None = None,
        ur_hist: torch.Tensor | None = None,
    ):
        dt_pair = self._pair_dt(ti, tin, like=zi)
        dz = (zin - zi) / dt_pair
        z_mean = 0.5 * (zin + zi)
        res = dz - self.g(
            z_mean,
            reduced_velocity=reduced_velocity,
            z_hist=z_hist,
            ur_hist=ur_hist,
        )
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

    def res_loss_Euler_per_sample(
        self,
        zi,
        ti,
        zin,
        tin,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
        z_hist: torch.Tensor | None = None,
        ur_hist: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dt_pair = self._pair_dt(ti, tin, like=zi)
        dz = (zin - zi) / dt_pair
        z_mean = 0.5 * (zin + zi)
        res = dz - self.g(
            z_mean,
            reduced_velocity=reduced_velocity,
            z_hist=z_hist,
            ur_hist=ur_hist,
        )
        res_scaled = res / self.res_scale
        if self.force_output == "coefficient":
            f0 = self._force_scale_from_reduced_velocity(
                reduced_velocity,
                like=res_scaled[..., 1:2],
                state=z_mean,
            )
            res_scaled = res_scaled.clone()
            res_scaled[..., 1:2] = res_scaled[..., 1:2] / f0
        return torch.sum(res_scaled**2, dim=1)

    def avg_force_Euler(
        self,
        zi,
        ti,
        zin,
        tin,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
        z_hist: torch.Tensor | None = None,
        ur_hist: torch.Tensor | None = None,
    ):
        z_mean = 0.5 * (zin + zi)
        forces = self.f(
            z_mean,
            reduced_velocity=reduced_velocity,
            z_hist=z_hist,
            ur_hist=ur_hist,
        )
        loss = torch.mean(torch.linalg.norm(forces, ord=1, dim=1))
        return loss

    def avg_force_Euler_per_sample(
        self,
        zi,
        ti,
        zin,
        tin,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
        z_hist: torch.Tensor | None = None,
        ur_hist: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z_mean = 0.5 * (zin + zi)
        forces = self.f(
            z_mean,
            reduced_velocity=reduced_velocity,
            z_hist=z_hist,
            ur_hist=ur_hist,
        )
        return torch.linalg.norm(forces, ord=1, dim=1)

    def avg_force_coeff_Euler(
        self,
        zi,
        ti,
        zin,
        tin,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
        z_hist: torch.Tensor | None = None,
        ur_hist: torch.Tensor | None = None,
    ):
        z_mean = 0.5 * (zin + zi)
        force_coeff = self.u_theta_coeff(
            z_mean,
            reduced_velocity=reduced_velocity,
            z_hist=z_hist,
            ur_hist=ur_hist,
        )
        return torch.mean(torch.linalg.norm(force_coeff, ord=1, dim=1))

    def avg_force_coeff_Euler_per_sample(
        self,
        zi,
        ti,
        zin,
        tin,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
        z_hist: torch.Tensor | None = None,
        ur_hist: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z_mean = 0.5 * (zin + zi)
        force_coeff = self.u_theta_coeff(
            z_mean,
            reduced_velocity=reduced_velocity,
            z_hist=z_hist,
            ur_hist=ur_hist,
        )
        return torch.linalg.norm(force_coeff, ord=1, dim=1)

    def avg_drag_vortex_coeff_Euler(
        self,
        zi,
        ti,
        zin,
        tin,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
        z_hist: torch.Tensor | None = None,
        ur_hist: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        drag_per, vortex_per = self.avg_drag_vortex_coeff_Euler_per_sample(
            zi,
            ti,
            zin,
            tin,
            reduced_velocity=reduced_velocity,
            z_hist=z_hist,
            ur_hist=ur_hist,
        )
        return torch.mean(drag_per), torch.mean(vortex_per)

    def avg_drag_vortex_coeff_Euler_per_sample(
        self,
        zi,
        ti,
        zin,
        tin,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
        z_hist: torch.Tensor | None = None,
        ur_hist: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z_mean = 0.5 * (zin + zi)
        raw = self._force_net_raw(
            z_mean,
            reduced_velocity=reduced_velocity,
            z_hist=z_hist,
            ur_hist=ur_hist,
        )
        c_drag, c_vortex = self._two_head_vivana_coeff_from_raw(raw)
        return torch.sum(torch.abs(c_drag), dim=1), torch.sum(torch.abs(c_vortex), dim=1)


    def res_loss_SRK4(
        self,
        zi,
        ti,
        zin,
        tin,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
        z_hist: torch.Tensor | None = None,
        ur_hist: torch.Tensor | None = None,
    ):
        dt = self.dt
        # constants from the scheme
        a = 0.5
        b = math.sqrt(3.0) / 6.0

        # finite difference
        dz_fd = (zin - zi) / dt              # (B, d)

        # midpoint between zn and zn+1
        z_mid = 0.5 * (zi + zin)             # (B, d)

        # stage convex combos
        z_a_plus  = (0.5 + b) * zi + (0.5 - b) * zin   # (B, d)
        z_a_minus = (0.5 - b) * zi + (0.5 + b) * zin   # (B, d)

        # stage evaluations of g
        g_a_plus = self.g(
            z_a_plus,
            reduced_velocity=reduced_velocity,
            z_hist=z_hist,
            ur_hist=ur_hist,
        )                  # (B, d)
        g_a_minus = self.g(
            z_a_minus,
            reduced_velocity=reduced_velocity,
            z_hist=z_hist,
            ur_hist=ur_hist,
        )                 # (B, d)

        # two corrected midpoints
        z_corr_minus = z_mid - b * dt * g_a_plus      # (B, d)
        z_corr_plus  = z_mid + b * dt * g_a_minus     # (B, d)

        # final two g-evals
        g1 = self.g(
            z_corr_minus,
            reduced_velocity=reduced_velocity,
            z_hist=z_hist,
            ur_hist=ur_hist,
        )                     # (B, d)
        g2 = self.g(
            z_corr_plus,
            reduced_velocity=reduced_velocity,
            z_hist=z_hist,
            ur_hist=ur_hist,
        )                      # (B, d)

        dz_model = 0.5 * g1 + 0.5 * g2                # (B, d)

        # residual
        res = dz_fd - dz_model                        # (B, d)

        # scale like before, but for time-derivatives
        res_scaled = res / self.res_scale
        if self.force_output == "coefficient":
            f0 = self._force_scale_from_reduced_velocity(
                reduced_velocity,
                like=res_scaled[..., 1:2],
                state=z_mid,
            )
            res_scaled = res_scaled.clone()
            res_scaled[..., 1:2] = res_scaled[..., 1:2] / f0

        loss = torch.mean(torch.sum(res_scaled**2, dim=1))
        return loss

    def res_loss_SRK4_per_sample(
        self,
        zi,
        ti,
        zin,
        tin,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
        z_hist: torch.Tensor | None = None,
        ur_hist: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dt = self.dt
        a = 0.5
        b = math.sqrt(3.0) / 6.0

        dz_fd = (zin - zi) / dt
        z_mid = 0.5 * (zi + zin)
        z_a_plus = (0.5 + b) * zi + (0.5 - b) * zin
        z_a_minus = (0.5 - b) * zi + (0.5 + b) * zin

        g_a_plus = self.g(z_a_plus, reduced_velocity=reduced_velocity, z_hist=z_hist, ur_hist=ur_hist)
        g_a_minus = self.g(z_a_minus, reduced_velocity=reduced_velocity, z_hist=z_hist, ur_hist=ur_hist)

        z_corr_minus = z_mid - b * dt * g_a_plus
        z_corr_plus = z_mid + b * dt * g_a_minus

        g1 = self.g(z_corr_minus, reduced_velocity=reduced_velocity, z_hist=z_hist, ur_hist=ur_hist)
        g2 = self.g(z_corr_plus, reduced_velocity=reduced_velocity, z_hist=z_hist, ur_hist=ur_hist)

        dz_model = 0.5 * g1 + 0.5 * g2
        res = dz_fd - dz_model
        res_scaled = res / self.res_scale
        if self.force_output == "coefficient":
            f0 = self._force_scale_from_reduced_velocity(
                reduced_velocity,
                like=res_scaled[..., 1:2],
                state=z_mid,
            )
            res_scaled = res_scaled.clone()
            res_scaled[..., 1:2] = res_scaled[..., 1:2] / f0
        return torch.sum(res_scaled**2, dim=1)
    
    def avg_force_SRK4(
        self,
        zi,
        ti,
        zin,
        tin,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
        z_hist: torch.Tensor | None = None,
        ur_hist: torch.Tensor | None = None,
    ):
        dt = self.dt
        b = math.sqrt(3.0) / 6.0

        # same stage points as in res_loss
        z_a_plus  = (0.5 + b) * zi + (0.5 - b) * zin
        z_a_minus = (0.5 - b) * zi + (0.5 + b) * zin

        # evaluate learned force at both stages
        f1 = self.f(
            z_a_plus,
            reduced_velocity=reduced_velocity,
            z_hist=z_hist,
            ur_hist=ur_hist,
        )    # assume (B, 1) or (B, 2)
        f2 = self.f(
            z_a_minus,
            reduced_velocity=reduced_velocity,
            z_hist=z_hist,
            ur_hist=ur_hist,
        )

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
        z_hist: torch.Tensor | None = None,
        ur_hist: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b = math.sqrt(3.0) / 6.0
        z_a_plus = (0.5 + b) * zi + (0.5 - b) * zin
        z_a_minus = (0.5 - b) * zi + (0.5 + b) * zin
        f1 = self.f(z_a_plus, reduced_velocity=reduced_velocity, z_hist=z_hist, ur_hist=ur_hist)
        f2 = self.f(z_a_minus, reduced_velocity=reduced_velocity, z_hist=z_hist, ur_hist=ur_hist)
        return 0.5 * torch.sum(torch.abs(f1), dim=1) + 0.5 * torch.sum(torch.abs(f2), dim=1)

    def avg_force_coeff_SRK4(
        self,
        zi,
        ti,
        zin,
        tin,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
        z_hist: torch.Tensor | None = None,
        ur_hist: torch.Tensor | None = None,
    ):
        dt = self.dt
        b = math.sqrt(3.0) / 6.0

        z_a_plus = (0.5 + b) * zi + (0.5 - b) * zin
        z_a_minus = (0.5 - b) * zi + (0.5 + b) * zin

        f1 = self.f(z_a_plus, reduced_velocity=reduced_velocity, z_hist=z_hist, ur_hist=ur_hist)
        f2 = self.f(z_a_minus, reduced_velocity=reduced_velocity, z_hist=z_hist, ur_hist=ur_hist)
        f0 = self._force_scale_from_reduced_velocity(
            reduced_velocity,
            like=f1,
            state=z_a_plus,
        )

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
        z_hist: torch.Tensor | None = None,
        ur_hist: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b = math.sqrt(3.0) / 6.0
        z_a_plus = (0.5 + b) * zi + (0.5 - b) * zin
        z_a_minus = (0.5 - b) * zi + (0.5 + b) * zin
        f1 = self.f(z_a_plus, reduced_velocity=reduced_velocity, z_hist=z_hist, ur_hist=ur_hist)
        f2 = self.f(z_a_minus, reduced_velocity=reduced_velocity, z_hist=z_hist, ur_hist=ur_hist)
        f0 = self._force_scale_from_reduced_velocity(
            reduced_velocity,
            like=f1,
            state=z_a_plus,
        )
        f1c = f1 / f0
        f2c = f2 / f0
        return 0.5 * torch.sum(torch.abs(f1c), dim=1) + 0.5 * torch.sum(torch.abs(f2c), dim=1)

    def avg_drag_vortex_coeff_SRK4(
        self,
        zi,
        ti,
        zin,
        tin,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
        z_hist: torch.Tensor | None = None,
        ur_hist: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        drag_per, vortex_per = self.avg_drag_vortex_coeff_SRK4_per_sample(
            zi,
            ti,
            zin,
            tin,
            reduced_velocity=reduced_velocity,
            z_hist=z_hist,
            ur_hist=ur_hist,
        )
        return torch.mean(drag_per), torch.mean(vortex_per)

    def avg_drag_vortex_coeff_SRK4_per_sample(
        self,
        zi,
        ti,
        zin,
        tin,
        reduced_velocity: torch.Tensor | np.ndarray | float | None = None,
        z_hist: torch.Tensor | None = None,
        ur_hist: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b = math.sqrt(3.0) / 6.0
        z_a_plus = (0.5 + b) * zi + (0.5 - b) * zin
        z_a_minus = (0.5 - b) * zi + (0.5 + b) * zin
        raw1 = self._force_net_raw(
            z_a_plus,
            reduced_velocity=reduced_velocity,
            z_hist=z_hist,
            ur_hist=ur_hist,
        )
        raw2 = self._force_net_raw(
            z_a_minus,
            reduced_velocity=reduced_velocity,
            z_hist=z_hist,
            ur_hist=ur_hist,
        )
        c_drag1, c_vortex1 = self._two_head_vivana_coeff_from_raw(raw1)
        c_drag2, c_vortex2 = self._two_head_vivana_coeff_from_raw(raw2)
        drag_per = 0.5 * torch.sum(torch.abs(c_drag1), dim=1) + 0.5 * torch.sum(torch.abs(c_drag2), dim=1)
        vortex_per = 0.5 * torch.sum(torch.abs(c_vortex1), dim=1) + 0.5 * torch.sum(torch.abs(c_vortex2), dim=1)
        return drag_per, vortex_per
    
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
    fig, axes = plt.subplots(4, 1, figsize=(6, 12), sharex=False)
    ax_full, ax_diff, ax_zoom, ax_middle = axes
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

    mid_start, mid_end = middle_window
    ax_middle.plot(t[middle_mask], y_true_norm[middle_mask], label="y/D (true)")
    ax_middle.plot(t[middle_mask], y_pred_norm[middle_mask], label="y/D (pred)")
    ax_middle.set_xlabel("time")
    ax_middle.set_ylabel("y/D")
    ax_middle.grid(True, alpha=0.3)
    ax_middle.set_title(f"Normalized rollout ({mid_start}-{mid_end}s) epoch {epoch+1}{ur_title}{title_suffix}")
    ax_middle.legend(loc="upper right")

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
    drag_coeff_pred: np.ndarray | None = None,
    vortex_coeff_pred: np.ndarray | None = None,
    reduced_velocity: float | None = None,
    *,
    tag_prefix: str = "val/rollout",
    step: int | None = None,
    title_suffix: str = "",
):
    fig, axes = plt.subplots(4, 1, figsize=(6, 12), sharex=False)
    ax_full, ax_diff, ax_zoom, ax_middle = axes
    ur_title = f" (U_r={float(reduced_velocity):.3f})" if reduced_velocity is not None else ""
    ax_full.plot(t, force_coeff_true, label="C_F (true)", color="tab:blue", alpha=0.7)
    ax_full.plot(t, force_coeff_pred, label="C_F (pred)", color="tab:purple")
    if drag_coeff_pred is not None:
        ax_full.plot(t, drag_coeff_pred, label="C_drag (pred)", color="tab:green", linestyle="--")
    if vortex_coeff_pred is not None:
        ax_full.plot(t, vortex_coeff_pred, label="C_vortex (pred)", color="tab:red", linestyle=":")
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
    if drag_coeff_pred is not None:
        ax_zoom.plot(t[zoom_mask], drag_coeff_pred[zoom_mask], label="C_drag (pred)", color="tab:green", linestyle="--")
    if vortex_coeff_pred is not None:
        ax_zoom.plot(t[zoom_mask], vortex_coeff_pred[zoom_mask], label="C_vortex (pred)", color="tab:red", linestyle=":")
    ax_zoom.set_xlabel("time")
    ax_zoom.set_ylabel("C_F")
    ax_zoom.grid(True, alpha=0.3)
    ax_zoom.set_title(f"Force coefficient rollout (first 1s) epoch {epoch+1}{ur_title}{title_suffix}")
    ax_zoom.legend(loc="upper right")

    mid_start, mid_end = middle_window
    ax_middle.plot(t[middle_mask], force_coeff_true[middle_mask], label="C_F (true)", color="tab:blue", alpha=0.7)
    ax_middle.plot(t[middle_mask], force_coeff_pred[middle_mask], label="C_F (pred)", color="tab:purple")
    if drag_coeff_pred is not None:
        ax_middle.plot(
            t[middle_mask],
            drag_coeff_pred[middle_mask],
            label="C_drag (pred)",
            color="tab:green",
            linestyle="--",
        )
    if vortex_coeff_pred is not None:
        ax_middle.plot(
            t[middle_mask],
            vortex_coeff_pred[middle_mask],
            label="C_vortex (pred)",
            color="tab:red",
            linestyle=":",
        )
    ax_middle.set_xlabel("time")
    ax_middle.set_ylabel("C_F")
    ax_middle.grid(True, alpha=0.3)
    ax_middle.set_title(f"Force coefficient rollout ({mid_start}-{mid_end}s) epoch {epoch+1}{ur_title}{title_suffix}")
    ax_middle.legend(loc="upper right")

    plt.tight_layout()
    writer.add_figure(f"{tag_prefix}_force", fig, epoch + 1 if step is None else step)
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
        (DISP_ROLLOUT_NRMSE_KEY, DISP_ROLLOUT_NRMSE_KEY),
        (FORCE_ROLLOUT_NRMSE_KEY, FORCE_ROLLOUT_NRMSE_KEY),
        (FORCE_MAPPING_NRMSE_KEY, FORCE_MAPPING_NRMSE_KEY),
        (DOMINANT_FREQ_REL_ERROR_KEY, DOMINANT_FREQ_REL_ERROR_KEY),
        (MEAN_DISP_AMP_REL_ERROR_KEY, MEAN_DISP_AMP_REL_ERROR_KEY),
        (DISP_SPECTRAL_SHAPE_ERROR_KEY, DISP_SPECTRAL_SHAPE_ERROR_KEY),
        (FORCE_SPECTRAL_SHAPE_ERROR_KEY, FORCE_SPECTRAL_SHAPE_ERROR_KEY),
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

    # Also log average Force mapping NRMSE vs reduced velocity (grouped by U_r).
    force_key = FORCE_MAPPING_NRMSE_KEY
    grouped: dict[float, list[float]] = {}
    for ur_val, metrics in pairs:
        if force_key not in metrics:
            continue
        y_val = float(metrics[force_key])
        if not np.isfinite(y_val) or y_val <= 0.0:
            continue
        grouped.setdefault(float(ur_val), []).append(y_val)
    if grouped:
        xs = sorted(grouped.keys())
        ys = [float(np.mean(grouped[x])) for x in xs]
        fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4))
        ax2.plot(xs, ys, marker="o", label=f"Avg {FORCE_MAPPING_NRMSE_KEY}")
        ax2.set_xlabel("Reduced velocity (U_r)")
        ax2.set_ylabel("Error")
        ax2.set_yscale("log")
        ax2.set_title(f"Avg {FORCE_MAPPING_NRMSE_KEY} vs U_r")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="best")
        plt.tight_layout()
        writer.add_figure(f"final_val/{FORCE_MAPPING_NRMSE_KEY}_avg_vs_U_r", fig2, epoch)
        plt.close(fig2)

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
    if velocity is not None and np.asarray(velocity).shape[0] != t.shape[0]:
        raise ValueError("Velocity array must have the same length as the time vector.")
    if hamiltonian is not None and np.asarray(hamiltonian).shape[0] != t.shape[0]:
        raise ValueError("Hamiltonian array must have the same length as the time vector when provided.")
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
    history_len: int = 0,
    filter_too_short_series: bool = False,
    min_required_samples: int | None = None,
) -> tuple[DataLoader, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]], int]:
    if not series_data:
        raise ValueError("series_data must contain at least one (y, t, dt, vel, force, U_r) tuple.")
    sequence_tensors: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    datasets: list[TensorDataset | ConcatDataset] = []
    min_length: int | None = None
    required_len = int(history_len) + 2 if int(history_len) > 0 else 2
    if min_required_samples is not None:
        required_len = max(required_len, int(min_required_samples))
    if required_len < 2:
        required_len = 2
    skipped_short: list[tuple[int, int]] = []
    for idx, (y_np, t_np, dt_value, vel_np, force_np, ur_np) in enumerate(series_data):
        seq_len_raw = int(np.asarray(y_np, dtype=float).reshape(-1).shape[0])
        if bool(filter_too_short_series) and seq_len_raw < required_len:
            skipped_short.append((idx, seq_len_raw))
            continue
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
                history_len=history_len,
            )
        )
        seq_len = y_tensor.shape[0]
        min_length = seq_len if min_length is None else min(min_length, seq_len)
    if skipped_short:
        details = ", ".join([f"idx={i}:len={n}" for i, n in skipped_short[:8]])
        suffix = " ..." if len(skipped_short) > 8 else ""
        print(
            f"Filtered out {len(skipped_short)} short time series before dataloader build "
            f"(required_len={required_len}): {details}{suffix}"
        )
    if not datasets:
        raise ValueError(
            "No usable time series remain after short-series filtering. "
            f"required_len={required_len}, original_count={len(series_data)}."
        )
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
    traj_scale: float | None = None,
    history_len: int = 0,
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
    hist = max(0, int(history_len))
    if hist > 0 and z.shape[0] < (hist + 2):
        raise ValueError(
            f"Not enough samples ({int(z.shape[0])}) for history_len={hist}. Need at least {hist + 2}."
        )
    scale_tensor: torch.Tensor | None = None
    if hist == 0:
        if traj_scale is not None:
            scale_tensor = torch.full((z.shape[0] - 1,), float(traj_scale), dtype=torch.float32)
        if force_tensor is None:
            if scale_tensor is None:
                return TensorDataset(
                    z[:-1],
                    t_tensor[:-1].unsqueeze(1),
                    z[1:],
                    t_tensor[1:].unsqueeze(1),
                    ur[:-1],
                )
            return TensorDataset(
                z[:-1],
                t_tensor[:-1].unsqueeze(1),
                z[1:],
                t_tensor[1:].unsqueeze(1),
                ur[:-1],
                scale_tensor,
            )
        if force_tensor.shape[0] != z.shape[0]:
            raise ValueError("force_tensor must match the sequence length.")
        if scale_tensor is None:
            return TensorDataset(
                z[:-1],
                t_tensor[:-1].unsqueeze(1),
                z[1:],
                t_tensor[1:].unsqueeze(1),
                ur[:-1],
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
            scale_tensor,
        )

    idx = torch.arange(hist, z.shape[0] - 1, dtype=torch.long)
    z_i = z[idx]
    z_next = z[idx + 1]
    t_i = t_tensor[idx].unsqueeze(1)
    t_next = t_tensor[idx + 1].unsqueeze(1)
    ur_i = ur[idx]

    z_hist = z.unfold(0, hist + 1, 1).permute(0, 2, 1)[:-1].contiguous()
    ur_hist = ur.unfold(0, hist + 1, 1).permute(0, 2, 1)[:-1].contiguous()

    if traj_scale is not None:
        scale_tensor = torch.full((z_i.shape[0],), float(traj_scale), dtype=torch.float32)

    if force_tensor is None:
        if scale_tensor is None:
            return TensorDataset(
                z_i,
                t_i,
                z_next,
                t_next,
                ur_i,
                z_hist,
                ur_hist,
            )
        return TensorDataset(
            z_i,
            t_i,
            z_next,
            t_next,
            ur_i,
            z_hist,
            ur_hist,
            scale_tensor,
        )
    if force_tensor.shape[0] != z.shape[0]:
        raise ValueError("force_tensor must match the sequence length.")
    if scale_tensor is None:
        return TensorDataset(
            z_i,
            t_i,
            z_next,
            t_next,
            ur_i,
            z_hist,
            ur_hist,
            force_tensor[idx].unsqueeze(1),
            force_tensor[idx + 1].unsqueeze(1),
        )
    return TensorDataset(
        z_i,
        t_i,
        z_next,
        t_next,
        ur_i,
        z_hist,
        ur_hist,
        force_tensor[idx].unsqueeze(1),
        force_tensor[idx + 1].unsqueeze(1),
        scale_tensor,
    )

def build_rollout_dataset(
    y_data_t: torch.Tensor,
    vel: torch.Tensor,
    m_eff: float,
    t_tensor: torch.Tensor,
    rollout_steps: int,
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

    return TensorDataset(z0_batch, t_seq_batch, z_traj_batch)

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
    validation_start_idx: int = 0,
) -> dict[str, np.ndarray]:
    """Roll the model forward over the full time grid and return normalised traces."""
    start_idx = max(0, int(validation_start_idx))
    if start_idx >= int(y0.shape[0]):
        raise ValueError(
            f"validation_start_idx={start_idx} is out of bounds for sequence length {int(y0.shape[0])}."
        )
    p0 = vel[start_idx] * m_eff
    z0 = torch.stack((y0[start_idx], p0), dim=0).unsqueeze(0).to(device)
    t_arr = np.asarray(t, dtype=np.float32).reshape(-1)
    total_len = int(y0.shape[0])
    expected_post_start_len = max(0, total_len - start_idx)
    if start_idx > 0:
        # Support both call patterns:
        # 1) t is full series -> trim by start_idx here.
        # 2) t is already post-start series -> do not trim again.
        if int(t_arr.size) == total_len:
            t_eval = t_arr[start_idx:]
        elif int(t_arr.size) == expected_post_start_len:
            t_eval = t_arr
        else:
            raise ValueError(
                f"Time vector length ({int(t_arr.size)}) is incompatible with validation_start_idx={start_idx} "
                f"and sequence length {total_len}."
            )
    else:
        t_eval = t_arr
    t_seq = torch.from_numpy(t_eval).to(device=device).unsqueeze(0)
    rv_val = reduced_velocity
    if torch.is_tensor(rv_val):
        rv_val = float(rv_val.reshape(-1)[0].detach().cpu())
    else:
        rv_val = float(np.asarray(rv_val).reshape(-1)[0])
    rv_tensor = torch.tensor(rv_val, dtype=z0.dtype, device=device).view(1, 1)

    z_hist_init = None
    ur_hist_init = None
    if bool(getattr(model, "is_tcn_force_model", False)):
        history_len = int(getattr(model, "history_len", 0))
        if start_idx < history_len:
            available_s = float(start_idx * dt)
            needed_s = float(history_len * dt)
            start_t = float(np.asarray(t, dtype=float).reshape(-1)[0])
            raise ValueError(
                f"validation start at t={start_t + start_idx * dt:.6g}s (index={start_idx}) is too early "
                f"for TCN history_len={history_len}. Need at least {needed_s:.6g}s ({history_len} samples) "
                f"before validation start, but only {available_s:.6g}s are available."
            )
        hist_start = start_idx - history_len
        z_full = torch.stack((y0, vel * m_eff), dim=1).to(device=device)
        z_hist_init = z_full[hist_start : start_idx + 1, :].unsqueeze(0)
        ur_hist_init = rv_tensor.unsqueeze(1).expand(1, history_len + 1, 1).contiguous()

    with torch.no_grad():
        use_two_head_vivana = bool(getattr(model, "use_two_head_vivana", False))
        if use_two_head_vivana:
            z_pred, f_hist, c_drag_hist, c_vortex_hist = model.rollout(
                z0,
                t_seq,
                dt,
                reduced_velocity=rv_tensor,
                z_hist_init=z_hist_init,
                ur_hist_init=ur_hist_init,
                record_coeff_parts=True,
            )
        else:
            z_pred, f_hist = model.rollout(
                z0,
                t_seq,
                dt,
                reduced_velocity=rv_tensor,
                z_hist_init=z_hist_init,
                ur_hist_init=ur_hist_init,
            )
            c_drag_hist = None
            c_vortex_hist = None
        if f_hist is None:
            force_total_t = model.u_theta_sequence(
                z_pred,
                reduced_velocity=rv_tensor.unsqueeze(1).expand(1, z_pred.shape[1], 1),
            )
        else:
            force_total_t = f_hist
        rv_seq = rv_tensor.unsqueeze(1).expand(1, z_pred.shape[1], 1)
        if model.is_tcn_force_model:
            force_model_t = model.learned_force_sequence(z_pred, reduced_velocity=rv_seq)
        else:
            force_model_t = model.learned_force(z_pred, reduced_velocity=rv_seq)
        force_drag_t = torch.zeros_like(force_total_t)
        drag_coeff_t = c_drag_hist
        vortex_coeff_t = c_vortex_hist
        hamiltonian_model_t = model.H(z_pred)

    y_samples_arr = z_pred[0, :, 0].detach().cpu().numpy()
    p_samples_arr = z_pred[0, :, 1].detach().cpu().numpy()
    y_pred_norm = y_samples_arr / D
    p_pred_norm = (p_samples_arr / m_eff) / (np.sqrt(k / m_eff) * D)
    force_total_arr = force_total_t[0, :, 0].detach().cpu().numpy()
    force_drag_arr = force_drag_t[0, :, 0].detach().cpu().numpy()
    force_model_arr = force_model_t[0, :, 0].detach().cpu().numpy()
    hamiltonian_model_arr = hamiltonian_model_t[0, :].detach().cpu().numpy()
    result = {
        "y_norm": y_pred_norm,
        "p_norm": p_pred_norm,
        "force_total": force_total_arr,
        "force_drag": force_drag_arr,
        "force_model": force_model_arr,
        "hamiltonian_model": hamiltonian_model_arr,
    }
    if drag_coeff_t is not None and vortex_coeff_t is not None:
        result["drag_coeff_pred"] = drag_coeff_t[0, :, 0].detach().cpu().numpy()
        result["vortex_coeff_pred"] = vortex_coeff_t[0, :, 0].detach().cpu().numpy()
    return result
