import math
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

from rainflow import count_cycles as _rainflow_count_cycles

try:
    from scipy.signal import savgol_filter
except ImportError:
    savgol_filter = None

from architectures import FourierFeatures, ODEPirateNet

@dataclass
class DataConfig:
    file: str = "data.npz"
    steadystate: bool = False
    steadystate_time_threshold: float = 10.0
    reduce_time: bool = False
    reduction_factor: int = 1
    middle_time_plot: list[float] = field(default_factory=lambda: [15.0, 17.0])
    use_generated_train_series: bool = False
    train_series_dir: str = "Data_Gen/generated_series"
    context_len: int = 20

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
    learn_hamiltonian: bool = False
    discover_damping: bool = False
    use_pirate_force: bool = False
    pirate_force_kwargs: dict[str, Any] = field(default_factory=dict)
    use_fourier_features: bool = False
    fourier_features: int = 64
    fourier_sigma: float = 1.0
    use_feature_engineering: bool = False
    q_scale: float | None = None
    p_scale: float | None = None

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
    force_context_mode: str = "none"  # options: none, flatten, cnn
    context_cnn_channels: int = 64
    cnn_kwargs: dict[str, Any] = field(default_factory=dict)
    append_current_state: bool = False

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
    decay_steps: int = 1000
    min_lr: float = 1e-5
    scheduler_type: str = "cosine"  # or "exponential"

@dataclass
class TrainingConfig:
    batch_size: int = 32
    force_reg: float = 1e-2
    max_grad_norm: float = 1e4
    lr: float = 1e-3
    optimizer: str = "adam"
    weight_decay: float = 0.0
    epochs: int = 2000
    rollout_every_epoch: int = 50
    use_lr_scheduler: bool = False
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    use_gradnorm: bool = False
    gradnorm_alpha: float = 0.9
    gradnorm_eps: float = 1e-8
    gradnorm_min_weight: float = 0.1
    gradnorm_max_weight: float = 10.0

@dataclass
class LoggingConfig:
    run_dir_root: str = "HNNruns"

@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data


def parse_config(raw: dict[str, Any]) -> Config:
    data_cfg = raw.get("data", {}) or {}
    model_cfg = raw.get("model", {}) or {}
    architecture_cfg = dict(raw.get("architecture", {}) or {})
    smoothing_cfg = raw.get("smoothing", {}) or {}
    training_cfg = raw.get("training", {}) or {}
    logging_cfg = raw.get("logging", {}) or {}

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

    data = DataConfig(**data_cfg)
    model = ModelConfig(**model_cfg)
    architecture = ArchitectureConfig(**architecture_cfg)
    smoothing = SmoothingConfig(**smoothing_cfg)
    scheduler_dict = training_cfg.get("scheduler", {}) or {}
    scheduler = SchedulerConfig(**scheduler_dict)
    training_fields = {k: v for k, v in training_cfg.items() if k != "scheduler"}
    training = TrainingConfig(**training_fields, scheduler=scheduler)
    logging = LoggingConfig(**logging_cfg)
    return Config(
        data=data,
        model=model,
        architecture=architecture,
        smoothing=smoothing,
        training=training,
        logging=logging,
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
) -> dict[str, float]:
    context_mode = getattr(model, "force_context_mode", "none")
    model_context_len = max(1, int(getattr(model, "context_len", 1)))
    start_idx = max(0, model_context_len - 1)

    rollout = rollout_model(
        model,
        y_data_t,
        val_vel,
        m_eff,
        dt,
        t,
        D,
        k,
        device,
        start_idx=start_idx,
    )
    metrics: dict[str, float] = {}
    y_pred_raw = rollout["y_norm"] * D
    y_true_raw_aligned = y_data_raw[start_idx : start_idx + y_pred_raw.shape[0]]
    y_true_norm_aligned = y_true_norm[start_idx : start_idx + y_pred_raw.shape[0]]
    disp_std_raw = float(np.std(y_true_raw_aligned))
    if not np.isfinite(disp_std_raw) or disp_std_raw <= 0.0:
        disp_std_raw = 1.0
    rel_rmse_disp = float(np.sqrt(np.mean((y_pred_raw - y_true_raw_aligned) ** 2))) / disp_std_raw
    metrics["rel_rmse_y"] = rel_rmse_disp
    force_total_pred = np.asarray(rollout["force_total"]).reshape(-1)
    force_target_full = np.asarray(force_data).reshape(-1)
    force_target = force_target_full[start_idx : start_idx + force_total_pred.shape[0]]
    min_len = min(force_total_pred.shape[0], force_target.shape[0])
    if min_len > 0:
        rmse_force = float(
            np.sqrt(np.mean((force_total_pred[:min_len] - force_target[:min_len]) ** 2))
        )
        force_std = float(np.std(force_target[:min_len]))
        if not np.isfinite(force_std) or force_std <= 0.0:
            force_std = 1.0
        metrics["rel_rmse_force_total"] = rmse_force / force_std
        force_model_aligned = force_total_pred[:min_len]
        force_true_aligned = force_target[:min_len]
        damage_true = fatigue_damage(force_true_aligned)
        damage_model = fatigue_damage(force_model_aligned)
        damage_rel_err = relative_error(damage_model, damage_true)
        if np.isfinite(damage_rel_err):
            metrics["force_fatigue_damage_rel_error"] = damage_rel_err
    else:
        force_model_aligned = force_total_pred
        force_true_aligned = force_target
    half_idx_disp = len(y_true_norm_aligned) // 2
    y_true_half = y_true_norm_aligned[half_idx_disp:]
    y_model_half = rollout["y_norm"][half_idx_disp:]
    half_idx_force = force_true_aligned.size // 2
    force_true_half = force_true_aligned[half_idx_force:]
    force_model_half = force_model_aligned[half_idx_force:]
    if force_true_half.size > 0 and force_model_half.size > 0:
        spectral_rel_err = spectral_relative_error(force_true_half, force_model_half, dt)
        if np.isfinite(spectral_rel_err):
            metrics["force_spectral_rel_error_second_half"] = spectral_rel_err
    with torch.no_grad():
        z_true_full = torch.stack((y_data_t, val_vel * m_eff), dim=1)
        z_true = z_true_full[start_idx:]
        context_mode = getattr(model, "force_context_mode", "none")
        model_context_len = max(1, int(getattr(model, "context_len", 1)))
        if z_true.shape[0] == 0:
            force_on_data = np.asarray([])
        elif context_mode != "none" and model_context_len > 1:
            contexts: list[torch.Tensor] = []
            x_batch_list: list[torch.Tensor] = []
            total_len = z_true.shape[0]
            for end in range(1, total_len + 1):
                start_window = max(0, end - model_context_len)
                ctx = z_true[start_window:end]
                if ctx.shape[0] < model_context_len:
                    pad = ctx[0:1].repeat(model_context_len - ctx.shape[0], 1)
                    ctx = torch.cat((pad, ctx), dim=0)
                contexts.append(ctx)
                x_batch_list.append(ctx[-1])
            context_batch = torch.stack(contexts, dim=0).to(device)  # (N, context_len, 2)
            x_batch = torch.stack(x_batch_list, dim=0).to(device)    # (N, 2)
            force_on_data = model.u_theta(x_batch, context_z=context_batch).squeeze(-1).detach().cpu().numpy()
        else:
            force_on_data = model.u_theta(z_true).squeeze(-1).detach().cpu().numpy()
    min_len_data = min(force_on_data.shape[0], force_target.shape[0])
    if min_len_data > 0:
        force_data_pred = force_on_data[:min_len_data]
        force_data_true = force_target[:min_len_data]
        rmse_force_data = float(np.sqrt(np.mean((force_data_pred - force_data_true) ** 2)))
        force_std_data = float(np.std(force_data_true))
        if not np.isfinite(force_std_data) or force_std_data <= 0.0:
            force_std_data = 1.0
        metrics["rel_rmse_force_on_data"] = rmse_force_data / force_std_data
        damage_true_data = fatigue_damage(force_data_true)
        damage_pred_data = fatigue_damage(force_data_pred)
        damage_rel_data = relative_error(damage_pred_data, damage_true_data)
        if np.isfinite(damage_rel_data):
            metrics["force_fatigue_damage_rel_error_on_data"] = damage_rel_data
    for name, value in metrics.items():
        writer.add_scalar(f"val/{name}", value, epoch)
    t_aligned = t[start_idx:]
    zoom_mask = create_zoom_mask(t_aligned)
    middle_mask = create_window_mask(t_aligned, middle_time_plot)
    log_displacement_plots(
        writer,
        epoch,
        t_aligned,
        y_true_norm_aligned,
        rollout["y_norm"],
        rollout["p_norm"],
        zoom_mask,
        middle_mask,
        middle_time_plot,
    )
    log_force_plots(
        writer,
        epoch,
        t_aligned,
        rollout["force_total"],
        rollout["force_drag"],
        rollout["force_model"],
        force_target,
        zoom_mask,
        middle_mask,
        middle_time_plot,
        model.include_physical_drag,
    )
    log_hamiltonian_plots(
        writer,
        epoch,
        t_aligned,
        rollout["hamiltonian_model"],
        zoom_mask,
        middle_mask,
        middle_time_plot,
        hamiltonian_data=hamiltonian_data[start_idx:] if hamiltonian_data is not None else None,
    )
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


def fatigue_damage(signal: np.ndarray, exponent: float = 3.0) -> float:
    """Return cumulative fatigue damage using rainflow counting and range^exponent."""
    signal = np.asarray(signal, dtype=float)
    if signal.size < 3 or not np.all(np.isfinite(signal)):
        return float("nan")
    cycles = _rainflow_count_cycles(signal)
    damage = 0.0
    for cycle in cycles:
        if len(cycle) == 3:
            rng, mean, count = cycle
        else:
            rng, count = cycle
        rng = abs(float(rng))
        cnt = float(count)
        if rng > 0.0 and cnt > 0.0:
            damage += (rng ** exponent) * cnt
    return float(damage)

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
        learn_hamiltonian: bool = False,
        use_pirate_force: bool = False,
        pirate_force_kwargs: dict | None = None,
        use_fourier_features: bool = False,
        fourier_features: int = 64,
        fourier_sigma: float = 1.0,
        use_feature_engineering: bool = False,
        force_net_type: str | None = None,
        residual_kwargs: dict[str, Any] | None = None,
        mlp_kwargs: dict[str, Any] | None = None,
        context_len: int = 20,
        force_context_mode: str = "none",
        context_cnn_channels: int = 64,
        context_cnn_kwargs: dict[str, Any] | None = None,
        append_current_state: bool = False,
    ):
        super().__init__()
        self.dt = dt
        self.m = m
        self.k = k
        self.U = U
        self.rho = rho
        self.D = D
        self.max_damping_ratio = torch.tensor(max_damping_ratio)
        self.q_scale = q_scale
        self.p_scale = p_scale
        self.discover_damping = bool(discover_damping)
        self.include_physical_drag = bool(include_physical_drag)
        self.learn_hamiltonian = bool(learn_hamiltonian)
        self.use_feature_engineering = bool(use_feature_engineering)
        self.engineered_feature_dim = 7
        self.force_input_dim = self.engineered_feature_dim if self.use_feature_engineering else 2
        self.context_len = max(1, int(context_len))
        self.force_context_mode = str(force_context_mode).lower()
        if self.force_context_mode not in {"none", "flatten", "cnn"}:
            raise ValueError("force_context_mode must be one of {'none', 'flatten', 'cnn'}.")
        self.context_cnn_channels = int(context_cnn_channels)
        self.context_cnn_kwargs = dict(context_cnn_kwargs or {})
        self.append_current_state = bool(append_current_state)

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

        # NN for instantaneous force u(x)
        pirate_force_kwargs = pirate_force_kwargs or {}
        self.use_fourier_features = bool(use_fourier_features)
        self.fourier_features = int(fourier_features)
        self.fourier_sigma = float(fourier_sigma)
        if self.force_context_mode != "none" and self.use_fourier_features:
            # simplify: disable Fourier features when using context encoders
            self.use_fourier_features = False
        if self.force_context_mode != "none" and self.use_feature_engineering:
            # still allowed, but context operates on engineered features
            pass
        self.force_embed = None
        base_force_dim = self.force_input_dim
        force_in_features = base_force_dim
        context_out_dim = None
        selected_net = force_net_type if force_net_type not in (None, "") else ("pirate" if use_pirate_force else "residual")
        net_type = str(selected_net).lower()
        valid_types = {"residual", "mlp", "pirate"}
        if net_type not in valid_types:
            raise ValueError(f"force_net_type must be one of {valid_types}, got '{force_net_type}'.")
        self.use_pirate_force = net_type == "pirate"
        self.residual_net = net_type == "residual"
        self.context_cnn = None
        if self.force_context_mode != "none" and self.use_pirate_force:
            raise ValueError("force_context_mode requires force_net_type to be 'mlp' or 'residual', not 'pirate'.")
        if self.force_context_mode == "flatten":
            force_in_features = self.force_input_dim * self.context_len
            if self.append_current_state:
                force_in_features += self.force_input_dim
            self.force_embed = None
        elif self.force_context_mode == "cnn":
            self.force_embed = None
            context_in_channels = self.force_input_dim

            cnn_cfg = self.context_cnn_kwargs
            channels_cfg = cnn_cfg.get("channels", self.context_cnn_channels)
            if isinstance(channels_cfg, int):
                channels = [int(channels_cfg)]
            elif isinstance(channels_cfg, (list, tuple)):
                channels = [int(c) for c in channels_cfg]
            else:
                raise ValueError("cnn_kwargs.channels must be int or list of ints.")
            if not channels:
                channels = [self.context_cnn_channels]

            def _expand_param(val, n, default):
                if isinstance(val, (list, tuple)):
                    vals = [int(v) for v in val]
                    if len(vals) < n:
                        vals = vals + [vals[-1]] * (n - len(vals))
                    else:
                        vals = vals[:n]
                    return vals
                if val is None:
                    return [default] * n
                return [int(val)] * n

            kernel_sizes = _expand_param(cnn_cfg.get("kernel_sizes", 3), len(channels), 3)
            default_dilations = [2, 4, 4] if len(channels) >= 3 else [1] * len(channels)
            dilations = _expand_param(cnn_cfg.get("dilations", default_dilations), len(channels), 1)
            paddings_raw = cnn_cfg.get("paddings")
            paddings = (
                _expand_param(paddings_raw, len(channels), 0) if paddings_raw is not None else None
            )
            strides = _expand_param(cnn_cfg.get("strides", 1), len(channels), 1)

            pool_type = str(cnn_cfg.get("pool", "global")).lower()
            pool_kernel = int(cnn_cfg.get("pool_kernel", 2))
            pool_stride = int(cnn_cfg.get("pool_stride", pool_kernel))

            layers: list[nn.Module] = []
            in_ch = context_in_channels
            seq_len = self.context_len
            for out_ch, ksz, dil, stride, idx in zip(
                channels, kernel_sizes, dilations, strides, range(len(channels))
            ):
                if ksz < 1:
                    raise ValueError("kernel_sizes must be >= 1")
                if stride < 1:
                    raise ValueError("strides must be >= 1")
                pad = paddings[idx] if paddings is not None else int(dil * (ksz - 1) / 2)
                layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=ksz, padding=pad, dilation=dil, stride=stride))
                layers.append(nn.GELU())
                # update sequence length estimate
                seq_len = int(math.floor((seq_len + 2 * pad - dil * (ksz - 1) - 1) / stride + 1))
                seq_len = max(seq_len, 1)
                in_ch = out_ch

            if pool_type == "global":
                layers.append(nn.AdaptiveAvgPool1d(1))
                seq_len = 1
            elif pool_type == "avg":
                layers.append(nn.AvgPool1d(kernel_size=pool_kernel, stride=pool_stride))
                seq_len = int(math.floor((seq_len - pool_kernel) / pool_stride + 1))
                seq_len = max(seq_len, 1)
            elif pool_type == "max":
                layers.append(nn.MaxPool1d(kernel_size=pool_kernel, stride=pool_stride))
                seq_len = int(math.floor((seq_len - pool_kernel) / pool_stride + 1))
                seq_len = max(seq_len, 1)
            elif pool_type != "none":
                raise ValueError("pool must be one of {'global', 'avg', 'max', 'none'}.")

            layers.append(nn.Flatten(start_dim=1))
            self.context_cnn = nn.Sequential(*layers)
            force_in_features = in_ch * seq_len
            if self.append_current_state:
                force_in_features += self.force_input_dim
            # keep attribute for compatibility
            self.context_cnn_channels = in_ch
        elif self.use_fourier_features:
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

        pirate_cfg = dict(pirate_force_kwargs) if pirate_force_kwargs is not None else {}
        if self.use_pirate_force:
            pirate_args = {
                "input_size": base_force_dim,
                "output_size": 1,
                "depth": int(pirate_cfg.pop("depth", pirate_cfg.pop("pirate_layers", 2))),
                "fourier_features": int(pirate_cfg.pop("fourier_features", 64)),
                "sigma": float(pirate_cfg.pop("sigma", 1.0)),
                "use_rwf": bool(pirate_cfg.pop("use_rwf", True)),
                "activation": pirate_cfg.pop("activation", "tanh"),
            }
            pirate_args.update(pirate_cfg)
            self.u_net = ODEPirateNet(**pirate_args)
        elif self.residual_net:
            layers = [nn.Linear(force_in_features, self.residual_hidden)]
            for _ in range(self.residual_layers):
                layers.append(Residual(self.residual_hidden, activation=self.residual_activation))
            layers.append(nn.Linear(self.residual_hidden, 1))
            self.u_net = nn.Sequential(*layers)
        else:
            mlp_layers: list[nn.Module] = []
            in_features = force_in_features
            mlp_act_cls = _activation_factory(self.mlp_activation)
            for _ in range(self.mlp_layers):
                mlp_layers.append(nn.Linear(in_features, self.mlp_hidden))
                mlp_layers.append(mlp_act_cls())
                in_features = self.mlp_hidden
            mlp_layers.append(nn.Linear(self.mlp_hidden, 1))
            self.u_net = nn.Sequential(*mlp_layers)

        if self.learn_hamiltonian:
            h_in_features = self.force_input_dim
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
        context_len: int = 20,
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
        learn_hamiltonian = bool(cfg.get("learn_hamiltonian", False))
        use_pirate_force = bool(cfg.get("use_pirate_force", False))
        pirate_force_kwargs = cfg.get("pirate_force_kwargs", {}) or {}
        use_fourier_features = bool(cfg.get("use_fourier_features", False))
        fourier_features = int(cfg.get("fourier_features", 64))
        fourier_sigma = float(cfg.get("fourier_sigma", 1.0))
        use_feature_engineering = bool(cfg.get("use_feature_engineering", False))
        arch_cfg = arch_cfg or {}
        force_net_type = arch_cfg.get("force_net_type")
        force_context_mode = str(arch_cfg.get("force_context_mode", "none"))
        cnn_kwargs_cfg = arch_cfg.get("cnn_kwargs", {}) or {}
        channels_cfg_val = cnn_kwargs_cfg.get("channels", arch_cfg.get("context_cnn_channels", 64))
        if isinstance(channels_cfg_val, (list, tuple)):
            context_cnn_channels = int(channels_cfg_val[0]) if channels_cfg_val else 64
        else:
            context_cnn_channels = int(channels_cfg_val)
        append_current_state = bool(arch_cfg.get("append_current_state", False))
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
            learn_hamiltonian=learn_hamiltonian,
            use_pirate_force=use_pirate_force,
            pirate_force_kwargs=combined_pirate_kwargs,
            use_fourier_features=use_fourier_features,
            fourier_features=fourier_features,
            fourier_sigma=fourier_sigma,
            use_feature_engineering=use_feature_engineering,
            force_net_type=force_net_type,
            residual_kwargs=residual_kwargs,
            mlp_kwargs=mlp_kwargs,
            context_len=context_len,
            force_context_mode=force_context_mode,
            context_cnn_channels=context_cnn_channels,
            context_cnn_kwargs=cnn_kwargs_cfg,
            append_current_state=append_current_state,
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
            c_eff = 2.0 * zeta * torch.sqrt(torch.tensor(self.k * self.m, device=x.device, dtype=x.dtype))
        else:
            c_eff = self.fixed_c.to(device=x.device, dtype=x.dtype)
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

    def _encode_context(self, context_z: torch.Tensor) -> torch.Tensor:
        if context_z is None:
            raise ValueError("context_z is required when using a context encoder.")
        if context_z.dim() < 3:
            raise ValueError(f"context_z must have shape (B, context_len, dim); got {context_z.shape}.")
        base_feats = self._base_features(context_z)
        if self.force_context_mode == "flatten":
            return base_feats.reshape(base_feats.shape[0], -1)
        if self.force_context_mode == "cnn":
            if self.context_cnn is None:
                raise ValueError("context_cnn not initialized.")
            # (B, T, C) -> (B, C, T)
            permuted = base_feats.permute(0, 2, 1)
            return self.context_cnn(permuted)
        raise ValueError(f"Unsupported force_context_mode '{self.force_context_mode}'.")

    def _force_features(self, x: torch.Tensor, context_z: torch.Tensor | None = None) -> torch.Tensor:
        base_features = self._base_features(x)
        if self.force_context_mode == "none":
            return self.force_embed(base_features) if self.force_embed is not None else base_features
        context_features = self._encode_context(context_z)
        if self.append_current_state:
            context_features = torch.cat((context_features, base_features.reshape(base_features.shape[0], -1)), dim=1)
        return context_features

    def u_theta1(self, x, context_z: torch.Tensor | None = None):
        features = self._force_features(x, context_z=context_z)
        return self.u_net(features) * self.k * self.D
    
    def u_theta2(self, x, context_z: torch.Tensor | None = None):
        return self.u_theta1(x, context_z=context_z) + self.drag_force(x)

    def learned_force(self, x, context_z: torch.Tensor | None = None):
        return self.u_theta1(x, context_z=context_z)
    
    def u_theta(self, x, context_z: torch.Tensor | None = None):
        return self.u_theta2(x, context_z=context_z) if self.include_physical_drag else self.u_theta1(
            x, context_z=context_z
        )
    
    def f(self, x, context_z: torch.Tensor | None = None):
        u = self.u_theta(x, context_z=context_z)
        G = self.G.to(x.device).to(x.dtype)                        # (..., 1)
        Gu = torch.einsum('ij,...j->...i', G, u)
        return Gu

    def g(self, x, context_z: torch.Tensor | None = None):
        gH = self.grad_H(x)                         # (..., 2)
        R = self.R(x)                               # (..., 2, 2)

        J = self.J.to(x.device).to(x.dtype)

        JgH = torch.einsum('ij,...j->...i', J, gH) #Just  J @ gH, with batch handling
        RgH = torch.einsum('...ij,...j->...i', R, gH)

        core = JgH - RgH

        return core + self.f(x, context_z=context_z)

    def step_euler(self, x, dt, context_z: torch.Tensor | None = None):
        return x + dt * self.g(x, context_z=context_z)

    def step_rk4(self, x, t, dt, context_z: torch.Tensor | None = None):
        x_next, _ = self.rk4_step(x, t, dt, context_z=context_z)
        return x_next

    def rk4_step(self, x, t, dt, context_z: torch.Tensor | None = None):
        """
        Perform one Runge-Kutta 4 integration step and return both the next state
        and the averaged force over the step.
        """
        k1 = self.g(x, context_z=context_z)
        force1 = self.u_theta(x, context_z=context_z)

        x2 = x + 0.5 * dt * k1
        k2 = self.g(x2, context_z=context_z)
        force2 = self.u_theta(x2, context_z=context_z)

        x3 = x + 0.5 * dt * k2
        k3 = self.g(x3, context_z=context_z)
        force3 = self.u_theta(x3, context_z=context_z)

        x4 = x + dt * k3
        k4 = self.g(x4, context_z=context_z)
        force4 = self.u_theta(x4, context_z=context_z)

        x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        force_avg = (force1 + 2.0 * force2 + 2.0 * force3 + force4) / 6.0
        return x_next, force_avg

    def rollout(self, z0, t_seq, dt):
        """
        z0: (B, state_dim)    starting state from data
        t_seq: (B, K+1)       absolute times t0..tK
        returns:
        Z_pred: (B, K+1, state_dim)  predictions incl. z0
        F_hist: (B, K+1, 1)          optional, learned force per step
        """
        B = z0.shape[0]
        state_dim = z0.shape[-1]
        K = t_seq.shape[1] - 1

        Z_pred = [z0]
        F_hist = []

        z = z0
        for k in range(K):
            t = t_seq[:, k]
            z, Fk = self.rk4_step(z, t, dt)
            Z_pred.append(z)
            F_hist.append(Fk)

        Z_pred = torch.stack(Z_pred, dim=1)            # (B,K+1,D)
        if F_hist:
            initial_force = self.u_theta(z0)
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

    
    def res_loss(self, zi, ti, zin, tin, context_z: torch.Tensor | None = None):
        return self.res_loss_SRK4(zi, ti, zin, tin, context_z=context_z)
    
    def avg_force(self, zi, ti, zin, tin, context_z: torch.Tensor | None = None):
        return self.avg_force_SRK4(zi, ti, zin, tin, context_z=context_z)
    
    def res_loss_Euler(self, zi, ti, zin, tin, context_z: torch.Tensor | None = None):
        dz = (zin-zi)/self.dt
        z_mean = 0.5*(zin+zi)
        res = dz - self.g(z_mean, context_z=context_z)
        scale = torch.tensor((self.q_scale, self.p_scale), device=res.device, dtype=res.dtype)
        res_scaled = res / scale
        loss = torch.mean(torch.sum(res_scaled**2, dim=1))
        return loss

    def avg_force_Euler(self, zi, ti, zin, tin, context_z: torch.Tensor | None = None):
        z_mean = 0.5*(zin+zi)
        forces = self.learned_force(z_mean, context_z=context_z)
        loss = torch.mean(torch.linalg.norm(forces, ord=1, dim=1))
        return loss
    

    def res_loss_SRK4(self, zi, ti, zin, tin, context_z: torch.Tensor | None = None):
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
        g_a_plus  = self.g(z_a_plus, context_z=context_z)                  # (B, d)
        g_a_minus = self.g(z_a_minus, context_z=context_z)                 # (B, d)

        # two corrected midpoints
        z_corr_minus = z_mid - b * dt * g_a_plus      # (B, d)
        z_corr_plus  = z_mid + b * dt * g_a_minus     # (B, d)

        # final two g-evals
        g1 = self.g(z_corr_minus, context_z=context_z)                     # (B, d)
        g2 = self.g(z_corr_plus, context_z=context_z)                      # (B, d)

        dz_model = 0.5 * g1 + 0.5 * g2                # (B, d)

        # residual
        res = dz_fd - dz_model                        # (B, d)

        # scale like before, but for time-derivatives
        res_scale = torch.tensor(
            (self.q_scale, self.p_scale),
            device=res.device, dtype=res.dtype
        )
        res_scaled = res / res_scale

        loss = torch.mean(torch.sum(res_scaled**2, dim=1))
        return loss
    
    def avg_force_SRK4(self, zi, ti, zin, tin, context_z: torch.Tensor | None = None):
        dt = self.dt
        b = math.sqrt(3.0) / 6.0

        # same stage points as in res_loss
        z_a_plus  = (0.5 + b) * zi + (0.5 - b) * zin
        z_a_minus = (0.5 - b) * zi + (0.5 + b) * zin

        # evaluate learned force at both stages
        f1 = self.f(z_a_plus, context_z=context_z)    # assume (B, 1) or (B, 2)
        f2 = self.f(z_a_minus, context_z=context_z)

        loss = 0.5 * torch.mean(torch.sum(torch.abs(f1), dim=1)) \
            + 0.5 * torch.mean(torch.sum(torch.abs(f2), dim=1))
        return loss
    
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
):
    fig, axes = plt.subplots(4, 1, figsize=(6, 12), sharex=False)
    ax_full, ax_diff, ax_zoom, ax_middle = axes

    ax_full.plot(t, y_true_norm, label="y/D (true)")
    ax_full.plot(t, y_pred_norm, label="y/D (pred)")
    ax_full.set_xlabel("time")
    ax_full.set_ylabel("y/D")
    ax_full.grid(True, alpha=0.3)
    ax_full.set_title(f"Normalized rollout at epoch {epoch+1}")
    ax_full.legend(loc="upper right")

    diff_y_norm = y_pred_norm - y_true_norm
    ax_diff.plot(t, diff_y_norm, label="Δ(y/D)", color="tab:orange")
    ax_diff.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax_diff.set_xlabel("time")
    ax_diff.set_ylabel("Δy/D")
    ax_diff.grid(True, alpha=0.3)
    ax_diff.set_title(f"Difference (pred - true) epoch {epoch+1}")
    ax_diff.legend(loc="upper right")

    ax_zoom.plot(t[zoom_mask], y_true_norm[zoom_mask], label="y/D (true)")
    ax_zoom.plot(t[zoom_mask], y_pred_norm[zoom_mask], label="y/D (pred)")
    ax_zoom.set_xlabel("time")
    ax_zoom.set_ylabel("y/D")
    ax_zoom.grid(True, alpha=0.3)
    ax_zoom.set_title(f"Normalized rollout (first 1s) epoch {epoch+1}")
    ax_zoom.legend(loc="upper right")

    mid_start, mid_end = middle_window
    ax_middle.plot(t[middle_mask], y_true_norm[middle_mask], label="y/D (true)")
    ax_middle.plot(t[middle_mask], y_pred_norm[middle_mask], label="y/D (pred)")
    ax_middle.set_xlabel("time")
    ax_middle.set_ylabel("y/D")
    ax_middle.grid(True, alpha=0.3)
    ax_middle.set_title(f"Normalized rollout ({mid_start}-{mid_end}s) epoch {epoch+1}")
    ax_middle.legend(loc="upper right")

    plt.tight_layout()
    writer.add_figure("val/rollout_displacement", fig, epoch + 1)
    plt.close(fig)

def log_force_plots(
    writer,
    epoch,
    t,
    force_total,
    force_drag,
    force_model,
    force_data,
    zoom_mask,
    middle_mask,
    middle_window,
    include_physical_drag: bool,
):
    fig, axes = plt.subplots(4, 1, figsize=(6, 12), sharex=False)
    ax_full, ax_diff, ax_zoom, ax_middle = axes
    total_label = "F_total (model + drag)" if include_physical_drag else "F_total (model)"
    model_label = "F_model (wake)" if include_physical_drag else "F_model"

    ax_full.plot(t, force_total, label=total_label, color="tab:purple")
    if include_physical_drag:
        ax_full.plot(t, force_drag, label="F_drag", color="tab:red", linestyle="--")
    ax_full.plot(t, force_model, label=model_label, color="tab:green", linestyle=":")
    ax_full.plot(t, force_data, label="F_data", color="tab:blue", alpha=0.7)
    ax_full.set_xlabel("time")
    ax_full.set_ylabel("Force")
    ax_full.grid(True, alpha=0.3)
    ax_full.set_title(f"Force rollout at epoch {epoch+1}")
    ax_full.legend(loc="upper right")

    diff_force = force_total - force_data
    ax_diff.plot(t, diff_force, label="ΔF_total", color="tab:orange")
    ax_diff.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax_diff.set_xlabel("time")
    ax_diff.set_ylabel("ΔForce")
    ax_diff.grid(True, alpha=0.3)
    ax_diff.set_title(f"Force difference (model - data) epoch {epoch+1}")
    ax_diff.legend(loc="upper right")

    ax_zoom.plot(t[zoom_mask], force_total[zoom_mask], label=total_label, color="tab:purple")
    if include_physical_drag:
        ax_zoom.plot(t[zoom_mask], force_drag[zoom_mask], label="F_drag", color="tab:red", linestyle="--")
    ax_zoom.plot(t[zoom_mask], force_model[zoom_mask], label=model_label, color="tab:green", linestyle=":")
    ax_zoom.plot(t[zoom_mask], force_data[zoom_mask], label="F_data", color="tab:blue", alpha=0.7)
    ax_zoom.set_xlabel("time")
    ax_zoom.set_ylabel("Force")
    ax_zoom.grid(True, alpha=0.3)
    ax_zoom.set_title(f"Force rollout (first 1s) epoch {epoch+1}")
    ax_zoom.legend(loc="upper right")

    mid_start, mid_end = middle_window
    ax_middle.plot(t[middle_mask], force_total[middle_mask], label=total_label, color="tab:purple")
    if include_physical_drag:
        ax_middle.plot(
            t[middle_mask],
            force_drag[middle_mask],
            label="F_drag",
            color="tab:red",
            linestyle="--",
        )
    ax_middle.plot(
        t[middle_mask],
        force_model[middle_mask],
        label=model_label,
        color="tab:green",
        linestyle=":",
    )
    ax_middle.plot(t[middle_mask], force_data[middle_mask], label="F_data", color="tab:blue", alpha=0.7)
    ax_middle.set_xlabel("time")
    ax_middle.set_ylabel("Force")
    ax_middle.grid(True, alpha=0.3)
    ax_middle.set_title(f"Force rollout ({mid_start}-{mid_end}s) epoch {epoch+1}")
    ax_middle.legend(loc="upper right")

    plt.tight_layout()
    writer.add_figure("val/rollout_force", fig, epoch + 1)
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
):
    fig, axes = plt.subplots(4, 1, figsize=(6, 12), sharex=False)
    ax_full, ax_diff, ax_zoom, ax_middle = axes
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
    ax_full.set_title(f"Hamiltonian rollout at epoch {epoch+1}")
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
    ax_diff.set_title(f"Hamiltonian difference epoch {epoch+1}")
    ax_diff.legend(loc="upper right")

    ax_zoom.plot(t[zoom_mask], h_model_rel[zoom_mask], **model_kwargs.copy())
    if hamiltonian_data is not None:
        ax_zoom.plot(t[zoom_mask], h_data_rel[zoom_mask], **data_kwargs.copy())
    ax_zoom.set_xlabel("time")
    ax_zoom.set_ylabel("Hamiltonian")
    ax_zoom.grid(True, alpha=0.3)
    ax_zoom.set_title(f"Hamiltonian (first 1s) epoch {epoch+1}")
    ax_zoom.legend(loc="upper right")

    mid_start, mid_end = middle_window
    ax_middle.plot(t[middle_mask], h_model_rel[middle_mask], **model_kwargs.copy())
    if hamiltonian_data is not None:
        ax_middle.plot(t[middle_mask], h_data_rel[middle_mask], **data_kwargs.copy())
    ax_middle.set_xlabel("time")
    ax_middle.set_ylabel("Hamiltonian")
    ax_middle.grid(True, alpha=0.3)
    ax_middle.set_title(f"Hamiltonian ({mid_start}-{mid_end}s) epoch {epoch+1}")
    ax_middle.legend(loc="upper right")

    plt.tight_layout()
    writer.add_figure("val/rollout_hamiltonian", fig, epoch + 1)
    plt.close(fig)

def preprocess_timeseries(
    t: np.ndarray,
    y: np.ndarray,
    force: np.ndarray,
    hamiltonian: np.ndarray,
    data_cfg: DataConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Apply optional steady-state trimming and uniform decimation to time-series arrays.
    """
    if t.size == 0:
        return t, y, force, hamiltonian, float("nan")
    mask = np.ones_like(t, dtype=bool)
    if data_cfg.steadystate:
        mask &= t > float(data_cfg.steadystate_time_threshold)
    t_proc = t[mask]
    y_proc = y[mask]
    f_proc = force[mask]
    h_proc = hamiltonian[mask]
    step = max(1, int(data_cfg.reduction_factor if data_cfg.reduce_time else 1))
    if step > 1:
        t_proc = t_proc[::step]
        y_proc = y_proc[::step]
        f_proc = f_proc[::step]
        h_proc = h_proc[::step]
    dt_value = float(t_proc[1] - t_proc[0]) if t_proc.size > 1 else float("nan")
    return t_proc, y_proc, f_proc, h_proc, dt_value


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
    series_data: list[tuple[np.ndarray, np.ndarray, float]],
    m_eff: float,
    batch_size: int,
    device: torch.device,
    smoothing_cfg: SmoothingConfig | None = None,
    context_len: int = 20,
    shuffle: bool = True,
) -> tuple[DataLoader, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], int]:
    if not series_data:
        raise ValueError("series_data must contain at least one (y, t, dt) tuple.")
    sequence_tensors: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    datasets: list[TensorDataset | ConcatDataset] = []
    min_length: int | None = None
    for y_np, t_np, dt_value in series_data:
        y_tensor, vel_tensor, t_tensor = prepare_sequence_tensors(
            y_np,
            t_np,
            dt_value,
            m_eff,
            device,
            smoothing_cfg=smoothing_cfg,
        )
        sequence_tensors.append((y_tensor, vel_tensor, t_tensor))
        dataset = build_dataset(y_tensor, vel_tensor, m_eff, t_tensor, context_len=context_len)
        datasets.append(dataset)
        dataset_len = len(dataset)
        min_length = dataset_len if min_length is None else min(min_length, dataset_len)
    dataset = combine_datasets(datasets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader, sequence_tensors, min_length if min_length is not None else 0


def prepare_sequence_tensors(
    y_np: np.ndarray,
    t_np: np.ndarray,
    dt: float,
    m_eff: float,
    device: torch.device,
    smoothing_cfg: SmoothingConfig | None = None,
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
    vel_np = compute_velocity_numpy(
        y_arr,
        dt_value,
        use_savgol=smoothing_cfg.use_savgol_smoothing,
        savgol_window=smoothing_cfg.window_length,
        savgol_polyorder=smoothing_cfg.polyorder,
    )
    y_tensor = torch.from_numpy(y_arr).float().to(device)
    vel_tensor = torch.from_numpy(vel_np).float().to(device)
    t_tensor = torch.from_numpy(t_arr).float().to(device)
    return y_tensor, vel_tensor, t_tensor


def load_training_series(
    y_eval: np.ndarray,
    t_eval: np.ndarray,
    dt_eval: float,
    use_generated: bool,
    series_dir: Path,
    m_eff: float,
    device: torch.device,
    smoothing_cfg: SmoothingConfig | None = None,
) -> tuple[list[tuple[np.ndarray, np.ndarray, float]], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    train_series_raw: list[tuple[np.ndarray, np.ndarray, float]] = []
    if use_generated:
        if not series_dir.exists():
            raise FileNotFoundError(f"Training series directory '{series_dir}' does not exist.")
        series_files = sorted(series_dir.glob("*.npz"))
        if not series_files:
            raise FileNotFoundError(f"No '.npz' files found in training series directory '{series_dir}'.")
        for series_file in series_files:
            series_data = np.load(series_file)
            series_t = np.asarray(series_data["a"])
            series_y = np.asarray(series_data["b"])
            if series_t.ndim != 1 or series_y.ndim != 1:
                raise ValueError(f"Series '{series_file}' must contain 1D 'a' and 'b' arrays.")
            if series_t.shape[0] != series_y.shape[0]:
                raise ValueError(f"Series '{series_file}' has mismatched lengths.")
            if series_t.shape[0] < 2:
                raise ValueError(f"Series '{series_file}' is too short to build training samples.")
            series_dt = float(series_t[1] - series_t[0])
            if not np.allclose(np.diff(series_t), series_dt, rtol=1e-6, atol=1e-9):
                raise ValueError(f"Series '{series_file}' time vector is not uniform.")
            if not np.isclose(series_dt, dt_eval, rtol=1e-6, atol=1e-9):
                series_y, series_t = resample_uniform_series(series_t, series_y, dt_eval)
                series_dt = dt_eval
            train_series_raw.append((series_y, series_t, series_dt))
    else:
        train_series_raw.append((y_eval, t_eval, dt_eval))

    eval_tensors = prepare_sequence_tensors(
        y_eval,
        t_eval,
        dt_eval,
        m_eff,
        device,
        smoothing_cfg=smoothing_cfg,
    )
    return train_series_raw, eval_tensors


def build_dataset(
    y_data_t: torch.Tensor,
    vel: torch.Tensor,
    m_eff: float,
    t_tensor: torch.Tensor,
    context_len: int = 20,
) -> TensorDataset:
    """Construct sliding windows of past states/times and the next-state target."""
    if context_len < 1:
        raise ValueError("context_len must be at least 1")
    z = torch.stack((y_data_t, vel * m_eff), dim=1)
    total_steps = z.shape[0]
    if total_steps <= context_len:
        raise ValueError(f"Not enough timesteps ({total_steps}) for context_len={context_len}.")

    contexts_z: list[torch.Tensor] = []
    contexts_t: list[torch.Tensor] = []
    targets_z: list[torch.Tensor] = []
    targets_t: list[torch.Tensor] = []
    for start in range(total_steps - context_len):
        end = start + context_len
        contexts_z.append(z[start:end])                  # (context_len, 2)
        contexts_t.append(t_tensor[start:end])           # (context_len,)
        targets_z.append(z[end])                         # (2,)
        targets_t.append(t_tensor[end])                  # ()

    context_batch = torch.stack(contexts_z, dim=0)               # (N, context_len, 2)
    context_t_batch = torch.stack(contexts_t, dim=0)             # (N, context_len)
    target_z_batch = torch.stack(targets_z, dim=0)               # (N, 2)
    target_t_batch = torch.stack(targets_t, dim=0).unsqueeze(1)  # (N, 1)

    return TensorDataset(context_batch, context_t_batch, target_z_batch, target_t_batch)

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
    m_eff: float,
    dt: float,
    t: np.ndarray,
    D: float,
    k: float,
    device: torch.device,
    start_idx: int = 0,
) -> dict[str, np.ndarray]:
    """Roll the model forward over the time grid starting at start_idx."""
    total_steps = len(t) - start_idx
    if total_steps <= 0:
        raise ValueError("start_idx is beyond available time steps.")

    context_len = max(1, int(getattr(model, "context_len", 1)))
    use_context = getattr(model, "force_context_mode", "none") != "none" and context_len > 1

    z_all = torch.stack((y0, vel * m_eff), dim=1)  # (T, 2)
    if z_all.shape[0] <= start_idx:
        raise ValueError("Not enough samples to run rollout from start_idx.")

    if use_context:
        start_for_ctx = max(0, start_idx - context_len + 1)
        context_window = z_all[start_for_ctx : start_idx + 1]
        if context_window.shape[0] < context_len:
            pad = context_window[0].unsqueeze(0).repeat(context_len - context_window.shape[0], 1)
            context_window = torch.cat((pad, context_window), dim=0)
        context_window = context_window.to(device)
        state = context_window[-1:].clone()  # (1, 2)
    else:
        p0 = vel[start_idx] * m_eff
        state = torch.stack((y0[start_idx], p0), dim=0).unsqueeze(0).to(device)

    y_samples: list[float] = []
    p_samples: list[float] = []
    force_total: list[float] = []
    force_drag: list[float] = []
    force_model: list[float] = []
    hamiltonian_model_vals: list[float] = []

    with torch.no_grad():
        for step_idx in range(total_steps):
            y_samples.append(float(state[0, 0].detach().cpu()))
            p_samples.append(float(state[0, 1].detach().cpu()))
            if use_context:
                ctx = context_window.unsqueeze(0)  # (1, context_len, 2)
                model_force = float(model.learned_force(state, context_z=ctx).squeeze().detach().cpu())
                total_force_val = float(model.u_theta(state, context_z=ctx).squeeze().detach().cpu())
            else:
                model_force = float(model.learned_force(state).squeeze().detach().cpu())
                total_force_val = float(model.u_theta(state).squeeze().detach().cpu())
            if model.include_physical_drag:
                drag_force = float(model.drag_force(state).squeeze().detach().cpu())
            else:
                drag_force = 0.0
            H_val = float(model.H(state).detach().cpu())
            force_model.append(model_force)
            force_drag.append(drag_force)
            force_total.append(total_force_val)
            hamiltonian_model_vals.append(H_val)

            # propagate to next state unless at the end
            if step_idx == total_steps - 1:
                break
            if use_context:
                ctx = context_window.unsqueeze(0)
                state, _ = model.rk4_step(state, t, dt, context_z=ctx)
                # update context window with predicted next state
                context_window = torch.cat((context_window[1:], state.detach()), dim=0)
            else:
                state = model.step_rk4(state, t, dt)

    y_samples = np.asarray(y_samples)
    p_samples = np.asarray(p_samples)
    y_pred_norm = y_samples / D
    p_pred_norm = (p_samples / m_eff) / (np.sqrt(k / m_eff) * D)
    force_total_arr = np.asarray(force_total)
    force_drag_arr = np.asarray(force_drag)
    force_model_arr = np.asarray(force_model)
    hamiltonian_model_arr = np.asarray(hamiltonian_model_vals)
    return {
        "y_norm": y_pred_norm,
        "p_norm": p_pred_norm,
        "force_total": force_total_arr,
        "force_drag": force_drag_arr,
        "force_model": force_model_arr,
        "hamiltonian_model": hamiltonian_model_arr,
    }
