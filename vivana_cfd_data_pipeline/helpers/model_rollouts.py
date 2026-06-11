from __future__ import annotations

import csv
import importlib
import math
import pickle
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any, Callable, Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import scipy.signal as signal
import torch
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
CFD_DATA_DIR = Path(__file__).resolve().parents[1]
if str(CFD_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(CFD_DATA_DIR))

from vivana_cfd_data_pipeline.scripts.training_npz_loader import (
    iter_all_npz_files as shared_iter_all_npz_files,
    iter_npz_files as shared_iter_npz_files,
    load_series as shared_load_series,
)

from training.training_utils import (
    AGGREGATE_VALIDATION_ERROR_KEY,
    DOMINANT_FREQ_REL_ERROR_KEY,
    DISP_STD_REL_ERROR_KEY,
    FORCE_DOMINANT_FREQ_REL_ERROR_KEY,
    FORCE_STD_REL_ERROR_KEY,
    PHVIV,
    dominant_frequency,
    normalize_method_name,
    parse_config,
    relative_error,
    resolve_phnn_input_scaling_mode,
    resolve_td_correction_params,
    resolve_td_correction_mode,
    resolve_td_force_input_source,
    resolve_td_phase_input_source,
    resolve_td_memory_config,
    structural_step_constant_force_torch,
    td_correction_mode_flags,
    td_baseline_step_torch,
    _recompute_td_baseline_on_grid,
)
from training.methods.correction.trainer import _td_step_with_corrections

try:
    from training.methods.standalone.trainer import LatentRNNForceModel
except ImportError:
    LatentRNNForceModel = None

try:
    from training.training_utils import build_td_parameter_model
except ImportError:
    build_td_parameter_model = None

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman", "Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
    }
)


DEFAULT_ORDERED_SPLIT_DIRS = ("train", "val")
DEFAULT_NEWMARK_BETA = 0.25
DEFAULT_NEWMARK_GAMMA = 0.5
DEFAULT_CFD_FORCE_TOTAL_SCALE = 4.0
DEFAULT_DATASET_ROOT_RELATIVE_PATHS = (
    Path("vivana_cfd_data_pipeline") / "generated" / "td_burnin_trimmed",
    Path("vivana_cfd_data_pipeline") / "generated" / "td_burnin_trimmed_v1",
    Path("vivana_cfd_data_pipeline") / "generated" / "td_burnin_trimmed_all",
    Path("vivana_cfd_data_pipeline") / "generated" / "td_burnin_trimmed_alltimeseries",
    Path("npz_exports_td_burnin_trimmed_v1"),
    Path("npz_exports_td_burnin_trimmed"),
    Path("npz_exports_td_burnin_trimmed_all"),
)
DEFAULT_CFD_METADATA_PATH = REPO_ROOT / "vivana_cfd_data_pipeline" / "metadata" / "CFD_metadata.csv"
_METADATA_CACHE: dict[str, dict[str, str]] | None = None

VALIDATION_ERROR_KEYS = [
    ("Force mapping NRMSE", "Force mapping NRMSE"),
    (DISP_STD_REL_ERROR_KEY, DISP_STD_REL_ERROR_KEY),
    (DOMINANT_FREQ_REL_ERROR_KEY, DOMINANT_FREQ_REL_ERROR_KEY),
    (FORCE_DOMINANT_FREQ_REL_ERROR_KEY, FORCE_DOMINANT_FREQ_REL_ERROR_KEY),
    (FORCE_STD_REL_ERROR_KEY, FORCE_STD_REL_ERROR_KEY),
]

RK4_REFERENCE_METRICS = {
    DISP_STD_REL_ERROR_KEY,
    DOMINANT_FREQ_REL_ERROR_KEY,
    FORCE_DOMINANT_FREQ_REL_ERROR_KEY,
    FORCE_STD_REL_ERROR_KEY,
}

DISPLACEMENT_MAE_KEY = "MAE"
VALIDATION_COMPONENT_METRIC_KEYS = [
    DOMINANT_FREQ_REL_ERROR_KEY,
    DISP_STD_REL_ERROR_KEY,
    FORCE_DOMINANT_FREQ_REL_ERROR_KEY,
    FORCE_STD_REL_ERROR_KEY,
]
VALIDATION_TRACKED_METRIC_KEYS = [metric_label for _, metric_label in VALIDATION_ERROR_KEYS] + [DISPLACEMENT_MAE_KEY]
VALIDATION_SUMMARY_METRIC_KEYS = VALIDATION_COMPONENT_METRIC_KEYS + [AGGREGATE_VALIDATION_ERROR_KEY]
VALIDATION_PLOT_YLABELS = {
    "Force mapping NRMSE": r"$\mathrm{NRMSE}_F$",
    DISP_STD_REL_ERROR_KEY: r"$\varepsilon_{\sigma}^{y}$ [%]",
    DOMINANT_FREQ_REL_ERROR_KEY: r"$\varepsilon_{\omega}^{y}$ [%]",
    FORCE_STD_REL_ERROR_KEY: r"$\varepsilon_{\sigma}^{F}$ [%]",
    FORCE_DOMINANT_FREQ_REL_ERROR_KEY: r"$\varepsilon_{\omega}^{F}$ [%]",
}


def _parse_checkpoint_config(raw_config: dict[str, Any] | Any) -> Any:
    try:
        return parse_config(raw_config)
    except TypeError as exc:
        message = str(exc)
        if "ModelConfig.__init__()" not in message or "input_scaling_mode" not in message:
            raise
        # Notebook kernels can retain an old training_utils module even after the file changed.
        import training.training_utils as hnn_helper_module

        hnn_helper_module = importlib.reload(hnn_helper_module)
        return hnn_helper_module.parse_config(raw_config)


def dataset_root_candidates(
    cwd: str | Path,
    *,
    repo_root: str | Path | None = None,
    extra_candidates: Sequence[str | Path] | None = None,
) -> list[Path]:
    cwd_path = Path(cwd).resolve()
    base_candidates = [cwd_path, cwd_path.parent]
    if repo_root is not None:
        repo_root_path = Path(repo_root).resolve()
        base_candidates.extend([repo_root_path, repo_root_path.parent])
    candidates: list[Path] = []
    seen: set[Path] = set()
    for raw_path in extra_candidates or ():
        candidate = Path(raw_path).expanduser().resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        candidates.append(candidate)
    for base in base_candidates:
        for rel in DEFAULT_DATASET_ROOT_RELATIVE_PATHS:
            candidate = (base / rel).resolve()
            if candidate in seen:
                continue
            seen.add(candidate)
            candidates.append(candidate)
    return candidates


def resolve_dataset_root(
    cwd: str | Path,
    *,
    repo_root: str | Path | None = None,
    extra_candidates: Sequence[str | Path] | None = None,
) -> Path:
    for candidate in dataset_root_candidates(cwd, repo_root=repo_root, extra_candidates=extra_candidates):
        if not candidate.exists():
            continue
        if any((candidate / split).glob("*.npz") for split in DEFAULT_ORDERED_SPLIT_DIRS if (candidate / split).exists()):
            return candidate
        if any(candidate.glob("*.npz")):
            return candidate
    tried = "\n".join(str(path) for path in dataset_root_candidates(cwd, repo_root=repo_root, extra_candidates=extra_candidates))
    raise FileNotFoundError(f"Could not locate dataset root. Tried:\n{tried}")


def _metadata_rows() -> dict[str, dict[str, str]]:
    global _METADATA_CACHE
    if _METADATA_CACHE is not None:
        return _METADATA_CACHE
    rows: dict[str, dict[str, str]] = {}
    if DEFAULT_CFD_METADATA_PATH.exists():
        with DEFAULT_CFD_METADATA_PATH.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                case_name = str(row.get("case_name", "")).strip()
                if case_name:
                    rows[case_name] = row
    _METADATA_CACHE = rows
    return rows


def _metadata_float(row: dict[str, str] | None, key: str) -> float | None:
    if not row:
        return None
    raw = str(row.get(key, "")).strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if np.isfinite(value) else None


def _progress(iterable: Any, *, total: int | None = None, desc: str | None = None, leave: bool = False) -> Any:
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=leave)


def _force_mapping_nrmse(force_pred: np.ndarray, force_true: np.ndarray) -> float:
    force_pred_arr = np.asarray(force_pred, dtype=float).reshape(-1)
    force_true_arr = np.asarray(force_true, dtype=float).reshape(-1)
    n = min(force_pred_arr.size, force_true_arr.size)
    if n < 1:
        return float("nan")
    force_pred_arr = force_pred_arr[:n]
    force_true_arr = force_true_arr[:n]
    denom = float(np.std(force_true_arr))
    if not np.isfinite(denom) or denom <= 0.0:
        denom = 1.0
    return float(np.sqrt(np.mean((force_pred_arr - force_true_arr) ** 2)) / denom)


@dataclass
class LoadedTrainingModel:
    label: str
    path: Path
    method_name: str
    kind: str
    checkpoint: dict[str, Any]
    config: Any
    method_cfg: dict[str, Any]
    model: torch.nn.Module
    base_td_params: dict[str, float]
    td_memory_cfg: dict[str, Any] | None
    correction_mode: str
    mean_active: bool
    predict_sigma: bool
    fhat_active: bool
    use_td_force_input: bool
    td_force_input_source: str
    use_td_fhat_input: bool
    use_acceleration_input: bool
    use_phi_input: bool
    phi_input_source: str
    use_sigma_inputs: bool
    fhat_bound_multiplier: float
    sigma_min: float
    force_zero_output: bool


def _load_state(model: torch.nn.Module, state: dict[str, Any]) -> None:
    clean_state = dict(state)
    if any(key.startswith("_orig_mod.") for key in clean_state):
        clean_state = {key.removeprefix("_orig_mod."): value for key, value in clean_state.items()}
    if any(key.startswith("module.") for key in clean_state):
        clean_state = {key.removeprefix("module."): value for key, value in clean_state.items()}
    model.load_state_dict(clean_state, strict=False)


def _load_state_strict(model: torch.nn.Module, state: dict[str, Any]) -> None:
    clean_state = dict(state)
    if any(key.startswith("_orig_mod.") for key in clean_state):
        clean_state = {key.removeprefix("_orig_mod."): value for key, value in clean_state.items()}
    if any(key.startswith("module.") for key in clean_state):
        clean_state = {key.removeprefix("module."): value for key, value in clean_state.items()}
    model.load_state_dict(clean_state, strict=True)


def _coerce_model_spec(spec: str | Path | dict[str, Any], *, repo_root: Path) -> tuple[Path, str]:
    if isinstance(spec, (str, Path)):
        path = Path(spec)
        label = path.stem
    elif isinstance(spec, dict):
        raw_path = spec.get("path")
        if raw_path is None:
            raise ValueError("Each model spec dict must contain a 'path'.")
        path = Path(raw_path)
        label = str(spec.get("label", path.stem))
    else:
        raise TypeError(f"Unsupported model spec type: {type(spec)!r}")
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    return path, label


def _default_model_specs(repo_root: Path) -> list[dict[str, Any]]:
    model_dir = repo_root / "models"
    pt_files = sorted(model_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {model_dir}")
    chosen = pt_files[0]
    return [{"path": chosen, "label": chosen.stem}]


def _module_device(module: torch.nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        for buffer in module.buffers():
            return buffer.device
    return torch.device("cpu")


def _module_dtype(module: torch.nn.Module, fallback: torch.dtype = torch.float32) -> torch.dtype:
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        for buffer in module.buffers():
            return buffer.dtype
    return fallback


def _load_checkpoint_file(path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception:
        with path.open("rb") as handle:
            checkpoint = pickle.load(handle)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint at {path} did not contain a dict.")
    return checkpoint


class Polynomial3DCorrectionModel(torch.nn.Module):
    def __init__(self, checkpoint: dict[str, Any], *, device: torch.device) -> None:
        super().__init__()
        self.force_output = str(checkpoint.get("force_output", "coefficient"))
        self.use_td_force_input = False
        self.predict_sigma_default = bool(checkpoint.get("predict_sigma", True))
        self.register_buffer(
            "diameter_buffer",
            torch.tensor(float(checkpoint.get("diameter_m", 1.0)), dtype=torch.float32, device=device),
        )
        mean_model = dict(checkpoint["mean_model"])
        sigma_model = dict(checkpoint.get("sigma_model", {}))
        self._register_poly_model("mean", mean_model, device=device)
        if sigma_model:
            self._register_poly_model("sigma", sigma_model, device=device)
            self.has_sigma_model = True
        else:
            self.has_sigma_model = False

    @property
    def diameter(self) -> float:
        return float(self.diameter_buffer.detach().cpu().item())

    def _register_poly_model(self, prefix: str, model: dict[str, Any], *, device: torch.device) -> None:
        coeffs = torch.as_tensor(model["coeffs"], dtype=torch.float32, device=device).reshape(-1)
        powers = torch.as_tensor(model["powers"], dtype=torch.int64, device=device)
        self.register_buffer(f"{prefix}_coeffs", coeffs)
        self.register_buffer(f"{prefix}_powers", powers)
        self.register_buffer(
            f"{prefix}_center_scale",
            torch.tensor(
                [
                    float(model["q_center"]),
                    float(model["p_center"]),
                    float(model["ur_center"]),
                    float(model["q_scale"]),
                    float(model["p_scale"]),
                    float(model["ur_scale"]),
                ],
                dtype=torch.float32,
                device=device,
            ),
        )
        setattr(self, f"{prefix}_order", int(model["order"]))

    def _eval_poly(self, prefix: str, *, q_norm: torch.Tensor, p_norm: torch.Tensor, ur_value: torch.Tensor) -> torch.Tensor:
        coeffs = getattr(self, f"{prefix}_coeffs")
        powers = getattr(self, f"{prefix}_powers")
        center_scale = getattr(self, f"{prefix}_center_scale")
        q_scaled = (q_norm - center_scale[0]) / torch.clamp(center_scale[3], min=1e-12)
        p_scaled = (p_norm - center_scale[1]) / torch.clamp(center_scale[4], min=1e-12)
        ur_scaled = (ur_value - center_scale[2]) / torch.clamp(center_scale[5], min=1e-12)
        basis = (
            torch.pow(q_scaled, powers[:, 0].view(1, -1))
            * torch.pow(p_scaled, powers[:, 1].view(1, -1))
            * torch.pow(ur_scaled, powers[:, 2].view(1, -1))
        )
        return basis @ coeffs.view(-1, 1)

    def predict(
        self,
        *,
        z: torch.Tensor,
        reduced_velocity: torch.Tensor,
        structural_mass: torch.Tensor | float,
        stiffness: torch.Tensor | float,
        predict_sigma: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mass_t = torch.as_tensor(structural_mass, dtype=z.dtype, device=z.device)
        stiffness_t = torch.as_tensor(stiffness, dtype=z.dtype, device=z.device)
        if mass_t.ndim == 0:
            mass_t = mass_t.view(1, 1).expand(z.shape[0], 1)
        if stiffness_t.ndim == 0:
            stiffness_t = stiffness_t.view(1, 1).expand(z.shape[0], 1)
        velocity = z[:, 1:2] / torch.clamp(mass_t, min=1e-12)
        omega = torch.sqrt(torch.clamp(stiffness_t / torch.clamp(mass_t, min=1e-12), min=1e-12))
        diameter_t = self.diameter_buffer.to(device=z.device, dtype=z.dtype)
        q_norm = z[:, 0:1] / torch.clamp(diameter_t, min=1e-12)
        p_norm = velocity / torch.clamp(omega * diameter_t, min=1e-12)
        ur_t = torch.as_tensor(reduced_velocity, dtype=z.dtype, device=z.device).reshape(-1, 1)
        corr_mu = self._eval_poly("mean", q_norm=q_norm, p_norm=p_norm, ur_value=ur_t)
        if predict_sigma and self.has_sigma_model:
            sigma = self._eval_poly("sigma", q_norm=q_norm, p_norm=p_norm, ur_value=ur_t)
            sigma = torch.clamp(sigma, min=0.0)
        else:
            sigma = torch.zeros_like(corr_mu)
        return corr_mu, sigma


def load_trained_model_sources(
    model_specs: Sequence[str | Path | dict[str, Any]] | None,
    *,
    repo_root: str | Path,
    device: str | torch.device = "cpu",
) -> list[LoadedTrainingModel]:
    repo_root_path = Path(repo_root).resolve()
    device_obj = torch.device(device)
    effective_specs = list(model_specs or _default_model_specs(repo_root_path))
    sources: list[LoadedTrainingModel] = []
    for spec in effective_specs:
        path, label = _coerce_model_spec(spec, repo_root=repo_root_path)
        if not path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {path}")
        checkpoint = _load_checkpoint_file(path)
        config = _parse_checkpoint_config(checkpoint.get("config", {}))
        method_name = normalize_method_name(checkpoint.get("method", getattr(config, "method", "correction")))
        if method_name == "standalone":
            method_cfg = dict(config.standalone or {})
        else:
            method_cfg = dict(config.correction or {})
        td_correction_cfg = dict(config.correction or {}) if method_name == "standalone" else method_cfg
        correction_mode = resolve_td_correction_mode(td_correction_cfg)
        mode_flags = td_correction_mode_flags(correction_mode)
        mean_active = bool(mode_flags["mean_active"])
        predict_sigma = False
        fhat_active = bool(mode_flags["fhat_active"])
        td_force_input_source = resolve_td_force_input_source(
            checkpoint.get("td_force_input_source", td_correction_cfg.get("use_td_force_input", checkpoint.get("use_td_force_input", False)))
        )
        use_td_force_input = td_force_input_source != "none"
        use_td_fhat_input = bool(td_correction_cfg.get("use_td_fhat_input", checkpoint.get("use_td_fhat_input", False)))
        use_acceleration_input = bool(td_correction_cfg.get("use_acceleration_input", checkpoint.get("use_acceleration_input", False)))
        phi_input_source = resolve_td_phase_input_source(
            td_correction_cfg.get(
                "phi_input_source",
                checkpoint.get("phi_input_source", td_correction_cfg.get("use_phi_input", checkpoint.get("use_phi_input", False))),
            )
        )
        use_phi_input = phi_input_source != "none"
        use_sigma_inputs = False
        fhat_bound_multiplier = float(td_correction_cfg.get("fhat_bound_multiplier", checkpoint.get("fhat_bound_multiplier", 1.5)))
        force_zero_output = bool(td_correction_cfg.get("force_zero_output", checkpoint.get("force_zero_output", False)))
        sigma_min = 0.0
        td_memory_cfg = resolve_td_memory_config(td_correction_cfg)
        base_td_params = dict(checkpoint.get("base_td_params", resolve_td_correction_params(td_correction_cfg)))
        if fhat_active and use_td_force_input:
            raise ValueError(
                f"{path.name}: use_td_force_input=true is invalid for correction_mode values that include fhat correction."
            )
        if bool(checkpoint.get("poly3d_correction_model", False)):
            model = Polynomial3DCorrectionModel(checkpoint, device=device_obj)
            model.eval()
            method_name = "poly3d"
            kind = "poly3d_correction"
            predict_sigma = False
            correction_mode = "mean_only"
            mean_active = True
            fhat_active = False
            use_td_force_input = False
            td_force_input_source = "none"
            use_acceleration_input = False
            use_phi_input = False
            phi_input_source = "none"
            use_sigma_inputs = False
            fhat_bound_multiplier = 1.5
            sigma_min = 0.0
            force_zero_output = False
            td_memory_cfg = None
        elif checkpoint.get("td_parameter_model") is not None or "resolved_td_params" in checkpoint:
            if build_td_parameter_model is None:
                raise ImportError(
                    "This checkpoint requires TD-parameter-model support, but "
                    "`build_td_parameter_model` is not available in the current training_utils.py. "
                    "Use the TD-delta branch or load an older PHNN correction checkpoint instead."
                )
            model = build_td_parameter_model(method_cfg, device=device_obj)
            setattr(model, "rho", float(getattr(config.model, "rho", 1.0)))
            setattr(model, "diameter", float(getattr(config.model, "D", 1.0)))
            _load_state(model, checkpoint["model_state"])
            model.eval()
            kind = "td_parameter_model"
            correction_mode = "mean_only"
            mean_active = True
            predict_sigma = False
            fhat_active = False
            use_td_force_input = False
            td_force_input_source = "none"
            use_acceleration_input = False
            use_phi_input = False
            phi_input_source = "none"
            use_sigma_inputs = False
            fhat_bound_multiplier = 1.5
            sigma_min = 0.0
            force_zero_output = False
            td_memory_cfg = None
        elif method_name == "standalone":
            if LatentRNNForceModel is None:
                raise ImportError(
                    f"{path.name}: latent_rnn checkpoint support is unavailable because "
                    "`training.methods.standalone.trainer.LatentRNNForceModel` could not be imported."
                )
            lrnn_cfg = dict(config.standalone or {})
            latent_dim = int(lrnn_cfg.get("latent_dim", 3))
            encoder_length = int(lrnn_cfg.get("encoder_length", 50))
            if latent_dim < 1:
                raise ValueError(f"{path.name}: latent_rnn.latent_dim must be >= 1.")
            if encoder_length < 1:
                raise ValueError(f"{path.name}: latent_rnn.encoder_length must be >= 1.")
            encoder_type = str(lrnn_cfg.get("encoder_type", "gru")).strip().lower()
            if encoder_type != "gru":
                raise ValueError(f"{path.name}: latent_rnn encoder_type={encoder_type!r} is not supported here.")
            latent_update = str(lrnn_cfg.get("latent_update", "dt_scaled")).strip().lower()
            if latent_update != "dt_scaled":
                raise ValueError(f"{path.name}: latent_rnn latent_update={latent_update!r} is not supported here.")
            raw_latent_time_scale = checkpoint.get("latent_time_scale", lrnn_cfg.get("latent_time_scale", "auto"))
            if isinstance(raw_latent_time_scale, str) and raw_latent_time_scale.strip().lower() == "auto":
                raise ValueError(
                    f"{path.name}: latent_time_scale='auto' was not saved in the checkpoint. "
                    "Re-save the latent_rnn checkpoint with the resolved latent_time_scale."
                )
            latent_time_scale = float(raw_latent_time_scale)
            if not np.isfinite(latent_time_scale) or latent_time_scale <= 0.0:
                raise ValueError(f"{path.name}: latent_time_scale must be positive and finite.")
            model_cfg_obj = config.model
            arch_cfg = config.architecture
            input_scaling_mode = resolve_phnn_input_scaling_mode(getattr(model_cfg_obj, "input_scaling_mode", "current"))
            ur_scale = 10.0 if getattr(model_cfg_obj, "ur_scale", None) is None else float(model_cfg_obj.ur_scale)
            include_acceleration = bool(lrnn_cfg.get("encoder_include_acceleration", True))
            model = LatentRNNForceModel(
                latent_dim=latent_dim,
                encoder_input_dim=3 + (1 if include_acceleration else 0),
                encoder_hidden=int(lrnn_cfg.get("encoder_hidden", 128)),
                encoder_layers=int(lrnn_cfg.get("encoder_layers", 1)),
                encoder_dropout=float(lrnn_cfg.get("encoder_dropout", 0.0)),
                backbone_input_dim=3 + latent_dim,
                architecture_cfg=arch_cfg,
                rho=float(getattr(model_cfg_obj, "rho", 1.0)),
                diameter=float(getattr(model_cfg_obj, "D", 1.0)),
                coefficient_output_bound=(
                    None
                    if getattr(model_cfg_obj, "coefficient_output_bound", None) is None
                    else float(getattr(model_cfg_obj, "coefficient_output_bound"))
                ),
                input_scaling_mode=input_scaling_mode,
                ur_scale=ur_scale,
                latent_time_scale=latent_time_scale,
                corr_init_mode=str(lrnn_cfg.get("corr_init_mode", getattr(model_cfg_obj, "corr_init_mode", "zero"))),
            ).to(device_obj)
            _load_state_strict(model, checkpoint["model_state"])
            model.eval()
            kind = "latent_rnn"
            correction_mode = "latent_rnn"
            mean_active = True
            predict_sigma = False
            fhat_active = False
            use_td_force_input = False
            td_force_input_source = "none"
            use_td_fhat_input = False
            use_acceleration_input = include_acceleration
            use_phi_input = False
            phi_input_source = "none"
            use_sigma_inputs = False
            fhat_bound_multiplier = 1.5
            sigma_min = 0.0
            force_zero_output = False
            td_memory_cfg = None
        elif method_name == "correction":
            dt = float(checkpoint.get("dt", 1.0))
            model_dict = asdict(config.model)
            raw_model_cfg = {}
            raw_config = checkpoint.get("config", {})
            if isinstance(raw_config, dict) and isinstance(raw_config.get("model"), dict):
                raw_model_cfg = raw_config["model"]
            for key in ("structural_mass", "Ca", "k", "damping_c"):
                if key in raw_model_cfg:
                    model_dict[key] = raw_model_cfg[key]
            model_dict["use_td_force_input"] = use_td_force_input
            model_dict["use_td_fhat_input"] = use_td_fhat_input
            model_dict["use_acceleration_input"] = use_acceleration_input
            model_dict["use_phi_input"] = use_phi_input
            model_dict["phi_input_source"] = None if not use_phi_input else phi_input_source
            model_dict["use_sigma_inputs"] = use_sigma_inputs
            model_dict["use_stochastic_process_noise"] = predict_sigma
            model_dict["correction_mode"] = correction_mode
            arch_dict = asdict(config.architecture)
            model, _derived = PHVIV.from_config(dt=dt, cfg=model_dict, arch_cfg=arch_dict, device=device_obj)
            setattr(model, "correction_mode", correction_mode)
            setattr(model, "td_force_input_source", td_force_input_source)
            setattr(model, "fhat_bound_multiplier", float(fhat_bound_multiplier))
            setattr(model, "force_zero_output", force_zero_output)
            _load_state(model, checkpoint["model_state"])
            model.eval()
            kind = "phnn_correction"
        else:
            raise ValueError(
                f"{path.name}: unsupported checkpoint method {method_name!r}; refusing to route it through the PHNN loader."
            )

        sources.append(
            LoadedTrainingModel(
                label=label,
                path=path,
                method_name=method_name,
                kind=kind,
                checkpoint=checkpoint,
                config=config,
                method_cfg=method_cfg,
                model=model,
                base_td_params=base_td_params,
                td_memory_cfg=td_memory_cfg,
                correction_mode=correction_mode,
                mean_active=mean_active,
                predict_sigma=predict_sigma,
                fhat_active=fhat_active,
                use_td_force_input=use_td_force_input,
                td_force_input_source=td_force_input_source,
                use_td_fhat_input=use_td_fhat_input,
                use_acceleration_input=use_acceleration_input,
                use_phi_input=use_phi_input,
                phi_input_source=phi_input_source,
                use_sigma_inputs=use_sigma_inputs,
                fhat_bound_multiplier=fhat_bound_multiplier,
                sigma_min=sigma_min,
                force_zero_output=force_zero_output,
            )
        )
    return sources


def resolve_effective_td_params(source: LoadedTrainingModel, *, ur_value: float) -> dict[str, float]:
    if source.kind != "td_parameter_model":
        return dict(source.base_td_params)
    device = _module_device(source.model)
    dtype = _module_dtype(source.model)
    with torch.no_grad():
        reduced_velocity = torch.tensor([[float(ur_value)]], device=device, dtype=dtype)
        params = source.model(reduced_velocity, like=reduced_velocity)
    return {
        name: float(torch.as_tensor(value).detach().cpu().reshape(-1)[0])
        for name, value in params.items()
    }


def _resolve_td_params_for_dt(
    params: dict[str, float],
    *,
    dt: float,
    td_memory_tau_s: float | str | None,
    flow_speed: float | None = None,
    diameter: float | None = None,
) -> dict[str, float]:
    resolved = dict(params)
    if td_memory_tau_s is None:
        return resolved
    dt_value = float(dt)
    if not np.isfinite(dt_value) or dt_value <= 0.0:
        raise ValueError("dt must be positive and finite when resolving n_memory from tau.")
    if isinstance(td_memory_tau_s, str):
        tau_mode = td_memory_tau_s.strip().lower()
        tau_over_tref = 2.0
        if tau_mode == "auto":
            tau_over_tref = 2.0
        elif tau_mode.startswith("auto:"):
            tau_over_tref = float(tau_mode.split(":", 1)[1])
        elif tau_mode.startswith("tau_over_tref:"):
            tau_over_tref = float(tau_mode.split(":", 1)[1])
        else:
            raise ValueError(
                "td_memory_tau_s must be None, a positive number, 'auto', or 'tau_over_tref:<value>'."
            )
        flow_speed_value = float(flow_speed) if flow_speed is not None else float("nan")
        diameter_value = float(diameter) if diameter is not None else float("nan")
        fhat0 = float(params.get("fhat0", float("nan")))
        if not np.isfinite(flow_speed_value) or abs(flow_speed_value) <= 0.0:
            raise ValueError("Need a finite non-zero flow speed to resolve td_memory_tau_s from tau/T_ref.")
        if not np.isfinite(diameter_value) or diameter_value <= 0.0:
            raise ValueError("Need a finite positive diameter to resolve td_memory_tau_s from tau/T_ref.")
        if not np.isfinite(fhat0) or fhat0 <= 0.0:
            raise ValueError("Need a finite positive fhat0 to resolve td_memory_tau_s from tau/T_ref.")
        if not np.isfinite(tau_over_tref) or tau_over_tref <= 0.0:
            raise ValueError("tau/T_ref must be positive and finite when td_memory_tau_s is string-configured.")
        tau_value = tau_over_tref / (fhat0 * abs(flow_speed_value) / diameter_value)
    else:
        tau_value = float(td_memory_tau_s)
    if not np.isfinite(tau_value) or tau_value <= 0.0:
        raise ValueError("td_memory_tau_s must be positive and finite when provided.")
    resolved["n_memory"] = max(1.0, float(round(tau_value / dt_value)))
    return resolved


def _source_td_memory_tau_spec(
    source: LoadedTrainingModel,
    *,
    td_memory_tau_s: float | str | None,
) -> float | str | None:
    if td_memory_tau_s is not None:
        return td_memory_tau_s
    cfg = dict(source.td_memory_cfg or {})
    mode = str(cfg.get("mode", "fixed_n_memory")).strip().lower()
    if mode == "fixed_n_memory":
        return None
    if mode == "fixed_tau":
        tau_s = cfg.get("tau_s", None)
        return None if tau_s is None else float(tau_s)
    if mode == "tau_over_tref":
        return f"tau_over_tref:{float(cfg.get('tau_over_tref', 4.0))}"
    return None


def _td_memory_cfg_from_tau_spec(
    source: LoadedTrainingModel,
    *,
    td_memory_tau_s: float | str | None,
) -> dict[str, Any]:
    if td_memory_tau_s is None:
        return resolve_td_memory_config(source.td_memory_cfg)
    if isinstance(td_memory_tau_s, str):
        tau_mode = td_memory_tau_s.strip().lower()
        if tau_mode == "auto":
            return resolve_td_memory_config({"td_memory_mode": "tau_over_tref", "td_tau_over_tref": 2.0})
        if tau_mode.startswith("auto:"):
            return resolve_td_memory_config(
                {"td_memory_mode": "tau_over_tref", "td_tau_over_tref": float(tau_mode.split(":", 1)[1])}
            )
        if tau_mode.startswith("tau_over_tref:"):
            return resolve_td_memory_config(
                {"td_memory_mode": "tau_over_tref", "td_tau_over_tref": float(tau_mode.split(":", 1)[1])}
            )
        raise ValueError("td_memory_tau_s string must be 'auto', 'auto:<value>', or 'tau_over_tref:<value>'.")
    return resolve_td_memory_config({"td_memory_mode": "fixed_tau", "td_memory_tau_s": float(td_memory_tau_s)})


def _preferred_scalar_from_npz(data: Any, keys: Sequence[str], *, required: bool = True) -> float | None:
    for key in keys:
        if key not in data:
            continue
        arr = np.asarray(data[key], dtype=float).reshape(-1)
        if arr.size == 1 and np.isfinite(arr[0]):
            return float(arr[0])
    if required:
        raise KeyError(f"Expected one finite scalar from keys {list(keys)}.")
    return None


def _resolve_force_span(npz_path: Path, data: Any) -> float:
    span = _preferred_scalar_from_npz(
        data,
        ("physical_span_m", "raw_force_span_scale_applied", "span_m", "python_span_m"),
        required=False,
    )
    if span is not None and np.isfinite(span) and span > 0.0:
        return float(span)
    metadata_row = _metadata_rows().get(npz_path.stem)
    meta_span = _metadata_float(metadata_row, "span_m")
    if meta_span is not None and meta_span > 0.0:
        return float(meta_span)
    return float(DEFAULT_CFD_FORCE_TOTAL_SCALE)


def _series_label_ur(traj: dict[str, Any]) -> float:
    for key in ("ur_label", "ur_stored", "ur"):
        values = traj.get(key)
        if values is None:
            continue
        arr = np.asarray(values, dtype=float).reshape(-1)
        finite = arr[np.isfinite(arr)]
        if finite.size > 0:
            return float(finite[0])
    return float("nan")


def _compute_effective_ur(flow_speed: np.ndarray, *, stiffness: float, effective_mass: float, diameter: float) -> float:
    flow_speed_arr = np.asarray(flow_speed, dtype=float).reshape(-1)
    finite_speed = flow_speed_arr[np.isfinite(flow_speed_arr)]
    if finite_speed.size == 0:
        return float("nan")
    if not (
        np.isfinite(stiffness)
        and stiffness > 0.0
        and np.isfinite(effective_mass)
        and effective_mass > 0.0
        and np.isfinite(diameter)
        and diameter > 0.0
    ):
        return float("nan")
    natural_frequency_hz = float(np.sqrt(stiffness / effective_mass) / (2.0 * np.pi))
    if not np.isfinite(natural_frequency_hz) or natural_frequency_hz <= 0.0:
        return float("nan")
    reference_speed = float(np.median(finite_speed))
    return float(reference_speed / (natural_frequency_hz * diameter))


def load_series(npz_path: str | Path) -> dict[str, Any]:
    return shared_load_series(npz_path)


def _limit_files(files: list[Path], *, max_files_per_split: int | None) -> list[Path]:
    if max_files_per_split is None:
        return files
    return files[: int(max_files_per_split)]


def iter_npz_files(
    root: str | Path,
    split: str,
    *,
    split_dirs: Sequence[str] = DEFAULT_ORDERED_SPLIT_DIRS,
    max_files_per_split: int | None = None,
) -> list[Path]:
    return shared_iter_npz_files(
        root,
        split,
        split_dirs=split_dirs,
        max_files_per_split=max_files_per_split,
    )


def iter_all_npz_files(
    root: str | Path,
    *,
    split_dirs: Sequence[str] = DEFAULT_ORDERED_SPLIT_DIRS,
    max_files_per_split: int | None = None,
) -> list[Path]:
    return shared_iter_all_npz_files(
        root,
        split_dirs=split_dirs,
        max_files_per_split=max_files_per_split,
    )


def _summary_template_anchor(series_group: Sequence[dict[str, Any]], *, ur_value: float) -> dict[str, Any]:
    if not series_group:
        raise ValueError("Need at least one series to build a summary-template anchor.")

    def _mean_scalar(key: str) -> float:
        return float(np.mean([float(series[key]) for series in series_group], dtype=float))

    def _mean_initial_signal(key: str) -> float:
        values = [float(np.asarray(series[key], dtype=float).reshape(-1)[0]) for series in series_group]
        return float(np.mean(values, dtype=float))

    def _mean_model_ur() -> float:
        values = [_series_reduced_velocity(series) for series in series_group]
        return float(np.mean(values, dtype=float))

    def _mean_raw_dt() -> float:
        values = []
        for series in series_group:
            time = np.asarray(series["time"], dtype=float).reshape(-1)
            if time.size >= 2:
                values.append(float(np.median(np.diff(time))))
        if not values:
            raise ValueError("Need at least two time samples to define summary-template dt.")
        return float(np.mean(values, dtype=float))

    ctx0_stack = np.vstack([np.asarray(series["td_context"], dtype=float)[0, :5] for series in series_group])
    ctx0 = np.mean(ctx0_stack, axis=0, dtype=float)
    flow_speed_values = []
    for series in series_group:
        flow_speed_hist = np.asarray(series["td_context"], dtype=float)[:, 4].reshape(-1)
        finite_speed = flow_speed_hist[np.isfinite(flow_speed_hist)]
        if finite_speed.size > 0:
            flow_speed_values.append(float(np.median(finite_speed)))
    if not flow_speed_values:
        raise ValueError("Anchor series does not have a valid flow-speed history.")
    ctx0[4] = float(np.mean(flow_speed_values, dtype=float))

    return {
        "name": f"anchor_ur_{float(ur_value):.6f}",
        "displacement": np.asarray([_mean_initial_signal("displacement")], dtype=float),
        "velocity": np.asarray([_mean_initial_signal("velocity")], dtype=float),
        "force_td_stored": np.asarray([_mean_initial_signal("force_td_stored")], dtype=float),
        "td_context": np.asarray([ctx0], dtype=float),
        "rho": _mean_scalar("rho"),
        "diameter": _mean_scalar("diameter"),
        "stiffness": _mean_scalar("stiffness"),
        "effective_mass": _mean_scalar("effective_mass"),
        "dry_mass": _mean_scalar("dry_mass"),
        "damping": _mean_scalar("damping"),
        "span": _mean_scalar("span"),
        "ur_model": _mean_model_ur(),
        "raw_dt": _mean_raw_dt(),
        "ur_effective": float(ur_value),
    }


def _interpolate_summary_templates(
    lower_template: dict[str, Any],
    upper_template: dict[str, Any],
    *,
    alpha: float,
    target_ur: float,
) -> dict[str, Any]:
    weight = float(np.clip(alpha, 0.0, 1.0))

    def _interp_scalar(key: str) -> float:
        return float((1.0 - weight) * float(lower_template[key]) + weight * float(upper_template[key]))

    def _interp_array(key: str) -> np.ndarray:
        lower_arr = np.asarray(lower_template[key], dtype=float)
        upper_arr = np.asarray(upper_template[key], dtype=float)
        return ((1.0 - weight) * lower_arr + weight * upper_arr).astype(float, copy=False)

    template = {
        "name": f"interp_ur_{float(target_ur):.6f}",
        "displacement": _interp_array("displacement"),
        "velocity": _interp_array("velocity"),
        "force_td_stored": _interp_array("force_td_stored"),
        "td_context": _interp_array("td_context"),
        "rho": _interp_scalar("rho"),
        "diameter": _interp_scalar("diameter"),
        "stiffness": _interp_scalar("stiffness"),
        "effective_mass": _interp_scalar("effective_mass"),
        "dry_mass": _interp_scalar("dry_mass"),
        "damping": _interp_scalar("damping"),
        "span": _interp_scalar("span"),
        "ur_model": _interp_scalar("ur_model"),
        "raw_dt": _interp_scalar("raw_dt"),
        "ur_effective": float(target_ur),
    }
    template["td_context"][0, 4] = float(template["td_context"][0, 4])
    return template


def _build_summary_sweep_template(series_list: Sequence[dict[str, Any]], target_ur: float) -> dict[str, Any]:
    grouped: dict[float, list[dict[str, Any]]] = {}
    for series in series_list:
        ur_value = float(series["ur_effective"])
        if not np.isfinite(ur_value):
            continue
        grouped.setdefault(float(round(ur_value, 6)), []).append(series)
    if not grouped:
        raise ValueError("No finite U_r values found when building the summary sweep template.")

    anchor_urs = np.asarray(sorted(grouped.keys()), dtype=float)
    anchor_templates = {
        ur_key: _summary_template_anchor(grouped[float(ur_key)], ur_value=float(ur_key))
        for ur_key in anchor_urs
    }
    target_ur_value = float(target_ur)
    if anchor_urs.size == 1:
        return dict(anchor_templates[float(anchor_urs[0])])

    insert_idx = int(np.searchsorted(anchor_urs, target_ur_value, side="left"))
    if insert_idx <= 0:
        return dict(anchor_templates[float(anchor_urs[0])])
    if insert_idx >= anchor_urs.size:
        return dict(anchor_templates[float(anchor_urs[-1])])

    lower_ur = float(anchor_urs[insert_idx - 1])
    upper_ur = float(anchor_urs[insert_idx])
    if np.isclose(target_ur_value, lower_ur):
        return dict(anchor_templates[lower_ur])
    if np.isclose(target_ur_value, upper_ur):
        return dict(anchor_templates[upper_ur])

    span = upper_ur - lower_ur
    if not np.isfinite(span) or span <= 0.0:
        return dict(anchor_templates[lower_ur])
    alpha = (target_ur_value - lower_ur) / span
    return _interpolate_summary_templates(
        anchor_templates[lower_ur],
        anchor_templates[upper_ur],
        alpha=alpha,
        target_ur=target_ur_value,
    )


def _resolve_mass_value(series: dict[str, Any], *, mass_source: str) -> float:
    source = str(mass_source).strip().lower()
    if source == "dry":
        return float(series["dry_mass"])
    if source == "effective":
        return float(series["effective_mass"])
    raise ValueError("mass_source must be 'dry' or 'effective'.")


def _series_reduced_velocity(series: dict[str, Any]) -> float:
    ur_series = np.asarray(series.get("ur", []), dtype=float).reshape(-1)
    finite = ur_series[np.isfinite(ur_series)]
    if finite.size > 0:
        return float(finite[0])
    ur_effective = float(series.get("ur_effective", float("nan")))
    if np.isfinite(ur_effective):
        return float(ur_effective)
    raise ValueError(f"Series {series.get('name', '<unknown>')} does not contain a finite reduced velocity.")


def _latent_encoder_length(source: LoadedTrainingModel) -> int:
    return max(1, int(source.method_cfg.get("encoder_length", 50)))


def _latent_include_acceleration(source: LoadedTrainingModel) -> bool:
    return bool(source.method_cfg.get("encoder_include_acceleration", True))


def _latent_series_mass(series: dict[str, Any], *, mass_source: str) -> float:
    source = str(mass_source).strip().lower()
    if source == "effective":
        return float(series["effective_mass"])
    return float(series["dry_mass"])


def _latent_encoder_features_for_series(
    source: LoadedTrainingModel,
    series: dict[str, Any],
    *,
    mass_source: str,
) -> torch.Tensor:
    model = source.model
    y = torch.from_numpy(np.ascontiguousarray(np.asarray(series["displacement"], dtype=float))).float().unsqueeze(1)
    dy = torch.from_numpy(np.ascontiguousarray(np.asarray(series["velocity"], dtype=float))).float().unsqueeze(1)
    ddy = torch.from_numpy(np.ascontiguousarray(np.asarray(series["acceleration"], dtype=float))).float().unsqueeze(1)
    y_arr = np.asarray(series["displacement"], dtype=float).reshape(-1)
    ur_arr = np.asarray(series["ur"], dtype=float)
    if ur_arr.ndim == 0 or ur_arr.reshape(-1).size == 1:
        ur_np = np.full((y_arr.size,), float(ur_arr.reshape(-1)[0]), dtype=float)
    else:
        ur_np = ur_arr.reshape(-1)
    ur = torch.from_numpy(np.ascontiguousarray(ur_np)).float().reshape(-1, 1)
    flow_speed = torch.from_numpy(np.ascontiguousarray(np.asarray(series["td_context"], dtype=float)[:, 4])).float().unsqueeze(1)
    mass = _latent_series_mass(series, mass_source=mass_source)
    stiffness = float(series["stiffness"])
    diameter = float(getattr(model, "D", series["diameter"]))
    ur_scale = float(getattr(model, "ur_scale", 10.0))
    mode = resolve_phnn_input_scaling_mode(str(getattr(model, "input_scaling_mode", "current")))
    y_feat = y / max(diameter, 1.0e-12)
    if mode == "convective":
        u = torch.clamp(torch.abs(flow_speed), min=1.0e-12)
        v_feat = dy / u
        ur_feat = flow_speed / max(diameter, 1.0e-12)
        acc_feat = ddy * diameter / torch.clamp(u * u, min=1.0e-12)
    else:
        p_scale = math.sqrt(max(mass * stiffness, 1.0e-12)) * diameter
        v_feat = (dy * mass) / max(p_scale, 1.0e-12)
        ur_feat = ur / max(ur_scale, 1.0e-12)
        omega_n = math.sqrt(max(stiffness / max(mass, 1.0e-12), 1.0e-12))
        acc_feat = ddy / max((omega_n * omega_n) * diameter, 1.0e-12)
    parts = [y_feat, v_feat, ur_feat]
    if _latent_include_acceleration(source):
        parts.append(acc_feat)
    return torch.cat(parts, dim=1)


def _encode_latent_history(
    source: LoadedTrainingModel,
    history: torch.Tensor,
) -> np.ndarray:
    device = _module_device(source.model)
    dtype = _module_dtype(source.model)
    with torch.no_grad():
        h0 = source.model.encode(history.unsqueeze(0).to(device=device, dtype=dtype))
    return h0.detach().cpu().numpy().reshape(-1).astype(float, copy=True)


def _latent_state_from_series_history(
    source: LoadedTrainingModel,
    series: dict[str, Any],
    *,
    mass_source: str,
    start_idx: int = 0,
) -> np.ndarray:
    enc_len = _latent_encoder_length(source)
    features = _latent_encoder_features_for_series(source, series, mass_source=mass_source)
    start = max(0, int(start_idx))
    end = start + enc_len
    if int(features.shape[0]) < end:
        raise ValueError(
            f"Series {series.get('name', '<unknown>')} has {int(features.shape[0])} sample(s), "
            f"but latent_rnn encoder_length={enc_len} requires at least {end}."
        )
    return _encode_latent_history(source, features[start:end])


def latent_reference_initial_state(
    source: LoadedTrainingModel,
    series_list: Sequence[dict[str, Any]],
    target_ur: float,
    *,
    mass_source: str,
    max_groups: int = 2,
) -> np.ndarray:
    """Encode nearby real CFD histories and average them into a generated-rollout latent IC."""
    if source.kind != "latent_rnn":
        raise ValueError("latent_reference_initial_state is only valid for latent_rnn sources.")
    enc_len = _latent_encoder_length(source)
    candidates: list[tuple[float, dict[str, Any]]] = []
    for series in series_list:
        time_arr = np.asarray(series["time"], dtype=float).reshape(-1)
        if time_arr.size < enc_len:
            continue
        ur_value = float(series.get("ur_effective", _series_reduced_velocity(series)))
        if np.isfinite(ur_value):
            candidates.append((abs(ur_value - float(target_ur)), series))
    if not candidates:
        raise ValueError("No real series is long enough to initialize the latent_rnn encoder state.")
    h_values = []
    for _dist, series in sorted(candidates, key=lambda item: item[0])[: max(1, int(max_groups))]:
        h_values.append(_latent_state_from_series_history(source, series, mass_source=mass_source))
    return np.mean(np.vstack(h_values), axis=0, dtype=float)


def _wrap_angle_rad(angle: float) -> float:
    angle_value = float(angle)
    return float(math.atan2(math.sin(angle_value), math.cos(angle_value)))


def td_baseline_step_numpy(
    *,
    velocity: float,
    acceleration: float,
    td_context: np.ndarray,
    dt: float,
    rho: float,
    diameter: float,
    params: dict[str, float],
) -> tuple[float, np.ndarray]:
    ddy = float(td_context[0])
    phi_vy = _wrap_angle_rad(td_context[1])
    sig_dy = float(td_context[2])
    sig_ddy = float(td_context[3])
    flow_speed = float(td_context[4])
    n_memory = max(1.0, float(params["n_memory"]))
    dt_value = float(dt)
    velocity_value = float(velocity)
    acceleration_value = float(acceleration)

    speed_mag = math.sqrt(max(flow_speed * flow_speed + velocity_value * velocity_value, 1.0e-12))
    projection = flow_speed / speed_mag
    dy_r = velocity_value * projection
    ddy_r = ddy * projection

    sig_dy_next = math.sqrt(
        max(((n_memory - 1.0) / n_memory) * (sig_dy * sig_dy) + (dy_r * dy_r) / n_memory, 1.0e-12)
    )
    sig_ddy_next = math.sqrt(
        max(((n_memory - 1.0) / n_memory) * (sig_ddy * sig_ddy) + (ddy_r * ddy_r) / n_memory, 1.0e-12)
    )

    cos_phi_dy = dy_r / max(sig_dy_next, 1.0e-12)
    sin_phi_dy = -ddy_r / max(sig_ddy_next, 1.0e-12)
    phi_dy = math.atan2(sin_phi_dy, cos_phi_dy)

    theta = math.atan2(math.sin(phi_dy - phi_vy), math.cos(phi_dy - phi_vy))
    if theta <= 0.0:
        fhat = float(params["fhat0"]) + (float(params["fhat0"]) - float(params["fhat_min"])) * math.sin(theta)
    else:
        fhat = float(params["fhat0"]) + (float(params["fhat_max"]) - float(params["fhat0"])) * math.sin(theta)
    omega_vy = 2.0 * math.pi * fhat * speed_mag / float(diameter)
    phi_vy_next = _wrap_angle_rad(phi_vy + dt_value * omega_vy)

    fdy = -0.5 * float(rho) * float(diameter) * float(params["Cd"]) * speed_mag * velocity_value
    fcv = 0.5 * float(rho) * float(diameter) * float(params["Cv"]) * speed_mag * flow_speed * math.cos(phi_vy_next)
    fca = -0.25 * float(rho) * float(params["Ca"]) * math.pi * (float(diameter) ** 2) * acceleration_value
    force_total = fca + fcv + fdy

    next_context = np.asarray([acceleration_value, phi_vy_next, sig_dy_next, sig_ddy_next, flow_speed], dtype=float)
    return float(force_total), next_context


def structural_step_constant_force_numpy(
    *,
    y: float,
    velocity: float,
    force: float,
    dt: float,
    mass: float,
    damping_c: float,
    stiffness: float,
) -> tuple[float, float, float]:
    y_value = float(y)
    velocity_value = float(velocity)
    force_value = float(force)
    dt_value = float(dt)
    mass_value = float(mass)
    damping_value = float(damping_c)
    stiffness_value = float(stiffness)

    def accel(y_state: float, v_state: float) -> float:
        return (force_value - damping_value * v_state - stiffness_value * y_state) / mass_value

    k1_y = velocity_value
    k1_v = accel(y_value, velocity_value)

    y2 = y_value + 0.5 * dt_value * k1_y
    v2 = velocity_value + 0.5 * dt_value * k1_v
    k2_y = v2
    k2_v = accel(y2, v2)

    y3 = y_value + 0.5 * dt_value * k2_y
    v3 = velocity_value + 0.5 * dt_value * k2_v
    k3_y = v3
    k3_v = accel(y3, v3)

    y4 = y_value + dt_value * k3_y
    v4 = velocity_value + dt_value * k3_v
    k4_y = v4
    k4_v = accel(y4, v4)

    y_next = y_value + (dt_value / 6.0) * (k1_y + 2.0 * k2_y + 2.0 * k3_y + k4_y)
    v_next = velocity_value + (dt_value / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
    a_next = accel(y_next, v_next)
    return float(y_next), float(v_next), float(a_next)


def _vivana_added_mass_coeff_per_m(*, rho: float, diameter: float, td_params: dict[str, float]) -> float:
    return 0.25 * float(rho) * float(td_params["Ca"]) * math.pi * float(diameter) ** 2


def _vivana_added_mass_coeff_torch(
    *,
    rho: float,
    diameter: float,
    td_params: dict[str, float],
    like: torch.Tensor,
) -> torch.Tensor:
    value = _vivana_added_mass_coeff_per_m(rho=rho, diameter=diameter, td_params=td_params)
    return torch.full_like(like, float(value))


def _vivana_coupled_diagnostics(
    *,
    y: float,
    velocity: float,
    phi_vy: float,
    q_dy: float,
    q_ddy: float,
    dt: float,
    mass: float,
    damping: float,
    stiffness: float,
    rho: float,
    diameter: float,
    span: float,
    flow_speed: float,
    params: dict[str, float],
) -> dict[str, float]:
    sig_dy = math.sqrt(max(float(q_dy), 0.0))
    sig_ddy = math.sqrt(max(float(q_ddy), 0.0))
    phi_vy_value = _wrap_angle_rad(phi_vy)
    speed_mag = math.sqrt(max(float(flow_speed) * float(flow_speed) + float(velocity) * float(velocity), 1.0e-12))
    projection = float(flow_speed) / speed_mag
    dy_r = float(velocity) * projection

    force_drag_per_m = -0.5 * float(rho) * float(diameter) * float(params["Cd"]) * speed_mag * float(velocity)
    force_cv_per_m = (
        0.5
        * float(rho)
        * float(diameter)
        * float(params["Cv"])
        * speed_mag
        * float(flow_speed)
        * math.cos(phi_vy_value)
    )
    added_mass_coeff_per_m = 0.25 * float(rho) * float(params["Ca"]) * math.pi * float(diameter) ** 2
    acceleration = (
        force_cv_per_m
        + force_drag_per_m
        - float(damping) * float(velocity)
        - float(stiffness) * float(y)
    ) / max(float(mass) + added_mass_coeff_per_m, 1.0e-12)
    ddy_r = acceleration * projection

    cos_phi_dy = dy_r / max(sig_dy, 1.0e-12)
    sin_phi_dy = -ddy_r / max(sig_ddy, 1.0e-12)
    phi_dy = math.atan2(sin_phi_dy, cos_phi_dy)
    theta = math.atan2(math.sin(phi_dy - phi_vy_value), math.cos(phi_dy - phi_vy_value))
    if theta <= 0.0:
        fhat = float(params["fhat0"]) + (float(params["fhat0"]) - float(params["fhat_min"])) * math.sin(theta)
    else:
        fhat = float(params["fhat0"]) + (float(params["fhat_max"]) - float(params["fhat0"])) * math.sin(theta)
    omega_vy = 2.0 * math.pi * fhat * speed_mag / float(diameter)

    force_added_mass_per_m = -added_mass_coeff_per_m * acceleration
    force_total_per_m = force_cv_per_m + force_drag_per_m + force_added_mass_per_m
    tau = max(float(params["n_memory"]) * float(dt), 1.0e-12)
    q_dy_dot = (dy_r * dy_r - float(q_dy)) / tau
    q_ddy_dot = (ddy_r * ddy_r - float(q_ddy)) / tau
    return {
        "acceleration": float(acceleration),
        "force_total_per_m": float(force_total_per_m),
        "force_total": float(force_total_per_m * float(span)),
        "omega_vy": float(omega_vy),
        "q_dy_dot": float(q_dy_dot),
        "q_ddy_dot": float(q_ddy_dot),
    }


def _simulate_vivana_rk4_coupled(
    *,
    time: np.ndarray,
    initial_displacement: float,
    initial_velocity: float,
    initial_acceleration: float,
    initial_phi_vy: float,
    initial_sig_dy: float,
    initial_sig_ddy: float,
    initial_force_per_m: float,
    mass: float,
    damping: float,
    stiffness: float,
    rho: float,
    diameter: float,
    span: float,
    flow_speed: float,
    params: dict[str, float],
) -> dict[str, np.ndarray]:
    time_arr = np.asarray(time, dtype=float).reshape(-1)
    if time_arr.size < 2:
        raise ValueError("Need at least two time samples for Vivana RK4-coupled rollout.")
    dt = float(np.median(np.diff(time_arr)))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("Need a positive finite dt for Vivana RK4-coupled rollout.")

    n = time_arr.size
    displacement = np.empty((n,), dtype=float)
    velocity = np.empty((n,), dtype=float)
    acceleration = np.empty((n,), dtype=float)
    force_per_m = np.empty((n,), dtype=float)
    force_total = np.empty((n,), dtype=float)
    phi_vy = np.empty((n,), dtype=float)
    q_dy = np.empty((n,), dtype=float)
    q_ddy = np.empty((n,), dtype=float)

    displacement[0] = float(initial_displacement)
    velocity[0] = float(initial_velocity)
    acceleration[0] = float(initial_acceleration)
    force_per_m[0] = float(initial_force_per_m)
    force_total[0] = float(initial_force_per_m) * float(span)
    phi_vy[0] = _wrap_angle_rad(initial_phi_vy)
    q_dy[0] = max(float(initial_sig_dy) ** 2, 0.0)
    q_ddy[0] = max(float(initial_sig_ddy) ** 2, 0.0)

    for idx in range(n - 1):
        state = np.asarray([displacement[idx], velocity[idx], phi_vy[idx], q_dy[idx], q_ddy[idx]], dtype=float)

        def rhs(state_vec: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
            diag = _vivana_coupled_diagnostics(
                y=float(state_vec[0]),
                velocity=float(state_vec[1]),
                phi_vy=float(state_vec[2]),
                q_dy=float(state_vec[3]),
                q_ddy=float(state_vec[4]),
                dt=dt,
                mass=mass,
                damping=damping,
                stiffness=stiffness,
                rho=rho,
                diameter=diameter,
                span=span,
                flow_speed=flow_speed,
                params=params,
            )
            deriv = np.asarray(
                [
                    float(state_vec[1]),
                    diag["acceleration"],
                    diag["omega_vy"],
                    diag["q_dy_dot"],
                    diag["q_ddy_dot"],
                ],
                dtype=float,
            )
            return deriv, diag

        k1, diag1 = rhs(state)
        k2, _ = rhs(state + 0.5 * dt * k1)
        k3, _ = rhs(state + 0.5 * dt * k2)
        k4, _ = rhs(state + dt * k3)
        next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        next_state[2] = _wrap_angle_rad(next_state[2])
        next_state[3] = max(float(next_state[3]), 0.0)
        next_state[4] = max(float(next_state[4]), 0.0)

        displacement[idx + 1] = float(next_state[0])
        velocity[idx + 1] = float(next_state[1])
        phi_vy[idx + 1] = float(next_state[2])
        q_dy[idx + 1] = float(next_state[3])
        q_ddy[idx + 1] = float(next_state[4])
        next_diag = _vivana_coupled_diagnostics(
            y=float(next_state[0]),
            velocity=float(next_state[1]),
            phi_vy=float(next_state[2]),
            q_dy=float(next_state[3]),
            q_ddy=float(next_state[4]),
            dt=dt,
            mass=mass,
            damping=damping,
            stiffness=stiffness,
            rho=rho,
            diameter=diameter,
            span=span,
            flow_speed=flow_speed,
            params=params,
        )
        acceleration[idx + 1] = next_diag["acceleration"]
        force_per_m[idx + 1] = diag1["force_total_per_m"]
        force_total[idx + 1] = diag1["force_total"]

    return {
        "time": time_arr,
        "displacement": displacement,
        "velocity": velocity,
        "acceleration": acceleration,
        "force_per_m": force_per_m,
        "force_total": force_total,
        "phi_vy": phi_vy,
        "sig_dy": np.sqrt(np.clip(q_dy, a_min=0.0, a_max=None)),
        "sig_ddy": np.sqrt(np.clip(q_ddy, a_min=0.0, a_max=None)),
    }


def simulate_structural_response_with_force_rk4(
    series: dict[str, Any],
    force_series: np.ndarray,
    *,
    mass_source: str = "dry",
) -> dict[str, np.ndarray]:
    time = np.asarray(series["time"], dtype=float)
    displacement_true = np.asarray(series["displacement"], dtype=float)
    velocity_true = np.asarray(series["velocity"], dtype=float)
    applied_force = np.asarray(force_series, dtype=float).reshape(-1)

    mass = _resolve_mass_value(series, mass_source=mass_source)
    damping = float(series["damping"])
    stiffness = float(series["stiffness"])
    n = time.size
    if applied_force.size != n:
        raise ValueError(f"Expected force series of length {n}, got {applied_force.size}.")

    displacement = np.empty((n,), dtype=float)
    velocity = np.empty((n,), dtype=float)
    acceleration = np.empty((n,), dtype=float)
    displacement[0] = float(displacement_true[0])
    velocity[0] = float(velocity_true[0])
    acceleration[0] = (applied_force[0] - damping * velocity[0] - stiffness * displacement[0]) / mass

    for idx in range(n - 1):
        dt = float(time[idx + 1] - time[idx])
        y_next, v_next, a_next = structural_step_constant_force_numpy(
            y=displacement[idx],
            velocity=velocity[idx],
            force=applied_force[idx],
            dt=dt,
            mass=mass,
            damping_c=damping,
            stiffness=stiffness,
        )
        displacement[idx + 1] = y_next
        velocity[idx + 1] = v_next
        acceleration[idx + 1] = a_next

    return {
        "displacement": displacement,
        "velocity": velocity,
        "acceleration": acceleration,
    }


def simulate_structural_response_with_force_newmark(
    series: dict[str, Any],
    force_series: np.ndarray,
    *,
    mass_source: str = "dry",
    beta: float = DEFAULT_NEWMARK_BETA,
    gamma: float = DEFAULT_NEWMARK_GAMMA,
) -> dict[str, np.ndarray]:
    time = np.asarray(series["time"], dtype=float)
    displacement_true = np.asarray(series["displacement"], dtype=float)
    velocity_true = np.asarray(series["velocity"], dtype=float)
    applied_force = np.asarray(force_series, dtype=float).reshape(-1)

    mass = _resolve_mass_value(series, mass_source=mass_source)
    damping = float(series["damping"])
    stiffness = float(series["stiffness"])
    n = time.size
    if applied_force.size != n:
        raise ValueError(f"Expected force series of length {n}, got {applied_force.size}.")
    if beta <= 0.0:
        raise ValueError("beta must be positive.")

    displacement = np.empty((n,), dtype=float)
    velocity = np.empty((n,), dtype=float)
    acceleration = np.empty((n,), dtype=float)
    displacement[0] = float(displacement_true[0])
    velocity[0] = float(velocity_true[0])
    acceleration[0] = (applied_force[0] - damping * velocity[0] - stiffness * displacement[0]) / mass

    for idx in range(n - 1):
        dt = float(time[idx + 1] - time[idx])
        a0 = 1.0 / (beta * dt * dt)
        a1 = gamma / (beta * dt)
        a2 = 1.0 / (beta * dt)
        a3 = 1.0 / (2.0 * beta) - 1.0
        a4 = gamma / beta - 1.0
        a5 = dt * (gamma / (2.0 * beta) - 1.0)
        k_eff = stiffness + a0 * mass + a1 * damping
        rhs = (
            applied_force[idx + 1]
            + mass * (a0 * displacement[idx] + a2 * velocity[idx] + a3 * acceleration[idx])
            + damping * (a1 * displacement[idx] + a4 * velocity[idx] + a5 * acceleration[idx])
        )
        y_next = rhs / k_eff
        a_next = a0 * (y_next - displacement[idx]) - a2 * velocity[idx] - a3 * acceleration[idx]
        v_next = velocity[idx] + dt * ((1.0 - gamma) * acceleration[idx] + gamma * a_next)
        displacement[idx + 1] = y_next
        velocity[idx + 1] = v_next
        acceleration[idx + 1] = a_next

    return {
        "displacement": displacement,
        "velocity": velocity,
        "acceleration": acceleration,
    }


def simulate_vivana_td(
    series: dict[str, Any],
    *,
    td_params: dict[str, float],
    mass_source: str = "dry",
    td_memory_tau_s: float | str | None = None,
) -> dict[str, np.ndarray]:
    time = np.asarray(series["time"], dtype=float)
    displacement_true = np.asarray(series["displacement"], dtype=float)
    velocity_true = np.asarray(series["velocity"], dtype=float)
    td_context = np.asarray(series["td_context"], dtype=float)
    mass = _resolve_mass_value(series, mass_source=mass_source)
    damping = float(series["damping"])
    stiffness = float(series["stiffness"])
    rho = float(series["rho"])
    diameter = float(series["diameter"])
    span = float(series["span"])

    step_td_params = _resolve_td_params_for_dt(
        td_params,
        dt=float(np.median(np.diff(time))),
        td_memory_tau_s=td_memory_tau_s,
        flow_speed=float(td_context[0, 4]),
        diameter=diameter,
    )
    coupled = _simulate_vivana_rk4_coupled(
        time=time,
        initial_displacement=float(displacement_true[0]),
        initial_velocity=float(velocity_true[0]),
        initial_acceleration=float(td_context[0, 0]),
        initial_phi_vy=float(td_context[0, 1]),
        initial_sig_dy=float(td_context[0, 2]),
        initial_sig_ddy=float(td_context[0, 3]),
        initial_force_per_m=float(series["force_td_stored"][0]),
        mass=mass,
        damping=damping,
        stiffness=stiffness,
        rho=rho,
        diameter=diameter,
        span=span,
        flow_speed=float(td_context[0, 4]),
        params=step_td_params,
    )

    forced_truth_rk4 = simulate_structural_response_with_force_rk4(series, series["force_per_m"], mass_source=mass_source)
    forced_truth_newmark = simulate_structural_response_with_force_newmark(series, series["force_per_m"], mass_source=mass_source)
    return {
        "displacement_td": np.asarray(coupled["displacement"], dtype=float),
        "velocity_td": np.asarray(coupled["velocity"], dtype=float),
        "force_td": np.asarray(coupled["force_per_m"], dtype=float),
        "force_td_total": np.asarray(coupled["force_total"], dtype=float),
        "displacement_rk4_truth_force": forced_truth_rk4["displacement"],
        "velocity_rk4_truth_force": forced_truth_rk4["velocity"],
        "acceleration_rk4_truth_force": forced_truth_rk4["acceleration"],
        "displacement_newmark_truth_force": forced_truth_newmark["displacement"],
        "velocity_newmark_truth_force": forced_truth_newmark["velocity"],
        "acceleration_newmark_truth_force": forced_truth_newmark["acceleration"],
    }


def simulate_vivana_td_stepwise(
    series: dict[str, Any],
    *,
    td_params: dict[str, float],
    mass_source: str = "dry",
    td_memory_tau_s: float | str | None = None,
    dtype: torch.dtype = torch.float32,
    force_phase_convention: str = "next",
    use_vivana_added_mass_lhs: bool = False,
) -> dict[str, np.ndarray]:
    """Vivana-TD rollout using the same step structure as trained-model rollouts.

    The TD hidden state and force are updated once per time step. The resulting
    force is then held constant while the structural state is advanced with RK4.
    """
    time = np.asarray(series["time"], dtype=float).reshape(-1)
    displacement_true = np.asarray(series["displacement"], dtype=float).reshape(-1)
    velocity_true = np.asarray(series["velocity"], dtype=float).reshape(-1)
    td_context = np.asarray(series["td_context"], dtype=float)
    if time.size < 2:
        raise ValueError("Need at least two time samples for Vivana-TD stepwise rollout.")

    mass = _resolve_mass_value(series, mass_source=mass_source)
    damping = float(series["damping"])
    stiffness = float(series["stiffness"])
    rho = float(series["rho"])
    diameter = float(series["diameter"])
    span = float(series["span"])

    device = torch.device("cpu")
    mass_t = torch.tensor([[float(mass)]], device=device, dtype=dtype)
    damping_t = torch.tensor([[float(damping)]], device=device, dtype=dtype)
    stiffness_t = torch.tensor([[float(stiffness)]], device=device, dtype=dtype)
    z = torch.tensor(
        [[float(displacement_true[0]), float(velocity_true[0]) * float(mass)]],
        device=device,
        dtype=dtype,
    )
    ctx = torch.tensor(td_context[0, :5], device=device, dtype=dtype).view(1, 5).clone()

    n = int(time.size)
    displacement = np.empty((n,), dtype=float)
    velocity = np.empty((n,), dtype=float)
    acceleration = np.empty((n,), dtype=float)
    force_per_m = np.empty((n,), dtype=float)
    force_total = np.empty((n,), dtype=float)
    force_cv_per_m = np.empty((n,), dtype=float)
    force_drag_per_m = np.empty((n,), dtype=float)
    force_added_mass_per_m = np.empty((n,), dtype=float)
    displacement[0] = float(displacement_true[0])
    velocity[0] = float(velocity_true[0])
    acceleration[0] = float(ctx[0, 0].detach().cpu().item())
    force_per_m[0] = 0.0
    force_total[0] = 0.0
    force_cv_per_m[0] = 0.0
    force_drag_per_m[0] = 0.0
    force_added_mass_per_m[0] = 0.0

    with torch.no_grad():
        for idx in range(n - 1):
            dt = float(time[idx + 1] - time[idx])
            step_td_params = _resolve_td_params_for_dt(
                td_params,
                dt=dt,
                td_memory_tau_s=td_memory_tau_s,
                flow_speed=float(ctx[0, 4].item()),
                diameter=diameter,
            )
            velocity_t = z[:, 1:2] / mass_t
            td_force_next, ctx_next, td_diagnostics = td_baseline_step_torch(
                velocity=velocity_t,
                acceleration=ctx[:, 0:1],
                td_context=ctx,
                dt=dt,
                rho=float(rho),
                diameter=float(diameter),
                params=step_td_params,
                force_phase_convention=force_phase_convention,
                return_diagnostics=True,
            )
            step_mass_t = mass_t
            step_force_t = td_force_next
            td_force_record_t = td_force_next
            force_added_mass_record_t = td_diagnostics["force_ca"]
            if bool(use_vivana_added_mass_lhs):
                added_mass_t = _vivana_added_mass_coeff_torch(
                    rho=float(rho),
                    diameter=float(diameter),
                    td_params=step_td_params,
                    like=mass_t,
                )
                step_mass_t = mass_t + added_mass_t
                step_force_t = td_force_next - td_diagnostics["force_ca"]
            y_next, v_next, a_next = structural_step_constant_force_torch(
                y=z[:, 0:1],
                velocity=velocity_t,
                force=step_force_t,
                dt=dt,
                mass=step_mass_t,
                damping_c=damping_t,
                stiffness=stiffness_t,
            )
            if bool(use_vivana_added_mass_lhs):
                force_added_mass_record_t = -added_mass_t * a_next
                td_force_record_t = step_force_t + force_added_mass_record_t
            z = torch.cat([y_next, v_next * mass_t], dim=1)
            ctx = ctx_next.clone()
            ctx[:, 0:1] = a_next

            displacement[idx + 1] = float(y_next.detach().cpu().reshape(-1)[0])
            velocity[idx + 1] = float(v_next.detach().cpu().reshape(-1)[0])
            acceleration[idx + 1] = float(a_next.detach().cpu().reshape(-1)[0])
            force_per_m[idx + 1] = float(td_force_record_t.detach().cpu().reshape(-1)[0])
            force_total[idx + 1] = force_per_m[idx + 1] * float(span)
            force_cv_per_m[idx + 1] = float(td_diagnostics["force_cv"].detach().cpu().reshape(-1)[0])
            force_drag_per_m[idx + 1] = float(td_diagnostics["force_drag"].detach().cpu().reshape(-1)[0])
            force_added_mass_per_m[idx + 1] = float(force_added_mass_record_t.detach().cpu().reshape(-1)[0])

    force_per_m[0] = 0.0
    force_total[0] = 0.0
    force_cv_per_m[0] = 0.0
    force_drag_per_m[0] = 0.0
    force_added_mass_per_m[0] = 0.0

    forced_truth_rk4 = simulate_structural_response_with_force_rk4(series, series["force_per_m"], mass_source=mass_source)
    forced_truth_newmark = simulate_structural_response_with_force_newmark(series, series["force_per_m"], mass_source=mass_source)
    return {
        "displacement_td": displacement,
        "velocity_td": velocity,
        "acceleration_td": acceleration,
        "force_td": force_per_m,
        "force_td_total": force_total,
        "force_cv_td": force_cv_per_m,
        "force_drag_td": force_drag_per_m,
        "force_added_mass_td": force_added_mass_per_m,
        "displacement_rk4_truth_force": forced_truth_rk4["displacement"],
        "velocity_rk4_truth_force": forced_truth_rk4["velocity"],
        "acceleration_rk4_truth_force": forced_truth_rk4["acceleration"],
        "displacement_newmark_truth_force": forced_truth_newmark["displacement"],
        "velocity_newmark_truth_force": forced_truth_newmark["velocity"],
        "acceleration_newmark_truth_force": forced_truth_newmark["acceleration"],
    }


def _simulate_rollout(
    *,
    source: LoadedTrainingModel,
    time: np.ndarray,
    initial_displacement: float,
    initial_velocity: float,
    td_context0: np.ndarray,
    ur_value: float,
    mass: float,
    damping: float,
    stiffness: float,
    rho: float,
    diameter: float,
    span: float,
    td_memory_tau_s: float | str | None = None,
    stochastic: bool = False,
    initial_latent: np.ndarray | None = None,
    dtype: torch.dtype | None = None,
    force_phase_convention: str = "next",
    use_vivana_added_mass_lhs: bool = False,
) -> dict[str, np.ndarray]:
    time_arr = np.asarray(time, dtype=float).reshape(-1)
    if time_arr.size < 2:
        raise ValueError("Need at least two time samples for rollout.")
    td_context_init = np.asarray(td_context0, dtype=float).reshape(-1)
    if td_context_init.size < 5:
        raise ValueError("td_context0 must contain at least five entries.")

    if source.kind == "latent_rnn":
        if initial_latent is None:
            raise ValueError(
                f"{source.label}: latent_rnn rollout requires an initial latent state. "
                "Use real-series encoding or latent_reference_initial_state for generated sweeps."
            )
        return _simulate_latent_rnn_rollout(
            source=source,
            time=time_arr,
            initial_displacement=float(initial_displacement),
            initial_velocity=float(initial_velocity),
            initial_latent=initial_latent,
            td_context0=td_context_init,
            ur_value=float(ur_value),
            mass=float(mass),
            damping=float(damping),
            stiffness=float(stiffness),
            diameter=float(diameter),
            span=float(span),
            dtype=dtype,
        )

    device = _module_device(source.model)
    dtype = _module_dtype(source.model) if dtype is None else dtype
    mass_t = torch.tensor([[float(mass)]], device=device, dtype=dtype)
    damping_t = torch.tensor([[float(damping)]], device=device, dtype=dtype)
    stiffness_t = torch.tensor([[float(stiffness)]], device=device, dtype=dtype)
    ur_t = torch.tensor([[float(ur_value)]], device=device, dtype=dtype)
    z = torch.tensor([[float(initial_displacement), float(initial_velocity) * float(mass)]], device=device, dtype=dtype)
    ctx = torch.tensor(td_context_init[:5], device=device, dtype=dtype).view(1, 5).clone()
    displacement = np.empty((time_arr.size,), dtype=float)
    velocity = np.empty((time_arr.size,), dtype=float)
    td_force_per_m = np.empty((time_arr.size,), dtype=float)
    td_force_total = np.empty((time_arr.size,), dtype=float)
    delta_force_per_m = np.empty((time_arr.size,), dtype=float)
    delta_force_total = np.empty((time_arr.size,), dtype=float)
    delta_fhat = np.empty((time_arr.size,), dtype=float)
    fhat_td = np.empty((time_arr.size,), dtype=float)
    fhat_corr = np.empty((time_arr.size,), dtype=float)
    force_per_m = np.empty((time_arr.size,), dtype=float)
    force_total = np.empty((time_arr.size,), dtype=float)
    displacement[0] = float(initial_displacement)
    velocity[0] = float(initial_velocity)
    td_force_per_m[0] = 0.0
    td_force_total[0] = 0.0
    delta_force_per_m[0] = 0.0
    delta_force_total[0] = 0.0
    delta_fhat[0] = 0.0
    fhat_td[0] = 0.0
    fhat_corr[0] = 0.0
    force_per_m[0] = 0.0
    force_total[0] = 0.0

    if source.kind == "td_parameter_model":
        td_params = resolve_effective_td_params(source, ur_value=float(ur_value))
    else:
        td_params = dict(source.base_td_params)

    effective_td_memory_tau = _source_td_memory_tau_spec(source, td_memory_tau_s=td_memory_tau_s)
    effective_td_memory_cfg = _td_memory_cfg_from_tau_spec(source, td_memory_tau_s=td_memory_tau_s)
    with torch.no_grad():
        for idx in range(time_arr.size - 1):
            dt = float(time_arr[idx + 1] - time_arr[idx])
            step_td_params = _resolve_td_params_for_dt(
                td_params,
                dt=dt,
                td_memory_tau_s=effective_td_memory_tau,
                flow_speed=float(ctx[0, 4].item()),
                diameter=diameter,
            )
            velocity_t = z[:, 1:2] / mass_t
            y_current_t = z[:, 0:1]
            added_mass_t = _vivana_added_mass_coeff_torch(
                rho=float(rho),
                diameter=float(diameter),
                td_params=step_td_params,
                like=mass_t,
            )
            td_force_ca_for_lhs_t = -added_mass_t * ctx[:, 0:1]
            if source.kind == "td_parameter_model":
                td_force_next, ctx_next, td_diag = td_baseline_step_torch(
                    velocity=velocity_t,
                    acceleration=ctx[:, 0:1],
                    td_context=ctx,
                    dt=dt,
                    rho=float(rho),
                    diameter=float(diameter),
                    params=step_td_params,
                    force_phase_convention=force_phase_convention,
                    return_diagnostics=True,
                )
                delta_force_per_m_t = torch.zeros_like(td_force_next)
                total_force_per_m_t = td_force_next
                delta_fhat_t = torch.zeros_like(td_force_next)
                fhat_td_t = td_diag["fhat_td"]
                fhat_corr_t = td_diag["fhat_td"]
                y_next, v_next, a_next = structural_step_constant_force_torch(
                    y=z[:, 0:1],
                    velocity=velocity_t,
                    force=total_force_per_m_t,
                    dt=dt,
                    mass=mass_t,
                    damping_c=damping_t,
                    stiffness=stiffness_t,
                )
                z = torch.cat([y_next, v_next * mass_t], dim=1)
                ctx = ctx_next.clone()
                ctx[:, 0:1] = a_next
            elif source.kind == "poly3d_correction":
                td_force_next, ctx_next, td_diag = td_baseline_step_torch(
                    velocity=velocity_t,
                    acceleration=ctx[:, 0:1],
                    td_context=ctx,
                    dt=dt,
                    rho=float(rho),
                    diameter=float(diameter),
                    params=step_td_params,
                    force_phase_convention=force_phase_convention,
                    return_diagnostics=True,
                )
                corr_mu, sigma_corr = source.model.predict(
                    z=z,
                    reduced_velocity=ur_t,
                    structural_mass=mass_t,
                    stiffness=stiffness_t,
                    predict_sigma=source.predict_sigma,
                )
                if stochastic and source.predict_sigma:
                    noise = torch.randn(corr_mu.shape, device=corr_mu.device, dtype=corr_mu.dtype)
                    delta_force_per_m_t = corr_mu + sigma_corr * noise
                else:
                    delta_force_per_m_t = corr_mu
                total_force_per_m_t = td_force_next + delta_force_per_m_t
                delta_fhat_t = torch.zeros_like(td_force_next)
                fhat_td_t = td_diag["fhat_td"]
                fhat_corr_t = td_diag["fhat_td"]
                y_next, v_next, a_next = structural_step_constant_force_torch(
                    y=z[:, 0:1],
                    velocity=velocity_t,
                    force=total_force_per_m_t,
                    dt=dt,
                    mass=mass_t,
                    damping_c=damping_t,
                    stiffness=stiffness_t,
                )
                z = torch.cat([y_next, v_next * mass_t], dim=1)
                ctx = ctx_next.clone()
                ctx[:, 0:1] = a_next
            elif source.kind == "phnn_correction":
                step = _td_step_with_corrections(
                    model=source.model,
                    z=z,
                    reduced_velocity=ur_t,
                    td_context=ctx,
                    dt=dt,
                    structural_mass=mass_t,
                    damping_c=damping_t,
                    stiffness=stiffness_t,
                    td_params=td_params,
                    td_memory_cfg=effective_td_memory_cfg,
                    mean_active=source.mean_active,
                    sigma_active=source.predict_sigma,
                    fhat_active=source.fhat_active,
                    td_force_input_source=source.td_force_input_source,
                    fhat_bound_multiplier=source.fhat_bound_multiplier,
                    force_zero_output=source.force_zero_output,
                    rollout_stochastic=stochastic,
                    force_phase_convention=force_phase_convention,
                )
                td_force_next = step["td_force_next"]
                delta_force_per_m_t = step["corr_force"]
                total_force_per_m_t = step["total_force_next"]
                delta_fhat_t = step["delta_fhat"]
                fhat_td_t = step["fhat_td"]
                fhat_corr_t = step["fhat_corr"]
                z = step["z_next_mean"]
                ctx = step["td_context_next"]
                y_next = z[:, 0:1]
                v_next = z[:, 1:2] / mass_t
            else:
                raise ValueError(f"Unsupported model kind for rollout: {source.kind}")

            if bool(use_vivana_added_mass_lhs):
                td_force_without_added_mass_t = td_force_next - td_force_ca_for_lhs_t
                force_without_added_mass_t = total_force_per_m_t - td_force_ca_for_lhs_t
                y_next, v_next, a_next = structural_step_constant_force_torch(
                    y=y_current_t,
                    velocity=velocity_t,
                    force=force_without_added_mass_t,
                    dt=dt,
                    mass=mass_t + added_mass_t,
                    damping_c=damping_t,
                    stiffness=stiffness_t,
                )
                vivana_added_mass_force_t = -added_mass_t * a_next
                td_force_next = td_force_without_added_mass_t + vivana_added_mass_force_t
                total_force_per_m_t = force_without_added_mass_t + vivana_added_mass_force_t
                z = torch.cat([y_next, v_next * mass_t], dim=1)
                ctx = ctx.clone()
                ctx[:, 0:1] = a_next

            td_force_total_t = td_force_next * float(span)
            delta_force_total_t = delta_force_per_m_t * float(span)
            total_force_t = total_force_per_m_t * float(span)
            displacement[idx + 1] = float(y_next.detach().cpu().reshape(-1)[0])
            velocity[idx + 1] = float(v_next.detach().cpu().reshape(-1)[0])
            td_force_per_m[idx + 1] = float(td_force_next.detach().cpu().reshape(-1)[0])
            td_force_total[idx + 1] = float(td_force_total_t.detach().cpu().reshape(-1)[0])
            delta_force_per_m[idx + 1] = float(delta_force_per_m_t.detach().cpu().reshape(-1)[0])
            delta_force_total[idx + 1] = float(delta_force_total_t.detach().cpu().reshape(-1)[0])
            delta_fhat[idx + 1] = float(delta_fhat_t.detach().cpu().reshape(-1)[0])
            fhat_td[idx + 1] = float(fhat_td_t.detach().cpu().reshape(-1)[0])
            fhat_corr[idx + 1] = float(fhat_corr_t.detach().cpu().reshape(-1)[0])
            force_per_m[idx + 1] = float(total_force_per_m_t.detach().cpu().reshape(-1)[0])
            force_total[idx + 1] = float(total_force_t.detach().cpu().reshape(-1)[0])

    td_force_per_m[0] = 0.0
    td_force_total[0] = 0.0
    delta_force_per_m[0] = 0.0
    delta_force_total[0] = 0.0
    delta_fhat[0] = 0.0
    fhat_td[0] = 0.0
    fhat_corr[0] = 0.0
    force_per_m[0] = 0.0
    force_total[0] = 0.0

    return {
        "time": time_arr,
        "displacement": displacement,
        "velocity": velocity,
        "td_force": td_force_per_m,
        "td_force_total": td_force_total,
        "delta_force": delta_force_per_m,
        "delta_force_total": delta_force_total,
        "delta_fhat": delta_fhat,
        "fhat_td": fhat_td,
        "fhat_corr": fhat_corr,
        "force": force_per_m,
        "force_total": force_total,
        "final_state": {
            "displacement": float(displacement[-1]),
            "velocity": float(velocity[-1]),
            "force": float(force_per_m[-1]),
            "delta_fhat": float(delta_fhat[-1]),
            "td_context": ctx.detach().cpu().numpy().reshape(-1).astype(float, copy=True),
        },
    }


def simulate_checkpoint_series_rollout(
    source: LoadedTrainingModel,
    series: dict[str, Any],
    *,
    mass_source: str = "dry",
    td_memory_tau_s: float | str | None = None,
    dtype: torch.dtype | None = None,
    force_phase_convention: str = "next",
    use_vivana_added_mass_lhs: bool = False,
    latent_encoder_series: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    if source.kind == "latent_rnn":
        time_arr = np.asarray(series["time"], dtype=float).reshape(-1)
        enc_len = _latent_encoder_length(source)
        encoder_source = series if latent_encoder_series is None else latent_encoder_series
        encoder_time = np.asarray(encoder_source["time"], dtype=float).reshape(-1)
        if encoder_time.size < enc_len:
            raise ValueError(
                f"Series {encoder_source.get('name', series.get('name', '<unknown>'))} is too short for latent_rnn encoding "
                f"with encoder_length={enc_len}."
            )
        encoder_end_time = float(encoder_time[enc_len - 1])
        rollout_start_idx = int(np.searchsorted(time_arr, encoder_end_time, side="left"))
        if rollout_start_idx >= time_arr.size - 1:
            raise ValueError(
                f"Series {series.get('name', '<unknown>')} is too short for latent_rnn rollout after "
                f"encoding through t={encoder_end_time:g}."
            )
        latent0 = _latent_state_from_series_history(source, encoder_source, mass_source=mass_source, start_idx=0)
        rollout = _simulate_rollout(
            source=source,
            time=time_arr[rollout_start_idx:],
            initial_displacement=float(np.asarray(series["displacement"], dtype=float).reshape(-1)[rollout_start_idx]),
            initial_velocity=float(np.asarray(series["velocity"], dtype=float).reshape(-1)[rollout_start_idx]),
            td_context0=np.asarray(series["td_context"], dtype=float)[rollout_start_idx],
            ur_value=_series_reduced_velocity(series),
            mass=_resolve_mass_value(series, mass_source=mass_source),
            damping=float(series["damping"]),
            stiffness=float(series["stiffness"]),
            rho=float(series["rho"]),
            diameter=float(series["diameter"]),
            span=float(series["span"]),
            td_memory_tau_s=td_memory_tau_s,
            initial_latent=latent0,
            dtype=dtype,
            force_phase_convention=force_phase_convention,
            use_vivana_added_mass_lhs=use_vivana_added_mass_lhs,
        )
        full = dict(rollout)
        prefix_len = rollout_start_idx
        prefix_map = {
            "displacement": np.asarray(series["displacement"], dtype=float).reshape(-1)[:prefix_len],
            "velocity": np.asarray(series["velocity"], dtype=float).reshape(-1)[:prefix_len],
            "force": np.asarray(series["force_per_m"], dtype=float).reshape(-1)[:prefix_len],
            "force_total": np.asarray(series["force_total"], dtype=float).reshape(-1)[:prefix_len],
            "td_force": np.asarray(series["force_td_stored"], dtype=float).reshape(-1)[:prefix_len],
            "td_force_total": np.asarray(series["force_td_stored"], dtype=float).reshape(-1)[:prefix_len] * float(series["span"]),
            "delta_force": np.zeros((prefix_len,), dtype=float),
            "delta_force_total": np.zeros((prefix_len,), dtype=float),
            "delta_fhat": np.zeros((prefix_len,), dtype=float),
            "fhat_td": np.zeros((prefix_len,), dtype=float),
            "fhat_corr": np.zeros((prefix_len,), dtype=float),
        }
        full["time"] = time_arr.copy()
        for key, prefix in prefix_map.items():
            if key in rollout:
                full[key] = np.concatenate([prefix, np.asarray(rollout[key], dtype=float).reshape(-1)])
        if "latent" in rollout:
            first_latent = np.repeat(latent0.reshape(1, -1), prefix_len, axis=0)
            full["latent"] = np.vstack([first_latent, np.asarray(rollout["latent"], dtype=float)])
        full["evaluation_start_idx"] = int(rollout_start_idx + 1)
        return full

    return _simulate_rollout(
        source=source,
        time=np.asarray(series["time"], dtype=float),
        initial_displacement=float(np.asarray(series["displacement"], dtype=float).reshape(-1)[0]),
        initial_velocity=float(np.asarray(series["velocity"], dtype=float).reshape(-1)[0]),
        td_context0=np.asarray(series["td_context"], dtype=float)[0],
        ur_value=_series_reduced_velocity(series),
        mass=_resolve_mass_value(series, mass_source=mass_source),
        damping=float(series["damping"]),
        stiffness=float(series["stiffness"]),
        rho=float(series["rho"]),
        diameter=float(series["diameter"]),
        span=float(series["span"]),
        td_memory_tau_s=td_memory_tau_s,
        dtype=dtype,
        force_phase_convention=force_phase_convention,
        use_vivana_added_mass_lhs=use_vivana_added_mass_lhs,
    )


_ROLLOUT_ARRAY_KEYS: tuple[str, ...] = (
    "time", "displacement", "velocity",
    "td_force", "td_force_total",
    "delta_force", "delta_force_total",
    "force", "force_total",
)


def _perturbed_initial_state(template: dict[str, Any], perturbation_fraction: float = 0.1) -> dict[str, Any]:
    diameter = float(template["diameter"])
    flow_speed_hist = np.asarray(template["td_context"][:, 4], dtype=float).reshape(-1)
    finite_speed = flow_speed_hist[np.isfinite(flow_speed_hist)]
    flow_speed = float(np.median(finite_speed)) if finite_speed.size > 0 else 1.0
    ctx = np.zeros(5, dtype=float)
    ctx[2] = 1.0e-6  # sig_dy: small non-zero to avoid div-by-zero in TD model
    ctx[3] = 1.0e-6  # sig_ddy
    ctx[4] = flow_speed
    return {
        "displacement": float(perturbation_fraction) * diameter,
        "velocity": 0.0,
        "force": 0.0,
        "td_context": ctx,
    }


def _find_steady_state_onset(
    displacement: np.ndarray,
    dt: float,
    estimated_period_s: float,
    n_cycles: int = 5,
    amp_rel_tol: float = 0.05,
) -> int:
    """Return the first index where amplitude is considered converged.

    Scans forward with steps of half a period.  At each candidate index i,
    compares std(disp[i : i+window]) with std(disp[i+window : i+2*window]).
    Returns i the first time they agree within amp_rel_tol, or the start of
    the final window if no convergence is detected.
    """
    n = len(displacement)
    samples_per_cycle = max(1, int(round(estimated_period_s / max(dt, 1.0e-12))))
    window = max(n_cycles * samples_per_cycle, 4)
    step = max(1, samples_per_cycle // 2)

    if n < 2 * window + 1:
        return 0

    for i in range(0, n - 2 * window, step):
        seg1 = displacement[i : i + window]
        seg2 = displacement[i + window : i + 2 * window]
        amp1 = float(np.std(seg1))
        amp2 = float(np.std(seg2))
        if amp1 < 1.0e-8 and amp2 < 1.0e-8:
            return i
        if amp1 > 1.0e-8 and abs(amp2 - amp1) / amp1 < amp_rel_tol:
            return i

    return max(0, n - window)


def _trim_rollout_to_onset(
    rollout: dict[str, Any],
    onset_idx: int,
    kept_seconds: float | None,
) -> dict[str, Any]:
    """Slice rollout arrays to [onset_idx, onset_idx + kept_samples] and re-zero time."""
    time_arr = np.asarray(rollout["time"], dtype=float).reshape(-1)
    n = len(time_arr)
    start = min(max(0, int(onset_idx)), n - 1)
    if kept_seconds is not None and kept_seconds > 0.0 and n - start > 1:
        dt_est = float(np.median(np.diff(time_arr[start:])))
        if dt_est > 0.0:
            end = min(n, start + int(round(float(kept_seconds) / dt_est)))
        else:
            end = n
    else:
        end = n
    if end - start < 4:
        start, end = 0, n
    trimmed = dict(rollout)
    for key in _ROLLOUT_ARRAY_KEYS:
        if key in rollout:
            arr = np.asarray(rollout[key], dtype=float).reshape(-1)
            trimmed[key] = arr[start:end]
    trimmed["time"] = trimmed["time"] - float(trimmed["time"][0])
    return trimmed


def generate_vivana_summary_rollout(
    template: dict[str, Any],
    target_ur: float,
    *,
    generation_dt: float,
    generation_duration_s: float,
    transient_seconds: float,
    td_params: dict[str, float],
    mass_source: str = "dry",
    initial_state: dict[str, Any] | None = None,
    td_memory_tau_s: float | str | None = None,
) -> dict[str, np.ndarray]:
    flow_speed_hist = np.asarray(template["td_context"][:, 4], dtype=float).reshape(-1)
    finite_speed = flow_speed_hist[np.isfinite(flow_speed_hist)]
    if finite_speed.size == 0:
        raise ValueError(f"Template {template['name']} does not have a valid flow-speed history.")
    target_ur_value = float(target_ur)
    if not np.isfinite(target_ur_value) or target_ur_value <= 0.0:
        raise ValueError("target_ur must be positive and finite.")

    effective_mass = float(template["effective_mass"])
    diameter = float(template["diameter"])
    rho = float(template["rho"])
    damping = float(template["damping"])
    dry_mass = float(template["dry_mass"])
    span = float(template["span"])
    flow_speed_const = float(np.median(finite_speed))
    target_natural_frequency_hz = flow_speed_const / (target_ur_value * diameter)
    target_stiffness = effective_mass * (2.0 * np.pi * target_natural_frequency_hz) ** 2

    source = str(mass_source).strip().lower()
    if source == "dry":
        mass = dry_mass
    elif source == "effective":
        mass = effective_mass
    else:
        raise ValueError("mass_source must be 'dry' or 'effective'.")

    n_steps = int(np.ceil(float(generation_duration_s) / float(generation_dt))) + 1
    time_full = float(generation_dt) * np.arange(n_steps, dtype=float)
    if initial_state is None:
        initial_displacement = float(template["displacement"][0])
        initial_velocity = float(template["velocity"][0])
        initial_force = float(template["force_td_stored"][0])
        ctx = np.asarray(template["td_context"][0], dtype=float).copy()
    else:
        initial_displacement = float(initial_state["displacement"])
        initial_velocity = float(initial_state["velocity"])
        initial_force = float(initial_state.get("force", template["force_td_stored"][0]))
        ctx = np.asarray(initial_state["td_context"], dtype=float).reshape(-1)[:5].copy()
    ctx[4] = flow_speed_const
    step_td_params = _resolve_td_params_for_dt(
        td_params,
        dt=float(generation_dt),
        td_memory_tau_s=td_memory_tau_s,
        flow_speed=flow_speed_const,
        diameter=diameter,
    )
    coupled = _simulate_vivana_rk4_coupled(
        time=time_full,
        initial_displacement=initial_displacement,
        initial_velocity=initial_velocity,
        initial_acceleration=float(ctx[0]),
        initial_phi_vy=float(ctx[1]),
        initial_sig_dy=float(ctx[2]),
        initial_sig_ddy=float(ctx[3]),
        initial_force_per_m=initial_force,
        mass=mass,
        damping=damping,
        stiffness=target_stiffness,
        rho=rho,
        diameter=diameter,
        span=span,
        flow_speed=flow_speed_const,
        params=step_td_params,
    )
    full_displacement = np.asarray(coupled["displacement"], dtype=float)
    full_velocity = np.asarray(coupled["velocity"], dtype=float)
    full_force_td = np.asarray(coupled["force_per_m"], dtype=float)
    full_force_td_total = np.asarray(coupled["force_total"], dtype=float)

    keep_mask = time_full >= float(transient_seconds)
    if np.count_nonzero(keep_mask) < 4:
        raise ValueError("Trimmed Vivana-TD rollout is too short to analyze.")
    kept_count = int(np.count_nonzero(keep_mask))
    return {
        "time": time_full[keep_mask] - float(transient_seconds),
        "displacement": full_displacement[keep_mask],
        "velocity": full_velocity[keep_mask],
        "td_force": full_force_td[keep_mask],
        "td_force_total": full_force_td_total[keep_mask],
        "delta_force": np.zeros((kept_count,), dtype=float),
        "delta_force_total": np.zeros((kept_count,), dtype=float),
        "force": full_force_td[keep_mask],
        "force_total": full_force_td_total[keep_mask],
        "stiffness": float(target_stiffness),
        "effective_mass": float(effective_mass),
        "ur_effective": float(target_ur_value),
        "final_state": {
            "displacement": float(full_displacement[-1]),
            "velocity": float(full_velocity[-1]),
            "force": float(full_force_td[-1]),
            "td_context": np.asarray(
                [
                    float(np.asarray(coupled["acceleration"], dtype=float)[-1]),
                    float(np.asarray(coupled["phi_vy"], dtype=float)[-1]),
                    float(np.asarray(coupled["sig_dy"], dtype=float)[-1]),
                    float(np.asarray(coupled["sig_ddy"], dtype=float)[-1]),
                    float(flow_speed_const),
                ],
                dtype=float,
            ),
        },
    }


def generate_checkpoint_summary_rollout(
    source: LoadedTrainingModel,
    template: dict[str, Any],
    target_ur: float,
    *,
    generation_dt: float,
    generation_duration_s: float,
    transient_seconds: float,
    mass_source: str = "dry",
    initial_state: dict[str, Any] | None = None,
    td_memory_tau_s: float | str | None = None,
    stochastic: bool = False,
    force_phase_convention: str = "next",
    use_vivana_added_mass_lhs: bool = False,
) -> dict[str, np.ndarray]:
    flow_speed_hist = np.asarray(template["td_context"][:, 4], dtype=float).reshape(-1)
    finite_speed = flow_speed_hist[np.isfinite(flow_speed_hist)]
    if finite_speed.size == 0:
        raise ValueError(f"Template {template['name']} does not have a valid flow-speed history.")
    target_ur_value = float(target_ur)
    if not np.isfinite(target_ur_value) or target_ur_value <= 0.0:
        raise ValueError("target_ur must be positive and finite.")

    effective_mass = float(template["effective_mass"])
    diameter = float(template["diameter"])
    rho = float(template["rho"])
    damping = float(template["damping"])
    span = float(template["span"])
    flow_speed_const = float(np.median(finite_speed))
    target_natural_frequency_hz = flow_speed_const / (target_ur_value * diameter)
    target_stiffness = effective_mass * (2.0 * np.pi * target_natural_frequency_hz) ** 2
    if "ur_model" in template:
        model_ur_value = float(template["ur_model"])
    else:
        model_ur_value = _series_reduced_velocity(template)

    n_steps = int(np.ceil(float(generation_duration_s) / float(generation_dt))) + 1
    time_full = float(generation_dt) * np.arange(n_steps, dtype=float)
    if initial_state is None:
        initial_displacement = float(np.asarray(template["displacement"], dtype=float).reshape(-1)[0])
        initial_velocity = float(np.asarray(template["velocity"], dtype=float).reshape(-1)[0])
        td_context0 = np.asarray(template["td_context"], dtype=float)[0].copy()
    else:
        initial_displacement = float(initial_state["displacement"])
        initial_velocity = float(initial_state["velocity"])
        td_context0 = np.asarray(initial_state["td_context"], dtype=float).reshape(-1)[:5].copy()
    td_context0[4] = flow_speed_const
    initial_latent = None
    if source.kind == "latent_rnn":
        if initial_state is not None and "latent" in initial_state:
            initial_latent = np.asarray(initial_state["latent"], dtype=float).reshape(-1)
        elif initial_state is not None and "latent_state" in initial_state:
            initial_latent = np.asarray(initial_state["latent_state"], dtype=float).reshape(-1)
        else:
            initial_latent = np.zeros((int(getattr(source.model, "latent_dim", 1)),), dtype=float)
    rollout = _simulate_rollout(
        source=source,
        time=time_full,
        initial_displacement=initial_displacement,
        initial_velocity=initial_velocity,
        td_context0=td_context0,
        ur_value=model_ur_value,
        mass=_resolve_mass_value(template, mass_source=mass_source),
        damping=damping,
        stiffness=float(target_stiffness),
        rho=rho,
        diameter=diameter,
        span=span,
        td_memory_tau_s=td_memory_tau_s,
        stochastic=stochastic,
        initial_latent=initial_latent,
        force_phase_convention=force_phase_convention,
        use_vivana_added_mass_lhs=use_vivana_added_mass_lhs,
    )
    keep_mask = time_full >= float(transient_seconds)
    if np.count_nonzero(keep_mask) < 4:
        raise ValueError("Trimmed checkpoint rollout is too short to analyze.")
    return {
        "time": rollout["time"][keep_mask] - float(transient_seconds),
        "displacement": rollout["displacement"][keep_mask],
        "velocity": rollout["velocity"][keep_mask],
        "td_force": rollout["td_force"][keep_mask],
        "td_force_total": rollout["td_force_total"][keep_mask],
        "delta_force": rollout["delta_force"][keep_mask],
        "delta_force_total": rollout["delta_force_total"][keep_mask],
        "force": rollout["force"][keep_mask],
        "force_total": rollout["force_total"][keep_mask],
        "stiffness": float(target_stiffness),
        "effective_mass": float(effective_mass),
        "ur_effective": float(target_ur_value),
        "ur_model": float(model_ur_value),
        "final_state": dict(rollout["final_state"]),
    }


def compute_psd_welch(time: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    time_arr = np.asarray(time, dtype=float).reshape(-1)
    values_arr = np.asarray(values, dtype=float).reshape(-1)
    n = min(time_arr.size, values_arr.size)
    if n < 4:
        raise ValueError("Need at least four samples to compute a PSD.")
    time_arr = time_arr[:n]
    values_arr = values_arr[:n]
    dt = float(np.median(np.diff(time_arr)))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("Need a positive finite dt to compute a PSD.")
    fs = 1.0 / dt
    values_centered = values_arr - float(np.mean(values_arr))
    nperseg = n
    noverlap = 0 if nperseg < 16 else min(nperseg // 2, nperseg - 1)
    nfft = max(8 * nperseg, nperseg)
    freqs, psd = signal.welch(
        values_centered,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend="constant",
        scaling="density",
    )
    return freqs, psd


def suggest_psd_xlim(*freq_psd_pairs: tuple[np.ndarray, np.ndarray], harmonics: float = 2.0, min_xmax: float = 0.1) -> float:
    dominant_freqs: list[float] = []
    nyquist_candidates: list[float] = []
    for freqs, psd in freq_psd_pairs:
        freqs_arr = np.asarray(freqs, dtype=float).reshape(-1)
        psd_arr = np.asarray(psd, dtype=float).reshape(-1)
        n = min(freqs_arr.size, psd_arr.size)
        if n < 2:
            continue
        freqs_arr = freqs_arr[:n]
        psd_arr = psd_arr[:n]
        positive = freqs_arr > 0.0
        if not np.any(positive):
            continue
        freqs_pos = freqs_arr[positive]
        psd_pos = psd_arr[positive]
        dom_idx = int(np.argmax(psd_pos))
        dominant_freqs.append(float(freqs_pos[dom_idx]))
        nyquist_candidates.append(float(freqs_arr[-1]))
    if not dominant_freqs:
        return float(min_xmax)
    dominant = max(dominant_freqs)
    nyquist = max(nyquist_candidates) if nyquist_candidates else dominant * harmonics
    xmax = max(min_xmax, harmonics * dominant)
    return float(min(xmax, nyquist))


def normalize_psd_area(freqs: np.ndarray, psd: np.ndarray) -> np.ndarray:
    freqs_arr = np.asarray(freqs, dtype=float).reshape(-1)
    psd_arr = np.asarray(psd, dtype=float).reshape(-1)
    n = min(freqs_arr.size, psd_arr.size)
    if n < 2:
        return psd_arr[:n]
    freqs_arr = freqs_arr[:n]
    psd_arr = np.clip(psd_arr[:n], a_min=0.0, a_max=None)
    area = float(np.trapz(psd_arr, freqs_arr))
    if not np.isfinite(area) or area <= 0.0:
        return psd_arr
    return psd_arr / area


def dominant_frequency_from_signal(
    time: np.ndarray,
    values: np.ndarray,
    *,
    reference_frequency_hz: float | None = None,
    reference_peak_min_relative_height: float = 0.85,
) -> float:
    time_arr = np.asarray(time, dtype=float).reshape(-1)
    values_arr = np.asarray(values, dtype=float).reshape(-1)
    n = min(time_arr.size, values_arr.size)
    if n < 4:
        return float("nan")
    time_arr = time_arr[:n]
    values_arr = values_arr[:n]
    dt = float(np.median(np.diff(time_arr)))
    if not np.isfinite(dt) or dt <= 0.0:
        return float("nan")
    # Keep summary plots consistent with the dominant-frequency relative error metric.
    # The reference-frequency arguments are intentionally ignored here.
    return float(dominant_frequency(values_arr, dt))


def displacement_peak_amplitudes(time: np.ndarray, displacement: np.ndarray, dominant_frequency_hz: float) -> np.ndarray:
    time_arr = np.asarray(time, dtype=float).reshape(-1)
    displacement_arr = np.asarray(displacement, dtype=float).reshape(-1)
    abs_disp = np.abs(displacement_arr)
    if abs_disp.size == 0:
        return np.asarray([], dtype=float)
    dt = float(np.median(np.diff(time_arr))) if time_arr.size >= 2 else 1.0
    if np.isfinite(dominant_frequency_hz) and dominant_frequency_hz > 0.0 and np.isfinite(dt) and dt > 0.0:
        period = 1.0 / dominant_frequency_hz
        min_distance = max(1, int(round(0.35 * period / dt)))
    else:
        min_distance = 1
    prominence = max(1.0e-12, 0.05 * float(np.std(abs_disp)))
    peaks, _ = signal.find_peaks(abs_disp, distance=min_distance, prominence=prominence)
    amplitudes = abs_disp[peaks]
    if amplitudes.size == 0:
        amplitudes = np.asarray([float(np.max(abs_disp))], dtype=float)
    return amplitudes.astype(float)


def instantaneous_phase_lag_deg_samples(time: np.ndarray, reference: np.ndarray, target: np.ndarray, frequency_hz: float) -> np.ndarray:
    time_arr = np.asarray(time, dtype=float).reshape(-1)
    reference_arr = np.asarray(reference, dtype=float).reshape(-1)
    target_arr = np.asarray(target, dtype=float).reshape(-1)
    n = min(time_arr.size, reference_arr.size, target_arr.size)
    if n < 16 or not np.isfinite(frequency_hz) or frequency_hz <= 0.0:
        return np.asarray([], dtype=float)

    time_arr = time_arr[:n]
    reference_arr = reference_arr[:n] - float(np.mean(reference_arr[:n]))
    target_arr = target_arr[:n] - float(np.mean(target_arr[:n]))
    dt = float(np.median(np.diff(time_arr)))
    if not np.isfinite(dt) or dt <= 0.0:
        return np.asarray([], dtype=float)
    sample_rate_hz = 1.0 / dt
    nyquist_hz = 0.5 * sample_rate_hz

    low_hz = max(0.05, 0.65 * float(frequency_hz))
    high_hz = min(0.95 * nyquist_hz, 1.35 * float(frequency_hz))
    if high_hz <= low_hz:
        low_hz = max(0.05, 0.50 * float(frequency_hz))
        high_hz = min(0.95 * nyquist_hz, 1.50 * float(frequency_hz))
    if high_hz <= low_hz:
        return np.asarray([], dtype=float)

    try:
        sos = signal.butter(4, [low_hz, high_hz], btype="bandpass", fs=sample_rate_hz, output="sos")
        reference_filtered = signal.sosfiltfilt(sos, reference_arr)
        target_filtered = signal.sosfiltfilt(sos, target_arr)
    except ValueError:
        return np.asarray([], dtype=float)

    reference_analytic = signal.hilbert(reference_filtered)
    target_analytic = signal.hilbert(target_filtered)
    reference_amp = np.abs(reference_analytic)
    target_amp = np.abs(target_analytic)
    reference_threshold = max(1.0e-12, 0.05 * float(np.sqrt(np.mean(reference_amp**2))))
    target_threshold = max(1.0e-12, 0.05 * float(np.sqrt(np.mean(target_amp**2))))

    lag_rad = np.angle(target_analytic) - np.angle(reference_analytic)
    lag_deg = np.degrees(np.angle(np.exp(1j * lag_rad)))
    edge_trim = int(np.ceil(sample_rate_hz / max(float(frequency_hz), 1.0e-9)))

    mask = np.isfinite(lag_deg) & np.isfinite(reference_amp) & np.isfinite(target_amp)
    mask &= reference_amp >= reference_threshold
    mask &= target_amp >= target_threshold
    if 2 * edge_trim < n:
        mask[:edge_trim] = False
        mask[-edge_trim:] = False
    return np.asarray(lag_deg[mask], dtype=float)


def compute_summary_metrics(
    time: np.ndarray,
    displacement: np.ndarray,
    velocity: np.ndarray,
    force: np.ndarray,
    *,
    stiffness: float,
    effective_mass: float,
) -> dict[str, Any]:
    time_arr = np.asarray(time, dtype=float).reshape(-1)
    displacement_arr = np.asarray(displacement, dtype=float).reshape(-1)
    velocity_arr = np.asarray(velocity, dtype=float).reshape(-1)
    force_arr = np.asarray(force, dtype=float).reshape(-1)
    force_dominant_frequency_hz = dominant_frequency_from_signal(time_arr, force_arr)
    dominant_frequency_hz = dominant_frequency_from_signal(
        time_arr,
        displacement_arr,
        reference_frequency_hz=force_dominant_frequency_hz,
    )
    if np.isfinite(stiffness) and stiffness > 0.0 and np.isfinite(effective_mass) and effective_mass > 0.0:
        natural_frequency_hz = float(np.sqrt(stiffness / effective_mass) / (2.0 * np.pi))
        dominant_frequency_ratio = float(dominant_frequency_hz / natural_frequency_hz) if natural_frequency_hz > 0.0 else float("nan")
        force_dominant_frequency_ratio = (
            float(force_dominant_frequency_hz / natural_frequency_hz)
            if natural_frequency_hz > 0.0
            else float("nan")
        )
    else:
        natural_frequency_hz = float("nan")
        dominant_frequency_ratio = float("nan")
        force_dominant_frequency_ratio = float("nan")
    phase_force_displacement_deg_samples = instantaneous_phase_lag_deg_samples(
        time_arr,
        displacement_arr,
        force_arr,
        dominant_frequency_hz,
    )
    phase_force_velocity_deg_samples = instantaneous_phase_lag_deg_samples(
        time_arr,
        velocity_arr,
        force_arr,
        dominant_frequency_hz,
    )
    return {
        "disp_std": float(np.std(displacement_arr)),
        "force_std": float(np.std(force_arr)),
        "dominant_frequency_hz": float(dominant_frequency_hz),
        "force_dominant_frequency_hz": float(force_dominant_frequency_hz),
        "natural_frequency_hz": float(natural_frequency_hz),
        "dominant_frequency_ratio": float(dominant_frequency_ratio),
        "force_dominant_frequency_ratio": float(force_dominant_frequency_ratio),
        "peak_amplitudes": displacement_peak_amplitudes(time_arr, displacement_arr, dominant_frequency_hz),
        "phase_force_displacement_deg_samples": phase_force_displacement_deg_samples,
        "phase_force_velocity_deg_samples": phase_force_velocity_deg_samples,
    }


def compute_validation_style_error_metrics(
    *,
    time: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    force_true: np.ndarray | None = None,
    force_pred: np.ndarray | None = None,
) -> dict[str, float]:
    time_arr = np.asarray(time, dtype=float).reshape(-1)
    y_true_arr = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred_arr = np.asarray(y_pred, dtype=float).reshape(-1)
    force_true_arr = None if force_true is None else np.asarray(force_true, dtype=float).reshape(-1)
    force_pred_arr = None if force_pred is None else np.asarray(force_pred, dtype=float).reshape(-1)

    min_len_y = min(y_true_arr.size, y_pred_arr.size)
    min_len_force = 0 if force_true_arr is None or force_pred_arr is None else min(force_true_arr.size, force_pred_arr.size)
    dt = float(np.median(np.diff(time_arr))) if time_arr.size >= 2 else float("nan")

    metrics = {
        "Force mapping NRMSE": float("nan"),
        DISP_STD_REL_ERROR_KEY: float("nan"),
        DOMINANT_FREQ_REL_ERROR_KEY: float("nan"),
        FORCE_DOMINANT_FREQ_REL_ERROR_KEY: float("nan"),
        FORCE_STD_REL_ERROR_KEY: float("nan"),
        DISPLACEMENT_MAE_KEY: float("nan"),
        AGGREGATE_VALIDATION_ERROR_KEY: float("nan"),
    }

    if min_len_y >= 1:
        y_true_aligned = y_true_arr[:min_len_y]
        y_pred_aligned = y_pred_arr[:min_len_y]
        metrics[DISPLACEMENT_MAE_KEY] = float(np.mean(np.abs(y_pred_aligned - y_true_aligned)))

    if min_len_force >= 1:
        metrics["Force mapping NRMSE"] = float(_force_mapping_nrmse(force_pred_arr[:min_len_force], force_true_arr[:min_len_force]))

    if min_len_y > 1 and np.isfinite(dt) and dt > 0.0:
        y_true_aligned = y_true_arr[:min_len_y]
        y_pred_aligned = y_pred_arr[:min_len_y]
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

    if min_len_force > 1 and np.isfinite(dt) and dt > 0.0:
        force_true_aligned = force_true_arr[:min_len_force]
        force_pred_aligned = force_pred_arr[:min_len_force]
        true_force_dom = dominant_frequency(force_true_aligned, dt)
        pred_force_dom = dominant_frequency(force_pred_aligned, dt)
        force_dom_rel = relative_error(pred_force_dom, true_force_dom)
        if np.isfinite(force_dom_rel):
            metrics[FORCE_DOMINANT_FREQ_REL_ERROR_KEY] = abs(float(force_dom_rel))

        true_force_std = float(np.std(force_true_aligned))
        pred_force_std = float(np.std(force_pred_aligned))
        force_std_rel = relative_error(pred_force_std, true_force_std)
        if np.isfinite(force_std_rel):
            metrics[FORCE_STD_REL_ERROR_KEY] = abs(float(force_std_rel))

    aggregate_values = [
        float(metrics[key])
        for key in VALIDATION_COMPONENT_METRIC_KEYS
        if np.isfinite(float(metrics.get(key, float("nan"))))
    ]
    if len(aggregate_values) == len(VALIDATION_COMPONENT_METRIC_KEYS):
        metrics[AGGREGATE_VALIDATION_ERROR_KEY] = float(np.mean(aggregate_values))

    return metrics


def _aggregate_grouped_validation_metric(grouped_errors: dict[float, list[float]]) -> dict[str, float | int] | None:
    ur_keys: list[float] = []
    ur_mean_errors: list[float] = []
    case_count = 0
    for ur_key in sorted(grouped_errors.keys()):
        values = np.asarray(grouped_errors[float(ur_key)], dtype=float).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        ur_keys.append(float(ur_key))
        ur_mean_errors.append(float(np.mean(values)))
        case_count += int(values.size)
    if not ur_mean_errors:
        return None
    ur_mean_errors_arr = np.asarray(ur_mean_errors, dtype=float)
    max_idx = int(np.argmax(ur_mean_errors_arr))
    return {
        "effective_ur_count": int(len(ur_keys)),
        "case_count": int(case_count),
        "mean_over_effective_ur": float(np.mean(ur_mean_errors_arr)),
        "max_over_effective_ur": float(np.max(ur_mean_errors_arr)),
        "worst_effective_ur": float(ur_keys[max_idx]),
    }


def _series_raw_ur_value(series: dict[str, Any]) -> float:
    raw_ur = series.get("ur", float("nan"))
    try:
        ur_arr = np.asarray(raw_ur, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return float("nan")
    finite = ur_arr[np.isfinite(ur_arr)]
    return float(finite[0]) if finite.size else float("nan")


def _series_validation_characteristics(series: dict[str, Any]) -> dict[str, float]:
    time_arr = np.asarray(series["time"], dtype=float).reshape(-1)
    displacement_arr = np.asarray(series["displacement"], dtype=float).reshape(-1)
    force_arr = np.asarray(series["force_per_m"], dtype=float).reshape(-1)
    dt = float(np.median(np.diff(time_arr))) if time_arr.size >= 2 else float("nan")
    characteristics = {
        DISP_STD_REL_ERROR_KEY: float(np.std(displacement_arr)),
        FORCE_STD_REL_ERROR_KEY: float(np.std(force_arr)),
        DOMINANT_FREQ_REL_ERROR_KEY: float("nan"),
        FORCE_DOMINANT_FREQ_REL_ERROR_KEY: float("nan"),
    }
    if np.isfinite(dt) and dt > 0.0 and displacement_arr.size > 1:
        characteristics[DOMINANT_FREQ_REL_ERROR_KEY] = float(dominant_frequency(displacement_arr, dt))
    if np.isfinite(dt) and dt > 0.0 and force_arr.size > 1:
        characteristics[FORCE_DOMINANT_FREQ_REL_ERROR_KEY] = float(dominant_frequency(force_arr, dt))
    return characteristics


def _build_inherent_dataset_errors(case_data: Sequence[dict[str, Any]]) -> dict[str, dict[float, list[float]]]:
    """Estimate CFD scatter by comparing each case to the CFD mean at the same effective U_r."""
    values_by_ur: dict[float, dict[str, list[float]]] = {}
    for entry in case_data:
        series = entry.get("validation_series", entry["series"])
        ur_key = float(round(float(series["ur_effective"]), 6))
        grouped = values_by_ur.setdefault(ur_key, {label: [] for label in VALIDATION_COMPONENT_METRIC_KEYS})
        characteristics = _series_validation_characteristics(series)
        for metric_label in VALIDATION_COMPONENT_METRIC_KEYS:
            value = float(characteristics.get(metric_label, float("nan")))
            if np.isfinite(value):
                grouped[metric_label].append(value)

    errors_by_metric: dict[str, dict[float, list[float]]] = {label: {} for label in VALIDATION_TRACKED_METRIC_KEYS}
    for ur_key, metrics_by_name in values_by_ur.items():
        for metric_label in VALIDATION_COMPONENT_METRIC_KEYS:
            values = np.asarray(metrics_by_name.get(metric_label, []), dtype=float).reshape(-1)
            values = values[np.isfinite(values)]
            if values.size < 2:
                continue
            group_mean = float(np.mean(values))
            for value in values:
                error = abs(float(relative_error(float(value), group_mean)))
                if np.isfinite(error):
                    errors_by_metric[metric_label].setdefault(float(ur_key), []).append(error)
    return errors_by_metric


def _build_inherent_raw_ur_errors(case_data: Sequence[dict[str, Any]], raw_ur: float) -> dict[str, list[float]]:
    """Estimate CFD scatter within a raw U_r subset by comparing each case to that subset mean."""
    metrics_by_name: dict[str, list[float]] = {label: [] for label in VALIDATION_COMPONENT_METRIC_KEYS}
    for entry in case_data:
        series = entry.get("validation_series", entry["series"])
        raw_ur_value = _series_raw_ur_value(series)
        if not (np.isfinite(raw_ur_value) and np.isclose(raw_ur_value, float(raw_ur), rtol=0.0, atol=1.0e-9)):
            continue
        characteristics = _series_validation_characteristics(series)
        for metric_label in VALIDATION_COMPONENT_METRIC_KEYS:
            value = float(characteristics.get(metric_label, float("nan")))
            if np.isfinite(value):
                metrics_by_name[metric_label].append(value)

    errors_by_metric: dict[str, list[float]] = {label: [] for label in VALIDATION_COMPONENT_METRIC_KEYS}
    for metric_label, values_list in metrics_by_name.items():
        values = np.asarray(values_list, dtype=float).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size < 2:
            continue
        group_mean = float(np.mean(values))
        for value in values:
            error = abs(float(relative_error(float(value), group_mean)))
            if np.isfinite(error):
                errors_by_metric[metric_label].append(error)
    return errors_by_metric


def _build_validation_summary_rows(
    *,
    dataset_errors_by_metric: dict[str, dict[float, list[float]]] | None = None,
    baseline_errors_by_metric: dict[str, dict[float, list[float]]],
    model_errors_by_metric: dict[str, dict[str, dict[float, list[float]]]],
    sources: Sequence[LoadedTrainingModel],
    unseen_raw_ur: float | None = None,
    unseen_errors_by_method: dict[str, dict[str, list[float]]] | None = None,
    modified_errors_by_metric: dict[str, dict[float, list[float]]] | None = None,
    modified_td_label: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    method_specs: list[tuple[str, dict[str, dict[float, list[float]]]]] = []
    if dataset_errors_by_metric is not None:
        method_specs.append(("Inherent CFD scatter (U_r mean)", dataset_errors_by_metric))
    method_specs.append(("VIVANA-TD baseline", baseline_errors_by_metric))
    if modified_errors_by_metric is not None and modified_td_label is not None:
        method_specs.append((modified_td_label, modified_errors_by_metric))
    method_specs.extend((source.label, model_errors_by_metric[source.label]) for source in sources)
    for method_label, metrics_by_name in method_specs:
        row: dict[str, Any] = {"Method": str(method_label)}
        effective_ur_counts: list[int] = []
        case_counts: list[int] = []
        has_finite_metric = False
        component_values: list[float] = []
        for metric_label in VALIDATION_COMPONENT_METRIC_KEYS:
            stats = _aggregate_grouped_validation_metric(metrics_by_name.get(metric_label, {}))
            if stats is None:
                row[str(metric_label)] = float("nan")
                continue
            metric_value = float(stats["mean_over_effective_ur"])
            row[str(metric_label)] = metric_value
            component_values.append(metric_value)
            effective_ur_counts.append(int(stats["effective_ur_count"]))
            case_counts.append(int(stats["case_count"]))
            has_finite_metric = True
        row[AGGREGATE_VALIDATION_ERROR_KEY] = (
            float(np.mean(component_values))
            if len(component_values) == len(VALIDATION_COMPONENT_METRIC_KEYS)
            else float("nan")
        )
        if unseen_raw_ur is not None:
            column_name = f"{AGGREGATE_VALIDATION_ERROR_KEY} (U_r={float(unseen_raw_ur):.6g} only)"
            unseen_component_values: list[float] = []
            unseen_metrics = {} if unseen_errors_by_method is None else unseen_errors_by_method.get(method_label, {})
            for metric_label in VALIDATION_COMPONENT_METRIC_KEYS:
                unseen_values = np.asarray(unseen_metrics.get(metric_label, []), dtype=float).reshape(-1)
                unseen_values = unseen_values[np.isfinite(unseen_values)]
                if unseen_values.size:
                    unseen_component_values.append(float(np.mean(unseen_values)))
            row[column_name] = (
                float(np.mean(unseen_component_values))
                if len(unseen_component_values) == len(VALIDATION_COMPONENT_METRIC_KEYS)
                else float("nan")
            )
        if not has_finite_metric:
            continue
        row["Effective U_r count"] = int(max(effective_ur_counts)) if effective_ur_counts else 0
        row["Case count"] = int(max(case_counts)) if case_counts else 0
        rows.append(row)
    return rows


def _display_validation_summary_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No finite validation summary rows available.")
        return
    print(
        "Block 10 validation summary table "
        "(four trainer-style rollout validation errors, aggregated from the mean metric value "
        "at each effective reduced velocity; aggregate = mean of the four errors; "
        "U_r-specific aggregate column filters to that raw reduced-velocity label before averaging the four errors; "
        "inherent CFD scatter compares each CFD case with the CFD mean at the same U_r, "
        "and uses the raw-U_r subset mean for the U_r-specific column):"
    )
    try:
        import pandas as pd

        from IPython.display import display

        display(pd.DataFrame(rows))
        return
    except Exception:
        pass

    headers = list(rows[0].keys())

    def _format_value(value: Any) -> str:
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            value_float = float(value)
            return "nan" if not np.isfinite(value_float) else f"{value_float:.6g}"
        return str(value)

    widths: dict[str, int] = {}
    for header in headers:
        rendered = [_format_value(row.get(header, "")) for row in rows]
        widths[header] = max(len(header), *(len(text) for text in rendered))
    header_line = " | ".join(header.ljust(widths[header]) for header in headers)
    separator_line = "-+-".join("-" * widths[header] for header in headers)
    print(header_line)
    print(separator_line)
    for row in rows:
        print(" | ".join(_format_value(row.get(header, "")).ljust(widths[header]) for header in headers))


def global_generation_grid(
    summary_series: list[dict[str, Any]],
    *,
    transient_seconds: float,
    kept_duration_s: float | None = None,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    if not summary_series:
        raise ValueError("Need at least one CFD series to build the common generation grid.")
    dt_values: list[float] = []
    durations: list[float] = []
    for series in summary_series:
        time = np.asarray(series["time"], dtype=float).reshape(-1)
        if time.size < 2:
            continue
        dt_values.append(float(np.min(np.diff(time))))
        durations.append(float(time[-1] - time[0]))
    if not dt_values or not durations:
        raise ValueError("Could not determine generation dt/duration from the CFD series.")
    generation_dt = float(np.min(dt_values))
    if kept_duration_s is None:
        generation_duration_s = float(np.max(durations) + transient_seconds)
    else:
        kept_duration_value = float(kept_duration_s)
        if not np.isfinite(kept_duration_value) or kept_duration_value <= 0.0:
            raise ValueError("kept_duration_s must be positive and finite when provided.")
        generation_duration_s = float(kept_duration_value + transient_seconds)
    n_steps = int(np.ceil(generation_duration_s / generation_dt)) + 1
    time_full = generation_dt * np.arange(n_steps, dtype=float)
    keep_mask = time_full >= transient_seconds
    return generation_dt, generation_duration_s, time_full, keep_mask


def _summary_target_urs(
    series_list: Sequence[dict[str, Any]],
    *,
    fine_ur_step: float,
    summary_ur_range: tuple[float, float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if float(fine_ur_step) <= 0.0:
        raise ValueError("fine_ur_step must be positive.")
    actual_urs = np.asarray(sorted({float(series["ur_effective"]) for series in series_list}), dtype=float)
    if actual_urs.size == 0:
        raise ValueError("No finite U_r values found for the summary sweep.")
    if summary_ur_range is None:
        ur_min = float(np.min(actual_urs))
        ur_max = float(np.max(actual_urs))
    else:
        if len(summary_ur_range) != 2:
            raise ValueError("summary_ur_range must contain exactly two values: (min_ur, max_ur).")
        ur_min = float(summary_ur_range[0])
        ur_max = float(summary_ur_range[1])
        if not np.isfinite(ur_min) or not np.isfinite(ur_max):
            raise ValueError("summary_ur_range values must be finite.")
        if ur_max < ur_min:
            raise ValueError("summary_ur_range max must be greater than or equal to min.")
    fine_urs = np.arange(ur_min, ur_max + 0.5 * float(fine_ur_step), float(fine_ur_step))
    range_tol = 1.0e-9
    exact_urs = actual_urs[(actual_urs >= ur_min - range_tol) & (actual_urs <= ur_max + range_tol)]
    target_urs = np.asarray(sorted({round(float(ur), 6) for ur in np.concatenate([fine_urs, exact_urs])}), dtype=float)
    exact_urs = np.asarray(sorted({round(float(ur), 6) for ur in exact_urs}), dtype=float)
    return target_urs, exact_urs


def _clear_directory_contents(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()




def sorted_group_stats(grouped: dict[float, list[float]]) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    urs = np.asarray(sorted(grouped.keys()), dtype=float)
    values = [np.asarray(grouped[float(ur)], dtype=float) for ur in urs]
    means = np.asarray([float(np.mean(v)) for v in values], dtype=float) if values else np.asarray([], dtype=float)
    return urs, means, values


def _group_plot_stats(grouped_errors: dict[float, list[float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    urs, _, values = sorted_group_stats(grouped_errors)
    if urs.size == 0:
        raise ValueError("No finite grouped values available to plot.")
    mean_vals = np.asarray([float(np.mean(v)) for v in values], dtype=float)
    min_vals = np.asarray([float(np.min(v)) for v in values], dtype=float)
    max_vals = np.asarray([float(np.max(v)) for v in values], dtype=float)
    std_vals = np.asarray([float(np.std(v)) for v in values], dtype=float)
    positive_values = np.concatenate([arr[np.isfinite(arr) & (arr > 0.0)] for arr in values if arr.size > 0]) if values else np.asarray([], dtype=float)
    return urs, mean_vals, min_vals, max_vals, std_vals, positive_values


def _wrap_phase_deg(values: np.ndarray) -> np.ndarray:
    values_arr = np.asarray(values, dtype=float)
    return ((values_arr + 180.0) % 360.0) - 180.0


def _circular_summary(grouped: dict[float, list[float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    urs = np.asarray(sorted(grouped.keys()), dtype=float)
    if urs.size == 0:
        empty = np.asarray([], dtype=float)
        return urs, empty, empty, empty, empty, empty
    mean_vals = []
    min_vals = []
    max_vals = []
    lower_std = []
    upper_std = []
    for ur in urs:
        vals = _wrap_phase_deg(np.asarray(grouped[float(ur)], dtype=float).reshape(-1))
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            mean_vals.append(float("nan"))
            min_vals.append(float("nan"))
            max_vals.append(float("nan"))
            lower_std.append(float("nan"))
            upper_std.append(float("nan"))
            continue
        angles = np.deg2rad(vals)
        resultant = np.mean(np.exp(1j * angles))
        mean_angle = float(np.angle(resultant))
        mean_deg = float(np.rad2deg(mean_angle))
        deltas_deg = np.rad2deg(np.angle(np.exp(1j * (angles - mean_angle))))
        resultant_length = float(np.clip(np.abs(resultant), 1.0e-12, 1.0))
        std_deg = float(np.rad2deg(np.sqrt(max(0.0, -2.0 * np.log(resultant_length)))))
        mean_vals.append(mean_deg)
        min_vals.append(float(np.clip(mean_deg + np.min(deltas_deg), -180.0, 180.0)))
        max_vals.append(float(np.clip(mean_deg + np.max(deltas_deg), -180.0, 180.0)))
        lower_std.append(float(np.clip(mean_deg - std_deg, -180.0, 180.0)))
        upper_std.append(float(np.clip(mean_deg + std_deg, -180.0, 180.0)))
    return (
        urs,
        np.asarray(mean_vals, dtype=float),
        np.asarray(min_vals, dtype=float),
        np.asarray(max_vals, dtype=float),
        np.asarray(lower_std, dtype=float),
        np.asarray(upper_std, dtype=float),
    )


_FIXED_MODEL_STYLES: dict[str, dict[str, Any]] = {
    "Force correction":     {"color": "#0072B2", "marker": "s"},
    "Frequency correction": {"color": "#D55E00", "marker": "o"},
    "Combined correction":  {"color": "#009E73", "marker": "^"},
    "Standalone model":     {"color": "#882255", "marker": "D"},
}

_FALLBACK_PALETTE = [
    "#CC79A7", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan",
]
_FALLBACK_MARKERS = ["D", "P", "h", "H", "*", "X", "v"]


def _model_style_map(labels: Sequence[str]) -> dict[str, dict[str, Any]]:
    styles: dict[str, dict[str, Any]] = {}
    fallback_idx = 0
    for label in labels:
        key = str(label)
        if key in _FIXED_MODEL_STYLES:
            fixed = _FIXED_MODEL_STYLES[key]
            styles[key] = {
                "color": fixed["color"],
                "linewidth": 1.6,
                "alpha": 1.0,
                "marker": fixed["marker"],
                "short_label": key,
            }
        else:
            styles[key] = {
                "color": _FALLBACK_PALETTE[fallback_idx % len(_FALLBACK_PALETTE)],
                "linewidth": 1.25,
                "alpha": 1.0,
                "marker": _FALLBACK_MARKERS[fallback_idx % len(_FALLBACK_MARKERS)],
                "short_label": key,
            }
            fallback_idx += 1
    return styles


def _plot_grouped_scalar(
    ax: Any,
    grouped: dict[float, list[float]],
    *,
    color: str,
    label: str,
    alpha: float,
    size: float,
    marker: str | None,
    linewidth: float = 1.8,
    linestyle: str = "-",
    marker_size: float = 3.2,
    show_scatter: bool = True,
    show_line: bool = True,
    facecolors: str | None = None,
    edgecolors: str | None = None,
    edge_linewidths: float = 1.5,
    highlight_keys: Sequence[float] | None = None,
) -> None:
    urs, means, values = sorted_group_stats(grouped)
    if urs.size == 0:
        return
    scatter_fc = facecolors if facecolors is not None else color
    scatter_ec = edgecolors if edgecolors is not None else color
    if show_scatter:
        for ur, series_values in zip(urs, values):
            ax.scatter(
                np.full(series_values.shape, ur),
                series_values,
                facecolors=scatter_fc,
                edgecolors=scatter_ec,
                linewidths=edge_linewidths,
                alpha=alpha,
                s=size,
                marker=marker or "o",
                zorder=4,
            )
    if show_line:
        # Label lives on the line so the legend shows a line handle.
        plot_kwargs = {"color": color, "linewidth": linewidth, "label": label, "linestyle": linestyle}
        if marker is not None:
            plot_kwargs["marker"] = marker
            plot_kwargs["markersize"] = marker_size
        ax.plot(urs, means, **plot_kwargs)
    else:
        # No line: add a fully-opaque dummy scatter so the legend entry is crisp.
        ax.scatter([], [], facecolors=scatter_fc, edgecolors=scatter_ec, linewidths=edge_linewidths,
                   alpha=1.0, s=size, marker=marker or "o", label=label)
    if highlight_keys is not None:
        highlight_set = {round(float(value), 6) for value in highlight_keys if np.isfinite(float(value))}
        mask = np.asarray([round(float(ur), 6) in highlight_set for ur in urs], dtype=bool)
        if np.any(mask):
            ax.scatter(
                urs[mask],
                means[mask],
                marker=marker or "o",
                s=max(float(size) * 1.6, 24.0),
                facecolors=color,
                edgecolors="black",
                linewidths=1.1,
                zorder=6,
            )


def _series_natural_frequency_hz(series: dict[str, Any]) -> float:
    try:
        effective_mass = float(series["effective_mass"])
        stiffness = float(series["stiffness"])
        natural_frequency_hz = float(np.sqrt(stiffness / effective_mass) / (2.0 * np.pi))
        if np.isfinite(natural_frequency_hz) and natural_frequency_hz > 0.0:
            return natural_frequency_hz
    except Exception:
        pass
    try:
        flow_speed = float(np.median(np.asarray(series["td_context"], dtype=float)[:, 4]))
        ur_effective = float(series["ur_effective"])
        diameter = float(series["diameter"])
        natural_frequency_hz = flow_speed / (ur_effective * diameter)
        if np.isfinite(natural_frequency_hz) and natural_frequency_hz > 0.0:
            return natural_frequency_hz
    except Exception:
        pass
    return float("nan")


def _frequency_ratio_from_signal(time: np.ndarray, values: np.ndarray, natural_frequency_hz: float) -> float:
    dominant_frequency_hz = dominant_frequency_from_signal(time, values)
    if not (np.isfinite(dominant_frequency_hz) and np.isfinite(natural_frequency_hz) and natural_frequency_hz > 0.0):
        return float("nan")
    return float(dominant_frequency_hz / natural_frequency_hz)


def _validation_ur_ticks(case_data: Sequence[dict[str, Any]]) -> np.ndarray:
    values: list[float] = []
    for entry in case_data:
        series = entry.get("validation_series", entry["series"])
        try:
            value = float(round(float(series["ur_effective"]), 6))
        except Exception:
            continue
        if np.isfinite(value):
            values.append(value)
    return np.asarray(sorted(set(values)), dtype=float)


def _save_block10_figure(fig: Any, output_dir: Path, stem: str) -> None:
    _save_section7_figure(fig, output_dir, stem)


def _save_section7_figure(fig: Any, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.pdf", format="pdf", dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)


def _scaled_grouped_values(grouped: dict[float, list[float]], scale: float) -> dict[float, list[float]]:
    return {
        float(ur): [float(scale) * float(value) for value in values if np.isfinite(float(value))]
        for ur, values in grouped.items()
    }


def _add_section7_heldout_marker(ax: Any, *, ur_value: float = 6.46) -> None:
    half_width = 0.175
    ax.axvspan(
        float(ur_value) - half_width,
        float(ur_value) + half_width,
        facecolor="0.4",
        alpha=0.10,
        edgecolor="none",
        zorder=0,
    )


def _block10_legend_columns(label_count: int) -> int:
    return max(1, min(3, int(label_count)))


def _format_plain_log_tick(value: float, _position: int | None = None) -> str:
    value_float = float(value)
    if not np.isfinite(value_float) or value_float <= 0.0:
        return ""
    if value_float >= 1.0 and np.isclose(value_float, round(value_float)):
        return f"{int(round(value_float))}"
    return f"{value_float:g}"


def _set_block10_error_axis_limits(ax: Any, plotted_positive_values: np.ndarray) -> None:
    values = np.asarray(plotted_positive_values, dtype=float).reshape(-1)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return
    ymin = float(np.min(values))
    ymax = float(np.max(values))
    if not np.isfinite(ymin) or not np.isfinite(ymax):
        return
    if ymax <= ymin:
        ymax = ymin * 1.25
    else:
        ymax *= 1.05
    ax.set_ylim(bottom=ymin, top=ymax)
    tick_candidates = np.asarray(
        [
            0.001, 0.003,
            0.01, 0.03,
            0.1, 0.3,
            1.0, 3.0,
            10.0, 30.0,
            100.0, 300.0,
            1000.0,
        ],
        dtype=float,
    )
    ticks = tick_candidates[(tick_candidates >= ymin * 0.999) & (tick_candidates <= ymax * 1.001)]
    if ticks.size > 0:
        ax.yaxis.set_major_locator(mticker.FixedLocator(ticks))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_format_plain_log_tick))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.margins(y=0.0)


def _apply_block10_axes(
    ax: Any,
    *,
    ylabel: str,
    ur_ticks: np.ndarray,
    yscale: str | None = None,
    heldout_ur: float = 6.46,
) -> None:
    ax.set_xlabel(r"Reduced velocity $U_r$")
    ax.set_ylabel(ylabel, rotation=90, labelpad=2, va="center")
    ax.yaxis.set_label_coords(-0.085, 0.5)
    if yscale is not None:
        ax.set_yscale(yscale)
    if ur_ticks.size > 0:
        ax.set_xticks(ur_ticks)
        ax.set_xticklabels([f"{float(value):.3g}" for value in ur_ticks])
    _add_section7_heldout_marker(ax, ur_value=float(heldout_ur))
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.grid(True, which="major", color="0.88", linewidth=0.5, alpha=0.75)
    ax.grid(True, which="minor", color="0.94", linewidth=0.35, alpha=0.45)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_edgecolor("0.65")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=_block10_legend_columns(len(labels)),
            frameon=False,
            handlelength=2.0,
            columnspacing=1.2,
        )


def _plot_peak_summary(
    ax: Any,
    grouped: dict[float, list[float]],
    *,
    color: str,
    label: str,
    marker: str | None,
    alpha_fill: float = 0.10,
    alpha_std: float = 0.22,
    linewidth: float = 2.0,
    linestyle: str = "-",
    show_band: bool = True,
    highlight_keys: Sequence[float] | None = None,
) -> None:
    urs, _, values = sorted_group_stats(grouped)
    if urs.size == 0:
        return
    mean_vals = np.asarray([float(np.mean(v)) for v in values], dtype=float)
    min_vals = np.asarray([float(np.min(v)) for v in values], dtype=float)
    max_vals = np.asarray([float(np.max(v)) for v in values], dtype=float)
    std_vals = np.asarray([float(np.std(v)) for v in values], dtype=float)
    if show_band:
        ax.fill_between(urs, min_vals, max_vals, color=color, alpha=alpha_fill)
        ax.fill_between(urs, mean_vals - std_vals, mean_vals + std_vals, color=color, alpha=alpha_std)
    plot_kwargs = {"color": color, "linewidth": linewidth, "label": label, "linestyle": linestyle}
    if marker is not None:
        plot_kwargs["marker"] = marker
        plot_kwargs["markersize"] = 5
    ax.plot(urs, mean_vals, **plot_kwargs)
    if highlight_keys is not None:
        highlight_set = {round(float(value), 6) for value in highlight_keys if np.isfinite(float(value))}
        mask = np.asarray([round(float(ur), 6) in highlight_set for ur in urs], dtype=bool)
        if np.any(mask):
            ax.scatter(
                urs[mask],
                mean_vals[mask],
                marker=marker or "o",
                s=42.0,
                facecolors=color,
                edgecolors="black",
                linewidths=1.1,
                zorder=6,
            )


def _plot_phase_summary(
    ax: Any,
    grouped: dict[float, list[float]],
    *,
    color: str,
    label: str,
    marker: str | None,
    alpha_fill: float = 0.10,
    alpha_std: float = 0.22,
    linewidth: float = 2.0,
    linestyle: str = "-",
    show_band: bool = True,
    highlight_keys: Sequence[float] | None = None,
) -> None:
    urs, mean_vals, min_vals, max_vals, lower_std, upper_std = _circular_summary(grouped)
    if urs.size == 0:
        return
    mask = np.isfinite(mean_vals) & np.isfinite(min_vals) & np.isfinite(max_vals) & np.isfinite(lower_std) & np.isfinite(upper_std)
    urs = urs[mask]
    mean_vals = mean_vals[mask]
    min_vals = min_vals[mask]
    max_vals = max_vals[mask]
    lower_std = lower_std[mask]
    upper_std = upper_std[mask]
    if urs.size == 0:
        return
    if show_band:
        ax.fill_between(urs, min_vals, max_vals, color=color, alpha=alpha_fill)
        ax.fill_between(urs, lower_std, upper_std, color=color, alpha=alpha_std)
    plot_kwargs = {"color": color, "linewidth": linewidth, "label": label, "linestyle": linestyle}
    if marker is not None:
        plot_kwargs["marker"] = marker
        plot_kwargs["markersize"] = 5
    ax.plot(urs, mean_vals, **plot_kwargs)
    if highlight_keys is not None:
        highlight_set = {round(float(value), 6) for value in highlight_keys if np.isfinite(float(value))}
        mask = np.asarray([round(float(ur), 6) in highlight_set for ur in urs], dtype=bool)
        if np.any(mask):
            ax.scatter(
                urs[mask],
                mean_vals[mask],
                marker=marker or "o",
                s=42.0,
                facecolors=color,
                edgecolors="black",
                linewidths=1.1,
                zorder=6,
            )


def _plot_validation_band(ax: Any, grouped_errors: dict[float, list[float]], *, color: str, label: str, marker: str | None, floor: float, alpha_fill: float = 0.10, alpha_std: float = 0.22, linewidth: float = 2.0, linestyle: str = "-") -> None:
    urs, mean_vals, min_vals, max_vals, std_vals, _positive_values = _group_plot_stats(grouped_errors)
    min_plot = np.maximum(min_vals, floor)
    lower_std = np.maximum(mean_vals - std_vals, floor)
    mean_plot = np.maximum(mean_vals, floor)
    max_plot = np.maximum(max_vals, floor)
    ax.fill_between(urs, min_plot, max_plot, color=color, alpha=alpha_fill)
    ax.fill_between(urs, lower_std, np.maximum(mean_vals + std_vals, floor), color=color, alpha=alpha_std)
    plot_kwargs = {"color": color, "linewidth": linewidth, "label": label, "linestyle": linestyle}
    if marker is not None:
        plot_kwargs["marker"] = marker
        plot_kwargs["markersize"] = 3.2
    ax.plot(urs, mean_plot, **plot_kwargs)


def _compute_reference_truth_rollout(series: dict[str, Any], *, mass_source: str) -> dict[str, np.ndarray]:
    return simulate_structural_response_with_force_rk4(series, series["force_per_m"], mass_source=mass_source)


def _simulate_latent_rnn_rollout(
    *,
    source: LoadedTrainingModel,
    time: np.ndarray,
    initial_displacement: float,
    initial_velocity: float,
    initial_latent: np.ndarray,
    td_context0: np.ndarray,
    ur_value: float,
    mass: float,
    damping: float,
    stiffness: float,
    diameter: float,
    span: float,
    dtype: torch.dtype | None = None,
) -> dict[str, np.ndarray]:
    time_arr = np.asarray(time, dtype=float).reshape(-1)
    if time_arr.size < 2:
        raise ValueError("Need at least two time samples for latent_rnn rollout.")
    h0_arr = np.asarray(initial_latent, dtype=float).reshape(1, -1)
    latent_dim = int(getattr(source.model, "latent_dim", h0_arr.shape[1]))
    if h0_arr.shape[1] != latent_dim:
        raise ValueError(f"latent_rnn initial latent state has dim {h0_arr.shape[1]}, expected {latent_dim}.")
    ctx_init = np.asarray(td_context0, dtype=float).reshape(-1)
    if ctx_init.size < 5:
        raise ValueError("td_context0 must contain at least five entries for latent_rnn rollout.")

    device = _module_device(source.model)
    dtype = _module_dtype(source.model) if dtype is None else dtype
    mass_t = torch.tensor([[float(mass)]], device=device, dtype=dtype)
    damping_t = torch.tensor([[float(damping)]], device=device, dtype=dtype)
    stiffness_t = torch.tensor([[float(stiffness)]], device=device, dtype=dtype)
    ur_t = torch.tensor([[float(ur_value)]], device=device, dtype=dtype)
    flow_t = torch.tensor([[float(ctx_init[4])]], device=device, dtype=dtype)
    z0 = torch.tensor([[float(initial_displacement), float(initial_velocity) * float(mass)]], device=device, dtype=dtype)
    h0 = torch.as_tensor(h0_arr, device=device, dtype=dtype)
    t_seq = torch.as_tensor(time_arr - float(time_arr[0]), device=device, dtype=dtype).view(1, -1)
    with torch.no_grad():
        rollout = source.model.rollout_from_latent(
            h=h0,
            z0=z0,
            t_seq=t_seq,
            ur=ur_t,
            mass=mass_t,
            damping_c=damping_t,
            stiffness=stiffness_t,
            flow_speed=flow_t,
        )

    z_hist = rollout["z"][0].detach().cpu().numpy().astype(float, copy=False)
    h_hist = rollout["h"][0].detach().cpu().numpy().astype(float, copy=False)
    force_steps = rollout["force"][0, :, 0].detach().cpu().numpy().astype(float, copy=False)
    displacement = z_hist[:, 0].copy()
    velocity = z_hist[:, 1].copy() / max(float(mass), 1.0e-12)
    force_per_m = np.empty((time_arr.size,), dtype=float)
    if force_steps.size:
        force_per_m[1:] = force_steps[: time_arr.size - 1]
        force_per_m[0] = 0.0
    else:
        force_per_m.fill(0.0)
    force_total = force_per_m * float(span)
    zero = np.zeros_like(force_per_m)
    acceleration_final = (
        (float(force_per_m[-1]) - float(damping) * float(velocity[-1]) - float(stiffness) * float(displacement[-1]))
        / max(float(mass), 1.0e-12)
    )
    final_ctx = ctx_init[:5].astype(float, copy=True)
    final_ctx[0] = float(acceleration_final)
    return {
        "time": time_arr,
        "displacement": displacement,
        "velocity": velocity,
        "td_force": zero.copy(),
        "td_force_total": zero.copy(),
        "delta_force": force_per_m.copy(),
        "delta_force_total": force_total.copy(),
        "delta_fhat": zero.copy(),
        "fhat_td": zero.copy(),
        "fhat_corr": zero.copy(),
        "force": force_per_m,
        "force_total": force_total,
        "latent": h_hist,
        "final_state": {
            "displacement": float(displacement[-1]),
            "velocity": float(velocity[-1]),
            "force": float(force_per_m[-1]),
            "delta_fhat": 0.0,
            "td_context": final_ctx,
            "latent": h_hist[-1].astype(float, copy=True),
        },
    }


def _reduce_series_for_validation(
    series: dict[str, Any],
    *,
    reduce_time: bool,
    reduction_factor: int,
    cut_start_seconds: float = 0.0,
    td_params: dict[str, float] | None = None,
    td_memory_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not bool(reduce_time):
        return series
    rf = max(1, int(reduction_factor))
    if rf <= 1 and float(cut_start_seconds) <= 0.0:
        return series

    time = np.asarray(series["time"], dtype=float).reshape(-1)
    if time.size < 2:
        return series
    offset = 0
    idx = np.arange(offset, time.size, rf, dtype=int)
    if idx.size < 2:
        raise ValueError(f"Series {series.get('name', '<unknown>')} is too short after validation reduction.")
    if float(cut_start_seconds) > 0.0:
        reduced_time = time[idx]
        mask = reduced_time >= (float(reduced_time[0]) + float(cut_start_seconds))
        idx = idx[mask]
        if idx.size < 2:
            raise ValueError(
                f"Series {series.get('name', '<unknown>')} is too short after validation cut_start_seconds."
            )

    reduced = dict(series)
    time_aligned_keys = (
        "time",
        "displacement",
        "velocity",
        "acceleration",
        "force_total",
        "force_per_m",
        "force_td_stored",
        "td_context",
    )
    for key in time_aligned_keys:
        if key not in reduced:
            continue
        arr = np.asarray(reduced[key])
        if arr.shape[0] == time.size:
            reduced[key] = arr[idx].copy()
    memory_cfg = resolve_td_memory_config(td_memory_cfg)
    if td_params is not None and memory_cfg["mode"] != "fixed_n_memory":
        reduced_time = np.asarray(reduced["time"], dtype=float).reshape(-1)
        reduced_context = np.asarray(reduced["td_context"], dtype=float)
        recomputed_td = _recompute_td_baseline_on_grid(
            t=reduced_time,
            dy=np.asarray(reduced["velocity"], dtype=float).reshape(-1),
            ddy=np.asarray(reduced["acceleration"], dtype=float).reshape(-1),
            flow_speed=reduced_context[:, 4],
            force_td0=float(np.asarray(reduced["force_td_stored"], dtype=float).reshape(-1)[0]),
            phi_td0=float(reduced_context[0, 1]),
            sig_dy_td0=float(reduced_context[0, 2]),
            sig_ddy_td0=float(reduced_context[0, 3]),
            rho=float(reduced["rho"]),
            diameter=float(reduced["diameter"]),
            td_params=td_params,
            td_memory_cfg=memory_cfg,
        )
        reduced["force_td_stored"] = np.asarray(recomputed_td["force_td"], dtype=float)
        reduced["td_context"] = np.stack(
            [
                np.asarray(reduced["acceleration"], dtype=float).reshape(-1),
                np.asarray(recomputed_td["phi_td"], dtype=float).reshape(-1),
                np.asarray(recomputed_td["sig_dy_td"], dtype=float).reshape(-1),
                np.asarray(recomputed_td["sig_ddy_td"], dtype=float).reshape(-1),
                reduced_context[:, 4],
            ],
            axis=1,
        )
    ur_value = reduced.get("ur", None)
    if isinstance(ur_value, np.ndarray) and ur_value.shape[:1] == (time.size,):
        reduced["ur"] = np.asarray(ur_value)[idx].copy()
    reduced["validation_reduction_factor"] = int(rf)
    reduced["validation_reduction_offset"] = int(offset)
    return reduced


def _build_case_rollouts(
    *,
    dataset_root: Path,
    sources: Sequence[LoadedTrainingModel],
    baseline_td_params: dict[str, float],
    td_mass_source: str,
    td_memory_tau_s: float | str | None,
    max_files_per_split: int | None,
    ordered_split_dirs: Sequence[str],
    load_sima_series_for_npz: Callable[[dict[str, Any]], dict[str, Any] | None] | None,
    include_rollouts: bool,
    validation_reduce_time: bool = False,
    validation_reduction_factor: int = 1,
    validation_cut_start_seconds: float = 0.0,
    force_phase_convention: str = "next",
    use_vivana_added_mass_lhs: bool = False,
) -> list[dict[str, Any]]:
    files = iter_all_npz_files(
        dataset_root,
        split_dirs=ordered_split_dirs,
        max_files_per_split=max_files_per_split,
    )
    case_data: list[dict[str, Any]] = []
    progress_desc = "Building case rollouts" if include_rollouts else "Loading case data"
    for npz_path in _progress(files, total=len(files), desc=progress_desc):
        series = load_series(npz_path)
        validation_series = _reduce_series_for_validation(
            series,
            reduce_time=bool(validation_reduce_time),
            reduction_factor=int(validation_reduction_factor),
            cut_start_seconds=float(validation_cut_start_seconds),
            td_params=(sources[0].base_td_params if sources else baseline_td_params),
            td_memory_cfg=(sources[0].td_memory_cfg if sources else None),
        )
        baseline_rollout = None
        model_rollouts = None
        if include_rollouts:
            baseline_rollout = simulate_vivana_td_stepwise(
                series,
                td_params=baseline_td_params,
                mass_source=td_mass_source,
                td_memory_tau_s=td_memory_tau_s,
                force_phase_convention=force_phase_convention,
                use_vivana_added_mass_lhs=use_vivana_added_mass_lhs,
            )
            model_rollouts = {
                source.label: simulate_checkpoint_series_rollout(
                    source,
                    series,
                    mass_source=td_mass_source,
                    td_memory_tau_s=td_memory_tau_s,
                    force_phase_convention=force_phase_convention,
                    use_vivana_added_mass_lhs=use_vivana_added_mass_lhs,
                )
                for source in sources
            }
        sima_series = load_sima_series_for_npz(series) if load_sima_series_for_npz is not None else None
        case_data.append(
            {
                "series": series,
                "validation_series": validation_series,
                "baseline_rollout": baseline_rollout,
                "model_rollouts": model_rollouts,
                "sima_series": sima_series,
            }
        )
    return case_data


def _plot_case_psd_overlays(
    *,
    case_data: list[dict[str, Any]],
    td_mass_source: str,
    figsize: tuple[float, float],
    sources: Sequence[LoadedTrainingModel],
    show_vivana_td_baseline: bool = True,
) -> None:
    if not case_data:
        raise ValueError("No NPZ files found to plot.")
    n_cases = len(case_data)
    ncols = 2 if n_cases > 1 else 1
    nrows = math.ceil(n_cases / ncols)
    model_styles = _model_style_map([source.label for source in sources])

    fig_psd_disp, axes_psd_disp = plt.subplots(nrows, ncols, figsize=(figsize[0], 3.8 * nrows), squeeze=False)
    fig_psd_disp.suptitle(f"All files | Displacement PSD | mass={td_mass_source}", fontsize=14)
    fig_psd_force, axes_psd_force = plt.subplots(nrows, ncols, figsize=(figsize[0], 3.8 * nrows), squeeze=False)
    fig_psd_force.suptitle(f"All files | Force-Per-Meter PSD | mass={td_mass_source}", fontsize=14)
    flat_psd_disp = axes_psd_disp.reshape(-1)
    flat_psd_force = axes_psd_force.reshape(-1)

    for idx, entry in enumerate(case_data):
        series = entry["series"]
        baseline_rollout = entry["baseline_rollout"]
        model_rollouts = entry["model_rollouts"]
        sima_series = entry["sima_series"]
        title = f"{series['name']} | U_r,eff={series['ur_effective']:.4f}"

        freq_disp_true, psd_disp_true = compute_psd_welch(series["time"], series["displacement"])
        freq_disp_td, psd_disp_td = compute_psd_welch(series["time"], baseline_rollout["displacement_td"])
        freq_force_true, psd_force_true = compute_psd_welch(series["time"], series["force_per_m"])
        freq_force_td, psd_force_td = compute_psd_welch(series["time"], baseline_rollout["force_td"])
        disp_pairs = [(freq_disp_true, psd_disp_true), (freq_disp_td, psd_disp_td)]
        force_pairs = [(freq_force_true, psd_force_true), (freq_force_td, psd_force_td)]

        model_psd_disp: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        model_psd_force: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for source in sources:
            rollout = model_rollouts[source.label]
            freq_disp_model, psd_disp_model = compute_psd_welch(series["time"], rollout["displacement"])
            freq_force_model, psd_force_model = compute_psd_welch(series["time"], rollout["force"])
            model_psd_disp[source.label] = (freq_disp_model, psd_disp_model)
            model_psd_force[source.label] = (freq_force_model, psd_force_model)
            disp_pairs.append((freq_disp_model, psd_disp_model))
            force_pairs.append((freq_force_model, psd_force_model))

        freq_disp_sima = psd_disp_sima = freq_force_sima = psd_force_sima = None
        if sima_series is not None:
            freq_disp_sima, psd_disp_sima = compute_psd_welch(sima_series["time"], sima_series["displacement"])
            freq_force_sima, psd_force_sima = compute_psd_welch(sima_series["time"], sima_series["force_per_m"])
            disp_pairs.append((freq_disp_sima, psd_disp_sima))
            force_pairs.append((freq_force_sima, psd_force_sima))

        ax_psd_disp = flat_psd_disp[idx]
        ax_psd_force = flat_psd_force[idx]

        ax_psd_disp.plot(
            freq_disp_true,
            normalize_psd_area(freq_disp_true, psd_disp_true),
            label="CFD reference",
            color="black",
            linewidth=2.3,
        )
        ax_psd_force.plot(
            freq_force_true,
            normalize_psd_area(freq_force_true, psd_force_true),
            label="CFD reference",
            color="black",
            linewidth=2.3,
        )
        if bool(show_vivana_td_baseline):
            ax_psd_disp.plot(
                freq_disp_td,
                normalize_psd_area(freq_disp_td, psd_disp_td),
                label="VIVANA-TD baseline",
                color="0.35",
                linestyle="--",
                linewidth=1.6,
            )
            ax_psd_force.plot(
                freq_force_td,
                normalize_psd_area(freq_force_td, psd_force_td),
                label="VIVANA-TD baseline",
                color="0.35",
                linestyle="--",
                linewidth=1.6,
            )

        for source in sources:
            style = model_styles[source.label]
            freq_disp_model, psd_disp_model = model_psd_disp[source.label]
            freq_force_model, psd_force_model = model_psd_force[source.label]
            ax_psd_disp.plot(
                freq_disp_model,
                normalize_psd_area(freq_disp_model, psd_disp_model),
                label=source.label,
                color=style["color"],
                linewidth=style["linewidth"],
            )
            ax_psd_force.plot(
                freq_force_model,
                normalize_psd_area(freq_force_model, psd_force_model),
                label=source.label,
                color=style["color"],
                linewidth=style["linewidth"],
            )

        if sima_series is not None:
            ax_psd_disp.plot(
                freq_disp_sima,
                normalize_psd_area(freq_disp_sima, psd_disp_sima),
                label="SIMA",
                color="tab:green",
                linewidth=1.2,
            )
            ax_psd_force.plot(
                freq_force_sima,
                normalize_psd_area(freq_force_sima, psd_force_sima),
                label="SIMA",
                color="tab:green",
                linewidth=1.2,
            )

        xlim_disp = suggest_psd_xlim(*disp_pairs)
        xlim_force = suggest_psd_xlim(*force_pairs)

        ax_psd_disp.set_title(title)
        ax_psd_disp.set_ylabel("Area-normalized PSD")
        ax_psd_disp.set_xlabel("Frequency [Hz]")
        ax_psd_disp.grid(True, which="major", color="0.88", linewidth=0.5, alpha=0.75)
        ax_psd_disp.grid(True, which="minor", color="0.94", linewidth=0.35, alpha=0.45)
        for spine in ax_psd_disp.spines.values():
            spine.set_linewidth(0.6)
            spine.set_edgecolor("0.65")
        ax_psd_disp.legend(loc="best", frameon=False)
        ax_psd_disp.set_xlim(0.0, xlim_disp)

        ax_psd_force.set_title(title)
        ax_psd_force.set_ylabel("Area-normalized PSD")
        ax_psd_force.set_xlabel("Frequency [Hz]")
        ax_psd_force.grid(True, which="major", color="0.88", linewidth=0.5, alpha=0.75)
        ax_psd_force.grid(True, which="minor", color="0.94", linewidth=0.35, alpha=0.45)
        for spine in ax_psd_force.spines.values():
            spine.set_linewidth(0.6)
            spine.set_edgecolor("0.65")
        ax_psd_force.legend(loc="best", frameon=False)
        ax_psd_force.set_xlim(0.0, xlim_force)

    for extra_ax in flat_psd_disp[n_cases:]:
        extra_ax.axis("off")
    for extra_ax in flat_psd_force[n_cases:]:
        extra_ax.axis("off")

    fig_psd_disp.tight_layout()
    fig_psd_disp.subplots_adjust(top=0.92)
    plt.show()
    fig_psd_force.tight_layout()
    fig_psd_force.subplots_adjust(top=0.92)
    plt.show()


def _append_summary_metrics(
    grouped: dict[str, dict[float, list[float]]],
    *,
    key: float,
    metrics: dict[str, Any],
    diameter: float | None = None,
    dyn_pressure_D: float | None = None,
) -> None:
    grouped["disp_std"].setdefault(key, []).append(metrics["disp_std"])
    grouped["force_std"].setdefault(key, []).append(metrics["force_std"])
    if diameter is not None and np.isfinite(diameter) and diameter > 0.0:
        grouped["disp_std_over_D"].setdefault(key, []).append(metrics["disp_std"] / diameter)
    if dyn_pressure_D is not None and np.isfinite(dyn_pressure_D) and dyn_pressure_D > 0.0:
        grouped["cf_std"].setdefault(key, []).append(metrics["force_std"] / dyn_pressure_D)
    grouped["dominant_frequency_ratio"].setdefault(key, []).append(metrics["dominant_frequency_ratio"])
    grouped["force_dominant_frequency_ratio"].setdefault(key, []).append(metrics["force_dominant_frequency_ratio"])
    grouped["peak_amp"].setdefault(key, []).extend(np.asarray(metrics["peak_amplitudes"], dtype=float).tolist())

    phase_force_disp_samples = np.asarray(metrics["phase_force_displacement_deg_samples"], dtype=float)
    phase_force_vel_samples = np.asarray(metrics["phase_force_velocity_deg_samples"], dtype=float)
    if phase_force_disp_samples.size:
        grouped["phase_force_disp"].setdefault(key, []).extend(phase_force_disp_samples.tolist())
    if phase_force_vel_samples.size:
        grouped["phase_force_vel"].setdefault(key, []).extend(phase_force_vel_samples.tolist())


def _empty_summary_grouped() -> dict[str, dict[float, list[float]]]:
    return {
        "disp_std": {},
        "disp_std_over_D": {},
        "force_std": {},
        "cf_std": {},
        "dominant_frequency_ratio": {},
        "force_dominant_frequency_ratio": {},
        "peak_amp": {},
        "phase_force_disp": {},
        "phase_force_vel": {},
    }


def _normalize_summary_generation_dt_specs(
    *,
    summary_generation_dt: float | None,
    summary_generation_dt_specs: Sequence[float | None | dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    default_linestyles = ["-", "--", ":", "-."]
    raw_specs: Sequence[float | None | dict[str, Any]]
    if summary_generation_dt_specs:
        raw_specs = summary_generation_dt_specs
    else:
        raw_specs = [summary_generation_dt]

    specs: list[dict[str, Any]] = []
    for idx, raw_spec in enumerate(raw_specs):
        if isinstance(raw_spec, dict):
            dt_value = raw_spec.get("dt")
            label = str(raw_spec.get("label", "auto dt" if dt_value is None else f"dt={float(dt_value):.6g}"))
            linestyle = str(raw_spec.get("linestyle", default_linestyles[idx % len(default_linestyles)]))
        else:
            dt_value = raw_spec
            label = "auto dt" if dt_value is None else f"dt={float(dt_value):.6g}"
            linestyle = default_linestyles[idx % len(default_linestyles)]
        specs.append(
            {
                "key": f"variant_{idx}",
                "dt": None if dt_value is None else float(dt_value),
                "label": label,
                "linestyle": linestyle,
            }
        )
    return specs


def _plot_td_frequency_ratio_guides(
    ax: Any,
    *,
    ur_min: float,
    ur_max: float,
    td_params: dict[str, float],
) -> None:
    if not np.isfinite(ur_min) or not np.isfinite(ur_max) or ur_max <= ur_min:
        return
    guide_specs = [
        ("fhat_min", "Vivana-TD f_min U_r", "0.55", ":"),
        ("fhat0", "Vivana-TD f_0 U_r", "0.35", "--"),
        ("fhat_max", "Vivana-TD f_max U_r", "0.15", "-."),
    ]
    ur_values = np.linspace(float(ur_min), float(ur_max), 256, dtype=float)
    for key, label, color, linestyle in guide_specs:
        coeff = float(td_params.get(key, float("nan")))
        if not np.isfinite(coeff):
            continue
        ax.plot(
            ur_values,
            coeff * ur_values,
            color=color,
            linestyle=linestyle,
            linewidth=1.1,
            alpha=0.9,
            label=label,
        )


def _plot_summary_metrics(
    *,
    cfd_grouped: dict[str, dict[float, list[float]]],
    sima_grouped: dict[str, dict[float, list[float]]] | None,
    summary_variants: Sequence[dict[str, Any]],
    fine_ur_step: float,
    summary_mass_source: str,
    sources: Sequence[LoadedTrainingModel],
    baseline_td_params: dict[str, float],
    show_td_frequency_guides: bool = True,
    show_vivana_td_baseline: bool = True,
    output_dir: Path | None = None,
) -> None:
    model_labels: list[str] = [source.label for source in sources]
    for variant in summary_variants:
        for label in (variant.get("model_grouped") or {}).keys():
            if label not in model_labels:
                model_labels.append(str(label))
    model_styles = _model_style_map(model_labels)
    base_model_styles = _model_style_map([source.label for source in sources])
    for label in model_labels:
        label_str = str(label)
        for suffix in (" low to high", " high to low"):
            if label_str.endswith(suffix):
                base_label = label_str[: -len(suffix)]
                if base_label in base_model_styles:
                    model_styles[label_str] = dict(base_model_styles[base_label])
                    model_styles[label_str]["marker"] = "o" if suffix == " low to high" else "s"
                    model_styles[label_str]["direction_linestyle"] = ":" if suffix == " low to high" else "--"
                break

    _scalar_specs = [
        (r"$\sigma_{y/D}$",        "disp_std_over_D"),
        (r"$\omega_y/\omega_n$",   "dominant_frequency_ratio"),
        (r"$\sigma_{C_F}$",        "cf_std"),
        (r"$\omega_F/\omega_n$",   "force_dominant_frequency_ratio"),
    ]
    fig_s9, axes_s9 = plt.subplots(len(_scalar_specs), 1, figsize=(5.85, 6.4), sharex=True)
    axes_s9 = list(np.atleast_1d(axes_s9))
    for _s9_idx, (_s9_ax, (_s9_ylabel, _s9_key)) in enumerate(zip(axes_s9, _scalar_specs)):
        _plot_grouped_scalar(
            _s9_ax,
            cfd_grouped[_s9_key],
            color="black",
            label="CFD reference",
            alpha=1.0,
            size=95,
            marker="o",
            linewidth=2.5,
            linestyle="-",
            show_scatter=True,
            show_line=False,
            facecolors="none",
            edgecolors="black",
            edge_linewidths=1.8,
        )
        if sima_grouped is not None and sima_grouped[_s9_key]:
            _plot_grouped_scalar(
                _s9_ax,
                sima_grouped[_s9_key],
                color="#CC79A7",
                label="SIMA",
                alpha=0.45,
                size=0,
                marker=None,
                linewidth=1.0,
                show_scatter=False,
            )
        for _v_idx, _variant in enumerate(summary_variants):
            if bool(show_vivana_td_baseline):
                _plot_grouped_scalar(
                    _s9_ax, _variant["baseline_grouped"][_s9_key],
                    color="0.35", label="VIVANA-TD baseline" if _v_idx == 0 else "_nolegend_",
                    alpha=1.0, size=12, marker=None,
                    linewidth=1.8 if _v_idx == 0 else 1.6, linestyle="--", show_scatter=False,
                )
            for _ml in model_labels:
                _st = model_styles[_ml]
                _g = (_variant["model_grouped"].get(_ml) or {}).get(_s9_key, {})
                if not _g:
                    continue
                _plot_grouped_scalar(
                    _s9_ax, _g,
                    color=_st["color"], label=_ml if _v_idx == 0 else "_nolegend_",
                    alpha=1.0, size=0, marker=None, linewidth=1.6,
                    linestyle=str(_st.get("direction_linestyle", "-")),
                    show_scatter=False,
                )
        _s9_ax.set_ylabel(_s9_ylabel, rotation=90, labelpad=2, va="center")
        _s9_ax.yaxis.set_label_coords(-0.085, 0.5)
        if _s9_key == "cf_std":
            _s9_ax.set_ylim(0.0, 2.5)
        _add_section7_heldout_marker(_s9_ax)
        _s9_ax.grid(True, which="major", color="0.88", linewidth=0.5, alpha=0.75)
        _s9_ax.grid(True, which="minor", color="0.94", linewidth=0.35, alpha=0.45)
        for _sp in _s9_ax.spines.values():
            _sp.set_linewidth(0.6)
            _sp.set_edgecolor("0.65")
        if _s9_idx == 0:
            _h, _l = _s9_ax.get_legend_handles_labels()
            if _h:
                _s9_ax.legend(
                    _h,
                    _l,
                    loc="lower center",
                    bbox_to_anchor=(0.5, 1.02),
                    ncol=_block10_legend_columns(len(_l)),
                    frameon=False,
                    handlelength=2.0,
                    columnspacing=1.2,
                )
        if _s9_idx == len(axes_s9) - 1:
            _s9_ax.set_xlabel(r"Reduced velocity $U_r$")
    # Tighten x range and add boundary ticks from actual data.
    _s9_x: list[float] = []
    for _ln in axes_s9[0].get_lines():
        _xd = np.asarray(_ln.get_xdata(), dtype=float)
        _s9_x.extend(float(v) for v in _xd if np.isfinite(v))
    for _col in axes_s9[0].collections:
        try:
            _off = np.asarray(_col.get_offsets(), dtype=float)
            _s9_x.extend(float(v) for v in _off[:, 0] if np.isfinite(v))
        except Exception:
            pass
    if _s9_x:
        _xlo, _xhi = float(np.min(_s9_x)), float(np.max(_s9_x))
        _pad = 0.25
        _base = np.arange(2.0, _xhi + 0.01, 2.0)
        _xticks = np.unique(np.round(np.concatenate([[_xlo], _base, [_xhi]]), 6))
        for _ax in axes_s9:
            _ax.set_xlim(_xlo - _pad, _xhi + _pad)
            _ax.set_xticks(_xticks)
            _ax.set_xticklabels([f"{v:.3g}" for v in _xticks])
    plt.tight_layout()
    fig_s9.subplots_adjust(hspace=0.1)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _save_section7_figure(fig_s9, output_dir, "block9_scalar_metrics")
    plt.show()


def _build_summary_metrics(
    *,
    case_data: list[dict[str, Any]],
    sources: Sequence[LoadedTrainingModel],
    baseline_td_params: dict[str, float],
    fine_ur_step: float,
    summary_mass_source: str,
    transient_seconds: float,
    summary_generation_dt: float | None,
    summary_generation_duration_s: float | None,
    summary_kept_duration_s: float | None,
    summary_ur_range: tuple[float, float] | None,
    summary_first_case_extra_duration_s: float = 0.0,
    summary_generation_dt_from_reduction_factor: bool = False,
    summary_generation_dt_reduction_factor: int = 1,
    td_memory_tau_s: float | str | None = None,
    collect_generated_rollouts: bool = False,
    use_perturbed_start: bool = False,
    perturbation_fraction: float = 0.1,
    steady_state_n_cycles: int = 5,
    steady_state_amp_rel_tol: float = 0.05,
    steady_state_max_seconds: float = 800.0,
    stochastic_rollout: bool = False,
    force_phase_convention: str = "next",
    use_vivana_added_mass_lhs: bool = False,
) -> tuple[
    dict[str, dict[float, list[float]]],
    dict[str, dict[float, list[float]]],
    dict[str, dict[float, list[float]]] | None,
    dict[str, dict[str, dict[float, list[float]]]],
    float,
    float,
    np.ndarray,
    dict[float, float],
    dict[float, dict[str, Any]] | None,
    dict[str, dict[float, float]],
]:
    series_list = [entry["series"] for entry in case_data if np.isfinite(float(entry["series"]["ur_effective"]))]
    if not series_list:
        raise ValueError("No finite U_r values found, skipping summary-vs-U_r plots.")

    cfd_grouped = _empty_summary_grouped()
    sima_grouped = _empty_summary_grouped()
    any_sima = False
    for entry in case_data:
        series = entry["series"]
        if not np.isfinite(float(series["ur_effective"])):
            continue
        key = float(round(float(series["ur_effective"]), 6))
        metrics = compute_summary_metrics(
            series["time"],
            series["displacement"],
            series["velocity"],
            series["force_per_m"],
            stiffness=float(series["stiffness"]),
            effective_mass=float(series["effective_mass"]),
        )
        _cfd_D = float(series["diameter"])
        _cfd_U = float(np.median(np.asarray(series["td_context"], dtype=float)[:, 4]))
        _cfd_qD = 0.5 * float(series["rho"]) * _cfd_U ** 2 * _cfd_D
        _append_summary_metrics(cfd_grouped, key=key, metrics=metrics, diameter=_cfd_D, dyn_pressure_D=_cfd_qD)

        sima_series = entry["sima_series"]
        if sima_series is not None:
            any_sima = True
            sima_metrics = compute_summary_metrics(
                sima_series["time"],
                sima_series["displacement"],
                sima_series["velocity"],
                sima_series["force_per_m"],
                stiffness=float(series["stiffness"]),
                effective_mass=float(series["effective_mass"]),
            )
            _append_summary_metrics(sima_grouped, key=key, metrics=sima_metrics, diameter=_cfd_D, dyn_pressure_D=_cfd_qD)

    use_reduction_factor_dt = bool(summary_generation_dt_from_reduction_factor)
    reduction_factor_for_dt = int(summary_generation_dt_reduction_factor)
    if reduction_factor_for_dt <= 0:
        raise ValueError("summary_generation_dt_reduction_factor must be positive.")

    if summary_generation_dt is None or summary_generation_duration_s is None:
        grid_dt, grid_duration, _time_full, _keep_mask = global_generation_grid(
            series_list,
            transient_seconds=transient_seconds,
            kept_duration_s=summary_kept_duration_s,
        )
        if summary_generation_dt is None and not use_reduction_factor_dt:
            summary_generation_dt = grid_dt
        if summary_generation_duration_s is None:
            summary_generation_duration_s = grid_duration

    target_urs, exact_dataset_urs = _summary_target_urs(
        series_list,
        fine_ur_step=float(fine_ur_step),
        summary_ur_range=summary_ur_range,
    )
    first_case_extra_duration_s = float(summary_first_case_extra_duration_s)
    if not np.isfinite(first_case_extra_duration_s) or first_case_extra_duration_s < 0.0:
        raise ValueError("summary_first_case_extra_duration_s must be finite and non-negative.")

    baseline_grouped = _empty_summary_grouped()
    model_grouped = {source.label: _empty_summary_grouped() for source in sources}
    generated_rollouts: dict[float, dict[str, Any]] | None = {} if collect_generated_rollouts else None
    baseline_continuation_state: dict[str, Any] | None = None
    model_continuation_states: dict[str, dict[str, Any] | None] = {source.label: None for source in sources}
    generation_dt_by_ur: dict[float, float] = {}
    convergence_time_by_source: dict[str, dict[float, float]] = {
        "Vivana-TD baseline": {},
        **{source.label: {} for source in sources},
    }
    for target_idx, target_ur in enumerate(_progress(target_urs, total=int(target_urs.size), desc="Sweeping reduced velocity")):
        # Use a bracketed interpolation between neighboring CFD anchors instead
        # of a nearest-template switch, which creates visible kinks in the sweep.
        template = _build_summary_sweep_template(series_list, float(target_ur))
        if use_reduction_factor_dt:
            generation_dt_for_target = float(template["raw_dt"]) * float(reduction_factor_for_dt)
        else:
            generation_dt_for_target = float(summary_generation_dt)
        if not np.isfinite(generation_dt_for_target) or generation_dt_for_target <= 0.0:
            raise ValueError(f"Invalid summary generation dt at U_r={float(target_ur):.6g}: {generation_dt_for_target}")
        key = float(round(float(target_ur), 6))
        generation_dt_by_ur[key] = float(generation_dt_for_target)

        if use_perturbed_start:
            baseline_init = _perturbed_initial_state(template, float(perturbation_fraction))
            baseline_rollout = generate_vivana_summary_rollout(
                template,
                float(target_ur),
                generation_dt=float(generation_dt_for_target),
                generation_duration_s=float(steady_state_max_seconds),
                transient_seconds=0.0,
                td_params=baseline_td_params,
                mass_source=summary_mass_source,
                initial_state=baseline_init,
                td_memory_tau_s=td_memory_tau_s,
            )
            fn_hz = math.sqrt(float(baseline_rollout["stiffness"]) / float(baseline_rollout["effective_mass"])) / (2.0 * math.pi)
            period_s = 1.0 / max(fn_hz, 1.0e-6)
            onset_idx = _find_steady_state_onset(
                baseline_rollout["displacement"],
                float(generation_dt_for_target),
                period_s,
                n_cycles=int(steady_state_n_cycles),
                amp_rel_tol=float(steady_state_amp_rel_tol),
            )
            convergence_time_by_source["Vivana-TD baseline"][key] = float(onset_idx) * float(generation_dt_for_target)
            baseline_rollout = _trim_rollout_to_onset(baseline_rollout, onset_idx, summary_kept_duration_s)
        else:
            extra_duration = first_case_extra_duration_s if target_idx == 0 else 0.0
            rollout_duration_s = float(summary_generation_duration_s) + extra_duration
            rollout_transient_seconds = float(transient_seconds) + extra_duration
            baseline_rollout = generate_vivana_summary_rollout(
                template,
                float(target_ur),
                generation_dt=float(generation_dt_for_target),
                generation_duration_s=rollout_duration_s,
                transient_seconds=rollout_transient_seconds,
                td_params=baseline_td_params,
                mass_source=summary_mass_source,
                initial_state=baseline_continuation_state,
                td_memory_tau_s=td_memory_tau_s,
            )
            baseline_continuation_state = dict(baseline_rollout["final_state"])

        baseline_metrics = compute_summary_metrics(
            baseline_rollout["time"],
            baseline_rollout["displacement"],
            baseline_rollout["velocity"],
            baseline_rollout["force"],
            stiffness=float(baseline_rollout["stiffness"]),
            effective_mass=float(baseline_rollout["effective_mass"]),
        )
        _tmpl_D = float(template["diameter"])
        _tmpl_U = float(np.asarray(template["td_context"], dtype=float).reshape(-1, 5)[0, 4])
        _tmpl_qD = 0.5 * float(template["rho"]) * _tmpl_U ** 2 * _tmpl_D
        _append_summary_metrics(baseline_grouped, key=key, metrics=baseline_metrics, diameter=_tmpl_D, dyn_pressure_D=_tmpl_qD)
        if generated_rollouts is not None:
            generated_rollouts[key] = {
                "baseline": baseline_rollout,
                "models": {},
            }

        for source in sources:
            if use_perturbed_start:
                model_init = _perturbed_initial_state(template, float(perturbation_fraction))
                if source.kind == "latent_rnn":
                    previous_state = model_continuation_states.get(source.label)
                    if previous_state is not None and "latent" in previous_state:
                        model_init["latent"] = np.asarray(previous_state["latent"], dtype=float).reshape(-1).copy()
                    else:
                        model_init["latent"] = latent_reference_initial_state(
                            source,
                            series_list,
                            float(target_ur),
                            mass_source=summary_mass_source,
                        )
                model_rollout = generate_checkpoint_summary_rollout(
                    source,
                    template,
                    float(target_ur),
                    generation_dt=float(generation_dt_for_target),
                    generation_duration_s=float(steady_state_max_seconds),
                    transient_seconds=0.0,
                    mass_source=summary_mass_source,
                    initial_state=model_init,
                    td_memory_tau_s=td_memory_tau_s,
                    stochastic=stochastic_rollout,
                    force_phase_convention=force_phase_convention,
                    use_vivana_added_mass_lhs=use_vivana_added_mass_lhs,
                )
                fn_hz = math.sqrt(float(model_rollout["stiffness"]) / float(model_rollout["effective_mass"])) / (2.0 * math.pi)
                period_s = 1.0 / max(fn_hz, 1.0e-6)
                onset_idx = _find_steady_state_onset(
                    model_rollout["displacement"],
                    float(generation_dt_for_target),
                    period_s,
                    n_cycles=int(steady_state_n_cycles),
                    amp_rel_tol=float(steady_state_amp_rel_tol),
                )
                convergence_time_by_source[source.label][key] = float(onset_idx) * float(generation_dt_for_target)
                model_rollout = _trim_rollout_to_onset(model_rollout, onset_idx, summary_kept_duration_s)
                if source.kind == "latent_rnn":
                    model_continuation_states[source.label] = dict(model_rollout["final_state"])
            else:
                extra_duration = first_case_extra_duration_s if target_idx == 0 else 0.0
                rollout_duration_s = float(summary_generation_duration_s) + extra_duration
                rollout_transient_seconds = float(transient_seconds) + extra_duration
                model_initial_state = model_continuation_states[source.label]
                if source.kind == "latent_rnn" and model_initial_state is None:
                    model_initial_state = {
                        "displacement": float(np.asarray(template["displacement"], dtype=float).reshape(-1)[0]),
                        "velocity": float(np.asarray(template["velocity"], dtype=float).reshape(-1)[0]),
                        "td_context": np.asarray(template["td_context"], dtype=float).reshape(-1)[:5].copy(),
                        "latent": latent_reference_initial_state(
                            source,
                            series_list,
                            float(target_ur),
                            mass_source=summary_mass_source,
                        ),
                    }
                model_rollout = generate_checkpoint_summary_rollout(
                    source,
                    template,
                    float(target_ur),
                    generation_dt=float(generation_dt_for_target),
                    generation_duration_s=rollout_duration_s,
                    transient_seconds=rollout_transient_seconds,
                    mass_source=summary_mass_source,
                    initial_state=model_initial_state,
                    td_memory_tau_s=td_memory_tau_s,
                    stochastic=stochastic_rollout,
                    force_phase_convention=force_phase_convention,
                    use_vivana_added_mass_lhs=use_vivana_added_mass_lhs,
                )
                model_continuation_states[source.label] = dict(model_rollout["final_state"])

            model_metrics = compute_summary_metrics(
                model_rollout["time"],
                model_rollout["displacement"],
                model_rollout["velocity"],
                model_rollout["force"],
                stiffness=float(model_rollout["stiffness"]),
                effective_mass=float(model_rollout["effective_mass"]),
            )
            _append_summary_metrics(model_grouped[source.label], key=key, metrics=model_metrics, diameter=_tmpl_D, dyn_pressure_D=_tmpl_qD)
            if generated_rollouts is not None:
                generated_rollouts[key]["models"][source.label] = model_rollout

    return (
        cfd_grouped,
        baseline_grouped,
        (sima_grouped if any_sima else None),
        model_grouped,
        float(min(generation_dt_by_ur.values()) if generation_dt_by_ur else summary_generation_dt),
        float(summary_generation_duration_s),
        exact_dataset_urs,
        generation_dt_by_ur,
        generated_rollouts,
        convergence_time_by_source,
    )


def _build_hysteresis_summary_metrics(
    *,
    case_data: list[dict[str, Any]],
    source: LoadedTrainingModel,
    baseline_td_params: dict[str, float],
    target_urs: np.ndarray,
    summary_mass_source: str,
    summary_generation_dt: float | None,
    summary_generation_dt_from_reduction_factor: bool,
    summary_generation_dt_reduction_factor: int,
    td_memory_tau_s: float | str | None,
    include_vivana: bool,
    step_transient_seconds: float,
    kept_duration_s: float,
    perturbation_fraction: float,
    steady_state_n_cycles: int,
    steady_state_amp_rel_tol: float,
    steady_state_max_seconds: float,
    stochastic_rollout: bool,
    force_phase_convention: str = "next",
    use_vivana_added_mass_lhs: bool = False,
) -> tuple[dict[str, dict[str, dict[float, list[float]]]], dict[str, Any]]:
    series_list = [entry["series"] for entry in case_data if np.isfinite(float(entry["series"]["ur_effective"]))]
    if not series_list:
        raise ValueError("No finite U_r series are available for the hysteresis sweep.")
    target_urs = np.asarray(target_urs, dtype=float).reshape(-1)
    if target_urs.size == 0:
        raise ValueError("No U_r points are available for the hysteresis sweep.")

    use_reduction_factor_dt = bool(summary_generation_dt_from_reduction_factor)
    reduction_factor_for_dt = int(summary_generation_dt_reduction_factor)
    if reduction_factor_for_dt <= 0:
        raise ValueError("summary_generation_dt_reduction_factor must be positive.")
    if not use_reduction_factor_dt and summary_generation_dt is None:
        summary_generation_dt, _, _, _ = global_generation_grid(
            series_list,
            transient_seconds=float(step_transient_seconds),
            kept_duration_s=float(kept_duration_s),
        )

    def _dt_for_template(template: dict[str, Any]) -> float:
        if use_reduction_factor_dt:
            return float(template["raw_dt"]) * float(reduction_factor_for_dt)
        return float(summary_generation_dt)

    def _generate(
        method: str,
        template: dict[str, Any],
        target_ur: float,
        generation_dt: float,
        generation_duration_s: float,
        transient_seconds: float,
        initial_state: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if method == "vivana_td":
            return generate_vivana_summary_rollout(
                template,
                float(target_ur),
                generation_dt=float(generation_dt),
                generation_duration_s=float(generation_duration_s),
                transient_seconds=float(transient_seconds),
                td_params=dict(baseline_td_params),
                mass_source=summary_mass_source,
                initial_state=initial_state,
                td_memory_tau_s=td_memory_tau_s,
            )
        if method == "trained_model":
            return generate_checkpoint_summary_rollout(
                source,
                template,
                float(target_ur),
                generation_dt=float(generation_dt),
                generation_duration_s=float(generation_duration_s),
                transient_seconds=float(transient_seconds),
                mass_source=summary_mass_source,
                initial_state=initial_state,
                td_memory_tau_s=td_memory_tau_s,
                stochastic=bool(stochastic_rollout),
                force_phase_convention=force_phase_convention,
                use_vivana_added_mass_lhs=use_vivana_added_mass_lhs,
            )
        raise ValueError(f"Unknown hysteresis method: {method!r}")

    def _settled_endpoint(
        method: str,
        template: dict[str, Any],
        target_ur: float,
        generation_dt: float,
    ) -> tuple[dict[str, Any], float]:
        initial_state = _perturbed_initial_state(template, float(perturbation_fraction))
        if method == "trained_model" and source.kind == "latent_rnn":
            initial_state["latent"] = latent_reference_initial_state(
                source,
                series_list,
                float(target_ur),
                mass_source=summary_mass_source,
            )
        rollout = _generate(
            method,
            template,
            float(target_ur),
            float(generation_dt),
            float(steady_state_max_seconds),
            0.0,
            initial_state,
        )
        fn_hz = math.sqrt(float(rollout["stiffness"]) / float(rollout["effective_mass"])) / (2.0 * math.pi)
        period_s = 1.0 / max(fn_hz, 1.0e-6)
        onset_idx = _find_steady_state_onset(
            rollout["displacement"],
            float(generation_dt),
            period_s,
            n_cycles=int(steady_state_n_cycles),
            amp_rel_tol=float(steady_state_amp_rel_tol),
        )
        return _trim_rollout_to_onset(rollout, onset_idx, kept_duration_s), float(onset_idx) * float(generation_dt)

    def _append_rollout(grouped: dict[str, dict[float, list[float]]], key: float, rollout: dict[str, Any]) -> None:
        metrics = compute_summary_metrics(
            rollout["time"],
            rollout["displacement"],
            rollout["velocity"],
            rollout["force"],
            stiffness=float(rollout["stiffness"]),
            effective_mass=float(rollout["effective_mass"]),
        )
        _append_summary_metrics(grouped, key=key, metrics=metrics)

    methods = ["trained_model"]
    if include_vivana:
        methods.append("vivana_td")

    grouped_by_label: dict[str, dict[str, dict[float, list[float]]]] = {}
    details: dict[str, Any] = {
        "source_label": source.label,
        "target_urs": target_urs,
        "endpoint_settle_s": {},
        "step_transient_seconds": float(step_transient_seconds),
        "kept_duration_s": float(kept_duration_s),
    }
    direction_specs = [
        ("low to high", "low_to_high", target_urs),
        ("high to low", "high_to_low", target_urs[::-1]),
    ]
    for method in methods:
        base_label = source.label if method == "trained_model" else "Vivana-TD"
        for direction_label, direction_key, ur_order in direction_specs:
            plot_label = f"{base_label} {direction_label}"
            grouped = _empty_summary_grouped()
            continuation_state: dict[str, Any] | None = None
            endpoint_times: dict[float, float] = {}
            step_duration_s = float(step_transient_seconds) + float(kept_duration_s)
            for idx, target_ur in enumerate(_progress(ur_order, total=len(ur_order), desc=f"Hysteresis {plot_label}")):
                target_ur_value = float(target_ur)
                key = float(round(target_ur_value, 6))
                template = _build_summary_sweep_template(series_list, target_ur_value)
                generation_dt = _dt_for_template(template)
                if not np.isfinite(generation_dt) or generation_dt <= 0.0:
                    raise ValueError(f"Invalid hysteresis dt at U_r={target_ur_value:.6g}: {generation_dt}")
                if idx == 0:
                    rollout, endpoint_settle_s = _settled_endpoint(method, template, target_ur_value, generation_dt)
                    endpoint_times[key] = endpoint_settle_s
                else:
                    rollout = _generate(
                        method,
                        template,
                        target_ur_value,
                        generation_dt,
                        step_duration_s,
                        float(step_transient_seconds),
                        continuation_state,
                    )
                continuation_state = dict(rollout["final_state"])
                _append_rollout(grouped, key, rollout)
            grouped_by_label[plot_label] = grouped
            details["endpoint_settle_s"][plot_label] = endpoint_times
    return grouped_by_label, details


def _display_exact_ur_sweep_diagnostic(
    *,
    case_data: list[dict[str, Any]],
    summary_variants: Sequence[dict[str, Any]],
    sources: Sequence[LoadedTrainingModel],
    baseline_td_params: dict[str, float],
    td_mass_source: str,
    summary_mass_source: str,
    transient_seconds: float,
    td_memory_tau_s: float | str | None,
    force_phase_convention: str = "next",
    use_vivana_added_mass_lhs: bool = False,
) -> list[dict[str, Any]]:
    series_list = [entry["series"] for entry in case_data if np.isfinite(float(entry["series"]["ur_effective"]))]
    if not series_list or not summary_variants:
        return []

    rows: list[dict[str, Any]] = []

    def _safe_ratio(numerator: float, denominator: float) -> float:
        if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(float(denominator)) <= 1.0e-12:
            return float("nan")
        return float(numerator) / float(denominator)

    def _sweep_std(
        *,
        variant: dict[str, Any],
        method: str,
        ur_key: float,
    ) -> float:
        if method == "Vivana-TD baseline":
            values = variant["baseline_grouped"]["disp_std"].get(ur_key, [])
        else:
            grouped_by_method = variant.get("standard_model_grouped", variant.get("model_grouped", {}))
            values = grouped_by_method.get(method, {}).get("disp_std", {}).get(ur_key, [])
        finite_values = np.asarray(values, dtype=float)
        finite_values = finite_values[np.isfinite(finite_values)]
        return float(np.mean(finite_values)) if finite_values.size else float("nan")

    def _variant_generation_dt(variant: dict[str, Any], target_ur: float) -> float:
        dt_by_ur = variant.get("generation_dt_by_ur", {})
        ur_key = float(round(float(target_ur), 6))
        if isinstance(dt_by_ur, dict) and ur_key in dt_by_ur:
            return float(dt_by_ur[ur_key])
        return float(variant["generation_dt"])

    def _generated_case_replay_rollout(
        *,
        method: str,
        source: LoadedTrainingModel | None,
        series: dict[str, Any],
        target_ur: float,
    ) -> dict[str, np.ndarray]:
        flow_speed_hist = np.asarray(series["td_context"], dtype=float)[:, 4].reshape(-1)
        finite_speed = flow_speed_hist[np.isfinite(flow_speed_hist)]
        if finite_speed.size == 0:
            raise ValueError(f"Series {series.get('name', '<unknown>')} does not have a valid flow-speed history.")
        flow_speed_const = float(np.median(finite_speed))
        effective_mass = float(series["effective_mass"])
        diameter = float(series["diameter"])
        target_stiffness = effective_mass * (2.0 * np.pi * flow_speed_const / (float(target_ur) * diameter)) ** 2
        model_ur_value = _series_reduced_velocity(series)
        td_context0 = np.asarray(series["td_context"], dtype=float)[0].copy()
        td_context0[4] = flow_speed_const
        if method == "Vivana-TD baseline":
            step_td_params = _resolve_td_params_for_dt(
                baseline_td_params,
                dt=float(np.median(np.diff(np.asarray(series["time"], dtype=float)))),
                td_memory_tau_s=td_memory_tau_s,
                flow_speed=flow_speed_const,
                diameter=diameter,
            )
            coupled = _simulate_vivana_rk4_coupled(
                time=np.asarray(series["time"], dtype=float),
                initial_displacement=float(np.asarray(series["displacement"], dtype=float).reshape(-1)[0]),
                initial_velocity=float(np.asarray(series["velocity"], dtype=float).reshape(-1)[0]),
                initial_acceleration=float(td_context0[0]),
                initial_phi_vy=float(td_context0[1]),
                initial_sig_dy=float(td_context0[2]),
                initial_sig_ddy=float(td_context0[3]),
                initial_force_per_m=float(np.asarray(series["force_td_stored"], dtype=float).reshape(-1)[0]),
                mass=_resolve_mass_value(series, mass_source=summary_mass_source),
                damping=float(series["damping"]),
                stiffness=float(target_stiffness),
                rho=float(series["rho"]),
                diameter=diameter,
                span=float(series["span"]),
                flow_speed=flow_speed_const,
                params=step_td_params,
            )
            return {"displacement": np.asarray(coupled["displacement"], dtype=float)}

        if source is None:
            raise ValueError("source is required for trained-model diagnostics.")
        initial_latent = (
            _latent_state_from_series_history(source, series, mass_source=summary_mass_source)
            if source.kind == "latent_rnn"
            else None
        )
        return _simulate_rollout(
            source=source,
            time=np.asarray(series["time"], dtype=float),
            initial_displacement=float(np.asarray(series["displacement"], dtype=float).reshape(-1)[0]),
            initial_velocity=float(np.asarray(series["velocity"], dtype=float).reshape(-1)[0]),
            td_context0=td_context0,
            ur_value=model_ur_value,
            mass=_resolve_mass_value(series, mass_source=summary_mass_source),
            damping=float(series["damping"]),
            stiffness=float(target_stiffness),
            rho=float(series["rho"]),
            diameter=diameter,
            span=float(series["span"]),
            td_memory_tau_s=td_memory_tau_s,
            initial_latent=initial_latent,
            force_phase_convention=force_phase_convention,
            use_vivana_added_mass_lhs=use_vivana_added_mass_lhs,
        )

    def _method_rollouts(
        *,
        method: str,
        source: LoadedTrainingModel | None,
        validation_series: dict[str, Any],
        anchor_template: dict[str, Any],
        target_ur: float,
        variant: dict[str, Any],
    ) -> tuple[float, float, float, float, float]:
        variant_generation_dt = _variant_generation_dt(variant, target_ur)
        generated_grid_duration = max(
            float(variant_generation_dt) * float(max(1, np.asarray(validation_series["time"], dtype=float).size - 1)),
            float(variant_generation_dt),
        )
        if method == "Vivana-TD baseline":
            validation_rollout = simulate_vivana_td_stepwise(
                validation_series,
                td_params=baseline_td_params,
                mass_source=td_mass_source,
                td_memory_tau_s=td_memory_tau_s,
                force_phase_convention=force_phase_convention,
                use_vivana_added_mass_lhs=use_vivana_added_mass_lhs,
            )
            generated_case_replay_rollout = _generated_case_replay_rollout(
                method=method,
                source=source,
                series=validation_series,
                target_ur=target_ur,
            )
            generated_grid_same_length_rollout = generate_vivana_summary_rollout(
                validation_series,
                target_ur,
                generation_dt=float(variant_generation_dt),
                generation_duration_s=generated_grid_duration,
                transient_seconds=0.0,
                td_params=baseline_td_params,
                mass_source=summary_mass_source,
                initial_state=None,
                td_memory_tau_s=td_memory_tau_s,
            )
            case_reset_rollout = generate_vivana_summary_rollout(
                validation_series,
                target_ur,
                generation_dt=float(variant_generation_dt),
                generation_duration_s=float(variant["generation_duration_s"]),
                transient_seconds=float(transient_seconds),
                td_params=baseline_td_params,
                mass_source=summary_mass_source,
                initial_state=None,
                td_memory_tau_s=td_memory_tau_s,
            )
            anchor_reset_rollout = generate_vivana_summary_rollout(
                anchor_template,
                target_ur,
                generation_dt=float(variant_generation_dt),
                generation_duration_s=float(variant["generation_duration_s"]),
                transient_seconds=float(transient_seconds),
                td_params=baseline_td_params,
                mass_source=summary_mass_source,
                initial_state=None,
                td_memory_tau_s=td_memory_tau_s,
            )
            return (
                float(np.std(validation_rollout["displacement_td"])),
                float(np.std(generated_case_replay_rollout["displacement"])),
                float(np.std(generated_grid_same_length_rollout["displacement"])),
                float(np.std(case_reset_rollout["displacement"])),
                float(np.std(anchor_reset_rollout["displacement"])),
            )

        if source is None:
            raise ValueError("source is required for trained-model diagnostics.")
        validation_initial_state = None
        anchor_initial_state = None
        if source.kind == "latent_rnn":
            validation_initial_state = {
                "displacement": float(np.asarray(validation_series["displacement"], dtype=float).reshape(-1)[0]),
                "velocity": float(np.asarray(validation_series["velocity"], dtype=float).reshape(-1)[0]),
                "td_context": np.asarray(validation_series["td_context"], dtype=float).reshape(-1)[:5].copy(),
                "latent": latent_reference_initial_state(
                    source,
                    [validation_series],
                    target_ur,
                    mass_source=summary_mass_source,
                    max_groups=1,
                ),
            }
            anchor_initial_state = {
                "displacement": float(np.asarray(anchor_template["displacement"], dtype=float).reshape(-1)[0]),
                "velocity": float(np.asarray(anchor_template["velocity"], dtype=float).reshape(-1)[0]),
                "td_context": np.asarray(anchor_template["td_context"], dtype=float).reshape(-1)[:5].copy(),
                "latent": latent_reference_initial_state(
                    source,
                    series_list,
                    target_ur,
                    mass_source=summary_mass_source,
                ),
            }
        validation_rollout = simulate_checkpoint_series_rollout(
            source,
            validation_series,
            mass_source=td_mass_source,
            td_memory_tau_s=td_memory_tau_s,
            force_phase_convention=force_phase_convention,
            use_vivana_added_mass_lhs=use_vivana_added_mass_lhs,
        )
        generated_case_replay_rollout = _generated_case_replay_rollout(
            method=method,
            source=source,
            series=validation_series,
            target_ur=target_ur,
        )
        generated_grid_same_length_rollout = generate_checkpoint_summary_rollout(
            source,
            validation_series,
            target_ur,
            generation_dt=float(variant_generation_dt),
            generation_duration_s=generated_grid_duration,
            transient_seconds=0.0,
            mass_source=summary_mass_source,
            initial_state=validation_initial_state,
            td_memory_tau_s=td_memory_tau_s,
            force_phase_convention=force_phase_convention,
            use_vivana_added_mass_lhs=use_vivana_added_mass_lhs,
        )
        case_reset_rollout = generate_checkpoint_summary_rollout(
            source,
            validation_series,
            target_ur,
            generation_dt=float(variant_generation_dt),
            generation_duration_s=float(variant["generation_duration_s"]),
            transient_seconds=float(transient_seconds),
            mass_source=summary_mass_source,
            initial_state=validation_initial_state,
            td_memory_tau_s=td_memory_tau_s,
            force_phase_convention=force_phase_convention,
            use_vivana_added_mass_lhs=use_vivana_added_mass_lhs,
        )
        anchor_reset_rollout = generate_checkpoint_summary_rollout(
            source,
            anchor_template,
            target_ur,
            generation_dt=float(variant_generation_dt),
            generation_duration_s=float(variant["generation_duration_s"]),
            transient_seconds=float(transient_seconds),
            mass_source=summary_mass_source,
            initial_state=anchor_initial_state,
            td_memory_tau_s=td_memory_tau_s,
            force_phase_convention=force_phase_convention,
        )
        return (
            float(np.std(validation_rollout["displacement"])),
            float(np.std(generated_case_replay_rollout["displacement"])),
            float(np.std(generated_grid_same_length_rollout["displacement"])),
            float(np.std(case_reset_rollout["displacement"])),
            float(np.std(anchor_reset_rollout["displacement"])),
        )

    method_specs: list[tuple[str, LoadedTrainingModel | None]] = [("Vivana-TD baseline", None)]
    method_specs.extend((source.label, source) for source in sources)

    for variant in summary_variants:
        exact_ur_set = {round(float(value), 6) for value in variant.get("exact_dataset_urs", [])}
        if not exact_ur_set:
            continue
        for entry in case_data:
            validation_series = entry.get("validation_series", entry["series"])
            ur_key = float(round(float(validation_series["ur_effective"]), 6))
            if ur_key not in exact_ur_set:
                continue
            anchor_template = _build_summary_sweep_template(series_list, ur_key)
            cfd_validation_std = float(np.std(np.asarray(validation_series["displacement"], dtype=float)))
            cfd_raw_std = float(np.std(np.asarray(entry["series"]["displacement"], dtype=float)))
            target_stiffness = (
                float(validation_series["effective_mass"])
                * (2.0 * np.pi * float(np.median(np.asarray(validation_series["td_context"], dtype=float)[:, 4])) / (ur_key * float(validation_series["diameter"]))) ** 2
            )
            stiffness_rel_diff = _safe_ratio(target_stiffness - float(validation_series["stiffness"]), float(validation_series["stiffness"]))
            model_input_ur = _series_reduced_velocity(validation_series)
            for method, source in method_specs:
                (
                    validation_std,
                    generated_case_replay_std,
                    generated_grid_same_length_std,
                    case_reset_std,
                    anchor_reset_std,
                ) = _method_rollouts(
                    method=method,
                    source=source,
                    validation_series=validation_series,
                    anchor_template=anchor_template,
                    target_ur=ur_key,
                    variant=variant,
                )
                sweep_std = _sweep_std(variant=variant, method=method, ur_key=ur_key)
                rows.append(
                    {
                        "dt variant": str(variant["label"]),
                        "case": str(validation_series.get("name", "<unknown>")),
                        "U_r": ur_key,
                        "model input U_r": model_input_ur,
                        "effective/model U_r ratio": _safe_ratio(ur_key, model_input_ur),
                        "method": method,
                        "CFD std raw": cfd_raw_std,
                        "CFD std validation": cfd_validation_std,
                        "validation replay std": validation_std,
                        "generated setup on validation grid std": generated_case_replay_std,
                        "generated dt same sample-count std": generated_grid_same_length_std,
                        "case synthetic reset std": case_reset_std,
                        "anchor reset std": anchor_reset_std,
                        "sweep continuation std": sweep_std,
                        "generated setup / validation": _safe_ratio(generated_case_replay_std, validation_std),
                        "generated dt / generated setup": _safe_ratio(generated_grid_same_length_std, generated_case_replay_std),
                        "long reset / generated dt": _safe_ratio(case_reset_std, generated_grid_same_length_std),
                        "anchor reset / case reset": _safe_ratio(anchor_reset_std, case_reset_std),
                        "sweep / anchor reset": _safe_ratio(sweep_std, anchor_reset_std),
                        "sweep / validation": _safe_ratio(sweep_std, validation_std),
                        "target stiffness rel diff": stiffness_rel_diff,
                    }
                )

    if not rows:
        return []

    return rows


def _plot_validation_error_metrics(
    *,
    case_data: list[dict[str, Any]],
    td_mass_source: str,
    baseline_td_params: dict[str, float],
    sources: Sequence[LoadedTrainingModel],
    td_memory_tau_s: float | str | None,
    unseen_raw_ur: float | None = 5.75,
    output_dir: Path | None = REPO_ROOT / "figs" / "block10",
    modified_td_params: dict[str, float] | None = None,
    modified_td_label: str = "VIVANA-TD modified",
    force_phase_convention: str = "next",
    use_vivana_added_mass_lhs: bool = False,
    heldout_ur: float = 6.46,
) -> dict[str, Any]:
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    model_styles = _model_style_map([source.label for source in sources])
    baseline_errors_by_metric: dict[str, dict[float, list[float]]] = {label: {} for label in VALIDATION_TRACKED_METRIC_KEYS}
    modified_errors_by_metric: dict[str, dict[float, list[float]]] = {label: {} for label in VALIDATION_TRACKED_METRIC_KEYS} if modified_td_params is not None else {}
    rk4_errors_by_metric: dict[str, dict[float, list[float]]] = {label: {} for label in RK4_REFERENCE_METRICS}
    model_errors_by_metric: dict[str, dict[str, dict[float, list[float]]]] = {
        source.label: {label: {} for label in VALIDATION_TRACKED_METRIC_KEYS} for source in sources
    }
    baseline_agg_by_ur: dict[float, list[float]] = {}
    modified_agg_by_ur: dict[float, list[float]] = {}
    model_agg_by_ur: dict[str, dict[float, list[float]]] = {source.label: {} for source in sources}
    validation_disp_std_by_method: dict[str, dict[float, list[float]]] = {
        "CFD reference": {},
        "VIVANA-TD baseline": {},
    }
    validation_disp_std_by_method.update({source.label: {} for source in sources})
    validation_force_std_by_method: dict[str, dict[float, list[float]]] = {
        "CFD reference": {},
        "VIVANA-TD baseline": {},
    }
    validation_force_std_by_method.update({source.label: {} for source in sources})
    validation_disp_freq_by_method: dict[str, dict[float, list[float]]] = {
        "CFD reference": {},
        "VIVANA-TD baseline": {},
    }
    validation_disp_freq_by_method.update({source.label: {} for source in sources})
    validation_force_freq_by_method: dict[str, dict[float, list[float]]] = {
        "CFD reference": {},
        "VIVANA-TD baseline": {},
    }
    validation_force_freq_by_method.update({source.label: {} for source in sources})

    def _record_validation_scalar(
        grouped: dict[str, dict[float, list[float]]],
        method: str,
        ur: float,
        value: float,
    ) -> None:
        value_float = float(value)
        if np.isfinite(value_float):
            grouped[method].setdefault(float(ur), []).append(value_float)

    validation_ur_ticks_arr = _validation_ur_ticks(case_data)
    dataset_errors_by_metric = _build_inherent_dataset_errors(case_data)
    unseen_raw_ur_value = None if unseen_raw_ur is None else float(unseen_raw_ur)
    unseen_errors_by_method: dict[str, dict[str, list[float]]] = {
        "Inherent CFD scatter (U_r mean)": (
            {label: [] for label in VALIDATION_COMPONENT_METRIC_KEYS}
            if unseen_raw_ur_value is None
            else _build_inherent_raw_ur_errors(case_data, unseen_raw_ur_value)
        ),
        "VIVANA-TD baseline": {label: [] for label in VALIDATION_COMPONENT_METRIC_KEYS},
        **({modified_td_label: {label: [] for label in VALIDATION_COMPONENT_METRIC_KEYS}} if modified_td_params is not None else {}),
        **{source.label: {label: [] for label in VALIDATION_COMPONENT_METRIC_KEYS} for source in sources},
    }

    for entry in _progress(case_data, total=len(case_data), desc="Validation metrics"):
        series = entry.get("validation_series", entry["series"])
        ur_key = float(round(float(series["ur_effective"]), 6))
        raw_ur_value = _series_raw_ur_value(series)
        is_unseen_raw_ur = (
            unseen_raw_ur_value is not None
            and np.isfinite(raw_ur_value)
            and np.isclose(raw_ur_value, unseen_raw_ur_value, rtol=0.0, atol=1.0e-9)
        )
        use_precomputed_rollouts = series is entry.get("series")
        baseline_rollout = entry.get("baseline_rollout") if use_precomputed_rollouts else None
        if baseline_rollout is None:
            baseline_rollout = simulate_vivana_td_stepwise(
                series,
                td_params=baseline_td_params,
                mass_source=td_mass_source,
                td_memory_tau_s=td_memory_tau_s,
                force_phase_convention=force_phase_convention,
                use_vivana_added_mass_lhs=use_vivana_added_mass_lhs,
            )
        truth_rollout = _compute_reference_truth_rollout(series, mass_source=td_mass_source)
        natural_frequency_hz = _series_natural_frequency_hz(series)
        _record_validation_scalar(validation_disp_std_by_method, "CFD reference", ur_key, np.std(series["displacement"]))
        _record_validation_scalar(
            validation_disp_std_by_method,
            "VIVANA-TD baseline",
            ur_key,
            np.std(baseline_rollout["displacement_td"]),
        )
        _record_validation_scalar(validation_force_std_by_method, "CFD reference", ur_key, np.std(series["force_per_m"]))
        _record_validation_scalar(
            validation_force_std_by_method,
            "VIVANA-TD baseline",
            ur_key,
            np.std(baseline_rollout["force_td"]),
        )
        _record_validation_scalar(
            validation_disp_freq_by_method,
            "CFD reference",
            ur_key,
            _frequency_ratio_from_signal(series["time"], series["displacement"], natural_frequency_hz),
        )
        _record_validation_scalar(
            validation_disp_freq_by_method,
            "VIVANA-TD baseline",
            ur_key,
            _frequency_ratio_from_signal(series["time"], baseline_rollout["displacement_td"], natural_frequency_hz),
        )
        _record_validation_scalar(
            validation_force_freq_by_method,
            "CFD reference",
            ur_key,
            _frequency_ratio_from_signal(series["time"], series["force_per_m"], natural_frequency_hz),
        )
        _record_validation_scalar(
            validation_force_freq_by_method,
            "VIVANA-TD baseline",
            ur_key,
            _frequency_ratio_from_signal(series["time"], baseline_rollout["force_td"], natural_frequency_hz),
        )

        baseline_metrics = compute_validation_style_error_metrics(
            time=series["time"],
            y_true=series["displacement"],
            y_pred=baseline_rollout["displacement_td"],
            force_true=series["force_per_m"],
            force_pred=baseline_rollout["force_td"],
        )
        rk4_metrics = compute_validation_style_error_metrics(
            time=series["time"],
            y_true=series["displacement"],
            y_pred=truth_rollout["displacement"],
        )
        for metric_label in VALIDATION_TRACKED_METRIC_KEYS:
            value = float(baseline_metrics.get(metric_label, float("nan")))
            if np.isfinite(value):
                baseline_errors_by_metric[metric_label].setdefault(ur_key, []).append(value)
        _agg_val = float(baseline_metrics.get(AGGREGATE_VALIDATION_ERROR_KEY, float("nan")))
        if np.isfinite(_agg_val):
            baseline_agg_by_ur.setdefault(ur_key, []).append(_agg_val)
        if is_unseen_raw_ur:
            for metric_label in VALIDATION_COMPONENT_METRIC_KEYS:
                value = float(baseline_metrics.get(metric_label, float("nan")))
                if np.isfinite(value):
                    unseen_errors_by_method["VIVANA-TD baseline"][metric_label].append(value)
        if modified_td_params is not None:
            modified_rollout = simulate_vivana_td_stepwise(
                series,
                td_params=modified_td_params,
                mass_source=td_mass_source,
                td_memory_tau_s=td_memory_tau_s,
                force_phase_convention=force_phase_convention,
                use_vivana_added_mass_lhs=use_vivana_added_mass_lhs,
            )
            modified_metrics = compute_validation_style_error_metrics(
                time=series["time"],
                y_true=series["displacement"],
                y_pred=modified_rollout["displacement_td"],
                force_true=series["force_per_m"],
                force_pred=modified_rollout["force_td"],
            )
            for metric_label in VALIDATION_TRACKED_METRIC_KEYS:
                value = float(modified_metrics.get(metric_label, float("nan")))
                if np.isfinite(value):
                    modified_errors_by_metric[metric_label].setdefault(ur_key, []).append(value)
            _agg_val = float(modified_metrics.get(AGGREGATE_VALIDATION_ERROR_KEY, float("nan")))
            if np.isfinite(_agg_val):
                modified_agg_by_ur.setdefault(ur_key, []).append(_agg_val)
            if is_unseen_raw_ur:
                for metric_label in VALIDATION_COMPONENT_METRIC_KEYS:
                    value = float(modified_metrics.get(metric_label, float("nan")))
                    if np.isfinite(value):
                        unseen_errors_by_method[modified_td_label][metric_label].append(value)
        for metric_key, metric_label in VALIDATION_ERROR_KEYS:
            if metric_label in RK4_REFERENCE_METRICS:
                rk4_value = float(rk4_metrics.get(metric_key, float("nan")))
                if np.isfinite(rk4_value):
                    rk4_errors_by_metric[metric_label].setdefault(ur_key, []).append(rk4_value)

        for source in sources:
            model_rollouts = entry.get("model_rollouts") if use_precomputed_rollouts else None
            if model_rollouts is None or source.label not in model_rollouts:
                rollout = simulate_checkpoint_series_rollout(
                    source,
                    series,
                    mass_source=td_mass_source,
                    td_memory_tau_s=td_memory_tau_s,
                    force_phase_convention=force_phase_convention,
                    use_vivana_added_mass_lhs=use_vivana_added_mass_lhs,
                )
            else:
                rollout = model_rollouts[source.label]
            eval_start = max(0, int(rollout.get("evaluation_start_idx", 0)))
            eval_time = np.asarray(series["time"], dtype=float).reshape(-1)[eval_start:]
            eval_displacement = np.asarray(rollout["displacement"], dtype=float).reshape(-1)[eval_start:]
            eval_force = np.asarray(rollout["force"], dtype=float).reshape(-1)[eval_start:]
            _record_validation_scalar(
                validation_disp_std_by_method,
                source.label,
                ur_key,
                np.std(eval_displacement),
            )
            _record_validation_scalar(validation_force_std_by_method, source.label, ur_key, np.std(eval_force))
            _record_validation_scalar(
                validation_disp_freq_by_method,
                source.label,
                ur_key,
                _frequency_ratio_from_signal(eval_time, eval_displacement, natural_frequency_hz),
            )
            _record_validation_scalar(
                validation_force_freq_by_method,
                source.label,
                ur_key,
                _frequency_ratio_from_signal(eval_time, eval_force, natural_frequency_hz),
            )
            model_metrics = compute_validation_style_error_metrics(
                time=eval_time,
                y_true=np.asarray(series["displacement"], dtype=float).reshape(-1)[eval_start:],
                y_pred=eval_displacement,
                force_true=np.asarray(series["force_per_m"], dtype=float).reshape(-1)[eval_start:],
                force_pred=eval_force,
            )
            for metric_label in VALIDATION_TRACKED_METRIC_KEYS:
                value = float(model_metrics.get(metric_label, float("nan")))
                if np.isfinite(value):
                    model_errors_by_metric[source.label][metric_label].setdefault(ur_key, []).append(value)
            _agg_val = float(model_metrics.get(AGGREGATE_VALIDATION_ERROR_KEY, float("nan")))
            if np.isfinite(_agg_val):
                model_agg_by_ur[source.label].setdefault(ur_key, []).append(_agg_val)
            if is_unseen_raw_ur:
                for metric_label in VALIDATION_COMPONENT_METRIC_KEYS:
                    value = float(model_metrics.get(metric_label, float("nan")))
                    if np.isfinite(value):
                        unseen_errors_by_method[source.label][metric_label].append(value)

    # Combined 4-panel figure for the 4 component error metrics (Force mapping NRMSE excluded).
    _component_metric_labels = [
        DISP_STD_REL_ERROR_KEY,
        DOMINANT_FREQ_REL_ERROR_KEY,
        FORCE_STD_REL_ERROR_KEY,
        FORCE_DOMINANT_FREQ_REL_ERROR_KEY,
    ]
    fig_comp, axes_comp = plt.subplots(
        len(_component_metric_labels), 1,
        figsize=(5.85, 6.4),
        sharex=True,
    )
    axes_comp = list(np.atleast_1d(axes_comp))
    for _ax_idx, (_ax, _metric_label) in enumerate(zip(axes_comp, _component_metric_labels)):
        if not baseline_errors_by_metric[_metric_label]:
            _ax.set_visible(False)
            continue
        _baseline_plot_grouped = _scaled_grouped_values(baseline_errors_by_metric[_metric_label], 100.0)
        _model_plot_grouped_by_label = {
            _src.label: _scaled_grouped_values(model_errors_by_metric[_src.label][_metric_label], 100.0)
            for _src in sources
        }
        _plotted_positive_sets: list[np.ndarray] = []
        _, _bmeans, _, _, _, _bp = _group_plot_stats(_baseline_plot_grouped)
        _bmeans = _bmeans[np.isfinite(_bmeans) & (_bmeans > 0.0)]
        if _bmeans.size > 0:
            _plotted_positive_sets.append(_bmeans)
        for _src in sources:
            _g = _model_plot_grouped_by_label[_src.label]
            if _g:
                _, _mmeans, _, _, _, _mp = _group_plot_stats(_g)
                _mmeans = _mmeans[np.isfinite(_mmeans) & (_mmeans > 0.0)]
                if _mmeans.size > 0:
                    _plotted_positive_sets.append(_mmeans)
        _merged = np.concatenate(_plotted_positive_sets) if _plotted_positive_sets else np.asarray([], dtype=float)
        _floor = float(max(1.0e-12, np.min(_merged))) if _merged.size > 0 else 1.0e-12
        _plot_validation_band(_ax, _baseline_plot_grouped, color="0.35", label="VIVANA-TD baseline", marker=None, floor=_floor, linestyle="--", linewidth=1.4, alpha_fill=0.0, alpha_std=0.0)
        for _src in sources:
            _g = _model_plot_grouped_by_label[_src.label]
            if not _g:
                continue
            _st = model_styles[_src.label]
            _plot_validation_band(_ax, _g, color=_st["color"], label=_src.label, marker=_st["marker"], floor=_floor, alpha_fill=0.0, alpha_std=0.0, linewidth=1.6)
        _ax.set_ylabel(VALIDATION_PLOT_YLABELS.get(_metric_label, _metric_label), rotation=90, labelpad=2, va="center")
        _ax.yaxis.set_label_coords(-0.085, 0.5)
        _ax.set_yscale("log")
        _set_block10_error_axis_limits(_ax, _merged)
        _add_section7_heldout_marker(_ax, ur_value=float(heldout_ur))
        _ax.grid(True, which="major", color="0.88", linewidth=0.5, alpha=0.75)
        _ax.grid(True, which="minor", color="0.94", linewidth=0.35, alpha=0.45)
        for _sp in _ax.spines.values():
            _sp.set_linewidth(0.6)
            _sp.set_edgecolor("0.65")
        if _ax_idx == 0:
            _handles, _hlabels = _ax.get_legend_handles_labels()
            if _handles:
                _ax.legend(_handles, _hlabels, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=_block10_legend_columns(len(_hlabels)), frameon=False, handlelength=2.0, columnspacing=1.2)
        if _ax_idx == len(axes_comp) - 1:
            _ax.set_xlabel(r"Reduced velocity $U_r$")
        if validation_ur_ticks_arr.size > 0:
            _ax.set_xticks(validation_ur_ticks_arr)
            _ax.set_xticklabels([f"{float(_v):.3g}" for _v in validation_ur_ticks_arr])
    plt.tight_layout()
    fig_comp.subplots_adjust(hspace=0.1)
    if output_dir is not None:
        _save_block10_figure(fig_comp, output_dir, "block10_component_errors")
    plt.show()

    # Aggregate error vs reduced velocity.
    if baseline_agg_by_ur:
        fig_agg, ax_agg = plt.subplots(figsize=(5.85, 3.1))
        _baseline_agg_plot = _scaled_grouped_values(baseline_agg_by_ur, 100.0)
        _model_agg_plot_by_label = {
            _src.label: _scaled_grouped_values(model_agg_by_ur[_src.label], 100.0)
            for _src in sources
        }
        _agg_plotted_positive_sets: list[np.ndarray] = []
        _, _abmeans, _, _, _, _abp = _group_plot_stats(_baseline_agg_plot)
        _abmeans = _abmeans[np.isfinite(_abmeans) & (_abmeans > 0.0)]
        if _abmeans.size > 0:
            _agg_plotted_positive_sets.append(_abmeans)
        for _src in sources:
            _g = _model_agg_plot_by_label[_src.label]
            if _g:
                _, _ammeans, _, _, _, _amp = _group_plot_stats(_g)
                _ammeans = _ammeans[np.isfinite(_ammeans) & (_ammeans > 0.0)]
                if _ammeans.size > 0:
                    _agg_plotted_positive_sets.append(_ammeans)
        _agg_merged = np.concatenate(_agg_plotted_positive_sets) if _agg_plotted_positive_sets else np.asarray([], dtype=float)
        _agg_floor = float(max(1.0e-12, np.min(_agg_merged))) if _agg_merged.size > 0 else 1.0e-12
        _plot_validation_band(ax_agg, _baseline_agg_plot, color="0.35", label="VIVANA-TD baseline", marker=None, floor=_agg_floor, linestyle="--", linewidth=1.4, alpha_fill=0.0, alpha_std=0.0)
        for _src in sources:
            _g = _model_agg_plot_by_label[_src.label]
            if not _g:
                continue
            _st = model_styles[_src.label]
            _plot_validation_band(ax_agg, _g, color=_st["color"], label=_src.label, marker=_st["marker"], floor=_agg_floor, alpha_fill=0.0, alpha_std=0.0, linewidth=1.6)
        _apply_block10_axes(
            ax_agg,
            ylabel=r"$\bar{\varepsilon}$ [%]",
            ur_ticks=validation_ur_ticks_arr,
            yscale="log",
            heldout_ur=heldout_ur,
        )
        _set_block10_error_axis_limits(ax_agg, _agg_merged)
        plt.tight_layout()
        fig_agg.subplots_adjust(top=0.82)
        if output_dir is not None:
            _save_block10_figure(fig_agg, output_dir, "block10_aggregate_error")
        plt.show()

    fig_std, ax_std = plt.subplots(figsize=(5.85, 3.1))
    _plot_grouped_scalar(
        ax_std,
        validation_disp_std_by_method["CFD reference"],
        color="black",
        label="CFD reference",
        alpha=1.0,
        size=95,
        marker="o",
        linewidth=0,
        show_scatter=True,
        show_line=False,
        facecolors="none",
        edgecolors="black",
        edge_linewidths=1.8,
    )
    _plot_grouped_scalar(
        ax_std,
        validation_disp_std_by_method["VIVANA-TD baseline"],
        color="0.35",
        label="VIVANA-TD baseline",
        alpha=0.0,
        size=0,
        marker=None,
        linewidth=1.4,
        linestyle="--",
        show_scatter=False,
    )
    for source in sources:
        grouped = validation_disp_std_by_method[source.label]
        if not grouped:
            continue
        style = model_styles[source.label]
        _plot_grouped_scalar(
            ax_std,
            grouped,
            color=style["color"],
            label=source.label,
            alpha=0.18,
            size=9,
            marker=style["marker"],
            linewidth=1.6,
            show_scatter=False,
        )
    _apply_block10_axes(
        ax_std,
        ylabel=r"$\sigma_{y/D}$",
        ur_ticks=validation_ur_ticks_arr,
        heldout_ur=heldout_ur,
    )
    plt.tight_layout()
    fig_std.subplots_adjust(top=0.82)
    if output_dir is not None:
        _save_block10_figure(fig_std, output_dir, "block10_disp_std")
    plt.show()

    scalar_plot_specs = [
        (validation_force_std_by_method,   r"$\sigma_{C_F}$",      "force_std"),
        (validation_disp_freq_by_method,   r"$\omega_y/\omega_n$",     "disp_freq_ratio"),
        (validation_force_freq_by_method,  r"$\omega_F/\omega_n$",     "force_freq_ratio"),
    ]
    for grouped_by_method, ylabel, fig_name in scalar_plot_specs:
        fig_scalar, ax_scalar = plt.subplots(figsize=(5.85, 3.1))
        _plot_grouped_scalar(
            ax_scalar,
            grouped_by_method["CFD reference"],
            color="black",
            label="CFD reference",
            alpha=1.0,
            size=95,
            marker="o",
            linewidth=0,
            show_scatter=True,
            show_line=False,
            facecolors="none",
            edgecolors="black",
            edge_linewidths=1.8,
        )
        _plot_grouped_scalar(
            ax_scalar,
            grouped_by_method["VIVANA-TD baseline"],
            color="0.35",
            label="VIVANA-TD baseline",
            alpha=0.0,
            size=0,
            marker=None,
            linewidth=1.4,
            linestyle="--",
            show_scatter=False,
        )
        for source in sources:
            grouped = grouped_by_method[source.label]
            if not grouped:
                continue
            style = model_styles[source.label]
            _plot_grouped_scalar(
                ax_scalar,
                grouped,
                color=style["color"],
                label=source.label,
                alpha=0.18,
                size=9,
                marker=style["marker"],
                linewidth=1.6,
                show_scatter=False,
            )
        _apply_block10_axes(
            ax_scalar,
            ylabel=ylabel,
            ur_ticks=validation_ur_ticks_arr,
            heldout_ur=heldout_ur,
        )
        plt.tight_layout()
        fig_scalar.subplots_adjust(top=0.82)
        if output_dir is not None:
            _save_block10_figure(fig_scalar, output_dir, f"block10_{fig_name}")
        plt.show()

    summary_rows = _build_validation_summary_rows(
        dataset_errors_by_metric=dataset_errors_by_metric,
        baseline_errors_by_metric=baseline_errors_by_metric,
        model_errors_by_metric=model_errors_by_metric,
        sources=sources,
        unseen_raw_ur=unseen_raw_ur_value,
        unseen_errors_by_method=unseen_errors_by_method,
        modified_errors_by_metric=modified_errors_by_metric if modified_td_params is not None else None,
        modified_td_label=modified_td_label if modified_td_params is not None else None,
    )
    _display_validation_summary_table(summary_rows)
    return {
        "dataset_errors_by_metric": dataset_errors_by_metric,
        "baseline_errors_by_metric": baseline_errors_by_metric,
        "rk4_errors_by_metric": rk4_errors_by_metric,
        "model_errors_by_metric": model_errors_by_metric,
        "case_matched_validation_disp_std": validation_disp_std_by_method,
        "case_matched_validation_force_std": validation_force_std_by_method,
        "case_matched_validation_disp_freq": validation_disp_freq_by_method,
        "case_matched_validation_force_freq": validation_force_freq_by_method,
        "unseen_raw_ur": unseen_raw_ur_value,
        "unseen_errors_by_method": unseen_errors_by_method,
        "summary_rows": summary_rows,
        "summary_metric_keys": list(VALIDATION_SUMMARY_METRIC_KEYS),
        "tracked_metric_keys": list(VALIDATION_TRACKED_METRIC_KEYS),
    }


def plot_trained_model_validation_analysis(
    *,
    case_data: list[dict[str, Any]],
    td_mass_source: str,
    baseline_td_params: dict[str, float],
    sources: Sequence[LoadedTrainingModel],
    td_memory_tau_s: float | str | None,
    unseen_raw_ur: float | None = 5.75,
    output_dir: Path | None = REPO_ROOT / "figs" / "block10",
    modified_td_params: dict[str, float] | None = None,
    modified_td_label: str = "VIVANA-TD modified",
    force_phase_convention: str = "next",
    use_vivana_added_mass_lhs: bool = False,
    heldout_ur: float = 6.46,
) -> dict[str, Any]:
    return _plot_validation_error_metrics(
        case_data=case_data,
        td_mass_source=td_mass_source,
        baseline_td_params=dict(baseline_td_params),
        sources=sources,
        td_memory_tau_s=td_memory_tau_s,
        unseen_raw_ur=unseen_raw_ur,
        output_dir=output_dir,
        modified_td_params=dict(modified_td_params) if modified_td_params is not None else None,
        modified_td_label=modified_td_label,
        force_phase_convention=force_phase_convention,
        use_vivana_added_mass_lhs=use_vivana_added_mass_lhs,
        heldout_ur=heldout_ur,
    )


def _plot_convergence_times(
    *,
    convergence_time_by_source: dict[str, dict[float, float]],
    sources: Sequence[LoadedTrainingModel],
    steady_state_max_seconds: float,
) -> None:
    source_styles: list[tuple[str, str, str]] = [
        ("Vivana-TD baseline", "#ff7f0e", "-"),
    ]
    for source in sources:
        source_styles.append((source.label, None, "--"))

    fig, ax = plt.subplots(figsize=(9, 4))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    auto_color_idx = 0
    for label, color, ls in source_styles:
        data = convergence_time_by_source.get(label, {})
        if not data:
            continue
        urs = np.asarray(sorted(data.keys()), dtype=float)
        times = np.asarray([data[u] for u in urs], dtype=float)
        if color is None:
            color = color_cycle[auto_color_idx % len(color_cycle)]
            auto_color_idx += 1
        ax.plot(urs, times, linestyle=ls, color=color, linewidth=1.8, marker="o", markersize=4, label=label)

    ax.axhline(steady_state_max_seconds, color="k", linewidth=0.8, linestyle=":", alpha=0.5, label=f"Max ({steady_state_max_seconds:.0f} s)")
    ax.set_xlabel("Reduced velocity $U_r$")
    ax.set_ylabel("Time to steady state [s]")
    ax.set_title("Convergence time by reduced velocity")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def run_trained_model_analysis(
    *,
    dataset_root: str | Path,
    repo_root: str | Path,
    model_specs: Sequence[str | Path | dict[str, Any]] | None = None,
    td_mass_source: str,
    baseline_td_params: dict[str, float],
    summary_mass_source: str,
    fine_ur_step: float,
    transient_seconds: float,
    figsize: tuple[float, float] = (14.0, 8.0),
    max_files_per_split: int | None = None,
    ordered_split_dirs: Sequence[str] = DEFAULT_ORDERED_SPLIT_DIRS,
    load_sima_series_for_npz: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
    summary_generation_dt: float | None = None,
    summary_generation_dt_specs: Sequence[float | None | dict[str, Any]] | None = None,
    summary_generation_duration_s: float | None = None,
    summary_kept_duration_s: float | None = None,
    summary_ur_range: tuple[float, float] | None = None,
    summary_first_case_extra_duration_s: float = 0.0,
    summary_generation_dt_from_reduction_factor: bool = False,
    summary_generation_dt_reduction_factor: int = 1,
    td_memory_tau_s: float | str | None = None,
    summary_output_dir: str | Path | None = None,
    show_td_frequency_guides: bool = True,
    show_spectral_plots: bool = True,
    show_validation_plots: bool = True,
    validation_reduce_time: bool = False,
    validation_reduction_factor: int = 1,
    validation_cut_start_seconds: float = 0.0,
    use_perturbed_start: bool = False,
    perturbation_fraction: float = 0.1,
    steady_state_n_cycles: int = 5,
    steady_state_amp_rel_tol: float = 0.05,
    steady_state_max_seconds: float = 800.0,
    stochastic_rollout: bool = False,
    summary_hysteresis: bool = False,
    summary_hysteresis_include_vivana: bool = True,
    summary_hysteresis_step_transient_seconds: float | None = None,
    summary_hysteresis_kept_seconds: float | None = None,
    summary_hysteresis_random_seed: int | None = None,
    show_vivana_td_baseline: bool = True,
    force_phase_convention: str = "next",
    use_vivana_added_mass_lhs: bool = False,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    dataset_root_path = Path(dataset_root).resolve()
    repo_root_path = Path(repo_root).resolve()
    sources = load_trained_model_sources(model_specs, repo_root=repo_root_path, device=device)
    if not sources:
        raise ValueError("No trained model checkpoints were provided.")
    if bool(summary_hysteresis):
        if summary_hysteresis_random_seed is not None:
            np.random.seed(int(summary_hysteresis_random_seed))
            torch.manual_seed(int(summary_hysteresis_random_seed))

    print("Loaded trained models:")
    for source in sources:
        if source.kind == "td_parameter_model":
            descriptor = "learned TD-params"
        elif source.kind == "poly3d_correction":
            descriptor = "3D polynomial surrogate"
        elif source.kind == "latent_rnn":
            descriptor = f"latent RNN force model (latent_dim={int(getattr(source.model, 'latent_dim', -1))})"
        else:
            descriptor = "PHNN correction"
        print(f"  - {source.label}: {source.path} ({descriptor})")
    if td_memory_tau_s is not None:
        if isinstance(td_memory_tau_s, str):
            print(f"Using TD memory tau override: tau={td_memory_tau_s!r}, with n_memory=tau/dt per rollout")
        else:
            print(f"Using TD memory tau override: tau={float(td_memory_tau_s):.6g} s, with n_memory=tau/dt per rollout")
    if bool(validation_reduce_time):
        print(
            "Validation metrics use reduced time grid:"
            f" reduction_factor={int(validation_reduction_factor)},"
            f" cut_start_seconds={float(validation_cut_start_seconds):.6g}"
        )
    if bool(use_vivana_added_mass_lhs):
        print("Rollout integration uses the Vivana-TD added-mass coefficient on the LHS.")

    case_data = _build_case_rollouts(
        dataset_root=dataset_root_path,
        sources=sources,
        baseline_td_params=dict(baseline_td_params),
        td_mass_source=td_mass_source,
        td_memory_tau_s=td_memory_tau_s,
        max_files_per_split=max_files_per_split,
        ordered_split_dirs=ordered_split_dirs,
        load_sima_series_for_npz=load_sima_series_for_npz,
        include_rollouts=show_spectral_plots,
        validation_reduce_time=bool(validation_reduce_time),
        validation_reduction_factor=int(validation_reduction_factor),
        validation_cut_start_seconds=float(validation_cut_start_seconds),
        force_phase_convention=force_phase_convention,
        use_vivana_added_mass_lhs=bool(use_vivana_added_mass_lhs),
    )
    if show_spectral_plots:
        _plot_case_psd_overlays(
            case_data=case_data,
            td_mass_source=td_mass_source,
            figsize=figsize,
            sources=sources,
            show_vivana_td_baseline=bool(show_vivana_td_baseline),
        )

    if bool(summary_generation_dt_from_reduction_factor):
        dt_specs = [
            {
                "key": "reduction_factor_dt",
                "dt": None,
                "label": f"dt from reduction factor={int(summary_generation_dt_reduction_factor)}",
                "linestyle": "--",
            }
        ]
        print(
            "Model summary generation dt uses dataset raw dt times reduction factor:"
            f" reduction_factor={int(summary_generation_dt_reduction_factor)}"
        )
    else:
        dt_specs = _normalize_summary_generation_dt_specs(
            summary_generation_dt=summary_generation_dt,
            summary_generation_dt_specs=summary_generation_dt_specs,
        )
    cfd_grouped: dict[str, dict[float, list[float]]] | None = None
    sima_grouped: dict[str, dict[float, list[float]]] | None = None
    summary_variants: list[dict[str, Any]] = []
    summary_hysteresis_results: dict[str, Any] | None = None
    for dt_spec in dt_specs:
        (
            cfd_grouped_variant,
            baseline_grouped,
            sima_grouped_variant,
            model_grouped,
            resolved_summary_generation_dt,
            resolved_summary_generation_duration_s,
            exact_dataset_urs,
            generation_dt_by_ur,
            generated_rollouts,
            convergence_time_by_source,
        ) = _build_summary_metrics(
            case_data=case_data,
            sources=sources,
            baseline_td_params=dict(baseline_td_params),
            fine_ur_step=float(fine_ur_step),
            summary_mass_source=summary_mass_source,
            transient_seconds=float(transient_seconds),
            summary_generation_dt=dt_spec["dt"],
            summary_generation_duration_s=summary_generation_duration_s,
            summary_kept_duration_s=summary_kept_duration_s,
            summary_ur_range=summary_ur_range,
            summary_first_case_extra_duration_s=summary_first_case_extra_duration_s,
            summary_generation_dt_from_reduction_factor=bool(summary_generation_dt_from_reduction_factor),
            summary_generation_dt_reduction_factor=int(summary_generation_dt_reduction_factor),
            td_memory_tau_s=td_memory_tau_s,
            collect_generated_rollouts=False,
            use_perturbed_start=bool(use_perturbed_start),
            perturbation_fraction=float(perturbation_fraction),
            steady_state_n_cycles=int(steady_state_n_cycles),
            steady_state_amp_rel_tol=float(steady_state_amp_rel_tol),
            steady_state_max_seconds=float(steady_state_max_seconds),
            stochastic_rollout=bool(stochastic_rollout),
            force_phase_convention=force_phase_convention,
            use_vivana_added_mass_lhs=bool(use_vivana_added_mass_lhs),
        )
        if cfd_grouped is None:
            cfd_grouped = cfd_grouped_variant
        if sima_grouped is None:
            sima_grouped = sima_grouped_variant
        variant_label = str(dt_spec["label"])
        dt_values_for_variant = np.asarray(list(generation_dt_by_ur.values()), dtype=float)
        if bool(summary_generation_dt_from_reduction_factor) and dt_values_for_variant.size:
            variant_label = (
                f"{variant_label}"
                f" (range {float(np.min(dt_values_for_variant)):.6g}-{float(np.max(dt_values_for_variant)):.6g})"
            )
        elif dt_spec["dt"] is None:
            variant_label = f"{variant_label} (resolved {resolved_summary_generation_dt:.6g})"
        print(
            "Model summary generation grid:"
            f" {variant_label},"
            f" total duration={resolved_summary_generation_duration_s:.3f} s,"
            f" transient trimmed={float(transient_seconds):.1f} s,"
            f" kept duration={resolved_summary_generation_duration_s - float(transient_seconds):.3f} s,"
            f" exact dataset U_r points={int(np.asarray(exact_dataset_urs).size)}"
        )
        hysteresis_grouped = None
        hysteresis_details = None
        if bool(summary_hysteresis):
            target_urs_for_hysteresis = np.asarray(sorted(baseline_grouped["disp_std"].keys()), dtype=float)
            if target_urs_for_hysteresis.size == 0:
                raise ValueError("No generated U_r points are available for the hysteresis overlay.")
            hysteresis_grouped = {}
            hysteresis_details = {"sources": {}}
            for source_idx, source in enumerate(sources):
                source_hysteresis_grouped, source_hysteresis_details = _build_hysteresis_summary_metrics(
                    case_data=case_data,
                    source=source,
                    baseline_td_params=dict(baseline_td_params),
                    target_urs=target_urs_for_hysteresis,
                    summary_mass_source=summary_mass_source,
                    summary_generation_dt=dt_spec["dt"],
                    summary_generation_dt_from_reduction_factor=bool(summary_generation_dt_from_reduction_factor),
                    summary_generation_dt_reduction_factor=int(summary_generation_dt_reduction_factor),
                    td_memory_tau_s=td_memory_tau_s,
                    include_vivana=bool(summary_hysteresis_include_vivana and source_idx == 0),
                    step_transient_seconds=(
                        float(summary_hysteresis_step_transient_seconds)
                        if summary_hysteresis_step_transient_seconds is not None
                        else float(summary_kept_duration_s if summary_kept_duration_s is not None else resolved_summary_generation_duration_s - float(transient_seconds))
                    ),
                    kept_duration_s=(
                        float(summary_hysteresis_kept_seconds)
                        if summary_hysteresis_kept_seconds is not None
                        else float(summary_kept_duration_s if summary_kept_duration_s is not None else resolved_summary_generation_duration_s - float(transient_seconds))
                    ),
                    perturbation_fraction=float(perturbation_fraction),
                    steady_state_n_cycles=int(steady_state_n_cycles),
                    steady_state_amp_rel_tol=float(steady_state_amp_rel_tol),
                    steady_state_max_seconds=float(steady_state_max_seconds),
                    stochastic_rollout=bool(stochastic_rollout),
                    force_phase_convention=force_phase_convention,
                    use_vivana_added_mass_lhs=bool(use_vivana_added_mass_lhs),
                )
                hysteresis_grouped.update(source_hysteresis_grouped)
                hysteresis_details["sources"][source.label] = source_hysteresis_details
            model_grouped_for_plot = hysteresis_grouped
            if summary_hysteresis_results is None:
                summary_hysteresis_results = {
                    "sources": sources,
                    "variants": {},
                }
            summary_hysteresis_results["variants"][str(dt_spec["key"])] = {
                "grouped": hysteresis_grouped,
                "details": hysteresis_details,
            }
        else:
            model_grouped_for_plot = model_grouped
        summary_variants.append(
            {
                "key": dt_spec["key"],
                "label": variant_label,
                "linestyle": dt_spec["linestyle"],
                "requested_generation_dt": dt_spec["dt"],
                "generation_dt": resolved_summary_generation_dt,
                "generation_dt_by_ur": generation_dt_by_ur,
                "generation_dt_from_reduction_factor": bool(summary_generation_dt_from_reduction_factor),
                "generation_dt_reduction_factor": int(summary_generation_dt_reduction_factor),
                "generation_duration_s": resolved_summary_generation_duration_s,
                "exact_dataset_urs": exact_dataset_urs,
                "baseline_grouped": baseline_grouped,
                "model_grouped": model_grouped_for_plot,
                "standard_model_grouped": model_grouped,
                "generated_rollouts": generated_rollouts,
                "hysteresis_grouped": hysteresis_grouped,
                "hysteresis_details": hysteresis_details,
            }
        )
    if cfd_grouped is None:
        raise ValueError("No summary metrics were generated for the requested dt variants.")
    _plot_summary_metrics(
        cfd_grouped=cfd_grouped,
        sima_grouped=sima_grouped,
        summary_variants=summary_variants,
        fine_ur_step=float(fine_ur_step),
        summary_mass_source=summary_mass_source,
        sources=sources,
        baseline_td_params=dict(baseline_td_params),
        show_td_frequency_guides=bool(show_td_frequency_guides),
        show_vivana_td_baseline=bool(show_vivana_td_baseline),
        output_dir=Path(summary_output_dir) if summary_output_dir is not None else None,
    )
    if bool(use_perturbed_start) and convergence_time_by_source:
        _plot_convergence_times(
            convergence_time_by_source=convergence_time_by_source,
            sources=sources,
            steady_state_max_seconds=float(steady_state_max_seconds),
        )
    exact_ur_sweep_diagnostic = _display_exact_ur_sweep_diagnostic(
        case_data=case_data,
        summary_variants=summary_variants,
        sources=sources,
        baseline_td_params=dict(baseline_td_params),
        td_mass_source=td_mass_source,
        summary_mass_source=summary_mass_source,
        transient_seconds=float(transient_seconds),
        td_memory_tau_s=td_memory_tau_s,
        force_phase_convention=force_phase_convention,
        use_vivana_added_mass_lhs=bool(use_vivana_added_mass_lhs),
    )
    validation = None
    if bool(show_validation_plots):
        validation = plot_trained_model_validation_analysis(
            case_data=case_data,
            td_mass_source=td_mass_source,
            baseline_td_params=dict(baseline_td_params),
            sources=sources,
            td_memory_tau_s=td_memory_tau_s,
            force_phase_convention=force_phase_convention,
            use_vivana_added_mass_lhs=bool(use_vivana_added_mass_lhs),
        )

    return {
        "sources": sources,
        "case_data": case_data,
        "summary": {
            "cfd_grouped": cfd_grouped,
            "sima_grouped": sima_grouped,
            "variants": summary_variants,
            "baseline_grouped": summary_variants[0]["baseline_grouped"],
            "model_grouped": summary_variants[0]["model_grouped"],
            "generation_dt": summary_variants[0]["generation_dt"],
            "generation_duration_s": summary_variants[0]["generation_duration_s"],
            "kept_duration_s": summary_variants[0]["generation_duration_s"] - float(transient_seconds),
            "summary_ur_range": summary_ur_range,
            "summary_first_case_extra_duration_s": float(summary_first_case_extra_duration_s),
            "td_memory_tau_s": td_memory_tau_s if isinstance(td_memory_tau_s, str) or td_memory_tau_s is None else float(td_memory_tau_s),
            "show_td_frequency_guides": bool(show_td_frequency_guides),
            "show_vivana_td_baseline": bool(show_vivana_td_baseline),
            "show_validation_plots": bool(show_validation_plots),
            "use_vivana_added_mass_lhs": bool(use_vivana_added_mass_lhs),
            "exact_ur_sweep_diagnostic": exact_ur_sweep_diagnostic,
            "hysteresis": summary_hysteresis_results,
        },
        "validation": validation,
        "validation_settings": {
            "reduce_time": bool(validation_reduce_time),
            "reduction_factor": int(validation_reduction_factor),
            "cut_start_seconds": float(validation_cut_start_seconds),
        },
        "convergence_time_by_source": convergence_time_by_source,
        "hysteresis": summary_hysteresis_results,
        "use_vivana_added_mass_lhs": bool(use_vivana_added_mass_lhs),
    }
