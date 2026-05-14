"""Plot Vivana-TD and learned corrected fhat synchronization heatmaps."""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DIMENSIONS = ("ur", "y", "dy")

# Edit these values and run this file directly.
CHECKPOINT_PATH = Path("models/fhat/graph_making/final.pt")
DATA_ROOT = Path("CFD_Data/npz_exports_td_burnin_trimmed4")
TRAIN_SPLITS = ("train",)
UNSEEN_DATA_ROOT: Path | None = None  # None means use DATA_ROOT.
UNSEEN_SPLITS = ("val_seen",)
EXTRA_UNSEEN_UR_VALUES: tuple[float, ...] = (
    3.36965056,
    5.05450547,
    6.73904035,
    8.42459486,
    10.10828857,
)

# Set LEGACY_SPLITS to a tuple such as ("train", "val_seen") to treat all
# selected data as training coverage and disable unseen hatching.
LEGACY_SPLITS: tuple[str, ...] | None = None

DIMENSION = "all"  # "all", "ur", "y", or "dy"
BINS_THETA = 90
BINS_OTHER = 70
OUT_DIR = Path("figs/fhat_sync_graph_heatmaps")
MAX_FILES: int | None = None  # Optional per-group file cap for smoke runs.
BATCH_SIZE = 65536
DEVICE = "cpu"
FILL_EMPTY_BINS = True


def configured_dimensions() -> tuple[str, ...]:
    available = (DIMENSIONS,) if isinstance(DIMENSIONS, str) else tuple(DIMENSIONS)
    selected = (DIMENSION,) if isinstance(DIMENSION, str) and DIMENSION != "all" else available
    invalid = [dim for dim in selected if dim not in {"ur", "y", "dy"}]
    if invalid:
        raise ValueError(f"Invalid dimension(s) {invalid}; use 'ur', 'y', 'dy', or DIMENSION='all'.")
    if not selected:
        raise ValueError("No dimensions selected for plotting.")
    return selected


def require_runtime_modules() -> dict[str, Any]:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "This script requires PyTorch to load and evaluate the checkpoint. "
            "Install the project requirements or run it from an environment where 'import torch' works."
        ) from exc

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize

        from HNN_helper import (
            PHVIV,
            _broadcast_td_hidden_param_torch,
            load_td_correction_trajectories,
            parse_config,
            resolve_td_correction_params,
            resolve_td_fhat_correction_bounds,
            resolve_td_memory_config,
            resolve_td_n_memory_torch,
            td_baseline_step_torch,
            td_hidden_inputs_from_context_torch,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Missing required module '{exc.name}'. Install the project requirements before running this script."
        ) from exc

    return {
        "torch": torch,
        "plt": plt,
        "Normalize": Normalize,
        "PHVIV": PHVIV,
        "_broadcast_td_hidden_param_torch": _broadcast_td_hidden_param_torch,
        "load_td_correction_trajectories": load_td_correction_trajectories,
        "parse_config": parse_config,
        "resolve_td_correction_params": resolve_td_correction_params,
        "resolve_td_fhat_correction_bounds": resolve_td_fhat_correction_bounds,
        "resolve_td_memory_config": resolve_td_memory_config,
        "resolve_td_n_memory_torch": resolve_td_n_memory_torch,
        "td_baseline_step_torch": td_baseline_step_torch,
        "td_hidden_inputs_from_context_torch": td_hidden_inputs_from_context_torch,
    }


def checkpoint_config_dict(ckpt: dict[str, Any]) -> dict[str, Any]:
    raw_cfg = ckpt.get("config", {})
    if isinstance(raw_cfg, dict):
        return raw_cfg
    if hasattr(raw_cfg, "__dict__"):
        return dict(raw_cfg.__dict__)
    raise TypeError(f"Unsupported checkpoint config type: {type(raw_cfg)}")


def torch_load_checkpoint(torch: Any, checkpoint_path: Path, map_location: Any) -> dict[str, Any]:
    try:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=map_location)


def collect_npz_paths(data_root: Path, splits: list[str], max_files: int | None) -> list[Path]:
    paths: list[Path] = []
    for split in splits:
        split_dir = data_root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory does not exist: {split_dir}")
        paths.extend(sorted(split_dir.glob("*.npz")))
    if not paths:
        raise FileNotFoundError(f"No .npz files found under {data_root} for splits {splits}.")
    if max_files is not None:
        if max_files < 1:
            raise ValueError("--max-files must be positive when provided.")
        paths = paths[:max_files]
    return paths


def merge_sample_dicts(sample_dicts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not sample_dicts:
        raise ValueError("No sample dictionaries were provided.")
    keys = sample_dicts[0].keys()
    return {key: np.concatenate([samples[key] for samples in sample_dicts]) for key in keys}


def load_checkpoint_model(
    *,
    modules: dict[str, Any],
    checkpoint_path: Path,
    first_traj: dict[str, Any],
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    torch = modules["torch"]
    PHVIV = modules["PHVIV"]
    parse_config = modules["parse_config"]
    resolve_td_fhat_correction_bounds = modules["resolve_td_fhat_correction_bounds"]

    ckpt = torch_load_checkpoint(torch, checkpoint_path, map_location=device)
    cfg = parse_config(ckpt["config"])
    cfg_dict = checkpoint_config_dict(ckpt)
    hnn_cfg = dict(cfg_dict.get("hnn", {}))
    model_cfg = asdict(cfg.model)
    arch_cfg = asdict(cfg.architecture)

    model_cfg["structural_mass"] = float(np.asarray(first_traj["dry_mass_kg"]).reshape(()))
    model_cfg["k"] = float(np.asarray(first_traj["stiffness_n_m"]).reshape(()))
    model_cfg["damping_c"] = float(np.asarray(first_traj["damping_c"]).reshape(()))
    model_cfg["Ca"] = 0.0
    model_cfg["use_stochastic_process_noise"] = bool(ckpt.get("predict_sigma", False))
    model_cfg["correction_mode"] = ckpt.get("correction_mode", hnn_cfg.get("correction_mode", "fhat_only"))
    model_cfg["use_td_force_input"] = ckpt.get("td_force_input_source", ckpt.get("use_td_force_input", False))
    model_cfg["use_td_fhat_input"] = bool(ckpt.get("use_td_fhat_input", hnn_cfg.get("use_td_fhat_input", False)))
    model_cfg["use_acceleration_input"] = bool(
        ckpt.get("use_acceleration_input", hnn_cfg.get("use_acceleration_input", False))
    )
    model_cfg["use_phi_input"] = bool(ckpt.get("use_phi_input", hnn_cfg.get("use_phi_input", False)))
    model_cfg["phi_input_source"] = ckpt.get("phi_input_source", hnn_cfg.get("use_phi_input", "theta"))
    model_cfg["use_sigma_inputs"] = bool(ckpt.get("use_sigma_inputs", hnn_cfg.get("use_sigma_inputs", False)))
    if "input_configs" in ckpt:
        model_cfg["input_configs"] = ckpt["input_configs"]
    arch_cfg["shared_td_correction_trunk"] = bool(
        ckpt.get("shared_td_correction_trunk", arch_cfg.get("shared_td_correction_trunk", False))
    )

    model, _derived = PHVIV.from_config(
        dt=float(ckpt.get("dt", 1.0)),
        cfg=model_cfg,
        arch_cfg=arch_cfg,
        device=device,
    )
    setattr(model, "correction_mode", model_cfg["correction_mode"])
    setattr(model, "td_force_input_source", ckpt.get("td_force_input_source", "none"))
    setattr(model, "fhat_bound_multiplier", float(ckpt.get("fhat_bound_multiplier", hnn_cfg.get("fhat_bound_multiplier", 1.5))))
    if "fhat_correction_bounds" in ckpt:
        fhat_correction_bounds = ckpt["fhat_correction_bounds"]
    else:
        fhat_correction_bounds = resolve_td_fhat_correction_bounds(hnn_cfg)
    setattr(model, "fhat_correction_bounds", fhat_correction_bounds)
    setattr(model, "force_zero_output", bool(ckpt.get("force_zero_output", hnn_cfg.get("force_zero_output", False))))

    state = {
        key.removeprefix("_orig_mod.").removeprefix("module."): value
        for key, value in ckpt["model_state"].items()
    }
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, ckpt


def td_p_scale_tensor(
    *,
    modules: dict[str, Any],
    model: Any,
    reduced_velocity: Any,
    structural_mass: Any,
    stiffness: Any,
    like: Any,
) -> Any:
    torch = modules["torch"]
    broadcast = modules["_broadcast_td_hidden_param_torch"]
    mass_t = broadcast(structural_mass, like=like, name="structural_mass")
    stiffness_t = broadcast(stiffness, like=like, name="stiffness")
    if getattr(model, "input_scaling_mode", "current") == "convective":
        rv_raw = model._prepare_reduced_velocity_raw(reduced_velocity, like=like)
        if rv_raw is None:
            raise ValueError("reduced_velocity is required for convective PHNN momentum scaling.")
        u_flow = torch.clamp(torch.abs(rv_raw * float(model.D)), min=1e-12)
        return torch.clamp(mass_t * u_flow, min=1e-12)
    return torch.sqrt(torch.clamp(mass_t * stiffness_t, min=1e-12)) * float(model.D)


def state_for_model_scaling(
    *,
    modules: dict[str, Any],
    model: Any,
    z: Any,
    reduced_velocity: Any,
    structural_mass: Any,
    stiffness: Any,
) -> Any:
    torch = modules["torch"]
    p_scale_actual = td_p_scale_tensor(
        modules=modules,
        model=model,
        reduced_velocity=reduced_velocity,
        structural_mass=structural_mass,
        stiffness=stiffness,
        like=z[..., :1],
    )
    p_scale_model = torch.as_tensor(float(model.nn_p_scale), device=z.device, dtype=z.dtype)
    p_model = z[..., 1:2] * (p_scale_model / p_scale_actual)
    return torch.cat([z[..., 0:1], p_model], dim=-1)


def fhat_head_phi_input(
    *,
    modules: dict[str, Any],
    model: Any,
    td_context: Any,
    velocity: Any,
    structural_mass: Any,
    stiffness: Any,
) -> Any:
    input_configs = getattr(model, "td_input_configs", {})
    fhat_cfg = input_configs.get("fhat", input_configs.get("mean", {}))
    if not bool(fhat_cfg.get("use_phi_input", False)):
        return None
    phase_source = fhat_cfg.get("phase_input_source", fhat_cfg.get("phi_input_source", "theta"))
    phi_input, _sigma_inputs, _acceleration_input = modules["td_hidden_inputs_from_context_torch"](
        td_context=td_context,
        structural_mass=structural_mass,
        stiffness=stiffness,
        diameter=float(model.D),
        velocity=velocity,
        phase_input_source=phase_source,
        input_scaling_mode=getattr(model, "input_scaling_mode", "current"),
    )
    return phi_input


def evaluate_fhat_samples(
    *,
    modules: dict[str, Any],
    model: Any,
    trajectories: list[dict[str, Any]],
    td_params: dict[str, float],
    td_memory_cfg: dict[str, Any],
    device: Any,
    batch_size: int,
    is_unseen: bool,
    override_ur: float | None = None,
) -> dict[str, np.ndarray]:
    torch = modules["torch"]
    resolve_td_n_memory_torch = modules["resolve_td_n_memory_torch"]
    td_baseline_step_torch = modules["td_baseline_step_torch"]

    sample_blocks: list[dict[str, np.ndarray]] = []
    with torch.no_grad():
        for traj in trajectories:
            t = np.asarray(traj["t"], dtype=np.float32).reshape(-1)
            if t.size < 2:
                continue
            n = t.size - 1
            y = np.asarray(traj["y"], dtype=np.float32).reshape(-1)[:n]
            dy = np.asarray(traj["dy"], dtype=np.float32).reshape(-1)[:n]
            if override_ur is None:
                ur = np.asarray(traj["ur"], dtype=np.float32).reshape(-1)[:n]
            else:
                ur = np.full(n, float(override_ur), dtype=np.float32)
            td_context = np.asarray(traj["td_context"], dtype=np.float32)[:n]
            dt = np.diff(t).astype(np.float32).reshape(-1, 1)
            dry_mass = np.full((n, 1), float(np.asarray(traj["dry_mass_kg"]).reshape(())), dtype=np.float32)
            stiffness = np.full((n, 1), float(np.asarray(traj["stiffness_n_m"]).reshape(())), dtype=np.float32)

            theta_parts: list[np.ndarray] = []
            fhat_td_parts: list[np.ndarray] = []
            fhat_corr_parts: list[np.ndarray] = []
            delta_parts: list[np.ndarray] = []
            for start in range(0, n, batch_size):
                stop = min(start + batch_size, n)
                y_b = torch.from_numpy(y[start:stop, None]).to(device)
                dy_b = torch.from_numpy(dy[start:stop, None]).to(device)
                ur_b = torch.from_numpy(ur[start:stop, None]).to(device)
                context_b = torch.from_numpy(td_context[start:stop]).to(device)
                dt_b = torch.from_numpy(dt[start:stop]).to(device)
                mass_b = torch.from_numpy(dry_mass[start:stop]).to(device)
                stiffness_b = torch.from_numpy(stiffness[start:stop]).to(device)

                z = torch.cat([y_b, mass_b * dy_b], dim=1)
                velocity = z[:, 1:2] / mass_b
                step_params = dict(td_params)
                step_params["n_memory"] = resolve_td_n_memory_torch(
                    td_params,
                    dt=dt_b,
                    flow_speed=context_b[:, 4:5],
                    diameter=float(model.D),
                    memory_cfg=td_memory_cfg,
                )
                z_model = state_for_model_scaling(
                    modules=modules,
                    model=model,
                    z=z,
                    reduced_velocity=ur_b,
                    structural_mass=mass_b,
                    stiffness=stiffness_b,
                )
                phi_input = fhat_head_phi_input(
                    modules=modules,
                    model=model,
                    td_context=context_b,
                    velocity=velocity,
                    structural_mass=mass_b,
                    stiffness=stiffness_b,
                )
                raw_delta_fhat = model._fhat_net_raw(
                    z_model,
                    reduced_velocity=ur_b,
                    phi_input=phi_input,
                )
                bounds = getattr(model, "fhat_correction_bounds", None)
                if bounds is None:
                    bound_min = None
                    bound_max = None
                else:
                    bound_min, bound_max = bounds
                _td_force, _td_context_next, td_diag = td_baseline_step_torch(
                    velocity=velocity,
                    acceleration=context_b[:, 0:1],
                    td_context=context_b,
                    dt=dt_b,
                    rho=float(model.rho),
                    diameter=float(model.D),
                    params=step_params,
                    raw_delta_fhat=raw_delta_fhat,
                    fhat_bound_multiplier=float(getattr(model, "fhat_bound_multiplier", 1.5)),
                    fhat_bound_min=bound_min,
                    fhat_bound_max=bound_max,
                    return_diagnostics=True,
                )
                theta_parts.append(td_diag["theta_td"].detach().cpu().numpy().reshape(-1))
                fhat_td_parts.append(td_diag["fhat_td"].detach().cpu().numpy().reshape(-1))
                fhat_corr_parts.append(td_diag["fhat_corr"].detach().cpu().numpy().reshape(-1))
                delta_parts.append(td_diag["delta_fhat"].detach().cpu().numpy().reshape(-1))

            sample_blocks.append(
                {
                    "theta": np.concatenate(theta_parts),
                    "fhat_td": np.concatenate(fhat_td_parts),
                    "fhat_corr": np.concatenate(fhat_corr_parts),
                    "delta_fhat": np.concatenate(delta_parts),
                    "ur": ur,
                    "y": y,
                    "dy": dy,
                    "is_unseen": np.full_like(ur, bool(is_unseen), dtype=bool),
                }
            )

    if not sample_blocks:
        raise ValueError("No valid trajectory samples were available for fhat evaluation.")
    return {key: np.concatenate([block[key] for block in sample_blocks]) for key in sample_blocks[0]}


def bin_average(
    *,
    theta: np.ndarray,
    other: np.ndarray,
    values: np.ndarray,
    theta_edges: np.ndarray,
    other_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(theta) & np.isfinite(other) & np.isfinite(values)
    sums, _, _ = np.histogram2d(
        other[finite],
        theta[finite],
        bins=(other_edges, theta_edges),
        weights=values[finite],
    )
    counts, _, _ = np.histogram2d(other[finite], theta[finite], bins=(other_edges, theta_edges))
    with np.errstate(divide="ignore", invalid="ignore"):
        means = np.where(counts > 0.0, sums / counts, np.nan)
    return means, counts


def bin_counts(
    *,
    theta: np.ndarray,
    other: np.ndarray,
    theta_edges: np.ndarray,
    other_edges: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    finite = np.isfinite(theta) & np.isfinite(other) & mask
    counts, _, _ = np.histogram2d(other[finite], theta[finite], bins=(other_edges, theta_edges))
    return counts


def finite_min_max(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Cannot determine plot range from all-NaN values.")
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if np.isclose(lo, hi):
        pad = max(1.0e-6, abs(lo) * 0.05)
        lo -= pad
        hi += pad
    return lo, hi


def categorical_edges(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Cannot determine categorical plot rows from all-NaN values.")
    centers = np.asarray(sorted(np.unique(finite)), dtype=float)
    if centers.size == 1:
        pad = max(0.5, abs(float(centers[0])) * 0.05)
        return centers, np.asarray([centers[0] - pad, centers[0] + pad], dtype=float)
    mids = 0.5 * (centers[:-1] + centers[1:])
    first_pad = max(0.5 * (centers[1] - centers[0]), 1.0e-6)
    last_pad = max(0.5 * (centers[-1] - centers[-2]), 1.0e-6)
    edges = np.concatenate([[centers[0] - first_pad], mids, [centers[-1] + last_pad]])
    return centers, edges


def vivana_fhat_from_theta(theta: np.ndarray, td_params: dict[str, float]) -> np.ndarray:
    theta_arr = np.asarray(theta, dtype=float)
    return np.where(
        theta_arr <= 0.0,
        float(td_params["fhat0"]) + (float(td_params["fhat0"]) - float(td_params["fhat_min"])) * np.sin(theta_arr),
        float(td_params["fhat0"]) + (float(td_params["fhat_max"]) - float(td_params["fhat0"])) * np.sin(theta_arr),
    )


def vivana_baseline_grid(
    *,
    theta_edges: np.ndarray,
    other_edges: np.ndarray,
    td_params: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    row = vivana_fhat_from_theta(theta_centers, td_params)
    grid = np.repeat(row[None, :], repeats=other_edges.size - 1, axis=0)
    counts = np.ones_like(grid)
    return grid, counts


def fill_grid_nearest(grid: np.ndarray) -> np.ndarray:
    filled = np.asarray(grid, dtype=float).copy()
    finite_mask = np.isfinite(filled)
    if np.all(finite_mask):
        return filled
    if not np.any(finite_mask):
        return filled
    finite_coords = np.argwhere(finite_mask)
    finite_values = filled[finite_mask]
    missing_coords = np.argwhere(~finite_mask)
    for coord in missing_coords:
        distances = np.sum((finite_coords - coord) ** 2, axis=1)
        filled[tuple(coord)] = finite_values[int(np.argmin(distances))]
    return filled


def fill_periodic_theta_rows(grid: np.ndarray) -> np.ndarray:
    filled = np.asarray(grid, dtype=float).copy()
    x_all = np.arange(filled.shape[1], dtype=float)
    period = float(filled.shape[1])
    for row_idx in range(filled.shape[0]):
        row = filled[row_idx]
        finite = np.isfinite(row)
        if np.count_nonzero(finite) == 0:
            continue
        if np.count_nonzero(finite) == 1:
            row[~finite] = row[finite][0]
            continue
        x_known = x_all[finite]
        y_known = row[finite]
        x_ext = np.concatenate([x_known - period, x_known, x_known + period])
        y_ext = np.concatenate([y_known, y_known, y_known])
        order = np.argsort(x_ext)
        row[~finite] = np.interp(x_all[~finite], x_ext[order], y_ext[order])
    return filled


def fill_empty_heatmap_bins(grid: np.ndarray, *, dimension: str) -> np.ndarray:
    if not FILL_EMPTY_BINS:
        return grid
    if dimension == "ur":
        return fill_grid_nearest(fill_periodic_theta_rows(grid))
    return fill_grid_nearest(grid)


def add_coverage_overlay(
    *,
    ax: Any,
    theta_edges: np.ndarray,
    other_edges: np.ndarray,
    train_present: np.ndarray,
    unseen_present: np.ndarray,
) -> None:
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    other_centers = 0.5 * (other_edges[:-1] + other_edges[1:])
    if np.any(unseen_present):
        ax.contourf(
            theta_centers,
            other_centers,
            unseen_present.astype(float),
            levels=[0.5, 1.5],
            colors="none",
            hatches=["///"],
            alpha=0.0,
        )
    if np.any(train_present):
        ax.contour(
            theta_centers,
            other_centers,
            train_present.astype(float),
            levels=[0.5],
            colors="black",
            linewidths=0.35,
            alpha=0.7,
        )


def plot_dimension(
    *,
    modules: dict[str, Any],
    samples: dict[str, np.ndarray],
    td_params: dict[str, float],
    dimension: str,
    bins_theta: int,
    bins_other: int,
    out_dir: Path,
) -> Path:
    plt = modules["plt"]
    Normalize = modules["Normalize"]
    theta_edges = np.linspace(-np.pi, np.pi, bins_theta + 1)
    other_centers = None
    if dimension == "ur":
        other_centers, other_edges = categorical_edges(samples[dimension])
    else:
        other_lo, other_hi = finite_min_max(samples[dimension])
        other_edges = np.linspace(other_lo, other_hi, bins_other + 1)

    baseline, baseline_counts = vivana_baseline_grid(
        theta_edges=theta_edges,
        other_edges=other_edges,
        td_params=td_params,
    )
    corrected, corrected_counts = bin_average(
        theta=samples["theta"],
        other=samples[dimension],
        values=samples["fhat_corr"],
        theta_edges=theta_edges,
        other_edges=other_edges,
    )
    train_mask = ~samples["is_unseen"].astype(bool)
    unseen_mask = samples["is_unseen"].astype(bool)
    train_counts = bin_counts(
        theta=samples["theta"],
        other=samples[dimension],
        theta_edges=theta_edges,
        other_edges=other_edges,
        mask=train_mask,
    )
    unseen_counts = bin_counts(
        theta=samples["theta"],
        other=samples[dimension],
        theta_edges=theta_edges,
        other_edges=other_edges,
        mask=unseen_mask,
    )
    train_present = train_counts > 0.0
    unseen_present = unseen_counts > 0.0

    combined = np.concatenate([baseline[np.isfinite(baseline)], corrected[np.isfinite(corrected)]])
    if combined.size == 0:
        raise ValueError(f"No finite binned fhat values for dimension '{dimension}'.")
    norm = Normalize(vmin=float(np.min(combined)), vmax=float(np.max(combined)))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="white")
    baseline_plot = fill_empty_heatmap_bins(baseline, dimension=dimension)
    corrected_plot = fill_empty_heatmap_bins(corrected, dimension=dimension)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True, constrained_layout=True)
    meshes = []
    for ax, grid, title in zip(axes, (baseline_plot, corrected_plot), ("Vivana-TD", "Corrected")):
        mesh = ax.pcolormesh(theta_edges, other_edges, grid, shading="auto", cmap=cmap, norm=norm)
        meshes.append(mesh)
        add_coverage_overlay(
            ax=ax,
            theta_edges=theta_edges,
            other_edges=other_edges,
            train_present=train_present,
            unseen_present=unseen_present,
        )
        ax.set_title(title)
        ax.set_xlabel(r"$\theta$ [rad]")
        ax.set_xlim(-np.pi, np.pi)
    axes[0].set_ylabel({"ur": r"$U_r$", "y": "y", "dy": "dy"}[dimension])
    if other_centers is not None:
        for ax in axes:
            ax.set_yticks(other_centers)
            ax.set_yticklabels([f"{value:g}" for value in other_centers])
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor="none", edgecolor="black", linewidth=0.7, label="train bins"),
        Patch(facecolor="none", edgecolor="0.3", hatch="///", label="unseen bins"),
    ]
    axes[1].legend(handles=legend_handles, loc="upper right", frameon=True, fontsize=8)
    fig.colorbar(meshes[-1], ax=axes, label="fhat")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"fhat_sync_graph_theta_vs_{dimension}.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    samples_n = int(np.count_nonzero(np.isfinite(samples["theta"]) & np.isfinite(samples[dimension])))
    train_n = int(np.count_nonzero(np.isfinite(samples["theta"]) & np.isfinite(samples[dimension]) & train_mask))
    unseen_n = int(np.count_nonzero(np.isfinite(samples["theta"]) & np.isfinite(samples[dimension]) & unseen_mask))
    nonempty = int(np.count_nonzero((baseline_counts + corrected_counts) > 0.0))
    delta_check = np.nanmax(np.abs((samples["fhat_td"] + samples["delta_fhat"]) - samples["fhat_corr"]))
    print(
        f"{dimension}: saved {out_path} | samples={samples_n} train_samples={train_n} "
        f"unseen_samples={unseen_n} nonempty_bins={nonempty} "
        f"fhat_range=[{float(np.nanmin(combined)):.6g}, {float(np.nanmax(combined)):.6g}] "
        f"max|td+delta-corr|={float(delta_check):.3e}"
    )
    return out_path


def main() -> int:
    dimensions = configured_dimensions()
    if BINS_THETA < 2 or BINS_OTHER < 2:
        raise ValueError("BINS_THETA and BINS_OTHER must both be at least 2.")
    if BATCH_SIZE < 1:
        raise ValueError("BATCH_SIZE must be positive.")
    if MAX_FILES is not None and MAX_FILES < 1:
        raise ValueError("MAX_FILES must be positive when provided.")

    modules = require_runtime_modules()
    torch = modules["torch"]
    device = torch.device(DEVICE)

    checkpoint_path = CHECKPOINT_PATH
    data_root = DATA_ROOT
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    ckpt_preview = torch_load_checkpoint(torch, checkpoint_path, map_location="cpu")
    cfg_dict = checkpoint_config_dict(ckpt_preview)
    hnn_cfg = dict(cfg_dict.get("hnn", {}))
    data_cfg = dict(cfg_dict.get("data", {}))
    td_params = modules["resolve_td_correction_params"](hnn_cfg)
    td_memory_cfg = modules["resolve_td_memory_config"](hnn_cfg)
    reduction_factor = int(data_cfg.get("reduction_factor", 20))

    if LEGACY_SPLITS is not None:
        train_paths = collect_npz_paths(data_root, list(LEGACY_SPLITS), MAX_FILES)
        unseen_paths: list[Path] = []
        print(f"Loading {len(train_paths)} trajectory file(s) from {data_root} splits={LEGACY_SPLITS}")
    else:
        unseen_data_root = data_root if UNSEEN_DATA_ROOT is None else UNSEEN_DATA_ROOT
        train_paths = collect_npz_paths(data_root, list(TRAIN_SPLITS), MAX_FILES)
        unseen_paths = collect_npz_paths(unseen_data_root, list(UNSEEN_SPLITS), MAX_FILES)
        print(
            f"Loading train={len(train_paths)} file(s) from {data_root} splits={TRAIN_SPLITS}; "
            f"unseen={len(unseen_paths)} file(s) from {unseen_data_root} splits={UNSEEN_SPLITS}"
        )

    train_trajectories = modules["load_td_correction_trajectories"](
        paths=train_paths,
        cut_start_seconds=0.0,
        reduce_time=True,
        reduction_factor=reduction_factor,
        stagger_reduced_time=False,
        ur_source="stored",
        td_params=td_params,
        td_memory_cfg=td_memory_cfg,
        recompute_td_observables_from_phi=bool(hnn_cfg.get("recompute_td_observables_from_phi", False)),
    )
    unseen_trajectories = (
        modules["load_td_correction_trajectories"](
            paths=unseen_paths,
            cut_start_seconds=0.0,
            reduce_time=True,
            reduction_factor=reduction_factor,
            stagger_reduced_time=False,
            ur_source="stored",
            td_params=td_params,
            td_memory_cfg=td_memory_cfg,
            recompute_td_observables_from_phi=bool(hnn_cfg.get("recompute_td_observables_from_phi", False)),
        )
        if unseen_paths
        else []
    )
    model, _ckpt = load_checkpoint_model(
        modules=modules,
        checkpoint_path=checkpoint_path,
        first_traj=train_trajectories[0],
        device=device,
    )
    print(
        "Evaluating fhat head on real sampled states "
        f"(train_trajectories={len(train_trajectories)}, unseen_trajectories={len(unseen_trajectories)}, "
        f"reduction_factor={reduction_factor}, device={device})"
    )
    sample_groups = [
        evaluate_fhat_samples(
            modules=modules,
            model=model,
            trajectories=train_trajectories,
            td_params=td_params,
            td_memory_cfg=td_memory_cfg,
            device=device,
            batch_size=int(BATCH_SIZE),
            is_unseen=False,
        )
    ]
    if unseen_trajectories:
        sample_groups.append(
            evaluate_fhat_samples(
                modules=modules,
                model=model,
                trajectories=unseen_trajectories,
                td_params=td_params,
                td_memory_cfg=td_memory_cfg,
                device=device,
                batch_size=int(BATCH_SIZE),
                is_unseen=True,
            )
        )
    base_samples = merge_sample_dicts(sample_groups)
    ur_samples = base_samples
    if EXTRA_UNSEEN_UR_VALUES:
        extra_source_trajectories = train_trajectories + unseen_trajectories
        extra_groups = [base_samples]
        for ur_value in EXTRA_UNSEEN_UR_VALUES:
            extra_groups.append(
                evaluate_fhat_samples(
                    modules=modules,
                    model=model,
                    trajectories=extra_source_trajectories,
                    td_params=td_params,
                    td_memory_cfg=td_memory_cfg,
                    device=device,
                    batch_size=int(BATCH_SIZE),
                    is_unseen=True,
                    override_ur=float(ur_value),
                )
            )
        ur_samples = merge_sample_dicts(extra_groups)

    for dim in dimensions:
        plot_dimension(
            modules=modules,
            samples=ur_samples if dim == "ur" else base_samples,
            td_params=td_params,
            dimension=dim,
            bins_theta=int(BINS_THETA),
            bins_other=int(BINS_OTHER),
            out_dir=OUT_DIR,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
