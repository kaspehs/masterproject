"""Plot LOO (leave-one-Ur-out) rollout errors as a function of reduced velocity.

Edit LOO_EXPERIMENTS to overlay multiple sets of LOO models on the same axes.
VIVANA-TD pure baseline is always shown as a dashed black line.
"""
from __future__ import annotations

import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from HNN_helper import (
    AGGREGATE_VALIDATION_ERROR_KEY,
    DISP_STD_REL_ERROR_KEY,
    DOMINANT_FREQ_REL_ERROR_KEY,
    FORCE_DOMINANT_FREQ_REL_ERROR_KEY,
    FORCE_STD_REL_ERROR_KEY,
    PHVIV,
    compute_validation_metrics,
    load_td_correction_trajectories,
    parse_config,
    resolve_td_correction_params,
    resolve_td_memory_config,
    structural_step_constant_force_torch,
    td_baseline_step_torch,
)
from methods.hnn.trainer import (
    _td_correction_state_rollout,
    _td_flow_feature_from_traj,
)

# ── configuration ──────────────────────────────────────────────────────────────

DATA_DIR = ROOT / "CFD_Data" / "npz_exports_td_burnin_trimmed_alltimeseries"

LOO_EXPERIMENTS: list[dict[str, Any]] = [
    {
        "label": "fhat LOO",
        "model_dir": ROOT / "models" / "fhat" / "loo",
        "color": "steelblue",
        "marker": "o",
    },
    # Add more sets here, e.g.:
    # {
    #     "label": "mean LOO",
    #     "model_dir": ROOT / "models" / "mean" / "loo",
    #     "color": "tomato",
    #     "marker": "s",
    # },
]

OUTPUT_PATH = ROOT / "figs" / "loo_errors.png"

# Mapping: checkpoint suffix → (Ur float value, NPZ glob pattern)
UR_MAP: dict[str, tuple[float, str]] = {
    "ur2":   (2.0,  "comb_Ur2__*.npz"),
    "ur4":   (4.0,  "comb_Ur4__*.npz"),
    "ur5":   (5.0,  "comb_Ur5__*.npz"),
    "ur575": (5.75, "comb_Ur575__*.npz"),
    "ur7":   (7.0,  "comb_Ur7__*.npz"),
    "ur8":   (8.0,  "comb_Ur8__*.npz"),
    "ur10":  (10.0, "comb_Ur10__*.npz"),
}

METRICS: list[tuple[str, str]] = [
    (DOMINANT_FREQ_REL_ERROR_KEY,       "Disp. dominant freq.\nrelative error"),
    (DISP_STD_REL_ERROR_KEY,            "Disp. std\nrelative error"),
    (FORCE_DOMINANT_FREQ_REL_ERROR_KEY, "Force dominant freq.\nrelative error"),
    (FORCE_STD_REL_ERROR_KEY,           "Force std\nrelative error"),
    (AGGREGATE_VALIDATION_ERROR_KEY,    "Aggregate error"),
]

_LOO_NAME_RE = re.compile(r"LOO(ur\d+)_", re.IGNORECASE)
_UR_ORDER = sorted(UR_MAP.keys(), key=lambda s: UR_MAP[s][0])

# ── helpers ────────────────────────────────────────────────────────────────────

def _find_loo_checkpoints(model_dir: Path) -> dict[str, Path]:
    """Return {ur_suffix: checkpoint_path} for all LOO checkpoints in model_dir."""
    result: dict[str, Path] = {}
    for pt in sorted(model_dir.glob("*.pt")):
        m = _LOO_NAME_RE.search(pt.name)
        if m is None:
            continue
        suffix = m.group(1).lower()
        if suffix in UR_MAP:
            result[suffix] = pt
    return result


def _load_model(path: Path, device: torch.device) -> tuple[PHVIV, dict]:
    ckpt = torch.load(path, map_location=device)
    config = parse_config(ckpt["config"])
    model, _ = PHVIV.from_config(
        dt=float(ckpt["dt"]),
        cfg=asdict(config.model),
        arch_cfg=asdict(config.architecture),
        device=device,
    )
    state = {
        k.removeprefix("_orig_mod.").removeprefix("module."): v
        for k, v in ckpt["model_state"].items()
    }
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, ckpt


def _load_trajs(suffix: str) -> list[dict[str, np.ndarray]]:
    _, glob_pat = UR_MAP[suffix]
    paths = sorted(DATA_DIR.glob(glob_pat))
    if not paths:
        raise FileNotFoundError(f"No files matching '{glob_pat}' in {DATA_DIR}")
    return load_td_correction_trajectories(paths=paths)


def _avg_metrics(per_traj: list[dict[str, float]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, _ in METRICS:
        vals = [m[key] for m in per_traj if key in m and np.isfinite(float(m[key]))]
        result[key] = float(np.mean(vals)) if vals else float("nan")
    return result


def _eval_loo_model(
    model: PHVIV,
    ckpt: dict,
    trajs: list[dict],
    device: torch.device,
) -> dict[str, float]:
    hnn_cfg = dict(ckpt["config"].get("hnn", {}))
    td_params = resolve_td_correction_params(hnn_cfg)
    td_memory_cfg = resolve_td_memory_config(hnn_cfg)
    mass_key = "dry_mass_kg" if str(hnn_cfg.get("td_mass_source", "dry")) == "dry" else "effective_mass_kg"
    mean_active = bool(ckpt.get("mean_active", False))
    fhat_active = bool(ckpt.get("fhat_active", False))
    predict_sigma = bool(ckpt.get("predict_sigma", False))
    td_force_src = str(ckpt.get("td_force_input_source", "none"))
    fhat_bound_mult = float(ckpt.get("fhat_bound_multiplier", 1.0))
    input_scaling = str(getattr(model, "input_scaling_mode", "current"))

    per_traj: list[dict[str, float]] = []
    with torch.no_grad():
        for traj in trajs:
            mass = float(np.asarray(traj[mass_key]).reshape(()))
            damping = float(np.asarray(traj["damping_c"]).reshape(()))
            stiffness = float(np.asarray(traj["stiffness_n_m"]).reshape(()))
            t_np = np.asarray(traj["t"], dtype=float)
            dt = float(t_np[1] - t_np[0])

            y_t = torch.from_numpy(np.ascontiguousarray(traj["y"])).float().to(device)
            dy_t = torch.from_numpy(np.ascontiguousarray(traj["dy"])).float().to(device)
            td_ctx = torch.from_numpy(np.ascontiguousarray(traj["td_context"])).float().to(device)

            ur_np = _td_flow_feature_from_traj(traj, input_scaling_mode=input_scaling, diameter=float(model.D))
            ur_t = torch.from_numpy(np.ascontiguousarray(ur_np)).float().unsqueeze(1).to(device)

            z0 = torch.cat([y_t[0:1].unsqueeze(1), dy_t[0:1].unsqueeze(1) * mass], dim=1)
            steps = int(y_t.shape[0] - 1)

            z_seq, force_seq, _, _, _ = _td_correction_state_rollout(
                model=model,
                z0=z0,
                ur0=ur_t[0:1],
                td_context0=td_ctx[0:1],
                steps=steps,
                dt=dt,
                structural_mass=torch.full((1, 1), mass, dtype=torch.float32, device=device),
                damping_c=torch.full((1, 1), damping, dtype=torch.float32, device=device),
                stiffness=torch.full((1, 1), stiffness, dtype=torch.float32, device=device),
                td_params=td_params,
                td_memory_cfg=td_memory_cfg,
                mean_active=mean_active,
                sigma_active=predict_sigma,
                fhat_active=fhat_active,
                td_force_input_source=td_force_src,
                fhat_bound_multiplier=fhat_bound_mult,
            )

            y_pred = z_seq[0, :, 0].cpu().numpy()
            vel_pred = (z_seq[0, :, 1] / mass).cpu().numpy()
            force_np = force_seq[0, :, 0].cpu().numpy()

            rollout = {
                "y_norm": y_pred / float(model.D),
                "p_norm": vel_pred / (np.sqrt(stiffness / mass) * float(model.D)),
                "force_total": force_np,
            }
            metrics = compute_validation_metrics(
                model=model,
                y_data_t=y_t,
                val_vel=dy_t,
                reduced_velocity=ur_t[:, 0],
                m_eff=mass,
                dt=dt,
                t=t_np,
                y_data_raw=np.asarray(traj["y"], dtype=float),
                force_data=np.asarray(traj["force_per_m"], dtype=float),
                D=float(model.D),
                k=stiffness,
                device=device,
                rollout=rollout,
            )
            per_traj.append(metrics)

    return _avg_metrics(per_traj)


def _eval_td_baseline(
    trajs: list[dict],
    td_params: dict[str, float],
    rho: float,
    diameter: float,
    device: torch.device,
) -> dict[str, float]:
    per_traj: list[dict[str, float]] = []
    with torch.no_grad():
        for traj in trajs:
            mass = float(np.asarray(traj["dry_mass_kg"]).reshape(()))
            damping = float(np.asarray(traj["damping_c"]).reshape(()))
            stiffness = float(np.asarray(traj["stiffness_n_m"]).reshape(()))
            t_np = np.asarray(traj["t"], dtype=float)
            dt = float(t_np[1] - t_np[0])

            y_t = torch.from_numpy(np.ascontiguousarray(traj["y"])).float().to(device)
            dy_t = torch.from_numpy(np.ascontiguousarray(traj["dy"])).float().to(device)
            ur_t = torch.from_numpy(np.ascontiguousarray(traj["ur"])).float().unsqueeze(1).to(device)
            td_ctx = torch.from_numpy(np.ascontiguousarray(traj["td_context"])).float().to(device)

            mass_t = torch.full((1, 1), mass, dtype=torch.float32, device=device)
            damp_t = torch.full((1, 1), damping, dtype=torch.float32, device=device)
            stiff_t = torch.full((1, 1), stiffness, dtype=torch.float32, device=device)

            z_cur = torch.cat([y_t[0:1].unsqueeze(1), dy_t[0:1].unsqueeze(1) * mass], dim=1)
            td_ctx_cur = td_ctx[0:1].clone()
            y_hist = [float(y_t[0].item())]
            f_hist: list[float] = []

            for _ in range(int(y_t.shape[0] - 1)):
                velocity = z_cur[:, 1:2] / mass
                td_force_next, td_ctx_next = td_baseline_step_torch(
                    velocity=velocity,
                    acceleration=td_ctx_cur[:, 0:1],
                    td_context=td_ctx_cur,
                    dt=dt,
                    rho=rho,
                    diameter=diameter,
                    params=td_params,
                )
                y_next, v_next, a_next = structural_step_constant_force_torch(
                    y=z_cur[:, 0:1],
                    velocity=velocity,
                    force=td_force_next,
                    dt=dt,
                    mass=mass_t,
                    damping_c=damp_t,
                    stiffness=stiff_t,
                )
                z_cur = torch.cat([y_next, v_next * mass], dim=1)
                td_ctx_cur = td_ctx_next.clone()
                td_ctx_cur[:, 0:1] = a_next
                y_hist.append(float(y_next[0, 0].item()))
                f_hist.append(float(td_force_next[0, 0].item()))

            y_pred = np.array(y_hist, dtype=float)
            force_np = np.array(f_hist, dtype=float)

            rollout = {
                "y_norm": y_pred / diameter,
                "p_norm": np.zeros_like(y_pred),
                "force_total": force_np,
            }
            metrics = compute_validation_metrics(
                model=None,
                y_data_t=y_t,
                val_vel=dy_t,
                reduced_velocity=ur_t[:, 0],
                m_eff=mass,
                dt=dt,
                t=t_np,
                y_data_raw=np.asarray(traj["y"], dtype=float),
                force_data=np.asarray(traj["force_per_m"], dtype=float),
                D=diameter,
                k=stiffness,
                device=device,
                rollout=rollout,
            )
            per_traj.append(metrics)

    return _avg_metrics(per_traj)


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    device = torch.device("cpu")

    print("Loading trajectories...")
    trajs_by_ur: dict[str, list[dict]] = {}
    for suffix in _UR_ORDER:
        try:
            trajs_by_ur[suffix] = _load_trajs(suffix)
            n = len(trajs_by_ur[suffix])
            print(f"  Ur={UR_MAP[suffix][0]:g}: {n} timeseries")
        except FileNotFoundError as exc:
            print(f"  Skipping {suffix}: {exc}")

    # Physical params for baseline (extracted from first loaded model)
    ref_td_params: dict[str, float] | None = None
    ref_rho: float | None = None
    ref_diameter: float | None = None

    # Evaluate LOO experiments
    experiment_results: list[dict] = []
    for exp in LOO_EXPERIMENTS:
        label = str(exp["label"])
        model_dir = Path(exp["model_dir"])
        print(f"\nEvaluating '{label}' from {model_dir.name}/")
        checkpoints = _find_loo_checkpoints(model_dir)
        if not checkpoints:
            print(f"  No LOO checkpoints found.")
            continue

        ur_values: list[float] = []
        metric_arrays: dict[str, list[float]] = {key: [] for key, _ in METRICS}

        for suffix in _UR_ORDER:
            if suffix not in checkpoints or suffix not in trajs_by_ur:
                continue
            ur_val = UR_MAP[suffix][0]
            print(f"  Ur={ur_val:g} ...", end=" ", flush=True)
            model, ckpt = _load_model(checkpoints[suffix], device)

            if ref_td_params is None:
                hnn_cfg = dict(ckpt["config"].get("hnn", {}))
                ref_td_params = resolve_td_correction_params(hnn_cfg)
                ref_rho = float(getattr(parse_config(ckpt["config"]).model, "rho", 1000.0))
                ref_diameter = float(model.D)

            metrics = _eval_loo_model(model, ckpt, trajs_by_ur[suffix], device)
            ur_values.append(ur_val)
            for key, _ in METRICS:
                metric_arrays[key].append(metrics.get(key, float("nan")))
            print("done")

        experiment_results.append({
            "label": label,
            "color": exp.get("color", "steelblue"),
            "marker": exp.get("marker", "o"),
            "ur_values": ur_values,
            "metrics": metric_arrays,
        })

    # Evaluate VIVANA-TD baseline
    td_ur_values: list[float] = []
    td_metric_arrays: dict[str, list[float]] = {key: [] for key, _ in METRICS}
    if ref_td_params is not None and ref_rho is not None and ref_diameter is not None:
        print("\nEvaluating VIVANA-TD baseline...")
        for suffix in _UR_ORDER:
            if suffix not in trajs_by_ur:
                continue
            ur_val = UR_MAP[suffix][0]
            print(f"  Ur={ur_val:g} ...", end=" ", flush=True)
            metrics = _eval_td_baseline(trajs_by_ur[suffix], ref_td_params, ref_rho, ref_diameter, device)
            td_ur_values.append(ur_val)
            for key, _ in METRICS:
                td_metric_arrays[key].append(metrics.get(key, float("nan")))
            print("done")
    else:
        print("\nSkipping VIVANA-TD baseline (no model loaded to extract physical params).")

    # ── plot ──────────────────────────────────────────────────────────────────
    n = len(METRICS)
    fig, axes = plt.subplots(1, n, figsize=(3.8 * n, 4.2))
    if n == 1:
        axes = [axes]

    all_ur_ticks = sorted({v for v, _ in UR_MAP.values()})

    for ax, (metric_key, metric_label) in zip(axes, METRICS):
        if td_ur_values:
            ax.plot(
                td_ur_values,
                td_metric_arrays[metric_key],
                color="black",
                linestyle="--",
                marker="x",
                linewidth=1.4,
                markersize=6,
                label="VIVANA-TD baseline",
                zorder=2,
            )
        for res in experiment_results:
            ax.plot(
                res["ur_values"],
                res["metrics"][metric_key],
                color=res["color"],
                marker=res["marker"],
                linestyle="-",
                linewidth=1.4,
                markersize=6,
                label=res["label"],
                zorder=3,
            )
        ax.set_xlabel("Reduced velocity $U_r$")
        ax.set_ylabel(metric_label)
        ax.set_xticks(all_ur_ticks)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)),
               bbox_to_anchor=(0.5, 1.02), frameon=True)
    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    print(f"\nSaved → {OUTPUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
