from __future__ import annotations

import csv
import math
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


def _bootstrap_paths() -> Path:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    for p in (project_root, script_dir):
        p_str = str(p)
        if p_str not in sys.path:
            sys.path.insert(0, p_str)
    return project_root


PROJECT_ROOT = _bootstrap_paths()

from HNN_helper import PHVIV, compute_velocity_numpy, parse_config, rollout_model  # noqa: E402
from methods.vpinn.trainer import (  # noqa: E402
    ScaledForceWrapper,
    _as_diag_param,
    _build_force_model,
    _m_eff_from_model_cfg,
    rollout_rk4,
)
from Experimental_Data import export_mat_to_npz_channels as mat_export  # noqa: E402

# ----------------------
# User configuration
# ----------------------
# Folder containing source .mat files.
MAT_DIR = PROJECT_ROOT / "Experimental_Data/CrossFlow/RawDatCleanedCorrectedSmoothedData"
# TensorBoard run folder for the trained model to evaluate.
TB_DIR = PROJECT_ROOT / "Experimental_Data/eval_models/phnnE2_pirate_sym3_0303-124718"
# Optional explicit checkpoint path. If None, checkpoint is auto-resolved from TB_DIR.
CHECKPOINT_PATH: Path | None = None
# Output folder for figures and CSV summary.
OUTPUT_DIR = PROJECT_ROOT / "Experimental_Data/rollout_eval_outputs"
MAT_GLOB = "*.mat"
# MAT variable name to read. Set to None for auto-detect.
DATA_VARIABLE: str | None = "data"
DEVICE = "cpu"  # "cpu" or "cuda"
# Choose rollout U_r input:
# - "mean": use one constant mean U_r value per file
# - "instantaneous": use per-sample instantaneous U_r
UR_INPUT_MODE = "mean"

def _load_state(model: torch.nn.Module, state: dict[str, Any]) -> None:
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    if any(k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)


def _resolve_checkpoint(tb_dir: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"Checkpoint '{explicit}' not found.")
        return explicit.resolve()

    tb_dir = tb_dir.resolve()
    async_dir = tb_dir / "async_validation"
    if async_dir.exists():
        async_ckpts = sorted(async_dir.glob("epoch_*.pt"))
        if async_ckpts:
            return async_ckpts[-1].resolve()

    run_name = tb_dir.name
    candidate = (PROJECT_ROOT / "models" / f"{run_name}.pt").resolve()
    if candidate.exists():
        return candidate

    local_ckpts = sorted(tb_dir.glob("*.pt"))
    if local_ckpts:
        return local_ckpts[-1].resolve()

    raise FileNotFoundError(
        "Could not resolve checkpoint. Checked:\n"
        f"1) explicit path\n2) {async_dir}\n3) {candidate}\n4) {tb_dir}/*.pt"
    )


def _safe_stem(path: Path) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in path.stem)


def _load_mat_timeseries(mat_path: Path, data_variable: str | None) -> dict[str, np.ndarray]:
    matrix, channel_names = mat_export._load_data_matrix(mat_path, data_variable)
    channels = mat_export._extract_channels(matrix, channel_names)

    time = np.asarray(channels["time"], dtype=float).reshape(-1)
    disp = np.asarray(channels["y"], dtype=float).reshape(-1)
    vel = np.asarray(channels["dy"], dtype=float).reshape(-1)
    ur_inst = np.asarray(channels["U_r_inst"], dtype=float).reshape(-1)
    ur_mode = str(UR_INPUT_MODE).strip().lower()
    if ur_mode == "mean":
        ur_mean = float(np.mean(ur_inst))
        ur = np.full(ur_inst.shape, ur_mean, dtype=float)
    elif ur_mode == "instantaneous":
        ur = ur_inst
    else:
        raise ValueError("UR_INPUT_MODE must be one of: 'mean', 'instantaneous'")
    force = np.asarray(channels["cf_force"], dtype=float).reshape(-1)

    n = int(min(time.size, disp.size, vel.size, ur.size, force.size))
    if n < 3:
        raise ValueError(f"{mat_path.name}: too few aligned samples ({n}).")
    return {
        "time": time[:n],
        "y_true": disp[:n],
        "v_true": vel[:n],
        "ur": ur[:n],
        "force_true": force[:n],
    }


def _align(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = int(min(a.size, b.size))
    return np.asarray(a[:n], dtype=float), np.asarray(b[:n], dtype=float)


def _error_metrics(pred: np.ndarray, true: np.ndarray, prefix: str) -> dict[str, float]:
    pred_a, true_a = _align(pred, true)
    if pred_a.size == 0:
        return {}
    diff = pred_a - true_a
    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))
    std_true = float(np.std(true_a))
    if std_true <= 0.0:
        std_true = 1.0
    nrmse = rmse / std_true
    return {
        f"{prefix}_rmse": rmse,
        f"{prefix}_nrmse": nrmse,
        f"{prefix}_mae": mae,
        f"{prefix}_bias": bias,
        f"{prefix}_max_abs": float(np.max(np.abs(diff))),
    }


def _build_phnn_model_for_dt(ckpt: dict[str, Any], cfg: Any, dt: float, device: torch.device) -> tuple[PHVIV, dict[str, float]]:
    model_dict = asdict(cfg.model)
    arch_dict = asdict(cfg.architecture)
    model, derived = PHVIV.from_config(dt=dt, cfg=model_dict, arch_cfg=arch_dict, device=device)
    _load_state(model, ckpt["model_state"])
    model.eval()
    return model, derived


def _build_vpinn_model(
    ckpt: dict[str, Any],
    cfg: Any,
    device: torch.device,
    d: int,
) -> tuple[torch.nn.Module, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    vp = dict(cfg.vpinn or {})
    force_representation = str(vp.get("force_representation", "force")).strip().lower()
    use_force_coeff = force_representation == "coefficient"

    m = _as_diag_param(vp.get("m", _m_eff_from_model_cfg(cfg.model)), d, device, "m")
    c = _as_diag_param(vp.get("c", getattr(cfg.model, "damping_c", 1e-4)), d, device, "c")
    k = _as_diag_param(vp.get("k", getattr(cfg.model, "k", 1218.0)), d, device, "k")

    input_dim = 2 * d + 1
    output_dim = d
    base_model = _build_force_model(cfg, input_dim=input_dim, output_dim=output_dim).to(device)
    use_input_scaling = bool(vp.get("use_input_scaling", False))
    if use_input_scaling:
        D_val = float(getattr(cfg.model, "D", 1.0))
        x_scale = D_val if np.isfinite(D_val) and D_val != 0.0 else 1.0
        omega = torch.sqrt(torch.clamp(k / m, min=1e-12))
        v_scale = omega * float(x_scale)
        ur_scale = float(vp.get("ur_scale", 10.0))
        f_scale = 1.0 if use_force_coeff else k * float(x_scale)
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
    return model, m, c, k, use_force_coeff


def _vpinn_f0_from_ur(cfg: Any, ur_series: np.ndarray) -> float:
    m_eff = float(_m_eff_from_model_cfg(cfg.model))
    k_val = float(getattr(cfg.model, "k", 1218.0))
    D = float(getattr(cfg.model, "D", 0.1))
    rho = float(getattr(cfg.model, "rho", 1000.0))
    if m_eff <= 0.0 or k_val <= 0.0:
        raise ValueError("Invalid model m/k for F0 fallback.")
    fn_hz = math.sqrt(k_val / m_eff) / (2.0 * math.pi)
    ur_finite = np.asarray(ur_series, dtype=float).reshape(-1)
    ur_finite = ur_finite[np.isfinite(ur_finite)]
    if ur_finite.size == 0:
        raise ValueError("Cannot compute F0 fallback: no finite U_r values.")
    U = float(np.mean(ur_finite)) * fn_hz * D
    f0 = 0.5 * rho * D * U * U
    if not np.isfinite(f0) or f0 <= 0.0:
        raise ValueError(f"Invalid F0 fallback value: {f0}")
    return float(f0)


def _plot_series(
    *,
    out_path: Path,
    title: str,
    t: np.ndarray,
    y_true: np.ndarray,
    y_roll: np.ndarray,
    f_true: np.ndarray,
    f_roll: np.ndarray,
    f_map: np.ndarray,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib") from exc

    t_y = t[: min(y_true.size, y_roll.size)]
    yt, yr = _align(y_true, y_roll)
    t_f = t[: min(f_true.size, f_roll.size, f_map.size)]
    ft, fr = _align(f_true, f_roll)
    fm = np.asarray(f_map[: t_f.size], dtype=float)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=False)
    ax1, ax2, ax3, ax4 = axes

    ax1.plot(t_y, yt, label="true")
    ax1.plot(t_y, yr, label="rollout")
    ax1.set_title(f"{title} | displacement")
    ax1.set_ylabel("y")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    ax2.plot(t_y, yr - yt, color="tab:red")
    ax2.set_title("displacement error (rollout - true)")
    ax2.set_ylabel("dy")
    ax2.grid(True, alpha=0.3)

    ax3.plot(t_f, ft, label="true force")
    ax3.plot(t_f, fr, label="rollout force")
    ax3.plot(t_f, fm, label="mapping force")
    ax3.set_title("force timeseries")
    ax3.set_ylabel("force")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="best")

    ax4.plot(t_f, fr - ft, label="rollout - true")
    ax4.plot(t_f, fm - ft, label="mapping - true")
    ax4.set_title("force errors")
    ax4.set_xlabel("time [s]")
    ax4.set_ylabel("dF")
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc="best")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    mat_dir = MAT_DIR.resolve()
    tb_dir = TB_DIR.resolve()
    output_dir = OUTPUT_DIR.resolve()

    if not mat_dir.exists():
        raise FileNotFoundError(f"MAT directory '{mat_dir}' not found.")
    if not tb_dir.exists():
        raise FileNotFoundError(f"TensorBoard directory '{tb_dir}' not found.")

    checkpoint_path = _resolve_checkpoint(tb_dir, CHECKPOINT_PATH)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = parse_config(ckpt.get("config", {}))
    method = str(ckpt.get("method", cfg.method)).strip().lower()
    device = torch.device(DEVICE if DEVICE == "cuda" and torch.cuda.is_available() else "cpu")

    mat_files = sorted(mat_dir.glob(MAT_GLOB))
    if not mat_files:
        raise FileNotFoundError(f"No MAT files found in '{mat_dir}' with pattern '{MAT_GLOB}'.")

    data_variable = DATA_VARIABLE
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Method: {method}")
    print(f"MAT files: {len(mat_files)}")
    print(f"Output dir: {output_dir}")
    print(f"U_r input mode: {UR_INPUT_MODE}")

    rows: list[dict[str, float | str]] = []
    phnn_cache: dict[float, tuple[PHVIV, dict[str, float]]] = {}
    vpinn_bundle: tuple[torch.nn.Module, torch.Tensor, torch.Tensor, torch.Tensor, bool] | None = None

    for mat_path in mat_files:
        ts = _load_mat_timeseries(mat_path, data_variable)
        t = ts["time"]
        y_true = ts["y_true"]
        v_true = ts["v_true"]
        ur = ts["ur"]
        force_true = ts["force_true"]
        dt = float(np.median(np.diff(t)))
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"{mat_path.name}: invalid dt={dt}")

        if method in {"hnn", "phnn"}:
            if dt not in phnn_cache:
                phnn_cache[dt] = _build_phnn_model_for_dt(ckpt, cfg, dt=dt, device=device)
            model, derived = phnn_cache[dt]

            y_t = torch.from_numpy(y_true).float().to(device)
            v_eval = compute_velocity_numpy(
                y_true,
                dt,
                use_savgol=bool(cfg.smoothing.use_savgol_smoothing),
                savgol_window=int(cfg.smoothing.window_length),
                savgol_polyorder=int(cfg.smoothing.polyorder),
            )
            v_t = torch.from_numpy(v_eval).float().to(device)
            ur_t = torch.from_numpy(ur).float().to(device)

            rollout = rollout_model(
                model,
                y_t,
                v_t,
                ur_t,
                float(derived["m_eff"]),
                dt,
                t,
                float(derived["D"]),
                float(derived["k"]),
                device,
            )
            y_roll = np.asarray(rollout["y_norm"], dtype=float) * float(derived["D"])
            force_roll = np.asarray(rollout["force_total"], dtype=float).reshape(-1)

            with torch.no_grad():
                z_true = torch.stack((y_t, v_t * float(derived["m_eff"])), dim=1)
                force_map = model.u_theta(z_true, reduced_velocity=ur_t).squeeze(-1).detach().cpu().numpy()

        elif method == "vpinn":
            if vpinn_bundle is None:
                vpinn_bundle = _build_vpinn_model(ckpt, cfg, device, d=1)
            model, m, c, k, use_force_coeff = vpinn_bundle

            x_t = torch.from_numpy(y_true.reshape(-1, 1)).float().to(device)
            v_t = torch.from_numpy(v_true.reshape(-1, 1)).float().to(device)
            ur_t = torch.from_numpy(ur.reshape(-1, 1)).float().to(device)

            f0 = None
            f0_scalar = None
            if use_force_coeff:
                f0_scalar = _vpinn_f0_from_ur(cfg, ur)
                f0 = torch.tensor([[float(f0_scalar)]], dtype=torch.float32, device=device)

            steps = int(x_t.shape[0] - 1)
            if steps < 1:
                raise ValueError(f"{mat_path.name}: not enough samples for rollout.")
            x_seq, _v_seq, f_seq = rollout_rk4(
                model=model,
                x0=x_t[0:1, :],
                v0=v_t[0:1, :],
                ur0=ur_t[0:1, :],
                steps=steps,
                dt=dt,
                m=m,
                c=c,
                k=k,
                f0=f0,
            )
            y_roll = x_seq[0, :, 0].detach().cpu().numpy()
            force_roll = f_seq[0, :, 0].detach().cpu().numpy()
            with torch.no_grad():
                force_map = model(torch.cat([x_t, v_t, ur_t], dim=-1))[:, 0].detach().cpu().numpy()
            if use_force_coeff and f0_scalar is not None:
                force_roll = force_roll * float(f0_scalar)
                force_map = force_map * float(f0_scalar)
        else:
            raise ValueError(f"Unsupported checkpoint method '{method}'.")

        metrics: dict[str, float | str] = {"file": mat_path.name, "U_r_mean": float(np.mean(ur))}
        metrics.update(_error_metrics(y_roll, y_true, "disp_rollout"))
        metrics.update(_error_metrics(force_roll, force_true, "force_rollout"))
        metrics.update(_error_metrics(force_map, force_true, "force_mapping"))
        rows.append(metrics)

        fig_dir = output_dir / "figures"
        fig_path = fig_dir / f"{_safe_stem(mat_path)}.png"
        _plot_series(
            out_path=fig_path,
            title=mat_path.stem,
            t=t,
            y_true=y_true,
            y_roll=y_roll,
            f_true=force_true,
            f_roll=force_roll,
            f_map=force_map,
        )
        print(f"[OK] {mat_path.name}: wrote {fig_path}")

    summary_path = output_dir / "metrics_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
