from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
try:
    from scipy.signal import welch
except ImportError:
    welch = None

from HNN_helper import (
    DISP_SPECTRAL_REL_ERROR_KEY,
    DISP_STD_REL_ERROR_KEY,
    DOMINANT_FREQ_REL_ERROR_KEY,
    FORCE_MAPPING_NRMSE_KEY,
    FORCE_SPECTRAL_REL_ERROR_KEY,
    ROLLOUT_DIVERGED_KEY,
    compute_validation_metrics,
    load_config,
    load_td_correction_trajectories,
    parse_config,
    resolve_cut_start_seconds,
    resolve_td_correction_params,
    structural_step_constant_force_torch,
    td_baseline_step_torch,
)

# Script config: edit these and run the file directly.
CONFIG_PATH = Path("template_configs/vpinn_smoke.yml")
SPLIT = "val"
OUTPUT_CSV: Path | None = Path("CFD_Data/analysis/vivana_td_baseline_val.csv")
PLOT_OUTPUT_DIR: Path | None = Path("CFD_Data/analysis/vivana_td_baseline_plots")
SHOW_PROGRESS = True
PROGRESS_STEP_PERCENT = 5


def _format_value(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{float(value):.6e}"


def _mass_key(method_cfg: dict[str, Any], *, key_name: str) -> str:
    source = str(method_cfg.get(key_name, "dry")).strip().lower()
    if source == "dry":
        return "dry_mass_kg"
    if source == "effective":
        return "effective_mass_kg"
    raise ValueError(f"{key_name} must be one of: dry, effective.")


def _progress_bar(percent: int, *, width: int = 24) -> str:
    filled = max(0, min(width, int(round((float(percent) / 100.0) * width))))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _maybe_print_progress(*, label: str, completed: int, total: int, last_percent: int) -> int:
    if not SHOW_PROGRESS or total <= 0:
        return last_percent
    percent = int((100.0 * float(completed)) / float(total))
    percent = max(0, min(100, percent))
    step = max(1, int(PROGRESS_STEP_PERCENT))
    should_emit = percent >= 100 or last_percent < 0 or (percent - last_percent) >= step
    if not should_emit:
        return last_percent
    end = "\n" if percent >= 100 else "\r"
    print(f"{label} {_progress_bar(percent)} {percent:3d}% ({completed}/{total})", end=end, flush=True)
    return percent


def _load_split_trajectories(config: Any, split: str) -> list[dict[str, np.ndarray]]:
    data_cfg = config.data
    method_name = str(getattr(config, "method", "")).strip().lower()
    if method_name in {"hnn", "phnn"}:
        method_cfg = dict(getattr(config, "hnn", {}) or {})
    elif method_name == "vpinn":
        method_cfg = dict(getattr(config, "vpinn", {}) or {})
    else:
        method_cfg = {}
    ur_source = str(method_cfg.get("td_mass_source", "dry")).strip().lower()
    split = str(split).strip().lower()
    split_dir = Path(data_cfg.train_series_dir) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Expected split directory '{split_dir}'.")
    paths = sorted(split_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No '.npz' files found in '{split_dir}'.")
    return load_td_correction_trajectories(
        paths=paths,
        cut_start_seconds=resolve_cut_start_seconds(data_cfg, split),
        reduce_time=bool(getattr(data_cfg, "reduce_time", False)),
        reduction_factor=int(getattr(data_cfg, "reduction_factor", 1)),
        ur_source=ur_source,
    )


def _baseline_hnn_rollout(
    *,
    z0: torch.Tensor,
    td_context0: torch.Tensor,
    steps: int,
    dt: float,
    structural_mass: torch.Tensor,
    damping_c: torch.Tensor,
    stiffness: torch.Tensor,
    rho: float,
    diameter: float,
    td_params: dict[str, float],
    progress_label: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    z = z0
    td_context = td_context0
    z_hist = [z0]
    force_hist: list[torch.Tensor] = []
    progress = -1
    for step_idx in range(int(steps)):
        velocity = z[:, 1:2] / structural_mass
        td_force_next, td_context_next = td_baseline_step_torch(
            velocity=velocity,
            acceleration=td_context[:, 0:1],
            td_context=td_context,
            dt=dt,
            rho=rho,
            diameter=diameter,
            params=td_params,
        )
        y_next, v_next, a_next = structural_step_constant_force_torch(
            y=z[:, 0:1],
            velocity=velocity,
            force=td_force_next,
            dt=dt,
            mass=structural_mass,
            damping_c=damping_c,
            stiffness=stiffness,
        )
        z = torch.cat([y_next, v_next * structural_mass], dim=1)
        td_context = td_context_next.clone()
        td_context[:, 0:1] = a_next
        z_hist.append(z)
        force_hist.append(td_force_next)
        if progress_label is not None:
            progress = _maybe_print_progress(
                label=progress_label,
                completed=step_idx + 1,
                total=max(1, int(steps)),
                last_percent=progress,
            )
    z_seq = torch.stack(z_hist, dim=1)
    force_seq = torch.stack(force_hist, dim=1) if force_hist else z0.new_zeros((z0.shape[0], 0, 1))
    return z_seq, force_seq


def _baseline_vpinn_rollout(
    *,
    x0: torch.Tensor,
    v0: torch.Tensor,
    td_context0: torch.Tensor,
    steps: int,
    dt: float,
    m: torch.Tensor,
    c: torch.Tensor,
    k: torch.Tensor,
    rho: float,
    diameter: float,
    td_params: dict[str, float],
    progress_label: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = x0
    v = v0
    td_context = td_context0
    xs = [x]
    vs = [v]
    fs: list[torch.Tensor] = []
    progress = -1
    for step_idx in range(int(steps)):
        td_force_next, td_context_next = td_baseline_step_torch(
            velocity=v,
            acceleration=td_context[:, 0:1],
            td_context=td_context,
            dt=dt,
            rho=rho,
            diameter=diameter,
            params=td_params,
        )
        x, v, a_next = structural_step_constant_force_torch(
            y=x,
            velocity=v,
            force=td_force_next,
            dt=dt,
            mass=m,
            damping_c=c,
            stiffness=k,
        )
        td_context = td_context_next.clone()
        td_context[:, 0:1] = a_next
        xs.append(x)
        vs.append(v)
        fs.append(td_force_next)
        if progress_label is not None:
            progress = _maybe_print_progress(
                label=progress_label,
                completed=step_idx + 1,
                total=max(1, int(steps)),
                last_percent=progress,
            )
    x_seq = torch.stack(xs, dim=1)
    v_seq = torch.stack(vs, dim=1)
    f_seq = torch.stack(fs, dim=1) if fs else x0.new_zeros((x0.shape[0], 0, x0.shape[1]))
    return x_seq, v_seq, f_seq


def _force_mapping_nrmse(force_pred: np.ndarray, force_true: np.ndarray) -> float:
    force_pred = np.asarray(force_pred, dtype=float).reshape(-1)
    force_true = np.asarray(force_true, dtype=float).reshape(-1)
    n = min(force_pred.size, force_true.size)
    if n < 1:
        return float("nan")
    force_pred = force_pred[:n]
    force_true = force_true[:n]
    denom = float(np.std(force_true))
    if not np.isfinite(denom) or denom <= 0.0:
        denom = 1.0
    return float(np.sqrt(np.mean((force_pred - force_true) ** 2)) / denom)


def _force_series_with_initial(force_seq: torch.Tensor) -> np.ndarray:
    force_np = force_seq[0, :, 0].detach().cpu().numpy()
    if force_np.size == 0:
        return force_np
    return np.concatenate([force_np[:1], force_np], axis=0)


def _power_spectrum(signal: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(signal, dtype=float).reshape(-1)
    if signal.size < 2 or not np.isfinite(dt) or dt <= 0.0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    centered = signal - np.mean(signal)
    if welch is not None:
        nperseg = min(1024, centered.size)
        if nperseg < 2:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        freqs, psd = welch(centered, fs=1.0 / float(dt), nperseg=nperseg)
        return np.asarray(freqs, dtype=float), np.asarray(psd, dtype=float)
    freqs = np.fft.rfftfreq(centered.size, d=float(dt))
    fft_vals = np.fft.rfft(centered)
    power = (np.abs(fft_vals) ** 2) / max(1, centered.size)
    return np.asarray(freqs, dtype=float), np.asarray(power, dtype=float)


def _save_rollout_plots(
    *,
    traj: dict[str, np.ndarray],
    y_pred: np.ndarray,
    force_pred: np.ndarray,
    plot_dir: Path,
) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    case_name = Path(str(traj["name"])).stem
    t = np.asarray(traj["t"], dtype=float).reshape(-1)
    y_true = np.asarray(traj["y"], dtype=float).reshape(-1)
    force_true = np.asarray(traj["force_per_m"], dtype=float).reshape(-1)
    dt = float(t[1] - t[0]) if t.size > 1 else float("nan")
    ur_value = float(np.asarray(traj["ur"]).reshape(-1)[0])

    y_len = min(t.size, y_true.size, y_pred.size)
    f_len = min(t.size, force_true.size, force_pred.size)

    disp_freqs_true, disp_psd_true = _power_spectrum(y_true[:y_len], dt)
    disp_freqs_pred, disp_psd_pred = _power_spectrum(y_pred[:y_len], dt)
    force_freqs_true, force_psd_true = _power_spectrum(force_true[:f_len], dt)
    force_freqs_pred, force_psd_pred = _power_spectrum(force_pred[:f_len], dt)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"{case_name} | U_r={ur_value:.6g}")

    ax = axes[0, 0]
    ax.plot(t[:y_len], y_true[:y_len], label="Ground truth", linewidth=1.4)
    ax.plot(t[:y_len], y_pred[:y_len], label="Vivana-TD", linewidth=1.2)
    ax.set_title("Displacement Rollout")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(t[:f_len], force_true[:f_len], label="Ground truth", linewidth=1.4)
    ax.plot(t[:f_len], force_pred[:f_len], label="Vivana-TD", linewidth=1.2)
    ax.set_title("Force Rollout")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Force per meter")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    if disp_freqs_true.size > 0:
        ax.semilogy(disp_freqs_true, np.maximum(disp_psd_true, 1e-16), label="Ground truth", linewidth=1.4)
    if disp_freqs_pred.size > 0:
        ax.semilogy(disp_freqs_pred, np.maximum(disp_psd_pred, 1e-16), label="Vivana-TD", linewidth=1.2)
    ax.set_title("Displacement Spectrum")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    if force_freqs_true.size > 0:
        ax.semilogy(force_freqs_true, np.maximum(force_psd_true, 1e-16), label="Ground truth", linewidth=1.4)
    if force_freqs_pred.size > 0:
        ax.semilogy(force_freqs_pred, np.maximum(force_psd_pred, 1e-16), label="Vivana-TD", linewidth=1.2)
    ax.set_title("Force Spectrum")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(plot_dir / f"{case_name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _base_row(traj: dict[str, np.ndarray], dt: float) -> dict[str, float | str]:
    t_np = np.asarray(traj["t"], dtype=float).reshape(-1)
    return {
        "timeseries": str(traj["name"]),
        "ur_value": float(np.asarray(traj["ur"]).reshape(-1)[0]),
        "n_samples": float(int(t_np.shape[0])),
        "dt": float(dt),
        "duration_s": float(t_np[-1] - t_np[0]) if t_np.size > 1 else 0.0,
    }


def _metric_row(metrics: dict[str, float]) -> dict[str, float]:
    return {
        FORCE_MAPPING_NRMSE_KEY: float(metrics.get(FORCE_MAPPING_NRMSE_KEY, float("nan"))),
        DOMINANT_FREQ_REL_ERROR_KEY: float(metrics.get(DOMINANT_FREQ_REL_ERROR_KEY, float("nan"))),
        DISP_STD_REL_ERROR_KEY: float(metrics.get(DISP_STD_REL_ERROR_KEY, float("nan"))),
        DISP_SPECTRAL_REL_ERROR_KEY: float(metrics.get(DISP_SPECTRAL_REL_ERROR_KEY, float("nan"))),
        FORCE_SPECTRAL_REL_ERROR_KEY: float(metrics.get(FORCE_SPECTRAL_REL_ERROR_KEY, float("nan"))),
        ROLLOUT_DIVERGED_KEY: float(metrics.get(ROLLOUT_DIVERGED_KEY, float("nan"))),
    }


def _evaluate_hnn_baseline(config: Any, trajs: list[dict[str, np.ndarray]]) -> list[dict[str, float | str]]:
    hnn_cfg = dict(config.hnn or {})
    model_cfg = config.model
    td_params = resolve_td_correction_params(hnn_cfg)
    mass_key = _mass_key(hnn_cfg, key_name="td_mass_source")
    rho = float(getattr(model_cfg, "rho", 1000.0))
    diameter = float(getattr(model_cfg, "D", 0.1))
    device = torch.device("cpu")
    total_trajs = len(trajs)

    rows: list[dict[str, float | str]] = []
    with torch.no_grad():
        overall_progress = -1
        for traj_idx, traj in enumerate(trajs, start=1):
            if SHOW_PROGRESS:
                print(f"\nTrajectory {traj_idx}/{total_trajs}: {traj['name']}")
            t = torch.from_numpy(np.ascontiguousarray(traj["t"])).float()
            y = torch.from_numpy(np.ascontiguousarray(traj["y"])).float()
            dy = torch.from_numpy(np.ascontiguousarray(traj["dy"])).float()
            ur = torch.from_numpy(np.ascontiguousarray(traj["ur"])).float().unsqueeze(1)
            td_context = torch.from_numpy(np.ascontiguousarray(traj["td_context"])).float()

            dt = float(t[1] - t[0])
            mass_value = float(np.asarray(traj[mass_key]).reshape(()))
            damping_value = float(np.asarray(traj["damping_c"]).reshape(()))
            stiffness_value = float(np.asarray(traj["stiffness_n_m"]).reshape(()))
            z = torch.cat(
                [
                    y.unsqueeze(1),
                    dy.unsqueeze(1) * torch.full((y.shape[0], 1), mass_value, dtype=torch.float32),
                ],
                dim=1,
            )
            z_seq, force_seq = _baseline_hnn_rollout(
                z0=z[0:1],
                td_context0=td_context[0:1],
                steps=max(0, int(z.shape[0] - 1)),
                dt=dt,
                structural_mass=torch.full((1, 1), mass_value, dtype=torch.float32),
                damping_c=torch.full((1, 1), damping_value, dtype=torch.float32),
                stiffness=torch.full((1, 1), stiffness_value, dtype=torch.float32),
                rho=rho,
                diameter=diameter,
                td_params=td_params,
                progress_label="  rollout ",
            )

            y_pred = z_seq[0, :, 0].detach().cpu().numpy()
            vel_pred = (z_seq[0, :, 1] / mass_value).detach().cpu().numpy()
            rollout = {
                "y_norm": y_pred / float(diameter),
                "p_norm": vel_pred / (np.sqrt(stiffness_value / mass_value) * float(diameter)),
                "force_total": _force_series_with_initial(force_seq),
            }
            metrics = compute_validation_metrics(
                model=None,
                y_data_t=y,
                val_vel=dy,
                reduced_velocity=ur,
                m_eff=mass_value,
                dt=dt,
                t=np.asarray(traj["t"], dtype=float),
                y_data_raw=np.asarray(traj["y"], dtype=float),
                force_data=np.asarray(traj["force_per_m"], dtype=float),
                D=float(diameter),
                k=float(stiffness_value),
                device=device,
                rollout=rollout,
            )
            metrics[FORCE_MAPPING_NRMSE_KEY] = _force_mapping_nrmse(
                np.asarray(traj["force_td_per_m"], dtype=float),
                np.asarray(traj["force_per_m"], dtype=float),
            )
            if PLOT_OUTPUT_DIR is not None:
                _save_rollout_plots(
                    traj=traj,
                    y_pred=y_pred,
                    force_pred=np.asarray(rollout["force_total"], dtype=float),
                    plot_dir=PLOT_OUTPUT_DIR / str(SPLIT).strip().lower(),
                )

            row = _base_row(traj, dt)
            row.update(_metric_row(metrics))
            rows.append(row)
            overall_progress = _maybe_print_progress(
                label="overall ",
                completed=traj_idx,
                total=total_trajs,
                last_percent=overall_progress,
            )
    return rows


def _evaluate_vpinn_baseline(config: Any, trajs: list[dict[str, np.ndarray]]) -> list[dict[str, float | str]]:
    vp = dict(config.vpinn or {})
    model_cfg = config.model
    td_params = resolve_td_correction_params(vp)
    mass_key = _mass_key(vp, key_name="td_mass_source")
    rho = float(getattr(model_cfg, "rho", 1000.0))
    diameter = float(getattr(model_cfg, "D", 0.1))
    device = torch.device("cpu")
    total_trajs = len(trajs)

    rows: list[dict[str, float | str]] = []
    with torch.no_grad():
        overall_progress = -1
        for traj_idx, traj in enumerate(trajs, start=1):
            if SHOW_PROGRESS:
                print(f"\nTrajectory {traj_idx}/{total_trajs}: {traj['name']}")
            t = torch.from_numpy(np.ascontiguousarray(traj["t"])).float()
            x = torch.from_numpy(np.ascontiguousarray(traj["y"])).float().unsqueeze(1)
            v = torch.from_numpy(np.ascontiguousarray(traj["dy"])).float().unsqueeze(1)
            ur = torch.from_numpy(np.ascontiguousarray(traj["ur"])).float().unsqueeze(1)
            td_context = torch.from_numpy(np.ascontiguousarray(traj["td_context"])).float()

            dt = float(t[1] - t[0])
            mass_value = float(np.asarray(traj[mass_key]).reshape(()))
            damping_value = float(np.asarray(traj["damping_c"]).reshape(()))
            stiffness_value = float(np.asarray(traj["stiffness_n_m"]).reshape(()))
            x_seq, v_seq, f_seq = _baseline_vpinn_rollout(
                x0=x[0:1],
                v0=v[0:1],
                td_context0=td_context[0:1],
                steps=max(0, int(x.shape[0] - 1)),
                dt=dt,
                m=torch.full((1, 1), mass_value, dtype=torch.float32),
                c=torch.full((1, 1), damping_value, dtype=torch.float32),
                k=torch.full((1, 1), stiffness_value, dtype=torch.float32),
                rho=rho,
                diameter=diameter,
                td_params=td_params,
                progress_label="  rollout ",
            )

            x_pred = x_seq[0, :, 0].detach().cpu().numpy()
            v_pred = v_seq[0, :, 0].detach().cpu().numpy()
            rollout = {
                "y_norm": x_pred / float(diameter),
                "p_norm": v_pred / (np.sqrt(stiffness_value / mass_value) * float(diameter)),
                "force_total": _force_series_with_initial(f_seq),
            }
            metrics = compute_validation_metrics(
                model=None,
                y_data_t=x.squeeze(-1),
                val_vel=v.squeeze(-1),
                reduced_velocity=ur,
                m_eff=mass_value,
                dt=dt,
                t=np.asarray(traj["t"], dtype=float),
                y_data_raw=np.asarray(traj["y"], dtype=float),
                force_data=np.asarray(traj["force_per_m"], dtype=float),
                D=float(diameter),
                k=float(stiffness_value),
                device=device,
                rollout=rollout,
            )
            metrics[FORCE_MAPPING_NRMSE_KEY] = _force_mapping_nrmse(
                np.asarray(traj["force_td_per_m"], dtype=float),
                np.asarray(traj["force_per_m"], dtype=float),
            )
            if PLOT_OUTPUT_DIR is not None:
                _save_rollout_plots(
                    traj=traj,
                    y_pred=x_pred,
                    force_pred=np.asarray(rollout["force_total"], dtype=float),
                    plot_dir=PLOT_OUTPUT_DIR / str(SPLIT).strip().lower(),
                )

            row = _base_row(traj, dt)
            row.update(_metric_row(metrics))
            rows.append(row)
            overall_progress = _maybe_print_progress(
                label="overall ",
                completed=traj_idx,
                total=total_trajs,
                last_percent=overall_progress,
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_table(method: str, split: str, rows: list[dict[str, float | str]]) -> None:
    print(f"Vivana-TD baseline validation errors ({method}, split={split})")
    if not rows:
        print("No rows to display.")
        return

    headers = list(rows[0].keys())
    widths: dict[str, int] = {header: len(header) for header in headers}
    rendered_rows: list[dict[str, str]] = []
    for row in rows:
        rendered: dict[str, str] = {}
        for key, value in row.items():
            if isinstance(value, str):
                text = value
            elif key == "ur_value":
                text = f"{float(value):.6g}" if np.isfinite(value) else "nan"
            elif key == "n_samples":
                text = str(int(value)) if np.isfinite(value) else "nan"
            else:
                text = _format_value(float(value))
            rendered[key] = text
            widths[key] = max(widths[key], len(text))
        rendered_rows.append(rendered)

    print("  ".join(f"{header:{widths[header]}}" for header in headers))
    print("  ".join("-" * widths[header] for header in headers))
    for row in rendered_rows:
        print("  ".join(f"{row[header]:{widths[header]}}" for header in headers))

    metric_keys = [
        FORCE_MAPPING_NRMSE_KEY,
        DOMINANT_FREQ_REL_ERROR_KEY,
        DISP_STD_REL_ERROR_KEY,
        DISP_SPECTRAL_REL_ERROR_KEY,
        FORCE_SPECTRAL_REL_ERROR_KEY,
        ROLLOUT_DIVERGED_KEY,
    ]
    print("\nMean over timeseries:")
    for key in metric_keys:
        vals = [float(row[key]) for row in rows if np.isfinite(float(row[key]))]
        mean_val = float(np.mean(vals)) if vals else float("nan")
        print(f"  {key}: {_format_value(mean_val)}")


def main() -> None:
    raw_cfg = load_config(CONFIG_PATH)
    config = parse_config(raw_cfg)
    split = str(SPLIT).strip().lower()
    if split not in {"train", "val"}:
        raise ValueError("SPLIT must be 'train' or 'val'.")
    trajs = _load_split_trajectories(config, split)

    method = str(getattr(config, "method", "hnn")).strip().lower()
    if method == "hnn":
        hnn_cfg = dict(config.hnn or {})
        if "use_td_correction" in hnn_cfg and not bool(hnn_cfg.get("use_td_correction", True)):
            raise ValueError("PHNN now only supports TD-correction baseline evaluation. Remove hnn.use_td_correction or set it to true.")
        rows = _evaluate_hnn_baseline(config, trajs)
    elif method == "vpinn":
        vpinn_cfg = dict(config.vpinn or {})
        if "use_td_correction" in vpinn_cfg and not bool(vpinn_cfg.get("use_td_correction", True)):
            raise ValueError("VPINN now only supports TD-correction baseline evaluation. Remove vpinn.use_td_correction or set it to true.")
        rows = _evaluate_vpinn_baseline(config, trajs)
    else:
        raise ValueError("This script currently supports only method=hnn or method=vpinn in TD-correction mode.")

    _print_table(method, split, rows)
    if OUTPUT_CSV is not None:
        _write_csv(OUTPUT_CSV, rows)
        print(f"\nWrote CSV to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
