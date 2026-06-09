from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

# Matplotlib/fontconfig need writable cache/config directories in this environment.
cache_root = Path("/tmp") / "phase_burnin_plot_cache"
cache_root.mkdir(parents=True, exist_ok=True)
if "MPLCONFIGDIR" not in os.environ:
    mpl_dir = cache_root / "mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)
if "XDG_CACHE_HOME" not in os.environ:
    xdg_cache = cache_root / "xdg"
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(xdg_cache)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

try:
    from td_hidden_state import (
        compute_theta_series,
        initial_hidden_sigmas,
        initial_phi_dy,
        replay_hidden_state_with_cfd_motion,
        wrap_phase,
    )
except ModuleNotFoundError:
    from vivana_cfd_data_pipeline.vivana_td.td_hidden_state import (
        compute_theta_series,
        initial_hidden_sigmas,
        initial_phi_dy,
        replay_hidden_state_with_cfd_motion,
        wrap_phase,
    )


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
DATA_ROOT = THIS_DIR.parent
RAW_NPZ_DIR = DATA_ROOT / "generated" / "cfd_npz_exports"
TRIMMED_ROOT_CANDIDATES = (
    DATA_ROOT / "generated" / "td_burnin_trimmed",
)
DEFAULT_CASE = "comb_Ur5__1Hydro"
DEFAULT_OUTPUT_DIR = THIS_DIR / "analysis" / "phase_burnin_schematic"
CASE_PARAMSET_RE = re.compile(
    r"cv(?P<Cv>[-0-9.]+)_cd(?P<Cd>[-0-9.]+)_ca(?P<Ca>[-0-9.]+)_c(?P<C>[-0-9.]+)"
    r"_fhat0(?P<fhat0>[-0-9.]+)_band(?P<fhat_min>[-0-9.]+)-(?P<fhat_max>[-0-9.]+)"
)
PHASE_TICKS = [-np.pi, -np.pi / 2.0, 0.0, np.pi / 2.0, np.pi]
PHASE_TICKLABELS = [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"]


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return float(text)


def _normalize(signal: np.ndarray) -> np.ndarray:
    scale = float(np.max(np.abs(signal))) if signal.size else 1.0
    return np.asarray(signal, dtype=float) / max(scale, np.finfo(float).eps)


def _load_manifest_row(case_name: str) -> tuple[Path, dict[str, str]]:
    for root in TRIMMED_ROOT_CANDIDATES:
        manifest_path = root / "burnin_manifest.csv"
        if not manifest_path.exists():
            continue
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("case_name") == case_name:
                    return manifest_path, row
    raise FileNotFoundError(f"Could not find burn-in manifest entry for case '{case_name}'.")


def _find_trimmed_npz(case_name: str) -> Path:
    for root in TRIMMED_ROOT_CANDIDATES:
        if not root.exists():
            continue
        matches = sorted(root.rglob(f"{case_name}.npz"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not find trimmed NPZ for case '{case_name}'.")


def _parse_paramset_id(paramset_id: str) -> dict[str, float]:
    match = CASE_PARAMSET_RE.fullmatch(str(paramset_id).strip())
    if match is None:
        raise ValueError(f"Could not parse TD paramset id '{paramset_id}'.")
    return {key: float(value) for key, value in match.groupdict().items()}


def _compute_motion_phase(
    dy: np.ndarray,
    ddy: np.ndarray,
    sig_dy: np.ndarray,
    sig_ddy: np.ndarray,
    flow_speed_m_s: float,
) -> np.ndarray:
    dy_arr = np.asarray(dy, dtype=float)
    ddy_arr = np.asarray(ddy, dtype=float)
    sig_dy_arr = np.asarray(sig_dy, dtype=float)
    sig_ddy_arr = np.asarray(sig_ddy, dtype=float)
    speed_mag = np.sqrt(float(flow_speed_m_s) ** 2 + dy_arr**2)
    projection = float(flow_speed_m_s) / np.maximum(speed_mag, np.finfo(float).eps)
    dy_r = dy_arr * projection
    ddy_r = ddy_arr * projection
    cos_phi_dy = dy_r / np.clip(sig_dy_arr, np.finfo(float).eps, None)
    sin_phi_dy = -ddy_r / np.clip(sig_ddy_arr, np.finfo(float).eps, None)
    return np.angle(cos_phi_dy + 1j * sin_phi_dy)


def _resolve_memory(
    manifest_row: dict[str, str],
    *,
    dt: float,
    flow_speed_m_s: float,
    diameter_m: float,
    fhat0: float,
) -> tuple[int, float, str]:
    mode = str(manifest_row.get("td_memory_mode") or "tau_over_tref").strip() or "tau_over_tref"
    n_memory_resolved = _maybe_float(manifest_row.get("td_n_memory_resolved"))
    tau_seconds = _maybe_float(manifest_row.get("td_tau_s_resolved"))
    if tau_seconds is None:
        tau_seconds = _maybe_float(manifest_row.get("td_memory_tau_s"))

    if n_memory_resolved is None:
        if mode == "fixed_tau":
            if tau_seconds is None:
                raise ValueError("Need td_memory_tau_s or td_tau_s_resolved for fixed_tau mode.")
            n_memory_resolved = float(tau_seconds) / float(dt)
        elif mode == "fixed_n_memory":
            n_memory_resolved = 500.0
            tau_seconds = float(n_memory_resolved) * float(dt)
        else:
            tau_over_tref = _maybe_float(manifest_row.get("td_tau_over_tref"))
            tau_over_tref = 4.0 if tau_over_tref is None else float(tau_over_tref)
            tau_seconds = float(tau_over_tref) * float(diameter_m) / max(
                float(fhat0) * abs(float(flow_speed_m_s)),
                np.finfo(float).eps,
            )
            n_memory_resolved = float(tau_seconds) / float(dt)

    if tau_seconds is None:
        tau_seconds = float(n_memory_resolved) * float(dt)
    return max(1, int(round(float(n_memory_resolved)))), float(tau_seconds), mode


def _reconstruct_replay(
    raw: np.lib.npyio.NpzFile,
    manifest_row: dict[str, str],
) -> dict[str, Any]:
    time_dim = np.asarray(raw["time_dim"], dtype=float)
    y_dim = np.asarray(raw["y_disp_dim"], dtype=float)
    dy_dim = np.asarray(raw["y_vel_dim"], dtype=float)
    ddy_dim = np.asarray(raw["y_acc_dim"], dtype=float)
    dt_dim = float(np.asarray(raw["dt_dim"]).reshape(()))
    flow_speed_m_s = float(np.asarray(raw["flow_speed_m_s"]).reshape(()))
    rho_kg_m3 = float(np.asarray(raw["rho_kg_m3"]).reshape(()))
    diameter_m = float(np.asarray(raw["diameter_m"]).reshape(()))

    params = _parse_paramset_id(str(manifest_row["td_paramset_id"]))
    n_memory, tau_seconds, memory_mode = _resolve_memory(
        manifest_row,
        dt=dt_dim,
        flow_speed_m_s=flow_speed_m_s,
        diameter_m=diameter_m,
        fhat0=float(params["fhat0"]),
    )

    sig_dy_loc0, sig_ddy_loc0 = initial_hidden_sigmas(
        case_like={"dy_dim": dy_dim, "ddy_dim": ddy_dim, "dt_dim": np.asarray(dt_dim, dtype=float)},
        start_idx=0,
        flow_speed_m_s=flow_speed_m_s,
        n_memory=int(n_memory),
        mode="lookahead_rms",
        window_seconds=float(tau_seconds),
    )
    phi_dy0 = initial_phi_dy(
        dy0=float(dy_dim[0]),
        ddy0=float(ddy_dim[0]),
        sig_dy_loc0=float(sig_dy_loc0),
        sig_ddy_loc0=float(sig_ddy_loc0),
        flow_speed_m_s=flow_speed_m_s,
    )
    theta0_export = _maybe_float(manifest_row.get("theta0_export"))
    theta0_export = 0.0 if theta0_export is None else float(theta0_export)
    phi_vy0 = float(wrap_phase(np.asarray([phi_dy0 - theta0_export], dtype=float))[0])

    replay_params = dict(params)
    replay_params["n_memory"] = float(n_memory)
    replay = replay_hidden_state_with_cfd_motion(
        time=time_dim,
        y=y_dim,
        dy=dy_dim,
        ddy=ddy_dim,
        flow_speed_m_s=flow_speed_m_s,
        rho_kg_m3=rho_kg_m3,
        diameter_m=diameter_m,
        params=replay_params,
        phi_vy0=float(phi_vy0),
        sig_dy_loc0=float(sig_dy_loc0),
        sig_ddy_loc0=float(sig_ddy_loc0),
        n_memory=int(n_memory),
    )
    theta = compute_theta_series(
        dy=dy_dim,
        ddy=ddy_dim,
        phi_vy=np.asarray(replay["phi_vy"], dtype=float),
        sig_dy_loc=np.asarray(replay["sig_dy_loc"], dtype=float),
        sig_ddy_loc=np.asarray(replay["sig_ddy_loc"], dtype=float),
        flow_speed_m_s=flow_speed_m_s,
        mode="principal",
    )
    phi_dy = _compute_motion_phase(
        dy=dy_dim,
        ddy=ddy_dim,
        sig_dy=np.asarray(replay["sig_dy_loc"], dtype=float),
        sig_ddy=np.asarray(replay["sig_ddy_loc"], dtype=float),
        flow_speed_m_s=flow_speed_m_s,
    )
    return {
        "time_dim": time_dim,
        "y_dim": y_dim,
        "dy_dim": dy_dim,
        "ddy_dim": ddy_dim,
        "flow_speed_m_s": flow_speed_m_s,
        "rho_kg_m3": rho_kg_m3,
        "diameter_m": diameter_m,
        "params": replay_params,
        "memory_mode": memory_mode,
        "n_memory": int(n_memory),
        "tau_seconds": float(tau_seconds),
        "phi_vy0": float(phi_vy0),
        "sig_dy_loc0": float(sig_dy_loc0),
        "sig_ddy_loc0": float(sig_ddy_loc0),
        "phi_dy": np.asarray(phi_dy, dtype=float),
        "theta": np.asarray(theta, dtype=float),
        "replay": replay,
    }


def _draw_flow_box(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    body: str,
    facecolor: str,
) -> None:
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.014,rounding_size=0.02",
        linewidth=1.2,
        edgecolor="#243447",
        facecolor=facecolor,
        transform=ax.transAxes,
    )
    ax.add_patch(box)
    ax.text(
        x + 0.02,
        y + height - 0.07,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.4,
        fontweight="bold",
        color="#13212f",
    )
    ax.text(
        x + 0.02,
        y + 0.06,
        body,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.0,
        color="#203040",
        linespacing=1.25,
    )


def _style_phase_axis(ax: plt.Axes) -> None:
    ax.set_yticks(PHASE_TICKS)
    ax.set_yticklabels(PHASE_TICKLABELS)
    ax.set_ylim(-np.pi - 0.2, np.pi + 0.2)
    ax.grid(True, alpha=0.28)


def make_figure(
    *,
    case_name: str,
    manifest_path: Path,
    manifest_row: dict[str, str],
    raw: np.lib.npyio.NpzFile,
    trimmed: np.lib.npyio.NpzFile,
    replay_payload: dict[str, Any],
    output_path: Path,
    pre_cut_seconds: float,
    post_cut_seconds: float,
) -> Path:
    raw_time = np.asarray(raw["time_dim"], dtype=float)
    raw_y = np.asarray(raw["y_disp_dim"], dtype=float)
    raw_dy = np.asarray(raw["y_vel_dim"], dtype=float)
    raw_ddy = np.asarray(raw["y_acc_dim"], dtype=float)
    cut_idx = int(np.asarray(trimmed["burnin_start_idx"]).reshape(()))
    cut_time = float(np.asarray(trimmed["burnin_start_time_dim"]).reshape(()))
    cut_rel = float(cut_time - raw_time[0])

    replay = replay_payload["replay"]
    replay_phi_vy_wrapped = np.asarray(wrap_phase(np.asarray(replay["phi_vy"], dtype=float)), dtype=float)
    replay_phi_dy = np.asarray(replay_payload["phi_dy"], dtype=float)
    replay_theta = np.asarray(replay_payload["theta"], dtype=float)
    replay_fhat = np.asarray(replay["fhat"], dtype=float)

    trimmed_time = np.asarray(trimmed["time_dim"], dtype=float)
    trimmed_phi_vy = np.asarray(trimmed["phi_vy_td"], dtype=float)
    trimmed_phi_vy_wrapped = np.asarray(wrap_phase(trimmed_phi_vy), dtype=float)
    trimmed_theta = np.asarray(trimmed["theta_td"], dtype=float)
    trimmed_sig_dy = np.asarray(trimmed["sig_dy_loc_td"], dtype=float)
    trimmed_sig_ddy = np.asarray(trimmed["sig_ddy_loc_td"], dtype=float)

    post_window_end = min(float(raw_time[-1] - raw_time[0]), cut_rel + float(post_cut_seconds))
    burn_mask = (raw_time - raw_time[0]) <= post_window_end
    motion_time_rel = raw_time[burn_mask] - raw_time[0]

    phase_window_start = max(cut_time - float(pre_cut_seconds), raw_time[0])
    phase_mask = (raw_time >= phase_window_start) & (raw_time <= cut_time)
    phase_time_to_cut = raw_time[phase_mask] - cut_time

    handoff_pre_mask = (raw_time >= phase_window_start) & (raw_time <= cut_time)
    handoff_post_mask = trimmed_time <= cut_time + float(post_cut_seconds) + 1.0e-12
    handoff_pre_time = raw_time[handoff_pre_mask] - cut_time
    handoff_post_time = trimmed_time[handoff_post_mask] - cut_time

    fig = plt.figure(figsize=(15.5, 11.2))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.85, 1.0, 1.0], hspace=0.34, wspace=0.22)

    ax_schema = fig.add_subplot(gs[0, :])
    ax_motion = fig.add_subplot(gs[1, 0])
    ax_phase = fig.add_subplot(gs[1, 1])
    ax_theta = fig.add_subplot(gs[2, 0])
    ax_handoff = fig.add_subplot(gs[2, 1])

    fig.patch.set_facecolor("#f7f5ef")
    for ax in (ax_motion, ax_phase, ax_theta, ax_handoff):
        ax.set_facecolor("#fffdf8")

    ax_schema.set_axis_off()
    _draw_flow_box(
        ax_schema,
        x=0.02,
        y=0.42,
        width=0.18,
        height=0.42,
        title="1. Raw CFD Motion",
        body="Ground-truth\ncross-flow motion\n$y_{CFD}(t)$.",
        facecolor="#d8ecff",
    )
    _draw_flow_box(
        ax_schema,
        x=0.22,
        y=0.42,
        width=0.18,
        height=0.42,
        title="2. Differentiate",
        body="Differentiate to\n$dy_{CFD}(t)$ and\n$ddy_{CFD}(t)$.",
        facecolor="#ffe9c7",
    )
    _draw_flow_box(
        ax_schema,
        x=0.42,
        y=0.42,
        width=0.18,
        height=0.42,
        title="3. Replay TD",
        body="Feed the CFD motion\ninto the TD hidden-state\nupdate only.",
        facecolor="#dff6e3",
    )
    _draw_flow_box(
        ax_schema,
        x=0.62,
        y=0.42,
        width=0.17,
        height=0.42,
        title="4. Sync",
        body="Compute $\\phi_{dy}$,\nthen $\\theta=wrap(\\phi_{dy}-\\phi_{vy})$,\nand advance $\\phi_{vy}$.",
        facecolor="#f5def7",
    )
    _draw_flow_box(
        ax_schema,
        x=0.81,
        y=0.42,
        width=0.17,
        height=0.42,
        title="5. Export",
        body="At the cut,\nkeep $\\phi_{vy}$,\n$\\sigma_{dy}$ and\n$\\sigma_{ddy}$.",
        facecolor="#f9dfd8",
    )

    arrow_y = 0.63
    for x0, x1 in ((0.20, 0.22), (0.40, 0.42), (0.60, 0.62), (0.79, 0.81)):
        ax_schema.add_patch(
            FancyArrowPatch(
                (x0, arrow_y),
                (x1, arrow_y),
                transform=ax_schema.transAxes,
                arrowstyle="-|>",
                mutation_scale=16,
                linewidth=1.6,
                color="#314a5f",
            )
        )

    formula_text = (
        r"$\phi_{dy}=\angle\left(\frac{dy_r}{\sigma_{dy}} - i\frac{ddy_r}{\sigma_{ddy}}\right)$"
        "\n"
        r"$\phi_{vy}^{k+1}=\phi_{vy}^{k}+\omega_{vy}(\theta_k)\,\Delta t,\quad "
        r"\omega_{vy}=2\pi\,\hat f(\theta_k)\,|V_k|/D$"
        "\n"
        r"Trimmed export stores phi_vy_td = replayed phi_vy from the cut onward."
    )
    ax_schema.text(
        0.02,
        0.08,
        formula_text,
        transform=ax_schema.transAxes,
        ha="left",
        va="bottom",
        fontsize=11.0,
        color="#1c2a36",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f2eee2", edgecolor="#d1c6a8"),
    )
    ax_schema.text(
        0.98,
        0.08,
        f"Manifest: {manifest_path.relative_to(REPO_ROOT)}",
        transform=ax_schema.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.3,
        color="#54616d",
    )

    motion_y = _normalize(raw_y[burn_mask])
    motion_dy = _normalize(raw_dy[burn_mask])
    motion_ddy = _normalize(raw_ddy[burn_mask])
    ax_motion.axvspan(0.0, cut_rel, color="#f5c8b8", alpha=0.35, label="Burn-in region removed")
    ax_motion.axvline(cut_rel, color="#c0392b", linestyle="--", linewidth=1.5, label="Trim cut")
    ax_motion.plot(motion_time_rel, motion_y, color="#1f77b4", linewidth=1.8, label=r"$y_{CFD}$ (normalized)")
    ax_motion.plot(motion_time_rel, motion_dy, color="#e67e22", linewidth=1.4, label=r"$dy_{CFD}$ (normalized)")
    ax_motion.plot(motion_time_rel, motion_ddy, color="#16a085", linewidth=1.4, label=r"$ddy_{CFD}$ (normalized)")
    ax_motion.set_title("Ground-truth CFD motion used during burn-in")
    ax_motion.set_xlabel("Time from raw-series start [s]")
    ax_motion.set_ylabel("Normalized amplitude [-]")
    ax_motion.grid(True, alpha=0.28)
    ax_motion.legend(loc="upper left", fontsize=9.4, frameon=True)
    ax_motion.text(
        0.98,
        0.05,
        f"Burn-in removed: {cut_rel:.3f} s\ntrim index: {cut_idx}",
        transform=ax_motion.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.6,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#fff6ef", edgecolor="#d8b5a6"),
    )

    ax_phase.axvline(0.0, color="#c0392b", linestyle="--", linewidth=1.4)
    ax_phase.plot(phase_time_to_cut, replay_phi_dy[phase_mask], color="#1f77b4", linewidth=1.8, label=r"$\phi_{dy}$ from CFD")
    ax_phase.plot(
        phase_time_to_cut,
        replay_phi_vy_wrapped[phase_mask],
        color="#8e44ad",
        linewidth=1.8,
        label=r"Replayed $\phi_{vy}$",
    )
    _style_phase_axis(ax_phase)
    ax_phase.set_title("Phase alignment during the final burn-in seconds")
    ax_phase.set_xlabel("Time relative to trim cut [s]")
    ax_phase.set_ylabel("Wrapped phase [rad]")
    ax_phase.legend(loc="upper left", fontsize=9.4, frameon=True)
    ax_phase.text(
        0.98,
        0.05,
        r"$\theta = wrap(\phi_{dy}-\phi_{vy})$ drives $\hat f$ and $\omega_{vy}$",
        transform=ax_phase.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.4,
        color="#2b3d4f",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#f4eef8", edgecolor="#d4c1df"),
    )

    theta_time_rel = raw_time[: cut_idx + 1] - raw_time[0]
    ax_theta.axvspan(0.0, cut_rel, color="#f5c8b8", alpha=0.35)
    ax_theta.axvline(cut_rel, color="#c0392b", linestyle="--", linewidth=1.5)
    ax_theta.axhline(0.0, color="#566573", linestyle=":", linewidth=1.0)
    ax_theta.plot(theta_time_rel, replay_theta[: cut_idx + 1], color="#2e86c1", linewidth=1.8, label=r"Replay $\theta$")
    trim_theta_time_rel = trimmed_time - raw_time[0]
    trim_theta_mask = trim_theta_time_rel <= post_window_end
    ax_theta.plot(
        trim_theta_time_rel[trim_theta_mask],
        trimmed_theta[trim_theta_mask],
        color="#117a65",
        linewidth=1.6,
        label=r"Stored $\theta_{td}$ after cut",
    )
    ax_theta.set_title(r"Phase lag $\theta$ is what the burn-in is settling")
    ax_theta.set_xlabel("Time from raw-series start [s]")
    ax_theta.set_ylabel(r"$\theta$ [rad]")
    ax_theta.set_yticks(PHASE_TICKS)
    ax_theta.set_yticklabels(PHASE_TICKLABELS)
    ax_theta.grid(True, alpha=0.28)
    ax_theta.legend(loc="upper left", fontsize=9.4, frameon=True)
    ax_theta_r = ax_theta.twinx()
    ax_theta_r.plot(theta_time_rel, replay_fhat[: cut_idx + 1], color="#d35400", linewidth=1.2, alpha=0.55)
    ax_theta_r.set_ylabel(r"$\hat f$ [-]", color="#a04000")
    ax_theta_r.tick_params(axis="y", colors="#a04000")

    ax_handoff.axvspan(-float(pre_cut_seconds), 0.0, color="#fbe6df", alpha=0.65, label="Raw burn-in replay")
    ax_handoff.axvspan(0.0, float(post_cut_seconds), color="#e6f5ea", alpha=0.55, label="Trimmed export")
    ax_handoff.axvline(0.0, color="#c0392b", linestyle="--", linewidth=1.5)
    ax_handoff.plot(
        handoff_pre_time,
        replay_phi_vy_wrapped[handoff_pre_mask],
        color="#8e44ad",
        linewidth=1.8,
        label=r"Replay $\phi_{vy}$ before cut",
    )
    ax_handoff.plot(
        handoff_post_time,
        trimmed_phi_vy_wrapped[handoff_post_mask],
        color="#239b56",
        linewidth=1.8,
        label=r"Stored $\phi_{vy,td}$ after cut",
    )
    _style_phase_axis(ax_handoff)
    ax_handoff.set_xlim(-float(pre_cut_seconds), float(post_cut_seconds))
    ax_handoff.set_title(r"Handoff of the hidden phase into the trimmed training series")
    ax_handoff.set_xlabel("Time relative to trim cut [s]")
    ax_handoff.set_ylabel(r"Wrapped $\phi_{vy}$ [rad]")
    ax_handoff.legend(loc="upper left", fontsize=9.1, frameon=True)
    ax_handoff.text(
        0.98,
        0.05,
        "\n".join(
            [
                f"stored phi_vy_td[0] = {trimmed_phi_vy[0]:.3f} rad",
                f"wrapped phi_vy_td[0] = {trimmed_phi_vy_wrapped[0]:.3f} rad",
                f"sig_dy_loc_td[0] = {trimmed_sig_dy[0]:.3f}",
                f"sig_ddy_loc_td[0] = {trimmed_sig_ddy[0]:.3f}",
                f"theta_td[0] = {trimmed_theta[0]:.3f} rad",
            ]
        ),
        transform=ax_handoff.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.4,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#fffdf8", edgecolor="#c8c0af"),
    )

    fig.suptitle(
        "\n".join(
            [
                f"Phase burn-in schematic for {case_name}",
                (
                    f"Use raw CFD motion to replay the TD hidden state until the trim cut, "
                    f"then export the resulting phase as phi_vy_td."
                ),
            ]
        ),
        fontsize=14.2,
        y=0.985,
    )
    fig.subplots_adjust(top=0.91, left=0.06, right=0.98, bottom=0.06)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a schematic figure explaining the TD phase burn-in driven by CFD motion.",
    )
    parser.add_argument("--case-name", default=DEFAULT_CASE, help="Case name without .npz suffix.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output image path. Defaults to vivana_cfd_data_pipeline/outputs/analysis/phase_burnin_schematic/<case>_phase_burnin_schematic.png",
    )
    parser.add_argument(
        "--pre-cut-seconds",
        type=float,
        default=10.0,
        help="Seconds before the trim cut to show in the phase and handoff panels.",
    )
    parser.add_argument(
        "--post-cut-seconds",
        type=float,
        default=6.0,
        help="Seconds after the trim cut to show in the handoff panels.",
    )
    args = parser.parse_args()

    raw_path = RAW_NPZ_DIR / f"{args.case_name}.npz"
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw CFD NPZ not found: {raw_path}")

    manifest_path, manifest_row = _load_manifest_row(args.case_name)
    trimmed_path = _find_trimmed_npz(args.case_name)
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else (DEFAULT_OUTPUT_DIR / f"{args.case_name}_phase_burnin_schematic.png").resolve()
    )

    raw = np.load(raw_path, allow_pickle=True)
    trimmed = np.load(trimmed_path, allow_pickle=True)
    replay_payload = _reconstruct_replay(raw, manifest_row)

    saved_path = make_figure(
        case_name=args.case_name,
        manifest_path=manifest_path,
        manifest_row=manifest_row,
        raw=raw,
        trimmed=trimmed,
        replay_payload=replay_payload,
        output_path=output_path,
        pre_cut_seconds=float(args.pre_cut_seconds),
        post_cut_seconds=float(args.post_cut_seconds),
    )
    print(saved_path)


if __name__ == "__main__":
    main()
