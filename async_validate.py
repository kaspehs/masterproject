"""
Asynchronous validation runner.

Loads a checkpoint saved during training and runs validation on the val split.
Intended to be spawned as a child process so training can continue.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional
import os
import sys

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from dataclasses import asdict

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from HNN_helper import (
    PHVIV,
    build_dataloader_from_series,
    compute_validation_metrics,
    load_training_series,
    log_validation_epoch,
    parse_config,
    preprocess_timeseries,
    resolve_cut_start_seconds,
)
from methods.vpinn.trainer import (
    _apply_per_traj_scale,
    _force_mapping_nrmse_over_trajs,
    _load_metadata_map,
    ScaledForceWrapper,
    WindowDataset,
    _as_diag_param,
    _build_force_model,
    _evaluate_epoch,
    _infer_dt_target_from_data_cfg,
    _load_trajectory,
    _m_eff_from_model_cfg,
    _test_functions,
    _log_rollout_validation,
)


def _set_threading(num_threads: int) -> None:
    num_threads = max(1, int(num_threads))
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(max(1, min(4, num_threads)))


def _rollout_index(epoch: int, rollout_every: int, num_series: int, cycle: bool) -> int:
    if num_series <= 0:
        return 0
    if not cycle:
        return 0
    step = max(0, (epoch + 1) // max(1, int(rollout_every)) - 1)
    return int(step % num_series)


def _load_checkpoint(path: Path) -> tuple[dict[str, Any], Any, str]:
    ckpt = torch.load(path, map_location="cpu")
    cfg_raw = ckpt.get("config", {})
    cfg = parse_config(cfg_raw)
    method = str(ckpt.get("method", cfg.method)).strip().lower()
    return ckpt, cfg, method


def _load_state(model: torch.nn.Module, state: dict[str, Any]) -> None:
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    if any(k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)


def _run_hnn_validation(
    *,
    ckpt: dict[str, Any],
    cfg: Any,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    rollout_every: int,
    cycle_rollout: bool,
    do_losses: bool,
    do_rollout: bool,
    num_workers: int,
) -> None:
    data_cfg = cfg.data
    hnn_cfg = dict(cfg.hnn or {})
    velocity_source = str(hnn_cfg.get("velocity_source", "compute")).strip().lower()
    per_traj_norm = str(hnn_cfg.get("per_traj_norm", "none")).strip().lower()
    per_traj_norm_eps = float(hnn_cfg.get("per_traj_norm_eps", 1e-8))
    if per_traj_norm not in {"none", "force_rms"}:
        raise ValueError("hnn.per_traj_norm must be one of: none, force_rms.")
    loss_cfg = cfg.loss

    if bool(getattr(data_cfg, "use_generated_train_series", False)):
        val_dir = Path(data_cfg.train_series_dir) / "val"
        if not val_dir.exists():
            raise FileNotFoundError(f"Validation directory '{val_dir}' not found.")
        val_files = sorted(val_dir.glob("*.npz"))
        if not val_files:
            raise FileNotFoundError(f"No '.npz' files found in '{val_dir}'.")
        data_path = val_files[0]
    else:
        data_path = Path(data_cfg.file)

    with np.load(data_path) as data:
        t = np.asarray(data["a"])
        y_data = np.asarray(data["b"])
        F_data = np.asarray(data["c"])
        H_data = np.asarray(data["d"]) if "d" in data else np.zeros_like(y_data)
        if "U_r" not in data:
            raise KeyError(f"{data_path} is missing reduced velocity 'U_r'.")
        reduced_velocity = np.asarray(data["U_r"])
        vel_data = None
        for key in ("e", "dy", "v"):
            if key in data:
                vel_data = np.asarray(data[key])
                break

    val_cut = resolve_cut_start_seconds(data_cfg, "val")
    t, y_data, F_data, hamiltonian_data, vel_data, dt = preprocess_timeseries(
        t,
        y_data,
        F_data,
        H_data,
        data_cfg,
        velocity=vel_data,
        cut_start_seconds=val_cut,
    )

    model_dict = asdict(cfg.model)
    arch_dict = asdict(cfg.architecture)
    model, derived = PHVIV.from_config(dt=float(dt), cfg=model_dict, arch_cfg=arch_dict, device=device)
    _load_state(model, ckpt["model_state"])
    model.eval()

    m_eff = float(derived["m_eff"])
    D = float(derived["D"])
    k = float(derived["k"])

    if bool(getattr(data_cfg, "use_generated_train_series", False)):
        series_dir = Path(data_cfg.train_series_dir) / "val"
        val_series_raw, _ = load_training_series(
            y_data,
            t,
            dt,
            True,
            series_dir,
            m_eff,
            device,
            smoothing_cfg=cfg.smoothing,
            velocity_source=velocity_source,
            eval_velocity=vel_data,
            eval_reduced_velocity=reduced_velocity,
            require_force=True,
            eval_force=F_data,
            cut_start_seconds=val_cut,
        )
    else:
        val_series_raw, _ = load_training_series(
            y_data,
            t,
            dt,
            False,
            Path("."),
            m_eff,
            device,
            smoothing_cfg=cfg.smoothing,
            velocity_source=velocity_source,
            eval_velocity=vel_data,
            eval_reduced_velocity=reduced_velocity,
            require_force=True,
            eval_force=F_data,
            cut_start_seconds=val_cut,
        )

    val_loader, val_sequences, _ = build_dataloader_from_series(
        val_series_raw,
        m_eff=m_eff,
        batch_size=int(cfg.training.batch_size),
        device=device,
        smoothing_cfg=cfg.smoothing,
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=(device.type == "cuda"),
        per_traj_norm=per_traj_norm,
        per_traj_norm_eps=per_traj_norm_eps,
    )

    if do_losses:
        amp_enabled = bool(cfg.precision.use_amp) and device.type == "cuda"
        loss_metrics = _evaluate_val_losses(
            model=model,
            loader=val_loader,
            device=device,
            non_blocking=(device.type == "cuda"),
            force_reg=float(loss_cfg.force_reg),
            force_reg_on_coeff=bool(getattr(loss_cfg, "force_reg_on_coeff", False)),
            use_force_data_loss=bool(getattr(loss_cfg, "use_force_data_loss", False)),
            force_data_weight=float(getattr(loss_cfg, "force_data_weight", 1.0)),
            amp_enabled=amp_enabled,
            amp_dtype=_amp_dtype(cfg.precision.amp_dtype),
            per_traj_norm_eps=per_traj_norm_eps,
        )
        for name, value in loss_metrics.items():
            writer.add_scalar(f"val/{name}", value, epoch)

    if do_rollout:
        metrics_sum: dict[str, float] = {}
        count = 0
        for series_raw, sequence in zip(val_series_raw, val_sequences):
            y_np, t_np, dt_value, _vel_np, force_np, _ur_np = series_raw
            y_tensor, vel_tensor, _t_tensor, ur_tensor = sequence
            metrics = compute_validation_metrics(
                model=model,
                y_data_t=y_tensor,
                val_vel=vel_tensor,
                reduced_velocity=ur_tensor,
                m_eff=m_eff,
                dt=dt_value,
                t=t_np,
                y_data_raw=y_np,
                force_data=force_np,
                D=D,
                k=k,
                device=device,
                log_extra_metrics=bool(getattr(cfg.monitoring, "log_extra_validation_metrics", False)),
            )
            for name, value in metrics.items():
                metrics_sum[name] = metrics_sum.get(name, 0.0) + float(value)
            count += 1
        if count > 0:
            for name, total in metrics_sum.items():
                writer.add_scalar(f"val/{name}", total / float(count), epoch)

        rollout_idx = _rollout_index(epoch, rollout_every, len(val_series_raw), cycle_rollout)
        y_np, t_np, dt_value, _vel_np, force_np, _ur_np = val_series_raw[rollout_idx]
        y_tensor, vel_tensor, _t_tensor, ur_tensor = val_sequences[rollout_idx]
        log_validation_epoch(
            writer,
            epoch,
            model,
            y_tensor,
            vel_tensor,
            ur_tensor,
            m_eff,
            dt_value,
            t_np,
            y_np / D,
            y_np,
            force_np,
            D,
            k,
            device,
            getattr(data_cfg, "middle_time_plot", [0.0, 1.0]),
            hamiltonian_data,
            log_extra_metrics=bool(getattr(cfg.monitoring, "log_extra_validation_metrics", False)),
            log_metrics=False,
        )
    else:
        force_map_sum = 0.0
        count = 0
        for series_raw, sequence in zip(val_series_raw, val_sequences):
            _y_np, _t_np, _dt_value, _vel_np, force_np, _ur_np = series_raw
            if force_np is None:
                continue
            y_tensor, vel_tensor, _t_tensor, ur_tensor = sequence
            z_true = torch.stack((y_tensor, vel_tensor * m_eff), dim=1).to(device=device)
            rv = ur_tensor.to(device=device)
            with torch.no_grad():
                force_on_data = model.u_theta(z_true, reduced_velocity=rv).squeeze(-1).detach().cpu().numpy()
            force_target = np.asarray(force_np).reshape(-1)
            min_len = min(force_on_data.shape[0], force_target.shape[0])
            if min_len <= 0:
                continue
            force_pred = force_on_data[:min_len]
            force_true = force_target[:min_len]
            rmse = float(np.sqrt(np.mean((force_pred - force_true) ** 2)))
            force_std = float(np.std(force_true))
            if force_std <= 0.0:
                force_std = 1.0
            force_map_sum += rmse / force_std
            count += 1
        if count > 0:
            writer.add_scalar("val/force_mapping_nrmse_on_data", force_map_sum / float(count), epoch)


def _run_vpinn_validation(
    *,
    ckpt: dict[str, Any],
    cfg: Any,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    rollout_every: int,
    cycle_rollout: bool,
    do_losses: bool,
    do_rollout: bool,
    num_workers: int,
) -> None:
    data_cfg = cfg.data
    vp = dict(cfg.vpinn or {})
    velocity_source = str(vp.get("velocity_source", "compute")).strip().lower()
    force_representation = str(vp.get("force_representation", "force")).strip().lower()
    if force_representation not in {"force", "coefficient"}:
        raise ValueError("vpinn.force_representation must be one of: force, coefficient.")
    num_poly = int(vp.get("num_poly_test", 2))
    num_sine = int(vp.get("num_sine_test", 0))

    if bool(getattr(data_cfg, "use_generated_train_series", False)):
        val_dir = Path(data_cfg.train_series_dir) / "val"
        if not val_dir.exists():
            raise FileNotFoundError(f"Validation directory '{val_dir}' not found.")
        val_files = sorted(val_dir.glob("*.npz"))
        if not val_files:
            raise FileNotFoundError(f"No '.npz' files found in '{val_dir}'.")
        sources = val_files
    else:
        sources = [Path(data_cfg.file)]

    val_trajs: list[dict[str, Any]] = []
    dt_ref: Optional[float] = None
    cut_start_seconds = float(
        data_cfg.cut_start_seconds_val
        if getattr(data_cfg, "cut_start_seconds_val", None) is not None
        else getattr(data_cfg, "cut_start_seconds", 0.0)
    )
    dt_target = vp.get("dt_target", None)
    if dt_target is None:
        dt_target = _infer_dt_target_from_data_cfg(data_cfg)
    dt_target = None if dt_target is None else float(dt_target)

    f0_lookup = None
    if force_representation == "coefficient":
        if bool(getattr(data_cfg, "use_generated_train_series", False)):
            meta_path = Path(data_cfg.train_series_dir) / "metadata.json"
        else:
            meta_path = Path(data_cfg.file).resolve().parent / "metadata.json"
        f0_lookup = _load_metadata_map(meta_path)

    for path in sources:
        traj, dt = _load_trajectory(
            path=path,
            dt_target=dt_target,
            velocity_source=velocity_source,
            smoothing_cfg=cfg.smoothing,
            reduce_time=bool(getattr(data_cfg, "reduce_time", False)),
            reduction_factor=int(getattr(data_cfg, "reduction_factor", 1)),
            cut_start_seconds=cut_start_seconds,
            force_representation=force_representation,
            f0_lookup=f0_lookup,
            rho=float(getattr(cfg.model, "rho", 1000.0)),
            D=float(getattr(cfg.model, "D", 0.1)),
        )
        if dt_ref is None:
            dt_ref = dt
        elif not np.isclose(dt, float(dt_ref), rtol=1e-9, atol=1e-12):
            raise ValueError(f"{path} has dt={dt} but expected dt={dt_ref}.")
        val_trajs.append(traj)
    if dt_ref is None:
        raise ValueError("No validation trajectories loaded.")
    dt = float(dt_ref)

    d = int(val_trajs[0]["x"].shape[-1])
    m = _as_diag_param(vp.get("m", _m_eff_from_model_cfg(cfg.model)), d, device, "m")
    c = _as_diag_param(vp.get("c", getattr(cfg.model, "damping_c", 1e-4)), d, device, "c")
    k = _as_diag_param(vp.get("k", getattr(cfg.model, "k", 1218.0)), d, device, "k")

    input_dim = 2 * d + 1
    output_dim = d
    base_model = _build_force_model(cfg, input_dim=input_dim, output_dim=output_dim).to(device)
    # Keep validation backward-compatible with older checkpoints/configs:
    # scaling must be explicitly enabled, otherwise state_dict keys won't match.
    use_input_scaling = bool(vp.get("use_input_scaling", False))
    if use_input_scaling:
        D_val = float(getattr(cfg.model, "D", 1.0))
        x_scale = D_val if np.isfinite(D_val) and D_val != 0.0 else 1.0
        omega = torch.sqrt(torch.clamp(k / m, min=1e-12))
        v_scale = omega * float(x_scale)
        ur_scale = float(vp.get("ur_scale", 10.0))
        f_scale = 1.0 if force_representation == "coefficient" else k * float(x_scale)
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

    w, wdot, alpha = _test_functions(int(vp.get("window_M", 50)), dt, num_poly=num_poly, num_sine=num_sine)
    w = w.to(device)
    wdot = wdot.to(device)
    alpha = alpha.to(device)

    per_traj_norm = str(vp.get("per_traj_norm", "none")).strip().lower()
    per_traj_norm_eps = float(vp.get("per_traj_norm_eps", 1e-8))
    if per_traj_norm not in {"none", "force_rms", "residual_rms"}:
        raise ValueError("vpinn.per_traj_norm must be one of: none, force_rms, residual_rms.")
    if per_traj_norm != "none":
        _apply_per_traj_scale(
            val_trajs,
            mode=per_traj_norm,
            dt=dt,
            m=m,
            c=c,
            k=k,
            w=w,
            wdot=wdot,
            alpha=alpha,
            window_M=int(vp.get("window_M", 50)),
            stride=int(vp.get("stride", 1)),
            eps=per_traj_norm_eps,
        )

    return_scale = per_traj_norm != "none"
    val_dataset = WindowDataset(
        val_trajs,
        window_intervals=int(vp.get("window_M", 50)),
        stride=int(vp.get("stride", 1)),
        return_scale=return_scale,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    if do_losses:
        amp_enabled = bool(cfg.precision.use_amp) and device.type == "cuda"
        val_metrics = _evaluate_epoch(
            model=model,
            loader=val_loader,
            device=device,
            non_blocking=(device.type == "cuda"),
            dt=dt,
            m=m,
            c=c,
            k=k,
            wf=float(vp.get("wf", 1.0)),
            ww=float(vp.get("ww", 1.0)),
            use_force_loss=bool(vp.get("use_force_loss", True)),
            use_weak_loss=bool(vp.get("use_weak_loss", True)),
            w=w,
            wdot=wdot,
            alpha=alpha,
            amp_enabled=amp_enabled,
            amp_dtype=_amp_dtype(cfg.precision.amp_dtype),
            per_traj_norm_eps=per_traj_norm_eps,
            expect_scale=return_scale,
            expect_f0=(force_representation == "coefficient"),
        )
        for name, value in val_metrics.items():
            writer.add_scalar(f"val/{name}", value, epoch)
        force_map = _force_mapping_nrmse_over_trajs(model=model, val_trajs=val_trajs, device=device)
        if force_map is not None:
            for k_name, v_value in force_map.items():
                writer.add_scalar(f"val/{k_name}", v_value, epoch)

    if do_rollout:
        rollout_idx = _rollout_index(epoch, rollout_every, len(val_trajs), cycle_rollout)
        _log_rollout_validation(
            writer=writer,
            epoch=epoch,
            model=model,
            traj=val_trajs[rollout_idx],
            dt=dt,
            m=m,
            c=c,
            k=k,
            D=float(getattr(cfg.model, "D", 1.0)),
            middle_time_plot=getattr(data_cfg, "middle_time_plot", [0.0, 1.0]),
            device=device,
        )


def _evaluate_val_losses(
    *,
    model: PHVIV,
    loader: Any,
    device: torch.device,
    non_blocking: bool,
    force_reg: float,
    use_force_data_loss: bool,
    force_data_weight: float,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    per_traj_norm_eps: float,
    force_reg_on_coeff: bool,
) -> dict[str, float]:
    model.eval()
    amp_enabled = bool(amp_enabled) and device.type == "cuda"
    loss_sum = torch.zeros((), device=device)
    res_sum = torch.zeros((), device=device)
    force_sum = torch.zeros((), device=device)
    data_sum = torch.zeros((), device=device)
    batches = 0
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 5:
                z_i, t_i, z_next, t_next, ur_i = batch
                f_i = None
                f_next = None
                scale = None
            elif len(batch) == 6:
                z_i, t_i, z_next, t_next, ur_i, scale = batch
                f_i = None
                f_next = None
            elif len(batch) == 7:
                z_i, t_i, z_next, t_next, ur_i, f_i, f_next = batch
                scale = None
            elif len(batch) == 8:
                z_i, t_i, z_next, t_next, ur_i, f_i, f_next, scale = batch
            else:
                raise ValueError("Unexpected batch format from dataloader.")
            z_i = z_i.to(device, non_blocking=non_blocking)
            t_i = t_i.to(device, non_blocking=non_blocking)
            z_next = z_next.to(device, non_blocking=non_blocking)
            t_next = t_next.to(device, non_blocking=non_blocking)
            ur_i = ur_i.to(device, non_blocking=non_blocking)
            if f_i is not None:
                f_i = f_i.to(device, non_blocking=non_blocking)
            if f_next is not None:
                f_next = f_next.to(device, non_blocking=non_blocking)
            if scale is not None:
                scale = scale.to(device, non_blocking=non_blocking).view(-1)

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                if scale is None:
                    res_loss = model.res_loss(z_i, t_i, z_next, t_next, reduced_velocity=ur_i)
                    if force_reg_on_coeff:
                        avg_force = model.avg_force_coeff(z_i, t_i, z_next, t_next, reduced_velocity=ur_i)
                    else:
                        avg_force = model.avg_force(z_i, t_i, z_next, t_next, reduced_velocity=ur_i)
                else:
                    per_res = model.res_loss_per_sample(z_i, t_i, z_next, t_next, reduced_velocity=ur_i)
                    if force_reg_on_coeff:
                        per_force = model.avg_force_coeff_per_sample(z_i, t_i, z_next, t_next, reduced_velocity=ur_i)
                    else:
                        per_force = model.avg_force_per_sample(z_i, t_i, z_next, t_next, reduced_velocity=ur_i)
                    denom = scale * scale + float(per_traj_norm_eps)
                    res_loss = torch.mean(per_res / denom)
                    avg_force = torch.mean(per_force / denom)
                force_loss = float(force_reg) * avg_force
                if use_force_data_loss:
                    if f_i is None or f_next is None:
                        raise ValueError(
                            "use_force_data_loss is True but the dataloader did not provide force labels."
                        )
                    z_mid = 0.5 * (z_i + z_next)
                    f_mid = 0.5 * (f_i + f_next)
                    f_pred = model.u_theta(z_mid, reduced_velocity=ur_i)
                    per_data = torch.mean((f_pred - f_mid) ** 2, dim=1)
                    if scale is not None:
                        per_data = per_data / (scale * scale + float(per_traj_norm_eps))
                    data_force_loss = torch.mean(per_data)
                else:
                    data_force_loss = res_loss.new_tensor(0.0)
                total = res_loss + force_loss + float(force_data_weight) * data_force_loss

            loss_sum = loss_sum + total.detach().float()
            res_sum = res_sum + res_loss.detach().float()
            force_sum = force_sum + force_loss.detach().float()
            data_sum = data_sum + data_force_loss.detach().float()
            batches += 1

    denom = float(max(batches, 1))
    return {
        "loss_total": float((loss_sum / denom).detach().cpu()),
        "loss_physics": float((res_sum / denom).detach().cpu()),
        "loss_reg": float((force_sum / denom).detach().cpu()),
        "loss_data": float((data_sum / denom).detach().cpu()),
    }


def _amp_dtype(name: str) -> torch.dtype:
    key = str(name).lower()
    if key == "fp16":
        return torch.float16
    if key == "bf16":
        return torch.bfloat16
    return torch.bfloat16


def main() -> None:
    parser = argparse.ArgumentParser(description="Async validation runner.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--rollout-every", type=int, default=1)
    parser.add_argument("--cycle-rollout", type=int, default=0)
    parser.add_argument("--do-losses", type=int, default=1)
    parser.add_argument("--do-rollout", type=int, default=1)
    args = parser.parse_args()

    _set_threading(int(args.num_threads))
    device = torch.device(str(args.device))

    ckpt, cfg, method = _load_checkpoint(args.checkpoint)

    writer = SummaryWriter(log_dir=str(args.log_dir))
    try:
        if method in {"hnn", "phnn"}:
            _run_hnn_validation(
                ckpt=ckpt,
                cfg=cfg,
                device=device,
                writer=writer,
                epoch=int(args.epoch),
                rollout_every=int(args.rollout_every),
                cycle_rollout=bool(int(args.cycle_rollout)),
                do_losses=bool(int(args.do_losses)),
                do_rollout=bool(int(args.do_rollout)),
                num_workers=int(args.num_workers),
            )
        elif method == "vpinn":
            _run_vpinn_validation(
                ckpt=ckpt,
                cfg=cfg,
                device=device,
                writer=writer,
                epoch=int(args.epoch),
                rollout_every=int(args.rollout_every),
                cycle_rollout=bool(int(args.cycle_rollout)),
                do_losses=bool(int(args.do_losses)),
                do_rollout=bool(int(args.do_rollout)),
                num_workers=int(args.num_workers),
            )
        else:
            raise ValueError(f"Unsupported method '{method}'.")
    finally:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()
