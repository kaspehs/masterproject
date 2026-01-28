from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.optim as optim
import torch.nn.utils as nn_utils
from torch.utils.tensorboard import SummaryWriter

from core.logging import setup_writer
from core.optim import setup_optimizer_and_scheduler
from core.runtime import (
    configure_tf32,
    maybe_compile_model,
    select_device,
    set_num_threads_from_slurm,
    setup_amp,
)
from HNN_helper import (
    Config,
    GradNormBalancer,
    PHVIV,
    build_dataloader_from_series,
    compute_validation_metrics,
    compute_model_grad_norm,
    load_training_series,
    log_training_metrics,
    log_validation_epoch,
    preprocess_timeseries,
)


def _train_one_epoch(
    *,
    model: torch.nn.Module,
    opt: optim.Optimizer,
    train_loader: Any,
    device: torch.device,
    non_blocking: bool,
    max_grad_norm: float,
    force_reg: float,
    use_force_data_loss: bool,
    force_data_weight: float,
    gradnorm_balancer: Optional[GradNormBalancer],
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    scaler: torch.cuda.amp.GradScaler,
    log_component_grad_norms: bool,
) -> dict[str, float]:
    batch_count = 0
    loss_sum = torch.zeros((), device=device)
    res_loss_sum = torch.zeros((), device=device)
    force_loss_sum = torch.zeros((), device=device)
    force_data_loss_sum = torch.zeros((), device=device)
    grad_norm_sum = torch.zeros((), device=device)
    avg_force_sum = torch.zeros((), device=device)
    res_grad_component_sum = torch.zeros((), device=device)
    force_grad_component_sum = torch.zeros((), device=device)
    gradnorm_res_weight_sum = torch.zeros((), device=device)
    gradnorm_force_weight_sum = torch.zeros((), device=device)
    gradnorm_data_weight_sum = torch.zeros((), device=device) if use_force_data_loss else None
    gradnorm_weight_count = 0

    for batch in train_loader:
        if len(batch) == 5:
            z_i, t_i, z_next, t_next, ur_i = batch
            f_i = None
            f_next = None
        elif len(batch) == 7:
            z_i, t_i, z_next, t_next, ur_i, f_i, f_next = batch
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

        opt.zero_grad()

        with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
            res_loss = model.res_loss(z_i, t_i, z_next, t_next, reduced_velocity=ur_i)
            avg_force = model.avg_force(z_i, t_i, z_next, t_next, reduced_velocity=ur_i)
            base_force_loss = avg_force
            if use_force_data_loss:
                if f_i is None or f_next is None:
                    raise ValueError(
                        "use_force_data_loss is True but the dataloader did not provide force labels."
                    )
                z_mid = 0.5 * (z_i + z_next)
                f_mid = 0.5 * (f_i + f_next)
                f_pred = model.u_theta(z_mid, reduced_velocity=ur_i)
                data_force_loss = torch.mean((f_pred - f_mid) ** 2)
            else:
                data_force_loss = res_loss.new_tensor(0.0)

            if gradnorm_balancer is not None:
                weights = gradnorm_balancer.update(
                    {
                        "residual": res_loss.float(),
                        "force": base_force_loss.float(),
                        "data": data_force_loss.float() if use_force_data_loss else res_loss.float(),
                    }
                )
                res_weight = weights["residual"]
                force_weight = weights["force"]
                data_weight = weights.get("data", res_loss.new_tensor(1.0))
                gradnorm_res_weight_sum = gradnorm_res_weight_sum + res_weight
                gradnorm_force_weight_sum = gradnorm_force_weight_sum + force_weight
                if gradnorm_data_weight_sum is not None:
                    gradnorm_data_weight_sum = gradnorm_data_weight_sum + data_weight
                gradnorm_weight_count += 1
            else:
                res_weight = res_loss.new_tensor(1.0)
                force_weight = res_loss.new_tensor(1.0)
                data_weight = res_loss.new_tensor(1.0)

            weighted_res = res_weight * res_loss
            force_loss = float(force_reg) * base_force_loss
            weighted_force = force_weight * force_loss
            weighted_data = data_weight * (float(force_data_weight) * data_force_loss)
            loss = (weighted_res + weighted_force + weighted_data).float()

        if log_component_grad_norms and scaler.is_enabled():
            raise ValueError(
                "monitoring.log_component_grad_norms is not supported with AMP fp16 (GradScaler enabled)."
            )
        if log_component_grad_norms:
            weighted_res.backward(retain_graph=True)
            res_grad_component_sum = res_grad_component_sum + torch.as_tensor(
                compute_model_grad_norm(model), device=device
            )
            model.zero_grad(set_to_none=True)
            weighted_force.backward(retain_graph=True)
            force_grad_component_sum = force_grad_component_sum + torch.as_tensor(
                compute_model_grad_norm(model), device=device
            )
            model.zero_grad(set_to_none=True)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
        else:
            loss.backward()

        grad_norm = nn_utils.clip_grad_norm_(model.parameters(), max_norm=float(max_grad_norm))
        if isinstance(grad_norm, torch.Tensor):
            grad_norm_sum = grad_norm_sum + grad_norm.detach()
        else:
            grad_norm_sum = grad_norm_sum + torch.tensor(float(grad_norm), device=device)

        if scaler.is_enabled():
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()

        batch_count += 1
        loss_sum = loss_sum + loss.detach()
        res_loss_sum = res_loss_sum + res_loss.detach().float()
        force_loss_sum = force_loss_sum + force_loss.detach().float()
        force_data_loss_sum = force_data_loss_sum + data_force_loss.detach().float()
        avg_force_sum = avg_force_sum + avg_force.detach().float()

    denom = float(max(batch_count, 1))
    metrics: dict[str, float] = {
        "mean_loss": float((loss_sum / denom).detach().cpu()),
        "mean_res_loss": float((res_loss_sum / denom).detach().cpu()),
        "mean_force_loss": float((force_loss_sum / denom).detach().cpu()),
        "mean_force_data_loss": float((force_data_loss_sum / denom).detach().cpu()),
        "mean_grad_norm": float((grad_norm_sum / denom).detach().cpu()),
        "mean_force": float((avg_force_sum / denom).detach().cpu()),
        "mean_res_grad_component": float((res_grad_component_sum / denom).detach().cpu()),
        "mean_force_grad_component": float((force_grad_component_sum / denom).detach().cpu()),
    }
    if gradnorm_weight_count > 0:
        metrics["mean_gradnorm_weight_residual"] = float(
            (gradnorm_res_weight_sum / float(gradnorm_weight_count)).detach().cpu()
        )
        metrics["mean_gradnorm_weight_force"] = float(
            (gradnorm_force_weight_sum / float(gradnorm_weight_count)).detach().cpu()
        )
        if gradnorm_data_weight_sum is not None:
            metrics["mean_gradnorm_weight_data"] = float(
                (gradnorm_data_weight_sum / float(gradnorm_weight_count)).detach().cpu()
            )
    return metrics


def _validate_if_needed(
    *,
    writer: SummaryWriter,
    epoch: int,
    rollout_every_epochs: int,
    model: PHVIV,
    y_data_t: torch.Tensor,
    val_vel: torch.Tensor,
    reduced_velocity: torch.Tensor,
    val_series_raw: list[tuple[np.ndarray, np.ndarray, float, np.ndarray | None, np.ndarray | None, np.ndarray]] | None,
    val_sequences: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] | None,
    val_loader: Any | None,
    cycle_validation_rollout: bool,
    m_eff: float,
    dt: float,
    t: np.ndarray,
    y_true_norm: np.ndarray,
    y_data: np.ndarray,
    force_data: np.ndarray,
    D: float,
    k: float,
    device: torch.device,
    middle_time_plot,
    hamiltonian_data,
    log_extra_validation_metrics: bool,
    force_reg: float,
    use_force_data_loss: bool,
    force_data_weight: float,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> None:
    if rollout_every_epochs <= 0:
        return
    if (epoch + 1) % int(rollout_every_epochs) != 0:
        return
    if val_loader is not None:
        val_loss_metrics = _evaluate_val_losses(
            model=model,
            loader=val_loader,
            device=device,
            non_blocking=(device.type == "cuda"),
            force_reg=force_reg,
            use_force_data_loss=use_force_data_loss,
            force_data_weight=force_data_weight,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        for name, value in val_loss_metrics.items():
            writer.add_scalar(f"val/{name}", value, epoch + 1)

    if val_series_raw is not None and val_sequences is not None:
        metrics_sum: dict[str, float] = {}
        count = 0
        for series_raw, sequence in zip(val_series_raw, val_sequences):
            y_np, t_np, dt_value, _vel_np, force_np, _ur_np = series_raw
            y_tensor, vel_tensor, _t_tensor, ur_tensor = sequence
            if force_np is None:
                continue
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
                log_extra_metrics=log_extra_validation_metrics,
            )
            for name, value in metrics.items():
                metrics_sum[name] = metrics_sum.get(name, 0.0) + float(value)
            count += 1
        if count > 0:
            for name, total in metrics_sum.items():
                writer.add_scalar(f"val/{name}", total / float(count), epoch + 1)
        if cycle_validation_rollout:
            step = max(0, (epoch + 1) // max(1, int(rollout_every_epochs)) - 1)
            rollout_idx = step % len(val_series_raw)
        else:
            rollout_idx = 0
        series_raw = val_series_raw[rollout_idx]
        sequence = val_sequences[rollout_idx]
        y_np, t_np, dt_value, _vel_np, force_np, _ur_np = series_raw
        y_tensor, vel_tensor, _t_tensor, ur_tensor = sequence
        if force_np is None:
            return
        log_validation_epoch(
            writer,
            epoch + 1,
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
            middle_time_plot,
            hamiltonian_data,
            log_extra_metrics=log_extra_validation_metrics,
            log_metrics=False,
        )
        return
    log_validation_epoch(
        writer,
        epoch + 1,
        model,
        y_data_t,
        val_vel,
        reduced_velocity,
        m_eff,
        dt,
        t,
        y_true_norm,
        y_data,
        force_data,
        D,
        k,
        device,
        middle_time_plot,
        hamiltonian_data,
        log_extra_metrics=log_extra_validation_metrics,
    )


def _prune_async_processes(processes: list[subprocess.Popen]) -> list[subprocess.Popen]:
    return [proc for proc in processes if proc.poll() is None]


def _launch_async_validation(
    *,
    processes: list[subprocess.Popen],
    max_concurrent: int,
    checkpoint_path: Path,
    epoch: int,
    writer: SummaryWriter,
    async_device: str,
    async_num_workers: int,
    async_num_threads: int,
    rollout_every_epochs: int,
    cycle_validation_rollout: bool,
    do_losses: bool,
    do_rollout: bool,
) -> list[subprocess.Popen]:
    processes = _prune_async_processes(processes)
    if max_concurrent > 0 and len(processes) >= max_concurrent:
        return processes
    script_path = Path(__file__).resolve().parents[2] / "async_validate.py"
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(async_num_threads)
    env["MKL_NUM_THREADS"] = str(async_num_threads)
    env["OPENBLAS_NUM_THREADS"] = str(async_num_threads)
    env["NUMEXPR_NUM_THREADS"] = str(async_num_threads)
    args = [
        sys.executable,
        str(script_path),
        "--checkpoint",
        str(checkpoint_path),
        "--log-dir",
        str(writer.log_dir),
        "--epoch",
        str(epoch + 1),
        "--device",
        str(async_device),
        "--num-threads",
        str(async_num_threads),
        "--num-workers",
        str(async_num_workers),
        "--rollout-every",
        str(int(rollout_every_epochs)),
        "--cycle-rollout",
        "1" if cycle_validation_rollout else "0",
        "--do-losses",
        "1" if do_losses else "0",
        "--do-rollout",
        "1" if do_rollout else "0",
    ]
    processes.append(subprocess.Popen(args, env=env))
    return processes


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
) -> dict[str, float]:
    was_training = model.training
    model.eval()
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
            elif len(batch) == 7:
                z_i, t_i, z_next, t_next, ur_i, f_i, f_next = batch
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

            with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
                res_loss = model.res_loss(z_i, t_i, z_next, t_next, reduced_velocity=ur_i)
                avg_force = model.avg_force(z_i, t_i, z_next, t_next, reduced_velocity=ur_i)
                force_loss = float(force_reg) * avg_force
                if use_force_data_loss:
                    if f_i is None or f_next is None:
                        raise ValueError(
                            "use_force_data_loss is True but the dataloader did not provide force labels."
                        )
                    z_mid = 0.5 * (z_i + z_next)
                    f_mid = 0.5 * (f_i + f_next)
                    f_pred = model.u_theta(z_mid, reduced_velocity=ur_i)
                    data_force_loss = torch.mean((f_pred - f_mid) ** 2)
                else:
                    data_force_loss = res_loss.new_tensor(0.0)
                total = res_loss + force_loss + float(force_data_weight) * data_force_loss

            loss_sum = loss_sum + total.detach().float()
            res_sum = res_sum + res_loss.detach().float()
            force_sum = force_sum + force_loss.detach().float()
            data_sum = data_sum + data_force_loss.detach().float()
            batches += 1

    if was_training:
        model.train()
    denom = float(max(batches, 1))
    return {
        "loss_total": float((loss_sum / denom).detach().cpu()),
        "loss_physics": float((res_sum / denom).detach().cpu()),
        "loss_reg": float((force_sum / denom).detach().cpu()),
        "loss_data": float((data_sum / denom).detach().cpu()),
    }


def train(config: Config, config_name: str) -> None:
    data_cfg = config.data
    middle_time_plot = data_cfg.middle_time_plot
    use_generated_train_series = data_cfg.use_generated_train_series
    train_series_root = Path(data_cfg.train_series_dir)
    train_series_dir = train_series_root

    if use_generated_train_series:
        train_dir = train_series_root / "train"
        val_dir = train_series_root / "val"
        if not val_dir.exists():
            raise FileNotFoundError(
                f"Expected validation data in '{val_dir}'. data.npz is no longer used for validation."
            )
        if not train_dir.exists():
            raise FileNotFoundError(f"Expected training data in '{train_dir}'.")
        val_files = sorted(val_dir.glob("*.npz"))
        if not val_files:
            raise FileNotFoundError(f"No '.npz' files found in validation directory '{val_dir}'.")
        data_path = val_files[0]
        train_series_dir = train_dir
    else:
        data_path = Path(data_cfg.file)

    data = np.load(data_path)
    t = data["a"]
    y_data = data["b"]
    F_data = data["c"]
    H_data = data["d"]
    if "U_r" not in data:
        raise KeyError(f"{data_path} is missing reduced velocity 'U_r'.")
    reduced_velocity = data["U_r"]
    vel_data = None
    for key in ("e", "dy", "v"):
        if key in data:
            vel_data = data[key]
            break

    t, y_data, F_data, hamiltonian_data, vel_data, dt = preprocess_timeseries(
        t,
        y_data,
        F_data,
        H_data,
        data_cfg,
        velocity=vel_data,
    )

    model_cfg = config.model
    smoothing_cfg = config.smoothing
    hnn_cfg = dict(config.hnn or {})
    velocity_source = str(hnn_cfg.get("velocity_source", "compute")).strip().lower()

    training_cfg = config.training
    optim_cfg = config.optim
    loss_cfg = config.loss
    runtime_cfg = config.runtime
    precision_cfg = config.precision
    compile_cfg = config.compile
    monitoring_cfg = config.monitoring

    batch_size = int(training_cfg.batch_size)
    max_grad_norm = float(training_cfg.max_grad_norm)
    epochs = int(training_cfg.epochs)

    lr = float(optim_cfg.lr)
    use_lr_scheduler = bool(optim_cfg.use_lr_scheduler)
    scheduler_cfg = optim_cfg.scheduler

    force_reg = float(loss_cfg.force_reg)
    use_force_data_loss = bool(getattr(loss_cfg, "use_force_data_loss", False))
    force_data_weight = float(getattr(loss_cfg, "force_data_weight", 1.0))

    rollout_every_epochs = int(monitoring_cfg.rollout_every_epochs)
    cycle_validation_rollout = bool(getattr(monitoring_cfg, "cycle_validation_rollout", False))
    log_every_epochs = max(1, int(monitoring_cfg.log_every_epochs))
    print_every_epochs = max(1, int(monitoring_cfg.print_every_epochs))
    log_component_grad_norms = bool(monitoring_cfg.log_component_grad_norms)
    log_extra_validation_metrics = bool(getattr(monitoring_cfg, "log_extra_validation_metrics", False))
    async_validation = bool(getattr(monitoring_cfg, "async_validation", False))
    async_device = str(getattr(monitoring_cfg, "async_validation_device", "cpu"))
    async_num_workers = int(getattr(monitoring_cfg, "async_validation_num_workers", 0))
    async_num_threads = int(getattr(monitoring_cfg, "async_validation_num_threads", 4))
    async_max_concurrent = int(getattr(monitoring_cfg, "async_validation_max_concurrent", 1))
    async_do_losses = bool(getattr(monitoring_cfg, "async_validation_do_losses", True))
    async_do_rollout = bool(getattr(monitoring_cfg, "async_validation_do_rollout", True))

    device = select_device(os.getenv("TRAIN_DEVICE", str(runtime_cfg.device)))
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA available: {torch.cuda.is_available()}, gpu0: {torch.cuda.get_device_name(0)}")
    configure_tf32(device, bool(precision_cfg.use_tf32))
    set_num_threads_from_slurm(default=1)
    non_blocking = device.type == "cuda"

    model_dict = asdict(model_cfg)
    arch_dict = asdict(config.architecture)
    model, derived_params = PHVIV.from_config(dt=dt, cfg=model_dict, arch_cfg=arch_dict, device=device)
    model = maybe_compile_model(model, bool(compile_cfg.use_compile), str(compile_cfg.compile_mode))
    D = derived_params["D"]
    k = derived_params["k"]
    m_eff = derived_params["m_eff"]

    train_series_raw, eval_tensors = load_training_series(
        y_data,
        t,
        dt,
        use_generated_train_series,
        train_series_dir,
        m_eff,
        device,
        smoothing_cfg=smoothing_cfg,
        velocity_source=velocity_source,
        eval_velocity=vel_data,
        eval_reduced_velocity=reduced_velocity,
        require_force=use_force_data_loss,
        eval_force=F_data,
        cut_start_seconds=float(getattr(data_cfg, "cut_start_seconds", 0.0)),
    )
    eval_y_tensor, eval_vel_tensor, eval_t_tensor, eval_ur_tensor = eval_tensors

    pin_memory = device.type == "cuda"
    num_workers = int(runtime_cfg.num_workers)
    train_loader, train_sequences, _ = build_dataloader_from_series(
        train_series_raw,
        m_eff=m_eff,
        batch_size=batch_size,
        device=device,
        smoothing_cfg=smoothing_cfg,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_series_raw: list[tuple[np.ndarray, np.ndarray, float, np.ndarray | None, np.ndarray | None, np.ndarray]] | None = None
    val_sequences: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] | None = None
    val_loader: Any | None = None
    if use_generated_train_series:
        val_dir = train_series_root / "val"
        if val_dir.exists():
            val_series_raw, _ = load_training_series(
                y_data,
                t,
                dt,
                True,
                val_dir,
                m_eff,
                device,
                smoothing_cfg=smoothing_cfg,
                velocity_source=velocity_source,
                eval_velocity=vel_data,
                eval_reduced_velocity=reduced_velocity,
                require_force=True,
                eval_force=F_data,
                cut_start_seconds=float(getattr(data_cfg, "cut_start_seconds", 0.0)),
            )
            val_loader, val_sequences, _ = build_dataloader_from_series(
                val_series_raw,
                m_eff=m_eff,
                batch_size=batch_size,
                device=device,
                smoothing_cfg=smoothing_cfg,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )

    if use_generated_train_series:
        y_data_t, val_vel, _t_tensor, val_ur = eval_y_tensor, eval_vel_tensor, eval_t_tensor, eval_ur_tensor
    else:
        y_data_t, val_vel, _t_tensor, val_ur = train_sequences[0]

    writer, run_name = setup_writer(config.logging.run_dir_root, config_name)
    async_processes: list[subprocess.Popen] = []
    async_dir = Path(writer.log_dir) / "async_validation"
    if async_validation:
        async_dir.mkdir(parents=True, exist_ok=True)

    y_true_norm = y_data / D
    force_data = F_data

    opt, lr_scheduler = setup_optimizer_and_scheduler(
        model,
        optim_cfg=optim_cfg,
        scheduler_cfg=scheduler_cfg,
        epochs=epochs,
    )

    gradnorm_balancer: Optional[GradNormBalancer] = None
    if bool(loss_cfg.use_gradnorm):
        names = ["residual", "force"]
        if use_force_data_loss:
            names.append("data")
        gradnorm_balancer = GradNormBalancer(
            model,
            names,
            alpha=float(loss_cfg.gradnorm_alpha),
            eps=float(loss_cfg.gradnorm_eps),
            min_weight=float(loss_cfg.gradnorm_min_weight),
            max_weight=float(loss_cfg.gradnorm_max_weight),
        )

    amp_enabled, amp_dtype, scaler = setup_amp(
        device, use_amp=bool(precision_cfg.use_amp), amp_dtype=str(precision_cfg.amp_dtype)
    )

    for epoch in range(epochs):
        if use_lr_scheduler:
            for group in opt.param_groups:
                group["lr"] = lr_scheduler.get_lr(epoch)

        epoch_metrics = _train_one_epoch(
            model=model,
            opt=opt,
            train_loader=train_loader,
            device=device,
            non_blocking=non_blocking,
            max_grad_norm=max_grad_norm,
            force_reg=force_reg,
            use_force_data_loss=use_force_data_loss,
            force_data_weight=force_data_weight,
            gradnorm_balancer=gradnorm_balancer,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            scaler=scaler,
            log_component_grad_norms=log_component_grad_norms,
        )

        mean_loss = epoch_metrics["mean_loss"]
        mean_res_loss = epoch_metrics["mean_res_loss"]
        mean_force_loss = epoch_metrics["mean_force_loss"]
        mean_force_data_loss = epoch_metrics["mean_force_data_loss"]
        mean_grad_norm = epoch_metrics["mean_grad_norm"]
        mean_force = epoch_metrics["mean_force"]
        mean_res_grad_component = epoch_metrics["mean_res_grad_component"]
        mean_force_grad_component = epoch_metrics["mean_force_grad_component"]

        drag_coeff = float(torch.exp(model.log_Cd).detach().cpu())
        current_lr = float(opt.param_groups[0]["lr"]) if opt.param_groups else lr

        damping_ratio_value = (
            float(torch.sigmoid(model.zeta_raw).detach().cpu()) * float(model.max_damping_ratio.detach().cpu())
            if getattr(model, "discover_damping", True)
            else float(model.fixed_damping_ratio)
        )

        train_metrics: dict[str, float] = {
            "loss_total": mean_loss,
            "loss_physics": mean_res_loss,
            "loss_reg": mean_force_loss,
            "loss_data": mean_force_data_loss,
            "lr": current_lr,
            "damping_ratio": damping_ratio_value,
            "grad_norm": mean_grad_norm,
            "drag_coefficient": drag_coeff,
            "avg_force": mean_force,
        }
        if log_component_grad_norms:
            train_metrics["grad_norm_residual_comp"] = mean_res_grad_component
            train_metrics["grad_norm_force_comp"] = mean_force_grad_component
        if "mean_gradnorm_weight_residual" in epoch_metrics:
            train_metrics["gradnorm_weight_physics"] = float(epoch_metrics["mean_gradnorm_weight_residual"])
        if "mean_gradnorm_weight_force" in epoch_metrics:
            train_metrics["gradnorm_weight_reg"] = float(epoch_metrics["mean_gradnorm_weight_force"])
        if "mean_gradnorm_weight_data" in epoch_metrics:
            train_metrics["gradnorm_weight_data"] = float(epoch_metrics["mean_gradnorm_weight_data"])

        log_this_epoch = (epoch % log_every_epochs) == 0 or epoch == (epochs - 1)
        if log_this_epoch:
            log_training_metrics(writer, epoch, train_metrics)
        print_this_epoch = (epoch % print_every_epochs) == 0 or epoch == (epochs - 1)
        if print_this_epoch:
            if use_force_data_loss:
                print(
                    f"Epoch {epoch}: loss={mean_loss:.4e}, res={mean_res_loss:.4e}, "
                    f"force={mean_force_loss:.4e}, data={mean_force_data_loss:.4e}"
                )
            else:
                print(f"Epoch {epoch}: loss={mean_loss:.4e}, res={mean_res_loss:.4e}, force={mean_force_loss:.4e}")

        should_validate = rollout_every_epochs > 0 and (
            (epoch + 1) % int(rollout_every_epochs) == 0 or epoch == (epochs - 1)
        )
        if async_validation and should_validate and (async_do_losses or async_do_rollout):
            async_processes = _prune_async_processes(async_processes)
            state_source = model
            if hasattr(model, "_orig_mod"):
                state_source = getattr(model, "_orig_mod")
            ckpt_path = async_dir / f"epoch_{epoch + 1:06d}.pt"
            torch.save(
                {
                    "model_state": state_source.state_dict(),
                    "config": asdict(config),
                    "run_name": run_name,
                    "dt": dt,
                    "method": str(config.method),
                },
                ckpt_path,
            )
            async_processes = _launch_async_validation(
                processes=async_processes,
                max_concurrent=async_max_concurrent,
                checkpoint_path=ckpt_path,
                epoch=epoch,
                writer=writer,
                async_device=async_device,
                async_num_workers=async_num_workers,
                async_num_threads=async_num_threads,
                rollout_every_epochs=rollout_every_epochs,
                cycle_validation_rollout=cycle_validation_rollout,
                do_losses=async_do_losses,
                do_rollout=async_do_rollout,
            )
        elif not async_validation:
            _validate_if_needed(
                writer=writer,
                epoch=epoch,
                rollout_every_epochs=rollout_every_epochs,
                model=model,
                y_data_t=y_data_t,
                val_vel=val_vel,
                reduced_velocity=val_ur,
                val_series_raw=val_series_raw,
                val_sequences=val_sequences,
                val_loader=val_loader,
                cycle_validation_rollout=cycle_validation_rollout,
                m_eff=m_eff,
                dt=dt,
                t=t,
                y_true_norm=y_true_norm,
                y_data=y_data,
                force_data=force_data,
                D=D,
                k=k,
                device=device,
                middle_time_plot=middle_time_plot,
                hamiltonian_data=hamiltonian_data,
                log_extra_validation_metrics=log_extra_validation_metrics,
                force_reg=force_reg,
                use_force_data_loss=use_force_data_loss,
                force_data_weight=force_data_weight,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{run_name}.pt"
    state_source = model
    if hasattr(model, "_orig_mod"):
        state_source = getattr(model, "_orig_mod")
    torch.save(
        {
            "model_state": state_source.state_dict(),
            "config": asdict(config),
            "run_name": run_name,
        },
        model_path,
    )
    print(f"Saved final model to {model_path}")

    writer.flush()
    writer.close()
