"""
Simulate the TD cross-flow model at a high internal sampling rate, then "log" at a
lower rate by taking every Nth sample (no filtering).

This matches a simplistic logger that only records every Nth point. Note that this
can introduce aliasing if the logged rate is too low relative to the dynamics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    from Data_Gen.simulate_td_model_cf import simulate_td_model_cf
except Exception:  # pragma: no cover
    from simulate_td_model_cf import simulate_td_model_cf  # type: ignore


# -------------------------
# Configure your experiment
# -------------------------
# Internal simulator sampling rate (dt=1e-4 -> 10kHz).
DT_SIM = 1e-4
# Total simulation time (seconds).
T_TOTAL = 20.0
# Logger rate (Hz). This script logs by taking every Nth point (no filtering).
LOGGER_HZ = 100.0
# Cases to simulate.
AMPS = [i for i in np.arange(-1.0, 1.0, 0.05)]
FHAT_START = -0.35
FHAT_STOP = 0.35
FHAT_STEP = 0.02
# RNG seed (None disables randomness; keep fixed for reproducibility).
SEED: int | None = None
# Integrator used by the simulator.
INTEGRATOR = "rk4"  # "rk4" | "euler"
# Optional slicing before logging (useful if you only want steady-state).
START_TIME: float | None = None
END_TIME: float | None = None
# Output folder for logged runs.
OUT_DIR = Path("Data_Gen/groundtruth_runs_100hz")


def _compute_stride(*, dt_sim: float, logger_hz: float, tol: float = 1e-12) -> tuple[int, float]:
    if dt_sim <= 0.0:
        raise ValueError("dt_sim must be > 0")
    if logger_hz <= 0.0:
        raise ValueError("logger_hz must be > 0")
    dt_logger = 1.0 / float(logger_hz)
    stride_float = dt_logger / float(dt_sim)
    stride = int(round(stride_float))
    if stride < 1:
        raise ValueError("Computed stride < 1; pick a lower logger_hz or smaller dt_sim.")
    if abs(stride_float - stride) > tol:
        raise ValueError(
            f"logger_hz ({logger_hz}) is not an integer divisor of dt_sim ({dt_sim}). "
            f"Need dt_logger/dt_sim to be integer; got {stride_float}."
        )
    return stride, dt_logger


def _downsample_nth(sim: dict[str, np.ndarray], stride: int) -> dict[str, np.ndarray]:
    def _maybe(name: str) -> np.ndarray | None:
        value = sim.get(name)
        if value is None:
            return None
        return np.asarray(value)

    out: dict[str, np.ndarray] = {}
    for key in ["time", "y", "dy", "ddy", "Fy", "F_total", "Fca", "Fcv", "Fdy", "H"]:
        arr = _maybe(key)
        if arr is None:
            continue
        out[key] = arr[::stride]
    return out


def simulate_logged(
    *,
    A_factor: float,
    fhat: float,
    dt_sim: float,
    T: float,
    logger_hz: float,
    seed: int | None,
    integrator: str,
    start_time: float | None,
    end_time: float | None,
) -> dict[str, Any]:
    stride, dt_logger = _compute_stride(dt_sim=dt_sim, logger_hz=logger_hz)
    sim = simulate_td_model_cf(
        A_factor=A_factor,
        fhat=fhat,
        dt=dt_sim,
        T=T,
        output_path=None,
        plot=False,
        seed=seed,
        verbose=False,
        integrator=integrator,
    )

    time = np.asarray(sim["time"])
    mask = np.ones_like(time, dtype=bool)
    if start_time is not None:
        mask &= time >= float(start_time)
    if end_time is not None:
        mask &= time <= float(end_time)
    if not np.all(mask):
        sim = {k: np.asarray(v)[mask] for k, v in sim.items()}

    logged = _downsample_nth(sim, stride=stride)
    return {
        "meta": {
            "A_factor": float(A_factor),
            "fhat": float(fhat),
            "dt_sim": float(dt_sim),
            "T": float(T),
            "logger_hz": float(logger_hz),
            "dt_logger": float(dt_logger),
            "stride": int(stride),
            "seed": None if seed is None else int(seed),
            "integrator": str(integrator),
            "start_time": None if start_time is None else float(start_time),
            "end_time": None if end_time is None else float(end_time),
        },
        "data": logged,
    }


def save_logged(out_dir: Path, payload: dict[str, Any]) -> Path:
    meta: dict[str, Any] = payload["meta"]
    data: dict[str, np.ndarray] = payload["data"]
    out_dir.mkdir(parents=True, exist_ok=True)

    amp = meta["A_factor"]
    fhat = meta["fhat"]
    seed = meta["seed"]
    hz = meta["logger_hz"]
    seed_part = "seedNone" if seed is None else f"seed{seed}"
    filename = f"td_cf_A{amp:.3f}_fhat{fhat:.5f}_{seed_part}_log{hz:.0f}Hz.npz"
    path = out_dir / filename

    np.savez(
        path,
        **data,
        A_factor=np.array(meta["A_factor"], dtype=float),
        fhat=np.array(meta["fhat"], dtype=float),
        dt_sim=np.array(meta["dt_sim"], dtype=float),
        dt_logger=np.array(meta["dt_logger"], dtype=float),
        logger_hz=np.array(meta["logger_hz"], dtype=float),
        stride=np.array(meta["stride"], dtype=int),
        seed=np.array(-1 if meta["seed"] is None else meta["seed"], dtype=int),
    )
    return path


def main() -> None:
    fhat_values = np.arange(FHAT_START, FHAT_STOP + 0.5 * FHAT_STEP, FHAT_STEP)
    total = 0
    for amp in AMPS:
        for fhat in fhat_values:
            payload = simulate_logged(
                A_factor=float(amp),
                fhat=float(fhat),
                dt_sim=float(DT_SIM),
                T=float(T_TOTAL),
                logger_hz=float(LOGGER_HZ),
                seed=SEED,
                integrator=str(INTEGRATOR),
                start_time=START_TIME,
                end_time=END_TIME,
            )
            out_path = save_logged(OUT_DIR, payload)
            total += 1
            if total % 50 == 0:
                print(f"Wrote {total} runs (latest: {out_path.name})")
    print(f"Done. Wrote {total} runs to {OUT_DIR}")


if __name__ == "__main__":
    main()
