from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

try:
    import Experimental_Data.plot_extracted_channels as extracted
except ModuleNotFoundError as exc:
    if getattr(exc, "name", "") != "Experimental_Data":
        raise
    # Support direct execution: python3 Experimental_Data/phase_analysis.py
    current_dir = Path(__file__).resolve().parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    import plot_extracted_channels as extracted


# Input
MAT_FILES = list(extracted.MAT_FILES)

# Keep data-loading behavior aligned with the channel-extraction script.
DATA_VARIABLE = extracted.DATA_VARIABLE
USE_RELATIVE_TIME = extracted.USE_RELATIVE_TIME
PLOT_FIRST_SECONDS = extracted.PLOT_FIRST_SECONDS

# Phase-drift settings
PHASE_WINDOW_SECONDS = 40.0
PHASE_WINDOW_OVERLAP = 0.5  # in [0, 1)
PHASE_FREQ_MIN_HZ = 0.1
PHASE_FREQ_MAX_HZ = 5.0
PHASE_MAX_LAG_SECONDS = 2.0
# Lag estimation mode: "phase" (recommended) or "xcorr" (legacy).
LAG_ESTIMATION_METHOD = "phase"
# When using phase-derived lag, unwrap lag by adding integer periods for continuity.
LAG_UNWRAP_BY_PERIOD = True
LAG_USE_SUBSAMPLE_PEAK = True
LAG_APPLY_SMOOTHING = True
LAG_MEDIAN_WINDOW = 5  # odd integer
LAG_EMA_ALPHA = 0.25  # in (0, 1]


def _wrap_phase_deg(phase_rad: np.ndarray) -> np.ndarray:
    return np.rad2deg((phase_rad + np.pi) % (2.0 * np.pi) - np.pi)


def _quadratic_peak_offset(y_prev: float, y_mid: float, y_next: float) -> float:
    denom = (y_prev - 2.0 * y_mid + y_next)
    if abs(denom) < 1e-12:
        return 0.0
    offset = 0.5 * (y_prev - y_next) / denom
    return float(np.clip(offset, -1.0, 1.0))


def _rolling_median(x: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    w = max(1, int(window))
    if w % 2 == 0:
        w += 1
    if arr.size < 3 or w <= 1:
        return arr.copy()
    half = w // 2
    out = np.empty_like(arr)
    for i in range(arr.size):
        lo = max(0, i - half)
        hi = min(arr.size, i + half + 1)
        out[i] = float(np.median(arr[lo:hi]))
    return out


def _ema(x: np.ndarray, alpha: float) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr.copy()
    a = float(alpha)
    if not (0.0 < a <= 1.0):
        raise ValueError("LAG_EMA_ALPHA must be in (0, 1].")
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, arr.size):
        out[i] = a * arr[i] + (1.0 - a) * out[i - 1]
    return out


def _stabilize_lag_seconds(lag_seconds: np.ndarray) -> np.ndarray:
    lag = np.asarray(lag_seconds, dtype=float).reshape(-1)
    if lag.size < 3:
        return lag
    med = _rolling_median(lag, int(LAG_MEDIAN_WINDOW))
    return _ema(med, float(LAG_EMA_ALPHA))


def _lag_derivative_per_second(t_seconds: np.ndarray, lag_seconds: np.ndarray) -> np.ndarray:
    t = np.asarray(t_seconds, dtype=float).reshape(-1)
    lag = np.asarray(lag_seconds, dtype=float).reshape(-1)
    if t.size != lag.size:
        n = min(t.size, lag.size)
        t = t[:n]
        lag = lag[:n]
    if t.size < 2:
        return np.zeros_like(t)
    return np.gradient(lag, t)


def _phase_drift_diagnostics(
    *,
    t: np.ndarray,
    sig_ref: np.ndarray,
    sig_cmp: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = np.asarray(t, dtype=float).reshape(-1)
    x = np.asarray(sig_ref, dtype=float).reshape(-1)
    y = np.asarray(sig_cmp, dtype=float).reshape(-1)
    n = min(t.size, x.size, y.size)
    t = t[:n]
    x = x[:n]
    y = y[:n]

    if not (0.0 <= float(PHASE_WINDOW_OVERLAP) < 1.0):
        raise ValueError("PHASE_WINDOW_OVERLAP must be in [0, 1).")
    n_win = int(round(float(PHASE_WINDOW_SECONDS) / dt))
    if n_win < 16:
        raise ValueError("PHASE_WINDOW_SECONDS too small for reliable FFT phase estimation.")
    n_step = max(1, int(round(n_win * (1.0 - float(PHASE_WINDOW_OVERLAP)))))

    max_lag = None
    if PHASE_MAX_LAG_SECONDS is not None:
        max_lag = max(1, int(round(float(PHASE_MAX_LAG_SECONDS) / dt)))

    t_centers: list[float] = []
    phase_diff_deg: list[float] = []
    lag_seconds: list[float] = []
    dom_freq_hz: list[float] = []
    prev_lag_s: float | None = None

    for start in range(0, n - n_win + 1, n_step):
        end = start + n_win
        tw = t[start:end]
        xw = x[start:end]
        yw = y[start:end]
        finite = np.isfinite(xw) & np.isfinite(yw) & np.isfinite(tw)
        if int(np.sum(finite)) < int(0.8 * n_win):
            continue
        tw = tw[finite]
        xw = xw[finite]
        yw = yw[finite]
        if xw.size < 16 or yw.size < 16:
            continue

        xw = xw - np.mean(xw)
        yw = yw - np.mean(yw)
        sx = float(np.std(xw))
        sy = float(np.std(yw))
        if sx < 1e-12 or sy < 1e-12:
            continue

        freqs = np.fft.rfftfreq(xw.size, d=dt)
        X = np.fft.rfft(xw)
        Y = np.fft.rfft(yw)
        band = (freqs >= float(PHASE_FREQ_MIN_HZ)) & (freqs <= float(PHASE_FREQ_MAX_HZ))
        if not np.any(band):
            continue
        idx_band = np.where(band)[0]
        k = idx_band[int(np.argmax(np.abs(X[idx_band])))]
        f_dom = float(freqs[k])
        phase_xy = float(np.angle(X[k]) - np.angle(Y[k]))
        phase_wrapped = float((phase_xy + np.pi) % (2.0 * np.pi) - np.pi)

        lag_method = str(LAG_ESTIMATION_METHOD).strip().lower()
        if lag_method == "phase":
            if f_dom <= 0.0:
                continue
            lag_s = float(phase_wrapped / (2.0 * np.pi * f_dom))
            if bool(LAG_UNWRAP_BY_PERIOD) and prev_lag_s is not None:
                period = 1.0 / f_dom
                if np.isfinite(period) and period > 0.0:
                    lag_s = float(lag_s + round((prev_lag_s - lag_s) / period) * period)
            if max_lag is not None:
                lag_s = float(np.clip(lag_s, -float(max_lag) * dt, float(max_lag) * dt))
        elif lag_method == "xcorr":
            xzn = xw / sx
            yzn = yw / sy
            corr = np.correlate(xzn, yzn, mode="full")
            lags = np.arange(-yzn.size + 1, xzn.size, dtype=int)
            if max_lag is not None:
                keep = np.abs(lags) <= int(max_lag)
                corr = corr[keep]
                lags = lags[keep]
            if corr.size == 0:
                continue
            i_peak = int(np.argmax(corr))
            if bool(LAG_USE_SUBSAMPLE_PEAK) and 0 < i_peak < (corr.size - 1):
                frac = _quadratic_peak_offset(
                    float(corr[i_peak - 1]),
                    float(corr[i_peak]),
                    float(corr[i_peak + 1]),
                )
            else:
                frac = 0.0
            lag_s = float((float(lags[i_peak]) + float(frac)) * dt)
        else:
            raise ValueError("LAG_ESTIMATION_METHOD must be one of: phase, xcorr")

        phase_deg = float(_wrap_phase_deg(np.array([phase_xy]))[0])
        if not (np.isfinite(phase_deg) and np.isfinite(lag_s) and np.isfinite(f_dom)):
            continue

        t_centers.append(float(np.mean(tw)))
        phase_diff_deg.append(phase_deg)
        lag_seconds.append(lag_s)
        dom_freq_hz.append(f_dom)
        prev_lag_s = lag_s

    lag_arr = np.asarray(lag_seconds, dtype=float)
    if bool(LAG_APPLY_SMOOTHING):
        lag_arr = _stabilize_lag_seconds(lag_arr)

    return (
        np.asarray(t_centers, dtype=float),
        np.asarray(phase_diff_deg, dtype=float),
        lag_arr,
        np.asarray(dom_freq_hz, dtype=float),
    )


def _select_channel(result: dict[str, object], channel_name: str) -> np.ndarray:
    channels = result["channels"]
    assert isinstance(channels, list)
    for name, values in channels:
        if str(name) == channel_name:
            return np.asarray(values, dtype=float)
    raise KeyError(f"Missing channel '{channel_name}' in result for {result['label']}.")


def _run_phase_analysis(result: dict[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = np.asarray(result["t"], dtype=float)
    mask = np.asarray(result["mask"], dtype=bool)
    dt = float(result["dt"])

    force = _select_channel(result, "Fy2 - Fy1 (scaled LB/LA, N)")
    disp = _select_channel(result, "Displacement y (m)")

    t_phase = t[mask]
    f_phase = force[mask]
    y_phase = disp[mask]

    phase_disp = _phase_drift_diagnostics(
        t=t_phase,
        sig_ref=f_phase,
        sig_cmp=y_phase,
        dt=dt,
    )
    return phase_disp


def _plot_phase_drift_all(results: list[dict[str, object]]) -> None:
    fig_phase, (ax_py, ax_f, ax_l, ax_dl) = plt.subplots(
        4, 1, figsize=(extracted.FIGURE_WIDTH, 8.8), sharex=True
    )
    plotted = False

    for result in results:
        label = str(result["label"])
        tc, ph_deg, lag_s, f_dom = _run_phase_analysis(result)
        tc = np.asarray(tc, dtype=float)
        ph_deg = np.asarray(ph_deg, dtype=float)
        lag_s = np.asarray(lag_s, dtype=float)
        f_dom = np.asarray(f_dom, dtype=float)
        finite = np.isfinite(tc) & np.isfinite(ph_deg) & np.isfinite(lag_s) & np.isfinite(f_dom)
        tc = tc[finite]
        ph_deg = ph_deg[finite]
        lag_s = lag_s[finite]
        f_dom = f_dom[finite]
        if tc.size == 0:
            print(f"{label}: phase-drift summary (vs y): insufficient valid windows.")
            continue

        ax_py.plot(tc, ph_deg, marker="o", markersize=2.8, linewidth=1.0, label=label)
        ax_l.plot(tc, lag_s, marker="o", markersize=2.8, linewidth=1.0, label=label)
        ax_f.plot(tc, f_dom, marker="o", markersize=2.8, linewidth=1.0, label=label)
        dlag_dt = _lag_derivative_per_second(tc, lag_s)
        ax_dl.plot(tc, dlag_dt, marker="o", markersize=2.4, linewidth=1.0, label=label)
        plotted = True

        if tc.size >= 2:
            slope_deg_per_s = float(np.polyfit(tc, ph_deg, 1)[0])
            print(
                f"{label}: phase-drift summary (vs y): start={ph_deg[0]:.2f} deg, end={ph_deg[-1]:.2f} deg, "
                f"slope={slope_deg_per_s:.4f} deg/s"
            )

    if not plotted:
        plt.close(fig_phase)
        print("Phase-drift summary (vs y): no valid phase windows found across selected files.")
        return

    ax_l.axhline(0.0, color="black", linewidth=0.9, alpha=0.7)
    ax_dl.axhline(0.0, color="black", linewidth=0.9, alpha=0.7)
    ax_l.grid(True)
    ax_f.grid(True)
    ax_py.grid(True)
    ax_dl.grid(True)
    ax_l.set_ylabel("Lag at max corr (s)")
    ax_f.set_ylabel("Dominant f (Hz)")
    ax_py.set_ylabel("Phase diff (deg)")
    ax_dl.set_ylabel("d(lag)/dt (s/s)")
    ax_dl.set_xlabel("Time (s)" if USE_RELATIVE_TIME else "Time")
    ax_py.set_title("Sliding-window phase: (Fy2 - Fy1) relative to displacement")
    lag_method = str(LAG_ESTIMATION_METHOD).strip().lower()
    if lag_method == "phase":
        ax_l.set_title("Sliding-window lag derived from phase/frequency")
    else:
        ax_l.set_title("Sliding-window lag from cross-correlation")
    ax_f.set_title("Dominant frequency used for phase estimate")
    ax_dl.set_title("Sliding-window lag drift rate")
    ax_py.axhline(0.0, color="black", linewidth=0.9, alpha=0.7)
    ax_py.legend(loc="best", fontsize="small")
    fig_phase.tight_layout()


def main() -> None:
    if not MAT_FILES:
        raise ValueError("MAT_FILES is empty.")

    # Ensure phase script can override these settings from the extraction script.
    extracted.DATA_VARIABLE = DATA_VARIABLE
    extracted.USE_RELATIVE_TIME = USE_RELATIVE_TIME
    extracted.PLOT_FIRST_SECONDS = PLOT_FIRST_SECONDS

    mat_files = [Path(p) for p in MAT_FILES]
    missing = [p for p in mat_files if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing MAT file(s): {missing}")

    results: list[dict[str, object]] = []
    for mat_file in mat_files:
        result = extracted._process_file(mat_file)
        results.append(result)

    _plot_phase_drift_all(results)
    plt.show()


if __name__ == "__main__":
    main()
