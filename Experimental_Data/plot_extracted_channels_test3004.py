from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, detrend, filtfilt, savgol_filter

import analyze_test3005 as analysis


# Input
MAT_FILES = [
    Path("Experimental_Data/CrossFlow/test3002.mat"),
    Path("Experimental_Data/CrossFlow/test3003.mat"),
    Path("Experimental_Data/CrossFlow/test3004.mat"),
    Path("Experimental_Data/CrossFlow/test3005.mat"),
    Path("Experimental_Data/CrossFlow/test3006.mat"),
    Path("Experimental_Data/CrossFlow/test3008.mat"),
    Path("Experimental_Data/CrossFlow/test3009.mat"),
    Path("Experimental_Data/CrossFlow/test3010.mat"),
    Path("Experimental_Data/CrossFlow/test3011.mat"),
    Path("Experimental_Data/CrossFlow/test3012.mat"),
    Path("Experimental_Data/CrossFlow/test3014.mat"),
]
DATA_VARIABLE = "data"  # Set to None to auto-detect first numeric 2D array.
USE_RELATIVE_TIME = True
PLOT_FIRST_SECONDS = 1000.0
PLOT_DETAILED_CHANNELS = False
PLOT_DETAILED_FOR_ALL_FILES = False
DETAILED_FILE_INDEX = 3  # MAT_FILES[3] -> test3005.mat

USE_DETREND_BEFORE_DERIVATIVES = False
DETREND_TYPE = "linear"  # "linear" or "constant"
SAVGOL_WINDOW_LENGTH_OVERRIDE = 21  # e.g. 71 (odd int). None -> use analysis.SAVGOL_WINDOW_LENGTH
DERIVATIVE_METHOD = "savgol"  # one of: "savgol", "filtfilt_gradient", "gradient"
FILTFILT_DERIV_ORDER = 4
FILTFILT_DERIV_CUTOFF_HZ = 3.0
USE_POST_FILTFILT_ON_ACCELERATION = False
POST_FILTFILT_ORDER = 4
POST_FILTFILT_CUTOFF_HZ = 3.0

FIGURE_WIDTH = 12.0
FIGURE_HEIGHT_PER_ROW = 1.8

ENABLE_PHASE_DRIFT_DIAGNOSTIC = True
PHASE_WINDOW_SECONDS = 20.0
PHASE_WINDOW_OVERLAP = 0.5  # in [0, 1)
PHASE_FREQ_MIN_HZ = 0.1
PHASE_FREQ_MAX_HZ = 5.0
PHASE_MAX_LAG_SECONDS = 2.0


def _lowpass_filtfilt(values: np.ndarray, *, dt: float, order: int, cutoff_hz: float, role: str) -> np.ndarray:
    nyquist = 0.5 / dt
    cutoff = float(cutoff_hz)
    if not (0.0 < cutoff < nyquist):
        raise ValueError(f"{role}: cutoff must be in (0, {nyquist:.6f}) Hz for dt={dt:.6e}, got {cutoff}.")
    b, a = butter(int(order), cutoff / nyquist, btype="low")
    return filtfilt(b, a, np.asarray(values, dtype=float))


def _fill_nonfinite_1d(values: np.ndarray, *, role: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = np.isfinite(arr)
    if np.all(finite):
        return arr
    if not np.any(finite):
        raise ValueError(f"{role}: signal has no finite samples.")
    idx = np.arange(arr.size, dtype=float)
    arr[~finite] = np.interp(idx[~finite], idx[finite], arr[finite])
    return arr


def _wrap_phase_deg(phase_rad: np.ndarray) -> np.ndarray:
    return np.rad2deg((phase_rad + np.pi) % (2.0 * np.pi) - np.pi)


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
        lag_s = float(lags[int(np.argmax(corr))] * dt)
        phase_deg = float(_wrap_phase_deg(np.array([phase_xy]))[0])
        if not (np.isfinite(phase_deg) and np.isfinite(lag_s) and np.isfinite(f_dom)):
            continue

        t_centers.append(float(np.mean(tw)))
        phase_diff_deg.append(phase_deg)
        lag_seconds.append(lag_s)
        dom_freq_hz.append(f_dom)

    return (
        np.asarray(t_centers, dtype=float),
        np.asarray(phase_diff_deg, dtype=float),
        np.asarray(lag_seconds, dtype=float),
        np.asarray(dom_freq_hz, dtype=float),
    )


def _compute_derivatives(disp: np.ndarray, *, dt: float) -> tuple[np.ndarray, np.ndarray, str]:
    disp_for_derivatives = np.asarray(disp, dtype=float)
    if USE_DETREND_BEFORE_DERIVATIVES:
        detrend_type = str(DETREND_TYPE).strip().lower()
        if detrend_type not in {"linear", "constant"}:
            raise ValueError("DETREND_TYPE must be 'linear' or 'constant'.")
        disp_for_derivatives = detrend(disp_for_derivatives, type=detrend_type)

    deriv_method = str(DERIVATIVE_METHOD).strip().lower()
    if deriv_method == "savgol":
        sg_window_target = (
            int(SAVGOL_WINDOW_LENGTH_OVERRIDE)
            if SAVGOL_WINDOW_LENGTH_OVERRIDE is not None
            else int(analysis.SAVGOL_WINDOW_LENGTH)
        )
        sg_window = analysis._savgol_window_length(
            disp_for_derivatives.size,
            sg_window_target,
            int(analysis.SAVGOL_POLYORDER),
        )
        if sg_window >= 0:
            vel_y = savgol_filter(
                disp_for_derivatives,
                window_length=sg_window,
                polyorder=int(analysis.SAVGOL_POLYORDER),
                deriv=1,
                delta=dt,
                mode="interp",
            )
            acc_y = savgol_filter(
                disp_for_derivatives,
                window_length=sg_window,
                polyorder=int(analysis.SAVGOL_POLYORDER),
                deriv=2,
                delta=dt,
                mode="interp",
            )
        else:
            vel_y = np.gradient(disp_for_derivatives, dt)
            acc_y = np.gradient(vel_y, dt)
        deriv_info = f"deriv=savgol(sg_window={sg_window})"
    elif deriv_method == "filtfilt_gradient":
        disp_lp = _lowpass_filtfilt(
            disp_for_derivatives,
            dt=dt,
            order=int(FILTFILT_DERIV_ORDER),
            cutoff_hz=float(FILTFILT_DERIV_CUTOFF_HZ),
            role="FILTFILT_DERIV",
        )
        vel_y = np.gradient(disp_lp, dt)
        acc_y = np.gradient(vel_y, dt)
        deriv_info = f"deriv=filtfilt_gradient(order={FILTFILT_DERIV_ORDER}, cutoff_hz={FILTFILT_DERIV_CUTOFF_HZ:g})"
    elif deriv_method == "gradient":
        vel_y = np.gradient(disp_for_derivatives, dt)
        acc_y = np.gradient(vel_y, dt)
        deriv_info = "deriv=gradient"
    else:
        raise ValueError("DERIVATIVE_METHOD must be one of: savgol, filtfilt_gradient, gradient")

    if USE_POST_FILTFILT_ON_ACCELERATION:
        acc_y = _lowpass_filtfilt(
            acc_y,
            dt=dt,
            order=int(POST_FILTFILT_ORDER),
            cutoff_hz=float(POST_FILTFILT_CUTOFF_HZ),
            role="POST_FILTFILT",
        )
        deriv_info += f", post_filtfilt_accel(order={POST_FILTFILT_ORDER}, cutoff_hz={POST_FILTFILT_CUTOFF_HZ:g})"

    return vel_y, acc_y, deriv_info


def _process_file(mat_file: Path) -> dict[str, object]:
    data, channel_names = analysis._load_data_matrix(mat_file, DATA_VARIABLE)

    time = analysis._select_column(data, channel_names, ["Time"], 0, role="time")
    ypos = analysis._select_column(data, channel_names, ["xpos1"], 23, role="CF displacement (xpos1)")
    fy_spring1 = analysis._select_column(data, channel_names, ["9130_FORCE_1"], 6, role="Fy_spring1")
    fy_spring2 = analysis._select_column(data, channel_names, ["9133_FORCE_4"], 9, role="Fy_spring2")

    time_arr = _fill_nonfinite_1d(np.asarray(time, dtype=float), role=f"{mat_file.name}: time")
    disp = _fill_nonfinite_1d(np.asarray(ypos, dtype=float), role=f"{mat_file.name}: displacement")
    fy1 = _fill_nonfinite_1d(np.asarray(fy_spring1, dtype=float), role=f"{mat_file.name}: Fy_spring1")
    fy2 = _fill_nonfinite_1d(np.asarray(fy_spring2, dtype=float), role=f"{mat_file.name}: Fy_spring2")

    t = time_arr - float(time_arr[0]) if USE_RELATIVE_TIME else time_arr
    if USE_RELATIVE_TIME:
        mask = t <= float(PLOT_FIRST_SECONDS)
    else:
        t0 = float(time_arr[0])
        mask = t <= (t0 + float(PLOT_FIRST_SECONDS))

    dt = float(np.nanmedian(np.diff(time_arr)))
    if dt <= 0.0 or not np.isfinite(dt):
        raise ValueError(f"{mat_file.name}: could not infer a valid positive dt.")

    vel_y, acc_y, deriv_info = _compute_derivatives(disp, dt=dt)

    fy1_center = fy1 * (analysis.LB / analysis.LA)
    fy2_center = fy2 * (analysis.LB / analysis.LA)
    m = float(analysis.M)
    m_added = float(analysis.ADDED_MASS_COEFF) * 0.25 * np.pi * float(analysis.RUO) * float(analysis.D) ** 2 * float(analysis.L)

    fy1_minus_fy2 = fy1_center - fy2_center
    fy2_minus_fy1 = fy2_center - fy1_center

    channels: list[tuple[str, np.ndarray]] = [
        ("Displacement y (m)", disp),
        ("Velocity y_dot (m/s)", vel_y),
        ("Acceleration y_ddot (m/s^2)", acc_y),
        ("Fy1 (scaled LB/LA, N)", fy1_center),
        ("Fy2 (scaled LB/LA, N)", fy2_center),
        ("Fy2 - Fy1 (scaled LB/LA, N)", fy2_minus_fy1),
    ]

    phase_acc = None
    phase_disp = None
    if ENABLE_PHASE_DRIFT_DIAGNOSTIC:
        t_phase = np.asarray(t, dtype=float)[mask]
        f_phase = np.asarray(fy2_minus_fy1, dtype=float)[mask]
        a_phase = np.asarray(acc_y, dtype=float)[mask]
        y_phase = np.asarray(disp, dtype=float)[mask]
        phase_acc = _phase_drift_diagnostics(
            t=t_phase,
            sig_ref=f_phase,
            sig_cmp=a_phase,
            dt=dt,
        )
        phase_disp = _phase_drift_diagnostics(
            t=t_phase,
            sig_ref=f_phase,
            sig_cmp=y_phase,
            dt=dt,
        )

    return {
        "path": mat_file,
        "label": mat_file.stem,
        "t": np.asarray(t, dtype=float),
        "mask": np.asarray(mask, dtype=bool),
        "dt": dt,
        "deriv_info": deriv_info,
        "channels": channels,
        "phase_acc": phase_acc,
        "phase_disp": phase_disp,
        "m": m,
        "m_added": m_added,
    }


def _plot_detailed_channels(result: dict[str, object]) -> None:
    channels = result["channels"]
    assert isinstance(channels, list)
    t = np.asarray(result["t"], dtype=float)
    mask = np.asarray(result["mask"], dtype=bool)
    t_plot = t[mask]

    n = len(channels)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(FIGURE_WIDTH, FIGURE_HEIGHT_PER_ROW * nrows),
        sharex=True,
    )
    axes = np.asarray(axes).reshape(-1)

    for i, item in enumerate(channels):
        name, values = item
        ax = axes[i]
        ax.plot(t_plot, np.asarray(values, dtype=float)[mask], linewidth=1.1)
        ax.set_title(str(name))
        ax.grid(True)
        ax.set_ylabel(str(name))

    for i in range(n, axes.size):
        axes[i].axis("off")

    for ax in axes:
        if ax.has_data():
            ax.set_xlabel("Time (s)" if USE_RELATIVE_TIME else "Time")

    label = str(result["label"])
    fig.suptitle(f"Extracted channels from {label}", fontsize=13)
    fig.tight_layout()


def _plot_phase_drift_all(results: list[dict[str, object]]) -> None:
    if not ENABLE_PHASE_DRIFT_DIAGNOSTIC:
        return

    fig_phase, (ax_p, ax_py, ax_f, ax_l) = plt.subplots(4, 1, figsize=(FIGURE_WIDTH, 8.6), sharex=True)
    plotted = False
    plotted_y = False

    for result in results:
        label = str(result["label"])
        phase = result["phase_acc"]
        if phase is None:
            continue
        tc, ph_deg, lag_s, f_dom = phase
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
            print(f"{label}: phase-drift summary (vs a_y): insufficient valid windows.")
            continue

        ax_p.plot(tc, ph_deg, marker="o", markersize=2.8, linewidth=1.0, label=label)
        ax_l.plot(tc, lag_s, marker="o", markersize=2.8, linewidth=1.0, label=label)
        ax_f.plot(tc, f_dom, marker="o", markersize=2.8, linewidth=1.0, label=label)
        plotted = True

        if tc.size >= 2:
            slope_deg_per_s = float(np.polyfit(tc, ph_deg, 1)[0])
            print(
                f"{label}: phase-drift summary (vs a_y): start={ph_deg[0]:.2f} deg, end={ph_deg[-1]:.2f} deg, "
                f"slope={slope_deg_per_s:.4f} deg/s"
            )

        phase_y = result["phase_disp"]
        if phase_y is not None:
            tc_y, ph_deg_y, _, _ = phase_y
            tc_y = np.asarray(tc_y, dtype=float)
            ph_deg_y = np.asarray(ph_deg_y, dtype=float)
            finite_y = np.isfinite(tc_y) & np.isfinite(ph_deg_y)
            tc_y = tc_y[finite_y]
            ph_deg_y = ph_deg_y[finite_y]
            if tc_y.size == 0:
                print(f"{label}: phase-drift summary (vs y): insufficient valid windows.")
            else:
                ax_py.plot(tc_y, ph_deg_y, marker="o", markersize=2.8, linewidth=1.0, label=label)
                plotted_y = True
                if tc_y.size >= 2:
                    slope_deg_per_s_y = float(np.polyfit(tc_y, ph_deg_y, 1)[0])
                    print(
                        f"{label}: phase-drift summary (vs y): start={ph_deg_y[0]:.2f} deg, "
                        f"end={ph_deg_y[-1]:.2f} deg, slope={slope_deg_per_s_y:.4f} deg/s"
                    )

    if not plotted:
        plt.close(fig_phase)
        print("Phase-drift summary (vs a_y): no valid phase windows found across selected files.")
        return

    ax_p.axhline(0.0, color="black", linewidth=0.9, alpha=0.7)
    ax_l.axhline(0.0, color="black", linewidth=0.9, alpha=0.7)
    ax_p.grid(True)
    ax_l.grid(True)
    ax_f.grid(True)
    ax_py.grid(True)
    ax_p.set_ylabel("Phase diff (deg)")
    ax_l.set_ylabel("Lag at max corr (s)")
    ax_f.set_ylabel("Dominant f (Hz)")
    ax_py.set_ylabel("Phase diff (deg)")
    ax_py.set_xlabel("Time (s)" if USE_RELATIVE_TIME else "Time")
    ax_p.set_title("Sliding-window phase: (Fy2 - Fy1) relative to acceleration")
    ax_l.set_title("Sliding-window lag from cross-correlation")
    ax_f.set_title("Dominant frequency used for phase estimate")
    ax_py.set_title("Sliding-window phase: (Fy2 - Fy1) relative to displacement")
    ax_py.axhline(0.0, color="black", linewidth=0.9, alpha=0.7)
    ax_p.legend(loc="best", fontsize="small")
    if plotted_y:
        ax_py.legend(loc="best", fontsize="small")
    else:
        print("Phase-drift summary (vs y): no valid phase windows found across selected files.")
    fig_phase.tight_layout()


def main() -> None:
    if not MAT_FILES:
        raise ValueError("MAT_FILES is empty.")

    mat_files = [Path(p) for p in MAT_FILES]
    missing = [p for p in mat_files if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing MAT file(s): {missing}")

    results: list[dict[str, object]] = []
    for mat_file in mat_files:
        result = _process_file(mat_file)
        results.append(result)

        print(
            f"\n{result['label']}: dt={result['dt']:.6e} s, "
            f"m={result['m']:.6f} kg, m_a={result['m_added']:.6f} kg, {result['deriv_info']}"
        )
        print("-" * 72)
        channels = result["channels"]
        mask = np.asarray(result["mask"], dtype=bool)
        for name, values in channels:
            v = np.asarray(values, dtype=float)
            mean_full = float(np.nanmean(v))
            if np.any(mask):
                mean_window = float(np.nanmean(v[mask]))
                print(f"{name:<36} full={mean_full: .6e} | first {PLOT_FIRST_SECONDS:g}s={mean_window: .6e}")
            else:
                print(f"{name:<36} full={mean_full: .6e} | first {PLOT_FIRST_SECONDS:g}s=nan")
        print("-" * 72)

    if PLOT_DETAILED_CHANNELS:
        if PLOT_DETAILED_FOR_ALL_FILES:
            for result in results:
                _plot_detailed_channels(result)
        else:
            idx = int(DETAILED_FILE_INDEX)
            if not (0 <= idx < len(results)):
                raise IndexError(f"DETAILED_FILE_INDEX={idx} out of range for {len(results)} files.")
            _plot_detailed_channels(results[idx])

    _plot_phase_drift_all(results)
    plt.show()


if __name__ == "__main__":
    main()
