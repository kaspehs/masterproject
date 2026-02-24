from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, detrend, filtfilt, savgol_filter

try:
    import Experimental_Data.analyze_experimental_data as analysis
except ModuleNotFoundError as exc:
    if getattr(exc, "name", "") != "Experimental_Data":
        raise
    current_dir = Path(__file__).resolve().parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    import analyze_experimental_data as analysis


# Input
MAT_FILES = [
    Path("Experimental_Data/CrossFlow/RawData/test3002.mat"),
    Path("Experimental_Data/CrossFlow/RawData/test3003.mat"),
    Path("Experimental_Data/CrossFlow/RawData/test3004.mat"),
    Path("Experimental_Data/CrossFlow/RawData/test3005.mat"),
    Path("Experimental_Data/CrossFlow/RawData/test3006.mat"),
    Path("Experimental_Data/CrossFlow/RawData/test3008.mat"),
    #Path("Experimental_Data/CrossFlow/RawData/test3009.mat"),
    Path("Experimental_Data/CrossFlow/RawData/test3010.mat"),
    Path("Experimental_Data/CrossFlow/RawData/test3011.mat"),
    Path("Experimental_Data/CrossFlow/RawData/test3012.mat"),
    Path("Experimental_Data/CrossFlow/RawData/test3014.mat"),
]
DATA_VARIABLE = "data"  # Set to None to auto-detect first numeric 2D array.
USE_RELATIVE_TIME = True
PLOT_FIRST_SECONDS = 1000.0
PLOT_DETAILED_CHANNELS = True
PLOT_DETAILED_FOR_ALL_FILES = False
DETAILED_FILE_INDEX = 3  # MAT_FILES[3] -> test3005.mat

USE_DETREND_BEFORE_DERIVATIVES = False
DETREND_TYPE = "linear"  # "linear" or "constant"
SAVGOL_WINDOW_LENGTH_OVERRIDE = None  # e.g. 71 (odd int). None -> use analysis.SAVGOL_WINDOW_LENGTH
DERIVATIVE_METHOD = "savgol"  # one of: "savgol", "filtfilt_gradient", "gradient"
FILTFILT_DERIV_ORDER = 4
FILTFILT_DERIV_CUTOFF_HZ = 3.0
USE_POST_FILTFILT_ON_ACCELERATION = False
POST_FILTFILT_ORDER = 4
POST_FILTFILT_CUTOFF_HZ = 3.0

FIGURE_WIDTH = 12.0
FIGURE_HEIGHT_PER_ROW = 1.8

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

    return {
        "path": mat_file,
        "label": mat_file.stem,
        "t": np.asarray(t, dtype=float),
        "mask": np.asarray(mask, dtype=bool),
        "dt": dt,
        "deriv_info": deriv_info,
        "channels": channels,
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

    fig.suptitle("Extracted channels", fontsize=13)
    fig.tight_layout()


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

    plt.show()


if __name__ == "__main__":
    main()
