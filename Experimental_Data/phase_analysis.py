from __future__ import annotations

from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

try:
    import Experimental_Data.plot_extracted_channels as extracted
    import Experimental_Data.analyze_experimental_data as analysis
except ModuleNotFoundError as exc:
    if getattr(exc, "name", "") != "Experimental_Data":
        raise
    # Support direct execution: python3 Experimental_Data/phase_analysis.py
    current_dir = Path(__file__).resolve().parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    import plot_extracted_channels as extracted
    import analyze_experimental_data as analysis


# Input (single source of truth in analyze_experimental_data.py)
# Supported sources here are raw sets because phase_analysis relies on
# extracted._process_file(), which reads raw channel names.
PHASE_MAT_SOURCE = "crossflow_raw"  # one of: crossflow_raw, crossflow, combined
_phase_source = str(PHASE_MAT_SOURCE).strip().lower()
if _phase_source == "combined":
    MAT_FILES = list(analysis.MAT_FILES_COMBINED_RAW)
elif _phase_source in {"crossflow_raw", "crossflow"}:
    MAT_FILES = list(analysis.MAT_FILES_CROSSFLOW_RAW)
else:
    raise ValueError("PHASE_MAT_SOURCE must be one of: crossflow_raw, crossflow, combined")

# Exclude files by test number parsed from filename, e.g. test3009.mat -> 3009.
EXCLUDE_TEST_NUMBERS: list[int] = [3009, 3002]

# Keep data-loading behavior aligned with the channel-extraction script.
DATA_VARIABLE = extracted.DATA_VARIABLE
USE_RELATIVE_TIME = extracted.USE_RELATIVE_TIME
PLOT_FIRST_SECONDS = extracted.PLOT_FIRST_SECONDS

# Phase-drift settings
PHASE_WINDOW_SECONDS = 20.0
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
LAG_ZERO_PHASE_WINDOW = 7  # odd integer

# Displacement correction settings
PHASE_CORRECTION_ENABLED = True
PHASE_CORRECTION_MODE = "common"  # "common" or "individual"
PHASE_CORRECTION_SIGN = "auto"  # "auto", +1, or -1
PHASE_CORRECTION_REMOVE_INITIAL_OFFSET = True
PHASE_COMMON_LAG_POLYORDER = 1
# Optional zero-phase smoothing around the warp step to reduce correction noise.
# Set to odd integer > 1 to enable.
PHASE_CORRECTION_PRE_SMOOTH_WINDOW = 1
# Post-correction Savitzky-Golay smoothing (applied after time-warp correction).
# Set window <= 1 to disable.
PHASE_CORRECTION_POST_SAVGOL_WINDOW = 21
PHASE_CORRECTION_POST_SAVGOL_POLYORDER = 3
# Optional plotting window for pre/post displacement figure.
# Interpreted in the displayed time base (`result['t']`), i.e. relative or absolute
# depending on USE_RELATIVE_TIME.
# Example: (390.0, 400.0)
DISPLACEMENT_PLOT_TIME_WINDOW: tuple[float, float] | list[float] | None = [390, 400]
PLOT_CORRECTED_DISPLACEMENT = True
PLOT_CORRECTED_ACCELERATION = True


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


def _zero_phase_boxcar(x: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    w = max(1, int(window))
    if w % 2 == 0:
        w += 1
    if arr.size < 3 or w <= 1:
        return arr.copy()

    if w > arr.size:
        w = arr.size if arr.size % 2 == 1 else arr.size - 1
        if w < 3:
            return arr.copy()

    half = w // 2
    padded = np.pad(arr, (half, half), mode="reflect")
    kernel = np.full(w, 1.0 / float(w), dtype=float)
    return np.convolve(padded, kernel, mode="valid")


def _savgol_smooth_1d(values: np.ndarray, *, window: int, polyorder: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size < 5:
        return arr.copy()
    w = int(window)
    if w <= 1:
        return arr.copy()
    if w % 2 == 0:
        w += 1
    if w > arr.size:
        w = arr.size if arr.size % 2 == 1 else arr.size - 1
    p = int(polyorder)
    if p < 0:
        p = 0
    if w <= p:
        w = p + 3
        if w % 2 == 0:
            w += 1
        if w > arr.size:
            w = arr.size if arr.size % 2 == 1 else arr.size - 1
    if w < 3 or w <= p:
        return arr.copy()
    return savgol_filter(arr, window_length=int(w), polyorder=int(p), mode="interp")


def _stabilize_lag_seconds(lag_seconds: np.ndarray) -> np.ndarray:
    lag = np.asarray(lag_seconds, dtype=float).reshape(-1)
    if lag.size < 3:
        return lag
    med = _rolling_median(lag, int(LAG_MEDIAN_WINDOW))
    return _zero_phase_boxcar(med, int(LAG_ZERO_PHASE_WINDOW))


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


def _extract_test_number(path: Path) -> int | None:
    stem = str(path.stem).lower()
    match = re.search(r"test(\d+)", stem)
    if match is None:
        return None
    return int(match.group(1))


def _filter_excluded_tests(paths: list[Path]) -> list[Path]:
    excluded_raw = list(EXCLUDE_TEST_NUMBERS)
    if not excluded_raw:
        return paths
    excluded = {int(v) for v in excluded_raw}
    kept: list[Path] = []
    for p in paths:
        test_no = _extract_test_number(Path(p))
        if test_no is not None and test_no in excluded:
            continue
        kept.append(Path(p))
    return kept


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


def _run_phase_analysis(
    result: dict[str, object],
    *,
    disp_override: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = np.asarray(result["t"], dtype=float)
    mask = np.asarray(result["mask"], dtype=bool)
    dt = float(result["dt"])

    force = _select_channel(result, "Fy2 - Fy1 (scaled LB/LA, N)")
    if disp_override is None:
        disp = _select_channel(result, "Displacement y (m)")
    else:
        disp = np.asarray(disp_override, dtype=float)

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


def _filter_finite_phase_series(
    tc: np.ndarray,
    ph_deg: np.ndarray,
    lag_s: np.ndarray,
    f_dom: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tc = np.asarray(tc, dtype=float).reshape(-1)
    ph_deg = np.asarray(ph_deg, dtype=float).reshape(-1)
    lag_s = np.asarray(lag_s, dtype=float).reshape(-1)
    f_dom = np.asarray(f_dom, dtype=float).reshape(-1)
    n = min(tc.size, ph_deg.size, lag_s.size, f_dom.size)
    tc = tc[:n]
    ph_deg = ph_deg[:n]
    lag_s = lag_s[:n]
    f_dom = f_dom[:n]
    finite = np.isfinite(tc) & np.isfinite(ph_deg) & np.isfinite(lag_s) & np.isfinite(f_dom)
    return tc[finite], ph_deg[finite], lag_s[finite], f_dom[finite]


def _phase_slope_deg_per_s(t: np.ndarray, phase_deg: np.ndarray) -> float | None:
    t = np.asarray(t, dtype=float).reshape(-1)
    phase_deg = np.asarray(phase_deg, dtype=float).reshape(-1)
    if t.size < 2 or phase_deg.size < 2:
        return None
    return float(np.polyfit(t, phase_deg, 1)[0])


def _collect_phase_series(
    results: list[dict[str, object]],
    *,
    disp_overrides: dict[str, np.ndarray] | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    phase_by_label: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for result in results:
        label = str(result["label"])
        disp_override = None if disp_overrides is None else disp_overrides.get(label)
        tc, ph_deg, lag_s, f_dom = _run_phase_analysis(result, disp_override=disp_override)
        tc, ph_deg, lag_s, f_dom = _filter_finite_phase_series(tc, ph_deg, lag_s, f_dom)
        phase_by_label[label] = (tc, ph_deg, lag_s, f_dom)
    return phase_by_label


def _fit_common_lag_model(
    phase_by_label: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, float, float]:
    rel_t_list: list[np.ndarray] = []
    lag_list: list[np.ndarray] = []
    for tc, _, lag_s, _ in phase_by_label.values():
        if tc.size < 2:
            continue
        rel_t_list.append(np.asarray(tc, dtype=float) - float(tc[0]))
        lag_list.append(np.asarray(lag_s, dtype=float))
    if not rel_t_list:
        raise ValueError("Could not build common lag model: no valid phase windows.")
    rel_t_all = np.concatenate(rel_t_list)
    lag_all = np.concatenate(lag_list)
    order = max(1, int(PHASE_COMMON_LAG_POLYORDER))
    coeff = np.polyfit(rel_t_all, lag_all, order)
    return coeff, float(np.min(rel_t_all)), float(np.max(rel_t_all))


def _evaluate_lag_curve(
    result: dict[str, object],
    *,
    tc: np.ndarray,
    lag_s: np.ndarray,
    common_model: tuple[np.ndarray, float, float] | None,
) -> np.ndarray:
    t = np.asarray(result["t"], dtype=float).reshape(-1)
    if t.size == 0:
        return t.copy()

    mode = str(PHASE_CORRECTION_MODE).strip().lower()
    if mode == "common":
        if common_model is None:
            raise ValueError("Common lag model is required when PHASE_CORRECTION_MODE='common'.")
        coeff, rel_t_min, rel_t_max = common_model
        rel_t = t - float(t[0])
        rel_t_eval = np.clip(rel_t, rel_t_min, rel_t_max)
        lag_eval = np.polyval(coeff, rel_t_eval)
    elif mode == "individual":
        if tc.size == 0 or lag_s.size == 0:
            lag_eval = np.zeros_like(t)
        elif tc.size == 1:
            lag_eval = np.full_like(t, float(lag_s[0]), dtype=float)
        else:
            tc_rel = tc - float(tc[0])
            t_rel = t - float(t[0])
            lag_eval = np.interp(t_rel, tc_rel, lag_s, left=float(lag_s[0]), right=float(lag_s[-1]))
    else:
        raise ValueError("PHASE_CORRECTION_MODE must be one of: common, individual")

    lag_eval = np.asarray(lag_eval, dtype=float).reshape(-1)
    if bool(PHASE_CORRECTION_REMOVE_INITIAL_OFFSET) and lag_eval.size > 0:
        lag_eval = lag_eval - float(lag_eval[0])
    return lag_eval


def _warp_signal_same_grid(
    *,
    t: np.ndarray,
    values: np.ndarray,
    lag_seconds: np.ndarray,
    sign: float,
) -> np.ndarray:
    t_arr = np.asarray(t, dtype=float).reshape(-1)
    v_arr = np.asarray(values, dtype=float).reshape(-1)
    lag_arr = np.asarray(lag_seconds, dtype=float).reshape(-1)
    n = min(t_arr.size, v_arr.size, lag_arr.size)
    t_arr = t_arr[:n]
    v_arr = v_arr[:n]
    lag_arr = lag_arr[:n]
    if n < 2:
        return v_arr.copy()

    finite = np.isfinite(t_arr) & np.isfinite(v_arr)
    if int(np.sum(finite)) < 2:
        return v_arr.copy()
    t_fit = t_arr[finite]
    v_fit = v_arr[finite]
    if np.any(np.diff(t_fit) <= 0.0):
        order = np.argsort(t_fit)
        t_fit = t_fit[order]
        v_fit = v_fit[order]

    t_query = t_arr + float(sign) * lag_arr
    t_query = np.clip(t_query, float(t_fit[0]), float(t_fit[-1]))
    return np.interp(t_query, t_fit, v_fit)


def _slope_score_abs_median(
    phase_by_label: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
) -> float:
    slopes: list[float] = []
    for tc, ph_deg, _, _ in phase_by_label.values():
        slope = _phase_slope_deg_per_s(tc, ph_deg)
        if slope is not None and np.isfinite(slope):
            slopes.append(abs(float(slope)))
    if not slopes:
        return float("inf")
    return float(np.median(np.asarray(slopes, dtype=float)))


def _resolve_phase_correction_sign(
    phase_by_label_before: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    results: list[dict[str, object]],
    lag_by_label: dict[str, np.ndarray],
) -> float:
    sign_cfg = PHASE_CORRECTION_SIGN
    if isinstance(sign_cfg, str) and sign_cfg.strip().lower() == "auto":
        candidates = (+1.0, -1.0)
    else:
        sign_val = float(sign_cfg)
        if sign_val not in (-1.0, 1.0):
            raise ValueError("PHASE_CORRECTION_SIGN must be 'auto', +1, or -1.")
        return sign_val

    best_sign = candidates[0]
    best_score = float("inf")
    for sign in candidates:
        disp_trial: dict[str, np.ndarray] = {}
        for result in results:
            label = str(result["label"])
            t = np.asarray(result["t"], dtype=float)
            y = _select_channel(result, "Displacement y (m)")
            lag_eval = lag_by_label[label]
            disp_trial[label] = _warp_signal_same_grid(t=t, values=y, lag_seconds=lag_eval, sign=sign)
        phase_trial = _collect_phase_series(results, disp_overrides=disp_trial)
        score = _slope_score_abs_median(phase_trial)
        if score < best_score:
            best_score = score
            best_sign = sign

    score_before = _slope_score_abs_median(phase_by_label_before)
    print(
        f"Phase-correction sign auto-selected: {best_sign:+.0f} "
        f"(median |slope|: before={score_before:.4f}, after={best_score:.4f} deg/s)"
    )
    return best_sign


def _build_corrected_displacement(
    results: list[dict[str, object]],
    phase_by_label_before: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> dict[str, np.ndarray]:
    common_model: tuple[np.ndarray, float, float] | None = None
    if str(PHASE_CORRECTION_MODE).strip().lower() == "common":
        common_model = _fit_common_lag_model(phase_by_label_before)
        coeff, rel_t_min, rel_t_max = common_model
        coeff_str = ", ".join(f"{c:.6e}" for c in coeff)
        print(
            f"Common lag model coeff=[{coeff_str}], valid relative-time window=[{rel_t_min:.3f}, {rel_t_max:.3f}] s"
        )

    lag_by_label: dict[str, np.ndarray] = {}
    for result in results:
        label = str(result["label"])
        tc, _, lag_s, _ = phase_by_label_before[label]
        lag_by_label[label] = _evaluate_lag_curve(result, tc=tc, lag_s=lag_s, common_model=common_model)

    sign = _resolve_phase_correction_sign(phase_by_label_before, results, lag_by_label)

    corrected: dict[str, np.ndarray] = {}
    for result in results:
        label = str(result["label"])
        t = np.asarray(result["t"], dtype=float)
        y = _select_channel(result, "Displacement y (m)")
        pre_w = int(PHASE_CORRECTION_PRE_SMOOTH_WINDOW)
        if pre_w > 1:
            y = _zero_phase_boxcar(y, pre_w)
        corrected[label] = _warp_signal_same_grid(
            t=t,
            values=y,
            lag_seconds=lag_by_label[label],
            sign=sign,
        )
        corrected[label] = _savgol_smooth_1d(
            np.asarray(corrected[label], dtype=float),
            window=int(PHASE_CORRECTION_POST_SAVGOL_WINDOW),
            polyorder=int(PHASE_CORRECTION_POST_SAVGOL_POLYORDER),
        )
    return corrected


def _print_phase_before_after_summary(
    phase_before: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    phase_after: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    print("\nPhase-slope summary (before -> after correction):")
    print("-" * 72)
    for label in phase_before:
        tc_b, ph_b, _, _ = phase_before[label]
        tc_a, ph_a, _, _ = phase_after[label]
        slope_b = _phase_slope_deg_per_s(tc_b, ph_b)
        slope_a = _phase_slope_deg_per_s(tc_a, ph_a)
        if slope_b is None or slope_a is None:
            print(f"{label}: insufficient valid windows.")
            continue
        print(f"{label:<12} {slope_b:+.5f} -> {slope_a:+.5f} deg/s")


def _plot_corrected_displacement(
    results: list[dict[str, object]],
    corrected_disp_by_label: dict[str, np.ndarray],
) -> None:
    if not corrected_disp_by_label:
        return
    n = len(results)
    ncols = 3 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(extracted.FIGURE_WIDTH, 2.2 * nrows + 1.0),
        sharex=False,
    )
    axes_arr = np.asarray(axes).reshape(-1)
    for i, result in enumerate(results):
        label = str(result["label"])
        t = np.asarray(result["t"], dtype=float)
        mask = np.asarray(result["mask"], dtype=bool)
        if DISPLACEMENT_PLOT_TIME_WINDOW is None:
            plot_mask = np.asarray(mask, dtype=bool)
        else:
            if len(DISPLACEMENT_PLOT_TIME_WINDOW) != 2:
                raise ValueError("DISPLACEMENT_PLOT_TIME_WINDOW must have exactly two values: (start, end).")
            t0 = float(DISPLACEMENT_PLOT_TIME_WINDOW[0])
            t1 = float(DISPLACEMENT_PLOT_TIME_WINDOW[1])
            t_lo = min(t0, t1)
            t_hi = max(t0, t1)
            plot_mask = (t >= t_lo) & (t <= t_hi)
        y = _select_channel(result, "Displacement y (m)")
        y_corr = np.asarray(corrected_disp_by_label[label], dtype=float)
        t_plot = t[plot_mask]
        y_plot = np.asarray(y, dtype=float)[plot_mask]
        y_corr_plot = np.asarray(y_corr, dtype=float)[plot_mask]
        if t_plot.size == 0:
            ax = axes_arr[i]
            ax.set_title(f"{label} (no samples in plot window)")
            ax.grid(True)
            ax.set_xlabel("Time (s)" if USE_RELATIVE_TIME else "Time")
            ax.set_ylabel("Displacement y (m)")
            continue
        marker_step = max(1, int(np.ceil(max(t_plot.size, 1) / 120.0)))
        ax = axes_arr[i]
        ax.plot(
            t_plot,
            y_plot,
            linestyle=":",
            marker="x",
            markevery=marker_step,
            markersize=3.0,
            linewidth=1.0,
            alpha=0.85,
            label="Original y",
        )
        ax.plot(
            t_plot,
            y_corr_plot,
            linestyle="-",
            marker="o",
            markevery=marker_step,
            markersize=3.0,
            linewidth=1.05,
            alpha=0.95,
            label="Corrected y",
        )
        ax.set_title(label)
        ax.grid(True)
        ax.set_xlabel("Time (s)" if USE_RELATIVE_TIME else "Time")
        ax.set_ylabel("Displacement y (m)")
        if i == 0:
            ax.legend(loc="best", fontsize="small")

    for i in range(n, axes_arr.size):
        axes_arr[i].axis("off")

    if DISPLACEMENT_PLOT_TIME_WINDOW is None:
        fig_title = "Displacement before/after phase-drift correction"
    else:
        t0 = float(DISPLACEMENT_PLOT_TIME_WINDOW[0])
        t1 = float(DISPLACEMENT_PLOT_TIME_WINDOW[1])
        fig_title = f"Displacement before/after phase-drift correction [{min(t0, t1):g}, {max(t0, t1):g}]"
    fig.suptitle(fig_title, fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))


def _plot_corrected_acceleration(
    results: list[dict[str, object]],
    corrected_disp_by_label: dict[str, np.ndarray],
) -> None:
    if not corrected_disp_by_label:
        return
    n = len(results)
    ncols = 3 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(extracted.FIGURE_WIDTH, 2.2 * nrows + 1.0),
        sharex=False,
    )
    axes_arr = np.asarray(axes).reshape(-1)
    for i, result in enumerate(results):
        label = str(result["label"])
        t = np.asarray(result["t"], dtype=float)
        mask = np.asarray(result["mask"], dtype=bool)
        if DISPLACEMENT_PLOT_TIME_WINDOW is None:
            plot_mask = np.asarray(mask, dtype=bool)
        else:
            if len(DISPLACEMENT_PLOT_TIME_WINDOW) != 2:
                raise ValueError("DISPLACEMENT_PLOT_TIME_WINDOW must have exactly two values: (start, end).")
            t0 = float(DISPLACEMENT_PLOT_TIME_WINDOW[0])
            t1 = float(DISPLACEMENT_PLOT_TIME_WINDOW[1])
            t_lo = min(t0, t1)
            t_hi = max(t0, t1)
            plot_mask = (t >= t_lo) & (t <= t_hi)

        y_corr = np.asarray(corrected_disp_by_label[label], dtype=float)
        dt = float(result["dt"])
        try:
            _, acc_corr, _ = extracted._compute_derivatives(y_corr, dt=dt)
        except Exception:
            vel_corr = np.gradient(y_corr, dt)
            acc_corr = np.gradient(vel_corr, dt)

        acc_raw = _select_channel(result, "Acceleration y_ddot (m/s^2)")

        t_plot = t[plot_mask]
        acc_raw_plot = np.asarray(acc_raw, dtype=float)[plot_mask]
        acc_corr_plot = np.asarray(acc_corr, dtype=float)[plot_mask]
        if t_plot.size == 0:
            ax = axes_arr[i]
            ax.set_title(f"{label} (no samples in plot window)")
            ax.grid(True)
            ax.set_xlabel("Time (s)" if USE_RELATIVE_TIME else "Time")
            ax.set_ylabel("Acceleration y_ddot (m/s^2)")
            continue

        marker_step = max(1, int(np.ceil(max(t_plot.size, 1) / 120.0)))
        ax = axes_arr[i]
        ax.plot(
            t_plot,
            acc_raw_plot,
            linestyle=":",
            marker="x",
            markevery=marker_step,
            markersize=3.0,
            linewidth=1.0,
            alpha=0.85,
            label="Original y_ddot",
        )
        ax.plot(
            t_plot,
            acc_corr_plot,
            linestyle="-",
            marker="o",
            markevery=marker_step,
            markersize=3.0,
            linewidth=1.05,
            alpha=0.95,
            label="Corrected y_ddot",
        )
        ax.set_title(label)
        ax.grid(True)
        ax.set_xlabel("Time (s)" if USE_RELATIVE_TIME else "Time")
        ax.set_ylabel("Acceleration y_ddot (m/s^2)")
        if i == 0:
            ax.legend(loc="best", fontsize="small")

    for i in range(n, axes_arr.size):
        axes_arr[i].axis("off")

    if DISPLACEMENT_PLOT_TIME_WINDOW is None:
        fig_title = "Acceleration before/after phase-drift correction"
    else:
        t0 = float(DISPLACEMENT_PLOT_TIME_WINDOW[0])
        t1 = float(DISPLACEMENT_PLOT_TIME_WINDOW[1])
        fig_title = f"Acceleration before/after phase-drift correction [{min(t0, t1):g}, {max(t0, t1):g}]"
    fig.suptitle(fig_title, fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))


def _plot_phase_drift_all(
    results: list[dict[str, object]],
    *,
    disp_overrides: dict[str, np.ndarray] | None = None,
    title_suffix: str = "",
) -> None:
    fig_phase, (ax_py, ax_f, ax_l, ax_dl) = plt.subplots(
        4, 1, figsize=(extracted.FIGURE_WIDTH, 8.8), sharex=True
    )
    plotted = False

    for result in results:
        label = str(result["label"])
        disp_override = None if disp_overrides is None else disp_overrides.get(label)
        tc, ph_deg, lag_s, f_dom = _run_phase_analysis(result, disp_override=disp_override)
        tc, ph_deg, lag_s, f_dom = _filter_finite_phase_series(tc, ph_deg, lag_s, f_dom)
        if tc.size == 0:
            print(f"{label}: phase-drift summary{title_suffix} (vs y): insufficient valid windows.")
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
                f"{label}: phase-drift summary{title_suffix} (vs y): start={ph_deg[0]:.2f} deg, end={ph_deg[-1]:.2f} deg, "
                f"slope={slope_deg_per_s:.4f} deg/s"
            )

    if not plotted:
        plt.close(fig_phase)
        print(f"Phase-drift summary{title_suffix} (vs y): no valid phase windows found across selected files.")
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
    suffix = f" {title_suffix}" if title_suffix else ""
    ax_py.set_title(f"Sliding-window phase: (Fy2 - Fy1) relative to displacement{suffix}")
    lag_method = str(LAG_ESTIMATION_METHOD).strip().lower()
    if lag_method == "phase":
        ax_l.set_title("Sliding-window lag derived from phase/frequency")
    else:
        ax_l.set_title("Sliding-window lag from cross-correlation")
    ax_f.set_title("Dominant frequency used for phase estimate")
    ax_dl.set_title("Sliding-window lag drift rate")
    ax_py.axhline(0.0, color="black", linewidth=0.9, alpha=0.7)
    handles, labels = ax_py.get_legend_handles_labels()
    if handles:
        ncols = min(4, max(1, len(labels)))
        fig_phase.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=ncols,
            fontsize="small",
            frameon=True,
        )
        fig_phase.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    else:
        fig_phase.tight_layout()


def main() -> None:
    if not MAT_FILES:
        raise ValueError("MAT_FILES is empty.")

    # Ensure phase script can override these settings from the extraction script.
    extracted.DATA_VARIABLE = DATA_VARIABLE
    extracted.USE_RELATIVE_TIME = USE_RELATIVE_TIME
    extracted.PLOT_FIRST_SECONDS = PLOT_FIRST_SECONDS

    mat_files = [Path(p) for p in MAT_FILES]
    mat_files = _filter_excluded_tests(mat_files)
    if not mat_files:
        raise ValueError("No MAT files left after applying EXCLUDE_TEST_NUMBERS.")
    missing = [p for p in mat_files if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing MAT file(s): {missing}")

    results: list[dict[str, object]] = []
    for mat_file in mat_files:
        result = extracted._process_file(mat_file)
        results.append(result)

    _plot_phase_drift_all(results)

    if bool(PHASE_CORRECTION_ENABLED):
        phase_before = _collect_phase_series(results)
        corrected_disp = _build_corrected_displacement(results, phase_before)
        phase_after = _collect_phase_series(results, disp_overrides=corrected_disp)
        _print_phase_before_after_summary(phase_before, phase_after)
        if bool(PLOT_CORRECTED_DISPLACEMENT):
            _plot_corrected_displacement(results, corrected_disp)
        if bool(PLOT_CORRECTED_ACCELERATION):
            _plot_corrected_acceleration(results, corrected_disp)
        _plot_phase_drift_all(results, disp_overrides=corrected_disp, title_suffix="(corrected displacement)")
    plt.show()


if __name__ == "__main__":
    main()
