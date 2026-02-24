from __future__ import annotations

from pathlib import Path
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.signal import find_peaks
from scipy.signal import hilbert
from scipy.signal import savgol_filter


# Input file settings
MAT_FILES_CROSSFLOW_RAW = [
    Path("Experimental_Data/CrossFlow/RawData/test3002.mat"),
    Path("Experimental_Data/CrossFlow/RawData/test3003.mat"),
    Path("Experimental_Data/CrossFlow/RawData/test3004.mat"),
    Path("Experimental_Data/CrossFlow/RawData/test3005.mat"),
    Path("Experimental_Data/CrossFlow/RawData/test3006.mat"),
    Path("Experimental_Data/CrossFlow/RawData/test3008.mat"),
    Path("Experimental_Data/CrossFlow/RawData/test3009.mat"),
    Path("Experimental_Data/CrossFlow/RawData/test3010.mat"),
    Path("Experimental_Data/CrossFlow/RawData/test3011.mat"),
    Path("Experimental_Data/CrossFlow/RawData/test3012.mat"),
    Path("Experimental_Data/CrossFlow/RawData/test3014.mat"),
]

MAT_FILES_CROSSFLOW_CORRECTED = [
    Path("Experimental_Data/CrossFlow/CorrectedData/test3002_corrected.mat"),
    Path("Experimental_Data/CrossFlow/CorrectedData/test3003_corrected.mat"),
    Path("Experimental_Data/CrossFlow/CorrectedData/test3004_corrected.mat"),
    Path("Experimental_Data/CrossFlow/CorrectedData/test3005_corrected.mat"),
    Path("Experimental_Data/CrossFlow/CorrectedData/test3006_corrected.mat"),
    Path("Experimental_Data/CrossFlow/CorrectedData/test3008_corrected.mat"),
    Path("Experimental_Data/CrossFlow/CorrectedData/test3009_corrected.mat"),
    Path("Experimental_Data/CrossFlow/CorrectedData/test3010_corrected.mat"),
    Path("Experimental_Data/CrossFlow/CorrectedData/test3011_corrected.mat"),
    Path("Experimental_Data/CrossFlow/CorrectedData/test3012_corrected.mat"),
    Path("Experimental_Data/CrossFlow/CorrectedData/test3014_corrected.mat"),
]

MAT_FILES_COMBINED_RAW = [
    Path("Experimental_Data/Combined/test4001.mat"),
    Path("Experimental_Data/Combined/test4002.mat"),
    Path("Experimental_Data/Combined/test4003.mat"),
    Path("Experimental_Data/Combined/test4004.mat"),
    Path("Experimental_Data/Combined/test4005.mat"),
    Path("Experimental_Data/Combined/test4006.mat"),
    Path("Experimental_Data/Combined/test4007.mat"),
    Path("Experimental_Data/Combined/test4008.mat"),
    Path("Experimental_Data/Combined/test4009.mat"),
    Path("Experimental_Data/Combined/test4010.mat"),
    Path("Experimental_Data/Combined/test4011.mat"),
    Path("Experimental_Data/Combined/test4012.mat"),
    Path("Experimental_Data/Combined/test4013.mat"),
    Path("Experimental_Data/Combined/test5002.mat"),
    Path("Experimental_Data/Combined/test5003.mat"),
    Path("Experimental_Data/Combined/test5004.mat"),
    Path("Experimental_Data/Combined/test5005.mat"),
]

# Backward-compatible aliases.
MAT_FILES_CROSSFLOW = MAT_FILES_CROSSFLOW_RAW
MAT_FILES_COMBINED = MAT_FILES_COMBINED_RAW

# Choose input set:
# - "crossflow_raw" (legacy alias: "crossflow")
# - "crossflow_corrected" (legacy aliases: "corrected", "crossflow_corr")
# - "combined"
MAT_SOURCE = "crossflow_corrected"
MAT_SOURCE_KEY = str(MAT_SOURCE).strip().lower()
if MAT_SOURCE_KEY == "combined":
    MAT_FILES = MAT_FILES_COMBINED_RAW
    MAT_GLOB_BASE_DIR = Path("Experimental_Data/Combined")
    DATA_VARIABLE_DEFAULT = "data"
elif MAT_SOURCE_KEY in {"crossflow_corrected", "corrected", "crossflow_corr"}:
    MAT_FILES = MAT_FILES_CROSSFLOW_CORRECTED
    MAT_GLOB_BASE_DIR = Path("Experimental_Data/CrossFlow/CorrectedData")
    DATA_VARIABLE_DEFAULT = "data_corrected"
else:
    MAT_FILES = MAT_FILES_CROSSFLOW_RAW
    MAT_GLOB_BASE_DIR = Path("Experimental_Data/CrossFlow/RawData")
    DATA_VARIABLE_DEFAULT = "data"

MAT_GLOB = None  # e.g. "test40*.mat" (merged with MAT_FILES if set)
DATA_VARIABLE = DATA_VARIABLE_DEFAULT  # Set to None to auto-detect the first 2D numeric array.
FIRST_WINDOW_SECONDS = 10.0
USE_RELATIVE_TIME = True  # Plot time starting at 0 for each file when True.

# Exclude files by test number parsed from filename, e.g. `test3009.mat`
# or `test3009_corrected.mat` -> 3009.
EXCLUDE_TEST_NUMBERS: list[int] = [3009, 3002]

# Base physical parameters
D = 0.1  # Diameter of the test cylinder (m)
L = 1.0  # Length of the cylinder (m)
RUO = 1025.0  # Water density (kg/m3)
FN = 1.119  # Natural frequency (Hz)
M = 16.79  # Mass (kg)
K = 1218.0  # Stiffness (N/m)
LB = 2.37  # Length between spring location and top end (m)
LA = 4.21 + 0.5  # Length between cylinder center and top end (m)

# Added-mass correction on measured CF force:
# External-force convention:
# F_hydro = F_wake - m_a * y_ddot  =>  F_wake = F_hydro + m_a * y_ddot
# where m_a = C_a * rho * pi * D^2 * L / 4
REMOVE_ADDED_MASS_FROM_CF = True
ADDED_MASS_COEFF = 1.0

# Optional inertia removal on CF force (external-force convention):
# F_hydro = F_reduced - m_remove * y_ddot  =>  F_reduced = F_hydro + m_remove * y_ddot
# where m_remove = M (+ m_a optionally).
REMOVE_INERTIA_FROM_CF = True
INERTIA_INCLUDE_ADDED_MASS = True

# Force signal representation for force-related plots:
# - "coefficient": plot C_F and C_D (default)
# - "force": plot raw forces in N
FORCE_SIGNAL_REPRESENTATION = "coefficient"  # one of: coefficient, force
# Coefficient normalization reference:
# - "mean_u": q_ref = 0.5*rho*L*D*(U_mean^2)
# - "mean_u_plus_dy2": q_ref(t) = 0.5*rho*L*D*(U_mean^2 + y_dot(t)^2)
# Backward-compatible alias: "instantaneous_true_velocity" -> "mean_u_plus_dy2" (without U(t)).
COEFF_NORMALIZATION_MODE = "mean_u_plus_dy2"  # one of: mean_u, mean_u_plus_dy2
COEFF_NORM_EPS = 1e-12
# Global CF sign convention (set to -1.0 to flip CF phase/sign everywhere).
CF_SIGN = 1.0

# Savitzky-Golay settings for velocity/acceleration estimation from displacement.
USE_SAVGOL_DERIVATIVES = True
SAVGOL_WINDOW_LENGTH = 71
SAVGOL_POLYORDER = 3

# Spectrum plotting
NORMALIZE_SPECTRA = True  # If True, each spectrum is divided by its own peak value.
SPECTRUM_NORM_EPS = 1e-12
SPECTRUM_PLOT_MAX_HZ = 4.0

# Dominant-frequency detection band (avoids near-DC drift being selected as "oscillation frequency").
DOM_FREQ_MIN_HZ = 0.1
DOM_FREQ_MAX_HZ = 5.0

# Mean phase-portrait settings
PHASE_MEAN_BINS = 180
PHASE_MIN_SAMPLES_PER_BIN = 6

# Reference natural frequency line in Figure 5, using wet mass with C_a = 1.0.
REF_CA_FOR_FN_LINE = 1.0


def _using_corrected_input() -> bool:
    return MAT_SOURCE_KEY in {"crossflow_corrected", "corrected", "crossflow_corr"}


def _cf_force_mode_label() -> str:
    if _using_corrected_input():
        base = "calculated_force (corrected MAT)"
        if CF_SIGN < 0.0:
            base = f"-({base})"
        return base
    base = "Fy2 - Fy1"
    if CF_SIGN < 0.0:
        base = f"-({base})"
    if REMOVE_INERTIA_FROM_CF:
        if INERTIA_INCLUDE_ADDED_MASS:
            return f"{base}, inertia removed (m + m_a)"
        return f"{base}, inertia removed (m)"
    if REMOVE_ADDED_MASS_FROM_CF:
        return f"{base}, added mass removed"
    return base


def _use_raw_force_signals() -> bool:
    mode = str(FORCE_SIGNAL_REPRESENTATION).strip().lower()
    if mode not in {"coefficient", "force"}:
        raise ValueError("FORCE_SIGNAL_REPRESENTATION must be one of: coefficient, force")
    return mode == "force"


def _use_mean_u_plus_dy2_norm() -> bool:
    mode = str(COEFF_NORMALIZATION_MODE).strip().lower()
    if mode == "instantaneous_true_velocity":
        warnings.warn(
            "COEFF_NORMALIZATION_MODE='instantaneous_true_velocity' is deprecated; "
            "using 'mean_u_plus_dy2' (U_mean^2 + y_dot^2)."
        )
        mode = "mean_u_plus_dy2"
    if mode not in {"mean_u", "mean_u_plus_dy2"}:
        raise ValueError("COEFF_NORMALIZATION_MODE must be one of: mean_u, mean_u_plus_dy2")
    return mode == "mean_u_plus_dy2"


def _cf_label() -> str:
    return "CF force F_y (N)" if _use_raw_force_signals() else "CF coefficient C_F (-)"


def _drag_label() -> str:
    return "Drag force F_D (N)" if _use_raw_force_signals() else "Drag coefficient C_D (-)"


def _cf_name() -> str:
    return "CF force" if _use_raw_force_signals() else "CF coefficient"


def _drag_name() -> str:
    return "Drag force" if _use_raw_force_signals() else "Drag coefficient"


def _maybe_fix_orientation(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 2:
        return arr
    # For time-series tables we expect rows >> cols. If cols are much larger than rows,
    # the matrix is likely transposed and should be flipped.
    if arr.shape[0] < arr.shape[1] and (arr.shape[1] / max(arr.shape[0], 1)) > 3.0:
        return arr.T
    return arr


def _norm_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _decode_matlab_char_array(arr: np.ndarray) -> str:
    flat = np.asarray(arr).reshape(-1)
    out_chars: list[str] = []
    for value in flat:
        code = int(value)
        if code == 0:
            continue
        out_chars.append(chr(code))
    return "".join(out_chars)


def _extract_channel_names_from_hdf5(f) -> list[str] | None:
    if "chan_names" not in f:
        return None
    refs = np.asarray(f["chan_names"])
    names: list[str] = []
    for ref in refs.reshape(-1):
        if not ref:
            names.append("")
            continue
        ds = f[ref]
        arr = np.asarray(ds)
        if arr.dtype.kind in {"U", "S"}:
            names.append(str(arr.reshape(-1)[0]))
        elif arr.dtype.kind in {"u", "i"}:
            names.append(_decode_matlab_char_array(arr))
        else:
            names.append(str(arr))
    return names


def _extract_channel_names_from_raw(raw: dict) -> list[str] | None:
    chan = raw.get("chan_names")
    if chan is None:
        return None
    arr = np.asarray(chan)
    names: list[str] = []
    for item in arr.reshape(-1):
        if isinstance(item, str):
            names.append(item)
            continue
        item_arr = np.asarray(item)
        if item_arr.dtype.kind in {"U", "S"}:
            names.append(str(item_arr.reshape(-1)[0]))
        elif item_arr.dtype.kind in {"u", "i"}:
            names.append(_decode_matlab_char_array(item_arr))
        else:
            names.append(str(item))
    return names


def _load_data_matrix_hdf5(mat_file: Path, variable_name: str | None = "data") -> tuple[np.ndarray, list[str] | None]:
    try:
        import h5py  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "This MAT file is MATLAB v7.3 (HDF5). Install h5py to read it: pip install h5py"
        ) from exc

    def _iter_datasets(group, prefix: str = ""):
        for key in group.keys():
            if key == "#refs#":
                continue
            obj = group[key]
            path = f"{prefix}{key}"
            if isinstance(obj, h5py.Dataset):
                yield path, obj
            elif isinstance(obj, h5py.Group):
                yield from _iter_datasets(obj, prefix=f"{path}/")

    with h5py.File(mat_file, "r") as f:
        if variable_name is not None:
            if variable_name not in f:
                raise KeyError(
                    f"Variable '{variable_name}' not found in {mat_file}. "
                    f"Available top-level keys: {sorted(list(f.keys()))}"
                )
            obj = f[variable_name]
            if not isinstance(obj, h5py.Dataset):
                raise ValueError(
                    f"Variable '{variable_name}' exists in {mat_file} but is not a numeric dataset."
                )
            arr = _maybe_fix_orientation(np.array(obj))
            if arr.ndim != 2:
                raise ValueError(f"Variable '{variable_name}' must be 2D, got shape {arr.shape}")
            return arr, _extract_channel_names_from_hdf5(f)

        for _, ds in _iter_datasets(f):
            arr = _maybe_fix_orientation(np.array(ds))
            if arr.ndim == 2 and np.issubdtype(arr.dtype, np.number):
                return arr, _extract_channel_names_from_hdf5(f)

    raise ValueError(f"No 2D numeric array found in {mat_file}")


def _load_data_matrix(mat_file: Path, variable_name: str | None = "data") -> tuple[np.ndarray, list[str] | None]:
    if not mat_file.exists():
        raise FileNotFoundError(f"MAT file not found: {mat_file}")

    try:
        raw = loadmat(mat_file, squeeze_me=True)
    except NotImplementedError:
        return _load_data_matrix_hdf5(mat_file, variable_name)

    if variable_name is not None:
        if variable_name not in raw:
            raise KeyError(
                f"Variable '{variable_name}' not found in {mat_file}. "
                f"Available keys: {sorted(k for k in raw.keys() if not k.startswith('__'))}"
            )
        arr = _maybe_fix_orientation(np.asarray(raw[variable_name]))
        if arr.ndim != 2:
            raise ValueError(f"Variable '{variable_name}' must be 2D, got shape {arr.shape}")
        return arr, _extract_channel_names_from_raw(raw)

    for key, value in raw.items():
        if key.startswith("__"):
            continue
        arr = _maybe_fix_orientation(np.asarray(value))
        if arr.ndim == 2 and np.issubdtype(arr.dtype, np.number):
            return arr, _extract_channel_names_from_raw(raw)

    raise ValueError(f"No 2D numeric array found in {mat_file}")


def _select_column(
    data: np.ndarray,
    channel_names: list[str] | None,
    aliases: list[str],
    fallback_idx: int,
    *,
    role: str,
) -> np.ndarray:
    if channel_names is not None:
        index_map = {_norm_name(name): idx for idx, name in enumerate(channel_names)}
        for alias in aliases:
            idx = index_map.get(_norm_name(alias))
            if idx is not None and 0 <= idx < data.shape[1]:
                return np.asarray(data[:, idx]).reshape(-1)
        warnings.warn(
            f"Could not find channel(s) {aliases} by name for '{role}'. "
            f"Falling back to fixed column {fallback_idx + 1}."
        )
    if not (0 <= fallback_idx < data.shape[1]):
        raise IndexError(
            f"Fallback index {fallback_idx} out of bounds for role '{role}' and shape {data.shape}."
        )
    return np.asarray(data[:, fallback_idx]).reshape(-1)


def _find_column_index(
    data: np.ndarray,
    channel_names: list[str] | None,
    aliases: list[str],
) -> int | None:
    if channel_names is None:
        return None
    index_map = {_norm_name(name): idx for idx, name in enumerate(channel_names)}
    for alias in aliases:
        idx = index_map.get(_norm_name(alias))
        if idx is not None and 0 <= idx < data.shape[1]:
            return int(idx)
    return None


def _load_optional_vector_from_mat(mat_file: Path, variable_names: list[str]) -> np.ndarray | None:
    try:
        raw = loadmat(mat_file, squeeze_me=True)
    except NotImplementedError:
        try:
            import h5py  # type: ignore
        except Exception:
            return None
        with h5py.File(mat_file, "r") as f:
            for name in variable_names:
                if name not in f:
                    continue
                obj = f[name]
                if isinstance(obj, h5py.Dataset):
                    arr = np.asarray(obj)
                    if arr.ndim == 0:
                        return np.asarray([float(arr)], dtype=float)
                    if arr.ndim in (1, 2) and np.issubdtype(arr.dtype, np.number):
                        return np.asarray(arr, dtype=float).reshape(-1)
        return None

    for name in variable_names:
        if name not in raw:
            continue
        arr = np.asarray(raw[name])
        if arr.ndim == 0:
            return np.asarray([float(arr)], dtype=float)
        if arr.ndim in (1, 2) and np.issubdtype(arr.dtype, np.number):
            return np.asarray(arr, dtype=float).reshape(-1)
    return None


def _fit_vector_length(values: np.ndarray, *, n: int, role: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == n:
        return arr
    if arr.size == 1:
        return np.full(n, float(arr[0]), dtype=float)
    if arr.size < n:
        raise ValueError(f"{role}: length {arr.size} is shorter than required length {n}.")
    return arr[:n]


def _fill_nonfinite_1d(values: np.ndarray, *, role: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = np.isfinite(arr)
    if not np.any(finite):
        raise ValueError(f"{role} contains no finite samples.")
    if np.all(finite):
        return arr
    idx = np.arange(arr.size, dtype=float)
    arr[~finite] = np.interp(idx[~finite], idx[finite], arr[finite])
    return arr


def _infer_dt(time: np.ndarray, *, role: str) -> float:
    t = np.asarray(time, dtype=float).reshape(-1)
    if t.size < 2:
        raise ValueError(f"{role}: not enough samples to infer dt.")
    diffs = np.diff(t)
    diffs = diffs[np.isfinite(diffs)]
    diffs = diffs[diffs > 0.0]
    if diffs.size == 0:
        raise ValueError(f"{role}: could not infer positive dt from time vector.")
    dt = float(np.median(diffs))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"{role}: invalid inferred dt={dt}.")
    return dt


def _spec(signal: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray, int]:
    # Simple single-sided power spectrum (FFT-based), analogous to MATLAB helper `spec`.
    n = int(signal.size)
    x = np.asarray(signal, dtype=float).reshape(-1)
    finite = np.isfinite(x)
    if not np.any(finite):
        raise ValueError("Signal contains no finite values for spectrum calculation.")
    if not np.all(finite):
        idx = np.arange(x.size, dtype=float)
        x = np.interp(idx, idx[finite], x[finite])
    x = x - np.mean(x)
    fft_vals = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    spectrum = np.nan_to_num((np.abs(fft_vals) ** 2) / max(n, 1), nan=0.0, posinf=0.0, neginf=0.0)
    n_half = int(freqs.size)
    return spectrum, freqs, n_half


def _prepare_signal_for_fft(signal: np.ndarray) -> np.ndarray:
    x = np.asarray(signal, dtype=float).reshape(-1)
    finite = np.isfinite(x)
    if not np.any(finite):
        raise ValueError("Signal contains no finite values for phase calculation.")
    if not np.all(finite):
        idx = np.arange(x.size, dtype=float)
        x = np.interp(idx, idx[finite], x[finite])
    return x - float(np.mean(x))


def _phase_lag_deg_at_frequency(
    reference: np.ndarray,
    response: np.ndarray,
    *,
    fs: float,
    target_hz: float,
) -> float:
    if not np.isfinite(target_hz) or target_hz <= 0.0:
        return float("nan")
    r = _prepare_signal_for_fft(reference)
    y = _prepare_signal_for_fft(response)
    n = int(min(r.size, y.size))
    if n < 4:
        return float("nan")
    r = r[:n]
    y = y[:n]
    freqs = np.fft.rfftfreq(n, d=1.0 / float(fs))
    if freqs.size == 0:
        return float("nan")
    k = int(np.argmin(np.abs(freqs - float(target_hz))))
    if k == 0 and freqs.size > 1:
        k = 1
    rf = np.fft.rfft(r)
    yf = np.fft.rfft(y)
    cross = yf[k] * np.conj(rf[k])
    if np.abs(cross) <= 1e-20:
        return float("nan")
    return float(np.angle(cross, deg=True))


def _normalize_spectrum(spec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(spec, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(arr)) if arr.size > 0 else 0.0
    if peak <= float(eps):
        return arr.copy()
    return arr / peak


def _dominant_frequency_and_spread(
    freq: np.ndarray,
    spec: np.ndarray,
    *,
    fmin: float,
    fmax: float,
) -> tuple[float, float]:
    f = np.asarray(freq, dtype=float).reshape(-1)
    s = np.nan_to_num(np.asarray(spec, dtype=float).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    band = (f >= float(fmin)) & (f <= float(fmax))
    if not np.any(band):
        # Fallback for short/coarsely sampled signals where the requested
        # frequency range does not align with available FFT bins.
        pos = f > 0.0
        if not np.any(pos):
            warnings.warn(
                f"No spectral bins in [{fmin}, {fmax}] Hz and no positive-frequency bins available; "
                "returning NaN dominant frequency."
            )
            return float("nan"), float("nan")
        fb = f[pos]
        sb = s[pos]
        warnings.warn(
            f"No spectral bins in [{fmin}, {fmax}] Hz. "
            f"Falling back to available positive-frequency bins [{float(np.min(fb)):.6g}, {float(np.max(fb)):.6g}] Hz."
        )
    else:
        fb = f[band]
        sb = s[band]
    idx_peak = int(np.argmax(sb))
    f_peak = float(fb[idx_peak])
    w_sum = float(np.sum(sb))
    if w_sum <= 0.0:
        # Fallback to one-bin uncertainty when spectral power is degenerate.
        bin_width = float(np.median(np.diff(fb))) if fb.size > 1 else 0.0
        return f_peak, max(bin_width, 0.0)
    mean_f = float(np.sum(fb * sb) / w_sum)
    var_f = float(np.sum(((fb - mean_f) ** 2) * sb) / w_sum)
    std_f = float(np.sqrt(max(var_f, 0.0)))
    return f_peak, std_f


def _displacement_amplitude_stats(y_nd: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_nd, dtype=float).reshape(-1)
    finite = np.isfinite(y)
    y = y[finite]
    if y.size == 0:
        return {
            "amp_mean": np.nan,
            "amp_std": np.nan,
            "amp_min": np.nan,
            "amp_max": np.nan,
        }
    y_centered = y - float(np.mean(y))
    peaks, _ = find_peaks(y_centered)
    troughs, _ = find_peaks(-y_centered)
    amp_samples = np.concatenate([np.abs(y_centered[peaks]), np.abs(y_centered[troughs])])
    if amp_samples.size < 4:
        amp_samples = np.abs(y_centered)
    amp_samples = amp_samples[np.isfinite(amp_samples)]
    if amp_samples.size == 0:
        return {
            "amp_mean": np.nan,
            "amp_std": np.nan,
            "amp_min": np.nan,
            "amp_max": np.nan,
        }
    return {
        "amp_mean": float(np.mean(amp_samples)),
        "amp_std": float(np.std(amp_samples)),
        "amp_min": float(np.min(amp_samples)),
        "amp_max": float(np.max(amp_samples)),
    }


def _interp_periodic(theta: np.ndarray, values: np.ndarray) -> np.ndarray:
    t = np.asarray(theta, dtype=float).reshape(-1)
    v = np.asarray(values, dtype=float).reshape(-1)
    valid = np.isfinite(v)
    if not np.any(valid):
        raise ValueError("Cannot interpolate periodic values: no valid samples.")
    if np.count_nonzero(valid) == 1:
        return np.full_like(t, float(v[valid][0]))
    tv = t[valid]
    vv = v[valid]
    tv_ext = np.concatenate([tv - 2.0 * np.pi, tv, tv + 2.0 * np.pi])
    vv_ext = np.concatenate([vv, vv, vv])
    order = np.argsort(tv_ext)
    return np.interp(t, tv_ext[order], vv_ext[order])


def _phase_binned_phase_portrait(
    y: np.ndarray,
    v: np.ndarray,
    *,
    n_bins: int,
    min_samples_per_bin: int,
) -> dict[str, np.ndarray]:
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    v_arr = np.asarray(v, dtype=float).reshape(-1)
    finite = np.isfinite(y_arr) & np.isfinite(v_arr)
    y_arr = y_arr[finite]
    v_arr = v_arr[finite]
    if y_arr.size < max(n_bins, 10):
        raise ValueError("Too few finite samples for phase-binned portrait.")

    y_centered = y_arr - float(np.mean(y_arr))
    if np.allclose(np.std(y_centered), 0.0):
        phase = np.mod(np.arctan2(v_arr - float(np.mean(v_arr)), y_centered + 1e-12), 2.0 * np.pi)
    else:
        phase = np.mod(np.angle(hilbert(y_centered)), 2.0 * np.pi)

    edges = np.linspace(0.0, 2.0 * np.pi, int(n_bins) + 1)
    theta = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.searchsorted(edges, phase, side="right") - 1
    bin_idx = np.clip(bin_idx, 0, int(n_bins) - 1)

    mean_y = np.full(int(n_bins), np.nan, dtype=float)
    mean_v = np.full(int(n_bins), np.nan, dtype=float)
    std_y = np.full(int(n_bins), np.nan, dtype=float)
    std_v = np.full(int(n_bins), np.nan, dtype=float)
    cov_yy = np.full(int(n_bins), np.nan, dtype=float)
    cov_vv = np.full(int(n_bins), np.nan, dtype=float)
    cov_yv = np.full(int(n_bins), np.nan, dtype=float)
    counts = np.zeros(int(n_bins), dtype=int)

    for b in range(int(n_bins)):
        mask_b = bin_idx == b
        c = int(np.count_nonzero(mask_b))
        counts[b] = c
        if c < int(min_samples_per_bin):
            continue
        yb = y_arr[mask_b]
        vb = v_arr[mask_b]
        mean_y[b] = float(np.mean(yb))
        mean_v[b] = float(np.mean(vb))
        std_y[b] = float(np.std(yb))
        std_v[b] = float(np.std(vb))
        if c >= 2:
            cmat = np.cov(np.stack([yb, vb], axis=0), ddof=1)
            cov_yy[b] = float(cmat[0, 0])
            cov_yv[b] = float(cmat[0, 1])
            cov_vv[b] = float(cmat[1, 1])
        else:
            cov_yy[b] = std_y[b] ** 2
            cov_yv[b] = 0.0
            cov_vv[b] = std_v[b] ** 2

    mean_y = _interp_periodic(theta, mean_y)
    mean_v = _interp_periodic(theta, mean_v)
    std_y = _interp_periodic(theta, std_y)
    std_v = _interp_periodic(theta, std_v)
    cov_yy = _interp_periodic(theta, cov_yy)
    cov_yv = _interp_periodic(theta, cov_yv)
    cov_vv = _interp_periodic(theta, cov_vv)

    dmy = np.roll(mean_y, -1) - np.roll(mean_y, 1)
    dmv = np.roll(mean_v, -1) - np.roll(mean_v, 1)
    tnorm = np.hypot(dmy, dmv)
    tnorm = np.where(tnorm <= 1e-12, 1.0, tnorm)
    nx = -dmv / tnorm
    ny = dmy / tnorm

    var_n = nx * (cov_yy * nx + cov_yv * ny) + ny * (cov_yv * nx + cov_vv * ny)
    sigma_n = np.sqrt(np.clip(var_n, 0.0, None))
    upper_y = mean_y + sigma_n * nx
    upper_v = mean_v + sigma_n * ny
    lower_y = mean_y - sigma_n * nx
    lower_v = mean_v - sigma_n * ny

    return {
        "theta": theta,
        "mean_y": mean_y,
        "mean_v": mean_v,
        "std_y": std_y,
        "std_v": std_v,
        "upper_y": upper_y,
        "upper_v": upper_v,
        "lower_y": lower_y,
        "lower_v": lower_v,
        "counts": counts.astype(float),
    }


def _savgol_window_length(n_samples: int, preferred_window: int, polyorder: int) -> int:
    window = int(preferred_window)
    if window % 2 == 0:
        window += 1
    max_valid = n_samples if (n_samples % 2 == 1) else (n_samples - 1)
    window = min(window, max_valid)
    min_valid = int(polyorder) + 2
    if min_valid % 2 == 0:
        min_valid += 1
    if window < min_valid:
        return -1
    return window


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


def _resolve_mat_files() -> list[Path]:
    files = [Path(p) for p in MAT_FILES]
    if MAT_GLOB:
        files.extend(sorted(Path(MAT_GLOB_BASE_DIR).glob(str(MAT_GLOB))))
    files = _filter_excluded_tests(files)
    unique: list[Path] = []
    seen: set[Path] = set()
    for p in files:
        pp = p if p.is_absolute() else p
        if pp in seen:
            continue
        seen.add(pp)
        unique.append(pp)
    if not unique:
        raise FileNotFoundError("No MAT files selected. Set MAT_FILES and/or MAT_GLOB.")
    missing = [p for p in unique if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing MAT file(s): {missing}")
    return unique


def _process_file(mat_file: Path) -> dict[str, object]:
    data, channel_names = _load_data_matrix(mat_file, DATA_VARIABLE)
    if _using_corrected_input():
        if data.shape[1] < 5:
            raise ValueError(f"{mat_file.name}: corrected input must have at least 5 columns, got {data.shape}")

        is_raw_layout = bool(data.shape[1] >= 25)
        y_fallback = 23 if is_raw_layout else 1

        time = _fill_nonfinite_1d(
            _select_column(data, channel_names, ["time", "Time"], 0, role="time"),
            role=f"{mat_file.name}: time",
        )
        ypos = _fill_nonfinite_1d(
            _select_column(
                data,
                channel_names,
                ["y_corrected", "xpos1", "y"],
                y_fallback,
                role="corrected displacement",
            ),
            role=f"{mat_file.name}: corrected displacement",
        )

        nt = int(min(time.size, ypos.size))
        if nt < 2:
            raise ValueError(f"{mat_file.name}: not enough samples to compute sampling frequency.")
        time = np.asarray(time[:nt], dtype=float)
        ypos = np.asarray(ypos[:nt], dtype=float)

        dt = _infer_dt(time, role=f"{mat_file.name}: corrected time")
        fs = 1.0 / dt

        idx_ur = _find_column_index(data, channel_names, ["u_r", "ur", "U_r"])
        if idx_ur is not None:
            ur_inst = _fill_nonfinite_1d(
                _fit_vector_length(data[:, idx_ur], n=nt, role=f"{mat_file.name}: reduced velocity"),
                role=f"{mat_file.name}: reduced velocity",
            )
        else:
            ur_aux = _load_optional_vector_from_mat(mat_file, ["U_r", "u_r", "ur"])
            if ur_aux is not None:
                ur_inst = _fill_nonfinite_1d(
                    _fit_vector_length(ur_aux, n=nt, role=f"{mat_file.name}: reduced velocity"),
                    role=f"{mat_file.name}: reduced velocity",
                )
            elif is_raw_layout:
                flow = _select_column(data, channel_names, ["Water_Speed"], 19, role="flow speed")
                flow = _fill_nonfinite_1d(
                    _fit_vector_length(flow, n=nt, role=f"{mat_file.name}: flow speed"),
                    role=f"{mat_file.name}: flow speed",
                )
                ur_inst = flow / (FN * D)
            else:
                raise ValueError(f"{mat_file.name}: could not resolve reduced-velocity signal.")

        idx_yvel = _find_column_index(data, channel_names, ["dy_corrected", "dy", "e"])
        if idx_yvel is not None:
            yvel = _fill_nonfinite_1d(
                _fit_vector_length(data[:, idx_yvel], n=nt, role=f"{mat_file.name}: corrected velocity"),
                role=f"{mat_file.name}: corrected velocity",
            )
        else:
            yvel_aux = _load_optional_vector_from_mat(mat_file, ["dy_corrected", "dy", "e"])
            if yvel_aux is not None:
                yvel = _fill_nonfinite_1d(
                    _fit_vector_length(yvel_aux, n=nt, role=f"{mat_file.name}: corrected velocity"),
                    role=f"{mat_file.name}: corrected velocity",
                )
            else:
                yvel = np.gradient(ypos, dt)

        idx_force = _find_column_index(data, channel_names, ["calculated_force", "f_total", "F_total", "c"])
        if idx_force is not None:
            fy_combined = _fill_nonfinite_1d(
                _fit_vector_length(data[:, idx_force], n=nt, role=f"{mat_file.name}: corrected force"),
                role=f"{mat_file.name}: corrected force",
            )
        else:
            force_aux = _load_optional_vector_from_mat(
                mat_file,
                ["calculated_force", "F_total", "f_total", "c"],
            )
            if force_aux is not None:
                fy_combined = _fill_nonfinite_1d(
                    _fit_vector_length(force_aux, n=nt, role=f"{mat_file.name}: corrected force"),
                    role=f"{mat_file.name}: corrected force",
                )
            elif is_raw_layout:
                fy_spr1 = _select_column(data, channel_names, ["9130_FORCE_1"], 6, role="Fy_spring1")
                fy_spr2 = _select_column(data, channel_names, ["9133_FORCE_4"], 9, role="Fy_spring2")
                fy_spr1 = _fill_nonfinite_1d(
                    _fit_vector_length(fy_spr1, n=nt, role=f"{mat_file.name}: Fy_spring1"),
                    role=f"{mat_file.name}: Fy_spring1",
                )
                fy_spr2 = _fill_nonfinite_1d(
                    _fit_vector_length(fy_spr2, n=nt, role=f"{mat_file.name}: Fy_spring2"),
                    role=f"{mat_file.name}: Fy_spring2",
                )
                fy1 = (fy_spr1 - np.mean(fy_spr1)) * LB / LA
                fy2 = (fy_spr2 - np.mean(fy_spr2)) * LB / LA
                fy_combined = fy2 - fy1
            else:
                raise ValueError(f"{mat_file.name}: could not resolve corrected force signal.")

        yacc = np.gradient(yvel, dt)

        ur = float(np.mean(ur_inst))
        umean = float(ur * FN * D)
        fdrag = np.zeros_like(fy_combined)

        if _use_mean_u_plus_dy2_norm():
            coeff_norm_mode_used = "mean_u_plus_dy2"
            q_ref_vec = 0.5 * RUO * L * D * ((umean**2) + np.asarray(yvel, dtype=float) ** 2)
            q_ref_vec = np.maximum(q_ref_vec, float(COEFF_NORM_EPS))
        else:
            coeff_norm_mode_used = "mean_u"
            q_ref_scalar = max(0.5 * RUO * L * D * (umean**2), float(COEFF_NORM_EPS))
            q_ref_vec = np.full_like(np.asarray(ypos, dtype=float), q_ref_scalar, dtype=float)
        q_ref = float(np.nanmean(q_ref_vec))
        cdrag_coeff = np.zeros_like(fy_combined)
        cd = float(np.nanmean(cdrag_coeff))

        finite_y = np.isfinite(ypos)
        y_mean = float(np.nanmean(ypos)) if np.any(finite_y) else 0.0
        y_nd = (ypos - y_mean) / D

        m_added = float(ADDED_MASS_COEFF) * 0.25 * np.pi * RUO * D * D * L
        m_inertia_removed = 0.0
        fy_combined = float(CF_SIGN) * fy_combined
        cfy_coeff = fy_combined / q_ref_vec

        if _use_raw_force_signals():
            cfy = fy_combined
            cdrag = fdrag
        else:
            cfy = cfy_coeff
            cdrag = cdrag_coeff

        spcy, fhiy, nhiy = _spec(ypos, fs)
        if NORMALIZE_SPECTRA:
            spcy = _normalize_spectrum(spcy, eps=SPECTRUM_NORM_EPS)
        freq = np.asarray(fhiy, dtype=float)
        spec = np.asarray(spcy, dtype=float)
        ydomfreq, ydomfreq_std = _dominant_frequency_and_spread(
            freq,
            spec,
            fmin=DOM_FREQ_MIN_HZ,
            fmax=DOM_FREQ_MAX_HZ,
        )
        phase_cfy_y_deg = _phase_lag_deg_at_frequency(y_nd, cfy, fs=fs, target_hz=ydomfreq)
        phase_cdrag_y_deg = _phase_lag_deg_at_frequency(y_nd, cdrag, fs=fs, target_hz=ydomfreq)

        time_plot = time - float(time[0]) if USE_RELATIVE_TIME else time
        t_end = float(FIRST_WINDOW_SECONDS) if USE_RELATIVE_TIME else float(time[0]) + float(FIRST_WINDOW_SECONDS)
        mask_early = time_plot <= t_end

        spc_cfy, f_cfy, n_cfy = _spec(cfy, fs)
        spc_cdrag, f_cdrag, n_cdrag = _spec(cdrag, fs)
        spc_ur, f_ur, n_ur = _spec(ur_inst, fs)
        if NORMALIZE_SPECTRA:
            spc_cfy = _normalize_spectrum(spc_cfy, eps=SPECTRUM_NORM_EPS)
            spc_cdrag = _normalize_spectrum(spc_cdrag, eps=SPECTRUM_NORM_EPS)
            spc_ur = _normalize_spectrum(spc_ur, eps=SPECTRUM_NORM_EPS)

        amp_stats = _displacement_amplitude_stats(y_nd)

        return {
            "path": mat_file,
            "label": mat_file.stem,
            "time": np.asarray(time, dtype=float),
            "y": np.asarray(ypos, dtype=float),
            "yacc": np.asarray(yacc, dtype=float),
            "cfy_force": np.asarray(fy_combined, dtype=float),
            "is_corrected_input": True,
            "time_plot": time_plot,
            "ur_inst": ur_inst,
            "y_nd": y_nd,
            "yvel": yvel,
            "cdrag": cdrag,
            "cfy": cfy,
            "mask_early": mask_early,
            "sp_disp": (fhiy, spcy, nhiy),
            "sp_cfy": (f_cfy, spc_cfy, n_cfy),
            "sp_cdrag": (f_cdrag, spc_cdrag, n_cdrag),
            "sp_ur": (f_ur, spc_ur, n_ur),
            "summary": {
                "umean": umean,
                "ur": ur,
                "cd": cd,
                "ydomfreq": ydomfreq,
                "ydomfreq_std": ydomfreq_std,
                "phase_cfy_y_deg": phase_cfy_y_deg,
                "phase_cdrag_y_deg": phase_cdrag_y_deg,
                "amp_mean": amp_stats["amp_mean"],
                "amp_std": amp_stats["amp_std"],
                "amp_min": amp_stats["amp_min"],
                "amp_max": amp_stats["amp_max"],
                "nt": nt,
                "dt": dt,
                "fs": fs,
                "m_added": m_added,
                "m_inertia_removed": float(m_inertia_removed),
                "coeff_norm_mode": coeff_norm_mode_used,
            },
        }

    if data.shape[1] < 25:
        raise ValueError(f"{mat_file.name}: expected at least 25 columns, got shape {data.shape}")

    time = _select_column(data, channel_names, ["Time"], 0, role="time")
    u = _select_column(data, channel_names, ["Water_Speed"], 19, role="flow speed")
    ypos = _select_column(
        data,
        channel_names,
        ["xpos1"],  # Keep original MATLAB convention: CF displacement from xpos1 channel.
        23,
        role="CF displacement (xpos1)",
    )
    fx_chain = _select_column(data, channel_names, ["9131_FORCE_2"], 7, role="Fx_chain")
    fx_spr = _select_column(data, channel_names, ["9132_FORCE_3"], 8, role="Fx_spring")
    fy_spr1 = _select_column(data, channel_names, ["9130_FORCE_1"], 6, role="Fy_spring1")
    fy_spr2 = _select_column(data, channel_names, ["9133_FORCE_4"], 9, role="Fy_spring2")

    umean = float(np.mean(u))
    ur = float(umean / (FN * D))
    ur_inst = u / (FN * D)
    fdrag = (fx_chain - fx_spr) * LB / LA
    fy1 = (fy_spr1 - np.mean(fy_spr1)) * LB / LA
    fy2 = (fy_spr2 - np.mean(fy_spr2)) * LB / LA
    fy_combined = fy2 - fy1

    nt = int(ypos.size)
    if nt < 2:
        raise ValueError(f"{mat_file.name}: not enough samples to compute sampling frequency.")
    dt = _infer_dt(time, role=f"{mat_file.name}: raw time")
    fs = 1.0 / dt
    if USE_SAVGOL_DERIVATIVES:
        sg_window = _savgol_window_length(nt, SAVGOL_WINDOW_LENGTH, SAVGOL_POLYORDER)
        if sg_window < 0:
            warnings.warn(
                f"{mat_file.name}: Savgol derivative window invalid for this data length; "
                "falling back to finite differences."
            )
            yvel = np.gradient(ypos, dt)
            yacc = np.gradient(yvel, dt)
        else:
            yvel = savgol_filter(
                ypos,
                window_length=sg_window,
                polyorder=int(SAVGOL_POLYORDER),
                deriv=1,
                delta=dt,
                mode="interp",
            )
            yacc = savgol_filter(
                ypos,
                window_length=sg_window,
                polyorder=int(SAVGOL_POLYORDER),
                deriv=2,
                delta=dt,
                mode="interp",
            )
    else:
        yvel = np.gradient(ypos, dt)
        yacc = np.gradient(yvel, dt)

    if _use_mean_u_plus_dy2_norm():
        coeff_norm_mode_used = "mean_u_plus_dy2"
        q_ref_vec = 0.5 * RUO * L * D * ((umean**2) + np.asarray(yvel, dtype=float) ** 2)
        q_ref_vec = np.maximum(q_ref_vec, float(COEFF_NORM_EPS))
    else:
        coeff_norm_mode_used = "mean_u"
        q_ref_scalar = max(0.5 * RUO * L * D * (umean**2), float(COEFF_NORM_EPS))
        q_ref_vec = np.full_like(np.asarray(ypos, dtype=float), q_ref_scalar, dtype=float)
    q_ref = float(np.nanmean(q_ref_vec))
    cdrag_coeff = fdrag / q_ref_vec
    cd = float(np.nanmean(cdrag_coeff))

    finite_y = np.isfinite(ypos)
    y_mean = float(np.nanmean(ypos)) if np.any(finite_y) else 0.0
    y_nd = (ypos - y_mean) / D

    m_added = float(ADDED_MASS_COEFF) * 0.25 * np.pi * RUO * D * D * L
    m_inertia_removed = 0.0
    if REMOVE_INERTIA_FROM_CF:
        m_inertia_removed = float(M)
        if INERTIA_INCLUDE_ADDED_MASS:
            m_inertia_removed += m_added
    elif REMOVE_ADDED_MASS_FROM_CF:
        # Legacy behavior: remove only added-mass inertia.
        m_inertia_removed = m_added
    if m_inertia_removed != 0.0:
        f_inertia = m_inertia_removed * yacc
        fy_combined = fy_combined + f_inertia
    fy_combined = float(CF_SIGN) * fy_combined
    cfy_coeff = fy_combined / q_ref_vec

    if _use_raw_force_signals():
        cfy = fy_combined
        cdrag = fdrag
    else:
        cfy = cfy_coeff
        cdrag = cdrag_coeff

    spcy, fhiy, nhiy = _spec(ypos, fs)
    if NORMALIZE_SPECTRA:
        spcy = _normalize_spectrum(spcy, eps=SPECTRUM_NORM_EPS)
    freq = np.asarray(fhiy, dtype=float)
    spec = np.asarray(spcy, dtype=float)
    ydomfreq, ydomfreq_std = _dominant_frequency_and_spread(
        freq,
        spec,
        fmin=DOM_FREQ_MIN_HZ,
        fmax=DOM_FREQ_MAX_HZ,
    )
    phase_cfy_y_deg = _phase_lag_deg_at_frequency(y_nd, cfy, fs=fs, target_hz=ydomfreq)
    phase_cdrag_y_deg = _phase_lag_deg_at_frequency(y_nd, cdrag, fs=fs, target_hz=ydomfreq)

    time_plot = time - float(time[0]) if USE_RELATIVE_TIME else time
    t_end = float(FIRST_WINDOW_SECONDS) if USE_RELATIVE_TIME else float(time[0]) + float(FIRST_WINDOW_SECONDS)
    mask_early = time_plot <= t_end

    spc_cfy, f_cfy, n_cfy = _spec(cfy, fs)
    spc_cdrag, f_cdrag, n_cdrag = _spec(cdrag, fs)
    spc_ur, f_ur, n_ur = _spec(ur_inst, fs)
    if NORMALIZE_SPECTRA:
        spc_cfy = _normalize_spectrum(spc_cfy, eps=SPECTRUM_NORM_EPS)
        spc_cdrag = _normalize_spectrum(spc_cdrag, eps=SPECTRUM_NORM_EPS)
        spc_ur = _normalize_spectrum(spc_ur, eps=SPECTRUM_NORM_EPS)

    amp_stats = _displacement_amplitude_stats(y_nd)

    return {
        "path": mat_file,
        "label": mat_file.stem,
        "time": np.asarray(time, dtype=float),
        "y": np.asarray(ypos, dtype=float),
        "yacc": np.asarray(yacc, dtype=float),
        "cfy_force": np.asarray(fy_combined, dtype=float),
        "is_corrected_input": False,
        "time_plot": time_plot,
        "ur_inst": ur_inst,
        "y_nd": y_nd,
        "yvel": yvel,
        "cdrag": cdrag,
        "cfy": cfy,
        "mask_early": mask_early,
        "sp_disp": (fhiy, spcy, nhiy),
        "sp_cfy": (f_cfy, spc_cfy, n_cfy),
        "sp_cdrag": (f_cdrag, spc_cdrag, n_cdrag),
        "sp_ur": (f_ur, spc_ur, n_ur),
        "summary": {
            "umean": umean,
            "ur": ur,
            "cd": cd,
            "ydomfreq": ydomfreq,
            "ydomfreq_std": ydomfreq_std,
            "phase_cfy_y_deg": phase_cfy_y_deg,
            "phase_cdrag_y_deg": phase_cdrag_y_deg,
            "amp_mean": amp_stats["amp_mean"],
            "amp_std": amp_stats["amp_std"],
            "amp_min": amp_stats["amp_min"],
            "amp_max": amp_stats["amp_max"],
            "nt": nt,
            "dt": dt,
            "fs": fs,
            "m_added": m_added,
            "m_inertia_removed": float(m_inertia_removed),
            "coeff_norm_mode": coeff_norm_mode_used,
        },
    }


def _add_legends(axes) -> None:
    for ax in np.asarray(axes).reshape(-1):
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best", fontsize="small")


def main() -> None:
    mat_files = _resolve_mat_files()
    entries = [_process_file(p) for p in mat_files]

    for entry in entries:
        s = entry["summary"]
        assert isinstance(s, dict)
        print(
            f"{entry['label']}: Umean={s['umean']:.6f} m/s, Ur={s['ur']:.6f}, "
            f"Cd={s['cd']:.6f}, Ydomfreq={s['ydomfreq']:.6f} +/- {s['ydomfreq_std']:.6f} Hz, "
            f"nt={s['nt']}, dt={s['dt']:.6f} s, Fs={s['fs']:.6f} Hz, "
            f"coeff_norm={s['coeff_norm_mode']}"
        )
        if REMOVE_INERTIA_FROM_CF:
            print(
                f"{entry['label']}: applied inertia removal with "
                f"m_remove={s['m_inertia_removed']:.6f} kg "
                f"(M={M:.6f} kg, m_a={s['m_added']:.6f} kg, include_m_a={INERTIA_INCLUDE_ADDED_MASS})"
            )
        elif REMOVE_ADDED_MASS_FROM_CF:
            print(
                f"{entry['label']}: applied added-mass removal with "
                f"C_a={ADDED_MASS_COEFF:.3f}, m_a={s['m_added']:.6f} kg"
            )

    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(entries), 2)))
    n_rows = len(entries)

    # Figure 1: full timeseries, small multiples (rows=file, cols=metrics)
    fig1, axes1 = plt.subplots(n_rows, 4, figsize=(14, max(1.1 * n_rows, 2.0)), sharex="col")
    axes1 = np.atleast_2d(axes1)
    col_titles_full = [
        "Measured reduced velocity U_r",
        "Measured CF displacement y/D (mean removed)",
        f"Measured {_drag_name()}",
        f"Measured {_cf_name()}",
    ]
    col_titles_full[3] = f"Measured {_cf_name()} ({_cf_force_mode_label()})"
    for j, title in enumerate(col_titles_full):
        axes1[0, j].set_title(title)

    for i, entry in enumerate(entries):
        color = colors[i % len(colors)]
        label = str(entry["label"])
        t = np.asarray(entry["time_plot"])
        ax_ur, ax_y, ax_cd, ax_cf = axes1[i, 0], axes1[i, 1], axes1[i, 2], axes1[i, 3]
        ax_ur.plot(t, np.asarray(entry["ur_inst"]), color=color)
        ax_y.plot(t, np.asarray(entry["y_nd"]), color=color)
        ax_cd.plot(t, np.asarray(entry["cdrag"]), color=color)
        ax_cf.plot(t, np.asarray(entry["cfy"]), color=color)
        for ax in (ax_ur, ax_y, ax_cd, ax_cf):
            ax.grid(True)
        ax_ur.set_ylabel(f"{label}\nU_r (-)")
        ax_y.set_ylabel("y/D (-)")
        ax_cd.set_ylabel(_drag_label())
        ax_cf.set_ylabel(_cf_label())

    for j in range(4):
        axes1[-1, j].set_xlabel("Time (s)")
    fig1.tight_layout()

    # Figure 3: phase + spectra overlay
    fig3, axes3 = plt.subplots(2, 2, figsize=(11, 8))
    ax_phase = axes3[0, 0]
    for i, entry in enumerate(entries):
        color = colors[i % len(colors)]
        label = str(entry["label"])
        y_nd = np.asarray(entry["y_nd"])
        yvel = np.asarray(entry["yvel"])
        ax_phase.plot(y_nd, yvel, color=color, alpha=0.35, label=label)
        ax_phase.scatter(y_nd[0], yvel[0], color=color, s=22, zorder=3)
    ax_phase.grid(True)
    ax_phase.set_xlabel("CF displacement y/D (mean removed, -)")
    ax_phase.set_ylabel("CF velocity dy/dt (m/s)")
    ax_phase.set_title("CF phase diagram")
    _add_legends([ax_phase])

    ax_fy = axes3[0, 1]
    ax_fd = axes3[1, 0]
    ax_ur = axes3[1, 1]
    for i, entry in enumerate(entries):
        color = colors[i % len(colors)]
        label = str(entry["label"])
        f_cfy, sp_cfy, n_cfy = entry["sp_cfy"]
        f_cdrag, sp_cdrag, n_cdrag = entry["sp_cdrag"]
        f_ur, sp_ur, n_ur = entry["sp_ur"]
        ax_fy.plot(np.asarray(f_cfy)[: int(n_cfy)], np.asarray(sp_cfy)[: int(n_cfy)], color=color, label=label)
        ax_fd.plot(
            np.asarray(f_cdrag)[: int(n_cdrag)],
            np.asarray(sp_cdrag)[: int(n_cdrag)],
            color=color,
            label=label,
        )
        ax_ur.plot(np.asarray(f_ur)[: int(n_ur)], np.asarray(sp_ur)[: int(n_ur)], color=color, label=label)
    ax_fy.set_xlim(0.1, SPECTRUM_PLOT_MAX_HZ)
    ax_fy.grid(True)
    ax_fy.set_xlabel("Frequency (Hz)")
    ax_fy.set_ylabel("Normalized spectrum" if NORMALIZE_SPECTRA else "Spectrum")
    ax_fy.set_title(f"{_cf_name()} spectrum ({_cf_force_mode_label()})")
    ax_fd.set_xlim(0.1, SPECTRUM_PLOT_MAX_HZ)
    ax_fd.grid(True)
    ax_fd.set_xlabel("Frequency (Hz)")
    ax_fd.set_ylabel("Normalized spectrum" if NORMALIZE_SPECTRA else "Spectrum")
    ax_fd.set_title(f"{_drag_name()} spectrum")
    ax_ur.set_xlim(0.1, SPECTRUM_PLOT_MAX_HZ)
    ax_ur.grid(True)
    ax_ur.set_xlabel("Frequency (Hz)")
    ax_ur.set_ylabel("Normalized spectrum" if NORMALIZE_SPECTRA else "Spectrum")
    ax_ur.set_title("Reduced velocity spectrum")
    _add_legends([ax_fy, ax_fd, ax_ur])
    fig3.tight_layout()

    # Figure 4: first N seconds, small multiples (rows=file, cols=metrics)
    fig4, axes4 = plt.subplots(n_rows, 4, figsize=(14, max(1.0 * n_rows, 2.0)), sharex="col")
    axes4 = np.atleast_2d(axes4)
    col_titles = [
        f"Reduced velocity U_r (first {FIRST_WINDOW_SECONDS:g} s)",
        f"CF displacement y/D (mean removed, first {FIRST_WINDOW_SECONDS:g} s)",
        f"{_drag_name()} (first {FIRST_WINDOW_SECONDS:g} s)",
        f"{_cf_name()} (first {FIRST_WINDOW_SECONDS:g} s)",
    ]
    col_titles[3] = f"{_cf_name()} ({_cf_force_mode_label()}, first {FIRST_WINDOW_SECONDS:g} s)"
    for j, title in enumerate(col_titles):
        axes4[0, j].set_title(title)

    for i, entry in enumerate(entries):
        color = colors[i % len(colors)]
        label = str(entry["label"])
        mask = np.asarray(entry["mask_early"], dtype=bool)
        ax_ur, ax_y, ax_cd, ax_cf = axes4[i, 0], axes4[i, 1], axes4[i, 2], axes4[i, 3]
        if np.any(mask):
            t = np.asarray(entry["time_plot"])[mask]
            ax_ur.plot(t, np.asarray(entry["ur_inst"])[mask], color=color)
            ax_y.plot(t, np.asarray(entry["y_nd"])[mask], color=color)
            ax_cd.plot(t, np.asarray(entry["cdrag"])[mask], color=color)
            ax_cf.plot(t, np.asarray(entry["cfy"])[mask], color=color)
        for ax in (ax_ur, ax_y, ax_cd, ax_cf):
            ax.grid(True)
        ax_ur.set_ylabel(f"{label}\nU_r (-)")
        ax_y.set_ylabel("y/D (-)")
        ax_cd.set_ylabel(_drag_label())
        ax_cf.set_ylabel(_cf_label())

    for j in range(4):
        axes4[-1, j].set_xlabel("Time (s)")
    fig4.tight_layout()

    # Figure 5: summary trends vs mean reduced velocity
    entries_sorted = sorted(entries, key=lambda e: float(e["summary"]["ur"]))
    ur_vals = np.array([float(e["summary"]["ur"]) for e in entries_sorted], dtype=float)
    ydom_vals = np.array([float(e["summary"]["ydomfreq"]) for e in entries_sorted], dtype=float)
    ydom_std_vals = np.array([float(e["summary"]["ydomfreq_std"]) for e in entries_sorted], dtype=float)
    amp_mean_vals = np.array([float(e["summary"]["amp_mean"]) for e in entries_sorted], dtype=float)
    amp_std_vals = np.array([float(e["summary"]["amp_std"]) for e in entries_sorted], dtype=float)
    amp_min_vals = np.array([float(e["summary"]["amp_min"]) for e in entries_sorted], dtype=float)
    amp_max_vals = np.array([float(e["summary"]["amp_max"]) for e in entries_sorted], dtype=float)
    labels_sorted = [str(e["label"]) for e in entries_sorted]

    fig5, axes5 = plt.subplots(1, 2, figsize=(12, 4.6))
    ax_f, ax_a = axes5

    m_added_ref = float(REF_CA_FOR_FN_LINE) * 0.25 * np.pi * RUO * D * D * L
    f_n_ref = (1.0 / (2.0 * np.pi)) * np.sqrt(K / (M + m_added_ref))

    freq_mask = np.isfinite(ur_vals) & np.isfinite(ydom_vals) & np.isfinite(ydom_std_vals)
    ur_freq = ur_vals[freq_mask]
    ydom_freq = ydom_vals[freq_mask]
    ydom_std_freq = ydom_std_vals[freq_mask]
    labels_freq = [name for name, keep in zip(labels_sorted, freq_mask) if keep]

    ax_f.fill_between(
        ur_freq,
        ydom_freq - ydom_std_freq,
        ydom_freq + ydom_std_freq,
        color="tab:blue",
        alpha=0.30,
        edgecolor="none",
        linewidth=0.0,
        label="±1σ spectral spread",
    )
    ax_f.errorbar(
        ur_freq,
        ydom_freq,
        yerr=ydom_std_freq,
        fmt="o",
        linestyle="none",
        capsize=3.0,
        color="tab:blue",
        label="Dominant frequency",
    )
    ax_f.axhline(
        f_n_ref,
        color="black",
        linewidth=1.5,
        linestyle="-",
        label=f"Natural frequency (C_a={REF_CA_FOR_FN_LINE:.1f})",
    )
    for x, yv, name in zip(ur_freq, ydom_freq, labels_freq):
        ax_f.annotate(name, (x, yv), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax_f.grid(True)
    ax_f.set_xlabel("Mean reduced velocity U_r (-)")
    ax_f.set_ylabel("Dominant CF oscillation frequency (Hz)")
    ax_f.set_title("Mean oscillation frequency vs mean reduced velocity")
    ax_f.legend(loc="best", fontsize="small")

    amp_mask = (
        np.isfinite(ur_vals)
        & np.isfinite(amp_mean_vals)
        & np.isfinite(amp_std_vals)
        & np.isfinite(amp_min_vals)
        & np.isfinite(amp_max_vals)
    )
    ur_amp = ur_vals[amp_mask]
    amp_mean = amp_mean_vals[amp_mask]
    amp_std = amp_std_vals[amp_mask]
    amp_min = amp_min_vals[amp_mask]
    amp_max = amp_max_vals[amp_mask]

    ax_a.errorbar(
        ur_amp,
        amp_mean,
        yerr=amp_std,
        fmt="o",
        linestyle="none",
        capsize=3.0,
        color="tab:orange",
        label="Mean amplitude",
    )
    ax_a.fill_between(
        ur_amp,
        amp_min,
        amp_max,
        color="tab:orange",
        alpha=0.30,
        edgecolor="none",
        linewidth=0.0,
        label="Min-Max band",
    )
    ax_a.fill_between(
        ur_amp,
        amp_mean - amp_std,
        amp_mean + amp_std,
        color="tab:orange",
        alpha=0.45,
        edgecolor="none",
        linewidth=0.0,
        label="±1σ band",
    )
    ax_a.grid(True)
    ax_a.set_xlabel("Mean reduced velocity U_r (-)")
    ax_a.set_ylabel("Displacement amplitude |y/D| (-)")
    ax_a.set_title("Displacement amplitude vs mean reduced velocity")
    ax_a.legend(loc="best", fontsize="small")

    fig5.tight_layout()

    # Figure 6: phase-binned mean phase portrait with uncertainty
    fig6, axes6 = plt.subplots(1, 3, figsize=(14, 4.8))
    ax_phase_mean, ax_y_phase, ax_v_phase = axes6

    for i, entry in enumerate(entries):
        color = colors[i % len(colors)]
        label = str(entry["label"])
        try:
            phase_stats = _phase_binned_phase_portrait(
                np.asarray(entry["y_nd"]),
                np.asarray(entry["yvel"]),
                n_bins=int(PHASE_MEAN_BINS),
                min_samples_per_bin=int(PHASE_MIN_SAMPLES_PER_BIN),
            )
        except ValueError as exc:
            warnings.warn(f"{label}: skipped mean phase plot ({exc})")
            continue

        theta = np.asarray(phase_stats["theta"], dtype=float)
        mean_y = np.asarray(phase_stats["mean_y"], dtype=float)
        mean_v = np.asarray(phase_stats["mean_v"], dtype=float)
        std_y = np.asarray(phase_stats["std_y"], dtype=float)
        std_v = np.asarray(phase_stats["std_v"], dtype=float)
        upper_y = np.asarray(phase_stats["upper_y"], dtype=float)
        upper_v = np.asarray(phase_stats["upper_v"], dtype=float)
        lower_y = np.asarray(phase_stats["lower_y"], dtype=float)
        lower_v = np.asarray(phase_stats["lower_v"], dtype=float)

        poly_x = np.concatenate([upper_y, lower_y[::-1]])
        poly_y = np.concatenate([upper_v, lower_v[::-1]])
        ax_phase_mean.fill(poly_x, poly_y, color=color, alpha=0.28, edgecolor="none")
        ax_phase_mean.plot(mean_y, mean_v, color=color, linewidth=1.8, label=label)

        phase_norm = theta / (2.0 * np.pi)
        ax_y_phase.fill_between(
            phase_norm,
            mean_y - std_y,
            mean_y + std_y,
            color=color,
            alpha=0.28,
            edgecolor="none",
        )
        ax_y_phase.plot(phase_norm, mean_y, color=color, linewidth=1.8, label=label)

        ax_v_phase.fill_between(
            phase_norm,
            mean_v - std_v,
            mean_v + std_v,
            color=color,
            alpha=0.28,
            edgecolor="none",
        )
        ax_v_phase.plot(phase_norm, mean_v, color=color, linewidth=1.8, label=label)

    ax_phase_mean.grid(True)
    ax_phase_mean.set_xlabel("CF displacement y/D (mean removed, -)")
    ax_phase_mean.set_ylabel("CF velocity dy/dt (m/s)")
    ax_phase_mean.set_title("Mean phase portrait with ±1σ tube")

    ax_y_phase.grid(True)
    ax_y_phase.set_xlabel("Phase / 2π (-)")
    ax_y_phase.set_ylabel("CF displacement y/D (mean removed, -)")
    ax_y_phase.set_title("Mean displacement vs phase")

    ax_v_phase.grid(True)
    ax_v_phase.set_xlabel("Phase / 2π (-)")
    ax_v_phase.set_ylabel("CF velocity dy/dt (m/s)")
    ax_v_phase.set_title("Mean velocity vs phase")

    _add_legends([ax_phase_mean])
    fig6.tight_layout()

    # Figure 7: hysteresis loops
    fig7, (ax_h_cf, ax_h_cd) = plt.subplots(1, 2, figsize=(12, 4.8))
    for i, entry in enumerate(entries):
        color = colors[i % len(colors)]
        label = str(entry["label"])
        y_nd = np.asarray(entry["y_nd"], dtype=float)
        cfy = np.asarray(entry["cfy"], dtype=float)
        cdrag = np.asarray(entry["cdrag"], dtype=float)
        mask_cf = np.isfinite(y_nd) & np.isfinite(cfy)
        mask_cd = np.isfinite(y_nd) & np.isfinite(cdrag)
        ax_h_cf.plot(y_nd[mask_cf], cfy[mask_cf], color=color, alpha=0.55, linewidth=1.0, label=label)
        ax_h_cd.plot(y_nd[mask_cd], cdrag[mask_cd], color=color, alpha=0.55, linewidth=1.0, label=label)
    ax_h_cf.grid(True)
    ax_h_cf.set_xlabel("Displacement y/D (mean removed, -)")
    ax_h_cf.set_ylabel(_cf_label())
    ax_h_cf.set_title(f"Hysteresis loop: {_cf_name()} vs y/D")
    ax_h_cd.grid(True)
    ax_h_cd.set_xlabel("Displacement y/D (mean removed, -)")
    ax_h_cd.set_ylabel(_drag_label())
    ax_h_cd.set_title(f"Hysteresis loop: {_drag_name()} vs y/D")
    _add_legends([ax_h_cf, ax_h_cd])
    fig7.tight_layout()

    # Figure 8: phase lag vs reduced velocity
    phase_cfy_vals = np.array([float(e["summary"]["phase_cfy_y_deg"]) for e in entries_sorted], dtype=float)
    phase_cdrag_vals = np.array([float(e["summary"]["phase_cdrag_y_deg"]) for e in entries_sorted], dtype=float)

    fig8, (ax_p_cf, ax_p_cd) = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)

    phase_cf_mask = np.isfinite(ur_vals) & np.isfinite(phase_cfy_vals)
    ur_phase_cf = ur_vals[phase_cf_mask]
    phase_cf = phase_cfy_vals[phase_cf_mask]
    labels_phase_cf = [name for name, keep in zip(labels_sorted, phase_cf_mask) if keep]
    ax_p_cf.scatter(
        ur_phase_cf,
        phase_cf,
        color="tab:blue",
        s=42,
        label=f"Phase({_cf_name()}) - Phase(y)",
    )
    ax_p_cf.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    for x, yv, name in zip(ur_phase_cf, phase_cf, labels_phase_cf):
        ax_p_cf.annotate(name, (x, yv), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax_p_cf.grid(True)
    ax_p_cf.set_xlabel("Mean reduced velocity U_r (-)")
    ax_p_cf.set_ylabel("Phase lag (deg)")
    ax_p_cf.set_title(f"Phase lag: {_cf_name()} relative to y")
    ax_p_cf.set_ylim(-185.0, 185.0)
    ax_p_cf.legend(loc="best", fontsize="small")

    phase_cd_mask = np.isfinite(ur_vals) & np.isfinite(phase_cdrag_vals)
    ur_phase_cd = ur_vals[phase_cd_mask]
    phase_cd = phase_cdrag_vals[phase_cd_mask]
    labels_phase_cd = [name for name, keep in zip(labels_sorted, phase_cd_mask) if keep]
    ax_p_cd.scatter(
        ur_phase_cd,
        phase_cd,
        color="tab:orange",
        s=42,
        label=f"Phase({_drag_name()}) - Phase(y)",
    )
    ax_p_cd.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    for x, yv, name in zip(ur_phase_cd, phase_cd, labels_phase_cd):
        ax_p_cd.annotate(name, (x, yv), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax_p_cd.grid(True)
    ax_p_cd.set_xlabel("Mean reduced velocity U_r (-)")
    ax_p_cd.set_title(f"Phase lag: {_drag_name()} relative to y")
    ax_p_cd.legend(loc="best", fontsize="small")

    fig8.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
