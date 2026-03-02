from __future__ import annotations

from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
from scipy.io import loadmat
from scipy.signal import savgol_filter


def bootstrap_import_paths(script_file: str | Path) -> tuple[Path, Path]:
    script_dir = Path(script_file).resolve().parent
    project_root = script_dir.parent
    for p in (project_root, script_dir):
        p_str = str(p)
        if p_str not in sys.path:
            sys.path.insert(0, p_str)
    return script_dir, project_root


def _build_analysis_fallback():
    class _AnalysisFallback:
        DATA_VARIABLE = "data"
        USE_RELATIVE_TIME = True
        TRIM_START_SECONDS = 0.0
        TRIM_END_SECONDS = 0.0
        REMOVE_INERTIA_FROM_CF = False
        INERTIA_INCLUDE_ADDED_MASS = True
        M = 16.79
        D = 0.1
        L = 1.0
        RUO = 1025.0
        FN = 1.119

        @staticmethod
        def _norm_name(name: str) -> str:
            return _norm_name(name)

        @staticmethod
        def _load_data_matrix(mat_file: Path, variable_name: str | None):
            return _load_data_matrix(Path(mat_file), variable_name)

        @staticmethod
        def _select_column(
            data: np.ndarray,
            channel_names: list[str] | None,
            aliases: list[str],
            fallback_idx: int,
            *,
            role: str,
        ) -> np.ndarray:
            return _select_column(data, channel_names, aliases, fallback_idx, role=role)

        @staticmethod
        def _fill_nonfinite_1d(values: np.ndarray, *, role: str) -> np.ndarray:
            return _fill_nonfinite_1d(values, role=role)

        @classmethod
        def _time_trim_mask(
            cls,
            time_values: np.ndarray,
            *,
            role: str,
            trim_start_seconds: float,
            trim_end_seconds: float,
        ) -> np.ndarray:
            t = np.asarray(time_values, dtype=float).reshape(-1)
            n = t.size
            if n == 0:
                return np.zeros(0, dtype=bool)
            mask = np.isfinite(t)
            if not np.any(mask):
                raise ValueError(f"{role}: time has no finite values.")

            t_f = t[mask]
            t0 = float(np.min(t_f))
            t1 = float(np.max(t_f))
            lo = t0 + max(0.0, float(trim_start_seconds))
            hi = t1 - max(0.0, float(trim_end_seconds))
            if hi < lo:
                raise ValueError(
                    f"{role}: invalid trim range (start={trim_start_seconds}, end={trim_end_seconds}) for [{t0}, {t1}]."
                )
            return mask & (t >= lo) & (t <= hi)

        @classmethod
        def _resolve_mat_files(cls) -> list[Path]:
            candidates = [
                Path("Experimental_Data/CrossFlow/RawData"),
                Path("CrossFlow/RawData"),
            ]
            for c in candidates:
                p = c.resolve()
                if p.exists() and p.is_dir():
                    return sorted(p.glob("*.mat"))
            return []

        @staticmethod
        def _spec(values: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray, int]:
            x = np.asarray(values, dtype=float).reshape(-1)
            x = x[np.isfinite(x)]
            n = int(x.size)
            if n < 2 or not np.isfinite(float(fs)) or float(fs) <= 0.0:
                return np.zeros(1, dtype=float), np.zeros(1, dtype=float), 1
            x = x - float(np.mean(x))
            yf = np.fft.rfft(x)
            freq = np.fft.rfftfreq(n, d=1.0 / float(fs))
            sp = (np.abs(yf) ** 2) / float(max(n, 1))
            return np.asarray(sp, dtype=float), np.asarray(freq, dtype=float), int(freq.size)

        @classmethod
        def _process_file(cls, mat_file: Path) -> dict[str, object]:
            data, channel_names = _load_data_matrix(Path(mat_file), cls.DATA_VARIABLE)
            if data.ndim != 2:
                raise ValueError(f"{Path(mat_file).name}: expected 2D matrix, got {data.shape}.")
            if data.shape[1] < 25:
                raise ValueError(f"{Path(mat_file).name}: expected at least 25 columns, got {data.shape}.")

            time = _fill_nonfinite_1d(
                _select_column(data, channel_names, ["Time", "time"], 0, role="time"),
                role=f"{Path(mat_file).name}: time",
            )
            flow = _fill_nonfinite_1d(
                _select_column(data, channel_names, ["Water_Speed"], 19, role="flow speed"),
                role=f"{Path(mat_file).name}: flow speed",
            )
            y = _fill_nonfinite_1d(
                _select_column(data, channel_names, ["xpos1", "y_corrected"], 23, role="displacement"),
                role=f"{Path(mat_file).name}: displacement",
            )
            fx_chain = _fill_nonfinite_1d(
                _select_column(data, channel_names, ["9131_FORCE_2"], 7, role="Fx_chain"),
                role=f"{Path(mat_file).name}: Fx_chain",
            )
            fx_spr = _fill_nonfinite_1d(
                _select_column(data, channel_names, ["9132_FORCE_3"], 8, role="Fx_spring"),
                role=f"{Path(mat_file).name}: Fx_spring",
            )
            fy1 = _fill_nonfinite_1d(
                _select_column(data, channel_names, ["9130_FORCE_1"], 6, role="Fy1"),
                role=f"{Path(mat_file).name}: Fy1",
            )
            fy2 = _fill_nonfinite_1d(
                _select_column(data, channel_names, ["9133_FORCE_4"], 9, role="Fy2"),
                role=f"{Path(mat_file).name}: Fy2",
            )

            n = int(min(time.size, flow.size, y.size, fx_chain.size, fx_spr.size, fy1.size, fy2.size))
            if n < 2:
                raise ValueError(f"{Path(mat_file).name}: not enough aligned samples.")
            time = np.asarray(time[:n], dtype=float)
            flow = np.asarray(flow[:n], dtype=float)
            y = np.asarray(y[:n], dtype=float)
            fx_chain = np.asarray(fx_chain[:n], dtype=float)
            fx_spr = np.asarray(fx_spr[:n], dtype=float)
            fy1 = np.asarray(fy1[:n], dtype=float)
            fy2 = np.asarray(fy2[:n], dtype=float)

            trim_mask = cls._time_trim_mask(
                time,
                role=f"{Path(mat_file).name}: trim",
                trim_start_seconds=float(cls.TRIM_START_SECONDS),
                trim_end_seconds=float(cls.TRIM_END_SECONDS),
            )
            if np.any(trim_mask):
                time = time[trim_mask]
                flow = flow[trim_mask]
                y = y[trim_mask]
                fx_chain = fx_chain[trim_mask]
                fx_spr = fx_spr[trim_mask]
                fy1 = fy1[trim_mask]
                fy2 = fy2[trim_mask]

            if cls.USE_RELATIVE_TIME and time.size > 0:
                time_plot = np.asarray(time - float(time[0]), dtype=float)
            else:
                time_plot = np.asarray(time, dtype=float)

            dt = float(np.median(np.diff(time))) if time.size >= 2 else float("nan")
            if not np.isfinite(dt) or dt <= 0.0:
                raise ValueError(f"{Path(mat_file).name}: invalid dt ({dt}).")
            fs = float(1.0 / dt)

            ur_inst = flow / (float(cls.FN) * float(cls.D))
            yvel = np.gradient(y, dt)
            yacc = np.gradient(yvel, dt)
            y_nd = (y - float(np.mean(y))) / float(cls.D)

            lb = 2.37
            la = 4.21 + 0.5
            f_drag = (fx_chain - fx_spr) * lb / la
            q_ref = 0.5 * float(cls.RUO) * float(cls.L) * float(cls.D) * (float(np.mean(flow)) ** 2)
            cdrag = f_drag / q_ref if np.isfinite(q_ref) and q_ref > 0.0 else np.full_like(f_drag, np.nan)

            sp_y, f_y, n_y = cls._spec(y, fs)
            if n_y > 1:
                pos = (f_y > 0.0) & np.isfinite(sp_y)
                if np.any(pos):
                    f_pos = f_y[pos]
                    s_pos = sp_y[pos]
                    k = int(np.argmax(s_pos))
                    ydom = float(f_pos[k])
                else:
                    ydom = float("nan")
            else:
                ydom = float("nan")

            summary = {
                "umean": float(np.mean(flow)),
                "ur": float(np.mean(ur_inst)),
                "cd": float(np.nanmean(cdrag)) if np.any(np.isfinite(cdrag)) else float("nan"),
                "ydomfreq": float(ydom),
                "ydomfreq_std": 0.0 if np.isfinite(ydom) else float("nan"),
                "nt": int(time.size),
                "dt": float(dt),
                "fs": float(fs),
                "coeff_norm_mode": "fallback",
                "m_added": 0.0,
                "m_inertia_removed": 0.0,
            }

            return {
                "path": Path(mat_file),
                "label": str(Path(mat_file).stem),
                "time": np.asarray(time, dtype=float),
                "time_plot": np.asarray(time_plot, dtype=float),
                "ur_inst": np.asarray(ur_inst, dtype=float),
                "y": np.asarray(y, dtype=float),
                "y_nd": np.asarray(y_nd, dtype=float),
                "yvel": np.asarray(yvel, dtype=float),
                "yacc": np.asarray(yacc, dtype=float),
                "cdrag": np.asarray(cdrag, dtype=float),
                "summary": summary,
            }

    return _AnalysisFallback


def import_analysis_and_phase(script_file: str | Path):
    bootstrap_import_paths(script_file)
    analysis = None
    phase = None

    # Import phase module independently.
    try:
        import Experimental_Data.phase_analysis as phase
    except ModuleNotFoundError:
        import phase_analysis as phase

    # Import analysis module if available, otherwise use local fallback.
    try:
        import Experimental_Data.analyze_experimental_data as analysis
    except ModuleNotFoundError:
        try:
            import analyze_experimental_data as analysis
        except ModuleNotFoundError:
            analysis = _build_analysis_fallback()

    return analysis, phase


def _build_extracted_fallback(analysis_module):
    class _ExtractedFallback:
        DATA_VARIABLE = analysis_module.DATA_VARIABLE
        USE_RELATIVE_TIME = bool(analysis_module.USE_RELATIVE_TIME)
        PLOT_FIRST_SECONDS = float(getattr(analysis_module, "FIRST_WINDOW_SECONDS", 10.0))
        FIGURE_WIDTH = 12.0

        @classmethod
        def _sync_analysis_config(cls) -> None:
            analysis_module.DATA_VARIABLE = cls.DATA_VARIABLE
            analysis_module.USE_RELATIVE_TIME = bool(cls.USE_RELATIVE_TIME)
            analysis_module.FIRST_WINDOW_SECONDS = float(cls.PLOT_FIRST_SECONDS)

        @classmethod
        def _process_file(cls, mat_file: Path) -> dict[str, object]:
            cls._sync_analysis_config()
            entry = analysis_module._process_file(mat_file)

            t = np.asarray(entry.get("time_plot", entry.get("time", [])), dtype=float).reshape(-1)
            ur = np.asarray(entry.get("ur_inst", []), dtype=float).reshape(-1)
            y = np.asarray(entry.get("y", []), dtype=float).reshape(-1)
            yacc = np.asarray(entry.get("yacc", []), dtype=float).reshape(-1)
            fy1 = np.asarray(entry.get("fy1_component", []), dtype=float).reshape(-1)
            fy2 = np.asarray(entry.get("fy2_component", []), dtype=float).reshape(-1)
            mask = np.asarray(entry.get("mask_early", []), dtype=bool).reshape(-1)

            n = int(min(t.size, ur.size, y.size, yacc.size, fy1.size, fy2.size))
            if n < 2:
                raise ValueError(f"{Path(mat_file).name}: not enough aligned samples in fallback phase loader.")

            t = t[:n]
            ur = ur[:n]
            y = y[:n]
            yacc = yacc[:n]
            fy1 = fy1[:n]
            fy2 = fy2[:n]
            if mask.size != n:
                mask = np.ones(n, dtype=bool)
            else:
                mask = mask[:n]

            fy_diff = np.asarray(fy2 - fy1, dtype=float)
            dt = float(np.median(np.diff(t)))
            if not np.isfinite(dt) or dt <= 0.0:
                dt = float(entry.get("summary", {}).get("dt", np.nan))
            if not np.isfinite(dt) or dt <= 0.0:
                raise ValueError(f"{Path(mat_file).name}: invalid dt in fallback phase loader.")

            channels = [
                ("Reduced velocity U_r (-)", np.asarray(ur, dtype=float)),
                ("Fy1 (scaled LB/LA, N)", np.asarray(fy1, dtype=float)),
                ("Fy2 (scaled LB/LA, N)", np.asarray(fy2, dtype=float)),
                ("Fy2 - Fy1 (scaled LB/LA, N)", np.asarray(fy_diff, dtype=float)),
                ("Displacement y (m)", np.asarray(y, dtype=float)),
                ("Acceleration y_ddot (m/s^2)", np.asarray(yacc, dtype=float)),
            ]

            return {
                "path": Path(entry.get("path", mat_file)),
                "label": str(entry.get("label", Path(mat_file).stem)),
                "t": np.asarray(t, dtype=float),
                "mask": np.asarray(mask, dtype=bool),
                "dt": float(dt),
                "channels": channels,
                "_fallback_from_analysis": True,
            }

        @staticmethod
        def _compute_derivatives(values: np.ndarray, *, dt: float) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
            y = np.asarray(values, dtype=float).reshape(-1)
            if y.size < 2:
                zeros = np.zeros_like(y)
                return zeros, zeros, {"mode": "fallback"}
            y_dot = np.gradient(y, float(dt))
            y_ddot = np.gradient(y_dot, float(dt))
            return np.asarray(y_dot, dtype=float), np.asarray(y_ddot, dtype=float), {"mode": "fallback"}

    return _ExtractedFallback


def _maybe_fix_orientation(arr: np.ndarray, *, min_cols: int = 25) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 2:
        return arr
    if arr.shape[1] < min_cols and arr.shape[0] >= min_cols:
        return arr.T
    if arr.shape[0] < arr.shape[1] and (arr.shape[1] / max(arr.shape[0], 1)) > 3.0:
        return arr.T
    return arr


def _norm_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _decode_matlab_char_array(arr: np.ndarray) -> str:
    flat = np.asarray(arr).reshape(-1)
    chars: list[str] = []
    for value in flat:
        code = int(value)
        if code == 0:
            continue
        chars.append(chr(code))
    return "".join(chars)


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
        if arr.size == 0:
            names.append("")
            continue
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
        if item_arr.size == 0:
            names.append("")
            continue
        if item_arr.dtype.kind in {"U", "S"}:
            names.append(str(item_arr.reshape(-1)[0]))
        elif item_arr.dtype.kind in {"u", "i"}:
            names.append(_decode_matlab_char_array(item_arr))
        else:
            names.append(str(item))
    return names


def _load_data_matrix_hdf5(mat_file: Path, variable_name: str | None) -> tuple[np.ndarray, list[str] | None]:
    try:
        import h5py  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            f"{mat_file.name} is MATLAB v7.3 (HDF5). Install h5py (pip install h5py)."
        ) from exc

    def _iter_datasets(group):
        for key in group.keys():
            if key == "#refs#":
                continue
            obj = group[key]
            if isinstance(obj, h5py.Dataset):
                yield obj
            elif isinstance(obj, h5py.Group):
                yield from _iter_datasets(obj)

    with h5py.File(mat_file, "r") as f:
        if variable_name is not None:
            if variable_name not in f:
                raise KeyError(
                    f"Variable '{variable_name}' not found in {mat_file}. "
                    f"Top-level keys: {sorted(list(f.keys()))}"
                )
            obj = f[variable_name]
            if not isinstance(obj, h5py.Dataset):
                raise ValueError(f"Variable '{variable_name}' is not a numeric dataset in {mat_file}.")
            arr = _maybe_fix_orientation(np.array(obj))
            if arr.ndim != 2:
                raise ValueError(f"Expected 2D matrix in {mat_file}, got {arr.shape}.")
            return arr, _extract_channel_names_from_hdf5(f)

        for ds in _iter_datasets(f):
            arr = _maybe_fix_orientation(np.array(ds))
            if arr.ndim == 2 and np.issubdtype(arr.dtype, np.number):
                return arr, _extract_channel_names_from_hdf5(f)

    raise ValueError(f"No 2D numeric dataset found in {mat_file}.")


def _load_data_matrix(mat_file: Path, variable_name: str | None) -> tuple[np.ndarray, list[str] | None]:
    try:
        raw = loadmat(mat_file, squeeze_me=True)
    except NotImplementedError:
        return _load_data_matrix_hdf5(mat_file, variable_name)
    except ValueError as exc:
        if "Unknown mat file type" in str(exc):
            return _load_data_matrix_hdf5(mat_file, variable_name)
        raise

    if variable_name is not None:
        if variable_name not in raw:
            keys = sorted(k for k in raw.keys() if not k.startswith("__"))
            raise KeyError(f"Variable '{variable_name}' not found in {mat_file}. Keys: {keys}")
        arr = _maybe_fix_orientation(np.asarray(raw[variable_name]))
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D matrix in {mat_file}, got {arr.shape}.")
        return arr, _extract_channel_names_from_raw(raw)

    for key, value in raw.items():
        if key.startswith("__"):
            continue
        arr = _maybe_fix_orientation(np.asarray(value))
        if arr.ndim == 2 and np.issubdtype(arr.dtype, np.number):
            return arr, _extract_channel_names_from_raw(raw)
    raise ValueError(f"No 2D numeric matrix found in {mat_file}.")


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
    if not (0 <= fallback_idx < data.shape[1]):
        raise IndexError(
            f"Fallback index {fallback_idx} out of bounds for role '{role}' and shape {data.shape}."
        )
    return np.asarray(data[:, fallback_idx]).reshape(-1)


def _fill_nonfinite_1d(values: np.ndarray, *, role: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = np.isfinite(arr)
    if not np.any(finite):
        raise ValueError(f"{role} contains no finite samples.")
    if np.all(finite):
        return arr
    idx = np.arange(arr.size, dtype=float)
    return np.interp(idx, idx[finite], arr[finite])


def _build_standalone_extracted_fallback():
    class _StandaloneExtractedFallback:
        DATA_VARIABLE = "data"
        USE_RELATIVE_TIME = True
        FIGURE_WIDTH = 12.0
        DERIV_SAVGOL_WINDOW = 71
        DERIV_SAVGOL_POLYORDER = 3

        @classmethod
        def _process_file(cls, mat_file: Path) -> dict[str, object]:
            data, channel_names = _load_data_matrix(Path(mat_file), cls.DATA_VARIABLE)
            if data.ndim != 2:
                raise ValueError(f"{Path(mat_file).name}: expected 2D data matrix, got {data.shape}.")
            if data.shape[1] < 25:
                raise ValueError(f"{Path(mat_file).name}: expected at least 25 columns, got {data.shape}.")

            time = _fill_nonfinite_1d(
                _select_column(data, channel_names, ["Time", "time"], 0, role="time"),
                role=f"{Path(mat_file).name}: time",
            )
            flow = _fill_nonfinite_1d(
                _select_column(data, channel_names, ["Water_Speed"], 19, role="flow speed"),
                role=f"{Path(mat_file).name}: flow speed",
            )
            y = _fill_nonfinite_1d(
                _select_column(data, channel_names, ["xpos1", "y_corrected"], 23, role="displacement"),
                role=f"{Path(mat_file).name}: displacement",
            )
            fy1 = _fill_nonfinite_1d(
                _select_column(data, channel_names, ["9130_FORCE_1"], 6, role="Fy1"),
                role=f"{Path(mat_file).name}: Fy1",
            )
            fy2 = _fill_nonfinite_1d(
                _select_column(data, channel_names, ["9133_FORCE_4"], 9, role="Fy2"),
                role=f"{Path(mat_file).name}: Fy2",
            )

            n = int(min(time.size, flow.size, y.size, fy1.size, fy2.size))
            if n < 2:
                raise ValueError(f"{Path(mat_file).name}: not enough aligned samples.")
            time = time[:n]
            flow = flow[:n]
            y = y[:n]
            fy1 = fy1[:n]
            fy2 = fy2[:n]

            dt = float(np.median(np.diff(time)))
            if not np.isfinite(dt) or dt <= 0.0:
                raise ValueError(f"{Path(mat_file).name}: invalid time axis (dt={dt}).")

            # Use same scaling convention as other scripts in this repo.
            lb = 2.37
            la = 4.21 + 0.5
            d = 0.1
            fn = 1.119
            ur = flow / (fn * d)
            fy1_scaled = fy1 * lb / la
            fy2_scaled = fy2 * lb / la
            fy_diff = fy2_scaled - fy1_scaled

            y_dot, y_ddot, _ = cls._compute_derivatives(y, dt=dt)

            t_plot = np.asarray(time - time[0], dtype=float) if bool(cls.USE_RELATIVE_TIME) else np.asarray(
                time, dtype=float
            )
            return {
                "path": Path(mat_file),
                "label": str(Path(mat_file).stem),
                "t": np.asarray(t_plot, dtype=float),
                "mask": np.ones(n, dtype=bool),
                "dt": float(dt),
                "channels": [
                    ("Reduced velocity U_r (-)", np.asarray(ur, dtype=float)),
                    ("Fy1 (scaled LB/LA, N)", np.asarray(fy1_scaled, dtype=float)),
                    ("Fy2 (scaled LB/LA, N)", np.asarray(fy2_scaled, dtype=float)),
                    ("Fy2 - Fy1 (scaled LB/LA, N)", np.asarray(fy_diff, dtype=float)),
                    ("Displacement y (m)", np.asarray(y, dtype=float)),
                    ("Acceleration y_ddot (m/s^2)", np.asarray(y_ddot, dtype=float)),
                ],
                "_fallback_standalone": True,
            }

        @staticmethod
        def _compute_derivatives(values: np.ndarray, *, dt: float) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
            y = np.asarray(values, dtype=float).reshape(-1)
            if y.size < 5:
                raise ValueError("Not enough samples for Savitzky-Golay derivatives.")
            delta = float(dt)
            if not np.isfinite(delta) or delta <= 0.0:
                raise ValueError(f"Invalid dt for Savitzky-Golay derivatives: {dt}")

            w = int(_StandaloneExtractedFallback.DERIV_SAVGOL_WINDOW)
            p = int(_StandaloneExtractedFallback.DERIV_SAVGOL_POLYORDER)
            if w % 2 == 0:
                w += 1
            if w > y.size:
                w = y.size if (y.size % 2 == 1) else (y.size - 1)
            if w <= p:
                w = p + 3
                if w % 2 == 0:
                    w += 1
                if w > y.size:
                    w = y.size if (y.size % 2 == 1) else (y.size - 1)
            if w < 5 or w <= p:
                raise ValueError(
                    "Could not configure Savitzky-Golay window/polyorder for derivative computation."
                )

            y_dot = savgol_filter(y, window_length=w, polyorder=p, deriv=1, delta=delta, mode="interp")
            y_ddot = savgol_filter(y, window_length=w, polyorder=p, deriv=2, delta=delta, mode="interp")
            return np.asarray(y_dot, dtype=float), np.asarray(y_ddot, dtype=float), {
                "mode": "savgol-standalone",
                "window_length": int(w),
                "polyorder": int(p),
            }

    return _StandaloneExtractedFallback


def import_analysis_and_extracted(
    script_file: str | Path,
    *,
    allow_extracted_fallback: bool = True,
    print_fallback_message: bool = True,
) -> tuple[Any, Any, bool]:
    bootstrap_import_paths(script_file)

    analysis = None
    try:
        import Experimental_Data.analyze_experimental_data as analysis
    except ModuleNotFoundError:
        try:
            import analyze_experimental_data as analysis
        except ModuleNotFoundError:
            analysis = None

    extracted = None
    try:
        import Experimental_Data.plot_extracted_channels as extracted  # type: ignore
    except ModuleNotFoundError:
        try:
            import plot_extracted_channels as extracted  # type: ignore
        except ModuleNotFoundError:
            extracted = None

    used_fallback = False
    if extracted is None and bool(allow_extracted_fallback):
        if analysis is not None:
            extracted = _build_extracted_fallback(analysis)
            fallback_msg = (
                "phase_analysis: 'plot_extracted_channels.py' not found; "
                "using analyze_experimental_data fallback for processing/derivatives."
            )
        else:
            extracted = _build_standalone_extracted_fallback()
            fallback_msg = (
                "phase_analysis: 'plot_extracted_channels.py' and "
                "'analyze_experimental_data.py' not found; "
                "using standalone MAT-processing fallback."
            )
        used_fallback = True
        if bool(print_fallback_message):
            print(fallback_msg)

    if extracted is None:
        raise ModuleNotFoundError(
            "Could not import plot_extracted_channels and no fallback was allowed/available."
        )

    return analysis, extracted, used_fallback


def extract_test_number(path: Path) -> int | None:
    stem = str(path.stem).lower()
    match = re.search(r"test(\d+)", stem)
    if match is None:
        return None
    return int(match.group(1))


def filter_excluded_tests(paths: list[Path], excluded_numbers: list[int] | tuple[int, ...]) -> list[Path]:
    excluded_raw = list(excluded_numbers)
    if not excluded_raw:
        return [Path(p) for p in paths]
    excluded = {int(v) for v in excluded_raw}
    kept: list[Path] = []
    for p in paths:
        test_no = extract_test_number(Path(p))
        if test_no is not None and test_no in excluded:
            continue
        kept.append(Path(p))
    return kept


def resolve_existing_dir(path_like: Path | str, *, script_file: str | Path) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        if p.exists() and p.is_dir():
            return p
        raise FileNotFoundError(f"Directory does not exist: {p}")

    # Try relative to current working directory first.
    cand_cwd = (Path.cwd() / p).resolve()
    if cand_cwd.exists() and cand_cwd.is_dir():
        return cand_cwd

    # Then try relative to the script directory.
    script_dir = Path(script_file).resolve().parent
    cand_script = (script_dir / p).resolve()
    if cand_script.exists() and cand_script.is_dir():
        return cand_script

    raise FileNotFoundError(
        f"Directory does not exist: {p} (tried {cand_cwd} and {cand_script})"
    )
