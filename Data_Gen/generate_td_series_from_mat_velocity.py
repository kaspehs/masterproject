from __future__ import annotations

import json
from pathlib import Path
import warnings

import numpy as np
from scipy.io import loadmat
from scipy.signal import savgol_filter

try:
    from utils import vforce_CF
    from simulate_td_model_cf import C, Ca, Cd, Cv, D, K, M, fhat0, fhat_max, fhat_min, n_memory, rho
except ModuleNotFoundError:
    from Data_Gen.utils import vforce_CF
    from Data_Gen.simulate_td_model_cf import C, Ca, Cd, Cv, D, K, M, fhat0, fhat_max, fhat_min, n_memory, rho


# -------------------------------------------------------------------
# 1) Input/Output config
# -------------------------------------------------------------------
INPUT_MAT_DIR = Path("Experimental_Data/CrossFlow/CleanedCorrectedSmoothedData")
INPUT_GLOB = "*.mat"
DATA_VARIABLE = "data"  # Set to None to auto-detect first 2D numeric matrix.

OUTPUT_DIR = Path("Data_Gen/generated_series_from_mat_velocity")
OVERWRITE = True

# Export layout:
# - "flat": one .npz per source .mat in OUTPUT_DIR
# - "split": split each generated series into fixed-length chunks and place
#            them in train/val/test subfolders.
EXPORT_LAYOUT = "split"  # one of: flat, split

# Chunking config (used when EXPORT_LAYOUT == "split")
SPLIT_CHUNK_SECONDS = 20.0
TRAIN_FRACTION = 0.80
VAL_FRACTION = 0.20
TEST_FRACTION = 0.0
SPLIT_SEED = 1234
MIN_CHUNK_SAMPLES = 3

# Time-series preprocessing
TRIM_START_SECONDS = 0.0
TRIM_END_SECONDS = 0.0
DOWNSAMPLE_STRIDE = 1

# Channel lookup in MAT data matrix
TIME_CHANNEL_ALIASES = ["Time", "time"]
FLOW_CHANNEL_ALIASES = ["Water_Speed", "water_speed", "U", "u", "FlowSpeed"]
TIME_FALLBACK_COL = 0
FLOW_FALLBACK_COL = 19

# Flow preprocessing
MIN_FLOW_SPEED = 1e-6
USE_SAVGOL_ON_FLOW = False
FLOW_SAVGOL_WINDOW = 31
FLOW_SAVGOL_POLYORDER = 3

# Reduced-velocity export mode
# - "mean": constant U_r channel (compatible with current loaders expecting
#           series-constant U_r)
# - "instantaneous": U_r(t) based on U(t)
UR_EXPORT_MODE = "mean"  # one of: mean, instantaneous

# TD simulation settings
INTEGRATOR = "rk4"  # one of: rk4, euler
INIT_A_FACTOR = 0.6
INIT_FHAT = 0.15
INIT_PHASE_RAD = 0.0


def _validate_config() -> None:
    layout = str(EXPORT_LAYOUT).strip().lower()
    if layout not in {"flat", "split"}:
        raise ValueError("EXPORT_LAYOUT must be one of: flat, split")
    if layout == "split":
        if not np.isfinite(float(SPLIT_CHUNK_SECONDS)) or float(SPLIT_CHUNK_SECONDS) <= 0.0:
            raise ValueError("SPLIT_CHUNK_SECONDS must be finite and > 0.")
        fracs = np.array([TRAIN_FRACTION, VAL_FRACTION, TEST_FRACTION], dtype=float)
        if not np.all(np.isfinite(fracs)) or np.any(fracs < 0.0):
            raise ValueError("TRAIN_FRACTION/VAL_FRACTION/TEST_FRACTION must be finite and >= 0.")
        if float(fracs.sum()) <= 0.0:
            raise ValueError("At least one split fraction must be > 0.")
    if int(DOWNSAMPLE_STRIDE) < 1:
        raise ValueError("DOWNSAMPLE_STRIDE must be >= 1.")
    mode = str(UR_EXPORT_MODE).strip().lower()
    if mode not in {"mean", "instantaneous"}:
        raise ValueError("UR_EXPORT_MODE must be one of: mean, instantaneous")
    integrator = str(INTEGRATOR).strip().lower()
    if integrator not in {"rk4", "euler"}:
        raise ValueError("INTEGRATOR must be one of: rk4, euler")


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


def _maybe_fix_orientation(arr: np.ndarray, *, min_cols: int = 25) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 2:
        return arr
    if arr.shape[1] < min_cols and arr.shape[0] >= min_cols:
        return arr.T
    if arr.shape[0] < arr.shape[1] and (arr.shape[1] / max(arr.shape[0], 1)) > 3.0:
        return arr.T
    return arr


def _load_data_matrix_hdf5(mat_file: Path, variable_name: str | None) -> tuple[np.ndarray, list[str] | None]:
    try:
        import h5py  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            f"{mat_file.name} looks like MATLAB v7.3 (HDF5). Install h5py (pip install h5py)."
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
            arr = _maybe_fix_orientation(np.array(obj))
            if arr.ndim != 2:
                raise ValueError(f"Expected 2D matrix in {mat_file}, got shape {arr.shape}.")
            return arr, _extract_channel_names_from_hdf5(f)

        for ds in _iter_datasets(f):
            arr = _maybe_fix_orientation(np.array(ds))
            if arr.ndim == 2 and np.issubdtype(arr.dtype, np.number):
                return arr, _extract_channel_names_from_hdf5(f)

    raise ValueError(f"No 2D numeric matrix found in {mat_file}.")


def _load_data_matrix(mat_file: Path, variable_name: str | None) -> tuple[np.ndarray, list[str] | None]:
    try:
        raw = loadmat(mat_file, squeeze_me=True)
    except (NotImplementedError, OSError):
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
            raise ValueError(f"Expected 2D matrix in {mat_file}, got shape {arr.shape}.")
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
                return np.asarray(data[:, idx], dtype=float).reshape(-1)
        warnings.warn(
            f"Could not find channel(s) {aliases} by name for '{role}'. "
            f"Falling back to fixed column {fallback_idx + 1}."
        )
    if not (0 <= fallback_idx < data.shape[1]):
        raise IndexError(f"Fallback index {fallback_idx} out of bounds for '{role}', shape={data.shape}.")
    return np.asarray(data[:, fallback_idx], dtype=float).reshape(-1)


def _fill_nonfinite_1d(values: np.ndarray, *, role: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = np.isfinite(arr)
    if not np.any(finite):
        raise ValueError(f"{role} contains no finite values.")
    if np.all(finite):
        return arr
    idx = np.arange(arr.size, dtype=float)
    return np.interp(idx, idx[finite], arr[finite])


def _safe_savgol_1d(values: np.ndarray, *, window_length: int, polyorder: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    n = int(arr.size)
    if n < 3:
        return arr.copy()
    window = int(window_length)
    if window % 2 == 0:
        window += 1
    max_valid = n if n % 2 == 1 else (n - 1)
    window = min(window, max_valid)
    min_valid = int(polyorder) + 2
    if min_valid % 2 == 0:
        min_valid += 1
    if window < min_valid:
        return arr.copy()
    return savgol_filter(arr, window_length=window, polyorder=int(polyorder), mode="interp")


def _prepare_time_and_flow(time_raw: np.ndarray, flow_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = _fill_nonfinite_1d(time_raw, role="time")
    u = _fill_nonfinite_1d(flow_raw, role="flow speed")
    if t.size != u.size:
        n = int(min(t.size, u.size))
        t = t[:n]
        u = u[:n]
    if t.size < 3:
        raise ValueError("Time series is too short (<3 samples).")

    order = np.argsort(t)
    t = t[order]
    u = u[order]

    dt = np.diff(t)
    keep = np.concatenate(([True], dt > 0.0))
    t = t[keep]
    u = u[keep]
    if t.size < 3:
        raise ValueError("Time series has too few strictly increasing samples.")

    if TRIM_START_SECONDS > 0.0:
        t0 = float(t[0] + TRIM_START_SECONDS)
        mask = t >= t0
        t = t[mask]
        u = u[mask]
    if TRIM_END_SECONDS > 0.0:
        t1 = float(t[-1] - TRIM_END_SECONDS)
        mask = t <= t1
        t = t[mask]
        u = u[mask]
    if t.size < 3:
        raise ValueError("Too few samples remain after trim.")

    stride = int(DOWNSAMPLE_STRIDE)
    if stride > 1:
        t = t[::stride]
        u = u[::stride]
    if t.size < 3:
        raise ValueError("Too few samples remain after downsampling.")

    if USE_SAVGOL_ON_FLOW:
        u = _safe_savgol_1d(u, window_length=int(FLOW_SAVGOL_WINDOW), polyorder=int(FLOW_SAVGOL_POLYORDER))
    u = np.maximum(np.asarray(u, dtype=float), float(MIN_FLOW_SPEED))
    return np.asarray(t, dtype=float), u


def _simulate_td_with_u_series(time: np.ndarray, u_series: np.ndarray) -> dict[str, np.ndarray]:
    t = np.asarray(time, dtype=float).reshape(-1)
    u = np.asarray(u_series, dtype=float).reshape(-1)
    if t.size != u.size:
        raise ValueError("time and u_series must have same length.")
    if t.size < 3:
        raise ValueError("Need at least 3 samples to simulate TD series.")

    n = int(t.size)
    y = np.zeros(n, dtype=float)
    dy = np.zeros(n, dtype=float)
    ddy = np.zeros(n, dtype=float)
    fy = np.zeros(n, dtype=float)
    fcv = np.zeros(n, dtype=float)
    fdy = np.zeros(n, dtype=float)
    fca = np.zeros(n, dtype=float)
    phi_vy = np.zeros(n, dtype=float)
    sig_dy_loc = np.zeros(n, dtype=float)
    sig_ddy_loc = np.zeros(n, dtype=float)

    u0 = float(max(u[0], MIN_FLOW_SPEED))
    amp = float(INIT_A_FACTOR) * float(D)
    omega_osc = 2.0 * np.pi * float(INIT_FHAT) * u0 / float(D)
    phase = float(INIT_PHASE_RAD)
    y[0] = amp * np.sin(phase)
    dy[0] = omega_osc * amp * np.cos(phase)
    ddy[0] = -(omega_osc**2) * amp * np.sin(phase)

    def acceleration(y_val: float, dy_val: float, force_val: float) -> float:
        return (1.0 / float(M)) * (-float(C) * dy_val - float(K) * y_val + force_val)

    def rk4_step(y_val: float, dy_val: float, force_val: float, dt_val: float) -> tuple[float, float]:
        def acc_local(y_state: float, dy_state: float) -> float:
            return acceleration(y_state, dy_state, force_val)

        k1_y = dy_val
        k1_v = acc_local(y_val, dy_val)

        y_mid = y_val + 0.5 * dt_val * k1_y
        v_mid = dy_val + 0.5 * dt_val * k1_v
        k2_y = v_mid
        k2_v = acc_local(y_mid, v_mid)

        y_mid = y_val + 0.5 * dt_val * k2_y
        v_mid = dy_val + 0.5 * dt_val * k2_v
        k3_y = v_mid
        k3_v = acc_local(y_mid, v_mid)

        y_end = y_val + dt_val * k3_y
        v_end = dy_val + dt_val * k3_v
        k4_y = v_end
        k4_v = acc_local(y_end, v_end)

        y_next = y_val + (dt_val / 6.0) * (k1_y + 2.0 * k2_y + 2.0 * k3_y + k4_y)
        v_next = dy_val + (dt_val / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
        return y_next, v_next

    integrator = str(INTEGRATOR).strip().lower()
    for i in range(n - 1):
        dt_i = float(t[i + 1] - t[i])
        if not np.isfinite(dt_i) or dt_i <= 0.0:
            raise ValueError(f"Non-positive dt detected at index {i}: {dt_i}")

        u_i = float(max(u[i], MIN_FLOW_SPEED))
        (
            fy[i + 1],
            phi_vy[i + 1],
            sig_dy_loc[i + 1],
            sig_ddy_loc[i + 1],
            fca[i + 1],
            fcv[i + 1],
            fdy[i + 1],
        ) = vforce_CF(
            float(Cv),
            float(Cd),
            float(Ca),
            float(fhat0),
            float(fhat_min),
            float(fhat_max),
            dt_i,
            int(n_memory),
            float(rho),
            u_i,
            float(D),
            float(dy[i]),
            float(ddy[i]),
            float(phi_vy[i]),
            float(sig_dy_loc[i]),
            float(sig_ddy_loc[i]),
        )

        if integrator == "rk4":
            y_next, dy_next = rk4_step(float(y[i]), float(dy[i]), float(fy[i + 1]), dt_i)
        else:
            y_next = float(y[i]) + dt_i * float(dy[i])
            dy_next = float(dy[i]) + dt_i * float(ddy[i])

        y[i + 1] = y_next
        dy[i + 1] = dy_next
        ddy[i + 1] = acceleration(y_next, dy_next, float(fy[i + 1]))

    h = 0.5 * float(K) * y**2 + 0.5 * (float(M) + float(D) ** 2 * np.pi * float(rho) * float(Ca) / 4.0) * dy**2
    f_total = np.asarray(fcv + fdy + fca, dtype=float)

    m_eff = float(M) + float(D) ** 2 * np.pi * float(rho) * float(Ca) / 4.0
    ur_inst = 2.0 * np.pi * u / float(D) * np.sqrt(m_eff / float(K))
    ur_mode = str(UR_EXPORT_MODE).strip().lower()
    if ur_mode == "mean":
        ur_export = np.full(ur_inst.shape, float(np.mean(ur_inst)), dtype=float)
    else:
        ur_export = ur_inst.copy()

    return {
        "time": t,
        "y": y,
        "dy": dy,
        "F_total": f_total,
        "H": h,
        "Fy": fy,
        "Fcv": fcv,
        "Fdy": fdy,
        "Fca": fca,
        "U_inst": u,
        "U_r_inst": ur_inst,
        "U_r": ur_export,
    }


def _build_payload(sim: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    # Keep keys compatible with current loaders and plotting scripts.
    return {
        "time": np.asarray(sim["time"], dtype=float),
        "y": np.asarray(sim["y"], dtype=float),
        "dy": np.asarray(sim["dy"], dtype=float),
        "F_total": np.asarray(sim["F_total"], dtype=float),
        "H": np.asarray(sim["H"], dtype=float),
        "U_r": np.asarray(sim["U_r"], dtype=float),
        "U_r_inst": np.asarray(sim["U_r_inst"], dtype=float),
        "U_inst": np.asarray(sim["U_inst"], dtype=float),
        "Fy": np.asarray(sim["Fy"], dtype=float),
        "Fcv": np.asarray(sim["Fcv"], dtype=float),
        "Fdy": np.asarray(sim["Fdy"], dtype=float),
        "Fca": np.asarray(sim["Fca"], dtype=float),
        # Legacy compatibility aliases
        "a": np.asarray(sim["time"], dtype=float),
        "b": np.asarray(sim["y"], dtype=float),
        "c": np.asarray(sim["F_total"], dtype=float),
        "d": np.asarray(sim["H"], dtype=float),
        "e": np.asarray(sim["dy"], dtype=float),
    }


def _slice_sim(sim: dict[str, np.ndarray], start: int, end: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, value in sim.items():
        out[key] = np.asarray(value)[start:end]
    return out


def _chunk_bounds(time: np.ndarray, *, chunk_seconds: float) -> list[tuple[int, int]]:
    t = np.asarray(time, dtype=float).reshape(-1)
    if t.size < 2:
        return []
    bounds: list[tuple[int, int]] = []
    start = 0
    while start < t.size:
        target_end_time = t[start] + float(chunk_seconds)
        end = int(np.searchsorted(t, target_end_time, side="left"))
        if end <= start:
            end = start + 1
        if (end - start) >= int(MIN_CHUNK_SAMPLES):
            bounds.append((start, end))
        start = end
    return bounds


def _assign_splits_per_source(n_chunks: int, rng: np.random.Generator) -> list[str]:
    if n_chunks <= 0:
        return []
    idx = rng.permutation(n_chunks)
    n_test = int(round(float(TEST_FRACTION) * n_chunks))
    n_val = int(round(float(VAL_FRACTION) * n_chunks))
    n_test = min(n_test, n_chunks)
    n_val = min(n_val, max(0, n_chunks - n_test))

    # Ensure at least one train chunk if possible.
    if (n_test + n_val) >= n_chunks and n_chunks > 1:
        if n_test > 0:
            n_test -= 1
        elif n_val > 0:
            n_val -= 1

    labels = np.full(n_chunks, "train", dtype=object)
    if n_test > 0:
        labels[idx[:n_test]] = "test"
    if n_val > 0:
        labels[idx[n_test : n_test + n_val]] = "val"
    return [str(v) for v in labels.tolist()]


def main() -> None:
    _validate_config()
    mat_files = sorted(INPUT_MAT_DIR.glob(INPUT_GLOB))
    if not mat_files:
        raise FileNotFoundError(f"No files matched '{INPUT_GLOB}' in {INPUT_MAT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    layout = str(EXPORT_LAYOUT).strip().lower()
    if layout == "split":
        for split_name in ("train", "val", "test"):
            (OUTPUT_DIR / split_name).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(SPLIT_SEED))
    records: list[dict[str, object]] = []

    print(f"Input MAT dir: {INPUT_MAT_DIR.resolve()}")
    print(f"Output dir: {OUTPUT_DIR.resolve()}")
    print(f"Files matched: {len(mat_files)}")
    print(f"Export layout: {layout}")
    if layout == "split":
        print(
            "Split config: "
            f"chunk={float(SPLIT_CHUNK_SECONDS):g}s, "
            f"train/val/test={float(TRAIN_FRACTION):.3f}/{float(VAL_FRACTION):.3f}/{float(TEST_FRACTION):.3f}, "
            f"seed={int(SPLIT_SEED)}"
        )

    for mat_path in mat_files:
        data, channel_names = _load_data_matrix(mat_path, DATA_VARIABLE)
        time_raw = _select_column(
            data,
            channel_names,
            aliases=list(TIME_CHANNEL_ALIASES),
            fallback_idx=int(TIME_FALLBACK_COL),
            role="time",
        )
        flow_raw = _select_column(
            data,
            channel_names,
            aliases=list(FLOW_CHANNEL_ALIASES),
            fallback_idx=int(FLOW_FALLBACK_COL),
            role="flow speed",
        )
        time, flow = _prepare_time_and_flow(time_raw, flow_raw)
        sim = _simulate_td_with_u_series(time, flow)

        if layout == "flat":
            out_path = OUTPUT_DIR / f"{mat_path.stem}_td_from_mat_u.npz"
            if out_path.exists() and not OVERWRITE:
                print(f"Skipping existing file: {out_path}")
                continue
            np.savez(out_path, **_build_payload(sim))
            print(f"Wrote {out_path}")
            records.append(
                {
                    "source_file": mat_path.name,
                    "output_file": out_path.name,
                    "split": "flat",
                    "samples": int(np.asarray(sim["time"]).size),
                    "u_mean": float(np.mean(sim["U_inst"])),
                    "u_std": float(np.std(sim["U_inst"])),
                    "ur_mode": str(UR_EXPORT_MODE),
                    "ur_mean": float(np.mean(sim["U_r"])),
                }
            )
            continue

        bounds = _chunk_bounds(np.asarray(sim["time"]), chunk_seconds=float(SPLIT_CHUNK_SECONDS))
        if not bounds:
            warnings.warn(f"{mat_path.name}: no chunks generated; skipping.")
            continue
        split_labels = _assign_splits_per_source(len(bounds), rng)
        for chunk_idx, (start, end) in enumerate(bounds):
            sim_chunk = _slice_sim(sim, start, end)
            split_name = split_labels[chunk_idx]
            out_path = OUTPUT_DIR / split_name / f"{mat_path.stem}_td_seg{chunk_idx:03d}.npz"
            if out_path.exists() and not OVERWRITE:
                print(f"Skipping existing file: {out_path}")
                continue
            np.savez(out_path, **_build_payload(sim_chunk))
            print(f"Wrote {out_path}")
            t_chunk = np.asarray(sim_chunk["time"], dtype=float)
            records.append(
                {
                    "source_file": mat_path.name,
                    "output_file": str(out_path.relative_to(OUTPUT_DIR)),
                    "split": split_name,
                    "chunk_index": int(chunk_idx),
                    "samples": int(t_chunk.size),
                    "chunk_start_s": float(t_chunk[0]),
                    "chunk_end_s": float(t_chunk[-1]),
                    "chunk_duration_s": float(t_chunk[-1] - t_chunk[0]),
                    "u_mean": float(np.mean(sim_chunk["U_inst"])),
                    "u_std": float(np.std(sim_chunk["U_inst"])),
                    "ur_mode": str(UR_EXPORT_MODE),
                    "ur_mean": float(np.mean(sim_chunk["U_r"])),
                }
            )

    metadata_path = OUTPUT_DIR / "metadata.json"
    metadata_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Wrote metadata: {metadata_path}")


if __name__ == "__main__":
    main()
