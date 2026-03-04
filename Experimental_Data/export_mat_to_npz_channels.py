from __future__ import annotations

import json
from pathlib import Path
import warnings

import numpy as np
from scipy.io import loadmat
from scipy.signal import savgol_filter


# Paths / selection
INPUT_DIR = Path("Experimental_Data/CrossFlow/RawData")
OUTPUT_DIR = Path("Experimental_Data/npz_exports")
INPUT_GLOB = "*.mat"  # Use "*.mat" for measurement files.
DATA_VARIABLE = "data"  # Set to None to auto-detect first 2D numeric dataset.
OVERWRITE = True

# Export layout:
# - "flat": one output file per source MAT file in OUTPUT_DIR.
# - "split": split each MAT file into fixed-length chunks, then write to train/val/test subfolders.
EXPORT_LAYOUT = "split"  # one of: flat, split
SPLIT_CHUNK_SECONDS = 60.0
TRAIN_FRACTION = 0.7
VAL_FRACTION = 0.2
TEST_FRACTION = 0.1
SPLIT_SEED = 0

# Physical / setup constants
D = 0.1  # Cylinder diameter (m)
L = 1.0  # Cylinder length (m)
RHO = 1025.0  # Water density (kg/m^3)
FN = 1.119  # Natural frequency (Hz)
LB = 2.37  # Length between spring location and top end (m)
LA = 4.21 + 0.5  # Length between cylinder center and top end (m)

# CF force channel choice: "fy1", "fy2", "sum", "diff"
CF_FORCE_MODE = "fy1"
ZERO_MEAN_CF_CHANNELS = True

# Exported reduced-velocity channel mode:
# - "mean": constant U_r channel based on mean flow speed.
# - "filtered_instantaneous": Savitzky-Golay filtered instantaneous U_r channel.
UR_CHANNEL_MODE = "mean"  # one of: mean, filtered_instantaneous
UR_FILTER_WINDOW_LENGTH = 31
UR_FILTER_POLYORDER = 3

# Export derivative channel for vpinn.velocity_source=file.
EXPORT_DY_CHANNEL = True


def _validate_split_config() -> None:
    layout = str(EXPORT_LAYOUT).strip().lower()
    if layout not in {"flat", "split"}:
        raise ValueError("EXPORT_LAYOUT must be one of: flat, split")
    if layout == "split":
        if not np.isfinite(float(SPLIT_CHUNK_SECONDS)) or float(SPLIT_CHUNK_SECONDS) <= 0.0:
            raise ValueError("SPLIT_CHUNK_SECONDS must be positive and finite.")
        fracs = np.array([TRAIN_FRACTION, VAL_FRACTION, TEST_FRACTION], dtype=float)
        if not np.all(np.isfinite(fracs)) or np.any(fracs < 0.0):
            raise ValueError("TRAIN_FRACTION/VAL_FRACTION/TEST_FRACTION must be finite and non-negative.")
        if float(np.sum(fracs)) <= 0.0:
            raise ValueError("At least one split fraction must be positive.")


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


def _build_cf_force(fy1: np.ndarray, fy2: np.ndarray, mode: str) -> np.ndarray:
    key = str(mode).strip().lower()
    if key == "fy1":
        return fy1
    if key == "fy2":
        return fy2
    if key == "sum":
        return fy1 + fy2
    if key == "diff":
        return fy1 - fy2
    raise ValueError("CF_FORCE_MODE must be one of: fy1, fy2, sum, diff")


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


def _fill_nonfinite_1d(values: np.ndarray, *, role: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = np.isfinite(arr)
    if not np.any(finite):
        raise ValueError(f"{role} contains no finite samples.")
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
    max_valid = n if (n % 2 == 1) else (n - 1)
    window = min(window, max_valid)
    min_valid = int(polyorder) + 2
    if min_valid % 2 == 0:
        min_valid += 1
    if window < min_valid:
        return arr.copy()
    return savgol_filter(arr, window_length=window, polyorder=int(polyorder), mode="interp")


def _build_reduced_velocity_channel(flow_speed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flow = _fill_nonfinite_1d(flow_speed, role="flow speed")
    ur_inst = flow / (FN * D)
    mode = str(UR_CHANNEL_MODE).strip().lower()
    if mode == "mean":
        ur_mean = float(np.mean(ur_inst))
        ur_export = np.full(ur_inst.shape, ur_mean, dtype=float)
    elif mode == "filtered_instantaneous":
        ur_export = _safe_savgol_1d(
            ur_inst,
            window_length=int(UR_FILTER_WINDOW_LENGTH),
            polyorder=int(UR_FILTER_POLYORDER),
        )
    else:
        raise ValueError("UR_CHANNEL_MODE must be one of: mean, filtered_instantaneous")
    return ur_export, ur_inst


def _extract_channels(data: np.ndarray, channel_names: list[str] | None) -> dict[str, np.ndarray | float]:
    if data.shape[1] < 25:
        raise ValueError(f"Expected at least 25 columns, got shape {data.shape}")

    # Prefer channel names for robust extraction across files with different layouts.
    time = _fill_nonfinite_1d(_select_column(data, channel_names, ["Time"], 0, role="time"), role="time")
    flow = _select_column(data, channel_names, ["Water_Speed"], 19, role="flow speed")
    y = _fill_nonfinite_1d(
        _select_column(
            data,
            channel_names,
            ["xpos1"],  # Keep convention used elsewhere in this repo.
            23,
            role="CF displacement (xpos1)",
        ),
        role="CF displacement",
    )
    fx_chain = _select_column(data, channel_names, ["9131_FORCE_2"], 7, role="Fx_chain")
    fx_spr = _select_column(data, channel_names, ["9132_FORCE_3"], 8, role="Fx_spring")
    fy_spr1 = _select_column(data, channel_names, ["9130_FORCE_1"], 6, role="Fy_spring1")
    fy_spr2 = _select_column(data, channel_names, ["9133_FORCE_4"], 9, role="Fy_spring2")

    if ZERO_MEAN_CF_CHANNELS:
        fy_spr1 = fy_spr1 - np.mean(fy_spr1)
        fy_spr2 = fy_spr2 - np.mean(fy_spr2)

    f_drag = (fx_chain - fx_spr) * LB / LA
    fy1 = fy_spr1 * LB / LA
    fy2 = fy_spr2 * LB / LA
    cf_force = _build_cf_force(fy1, fy2, CF_FORCE_MODE)

    flow_filled = _fill_nonfinite_1d(flow, role="flow speed")
    ur_channel, ur_inst = _build_reduced_velocity_channel(flow_filled)
    u_mean = float(np.mean(flow_filled))

    q_ref = 0.5 * RHO * L * D * (u_mean**2)
    if q_ref <= 0.0:
        raise ValueError("Invalid dynamic pressure reference (non-positive).")
    drag_coeff = f_drag / q_ref

    dt = float(time[1] - time[0]) if time.size > 1 else 0.0
    if dt <= 0.0:
        raise ValueError("Time channel must have at least two increasing samples.")
    dy = np.gradient(y, dt)

    return {
        "time": time,
        "y": y,
        "dy": dy,
        "cf_force": cf_force,
        "drag_coeff": drag_coeff,
        "U_r": ur_channel,
        "U_r_inst": ur_inst,
        "u_mean": u_mean,
        "u_std": float(np.std(flow_filled)),
        "ur_scalar": float(np.mean(ur_channel)),
        "ur_inst_mean": float(np.mean(ur_inst)),
        "ur_inst_std": float(np.std(ur_inst)),
    }


def _slice_channel_dict(
    channels: dict[str, np.ndarray | float],
    start: int,
    end: int,
) -> dict[str, np.ndarray | float]:
    out: dict[str, np.ndarray | float] = {}
    for key, value in channels.items():
        if isinstance(value, np.ndarray):
            out[key] = np.asarray(value)[start:end]
        else:
            out[key] = value
    return out


def _chunk_bounds(time: np.ndarray, *, chunk_seconds: float) -> list[tuple[int, int]]:
    t = np.asarray(time, dtype=float).reshape(-1)
    if t.size < 2:
        return []
    dt = float(t[1] - t[0])
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("Time channel must be increasing and have positive dt.")
    step = max(2, int(round(float(chunk_seconds) / dt)))
    bounds: list[tuple[int, int]] = []
    start = 0
    n = int(t.size)
    while start < n:
        end = min(start + step, n)
        if (end - start) >= 2:
            bounds.append((start, end))
        start = end
    return bounds


def _assign_splits_per_source(
    n_chunks: int,
    rng: np.random.Generator,
) -> list[str]:
    if n_chunks <= 0:
        return []
    idx = np.arange(n_chunks, dtype=int)
    rng.shuffle(idx)

    n_val = int(np.floor(float(VAL_FRACTION) * n_chunks))
    n_test = int(np.floor(float(TEST_FRACTION) * n_chunks))

    if n_chunks >= 2:
        n_val = max(1, n_val)
    if n_chunks >= 5 and float(TEST_FRACTION) > 0.0:
        n_test = max(1, n_test)

    if n_val >= n_chunks:
        n_val = n_chunks - 1
    if n_val + n_test >= n_chunks:
        n_test = max(0, n_chunks - n_val - 1)

    labels = np.full(n_chunks, "train", dtype=object)
    if n_test > 0:
        labels[idx[:n_test]] = "test"
    if n_val > 0:
        labels[idx[n_test : n_test + n_val]] = "val"
    return [str(v) for v in labels.tolist()]


def _build_payload(ch: dict[str, np.ndarray | float]) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {
        # vpinn-compatible keys
        "time": np.asarray(ch["time"], dtype=float),
        "y": np.asarray(ch["y"], dtype=float),
        "F_total": np.asarray(ch["cf_force"], dtype=float),
        "U_r": np.asarray(ch["U_r"], dtype=float),
        # Extra diagnostics / compatibility keys
        "cf_force": np.asarray(ch["cf_force"], dtype=float),
        "drag_coeff": np.asarray(ch["drag_coeff"], dtype=float),
        "U_r_inst": np.asarray(ch["U_r_inst"], dtype=float),
        "a": np.asarray(ch["time"], dtype=float),
        "b": np.asarray(ch["y"], dtype=float),
        "c": np.asarray(ch["cf_force"], dtype=float),
        "e": np.asarray(ch["dy"], dtype=float),
    }
    if EXPORT_DY_CHANNEL:
        payload["dy"] = np.asarray(ch["dy"], dtype=float)
    return payload


def main() -> None:
    _validate_split_config()
    mat_files = sorted(INPUT_DIR.glob(INPUT_GLOB))
    if not mat_files:
        raise FileNotFoundError(f"No files matched '{INPUT_GLOB}' in {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    rng = np.random.default_rng(int(SPLIT_SEED))
    layout = str(EXPORT_LAYOUT).strip().lower()
    if layout == "split":
        for split_name in ("train", "val", "test"):
            (OUTPUT_DIR / split_name).mkdir(parents=True, exist_ok=True)

    for mat_path in mat_files:
        data, channel_names = _load_data_matrix(mat_path, DATA_VARIABLE)
        ch = _extract_channels(data, channel_names)
        if layout == "flat":
            out_path = OUTPUT_DIR / f"{mat_path.stem}.npz"
            if out_path.exists() and not OVERWRITE:
                print(f"Skipping existing file: {out_path}")
                continue
            np.savez(out_path, **_build_payload(ch))
            print(f"Wrote {out_path}")
            records.append(
                {
                    "source_file": mat_path.name,
                    "output_file": out_path.name,
                    "split": "flat",
                    "chunk_index": 0,
                    "samples": int(np.asarray(ch["time"]).size),
                    "cf_force_mode": CF_FORCE_MODE,
                    "ur_channel_mode": str(UR_CHANNEL_MODE),
                    "u_mean": float(ch["u_mean"]),
                    "u_std": float(ch["u_std"]),
                    "ur_scalar": float(ch["ur_scalar"]),
                    "ur_inst_mean": float(ch["ur_inst_mean"]),
                    "ur_inst_std": float(ch["ur_inst_std"]),
                }
            )
            continue

        bounds = _chunk_bounds(np.asarray(ch["time"]), chunk_seconds=float(SPLIT_CHUNK_SECONDS))
        if not bounds:
            warnings.warn(f"{mat_path.name}: no valid chunks produced; skipping.")
            continue
        split_labels = _assign_splits_per_source(len(bounds), rng)
        for chunk_idx, (start, end) in enumerate(bounds):
            chunk = _slice_channel_dict(ch, start, end)
            split_name = split_labels[chunk_idx]
            out_dir = OUTPUT_DIR / split_name
            out_path = out_dir / f"{mat_path.stem}_seg{chunk_idx:03d}.npz"
            if out_path.exists() and not OVERWRITE:
                print(f"Skipping existing file: {out_path}")
                continue
            np.savez(out_path, **_build_payload(chunk))
            print(f"Wrote {out_path}")
            t_chunk = np.asarray(chunk["time"], dtype=float)
            records.append(
                {
                    "source_file": mat_path.name,
                    "output_file": str(out_path.relative_to(OUTPUT_DIR)),
                    "split": split_name,
                    "chunk_index": int(chunk_idx),
                    "chunk_start_s": float(t_chunk[0]),
                    "chunk_end_s": float(t_chunk[-1]),
                    "chunk_duration_s": float(t_chunk[-1] - t_chunk[0]),
                    "samples": int(t_chunk.size),
                    "cf_force_mode": CF_FORCE_MODE,
                    "ur_channel_mode": str(UR_CHANNEL_MODE),
                    "u_mean": float(chunk["u_mean"]),
                    "u_std": float(chunk["u_std"]),
                    "ur_scalar": float(chunk["ur_scalar"]),
                    "ur_inst_mean": float(chunk["ur_inst_mean"]),
                    "ur_inst_std": float(chunk["ur_inst_std"]),
                }
            )

    metadata_path = OUTPUT_DIR / "metadata.json"
    metadata_path.write_text(json.dumps(records, indent=2))
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
