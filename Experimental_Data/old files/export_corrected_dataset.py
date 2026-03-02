from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import sys
import warnings

import numpy as np
from scipy.io import savemat

try:
    import Experimental_Data.analyze_experimental_data as analysis
    import Experimental_Data.phase_analysis as phase
except ModuleNotFoundError as exc:
    if getattr(exc, "name", "") != "Experimental_Data":
        raise
    current_dir = Path(__file__).resolve().parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    import analyze_experimental_data as analysis
    import phase_analysis as phase


# Dataset export knobs
OUTPUT_DIR = Path("Experimental_Data/corrected_dataset")
TRAIN_FRACTION = 0.8
VAL_FRACTION = 0.2
SPLIT_SEED = 1234
OVERWRITE = True
DOWNSAMPLE_STRIDE = 1

# If True, delete the existing OUTPUT_DIR contents before exporting anything.
CLEAN_OUTPUT_DIR_BEFORE_EXPORT = True

# Chunking knobs for NPZ export.
# If CHUNK_LENGTH_SECONDS <= 0, each source series is exported as one sample.
CHUNK_LENGTH_SECONDS = 60.0
# If <= 0, uses CHUNK_LENGTH_SECONDS (non-overlapping chunks).
CHUNK_STRIDE_SECONDS = 0.0
# If True, include one last tail chunk ending at the last sample when possible.
INCLUDE_TAIL_CHUNK = True
MIN_CHUNK_SAMPLES = 256

# Split stratification knobs.
# - "none": random split
# - "ur_bin": stratify by chunk mean reduced velocity bins
# - "source_file": stratify by source file label
SPLIT_STRATIFY_MODE = "ur_bin"
SPLIT_STRATIFY_UR_BINS = 5

# Chunk reduced-velocity assignment:
# - True: each chunk uses one global/test mean U_r
# - False: each chunk uses its own local/chunk mean U_r
CHUNK_USE_GLOBAL_MEAN_UR = True

# Kinematics are computed using the same derivative routine as phase-analysis
# plotting (plot_extracted_channels._compute_derivatives).

# Exporting only the same active phase-analysis mask window can be useful.
USE_PHASE_MASK_WINDOW = True
# For corrected-dataset export, force a common linear lag model with extrapolation
# so correction is applied over full record duration (including times beyond the
# fitting window, e.g. > 980 s).
FORCE_COMMON_LINEAR_LAG_EXTRAPOLATION = True
# Optional additional end-trim used only for NPZ export path
# (downsampling/chunking + saved NPZ chunks). Corrected MAT export remains untrimmed.
NPZ_TRIM_START_SECONDS = float(getattr(analysis, "TRIM_START_SECONDS", 0.0))
NPZ_TRIM_END_SECONDS = float(getattr(analysis, "TRIM_END_SECONDS", 0.0))

# Exclude files by test number parsed from filename.
# These are additional exclusions applied after phase-analysis selection/exclusion.
# By default, mirror phase-analysis exclusion (idempotent if identical).
EXCLUDE_TEST_NUMBERS: list[int] = list(getattr(phase, "EXCLUDE_TEST_NUMBERS", []))

# Calculated-force knobs:
#   F_calc = GLOBAL_SIGN * (FY1_MULT * Fy1 + FY2_MULT * Fy2 + F_inertia)
# where:
#   F_inertia = INERTIA_SIGN * m_inertia * y_ddot_corrected
#   m_inertia = (M if INCLUDE_STRUCTURAL_MASS else 0)
#             + (C_a * pi/4*rho*D^2*L if INCLUDE_ADDED_MASS else 0)
FORCE_FY1_MULT = -1.0
FORCE_FY2_MULT = +1.0
FORCE_GLOBAL_SIGN = +1.0
FORCE_INCLUDE_INERTIA = True
FORCE_INERTIA_SIGN = +1.0
FORCE_INCLUDE_STRUCTURAL_MASS = True
FORCE_INCLUDE_ADDED_MASS = True
FORCE_ADDED_MASS_COEFF = float(analysis.ADDED_MASS_COEFF)

# Optional corrected MAT export:
# For each source MAT file, write a corrected MAT into:
#   - if source is in a RawData folder:
#       <source_parent_parent>/<CORRECTED_MAT_SUBDIR_NAME>/<source_stem>_corrected.mat
#   - otherwise:
#       <source_parent>/<CORRECTED_MAT_SUBDIR_NAME>/<source_stem>_corrected.mat
EXPORT_CORRECTED_MAT = True
CORRECTED_MAT_SUBDIR_NAME = "CorrectedData"

# Raw-layout corrected MAT knobs:
# Keep the same table shape/columns as raw input and inject corrected displacement
# into selected displacement column(s).
CORRECTED_DISPLACEMENT_ALIASES = ["xpos1"]
CORRECTED_DISPLACEMENT_FALLBACK_INDICES = [23]
RENAME_CORRECTED_DISPLACEMENT_CHANNELS = True


def _fill_nonfinite_1d(values: np.ndarray, *, role: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = np.isfinite(arr)
    if not np.any(finite):
        raise ValueError(f"{role}: signal has no finite samples.")
    if np.all(finite):
        return arr
    idx = np.arange(arr.size, dtype=float)
    arr[~finite] = np.interp(idx[~finite], idx[finite], arr[finite])
    return arr


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
    source_desc = "phase.MAT_FILES (fallback)"
    if hasattr(phase, "_resolve_phase_input_files"):
        phase_files, phase_source_desc = phase._resolve_phase_input_files()
        mat_files = [Path(p) for p in phase_files]
        source_desc = str(phase_source_desc)
    else:
        mat_files = [Path(p) for p in list(phase.MAT_FILES)]

    # Always apply the same exclusion step as phase_analysis first.
    if hasattr(phase, "_filter_excluded_tests"):
        mat_files = phase._filter_excluded_tests(mat_files)

    # Optional additional export-specific exclusions.
    mat_files = _filter_excluded_tests(mat_files)

    if not mat_files:
        raise ValueError(
            "No MAT files selected (check phase file selection settings and EXCLUDE_TEST_NUMBERS)."
        )
    missing = [p for p in mat_files if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing MAT file(s): {missing}")
    print(f"Export source files: {source_desc}")
    print(f"Export selected {len(mat_files)} file(s) after exclusions.")
    return mat_files


def _select_channel(result: dict[str, object], channel_name: str) -> np.ndarray:
    return np.asarray(phase._select_channel(result, channel_name), dtype=float)


def _flow_and_ur_for_result(
    *,
    mat_file: Path,
    t_result: np.ndarray,
    mask_result: np.ndarray,
    apply_mask_window: bool,
) -> tuple[np.ndarray, float, float, float]:
    data, channel_names = analysis._load_data_matrix(mat_file, phase.DATA_VARIABLE)
    time = analysis._select_column(data, channel_names, ["Time"], 0, role="time")
    flow = analysis._select_column(data, channel_names, ["Water_Speed"], 19, role="flow speed")

    time_arr = _fill_nonfinite_1d(np.asarray(time, dtype=float), role=f"{mat_file.name}: time")
    flow_arr = _fill_nonfinite_1d(np.asarray(flow, dtype=float), role=f"{mat_file.name}: flow speed")
    n = min(time_arr.size, flow_arr.size)
    flow_arr = flow_arr[:n]
    t_arr = time_arr[:n]
    if bool(phase.USE_RELATIVE_TIME):
        t_arr = t_arr - float(t_arr[0])

    mask = np.asarray(mask_result, dtype=bool).reshape(-1)
    n2 = min(mask.size, flow_arr.size, np.asarray(t_result, dtype=float).size)
    mask = mask[:n2]
    flow_arr = flow_arr[:n2]

    if bool(apply_mask_window):
        flow_used = flow_arr[mask]
    else:
        flow_used = flow_arr
    if flow_used.size < 2:
        raise ValueError(f"{mat_file.name}: not enough flow samples after masking.")

    ur_series = flow_used / (float(analysis.FN) * float(analysis.D))
    ur_mean = float(np.mean(ur_series))
    ur_std = float(np.std(ur_series))
    u_mean = float(np.mean(flow_used))
    return flow_used, ur_mean, ur_std, u_mean


def _split_labels(n_items: int, *, train_fraction: float, val_fraction: float, seed: int) -> list[str]:
    if n_items <= 0:
        return []
    fracs = np.asarray([train_fraction, val_fraction], dtype=float)
    if not np.all(np.isfinite(fracs)) or np.any(fracs < 0.0):
        raise ValueError("TRAIN_FRACTION and VAL_FRACTION must be finite and non-negative.")
    if float(np.sum(fracs)) <= 0.0:
        raise ValueError("At least one split fraction must be positive.")
    fracs = fracs / float(np.sum(fracs))

    if n_items == 1:
        return ["train"]

    n_val = int(round(float(fracs[1]) * n_items))
    n_val = min(max(1, n_val), n_items - 1)
    idx = np.arange(n_items, dtype=int)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(idx)

    labels = np.full(n_items, "train", dtype=object)
    labels[idx[:n_val]] = "val"
    return [str(v) for v in labels.tolist()]


def _split_labels_from_strata(
    strata: list[object],
    *,
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> list[str]:
    n_items = len(strata)
    if n_items <= 0:
        return []

    if str(SPLIT_STRATIFY_MODE).strip().lower() == "none":
        return _split_labels(
            n_items,
            train_fraction=float(train_fraction),
            val_fraction=float(val_fraction),
            seed=int(seed),
        )

    fracs = np.asarray([train_fraction, val_fraction], dtype=float)
    if not np.all(np.isfinite(fracs)) or np.any(fracs < 0.0):
        raise ValueError("TRAIN_FRACTION and VAL_FRACTION must be finite and non-negative.")
    if float(np.sum(fracs)) <= 0.0:
        raise ValueError("At least one split fraction must be positive.")
    val_frac = float(fracs[1] / np.sum(fracs))

    labels = np.full(n_items, "train", dtype=object)
    groups: dict[object, list[int]] = {}
    for i, key in enumerate(strata):
        groups.setdefault(key, []).append(i)

    rng = np.random.default_rng(int(seed))
    for idx_list in groups.values():
        idx_arr = np.asarray(idx_list, dtype=int)
        rng.shuffle(idx_arr)
        n_group = int(idx_arr.size)
        n_val = int(round(val_frac * n_group))
        if n_group >= 2 and val_frac > 0.0:
            n_val = max(1, n_val)
            n_val = min(n_val, n_group - 1)
        else:
            n_val = 0
        labels[idx_arr[:n_val]] = "val"

    if n_items > 1:
        n_val_total = int(np.sum(labels == "val"))
        target_val = int(round(val_frac * n_items))
        target_val = min(max(1, target_val), n_items - 1)
        if n_val_total < target_val:
            train_idx = np.where(labels == "train")[0]
            rng.shuffle(train_idx)
            promote = train_idx[: (target_val - n_val_total)]
            labels[promote] = "val"
        elif n_val_total > target_val:
            val_idx = np.where(labels == "val")[0]
            rng.shuffle(val_idx)
            demote = val_idx[: (n_val_total - target_val)]
            labels[demote] = "train"

    return [str(v) for v in labels.tolist()]


def _ur_bin_strata(values: np.ndarray, *, n_bins: int) -> list[int]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return []
    finite = np.isfinite(arr)
    strata = np.full(arr.size, -1, dtype=int)
    if not np.any(finite):
        return [int(v) for v in strata.tolist()]

    vals = arr[finite]
    n_bins_use = max(1, int(n_bins))
    if vals.size < 2 or np.allclose(vals, vals[0]):
        strata[finite] = 0
        return [int(v) for v in strata.tolist()]

    quantiles = np.linspace(0.0, 1.0, n_bins_use + 1)
    edges = np.quantile(vals, quantiles)
    edges = np.unique(edges)
    if edges.size <= 2:
        strata[finite] = 0
        return [int(v) for v in strata.tolist()]

    bins = np.digitize(vals, edges[1:-1], right=False)
    strata[finite] = bins.astype(int)
    return [int(v) for v in strata.tolist()]


def _chunk_series(
    *,
    t: np.ndarray,
    y: np.ndarray,
    dy: np.ndarray,
    ddy: np.ndarray,
    f: np.ndarray,
    ur: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]]:
    t_arr = np.asarray(t, dtype=float).reshape(-1)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    dy_arr = np.asarray(dy, dtype=float).reshape(-1)
    ddy_arr = np.asarray(ddy, dtype=float).reshape(-1)
    f_arr = np.asarray(f, dtype=float).reshape(-1)
    ur_arr = np.asarray(ur, dtype=float).reshape(-1)
    n = int(min(t_arr.size, y_arr.size, dy_arr.size, ddy_arr.size, f_arr.size, ur_arr.size))
    if n < 2:
        return []
    t_arr = t_arr[:n]
    y_arr = y_arr[:n]
    dy_arr = dy_arr[:n]
    ddy_arr = ddy_arr[:n]
    f_arr = f_arr[:n]
    ur_arr = ur_arr[:n]

    chunk_len_s = float(CHUNK_LENGTH_SECONDS)
    if chunk_len_s <= 0.0:
        return [(t_arr, y_arr, dy_arr, ddy_arr, f_arr, ur_arr, 0, n)]

    dt = float(np.median(np.diff(t_arr)))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("Invalid dt for chunking.")

    chunk_samples = max(2, int(round(chunk_len_s / dt)))
    stride_s = float(CHUNK_STRIDE_SECONDS)
    if stride_s <= 0.0:
        stride_s = chunk_len_s
    stride_samples = max(1, int(round(stride_s / dt)))

    chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]] = []
    for start in range(0, max(1, n - chunk_samples + 1), stride_samples):
        end = start + chunk_samples
        if end > n:
            break
        if (end - start) < int(MIN_CHUNK_SAMPLES):
            continue
        chunks.append(
            (
                t_arr[start:end],
                y_arr[start:end],
                dy_arr[start:end],
                ddy_arr[start:end],
                f_arr[start:end],
                ur_arr[start:end],
                start,
                end,
            )
        )

    if bool(INCLUDE_TAIL_CHUNK):
        end = n
        start = max(0, end - chunk_samples)
        if (end - start) >= int(MIN_CHUNK_SAMPLES):
            if not chunks or chunks[-1][6] != start:
                chunks.append(
                    (
                        t_arr[start:end],
                        y_arr[start:end],
                        dy_arr[start:end],
                        ddy_arr[start:end],
                        f_arr[start:end],
                        ur_arr[start:end],
                        start,
                        end,
                    )
                )

    if not chunks and n >= int(MIN_CHUNK_SAMPLES):
        chunks.append((t_arr, y_arr, dy_arr, ddy_arr, f_arr, ur_arr, 0, n))
    return chunks


def _compute_corrected_kinematics(y_corr: np.ndarray, *, dt: float) -> tuple[np.ndarray, np.ndarray]:
    y_arr = np.asarray(y_corr, dtype=float).reshape(-1)
    if y_arr.size < 2:
        return np.zeros_like(y_arr), np.zeros_like(y_arr)
    if not np.isfinite(float(dt)) or float(dt) <= 0.0:
        raise ValueError("dt must be finite and positive for kinematics.")
    try:
        y_dot, y_ddot, _ = phase.extracted._compute_derivatives(y_arr, dt=float(dt))
        return np.asarray(y_dot, dtype=float), np.asarray(y_ddot, dtype=float)
    except Exception as exc:
        warnings.warn(
            f"Falling back to finite differences for corrected kinematics because phase derivative routine failed: {exc}"
        )
        y_dot = np.gradient(y_arr, float(dt))
        y_ddot = np.gradient(y_dot, float(dt))
        return np.asarray(y_dot, dtype=float), np.asarray(y_ddot, dtype=float)


def _mass_added(ca: float) -> float:
    return float(ca) * 0.25 * np.pi * float(analysis.RUO) * float(analysis.D) ** 2 * float(analysis.L)


def _inertia_mass_used() -> float:
    m_val = 0.0
    if bool(FORCE_INCLUDE_STRUCTURAL_MASS):
        m_val += float(analysis.M)
    if bool(FORCE_INCLUDE_ADDED_MASS):
        m_val += _mass_added(float(FORCE_ADDED_MASS_COEFF))
    return float(m_val)


def _build_calculated_force(
    result: dict[str, object],
    *,
    y_ddot_corr: np.ndarray,
) -> tuple[np.ndarray, float]:
    fy1 = _select_channel(result, "Fy1 (scaled LB/LA, N)")
    fy2 = _select_channel(result, "Fy2 (scaled LB/LA, N)")
    acc = np.asarray(y_ddot_corr, dtype=float).reshape(-1)
    n = min(fy1.size, fy2.size, acc.size)
    fy1 = fy1[:n]
    fy2 = fy2[:n]
    acc = acc[:n]

    f_base = float(FORCE_FY1_MULT) * fy1 + float(FORCE_FY2_MULT) * fy2
    m_inertia = _inertia_mass_used()
    if bool(FORCE_INCLUDE_INERTIA) and m_inertia != 0.0:
        f_inertia = float(FORCE_INERTIA_SIGN) * m_inertia * acc
    else:
        f_inertia = np.zeros_like(f_base)
    f_calc = float(FORCE_GLOBAL_SIGN) * (f_base + f_inertia)
    return np.asarray(f_calc, dtype=float), float(m_inertia)


def _downsample(
    t: np.ndarray,
    y: np.ndarray,
    dy: np.ndarray,
    ddy: np.ndarray,
    f: np.ndarray,
    ur: np.ndarray,
    *,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    step = max(1, int(stride))
    t2 = np.asarray(t, dtype=float)[::step]
    y2 = np.asarray(y, dtype=float)[::step]
    dy2 = np.asarray(dy, dtype=float)[::step]
    ddy2 = np.asarray(ddy, dtype=float)[::step]
    f2 = np.asarray(f, dtype=float)[::step]
    ur2 = np.asarray(ur, dtype=float)[::step]
    if t2.size < 2:
        raise ValueError("Downsampling produced too few samples.")
    return t2, y2, dy2, ddy2, f2, ur2


def _build_payload(
    *,
    time: np.ndarray,
    y_corr: np.ndarray,
    dy_corr: np.ndarray,
    ddy_corr: np.ndarray,
    f_calc: np.ndarray,
    ur_scalar: float,
) -> dict[str, np.ndarray]:
    dy = np.asarray(dy_corr, dtype=float)
    ddy = np.asarray(ddy_corr, dtype=float)
    return {
        # Synthetic-style keys used by training loaders.
        "a": np.asarray(time, dtype=float),
        "b": np.asarray(y_corr, dtype=float),
        "c": np.asarray(f_calc, dtype=float),
        "e": np.asarray(dy, dtype=float),
        "ddy": np.asarray(ddy, dtype=float),
        "U_r": np.asarray(float(ur_scalar), dtype=float),
        # Explicit aliases.
        "time": np.asarray(time, dtype=float),
        "y": np.asarray(y_corr, dtype=float),
        "F_total": np.asarray(f_calc, dtype=float),
        "dy": np.asarray(dy, dtype=float),
        "d2y": np.asarray(ddy, dtype=float),
        "calculated_force": np.asarray(f_calc, dtype=float),
    }


def _clean_output_dir(path: Path) -> None:
    target = Path(path)
    target_str = str(target).strip()
    if target_str in {"", ".", "/"}:
        raise ValueError(f"Refusing to clean unsafe output path: {target}")
    if target.exists():
        shutil.rmtree(target)


def _export_corrected_mat(
    *,
    source_mat_path: Path,
    split_name: str,
    y_corr: np.ndarray,
    dy_corr: np.ndarray,
    f_calc: np.ndarray,
    overwrite: bool,
) -> Path:
    parent = source_mat_path.parent
    if parent.name.strip().lower() == "rawdata":
        export_root = parent.parent
    else:
        export_root = parent
    out_dir = export_root / str(CORRECTED_MAT_SUBDIR_NAME)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{source_mat_path.stem}_corrected.mat"
    if out_path.exists() and not bool(overwrite):
        raise FileExistsError(f"Refusing to overwrite existing file: {out_path}")

    source_data, source_channel_names = analysis._load_data_matrix(source_mat_path, phase.DATA_VARIABLE)
    data_corrected = np.asarray(source_data, dtype=float).copy()
    n_rows, n_cols = data_corrected.shape

    y = np.asarray(y_corr, dtype=float).reshape(-1)
    dy = np.asarray(dy_corr, dtype=float).reshape(-1)
    f = np.asarray(f_calc, dtype=float).reshape(-1)
    n = min(n_rows, y.size, dy.size, f.size)
    if n < 2:
        raise ValueError(f"{source_mat_path.name}: not enough samples to export corrected MAT.")

    y = y[:n]
    dy = dy[:n]
    f = f[:n]

    time_series = analysis._select_column(
        data_corrected[:n, :],
        source_channel_names,
        ["Time", "time"],
        0,
        role="corrected-mat export time",
    )
    time_series = _fill_nonfinite_1d(np.asarray(time_series, dtype=float), role=f"{source_mat_path.name}: time")
    dt = float(np.median(np.diff(time_series)))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"{source_mat_path.name}: invalid dt inferred for corrected MAT export.")

    disp_indices: list[int] = []
    if source_channel_names is not None:
        idx_map = {analysis._norm_name(name): i for i, name in enumerate(source_channel_names)}
        for alias in CORRECTED_DISPLACEMENT_ALIASES:
            idx = idx_map.get(analysis._norm_name(alias))
            if idx is not None and 0 <= idx < n_cols and idx not in disp_indices:
                disp_indices.append(int(idx))
    for idx in CORRECTED_DISPLACEMENT_FALLBACK_INDICES:
        ii = int(idx)
        if 0 <= ii < n_cols and ii not in disp_indices:
            disp_indices.append(ii)
    if not disp_indices:
        raise ValueError(
            f"{source_mat_path.name}: could not resolve any displacement column indices for correction."
        )

    data_corrected = data_corrected[:n, :]
    for idx in disp_indices:
        data_corrected[:, idx] = y

    channel_names_out = None
    if source_channel_names is not None:
        channel_names_out = list(source_channel_names)
        if bool(RENAME_CORRECTED_DISPLACEMENT_CHANNELS):
            for idx in disp_indices:
                channel_names_out[idx] = "y_corrected"

    flow = analysis._select_column(
        data_corrected,
        source_channel_names,
        ["Water_Speed", "water_speed"],
        19,
        role="corrected-mat export flow speed",
    )
    flow = _fill_nonfinite_1d(np.asarray(flow, dtype=float), role=f"{source_mat_path.name}: flow speed")
    ur = flow / (float(analysis.FN) * float(analysis.D))

    payload = {
        "data_corrected": np.asarray(data_corrected, dtype=float),
        # Keep MATLAB-style `data` for compatibility with raw file format.
        "data": np.asarray(data_corrected, dtype=float),
        "time": time_series.reshape(-1, 1),
        "y_corrected": y.reshape(-1, 1),
        "calculated_force": f.reshape(-1, 1),
        "U_r": ur.reshape(-1, 1),
        "dy_corrected": np.asarray(dy, dtype=float).reshape(-1, 1),
        "split": np.array([str(split_name)], dtype=object),
        "source_file": np.array([str(source_mat_path.name)], dtype=object),
    }
    if channel_names_out is not None:
        payload["chan_names"] = np.array(channel_names_out, dtype=object).reshape(-1, 1)
        payload["column_names"] = np.array(channel_names_out, dtype=object)

    savemat(out_path, payload, do_compression=True)
    return out_path


def main() -> None:
    phase.extracted.DATA_VARIABLE = phase.DATA_VARIABLE
    phase.extracted.USE_RELATIVE_TIME = phase.USE_RELATIVE_TIME
    if bool(FORCE_COMMON_LINEAR_LAG_EXTRAPOLATION):
        phase.PHASE_CORRECTION_MODE = "common"
        phase.PHASE_COMMON_LAG_POLYORDER = 1
        phase.PHASE_COMMON_LAG_EXTRAPOLATE = True

    mat_files = _resolve_mat_files()
    results: list[dict[str, object]] = []
    for mat_file in mat_files:
        result = phase.extracted._process_file(mat_file)
        results.append(result)

    if bool(phase.PHASE_CORRECTION_ENABLED):
        phase_before = phase._collect_phase_series(results)
        corrected_disp_by_label = phase._build_corrected_displacement(results, phase_before)
    else:
        corrected_disp_by_label = {
            str(result["label"]): _select_channel(result, "Displacement y (m)")
            for result in results
        }

    prepared_sources: list[dict[str, object]] = []
    for result in results:
        label = str(result["label"])
        mat_path = Path(result["path"])
        t = np.asarray(result["t"], dtype=float)
        mask = np.asarray(result["mask"], dtype=bool)
        y_corr = np.asarray(corrected_disp_by_label[label], dtype=float)
        dt_result = float(result["dt"])
        y_dot_corr, y_ddot_corr = _compute_corrected_kinematics(y_corr, dt=dt_result)
        f_calc, m_inertia_used = _build_calculated_force(result, y_ddot_corr=y_ddot_corr)

        n = min(t.size, mask.size, f_calc.size, y_corr.size, y_dot_corr.size, y_ddot_corr.size)
        t = t[:n]
        mask = mask[:n]
        f_calc = f_calc[:n]
        y_corr = y_corr[:n]
        y_dot_corr = y_dot_corr[:n]
        y_ddot_corr = y_ddot_corr[:n]
        y_corr_full = y_corr.copy()
        y_dot_corr_full = y_dot_corr.copy()
        f_calc_full = f_calc.copy()
        trim_mask = analysis._time_trim_mask(
            t,
            role=f"{mat_path.name}: export time trim",
            trim_start_seconds=float(NPZ_TRIM_START_SECONDS),
            trim_end_seconds=float(NPZ_TRIM_END_SECONDS),
        )
        export_mask = np.ones_like(mask, dtype=bool)
        if bool(USE_PHASE_MASK_WINDOW):
            export_mask &= np.asarray(mask, dtype=bool)
        if bool(float(NPZ_TRIM_START_SECONDS) > 0.0 or float(NPZ_TRIM_END_SECONDS) > 0.0):
            export_mask &= np.asarray(trim_mask, dtype=bool)

        if int(np.count_nonzero(export_mask)) < 2:
            warnings.warn(
                f"{mat_path.name}: skipping source after mask/trim "
                f"(remaining samples={int(np.count_nonzero(export_mask))})."
            )
            continue

        if not np.all(export_mask):
            t = t[export_mask]
            f_calc = f_calc[export_mask]
            y_corr = y_corr[export_mask]
            y_dot_corr = y_dot_corr[export_mask]
            y_ddot_corr = y_ddot_corr[export_mask]

        flow_used, ur_mean, ur_std, u_mean = _flow_and_ur_for_result(
            mat_file=mat_path,
            t_result=np.asarray(result["t"], dtype=float),
            mask_result=np.asarray(export_mask, dtype=bool),
            apply_mask_window=True,
        )

        ur_series = flow_used / (float(analysis.FN) * float(analysis.D))
        if bool(USE_PHASE_MASK_WINDOW):
            ur_for_export = ur_series
        else:
            ur_for_export = np.full_like(t, float(np.mean(ur_series)))
        if ur_for_export.size != t.size:
            ur_for_export = np.full_like(t, float(np.mean(ur_series)))

        try:
            t, y_corr, y_dot_corr, y_ddot_corr, f_calc, ur_for_export = _downsample(
                t,
                y_corr,
                y_dot_corr,
                y_ddot_corr,
                f_calc,
                ur_for_export,
                stride=int(DOWNSAMPLE_STRIDE),
            )
        except ValueError as exc:
            warnings.warn(f"{mat_path.name}: skipping source after downsampling ({exc})")
            continue

        chunks = _chunk_series(
            t=t,
            y=y_corr,
            dy=y_dot_corr,
            ddy=y_ddot_corr,
            f=f_calc,
            ur=ur_for_export,
        )
        if not chunks:
            warnings.warn(f"{mat_path.name}: no valid chunks produced; skipping source.")
            continue

        prepared_sources.append(
            {
                "label": label,
                "mat_path": mat_path,
                "y_corr_full": np.asarray(y_corr_full, dtype=float),
                "y_dot_corr_full": np.asarray(y_dot_corr_full, dtype=float),
                "f_calc_full": np.asarray(f_calc_full, dtype=float),
                "m_inertia_used": float(m_inertia_used),
                "u_mean_source": float(u_mean),
                "ur_mean_source": float(ur_mean),
                "ur_std_source": float(ur_std),
                "chunks": chunks,
            }
        )

    chunk_records: list[dict[str, object]] = []
    for source in prepared_sources:
        chunks = source["chunks"]
        assert isinstance(chunks, list)
        for j, chunk in enumerate(chunks, start=1):
            t_chunk, y_chunk, dy_chunk, ddy_chunk, f_chunk, ur_chunk, start_idx, end_idx = chunk
            ur_chunk_mean = float(np.mean(np.asarray(ur_chunk, dtype=float)))
            ur_source_mean = float(source["ur_mean_source"])
            ur_scalar = ur_source_mean if bool(CHUNK_USE_GLOBAL_MEAN_UR) else ur_chunk_mean
            chunk_records.append(
                {
                    "source_label": str(source["label"]),
                    "source_file": str(Path(source["mat_path"]).name),
                    "source_path": Path(source["mat_path"]),
                    "chunk_index": int(j),
                    "start_idx": int(start_idx),
                    "end_idx": int(end_idx),
                    "t": np.asarray(t_chunk, dtype=float),
                    "y": np.asarray(y_chunk, dtype=float),
                    "dy": np.asarray(dy_chunk, dtype=float),
                    "ddy": np.asarray(ddy_chunk, dtype=float),
                    "f": np.asarray(f_chunk, dtype=float),
                    "ur": np.asarray(ur_chunk, dtype=float),
                    "ur_scalar": float(ur_scalar),
                    "ur_scalar_chunk_mean": float(ur_chunk_mean),
                    "ur_scalar_source_mean": float(ur_source_mean),
                    "u_mean_source": float(source["u_mean_source"]),
                    "ur_mean_source": float(source["ur_mean_source"]),
                    "ur_std_source": float(source["ur_std_source"]),
                    "m_inertia_used": float(source["m_inertia_used"]),
                }
            )

    if not chunk_records:
        raise ValueError("No chunks available for export. Adjust chunking/downsampling settings.")

    split_mode = str(SPLIT_STRATIFY_MODE).strip().lower()
    if split_mode == "ur_bin":
        ur_values = np.asarray([float(c["ur_scalar"]) for c in chunk_records], dtype=float)
        strata = _ur_bin_strata(ur_values, n_bins=int(SPLIT_STRATIFY_UR_BINS))
    elif split_mode == "source_file":
        strata = [str(c["source_file"]) for c in chunk_records]
    else:
        strata = [0 for _ in chunk_records]

    split_names = _split_labels_from_strata(
        strata,
        train_fraction=float(TRAIN_FRACTION),
        val_fraction=float(VAL_FRACTION),
        seed=int(SPLIT_SEED),
    )

    if bool(CLEAN_OUTPUT_DIR_BEFORE_EXPORT):
        _clean_output_dir(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "train").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "val").mkdir(parents=True, exist_ok=True)

    corrected_mat_by_source: dict[str, Path | None] = {}
    if bool(EXPORT_CORRECTED_MAT):
        for source in prepared_sources:
            mat_path = Path(source["mat_path"])
            # Always export corrected MAT from full untrimmed source arrays.
            corrected_mat_path = _export_corrected_mat(
                source_mat_path=mat_path,
                split_name="all",
                y_corr=np.asarray(source["y_corr_full"], dtype=float),
                dy_corr=np.asarray(source["y_dot_corr_full"], dtype=float),
                f_calc=np.asarray(source["f_calc_full"], dtype=float),
                overwrite=bool(OVERWRITE),
            )
            corrected_mat_by_source[str(mat_path.name)] = corrected_mat_path
            print(f"Wrote {corrected_mat_path}")

    metadata: list[dict[str, object]] = []
    idx = 1
    for split_name, chunk in zip(split_names, chunk_records):
        t = np.asarray(chunk["t"], dtype=float)
        y_corr = np.asarray(chunk["y"], dtype=float)
        dy_corr = np.asarray(chunk["dy"], dtype=float)
        ddy_corr = np.asarray(chunk["ddy"], dtype=float)
        f_calc = np.asarray(chunk["f"], dtype=float)
        ur_chunk = np.asarray(chunk["ur"], dtype=float)
        ur_scalar = float(chunk["ur_scalar"])
        payload = _build_payload(
            time=t,
            y_corr=y_corr,
            dy_corr=dy_corr,
            ddy_corr=ddy_corr,
            f_calc=f_calc,
            ur_scalar=ur_scalar,
        )

        out_name = f"series_{idx:04d}_{chunk['source_label']}_chunk{int(chunk['chunk_index']):03d}.npz"
        out_path = OUTPUT_DIR / str(split_name) / out_name
        if out_path.exists() and not bool(OVERWRITE):
            raise FileExistsError(f"Refusing to overwrite existing file: {out_path}")
        np.savez(out_path, **payload)

        dt_out = float(np.median(np.diff(t)))
        corrected_mat_path = corrected_mat_by_source.get(str(chunk["source_file"]))
        metadata.append(
            {
                "index": int(idx),
                "split": str(split_name),
                "file": str(out_name),
                "source_file": str(chunk["source_file"]),
                "source_label": str(chunk["source_label"]),
                "source_chunk_index": int(chunk["chunk_index"]),
                "chunk_start_index": int(chunk["start_idx"]),
                "chunk_end_index": int(chunk["end_idx"]),
                "chunk_start_time_s": float(t[0]),
                "chunk_end_time_s": float(t[-1]),
                "U_r": float(ur_scalar),
                "U": float(ur_scalar * float(analysis.FN) * float(analysis.D)),
                "mean_reduced_velocity": float(ur_scalar),
                "chunk_mean_reduced_velocity": float(chunk["ur_scalar_chunk_mean"]),
                "std_reduced_velocity": float(np.std(ur_chunk)),
                "source_mean_reduced_velocity": float(chunk["ur_mean_source"]),
                "source_std_reduced_velocity": float(chunk["ur_std_source"]),
                "chunk_use_global_mean_ur": bool(CHUNK_USE_GLOBAL_MEAN_UR),
                "samples": int(t.size),
                "dt": dt_out,
                "duration_s": float(t[-1] - t[0]),
                "chunk_length_seconds_setting": float(CHUNK_LENGTH_SECONDS),
                "chunk_stride_seconds_setting": float(CHUNK_STRIDE_SECONDS),
                "split_stratify_mode": str(SPLIT_STRATIFY_MODE),
                "split_stratify_ur_bins": int(SPLIT_STRATIFY_UR_BINS),
                "phase_correction_enabled": bool(phase.PHASE_CORRECTION_ENABLED),
                "phase_correction_mode": str(phase.PHASE_CORRECTION_MODE),
                "phase_window_seconds": float(phase.PHASE_WINDOW_SECONDS),
                "phase_window_overlap": float(phase.PHASE_WINDOW_OVERLAP),
                "phase_lag_method": str(getattr(phase, "LAG_ESTIMATION_METHOD", "phase")),
                "phase_common_lag_polyorder": int(phase.PHASE_COMMON_LAG_POLYORDER),
                "phase_common_lag_extrapolate": bool(getattr(phase, "PHASE_COMMON_LAG_EXTRAPOLATE", False)),
                "force_common_linear_lag_extrapolation": bool(FORCE_COMMON_LINEAR_LAG_EXTRAPOLATION),
                "npz_trim_start_seconds_setting": float(NPZ_TRIM_START_SECONDS),
                "npz_trim_end_seconds_setting": float(NPZ_TRIM_END_SECONDS),
                # Backward-compatible keys:
                "trim_start_seconds_setting": float(NPZ_TRIM_START_SECONDS),
                "trim_end_seconds_setting": float(NPZ_TRIM_END_SECONDS),
                "force_fy1_mult": float(FORCE_FY1_MULT),
                "force_fy2_mult": float(FORCE_FY2_MULT),
                "force_global_sign": float(FORCE_GLOBAL_SIGN),
                "force_include_inertia": bool(FORCE_INCLUDE_INERTIA),
                "force_inertia_sign": float(FORCE_INERTIA_SIGN),
                "force_include_structural_mass": bool(FORCE_INCLUDE_STRUCTURAL_MASS),
                "force_include_added_mass": bool(FORCE_INCLUDE_ADDED_MASS),
                "force_added_mass_coeff": float(FORCE_ADDED_MASS_COEFF),
                "force_m_inertia_used": float(chunk["m_inertia_used"]),
                "corrected_mat_file": str(corrected_mat_path) if corrected_mat_path is not None else None,
            }
        )
        print(f"Wrote {out_path}")
        idx += 1

    metadata_path = OUTPUT_DIR / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
