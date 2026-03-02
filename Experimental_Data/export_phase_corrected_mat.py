from __future__ import annotations

from pathlib import Path
import shutil
import sys
import warnings

import numpy as np
from scipy.io import savemat

try:
    from Experimental_Data.script_helpers import (
        filter_excluded_tests as _filter_excluded_tests_common,
        import_analysis_and_extracted,
        import_analysis_and_phase,
    )
except ModuleNotFoundError:
    current_dir = Path(__file__).resolve().parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    from script_helpers import (
        filter_excluded_tests as _filter_excluded_tests_common,
        import_analysis_and_extracted,
        import_analysis_and_phase,
    )

analysis, phase = import_analysis_and_phase(__file__)
_analysis_for_extracted, extracted, _used_extracted_fallback = import_analysis_and_extracted(
    __file__,
    allow_extracted_fallback=True,
    print_fallback_message=False,
)


# -------------------------
# Export settings
# -------------------------
OUTPUT_DIR = Path("Experimental_Data/CrossFlow/CorrectedData")
OUTPUT_SUFFIX = "_corrected"
OVERWRITE = True
CLEAN_OUTPUT_DIR_BEFORE_EXPORT = True

# Keep selection aligned with phase_analysis.py.
# This script uses phase._resolve_phase_input_files() and phase._filter_excluded_tests().
# You can apply extra export-only exclusions below.
EXCLUDE_TEST_NUMBERS: list[int] = [3009, 3002]

# Displacement-column replacement settings.
CORRECTED_DISPLACEMENT_ALIASES = ["xpos1"]
CORRECTED_DISPLACEMENT_FALLBACK_INDICES = [23]
RENAME_CORRECTED_DISPLACEMENT_CHANNELS = False
CORRECTED_DISPLACEMENT_CHANNEL_NAME = "y_corrected"

# Deprecated/no-op: phase-analysis trim functionality was removed.
APPLY_PHASE_TRIM_TO_EXPORT = False

# MAT data variable override for loading input files.
# Use "data" for raw files and "data_corrected" for corrected files.
DATA_VARIABLE_OVERRIDE: str | None = "data"

# If phase correction fails (e.g., no valid phase windows), fallback to exporting
# original displacement unless this is set True.
FAIL_IF_PHASE_CORRECTION_UNAVAILABLE = True

# Use full-series mask for phase correction inside this exporter.
# This avoids inheriting short "early window" masks from upstream loaders.
FORCE_FULL_MASK_FOR_PHASE_CORRECTION = True


def _apply_extra_exclusions(paths: list[Path]) -> list[Path]:
    return _filter_excluded_tests_common([Path(p) for p in paths], EXCLUDE_TEST_NUMBERS)


def _resolve_input_files() -> tuple[list[Path], str]:
    source_desc = "phase.MAT_FILES (fallback)"
    if hasattr(phase, "_resolve_phase_input_files"):
        phase_files, phase_source = phase._resolve_phase_input_files()
        files = [Path(p) for p in phase_files]
        source_desc = str(phase_source)
    else:
        files = [Path(p) for p in list(getattr(phase, "MAT_FILES", []))]

    if hasattr(phase, "_filter_excluded_tests"):
        files = phase._filter_excluded_tests(files)

    files = _apply_extra_exclusions(files)
    if not files:
        raise ValueError("No MAT files selected after phase selection and exclusions.")
    missing = [p for p in files if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing MAT file(s): {missing}")
    return files, source_desc


def _resolve_displacement_indices(
    *,
    n_cols: int,
    channel_names: list[str] | None,
) -> list[int]:
    indices: list[int] = []
    if channel_names is not None:
        idx_map = {analysis._norm_name(name): i for i, name in enumerate(channel_names)}
        for alias in CORRECTED_DISPLACEMENT_ALIASES:
            idx = idx_map.get(analysis._norm_name(alias))
            if idx is not None and 0 <= idx < n_cols and idx not in indices:
                indices.append(int(idx))
    for idx in CORRECTED_DISPLACEMENT_FALLBACK_INDICES:
        ii = int(idx)
        if 0 <= ii < n_cols and ii not in indices:
            indices.append(ii)
    return indices


def _clean_output_dir(path: Path) -> None:
    target = Path(path)
    target_str = str(target).strip()
    if target_str in {"", ".", "/"}:
        raise ValueError(f"Refusing to clean unsafe output path: {target}")
    if target.exists():
        shutil.rmtree(target)


def _build_corrected_displacement_by_label(
    results: list[dict[str, object]],
) -> dict[str, np.ndarray]:
    if bool(phase.PHASE_CORRECTION_ENABLED):
        try:
            phase_before = phase._collect_phase_series(results)
            return phase._build_corrected_displacement(results, phase_before)
        except Exception as exc:
            if bool(FAIL_IF_PHASE_CORRECTION_UNAVAILABLE):
                raise
            warnings.warn(
                "Phase correction unavailable; falling back to original displacement for export. "
                f"Reason: {type(exc).__name__}: {exc}"
            )
    return {
        str(result["label"]): np.asarray(phase._select_channel(result, "Displacement y (m)"), dtype=float)
        for result in results
    }


def _compute_corrected_kinematics(y_corr: np.ndarray, *, dt: float) -> tuple[np.ndarray, np.ndarray]:
    y_arr = np.asarray(y_corr, dtype=float).reshape(-1)
    if y_arr.size < 5:
        raise ValueError("Corrected kinematics require Savitzky-Golay derivatives (need at least 5 samples).")
    if not np.isfinite(float(dt)) or float(dt) <= 0.0:
        raise ValueError("dt must be finite and positive for corrected kinematics.")

    try:
        deriv_out = extracted._compute_derivatives(y_arr, dt=float(dt))
    except Exception as exc:
        raise RuntimeError(
            "Corrected kinematics require Savitzky-Golay derivatives, "
            f"but derivative computation failed: {type(exc).__name__}: {exc}"
        ) from exc

    if not (isinstance(deriv_out, tuple) and len(deriv_out) >= 2):
        raise RuntimeError(f"Unexpected derivative return format: {type(deriv_out).__name__}")
    y_dot = np.asarray(deriv_out[0], dtype=float)
    y_ddot = np.asarray(deriv_out[1], dtype=float)
    deriv_meta = deriv_out[2] if len(deriv_out) >= 3 else {}
    mode = ""
    if isinstance(deriv_meta, dict):
        mode = str(deriv_meta.get("mode", "")).strip().lower()
    if "savgol" not in mode:
        raise RuntimeError(
            "Corrected kinematics require Savitzky-Golay derivatives; "
            f"got derivative mode='{mode or 'unknown'}'."
        )
    return y_dot, y_ddot


def _export_one(
    *,
    source_mat: Path,
    y_corr: np.ndarray,
    dy_corr: np.ndarray,
    ddy_corr: np.ndarray,
    output_dir: Path,
) -> Path:
    source_data, source_channel_names = analysis._load_data_matrix(source_mat, phase.DATA_VARIABLE)
    data_out = np.asarray(source_data, dtype=float).copy()
    n_rows, n_cols = data_out.shape

    y = np.asarray(y_corr, dtype=float).reshape(-1)
    dy = np.asarray(dy_corr, dtype=float).reshape(-1)
    ddy = np.asarray(ddy_corr, dtype=float).reshape(-1)
    n = int(min(n_rows, y.size, dy.size, ddy.size))
    if n < 2:
        raise ValueError(f"{source_mat.name}: not enough samples to export corrected MAT.")
    if y.size != n_rows:
        warnings.warn(
            f"{source_mat.name}: corrected displacement length ({y.size}) differs from source rows ({n_rows}); "
            f"exporting first {n} samples."
        )

    data_out = data_out[:n, :]
    y = y[:n]
    dy = dy[:n]
    ddy = ddy[:n]

    disp_indices = _resolve_displacement_indices(n_cols=n_cols, channel_names=source_channel_names)
    if not disp_indices:
        raise ValueError(f"{source_mat.name}: could not resolve displacement column(s) to overwrite.")
    for idx in disp_indices:
        data_out[:, idx] = y

    channel_names_out = None
    if source_channel_names is not None:
        channel_names_out = list(source_channel_names)
        if bool(RENAME_CORRECTED_DISPLACEMENT_CHANNELS):
            for idx in disp_indices:
                channel_names_out[idx] = str(CORRECTED_DISPLACEMENT_CHANNEL_NAME)

    time_series = analysis._select_column(
        data_out,
        source_channel_names,
        ["Time", "time"],
        0,
        role="corrected-mat export time",
    )
    time_series = analysis._fill_nonfinite_1d(np.asarray(time_series, dtype=float), role=f"{source_mat.name}: time")

    payload: dict[str, object] = {
        "data_corrected": np.asarray(data_out, dtype=float),
        "data": np.asarray(data_out, dtype=float),  # compatibility with raw-like readers
        "time": np.asarray(time_series, dtype=float).reshape(-1, 1),
        "y_corrected": np.asarray(y, dtype=float).reshape(-1, 1),
        "dy_corrected": np.asarray(dy, dtype=float).reshape(-1, 1),
        "ddy_corrected": np.asarray(ddy, dtype=float).reshape(-1, 1),
        "velocity_corrected": np.asarray(dy, dtype=float).reshape(-1, 1),
        "acceleration_corrected": np.asarray(ddy, dtype=float).reshape(-1, 1),
        "source_file": np.array([str(source_mat.name)], dtype=object),
    }
    if channel_names_out is not None:
        payload["chan_names"] = np.array(channel_names_out, dtype=object).reshape(-1, 1)
        payload["column_names"] = np.array(channel_names_out, dtype=object)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{source_mat.stem}{str(OUTPUT_SUFFIX)}.mat"
    if out_path.exists() and not bool(OVERWRITE):
        raise FileExistsError(f"Refusing to overwrite existing file: {out_path}")
    savemat(out_path, payload, do_compression=True)
    return out_path


def main() -> None:
    # Keep extraction behavior aligned with phase config.
    if DATA_VARIABLE_OVERRIDE is not None:
        phase.DATA_VARIABLE = DATA_VARIABLE_OVERRIDE
    extracted.DATA_VARIABLE = phase.DATA_VARIABLE
    extracted.USE_RELATIVE_TIME = phase.USE_RELATIVE_TIME
    if hasattr(extracted, "DERIV_SAVGOL_WINDOW"):
        extracted.DERIV_SAVGOL_WINDOW = int(phase.PHASE_CORRECTION_POST_SAVGOL_WINDOW)
    if hasattr(extracted, "DERIV_SAVGOL_POLYORDER"):
        extracted.DERIV_SAVGOL_POLYORDER = int(phase.PHASE_CORRECTION_POST_SAVGOL_POLYORDER)

    files, source_desc = _resolve_input_files()
    print(f"Corrected-MAT export source: {source_desc}")
    print(f"Corrected-MAT export DATA_VARIABLE: {phase.DATA_VARIABLE!r}")
    print(f"Selected {len(files)} file(s).")

    results: list[dict[str, object]] = []
    for mat_file in files:
        result = extracted._process_file(mat_file)
        if bool(FORCE_FULL_MASK_FOR_PHASE_CORRECTION):
            t_arr = np.asarray(result.get("t", []), dtype=float).reshape(-1)
            if t_arr.size > 0:
                result["mask"] = np.ones(t_arr.size, dtype=bool)
        if bool(APPLY_PHASE_TRIM_TO_EXPORT):
            warnings.warn(
                "APPLY_PHASE_TRIM_TO_EXPORT is ignored because phase trimming was removed from phase_analysis.py."
            )
        results.append(result)

    corrected_by_label = _build_corrected_displacement_by_label(results)

    out_dir = Path(OUTPUT_DIR).resolve()
    if bool(CLEAN_OUTPUT_DIR_BEFORE_EXPORT):
        _clean_output_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for result in results:
        source_mat = Path(result["path"])
        label = str(result["label"])
        y_corr = np.asarray(corrected_by_label[label], dtype=float)
        dt = float(result["dt"])
        y_dot_corr, y_ddot_corr = _compute_corrected_kinematics(y_corr, dt=dt)
        out_path = _export_one(
            source_mat=source_mat,
            y_corr=y_corr,
            dy_corr=y_dot_corr,
            ddy_corr=y_ddot_corr,
            output_dir=out_dir,
        )
        written.append(out_path)
        print(f"Wrote {out_path}")

    print(f"Done. Wrote {len(written)} corrected MAT file(s) to {out_dir}")


if __name__ == "__main__":
    main()
