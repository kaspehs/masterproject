from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

DATA_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DATA_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vivana_cfd_data_pipeline.helpers.cfd_io import (
    CfdRecord,
    apply_cleaning_manifest,
    build_cleaning_manifest,
    infer_ur_from_path,
    load_dog_file,
    nondimensionalize_records_from_metadata,
    remove_duplicate_timestamps,
    resolve_computed_ur_value,
    resolve_dry_structural_frequency_hz,
    resolve_flow_speed_m_s,
    resolve_stiffness_n_m,
    resolve_structural_frequency_hz,
)


# Export configuration
INPUT_DIR = DATA_ROOT / "raw"
OUTPUT_DIR = DATA_ROOT / "generated" / "cfd_npz_exports"
PATTERN = "*.dog"
MAX_FILES = None
OVERWRITE = True

# Existing notebook-driven preprocessing inputs
CLEANING_MANIFEST_PATH = DATA_ROOT / "metadata" / "cleaning_manifest.csv"
METADATA_PATH = DATA_ROOT / "metadata" / "CFD_metadata.csv"

# Nondimensional time convention for the exported *_nd channels:
# tau = omega_n * t
TIME_SCALE_LABEL = "omega_n"

# Precompute TD hidden sigma states from the previous WINDOW points.
TD_SIGMA_WINDOW_SAMPLES = 500


def _load_records(input_dir: Path, *, pattern: str, max_files: int | None) -> list[CfdRecord]:
    input_dir = input_dir.resolve()
    files = sorted(input_dir.rglob(pattern))
    if max_files is not None:
        files = files[: max(0, int(max_files))]
    if not files:
        raise FileNotFoundError(f"No files matched '{pattern}' under '{input_dir}'.")

    records: list[CfdRecord] = []
    for path in files:
        data, skipped = load_dog_file(path)
        case_name = path.relative_to(input_dir).with_suffix("").as_posix().replace("/", "__")
        records.append(
            CfdRecord(
                path=path,
                case_name=case_name,
                ur_value=infer_ur_from_path(path),
                data=data,
                skipped_lines=skipped,
            )
        )
    return records


def _load_cleaned_records(
    records: list[CfdRecord],
    manifest_path: Path,
) -> tuple[list[CfdRecord], pd.DataFrame, pd.DataFrame]:
    if manifest_path.exists():
        manifest_df = pd.read_csv(manifest_path)
    else:
        manifest_df = build_cleaning_manifest(records, manifest_path)
        print(f"Created default cleaning manifest at {manifest_path.resolve()}")
    cleaned_records, applied_df = apply_cleaning_manifest(records, manifest_df)
    dedup_records, dedup_df = remove_duplicate_timestamps(cleaned_records)
    return dedup_records, applied_df, dedup_df


def _resolve_time_scale_factor(
    metadata_row: dict[str, object],
    case_name: str,
) -> tuple[float, float]:
    structural_frequency_hz = resolve_structural_frequency_hz(metadata_row, case_name)
    omega_n = float(2.0 * np.pi * structural_frequency_hz)
    return omega_n, structural_frequency_hz


def _rolling_rms(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    n = values.size
    if n == 0:
        return np.asarray([], dtype=float)
    window = max(1, min(int(window), n))
    squared = values**2
    csum = np.concatenate(([0.0], np.cumsum(squared, dtype=float)))
    out = np.empty(n, dtype=float)
    for i in range(n):
        start = max(0, i - window + 1)
        count = i - start + 1
        out[i] = np.sqrt((csum[i + 1] - csum[start]) / float(count))
    if n >= window:
        out[:window] = out[window - 1]
    return out


def _precompute_td_sigmas(
    *,
    y_vel_dim: np.ndarray,
    y_acc_dim: np.ndarray,
    flow_speed_m_s: float,
    window_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y_vel_dim = np.asarray(y_vel_dim, dtype=float).reshape(-1)
    y_acc_dim = np.asarray(y_acc_dim, dtype=float).reshape(-1)
    if y_vel_dim.shape != y_acc_dim.shape:
        raise ValueError("y_vel_dim and y_acc_dim must have the same shape.")

    speed_mag = np.sqrt(float(flow_speed_m_s) ** 2 + y_vel_dim**2)
    projection = float(flow_speed_m_s) / np.maximum(speed_mag, np.finfo(float).eps)
    y_vel_local_dim = y_vel_dim * projection
    y_acc_local_dim = y_acc_dim * projection
    sig_dy_dim = _rolling_rms(y_vel_local_dim, int(window_samples))
    sig_ddy_dim = _rolling_rms(y_acc_local_dim, int(window_samples))
    return y_vel_local_dim, y_acc_local_dim, sig_dy_dim, sig_ddy_dim


def _build_payload(
    record_dim: CfdRecord,
    record_nd: CfdRecord,
    metadata_row: dict[str, object],
    action_row: dict[str, object] | None,
    dedup_row: dict[str, object] | None,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    def _maybe_float(key: str) -> float:
        value = metadata_row.get(key, np.nan)
        return float(value) if pd.notna(value) else float("nan")

    n = int(record_dim.data.shape[0])
    case_name = record_dim.case_name
    time_scale_factor, structural_frequency_hz = _resolve_time_scale_factor(
        metadata_row,
        case_name,
    )
    dry_structural_frequency_hz = np.nan
    try:
        dry_structural_frequency_hz = float(resolve_dry_structural_frequency_hz(metadata_row, case_name))
    except Exception:
        pass

    flow_speed_m_s = float(resolve_flow_speed_m_s(metadata_row, case_name))
    diameter_m = float(metadata_row["diameter_m"])
    rho_kg_m3 = float(metadata_row["rho_kg_m3"])
    span_m = float(metadata_row["span_m"])
    stiffness_n_m = float(resolve_stiffness_n_m(metadata_row, case_name, structural_frequency_hz=structural_frequency_hz))
    effective_mass_kg = float(metadata_row["effective_mass_kg"]) if pd.notna(metadata_row.get("effective_mass_kg", np.nan)) else np.nan
    dry_mass_kg = float(metadata_row["dry_mass_kg"]) if pd.notna(metadata_row.get("dry_mass_kg", np.nan)) else np.nan
    python_condition_set = metadata_row.get("python_condition_set", "")
    python_stiffness_n_m = _maybe_float("python_stiffness_n_m")
    python_mass_kg = _maybe_float("python_mass_kg")
    python_damping_c = _maybe_float("python_damping_c")
    python_rho_kg_m3 = _maybe_float("python_rho_kg_m3")
    python_flow_speed_m_s = _maybe_float("python_flow_speed_m_s")
    python_diameter_m = _maybe_float("python_diameter_m")
    python_n_memory = _maybe_float("python_n_memory")
    python_cv = _maybe_float("python_cv")
    python_cd = _maybe_float("python_cd")
    python_ca = _maybe_float("python_ca")
    python_fhat0 = _maybe_float("python_fhat0")
    python_fhat_min = _maybe_float("python_fhat_min")
    python_fhat_max = _maybe_float("python_fhat_max")
    python_dt_s = _maybe_float("python_dt_s")
    python_primary_integrator = metadata_row.get("python_primary_integrator", "")
    python_span_m = 1.0
    python_dry_mass_kg = python_mass_kg
    python_added_mass_kg = (
        0.25 * np.pi * python_rho_kg_m3 * python_ca * python_diameter_m**2
        if all(np.isfinite(value) for value in (python_rho_kg_m3, python_ca, python_diameter_m))
        else float("nan")
    )
    python_effective_mass_kg = (
        python_dry_mass_kg + python_added_mass_kg
        if all(np.isfinite(value) for value in (python_dry_mass_kg, python_added_mass_kg))
        else float("nan")
    )
    computed_ur = float(resolve_computed_ur_value(metadata_row, case_name))
    label_ur = float(record_dim.ur_value) if record_dim.ur_value is not None else np.nan
    omega_n = float(2.0 * np.pi * structural_frequency_hz)
    force_scale = float(0.5 * rho_kg_m3 * diameter_m**2 * span_m * flow_speed_m_s**2)
    vel_scale = float(omega_n * diameter_m)
    acc_scale = float(omega_n**2 * diameter_m)
    x_force_per_m_dim = np.asarray(record_dim.data[:, 7], dtype=float)
    y_force_per_m_dim = np.asarray(record_dim.data[:, 8], dtype=float)
    x_force_total_dim = x_force_per_m_dim * span_m
    y_force_total_dim = y_force_per_m_dim * span_m
    (
        y_vel_local_dim,
        y_acc_local_dim,
        sig_dy_dim,
        sig_ddy_dim,
    ) = _precompute_td_sigmas(
        y_vel_dim=np.asarray(record_dim.data[:, 4], dtype=float),
        y_acc_dim=np.asarray(record_dim.data[:, 6], dtype=float),
        flow_speed_m_s=flow_speed_m_s,
        window_samples=TD_SIGMA_WINDOW_SAMPLES,
    )
    y_vel_local_nd = y_vel_local_dim / vel_scale
    y_acc_local_nd = y_acc_local_dim / acc_scale
    sig_dy_nd = sig_dy_dim / vel_scale
    sig_ddy_nd = sig_ddy_dim / acc_scale

    ur_series = np.full((n,), computed_ur, dtype=float)
    ur_label_series = np.full((n,), label_ur, dtype=float)

    payload: dict[str, np.ndarray] = {
        # Compatibility aliases: default to nondimensional channels.
        "time": np.asarray(record_nd.time, dtype=float),
        "y": np.asarray(record_nd.data[:, 2], dtype=float),
        "dy": np.asarray(record_nd.data[:, 4], dtype=float),
        "ddy": np.asarray(record_nd.data[:, 6], dtype=float),
        "F_total": np.asarray(record_nd.data[:, 8], dtype=float),
        "cf_force": np.asarray(record_nd.data[:, 8], dtype=float),
        "U_r": ur_series,
        "a": np.asarray(record_nd.time, dtype=float),
        "b": np.asarray(record_nd.data[:, 2], dtype=float),
        "c": np.asarray(record_nd.data[:, 8], dtype=float),
        "e": np.asarray(record_nd.data[:, 4], dtype=float),
        # Explicit dimensional channels.
        "time_dim": np.asarray(record_dim.time, dtype=float),
        "x_disp_dim": np.asarray(record_dim.data[:, 1], dtype=float),
        "y_disp_dim": np.asarray(record_dim.data[:, 2], dtype=float),
        "x_vel_dim": np.asarray(record_dim.data[:, 3], dtype=float),
        "y_vel_dim": np.asarray(record_dim.data[:, 4], dtype=float),
        "x_acc_dim": np.asarray(record_dim.data[:, 5], dtype=float),
        "y_acc_dim": np.asarray(record_dim.data[:, 6], dtype=float),
        "x_force_dim": np.asarray(x_force_total_dim, dtype=float),
        "y_force_dim": np.asarray(y_force_total_dim, dtype=float),
        "x_force_per_m_dim": np.asarray(x_force_per_m_dim, dtype=float),
        "y_force_per_m_dim": np.asarray(y_force_per_m_dim, dtype=float),
        "y_vel_local_dim": np.asarray(y_vel_local_dim, dtype=float),
        "y_acc_local_dim": np.asarray(y_acc_local_dim, dtype=float),
        "sig_dy_dim": np.asarray(sig_dy_dim, dtype=float),
        "sig_ddy_dim": np.asarray(sig_ddy_dim, dtype=float),
        # Explicit nondimensional channels.
        "time_nd": np.asarray(record_nd.time, dtype=float),
        "x_disp_nd": np.asarray(record_nd.data[:, 1], dtype=float),
        "y_disp_nd": np.asarray(record_nd.data[:, 2], dtype=float),
        "x_vel_nd": np.asarray(record_nd.data[:, 3], dtype=float),
        "y_vel_nd": np.asarray(record_nd.data[:, 4], dtype=float),
        "x_acc_nd": np.asarray(record_nd.data[:, 5], dtype=float),
        "y_acc_nd": np.asarray(record_nd.data[:, 6], dtype=float),
        "x_force_nd": np.asarray(record_nd.data[:, 7], dtype=float),
        "y_force_nd": np.asarray(record_nd.data[:, 8], dtype=float),
        "y_vel_local_nd": np.asarray(y_vel_local_nd, dtype=float),
        "y_acc_local_nd": np.asarray(y_acc_local_nd, dtype=float),
        "sig_dy_nd": np.asarray(sig_dy_nd, dtype=float),
        "sig_ddy_nd": np.asarray(sig_ddy_nd, dtype=float),
        # Scalar metadata as 0d arrays for convenience in npz-only workflows.
        "U_r_label_series": ur_label_series,
        "U_r_computed_series": ur_series,
        "U_r_label_scalar": np.asarray(label_ur, dtype=float),
        "U_r_computed_scalar": np.asarray(computed_ur, dtype=float),
        "dt_dim": np.asarray(float(np.median(np.diff(record_dim.time))), dtype=float),
        "dt_nd": np.asarray(float(np.median(np.diff(record_nd.time))), dtype=float),
        "time_scale_factor": np.asarray(time_scale_factor, dtype=float),
        "structural_frequency_hz": np.asarray(structural_frequency_hz, dtype=float),
        "dry_structural_frequency_hz": np.asarray(dry_structural_frequency_hz, dtype=float),
        "omega_n": np.asarray(omega_n, dtype=float),
        "flow_speed_m_s": np.asarray(flow_speed_m_s, dtype=float),
        "diameter_m": np.asarray(diameter_m, dtype=float),
        "rho_kg_m3": np.asarray(rho_kg_m3, dtype=float),
        "span_m": np.asarray(span_m, dtype=float),
        "stiffness_n_m": np.asarray(stiffness_n_m, dtype=float),
        "effective_mass_kg": np.asarray(effective_mass_kg, dtype=float),
        "dry_mass_kg": np.asarray(dry_mass_kg, dtype=float),
        "python_condition_set": np.asarray(str(python_condition_set)),
        "python_span_m": np.asarray(float(python_span_m), dtype=float),
        "python_stiffness_n_m": np.asarray(python_stiffness_n_m, dtype=float),
        "python_mass_kg": np.asarray(python_mass_kg, dtype=float),
        "python_dry_mass_kg": np.asarray(python_dry_mass_kg, dtype=float),
        "python_effective_mass_kg": np.asarray(python_effective_mass_kg, dtype=float),
        "python_damping_c": np.asarray(python_damping_c, dtype=float),
        "python_rho_kg_m3": np.asarray(python_rho_kg_m3, dtype=float),
        "python_flow_speed_m_s": np.asarray(python_flow_speed_m_s, dtype=float),
        "python_diameter_m": np.asarray(python_diameter_m, dtype=float),
        "python_n_memory": np.asarray(python_n_memory, dtype=float),
        "python_cv": np.asarray(python_cv, dtype=float),
        "python_cd": np.asarray(python_cd, dtype=float),
        "python_ca": np.asarray(python_ca, dtype=float),
        "python_fhat0": np.asarray(python_fhat0, dtype=float),
        "python_fhat_min": np.asarray(python_fhat_min, dtype=float),
        "python_fhat_max": np.asarray(python_fhat_max, dtype=float),
        "python_dt_s": np.asarray(python_dt_s, dtype=float),
        "python_primary_integrator": np.asarray(str(python_primary_integrator)),
        "force_scale": np.asarray(force_scale, dtype=float),
        "velocity_scale": np.asarray(vel_scale, dtype=float),
        "acceleration_scale": np.asarray(acc_scale, dtype=float),
        "raw_force_span_scale_applied": np.asarray(span_m, dtype=float),
        "physical_span_m": np.asarray(span_m, dtype=float),
        "label_ur": np.asarray(label_ur, dtype=float),
        "computed_ur": np.asarray(computed_ur, dtype=float),
        "num_rows": np.asarray(n, dtype=int),
        "td_sigma_window_samples": np.asarray(int(TD_SIGMA_WINDOW_SAMPLES), dtype=int),
    }

    record_meta: dict[str, object] = {
        "case_name": case_name,
        "source_file": str(record_dim.path),
        "output_file": f"{case_name}.npz",
        "num_rows": n,
        "dt_dim": float(payload["dt_dim"]),
        "dt_nd": float(payload["dt_nd"]),
        "time_scale_mode": TIME_SCALE_LABEL,
        "time_scale_factor": float(time_scale_factor),
        "structural_frequency_hz": structural_frequency_hz,
        "dry_structural_frequency_hz": (None if not np.isfinite(dry_structural_frequency_hz) else dry_structural_frequency_hz),
        "omega_n": omega_n,
        "flow_speed_m_s": flow_speed_m_s,
        "diameter_m": diameter_m,
        "rho_kg_m3": rho_kg_m3,
        "span_m": span_m,
        "stiffness_n_m": stiffness_n_m,
        "effective_mass_kg": (None if not np.isfinite(effective_mass_kg) else effective_mass_kg),
        "dry_mass_kg": (None if not np.isfinite(dry_mass_kg) else dry_mass_kg),
        "python_condition_set": str(python_condition_set),
        "python_span_m": float(python_span_m),
        "python_stiffness_n_m": (None if not np.isfinite(python_stiffness_n_m) else python_stiffness_n_m),
        "python_mass_kg": (None if not np.isfinite(python_mass_kg) else python_mass_kg),
        "python_dry_mass_kg": (None if not np.isfinite(python_dry_mass_kg) else python_dry_mass_kg),
        "python_effective_mass_kg": (None if not np.isfinite(python_effective_mass_kg) else python_effective_mass_kg),
        "python_damping_c": (None if not np.isfinite(python_damping_c) else python_damping_c),
        "python_rho_kg_m3": (None if not np.isfinite(python_rho_kg_m3) else python_rho_kg_m3),
        "python_flow_speed_m_s": (None if not np.isfinite(python_flow_speed_m_s) else python_flow_speed_m_s),
        "python_diameter_m": (None if not np.isfinite(python_diameter_m) else python_diameter_m),
        "python_n_memory": (None if not np.isfinite(python_n_memory) else python_n_memory),
        "python_cv": (None if not np.isfinite(python_cv) else python_cv),
        "python_cd": (None if not np.isfinite(python_cd) else python_cd),
        "python_ca": (None if not np.isfinite(python_ca) else python_ca),
        "python_fhat0": (None if not np.isfinite(python_fhat0) else python_fhat0),
        "python_fhat_min": (None if not np.isfinite(python_fhat_min) else python_fhat_min),
        "python_fhat_max": (None if not np.isfinite(python_fhat_max) else python_fhat_max),
        "python_dt_s": (None if not np.isfinite(python_dt_s) else python_dt_s),
        "python_primary_integrator": str(python_primary_integrator),
        "force_scale": force_scale,
        "velocity_scale": vel_scale,
        "acceleration_scale": acc_scale,
        "force_input_convention": "per_unit_length_from_dog",
        "force_output_convention_dim": "total_force_over_span",
        "force_output_convention_nd": "total_force_over_span",
        "raw_force_span_scale_applied": span_m,
        "physical_span_m": span_m,
        "label_ur": (None if not np.isfinite(label_ur) else label_ur),
        "computed_ur": computed_ur,
        "compatibility_alias_source": "nondimensional",
        "td_sigma_window_samples": int(TD_SIGMA_WINDOW_SAMPLES),
    }
    if action_row is not None:
        record_meta.update(
            {
                "action_applied": action_row.get("action_applied"),
                "start_time_after": action_row.get("start_time_after"),
                "end_time_after": action_row.get("end_time_after"),
            }
        )
    if dedup_row is not None:
        record_meta.update(
            {
                "duplicate_rows_removed": int(dedup_row.get("duplicate_rows_removed", 0)),
                "has_duplicate_timestamps": bool(dedup_row.get("has_duplicate_timestamps", False)),
                "first_duplicate_time": (
                    None
                    if pd.isna(dedup_row.get("first_duplicate_time", np.nan))
                    else float(dedup_row.get("first_duplicate_time"))
                ),
            }
        )
    return payload, record_meta


def main() -> None:
    input_dir = INPUT_DIR.resolve()
    output_dir = OUTPUT_DIR.resolve()
    manifest_path = CLEANING_MANIFEST_PATH.resolve()
    metadata_path = METADATA_PATH.resolve()

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file '{metadata_path}' not found.")

    records = _load_records(input_dir, pattern=str(PATTERN), max_files=MAX_FILES)
    cleaned_records, applied_df, dedup_df = _load_cleaned_records(records, manifest_path)
    metadata_df = pd.read_csv(metadata_path)
    if "case_name" not in metadata_df.columns:
        raise ValueError(f"Metadata file '{metadata_path}' must contain a 'case_name' column.")

    nondim_records = nondimensionalize_records_from_metadata(
        cleaned_records,
        metadata_df,
        time_scale_mode="fn",
    )

    metadata_map = metadata_df.set_index("case_name").to_dict(orient="index")
    action_map = applied_df.set_index("case_name").to_dict(orient="index") if not applied_df.empty else {}
    dedup_map = dedup_df.set_index("case_name").to_dict(orient="index") if not dedup_df.empty else {}
    nondim_by_case = {record.case_name: record for record in nondim_records}

    output_dir.mkdir(parents=True, exist_ok=True)
    records_meta: list[dict[str, object]] = []
    for record_dim in cleaned_records:
        record_nd = nondim_by_case.get(record_dim.case_name)
        if record_nd is None:
            raise KeyError(f"Missing nondimensionalized record for case '{record_dim.case_name}'.")
        metadata_row = metadata_map.get(record_dim.case_name)
        if metadata_row is None:
            raise KeyError(f"Metadata file '{metadata_path}' is missing case '{record_dim.case_name}'.")
        payload, record_meta = _build_payload(
            record_dim,
            record_nd,
            metadata_row,
            action_row=action_map.get(record_dim.case_name),
            dedup_row=dedup_map.get(record_dim.case_name),
        )
        out_path = output_dir / f"{record_dim.case_name}.npz"
        if out_path.exists() and not bool(OVERWRITE):
            print(f"Skipping existing file: {out_path}")
            continue
        np.savez(out_path, **payload)
        records_meta.append(record_meta)
        print(f"Wrote {out_path}")

    metadata_out = output_dir / "metadata.json"
    metadata_out.write_text(json.dumps(records_meta, indent=2), encoding="utf-8")
    print(f"Wrote {metadata_out}")


if __name__ == "__main__":
    main()
