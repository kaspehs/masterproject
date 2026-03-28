from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

import numpy as np

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    tqdm = None

try:
    import analyze_vivana_td_burnin as burnin_config
    from td_hidden_state import (
        build_single_paramset_from_burnin_config,
        compute_force_spread_history,
        compute_theta_series,
        detect_burnin_start_index,
        initial_hidden_sigmas,
        initial_phi_dy,
        paramset_id,
        replay_hidden_state_with_cfd_motion,
        wrap_phase,
    )
except ModuleNotFoundError:
    try:
        from Data_Gen import analyze_vivana_td_burnin as burnin_config
        from Data_Gen.td_hidden_state import (
            build_single_paramset_from_burnin_config,
            compute_force_spread_history,
            compute_theta_series,
            detect_burnin_start_index,
            initial_hidden_sigmas,
            initial_phi_dy,
            paramset_id,
            replay_hidden_state_with_cfd_motion,
            wrap_phase,
        )
    except ModuleNotFoundError:
        from CFD_Data import analyze_vivana_td_burnin as burnin_config
        from CFD_Data.td_hidden_state import (
            build_single_paramset_from_burnin_config,
            compute_force_spread_history,
            compute_theta_series,
            detect_burnin_start_index,
            initial_hidden_sigmas,
            initial_phi_dy,
            paramset_id,
            replay_hidden_state_with_cfd_motion,
            wrap_phase,
        )


THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR if (THIS_DIR / "npz_exports").exists() else THIS_DIR.parent / "CFD_Data"


INPUT_NPZS: list[Path] | None = None
INPUT_NPZ_GLOB = str(DATA_DIR / "npz_exports" / "*.npz")
OUTPUT_DIR = DATA_DIR / "npz_exports_td_burnin_trimmed"
METADATA_PATH = DATA_DIR / "analysis" / "CFD_metadata.csv"
OVERWRITE = True
SKIP_NO_CONVERGENCE = True
FORCE_REL_STD_THRESHOLD = 1.0e-4
PERSISTENCE_SECONDS = 1.0
EXPORT_THETA0 = 0.0
SHOW_PROGRESS = True
MIN_OUTPUT_SAMPLES = 2
_METADATA_CACHE: dict[str, dict[str, str]] | None = None


def _progress(iterable, *, total: int | None = None, desc: str = ""):
    if SHOW_PROGRESS and tqdm is not None:
        return tqdm(iterable, total=total, desc=desc, leave=False)
    return iterable


def _resolve_input_npzs() -> list[Path]:
    if INPUT_NPZS is not None:
        paths = [Path(path).resolve() for path in INPUT_NPZS]
    else:
        paths = sorted(Path(path).resolve() for path in glob.glob(INPUT_NPZ_GLOB) if Path(path).suffix == ".npz")
    if not paths:
        raise FileNotFoundError("No CFD .npz files selected for TD burn-in preprocessing.")
    return paths


def _load_npz_payload(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=True) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _metadata_rows() -> dict[str, dict[str, str]]:
    global _METADATA_CACHE
    if _METADATA_CACHE is not None:
        return _METADATA_CACHE
    rows: dict[str, dict[str, str]] = {}
    if METADATA_PATH.exists():
        with METADATA_PATH.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                case_name = str(row.get("case_name", "")).strip()
                if case_name:
                    rows[case_name] = row
    _METADATA_CACHE = rows
    return rows


def _metadata_float(row: dict[str, str], key: str) -> float | None:
    raw = str(row.get(key, "")).strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if np.isfinite(value) else None


def _inject_python_metadata(payload: dict[str, np.ndarray], *, case_name: str) -> dict[str, np.ndarray]:
    row = _metadata_rows().get(str(case_name))
    if row is None:
        return payload

    augmented = {key: np.asarray(value).copy() for key, value in payload.items()}
    python_mass_kg = _metadata_float(row, "python_mass_kg")
    python_rho_kg_m3 = _metadata_float(row, "python_rho_kg_m3")
    python_diameter_m = _metadata_float(row, "python_diameter_m")
    python_ca = _metadata_float(row, "python_ca")
    python_effective_mass_kg = None
    if None not in (python_mass_kg, python_rho_kg_m3, python_diameter_m, python_ca):
        python_effective_mass_kg = float(
            python_mass_kg + 0.25 * np.pi * python_rho_kg_m3 * python_ca * python_diameter_m**2
        )

    numeric_fields = {
        "python_stiffness_n_m": _metadata_float(row, "python_stiffness_n_m"),
        "python_mass_kg": python_mass_kg,
        "python_dry_mass_kg": python_mass_kg,
        "python_effective_mass_kg": python_effective_mass_kg,
        "python_damping_c": _metadata_float(row, "python_damping_c"),
        "python_rho_kg_m3": python_rho_kg_m3,
        "python_flow_speed_m_s": _metadata_float(row, "python_flow_speed_m_s"),
        "python_diameter_m": python_diameter_m,
        "python_n_memory": _metadata_float(row, "python_n_memory"),
        "python_cv": _metadata_float(row, "python_cv"),
        "python_cd": _metadata_float(row, "python_cd"),
        "python_ca": python_ca,
        "python_fhat0": _metadata_float(row, "python_fhat0"),
        "python_fhat_min": _metadata_float(row, "python_fhat_min"),
        "python_fhat_max": _metadata_float(row, "python_fhat_max"),
        "python_dt_s": _metadata_float(row, "python_dt_s"),
    }
    for key, value in numeric_fields.items():
        if value is not None:
            augmented[key] = np.asarray(float(value), dtype=float)

    for key in ("python_condition_set", "python_primary_integrator"):
        raw = str(row.get(key, "")).strip()
        if raw:
            augmented[key] = np.asarray(raw)
    return augmented


def _time_aligned_keys(payload: dict[str, np.ndarray], n_samples: int) -> list[str]:
    keys: list[str] = []
    for key, value in payload.items():
        arr = np.asarray(value)
        if arr.ndim >= 1 and arr.shape[0] == int(n_samples):
            keys.append(key)
    return keys


def _sanitize_payload(payload: dict[str, np.ndarray], *, case_name: str) -> dict[str, np.ndarray]:
    required_arrays = ["time_dim", "y_disp_dim", "y_vel_dim", "y_acc_dim"]
    required_scalars = ["flow_speed_m_s", "rho_kg_m3", "diameter_m"]
    missing = [key for key in required_arrays + required_scalars if key not in payload]
    if missing:
        raise KeyError(f"CFD case '{case_name}' is missing required keys: {missing}")
    has_force_per_m = "y_force_per_m_dim" in payload
    has_force_total = "y_force_dim" in payload
    if not has_force_per_m and not has_force_total:
        raise KeyError(
            f"CFD case '{case_name}' is missing both 'y_force_per_m_dim' and 'y_force_dim'."
        )
    if not has_force_per_m and "span_m" not in payload:
        raise KeyError(
            f"CFD case '{case_name}' needs 'span_m' when only total force 'y_force_dim' is available."
        )

    time_dim = np.asarray(payload["time_dim"], dtype=float).reshape(-1)
    y_disp_dim = np.asarray(payload["y_disp_dim"], dtype=float).reshape(-1)
    y_vel_dim = np.asarray(payload["y_vel_dim"], dtype=float).reshape(-1)
    y_acc_dim = np.asarray(payload["y_acc_dim"], dtype=float).reshape(-1)
    if has_force_per_m:
        y_force_per_m_dim = np.asarray(payload["y_force_per_m_dim"], dtype=float).reshape(-1)
    else:
        span_m = float(np.asarray(payload["span_m"]).reshape(()))
        y_force_per_m_dim = np.asarray(payload["y_force_dim"], dtype=float).reshape(-1) / span_m
    if not (
        len(time_dim)
        == len(y_disp_dim)
        == len(y_vel_dim)
        == len(y_acc_dim)
        == len(y_force_per_m_dim)
    ):
        raise ValueError("Dimensional CFD channels must have the same length.")
    if len(time_dim) < 4:
        raise ValueError("Need at least 4 CFD samples for TD burn-in preprocessing.")

    sanitized = {key: np.asarray(value).copy() for key, value in payload.items()}
    sanitized["y_force_per_m_dim"] = np.asarray(y_force_per_m_dim, dtype=float)
    sanitized["force_per_m_dim"] = np.asarray(y_force_per_m_dim, dtype=float)
    keep_mask = np.ones(len(time_dim), dtype=bool)
    dup_mask = np.diff(time_dim) <= 0.0
    if np.any(dup_mask):
        if np.any(np.diff(time_dim) < 0.0):
            raise ValueError("CFD time vector must be nondecreasing.")
        keep_mask[1:] = ~dup_mask
        for key in _time_aligned_keys(sanitized, len(time_dim)):
            sanitized[key] = np.asarray(sanitized[key])[keep_mask]

    time_dim_clean = np.asarray(sanitized["time_dim"], dtype=float).reshape(-1)
    diffs = np.diff(time_dim_clean)
    if np.any(diffs <= 0.0):
        raise ValueError("CFD time vector must be strictly increasing after duplicate removal.")
    if diffs.size == 0:
        raise ValueError("Need at least two CFD samples after duplicate removal.")

    sanitized["dt_dim"] = np.asarray(float(np.median(diffs)), dtype=float)
    if "time_nd" in sanitized:
        time_nd = np.asarray(sanitized["time_nd"], dtype=float).reshape(-1)
        if len(time_nd) >= 2:
            sanitized["dt_nd"] = np.asarray(float(np.median(np.diff(time_nd))), dtype=float)
    sanitized["num_rows"] = np.asarray(int(time_dim_clean.size), dtype=int)
    return sanitized


def _trim_payload_arrays(payload: dict[str, np.ndarray], start_idx: int) -> dict[str, np.ndarray]:
    time_dim = np.asarray(payload["time_dim"], dtype=float).reshape(-1)
    n_samples = int(time_dim.size)
    trimmed: dict[str, np.ndarray] = {}
    for key, value in payload.items():
        arr = np.asarray(value)
        if arr.ndim >= 1 and arr.shape[0] == n_samples:
            trimmed[key] = arr[start_idx:].copy()
        else:
            trimmed[key] = arr.copy() if hasattr(arr, "copy") else np.asarray(arr)

    trimmed_time_dim = np.asarray(trimmed["time_dim"], dtype=float).reshape(-1)
    trimmed["num_rows"] = np.asarray(int(trimmed_time_dim.size), dtype=int)
    if trimmed_time_dim.size >= 2:
        trimmed["dt_dim"] = np.asarray(float(np.median(np.diff(trimmed_time_dim))), dtype=float)
    if "time_nd" in trimmed:
        time_nd = np.asarray(trimmed["time_nd"], dtype=float).reshape(-1)
        if time_nd.size >= 2:
            trimmed["dt_nd"] = np.asarray(float(np.median(np.diff(time_nd))), dtype=float)
    return trimmed


def _build_case_td_replay_inputs(
    payload: dict[str, np.ndarray],
    *,
    fallback_params: dict[str, float],
) -> tuple[dict[str, np.ndarray], dict[str, float], int]:
    replay_payload = {key: np.asarray(value).copy() for key, value in payload.items()}

    def _override_scalar(target_key: str, source_keys: tuple[str, ...]) -> None:
        value = _preferred_scalar(replay_payload, source_keys, default=None)
        if value is not None:
            replay_payload[target_key] = np.asarray(float(value), dtype=float)

    _override_scalar("flow_speed_m_s", ("python_flow_speed_m_s", "flow_speed_m_s"))
    _override_scalar("rho_kg_m3", ("python_rho_kg_m3", "rho_kg_m3"))
    _override_scalar("diameter_m", ("python_diameter_m", "diameter_m"))

    params = dict(fallback_params)
    for param_key, payload_keys in (
        ("Cv", ("python_cv",)),
        ("Cd", ("python_cd",)),
        ("Ca", ("python_ca",)),
        ("C", ("python_damping_c", "damping_c")),
        ("fhat_min", ("python_fhat_min",)),
        ("fhat0", ("python_fhat0",)),
        ("fhat_max", ("python_fhat_max",)),
    ):
        value = _preferred_scalar(replay_payload, payload_keys, default=None)
        if value is not None:
            params[param_key] = float(value)

    if not (params["fhat_min"] <= params["fhat0"] <= params["fhat_max"]):
        raise ValueError(
            "Invalid TD replay parameter set after metadata overrides: "
            f"{params['fhat_min']} <= {params['fhat0']} <= {params['fhat_max']} is required."
        )

    n_memory_value = _preferred_scalar(replay_payload, ("python_n_memory",), default=float(burnin_config.N_MEMORY))
    n_memory = max(1, int(round(float(n_memory_value))))
    return replay_payload, params, n_memory


def _representative_replay(
    payload: dict[str, np.ndarray],
    *,
    params: dict[str, float],
    theta0_export: float,
    n_memory: int,
) -> dict[str, np.ndarray]:
    flow_speed_m_s = float(np.asarray(payload["flow_speed_m_s"]).reshape(()))
    rho_kg_m3 = float(np.asarray(payload["rho_kg_m3"]).reshape(()))
    diameter_m = float(np.asarray(payload["diameter_m"]).reshape(()))
    sig_dy_loc0, sig_ddy_loc0 = initial_hidden_sigmas(
        case_like=payload,
        start_idx=0,
        flow_speed_m_s=flow_speed_m_s,
        n_memory=int(n_memory),
        mode=str(burnin_config.SIGMA_INIT_MODE),
        window_seconds=burnin_config.SIGMA_INIT_WINDOW_SECONDS,
    )
    phi_dy0 = initial_phi_dy(
        dy0=float(np.asarray(payload["y_vel_dim"], dtype=float)[0]),
        ddy0=float(np.asarray(payload["y_acc_dim"], dtype=float)[0]),
        sig_dy_loc0=sig_dy_loc0,
        sig_ddy_loc0=sig_ddy_loc0,
        flow_speed_m_s=flow_speed_m_s,
    )
    phi_vy0 = float(wrap_phase(np.asarray([phi_dy0 - float(theta0_export)]))[0])
    replay = replay_hidden_state_with_cfd_motion(
        time=np.asarray(payload["time_dim"], dtype=float),
        y=np.asarray(payload["y_disp_dim"], dtype=float),
        dy=np.asarray(payload["y_vel_dim"], dtype=float),
        ddy=np.asarray(payload["y_acc_dim"], dtype=float),
        flow_speed_m_s=flow_speed_m_s,
        rho_kg_m3=rho_kg_m3,
        diameter_m=diameter_m,
        params=params,
        phi_vy0=phi_vy0,
        sig_dy_loc0=sig_dy_loc0,
        sig_ddy_loc0=sig_ddy_loc0,
        n_memory=int(n_memory),
    )
    replay["theta_td"] = compute_theta_series(
        replay["dy"],
        replay["ddy"],
        replay["phi_vy"],
        replay["sig_dy_loc"],
        replay["sig_ddy_loc"],
        flow_speed_m_s=flow_speed_m_s,
        mode=str(burnin_config.PHASE_WRAP),
    )
    return replay


def _preferred_scalar(
    payload: dict[str, np.ndarray],
    keys: tuple[str, ...],
    default: float | None = None,
) -> float | None:
    for key in keys:
        if key in payload:
            value = float(np.asarray(payload[key]).reshape(()))
            if np.isfinite(value):
                return value
    if default is not None and np.isfinite(float(default)):
        return float(default)
    return None


def _manifest_row_base(npz_path: Path, payload: dict[str, np.ndarray], td_paramset_id: str) -> dict[str, object]:
    return {
        "case_name": npz_path.stem,
        "source_npz": str(npz_path),
        "output_npz": "",
        "status": "",
        "num_input_samples": int(np.asarray(payload["time_dim"]).reshape(-1).size),
        "num_output_samples": 0,
        "burnin_start_idx": None,
        "burnin_start_time_dim": None,
        "burnin_seconds_removed": None,
        "force_rel_std_threshold": float(FORCE_REL_STD_THRESHOLD),
        "persistence_seconds": float(PERSISTENCE_SECONDS),
        "force_rel_std_at_cut": None,
        "force_std_ref": None,
        "td_paramset_id": td_paramset_id,
        "theta0_export": float(EXPORT_THETA0),
        "num_theta0": int(np.asarray(burnin_config.THETA0_VALUES, dtype=float).size),
        "flow_speed_m_s": _preferred_scalar(
            payload,
            ("python_flow_speed_m_s", "flow_speed_m_s"),
        ),
        "dt_dim": float(np.asarray(payload["dt_dim"]).reshape(())),
        "stiffness_n_m": _preferred_scalar(
            payload,
            ("python_stiffness_n_m", "stiffness_n_m"),
        ),
        "effective_mass_kg": _preferred_scalar(
            payload,
            ("python_effective_mass_kg", "effective_mass_kg"),
        ),
        "dry_mass_kg": _preferred_scalar(
            payload,
            ("python_dry_mass_kg", "python_mass_kg", "dry_mass_kg"),
        ),
        "damping_c": _preferred_scalar(
            payload,
            ("python_damping_c", "damping_c"),
            default=float(build_single_paramset_from_burnin_config()["C"]),
        ),
        "error": "",
    }


def _write_manifest(rows: list[dict[str, object]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "burnin_manifest.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    csv_path = out_dir / "burnin_manifest.csv"
    fieldnames = [
        "case_name",
        "source_npz",
        "output_npz",
        "status",
        "num_input_samples",
        "num_output_samples",
        "burnin_start_idx",
        "burnin_start_time_dim",
        "burnin_seconds_removed",
        "force_rel_std_threshold",
        "persistence_seconds",
        "force_rel_std_at_cut",
        "force_std_ref",
        "td_paramset_id",
        "theta0_export",
        "num_theta0",
        "flow_speed_m_s",
        "dt_dim",
        "stiffness_n_m",
        "effective_mass_kg",
        "dry_mass_kg",
        "damping_c",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _prepare_output_payload(
    payload: dict[str, np.ndarray],
    representative: dict[str, np.ndarray],
    force_total_rel_std: np.ndarray,
    *,
    burnin_start_idx: int,
    burnin_start_time_dim: float,
    burnin_seconds_removed: float,
    force_std_ref: float,
    detection_status: str,
) -> dict[str, np.ndarray]:
    trimmed = _trim_payload_arrays(payload, burnin_start_idx)
    trimmed["phi_vy_td"] = np.asarray(representative["phi_vy"][burnin_start_idx:], dtype=float)
    trimmed["theta_td"] = np.asarray(representative["theta_td"][burnin_start_idx:], dtype=float)
    trimmed["sig_dy_loc_td"] = np.asarray(representative["sig_dy_loc"][burnin_start_idx:], dtype=float)
    trimmed["sig_ddy_loc_td"] = np.asarray(representative["sig_ddy_loc"][burnin_start_idx:], dtype=float)
    fy_td_per_m = np.asarray(representative["Fy"][burnin_start_idx:], dtype=float)
    f_total_td_per_m = np.asarray(representative["F_total"][burnin_start_idx:], dtype=float)
    fcv_td_per_m = np.asarray(representative["Fcv"][burnin_start_idx:], dtype=float)
    fdy_td_per_m = np.asarray(representative["Fdy"][burnin_start_idx:], dtype=float)
    fca_td_per_m = np.asarray(representative["Fca"][burnin_start_idx:], dtype=float)
    trimmed["Fy_td"] = np.asarray(fy_td_per_m, dtype=float)
    trimmed["F_total_td"] = np.asarray(f_total_td_per_m, dtype=float)
    trimmed["Fcv_td"] = np.asarray(fcv_td_per_m, dtype=float)
    trimmed["Fdy_td"] = np.asarray(fdy_td_per_m, dtype=float)
    trimmed["Fca_td"] = np.asarray(fca_td_per_m, dtype=float)
    trimmed["Fy_td_per_m"] = np.asarray(fy_td_per_m, dtype=float)
    trimmed["F_total_td_per_m"] = np.asarray(f_total_td_per_m, dtype=float)
    trimmed["Fcv_td_per_m"] = np.asarray(fcv_td_per_m, dtype=float)
    trimmed["Fdy_td_per_m"] = np.asarray(fdy_td_per_m, dtype=float)
    trimmed["Fca_td_per_m"] = np.asarray(fca_td_per_m, dtype=float)
    trimmed["force_total_rel_std_td"] = np.asarray(force_total_rel_std[burnin_start_idx:], dtype=float)
    trimmed["burnin_start_idx"] = np.asarray(int(burnin_start_idx), dtype=int)
    trimmed["burnin_start_time_dim"] = np.asarray(float(burnin_start_time_dim), dtype=float)
    trimmed["burnin_seconds_removed"] = np.asarray(float(burnin_seconds_removed), dtype=float)
    trimmed["burnin_force_rel_std_threshold"] = np.asarray(float(FORCE_REL_STD_THRESHOLD), dtype=float)
    trimmed["burnin_persistence_seconds"] = np.asarray(float(PERSISTENCE_SECONDS), dtype=float)
    trimmed["burnin_theta0_export"] = np.asarray(float(EXPORT_THETA0), dtype=float)
    trimmed["burnin_num_theta0"] = np.asarray(
        int(np.asarray(burnin_config.THETA0_VALUES, dtype=float).size),
        dtype=int,
    )
    trimmed["burnin_force_std_ref"] = np.asarray(float(force_std_ref), dtype=float)
    trimmed["burnin_detection_status"] = np.asarray(str(detection_status))
    trimmed["damping_c"] = np.asarray(
        _preferred_scalar(
            payload,
            ("python_damping_c", "damping_c"),
            default=float(build_single_paramset_from_burnin_config()["C"]),
        ),
        dtype=float,
    )
    trimmed["force_reference_convention_dim"] = np.asarray("per_unit_length", dtype="<U32")
    trimmed["force_td_convention_dim"] = np.asarray("per_unit_length", dtype="<U32")
    for key in (
        "span_m",
        "python_span_m",
        "physical_span_m",
        "raw_force_span_scale_applied",
        "x_force_dim",
        "y_force_dim",
        "Fy_td_total",
        "F_total_td_total",
        "Fcv_td_total",
        "Fdy_td_total",
        "Fca_td_total",
    ):
        trimmed.pop(key, None)
    return trimmed


def _process_single_file(npz_path: Path, *, params: dict[str, float]) -> dict[str, object]:
    raw_payload = _inject_python_metadata(_load_npz_payload(npz_path), case_name=npz_path.stem)
    payload = _sanitize_payload(raw_payload, case_name=npz_path.stem)
    replay_payload, replay_params, replay_n_memory = _build_case_td_replay_inputs(
        payload,
        fallback_params=params,
    )
    replay_paramset_id = paramset_id(replay_params)
    row = _manifest_row_base(npz_path, payload, replay_paramset_id)

    spread = compute_force_spread_history(
        case_payload=replay_payload,
        params=replay_params,
        theta0_values=np.asarray(burnin_config.THETA0_VALUES, dtype=float),
        sigma_init_mode=str(burnin_config.SIGMA_INIT_MODE),
        sigma_init_window_seconds=burnin_config.SIGMA_INIT_WINDOW_SECONDS,
        n_memory=int(replay_n_memory),
        progress=_progress if SHOW_PROGRESS else None,
        progress_desc=f"{npz_path.stem} theta0",
    )
    force_total_rel_std = np.asarray(spread["force_total_rel_std"], dtype=float)
    burnin_start_idx = detect_burnin_start_index(
        np.asarray(payload["time_dim"], dtype=float),
        force_total_rel_std,
        threshold=float(FORCE_REL_STD_THRESHOLD),
        persistence_seconds=float(PERSISTENCE_SECONDS),
    )
    row["force_std_ref"] = float(spread["force_std_ref"])

    if burnin_start_idx is None:
        row["status"] = "no_convergence"
        if SKIP_NO_CONVERGENCE:
            return row
        raise RuntimeError(f"No valid burn-in cut found for {npz_path.name}.")

    time_dim = np.asarray(payload["time_dim"], dtype=float)
    burnin_start_time_dim = float(time_dim[burnin_start_idx])
    burnin_seconds_removed = float(burnin_start_time_dim - float(time_dim[0]))
    row["burnin_start_idx"] = int(burnin_start_idx)
    row["burnin_start_time_dim"] = burnin_start_time_dim
    row["burnin_seconds_removed"] = burnin_seconds_removed
    row["force_rel_std_at_cut"] = float(force_total_rel_std[burnin_start_idx])

    if int(time_dim.size - burnin_start_idx) < int(MIN_OUTPUT_SAMPLES):
        row["status"] = "too_short_after_trim"
        return row

    representative = _representative_replay(
        replay_payload,
        params=replay_params,
        theta0_export=float(EXPORT_THETA0),
        n_memory=int(replay_n_memory),
    )
    output_payload = _prepare_output_payload(
        payload,
        representative,
        force_total_rel_std,
        burnin_start_idx=int(burnin_start_idx),
        burnin_start_time_dim=burnin_start_time_dim,
        burnin_seconds_removed=burnin_seconds_removed,
        force_std_ref=float(spread["force_std_ref"]),
        detection_status="ok",
    )

    output_path = OUTPUT_DIR.resolve() / npz_path.name
    if output_path.exists() and not bool(OVERWRITE):
        raise FileExistsError(f"Output file already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **output_payload)

    row["status"] = "ok"
    row["output_npz"] = str(output_path)
    row["num_output_samples"] = int(np.asarray(output_payload["time_dim"]).reshape(-1).size)
    return row


def main() -> None:
    params = build_single_paramset_from_burnin_config()
    td_paramset_id = paramset_id(params)
    rows: list[dict[str, object]] = []
    input_npzs = _resolve_input_npzs()

    for npz_path in _progress(input_npzs, total=len(input_npzs), desc="TD preprocess"):
        try:
            row = _process_single_file(npz_path, params=params)
        except Exception as exc:
            row = {
                "case_name": npz_path.stem,
                "source_npz": str(npz_path),
                "output_npz": "",
                "status": "invalid_input",
                "num_input_samples": 0,
                "num_output_samples": 0,
                "burnin_start_idx": None,
                "burnin_start_time_dim": None,
                "burnin_seconds_removed": None,
                "force_rel_std_threshold": float(FORCE_REL_STD_THRESHOLD),
                "persistence_seconds": float(PERSISTENCE_SECONDS),
                "force_rel_std_at_cut": None,
                "force_std_ref": None,
                "td_paramset_id": td_paramset_id,
                "theta0_export": float(EXPORT_THETA0),
                "num_theta0": int(np.asarray(burnin_config.THETA0_VALUES, dtype=float).size),
                "flow_speed_m_s": None,
                "dt_dim": None,
                "error": str(exc),
            }
            print(f"Skipping {npz_path.name}: {exc}")
        rows.append(row)

    _write_manifest(rows, OUTPUT_DIR.resolve())
    print(f"Wrote TD training-ready outputs to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
