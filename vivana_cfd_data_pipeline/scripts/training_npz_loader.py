from __future__ import annotations

import csv
import importlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np


DATA_PIPELINE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DATA_PIPELINE_ROOT.parent
DEFAULT_ORDERED_SPLIT_DIRS = ("train", "val")
DEFAULT_CFD_FORCE_TOTAL_SCALE = 4.0
DEFAULT_DATASET_ROOT_RELATIVE_PATHS = (
    Path("vivana_cfd_data_pipeline") / "generated" / "td_burnin_trimmed",
    Path("vivana_cfd_data_pipeline") / "generated" / "td_burnin_trimmed_v1",
    Path("vivana_cfd_data_pipeline") / "generated" / "td_burnin_trimmed_all",
    Path("vivana_cfd_data_pipeline") / "generated" / "td_burnin_trimmed_alltimeseries",
    Path("npz_exports_td_burnin_trimmed"),
    Path("npz_exports_td_burnin_trimmed_v1"),
    Path("npz_exports_td_burnin_trimmed_all"),
)
DEFAULT_CFD_METADATA_PATH = DATA_PIPELINE_ROOT / "metadata" / "CFD_metadata.csv"

_METADATA_CACHE: dict[str, dict[str, str]] | None = None


def _td_hidden_state_module():
    try:
        return importlib.import_module("vivana_cfd_data_pipeline.vivana_td.td_hidden_state")
    except ModuleNotFoundError:
        try:
            return importlib.import_module("td_hidden_state")
        except ModuleNotFoundError:
            return importlib.import_module("vivana_cfd_data_pipeline.vivana_td.td_hidden_state")


def dataset_root_candidates(
    cwd: str | Path,
    *,
    repo_root: str | Path | None = None,
    extra_candidates: Sequence[str | Path] | None = None,
) -> list[Path]:
    cwd_path = Path(cwd).resolve()
    base_candidates = [cwd_path, cwd_path.parent]
    if repo_root is not None:
        repo_root_path = Path(repo_root).resolve()
        base_candidates.extend([repo_root_path, repo_root_path.parent])
    candidates: list[Path] = []
    seen: set[Path] = set()
    for raw_path in extra_candidates or ():
        candidate = Path(raw_path).expanduser().resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        candidates.append(candidate)
    for base in base_candidates:
        for rel in DEFAULT_DATASET_ROOT_RELATIVE_PATHS:
            candidate = (base / rel).resolve()
            if candidate in seen:
                continue
            seen.add(candidate)
            candidates.append(candidate)
    return candidates


def resolve_dataset_root(
    cwd: str | Path,
    *,
    repo_root: str | Path | None = None,
    extra_candidates: Sequence[str | Path] | None = None,
) -> Path:
    candidates = dataset_root_candidates(cwd, repo_root=repo_root, extra_candidates=extra_candidates)
    for candidate in candidates:
        if not candidate.exists():
            continue
        if any((candidate / split).glob("*.npz") for split in DEFAULT_ORDERED_SPLIT_DIRS if (candidate / split).exists()):
            return candidate
        if any(candidate.glob("*.npz")):
            return candidate
    tried = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not locate dataset root. Tried:\n{tried}")


def _metadata_rows() -> dict[str, dict[str, str]]:
    global _METADATA_CACHE
    if _METADATA_CACHE is not None:
        return _METADATA_CACHE
    rows: dict[str, dict[str, str]] = {}
    if DEFAULT_CFD_METADATA_PATH.exists():
        with DEFAULT_CFD_METADATA_PATH.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                case_name = str(row.get("case_name", "")).strip()
                if case_name:
                    rows[case_name] = row
    _METADATA_CACHE = rows
    return rows


def _metadata_float(row: dict[str, str] | None, key: str) -> float | None:
    if not row:
        return None
    raw = str(row.get(key, "")).strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if np.isfinite(value) else None


def _extract_first_present(data: Any, keys: Sequence[str], *, path: Path, required: bool = True) -> np.ndarray | None:
    for key in keys:
        if key in data:
            return np.asarray(data[key])
    if required:
        raise KeyError(f"{path} is missing required keys {list(keys)}.")
    return None


def _preferred_scalar(data: Any, keys: Sequence[str], *, path: Path, required: bool = True) -> float | None:
    for key in keys:
        if key not in data:
            continue
        arr = np.asarray(data[key], dtype=float).reshape(-1)
        if arr.size == 1 and np.isfinite(arr[0]):
            return float(arr[0])
    if required:
        raise KeyError(f"{path} is missing a finite scalar for keys {list(keys)}.")
    return None


def _preferred_numeric_value(data: Any, keys: Sequence[str]) -> np.ndarray | None:
    for key in keys:
        if key not in data:
            continue
        arr = np.asarray(data[key], dtype=float)
        if np.any(np.isfinite(arr)):
            return arr
    return None


def _prepare_reduced_velocity_series(
    ur_raw: np.ndarray | float,
    length: int,
    *,
    name: str,
) -> np.ndarray:
    ur_arr = np.asarray(ur_raw, dtype=float)
    if ur_arr.ndim == 0:
        return np.full((length,), float(ur_arr), dtype=float)
    ur_flat = ur_arr.reshape(-1)
    ur_val = float(ur_flat[0])
    if not np.allclose(ur_flat, ur_val, rtol=1e-6, atol=1e-9):
        raise ValueError(f"{name} reduced velocity must be constant within a series.")
    return np.full((length,), ur_val, dtype=float)


def _resolve_force_span(path: Path, data: Any) -> float:
    span = _preferred_scalar(
        data,
        ("physical_span_m", "raw_force_span_scale_applied", "span_m", "python_span_m"),
        path=path,
        required=False,
    )
    if span is not None and span > 0.0:
        return float(span)
    metadata_row = _metadata_rows().get(path.stem)
    meta_span = _metadata_float(metadata_row, "span_m")
    if meta_span is not None and meta_span > 0.0:
        return float(meta_span)
    return float(DEFAULT_CFD_FORCE_TOTAL_SCALE)


def _resolve_vector(raw: np.ndarray | None, *, length: int, name: str) -> np.ndarray:
    if raw is None:
        return np.full((length,), np.nan, dtype=float)
    arr = np.asarray(raw, dtype=float)
    if arr.ndim == 0:
        return np.full((length,), float(arr), dtype=float)
    flat = arr.reshape(-1)
    if flat.shape[0] == 1:
        return np.full((length,), float(flat[0]), dtype=float)
    if flat.shape[0] != length:
        raise ValueError(f"{name} must be scalar or length-matched to time.")
    return flat


def _scalar_ur(ur_label_raw: np.ndarray | None, ur_raw: np.ndarray | None, *, length: int) -> float:
    for raw, name in ((ur_label_raw, "U_r_label"), (ur_raw, "U_r")):
        if raw is None:
            continue
        try:
            ur_arr = _prepare_reduced_velocity_series(raw, length, name=name)
        except ValueError:
            ur_arr = np.asarray(raw, dtype=float).reshape(-1)
        finite = ur_arr[np.isfinite(ur_arr)]
        if finite.size:
            return float(finite[0])
    return float("nan")


def _compute_effective_ur(flow_speed: np.ndarray, *, stiffness: float, effective_mass: float, diameter: float) -> float:
    flow_speed_arr = np.asarray(flow_speed, dtype=float).reshape(-1)
    finite_speed = flow_speed_arr[np.isfinite(flow_speed_arr)]
    if finite_speed.size == 0:
        return float("nan")
    if not (
        np.isfinite(stiffness)
        and stiffness > 0.0
        and np.isfinite(effective_mass)
        and effective_mass > 0.0
        and np.isfinite(diameter)
        and diameter > 0.0
    ):
        return float("nan")
    natural_frequency_hz = float(np.sqrt(stiffness / effective_mass) / (2.0 * np.pi))
    if not np.isfinite(natural_frequency_hz) or natural_frequency_hz <= 0.0:
        return float("nan")
    reference_speed = float(np.median(finite_speed))
    return float(reference_speed / (natural_frequency_hz * diameter))


def _reconstruct_td_channels(
    *,
    path: Path,
    data: Any,
    time: np.ndarray,
    displacement: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    force_total: np.ndarray,
    flow_speed: np.ndarray,
    rho: float,
    diameter: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    td_hidden = _td_hidden_state_module()
    cv = float(_preferred_scalar(data, ("python_cv",), path=path))
    cd = float(_preferred_scalar(data, ("python_cd",), path=path))
    ca = float(_preferred_scalar(data, ("python_ca",), path=path))
    damping_c = float(_preferred_scalar(data, ("python_damping_c",), path=path))
    fhat0 = float(_preferred_scalar(data, ("python_fhat0",), path=path))
    fhat_min = float(_preferred_scalar(data, ("python_fhat_min",), path=path))
    fhat_max = float(_preferred_scalar(data, ("python_fhat_max",), path=path))
    n_memory = int(round(float(_preferred_scalar(data, ("python_n_memory",), path=path))))
    if n_memory < 1:
        raise ValueError(f"{path} has invalid python_n_memory={n_memory}.")

    sig_dy_arr = _preferred_numeric_value(data, ("sig_dy_dim", "sig_dy_nd"))
    sig_ddy_arr = _preferred_numeric_value(data, ("sig_ddy_dim", "sig_ddy_nd"))
    if sig_dy_arr is None or sig_ddy_arr is None:
        raise KeyError(f"{path} is missing sigma channels needed to reconstruct TD state.")
    sig_dy = np.asarray(sig_dy_arr, dtype=float).reshape(-1)
    sig_ddy = np.asarray(sig_ddy_arr, dtype=float).reshape(-1)
    if sig_dy.size != time.size or sig_ddy.size != time.size:
        raise ValueError(f"{path} has mismatched sigma channel lengths for TD reconstruction.")

    finite_flow = flow_speed[np.isfinite(flow_speed)]
    flow_speed_scalar = float(np.median(finite_flow)) if finite_flow.size else float("nan")
    if not np.isfinite(flow_speed_scalar):
        raise ValueError(f"{path} is missing a finite flow speed for TD reconstruction.")

    phi_dy0 = float(
        td_hidden.initial_phi_dy(
            dy0=float(velocity[0]),
            ddy0=float(acceleration[0]),
            sig_dy_loc0=float(sig_dy[0]),
            sig_ddy_loc0=float(sig_ddy[0]),
            flow_speed_m_s=flow_speed_scalar,
        )
    )
    phi_vy0 = float(td_hidden.wrap_phase(np.asarray([phi_dy0], dtype=float))[0])
    replay = td_hidden.replay_hidden_state_with_cfd_motion(
        time=np.asarray(time, dtype=float),
        y=np.asarray(displacement, dtype=float),
        dy=np.asarray(velocity, dtype=float),
        ddy=np.asarray(acceleration, dtype=float),
        flow_speed_m_s=flow_speed_scalar,
        rho_kg_m3=float(rho),
        diameter_m=float(diameter),
        params={
            "Cv": cv,
            "Cd": cd,
            "Ca": ca,
            "C": damping_c,
            "fhat0": fhat0,
            "fhat_min": fhat_min,
            "fhat_max": fhat_max,
        },
        phi_vy0=phi_vy0,
        sig_dy_loc0=float(sig_dy[0]),
        sig_ddy_loc0=float(sig_ddy[0]),
        n_memory=n_memory,
    )
    phi_td = np.asarray(replay["phi_vy"], dtype=float).reshape(-1)
    sig_dy_td = np.asarray(replay["sig_dy_loc"], dtype=float).reshape(-1)
    sig_ddy_td = np.asarray(replay["sig_ddy_loc"], dtype=float).reshape(-1)
    force_td_stored = np.asarray(replay.get("F_total", force_total), dtype=float).reshape(-1)
    if any(arr.size != time.size for arr in (phi_td, sig_dy_td, sig_ddy_td, force_td_stored)):
        raise ValueError(f"{path} reconstructed TD channels do not match the CFD series length.")
    return force_td_stored, phi_td, sig_dy_td, sig_ddy_td


def load_series(npz_path: str | Path) -> dict[str, Any]:
    path = Path(npz_path)
    with np.load(path, allow_pickle=True) as data:
        time = np.asarray(_extract_first_present(data, ("time_dim", "a", "time"), path=path), dtype=float).reshape(-1)
        displacement = np.asarray(_extract_first_present(data, ("y_disp_dim", "b", "y"), path=path), dtype=float).reshape(-1)
        velocity = np.asarray(_extract_first_present(data, ("y_vel_dim", "dy", "e", "v"), path=path), dtype=float).reshape(-1)
        acceleration = np.asarray(_extract_first_present(data, ("y_acc_dim", "ddy"), path=path), dtype=float).reshape(-1)

        span = _resolve_force_span(path, data)
        force_per_m_raw = _preferred_numeric_value(data, ("y_force_per_m_dim", "force_per_m_dim"))
        force_total_raw = _preferred_numeric_value(data, ("y_force_dim", "c", "F_total", "force_total", "force"))
        if force_per_m_raw is None and force_total_raw is None:
            raise KeyError(f"{path} is missing CFD force channels for loading.")
        if force_per_m_raw is None:
            force_total = np.asarray(force_total_raw, dtype=float).reshape(-1)
            force_per_m = force_total / float(span)
        else:
            force_per_m = np.asarray(force_per_m_raw, dtype=float).reshape(-1)
            force_total = force_per_m * float(span)

        stiffness = float(
            _preferred_scalar(
                data,
                ("python_stiffness_n_m", "model_stiffness_n_m", "training_stiffness_n_m", "stiffness_n_m"),
                path=path,
            )
        )
        effective_mass = float(
            _preferred_scalar(
                data,
                ("python_effective_mass_kg", "model_effective_mass_kg", "training_effective_mass_kg", "effective_mass_kg"),
                path=path,
            )
        )
        dry_mass = float(
            _preferred_scalar(
                data,
                ("python_dry_mass_kg", "python_mass_kg", "model_dry_mass_kg", "training_dry_mass_kg", "dry_mass_kg"),
                path=path,
            )
        )
        damping = float(
            _preferred_scalar(
                data,
                ("python_damping_c", "model_damping_c", "training_damping_c", "damping_c"),
                path=path,
            )
        )
        rho = float(_preferred_scalar(data, ("python_rho_kg_m3", "rho_kg_m3"), path=path))
        diameter = float(_preferred_scalar(data, ("python_diameter_m", "diameter_m"), path=path))

        flow_speed = _resolve_vector(
            _preferred_numeric_value(data, ("python_flow_speed_m_s", "model_flow_speed_m_s", "training_flow_speed_m_s", "flow_speed_m_s")),
            length=time.size,
            name=f"{path} flow_speed_m_s",
        )
        if not np.all(np.isfinite(flow_speed)):
            finite_flow = flow_speed[np.isfinite(flow_speed)]
            fill_value = float(np.mean(finite_flow)) if finite_flow.size else float("nan")
            flow_speed = np.full((time.size,), fill_value, dtype=float)

        ur = _scalar_ur(
            _extract_first_present(data, ("U_r_label_series", "U_r_label_scalar"), path=path, required=False),
            _extract_first_present(data, ("U_r_computed_series", "U_r"), path=path, required=False),
            length=time.size,
        )

        force_td_per_m_raw = _preferred_numeric_value(data, ("F_total_td_per_m", "F_total_td"))
        force_td_total_raw = _preferred_numeric_value(data, ("F_total_td_total",))
        phi_td_raw = _extract_first_present(data, ("phi_vy_td",), path=path, required=False)
        sig_dy_td_raw = _extract_first_present(data, ("sig_dy_loc_td",), path=path, required=False)
        sig_ddy_td_raw = _extract_first_present(data, ("sig_ddy_loc_td",), path=path, required=False)
        if (
            force_td_per_m_raw is None
            and force_td_total_raw is None
            and phi_td_raw is None
            and sig_dy_td_raw is None
            and sig_ddy_td_raw is None
        ):
            force_td_stored, phi_td, sig_dy_td, sig_ddy_td = _reconstruct_td_channels(
                path=path,
                data=data,
                time=time,
                displacement=displacement,
                velocity=velocity,
                acceleration=acceleration,
                force_total=force_per_m,
                flow_speed=flow_speed,
                rho=rho,
                diameter=diameter,
            )
        else:
            if force_td_per_m_raw is None and force_td_total_raw is None:
                raise KeyError(f"{path} is missing TD force channels for loading.")
            if force_td_per_m_raw is None:
                force_td_stored = np.asarray(force_td_total_raw, dtype=float).reshape(-1) / float(span)
            else:
                force_td_stored = np.asarray(force_td_per_m_raw, dtype=float).reshape(-1)
            phi_td = np.asarray(_extract_first_present(data, ("phi_vy_td",), path=path), dtype=float).reshape(-1)
            sig_dy_td = np.asarray(_extract_first_present(data, ("sig_dy_loc_td",), path=path), dtype=float).reshape(-1)
            sig_ddy_td = np.asarray(_extract_first_present(data, ("sig_ddy_loc_td",), path=path), dtype=float).reshape(-1)

    expected = [
        displacement,
        velocity,
        acceleration,
        force_total,
        force_per_m,
        force_td_stored,
        phi_td,
        sig_dy_td,
        sig_ddy_td,
        flow_speed,
    ]
    if any(arr.shape[0] != time.size for arr in expected):
        raise ValueError(f"{path} has mismatched time-aligned array lengths.")
    if time.size < 2:
        raise ValueError(f"{path} is too short to simulate a rollout.")
    dt = float(time[1] - time[0])
    if not np.allclose(np.diff(time), dt, rtol=1e-6, atol=1e-9):
        raise ValueError(f"{path} time vector is not uniform.")

    td_context = np.stack([acceleration, phi_td, sig_dy_td, sig_ddy_td, flow_speed], axis=1)
    ur_effective = _compute_effective_ur(
        flow_speed,
        stiffness=stiffness,
        effective_mass=effective_mass,
        diameter=diameter,
    )
    return {
        "name": path.stem,
        "path": path,
        "time": time,
        "displacement": displacement,
        "velocity": velocity,
        "acceleration": acceleration,
        "force_total": force_total,
        "force_per_m": force_per_m,
        "force_td_stored": force_td_stored,
        "td_context": td_context,
        "rho": rho,
        "diameter": diameter,
        "stiffness": stiffness,
        "effective_mass": effective_mass,
        "dry_mass": dry_mass,
        "damping": damping,
        "span": span,
        "ur": ur,
        "ur_effective": ur_effective,
    }


def _limit_files(files: list[Path], *, max_files_per_split: int | None) -> list[Path]:
    if max_files_per_split is None:
        return files
    return files[: int(max_files_per_split)]


def iter_npz_files(
    root: str | Path,
    split: str,
    *,
    split_dirs: Sequence[str] = DEFAULT_ORDERED_SPLIT_DIRS,
    max_files_per_split: int | None = None,
) -> list[Path]:
    root_path = Path(root)
    split_dir = root_path / split
    if split_dir.exists():
        return _limit_files(sorted(split_dir.glob("*.npz")), max_files_per_split=max_files_per_split)
    root_files = _limit_files(sorted(root_path.glob("*.npz")), max_files_per_split=max_files_per_split)
    if root_files:
        return root_files if split == split_dirs[0] else []
    raise FileNotFoundError(f"Missing split directory: {split_dir}")


def iter_all_npz_files(
    root: str | Path,
    *,
    split_dirs: Sequence[str] = DEFAULT_ORDERED_SPLIT_DIRS,
    max_files_per_split: int | None = None,
) -> list[Path]:
    root_path = Path(root)
    files: list[Path] = []
    for split in split_dirs:
        split_dir = root_path / split
        if not split_dir.exists():
            continue
        files.extend(_limit_files(sorted(split_dir.glob("*.npz")), max_files_per_split=max_files_per_split))
    if files:
        return files
    return _limit_files(sorted(root_path.glob("*.npz")), max_files_per_split=max_files_per_split)
