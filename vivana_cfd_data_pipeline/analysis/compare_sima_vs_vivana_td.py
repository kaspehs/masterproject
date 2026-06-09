from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

try:
    from vivana_td_model import (
        DEFAULT_PARAMS as SHARED_DEFAULT_PARAMS,
        PYTHON_INTEGRATORS as SHARED_PYTHON_INTEGRATORS,
        VALID_INTEGRATORS as SHARED_VALID_INTEGRATORS,
        build_phase_burn_in_state_from_displacement,
        compute_theta_series as shared_compute_theta_series,
        replay_prescribed_motion as shared_replay_prescribed_motion,
        run_simulation as shared_run_simulation,
        standardize_rollout as shared_standardize_rollout,
        wrap_phase as shared_wrap_phase,
    )
except ModuleNotFoundError:
    from vivana_cfd_data_pipeline.vivana_td.vivana_td_model import (
        DEFAULT_PARAMS as SHARED_DEFAULT_PARAMS,
        PYTHON_INTEGRATORS as SHARED_PYTHON_INTEGRATORS,
        VALID_INTEGRATORS as SHARED_VALID_INTEGRATORS,
        build_phase_burn_in_state_from_displacement,
        compute_theta_series as shared_compute_theta_series,
        replay_prescribed_motion as shared_replay_prescribed_motion,
        run_simulation as shared_run_simulation,
        standardize_rollout as shared_standardize_rollout,
        wrap_phase as shared_wrap_phase,
    )


# ---------------------------------------------------------------------------
# User config: edit these values before running the script.
# ---------------------------------------------------------------------------

DATA_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = DATA_ROOT / "outputs" / "analysis" / "sima_vs_vivana_td"
OVERWRITE = True
SIMA_H5_PATH = DATA_ROOT / "VIVANAresults_validation.h5"
SIMA_CONDITIONSET_ROOT = "RIGID_Cylinder_changed/ConditionSet"
SIMA_SEGMENT_NAME = "segment_1"
SIMA_DIRECTION = "z"
SIMA_NODE_SELECTION = "center"  # one of: center, max_std, explicit
SIMA_NODE_INDEX = 1             # used when SIMA_NODE_SELECTION == "explicit"
SIMA_ELEMENT_SELECTION = "center"  # one of: center, explicit
SIMA_ELEMENT_INDEX = 1             # used when SIMA_ELEMENT_SELECTION == "explicit"
PRINT_H5_LAYOUT = False

# Case selection.
ANALYZE_ALL_CONDITIONSETS = True
CASES = [
    {
        "condition_set": "ConditionSet_6",
        "case_name": "ConditionSet_6",
    },
]

# Simulation window.
SIM_START_TIME_S = 0.0
HIDDEN_STATE_BURNIN_SECONDS = 100.0
SIM_DURATION_S: float | None = 60.0

DEFAULT_PARAMS = dict(SHARED_DEFAULT_PARAMS)

PYTHON_STIFFNESS_BY_RUN = {
    "ConditionSet_1": 118.44 / 4.0,
    "ConditionSet_2": 29.61 / 4.0,
    "ConditionSet_3": 18.95 / 4.0,
    "ConditionSet_4": 14.33 / 4.0,
    "ConditionSet_5": 9.67 / 4.0,
    "ConditionSet_6": 7.4 / 4.0,
    "ConditionSet_7": 4.74 / 4.0,
}

PRIMARY_INTEGRATOR = "rk4_coupled"
PYTHON_INTEGRATORS = SHARED_PYTHON_INTEGRATORS
VALID_INTEGRATORS = SHARED_VALID_INTEGRATORS


@dataclass
class SimaCaseData:
    case_name: str
    h5_path: Path
    condition_set: str
    node_index: int
    element_index: int
    element_length_m: float
    variables: dict[str, float]
    dt: float
    time_full: np.ndarray
    y_full: np.ndarray
    dy_full: np.ndarray
    ddy_full: np.ndarray
    start_window_idx: int
    compare_start_idx: int
    end_idx: int

    @property
    def time(self) -> np.ndarray:
        return self.time_full[self.compare_start_idx : self.end_idx]

    @property
    def y(self) -> np.ndarray:
        return self.y_full[self.compare_start_idx : self.end_idx]

    @property
    def dy(self) -> np.ndarray:
        return self.dy_full[self.compare_start_idx : self.end_idx]

    @property
    def ddy(self) -> np.ndarray:
        return self.ddy_full[self.compare_start_idx : self.end_idx]


def _coerce_float(value: object) -> float:
    if value is None:
        return float("nan")
    try:
        return float(np.asarray(value).reshape(()))
    except (TypeError, ValueError):
        text = str(value).strip()
        if not text:
            return float("nan")
        try:
            return float(text)
        except ValueError:
            return float("nan")


def _sanitize_name(name: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(name))


def _ensure_output_dir(path: Path) -> None:
    if path.exists() and not OVERWRITE:
        raise FileExistsError(f"Output directory already exists and OVERWRITE=False: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _iter_h5_layout(handle: h5py.File) -> list[str]:
    entries: list[str] = []

    def _visitor(name: str, obj: object) -> None:
        if isinstance(obj, h5py.Dataset):
            shape = "x".join(str(v) for v in obj.shape)
            entries.append(f"{name} [{shape}] dtype={obj.dtype}")

    handle.visititems(_visitor)
    return entries


def _condition_set_root_path(handle: h5py.File) -> str:
    configured = str(SIMA_CONDITIONSET_ROOT).strip("/")
    if configured in handle and isinstance(handle[configured], h5py.Group):
        return configured

    matches: list[str] = []

    def _visitor(name: str, obj: object) -> None:
        if not isinstance(obj, h5py.Group):
            return
        child_names = [str(key) for key in obj.keys()]
        if any(child_name.startswith("ConditionSet_") for child_name in child_names):
            matches.append(str(name).strip("/"))

    handle.visititems(_visitor)
    if not matches:
        raise KeyError(
            f"Could not find a SIMA condition-set root in '{handle.filename}'. "
            f"Configured path '{SIMA_CONDITIONSET_ROOT}' is missing."
        )

    preferred = [path for path in matches if path.endswith("/ConditionSet") or path == "ConditionSet"]
    candidates = preferred or matches
    if len(candidates) > 1:
        joined = ", ".join(sorted(candidates))
        raise KeyError(
            f"Multiple candidate SIMA condition-set roots found in '{handle.filename}': {joined}. "
            f"Set SIMA_CONDITIONSET_ROOT explicitly."
        )
    return candidates[0]


def _condition_set_names(handle: h5py.File) -> list[str]:
    root = handle[_condition_set_root_path(handle)]
    return sorted(name for name in root.keys() if str(name).startswith("ConditionSet_"))


def _condition_set_group(handle: h5py.File, condition_set: str) -> h5py.Group:
    return handle[f"{_condition_set_root_path(handle)}/{condition_set}"]


def _condition_set_variables(handle: h5py.File, condition_set: str) -> dict[str, float]:
    group = _condition_set_group(handle, condition_set)["Variables"]
    values: dict[str, float] = {}
    for key in group.keys():
        obj = group[key]
        if isinstance(obj, h5py.Dataset):
            values[str(key)] = float(np.asarray(obj).reshape(()))
    return values


def _node_numbers(handle: h5py.File, condition_set: str) -> list[int]:
    segment = _condition_set_group(handle, condition_set)[f"Dynamic/Cylinder/{SIMA_SEGMENT_NAME}"]
    nodes: list[int] = []
    for key in segment.keys():
        text = str(key)
        if text.startswith("node_"):
            try:
                nodes.append(int(text.split("_")[1]))
            except ValueError:
                continue
    return sorted(nodes)


def _element_numbers(handle: h5py.File, condition_set: str) -> list[int]:
    segment = _condition_set_group(handle, condition_set)[f"Dynamic/Cylinder/{SIMA_SEGMENT_NAME}"]
    elements: list[int] = []
    for key in segment.keys():
        text = str(key)
        if text.startswith("element_"):
            try:
                elements.append(int(text.split("_")[1]))
            except ValueError:
                continue
    return sorted(elements)


def _mid_node_number(handle: h5py.File, condition_set: str) -> int:
    nodes = _node_numbers(handle, condition_set)
    if not nodes:
        raise ValueError(f"No node_* groups found in SIMA condition set '{condition_set}'.")
    return int(nodes[len(nodes) // 2])


def _mid_element_number(handle: h5py.File, condition_set: str) -> int:
    elements = _element_numbers(handle, condition_set)
    if not elements:
        raise ValueError(f"No element_* groups found in SIMA condition set '{condition_set}'.")
    return int(elements[len(elements) // 2])


def _resolve_node_index(handle: h5py.File, condition_set: str) -> int:
    mode = str(SIMA_NODE_SELECTION).strip().lower()
    if mode == "explicit":
        return int(SIMA_NODE_INDEX)
    if mode == "center":
        return _mid_node_number(handle, condition_set)
    if mode == "max_std":
        best_node = None
        best_std = -float("inf")
        for node_index in _node_numbers(handle, condition_set):
            values = _read_h5_vector(
                handle,
                _node_displacement_path(condition_set, node_index, SIMA_DIRECTION),
                label=f"node_{node_index}_{SIMA_DIRECTION}_displacement",
            )
            std = float(np.std(values))
            if std > best_std:
                best_std = std
                best_node = node_index
        if best_node is None:
            raise ValueError(f"Could not resolve a valid displacement node in '{condition_set}'.")
        return int(best_node)
    raise ValueError("SIMA_NODE_SELECTION must be one of: center, max_std, explicit.")


def _resolve_element_index(handle: h5py.File, condition_set: str) -> int:
    mode = str(SIMA_ELEMENT_SELECTION).strip().lower()
    if mode == "explicit":
        return int(SIMA_ELEMENT_INDEX)
    if mode == "center":
        return _mid_element_number(handle, condition_set)
    raise ValueError("SIMA_ELEMENT_SELECTION must be one of: center, explicit.")


def _node_displacement_path(condition_set: str, node_index: int, direction: str) -> str:
    axis = str(direction).strip().lower()
    return (
        f"{SIMA_CONDITIONSET_ROOT}/{condition_set}/Dynamic/Cylinder/{SIMA_SEGMENT_NAME}/"
        f"node_{int(node_index)}/Displacement in {axis} - direction"
    )


def _dynamic_element_base_path(condition_set: str, element_index: int) -> str:
    return (
        f"{SIMA_CONDITIONSET_ROOT}/{condition_set}/Dynamic/Cylinder/{SIMA_SEGMENT_NAME}/"
        f"element_{int(element_index)}"
    )


def _static_end_path(condition_set: str, element_index: int, end: int) -> str:
    return (
        f"{SIMA_CONDITIONSET_ROOT}/{condition_set}/Static/Cylinder/{SIMA_SEGMENT_NAME}/"
        f"element_{int(element_index)}/end {int(end)}"
    )


def _resolve_h5_dataset(handle: h5py.File, dataset_path: str) -> h5py.Dataset:
    path_text = str(dataset_path).strip().lstrip("/")
    candidates = [path_text]
    configured_root = str(SIMA_CONDITIONSET_ROOT).strip("/")
    resolved_root = _condition_set_root_path(handle)
    if resolved_root != configured_root and path_text.startswith(configured_root + "/"):
        candidates.append(resolved_root + path_text[len(configured_root) :])

    for candidate in candidates:
        if candidate in handle:
            obj = handle[candidate]
            if isinstance(obj, h5py.Dataset):
                return obj

    matches: list[h5py.Dataset] = []

    def _visitor(name: str, obj: object) -> None:
        if isinstance(obj, h5py.Dataset) and (
            name == path_text or name.endswith("/" + path_text)
        ):
            matches.append(obj)

    handle.visititems(_visitor)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise KeyError(f"HDF5 dataset path '{dataset_path}' is ambiguous.")
    raise KeyError(f"HDF5 dataset path '{dataset_path}' was not found.")


def _read_h5_vector(handle: h5py.File, dataset_path: str, *, label: str) -> np.ndarray:
    values = np.asarray(_resolve_h5_dataset(handle, dataset_path), dtype=float)
    values = np.squeeze(values)
    if values.ndim != 1:
        raise ValueError(
            f"HDF5 dataset '{dataset_path}' for '{label}' must be 1D after squeeze, got shape {values.shape}."
        )
    return np.asarray(values, dtype=float)


def _center_signal(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr
    return arr - float(np.mean(arr))


def _sanitize_series(time: np.ndarray, *arrays: np.ndarray | None) -> tuple[np.ndarray, list[np.ndarray | None]]:
    if time.ndim != 1:
        raise ValueError("Time series must be 1D.")
    if time.size < 2:
        raise ValueError("Need at least two time samples.")
    order = np.argsort(time)
    time = np.asarray(time[order], dtype=float)
    sanitized: list[np.ndarray | None] = []
    for array in arrays:
        if array is None:
            sanitized.append(None)
            continue
        arr = np.asarray(array, dtype=float)
        if arr.ndim != 1 or arr.shape[0] != time.shape[0]:
            raise ValueError("All SIMA channels must be 1D and length-matched to time.")
        sanitized.append(arr[order])

    finite_mask = np.isfinite(time)
    for arr in sanitized:
        if arr is not None:
            finite_mask &= np.isfinite(arr)
    if int(np.count_nonzero(finite_mask)) < 2:
        raise ValueError("SIMA data does not contain at least two finite rows.")

    time = time[finite_mask]
    sanitized = [None if arr is None else arr[finite_mask] for arr in sanitized]

    keep_mask = np.ones(time.shape[0], dtype=bool)
    keep_mask[1:] = np.diff(time) > 0.0
    if int(np.count_nonzero(keep_mask)) < 2:
        raise ValueError("SIMA time vector became too short after duplicate removal.")

    time = time[keep_mask]
    sanitized = [None if arr is None else arr[keep_mask] for arr in sanitized]
    return time, sanitized


def _element_length_m(handle: h5py.File, condition_set: str, element_index: int) -> float:
    arc_1 = float(
        np.asarray(
            _resolve_h5_dataset(handle, f"{_static_end_path(condition_set, element_index, 1)}/Arc length")
        ).reshape(())
    )
    arc_2 = float(
        np.asarray(
            _resolve_h5_dataset(handle, f"{_static_end_path(condition_set, element_index, 2)}/Arc length")
        ).reshape(())
    )
    length = float(arc_2 - arc_1)
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError(f"Element length for {condition_set} element_{element_index} is not positive.")
    return length


def _selected_cases(handle: h5py.File) -> list[dict[str, str]]:
    if ANALYZE_ALL_CONDITIONSETS:
        return [{"condition_set": condition_set, "case_name": condition_set} for condition_set in _condition_set_names(handle)]
    if not CASES:
        raise ValueError("CASES is empty. Add at least one entry or enable ANALYZE_ALL_CONDITIONSETS.")

    selected: list[dict[str, str]] = []
    for raw_case in CASES:
        condition_set = str(raw_case.get("condition_set") or raw_case.get("case_name") or "").strip()
        if not condition_set:
            raise ValueError("Each CASES entry must contain 'condition_set' or 'case_name'.")
        case_name = str(raw_case.get("case_name") or condition_set).strip()
        selected.append({"condition_set": condition_set, "case_name": case_name})
    return selected


def _load_sima_case(handle: h5py.File, h5_path: Path, case_cfg: dict[str, str]) -> SimaCaseData:
    condition_set = str(case_cfg["condition_set"])
    case_name = str(case_cfg.get("case_name", condition_set))
    variables = _condition_set_variables(handle, condition_set)
    dt = float(variables.get("dt", DEFAULT_PARAMS["dt"]))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"Condition set '{condition_set}' does not contain a positive dt.")

    node_index = _resolve_node_index(handle, condition_set)
    element_index = _resolve_element_index(handle, condition_set)
    element_length_m = _element_length_m(handle, condition_set, element_index)

    y = _read_h5_vector(
        handle,
        _node_displacement_path(condition_set, node_index, SIMA_DIRECTION),
        label="y_disp",
    )
    y = _center_signal(y)
    time = np.arange(y.shape[0], dtype=float) * dt
    grad_edge_order = 2 if time.shape[0] >= 3 else 1
    dy = np.gradient(y, time, edge_order=grad_edge_order)
    ddy = np.gradient(dy, time, edge_order=grad_edge_order)

    time, sanitized = _sanitize_series(time, y, dy, ddy)
    y, dy, ddy = sanitized
    assert y is not None
    assert dy is not None
    assert ddy is not None

    start_window_idx = int(np.searchsorted(time, float(SIM_START_TIME_S), side="left"))
    compare_start_time = float(SIM_START_TIME_S) + max(0.0, float(HIDDEN_STATE_BURNIN_SECONDS))
    compare_start_idx = int(np.searchsorted(time, compare_start_time, side="left"))
    end_time = float(time[-1]) if SIM_DURATION_S is None else float(compare_start_time + float(SIM_DURATION_S))
    end_idx = int(np.searchsorted(time, end_time, side="right"))
    end_idx = max(end_idx, compare_start_idx + 2)

    if compare_start_idx >= time.shape[0] - 1:
        raise ValueError(f"Case '{case_name}' starts after the available SIMA time span.")
    if compare_start_idx <= start_window_idx:
        raise ValueError(
            f"Case '{case_name}' does not contain enough samples for "
            f"HIDDEN_STATE_BURNIN_SECONDS={HIDDEN_STATE_BURNIN_SECONDS}."
        )
    end_idx = min(end_idx, time.shape[0])
    if end_idx - compare_start_idx < 2:
        raise ValueError(f"Case '{case_name}' became too short after applying the requested time window.")

    return SimaCaseData(
        case_name=case_name,
        h5_path=h5_path,
        condition_set=condition_set,
        node_index=node_index,
        element_index=element_index,
        element_length_m=element_length_m,
        variables=variables,
        dt=dt,
        time_full=time,
        y_full=y,
        dy_full=dy,
        ddy_full=ddy,
        start_window_idx=start_window_idx,
        compare_start_idx=compare_start_idx,
        end_idx=end_idx,
    )


def _read_scalar_dataset(handle: h5py.File, dataset_path: str) -> float:
    return float(np.asarray(_resolve_h5_dataset(handle, dataset_path)).reshape(()))


def _read_force_channel_end_average(
    handle: h5py.File,
    condition_set: str,
    element_index: int,
    dataset_name: str,
) -> np.ndarray:
    base = _dynamic_element_base_path(condition_set, element_index)
    end_1 = _read_h5_vector(
        handle,
        f"{base}/{dataset_name}, global {SIMA_DIRECTION.lower()} direction, end 1",
        label=f"{dataset_name}_end_1",
    )
    end_2 = _read_h5_vector(
        handle,
        f"{base}/{dataset_name}, global {SIMA_DIRECTION.lower()} direction, end 2",
        label=f"{dataset_name}_end_2",
    )
    return 0.5 * (end_1 + end_2)


def _compute_sima_added_mass_force_per_m(params: dict[str, float | str], ddy: np.ndarray) -> np.ndarray:
    return (
        -0.25
        * float(params["rho"])
        * float(params["Ca"])
        * np.pi
        * float(params["D"]) ** 2
        * np.asarray(ddy, dtype=float)
    )


def _compute_python_drag_force_per_m(params: dict[str, float | str], dy: np.ndarray) -> np.ndarray:
    dy_arr = np.asarray(dy, dtype=float)
    speed_mag = np.sqrt(float(params["U"]) ** 2 + dy_arr ** 2)
    return -0.5 * float(params["rho"]) * float(params["D"]) * float(params["Cd"]) * speed_mag * dy_arr


def _load_sima_force_rollout(
    handle: h5py.File,
    sima: SimaCaseData,
    primary_params: dict[str, float | str],
) -> dict[str, np.ndarray]:
    length = float(sima.element_length_m)
    hydro_per_m_full = _read_force_channel_end_average(handle, sima.condition_set, sima.element_index, "Hydrodynamic load")
    morison_per_m_full = _read_force_channel_end_average(handle, sima.condition_set, sima.element_index, "Morison loads")
    cross_flow_per_m_full = _read_force_channel_end_average(handle, sima.condition_set, sima.element_index, "Cross-flow loads")

    start = int(sima.compare_start_idx)
    end = int(sima.end_idx)
    added_mass_per_m = _compute_sima_added_mass_force_per_m(primary_params, sima.ddy)

    hydro_per_m = np.asarray(hydro_per_m_full[start:end], dtype=float)
    morison_per_m = np.asarray(morison_per_m_full[start:end], dtype=float)
    cross_flow_per_m = np.asarray(cross_flow_per_m_full[start:end], dtype=float)

    return {
        "time": np.asarray(sima.time, dtype=float),
        "element_length_m": np.asarray(length, dtype=float),
        "hydrodynamic_total": hydro_per_m,
        "hydrodynamic_total_per_m": hydro_per_m,
        "hydrodynamic_total_raw": hydro_per_m * length,
        "morison_total": morison_per_m,
        "morison_total_per_m": morison_per_m,
        "morison_total_raw": morison_per_m * length,
        "cross_flow_total": cross_flow_per_m,
        "cross_flow_total_per_m": cross_flow_per_m,
        "cross_flow_total_raw": cross_flow_per_m * length,
        "added_mass_per_m": np.asarray(added_mass_per_m, dtype=float),
        "added_mass_raw": np.asarray(added_mass_per_m, dtype=float) * length,
    }


def _python_stiffness_for_condition(condition_set: str) -> float:
    override = PYTHON_STIFFNESS_BY_RUN.get(str(condition_set))
    if override is not None:
        return float(override)
    return float(DEFAULT_PARAMS["K"])


def _build_python_params(sima: SimaCaseData) -> dict[str, float | str]:
    params: dict[str, float | str] = dict(DEFAULT_PARAMS)
    mapping = {
        "M": "M",
        "U": "U",
        "D": "D",
        "cv": "Cv",
        "cd": "Cd",
        "ca": "Ca",
        "fnull": "fhat0",
        "fmin": "fhat_min",
        "fmax": "fhat_max",
    }
    for h5_name, py_name in mapping.items():
        if h5_name in sima.variables:
            params[py_name] = float(sima.variables[h5_name])

    params["dt"] = float(sima.dt)
    params["T"] = max(float(sima.dt), float(sima.time.shape[0]) * float(sima.dt))
    params["K"] = float(_python_stiffness_for_condition(sima.condition_set))
    return params


def _natural_frequency_hz(params: dict[str, float | str]) -> float:
    mass = float(params["M"])
    stiffness = float(params["K"])
    if not np.isfinite(mass) or mass <= 0.0 or not np.isfinite(stiffness) or stiffness <= 0.0:
        return float("nan")
    return float(np.sqrt(stiffness / mass) / (2.0 * np.pi))


def _reduced_velocity(params: dict[str, float | str]) -> float:
    mass = float(params["M"])
    rho = float(params["rho"])
    diameter = float(params["D"])
    stiffness = float(params["K"])
    velocity = float(params["U"])
    if (
        not np.isfinite(mass)
        or mass <= 0.0
        or not np.isfinite(rho)
        or rho <= 0.0
        or not np.isfinite(diameter)
        or diameter <= 0.0
        or not np.isfinite(stiffness)
        or stiffness <= 0.0
    ):
        return float("nan")
    return float(2.0 * np.pi * velocity / diameter * np.sqrt((mass + 0.25 * np.pi * rho * diameter**2) / stiffness))


def _wrap_phase(values: np.ndarray | float) -> np.ndarray:
    return shared_wrap_phase(values)


def _circular_mean(values: np.ndarray) -> float:
    values_arr = np.asarray(values, dtype=float).reshape(-1)
    values_arr = values_arr[np.isfinite(values_arr)]
    if values_arr.size == 0:
        return float("nan")
    resultant = np.mean(np.exp(1j * values_arr))
    if not np.isfinite(resultant.real) or not np.isfinite(resultant.imag):
        return float("nan")
    return float(np.angle(resultant))


def _compute_theta_series(
    dy: np.ndarray,
    ddy: np.ndarray,
    phi_vy: np.ndarray,
    sig_dy_loc: np.ndarray,
    sig_ddy_loc: np.ndarray,
    flow_speed: float,
) -> np.ndarray:
    return shared_compute_theta_series(
        dy=dy,
        ddy=ddy,
        phi_vy=phi_vy,
        sig_dy_loc=sig_dy_loc,
        sig_ddy_loc=sig_ddy_loc,
        flow_speed=flow_speed,
    )


def _run_simulation(
    params: dict[str, float | str] | None = None,
    initial_state: dict[str, float] | None = None,
) -> dict[str, np.ndarray | dict[str, float | str]]:
    return shared_run_simulation(params=params, initial_state=initial_state)


def _build_phase_burn_in_state(sima: SimaCaseData, python_params: dict[str, float | str]) -> dict[str, float]:
    return build_phase_burn_in_state_from_displacement(
        time_full=sima.time_full,
        y_full=sima.y_full,
        dt=float(sima.dt),
        start_window_idx=int(sima.start_window_idx),
        compare_start_idx=int(sima.compare_start_idx),
        end_idx=int(sima.end_idx),
        python_params=python_params,
    )


def _replay_prescribed_motion(
    *,
    time: np.ndarray,
    y: np.ndarray,
    dy: np.ndarray,
    ddy: np.ndarray,
    params: dict[str, float | str],
    phi_vy0: float,
    sig_dy_loc0: float,
    sig_ddy_loc0: float,
) -> dict[str, np.ndarray]:
    return shared_replay_prescribed_motion(
        time=time,
        y=y,
        dy=dy,
        ddy=ddy,
        params=params,
        phi_vy0=phi_vy0,
        sig_dy_loc0=sig_dy_loc0,
        sig_ddy_loc0=sig_ddy_loc0,
        relative_time=False,
    )


def _standardize_rollout(
    raw_result: dict[str, np.ndarray | dict[str, float | str]],
) -> dict[str, np.ndarray | dict[str, float | str] | str]:
    return shared_standardize_rollout(raw_result)


def _run_case_rollouts(sima: SimaCaseData, python_params: dict[str, float | str]) -> dict[str, dict[str, np.ndarray | dict[str, float | str] | str]]:
    initial_state = _build_phase_burn_in_state(sima, python_params)
    rollouts: dict[str, dict[str, np.ndarray | dict[str, float | str] | str]] = {}
    for integrator_key, _ in PYTHON_INTEGRATORS:
        run_result = _run_simulation(params={**python_params, "integrator": integrator_key}, initial_state=initial_state)
        rollouts[integrator_key] = _standardize_rollout(run_result)
    return rollouts


def _resample_uniform(time: np.ndarray, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    time_arr = np.asarray(time, dtype=float).reshape(-1)
    signal_arr = np.asarray(signal, dtype=float).reshape(-1)
    if time_arr.shape[0] != signal_arr.shape[0]:
        raise ValueError("Time and signal must have the same length for resampling.")
    if time_arr.shape[0] < 2:
        return time_arr, signal_arr
    dt = float(np.median(np.diff(time_arr)))
    if not np.isfinite(dt) or dt <= 0.0:
        return time_arr, signal_arr
    uniform_time = np.arange(float(time_arr[0]), float(time_arr[-1]) + 0.5 * dt, dt, dtype=float)
    uniform_signal = np.interp(uniform_time, time_arr, signal_arr)
    return uniform_time, uniform_signal


def _power_spectrum(time: np.ndarray, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    uniform_time, uniform_signal = _resample_uniform(time, signal)
    if uniform_time.shape[0] < 8:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    dt = float(np.median(np.diff(uniform_time)))
    centered = uniform_signal - float(np.mean(uniform_signal))
    if np.allclose(centered, 0.0):
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    nfft = 1 << int(np.ceil(np.log2(max(8, centered.shape[0]))))
    freqs = np.fft.rfftfreq(nfft, d=dt)
    psd = np.abs(np.fft.rfft(centered, n=nfft)) ** 2
    return np.asarray(freqs, dtype=float), np.asarray(psd, dtype=float)


def _dominant_frequency_hz(time: np.ndarray, signal: np.ndarray) -> float:
    freqs, psd = _power_spectrum(time, signal)
    if freqs.size <= 1 or psd.size <= 1:
        return float("nan")
    psd = psd.copy()
    psd[0] = 0.0
    idx = int(np.argmax(psd))
    if not np.isfinite(psd[idx]) or psd[idx] <= 0.0:
        return float("nan")
    return float(freqs[idx])


def _relative_error(value: float, reference: float) -> float:
    if not np.isfinite(reference) or abs(reference) <= np.finfo(float).eps:
        return float("nan")
    return float((value - reference) / abs(reference))


def _series_correlation(pred: np.ndarray, true: np.ndarray) -> float:
    pred_arr = np.asarray(pred, dtype=float).reshape(-1)
    true_arr = np.asarray(true, dtype=float).reshape(-1)
    if pred_arr.shape != true_arr.shape or pred_arr.size < 2:
        return float("nan")
    pred_std = float(np.std(pred_arr))
    true_std = float(np.std(true_arr))
    if pred_std <= np.finfo(float).eps or true_std <= np.finfo(float).eps:
        return float("nan")
    return float(np.corrcoef(pred_arr, true_arr)[0, 1])


def _interp_to_time(source_time: np.ndarray, source_signal: np.ndarray, target_time: np.ndarray) -> np.ndarray:
    return np.interp(
        np.asarray(target_time, dtype=float),
        np.asarray(source_time, dtype=float),
        np.asarray(source_signal, dtype=float),
    )


def _spectral_relative_error(true_time: np.ndarray, pred_time: np.ndarray, pred: np.ndarray, true: np.ndarray) -> float:
    freq_pred, psd_pred = _power_spectrum(pred_time, pred)
    freq_true, psd_true = _power_spectrum(true_time, true)
    if freq_pred.size == 0 or freq_true.size == 0:
        return float("nan")
    common_freqs = freq_true
    psd_pred_interp = np.interp(common_freqs, freq_pred, psd_pred, left=0.0, right=0.0)
    denom = float(np.linalg.norm(psd_true))
    if not np.isfinite(denom) or denom <= np.finfo(float).eps:
        return float("nan")
    return float(np.linalg.norm(psd_pred_interp - psd_true) / denom)


def _rmse(pred: np.ndarray, true: np.ndarray) -> float:
    pred_arr = np.asarray(pred, dtype=float)
    true_arr = np.asarray(true, dtype=float)
    if pred_arr.shape != true_arr.shape:
        raise ValueError("RMSE inputs must have the same shape.")
    return float(np.sqrt(np.mean((pred_arr - true_arr) ** 2)))


def _std_relative_error(pred: np.ndarray, true: np.ndarray) -> float:
    return _relative_error(float(np.std(pred)), float(np.std(true)))


def _nrmse_std(pred: np.ndarray, true: np.ndarray) -> float:
    denom = float(np.std(true))
    if not np.isfinite(denom) or denom <= np.finfo(float).eps:
        return float("nan")
    return float(_rmse(pred, true) / denom)


def _evaluate_rollout(
    *,
    prefix: str,
    sima_time: np.ndarray,
    y_true: np.ndarray,
    rollout: dict[str, np.ndarray | dict[str, float | str] | str],
) -> dict[str, float]:
    rollout_time = np.asarray(rollout["time"], dtype=float)
    y_pred = np.asarray(rollout["y"], dtype=float)
    y_true_interp = _interp_to_time(sima_time, y_true, rollout_time)
    result = {
        f"{prefix}_disp_rmse": _rmse(y_pred, y_true_interp),
        f"{prefix}_disp_nrmse_std": _nrmse_std(y_pred, y_true_interp),
        f"{prefix}_disp_std_rel_error": _std_relative_error(y_pred, y_true_interp),
        f"{prefix}_disp_dominant_freq_true_hz": _dominant_frequency_hz(sima_time, y_true),
        f"{prefix}_disp_dominant_freq_pred_hz": _dominant_frequency_hz(rollout_time, y_pred),
        f"{prefix}_disp_spectral_rel_error": _spectral_relative_error(sima_time, rollout_time, y_pred, y_true),
    }
    result[f"{prefix}_disp_dominant_freq_abs_error_hz"] = float(
        abs(float(result[f"{prefix}_disp_dominant_freq_pred_hz"]) - float(result[f"{prefix}_disp_dominant_freq_true_hz"]))
    )
    result[f"{prefix}_disp_dominant_freq_rel_error"] = _relative_error(
        float(result[f"{prefix}_disp_dominant_freq_pred_hz"]),
        float(result[f"{prefix}_disp_dominant_freq_true_hz"]),
    )
    return result


def _evaluate_force_component(
    *,
    prefix: str,
    sima_time: np.ndarray,
    sima_force: np.ndarray,
    rollout_time: np.ndarray,
    pred_force: np.ndarray,
) -> dict[str, float]:
    sima_interp = _interp_to_time(sima_time, sima_force, rollout_time)
    result = {
        f"{prefix}_rmse": _rmse(pred_force, sima_interp),
        f"{prefix}_nrmse_std": _nrmse_std(pred_force, sima_interp),
        f"{prefix}_std_true": float(np.std(sima_interp)),
        f"{prefix}_std_pred": float(np.std(pred_force)),
        f"{prefix}_std_rel_error": _std_relative_error(pred_force, sima_interp),
        f"{prefix}_mean_true": float(np.mean(sima_interp)),
        f"{prefix}_mean_pred": float(np.mean(pred_force)),
        f"{prefix}_corr": _series_correlation(pred_force, sima_interp),
        f"{prefix}_dominant_freq_true_hz": _dominant_frequency_hz(sima_time, sima_force),
        f"{prefix}_dominant_freq_pred_hz": _dominant_frequency_hz(rollout_time, pred_force),
        f"{prefix}_spectral_rel_error": _spectral_relative_error(sima_time, rollout_time, pred_force, sima_force),
    }
    result[f"{prefix}_dominant_freq_abs_error_hz"] = float(
        abs(float(result[f"{prefix}_dominant_freq_pred_hz"]) - float(result[f"{prefix}_dominant_freq_true_hz"]))
    )
    result[f"{prefix}_dominant_freq_rel_error"] = _relative_error(
        float(result[f"{prefix}_dominant_freq_pred_hz"]),
        float(result[f"{prefix}_dominant_freq_true_hz"]),
    )
    return result


def _evaluate_force_rollout(
    *,
    prefix: str,
    sima_forces: dict[str, np.ndarray],
    rollout: dict[str, np.ndarray | dict[str, float | str] | str],
    element_length_m: float,
) -> dict[str, float]:
    rollout_time = np.asarray(rollout["time"], dtype=float)
    pred_total = np.asarray(rollout["force_total_compare"], dtype=float) / float(element_length_m)
    pred_cross_flow = np.asarray(rollout["force_cv"], dtype=float) / float(element_length_m)
    pred_drag = np.asarray(rollout["force_drag"], dtype=float) / float(element_length_m)
    return {
        **_evaluate_force_component(
            prefix=f"{prefix}_force_total",
            sima_time=np.asarray(sima_forces["time"], dtype=float),
            sima_force=np.asarray(sima_forces["hydrodynamic_total_per_m"], dtype=float),
            rollout_time=rollout_time,
            pred_force=pred_total,
        ),
        **_evaluate_force_component(
            prefix=f"{prefix}_force_cross_flow",
            sima_time=np.asarray(sima_forces["time"], dtype=float),
            sima_force=np.asarray(sima_forces["cross_flow_total_per_m"], dtype=float),
            rollout_time=rollout_time,
            pred_force=pred_cross_flow,
        ),
        **_evaluate_force_component(
            prefix=f"{prefix}_force_morison",
            sima_time=np.asarray(sima_forces["time"], dtype=float),
            sima_force=np.asarray(sima_forces["morison_total_per_m"], dtype=float),
            rollout_time=rollout_time,
            pred_force=pred_drag,
        ),
    }


def _apply_primary_aliases(result: dict[str, float | str]) -> None:
    source_prefix = f"{PRIMARY_INTEGRATOR}_"
    for key, value in list(result.items()):
        if key.startswith(source_prefix):
            result["primary_" + key[len(source_prefix) :]] = value


def _evaluate_case(
    sima: SimaCaseData,
    sima_forces: dict[str, np.ndarray],
    rollouts: dict[str, dict[str, np.ndarray | dict[str, float | str] | str]],
    python_params: dict[str, float | str],
) -> dict[str, float | str]:
    sima_time = np.asarray(sima.time, dtype=float)
    y_true = np.asarray(sima.y, dtype=float)
    h5_stiffness = float(sima.variables.get("K", float("nan")))
    row: dict[str, float | str] = {
        "case_name": sima.case_name,
        "h5_path": str(sima.h5_path),
        "condition_set": sima.condition_set,
        "node_index": float(sima.node_index),
        "element_index": float(sima.element_index),
        "element_length_m": float(sima.element_length_m),
        "hidden_state_burnin_seconds": float(HIDDEN_STATE_BURNIN_SECONDS),
        "primary_integrator": PRIMARY_INTEGRATOR,
        "h5_stiffness_n_m": h5_stiffness,
        "python_stiffness_n_m": float(python_params["K"]),
        "python_mass": float(python_params["M"]),
        "python_damping_c": float(python_params["C"]),
        "python_rho": float(python_params["rho"]),
        "python_flow_speed_m_s": float(python_params["U"]),
        "python_diameter_m": float(python_params["D"]),
        "python_cv": float(python_params["Cv"]),
        "python_cd": float(python_params["Cd"]),
        "python_ca": float(python_params["Ca"]),
        "python_fhat0": float(python_params["fhat0"]),
        "python_fhat_min": float(python_params["fhat_min"]),
        "python_fhat_max": float(python_params["fhat_max"]),
        "python_n_memory": float(python_params["n_memory"]),
        "ur_value": _reduced_velocity(python_params),
        "f_n_hz": _natural_frequency_hz(python_params),
        "dt_mean": float(np.mean(np.diff(sima_time))),
        "duration_s": float(sima_time[-1] - sima_time[0]),
    }

    for integrator_key, _ in PYTHON_INTEGRATORS:
        rollout = rollouts[integrator_key]
        row[f"{integrator_key}_theta_mean_rad"] = _circular_mean(np.asarray(rollout["theta"], dtype=float))
        row.update(_evaluate_rollout(prefix=integrator_key, sima_time=sima_time, y_true=y_true, rollout=rollout))
        row.update(
            _evaluate_force_rollout(
                prefix=integrator_key,
                sima_forces=sima_forces,
                rollout=rollout,
                element_length_m=float(sima.element_length_m),
            )
        )

    _apply_primary_aliases(row)
    return row


def _save_case_npz(
    case_dir: Path,
    sima: SimaCaseData,
    sima_forces: dict[str, np.ndarray],
    rollouts: dict[str, dict[str, np.ndarray | dict[str, float | str] | str]],
    python_params: dict[str, float | str],
) -> None:
    data: dict[str, np.ndarray] = {
        "time": np.asarray(sima.time, dtype=float),
        "sima_y": np.asarray(sima.y, dtype=float),
        "sima_dy": np.asarray(sima.dy, dtype=float),
        "sima_ddy": np.asarray(sima.ddy, dtype=float),
        "sima_force_time": np.asarray(sima_forces["time"], dtype=float),
        "sima_force_total": np.asarray(sima_forces["hydrodynamic_total_per_m"], dtype=float),
        "sima_force_total_raw": np.asarray(sima_forces["hydrodynamic_total_raw"], dtype=float),
        "sima_force_cross_flow": np.asarray(sima_forces["cross_flow_total_per_m"], dtype=float),
        "sima_force_cross_flow_raw": np.asarray(sima_forces["cross_flow_total_raw"], dtype=float),
        "sima_force_morison": np.asarray(sima_forces["morison_total_per_m"], dtype=float),
        "sima_force_morison_raw": np.asarray(sima_forces["morison_total_raw"], dtype=float),
        "sima_force_added_mass": np.asarray(sima_forces["added_mass_per_m"], dtype=float),
        "sima_force_added_mass_raw": np.asarray(sima_forces["added_mass_raw"], dtype=float),
        "python_drag_on_sima_motion": _compute_python_drag_force_per_m(python_params, sima.dy),
        "element_length_m": np.asarray(float(sima.element_length_m), dtype=float),
        "h5_stiffness_n_m": np.asarray(float(sima.variables.get("K", float("nan"))), dtype=float),
        "python_stiffness_n_m": np.asarray(float(python_params["K"]), dtype=float),
        "python_mass": np.asarray(float(python_params["M"]), dtype=float),
        "python_flow_speed_m_s": np.asarray(float(python_params["U"]), dtype=float),
        "python_rho": np.asarray(float(python_params["rho"]), dtype=float),
        "python_diameter_m": np.asarray(float(python_params["D"]), dtype=float),
        "python_cv": np.asarray(float(python_params["Cv"]), dtype=float),
        "python_cd": np.asarray(float(python_params["Cd"]), dtype=float),
        "python_ca": np.asarray(float(python_params["Ca"]), dtype=float),
        "python_fhat0": np.asarray(float(python_params["fhat0"]), dtype=float),
        "python_fhat_min": np.asarray(float(python_params["fhat_min"]), dtype=float),
        "python_fhat_max": np.asarray(float(python_params["fhat_max"]), dtype=float),
        "python_n_memory": np.asarray(int(python_params["n_memory"]), dtype=int),
        "condition_set": np.asarray(sima.condition_set),
        "case_name": np.asarray(sima.case_name),
        "node_index": np.asarray(sima.node_index, dtype=int),
        "element_index": np.asarray(sima.element_index, dtype=int),
        "primary_integrator": np.asarray(PRIMARY_INTEGRATOR),
        "ur_value": np.asarray(_reduced_velocity(python_params), dtype=float),
        "f_n_hz": np.asarray(_natural_frequency_hz(python_params), dtype=float),
    }

    primary = rollouts[PRIMARY_INTEGRATOR]
    data.update(
        {
            "primary_time": np.asarray(primary["time"], dtype=float),
            "primary_y": np.asarray(primary["y"], dtype=float),
            "primary_dy": np.asarray(primary["dy"], dtype=float),
            "primary_ddy": np.asarray(primary["ddy"], dtype=float),
            "primary_force_total": np.asarray(primary["force_total"], dtype=float),
            "primary_force_total_compare": np.asarray(primary["force_total_compare"], dtype=float),
            "primary_force_total_per_m": np.asarray(primary["force_total"], dtype=float) / float(sima.element_length_m),
            "primary_force_total_compare_per_m": np.asarray(primary["force_total_compare"], dtype=float)
            / float(sima.element_length_m),
            "primary_force_cv": np.asarray(primary["force_cv"], dtype=float),
            "primary_force_cv_per_m": np.asarray(primary["force_cv"], dtype=float) / float(sima.element_length_m),
            "primary_force_drag": np.asarray(primary["force_drag"], dtype=float),
            "primary_force_drag_per_m": np.asarray(primary["force_drag"], dtype=float) / float(sima.element_length_m),
            "primary_force_added_mass": np.asarray(primary["force_added_mass"], dtype=float),
            "primary_force_added_mass_per_m": np.asarray(primary["force_added_mass"], dtype=float) / float(sima.element_length_m),
            "primary_phi_vy": np.asarray(primary["phi_vy"], dtype=float),
            "primary_sig_dy": np.asarray(primary["sig_dy"], dtype=float),
            "primary_sig_ddy": np.asarray(primary["sig_ddy"], dtype=float),
            "primary_theta": np.asarray(primary["theta"], dtype=float),
        }
    )

    for integrator_key, _ in PYTHON_INTEGRATORS:
        rollout = rollouts[integrator_key]
        data.update(
            {
                f"{integrator_key}_time": np.asarray(rollout["time"], dtype=float),
                f"{integrator_key}_y": np.asarray(rollout["y"], dtype=float),
                f"{integrator_key}_dy": np.asarray(rollout["dy"], dtype=float),
                f"{integrator_key}_ddy": np.asarray(rollout["ddy"], dtype=float),
                f"{integrator_key}_force_total": np.asarray(rollout["force_total"], dtype=float),
                f"{integrator_key}_force_total_compare": np.asarray(rollout["force_total_compare"], dtype=float),
                f"{integrator_key}_force_total_per_m": np.asarray(rollout["force_total"], dtype=float)
                / float(sima.element_length_m),
                f"{integrator_key}_force_total_compare_per_m": np.asarray(rollout["force_total_compare"], dtype=float)
                / float(sima.element_length_m),
                f"{integrator_key}_force_cv": np.asarray(rollout["force_cv"], dtype=float),
                f"{integrator_key}_force_cv_per_m": np.asarray(rollout["force_cv"], dtype=float)
                / float(sima.element_length_m),
                f"{integrator_key}_force_drag": np.asarray(rollout["force_drag"], dtype=float),
                f"{integrator_key}_force_drag_per_m": np.asarray(rollout["force_drag"], dtype=float)
                / float(sima.element_length_m),
                f"{integrator_key}_force_added_mass": np.asarray(rollout["force_added_mass"], dtype=float),
                f"{integrator_key}_force_added_mass_per_m": np.asarray(rollout["force_added_mass"], dtype=float)
                / float(sima.element_length_m),
                f"{integrator_key}_phi_vy": np.asarray(rollout["phi_vy"], dtype=float),
                f"{integrator_key}_sig_dy": np.asarray(rollout["sig_dy"], dtype=float),
                f"{integrator_key}_sig_ddy": np.asarray(rollout["sig_ddy"], dtype=float),
                f"{integrator_key}_theta": np.asarray(rollout["theta"], dtype=float),
            }
        )

    np.savez(case_dir / "comparison_timeseries.npz", **data)


def _save_case_plot(
    case_dir: Path,
    sima: SimaCaseData,
    primary_rollout: dict[str, np.ndarray | dict[str, float | str] | str],
    summary_row: dict[str, float | str],
) -> None:
    time = np.asarray(sima.time, dtype=float)
    y_true = np.asarray(sima.y, dtype=float)
    pred_time = np.asarray(primary_rollout["time"], dtype=float)
    y_pred = np.asarray(primary_rollout["y"], dtype=float)

    disp_freq_true, disp_psd_true = _power_spectrum(time, y_true)
    disp_freq_pred, disp_psd_pred = _power_spectrum(pred_time, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"{sima.case_name} | U_r={float(summary_row['ur_value']):.6g} | "
        f"{sima.condition_set} | node_{sima.node_index} {SIMA_DIRECTION} | "
        f"burn-in={float(HIDDEN_STATE_BURNIN_SECONDS):.0f}s | primary={PRIMARY_INTEGRATOR}\n"
        f"f_true={float(summary_row['primary_disp_dominant_freq_true_hz']):.6g} Hz | "
        f"f_pred={float(summary_row['primary_disp_dominant_freq_pred_hz']):.6g} Hz | "
        f"|df|={float(summary_row['primary_disp_dominant_freq_abs_error_hz']):.6g} Hz | "
        f"rel={float(summary_row['primary_disp_dominant_freq_rel_error']):.6g}"
    )

    ax = axes[0]
    ax.plot(time, y_true, label="SIMA", linewidth=1.5)
    ax.plot(pred_time, y_pred, label=f"Python {PRIMARY_INTEGRATOR}", linewidth=1.2)
    ax.set_title("Displacement")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(SIMA_DIRECTION)
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    if disp_freq_true.size > 0:
        ax.semilogy(disp_freq_true, np.maximum(disp_psd_true, 1.0e-16), label="SIMA", linewidth=1.5)
    if disp_freq_pred.size > 0:
        ax.semilogy(
            disp_freq_pred,
            np.maximum(disp_psd_pred, 1.0e-16),
            label=f"Python {PRIMARY_INTEGRATOR}",
            linewidth=1.2,
        )
    ax.set_title("Displacement Spectrum")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(case_dir / "comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_force_plot(
    case_dir: Path,
    sima: SimaCaseData,
    sima_forces: dict[str, np.ndarray],
    primary_rollout: dict[str, np.ndarray | dict[str, float | str] | str],
    python_params: dict[str, float | str],
) -> None:
    sima_time = np.asarray(sima_forces["time"], dtype=float)
    sima_total = np.asarray(sima_forces["hydrodynamic_total_per_m"], dtype=float)
    sima_cross_flow = np.asarray(sima_forces["cross_flow_total_per_m"], dtype=float)
    sima_morison = np.asarray(sima_forces["morison_total_per_m"], dtype=float)
    sima_added_mass = np.asarray(sima_forces["added_mass_per_m"], dtype=float)

    pred_time = np.asarray(primary_rollout["time"], dtype=float)
    pred_total = np.asarray(primary_rollout["force_total_compare"], dtype=float) / float(sima.element_length_m)
    pred_cross_flow = np.asarray(primary_rollout["force_cv"], dtype=float) / float(sima.element_length_m)
    pred_drag = np.asarray(primary_rollout["force_drag"], dtype=float) / float(sima.element_length_m)
    pred_added_mass = np.asarray(primary_rollout["force_added_mass"], dtype=float) / float(sima.element_length_m)
    python_drag_on_sima_motion = _compute_python_drag_force_per_m(python_params, sima.dy)

    fig, axes = plt.subplots(5, 1, figsize=(16, 18), sharex=True)
    fig.suptitle(
        f"{sima.case_name} force diagnostics | U_r={_reduced_velocity(python_params):.6g} | "
        f"{sima.condition_set} | element_{sima.element_index} {SIMA_DIRECTION} | primary={PRIMARY_INTEGRATOR}"
    )

    ax = axes[0]
    ax.plot(sima_time, sima_total, label="SIMA total", linewidth=1.5)
    ax.plot(pred_time, pred_total, label=f"Python {PRIMARY_INTEGRATOR} Fcv+Fdy (/m)", linewidth=1.1)
    ax.set_title("Total force comparison: SIMA vs Python (Fcv + Fdy) per m")
    ax.set_ylabel("Force / m")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(sima_time, sima_morison, label="SIMA Morison", linewidth=1.5)
    ax.plot(pred_time, pred_drag, label=f"Python {PRIMARY_INTEGRATOR} drag (/m)", linewidth=1.1)
    ax.plot(sima_time, python_drag_on_sima_motion, label="Python drag on SIMA motion", linewidth=1.0, alpha=0.9)
    ax.set_title("Morison/drag force: SIMA vs Python per m")
    ax.set_ylabel("Force / m")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[2]
    ax.plot(sima_time, sima_cross_flow, label="SIMA cross-flow", linewidth=1.5)
    ax.plot(pred_time, pred_cross_flow, label=f"Python {PRIMARY_INTEGRATOR} cross-flow (/m)", linewidth=1.1)
    ax.set_title("Cross-flow force: SIMA vs Python per m")
    ax.set_ylabel("Force / m")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[3]
    ax.plot(sima_time, sima_added_mass, label="SIMA computed added mass", linewidth=1.1)
    ax.plot(sima_time, sima_morison, label="SIMA Morison", linewidth=1.1)
    ax.plot(sima_time, python_drag_on_sima_motion, label="Python drag on SIMA motion", linewidth=1.1)
    ax.plot(pred_time, pred_drag, label=f"Python {PRIMARY_INTEGRATOR} drag", linewidth=1.1)
    ax.plot(pred_time, pred_added_mass, label=f"Python {PRIMARY_INTEGRATOR} added mass", linewidth=1.1)
    ax.set_title("Drag / added-mass diagnostics per m")
    ax.set_ylabel("Force / m")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[4]
    ax.plot(sima_time, sima_total, label="SIMA total", linewidth=1.4)
    ax.plot(sima_time, sima_morison, label="SIMA Morison", linewidth=1.2)
    ax.plot(sima_time, sima_cross_flow, label="SIMA cross-flow", linewidth=1.2)
    ax.set_title("SIMA force decomposition")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Force / m")
    ax.grid(True, alpha=0.3)
    ax.legend()

    for ax in axes:
        ax.set_xlim(float(sima_time[0]), float(sima_time[-1]))

    fig.tight_layout()
    fig.savefig(case_dir / "force_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_summary_csv(rows: list[dict[str, float | str]], output_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_errors_vs_ur(rows: list[dict[str, float | str]], output_path: Path) -> None:
    if not rows:
        return
    ur = np.asarray([float(row["ur_value"]) for row in rows], dtype=float)
    disp_spec = np.asarray([float(row["primary_disp_spectral_rel_error"]) for row in rows], dtype=float)
    disp_std = np.asarray([float(row["primary_disp_std_rel_error"]) for row in rows], dtype=float)
    disp_freq_abs = np.asarray([float(row["primary_disp_dominant_freq_abs_error_hz"]) for row in rows], dtype=float)
    disp_freq_rel = np.asarray([float(row["primary_disp_dominant_freq_rel_error"]) for row in rows], dtype=float)
    force_freq_abs = np.asarray([float(row["primary_force_total_dominant_freq_abs_error_hz"]) for row in rows], dtype=float)
    force_freq_rel = np.asarray([float(row["primary_force_total_dominant_freq_rel_error"]) for row in rows], dtype=float)

    order = np.argsort(ur)
    ur = ur[order]
    disp_spec = disp_spec[order]
    disp_std = disp_std[order]
    disp_freq_abs = disp_freq_abs[order]
    disp_freq_rel = disp_freq_rel[order]
    force_freq_abs = force_freq_abs[order]
    force_freq_rel = force_freq_rel[order]

    fig, axes = plt.subplots(3, 1, figsize=(10, 14), sharex=True)

    ax = axes[0]
    ax.plot(ur, disp_spec, marker="o", label="Primary spectral relative error")
    ax.plot(ur, disp_std, marker="o", label="Primary std signed relative error")
    ax.set_title(f"SIMA vs primary displacement errors vs U_r ({PRIMARY_INTEGRATOR})")
    ax.set_ylabel("Error")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(ur, disp_freq_abs, marker="o", label="|dominant frequency error| [Hz]")
    ax.plot(ur, disp_freq_rel, marker="o", label="dominant frequency signed relative error")
    ax.set_ylabel("Frequency error")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[2]
    ax.plot(ur, force_freq_abs, marker="o", label="|total-force dominant frequency error| [Hz]")
    ax.plot(ur, force_freq_rel, marker="o", label="total-force dominant frequency signed relative error")
    ax.set_xlabel("Reduced velocity (U_r)")
    ax.set_ylabel("Force freq. error")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_theta_mean_vs_ur(rows: list[dict[str, float | str]], output_path: Path) -> None:
    if not rows:
        return
    ur = np.asarray([float(row["ur_value"]) for row in rows], dtype=float)
    theta_mean = np.asarray([float(row["primary_theta_mean_rad"]) for row in rows], dtype=float)

    order = np.argsort(ur)
    ur = ur[order]
    theta_mean = theta_mean[order]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    ax.plot(ur, theta_mean, marker="o", linewidth=1.4, label=f"Primary mean theta ({PRIMARY_INTEGRATOR})")
    ax.set_title("Average theta vs reduced velocity")
    ax.set_xlabel("Reduced velocity (U_r)")
    ax.set_ylabel("Circular mean theta [rad]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _ensure_output_dir(OUTPUT_DIR)
    if not SIMA_H5_PATH.exists():
        raise FileNotFoundError(f"SIMA HDF5 file not found: {SIMA_H5_PATH}")

    with h5py.File(SIMA_H5_PATH, "r") as handle:
        if PRINT_H5_LAYOUT:
            layout_text = "\n".join(_iter_h5_layout(handle))
            print(layout_text)
            (OUTPUT_DIR / "h5_layout.txt").write_text(layout_text + "\n", encoding="utf-8")

        case_cfgs = _selected_cases(handle)
        summary_rows: list[dict[str, float | str]] = []

        for case_cfg in case_cfgs:
            sima = _load_sima_case(handle, SIMA_H5_PATH, case_cfg)
            python_params = _build_python_params(sima)
            rollouts = _run_case_rollouts(sima, python_params)
            sima_forces = _load_sima_force_rollout(handle, sima, python_params)
            summary_row = _evaluate_case(sima, sima_forces, rollouts, python_params)
            summary_rows.append(summary_row)

            case_dir = OUTPUT_DIR / _sanitize_name(sima.case_name)
            case_dir.mkdir(parents=True, exist_ok=True)
            _save_case_npz(case_dir, sima, sima_forces, rollouts, python_params)
            _save_case_plot(case_dir, sima, rollouts[PRIMARY_INTEGRATOR], summary_row)
            _save_force_plot(case_dir, sima, sima_forces, rollouts[PRIMARY_INTEGRATOR], python_params)

            print(
                f"Processed {sima.case_name} | {sima.condition_set} | "
                f"U_r={float(summary_row['ur_value']):.6g} | "
                f"primary_nrmse={float(summary_row['primary_disp_nrmse_std']):.6g} | "
                f"|df|={float(summary_row['primary_disp_dominant_freq_abs_error_hz']):.6g} Hz | "
                f"force_corr={float(summary_row['primary_force_total_corr']):.6g}"
            )

    _write_summary_csv(summary_rows, OUTPUT_DIR / "summary.csv")
    _save_errors_vs_ur(summary_rows, OUTPUT_DIR / "errors_vs_ur.png")
    _save_theta_mean_vs_ur(summary_rows, OUTPUT_DIR / "theta_mean_vs_ur.png")


if __name__ == "__main__":
    main()
