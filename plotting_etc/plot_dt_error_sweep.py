import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
COMPARE_HELPER_DIR = ROOT / "vivana_cfd_data_pipeline" / "working_comparison_script"
for helper_path in (ROOT, COMPARE_HELPER_DIR):
    if str(helper_path) not in sys.path:
        sys.path.insert(0, str(helper_path))

from compare_sima_python import (
    COMPARISON_START_TIME,
    DEFAULT_PARAMS,
    MPLCONFIGDIR,
    PLOT_WINDOW_TIME,
    PLOTS_DIR,
    build_phase_burn_in_state as _build_phase_burn_in_state,
    build_python_params,
    collect_trimmed_channels as _collect_trimmed_channels,
    discover_h5_cases,
    find_default_h5_path,
    load_sima_case,
    numeric_suffix,
    print_available_runs,
    sanitize_name,
    single_sided_spectrum,
    summarize_case,
    trim_series,
)
from vivana_cfd_data_pipeline.vivana_td.vivana_td_model import run_simulation as _old_vivana_run_simulation

TD_PARAMETER_KEYS = ("Cv", "Cd", "Ca", "fhat0", "fhat_min", "fhat_max")
_STEPWISE_BASELINE_CACHE = None


def load_stepwise_baseline_config():
    global _STEPWISE_BASELINE_CACHE
    if _STEPWISE_BASELINE_CACHE is not None:
        return _STEPWISE_BASELINE_CACHE

    from vivana_cfd_data_pipeline.helpers.model_rollouts import (
        load_trained_model_sources,
        simulate_vivana_td_stepwise,
    )
    from plotting_etc.plot_dt_model_error_sweep import (
        CORRECTION_MODEL_SPECS as MODEL_CORRECTION_MODEL_SPECS,
        ROLLOUT_DTYPE as MODEL_ROLLOUT_DTYPE,
        TD_MEMORY_TAU_SPEC as MODEL_TD_MEMORY_TAU_SPEC,
    )

    corr_specs = [
        {"path": str(spec["path"]), "label": spec["label"]}
        for spec in MODEL_CORRECTION_MODEL_SPECS
    ]
    sources = load_trained_model_sources(corr_specs, repo_root=ROOT, device="cpu")
    if not sources:
        raise RuntimeError("No model sources loaded; cannot resolve model-sweep VIVANA-TD baseline parameters.")

    baseline_source = sources[0]
    model = getattr(baseline_source, "model", None)
    if hasattr(model, "to"):
        model.to(dtype=MODEL_ROLLOUT_DTYPE)

    _STEPWISE_BASELINE_CACHE = {
        "td_params": dict(baseline_source.base_td_params),
        "td_memory_tau_s": MODEL_TD_MEMORY_TAU_SPEC,
        "dtype": MODEL_ROLLOUT_DTYPE,
        "force_phase_convention": TD_FORCE_PHASE_CONVENTION,
        "source_label": str(baseline_source.label),
        "simulate_vivana_td_stepwise": simulate_vivana_td_stepwise,
    }
    return _STEPWISE_BASELINE_CACHE


def build_stepwise_series_from_run_params(run_params, initial_state):
    dt = float(run_params["dt"])
    total_time = float(run_params["T"])
    n_samples = max(2, int(np.floor(total_time / dt)) + 1)
    time_offset = 0.0 if initial_state is None else float(initial_state.get("time_offset", 0.0))
    time = time_offset + np.arange(n_samples, dtype=float) * dt

    y0 = 0.0 if initial_state is None else float(initial_state.get("y", 0.0))
    dy0 = 0.0 if initial_state is None else float(initial_state.get("dy", 0.0))
    ddy0 = 0.0 if initial_state is None else float(initial_state.get("ddy", 0.0))
    phi_vy0 = 0.0 if initial_state is None else float(initial_state.get("phi_vy", 0.0))
    sig_dy0 = 0.0 if initial_state is None else float(initial_state.get("sig_dy_loc", 0.0))
    sig_ddy0 = 0.0 if initial_state is None else float(initial_state.get("sig_ddy_loc", 0.0))

    force_zero = np.zeros((n_samples,), dtype=float)
    effective_mass = float(run_params["M"]) + 0.25 * float(run_params["rho"]) * float(run_params["Ca"]) * np.pi * float(run_params["D"]) ** 2
    td_context = np.repeat(
        np.asarray([[ddy0, phi_vy0, sig_dy0, sig_ddy0, float(run_params["U"])]], dtype=float),
        n_samples,
        axis=0,
    )
    return {
        "name": "h5_stepwise_vivana_td",
        "time": time,
        "displacement": np.full((n_samples,), y0, dtype=float),
        "velocity": np.full((n_samples,), dy0, dtype=float),
        "acceleration": np.full((n_samples,), ddy0, dtype=float),
        "force_total": force_zero.copy(),
        "force_per_m": force_zero.copy(),
        "force_td_stored": force_zero.copy(),
        "td_context": td_context,
        "rho": float(run_params["rho"]),
        "diameter": float(run_params["D"]),
        "stiffness": float(run_params["K"]),
        "effective_mass": effective_mass,
        "dry_mass": float(run_params["M"]),
        "damping": float(run_params.get("C", 0.0)),
        "span": 1.0,
        "ur": float("nan"),
        "ur_effective": float("nan"),
    }


def run_model_stepwise_simulation(*, params, seed=None, initial_state=None):
    if seed is not None:
        np.random.default_rng(seed)
    baseline = load_stepwise_baseline_config()
    series = build_stepwise_series_from_run_params(params, initial_state)
    rollout = baseline["simulate_vivana_td_stepwise"](
        series,
        td_params=dict(baseline["td_params"]),
        mass_source=VIVANA_MASS_SOURCE,
        td_memory_tau_s=baseline["td_memory_tau_s"],
        dtype=baseline["dtype"],
        force_phase_convention=str(params.get("force_phase_convention", baseline["force_phase_convention"])),
        use_vivana_added_mass_lhs=bool(params.get("use_added_mass_lhs", VIVANA_ADDED_MASS_LHS)),
    )
    force = np.asarray(rollout["force_td"], dtype=float)
    displacement = np.asarray(rollout["displacement_td"], dtype=float)
    velocity = np.asarray(rollout["velocity_td"], dtype=float)
    acceleration = np.asarray(rollout["acceleration_td"], dtype=float)
    zeros = np.zeros_like(force)
    force_cv = np.asarray(rollout.get("force_cv_td", force), dtype=float)
    force_drag = np.asarray(rollout.get("force_drag_td", zeros), dtype=float)
    force_added_mass = np.asarray(rollout.get("force_added_mass_td", zeros), dtype=float)
    return {
        "params": dict(params),
        "time": np.asarray(series["time"], dtype=float),
        "y": displacement,
        "dy": velocity,
        "ddy": acceleration,
        "Fy": force,
        "Fcv": force_cv,
        "Fdy": force_drag,
        "Fca": force_added_mass,
        "phi_vy": np.full_like(force, np.nan),
        "sig_dy_loc": np.full_like(force, np.nan),
        "sig_ddy_loc": np.full_like(force, np.nan),
    }


def run_old_integrator_simulation(*, params, seed=None, initial_state=None):
    if seed is not None:
        np.random.default_rng(seed)
    return _old_vivana_run_simulation(params=params, initial_state=initial_state)


def run_simulation(*, params, seed=None, initial_state=None):
    solver_path = str(VIVANA_TD_SOLVER_PATH).strip().lower()
    if solver_path == "model_stepwise":
        return run_model_stepwise_simulation(params=params, seed=seed, initial_state=initial_state)
    if solver_path == "old_integrator":
        return run_old_integrator_simulation(params=params, seed=seed, initial_state=initial_state)
    raise ValueError(
        "VIVANA_TD_SOLVER_PATH must be 'old_integrator' or 'model_stepwise', "
        f"got {VIVANA_TD_SOLVER_PATH!r}."
    )


def build_shared_python_params(sima_case):
    params = dict(DEFAULT_PARAMS)
    params.update(build_python_params(sima_case))
    return params


def vivana_added_mass_from_params(params):
    return 0.25 * float(params["rho"]) * float(params["Ca"]) * np.pi * float(params["D"]) ** 2


def vivana_solve_mass_from_params(params):
    mass = float(params["M"])
    if bool(params.get("use_added_mass_lhs", False)):
        mass += vivana_added_mass_from_params(params)
    return mass


def print_td_parameter_comparison(sima_cases, baseline_config):
    baseline_params = dict(baseline_config["td_params"])
    print(f"VIVANA-TD baseline source: {baseline_config['source_label']}")
    print("TD parameter comparison: H5/SIMA-derived -> model-sweep baseline")
    for sima_case in sima_cases:
        h5_params = build_shared_python_params(sima_case)
        comparisons = []
        for key in TD_PARAMETER_KEYS:
            h5_value = float(h5_params[key])
            baseline_value = float(baseline_params[key])
            delta = h5_value - baseline_value
            comparisons.append(f"{key}={h5_value:.6g}->{baseline_value:.6g} (d={delta:+.3g})")
        print(f"  {sima_case['metadata']['display_name']}: " + ", ".join(comparisons))


def normalize_force_phase_convention(value):
    convention = str(value).strip().lower()
    if convention in {"current", "old", "previous"}:
        return "current"
    if convention in {"next", "advanced", "new"}:
        return "next"
    raise ValueError(f"force_phase_convention must be 'current' or 'next', got {value!r}.")


def build_python_run_params(
    python_params,
    dt,
    *,
    integrator=None,
    tau_memory=None,
    added_mass_lhs=None,
    force_phase_convention=None,
):
    run_params = dict(python_params)
    run_params["dt"] = float(dt)
    run_params["integrator"] = str(DEFAULT_OLD_VIVANA_INTEGRATOR if integrator is None else integrator)
    run_params["use_added_mass_lhs"] = bool(VIVANA_ADDED_MASS_LHS if added_mass_lhs is None else added_mass_lhs)
    run_params["force_phase_convention"] = normalize_force_phase_convention(
        TD_FORCE_PHASE_CONVENTION if force_phase_convention is None else force_phase_convention
    )
    if USE_STEADY_STATE_CONVERGENCE_WINDOW:
        run_params["T"] = float(MAX_ROLLOUT_SECONDS)
    mass_source = str(VIVANA_MASS_SOURCE).strip().lower()
    if bool(run_params["use_added_mass_lhs"]) and mass_source == "effective":
        raise ValueError(
            "Added-mass-LHS runs require VIVANA_MASS_SOURCE='dry' "
            "to avoid double-counting added mass."
        )
    if mass_source == "effective":
        run_params["M"] = float(run_params["M"]) + vivana_added_mass_from_params(run_params)
    elif mass_source != "dry":
        raise ValueError(f"VIVANA_MASS_SOURCE must be 'dry' or 'effective', got {VIVANA_MASS_SOURCE!r}.")
    if tau_memory is not None:
        run_params["n_memory"] = max(1.0, float(round(float(tau_memory) / float(dt))))
    return run_params


def build_phase_burn_in_state(sima_case, python_params, initial_condition_time, *, target_dt=None):
    return _build_phase_burn_in_state(sima_case, python_params, initial_condition_time)


def collect_trimmed_channels(result, start_time):
    channels = dict(_collect_trimmed_channels(result, start_time))
    if "Fy" in result:
        total_force, _ = trim_series(result["Fy"], result["time"], start_time)
        channels["total_force_cf"] = np.asarray(total_force, dtype=float)
    if "Fca" in result:
        added_mass, _ = trim_series(result["Fca"], result["time"], start_time)
        channels["added_mass_cf"] = np.asarray(added_mass, dtype=float)
    elif "total_force_cf" not in channels and "crossflow_cf" in channels and "morison_cf" in channels:
        channels["total_force_cf"] = np.asarray(channels["crossflow_cf"], dtype=float) + np.asarray(
            channels["morison_cf"],
            dtype=float,
        )
    return channels


def dominant_frequency_model_style(signal, dt):
    if dt <= 0.0:
        return float("nan")
    signal = np.asarray(signal, dtype=float).reshape(-1)
    if signal.size < 4:
        return float("nan")
    centered = signal - np.mean(signal)
    if np.allclose(centered, 0.0):
        return float("nan")
    freqs = np.fft.rfftfreq(int(centered.size), d=float(dt))
    power = np.abs(np.fft.rfft(centered)) ** 2
    if freqs.size < 2 or power.size < 2:
        return float("nan")
    power = np.asarray(power, dtype=float)
    power[0] = 0.0
    mask = np.isfinite(freqs) & np.isfinite(power) & (freqs > 0.0)
    if np.count_nonzero(mask) < 1:
        return float("nan")
    valid_indices = np.flatnonzero(mask)
    peak_index = int(valid_indices[int(np.argmax(power[mask]))])
    peak_power = float(power[peak_index])
    if not np.isfinite(peak_power) or peak_power <= 0.0:
        return float("nan")

    interpolated_index = float(peak_index)
    if 1 <= peak_index < (power.size - 1):
        y_prev = float(power[peak_index - 1])
        y_peak = float(power[peak_index])
        y_next = float(power[peak_index + 1])
        denom = y_prev - 2.0 * y_peak + y_next
        if np.isfinite(denom) and abs(denom) > 1.0e-18:
            delta = 0.5 * (y_prev - y_next) / denom
            if np.isfinite(delta):
                interpolated_index += float(np.clip(delta, -1.0, 1.0))

    df = float(freqs[1] - freqs[0])
    if not np.isfinite(df) or df <= 0.0:
        return float(freqs[peak_index])
    return float(max(interpolated_index * df, 0.0))


def summarize_signal_model_style(signal, dt):
    values = np.asarray(signal, dtype=float).reshape(-1)
    return {
        "dominant_freq_hz": dominant_frequency_model_style(values, dt),
        "std": float(np.std(values)) if values.size >= 2 else float("nan"),
    }


# Configure the input file and run selection here.
H5_PATH = Path(find_default_h5_path())
if not H5_PATH.exists() and (ROOT / "vivana_cfd_data_pipeline" / H5_PATH).exists():
    H5_PATH = ROOT / "vivana_cfd_data_pipeline" / H5_PATH
LIST_RUNS = False
RUN_SELECTIONS = None
EXCLUDED_RUN_SELECTIONS = ()

# Configure the sweep here.
# Set DT_SWEEP_MODE to "dt" for an absolute dt sweep, or "downsampling_factor"
# to sweep using multiples of each case's original SIMA dt.
DT_SWEEP_MODE = "downsampling_factor"
ERROR_SWEEP_TAU_RATIO = 4.0
COMPARE_TO_PYTHON_GROUND_TRUTH = False
COMPARE_TO_FACTOR_ONE_REFERENCE = True
PYTHON_GROUND_TRUTH_DT = 0.001
DT_VALUES = (
    0.001,
    0.006,
    0.012,
    0.06,
    0.12,
    0.24,
    0.48
)
DOWNSAMPLING_FACTORS = (
    1,
    2,
    5,
    10,
    20,
    40
)

# The model sweep's factor 1 is not the H5/SIMA dt=0.001 s.  It is the
# NPZ-export native dt used for training/validation.  Enable this so the old
# Vivana-only sweep uses the same physical dt grid: dt = NPZ_native_dt * factor.
MATCH_NPZ_EXPORT_BASE_DT = True
VIVANA_MASS_SOURCE = "dry"

# When enabled, solve (M + m_a) yddot + C ydot + K y = Fcv + Fdrag,
# then record Fca = -m_a yddot in the output force components. Keep
# VIVANA_MASS_SOURCE="dry" with this enabled to avoid adding m_a twice.
VIVANA_ADDED_MASS_LHS = False

# VIVANA-TD propagation path:
#   "old_integrator"  -> vivana_cfd_data_pipeline.vivana_td.vivana_td_model.run_simulation(...)
#   "model_stepwise" -> vivana_cfd_data_pipeline.helpers.model_rollouts.simulate_vivana_td_stepwise(...)
VIVANA_TD_SOLVER_PATH = "old_integrator"
DEFAULT_OLD_VIVANA_INTEGRATOR = "rk4"  # non-coupled constant-force RK4
TD_FORCE_PHASE_CONVENTION = "current"

# Plot each requested combination as its own curve.  Set this to an empty tuple
# to fall back to VIVANA_ADDED_MASS_LHS and TD_FORCE_PHASE_CONVENTION only.
CONVENTION_COMBINATIONS = (
    {"added_mass_lhs": False, "force_phase_convention": "current", "label": r"Force from $\phi_{vy,i}$"},
    {"added_mass_lhs": False, "force_phase_convention": "next", "label": r"Force from $\phi_{vy,i+1}$"},
)

# Optional replacement for the H5/SIMA-derived burn-in state.  When enabled,
# every dt run and its factor-1 reference starts from the same low-perturbation
# state used by the model dt sweep, then COMPARISON_START_TIME is treated as
# transient time before metrics are collected.
USE_SYNTHETIC_INITIAL_CONDITION = True
SYNTHETIC_INITIAL_TIME_OFFSET = 0.0
SYNTHETIC_DISPLACEMENT_OVER_D = 0.1
SYNTHETIC_VELOCITY = 0.0
SYNTHETIC_THETA = 0.0
SYNTHETIC_SIGMA_DISPLACEMENT_OVER_D = 0.1
USE_STEADY_STATE_CONVERGENCE_WINDOW = True
STEADY_STATE_N_CYCLES = 10
STEADY_STATE_AMP_REL_TOL = 0.05
METRIC_WINDOW_AFTER_CONVERGENCE_SECONDS = 100.0
MAX_ROLLOUT_SECONDS = 600.0

NPZ_EXPORT_BASE_DT_BY_CONDITION = {
    "ConditionSet_1": 0.006,  # comb_Ur2
    "ConditionSet_2": 0.006,  # comb_Ur4
    "ConditionSet_3": 0.008,  # comb_Ur5
    "ConditionSet_4": 0.008,  # comb_Ur575
    "ConditionSet_5": 0.012,  # comb_Ur7
    "ConditionSet_6": 0.012,  # comb_Ur8
    "ConditionSet_7": 0.012,  # comb_Ur10
}

INTEGRATOR_SPECS = (
   #{"integrator": "forward_euler", "label": "Forward Euler"},
    {"integrator": DEFAULT_OLD_VIVANA_INTEGRATOR, "label": "RK4"},
    #{"integrator": "newmark_beta", "label": "Newmark-beta"},
)


def build_run_variant_specs():
    raw_combinations = tuple(CONVENTION_COMBINATIONS) or (
        {
            "added_mass_lhs": bool(VIVANA_ADDED_MASS_LHS),
            "force_phase_convention": TD_FORCE_PHASE_CONVENTION,
            "label": (
                f"AM {'LHS' if bool(VIVANA_ADDED_MASS_LHS) else 'RHS'}, "
                f"phi={normalize_force_phase_convention(TD_FORCE_PHASE_CONVENTION)}"
            ),
        },
    )
    specs = []
    include_integrator_label = len(INTEGRATOR_SPECS) > 1
    for integrator_spec in INTEGRATOR_SPECS:
        for combo in raw_combinations:
            phase = normalize_force_phase_convention(combo["force_phase_convention"])
            added_mass_lhs = bool(combo["added_mass_lhs"])
            combo_label = str(
                combo.get(
                    "label",
                    f"AM {'LHS' if added_mass_lhs else 'RHS'}, phi={phase}",
                )
            )
            label = f"{integrator_spec['label']} / {combo_label}" if include_integrator_label else combo_label
            specs.append(
                {
                    "integrator": str(integrator_spec["integrator"]),
                    "integrator_label": str(integrator_spec["label"]),
                    "added_mass_lhs": added_mass_lhs,
                    "force_phase_convention": phase,
                    "label": label,
                    "key": (
                        str(integrator_spec["integrator"]),
                        added_mass_lhs,
                        phase,
                    ),
                }
            )
    return tuple(specs)


RUN_VARIANT_SPECS = build_run_variant_specs()

SUBPLOT_SPECS = (
    ("disp_cf", "dominant_freq_hz", "Displacement Dominant Frequency Error"),
    ("disp_cf", "std", "Displacement Standard Deviation Error"),
    ("total_force_cf", "dominant_freq_hz", "Total Force Dominant Frequency Error"),
    ("total_force_cf", "std", "Total Force Standard Deviation Error"),
)

GENERATE_FORCE_FREQUENCY_DIAGNOSTICS = True
DIAGNOSTIC_SPECTRUM_FMAX = 5.0
ALIGN_DIAGNOSTIC_TIME_SERIES = True
FIGURE_DPI = 300
SAVE_PNG_PREVIEW = False

THESIS_FIGSIZE_2X2 = (5.85, 4.8)
BASE_FONT_SIZE = 8
AXIS_LABEL_FONT_SIZE = 9
TICK_FONT_SIZE = 8
LEGEND_FONT_SIZE = 8
PANEL_LABEL_FONT_SIZE = 9
SPINE_COLOR = "0.65"
SPINE_LINE_WIDTH = 0.6
GRID_COLOR = "0.88"
GRID_MINOR_COLOR = "0.94"
ERROR_SCALE = 100.0
HELD_OUT_REDUCED_VELOCITY = 6.46

_YLABEL_SYMBOLS = {
    ("disp_cf", "std"): r"$\varepsilon^y_{\sigma}$ [%]",
    ("disp_cf", "dominant_freq_hz"): r"$\varepsilon^y_{\omega}$ [%]",
    ("total_force_cf", "std"): r"$\varepsilon^F_{\sigma}$ [%]",
    ("total_force_cf", "dominant_freq_hz"): r"$\varepsilon^F_{\omega}$ [%]",
}

VARIANT_STYLES = {
    r"Force from $\phi_{vy,i}$": {
        "color": "#8C564B",
        "linestyle": "-",
        "linewidth": 1.35,
        "marker": "o",
    },
    r"Force from $\phi_{vy,i+1}$": {
        "color": "#17BECF",
        "linestyle": "-",
        "linewidth": 1.35,
        "marker": "o",
    },
    "phi=current": {
        "color": "#8C564B",
        "linestyle": "-",
        "linewidth": 1.35,
        "marker": "o",
    },
    "phi=next": {
        "color": "#17BECF",
        "linestyle": "-",
        "linewidth": 1.35,
        "marker": "o",
    },
    "AM RHS, phi=current": {
        "color": "#8C564B",
        "linestyle": "-",
        "linewidth": 1.35,
        "marker": "o",
    },
    "AM RHS, phi=next": {
        "color": "#17BECF",
        "linestyle": "-",
        "linewidth": 1.35,
        "marker": "o",
    },
    "AM LHS, phi=current": {
        "color": "#9467BD",
        "linestyle": "-",
        "linewidth": 1.35,
        "marker": "o",
    },
    "AM LHS, phi=next": {
        "color": "#BCBD22",
        "linestyle": "-",
        "linewidth": 1.35,
        "marker": "o",
    },
}


def print_progress(current, total, sweep_label, integrator_label, width=36):
    filled = int(width * current / total)
    bar = "#" * filled + "-" * (width - filled)
    print(
        f"\rProgress [{bar}] {current:>3}/{total}  {sweep_label:<28}  integrator={integrator_label:<14}",
        end="",
        flush=True,
    )
    if current == total:
        print()


def normalize_selections(selections):
    if selections is None:
        return None
    if isinstance(selections, (str, int)):
        return (str(selections),)
    return tuple(str(selection) for selection in selections)


def selection_matches_case(case, selection):
    return selection in {case["name"], case["task_root"], str(case["index"])}


def resolve_selected_cases(cases, selections, excluded_selections):
    selections = normalize_selections(selections)
    excluded_selections = normalize_selections(excluded_selections) or ()

    if selections is None:
        selected_cases = list(cases)
    else:
        selected_cases = []
        for selection in selections:
            for case in cases:
                if selection_matches_case(case, selection):
                    selected_cases.append(case)
                    break
            else:
                raise SystemExit(f"Unknown run selection: {selection}")

    filtered_cases = []
    for case in selected_cases:
        if any(selection_matches_case(case, selection) for selection in excluded_selections):
            continue
        filtered_cases.append(case)

    unique_cases = []
    seen_task_roots = set()
    for case in sorted(filtered_cases, key=lambda item: numeric_suffix(item["name"])):
        if case["task_root"] in seen_task_roots:
            continue
        unique_cases.append(case)
        seen_task_roots.add(case["task_root"])

    if not unique_cases:
        raise SystemExit("No runnable conditions remain after applying the include/exclude configuration.")

    return unique_cases


def sweep_uses_downsampling():
    return DT_SWEEP_MODE == "downsampling_factor"


def uses_python_ground_truth_reference():
    return bool(COMPARE_TO_PYTHON_GROUND_TRUTH)


def uses_factor_one_reference():
    return bool(COMPARE_TO_FACTOR_ONE_REFERENCE)


def sorted_unique_positive(values, label):
    unique_values = sorted({float(value) for value in values})
    if not unique_values:
        raise ValueError(f"{label} must contain at least one value.")
    if any(value <= 0.0 for value in unique_values):
        raise ValueError(f"{label} must contain only positive values.")
    return tuple(unique_values)


def get_sweep_values():
    if sweep_uses_downsampling():
        return sorted_unique_positive(DOWNSAMPLING_FACTORS, "DOWNSAMPLING_FACTORS")
    if DT_SWEEP_MODE != "dt":
        raise ValueError('DT_SWEEP_MODE must be either "dt" or "downsampling_factor".')
    return sorted_unique_positive(DT_VALUES, "DT_VALUES")


def resolve_case_dt_for_sweep_value(sima_case, sweep_value):
    if sweep_uses_downsampling():
        if MATCH_NPZ_EXPORT_BASE_DT:
            name = str(sima_case["metadata"]["display_name"])
            if name not in NPZ_EXPORT_BASE_DT_BY_CONDITION:
                raise KeyError(f"No NPZ export base dt configured for {name!r}.")
            return float(NPZ_EXPORT_BASE_DT_BY_CONDITION[name]) * float(sweep_value)
        return float(sima_case["dynamic"]["dt"]) * float(sweep_value)
    return float(sweep_value)


def sweep_axis_label():
    if sweep_uses_downsampling():
        return "Downsampling factor"
    return "dt [s]"


def sweep_value_name():
    if sweep_uses_downsampling():
        return "downsampling"
    return "dt"


def format_sweep_value(value):
    value = float(value)
    if sweep_uses_downsampling() and np.isclose(value, round(value)):
        return f"{int(round(value))}"
    return f"{value:g}"


def progress_label_for_sweep(dt, sweep_value):
    if sweep_uses_downsampling():
        return f"factor={format_sweep_value(sweep_value)} (dt={dt:g})"
    return f"dt={dt:g}"


def reduced_velocity_sweep_descriptor(sweep_value):
    if sweep_uses_downsampling():
        return f"downsampling={format_sweep_value(sweep_value)}x of each case dt"
    return f"dt={format_sweep_value(sweep_value)}"


def output_token(value):
    text = format_sweep_value(value).replace(".", "p").replace("-", "m")
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in text)


def comparison_reference_label():
    if uses_factor_one_reference():
        return "Python factor-1 reference"
    if uses_python_ground_truth_reference():
        return f"Python ground truth (dt={PYTHON_GROUND_TRUTH_DT:g})"
    return "SIMA"


def comparison_output_suffix():
    if uses_factor_one_reference():
        return "_vs_factor1_reference"
    if uses_python_ground_truth_reference():
        return "_vs_python_ground_truth"
    return ""


def nominal_shedding_period(params):
    return float(params["D"]) / (float(params["U"]) * float(params["fhat0"]))


def tau_memory_for_case(sima_case):
    params = build_shared_python_params(sima_case)
    return float(ERROR_SWEEP_TAU_RATIO * nominal_shedding_period(params))


def wrap_angle(angle):
    return float(np.arctan2(np.sin(float(angle)), np.cos(float(angle))))


def build_synthetic_initial_state(run_params):
    diameter = float(run_params["D"])
    stiffness = float(run_params["K"])
    mass = vivana_solve_mass_from_params(run_params)
    flow_speed = float(run_params["U"])
    if not (np.isfinite(diameter) and diameter > 0.0):
        raise ValueError("Synthetic initial condition needs a positive finite D.")
    if not (np.isfinite(stiffness) and stiffness > 0.0 and np.isfinite(mass) and mass > 0.0):
        raise ValueError("Synthetic initial condition needs positive finite K and M.")

    omega_n = float(np.sqrt(stiffness / mass))
    y0 = float(SYNTHETIC_DISPLACEMENT_OVER_D) * diameter
    dy0 = float(SYNTHETIC_VELOCITY)
    ddy0 = -omega_n * omega_n * y0
    sig_dy0 = max(float(SYNTHETIC_SIGMA_DISPLACEMENT_OVER_D) * diameter * omega_n, 1.0e-12)
    sig_ddy0 = max(float(SYNTHETIC_SIGMA_DISPLACEMENT_OVER_D) * diameter * omega_n * omega_n, 1.0e-12)

    speed_mag = float(np.sqrt(max(flow_speed * flow_speed + dy0 * dy0, 1.0e-12)))
    projection = flow_speed / speed_mag
    dy_r = dy0 * projection
    ddy_r = ddy0 * projection
    phi_dy0 = float(np.arctan2(-ddy_r / sig_ddy0, dy_r / sig_dy0))
    phi_vy0 = wrap_angle(phi_dy0 - float(SYNTHETIC_THETA))

    return {
        "time_offset": float(SYNTHETIC_INITIAL_TIME_OFFSET),
        "y": y0,
        "dy": dy0,
        "ddy": ddy0,
        "phi_vy": phi_vy0,
        "sig_dy_loc": sig_dy0,
        "sig_ddy_loc": sig_ddy0,
    }


def build_initial_state_for_run(sima_case, run_params, dt, initial_state_cache):
    if USE_SYNTHETIC_INITIAL_CONDITION:
        cache_key = (
            "synthetic_initial_state",
            sima_case["metadata"]["task_root"],
            bool(run_params.get("use_added_mass_lhs", False)),
            str(run_params.get("force_phase_convention", TD_FORCE_PHASE_CONVENTION)),
            float(run_params["M"]),
            float(run_params["K"]),
            float(run_params["D"]),
            float(run_params["U"]),
        )
        if cache_key not in initial_state_cache:
            initial_state_cache[cache_key] = build_synthetic_initial_state(run_params)
        return dict(initial_state_cache[cache_key])

    return build_phase_burn_in_state(
        sima_case,
        run_params,
        COMPARISON_START_TIME,
        target_dt=dt,
    )


def collect_sima_channels(sima_case):
    dt = sima_case["dynamic"]["dt"]
    sima_time = np.arange(sima_case["dynamic"]["disp_cf"].size) * dt
    disp_trimmed, _ = trim_series(sima_case["dynamic"]["disp_cf"], sima_time, COMPARISON_START_TIME)
    morison_trimmed, _ = trim_series(sima_case["dynamic"]["morison_cf"], sima_time, COMPARISON_START_TIME)
    crossflow_trimmed, _ = trim_series(sima_case["dynamic"]["crossflow_cf"], sima_time, COMPARISON_START_TIME)
    return {
        "disp_cf": disp_trimmed,
        "total_force_cf": crossflow_trimmed + morison_trimmed,
    }


def collect_sima_force_components(sima_case):
    dt = sima_case["dynamic"]["dt"]
    sima_time = np.arange(sima_case["dynamic"]["disp_cf"].size) * dt
    hydro_trimmed, _ = trim_series(sima_case["dynamic"]["hydro_cf"], sima_time, COMPARISON_START_TIME)
    morison_trimmed, _ = trim_series(sima_case["dynamic"]["morison_cf"], sima_time, COMPARISON_START_TIME)
    crossflow_trimmed, _ = trim_series(sima_case["dynamic"]["crossflow_cf"], sima_time, COMPARISON_START_TIME)
    return {
        "time": trim_series(sima_case["dynamic"]["disp_cf"], sima_time, COMPARISON_START_TIME)[1],
        "disp_cf": trim_series(sima_case["dynamic"]["disp_cf"], sima_time, COMPARISON_START_TIME)[0],
        "hydro_cf": hydro_trimmed,
        "morison_cf": morison_trimmed,
        "crossflow_cf": crossflow_trimmed,
        "added_mass_cf": hydro_trimmed - crossflow_trimmed - morison_trimmed,
        "total_force_cf": crossflow_trimmed + morison_trimmed,
    }


def summarize_comparison_channels(channels, dt):
    return {
        "disp_cf": summarize_signal_model_style(channels["disp_cf"], dt),
        "total_force_cf": summarize_signal_model_style(channels["total_force_cf"], dt),
    }


def natural_period_from_run_params(run_params):
    stiffness = float(run_params["K"])
    mass = vivana_solve_mass_from_params(run_params)
    if not (np.isfinite(stiffness) and stiffness > 0.0 and np.isfinite(mass) and mass > 0.0):
        return float("nan")
    return float(2.0 * np.pi * np.sqrt(mass / stiffness))


def find_steady_state_onset(displacement, dt, estimated_period_s, n_cycles=5, amp_rel_tol=0.05):
    values = np.asarray(displacement, dtype=float).reshape(-1)
    n = values.size
    if not (np.isfinite(dt) and dt > 0.0 and np.isfinite(estimated_period_s) and estimated_period_s > 0.0):
        return 0
    samples_per_cycle = max(1, int(round(float(estimated_period_s) / max(float(dt), 1.0e-12))))
    window = max(int(n_cycles) * samples_per_cycle, 4)
    step = max(1, samples_per_cycle // 2)

    if n < 2 * window + 1:
        return 0

    for index in range(0, n - 2 * window, step):
        segment_1 = values[index : index + window]
        segment_2 = values[index + window : index + 2 * window]
        amp_1 = float(np.std(segment_1))
        amp_2 = float(np.std(segment_2))
        if amp_1 < 1.0e-8 and amp_2 < 1.0e-8:
            return int(index)
        if amp_1 > 1.0e-8 and abs(amp_2 - amp_1) / amp_1 < float(amp_rel_tol):
            return int(index)

    return max(0, n - window)


def find_convergence_onset(displacement, dt, estimated_period_s, n_cycles=5, amp_rel_tol=0.05):
    values = np.asarray(displacement, dtype=float).reshape(-1)
    n = values.size
    if not (np.isfinite(dt) and dt > 0.0 and np.isfinite(estimated_period_s) and estimated_period_s > 0.0):
        return None
    samples_per_cycle = max(1, int(round(float(estimated_period_s) / max(float(dt), 1.0e-12))))
    window = max(int(n_cycles) * samples_per_cycle, 4)
    step = max(1, samples_per_cycle // 2)

    if n < 2 * window + 1:
        return None

    for index in range(0, n - 2 * window, step):
        segment_1 = values[index : index + window]
        segment_2 = values[index + window : index + 2 * window]
        amp_1 = float(np.std(segment_1))
        amp_2 = float(np.std(segment_2))
        if amp_1 < 1.0e-8 and amp_2 < 1.0e-8:
            return int(index)
        if amp_1 > 1.0e-8 and abs(amp_2 - amp_1) / amp_1 < float(amp_rel_tol):
            return int(index)

    return None


def python_metric_start_time():
    if USE_SYNTHETIC_INITIAL_CONDITION and USE_STEADY_STATE_CONVERGENCE_WINDOW:
        return 0.0
    return float(COMPARISON_START_TIME)


def collect_python_metric_channels(result, context_label=None):
    start_time = python_metric_start_time()
    channels = collect_trimmed_channels(result, start_time)
    if not USE_STEADY_STATE_CONVERGENCE_WINDOW:
        return channels

    dt = float(result["params"]["dt"])
    period_s = natural_period_from_run_params(result["params"])
    label = f"{context_label} / " if context_label else ""
    onset_idx = find_convergence_onset(
        channels["disp_cf"],
        dt,
        period_s,
        n_cycles=STEADY_STATE_N_CYCLES,
        amp_rel_tol=STEADY_STATE_AMP_REL_TOL,
    )
    if onset_idx is None:
        print(
            f"  [warn] No convergence for {label}dt={dt:g} within {MAX_ROLLOUT_SECONDS:g} s; metrics set to NaN."
        )
        onset_idx = 0
        end_idx = 0
    else:
        window_samples = max(2, int(round(float(METRIC_WINDOW_AFTER_CONVERGENCE_SECONDS) / dt)))
        end_idx = int(onset_idx) + window_samples
        if end_idx > np.asarray(channels["disp_cf"]).size:
            print(
                f"  [warn] {label}dt={dt:g} converged too late for "
                f"{METRIC_WINDOW_AFTER_CONVERGENCE_SECONDS:g} s metric window; metrics set to NaN."
            )
            onset_idx = 0
            end_idx = 0
    trimmed = {}
    n = np.asarray(channels["disp_cf"]).size
    for key, value in channels.items():
        arr = np.asarray(value)
        if arr.ndim >= 1 and arr.shape[0] == n:
            trimmed[key] = arr[int(onset_idx) : int(end_idx)]
        else:
            trimmed[key] = value
    return trimmed


def collect_python_summary(result, context_label=None):
    trimmed = collect_python_metric_channels(result, context_label=context_label)
    return summarize_comparison_channels(trimmed, result["params"]["dt"])


def absolute_relative_error(value, reference):
    value = float(value)
    reference = float(reference)
    if not (np.isfinite(value) and np.isfinite(reference)) or np.isclose(reference, 0.0):
        return float("nan")
    return abs(value - reference) / abs(reference)


def reduced_velocity(sima_case):
    variables = sima_case["variables"]
    python_params = build_shared_python_params(sima_case)
    u = float(python_params["U"])
    d = float(python_params["D"])
    m = float(python_params["M"])
    k = float(python_params["K"])
    ca = float(python_params["Ca"])
    rho = float(python_params["rho"])
    added_mass = 0.25 * rho * ca * np.pi * d**2
    effective_mass = m + added_mass
    natural_frequency_hz = np.sqrt(k / effective_mass) / (2.0 * np.pi)
    return float(u / (natural_frequency_hz * d))


def case_context_label(sima_case, *, prefix=None, sweep_value=None, integrator_label=None):
    parts = []
    if prefix:
        parts.append(str(prefix))
    try:
        ur_label = f"Ur={reduced_velocity(sima_case):.4g}"
    except Exception:
        ur_label = "Ur=unknown"
    parts.append(f"{sima_case['metadata']['display_name']} ({ur_label})")
    if sweep_value is not None:
        parts.append(f"{sweep_value_name()}={format_sweep_value(sweep_value)}")
    if integrator_label is not None:
        parts.append(str(integrator_label))
    return " / ".join(parts)


def rms(signal):
    signal = np.asarray(signal, dtype=float)
    return float(np.sqrt(np.mean(signal**2)))


def correlation_coefficient(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a_centered = a - np.mean(a)
    b_centered = b - np.mean(b)
    denom = np.sqrt(np.sum(a_centered**2) * np.sum(b_centered**2))
    if denom <= 0.0:
        return np.nan
    return float(np.sum(a_centered * b_centered) / denom)


def align_signal_to_reference(reference_time, reference_signal, signal_time, signal):
    reference_time = np.asarray(reference_time, dtype=float)
    reference_signal = np.asarray(reference_signal, dtype=float)
    signal_time = np.asarray(signal_time, dtype=float)
    signal = np.asarray(signal, dtype=float)

    raw_resampled = np.interp(reference_time, signal_time, signal)
    ref_centered = reference_signal - np.mean(reference_signal)
    sig_centered = raw_resampled - np.mean(raw_resampled)
    correlation = np.correlate(sig_centered, ref_centered, mode="full")
    lag_samples = int(np.argmax(correlation) - (reference_signal.size - 1))
    dt = float(np.median(np.diff(reference_time)))
    lag_seconds = lag_samples * dt
    aligned_resampled = np.interp(reference_time + lag_seconds, signal_time, signal)

    return {
        "raw_resampled": raw_resampled,
        "aligned_resampled": aligned_resampled,
        "lag_seconds": lag_seconds,
        "raw_corr": correlation_coefficient(reference_signal, raw_resampled),
        "aligned_corr": correlation_coefficient(reference_signal, aligned_resampled),
    }


def build_reference_bundle(sima_case, variant_spec, reference_cache):
    integrator_key = variant_spec["integrator"]
    if uses_factor_one_reference():
        reference_sweep_value = 1.0 if sweep_uses_downsampling() else min(get_sweep_values())
        reference_dt = resolve_case_dt_for_sweep_value(sima_case, reference_sweep_value)
        cache_key = ("python_factor_one_reference", sima_case["metadata"]["task_root"], variant_spec["key"], reference_dt)
        if cache_key not in reference_cache:
            python_params = build_shared_python_params(sima_case)
            tau_memory = tau_memory_for_case(sima_case)
            initial_state_cache = reference_cache.setdefault("_initial_states", {})
            reference_params = build_python_run_params(
                python_params,
                reference_dt,
                integrator=integrator_key,
                tau_memory=tau_memory,
                added_mass_lhs=variant_spec["added_mass_lhs"],
                force_phase_convention=variant_spec["force_phase_convention"],
            )
            phase_burn_in_state = build_initial_state_for_run(
                sima_case,
                reference_params,
                reference_dt,
                initial_state_cache,
            )
            reference_result = run_simulation(
                params=reference_params,
                seed=0,
                initial_state=phase_burn_in_state,
            )
            reference_channels = collect_python_metric_channels(
                reference_result,
                context_label=case_context_label(
                    sima_case,
                    prefix="reference",
                    sweep_value=reference_sweep_value,
                    integrator_label=variant_spec["label"],
                ),
            )
            reference_cache[cache_key] = {
                "label": comparison_reference_label(),
                "dt": float(reference_result["params"]["dt"]),
                "result": reference_result,
                "channels": reference_channels,
                "summary": summarize_comparison_channels(reference_channels, reference_result["params"]["dt"]),
            }
        return reference_cache[cache_key]

    if uses_python_ground_truth_reference():
        cache_key = ("python_ground_truth", sima_case["metadata"]["task_root"], variant_spec["key"])
        if cache_key not in reference_cache:
            python_params = build_shared_python_params(sima_case)
            tau_memory = tau_memory_for_case(sima_case)
            initial_state_cache = reference_cache.setdefault("_initial_states", {})
            reference_params = build_python_run_params(
                python_params,
                PYTHON_GROUND_TRUTH_DT,
                integrator=integrator_key,
                tau_memory=tau_memory,
                added_mass_lhs=variant_spec["added_mass_lhs"],
                force_phase_convention=variant_spec["force_phase_convention"],
            )
            phase_burn_in_state = build_initial_state_for_run(
                sima_case,
                reference_params,
                PYTHON_GROUND_TRUTH_DT,
                initial_state_cache,
            )
            reference_result = run_simulation(
                params=reference_params,
                seed=0,
                initial_state=phase_burn_in_state,
            )
            reference_channels = collect_python_metric_channels(
                reference_result,
                context_label=case_context_label(
                    sima_case,
                    prefix="reference",
                    integrator_label=variant_spec["label"],
                ),
            )
            reference_cache[cache_key] = {
                "label": comparison_reference_label(),
                "dt": float(reference_result["params"]["dt"]),
                "result": reference_result,
                "channels": reference_channels,
                "summary": summarize_comparison_channels(reference_channels, reference_result["params"]["dt"]),
            }
        return reference_cache[cache_key]

    cache_key = ("sima", sima_case["metadata"]["task_root"])
    if cache_key not in reference_cache:
        reference_channels = collect_sima_force_components(sima_case)
        reference_cache[cache_key] = {
            "label": comparison_reference_label(),
            "dt": float(sima_case["dynamic"]["dt"]),
            "result": None,
            "channels": reference_channels,
            "summary": summarize_comparison_channels(reference_channels, sima_case["dynamic"]["dt"]),
        }
    return reference_cache[cache_key]


def build_factor_one_initial_state(sima_case, python_params, tau_memory, initial_state_cache, variant_spec):
    reference_sweep_value = 1.0 if sweep_uses_downsampling() else min(get_sweep_values())
    reference_dt = resolve_case_dt_for_sweep_value(sima_case, reference_sweep_value)
    cache_key = ("factor_one_initial_state", sima_case["metadata"]["task_root"], variant_spec["key"], reference_dt)
    if cache_key not in initial_state_cache:
        reference_params = build_python_run_params(
            python_params,
            reference_dt,
            integrator=variant_spec["integrator"],
            tau_memory=tau_memory,
            added_mass_lhs=variant_spec["added_mass_lhs"],
            force_phase_convention=variant_spec["force_phase_convention"],
        )
        initial_state_cache[cache_key] = build_initial_state_for_run(
            sima_case,
            reference_params,
            reference_dt,
            initial_state_cache,
        )
    return dict(initial_state_cache[cache_key])


def collect_force_frequency_error_record(sima_case, dt, integrator_label, python_summary, reference_bundle):
    reference_summary = reference_bundle["summary"]
    return {
        "case_name": sima_case["metadata"]["display_name"],
        "task_root": sima_case["metadata"]["task_root"],
        "integrator_label": integrator_label,
        "dt": float(dt),
        "reference_label": reference_bundle["label"],
        "error": absolute_relative_error(
            python_summary["total_force_cf"]["dominant_freq_hz"],
            reference_summary["total_force_cf"]["dominant_freq_hz"],
        ),
        "reference_freq_hz": reference_summary["total_force_cf"]["dominant_freq_hz"],
        "python_freq_hz": python_summary["total_force_cf"]["dominant_freq_hz"],
    }


def build_case_error_series(sima_case, sweep_values, total_runs, completed_runs, reference_cache):
    python_params = build_shared_python_params(sima_case)
    tau_memory = tau_memory_for_case(sima_case)
    initial_state_cache = reference_cache.setdefault("_initial_states", {})
    reference_by_variant = {
        spec["label"]: build_reference_bundle(sima_case, spec, reference_cache)
        for spec in RUN_VARIANT_SPECS
    }

    error_series = {
        spec["label"]: {
            "x": [],
            "dt": [],
            "errors": {subplot_spec[:2]: [] for subplot_spec in SUBPLOT_SPECS},
        }
        for spec in RUN_VARIANT_SPECS
    }
    force_frequency_records = []

    for sweep_value in sweep_values:
        dt = resolve_case_dt_for_sweep_value(sima_case, sweep_value)
        for spec in RUN_VARIANT_SPECS:
            completed_runs += 1
            print_progress(
                completed_runs,
                total_runs,
                progress_label_for_sweep(dt, sweep_value),
                f"{sima_case['metadata']['display_name']} / {spec['label']}",
            )
            run_params = build_python_run_params(
                python_params,
                dt,
                integrator=spec["integrator"],
                tau_memory=tau_memory,
                added_mass_lhs=spec["added_mass_lhs"],
                force_phase_convention=spec["force_phase_convention"],
            )
            if uses_factor_one_reference():
                phase_burn_in_state = build_factor_one_initial_state(
                    sima_case,
                    python_params,
                    tau_memory,
                    initial_state_cache,
                    spec,
                )
            else:
                phase_burn_in_state = build_initial_state_for_run(
                    sima_case,
                    run_params,
                    dt,
                    initial_state_cache,
                )
            run_result = run_simulation(
                params=run_params,
                seed=0,
                initial_state=phase_burn_in_state,
            )
            python_summary = collect_python_summary(
                run_result,
                context_label=case_context_label(
                    sima_case,
                    sweep_value=sweep_value,
                    integrator_label=spec["label"],
                ),
            )
            reference_bundle = reference_by_variant[spec["label"]]
            force_frequency_records.append(
                collect_force_frequency_error_record(
                    sima_case,
                    dt,
                    spec["label"],
                    python_summary,
                    reference_bundle,
                )
            )
            series = error_series[spec["label"]]
            series["x"].append(float(sweep_value))
            series["dt"].append(float(dt))
            for channel, metric, _ in SUBPLOT_SPECS:
                series["errors"][(channel, metric)].append(
                    absolute_relative_error(
                        python_summary[channel][metric],
                        reference_bundle["summary"][channel][metric],
                    )
                )

    return error_series, force_frequency_records, completed_runs


def build_reduced_velocity_error_series(sima_cases, sweep_value, reference_cache):
    error_series = {
        spec["label"]: {
            "reduced_velocity": [],
            "errors": {subplot_spec[:2]: [] for subplot_spec in SUBPLOT_SPECS},
        }
        for spec in RUN_VARIANT_SPECS
    }

    total_runs = len(sima_cases) * len(RUN_VARIANT_SPECS)
    completed_runs = 0

    for sima_case in sorted(sima_cases, key=reduced_velocity):
        dt = resolve_case_dt_for_sweep_value(sima_case, sweep_value)
        python_params = build_shared_python_params(sima_case)
        tau_memory = tau_memory_for_case(sima_case)
        initial_state_cache = reference_cache.setdefault("_initial_states", {})
        reference_by_variant = {
            spec["label"]: build_reference_bundle(sima_case, spec, reference_cache)
            for spec in RUN_VARIANT_SPECS
        }
        ur = reduced_velocity(sima_case)

        for spec in RUN_VARIANT_SPECS:
            completed_runs += 1
            print_progress(
                completed_runs,
                total_runs,
                progress_label_for_sweep(dt, sweep_value),
                f"Ur sweep / {sima_case['metadata']['display_name']} / {spec['label']}",
            )
            run_params = build_python_run_params(
                python_params,
                dt,
                integrator=spec["integrator"],
                tau_memory=tau_memory,
                added_mass_lhs=spec["added_mass_lhs"],
                force_phase_convention=spec["force_phase_convention"],
            )
            if uses_factor_one_reference():
                phase_burn_in_state = build_factor_one_initial_state(
                    sima_case,
                    python_params,
                    tau_memory,
                    initial_state_cache,
                    spec,
                )
            else:
                phase_burn_in_state = build_initial_state_for_run(
                    sima_case,
                    run_params,
                    dt,
                    initial_state_cache,
                )
            run_result = run_simulation(
                params=run_params,
                seed=0,
                initial_state=phase_burn_in_state,
            )
            python_summary = collect_python_summary(
                run_result,
                context_label=case_context_label(
                    sima_case,
                    prefix="Ur sweep",
                    sweep_value=sweep_value,
                    integrator_label=spec["label"],
                ),
            )
            reference_bundle = reference_by_variant[spec["label"]]

            series = error_series[spec["label"]]
            series["reduced_velocity"].append(ur)
            for channel, metric, _ in SUBPLOT_SPECS:
                series["errors"][(channel, metric)].append(
                    absolute_relative_error(
                        python_summary[channel][metric],
                        reference_bundle["summary"][channel][metric],
                    )
                )

    return error_series


def aggregate_error_series(sima_cases, sweep_values, reference_cache):
    aggregate_series = {
        spec["label"]: {
            "x": [float(sweep_value) for sweep_value in sweep_values],
            "errors": {
                subplot_spec[:2]: {"mean": [], "min": [], "max": []}
                for subplot_spec in SUBPLOT_SPECS
            },
        }
        for spec in RUN_VARIANT_SPECS
    }

    collected_errors = {
        spec["label"]: {
            subplot_spec[:2]: {float(sweep_value): [] for sweep_value in sweep_values}
            for subplot_spec in SUBPLOT_SPECS
        }
        for spec in RUN_VARIANT_SPECS
    }
    force_frequency_records = []

    total_runs = len(sima_cases) * len(sweep_values) * len(RUN_VARIANT_SPECS)
    completed_runs = 0

    for sima_case in sima_cases:
        case_error_series, case_force_frequency_records, completed_runs = build_case_error_series(
            sima_case, sweep_values, total_runs, completed_runs, reference_cache
        )
        force_frequency_records.extend(case_force_frequency_records)
        for spec in RUN_VARIANT_SPECS:
            label = spec["label"]
            case_x_values = case_error_series[label]["x"]
            for index, sweep_value in enumerate(case_x_values):
                for channel, metric, _ in SUBPLOT_SPECS:
                    collected_errors[label][(channel, metric)][sweep_value].append(
                        case_error_series[label]["errors"][(channel, metric)][index]
                    )

    for spec in RUN_VARIANT_SPECS:
        label = spec["label"]
        for channel, metric, _ in SUBPLOT_SPECS:
            stats = aggregate_series[label]["errors"][(channel, metric)]
            for sweep_value in aggregate_series[label]["x"]:
                values = np.asarray(collected_errors[label][(channel, metric)][sweep_value], dtype=float)
                finite_values = values[np.isfinite(values)]
                if finite_values.size:
                    stats["mean"].append(float(np.mean(finite_values)))
                    stats["min"].append(float(np.min(finite_values)))
                    stats["max"].append(float(np.max(finite_values)))
                else:
                    stats["mean"].append(float("nan"))
                    stats["min"].append(float("nan"))
                    stats["max"].append(float("nan"))

    return aggregate_series, force_frequency_records


def apply_thesis_rcparams():
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman", "Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": BASE_FONT_SIZE,
        "axes.labelsize": AXIS_LABEL_FONT_SIZE,
        "axes.titlesize": PANEL_LABEL_FONT_SIZE,
        "axes.linewidth": SPINE_LINE_WIDTH,
        "xtick.labelsize": TICK_FONT_SIZE,
        "ytick.labelsize": TICK_FONT_SIZE,
        "legend.fontsize": LEGEND_FONT_SIZE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": FIGURE_DPI,
    })


def variant_style(label):
    short_label = str(label).rsplit(" / ", 1)[-1]
    return dict(VARIANT_STYLES.get(short_label, {"color": "0.25", "linestyle": "-", "linewidth": 1.35, "marker": "o"}))


def apply_axes_style(ax):
    ax.grid(True, which="major", color=GRID_COLOR, linewidth=0.5, alpha=0.75)
    ax.grid(True, which="minor", color=GRID_MINOR_COLOR, linewidth=0.35, alpha=0.45)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(SPINE_LINE_WIDTH)
        spine.set_edgecolor(SPINE_COLOR)


def add_panel_label(ax, index):
    ax.text(
        0.02,
        0.96,
        f"({chr(ord('a') + int(index))})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=PANEL_LABEL_FONT_SIZE,
    )


def scaled_errors(values):
    return [
        float(value) * ERROR_SCALE if np.isfinite(value) else float("nan")
        for value in values
    ]


def save_figure(fig, output_path):
    fig.savefig(
        output_path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.02,
        dpi=FIGURE_DPI,
    )
    if SAVE_PNG_PREVIEW:
        fig.savefig(
            output_path.with_suffix(".png"),
            dpi=FIGURE_DPI,
            bbox_inches="tight",
            pad_inches=0.03,
        )


def plot_error_sweep(h5_path, sima_cases, error_series):
    MPLCONFIGDIR.mkdir(exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(MPLCONFIGDIR.resolve())

    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        raise SystemExit("matplotlib is required to generate the dt error sweep plot.") from exc

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter

    plots_dir = PLOTS_DIR / f"{sanitize_name(h5_path.stem)}_aggregate"
    plots_dir.mkdir(parents=True, exist_ok=True)
    base_name = "downsampling_error_sweep" if sweep_uses_downsampling() else "dt_error_sweep"
    output_name = f"fig_07_{base_name}{comparison_output_suffix()}.pdf"
    output_path = plots_dir / output_name

    apply_thesis_rcparams()
    all_x_ticks = sorted({x for spec in RUN_VARIANT_SPECS for x in error_series[spec["label"]]["x"]})
    x_ticks = all_x_ticks[1:]
    if not x_ticks:
        raise SystemExit("Need at least one non-reference sweep value to plot timestep errors.")

    def format_sweep_tick(value, _):
        for tick in x_ticks:
            if np.isclose(value, tick):
                return format_sweep_value(tick)
        return ""

    fig, axes = plt.subplots(2, 2, figsize=THESIS_FIGSIZE_2X2, sharex=True)
    for panel_index, (ax, (channel, metric, _title)) in enumerate(zip(axes.flat, SUBPLOT_SPECS)):
        for spec in RUN_VARIANT_SPECS:
            series = error_series[spec["label"]]
            stats = series["errors"][(channel, metric)]
            style = variant_style(spec["label"])
            for stat_key, linestyle in (("mean", "-"), ("max", "--")):
                ax.plot(
                    series["x"][1:],
                    scaled_errors(stats[stat_key][1:]),
                    marker=style["marker"],
                    markersize=3.0,
                    linewidth=style["linewidth"],
                    color=style["color"],
                    linestyle=linestyle,
                    label=spec["label"] if stat_key == "mean" else None,
                )
        if panel_index >= 2:
            ax.set_xlabel(sweep_axis_label())
        ax.set_ylabel(_YLABEL_SYMBOLS[(channel, metric)], labelpad=6)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(FixedLocator(x_ticks))
        ax.xaxis.set_major_formatter(FuncFormatter(format_sweep_tick))
        apply_axes_style(ax)
        add_panel_label(ax, panel_index)

    variant_handles = []
    for spec in RUN_VARIANT_SPECS:
        style = variant_style(spec["label"])
        variant_handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                linestyle="-",
                marker=style["marker"],
                markersize=3.0,
                linewidth=style["linewidth"],
                label=spec["label"],
            )
        )
    statistic_handles = [
        Line2D([0], [0], color="0.25", linestyle="-", linewidth=1.35, label="Mean"),
        Line2D([0], [0], color="0.25", linestyle="--", linewidth=1.35, label="Maximum"),
    ]
    fig.legend(
        handles=variant_handles + statistic_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=min(4, max(1, len(variant_handles) + len(statistic_handles))),
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        columnspacing=1.0,
        handletextpad=0.45,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94), pad=0.35, w_pad=0.8, h_pad=0.6)
    save_figure(fig, output_path)
    plt.close(fig)
    return [output_path]


def plot_reduced_velocity_error_sweep(h5_path, reduced_velocity_error_series, sweep_value):
    MPLCONFIGDIR.mkdir(exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(MPLCONFIGDIR.resolve())

    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        raise SystemExit("matplotlib is required to generate the reduced-velocity error plot.") from exc

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plots_dir = PLOTS_DIR / f"{sanitize_name(h5_path.stem)}_aggregate"
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_path = plots_dir / (
        f"fig_07_dt_error_sweep_by_reduced_velocity_"
        f"{sweep_value_name()}_{output_token(sweep_value)}{comparison_output_suffix()}.pdf"
    )

    apply_thesis_rcparams()
    fig, axes = plt.subplots(2, 2, figsize=THESIS_FIGSIZE_2X2, sharex=True)
    ur_ticks = sorted(
        {
            ur
            for spec in RUN_VARIANT_SPECS
            for ur in reduced_velocity_error_series[spec["label"]]["reduced_velocity"]
        }
    )

    for panel_index, (ax, (channel, metric, _title)) in enumerate(zip(axes.flat, SUBPLOT_SPECS)):
        ax.axvline(
            HELD_OUT_REDUCED_VELOCITY,
            color="0.5",
            linewidth=7.5,
            alpha=0.10,
            zorder=0,
        )
        if panel_index == 0:
            ax.text(
                HELD_OUT_REDUCED_VELOCITY,
                1.02,
                "held-out",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=LEGEND_FONT_SIZE,
                color="0.35",
            )
        for spec in RUN_VARIANT_SPECS:
            series = reduced_velocity_error_series[spec["label"]]
            style = variant_style(spec["label"])
            ax.plot(
                series["reduced_velocity"],
                scaled_errors(series["errors"][(channel, metric)]),
                marker=style["marker"],
                markersize=3.2,
                linewidth=style["linewidth"],
                color=style["color"],
                linestyle=style["linestyle"],
                label=spec["label"],
            )
        if panel_index >= 2:
            ax.set_xlabel(r"Reduced velocity $U_r$")
        ax.set_ylabel(_YLABEL_SYMBOLS[(channel, metric)], labelpad=6)
        ax.set_yscale("log")
        ax.set_xticks(ur_ticks, [f"{tick:.2f}" for tick in ur_ticks])
        ax.margins(x=0.04)
        apply_axes_style(ax)
        add_panel_label(ax, panel_index)

    integrator_handles = []
    for spec in RUN_VARIANT_SPECS:
        style = variant_style(spec["label"])
        integrator_handles.append(
            Line2D(
                [0], [0],
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=3.2,
                linewidth=style["linewidth"],
                label=spec["label"],
            )
        )
    fig.legend(
        handles=integrator_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=min(3, max(1, len(integrator_handles))),
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        columnspacing=1.0,
        handletextpad=0.45,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94), pad=0.35, w_pad=0.8, h_pad=0.6)
    save_figure(fig, output_path)
    plt.close(fig)
    return output_path


def rerun_case_result(sima_case, dt, variant_spec):
    python_params = build_shared_python_params(sima_case)
    tau_memory = tau_memory_for_case(sima_case)
    run_params = build_python_run_params(
        python_params,
        dt,
        integrator=variant_spec["integrator"],
        tau_memory=tau_memory,
        added_mass_lhs=variant_spec["added_mass_lhs"],
        force_phase_convention=variant_spec["force_phase_convention"],
    )
    phase_burn_in_state = build_initial_state_for_run(
        sima_case,
        run_params,
        dt,
        {},
    )
    return run_simulation(
        params=run_params,
        seed=0,
        initial_state=phase_burn_in_state,
    )


def generate_force_balance_diagnostic_plots(h5_path, sima_cases):
    MPLCONFIGDIR.mkdir(exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(MPLCONFIGDIR.resolve())

    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        raise SystemExit("matplotlib is required to generate the diagnostic plots.") from exc

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = PLOTS_DIR / f"{sanitize_name(h5_path.stem)}_aggregate"
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []
    summaries = []

    for sima_case in sima_cases:
        sima_components = collect_sima_force_components(sima_case)
        sima_sum = sima_components["crossflow_cf"] + sima_components["morison_cf"]
        sima_residual = sima_components["hydro_cf"] - sima_sum
        sima_time = sima_components["time"]
        sima_dt = sima_case["dynamic"]["dt"]
        sima_window = max(1, min(int(PLOT_WINDOW_TIME / sima_dt), sima_time.size))

        fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))
        fig.patch.set_facecolor("#fcfcf8")

        for ax in axes.flat:
            ax.set_facecolor("#fffdf7")

        axes[0, 0].plot(
            sima_time[-sima_window:],
            sima_components["hydro_cf"][-sima_window:],
            linewidth=1.8,
            color="#1f4e79",
            label="SIMA hydro_cf",
        )
        axes[0, 0].plot(
            sima_time[-sima_window:],
            sima_sum[-sima_window:],
            linewidth=1.6,
            color="#c0392b",
            linestyle="--",
            label="SIMA crossflow + morison",
        )
        axes[0, 0].set_title("SIMA force balance")
        axes[0, 0].set_xlabel("Time [s]")
        axes[0, 0].set_ylabel("Force [N]")
        axes[0, 0].grid(True, alpha=0.25)
        axes[0, 0].legend()

        axes[0, 1].plot(
            sima_time[-sima_window:],
            sima_residual[-sima_window:],
            linewidth=1.7,
            color="#117a65",
        )
        axes[0, 1].set_title("SIMA residual: hydro - (crossflow + morison)")
        axes[0, 1].set_xlabel("Time [s]")
        axes[0, 1].set_ylabel("Force [N]")
        axes[0, 1].grid(True, alpha=0.25)

        sima_freq_hydro, sima_amp_hydro = single_sided_spectrum(sima_components["hydro_cf"], sima_dt, skip_time=0.0)
        sima_freq_sum, sima_amp_sum = single_sided_spectrum(sima_sum, sima_dt, skip_time=0.0)
        sima_freq_res, sima_amp_res = single_sided_spectrum(sima_residual, sima_dt, skip_time=0.0)
        hydro_mask = sima_freq_hydro <= DIAGNOSTIC_SPECTRUM_FMAX
        sum_mask = sima_freq_sum <= DIAGNOSTIC_SPECTRUM_FMAX
        res_mask = sima_freq_res <= DIAGNOSTIC_SPECTRUM_FMAX

        axes[1, 0].plot(
            sima_freq_hydro[hydro_mask],
            sima_amp_hydro[hydro_mask],
            linewidth=1.8,
            color="#1f4e79",
            label="SIMA hydro_cf",
        )
        axes[1, 0].plot(
            sima_freq_sum[sum_mask],
            sima_amp_sum[sum_mask],
            linewidth=1.6,
            color="#c0392b",
            linestyle="--",
            label="SIMA crossflow + morison",
        )
        axes[1, 0].set_title("SIMA force balance spectrum")
        axes[1, 0].set_xlabel("Frequency [Hz]")
        axes[1, 0].set_ylabel("Amplitude")
        axes[1, 0].grid(True, alpha=0.25)
        axes[1, 0].legend()

        axes[1, 1].plot(
            sima_freq_res[res_mask],
            sima_amp_res[res_mask],
            linewidth=1.7,
            color="#117a65",
        )
        axes[1, 1].set_title("Residual spectrum")
        axes[1, 1].set_xlabel("Frequency [Hz]")
        axes[1, 1].set_ylabel("Amplitude")
        axes[1, 1].grid(True, alpha=0.25)

        residual_rms = rms(sima_residual)
        hydro_rms = rms(sima_components["hydro_cf"])
        relative_residual_pct = 100.0 * residual_rms / hydro_rms if hydro_rms > 0.0 else np.nan
        summaries.append(
            {
                "case_name": sima_case["metadata"]["display_name"],
                "task_root": sima_case["metadata"]["task_root"],
                "hydro_rms": hydro_rms,
                "residual_rms": residual_rms,
                "relative_residual_pct": relative_residual_pct,
            }
        )

        fig.suptitle(
            (
                f"SIMA force-balance diagnostic for {sima_case['metadata']['display_name']}: "
                f"residual RMS = {residual_rms:.6g} N "
                f"({relative_residual_pct:.3f}% of hydro RMS)"
            ),
            y=0.98,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

        filename = f"force_balance_diagnostic_{sanitize_name(sima_case['metadata']['display_name'])}.png"
        output_path = plots_dir / filename
        fig.savefig(output_path, dpi=FIGURE_DPI)
        plt.close(fig)
        output_paths.append(output_path)

    return output_paths, summaries


def generate_force_frequency_diagnostic_plots(h5_path, sima_cases, force_frequency_records, reference_cache):
    if not GENERATE_FORCE_FREQUENCY_DIAGNOSTICS:
        return []

    MPLCONFIGDIR.mkdir(exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(MPLCONFIGDIR.resolve())

    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        raise SystemExit("matplotlib is required to generate the diagnostic plots.") from exc

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = PLOTS_DIR / f"{sanitize_name(h5_path.stem)}_aggregate"
    plots_dir.mkdir(parents=True, exist_ok=True)
    sima_case_by_root = {
        sima_case["metadata"]["task_root"]: sima_case
        for sima_case in sima_cases
    }
    variant_by_label = {
        spec["label"]: spec
        for spec in RUN_VARIANT_SPECS
    }
    output_paths = []

    for spec in RUN_VARIANT_SPECS:
        matching_records = [
            record
            for record in force_frequency_records
            if record["integrator_label"] == spec["label"]
        ]
        if not matching_records:
            continue

        worst_record = max(matching_records, key=lambda record: record["error"])
        sima_case = sima_case_by_root[worst_record["task_root"]]
        variant_spec = variant_by_label[spec["label"]]
        run_result = rerun_case_result(
            sima_case,
            worst_record["dt"],
            variant_spec,
        )
        reference_bundle = build_reference_bundle(sima_case, variant_spec, reference_cache)

        reference_channels = reference_bundle["channels"]
        python_channels = collect_python_metric_channels(
            run_result,
            context_label=case_context_label(
                sima_case,
                prefix="diagnostic",
                integrator_label=spec["label"],
            ),
        )
        reference_dt = reference_bundle["dt"]
        python_dt = run_result["params"]["dt"]
        reference_time = reference_channels["time"]
        python_time = python_channels["time"]
        reference_label = reference_bundle["label"]

        component_specs = (
            ("disp_cf", "Displacement", "disp_cf", "Displacement [m]"),
            ("crossflow_cf", "Vortex lift force", "crossflow_cf", "Force [N/m]"),
            ("morison_cf", "Drag force", "morison_cf", "Force [N/m]"),
            ("added_mass_cf", "Added-mass force", "added_mass_cf", "Force [N/m]"),
        )

        fig, axes = plt.subplots(4, 1, figsize=(14, 14))
        fig.patch.set_facecolor("#fcfcf8")
        alignment_summaries = []
        for row_index, component_spec in enumerate(component_specs):
            if len(component_spec) == 4:
                sima_key, title, python_key, ylabel = component_spec
            else:
                sima_key, title, python_key = component_spec
                ylabel = "Force [N]"
            reference_signal = reference_channels[sima_key]
            python_signal = python_channels[python_key]
            reference_window = max(1, min(int(PLOT_WINDOW_TIME / reference_dt), reference_signal.size))
            python_window = max(1, min(int(PLOT_WINDOW_TIME / python_dt), python_signal.size))
            time_ax = axes[row_index]

            time_ax.set_facecolor("#fffdf7")
            time_ax.plot(
                reference_time[-reference_window:],
                reference_signal[-reference_window:],
                linewidth=1.8,
                color="#1f4e79",
                label=reference_label,
            )
            python_label = spec["label"]
            if ALIGN_DIAGNOSTIC_TIME_SERIES:
                aligned = align_signal_to_reference(
                    reference_time[-reference_window:],
                    reference_signal[-reference_window:],
                    python_time[-python_window:],
                    python_signal[-python_window:],
                )
                alignment_summaries.append(
                    {
                        "title": title,
                        "lag_seconds": aligned["lag_seconds"],
                        "raw_corr": aligned["raw_corr"],
                        "aligned_corr": aligned["aligned_corr"],
                    }
                )
                time_ax.plot(
                    reference_time[-reference_window:],
                    aligned["raw_resampled"],
                    linewidth=1.1,
                    linestyle=":",
                    color="#c0392b",
                    alpha=0.45,
                    label=f"{python_label} raw",
                )
                time_ax.plot(
                    reference_time[-reference_window:],
                    aligned["aligned_resampled"],
                    linewidth=1.6,
                    color="#c0392b",
                    label=f"{python_label} aligned",
                )
                time_ax.text(
                    0.02,
                    0.98,
                    (
                        f"lag = {aligned['lag_seconds']:.4g} s\n"
                        f"corr raw = {aligned['raw_corr']:.3f}\n"
                        f"corr aligned = {aligned['aligned_corr']:.3f}"
                    ),
                    transform=time_ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=9,
                    bbox={"facecolor": "#fffdf7", "alpha": 0.8, "edgecolor": "#d8d2c5"},
                )
            else:
                time_ax.plot(
                    python_time[-python_window:],
                    python_signal[-python_window:],
                    linewidth=1.6,
                    color="#c0392b",
                    label=python_label,
                )
            time_ax.set_title(f"{title} time history")
            time_ax.set_xlabel("Time [s]")
            time_ax.set_ylabel(ylabel)
            time_ax.grid(True, alpha=0.25)
            time_ax.legend()

        fig.suptitle(
            (
                f"Worst hydrodynamic-force frequency case for {spec['label']} vs {reference_bundle['label']}: "
                f"{worst_record['case_name']} at dt={worst_record['dt']:g}, "
                f"abs error={worst_record['error']:.3g}"
            ),
            y=0.98,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))

        filename = f"force_component_diagnostic_{sanitize_name(spec['label'])}{comparison_output_suffix()}.png"
        output_path = plots_dir / filename
        fig.savefig(output_path, dpi=FIGURE_DPI)
        plt.close(fig)
        output_paths.append(output_path)

        print(
            f"{spec['label']} lag summary vs {reference_bundle['label']} for {worst_record['case_name']} at dt={worst_record['dt']:g}:"
        )
        for summary in alignment_summaries:
            print(
                f"  {summary['title']:<26} lag={summary['lag_seconds']:.6f} s  "
                f"corr_raw={summary['raw_corr']:.3f}  corr_aligned={summary['aligned_corr']:.3f}"
            )

    return output_paths


def print_error_table(error_series):
    print(f"Aggregated errors by {sweep_value_name()} against {comparison_reference_label()} [%]")
    header = (
        f"{'variant':<24} {sweep_value_name():>14} "
        f"{'disp_freq_mean':>14} {'disp_freq_min':>14} {'disp_freq_max':>14} "
        f"{'disp_std_mean':>14} {'force_freq_mean':>16} {'force_std_mean':>16}"
    )
    print(header)
    for spec in RUN_VARIANT_SPECS:
        series = error_series[spec["label"]]
        for index, sweep_value in enumerate(series["x"]):
            disp_freq_stats = series["errors"][("disp_cf", "dominant_freq_hz")]
            disp_rms_stats = series["errors"][("disp_cf", "std")]
            force_freq_stats = series["errors"][("total_force_cf", "dominant_freq_hz")]
            force_rms_stats = series["errors"][("total_force_cf", "std")]
            print(
                f"{spec['label']:<24} "
                f"{format_sweep_value(sweep_value):>14} "
                f"{ERROR_SCALE * disp_freq_stats['mean'][index]:>14.3f} "
                f"{ERROR_SCALE * disp_freq_stats['min'][index]:>14.3f} "
                f"{ERROR_SCALE * disp_freq_stats['max'][index]:>14.3f} "
                f"{ERROR_SCALE * disp_rms_stats['mean'][index]:>14.3f} "
                f"{ERROR_SCALE * force_freq_stats['mean'][index]:>16.3f} "
                f"{ERROR_SCALE * force_rms_stats['mean'][index]:>16.3f}"
            )
    print()


def main():
    h5_path = Path(H5_PATH)
    cases = discover_h5_cases(h5_path)

    if LIST_RUNS:
        print_available_runs(cases)
        return

    selected_cases = resolve_selected_cases(cases, RUN_SELECTIONS, EXCLUDED_RUN_SELECTIONS)
    sima_cases = [load_sima_case(h5_path, case_info) for case_info in selected_cases]
    solver_path = str(VIVANA_TD_SOLVER_PATH).strip().lower()
    print(f"VIVANA-TD solver path: {solver_path}")
    mass_source = str(VIVANA_MASS_SOURCE).strip().lower()
    if any(bool(spec["added_mass_lhs"]) for spec in RUN_VARIANT_SPECS) and mass_source == "effective":
        raise SystemExit(
            "Added-mass-LHS variants require VIVANA_MASS_SOURCE='dry' "
            "to avoid double-counting added mass."
        )
    if solver_path == "model_stepwise":
        baseline_config = load_stepwise_baseline_config()
        print_td_parameter_comparison(sima_cases, baseline_config)
    elif solver_path != "old_integrator":
        raise SystemExit(
            "VIVANA_TD_SOLVER_PATH must be 'old_integrator' or 'model_stepwise', "
            f"got {VIVANA_TD_SOLVER_PATH!r}."
        )
    sweep_values = get_sweep_values()
    highest_sweep_value = max(sweep_values)
    reference_cache = {}

    error_series, force_frequency_records = aggregate_error_series(sima_cases, sweep_values, reference_cache)
    output_paths = plot_error_sweep(h5_path, sima_cases, error_series)
    reduced_velocity_error_series = build_reduced_velocity_error_series(sima_cases, highest_sweep_value, reference_cache)
    reduced_velocity_output_path = plot_reduced_velocity_error_sweep(
        h5_path,
        reduced_velocity_error_series,
        highest_sweep_value,
    )
    if uses_python_ground_truth_reference() or uses_factor_one_reference():
        force_balance_paths, force_balance_summaries = [], []
    else:
        force_balance_paths, force_balance_summaries = generate_force_balance_diagnostic_plots(
            h5_path,
            sima_cases,
        )
    diagnostic_paths = generate_force_frequency_diagnostic_plots(
        h5_path,
        sima_cases,
        force_frequency_records,
        reference_cache,
    )

    print(f"HDF5 file    : {h5_path}")
    print(f"Included runs: {', '.join(case['metadata']['display_name'] for case in sima_cases)}")
    print(f"Excluded runs: {', '.join(EXCLUDED_RUN_SELECTIONS) if EXCLUDED_RUN_SELECTIONS else '-'}")
    print(f"Reference    : {comparison_reference_label()}")
    print(f"Solver path  : {VIVANA_TD_SOLVER_PATH}")
    print(f"Mass source  : {VIVANA_MASS_SOURCE}")
    print(f"Variants     : {', '.join(spec['label'] for spec in RUN_VARIANT_SPECS)}")
    if str(VIVANA_TD_SOLVER_PATH).strip().lower() == "model_stepwise":
        print("TD force phase: per variant")
    print(f"tau/T_ref    : {ERROR_SWEEP_TAU_RATIO:g}")
    if USE_SYNTHETIC_INITIAL_CONDITION:
        print(
            "Initial state: synthetic "
            f"y0={SYNTHETIC_DISPLACEMENT_OVER_D:g}D, "
            f"dy0={SYNTHETIC_VELOCITY:g}, "
            f"theta0={SYNTHETIC_THETA:g}, "
            f"sig_dy={SYNTHETIC_SIGMA_DISPLACEMENT_OVER_D:g}D*omega_n, "
            f"sig_ddy={SYNTHETIC_SIGMA_DISPLACEMENT_OVER_D:g}D*omega_n^2"
        )
    else:
        print(f"Initial state: H5/SIMA burn-in at t={COMPARISON_START_TIME:g} s")
    if USE_STEADY_STATE_CONVERGENCE_WINDOW:
        print(
            "Metric trim  : steady-state onset from displacement std "
            f"({STEADY_STATE_AMP_REL_TOL * 100:g}% over {STEADY_STATE_N_CYCLES:g} periods), "
            f"then {METRIC_WINDOW_AFTER_CONVERGENCE_SECONDS:g} s window "
            f"(max rollout {MAX_ROLLOUT_SECONDS:g} s)"
        )
    else:
        print(f"Metric trim  : fixed t >= {python_metric_start_time():g} s")
    print(
        f"{sweep_value_name():<12}: "
        f"{', '.join(format_sweep_value(value) for value in sweep_values)}"
    )
    print("Plots written:")
    for output_path in output_paths:
        print(f"  {output_path.as_posix()}")
    print(
        f"Ur plot      : {reduced_velocity_output_path.as_posix()}  "
        f"({reduced_velocity_sweep_descriptor(highest_sweep_value)})"
    )
    if force_balance_paths:
        print("Force-balance diagnostics:")
        for force_balance_path in force_balance_paths:
            print(f"  {force_balance_path.as_posix()}")
        print("Force-balance residual summary:")
        for summary in force_balance_summaries:
            print(
                f"  {summary['case_name']}: residual_rms={summary['residual_rms']:.6g} N, "
                f"hydro_rms={summary['hydro_rms']:.6g} N, "
                f"relative={summary['relative_residual_pct']:.3f}%"
            )
    if diagnostic_paths:
        print("Diagnostic plots:")
        for diagnostic_path in diagnostic_paths:
            print(f"  {diagnostic_path.as_posix()}")
    print()
    print_error_table(error_series)


if __name__ == "__main__":
    main()
