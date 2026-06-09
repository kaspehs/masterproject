from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
CFD_DATA_DIR = Path(__file__).resolve().parents[1]
if str(CFD_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(CFD_DATA_DIR))

from vivana_cfd_data_pipeline.helpers import spectral_metrics as spectral_helpers
from vivana_cfd_data_pipeline.scripts.training_npz_loader import iter_all_npz_files, load_series, resolve_dataset_root


# Edit these values directly, then run this file.
DATASET_ROOT: Path | None = Path("vivana_cfd_data_pipeline/generated/cfd_npz_exports")
OUTPUT_DIR = Path("vivana_cfd_data_pipeline/outputs/analysis/vivana_td_ga")
LOSS_MODE = "mae"  # "mae" | "mse"
AGGREGATE_WEIGHTING = "per_ur_mean"  # "per_ur_mean" | "per_case_mean"
TD_MASS_SOURCE = "dry"  # "dry" | "effective"
TD_MEMORY_TAU_S: float | str | None = "tau_over_tref:4"
EXCLUDED_UR_VALUES: list[float] = []
MAX_CASES: int | None = None

VIVANA_TRANSIENT_SECONDS = 100.0
VIVANA_FIRST_CASE_EXTRA_SECONDS = 100.0
VIVANA_CONTINUATION_TRANSIENT_SECONDS = 200.0
VIVANA_CONTINUATION_KEPT_SECONDS = 300.0

POPULATION_SIZE = 48
GENERATIONS = 60
ELITE_COUNT = 2
TOURNAMENT_SIZE = 3
CROSSOVER_RATE = 0.5
MUTATION_RATE = 0.2
MUTATION_SIGMA_FRACTION = 0.10
PATIENCE = 15
RANDOM_SEED = 4

CV_BOUNDS = (1.0, 2.0)
CD_BOUNDS = (1.0, 2.0)
CA_BOUNDS = (0.5, 2.0)
FHAT0_BOUNDS = (0.05, 0.30)
FHAT_MIN_BOUNDS = (0.02, 0.30)
FHAT_MAX_BOUNDS = (0.05, 0.40)

BASELINE_PARAMS = {
    "Cv": 1.2,
    "Cd": 1.1,
    "Ca": 1.0,
    "fhat0": 0.18,
    "fhat_min": 0.11,
    "fhat_max": 0.26,
    "n_memory": 500.0,
}


FORCE_MAPPING_NRMSE_KEY = spectral_helpers.FORCE_MAPPING_NRMSE_KEY
VALIDATION_ERROR_KEYS = list(spectral_helpers.VALIDATION_ERROR_KEYS)
ACTIVE_VALIDATION_ERROR_KEYS = [
    (metric_key, metric_label)
    for metric_key, metric_label in VALIDATION_ERROR_KEYS
    if metric_key != FORCE_MAPPING_NRMSE_KEY
]
METRIC_LABELS = tuple(metric_label for _, metric_label in ACTIVE_VALIDATION_ERROR_KEYS)
PARAMETER_NAMES = ("Cv", "Cd", "Ca", "fhat0", "fhat_min", "fhat_max")
OBJECTIVE_PENALTY = 1.0e6
CACHE_KEY_DECIMALS = 12
UR_MATCH_TOL = 1.0e-6
STRICT_FHAT_EPS = 1.0e-6


@dataclass
class EvaluationContext:
    dataset_root: Path
    reference_series: list[dict[str, Any]]
    reference_ur_keys: tuple[float, ...]
    excluded_case_names: tuple[str, ...]
    summary_generation_dt: float
    summary_generation_duration_s: float
    first_case_duration_s: float
    first_case_transient_s: float
    continuation_duration_s: float
    continuation_transient_s: float


@dataclass
class CandidateEvaluation:
    objective: float
    metric_aggregates: dict[str, float]
    grouped_errors: dict[str, dict[float, list[float]]]
    failure: bool
    failure_reason: str | None = None


@dataclass
class OptimizationResult:
    context: EvaluationContext
    baseline_params: dict[str, float]
    baseline_evaluation: CandidateEvaluation
    best_params: dict[str, float]
    best_evaluation: CandidateEvaluation
    history_rows: list[dict[str, Any]]
    completed_generations: int
    cache_size: int
    relative_improvement: float


def reference_ur_value(series: dict[str, Any]) -> float:
    ur_effective = float(series.get("ur_effective", float("nan")))
    if np.isfinite(ur_effective):
        return ur_effective
    ur_value = float(series.get("ur", float("nan")))
    if np.isfinite(ur_value):
        return ur_value
    return float("nan")


def align_reference_series_to_rollout(reference_series: dict[str, Any], rollout: dict[str, Any]) -> dict[str, np.ndarray] | None:
    reference_time = np.asarray(reference_series["time"], dtype=float).reshape(-1)
    rollout_time = np.asarray(rollout["time"], dtype=float).reshape(-1)
    if reference_time.size < 4 or rollout_time.size < 4:
        return None
    reference_time_zeroed = reference_time - float(reference_time[0])
    rollout_time_zeroed = rollout_time - float(rollout_time[0])
    overlap_end = min(float(reference_time_zeroed[-1]), float(rollout_time_zeroed[-1]))
    if not np.isfinite(overlap_end) or overlap_end <= 0.0:
        return None
    rollout_mask = rollout_time_zeroed <= overlap_end
    if np.count_nonzero(rollout_mask) < 4:
        return None
    aligned_time = rollout_time_zeroed[rollout_mask]
    reference_force_coeff = spectral_helpers.series_force_coefficient(reference_series, reference_series["force_per_m"])
    rollout_force_coeff = spectral_helpers.series_force_coefficient(rollout, rollout["force"])
    return {
        "time": aligned_time,
        "y_true": np.interp(
            aligned_time,
            reference_time_zeroed,
            np.asarray(reference_series["displacement"], dtype=float).reshape(-1),
        ),
        "y_pred": np.asarray(rollout["displacement"], dtype=float).reshape(-1)[rollout_mask],
        "force_true": np.interp(aligned_time, reference_time_zeroed, reference_force_coeff),
        "force_pred": rollout_force_coeff[rollout_mask],
    }


def _float_round_key(value: float, *, decimals: int = 6) -> float:
    return float(round(float(value), int(decimals)))


def _matches_excluded_ur(series: dict[str, Any], excluded_ur_values: list[float]) -> bool:
    if not excluded_ur_values:
        return False
    candidates = []
    for key in ("ur_effective", "ur"):
        value = float(series.get(key, float("nan")))
        if np.isfinite(value):
            candidates.append(value)
    for candidate in candidates:
        for excluded in excluded_ur_values:
            if abs(float(candidate) - float(excluded)) <= UR_MATCH_TOL:
                return True
    return False


def _metric_column_name(metric_label: str) -> str:
    return (
        metric_label.lower()
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
        .replace("-", "_")
        .replace(" ", "_")
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(data), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(text)


def _resolve_seed_output_dir(output_dir: Path, random_seed: int) -> Path:
    seed_dir_name = f"seed_{int(random_seed)}"
    resolved_output_dir = Path(output_dir)
    if resolved_output_dir.name == seed_dir_name:
        return resolved_output_dir
    return resolved_output_dir / seed_dir_name


def _render_progress_bar(current: int, total: int, *, width: int = 24) -> str:
    total_int = max(int(total), 1)
    current_int = min(max(int(current), 0), total_int)
    filled = int(width * current_int / total_int)
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _print_progress_bars(
    *,
    generation_idx: int,
    generations: int,
    candidate_index: int,
    population_size: int,
    failure_count: int,
) -> None:
    generation_completed = min(max(int(candidate_index), 0), int(population_size))
    total_completed = int(generation_idx) * int(population_size) + generation_completed
    total_steps = int(generations) * int(population_size)
    generation_bar = _render_progress_bar(generation_completed, int(population_size))
    total_bar = _render_progress_bar(total_completed, total_steps)
    message = (
        f"\rGen {int(generation_idx) + 1}/{int(generations)} "
        f"{generation_bar} {generation_completed}/{int(population_size)} "
        f"| Total {total_bar} {total_completed}/{total_steps} "
        f"| failures={int(failure_count)}"
    )
    sys.stdout.write(message)
    sys.stdout.flush()


def _finish_progress_bars() -> None:
    sys.stdout.write("\n")
    sys.stdout.flush()


def _validated_bounds() -> dict[str, tuple[float, float]]:
    bounds = {
        "Cv": (float(CV_BOUNDS[0]), float(CV_BOUNDS[1])),
        "Cd": (float(CD_BOUNDS[0]), float(CD_BOUNDS[1])),
        "Ca": (float(CA_BOUNDS[0]), float(CA_BOUNDS[1])),
        "fhat0": (float(FHAT0_BOUNDS[0]), float(FHAT0_BOUNDS[1])),
        "fhat_min": (float(FHAT_MIN_BOUNDS[0]), float(FHAT_MIN_BOUNDS[1])),
        "fhat_max": (float(FHAT_MAX_BOUNDS[0]), float(FHAT_MAX_BOUNDS[1])),
    }
    for name, (lower, upper) in bounds.items():
        if not (np.isfinite(lower) and np.isfinite(upper) and lower < upper):
            raise ValueError(f"Invalid bounds for {name}: {(lower, upper)}")
    feasible_fhat0_lower = max(bounds["fhat0"][0], bounds["fhat_min"][0] + STRICT_FHAT_EPS)
    feasible_fhat0_upper = min(bounds["fhat0"][1], bounds["fhat_max"][1] - STRICT_FHAT_EPS)
    if feasible_fhat0_lower > feasible_fhat0_upper:
        raise ValueError("FHAT bounds do not allow any candidate with strict fhat_min < fhat0 < fhat_max.")
    return bounds


def _validate_runtime_settings(
    *,
    loss_mode: str,
    aggregate_weighting: str,
    mass_source: str,
    population_size: int,
    generations: int,
    elite_count: int,
    tournament_size: int,
    crossover_rate: float,
    mutation_rate: float,
    mutation_sigma_fraction: float,
) -> None:
    if str(loss_mode).strip().lower() not in {"mae", "mse"}:
        raise ValueError("LOSS_MODE must be 'mae' or 'mse'.")
    if str(aggregate_weighting).strip().lower() not in {"per_ur_mean", "per_case_mean"}:
        raise ValueError("AGGREGATE_WEIGHTING must be 'per_ur_mean' or 'per_case_mean'.")
    if str(mass_source).strip().lower() not in {"dry", "effective"}:
        raise ValueError("TD_MASS_SOURCE must be 'dry' or 'effective'.")
    if int(population_size) < 2:
        raise ValueError("POPULATION_SIZE must be >= 2.")
    if int(generations) < 1:
        raise ValueError("GENERATIONS must be >= 1.")
    if int(elite_count) < 1 or int(elite_count) > int(population_size):
        raise ValueError("ELITE_COUNT must be in [1, POPULATION_SIZE].")
    if int(tournament_size) < 2:
        raise ValueError("TOURNAMENT_SIZE must be >= 2.")
    for name, value in {
        "CROSSOVER_RATE": float(crossover_rate),
        "MUTATION_RATE": float(mutation_rate),
        "MUTATION_SIGMA_FRACTION": float(mutation_sigma_fraction),
    }.items():
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative.")
    if float(crossover_rate) > 1.0:
        raise ValueError("CROSSOVER_RATE must be <= 1.")
    if float(mutation_rate) > 1.0:
        raise ValueError("MUTATION_RATE must be <= 1.")


def candidate_from_params(params: dict[str, float]) -> np.ndarray:
    return np.asarray([float(params[name]) for name in PARAMETER_NAMES], dtype=float)


def _repair_strict_fhat_triplet(
    *,
    fhat_min: float,
    fhat0: float,
    fhat_max: float,
    bounds: dict[str, tuple[float, float]],
) -> tuple[float, float, float]:
    sorted_triplet = np.sort(np.asarray([fhat_min, fhat0, fhat_max], dtype=float))
    fhat_min_value = float(sorted_triplet[0])
    fhat0_value = float(sorted_triplet[1])
    fhat_max_value = float(sorted_triplet[2])

    fhat_min_lower, fhat_min_upper = bounds["fhat_min"]
    fhat0_lower, fhat0_upper = bounds["fhat0"]
    fhat_max_lower, fhat_max_upper = bounds["fhat_max"]

    feasible_fhat0_lower = max(float(fhat0_lower), float(fhat_min_lower) + STRICT_FHAT_EPS)
    feasible_fhat0_upper = min(float(fhat0_upper), float(fhat_max_upper) - STRICT_FHAT_EPS)
    if feasible_fhat0_lower > feasible_fhat0_upper:
        raise ValueError("FHAT bounds do not allow any candidate with strict fhat_min < fhat0 < fhat_max.")

    repaired_fhat0 = float(np.clip(fhat0_value, feasible_fhat0_lower, feasible_fhat0_upper))

    repaired_fhat_min_upper = min(float(fhat_min_upper), repaired_fhat0 - STRICT_FHAT_EPS)
    repaired_fhat_max_lower = max(float(fhat_max_lower), repaired_fhat0 + STRICT_FHAT_EPS)
    if repaired_fhat_min_upper < float(fhat_min_lower) or repaired_fhat_max_lower > float(fhat_max_upper):
        raise ValueError("Could not satisfy strict fhat ordering within the configured bounds.")

    repaired_fhat_min = float(np.clip(fhat_min_value, float(fhat_min_lower), repaired_fhat_min_upper))
    repaired_fhat_max = float(np.clip(fhat_max_value, repaired_fhat_max_lower, float(fhat_max_upper)))
    return repaired_fhat_min, repaired_fhat0, repaired_fhat_max


def repair_candidate(candidate: np.ndarray, bounds: dict[str, tuple[float, float]]) -> np.ndarray:
    repaired = np.asarray(candidate, dtype=float).reshape(-1).copy()
    if repaired.size != len(PARAMETER_NAMES):
        raise ValueError(f"Expected candidate with {len(PARAMETER_NAMES)} values, got {repaired.size}.")
    for idx, name in enumerate(PARAMETER_NAMES):
        lower, upper = bounds[name]
        if not np.isfinite(repaired[idx]):
            repaired[idx] = 0.5 * (lower + upper)
        repaired[idx] = float(np.clip(repaired[idx], lower, upper))

    idx_fhat0 = PARAMETER_NAMES.index("fhat0")
    idx_fhat_min = PARAMETER_NAMES.index("fhat_min")
    idx_fhat_max = PARAMETER_NAMES.index("fhat_max")
    repaired[idx_fhat_min], repaired[idx_fhat0], repaired[idx_fhat_max] = _repair_strict_fhat_triplet(
        fhat_min=float(repaired[idx_fhat_min]),
        fhat0=float(repaired[idx_fhat0]),
        fhat_max=float(repaired[idx_fhat_max]),
        bounds=bounds,
    )
    return repaired


def td_params_from_candidate(candidate: np.ndarray, *, baseline_params: dict[str, float], bounds: dict[str, tuple[float, float]]) -> dict[str, float]:
    repaired = repair_candidate(candidate, bounds)
    params = dict(baseline_params)
    for idx, name in enumerate(PARAMETER_NAMES):
        params[name] = float(repaired[idx])
    return params


def reduce_grouped_metric_values(grouped_errors: dict[float, list[float]], aggregate_weighting: str) -> float:
    weighting = str(aggregate_weighting).strip().lower()
    if weighting not in {"per_ur_mean", "per_case_mean"}:
        raise ValueError("aggregate_weighting must be 'per_ur_mean' or 'per_case_mean'.")
    per_ur_means: list[float] = []
    flat_values: list[np.ndarray] = []
    for ur_key in sorted(grouped_errors.keys()):
        values = np.asarray(grouped_errors[float(ur_key)], dtype=float).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        per_ur_means.append(float(np.mean(values)))
        flat_values.append(values)
    if weighting == "per_ur_mean":
        return float(np.mean(np.asarray(per_ur_means, dtype=float))) if per_ur_means else float("nan")
    if not flat_values:
        return float("nan")
    return float(np.mean(np.concatenate(flat_values)))


def combine_metric_aggregates(metric_aggregates: dict[str, float], loss_mode: str) -> float:
    values = np.asarray([float(metric_aggregates[label]) for label in METRIC_LABELS], dtype=float)
    if values.size != len(METRIC_LABELS) or not np.all(np.isfinite(values)):
        return float("nan")
    mode = str(loss_mode).strip().lower()
    if mode == "mae":
        return float(np.mean(values))
    if mode == "mse":
        return float(np.mean(values * values))
    raise ValueError("loss_mode must be 'mae' or 'mse'.")


def build_evaluation_context(
    *,
    dataset_root: Path | None = DATASET_ROOT,
    excluded_ur_values: list[float] | None = None,
    max_cases: int | None = MAX_CASES,
    transient_seconds: float = VIVANA_TRANSIENT_SECONDS,
    first_case_extra_seconds: float = VIVANA_FIRST_CASE_EXTRA_SECONDS,
    continuation_transient_seconds: float = VIVANA_CONTINUATION_TRANSIENT_SECONDS,
    continuation_kept_seconds: float = VIVANA_CONTINUATION_KEPT_SECONDS,
) -> EvaluationContext:
    candidate_roots = [] if dataset_root is None else [Path(dataset_root)]
    resolved_dataset_root = resolve_dataset_root(
        Path.cwd(),
        repo_root=REPO_ROOT,
        extra_candidates=candidate_roots,
    )
    excluded = [float(value) for value in (excluded_ur_values or [])]
    files = iter_all_npz_files(resolved_dataset_root)
    reference_series: list[dict[str, Any]] = []
    excluded_case_names: list[str] = []
    for npz_path in files:
        series = load_series(npz_path)
        if not np.isfinite(reference_ur_value(series)):
            continue
        if _matches_excluded_ur(series, excluded):
            excluded_case_names.append(str(series["name"]))
            continue
        reference_series.append(series)
    reference_series = sorted(reference_series, key=reference_ur_value)
    if max_cases is not None:
        reference_series = reference_series[: int(max_cases)]
    if not reference_series:
        raise ValueError("No finite-U_r CFD reference runs are available for optimization.")

    summary_generation_dt, summary_generation_duration_s, _, _ = spectral_helpers.global_generation_grid(
        reference_series,
        transient_seconds=float(transient_seconds),
    )
    first_case_duration_s = float(summary_generation_duration_s + float(first_case_extra_seconds))
    first_case_transient_s = float(float(transient_seconds) + float(first_case_extra_seconds))
    continuation_duration_s = float(float(continuation_transient_seconds) + float(continuation_kept_seconds))
    reference_ur_keys = tuple(
        sorted({_float_round_key(reference_ur_value(series)) for series in reference_series})
    )
    return EvaluationContext(
        dataset_root=resolved_dataset_root,
        reference_series=reference_series,
        reference_ur_keys=reference_ur_keys,
        excluded_case_names=tuple(excluded_case_names),
        summary_generation_dt=float(summary_generation_dt),
        summary_generation_duration_s=float(summary_generation_duration_s),
        first_case_duration_s=first_case_duration_s,
        first_case_transient_s=first_case_transient_s,
        continuation_duration_s=continuation_duration_s,
        continuation_transient_s=float(continuation_transient_seconds),
    )


def evaluate_vivana_td_params(
    td_params: dict[str, float],
    *,
    context: EvaluationContext,
    aggregate_weighting: str = AGGREGATE_WEIGHTING,
    loss_mode: str = LOSS_MODE,
    mass_source: str = TD_MASS_SOURCE,
    td_memory_tau_s: float | str | None = TD_MEMORY_TAU_S,
) -> CandidateEvaluation:
    grouped_errors: dict[str, dict[float, list[float]]] = {label: {} for label in METRIC_LABELS}
    try:
        continuation_state: dict[str, Any] | None = None
        for reference_idx, reference_series in enumerate(context.reference_series):
            target_ur = reference_ur_value(reference_series)
            ur_key = _float_round_key(target_ur)
            if reference_idx == 0:
                rollout_duration_s = context.first_case_duration_s
                rollout_transient_seconds = context.first_case_transient_s
            else:
                rollout_duration_s = context.continuation_duration_s
                rollout_transient_seconds = context.continuation_transient_s
            viv_rollout = spectral_helpers.generate_vivana_summary_rollout(
                reference_series,
                target_ur,
                generation_dt=context.summary_generation_dt,
                generation_duration_s=rollout_duration_s,
                transient_seconds=rollout_transient_seconds,
                td_params=td_params,
                mass_source=str(mass_source).strip().lower(),
                initial_state=continuation_state,
                td_memory_tau_s=td_memory_tau_s,
            )
            continuation_state = dict(viv_rollout["final_state"])
            aligned = align_reference_series_to_rollout(reference_series, viv_rollout)
            if aligned is None:
                raise ValueError(f"Could not align reference series '{reference_series['name']}' to the generated rollout.")
            error_metrics = spectral_helpers.compute_validation_style_error_metrics(
                time=aligned["time"],
                y_true=aligned["y_true"],
                y_pred=aligned["y_pred"],
                force_true=aligned["force_true"],
                force_pred=aligned["force_pred"],
            )
            for metric_key, metric_label in ACTIVE_VALIDATION_ERROR_KEYS:
                value = float(error_metrics.get(metric_key, float("nan")))
                if np.isfinite(value):
                    grouped_errors[metric_label].setdefault(ur_key, []).append(value)
    except Exception as exc:
        return CandidateEvaluation(
            objective=float(OBJECTIVE_PENALTY),
            metric_aggregates={label: float("nan") for label in METRIC_LABELS},
            grouped_errors=grouped_errors,
            failure=True,
            failure_reason=str(exc),
        )

    metric_aggregates = {
        label: reduce_grouped_metric_values(grouped_errors[label], aggregate_weighting)
        for label in METRIC_LABELS
    }
    objective = combine_metric_aggregates(metric_aggregates, loss_mode)
    if not np.isfinite(objective):
        return CandidateEvaluation(
            objective=float(OBJECTIVE_PENALTY),
            metric_aggregates=metric_aggregates,
            grouped_errors=grouped_errors,
            failure=True,
            failure_reason="No finite aggregate objective could be computed.",
        )
    return CandidateEvaluation(
        objective=float(objective),
        metric_aggregates=metric_aggregates,
        grouped_errors=grouped_errors,
        failure=False,
        failure_reason=None,
    )


def _candidate_cache_key(candidate: np.ndarray) -> tuple[float, ...]:
    candidate_arr = np.asarray(candidate, dtype=float).reshape(-1)
    return tuple(float(np.round(value, CACHE_KEY_DECIMALS)) for value in candidate_arr)


def _evaluate_candidate_cached(
    candidate: np.ndarray,
    *,
    evaluation_cache: dict[tuple[float, ...], CandidateEvaluation],
    bounds: dict[str, tuple[float, float]],
    baseline_params: dict[str, float],
    context: EvaluationContext,
    aggregate_weighting: str,
    loss_mode: str,
    mass_source: str,
    td_memory_tau_s: float | str | None,
) -> tuple[np.ndarray, CandidateEvaluation, bool]:
    repaired = repair_candidate(candidate, bounds)
    cache_key = _candidate_cache_key(repaired)
    if cache_key in evaluation_cache:
        return repaired, evaluation_cache[cache_key], True
    td_params = td_params_from_candidate(repaired, baseline_params=baseline_params, bounds=bounds)
    evaluation = evaluate_vivana_td_params(
        td_params,
        context=context,
        aggregate_weighting=aggregate_weighting,
        loss_mode=loss_mode,
        mass_source=mass_source,
        td_memory_tau_s=td_memory_tau_s,
    )
    evaluation_cache[cache_key] = evaluation
    return repaired, evaluation, False


def _build_initial_population(
    *,
    rng: np.random.Generator,
    population_size: int,
    bounds: dict[str, tuple[float, float]],
    baseline_params: dict[str, float],
) -> np.ndarray:
    population = np.empty((int(population_size), len(PARAMETER_NAMES)), dtype=float)
    population[0] = repair_candidate(candidate_from_params(baseline_params), bounds)
    for idx in range(1, int(population_size)):
        sample = []
        for name in PARAMETER_NAMES:
            lower, upper = bounds[name]
            sample.append(float(rng.uniform(lower, upper)))
        population[idx] = repair_candidate(np.asarray(sample, dtype=float), bounds)
    return population


def _tournament_select(
    *,
    rng: np.random.Generator,
    objectives: np.ndarray,
    tournament_size: int,
) -> int:
    candidate_indices = rng.integers(0, objectives.size, size=int(tournament_size))
    best_local = int(candidate_indices[int(np.argmin(objectives[candidate_indices]))])
    return best_local


def _crossover(
    *,
    rng: np.random.Generator,
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    crossover_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    if float(rng.random()) >= float(crossover_rate):
        return parent_a.copy(), parent_b.copy()
    mask = rng.random(parent_a.size) < 0.5
    child_a = np.where(mask, parent_a, parent_b)
    child_b = np.where(mask, parent_b, parent_a)
    return child_a.astype(float, copy=False), child_b.astype(float, copy=False)


def _mutate(
    *,
    rng: np.random.Generator,
    candidate: np.ndarray,
    bounds: dict[str, tuple[float, float]],
    mutation_rate: float,
    mutation_sigma_fraction: float,
) -> np.ndarray:
    mutated = np.asarray(candidate, dtype=float).reshape(-1).copy()
    for idx, name in enumerate(PARAMETER_NAMES):
        if float(rng.random()) >= float(mutation_rate):
            continue
        lower, upper = bounds[name]
        sigma = float(mutation_sigma_fraction) * (upper - lower)
        if sigma > 0.0:
            mutated[idx] += float(rng.normal(0.0, sigma))
    return repair_candidate(mutated, bounds)


def _build_history_row(
    *,
    generation: int,
    candidate_index: int,
    candidate: np.ndarray,
    evaluation: CandidateEvaluation,
    cache_hit: bool,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "generation": int(generation),
        "candidate_index": int(candidate_index),
        "cache_hit": bool(cache_hit),
        "failure": bool(evaluation.failure),
        "failure_reason": "" if evaluation.failure_reason is None else str(evaluation.failure_reason),
        "objective": float(evaluation.objective),
    }
    for idx, name in enumerate(PARAMETER_NAMES):
        row[name] = float(candidate[idx])
    for label in METRIC_LABELS:
        row[_metric_column_name(label)] = float(evaluation.metric_aggregates.get(label, float("nan")))
    return row


def _format_candidate_params(
    candidate: np.ndarray,
    *,
    baseline_params: dict[str, float],
    bounds: dict[str, tuple[float, float]],
) -> str:
    params = td_params_from_candidate(candidate, baseline_params=baseline_params, bounds=bounds)
    return str(params)


def run_genetic_optimization(
    *,
    context: EvaluationContext,
    baseline_params: dict[str, float],
    output_dir: Path | None = OUTPUT_DIR,
    loss_mode: str = LOSS_MODE,
    aggregate_weighting: str = AGGREGATE_WEIGHTING,
    mass_source: str = TD_MASS_SOURCE,
    td_memory_tau_s: float | str | None = TD_MEMORY_TAU_S,
    population_size: int = POPULATION_SIZE,
    generations: int = GENERATIONS,
    elite_count: int = ELITE_COUNT,
    tournament_size: int = TOURNAMENT_SIZE,
    crossover_rate: float = CROSSOVER_RATE,
    mutation_rate: float = MUTATION_RATE,
    mutation_sigma_fraction: float = MUTATION_SIGMA_FRACTION,
    patience: int = PATIENCE,
    random_seed: int = RANDOM_SEED,
    excluded_ur_values: list[float] | None = None,
    max_cases: int | None = MAX_CASES,
) -> OptimizationResult:
    _validate_runtime_settings(
        loss_mode=loss_mode,
        aggregate_weighting=aggregate_weighting,
        mass_source=mass_source,
        population_size=population_size,
        generations=generations,
        elite_count=elite_count,
        tournament_size=tournament_size,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        mutation_sigma_fraction=mutation_sigma_fraction,
    )
    bounds = _validated_bounds()
    rng = np.random.default_rng(int(random_seed))
    evaluation_cache: dict[tuple[float, ...], CandidateEvaluation] = {}
    history_rows: list[dict[str, Any]] = []

    baseline_candidate, baseline_evaluation, baseline_cache_hit = _evaluate_candidate_cached(
        candidate_from_params(baseline_params),
        evaluation_cache=evaluation_cache,
        bounds=bounds,
        baseline_params=baseline_params,
        context=context,
        aggregate_weighting=aggregate_weighting,
        loss_mode=loss_mode,
        mass_source=mass_source,
        td_memory_tau_s=td_memory_tau_s,
    )
    history_rows.append(
        _build_history_row(
            generation=-1,
            candidate_index=0,
            candidate=baseline_candidate,
            evaluation=baseline_evaluation,
            cache_hit=baseline_cache_hit,
        )
    )

    population = _build_initial_population(
        rng=rng,
        population_size=population_size,
        bounds=bounds,
        baseline_params=baseline_params,
    )
    best_candidate = baseline_candidate.copy()
    best_evaluation = baseline_evaluation
    completed_generations = 0
    stagnant_generations = 0

    for generation_idx in range(int(generations)):
        completed_generations = generation_idx + 1
        evaluated_population = np.empty_like(population)
        evaluations: list[CandidateEvaluation] = []
        failure_count = 0
        _print_progress_bars(
            generation_idx=generation_idx,
            generations=int(generations),
            candidate_index=0,
            population_size=int(population_size),
            failure_count=failure_count,
        )
        for candidate_index, candidate in enumerate(population):
            repaired_candidate, evaluation, cache_hit = _evaluate_candidate_cached(
                candidate,
                evaluation_cache=evaluation_cache,
                bounds=bounds,
                baseline_params=baseline_params,
                context=context,
                aggregate_weighting=aggregate_weighting,
                loss_mode=loss_mode,
                mass_source=mass_source,
                td_memory_tau_s=td_memory_tau_s,
            )
            evaluated_population[candidate_index] = repaired_candidate
            evaluations.append(evaluation)
            failure_count += int(evaluation.failure)
            history_rows.append(
                _build_history_row(
                    generation=generation_idx,
                    candidate_index=candidate_index,
                    candidate=repaired_candidate,
                    evaluation=evaluation,
                    cache_hit=cache_hit,
                )
            )
            _print_progress_bars(
                generation_idx=generation_idx,
                generations=int(generations),
                candidate_index=candidate_index + 1,
                population_size=int(population_size),
                failure_count=failure_count,
            )

        _finish_progress_bars()
        objectives = np.asarray([float(evaluation.objective) for evaluation in evaluations], dtype=float)
        ranking = np.argsort(objectives)
        generation_best_idx = int(ranking[0])
        generation_best_candidate = evaluated_population[generation_best_idx].copy()
        generation_best_evaluation = evaluations[generation_best_idx]
        if float(generation_best_evaluation.objective) < float(best_evaluation.objective):
            best_candidate = generation_best_candidate
            best_evaluation = generation_best_evaluation
            stagnant_generations = 0
        else:
            stagnant_generations += 1

        print(
            f"Generation {generation_idx + 1}/{int(generations)} | "
            f"best={float(generation_best_evaluation.objective):.6g} | "
            f"overall_best={float(best_evaluation.objective):.6g} | "
            f"failures={failure_count} | "
            f"best_params={_format_candidate_params(generation_best_candidate, baseline_params=baseline_params, bounds=bounds)}"
        )

        if generation_idx + 1 >= int(generations):
            break
        if int(patience) > 0 and stagnant_generations >= int(patience):
            print(f"Early stop after {generation_idx + 1} generation(s) without improvement.")
            break

        next_population = [evaluated_population[int(idx)].copy() for idx in ranking[: int(elite_count)]]
        while len(next_population) < int(population_size):
            parent_a = evaluated_population[
                _tournament_select(rng=rng, objectives=objectives, tournament_size=tournament_size)
            ]
            parent_b = evaluated_population[
                _tournament_select(rng=rng, objectives=objectives, tournament_size=tournament_size)
            ]
            child_a, child_b = _crossover(
                rng=rng,
                parent_a=parent_a,
                parent_b=parent_b,
                crossover_rate=crossover_rate,
            )
            next_population.append(
                _mutate(
                    rng=rng,
                    candidate=child_a,
                    bounds=bounds,
                    mutation_rate=mutation_rate,
                    mutation_sigma_fraction=mutation_sigma_fraction,
                )
            )
            if len(next_population) < int(population_size):
                next_population.append(
                    _mutate(
                        rng=rng,
                        candidate=child_b,
                        bounds=bounds,
                        mutation_rate=mutation_rate,
                        mutation_sigma_fraction=mutation_sigma_fraction,
                    )
                )
        population = np.asarray(next_population, dtype=float)

    best_params = td_params_from_candidate(best_candidate, baseline_params=baseline_params, bounds=bounds)
    baseline_objective = float(baseline_evaluation.objective)
    best_objective = float(best_evaluation.objective)
    if np.isfinite(baseline_objective) and baseline_objective > 0.0 and np.isfinite(best_objective):
        relative_improvement = float((baseline_objective - best_objective) / baseline_objective)
    else:
        relative_improvement = float("nan")

    result = OptimizationResult(
        context=context,
        baseline_params=td_params_from_candidate(
            baseline_candidate,
            baseline_params=baseline_params,
            bounds=bounds,
        ),
        baseline_evaluation=baseline_evaluation,
        best_params=best_params,
        best_evaluation=best_evaluation,
        history_rows=history_rows,
        completed_generations=completed_generations,
        cache_size=len(evaluation_cache),
        relative_improvement=relative_improvement,
    )
    if output_dir is not None:
        seed_output_dir = _resolve_seed_output_dir(Path(output_dir), int(random_seed))
        write_optimization_outputs(
            result,
            output_dir=seed_output_dir,
            base_output_dir=Path(output_dir),
            loss_mode=loss_mode,
            aggregate_weighting=aggregate_weighting,
            mass_source=mass_source,
            td_memory_tau_s=td_memory_tau_s,
            population_size=population_size,
            generations=generations,
            elite_count=elite_count,
            tournament_size=tournament_size,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            mutation_sigma_fraction=mutation_sigma_fraction,
            patience=patience,
            random_seed=random_seed,
            bounds=bounds,
            excluded_ur_values=[] if excluded_ur_values is None else list(excluded_ur_values),
            max_cases=max_cases,
        )
    return result


def write_optimization_outputs(
    result: OptimizationResult,
    *,
    output_dir: Path,
    base_output_dir: Path,
    loss_mode: str,
    aggregate_weighting: str,
    mass_source: str,
    td_memory_tau_s: float | str | None,
    population_size: int,
    generations: int,
    elite_count: int,
    tournament_size: int,
    crossover_rate: float,
    mutation_rate: float,
    mutation_sigma_fraction: float,
    patience: int,
    random_seed: int,
    bounds: dict[str, tuple[float, float]],
    excluded_ur_values: list[float],
    max_cases: int | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot = {
        "dataset_root": result.context.dataset_root,
        "base_output_dir": base_output_dir,
        "output_dir": output_dir,
        "loss_mode": str(loss_mode).strip().lower(),
        "aggregate_weighting": str(aggregate_weighting).strip().lower(),
        "mass_source": str(mass_source).strip().lower(),
        "td_memory_tau_s": td_memory_tau_s,
        "excluded_ur_values": [float(value) for value in excluded_ur_values],
        "max_cases": None if max_cases is None else int(max_cases),
        "vivana_transient_seconds": float(VIVANA_TRANSIENT_SECONDS),
        "vivana_first_case_extra_seconds": float(VIVANA_FIRST_CASE_EXTRA_SECONDS),
        "vivana_continuation_transient_seconds": float(VIVANA_CONTINUATION_TRANSIENT_SECONDS),
        "vivana_continuation_kept_seconds": float(VIVANA_CONTINUATION_KEPT_SECONDS),
        "population_size": int(population_size),
        "generations": int(generations),
        "elite_count": int(elite_count),
        "tournament_size": int(tournament_size),
        "crossover_rate": float(crossover_rate),
        "mutation_rate": float(mutation_rate),
        "mutation_sigma_fraction": float(mutation_sigma_fraction),
        "patience": int(patience),
        "random_seed": int(random_seed),
        "bounds": bounds,
        "baseline_params": result.baseline_params,
        "reference_ur_keys": result.context.reference_ur_keys,
        "reference_case_names": [str(series["name"]) for series in result.context.reference_series],
        "excluded_case_names": list(result.context.excluded_case_names),
        "summary_generation_dt": float(result.context.summary_generation_dt),
        "summary_generation_duration_s": float(result.context.summary_generation_duration_s),
        "first_case_duration_s": float(result.context.first_case_duration_s),
        "first_case_transient_s": float(result.context.first_case_transient_s),
        "continuation_duration_s": float(result.context.continuation_duration_s),
        "continuation_transient_s": float(result.context.continuation_transient_s),
    }
    best_result = {
        "best_params": result.best_params,
        "best_objective": float(result.best_evaluation.objective),
        "best_metric_aggregates": result.best_evaluation.metric_aggregates,
        "baseline_params": result.baseline_params,
        "baseline_objective": float(result.baseline_evaluation.objective),
        "baseline_metric_aggregates": result.baseline_evaluation.metric_aggregates,
        "relative_improvement": float(result.relative_improvement),
        "completed_generations": int(result.completed_generations),
        "cache_size": int(result.cache_size),
        "reference_ur_keys": result.context.reference_ur_keys,
        "excluded_case_names": list(result.context.excluded_case_names),
    }
    summary_lines = [
        f"Dataset root: {result.context.dataset_root}",
        f"Base output dir: {base_output_dir}",
        f"Run output dir: {output_dir}",
        f"Random seed: {int(random_seed)}",
        f"Included cases: {len(result.context.reference_series)}",
        f"Effective U_r keys: {list(result.context.reference_ur_keys)}",
        (
            "Excluded cases: "
            + (", ".join(result.context.excluded_case_names) if result.context.excluded_case_names else "none")
        ),
        f"Baseline objective: {float(result.baseline_evaluation.objective):.6g}",
        f"Best objective: {float(result.best_evaluation.objective):.6g}",
        f"Relative improvement: {float(result.relative_improvement):.6%}" if np.isfinite(result.relative_improvement) else "Relative improvement: nan",
        f"Best params: {result.best_params}",
    ]
    _write_json(output_dir / "run_config.json", config_snapshot)
    _write_json(output_dir / "best_result.json", best_result)
    _write_csv(output_dir / "history.csv", result.history_rows)
    _write_text(output_dir / "summary.txt", "\n".join(summary_lines) + "\n")


def run_configured_optimization(
    *,
    dataset_root: Path | None = DATASET_ROOT,
    output_dir: Path | None = OUTPUT_DIR,
    loss_mode: str = LOSS_MODE,
    aggregate_weighting: str = AGGREGATE_WEIGHTING,
    mass_source: str = TD_MASS_SOURCE,
    td_memory_tau_s: float | str | None = TD_MEMORY_TAU_S,
    excluded_ur_values: list[float] | None = None,
    max_cases: int | None = MAX_CASES,
    population_size: int = POPULATION_SIZE,
    generations: int = GENERATIONS,
    elite_count: int = ELITE_COUNT,
    tournament_size: int = TOURNAMENT_SIZE,
    crossover_rate: float = CROSSOVER_RATE,
    mutation_rate: float = MUTATION_RATE,
    mutation_sigma_fraction: float = MUTATION_SIGMA_FRACTION,
    patience: int = PATIENCE,
    random_seed: int = RANDOM_SEED,
) -> OptimizationResult:
    effective_excluded_ur_values = EXCLUDED_UR_VALUES if excluded_ur_values is None else excluded_ur_values
    context = build_evaluation_context(
        dataset_root=dataset_root,
        excluded_ur_values=effective_excluded_ur_values,
        max_cases=max_cases,
        transient_seconds=VIVANA_TRANSIENT_SECONDS,
        first_case_extra_seconds=VIVANA_FIRST_CASE_EXTRA_SECONDS,
        continuation_transient_seconds=VIVANA_CONTINUATION_TRANSIENT_SECONDS,
        continuation_kept_seconds=VIVANA_CONTINUATION_KEPT_SECONDS,
    )
    print(f"Resolved dataset root: {context.dataset_root}")
    print(f"Included cases: {len(context.reference_series)}")
    print(f"Effective U_r keys: {list(context.reference_ur_keys)}")
    if context.excluded_case_names:
        print(f"Excluded cases: {', '.join(context.excluded_case_names)}")
    else:
        print("Excluded cases: none")
    result = run_genetic_optimization(
        context=context,
        baseline_params=dict(BASELINE_PARAMS),
        output_dir=output_dir,
        loss_mode=loss_mode,
        aggregate_weighting=aggregate_weighting,
        mass_source=mass_source,
        td_memory_tau_s=td_memory_tau_s,
        population_size=population_size,
        generations=generations,
        elite_count=elite_count,
        tournament_size=tournament_size,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        mutation_sigma_fraction=mutation_sigma_fraction,
        patience=patience,
        random_seed=random_seed,
        excluded_ur_values=list(effective_excluded_ur_values),
        max_cases=max_cases,
    )
    print(f"Baseline objective: {float(result.baseline_evaluation.objective):.6g}")
    print(f"Best objective: {float(result.best_evaluation.objective):.6g}")
    print(f"Best params: {result.best_params}")
    if np.isfinite(result.relative_improvement):
        print(f"Relative improvement: {float(result.relative_improvement):.6%}")
    else:
        print("Relative improvement: nan")
    if output_dir is not None:
        print(f"Wrote optimization outputs to {Path(output_dir)}")
    return result


def main() -> None:
    run_configured_optimization()


if __name__ == "__main__":
    main()
