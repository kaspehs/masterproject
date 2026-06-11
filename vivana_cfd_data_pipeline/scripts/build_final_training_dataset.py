"""Build the final Vivana-TD training dataset from exported CFD NPZ files.

Pipeline:
  1. Replay Vivana-TD hidden state on each CFD export and trim burn-in.
  2. Assemble a delivery-style dataset root with train/ and val_seen/ splits.
  3. Optionally exclude one or more label U_r values from the final splits.
  4. Generate surrogate validation points from retained training anchors.
     Dataset-level U_r exclusions are also excluded from surrogate anchors.
  5. Optionally generate leave-one-U_r-out folders from the same final dataset.

The script intentionally writes to new ``*_final`` folders so older experiment
outputs remain available for comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


DATA_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DATA_ROOT.parent
for _path in (REPO_ROOT, DATA_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from vivana_cfd_data_pipeline.scripts import create_loo_ur_sets as loo
from vivana_cfd_data_pipeline.scripts import generate_surrogate_validation_points as surrogate


INPUT_NPZ_DIR = DATA_ROOT / "generated" / "cfd_npz_exports"
BURNIN_WORK_DIR = DATA_ROOT / "generated" / "td_burnin_trimmed_final_flat"
FINAL_DATASET_DIR = DATA_ROOT / "generated" / "final_dataset"
LOO_OUTPUT_ROOT = DATA_ROOT / "generated" / "loo_ur_td_burnin_trimmed_final"

METADATA_PATH = DATA_ROOT / "metadata" / "CFD_metadata.csv"
BURNIN_DIAGNOSTIC_DIR = DATA_ROOT / "outputs" / "td_preprocess_diagnostics_final"

UR_SOURCE = "label"
VAL_SEEN_POLICY = "first_per_ur"
# Optional label U_r exclusions for the main final train/val_seen dataset.
# These are the source of truth for final-dataset surrogate anchor exclusions.
DATASET_EXCLUDE_URS: tuple[float, ...] = ()
CREATE_LOO_DATASETS = False
HELD_OUT_URS = loo.HELD_OUT_URS

# Final Vivana-TD replay configuration used when making the training NPZ files.
# Keep this as the delivery source of truth for the TD hidden-state replay.
VIVANA_TD_PARAMS = {
    "Cv": 1.2,
    "Cd": 1.1,
    "Ca": 1.0,
    "C": 0.0,
    "fhat_min": 0.11,
    "fhat0": 0.18,
    "fhat_max": 0.26,
}

# Angle wrapping convention. "principal" means phases are wrapped to [-pi, pi).
PHASE_WRAP_CONVENTION = "principal"

# Force evaluation convention. "current" uses phi_vy at the start of the step;
# "advanced"/"next" uses phi_vy after the phase update.
FORCE_PHASE_CONVENTION = "current"

# Phase-spread initializations used to decide where the burn-in has converged.
BURNIN_THETA0_VALUES = np.linspace(-np.pi / 2.0, np.pi / 2.0, 7, endpoint=True)

# TD memory rule used during replay.
TD_MEMORY_MODE = "tau_over_tref"  # "fixed_n_memory" | "fixed_tau" | "tau_over_tref"
TD_TAU_OVER_TREF = 4.0
TD_MEMORY_TAU_S: float | None = None
N_MEMORY_FALLBACK = 500


@dataclass(frozen=True)
class SplitRow:
    split: str
    source: str
    destination: str
    label_ur: float
    reason: str


@dataclass(frozen=True)
class ExcludedRow:
    source: str
    label_ur: float
    reason: str


def _clean_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} already exists. Use --overwrite to rebuild it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _ensure_input_npzs(input_dir: Path) -> list[Path]:
    paths = sorted(Path(input_dir).glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No input CFD NPZ files found in {input_dir}")
    return paths


def _validate_vivana_td_config() -> None:
    required = {"Cv", "Cd", "Ca", "C", "fhat_min", "fhat0", "fhat_max"}
    missing = sorted(required - set(VIVANA_TD_PARAMS))
    if missing:
        raise KeyError(f"VIVANA_TD_PARAMS is missing required keys: {missing}")

    values = {key: float(VIVANA_TD_PARAMS[key]) for key in required}
    nonfinite = sorted(key for key, value in values.items() if not np.isfinite(value))
    if nonfinite:
        raise ValueError(f"VIVANA_TD_PARAMS has non-finite values for: {nonfinite}")
    if not (values["fhat_min"] <= values["fhat0"] <= values["fhat_max"]):
        raise ValueError("Require fhat_min <= fhat0 <= fhat_max in VIVANA_TD_PARAMS.")

    if PHASE_WRAP_CONVENTION != "principal":
        raise ValueError("Only PHASE_WRAP_CONVENTION='principal' is currently supported.")
    if FORCE_PHASE_CONVENTION not in {"current", "old", "previous", "advanced", "next", "new"}:
        raise ValueError("FORCE_PHASE_CONVENTION must be 'current' or 'advanced'/'next'.")
    if TD_MEMORY_MODE not in {"fixed_n_memory", "fixed_tau", "tau_over_tref"}:
        raise ValueError("TD_MEMORY_MODE must be one of: fixed_n_memory, fixed_tau, tau_over_tref.")


def _apply_vivana_td_config(burnin_module) -> None:
    """Push this script's delivery TD config into the reused burn-in modules."""
    _validate_vivana_td_config()

    burnin_config = burnin_module.burnin_config
    burnin_config.CV_VALUES = [float(VIVANA_TD_PARAMS["Cv"])]
    burnin_config.CD_VALUES = [float(VIVANA_TD_PARAMS["Cd"])]
    burnin_config.CA_VALUES = [float(VIVANA_TD_PARAMS["Ca"])]
    burnin_config.DAMPING_C_VALUES = [float(VIVANA_TD_PARAMS["C"])]
    burnin_config.FHAT_MIN_VALUES = [float(VIVANA_TD_PARAMS["fhat_min"])]
    burnin_config.FHAT0_VALUES = [float(VIVANA_TD_PARAMS["fhat0"])]
    burnin_config.FHAT_MAX_VALUES = [float(VIVANA_TD_PARAMS["fhat_max"])]
    burnin_config.PHASE_WRAP = str(PHASE_WRAP_CONVENTION)
    burnin_config.THETA0_VALUES = np.asarray(BURNIN_THETA0_VALUES, dtype=float)
    burnin_config.N_MEMORY = int(N_MEMORY_FALLBACK)
    burnin_config.TD_MEMORY_MODE = str(TD_MEMORY_MODE)
    burnin_config.TD_TAU_OVER_TREF = float(TD_TAU_OVER_TREF)
    burnin_config.TD_MEMORY_TAU_S = TD_MEMORY_TAU_S

    burnin_module.TD_MEMORY_MODE = str(TD_MEMORY_MODE)
    burnin_module.TD_TAU_OVER_TREF = float(TD_TAU_OVER_TREF)
    burnin_module.TD_MEMORY_TAU_S = TD_MEMORY_TAU_S
    burnin_module.FORCE_PHASE_CONVENTION = str(FORCE_PHASE_CONVENTION)


def _run_burnin_preprocessing(
    *,
    input_dir: Path,
    output_dir: Path,
    overwrite: bool,
    write_diagnostics: bool,
) -> None:
    from vivana_cfd_data_pipeline.scripts import prepare_cfd_npz_for_td_training as burnin

    _ensure_input_npzs(input_dir)
    _apply_vivana_td_config(burnin)

    burnin.INPUT_NPZS = None
    burnin.INPUT_NPZ_GLOB = str(Path(input_dir) / "*.npz")
    burnin.OUTPUT_DIR = Path(output_dir)
    burnin.METADATA_PATH = METADATA_PATH
    burnin.OVERWRITE = bool(overwrite)
    burnin.EXPORT_THETA0 = 0.0
    burnin.EXPORT_THETA0_VALUES = None
    burnin.WRITE_BURNIN_DIAGNOSTIC_PLOTS = bool(write_diagnostics)
    burnin.DIAGNOSTIC_PLOT_DIR = BURNIN_DIAGNOSTIC_DIR
    burnin._METADATA_CACHE = None
    burnin.main()


def _load_label_ur(path: Path) -> float:
    return float(surrogate.load_reduced_velocity(Path(path), ur_source=UR_SOURCE))


def _normalize_urs(values: Sequence[float] | None) -> tuple[float, ...]:
    if values is None:
        return ()
    normalized: list[float] = []
    for value in values:
        ur = float(value)
        if not np.isfinite(ur):
            raise ValueError(f"Reduced velocity exclusions must be finite, got {value!r}.")
        if not any(np.isclose(ur, existing, rtol=0.0, atol=surrogate.EXCLUDE_UR_ATOL) for existing in normalized):
            normalized.append(ur)
    return tuple(normalized)


def _is_excluded_ur(label_ur: float, exclude_urs: Sequence[float]) -> bool:
    exclude = np.asarray(list(exclude_urs), dtype=float).reshape(-1)
    if exclude.size == 0:
        return False
    return bool(np.any(np.isclose(float(label_ur), exclude, rtol=0.0, atol=surrogate.EXCLUDE_UR_ATOL)))


def _select_val_seen_files(paths: Sequence[Path]) -> list[Path]:
    if VAL_SEEN_POLICY != "first_per_ur":
        raise ValueError(f"Unsupported VAL_SEEN_POLICY={VAL_SEEN_POLICY!r}")

    selected: dict[float, Path] = {}
    for path in sorted(Path(p) for p in paths):
        key = round(_load_label_ur(path), 10)
        selected.setdefault(key, path)
    return [selected[key] for key in sorted(selected)]


def _copy_split_files(
    flat_dir: Path,
    final_dir: Path,
    *,
    overwrite: bool,
    exclude_urs: Sequence[float] = (),
) -> tuple[list[SplitRow], list[ExcludedRow]]:
    source_paths = sorted(Path(flat_dir).glob("*.npz"))
    if not source_paths:
        raise FileNotFoundError(f"No burn-in trimmed NPZ files found in {flat_dir}")

    _clean_dir(final_dir, overwrite=overwrite)
    train_dir = final_dir / "train"
    val_seen_dir = final_dir / "val_seen"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_seen_dir.mkdir(parents=True, exist_ok=True)

    excluded_rows: list[ExcludedRow] = []
    retained_paths: list[Path] = []
    for src_path in source_paths:
        label_ur = _load_label_ur(src_path)
        if _is_excluded_ur(label_ur, exclude_urs):
            excluded_rows.append(
                ExcludedRow(
                    source=str(src_path),
                    label_ur=float(label_ur),
                    reason="dataset_exclude_ur",
                )
            )
            continue
        retained_paths.append(src_path)

    if not retained_paths:
        raise ValueError("All burn-in trimmed NPZ files were excluded from the final dataset.")

    rows: list[SplitRow] = []
    for src_path in retained_paths:
        dst_path = train_dir / src_path.name
        shutil.copy2(src_path, dst_path)
        rows.append(
            SplitRow(
                split="train",
                source=str(src_path),
                destination=str(dst_path),
                label_ur=_load_label_ur(src_path),
                reason="all_burnin_trimmed_cases",
            )
        )

    for src_path in _select_val_seen_files(retained_paths):
        dst_path = val_seen_dir / src_path.name
        shutil.copy2(src_path, dst_path)
        rows.append(
            SplitRow(
                split="val_seen",
                source=str(src_path),
                destination=str(dst_path),
                label_ur=_load_label_ur(src_path),
                reason=VAL_SEEN_POLICY,
            )
        )

    _write_split_manifest(final_dir / "split_manifest.csv", rows)
    if excluded_rows:
        _write_exclusion_manifest(final_dir / "dataset_exclusion_manifest.csv", excluded_rows)
    return rows, excluded_rows


def _write_split_manifest(path: Path, rows: Sequence[SplitRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("split", "source", "destination", "label_ur", "reason"),
        )
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def _write_exclusion_manifest(path: Path, rows: Sequence[ExcludedRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("source", "label_ur", "reason"),
        )
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def _effective_surrogate_exclude_urs(dataset_exclude_urs: Sequence[float]) -> tuple[float, ...]:
    return _normalize_urs(dataset_exclude_urs)


def _generate_surrogate_files(final_dir: Path, *, dataset_exclude_urs: Sequence[float] = ()) -> dict[str, np.ndarray]:
    train_paths = sorted((Path(final_dir) / "train").glob("*.npz"))
    exclude_urs = _effective_surrogate_exclude_urs(dataset_exclude_urs)
    anchors = surrogate.collect_anchor_points(
        train_paths,
        ur_source=UR_SOURCE,
        exclude_urs=exclude_urs,
        exclude_ur_atol=surrogate.EXCLUDE_UR_ATOL,
    )
    points = surrogate.generate_surrogate_validation_points(
        anchors,
        target_urs=surrogate.TARGET_URS,
        points_per_interval=surrogate.POINTS_PER_INTERVAL,
        interpolation_kind=surrogate.INTERPOLATION_KIND,
        smoothing_strength=surrogate.SMOOTHING_STRENGTH,
        include_anchor_points=surrogate.INCLUDE_ANCHOR_POINTS,
    )
    surrogate.save_surrogate_points(
        points,
        npz_path=Path(final_dir) / "surrogate_validation_points.npz",
        csv_path=Path(final_dir) / "surrogate_validation_points.csv",
    )
    surrogate.save_diagnostic_plot(
        points,
        anchors=anchors,
        output_path=Path(final_dir) / "surrogate_validation_points_diagnostic.pdf",
    )
    return points


def _generate_loo_datasets(*, source_root: Path, output_root: Path) -> None:
    loo.SOURCE_ROOT = Path(source_root)
    loo.OUTPUT_ROOT = Path(output_root)
    loo.SPLITS = ("train", "val_seen")
    loo.HELD_OUT_URS = tuple(float(value) for value in HELD_OUT_URS)
    loo.UR_SOURCE = UR_SOURCE
    loo.main()


def _unique_urs(paths: Iterable[Path]) -> list[float]:
    return sorted({round(_load_label_ur(path), 10) for path in paths})


def _write_dataset_manifest(
    *,
    final_dir: Path,
    input_dir: Path,
    burnin_work_dir: Path,
    split_rows: Sequence[SplitRow],
    excluded_rows: Sequence[ExcludedRow],
    surrogate_points: dict[str, np.ndarray],
    create_loo: bool,
    loo_output_root: Path,
    dataset_exclude_urs: Sequence[float],
) -> None:
    train_paths = sorted((final_dir / "train").glob("*.npz"))
    val_seen_paths = sorted((final_dir / "val_seen").glob("*.npz"))
    manifest = {
        "input_npz_dir": str(input_dir),
        "burnin_work_dir": str(burnin_work_dir),
        "final_dataset_dir": str(final_dir),
        "train_dir": str(final_dir / "train"),
        "val_seen_dir": str(final_dir / "val_seen"),
        "surrogate_validation_npz": str(final_dir / "surrogate_validation_points.npz"),
        "surrogate_validation_csv": str(final_dir / "surrogate_validation_points.csv"),
        "surrogate_validation_diagnostic_plot": str(final_dir / "surrogate_validation_points_diagnostic.pdf"),
        "dataset_exclusion_manifest": str(final_dir / "dataset_exclusion_manifest.csv") if excluded_rows else None,
        "loo_output_root": str(loo_output_root) if create_loo else None,
        "ur_source": UR_SOURCE,
        "val_seen_policy": VAL_SEEN_POLICY,
        "dataset_exclude_urs": [float(value) for value in dataset_exclude_urs],
        "surrogate_exclude_urs": [float(value) for value in _effective_surrogate_exclude_urs(dataset_exclude_urs)],
        "held_out_urs": [float(value) for value in HELD_OUT_URS] if create_loo else [],
        "vivana_td": {
            "params": {key: float(value) for key, value in VIVANA_TD_PARAMS.items()},
            "phase_wrap_convention": str(PHASE_WRAP_CONVENTION),
            "force_phase_convention": str(FORCE_PHASE_CONVENTION),
            "burnin_theta0_values": [float(value) for value in np.asarray(BURNIN_THETA0_VALUES, dtype=float)],
            "td_memory_mode": str(TD_MEMORY_MODE),
            "td_tau_over_tref": float(TD_TAU_OVER_TREF),
            "td_memory_tau_s": None if TD_MEMORY_TAU_S is None else float(TD_MEMORY_TAU_S),
            "n_memory_fallback": int(N_MEMORY_FALLBACK),
        },
        "counts": {
            "input_npz": len(_ensure_input_npzs(input_dir)),
            "train_npz": len(train_paths),
            "val_seen_npz": len(val_seen_paths),
            "surrogate_rows": int(np.asarray(surrogate_points["ur"]).reshape(-1).size),
            "split_manifest_rows": len(split_rows),
            "excluded_npz": len(excluded_rows),
        },
        "train_label_urs": _unique_urs(train_paths),
        "val_seen_label_urs": _unique_urs(val_seen_paths),
    }
    with (final_dir / "dataset_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build final train/val_seen Vivana-TD datasets from CFD NPZ exports.",
    )
    parser.add_argument("--input-dir", type=Path, default=INPUT_NPZ_DIR)
    parser.add_argument("--burnin-work-dir", type=Path, default=BURNIN_WORK_DIR)
    parser.add_argument("--output-dir", type=Path, default=FINAL_DATASET_DIR)
    parser.add_argument("--loo-output-root", type=Path, default=LOO_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--burnin-diagnostics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--loo", action=argparse.BooleanOptionalAction, default=CREATE_LOO_DATASETS)
    parser.add_argument(
        "--exclude-ur",
        type=float,
        action="append",
        default=None,
        help=(
            "Exclude this label U_r from the main final train/val_seen dataset. "
            "Can be passed multiple times. The same U_r is also excluded from surrogate anchors."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_dir = Path(args.input_dir)
    burnin_work_dir = Path(args.burnin_work_dir)
    output_dir = Path(args.output_dir)
    loo_output_root = Path(args.loo_output_root)
    dataset_exclude_urs = _normalize_urs([*DATASET_EXCLUDE_URS, *(args.exclude_ur or [])])

    print(f"Building final TD training dataset from {input_dir}")
    if dataset_exclude_urs:
        print(f"Excluding label U_r from final train/val_seen dataset: {', '.join(f'{value:g}' for value in dataset_exclude_urs)}")
    _clean_dir(burnin_work_dir, overwrite=bool(args.overwrite))
    _run_burnin_preprocessing(
        input_dir=input_dir,
        output_dir=burnin_work_dir,
        overwrite=bool(args.overwrite),
        write_diagnostics=bool(args.burnin_diagnostics),
    )
    split_rows, excluded_rows = _copy_split_files(
        burnin_work_dir,
        output_dir,
        overwrite=bool(args.overwrite),
        exclude_urs=dataset_exclude_urs,
    )
    surrogate_points = _generate_surrogate_files(output_dir, dataset_exclude_urs=dataset_exclude_urs)
    if bool(args.loo):
        _generate_loo_datasets(source_root=output_dir, output_root=loo_output_root)
    _write_dataset_manifest(
        final_dir=output_dir,
        input_dir=input_dir,
        burnin_work_dir=burnin_work_dir,
        split_rows=split_rows,
        excluded_rows=excluded_rows,
        surrogate_points=surrogate_points,
        create_loo=bool(args.loo),
        loo_output_root=loo_output_root,
        dataset_exclude_urs=dataset_exclude_urs,
    )

    print(f"Wrote final dataset: {output_dir}")
    print(f"  train: {len(list((output_dir / 'train').glob('*.npz')))} files")
    print(f"  val_seen: {len(list((output_dir / 'val_seen').glob('*.npz')))} files")
    print(f"  surrogate rows: {int(np.asarray(surrogate_points['ur']).reshape(-1).size)}")
    if excluded_rows:
        print(f"  excluded by U_r: {len(excluded_rows)} files")
    if bool(args.loo):
        print(f"  leave-one-U_r-out datasets: {loo_output_root}")


if __name__ == "__main__":
    main()
