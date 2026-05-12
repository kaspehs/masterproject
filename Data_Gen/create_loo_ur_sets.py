"""Create leave-one-reduced-velocity-out TD training datasets.

Each generated folder contains copied ``train/`` and ``val_seen/`` splits from
``CFD_Data/npz_exports_td_burnin_trimmed4`` with one label U_r omitted. A
matching surrogate validation NPZ/CSV/diagnostic plot is generated from the
retained training files only.
"""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from generate_surrogate_validation_points import (
    EXCLUDE_UR_ATOL,
    INCLUDE_ANCHOR_POINTS,
    INTERPOLATION_KIND,
    POINTS_PER_INTERVAL,
    SMOOTHING_STRENGTH,
    TARGET_URS,
    collect_anchor_points,
    generate_surrogate_validation_points,
    load_reduced_velocity,
    save_diagnostic_plot,
    save_surrogate_points,
)


SOURCE_ROOT = Path("CFD_Data/npz_exports_td_burnin_trimmed4")
OUTPUT_ROOT = Path("CFD_Data/loo_ur_td_burnin_trimmed4")
SPLITS = ("train", "val_seen")
HELD_OUT_URS = (2.0, 4.0, 5.0, 5.75, 7.0, 8.0, 10.0)
UR_SOURCE = "label"
UR_ATOL = 1.0e-8


@dataclass(frozen=True)
class CopiedFile:
    split: str
    source: str
    destination: str
    label_ur: float


@dataclass(frozen=True)
class SkippedFile:
    split: str
    source: str
    label_ur: float
    reason: str


@dataclass(frozen=True)
class LooSummary:
    folder: str
    held_out_ur: float
    train_files: int
    val_seen_files: int
    surrogate_rows: int
    retained_train_urs: tuple[float, ...]
    retained_val_seen_urs: tuple[float, ...]
    surrogate_anchor_label_urs: tuple[float, ...]


def _ur_slug(ur: float) -> str:
    text = f"{float(ur):g}".replace(".", "")
    return f"Ur{text}"


def _is_held_out(label_ur: float, held_out_ur: float) -> bool:
    return bool(np.isclose(float(label_ur), float(held_out_ur), rtol=0.0, atol=UR_ATOL))


def _source_paths(split: str) -> list[Path]:
    split_dir = SOURCE_ROOT / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing source split directory: {split_dir}")
    return sorted(split_dir.glob("*.npz"))


def _unique_urs(paths: Iterable[Path]) -> tuple[float, ...]:
    values = [load_reduced_velocity(path, ur_source=UR_SOURCE) for path in paths]
    unique = sorted({round(float(value), 10) for value in values})
    return tuple(float(value) for value in unique)


def _write_csv_manifest(path: Path, copied: list[CopiedFile], skipped: list[SkippedFile]) -> None:
    rows: list[dict[str, object]] = []
    for item in copied:
        rows.append(
            {
                "action": "copied",
                "split": item.split,
                "label_ur": item.label_ur,
                "source": item.source,
                "destination": item.destination,
                "reason": "",
            }
        )
    for item in skipped:
        rows.append(
            {
                "action": "skipped",
                "split": item.split,
                "label_ur": item.label_ur,
                "source": item.source,
                "destination": "",
                "reason": item.reason,
            }
        )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("action", "split", "label_ur", "source", "destination", "reason"),
        )
        writer.writeheader()
        writer.writerows(rows)


def _prepare_loo_folder(held_out_ur: float) -> tuple[Path, list[CopiedFile], list[SkippedFile]]:
    loo_dir = OUTPUT_ROOT / f"loo_{_ur_slug(held_out_ur)}"
    if loo_dir.exists():
        shutil.rmtree(loo_dir)
    copied: list[CopiedFile] = []
    skipped: list[SkippedFile] = []

    for split in SPLITS:
        out_split_dir = loo_dir / split
        out_split_dir.mkdir(parents=True, exist_ok=True)
        for src_path in _source_paths(split):
            label_ur = load_reduced_velocity(src_path, ur_source=UR_SOURCE)
            if _is_held_out(label_ur, held_out_ur):
                skipped.append(
                    SkippedFile(
                        split=split,
                        source=str(src_path),
                        label_ur=float(label_ur),
                        reason="held_out_ur",
                    )
                )
                continue
            dst_path = out_split_dir / src_path.name
            shutil.copy2(src_path, dst_path)
            copied.append(
                CopiedFile(
                    split=split,
                    source=str(src_path),
                    destination=str(dst_path),
                    label_ur=float(label_ur),
                )
            )

    return loo_dir, copied, skipped


def _generate_surrogate_files(loo_dir: Path, held_out_ur: float) -> dict[str, np.ndarray]:
    train_paths = sorted((loo_dir / "train").glob("*.npz"))
    anchors = collect_anchor_points(
        train_paths,
        ur_source=UR_SOURCE,
        exclude_urs=(held_out_ur,),
        exclude_ur_atol=EXCLUDE_UR_ATOL,
    )
    points = generate_surrogate_validation_points(
        anchors,
        target_urs=TARGET_URS,
        points_per_interval=POINTS_PER_INTERVAL,
        interpolation_kind=INTERPOLATION_KIND,
        smoothing_strength=SMOOTHING_STRENGTH,
        include_anchor_points=INCLUDE_ANCHOR_POINTS,
    )
    save_surrogate_points(
        points,
        npz_path=loo_dir / "surrogate_validation_points.npz",
        csv_path=loo_dir / "surrogate_validation_points.csv",
    )
    save_diagnostic_plot(
        points,
        anchors=anchors,
        output_path=loo_dir / "surrogate_validation_points_diagnostic.png",
    )
    return points


def _validate_loo_folder(loo_dir: Path, held_out_ur: float) -> LooSummary:
    train_paths = sorted((loo_dir / "train").glob("*.npz"))
    val_seen_paths = sorted((loo_dir / "val_seen").glob("*.npz"))
    if len(_unique_urs(train_paths)) < 2:
        raise ValueError(f"{loo_dir} has fewer than two unique training U_r labels.")

    for split, paths in (("train", train_paths), ("val_seen", val_seen_paths)):
        for path in paths:
            label_ur = load_reduced_velocity(path, ur_source=UR_SOURCE)
            if _is_held_out(label_ur, held_out_ur):
                raise AssertionError(f"{path} in {split} still has held-out U_r={held_out_ur:g}.")

    required = [
        loo_dir / "surrogate_validation_points.npz",
        loo_dir / "surrogate_validation_points.csv",
        loo_dir / "surrogate_validation_points_diagnostic.png",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{loo_dir} is missing generated surrogate files: {missing}")

    with np.load(loo_dir / "surrogate_validation_points.npz", allow_pickle=True) as data:
        if "anchor_ur_label" not in data:
            raise KeyError(f"{loo_dir} surrogate NPZ is missing anchor_ur_label.")
        anchor_label_urs = tuple(float(value) for value in np.asarray(data["anchor_ur_label"], dtype=float).reshape(-1))
        for label_ur in anchor_label_urs:
            if _is_held_out(label_ur, held_out_ur):
                raise AssertionError(f"{loo_dir} surrogate anchors include held-out U_r={held_out_ur:g}.")
        surrogate_rows = int(np.asarray(data["ur"]).reshape(-1).size)

    return LooSummary(
        folder=str(loo_dir),
        held_out_ur=float(held_out_ur),
        train_files=len(train_paths),
        val_seen_files=len(val_seen_paths),
        surrogate_rows=surrogate_rows,
        retained_train_urs=_unique_urs(train_paths),
        retained_val_seen_urs=_unique_urs(val_seen_paths),
        surrogate_anchor_label_urs=anchor_label_urs,
    )


def _write_manifest(
    loo_dir: Path,
    held_out_ur: float,
    copied: list[CopiedFile],
    skipped: list[SkippedFile],
    summary: LooSummary,
) -> None:
    manifest = {
        "source_root": str(SOURCE_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "held_out_ur": float(held_out_ur),
        "ur_source": UR_SOURCE,
        "ur_atol": UR_ATOL,
        "surrogate_anchor_split": "train",
        "counts": {
            "copied_train": sum(1 for item in copied if item.split == "train"),
            "copied_val_seen": sum(1 for item in copied if item.split == "val_seen"),
            "skipped_train": sum(1 for item in skipped if item.split == "train"),
            "skipped_val_seen": sum(1 for item in skipped if item.split == "val_seen"),
            "surrogate_rows": summary.surrogate_rows,
        },
        "retained_train_urs": list(summary.retained_train_urs),
        "retained_val_seen_urs": list(summary.retained_val_seen_urs),
        "surrogate_anchor_label_urs": list(summary.surrogate_anchor_label_urs),
        "copied_files": [asdict(item) for item in copied],
        "skipped_files": [asdict(item) for item in skipped],
    }
    with (loo_dir / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    _write_csv_manifest(loo_dir / "manifest.csv", copied, skipped)


def _format_urs(values: tuple[float, ...]) -> str:
    return ";".join(f"{value:g}" for value in values)


def main() -> None:
    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(f"Missing source root: {SOURCE_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    summaries: list[LooSummary] = []
    for held_out_ur in HELD_OUT_URS:
        loo_dir, copied, skipped = _prepare_loo_folder(held_out_ur)
        _generate_surrogate_files(loo_dir, held_out_ur)
        summary = _validate_loo_folder(loo_dir, held_out_ur)
        _write_manifest(loo_dir, held_out_ur, copied, skipped, summary)
        summaries.append(summary)

    print("folder,held_out_ur,train_files,val_seen_files,surrogate_rows,retained_train_urs")
    for summary in summaries:
        print(
            f"{Path(summary.folder).name},"
            f"{summary.held_out_ur:g},"
            f"{summary.train_files},"
            f"{summary.val_seen_files},"
            f"{summary.surrogate_rows},"
            f"{_format_urs(summary.retained_train_urs)}"
        )
    print(f"Wrote {len(summaries)} leave-one-U_r-out dataset folders under {OUTPUT_ROOT}.")


if __name__ == "__main__":
    main()
