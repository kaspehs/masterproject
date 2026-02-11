"""Aggregate leave-one-U_r-out final errors from TensorBoard logs."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.util import tensor_util


DEFAULT_METRICS = (
    "Disp rollout NRMSE",
    "Force rollout NRMSE",
    "Force mapping NRMSE",
)

# Configuration: edit these values directly instead of passing CLI args.
LOG_ROOTS = [Path("logs/phnn_Ur1")]
OUTPUT_DIR = Path("figs/loo_ur")
UNSEEN_UR_TOL = 1e-6
METRICS_TO_PLOT = list(DEFAULT_METRICS)
USE_LOG_HEATMAP = True
USE_LOG_UNSEEN_Y = True
LOG_EPS = 1e-12

BY_UR_TAG_RE = re.compile(r"^final_val/by_ur/(?P<metric>.+)/U_r=(?P<ur>[-+0-9.eE]+)$")


def _slug(text: str) -> str:
    out = re.sub(r"[^A-Za-z0-9]+", "_", text.strip())
    return out.strip("_").lower() or "metric"


def _event_dirs(roots: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for event_path in root.rglob("events.out.tfevents.*"):
            run_dir = event_path.parent.resolve()
            if run_dir in seen:
                continue
            seen.add(run_dir)
            out.append(run_dir)
    return sorted(out)


def _read_text_tag(ea: EventAccumulator, tag: str) -> str | None:
    tensor_tags = ea.Tags().get("tensors", [])
    if not tensor_tags:
        return None
    candidate_tags = [tag, f"{tag}/text_summary"]
    candidate_tags.extend(t for t in tensor_tags if t.startswith(f"{tag}/"))
    selected_tag = None
    for cand in candidate_tags:
        if cand in tensor_tags:
            selected_tag = cand
            break
    if selected_tag is None:
        return None
    events = ea.Tensors(selected_tag)
    if not events:
        return None
    arr = tensor_util.make_ndarray(events[-1].tensor_proto)
    if arr.size == 0:
        return None
    value = arr.reshape(-1)[0]
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _load_json_text(ea: EventAccumulator, tag: str) -> dict[str, Any] | None:
    text = _read_text_tag(ea, tag)
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _extract_method_and_holdout(ea: EventAccumulator, run_dir: Path) -> tuple[str, float | None]:
    vp_cfg = _load_json_text(ea, "vpinn/config_vpinn")
    if vp_cfg is not None:
        hold = vp_cfg.get("train_exclude_ur", None)
        return "vpinn", _extract_single_ur(hold)
    h_cfg = _load_json_text(ea, "phnn/config_hnn")
    if h_cfg is not None:
        hold = h_cfg.get("train_exclude_ur", None)
        return "phnn", _extract_single_ur(hold)
    name = run_dir.name.lower()
    if "vpinn" in name:
        return "vpinn", None
    if "phnn" in name or "hnn" in name:
        return "phnn", None
    return "unknown", None


def _extract_single_ur(value: Any) -> float | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size != 1 or not np.isfinite(arr[0]):
        return None
    return float(arr[0])


def _parse_text_table(text: str) -> list[tuple[str, float, float]]:
    lines = [ln.strip() for ln in str(text).splitlines() if ln.strip().startswith("|")]
    if len(lines) < 3:
        return []
    header_cells = [c.strip() for c in lines[0].strip("|").split("|")]
    if not header_cells or header_cells[0] != "U_r":
        return []
    metrics = header_cells[1:]
    rows: list[tuple[str, float, float]] = []
    for ln in lines[2:]:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if len(cells) != len(header_cells):
            continue
        try:
            ur_val = float(cells[0])
        except ValueError:
            continue
        for metric, val_txt in zip(metrics, cells[1:]):
            try:
                value = float(val_txt)
            except ValueError:
                continue
            if not np.isfinite(value):
                continue
            rows.append((metric, float(ur_val), float(value)))
    return rows


def _extract_by_ur_records(ea: EventAccumulator) -> list[tuple[str, float, float]]:
    rows: list[tuple[str, float, float]] = []
    scalar_tags = ea.Tags().get("scalars", [])
    for tag in scalar_tags:
        m = BY_UR_TAG_RE.match(tag)
        if not m:
            continue
        metric = str(m.group("metric"))
        try:
            ur_val = float(m.group("ur"))
        except ValueError:
            continue
        events = ea.Scalars(tag)
        if not events:
            continue
        value = float(events[-1].value)
        if np.isfinite(value):
            rows.append((metric, ur_val, value))
    if rows:
        return rows
    table_text = _read_text_tag(ea, "final_val/errors_vs_ur_text")
    if not table_text:
        return []
    return _parse_text_table(table_text)


def _collect_records(run_dir: Path) -> list[dict[str, Any]]:
    ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0, "tensors": 0})
    ea.Reload()
    method, held_out_ur = _extract_method_and_holdout(ea, run_dir)
    by_ur = _extract_by_ur_records(ea)
    rows: list[dict[str, Any]] = []
    for metric, eval_ur, value in by_ur:
        rows.append(
            {
                "run_dir": str(run_dir),
                "run_name": run_dir.name,
                "method": method,
                "held_out_ur": held_out_ur,
                "eval_ur": float(eval_ur),
                "metric": metric,
                "value": float(value),
            }
        )
    return rows


def _write_csv(records: list[dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run_dir", "run_name", "method", "held_out_ur", "eval_ur", "metric", "value"]
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _plot_heatmap(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    method: str,
    out_path: Path,
) -> None:
    subset = [r for r in rows if r["metric"] == metric and r["method"] == method and r["held_out_ur"] is not None]
    if not subset:
        return
    held_vals = sorted({float(r["held_out_ur"]) for r in subset})
    eval_vals = sorted({float(r["eval_ur"]) for r in subset})
    h_idx = {u: i for i, u in enumerate(held_vals)}
    e_idx = {u: i for i, u in enumerate(eval_vals)}
    buckets: dict[tuple[int, int], list[float]] = defaultdict(list)
    for r in subset:
        i = h_idx[float(r["held_out_ur"])]
        j = e_idx[float(r["eval_ur"])]
        buckets[(i, j)].append(float(r["value"]))
    mat = np.full((len(held_vals), len(eval_vals)), np.nan, dtype=float)
    for (i, j), vals in buckets.items():
        mat[i, j] = float(np.mean(vals))

    fig, ax = plt.subplots(figsize=(7, 5))
    mat_plot = mat.copy()
    norm = None
    if USE_LOG_HEATMAP:
        finite_pos = mat_plot[np.isfinite(mat_plot) & (mat_plot > 0.0)]
        if finite_pos.size > 0:
            vmin = max(float(np.min(finite_pos)), float(LOG_EPS))
            vmax = float(np.max(finite_pos))
            mat_plot = np.where(mat_plot > 0.0, mat_plot, np.nan)
            if vmax > vmin:
                norm = LogNorm(vmin=vmin, vmax=vmax)
    im = ax.imshow(mat_plot, aspect="auto", interpolation="nearest", norm=norm)
    fig.colorbar(im, ax=ax, label=metric)
    ax.set_xticks(np.arange(len(eval_vals)))
    ax.set_xticklabels([f"{u:g}" for u in eval_vals], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(held_vals)))
    ax.set_yticklabels([f"{u:g}" for u in held_vals])
    ax.set_xlabel("Evaluated U_r")
    ax.set_ylabel("Excluded U_r (train)")
    ax.set_title(f"{method.upper()} | {metric}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_unseen_curve(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    out_path: Path,
    tol: float,
) -> None:
    subset = [r for r in rows if r["metric"] == metric and r["held_out_ur"] is not None]
    unseen = [
        r
        for r in subset
        if abs(float(r["eval_ur"]) - float(r["held_out_ur"])) <= float(tol)
    ]
    if not unseen:
        return
    grouped: dict[tuple[str, float], list[float]] = defaultdict(list)
    for r in unseen:
        grouped[(str(r["method"]), float(r["held_out_ur"]))].append(float(r["value"]))

    methods = sorted({m for m, _ in grouped.keys()})
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in methods:
        xs = sorted({u for m, u in grouped.keys() if m == method})
        ys = [float(np.mean(grouped[(method, u)])) for u in xs]
        yerr = [float(np.std(grouped[(method, u)])) for u in xs]
        ax.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3, label=method.upper())
    ax.set_xlabel("Excluded U_r (unseen during training)")
    ax.set_ylabel(metric)
    ax.set_title(f"Unseen-U_r generalization | {metric}")
    if USE_LOG_UNSEEN_Y:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    roots = [Path(p) for p in LOG_ROOTS]
    run_dirs = _event_dirs(roots)
    if not run_dirs:
        raise SystemExit("No TensorBoard event files found under the provided roots.")

    records: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        try:
            rows = _collect_records(run_dir)
        except Exception as exc:  # pragma: no cover - defensive for partial/corrupt logs
            print(f"Skipping {run_dir}: {exc}")
            continue
        if rows:
            records.extend(rows)

    if not records:
        raise SystemExit("No per-U_r final errors found. Expected by-ur scalars or final_val/errors_vs_ur_text.")

    out_dir = Path(OUTPUT_DIR)
    _write_csv(records, out_dir / "loo_ur_records.csv")

    methods = sorted({str(r["method"]) for r in records})
    for metric in METRICS_TO_PLOT:
        for method in methods:
            _plot_heatmap(
                records,
                metric=metric,
                method=method,
                out_path=out_dir / f"heatmap_{_slug(method)}_{_slug(metric)}.png",
            )
        _plot_unseen_curve(
            records,
            metric=metric,
            out_path=out_dir / f"unseen_curve_{_slug(metric)}.png",
            tol=float(UNSEEN_UR_TOL),
        )

    print(f"Processed {len(run_dirs)} runs.")
    print(f"Wrote: {out_dir / 'loo_ur_records.csv'}")
    print(f"Figures in: {out_dir}")


if __name__ == "__main__":
    main()
