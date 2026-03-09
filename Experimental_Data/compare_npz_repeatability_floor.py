from __future__ import annotations

import argparse
import csv
import math
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


DEFAULT_INPUT_DIR = Path("Experimental_Data/npz_exports_v2_60s_zero_mean")
DEFAULT_OUTPUT_DIR = Path("Experimental_Data/repeatability_floor_v2_60s")
DEFAULT_PATTERN = "*.npz"
DEFAULT_UR_DECIMALS = 0
DEFAULT_MAX_DURATION_S = 60.0
DT_MATCH_RTOL = 1e-6
DT_MATCH_ATOL = 1e-12
METRIC_SPECS = (
    ("dominant_freq_rel_error_disp", "Disp Dominant Freq Rel Error"),
    ("disp_psd_rel_error", "Disp PSD Rel Error"),
    ("force_psd_rel_error", "Force PSD Rel Error"),
    ("disp_std_rel_error", "Disp Std Rel Error"),
)


@dataclass(frozen=True)
class SeriesRecord:
    path: Path
    ur_value: float
    ur_key: float
    length: int
    dt: float
    time: np.ndarray
    displacement: np.ndarray
    force: np.ndarray


class ProgressTracker:
    def __init__(self, total: int, desc: str, unit: str) -> None:
        self.total = max(0, int(total))
        self.desc = str(desc)
        self.unit = str(unit)
        self.count = 0
        self.start = time.perf_counter()
        self.last_print = self.start
        self.bar = tqdm(total=self.total, desc=self.desc, unit=self.unit) if tqdm is not None else None
        if self.bar is None:
            print(f"{self.desc}: 0/{self.total} {self.unit}")

    def update(self, step: int = 1) -> None:
        self.count += int(step)
        if self.bar is not None:
            self.bar.update(step)
            return
        now = time.perf_counter()
        if self.count >= self.total or (now - self.last_print) >= 1.0:
            elapsed = max(now - self.start, 1e-9)
            rate = self.count / elapsed
            print(f"{self.desc}: {self.count}/{self.total} {self.unit} ({rate:.1f}/{self.unit}/s)")
            self.last_print = now

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()
            return
        elapsed = max(time.perf_counter() - self.start, 1e-9)
        rate = self.count / elapsed
        print(f"{self.desc}: done {self.count}/{self.total} {self.unit} in {elapsed:.1f}s ({rate:.1f}/{self.unit}/s)")


def dominant_frequency(signal: np.ndarray, dt: float) -> float:
    if dt <= 0.0:
        return float("nan")
    signal = np.asarray(signal, dtype=float).reshape(-1)
    if signal.size < 2:
        return float("nan")
    centered = signal - np.mean(signal)
    if np.allclose(centered, 0.0):
        return float("nan")
    fft_vals = np.fft.rfft(centered)
    freqs = np.fft.rfftfreq(centered.size, d=dt)
    if freqs.size <= 1:
        return float("nan")
    magnitudes = np.abs(fft_vals)
    magnitudes[0] = 0.0
    dominant_idx = int(np.argmax(magnitudes))
    dominant_mag = magnitudes[dominant_idx]
    if dominant_mag <= 0.0:
        return float("nan")
    return float(freqs[dominant_idx])


def spectral_relative_error(true_signal: np.ndarray, model_signal: np.ndarray, dt: float, eps: float = 1e-12) -> float:
    if dt <= 0.0:
        return float("nan")
    true_signal = np.asarray(true_signal, dtype=float).reshape(-1)
    model_signal = np.asarray(model_signal, dtype=float).reshape(-1)
    length = min(true_signal.size, model_signal.size)
    if length < 2:
        return float("nan")
    true_trim = true_signal[-length:]
    model_trim = model_signal[-length:]
    window = np.hanning(length)
    true_proc = (true_trim - np.mean(true_trim)) * window
    model_proc = (model_trim - np.mean(model_trim)) * window
    true_fft = np.abs(np.fft.rfft(true_proc))
    model_fft = np.abs(np.fft.rfft(model_proc))
    if true_fft.size == 0:
        return float("nan")
    true_fft[0] = 0.0
    model_fft[0] = 0.0
    denom = np.linalg.norm(true_fft)
    if denom <= eps:
        return float("nan")
    return float(np.linalg.norm(model_fft - true_fft) / (denom + eps))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate an empirical repeatability floor from experimental .npz files by comparing "
            "pairwise displacement/force metrics within the same reduced-velocity group."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Root folder containing .npz files.")
    parser.add_argument("--pattern", type=str, default=DEFAULT_PATTERN, help="Filename pattern used during recursive search.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where pairwise and summary CSV files will be written.",
    )
    parser.add_argument(
        "--ur-decimals",
        type=int,
        default=DEFAULT_UR_DECIMALS,
        help="Round mean U_r to this many decimals when grouping files.",
    )
    parser.add_argument(
        "--max-duration-s",
        type=float,
        default=DEFAULT_MAX_DURATION_S,
        help="Trim all kept series to at most this duration in seconds. Use <=0 to disable.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional debugging limit on the number of discovered files.",
    )
    return parser.parse_args()


def _load_1d_array(data: np.lib.npyio.NpzFile, keys: list[str]) -> np.ndarray:
    for key in keys:
        if key in data.files:
            return np.asarray(data[key], dtype=float).reshape(-1)
    raise KeyError(f"None of the expected keys were found: {keys}")


def _symmetric_relative_difference(a: float, b: float, eps: float = 1e-12) -> float:
    if not np.isfinite(a) or not np.isfinite(b):
        return float("nan")
    denom = max(abs(float(a)), abs(float(b)), float(eps))
    return float(abs(float(a) - float(b)) / denom)


def _stats_dict(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=float)
    if arr.size == 0:
        return {
            "count": 0.0,
            "median": float("nan"),
            "p25": float("nan"),
            "p10": float("nan"),
            "min": float("nan"),
            "mean": float("nan"),
        }
    return {
        "count": float(arr.size),
        "median": float(np.nanmedian(arr)),
        "p25": float(np.nanquantile(arr, 0.25)),
        "p10": float(np.nanquantile(arr, 0.10)),
        "min": float(np.nanmin(arr)),
        "mean": float(np.nanmean(arr)),
    }


def _discover_files(root: Path, pattern: str, max_files: int | None) -> list[Path]:
    files = sorted(root.rglob(pattern))
    if max_files is not None:
        files = files[: max(0, int(max_files))]
    return files


def _load_series_record(path: Path, ur_decimals: int, max_duration_s: float) -> SeriesRecord:
    with np.load(path, allow_pickle=True) as data:
        time_arr = _load_1d_array(data, ["time", "a"])
        y_arr = _load_1d_array(data, ["y", "b"])
        force_arr = _load_1d_array(data, ["F_total", "cf_force", "c"])
        ur_arr = _load_1d_array(data, ["U_r"])

    n = min(time_arr.size, y_arr.size, force_arr.size, ur_arr.size)
    if n < 2:
        raise ValueError(f"{path.name}: not enough samples after key alignment.")

    time_arr = np.asarray(time_arr[:n], dtype=float)
    y_arr = np.asarray(y_arr[:n], dtype=float)
    force_arr = np.asarray(force_arr[:n], dtype=float)
    ur_arr = np.asarray(ur_arr[:n], dtype=float)

    finite_mask = np.isfinite(time_arr) & np.isfinite(y_arr) & np.isfinite(force_arr) & np.isfinite(ur_arr)
    if not np.all(finite_mask):
        time_arr = time_arr[finite_mask]
        y_arr = y_arr[finite_mask]
        force_arr = force_arr[finite_mask]
        ur_arr = ur_arr[finite_mask]
    if time_arr.size < 2:
        raise ValueError(f"{path.name}: series became too short after finite-value filtering.")

    order = np.argsort(time_arr)
    time_arr = time_arr[order]
    y_arr = y_arr[order]
    force_arr = force_arr[order]
    ur_arr = ur_arr[order]

    diffs = np.diff(time_arr)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if diffs.size == 0:
        raise ValueError(f"{path.name}: invalid time axis.")
    dt = float(np.median(diffs))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"{path.name}: invalid dt={dt}.")

    if max_duration_s > 0.0:
        max_samples = int(math.floor(float(max_duration_s) / dt)) + 1
        max_samples = max(2, min(max_samples, time_arr.size))
        time_arr = time_arr[:max_samples]
        y_arr = y_arr[:max_samples]
        force_arr = force_arr[:max_samples]
        ur_arr = ur_arr[:max_samples]

    ur_value = float(np.mean(ur_arr))
    if not np.isfinite(ur_value):
        raise ValueError(f"{path.name}: invalid mean U_r.")

    return SeriesRecord(
        path=path,
        ur_value=ur_value,
        ur_key=round(ur_value, int(ur_decimals)),
        length=int(time_arr.size),
        dt=dt,
        time=time_arr,
        displacement=y_arr,
        force=force_arr,
    )


def _iter_pairs(records: list[SeriesRecord]) -> Iterator[tuple[SeriesRecord, SeriesRecord]]:
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            yield records[i], records[j]


def _pair_metrics(left: SeriesRecord, right: SeriesRecord) -> dict[str, float]:
    dom_left = dominant_frequency(left.displacement, left.dt)
    dom_right = dominant_frequency(right.displacement, right.dt)
    disp_std_left = float(np.std(left.displacement))
    disp_std_right = float(np.std(right.displacement))
    metrics = {
        "dominant_freq_rel_error_disp": _symmetric_relative_difference(dom_left, dom_right),
        "disp_std_rel_error": _symmetric_relative_difference(disp_std_left, disp_std_right),
        "disp_psd_rel_error": float("nan"),
        "force_psd_rel_error": float("nan"),
        "dt_match": 1.0 if np.isclose(left.dt, right.dt, rtol=DT_MATCH_RTOL, atol=DT_MATCH_ATOL) else 0.0,
    }
    if metrics["dt_match"] > 0.5:
        dt_common = 0.5 * (left.dt + right.dt)
        metrics["disp_psd_rel_error"] = spectral_relative_error(left.displacement, right.displacement, dt_common)
        metrics["force_psd_rel_error"] = spectral_relative_error(left.force, right.force, dt_common)
    return metrics


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", newline="", encoding="utf-8") as fh:
            fh.write("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_visualizations(
    output_dir: Path,
    plot_metric_values: dict[float, dict[str, list[float]]],
    summary_rows: list[dict[str, object]],
) -> None:
    if not plot_metric_values:
        return

    mpl_dir = output_dir / ".mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available; skipping visual outputs.")
        return

    ur_keys = sorted(plot_metric_values.keys())
    if not ur_keys:
        return

    summary_by_ur = {
        float(row["ur_group"]): row
        for row in summary_rows
        if isinstance(row.get("ur_group"), (int, float))
    }

    fig_dist, axes_dist = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    rng = np.random.default_rng(0)
    for ax, (metric_key, title) in zip(axes_dist.flat, METRIC_SPECS):
        values_by_group = []
        labels = []
        scatter_x = []
        scatter_y = []
        for idx, ur_key in enumerate(ur_keys, start=1):
            vals = np.asarray(plot_metric_values[ur_key].get(metric_key, []), dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            values_by_group.append(vals)
            labels.append(f"{ur_key:.4f}")
            jitter = rng.uniform(-0.12, 0.12, size=vals.size)
            scatter_x.append(np.full(vals.size, idx, dtype=float) + jitter)
            scatter_y.append(vals)
        if not values_by_group:
            ax.set_visible(False)
            continue
        ax.boxplot(values_by_group, labels=labels, showfliers=False)
        if scatter_x:
            ax.scatter(
                np.concatenate(scatter_x),
                np.concatenate(scatter_y),
                s=10,
                alpha=0.28,
                color="tab:blue",
                edgecolors="none",
            )
        ax.set_title(title)
        ax.set_xlabel("U_r group")
        ax.set_ylabel("Error")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.25)
    fig_dist.suptitle("Pairwise Repeatability Error Distributions by U_r", fontsize=14)
    fig_dist.savefig(output_dir / "metric_distributions_by_ur.png", dpi=180, bbox_inches="tight")
    plt.close(fig_dist)

    fig_summary, axes_summary = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    for ax, (metric_key, title) in zip(axes_summary.flat, METRIC_SPECS):
        xs: list[float] = []
        medians: list[float] = []
        p25s: list[float] = []
        p10s: list[float] = []
        mins: list[float] = []
        for ur_key in ur_keys:
            row = summary_by_ur.get(float(ur_key))
            if row is None:
                continue
            median = row.get(f"{metric_key}_median")
            p25 = row.get(f"{metric_key}_p25")
            p10 = row.get(f"{metric_key}_p10")
            min_v = row.get(f"{metric_key}_min")
            if not np.isfinite(float(median)):
                continue
            xs.append(float(ur_key))
            medians.append(float(median))
            p25s.append(float(p25))
            p10s.append(float(p10))
            mins.append(float(min_v))
        if not xs:
            ax.set_visible(False)
            continue
        x_arr = np.asarray(xs, dtype=float)
        med_arr = np.asarray(medians, dtype=float)
        p25_arr = np.asarray(p25s, dtype=float)
        p10_arr = np.asarray(p10s, dtype=float)
        min_arr = np.asarray(mins, dtype=float)
        ax.plot(x_arr, med_arr, marker="o", linewidth=2, color="tab:blue", label="median")
        ax.plot(x_arr, p25_arr, marker="s", linewidth=1.5, color="tab:orange", label="p25")
        ax.plot(x_arr, p10_arr, marker="^", linewidth=1.5, color="tab:green", label="p10")
        ax.scatter(x_arr, min_arr, s=28, color="tab:red", label="min", zorder=3)
        ax.set_title(title)
        ax.set_xlabel("U_r group")
        ax.set_ylabel("Error")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig_summary.suptitle("Repeatability Error Summary by U_r", fontsize=14)
    fig_summary.savefig(output_dir / "metric_summary_vs_ur.png", dpi=180, bbox_inches="tight")
    plt.close(fig_summary)


def main() -> None:
    args = _parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    file_paths = _discover_files(input_dir, args.pattern, args.max_files)
    if not file_paths:
        raise RuntimeError(f"No files matched '{args.pattern}' under {input_dir}")

    print(f"Discovered {len(file_paths)} files under {input_dir}")
    load_tracker = ProgressTracker(total=len(file_paths), desc="Loading series", unit="file")
    records: list[SeriesRecord] = []
    skipped_load: list[tuple[Path, str]] = []
    for path in file_paths:
        try:
            records.append(_load_series_record(path, ur_decimals=args.ur_decimals, max_duration_s=args.max_duration_s))
        except Exception as exc:
            skipped_load.append((path, str(exc)))
        finally:
            load_tracker.update()
    load_tracker.close()

    if skipped_load:
        print(f"Skipped {len(skipped_load)} files during loading.")

    groups: dict[float, list[SeriesRecord]] = defaultdict(list)
    for record in records:
        groups[record.ur_key].append(record)

    pairwise_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    plot_metric_values: dict[float, dict[str, list[float]]] = {}

    total_pairs = 0
    total_groups = 0
    for ur_key in sorted(groups.keys()):
        group_records = groups[ur_key]
        total_groups += 1
        length_counts = Counter(record.length for record in group_records)
        modal_length, modal_count = length_counts.most_common(1)[0]
        kept_records = [record for record in group_records if record.length == modal_length]
        dropped_length = len(group_records) - len(kept_records)

        print(
            f"U_r={ur_key:.{int(args.ur_decimals)}f}: "
            f"{len(group_records)} files, modal length={modal_length} ({modal_count} files), "
            f"dropped_length={dropped_length}"
        )

        if len(kept_records) < 2:
            summary_rows.append(
                {
                    "ur_group": ur_key,
                    "num_files_total": len(group_records),
                    "num_files_kept": len(kept_records),
                    "modal_length": modal_length,
                    "num_pairs": 0,
                    "num_pairs_dt_match": 0,
                    "dominant_freq_rel_error_disp_median": float("nan"),
                    "dominant_freq_rel_error_disp_p25": float("nan"),
                    "dominant_freq_rel_error_disp_p10": float("nan"),
                    "dominant_freq_rel_error_disp_min": float("nan"),
                    "disp_psd_rel_error_median": float("nan"),
                    "disp_psd_rel_error_p25": float("nan"),
                    "disp_psd_rel_error_p10": float("nan"),
                    "disp_psd_rel_error_min": float("nan"),
                    "force_psd_rel_error_median": float("nan"),
                    "force_psd_rel_error_p25": float("nan"),
                    "force_psd_rel_error_p10": float("nan"),
                    "force_psd_rel_error_min": float("nan"),
                    "disp_std_rel_error_median": float("nan"),
                    "disp_std_rel_error_p25": float("nan"),
                    "disp_std_rel_error_p10": float("nan"),
                    "disp_std_rel_error_min": float("nan"),
                }
            )
            continue

        num_pairs = len(kept_records) * (len(kept_records) - 1) // 2
        total_pairs += num_pairs
        tracker = ProgressTracker(total=num_pairs, desc=f"Pairwise U_r={ur_key:.{int(args.ur_decimals)}f}", unit="pair")
        group_metric_values: dict[str, list[float]] = defaultdict(list)
        dt_match_count = 0
        for left, right in _iter_pairs(kept_records):
            metrics = _pair_metrics(left, right)
            if metrics["dt_match"] > 0.5:
                dt_match_count += 1
            row = {
                "ur_group": ur_key,
                "file_a": str(left.path.relative_to(PROJECT_ROOT)),
                "file_b": str(right.path.relative_to(PROJECT_ROOT)),
                "length": modal_length,
                "dt_a": left.dt,
                "dt_b": right.dt,
                **metrics,
            }
            pairwise_rows.append(row)
            for key in ("dominant_freq_rel_error_disp", "disp_psd_rel_error", "force_psd_rel_error", "disp_std_rel_error"):
                group_metric_values[key].append(float(metrics[key]))
            tracker.update()
        tracker.close()
        plot_metric_values[float(ur_key)] = {
            key: [float(v) for v in group_metric_values[key]]
            for key, _ in METRIC_SPECS
        }

        dom_stats = _stats_dict(group_metric_values["dominant_freq_rel_error_disp"])
        disp_psd_stats = _stats_dict(group_metric_values["disp_psd_rel_error"])
        force_psd_stats = _stats_dict(group_metric_values["force_psd_rel_error"])
        disp_std_stats = _stats_dict(group_metric_values["disp_std_rel_error"])
        summary_rows.append(
            {
                "ur_group": ur_key,
                "num_files_total": len(group_records),
                "num_files_kept": len(kept_records),
                "modal_length": modal_length,
                "num_pairs": num_pairs,
                "num_pairs_dt_match": dt_match_count,
                "dominant_freq_rel_error_disp_median": dom_stats["median"],
                "dominant_freq_rel_error_disp_p25": dom_stats["p25"],
                "dominant_freq_rel_error_disp_p10": dom_stats["p10"],
                "dominant_freq_rel_error_disp_min": dom_stats["min"],
                "disp_psd_rel_error_median": disp_psd_stats["median"],
                "disp_psd_rel_error_p25": disp_psd_stats["p25"],
                "disp_psd_rel_error_p10": disp_psd_stats["p10"],
                "disp_psd_rel_error_min": disp_psd_stats["min"],
                "force_psd_rel_error_median": force_psd_stats["median"],
                "force_psd_rel_error_p25": force_psd_stats["p25"],
                "force_psd_rel_error_p10": force_psd_stats["p10"],
                "force_psd_rel_error_min": force_psd_stats["min"],
                "disp_std_rel_error_median": disp_std_stats["median"],
                "disp_std_rel_error_p25": disp_std_stats["p25"],
                "disp_std_rel_error_p10": disp_std_stats["p10"],
                "disp_std_rel_error_min": disp_std_stats["min"],
            }
        )

    per_ur_rows = [row for row in summary_rows if isinstance(row.get("ur_group"), (int, float))]

    def _mean_summary_column(name: str) -> float:
        vals = np.asarray(
            [float(row[name]) for row in per_ur_rows if np.isfinite(float(row.get(name, float("nan"))))],
            dtype=float,
        )
        if vals.size == 0:
            return float("nan")
        return float(np.mean(vals))

    summary_rows.append(
        {
            "ur_group": "ALL",
            "num_files_total": len(records),
            "num_files_kept": int(sum(int(row["num_files_kept"]) for row in summary_rows)),
            "modal_length": float("nan"),
            "num_pairs": total_pairs,
            "num_pairs_dt_match": int(sum(int(row["num_pairs_dt_match"]) for row in per_ur_rows)),
            "dominant_freq_rel_error_disp_median": _mean_summary_column("dominant_freq_rel_error_disp_median"),
            "dominant_freq_rel_error_disp_p25": _mean_summary_column("dominant_freq_rel_error_disp_p25"),
            "dominant_freq_rel_error_disp_p10": _mean_summary_column("dominant_freq_rel_error_disp_p10"),
            "dominant_freq_rel_error_disp_min": _mean_summary_column("dominant_freq_rel_error_disp_min"),
            "disp_psd_rel_error_median": _mean_summary_column("disp_psd_rel_error_median"),
            "disp_psd_rel_error_p25": _mean_summary_column("disp_psd_rel_error_p25"),
            "disp_psd_rel_error_p10": _mean_summary_column("disp_psd_rel_error_p10"),
            "disp_psd_rel_error_min": _mean_summary_column("disp_psd_rel_error_min"),
            "force_psd_rel_error_median": _mean_summary_column("force_psd_rel_error_median"),
            "force_psd_rel_error_p25": _mean_summary_column("force_psd_rel_error_p25"),
            "force_psd_rel_error_p10": _mean_summary_column("force_psd_rel_error_p10"),
            "force_psd_rel_error_min": _mean_summary_column("force_psd_rel_error_min"),
            "disp_std_rel_error_median": _mean_summary_column("disp_std_rel_error_median"),
            "disp_std_rel_error_p25": _mean_summary_column("disp_std_rel_error_p25"),
            "disp_std_rel_error_p10": _mean_summary_column("disp_std_rel_error_p10"),
            "disp_std_rel_error_min": _mean_summary_column("disp_std_rel_error_min"),
        }
    )

    skipped_rows = [{"file": str(path.relative_to(PROJECT_ROOT)), "reason": reason} for path, reason in skipped_load]
    _write_csv(output_dir / "pairwise_metrics.csv", pairwise_rows)
    _write_csv(output_dir / "group_summary.csv", summary_rows)
    _write_csv(output_dir / "skipped_files.csv", skipped_rows)
    _save_visualizations(output_dir, plot_metric_values, summary_rows)

    print("")
    print(f"Finished {total_groups} U_r groups and {total_pairs} kept pairs.")
    print(f"Wrote pairwise metrics to {output_dir / 'pairwise_metrics.csv'}")
    print(f"Wrote group summary to {output_dir / 'group_summary.csv'}")
    print(f"Wrote visual summaries to {output_dir / 'metric_distributions_by_ur.png'} and {output_dir / 'metric_summary_vs_ur.png'}")
    if skipped_rows:
        print(f"Wrote skipped file report to {output_dir / 'skipped_files.csv'}")


if __name__ == "__main__":
    main()
