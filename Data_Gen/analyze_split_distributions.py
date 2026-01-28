"""
Exploratory analysis for train/val/test splits of generated time series.

This script summarizes:
  - Reduced velocity (U_r), initial conditions, and metadata distributions
  - Time-series content statistics (mean/std/rms/amplitude/dominant frequency)
  - Per-split CSVs + optional plots
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional
import re

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional
    plt = None


# -------- Configuration --------
ROOT_DIR = Path("Data_Gen/generated_series_Ur")
OUTPUT_DIR = ROOT_DIR / "eda"
MAKE_PLOTS = True

# Histogram bins (shared across splits)
BINS_UR = 20
BINS_IC = 20
BINS_FREQ = 20


@dataclass
class SeriesStats:
    file: str
    split: str
    n: int
    dt: float
    ur: float
    a_factor: Optional[float]
    fhat: Optional[float]
    y0: float
    v0: float
    y_mean: float
    y_std: float
    y_rms: float
    y_min: float
    y_max: float
    y_amp: float
    f_mean: float
    f_std: float
    dom_freq: float


def _find_key(data: dict, keys: Iterable[str]) -> Optional[str]:
    for key in keys:
        if key in data:
            return key
    return None


def _parse_from_filename(path: Path) -> tuple[Optional[float], Optional[float], Optional[float]]:
    pat = re.compile(r"_A(?P<A>-?\d+(\.\d+)?)_fhat(?P<fhat>-?\d+(\.\d+)?)_Ur(?P<ur>-?\d+(\.\d+)?)")
    m = pat.search(path.stem)
    if not m:
        return None, None, None
    return float(m.group("A")), float(m.group("fhat")), float(m.group("ur"))


def _dominant_frequency(y: np.ndarray, dt: float) -> float:
    if y.size < 4 or dt <= 0.0:
        return float("nan")
    y = np.asarray(y, dtype=float)
    y = y - np.mean(y)
    spec = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(y.size, d=dt)
    if spec.size == 0:
        return float("nan")
    spec[0] = 0.0
    idx = int(np.argmax(spec))
    return float(freqs[idx]) if idx < freqs.size else float("nan")


def _load_series(path: Path, split: str) -> SeriesStats:
    with np.load(path) as data:
        t_key = _find_key(data, ("a", "time"))
        y_key = _find_key(data, ("b", "y"))
        f_key = _find_key(data, ("c", "F_total", "force_total", "force"))
        if t_key is None or y_key is None or f_key is None:
            raise ValueError(f"{path.name}: missing required arrays (time/displacement/force).")
        t = np.asarray(data[t_key], dtype=float)
        y = np.asarray(data[y_key], dtype=float)
        f = np.asarray(data[f_key], dtype=float)

        ur = float(np.asarray(data["U_r"]).reshape(-1)[0]) if "U_r" in data else float("nan")

        # optional metadata
        a_factor = float(np.asarray(data["A_factor"]).reshape(-1)[0]) if "A_factor" in data else None
        fhat = float(np.asarray(data["fhat"]).reshape(-1)[0]) if "fhat" in data else None

    if a_factor is None or fhat is None or not np.isfinite(ur):
        a_f, f_h, u_r = _parse_from_filename(path)
        if a_factor is None:
            a_factor = a_f
        if fhat is None:
            fhat = f_h
        if not np.isfinite(ur) and u_r is not None:
            ur = u_r

    if t.size < 2 or y.size != t.size or f.size != t.size:
        raise ValueError(f"{path.name}: invalid time series shapes.")

    dt = float(t[1] - t[0])
    y0 = float(y[0])
    if "dy" in data:
        v0 = float(np.asarray(data["dy"]).reshape(-1)[0])
    elif y.size >= 2 and dt > 0:
        v0 = float((y[1] - y[0]) / dt)
    else:
        v0 = 0.0

    y_mean = float(np.mean(y))
    y_std = float(np.std(y))
    y_rms = float(np.sqrt(np.mean(y * y)))
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    y_amp = 0.5 * (y_max - y_min)
    f_mean = float(np.mean(f))
    f_std = float(np.std(f))
    dom_freq = _dominant_frequency(y, dt)

    return SeriesStats(
        file=path.name,
        split=split,
        n=int(t.size),
        dt=dt,
        ur=float(ur),
        a_factor=a_factor,
        fhat=fhat,
        y0=y0,
        v0=v0,
        y_mean=y_mean,
        y_std=y_std,
        y_rms=y_rms,
        y_min=y_min,
        y_max=y_max,
        y_amp=y_amp,
        f_mean=f_mean,
        f_std=f_std,
        dom_freq=dom_freq,
    )


def _summarize(split: str, stats: list[SeriesStats]) -> None:
    def _arr(vals: Iterable[float]) -> np.ndarray:
        return np.asarray([v for v in vals if np.isfinite(v)], dtype=float)

    print(f"\n== {split} ==")
    print(f"count: {len(stats)}")
    if not stats:
        return

    ur = _arr(s.ur for s in stats)
    y0 = _arr(s.y0 for s in stats)
    v0 = _arr(s.v0 for s in stats)
    amp = _arr(s.y_amp for s in stats)
    domf = _arr(s.dom_freq for s in stats)

    def _fmt(name: str, arr: np.ndarray) -> None:
        if arr.size == 0:
            print(f"{name}: n/a")
            return
        print(f"{name}: mean={arr.mean():.4g}, std={arr.std():.4g}, min={arr.min():.4g}, max={arr.max():.4g}")

    _fmt("U_r", ur)
    _fmt("y0", y0)
    _fmt("v0", v0)
    _fmt("amp", amp)
    _fmt("dom_freq", domf)

    a_vals = _arr(s.a_factor for s in stats if s.a_factor is not None)
    fhat_vals = _arr(s.fhat for s in stats if s.fhat is not None)
    _fmt("A_factor", a_vals)
    _fmt("fhat", fhat_vals)

    # counts per U_r (rounded)
    if ur.size:
        ur_round = np.round(ur, 3)
        unique, counts = np.unique(ur_round, return_counts=True)
        print("U_r counts:", dict(zip(unique.tolist(), counts.tolist())))


def _to_csv(stats: list[SeriesStats], path: Path) -> None:
    if not stats:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(SeriesStats.__annotations__.keys())
    lines = [",".join(fields)]
    for s in stats:
        row = [getattr(s, f) for f in fields]
        row = ["" if v is None else v for v in row]
        lines.append(",".join(str(v) for v in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_hist(ax, values: np.ndarray, bins: int, title: str) -> None:
    if values.size == 0:
        ax.set_title(f"{title} (n/a)")
        ax.axis("off")
        return
    ax.hist(values, bins=bins, alpha=0.8)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def _plot_split_histograms(all_stats: Dict[str, list[SeriesStats]], out_dir: Path) -> None:
    if plt is None:
        print("matplotlib not available; skipping plots.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    def _vals(split: str, key: str) -> np.ndarray:
        vals = [getattr(s, key) for s in all_stats.get(split, [])]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        return np.asarray(vals, dtype=float)

    for split in all_stats.keys():
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        axes = axes.ravel()
        _plot_hist(axes[0], _vals(split, "ur"), BINS_UR, "U_r")
        _plot_hist(axes[1], _vals(split, "a_factor"), BINS_IC, "A_factor")
        _plot_hist(axes[2], _vals(split, "fhat"), BINS_IC, "fhat")
        _plot_hist(axes[3], _vals(split, "y0"), BINS_IC, "y0")
        _plot_hist(axes[4], _vals(split, "v0"), BINS_IC, "v0")
        _plot_hist(axes[5], _vals(split, "dom_freq"), BINS_FREQ, "dominant freq")
        fig.suptitle(f"{split} distributions")
        fig.tight_layout()
        fig.savefig(out_dir / f"{split}_distributions.png", dpi=200)
        plt.close(fig)


def main() -> None:
    splits = {"train": ROOT_DIR / "train", "val": ROOT_DIR / "val", "test": ROOT_DIR / "test"}
    all_stats: Dict[str, list[SeriesStats]] = {}
    for split, path in splits.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing split folder: {path}")
        files = sorted(path.glob("*.npz"))
        stats: list[SeriesStats] = []
        for f in files:
            try:
                stats.append(_load_series(f, split))
            except Exception as exc:
                print(f"[warn] {split}: {f.name}: {exc}")
        all_stats[split] = stats
        _summarize(split, stats)
        _to_csv(stats, OUTPUT_DIR / f"{split}_series_stats.csv")

    if MAKE_PLOTS:
        _plot_split_histograms(all_stats, OUTPUT_DIR / "plots")
    print(f"\nSaved CSVs and plots under {OUTPUT_DIR}.")


if __name__ == "__main__":
    main()
