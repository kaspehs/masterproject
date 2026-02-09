"""
Plot overlaid histograms for generated time-series (displacement or force).
Each series is shown in a different colour on the same axes for easy comparison.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CONFIG = {
    "series_dir": Path(__file__).parent / "generated_series_Ur_long",
    # If series_dir has train/val/test subfolders, select which to include.
    "splits": ["train", "val", "test"],
    # Optional time trim (seconds from series start)
    "cut_start_seconds": 25.0,
    # Reduced velocity filtering (use one of: list or range; None disables)
    "ur_include": [12.0],  # e.g. [2.0, 4.0, 6.0]
    "ur_range": None,    # e.g. (2.0, 6.0)
    "ur_tol": 1e-3,
    # Histogram settings
    "field": "disp",  # "disp" or "force"
    "bins": 100,
    "save": None,  # e.g. Path("figs/hist.png")
}


def _collect_series_files(series_dir: Path, splits: list[str]) -> list[Path]:
    split_dirs = []
    for name in splits:
        d = series_dir / name
        if d.exists() and d.is_dir():
            split_dirs.append(d)
    if split_dirs:
        files: list[Path] = []
        for d in split_dirs:
            files.extend(sorted(d.glob("*.npz")))
        return files
    return sorted(series_dir.glob("*.npz"))


def _extract_ur(arr: np.lib.npyio.NpzFile) -> float | None:
    if "U_r" in arr:
        ur = np.asarray(arr["U_r"], dtype=float)
        if ur.ndim == 0:
            return float(ur)
        ur_flat = ur.reshape(-1)
        if ur_flat.size > 0:
            return float(ur_flat[0])
    return None


def _ur_allowed(ur: float | None, *, ur_include, ur_range, ur_tol: float) -> bool:
    if ur is None:
        return False
    if ur_include is not None:
        return any(abs(float(ur) - float(u)) <= float(ur_tol) for u in ur_include)
    if ur_range is not None:
        lo, hi = ur_range
        return float(lo) <= float(ur) <= float(hi)
    return True


def _apply_cut(t: np.ndarray, *arrays: np.ndarray, cut_start_seconds: float):
    if cut_start_seconds <= 0.0:
        return (t, *arrays)
    t0 = float(t[0])
    mask = t >= (t0 + float(cut_start_seconds))
    t2 = t[mask]
    out = [arr[mask] for arr in arrays]
    return (t2, *out)


def load_series(
    series_dir: Path,
    *,
    cut_start_seconds: float = 0.0,
    ur_include=None,
    ur_range=None,
    ur_tol: float = 1e-3,
    splits: list[str] | None = None,
):
    splits = splits or []
    files = _collect_series_files(series_dir, splits)
    if not files:
        raise FileNotFoundError(f"No .npz files found in {series_dir}")
    series = []
    for path in files:
        arr = np.load(path)
        ur_val = _extract_ur(arr)
        if not _ur_allowed(ur_val, ur_include=ur_include, ur_range=ur_range, ur_tol=ur_tol):
            continue
        time = np.asarray(arr["a"])
        disp = np.asarray(arr["b"])
        force = np.asarray(arr["c"]) if "c" in arr else None
        if time.shape[0] != disp.shape[0]:
            raise ValueError(f"{path.name}: time/disp length mismatch.")
        if force is not None and force.shape[0] != time.shape[0]:
            raise ValueError(f"{path.name}: time/force length mismatch.")
        if force is None:
            time, disp = _apply_cut(time, disp, cut_start_seconds=cut_start_seconds)
        else:
            time, disp, force = _apply_cut(time, disp, force, cut_start_seconds=cut_start_seconds)
        if time.shape[0] != disp.shape[0]:
            raise ValueError(f"{path.name}: time/disp length mismatch after cut.")
        if force is not None and force.shape[0] != time.shape[0]:
            raise ValueError(f"{path.name}: time/force length mismatch after cut.")
        series.append(
            {
                "path": path,
                "time": time,
                "disp": disp,
                "force": force,
                "ur": ur_val,
            }
        )
    return series


def plot_hist(series, field: str, bins: int, save_path: Path | None):
    fig, ax = plt.subplots(figsize=(8, 5))
    all_values = np.concatenate([entry[field] for entry in series if entry[field] is not None])
    bin_edges = np.linspace(all_values.min(), all_values.max(), bins + 1)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    width = bin_edges[1] - bin_edges[0]
    tol = float(CONFIG.get("ur_tol", 1e-3))
    ur_keys = []
    for entry in series:
        ur_val = entry.get("ur", None)
        if ur_val is None:
            ur_keys.append(None)
        else:
            tol_val = tol if tol > 0.0 else 1e-6
            key = float(np.round(float(ur_val) / tol_val) * tol_val)
            ur_keys.append(key)
    unique_keys = sorted({k for k in ur_keys if k is not None})
    palette = plt.cm.viridis(np.linspace(0, 1, max(len(unique_keys), 1)))
    color_map = {key: palette[i] for i, key in enumerate(unique_keys)}
    default_color = (0.4, 0.4, 0.4, 0.7)
    cumulative = np.zeros_like(centers)
    for entry, key in zip(series, ur_keys):
        color = color_map.get(key, default_color)
        values = entry[field]
        if values is None:
            continue
        counts, _ = np.histogram(values, bins=bin_edges)
        ax.bar(
            centers,
            counts,
            width=width,
            bottom=cumulative,
            color=color,
            alpha=0.8,
            label=entry["path"].stem,
        )
        cumulative += counts
    ax.set_xlabel(field)
    ax.set_ylabel("Count")
    ax.set_title(f"Stacked histogram of {field} across series")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize="small")
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        print(f"Saved histogram to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def main():
    series = load_series(
        CONFIG["series_dir"],
        cut_start_seconds=float(CONFIG["cut_start_seconds"]),
        ur_include=CONFIG["ur_include"],
        ur_range=CONFIG["ur_range"],
        ur_tol=float(CONFIG["ur_tol"]),
        splits=list(CONFIG["splits"]),
    )
    plot_hist(series, field=str(CONFIG["field"]), bins=int(CONFIG["bins"]), save_path=CONFIG["save"])


if __name__ == "__main__":
    main()
