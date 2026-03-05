"""
Plot grouped histograms for generated time-series (displacement or force).
One subplot is created per mean-U_r group.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CONFIG = {
    "series_dir": Path(__file__).parent / "generated_series_from_mat_velocity",
    # If series_dir has train/val/test subfolders, select which to include.
    "splits": ["train", "val", "test"],
    # Optional time trim (seconds from series start)
    "cut_start_seconds": 0.0,
    # Reduced velocity filtering (use one of: list or range; None disables)
    "ur_include": None,  # e.g. [2.0, 4.0, 6.0]
    "ur_range": None,    # e.g. (2.0, 6.0)
    "ur_tol": 1e-3,
    # Group series by close mean reduced velocity and produce one histogram per group.
    # A series with mean U_r is assigned to key round(U_r / ur_group_width) * ur_group_width.
    "ur_group_width": 1.0,
    # Histogram settings
    "field": "force",  # "disp" or "force"
    "bins": 100,
    # Subplot grid for grouped histograms (all groups in one figure).
    # Example: 4 columns gives a 4xN layout.
    "subplot_cols": 4,
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


def _extract_ur_mean(arr: np.lib.npyio.NpzFile) -> float | None:
    if "U_r" in arr:
        ur = np.asarray(arr["U_r"], dtype=float)
        if ur.ndim == 0:
            return float(ur)
        ur_flat = ur.reshape(-1)
        if ur_flat.size > 0:
            return float(np.mean(ur_flat))
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
        ur_val = _extract_ur_mean(arr)
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
                "ur_mean": ur_val,
            }
        )
    return series


def _group_series_by_ur_mean(series, *, group_width: float, ur_tol: float) -> list[tuple[float, list[dict]]]:
    if group_width <= 0.0:
        raise ValueError("ur_group_width must be > 0.")
    groups: dict[float, list[dict]] = {}
    for entry in series:
        ur_val = entry.get("ur_mean", None)
        if ur_val is None:
            continue
        # Robust keying for floating values.
        step = float(group_width)
        key = float(np.round(float(ur_val) / step) * step)
        merged_key = None
        for existing in groups:
            if abs(existing - key) <= max(float(ur_tol), 1e-12):
                merged_key = existing
                break
        if merged_key is None:
            merged_key = key
            groups[merged_key] = []
        groups[merged_key].append(entry)
    return sorted(groups.items(), key=lambda kv: kv[0])


def _plot_hist_one_group(
    ax,
    group_key: float,
    entries: list[dict],
    *,
    field: str,
    bins: int,
    group_width: float,
):
    valid = [e for e in entries if e.get(field) is not None]
    if not valid:
        ax.set_visible(False)
        return
    all_values = np.concatenate([np.asarray(e[field], dtype=float).reshape(-1) for e in valid])
    if all_values.size == 0:
        ax.set_visible(False)
        return

    bin_edges = np.linspace(all_values.min(), all_values.max(), bins + 1)
    colors = plt.cm.viridis(np.linspace(0.1, 0.95, max(len(valid), 2)))

    for idx, entry in enumerate(valid):
        values = np.asarray(entry[field], dtype=float).reshape(-1)
        ax.hist(
            values,
            bins=bin_edges,
            alpha=0.35,
            color=colors[idx % len(colors)],
            label=entry["path"].stem,
            edgecolor="none",
        )

    lo = float(group_key) - 0.5 * float(group_width)
    hi = float(group_key) + 0.5 * float(group_width)
    ax.set_ylabel("Count")
    ax.set_title(f"{field} histogram for mean U_r in [{lo:.3f}, {hi:.3f})")
    ax.grid(True, alpha=0.3)


def plot_hist(series, field: str, bins: int, save_path: Path | None):
    group_width = float(CONFIG.get("ur_group_width", 0.5))
    ur_tol = float(CONFIG.get("ur_tol", 1e-3))
    grouped = _group_series_by_ur_mean(series, group_width=group_width, ur_tol=ur_tol)
    if not grouped:
        raise ValueError("No series remained after U_r filtering/grouping.")

    n_groups = len(grouped)
    ncols = int(max(1, CONFIG.get("subplot_cols", 4)))
    nrows = int(np.ceil(n_groups / ncols))
    fig_width = max(4.0 * ncols, 10.0)
    fig_height = max(2.8 * nrows, 4.0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), squeeze=False, sharex=True)
    axes_flat = axes.ravel()

    for ax, (key, entries) in zip(axes_flat, grouped):
        _plot_hist_one_group(
            ax,
            key,
            entries,
            field=field,
            bins=bins,
            group_width=group_width,
        )
    for ax in axes_flat[n_groups:]:
        ax.set_visible(False)
    for idx, ax in enumerate(axes_flat[:n_groups]):
        if idx // ncols == (nrows - 1):
            ax.set_xlabel(field)
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        print(f"Saved histogram figure to {save_path}")
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
