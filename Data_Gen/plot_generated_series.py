"""Visualize generated TD-model time series (phase portrait only by default)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


CONFIG = {
    # Data selection
    "series_dir": Path(__file__).parent / "generated_series_from_mat_velocity",
    # If series_dir has train/val/test subfolders, select which to include.
    "splits": ["train", "val", "test"],
    # Optional time trim (seconds from series start)
    "cut_start_seconds": 0.0,
    # Reduced velocity filtering (use one of: list or range; None disables)
    "ur_include": None,  # e.g. [2.0, 4.0, 6.0]
    "ur_range": None,    # e.g. (2.0, 6.0)
    "ur_tol": 1e-3,
    # Plotting
    "plot_timeseries": True,
    "plot_phase": True,
    "columns": 4,
    "save_timeseries": None,  # e.g. Path("figs/series.png")
    "save_phase": None,       # e.g. Path("figs/phase.png")
}


def _savgol_smooth(signal: np.ndarray, window: int, poly: int) -> np.ndarray:
    n = signal.size
    if n < 3 or window <= 1:
        return signal.copy()
    window = min(window, n - (1 - n % 2))
    if window % 2 == 0:
        window -= 1
    if window < 3 or poly >= window:
        return signal.copy()
    return savgol_filter(signal, window_length=window, polyorder=poly)


def _load_velocity(arr: np.lib.npyio.NpzFile) -> np.ndarray:
    for key in ("dy", "v", "e"):
        if key in arr:
            return np.asarray(arr[key]).reshape(-1)
    raise KeyError("No ground-truth velocity found (expected 'dy', 'v', or 'e').")


def _extract_ur(arr: np.lib.npyio.NpzFile, fallback: float | None = None) -> float | None:
    if "U_r" in arr:
        ur = np.asarray(arr["U_r"], dtype=float)
        if ur.ndim == 0:
            return float(ur)
        ur_flat = ur.reshape(-1)
        if ur_flat.size > 0:
            return float(ur_flat[0])
    return fallback


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


def load_series(
    series_dir: Path,
    window: int,
    poly: int,
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
    data: list[dict[str, object]] = []
    for f in files:
        arr = np.load(f)
        ur_val = _extract_ur(arr)
        if not _ur_allowed(ur_val, ur_include=ur_include, ur_range=ur_range, ur_tol=ur_tol):
            continue
        time = np.asarray(arr["a"])
        disp = np.asarray(arr["b"])
        force = np.asarray(arr["c"])
        if time.shape[0] != disp.shape[0] or time.shape[0] != force.shape[0]:
            raise ValueError(f"{f.name}: time/disp/force length mismatch.")
        vel = _load_velocity(arr)
        if vel.size < disp.size:
            raise ValueError("Velocity shorter than displacement.")
        vel = vel[: disp.size]
        time, disp, force, vel = _apply_cut(time, disp, force, vel, cut_start_seconds=cut_start_seconds)
        if time.shape[0] != disp.shape[0] or time.shape[0] != force.shape[0] or time.shape[0] != vel.shape[0]:
            raise ValueError(f"{f.name}: length mismatch after cut.")
        disp_smooth = _savgol_smooth(disp, window, poly)
        vel_smooth = vel
        data.append(
            {
                "path": f,
                "split": f.parent.name if f.parent != series_dir else "series",
                "time": time,
                "disp": disp,
                "force": force,
                "disp_smooth": disp_smooth,
                "vel_smooth": vel_smooth,
                "ur": ur_val,
                "is_validation": False,
            }
        )
    return data


def plot_series(series_data, columns: int = 4, save_path: Path | None = None):
    rows = int(np.ceil(len(series_data) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 3 * rows), sharex=False)
    axes = np.atleast_1d(axes).ravel()

    for ax, entry in zip(axes, series_data):
        disp = entry["disp_smooth"]
        force = entry["force"]
        disp_scale = np.max(np.abs(disp)) or 1.0
        force_scale = np.max(np.abs(force)) or 1.0
        ax.plot(entry["time"], disp / disp_scale, label=f"y / {disp_scale:.2e}")
        ax.plot(entry["time"], force / force_scale, label=f"F / {force_scale:.2e}")
        ax.set_title(entry["path"].name)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize="small")

    for ax in axes[len(series_data) :]:
        ax.axis("off")

    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        print(f"Saved figure to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def _ur_color_key(ur: float | None, ur_tol: float) -> float | None:
    if ur is None:
        return None
    tol = float(ur_tol) if float(ur_tol) > 0.0 else 1e-6
    return float(np.round(float(ur) / tol) * tol)


def plot_phase(series_data, save_path: Path | None = None, *, ur_tol: float = 1e-3):
    fig, ax = plt.subplots(figsize=(6, 6))
    ur_keys = [_ur_color_key(entry["ur"], ur_tol) for entry in series_data]
    unique_keys = sorted({k for k in ur_keys if k is not None})
    colors = plt.cm.viridis(np.linspace(0, 1, max(len(unique_keys), 1)))
    color_map = {key: colors[i] for i, key in enumerate(unique_keys)}
    default_color = (0.4, 0.4, 0.4, 0.7)
    for entry, key in zip(series_data, ur_keys):
        color = color_map.get(key, default_color)
        ax.plot(entry["disp_smooth"], entry["vel_smooth"], color=color, alpha=0.7, label=entry["path"].stem)
        ax.scatter(entry["disp_smooth"][0], entry["vel_smooth"][0], color=color, s=30)
    ax.set_xlabel("Displacement")
    ax.set_ylabel("Velocity")
    ax.set_title("Phase Portrait")
    ax.grid(True, alpha=0.3)
    #ax.legend(ncol=2, fontsize="small")
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        print(f"Saved phase plot to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def main():
    series = load_series(
        CONFIG["series_dir"],
        window=1,
        poly=1,
        cut_start_seconds=float(CONFIG["cut_start_seconds"]),
        ur_include=CONFIG["ur_include"],
        ur_range=CONFIG["ur_range"],
        ur_tol=float(CONFIG["ur_tol"]),
        splits=list(CONFIG["splits"]),
    )
    if bool(CONFIG["plot_timeseries"]):
        plot_series(series, columns=int(CONFIG["columns"]), save_path=CONFIG["save_timeseries"])
    if bool(CONFIG["plot_phase"]):
        plot_phase(series, save_path=CONFIG["save_phase"], ur_tol=float(CONFIG["ur_tol"]))


if __name__ == "__main__":
    main()
