"""Visualize generated TD-model time series with smoothing and phase portrait."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


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


def _savgol_velocity(signal: np.ndarray, dt: float, window: int, poly: int) -> np.ndarray:
    n = signal.size
    if n < 3 or dt <= 0.0:
        return np.zeros_like(signal)
    window = min(window, n - (1 - n % 2))
    if window % 2 == 0:
        window -= 1
    if window >= 3 and poly < window:
        try:
            return savgol_filter(
                signal,
                window_length=window,
                polyorder=poly,
                deriv=1,
                delta=dt,
                mode="interp",
            )
        except ValueError:
            pass
    vel = np.zeros_like(signal)
    vel[0] = (signal[1] - signal[0]) / dt if n >= 2 else 0.0
    vel[-1] = (signal[-1] - signal[-2]) / dt if n >= 2 else 0.0
    if n > 2:
        vel[1:-1] = (signal[2:] - signal[:-2]) / (2.0 * dt)
    return vel


def load_series(series_dir: Path, window: int, poly: int, val_file: Path | None = None):
    files = sorted(series_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {series_dir}")
    data: list[dict[str, object]] = []
    for f in files:
        arr = np.load(f)
        time = np.asarray(arr["a"])
        disp = np.asarray(arr["b"])
        force = np.asarray(arr["c"])
        dt = float(time[1] - time[0]) if time.size > 1 else 1.0
        disp_smooth = _savgol_smooth(disp, window, poly)
        vel_smooth = _savgol_velocity(disp, dt, window, poly)
        data.append(
            {
                "path": f,
                "time": time,
                "disp": disp,
                "force": force,
                "disp_smooth": disp_smooth,
                "vel_smooth": vel_smooth,
                "is_validation": False,
            }
        )
    if val_file is not None and val_file.exists():
        arr = np.load(val_file)
        time = np.asarray(arr["a"])
        disp = np.asarray(arr["b"])
        force = np.asarray(arr["c"]) if "c" in arr else np.zeros_like(disp)
        dt = float(time[1] - time[0]) if time.size > 1 else 1.0
        disp_smooth = _savgol_smooth(disp, window, poly)
        vel_smooth = _savgol_velocity(disp, dt, window, poly)
        data.append(
            {
                "path": val_file,
                "time": time,
                "disp": disp,
                "force": force,
                "disp_smooth": disp_smooth,
                "vel_smooth": vel_smooth,
                "is_validation": True,
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


def plot_phase(series_data, save_path: Path | None = None):
    fig, ax = plt.subplots(figsize=(6, 6))
    non_val = [entry for entry in series_data if not entry.get("is_validation")]
    colors = plt.cm.tab20(np.linspace(0, 1, len(non_val))) if non_val else []
    for entry, color in zip(non_val, colors):
        ax.plot(entry["disp_smooth"], entry["vel_smooth"], color=color, alpha=0.7, label=entry["path"].stem)
        ax.scatter(entry["disp_smooth"][0], entry["vel_smooth"][0], color=color, s=30)
    val_entries = [entry for entry in series_data if entry.get("is_validation")]
    for entry in val_entries:
        ax.plot(
            entry["disp_smooth"],
            entry["vel_smooth"],
            color="black",
            linewidth=2,
            alpha=0.9,
            label=f"Validation ({entry['path'].stem})",
        )
        ax.scatter(entry["disp_smooth"][0], entry["vel_smooth"][0], color="black", s=50)
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
    parser = argparse.ArgumentParser(description="Plot generated TD-model time series.")
    parser.add_argument("--series-dir", type=Path, default=Path(__file__).parent / "generated_series")
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--save", type=Path, default=None, help="Output path for time-series plot.")
    parser.add_argument("--phase-save", type=Path, default=None, help="Output path for phase plot.")
    parser.add_argument("--savgol-window", type=int, default=15)
    parser.add_argument("--savgol-poly", type=int, default=4)
    parser.add_argument("--validation-file", type=Path, default=Path("data.npz"))
    args = parser.parse_args()

    series = load_series(
        args.series_dir,
        window=args.savgol_window,
        poly=args.savgol_poly,
        val_file=args.validation_file,
    )
    plot_series(series, columns=args.columns, save_path=args.save)
    plot_phase(series, save_path=args.phase_save)


if __name__ == "__main__":
    main()
