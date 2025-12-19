"""
Plot overlaid histograms for generated time-series (displacement or force).
Each series is shown in a different colour on the same axes for easy comparison.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_series(series_dir: Path):
    files = sorted(series_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {series_dir}")
    series = []
    for path in files:
        arr = np.load(path)
        series.append(
            {
                "path": path,
                "time": np.asarray(arr["a"]),
                "disp": np.asarray(arr["b"]),
                "force": np.asarray(arr["c"]) if "c" in arr else None,
            }
        )
    return series


def plot_hist(series, field: str, bins: int, save_path: Path | None):
    fig, ax = plt.subplots(figsize=(8, 5))
    all_values = np.concatenate([entry[field] for entry in series if entry[field] is not None])
    bin_edges = np.linspace(all_values.min(), all_values.max(), bins + 1)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    width = bin_edges[1] - bin_edges[0]
    colors = plt.cm.tab20(np.linspace(0, 1, len(series)))
    cumulative = np.zeros_like(centers)
    for entry, color in zip(series, colors):
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
    parser = argparse.ArgumentParser(description="Plot histograms for generated TD-model data.")
    parser.add_argument(
        "--series-dir",
        type=Path,
        default=Path(__file__).parent / "generated_series",
        help="Directory containing *.npz series files.",
    )
    parser.add_argument(
        "--field",
        choices=("disp", "force"),
        default="disp",
        help="Which field to plot (displacement or force).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=100,
        help="Number of histogram bins.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional path to save the histogram figure.",
    )
    args = parser.parse_args()

    series = load_series(args.series_dir)
    plot_hist(series, field=args.field, bins=args.bins, save_path=args.save)


if __name__ == "__main__":
    main()
