#!/usr/bin/env python3
import numpy as np
from pathlib import Path

def main():
    series_dir = Path("Data_Gen/generated_series")
    files = sorted(series_dir.glob("*.npz"))
    if not files:
        print(f"No .npz files found under {series_dir}")
        return

    means = []
    for path in files:
        data = np.load(path)
        if "c" not in data:
            print(f"Skipping {path.name}: missing force array 'c'")
            continue
        force = np.asarray(data["c"])
        mean_abs = float(np.mean(np.abs(force)))
        print(f"{path.name}: mean |force| = {mean_abs:.6f}")
        means.append(mean_abs)

    if means:
        print(f"Overall mean |force| across {len(means)} series: {sum(means)/len(means):.6f}")
    else:
        print("No valid force arrays processed.")

if __name__ == "__main__":
    main()
