import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# ------------------------------------------------------------
#                 USER CONFIG SECTION
# ------------------------------------------------------------

LOG_PATHS = [
    "HNNruns/pirate_final1_1130-085457",
    "HNNruns/pirate_final_nosmoothing_1128-085640",
    "HNNruns/pirate_final_nocosine_1128-085640",
    "HNNruns/pirate_final_noreg_1128-085640",
    "HNNruns/pirate_final_rwf_1128-085640",
]

LABELS = [
    "Full Model",
    "No Smoothing",
    "No Scheduler",
    "No Force Reg.",
    "No RWF",
]

METRICS = [
    "val/rel_rmse_y",
    "val/rel_rmse_force_total",
    "train/residual_loss"
]

TITLES = [
    "NRMSEy",
    "NRMSEf",
    "Residual Loss"
]

MIN = [
    None, 
    None, 
    1e-3,
]

MAX = [None, 
       None, 
       1e1,
       ]

WINDOW = 20*20        # smoothing window
STD_WINDOW = 40*20  # std window (often 2× smoothing window)
OUTPUT_DIR = "figs"
DARK_MODE = False

# ---------------------------
# Utility functions
# ---------------------------

def load_scalar(log_path: str, tag: str):
    ea = EventAccumulator(log_path, size_guidance={'scalars': 0})
    ea.Reload()

    tags = ea.Tags().get("scalars", [])
    if tag not in tags:
        raise ValueError(f"Tag '{tag}' not found in {log_path}. Available: {tags}")

    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events], float)
    vals = np.array([e.value for e in events], float)

    # sort
    idx = np.argsort(steps)
    return steps[idx], vals[idx]


def smooth_savgol(values, window, poly=2):
    n = len(values)
    window = min(window, n - (1 - n % 2))
    if window < 3:
        return values
    if window % 2 == 0:
        window -= 1
    return savgol_filter(values, window_length=window, polyorder=poly)


def sliding_std(values, window):
    n = len(values)
    half = window // 2
    out = np.zeros_like(values)
    for i in range(n):
        left = max(0, i - half)
        right = min(n, i + half + 1)
        out[i] = np.std(values[left:right])
    return out


# ---------------------------
# NEW: log-space smoothing + band
# ---------------------------
def plot_metric(metric, title, min, max):
    if DARK_MODE:
        plt.style.use("dark_background")

    fig, ax = plt.subplots(figsize=(8, 5))

    eps = 1e-12

    for path, label in zip(LOG_PATHS, LABELS):
        try:
            steps, vals = load_scalar(path, metric)
        except ValueError as e:
            print(e)
            continue

        # ---- log-transform the values ----
        log_vals = np.log(vals + eps)

        # ---- smooth the mean in log-space ----
        log_mean = smooth_savgol(log_vals, WINDOW)

        # ---- compute std band in log-space ----
        log_std = sliding_std(log_vals, STD_WINDOW)

        # ---- convert back to linear space ----
        mean = np.exp(log_mean)
        lower = np.exp(log_mean - log_std)
        upper = np.exp(log_mean + log_std)

        # ---- plotting ----
        ax.plot(steps, mean, linewidth=2, label=label)
        ax.fill_between(steps, lower, upper, alpha=0.25)

    if min is not None or max is not None:
        ax.set_ylim(bottom=min, top=max)
    else:
        ax.autoscale(enable=True, axis="y")


    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel(title)
    ax.set_yscale("log")
    ax.grid(alpha=0.25, which="both")
    ax.legend()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fname = metric.replace("/", "_") + ".png"
    outpath = os.path.join(OUTPUT_DIR, fname)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

    print(f"Saved: {outpath}")


def main():
    for metric, title, min, max in zip(METRICS, TITLES, MIN, MAX):
        plot_metric(metric, title, min, max)


if __name__ == "__main__":
    main()
