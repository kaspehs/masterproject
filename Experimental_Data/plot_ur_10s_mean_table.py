from __future__ import annotations

import os
import sys
from importlib import reload
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Avoid matplotlib cache writes outside workspace on restricted environments.
_MPLCONFIGDIR_DEFAULT = Path("/tmp/matplotlib")
_MPLCONFIGDIR_DEFAULT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR_DEFAULT))
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

try:
    import Experimental_Data.analyze_experimental_data as analysis
except ModuleNotFoundError as exc:
    if getattr(exc, "name", "") != "Experimental_Data":
        raise
    current_dir = Path("Experimental_Data").resolve()
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    import analyze_experimental_data as analysis

analysis = reload(analysis)

# Edit these constants directly.
WINDOW_SECONDS = 5.0
MAX_TIME_SECONDS = 100.0
SORT_BY_MEAN_UR = True
TABLE_TITLE = "10 s Mean Reduced Velocity (U_r), First 100 s"
VALUE_DECIMALS = 3
SHOW_PLOT = True
SAVE_PNG = False
SAVE_PNG_PATH = Path("Experimental_Data/figs/ur_10s_mean_table.png")
SAVE_PNG_DPI = 300

# Input MAT-file selection
USE_MAT_DIR_SELECTION = True
MAT_DIR = Path("CrossFlow/CorrectedData")
MAT_DIR_PATTERN = "*.mat"
MAT_DIR_RECURSIVE = False


def _resolve_existing_dir(path_like: Path | str) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        if p.exists() and p.is_dir():
            return p
        raise FileNotFoundError(f"Directory does not exist: {p}")

    # Try relative to cwd first.
    cand_cwd = (Path.cwd() / p).resolve()
    if cand_cwd.exists() and cand_cwd.is_dir():
        return cand_cwd

    # Then try relative to Experimental_Data.
    cand_exp = (Path.cwd() / "Experimental_Data" / p).resolve()
    if cand_exp.exists() and cand_exp.is_dir():
        return cand_exp

    raise FileNotFoundError(
        f"Directory does not exist: {p} (tried {cand_cwd} and {cand_exp})"
    )


def _resolve_mat_files_for_table() -> tuple[list[Path], str]:
    if bool(USE_MAT_DIR_SELECTION):
        mat_dir = _resolve_existing_dir(MAT_DIR)
        pattern = str(MAT_DIR_PATTERN).strip() if str(MAT_DIR_PATTERN).strip() else "*.mat"
        if bool(MAT_DIR_RECURSIVE):
            files = sorted(mat_dir.rglob(pattern))
        else:
            files = sorted(mat_dir.glob(pattern))
        if not files:
            raise ValueError(f"No MAT files found in {mat_dir} with pattern '{pattern}'.")
        return [Path(p) for p in files], f"folder: {mat_dir} (pattern='{pattern}', recursive={bool(MAT_DIR_RECURSIVE)})"

    files = analysis._resolve_mat_files()
    if not files:
        raise ValueError("No MAT files from analysis._resolve_mat_files().")
    return [Path(p) for p in files], "analysis._resolve_mat_files()"


def _window_means(
    time_vec: np.ndarray,
    ur_vec: np.ndarray,
    window_seconds: float,
    max_time_seconds: float,
) -> np.ndarray:
    t = np.asarray(time_vec, dtype=float).reshape(-1)
    ur = np.asarray(ur_vec, dtype=float).reshape(-1)
    n = int(min(t.size, ur.size))
    if n == 0:
        return np.asarray([], dtype=float)

    t = t[:n]
    ur = ur[:n]
    finite = np.isfinite(t) & np.isfinite(ur)
    t = t[finite]
    ur = ur[finite]
    if t.size == 0:
        return np.asarray([], dtype=float)

    order = np.argsort(t)
    t = t[order]
    ur = ur[order]

    # Always define "first N seconds" relative to the first available sample.
    t = t - float(t[0])
    in_window = (t >= 0.0) & (t <= float(max_time_seconds))
    t = t[in_window]
    ur = ur[in_window]
    if t.size == 0:
        return np.asarray([], dtype=float)

    t0 = float(t[0])
    duration = float(t[-1] - t0)
    if duration <= 0.0:
        return np.asarray([float(np.nanmean(ur))], dtype=float)

    n_windows = int(np.ceil(duration / float(window_seconds)))
    out = np.full(n_windows, np.nan, dtype=float)
    for i in range(n_windows):
        start = t0 + i * float(window_seconds)
        end = start + float(window_seconds)
        if i == n_windows - 1:
            mask = (t >= start) & (t <= end)
        else:
            mask = (t >= start) & (t < end)
        if np.any(mask):
            out[i] = float(np.nanmean(ur[mask]))
    return out


def _format_cell(x: float) -> str:
    if not np.isfinite(x):
        return ""
    return f"{x:.{int(VALUE_DECIMALS)}f}"


def main() -> None:
    mat_files, source_desc = _resolve_mat_files_for_table()
    print(f"U_r mean table source: {source_desc}")
    print(f"Selected {len(mat_files)} MAT file(s).")
    entries = [analysis._process_file(p) for p in mat_files]

    if bool(SORT_BY_MEAN_UR):
        entries = sorted(
            entries,
            key=lambda e: float(np.nanmean(np.asarray(e["ur_inst"], dtype=float))),
        )

    row_labels: list[str] = []
    per_row_means: list[np.ndarray] = []
    dt_list: list[float] = []

    for entry in entries:
        label = str(entry["label"])
        t = np.asarray(entry["time_plot"], dtype=float)
        ur = np.asarray(entry["ur_inst"], dtype=float)
        means = _window_means(
            t,
            ur,
            window_seconds=float(WINDOW_SECONDS),
            max_time_seconds=float(MAX_TIME_SECONDS),
        )
        row_labels.append(label)
        per_row_means.append(means)

        if t.size >= 2:
            dt = float(np.nanmedian(np.diff(t)))
            dt_list.append(dt if np.isfinite(dt) and dt > 0 else np.nan)
        else:
            dt_list.append(np.nan)

    if not per_row_means:
        raise RuntimeError("No entries found; check MAT_FILES/MAT_GLOB settings in analyze_experimental_data.py")

    max_windows = max(1, int(np.ceil(float(MAX_TIME_SECONDS) / float(WINDOW_SECONDS))))
    table_values = np.full((len(per_row_means), max_windows), np.nan, dtype=float)
    for i, arr in enumerate(per_row_means):
        n_copy = min(arr.size, max_windows)
        table_values[i, :n_copy] = arr[:n_copy]

    col_labels = [
        f"{int(i * WINDOW_SECONDS)}-{int((i + 1) * WINDOW_SECONDS)} s"
        for i in range(max_windows)
    ]

    finite_vals = table_values[np.isfinite(table_values)]
    if finite_vals.size:
        vmin = float(np.nanmin(finite_vals))
        vmax = float(np.nanmax(finite_vals))
    else:
        vmin, vmax = 0.0, 1.0
    span = max(vmax - vmin, 1e-12)
    cmap = plt.get_cmap("viridis")

    cell_colours: list[list[tuple[float, float, float, float]]] = []
    cell_text: list[list[str]] = []
    for row in table_values:
        color_row = []
        text_row = []
        for x in row:
            if np.isfinite(x):
                z = (float(x) - vmin) / span
                color_row.append(cmap(z))
            else:
                color_row.append((0.94, 0.94, 0.94, 1.0))
            text_row.append(_format_cell(float(x)))
        cell_colours.append(color_row)
        cell_text.append(text_row)

    fig_w = max(10.0, 1.25 * max_windows + 2.8)
    fig_h = max(3.0, 0.48 * len(row_labels) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=cell_text,
        cellColours=cell_colours,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.35)
    ax.set_title(TABLE_TITLE)

    # Print change-point hint per test (largest jump between adjacent 10 s window means).
    print(
        f"\nLargest adjacent-window U_r change using {WINDOW_SECONDS:g}s means "
        f"(first {MAX_TIME_SECONDS:g}s)"
    )
    for i, label in enumerate(row_labels):
        row = table_values[i]
        finite = np.isfinite(row)
        if np.count_nonzero(finite) < 2:
            print(f"- {label}: not enough windows")
            continue
        valid_idx = np.flatnonzero(finite)
        vals = row[valid_idx]
        diffs = np.abs(np.diff(vals))
        k = int(np.argmax(diffs))
        left_idx = int(valid_idx[k])
        right_idx = int(valid_idx[k + 1])
        change_time_s = float(right_idx * WINDOW_SECONDS)

        dt = dt_list[i]
        if np.isfinite(dt) and dt > 0:
            suggested_drop = int(round(change_time_s / dt))
            print(
                f"- {label}: largest change between {left_idx*WINDOW_SECONDS:.0f}-{(left_idx+1)*WINDOW_SECONDS:.0f}s "
                f"and {right_idx*WINDOW_SECONDS:.0f}-{(right_idx+1)*WINDOW_SECONDS:.0f}s "
                f"(suggest DROP_FIRST_TIME_SAMPLES ~ {suggested_drop})"
            )
        else:
            print(
                f"- {label}: largest change between {left_idx*WINDOW_SECONDS:.0f}-{(left_idx+1)*WINDOW_SECONDS:.0f}s "
                f"and {right_idx*WINDOW_SECONDS:.0f}-{(right_idx+1)*WINDOW_SECONDS:.0f}s"
            )

    fig.tight_layout()
    if bool(SAVE_PNG):
        save_path = Path(SAVE_PNG_PATH).resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=int(SAVE_PNG_DPI), bbox_inches="tight")
        print(f"\nSaved figure to: {save_path}")
    if bool(SHOW_PLOT):
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
