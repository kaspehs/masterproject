from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np


# -------------------------
# Settings (edit these)
# -------------------------
INPUT_DIR = Path("Experimental_Data/npz_exports_v2")
OUTPUT_DIR = Path("Experimental_Data/npz_exports__v2_zero_mean")
INPUT_GLOB = "*.npz"
RECURSIVE = True

# If True, write back to source files in INPUT_DIR.
WRITE_INPLACE = False
OVERWRITE = True

# Either process specific keys or all numeric keys.
PROCESS_ALL_NUMERIC_KEYS = False
# If True, remove a Hann-window weighted mean instead of plain mean.
USE_HANN_WEIGHTED_MEAN = True
TARGET_KEYS = [
    "F_total",
    "cf_force",
    "c",
    "y",
    "b",
]


def _resolve_files() -> list[Path]:
    base = Path(INPUT_DIR)
    if not base.exists():
        raise FileNotFoundError(f"Input folder not found: {base.resolve()}")
    files = sorted(base.rglob(INPUT_GLOB) if RECURSIVE else base.glob(INPUT_GLOB))
    if not files:
        raise FileNotFoundError(f"No files matched '{INPUT_GLOB}' in {base.resolve()}")
    return [Path(p) for p in files]


def _demean_numeric_array(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values)
    if not np.issubdtype(arr.dtype, np.number):
        return arr, np.asarray([], dtype=float)

    work = np.asarray(arr, dtype=float)
    if work.ndim == 0:
        # Scalars are left unchanged.
        return arr, np.asarray([], dtype=float)

    if work.ndim == 1:
        if bool(USE_HANN_WEIGHTED_MEAN):
            mean = _weighted_mean_1d(work)
        else:
            mean = np.nanmean(work)
        if not np.isfinite(mean):
            mean = 0.0
        out = work - float(mean)
        return np.asarray(out, dtype=float), np.asarray([mean], dtype=float)

    if work.ndim == 2:
        if bool(USE_HANN_WEIGHTED_MEAN):
            means = _weighted_mean_2d_cols(work)
        else:
            means = np.nanmean(work, axis=0)
        means = np.where(np.isfinite(means), means, 0.0)
        out = work - means.reshape(1, -1)
        return np.asarray(out, dtype=float), np.asarray(means, dtype=float)

    # Higher-dimensional arrays are left unchanged.
    return arr, np.asarray([], dtype=float)


def _hann_weights(n: int) -> np.ndarray:
    n_int = int(n)
    if n_int <= 2:
        return np.ones(max(n_int, 0), dtype=float)
    w = np.hanning(n_int).astype(float)
    if not np.isfinite(np.sum(w)) or float(np.sum(w)) <= 0.0:
        return np.ones(n_int, dtype=float)
    return w


def _weighted_mean_1d(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float).reshape(-1)
    if x.size == 0:
        return float("nan")
    w = _hann_weights(x.size)
    finite = np.isfinite(x)
    if not np.any(finite):
        return float("nan")
    x_f = x[finite]
    w_f = w[finite]
    denom = float(np.sum(w_f))
    if not np.isfinite(denom) or denom <= 0.0:
        return float(np.nanmean(x_f))
    return float(np.sum(w_f * x_f) / denom)


def _weighted_mean_2d_cols(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if x.ndim != 2:
        return np.asarray([], dtype=float)
    n_rows, n_cols = x.shape
    w = _hann_weights(n_rows).reshape(-1, 1)
    finite = np.isfinite(x)
    wx = np.where(finite, x * w, 0.0)
    ww = np.where(finite, w, 0.0)
    num = np.sum(wx, axis=0)
    den = np.sum(ww, axis=0)
    out = np.full(n_cols, np.nan, dtype=float)
    good = np.isfinite(den) & (den > 0.0)
    out[good] = num[good] / den[good]
    return out


def _build_output_path(src: Path, *, input_base: Path) -> Path:
    if WRITE_INPLACE:
        return src
    rel = src.relative_to(input_base)
    return Path(OUTPUT_DIR) / rel


def _save_npz_atomic(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        np.savez(tmp_path, **payload)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def main() -> None:
    files = _resolve_files()
    input_base = Path(INPUT_DIR).resolve()
    if not WRITE_INPLACE:
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    changed_files = 0
    for src in files:
        out = _build_output_path(src.resolve(), input_base=input_base)
        if out.exists() and (not OVERWRITE):
            print(f"Skipping existing: {out}")
            continue

        with np.load(src, allow_pickle=False) as data:
            keys = list(data.files)
            if PROCESS_ALL_NUMERIC_KEYS:
                process_keys = set(keys)
            else:
                process_keys = set(str(k) for k in TARGET_KEYS)

            payload: dict[str, np.ndarray] = {}
            touched: list[str] = []
            for key in keys:
                arr = np.asarray(data[key])
                if key in process_keys and np.issubdtype(arr.dtype, np.number):
                    out_arr, removed_means = _demean_numeric_array(arr)
                    payload[key] = out_arr
                    if removed_means.size > 0:
                        touched.append(key)
                else:
                    payload[key] = arr

        _save_npz_atomic(out, payload)
        changed_files += 1
        touched_str = ", ".join(touched) if touched else "none"
        print(f"Wrote {out} | mean-removed keys: {touched_str}")

    print(f"Done. Processed {changed_files} file(s).")


if __name__ == "__main__":
    main()
