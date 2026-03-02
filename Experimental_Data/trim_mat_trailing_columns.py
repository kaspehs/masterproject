from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat, savemat

# Edit these constants directly.
ACTION = "trim_trailing"  # "trim_trailing" | "split_nan_gap"
INPUT_MAT_PATH = Path("Experimental_Data/CrossFlow/CorrectedData/test3006_corrected.mat")
DATA_KEY = "data"
# If None, time axis is inferred as the longest axis in `data`.
TIME_AXIS: int | None = None

# Trim mode settings.
OUTPUT_MAT_PATH: Path | None = Path("Experimental_Data/CrossFlow/CleanedCorrectedData/test3006_corrected.mat")
DROP_FIRST_TIME_SAMPLES = 6000
DROP_LAST_TIME_SAMPLES = 0
OVERWRITE_INPLACE = False

# Split mode settings.
SPLIT_OUTPUT_PREFIX: str | None = None  # Default: <input_stem>
USE_LARGEST_NAN_BLOCK = True
NAN_BLOCK_INDEX = 0  # used only when USE_LARGEST_NAN_BLOCK=False
MIN_SEGMENT_TIME_SAMPLES = 1
# In classic MAT (scipy loadmat/savemat) mode, also slice other numeric arrays
# that share the same time-length axis as DATA_KEY (e.g. time, y_corrected, U_r).
SLICE_MATCHING_NUMERIC_ARRAYS_FOR_MAT = True


def _resolve_time_axis(shape: tuple[int, ...], configured_axis: int | None) -> int:
    if configured_axis is None:
        return int(np.argmax(np.asarray(shape)))
    axis = int(configured_axis)
    if axis < 0:
        axis += len(shape)
    if not (0 <= axis < len(shape)):
        raise ValueError(f"TIME_AXIS={configured_axis} is invalid for shape {shape}.")
    return axis


def _compute_create_kwargs(ds: h5py.Dataset, new_shape: tuple[int, ...]) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "compression": ds.compression,
        "compression_opts": ds.compression_opts,
        "shuffle": bool(ds.shuffle),
        "fletcher32": bool(ds.fletcher32),
    }
    chunks = ds.chunks
    if chunks is not None:
        chunk_list = list(chunks)
        for i, size in enumerate(new_shape):
            chunk_list[i] = min(int(chunk_list[i]), int(size))
        kwargs["chunks"] = tuple(chunk_list)
    return kwargs


def _slice_along_axis(arr: np.ndarray, axis: int, start_idx: int, end_idx: int) -> np.ndarray:
    slices = [slice(None)] * arr.ndim
    slices[int(axis)] = slice(int(start_idx), int(end_idx))
    return arr[tuple(slices)]


def _is_hdf5_mat(path: Path) -> bool:
    return bool(h5py.is_hdf5(str(path)))


def _read_data_array(src_path: Path, data_key: str) -> np.ndarray:
    if _is_hdf5_mat(src_path):
        with h5py.File(src_path, "r") as src:
            if data_key not in src:
                raise KeyError(f"Dataset '{data_key}' not found in '{src_path}'.")
            return np.asarray(src[data_key])

    raw = loadmat(src_path, squeeze_me=False, struct_as_record=False)
    if data_key not in raw:
        keys = sorted(k for k in raw.keys() if not k.startswith("__"))
        raise KeyError(f"Variable '{data_key}' not found in '{src_path}'. Available keys: {keys}")
    return np.asarray(raw[data_key])


def _rewrite_data_slice_hdf5(
    *,
    src_path: Path,
    dst_path: Path,
    data_key: str,
    time_axis: int | None,
    start_idx: int,
    end_idx: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    shutil.copy2(src_path, dst_path)
    with h5py.File(dst_path, "r+") as dst:
        if data_key not in dst:
            raise KeyError(f"Dataset '{data_key}' not found in '{dst_path}'.")

        data_ds = dst[data_key]
        data = np.asarray(data_ds)
        if data.ndim != 2:
            raise ValueError(f"Dataset '{data_key}' must be 2D, got shape {data.shape}.")

        time_axis_idx = _resolve_time_axis(tuple(data.shape), time_axis)
        old_n = int(data.shape[time_axis_idx])
        start = int(start_idx)
        end = int(end_idx)
        if not (0 <= start < end <= old_n):
            raise ValueError(
                f"Invalid slice [{start}:{end}] for time axis length {old_n}."
            )

        data_slice = _slice_along_axis(data, time_axis_idx, start, end)

        create_kwargs = _compute_create_kwargs(data_ds, tuple(data_slice.shape))
        before_shape = tuple(data.shape)
        del dst[data_key]
        dst.create_dataset(data_key, data=data_slice, **create_kwargs)
        after_shape = tuple(data_slice.shape)
    return before_shape, after_shape


def _rewrite_data_slice_mat(
    *,
    src_path: Path,
    dst_path: Path,
    data_key: str,
    time_axis: int | None,
    start_idx: int,
    end_idx: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    raw = loadmat(src_path, squeeze_me=False, struct_as_record=False)
    if data_key not in raw:
        keys = sorted(k for k in raw.keys() if not k.startswith("__"))
        raise KeyError(f"Variable '{data_key}' not found in '{src_path}'. Available keys: {keys}")

    data = np.asarray(raw[data_key])
    if data.ndim != 2:
        raise ValueError(f"Variable '{data_key}' must be 2D, got shape {data.shape}.")

    time_axis_idx = _resolve_time_axis(tuple(data.shape), time_axis)
    old_n = int(data.shape[time_axis_idx])
    start = int(start_idx)
    end = int(end_idx)
    if not (0 <= start < end <= old_n):
        raise ValueError(f"Invalid slice [{start}:{end}] for time axis length {old_n}.")

    data_slice = _slice_along_axis(data, time_axis_idx, start, end)

    out: dict[str, object] = {}
    for key, value in raw.items():
        if key.startswith("__"):
            continue
        arr = np.asarray(value)
        if key == data_key:
            out[key] = data_slice
            continue
        if bool(SLICE_MATCHING_NUMERIC_ARRAYS_FOR_MAT) and arr.ndim >= 1 and np.issubdtype(arr.dtype, np.number):
            axes = [ax for ax, size in enumerate(arr.shape) if int(size) == old_n]
            if len(axes) == 1:
                out[key] = _slice_along_axis(arr, int(axes[0]), start, end)
                continue
        out[key] = value

    savemat(dst_path, out, do_compression=True)
    return tuple(data.shape), tuple(data_slice.shape)


def _rewrite_data_slice(
    *,
    src_path: Path,
    dst_path: Path,
    data_key: str,
    time_axis: int | None,
    start_idx: int,
    end_idx: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if _is_hdf5_mat(src_path):
        return _rewrite_data_slice_hdf5(
            src_path=src_path,
            dst_path=dst_path,
            data_key=data_key,
            time_axis=time_axis,
            start_idx=start_idx,
            end_idx=end_idx,
        )
    return _rewrite_data_slice_mat(
        src_path=src_path,
        dst_path=dst_path,
        data_key=data_key,
        time_axis=time_axis,
        start_idx=start_idx,
        end_idx=end_idx,
    )


def _nan_blocks_from_data(data: np.ndarray, *, time_axis: int) -> list[tuple[int, int, int]]:
    if time_axis == 0:
        nan_any = np.any(np.isnan(data), axis=1)
    else:
        nan_any = np.any(np.isnan(data), axis=0)
    idx = np.flatnonzero(nan_any)
    if idx.size == 0:
        return []

    blocks: list[tuple[int, int, int]] = []
    start = int(idx[0])
    prev = int(idx[0])
    for value in idx[1:]:
        i = int(value)
        if i == prev + 1:
            prev = i
            continue
        blocks.append((start, prev, prev - start + 1))
        start = i
        prev = i
    blocks.append((start, prev, prev - start + 1))
    return blocks


def _run_trim_mode(input_path: Path) -> None:
    if int(DROP_FIRST_TIME_SAMPLES) < 0:
        raise ValueError("DROP_FIRST_TIME_SAMPLES must be >= 0.")
    if int(DROP_LAST_TIME_SAMPLES) < 0:
        raise ValueError("DROP_LAST_TIME_SAMPLES must be >= 0.")

    data = _read_data_array(input_path, DATA_KEY)
    time_axis_idx = _resolve_time_axis(tuple(data.shape), TIME_AXIS)
    old_n = int(data.shape[time_axis_idx])

    start_idx = int(DROP_FIRST_TIME_SAMPLES)
    end_idx = old_n - int(DROP_LAST_TIME_SAMPLES)
    keep_n = end_idx - start_idx
    if keep_n < 1:
        raise ValueError(
            f"Cannot trim start={DROP_FIRST_TIME_SAMPLES} and end={DROP_LAST_TIME_SAMPLES} "
            f"time samples from axis length {old_n}."
        )

    if bool(OVERWRITE_INPLACE):
        with tempfile.TemporaryDirectory(prefix="trim-mat-") as tmpdir:
            tmp_path = Path(tmpdir) / input_path.name
            before, after = _rewrite_data_slice(
                src_path=input_path,
                dst_path=tmp_path,
                data_key=DATA_KEY,
                time_axis=TIME_AXIS,
                start_idx=start_idx,
                end_idx=end_idx,
            )
            tmp_path.replace(input_path)
            print(
                f"Updated in-place: {input_path}\n"
                f"  data shape: {before} -> {after}\n"
                f"  dropped leading time samples: {int(DROP_FIRST_TIME_SAMPLES)}\n"
                f"  dropped trailing time samples: {int(DROP_LAST_TIME_SAMPLES)}"
            )
        return

    output_path = OUTPUT_MAT_PATH
    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_trimmed.mat")
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    before, after = _rewrite_data_slice(
        src_path=input_path,
        dst_path=output_path,
        data_key=DATA_KEY,
        time_axis=TIME_AXIS,
        start_idx=start_idx,
        end_idx=end_idx,
    )
    print(
        f"Wrote: {output_path}\n"
        f"  data shape: {before} -> {after}\n"
        f"  dropped leading time samples: {int(DROP_FIRST_TIME_SAMPLES)}\n"
        f"  dropped trailing time samples: {int(DROP_LAST_TIME_SAMPLES)}"
    )


def _run_split_mode(input_path: Path) -> None:
    data = _read_data_array(input_path, DATA_KEY)
    time_axis_idx = _resolve_time_axis(tuple(data.shape), TIME_AXIS)
    old_n = int(data.shape[time_axis_idx])
    blocks = _nan_blocks_from_data(data, time_axis=time_axis_idx)

    if not blocks:
        raise ValueError("No NaN block found in the data time axis; nothing to split on.")

    if bool(USE_LARGEST_NAN_BLOCK):
        block = max(blocks, key=lambda item: item[2])
    else:
        idx = int(NAN_BLOCK_INDEX)
        if idx < 0 or idx >= len(blocks):
            raise IndexError(f"NAN_BLOCK_INDEX={idx} out of range for {len(blocks)} blocks.")
        block = blocks[idx]

    start, end, length = block
    left_start = 0
    left_end = start
    right_start = end + 1
    right_end = old_n

    left_len = left_end - left_start
    right_len = right_end - right_start
    min_len = int(MIN_SEGMENT_TIME_SAMPLES)
    if left_len < min_len or right_len < min_len:
        raise ValueError(
            f"Split would create too-short segment(s): left={left_len}, right={right_len}, "
            f"MIN_SEGMENT_TIME_SAMPLES={min_len}."
        )

    prefix = SPLIT_OUTPUT_PREFIX or input_path.stem
    out1 = input_path.with_name(f"{prefix}_part1.mat").resolve()
    out2 = input_path.with_name(f"{prefix}_part2.mat").resolve()
    out1.parent.mkdir(parents=True, exist_ok=True)

    before1, after1 = _rewrite_data_slice(
        src_path=input_path,
        dst_path=out1,
        data_key=DATA_KEY,
        time_axis=TIME_AXIS,
        start_idx=left_start,
        end_idx=left_end,
    )
    before2, after2 = _rewrite_data_slice(
        src_path=input_path,
        dst_path=out2,
        data_key=DATA_KEY,
        time_axis=TIME_AXIS,
        start_idx=right_start,
        end_idx=right_end,
    )

    print(
        f"Split source: {input_path}\n"
        f"  source data shape: {before1}\n"
        f"  selected NaN block: start={start}, end={end}, len={length}\n"
        f"  wrote part1: {out1} with shape {after1}\n"
        f"  wrote part2: {out2} with shape {after2}"
    )


def main() -> None:
    input_path = INPUT_MAT_PATH.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    action = str(ACTION).strip().lower()
    if action == "trim_trailing":
        _run_trim_mode(input_path)
        return
    if action == "split_nan_gap":
        _run_split_mode(input_path)
        return
    raise ValueError("ACTION must be one of: 'trim_trailing', 'split_nan_gap'.")


if __name__ == "__main__":
    main()
