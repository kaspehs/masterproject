"""Thesis-style time-series and spectrum plots for VIV model rollout comparisons.

This module is intentionally data-shape light: each rollout can be a dict-like
object or a pandas DataFrame.  The plotting functions only require named
columns/fields and can be adapted through ``field_map`` and ``styles``.

Typical usage is to import the functions from a notebook or analysis script and
pass the rollout arrays you already computed.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

_CACHE_ROOT = Path(tempfile.gettempdir()) / "masterproject_plot_cache"
(_CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
(_CACHE_ROOT / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory


ROOT = Path(__file__).resolve().parent.parent
CFD_DATA_DIR = ROOT / "vivana_cfd_data_pipeline"
for _path in (ROOT, CFD_DATA_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


DEFAULT_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "time": ("time", "t", "Time"),
    "cycles": ("cycles", "cycle", "oscillation_cycles", "n_cycles"),
    "natural_frequency_hz": ("natural_frequency_hz", "structural_frequency_hz", "f_n_hz", "fn_hz"),
    "y_over_D": ("y_over_D", "y/D", "yd", "yD", "displacement", "disp", "eta"),
    "C_F": ("C_F", "CF", "c_f", "force_coefficient", "force_coeff", "force_total", "force"),
    "delta_C_F": ("delta_C_F", "dC_F", "delta_CF", "delta_force", "force_correction"),
    "sigma_dy": ("sigma_dy", "sig_dy", "sig_dy_td", "sig_dy_loc", "sig_dy_loc_td"),
    "sigma_ddy": ("sigma_ddy", "sig_ddy", "sig_ddy_td", "sig_ddy_loc", "sig_ddy_loc_td"),
    "delta_freq": (
        "delta_freq",
        "delta_f_hat",
        "delta_fhat",
        "delta_fhat",
        "delta_omega",
        "delta_theta",
        "freq_correction",
        "frequency_correction",
        "internal_correction",
    ),
}

DEFAULT_STYLES: dict[str, dict[str, Any]] = {
    "cfd": {
        "label": "CFD reference",
        "color": "black",
        "linestyle": "-",
        "linewidth": 1.65,
        "zorder": 4,
    },
    "vivana": {
        "label": "VIVANA-TD baseline",
        "color": "0.45",
        "linestyle": "--",
        "linewidth": 1.25,
        "zorder": 3,
    },
    "force": {
        "label": "Force correction",
        "color": "#0072B2",
        "linestyle": "-",
        "linewidth": 1.35,
        "zorder": 3,
    },
    "frequency": {
        "label": "Frequency correction",
        "color": "#D55E00",
        "linestyle": "-",
        "linewidth": 1.35,
        "zorder": 3,
    },
    "combined": {
        "label": "Combined correction",
        "color": "#009E73",
        "linestyle": "-",
        "linewidth": 1.35,
        "zorder": 3,
    },
    "standalone": {
        "label": "Standalone model",
        "color": "#882255",
        "linestyle": "-",
        "linewidth": 1.35,
        "zorder": 3,
    },
}

RESPONSE_COMPARATORS: tuple[str, ...] = ("vivana", "force", "frequency", "combined", "standalone")
FORCE_CORRECTION_MODELS: tuple[str, ...] = ("force", "combined")
FREQUENCY_CORRECTION_MODELS: tuple[str, ...] = ("frequency", "combined")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit these values before running the script.
# ═══════════════════════════════════════════════════════════════════════════════

# "rollout_timeseries" -> standard response + correction rollout plots, plus spectra.
# "tau_sigma_sensitivity" -> offset plot of sigma_dy and sigma_ddy for tau/T_ref values.
PLOT_MODE: str = "rollout_timeseries"

DEFAULT_REAL_DATASET_ROOT = CFD_DATA_DIR / "generated" / "td_burnin_trimmed_alltimeseries"
DEFAULT_OUTPUT_DIR = ROOT / "figs" / "rollout_timeseries"
DEFAULT_DUMMY_OUTPUT_DIR = ROOT / "figs" / "rollout_timeseries_examples"

# Standard rollout-timeseries defaults.
DEFAULT_REAL_UR = 5.75
# Tau sigma sensitivity defaults.
DEFAULT_TAU_SIGMA_UR = 4.0
DEFAULT_TAU_OVER_TREF_VALUES: tuple[float, ...] = (0.01, 0.05, 0.1, 0.5, 1.0, 4.0)

# Shared plot/run defaults.
DEFAULT_CASE_INDEX = 0
DEFAULT_CYCLE_START = 0.0
DEFAULT_CYCLE_END = 10.0
TIMESERIES_CYCLES_AFTER_ENCODING = 10.0
BLOCK_10_VALIDATION_REDUCTION_FACTOR = 20
DEFAULT_SPECTRUM_MAX_F_OVER_FN = 4.0
DEFAULT_RESPONSE_SPECTRUM_F_OVER_FN_LIMITS: tuple[float, float] | None = (0.5, 3.0)
DEFAULT_CORRECTION_SPECTRUM_F_OVER_FN_LIMITS: tuple[float, float] | None = (0.5, 10.0)
DEFAULT_SPECTRUM_Y_MODE = "amplitude"
DEFAULT_SPECTRUM_ZERO_PAD_FACTOR = 4
BLOCK_9_10_SPINE_COLOR = "0.65"
BLOCK_9_10_SPINE_LINE_WIDTH = 0.6

# ═══════════════════════════════════════════════════════════════════════════════

BLOCK_9_10_MODEL_SPECS: tuple[dict[str, str | Path], ...] = (
    {
        "path": ROOT / "models" / "mean" / "multi_seed" / "best_seed.pt",
        "label": "Force correction",
    },
    {
        "path": ROOT / "models" / "fhat" / "multi_seed" / "best_seed.pt",
        "label": "Frequency correction",
    },
    {
        "path": ROOT / "models" / "combined" / "multi_seed" / "best_seed.pt",
        "label": "Combined correction",
    },
    {
        "path": ROOT / "models" / "latentrnn" / "best_seed.pt",
        "label": "Standalone model",
    },
)
BLOCK_9_10_TD_MEMORY_TAU_SECONDS = "tau_over_tref:4"
TD_MASS_SOURCE = "dry"
VIVANA_SUMMARY_MASS_SOURCE = "dry"
VIVANA_TD_BASELINE_PARAM_OVERRIDES = {
    "td_cv": 1.2,
    "td_cd": 1.1,
    "td_ca": 1.0,
    "td_fhat0": 0.18,
    "td_fhat_min": 0.11,
    "td_fhat_max": 0.26,
}
MODEL_LABEL_TO_ROLLOUT_KEY = {
    "Force correction": "force",
    "Frequency correction": "frequency",
    "Combined correction": "combined",
    "Standalone model": "standalone",
}


class _ProgressBar:
    def __init__(self, total: int, label: str, *, enabled: bool = False, width: int = 28) -> None:
        self.total = max(1, int(total))
        self.label = label
        self.enabled = bool(enabled)
        self.width = int(width)
        self.count = 0
        self.start_time = time.perf_counter()
        self._last_line_len = 0

    def update(self, message: str = "") -> None:
        if not self.enabled:
            return
        self.count = min(self.total, self.count + 1)
        elapsed = time.perf_counter() - self.start_time
        rate = self.count / elapsed if elapsed > 0.0 else 0.0
        remaining = (self.total - self.count) / rate if rate > 0.0 else float("inf")
        filled = int(round(self.width * self.count / self.total))
        bar = "#" * filled + "-" * (self.width - filled)
        eta = _format_duration(remaining)
        text = (
            f"\r{self.label} [{bar}] {self.count:>2}/{self.total:<2} "
            f"elapsed={_format_duration(elapsed)} eta={eta}"
        )
        if message:
            text += f"  {message}"
        padding = " " * max(0, self._last_line_len - len(text))
        print(text + padding, end="", flush=True)
        self._last_line_len = len(text)
        if self.count >= self.total:
            print()
            self._last_line_len = 0


def _format_duration(seconds: float) -> str:
    if not np.isfinite(seconds):
        return "--:--"
    seconds_i = max(0, int(round(seconds)))
    minutes, secs = divmod(seconds_i, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _merged_field_aliases(
    field_map: Mapping[str, str | Sequence[str]] | None,
) -> dict[str, tuple[str, ...]]:
    aliases = dict(DEFAULT_FIELD_ALIASES)
    if not field_map:
        return aliases

    for canonical, names in field_map.items():
        if isinstance(names, str):
            aliases[canonical] = (names,)
        else:
            aliases[canonical] = tuple(str(name) for name in names)
    return aliases


def _merged_styles(styles: Mapping[str, Mapping[str, Any]] | None) -> dict[str, dict[str, Any]]:
    merged = {key: dict(value) for key, value in DEFAULT_STYLES.items()}
    if not styles:
        return merged
    for key, value in styles.items():
        merged.setdefault(key, {})
        merged[key].update(dict(value))
    return merged


def _has_field(record: Any, name: str) -> bool:
    if isinstance(record, Mapping):
        return name in record
    columns = getattr(record, "columns", None)
    if columns is not None:
        return name in columns
    try:
        record[name]
    except Exception:
        return False
    return True


def _get_raw_field(record: Any, name: str) -> Any:
    if isinstance(record, Mapping):
        return record[name]
    return record[name]


def _resolve_field_name(
    record: Any,
    canonical: str,
    aliases: Mapping[str, tuple[str, ...]],
    *,
    required: bool = True,
) -> str | None:
    candidates = aliases.get(canonical, (canonical,))
    for candidate in candidates:
        if _has_field(record, candidate):
            return candidate
    if required:
        raise KeyError(
            f"Could not find field for {canonical!r}. Tried: {', '.join(candidates)}"
        )
    return None


def _as_1d_array(record: Any, field_name: str) -> np.ndarray:
    value = _get_raw_field(record, field_name)
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim > 1:
        arr = np.squeeze(arr)
    if arr.ndim != 1:
        raise ValueError(f"Field {field_name!r} must resolve to a 1D array.")
    return arr


def _record_label(record: Any, fallback: str) -> str:
    if isinstance(record, Mapping):
        label = record.get("label")
        if label is not None:
            return str(label)
    return fallback


def _finite_window_mask(x: np.ndarray, start: float | None, end: float | None) -> np.ndarray:
    mask = np.isfinite(x)
    if start is not None:
        mask &= x >= float(start)
    if end is not None:
        mask &= x <= float(end)
    if not np.any(mask):
        raise ValueError(
            "The requested time/cycle window contains no samples. "
            f"Window=({start}, {end}), data range=({np.nanmin(x):g}, {np.nanmax(x):g})."
        )
    return mask


def _window_axis_settings(
    rollouts: Mapping[str, Any],
    aliases: Mapping[str, tuple[str, ...]],
    *,
    x_axis: str,
    t_start: float | None,
    t_end: float | None,
    cycle_start: float | None,
    cycle_end: float | None,
) -> tuple[str, float | None, float | None, str]:
    if (t_start is not None or t_end is not None) and (
        cycle_start is not None or cycle_end is not None
    ):
        raise ValueError("Specify either a time window or a cycle window, not both.")

    if t_start is not None or t_end is not None:
        return "time", t_start, t_end, "Time"
    if cycle_start is not None or cycle_end is not None:
        return "cycles", cycle_start, cycle_end, "Oscillation cycles"

    preferred = str(x_axis)
    if "cfd" in rollouts and _resolve_field_name(
        rollouts["cfd"], preferred, aliases, required=False
    ):
        label = "Oscillation cycles" if preferred == "cycles" else "Time"
        return preferred, None, None, label

    if preferred != "time" and "cfd" in rollouts and _resolve_field_name(
        rollouts["cfd"], "time", aliases, required=False
    ):
        return "time", None, None, "Time"

    label = "Oscillation cycles" if preferred == "cycles" else "Time"
    return preferred, None, None, label


def extract_rollout_window(
    rollout: Any,
    *,
    x_axis: str = "cycles",
    start: float | None = None,
    end: float | None = None,
    field_map: Mapping[str, str | Sequence[str]] | None = None,
) -> dict[str, np.ndarray | str]:
    """Return a shallow, array-only copy of a rollout in a selected x-window.

    Parameters
    ----------
    rollout:
        Dict-like object or pandas DataFrame containing time-series fields.
    x_axis:
        Canonical x-axis field to window by, usually ``"cycles"`` or ``"time"``.
    start, end:
        Optional inclusive limits in x-axis units.
    field_map:
        Optional canonical-name to actual-field mapping.  For example:
        ``{"C_F": "total_force_cf", "delta_freq": "delta_f_hat"}``.

    Returns
    -------
    dict
        Contains all 1D fields with the same length as the x-axis, windowed by
        the same mask.  Non-array metadata is omitted except ``label``.
    """
    aliases = _merged_field_aliases(field_map)
    x_name = _resolve_field_name(rollout, x_axis, aliases, required=True)
    x = _as_1d_array(rollout, x_name)
    mask = _finite_window_mask(x, start, end)

    out: dict[str, np.ndarray | str] = {"x_axis": x_axis, x_axis: x[mask]}
    if isinstance(rollout, Mapping) and "label" in rollout:
        out["label"] = str(rollout["label"])

    names: list[str]
    if isinstance(rollout, Mapping):
        names = [str(name) for name in rollout.keys()]
    else:
        columns = getattr(rollout, "columns", ())
        names = [str(name) for name in columns]

    for name in names:
        if name == "label" or not _has_field(rollout, name):
            continue
        try:
            arr = _as_1d_array(rollout, name)
        except (TypeError, ValueError):
            continue
        if arr.shape == x.shape:
            out[name] = arr[mask]
    return out


def _windowed_xy(
    record: Any,
    y_canonical: str,
    *,
    x_canonical: str,
    start: float | None,
    end: float | None,
    aliases: Mapping[str, tuple[str, ...]],
) -> tuple[np.ndarray, np.ndarray]:
    x_name = _resolve_field_name(record, x_canonical, aliases, required=True)
    y_name = _resolve_field_name(record, y_canonical, aliases, required=True)
    x = _as_1d_array(record, x_name)
    y = _as_1d_array(record, y_name)
    if x.shape != y.shape:
        raise ValueError(
            f"Fields {x_name!r} and {y_name!r} must have the same shape; "
            f"got {x.shape} and {y.shape}."
        )
    mask = _finite_window_mask(x, start, end)
    return x[mask], y[mask]


def _robust_amplitude(values: Sequence[np.ndarray]) -> float:
    finite_parts = [np.asarray(v, dtype=float)[np.isfinite(v)] for v in values]
    finite_parts = [v for v in finite_parts if v.size]
    if not finite_parts:
        return 1.0
    merged = np.concatenate(finite_parts)
    amp = 0.5 * (np.nanpercentile(merged, 97.5) - np.nanpercentile(merged, 2.5))
    if not np.isfinite(amp) or amp <= 0.0:
        amp = float(np.nanmax(np.abs(merged))) if merged.size else 1.0
    return max(float(amp), 1e-12)


def _windowed_time_y(
    record: Any,
    y_canonical: str,
    *,
    window_x_canonical: str,
    start: float | None,
    end: float | None,
    aliases: Mapping[str, tuple[str, ...]],
) -> tuple[np.ndarray, np.ndarray, str]:
    x_name = _resolve_field_name(record, window_x_canonical, aliases, required=True)
    y_name = _resolve_field_name(record, y_canonical, aliases, required=True)
    time_name = _resolve_field_name(record, "time", aliases, required=False)
    spectrum_axis_name = time_name if time_name is not None else x_name

    x = _as_1d_array(record, x_name)
    y = _as_1d_array(record, y_name)
    spectrum_axis = _as_1d_array(record, spectrum_axis_name)
    if x.shape != y.shape or x.shape != spectrum_axis.shape:
        raise ValueError(
            f"Fields {x_name!r}, {y_name!r}, and {spectrum_axis_name!r} must have the same shape; "
            f"got {x.shape}, {y.shape}, and {spectrum_axis.shape}."
        )
    mask = _finite_window_mask(x, start, end)
    axis_label = r"$\omega$ [Hz]" if time_name is not None else rf"$\omega$ [1/{window_x_canonical}]"
    return spectrum_axis[mask], y[mask], axis_label


def _spectrum_axis_y(
    record: Any,
    y_canonical: str,
    *,
    window_x_canonical: str,
    start: float | None,
    end: float | None,
    aliases: Mapping[str, tuple[str, ...]],
    use_full_series: bool,
) -> tuple[np.ndarray, np.ndarray, str]:
    if not use_full_series:
        return _windowed_time_y(
            record,
            y_canonical,
            window_x_canonical=window_x_canonical,
            start=start,
            end=end,
            aliases=aliases,
        )

    y_name = _resolve_field_name(record, y_canonical, aliases, required=True)
    time_name = _resolve_field_name(record, "time", aliases, required=False)
    if time_name is not None:
        axis_name = time_name
        axis_label = r"$\omega$ [Hz]"
    else:
        axis_name = _resolve_field_name(record, window_x_canonical, aliases, required=True)
        axis_label = rf"$\omega$ [1/{window_x_canonical}]"

    axis = _as_1d_array(record, axis_name)
    y = _as_1d_array(record, y_name)
    if axis.shape != y.shape:
        raise ValueError(
            f"Fields {axis_name!r} and {y_name!r} must have the same shape; "
            f"got {axis.shape} and {y.shape}."
        )
    mask = np.isfinite(axis) & np.isfinite(y)
    if not np.any(mask):
        raise ValueError(f"Fields {axis_name!r} and {y_name!r} contain no finite spectrum samples.")
    return axis[mask], y[mask], axis_label


def _record_scalar(
    record: Any,
    canonical: str,
    aliases: Mapping[str, tuple[str, ...]],
) -> float | None:
    name = _resolve_field_name(record, canonical, aliases, required=False)
    if name is None:
        return None
    try:
        values = np.asarray(_get_raw_field(record, name), dtype=float).reshape(-1)
    except Exception:
        return None
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    value = float(finite[0])
    return value if value > 0.0 else None


def _spectrum_frequency_scale_hz(
    rollouts: Mapping[str, Any],
    aliases: Mapping[str, tuple[str, ...]],
    *,
    requested: float | None,
) -> float | None:
    if requested is not None and np.isfinite(float(requested)) and float(requested) > 0.0:
        return float(requested)
    ordered_keys = ["cfd", *[key for key in rollouts.keys() if key != "cfd"]]
    for key in ordered_keys:
        if key not in rollouts:
            continue
        value = _record_scalar(rollouts[key], "natural_frequency_hz", aliases)
        if value is not None:
            return value
    return None


def _scaled_spectrum_x(
    spectrum: tuple[np.ndarray, np.ndarray] | None,
    *,
    frequency_scale_hz: float | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    if spectrum is None:
        return None
    freqs, values = spectrum
    if frequency_scale_hz is None:
        return freqs, values
    scale = float(frequency_scale_hz)
    if not np.isfinite(scale) or scale <= 0.0:
        return freqs, values
    return np.asarray(freqs, dtype=float) / scale, np.asarray(values, dtype=float) * scale


def _spectrum_y_values(
    values: np.ndarray,
    *,
    mode: str,
) -> np.ndarray:
    values_arr = np.clip(np.asarray(values, dtype=float), a_min=0.0, a_max=None)
    mode_key = str(mode).strip().lower()
    if mode_key in {"amplitude", "sqrt", "asd"}:
        return np.sqrt(values_arr)
    if mode_key in {"power", "psd"}:
        return values_arr
    raise ValueError("spectrum_y_mode must be 'amplitude' or 'power'.")


def _valid_xlim(xlim: tuple[float, float] | Sequence[float] | None) -> tuple[float, float] | None:
    if xlim is None:
        return None
    if len(xlim) != 2:
        raise ValueError("Spectrum x-limits must be a two-value sequence.")
    left = float(xlim[0])
    right = float(xlim[1])
    if not (np.isfinite(left) and np.isfinite(right)) or right <= left:
        raise ValueError(f"Invalid spectrum x-limits: {xlim!r}.")
    return left, right


def _single_sided_spectrum(
    axis_values: np.ndarray,
    signal_values: np.ndarray,
    *,
    area_normalize: bool = False,
    use_hann_window: bool = True,
    zero_pad_factor: int = 1,
) -> tuple[np.ndarray, np.ndarray] | None:
    axis_arr = np.asarray(axis_values, dtype=float).reshape(-1)
    signal_arr = np.asarray(signal_values, dtype=float).reshape(-1)
    n = min(axis_arr.size, signal_arr.size)
    if n < 4:
        return None
    axis_arr = axis_arr[:n]
    signal_arr = signal_arr[:n]
    valid = np.isfinite(axis_arr) & np.isfinite(signal_arr)
    if np.count_nonzero(valid) < 4:
        return None
    axis_arr = axis_arr[valid]
    signal_arr = signal_arr[valid]
    order = np.argsort(axis_arr)
    axis_arr = axis_arr[order]
    signal_arr = signal_arr[order]

    diffs = np.diff(axis_arr)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if diffs.size == 0:
        return None
    dt = float(np.nanmedian(diffs))
    if not np.isfinite(dt) or dt <= 0.0:
        return None

    centered = signal_arr - float(np.nanmean(signal_arr))
    if np.allclose(centered, 0.0, equal_nan=False):
        return None
    window = np.hanning(centered.size) if use_hann_window else np.ones(centered.size)
    window_power = float(np.sum(window * window))
    if not np.isfinite(window_power) or window_power <= 0.0:
        return None
    pad_factor = max(1, int(zero_pad_factor))
    n_fft = int(centered.size) * pad_factor
    spectrum = np.abs(np.fft.rfft(centered * window, n=n_fft)) ** 2
    freqs = np.fft.rfftfreq(n_fft, d=dt)
    spectrum = spectrum / ((1.0 / dt) * window_power)
    if spectrum.size > 2:
        spectrum[1:-1] *= 2.0
    mask = np.isfinite(freqs) & np.isfinite(spectrum) & (freqs > 0.0)
    if np.count_nonzero(mask) < 2:
        return None
    freqs = np.asarray(freqs[mask], dtype=float)
    spectrum = np.clip(np.asarray(spectrum[mask], dtype=float), a_min=0.0, a_max=None)
    if not np.any(spectrum > 0.0):
        return None
    if area_normalize:
        integrate = getattr(np, "trapezoid", np.trapz)
        area = float(integrate(spectrum, freqs))
        if not np.isfinite(area) or area <= 0.0:
            return None
        spectrum = spectrum / area
    return freqs, spectrum


def _area_normalized_spectrum(
    spectrum: tuple[np.ndarray, np.ndarray] | None,
    *,
    xlim: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    if spectrum is None:
        return None
    freqs, values = spectrum
    freqs_arr = np.asarray(freqs, dtype=float).reshape(-1)
    values_arr = np.asarray(values, dtype=float).reshape(-1)
    n = min(freqs_arr.size, values_arr.size)
    if n < 2:
        return None
    freqs_arr = freqs_arr[:n]
    values_arr = np.clip(values_arr[:n], a_min=0.0, a_max=None)
    valid = np.isfinite(freqs_arr) & np.isfinite(values_arr) & (freqs_arr > 0.0)
    if xlim is not None:
        valid &= freqs_arr >= float(xlim[0])
        valid &= freqs_arr <= float(xlim[1])
    if np.count_nonzero(valid) < 2:
        return None
    integrate = getattr(np, "trapezoid", np.trapz)
    area = float(integrate(values_arr[valid], freqs_arr[valid]))
    if not np.isfinite(area) or area <= 0.0:
        return None
    return freqs_arr, values_arr / area


def _spectrum_height(spectra: Sequence[np.ndarray]) -> float:
    finite_parts = [np.asarray(v, dtype=float)[np.isfinite(v)] for v in spectra]
    finite_parts = [v for v in finite_parts if v.size]
    if not finite_parts:
        return 1.0
    merged = np.concatenate(finite_parts)
    height = float(np.nanmax(merged))
    if not np.isfinite(height) or height <= 0.0:
        height = float(np.nanpercentile(np.abs(merged), 99.0))
    return max(height, 1.0e-12)


def _spectrum_focus_xlim(
    spectra: Sequence[tuple[np.ndarray, np.ndarray] | None],
    *,
    dominant_multiplier: float = 3.0,
    min_visible_bins: int = 8,
    max_upper: float | None = None,
) -> tuple[float, float] | None:
    dominant_freqs: list[float] = []
    freq_steps: list[float] = []
    max_freq = 0.0

    for spectrum in spectra:
        if spectrum is None:
            continue
        freqs, values = spectrum
        freq_arr = np.asarray(freqs, dtype=float).reshape(-1)
        value_arr = np.asarray(values, dtype=float).reshape(-1)
        mask = np.isfinite(freq_arr) & np.isfinite(value_arr) & (freq_arr > 0.0) & (value_arr >= 0.0)
        if np.count_nonzero(mask) < 2:
            continue
        freq_valid = freq_arr[mask]
        value_valid = value_arr[mask]
        max_freq = max(max_freq, float(freq_valid[-1]))
        diffs = np.diff(freq_valid)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
        if diffs.size:
            freq_steps.append(float(np.nanmedian(diffs)))
        positive = value_valid > 0.0
        if np.any(positive):
            freq_positive = freq_valid[positive]
            value_positive = value_valid[positive]
            dominant_freqs.append(float(freq_positive[int(np.argmax(value_positive))]))

    if max_freq <= 0.0:
        return None
    if dominant_freqs:
        upper = float(max(dominant_freqs)) * float(dominant_multiplier)
    else:
        upper = max_freq
    if freq_steps:
        upper = max(upper, float(min_visible_bins) * max(freq_steps))
    if max_upper is not None and np.isfinite(float(max_upper)) and float(max_upper) > 0.0:
        upper = min(upper, float(max_upper))
    upper = min(max_freq, upper)
    if not np.isfinite(upper) or upper <= 0.0:
        upper = max_freq
    return 0.0, float(upper)


def _save_figure(
    fig: plt.Figure,
    output_dir: str | Path | None,
    basename: str | None,
    *,
    dpi: int,
    formats: Sequence[str] = ("pdf", "png"),
) -> list[Path]:
    if output_dir is None or basename is None:
        return []
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for fmt in formats:
        path = out_dir / f"{basename}.{fmt.lower().lstrip('.')}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        saved.append(path)
    return saved


def _format_ur(ur: float | str | None) -> str:
    if ur is None:
        return ""
    if isinstance(ur, str):
        return ur
    return f"{float(ur):g}"


def _filename_ur(ur: float | str | None) -> str:
    text = _format_ur(ur)
    if not text:
        return "selected"
    return text.replace(".", "p").replace("-", "m").replace(" ", "_")


def _set_thesis_rcparams(base_font_size: float) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Latin Modern Roman", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": base_font_size,
            "axes.labelsize": base_font_size + 1.0,
            "axes.titlesize": base_font_size + 1.0,
            "xtick.labelsize": base_font_size,
            "ytick.labelsize": base_font_size,
            "legend.fontsize": base_font_size,
            "axes.linewidth": 0.6,
            "lines.solid_capstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def _ensure_endpoint_xticks(
    ax: plt.Axes,
    *,
    start: float | None,
    end: float | None,
    x_label: str,
) -> None:
    if x_label != "Oscillation cycles":
        return
    endpoints = [value for value in (start, end) if value is not None and np.isfinite(float(value))]
    if not endpoints:
        return
    ticks = list(np.asarray(ax.get_xticks(), dtype=float).reshape(-1))
    for endpoint in endpoints:
        endpoint = float(endpoint)
        if not any(np.isclose(tick, endpoint, rtol=0.0, atol=1.0e-9) for tick in ticks):
            ticks.append(endpoint)
    ticks = sorted(tick for tick in ticks if np.isfinite(tick))
    ax.set_xticks(ticks)


def _apply_subplot_border(ax: plt.Axes, *, color: str = "0.15", linewidth: float = 0.8) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(color)
        spine.set_linewidth(linewidth)


def _apply_block9_10_spectral_frame(ax: plt.Axes) -> None:
    ax.grid(True, which="major", color="0.88", linewidth=0.5, alpha=0.75)
    ax.grid(True, which="minor", color="0.94", linewidth=0.35, alpha=0.45)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(BLOCK_9_10_SPINE_COLOR)
        spine.set_linewidth(BLOCK_9_10_SPINE_LINE_WIDTH)


def _model_line_handle(
    *,
    model_key: str,
    label: str,
    style: Mapping[str, Mapping[str, Any]],
    linewidth_fallback: float = 1.35,
) -> Line2D:
    model_style = style.get(model_key, {})
    return Line2D(
        [0],
        [0],
        color=model_style.get("color", "tab:blue"),
        linestyle=model_style.get("linestyle", "-"),
        linewidth=model_style.get("linewidth", linewidth_fallback),
        label=label,
    )


def _add_shared_model_legend(
    fig: plt.Figure,
    rollouts: Mapping[str, Any],
    *,
    model_order: Sequence[str],
    style: Mapping[str, Mapping[str, Any]],
    include_cfd: bool,
    ncol: int = 3,
) -> None:
    handles: list[Line2D] = []
    if include_cfd and "cfd" in rollouts:
        cfd_style = style.get("cfd", {})
        handles.append(
            _model_line_handle(
                model_key="cfd",
                label=str(cfd_style.get("label", "CFD reference")),
                style=style,
                linewidth_fallback=1.65,
            )
        )
    for model_key in model_order:
        if model_key not in rollouts:
            continue
        model_style = style.get(model_key, {})
        label = _record_label(rollouts[model_key], str(model_style.get("label", model_key)))
        handles.append(_model_line_handle(model_key=model_key, label=label, style=style))
    if not handles:
        return
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(max(1, int(ncol)), len(handles)),
        frameon=False,
        fontsize=plt.rcParams.get("legend.fontsize", 8),
        bbox_to_anchor=(0.5, 1.01),
        borderaxespad=0.0,
    )


def plot_rollout_response_timeseries(
    rollouts: Mapping[str, Any],
    *,
    ur: float | str | None = None,
    output_dir: str | Path | None = None,
    basename: str | None = None,
    t_start: float | None = None,
    t_end: float | None = None,
    cycle_start: float | None = 0.0,
    cycle_end: float | None = 10.0,
    x_axis: str = "cycles",
    field_map: Mapping[str, str | Sequence[str]] | None = None,
    styles: Mapping[str, Mapping[str, Any]] | None = None,
    comparators: Sequence[str] = RESPONSE_COMPARATORS,
    center_on_cfd: bool = True,
    x_origin: float = 0.0,
    figsize: tuple[float, float] = (5.85, 5.4),
    dpi: int = 300,
    base_font_size: float = 8.0,
    save: bool = True,
    save_formats: Sequence[str] = ("pdf", "png"),
    show: bool = False,
    progress: bool = False,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot displacement and force response rollouts in stacked comparison lanes.

    Each lane overlays the CFD reference with one comparator.  Signals are not
    amplitude-normalized.  By default, each lane is centered by the CFD mean in
    the plotted window before a vertical offset is applied; this keeps phase and
    amplitude differences visible without crowding the axes.
    """
    if "cfd" not in rollouts:
        raise KeyError("rollouts must contain a 'cfd' entry.")

    aliases = _merged_field_aliases(field_map)
    style = _merged_styles(styles)
    _set_thesis_rcparams(base_font_size)

    x_canonical, start, end, x_label = _window_axis_settings(
        rollouts,
        aliases,
        x_axis=x_axis,
        t_start=t_start,
        t_end=t_end,
        cycle_start=cycle_start,
        cycle_end=cycle_end,
    )
    x_display_origin = float(x_origin) if x_canonical == "cycles" else 0.0

    active_lanes = [key for key in comparators if key in rollouts]
    if not active_lanes:
        raise ValueError("No comparator rollouts were found for the response plot.")
    progress_bar = _ProgressBar(
        len(signals := (("y_over_D", r"$y/D$"), ("C_F", r"$C_F$"))) * len(active_lanes)
        + (1 if save and output_dir is not None else 0),
        "Response plot",
        enabled=progress,
    )

    fig, axes = plt.subplots(
        len(signals),
        1,
        figsize=figsize,
        sharex=True,
        constrained_layout=True,
    )
    axes = np.asarray(axes)

    xmins: list[float] = []
    xmaxs: list[float] = []

    for ax, (signal_key, y_label) in zip(axes, signals):
        lane_data: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]] = []
        centered_for_scale: list[np.ndarray] = []

        for lane_key in active_lanes:
            x_cfd, y_cfd = _windowed_xy(
                rollouts["cfd"],
                signal_key,
                x_canonical=x_canonical,
                start=start,
                end=end,
                aliases=aliases,
            )
            x_cmp, y_cmp = _windowed_xy(
                rollouts[lane_key],
                signal_key,
                x_canonical=x_canonical,
                start=start,
                end=end,
                aliases=aliases,
            )
            x_cfd = x_cfd - x_display_origin
            x_cmp = x_cmp - x_display_origin
            lane_center = float(np.nanmean(y_cfd)) if center_on_cfd else 0.0
            lane_data.append((lane_key, x_cfd, y_cfd, x_cmp, y_cmp, lane_center))
            centered_for_scale.extend((y_cfd - lane_center, y_cmp - lane_center))
            xmins.extend((float(np.nanmin(x_cfd)), float(np.nanmin(x_cmp))))
            xmaxs.extend((float(np.nanmax(x_cfd)), float(np.nanmax(x_cmp))))

        amp = _robust_amplitude(centered_for_scale)
        lane_gap = 3.0 * amp
        offsets = lane_gap * np.arange(len(active_lanes) - 1, -1, -1, dtype=float)

        for offset, (lane_key, x_cfd, y_cfd, x_cmp, y_cmp, lane_center) in zip(
            offsets, lane_data
        ):
            cfd_style = style["cfd"]
            cmp_style = style.get(lane_key, {})
            comparator_label = _record_label(
                rollouts[lane_key], str(cmp_style.get("label", lane_key))
            )

            ax.axhline(offset, color="0.88", linewidth=0.6, zorder=0)
            ax.plot(
                x_cfd,
                y_cfd - lane_center + offset,
                color=cfd_style.get("color", "black"),
                linestyle=cfd_style.get("linestyle", "-"),
                linewidth=cfd_style.get("linewidth", 1.6),
                zorder=cfd_style.get("zorder", 4),
            )
            ax.plot(
                x_cmp,
                y_cmp - lane_center + offset,
                color=cmp_style.get("color", "tab:blue"),
                linestyle=cmp_style.get("linestyle", "-"),
                linewidth=cmp_style.get("linewidth", 1.35),
                zorder=cmp_style.get("zorder", 3),
            )
            text_transform = blended_transform_factory(ax.transAxes, ax.transData)
            ax.text(
                1.01,
                offset,
                comparator_label,
                transform=text_transform,
                ha="left",
                va="center",
                fontsize=base_font_size,
                color=cmp_style.get("color", "0.2"),
                clip_on=False,
            )
            progress_bar.update(f"{y_label}: {comparator_label}")

        ax.set_ylabel(y_label, rotation=0, labelpad=22, va="center")
        ax.set_yticks(offsets)
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", color="0.9", linewidth=0.5)
        _apply_subplot_border(ax)
        ax.set_ylim(-1.45 * amp, offsets[0] + 1.45 * amp)

    if xmins and xmaxs:
        axes[-1].set_xlim(min(xmins), max(xmaxs))

    axes[-1].set_xlabel(x_label)
    _ensure_endpoint_xticks(
        axes[-1],
        start=(None if start is None else float(start) - x_display_origin),
        end=(None if end is None else float(end) - x_display_origin),
        x_label=x_label,
    )
    _add_shared_model_legend(
        fig,
        rollouts,
        model_order=active_lanes,
        style=style,
        include_cfd=True,
        ncol=3,
    )

    saved_paths: list[Path] = []
    if save:
        out_name = basename or f"rollout_response_timeseries_Ur_{_filename_ur(ur)}"
        saved_paths = _save_figure(fig, output_dir, out_name, dpi=dpi, formats=save_formats)
        progress_bar.update("saved files")
    fig._saved_paths = saved_paths  # type: ignore[attr-defined]

    if show:
        plt.show()
    return fig, axes


def plot_rollout_response_spectra(
    rollouts: Mapping[str, Any],
    *,
    ur: float | str | None = None,
    output_dir: str | Path | None = None,
    basename: str | None = None,
    t_start: float | None = None,
    t_end: float | None = None,
    cycle_start: float | None = 0.0,
    cycle_end: float | None = 10.0,
    x_axis: str = "cycles",
    field_map: Mapping[str, str | Sequence[str]] | None = None,
    styles: Mapping[str, Mapping[str, Any]] | None = None,
    comparators: Sequence[str] = RESPONSE_COMPARATORS,
    y_spectrum_scale_exclude: Sequence[str] = ("vivana",),
    area_normalize: bool = False,
    spectrum_full_series: bool = True,
    natural_frequency_hz: float | None = None,
    max_frequency_ratio: float | None = DEFAULT_SPECTRUM_MAX_F_OVER_FN,
    frequency_ratio_xlim: tuple[float, float] | None = DEFAULT_RESPONSE_SPECTRUM_F_OVER_FN_LIMITS,
    spectrum_y_mode: str = DEFAULT_SPECTRUM_Y_MODE,
    spectrum_zero_pad_factor: int = DEFAULT_SPECTRUM_ZERO_PAD_FACTOR,
    figsize: tuple[float, float] = (5.85, 5.4),
    dpi: int = 300,
    base_font_size: float = 8.0,
    save: bool = True,
    save_formats: Sequence[str] = ("pdf", "png"),
    show: bool = False,
    progress: bool = False,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot displacement and force spectra in response lanes."""
    if "cfd" not in rollouts:
        raise KeyError("rollouts must contain a 'cfd' entry.")

    aliases = _merged_field_aliases(field_map)
    style = _merged_styles(styles)
    _set_thesis_rcparams(base_font_size)
    frequency_scale_hz = _spectrum_frequency_scale_hz(
        rollouts,
        aliases,
        requested=natural_frequency_hz,
    )
    manual_xlim = _valid_xlim(frequency_ratio_xlim if frequency_scale_hz is not None else None)

    x_canonical, start, end, _ = _window_axis_settings(
        rollouts,
        aliases,
        x_axis=x_axis,
        t_start=t_start,
        t_end=t_end,
        cycle_start=cycle_start,
        cycle_end=cycle_end,
    )

    active_lanes = [key for key in comparators if key in rollouts]
    if not active_lanes:
        raise ValueError("No comparator rollouts were found for the response spectrum plot.")
    signals = (("y_over_D", r"$y/D$"), ("C_F", r"$C_F$"))
    progress_bar = _ProgressBar(
        len(signals) * len(active_lanes) + (1 if save and output_dir is not None else 0),
        "Response spectra",
        enabled=progress,
    )

    fig, axes = plt.subplots(
        len(signals),
        1,
        figsize=figsize,
        sharex=False,
        constrained_layout=True,
    )
    axes = np.asarray(axes)

    xlabel = r"$\omega/\omega_n$" if frequency_scale_hz is not None else r"$\omega$ [Hz]"
    for ax, (signal_key, y_label) in zip(axes, signals):
        lane_data: list[
            tuple[
                str,
                tuple[np.ndarray, np.ndarray] | None,
                tuple[np.ndarray, np.ndarray] | None,
            ]
        ] = []
        spectra_for_scale: list[np.ndarray] = []
        spectra_for_xlim: list[tuple[np.ndarray, np.ndarray] | None] = []

        for lane_key in active_lanes:
            t_cfd, y_cfd, axis_label = _spectrum_axis_y(
                rollouts["cfd"],
                signal_key,
                window_x_canonical=x_canonical,
                start=start,
                end=end,
                aliases=aliases,
                use_full_series=spectrum_full_series,
            )
            t_cmp, y_cmp, axis_label = _spectrum_axis_y(
                rollouts[lane_key],
                signal_key,
                window_x_canonical=x_canonical,
                start=start,
                end=end,
                aliases=aliases,
                use_full_series=spectrum_full_series,
            )
            if frequency_scale_hz is None:
                xlabel = axis_label
            cfd_spec = _scaled_spectrum_x(
                _single_sided_spectrum(
                    t_cfd,
                    y_cfd,
                    area_normalize=False,
                    zero_pad_factor=spectrum_zero_pad_factor,
                ),
                frequency_scale_hz=frequency_scale_hz,
            )
            cmp_spec = _scaled_spectrum_x(
                _single_sided_spectrum(
                    t_cmp,
                    y_cmp,
                    area_normalize=False,
                    zero_pad_factor=spectrum_zero_pad_factor,
                ),
                frequency_scale_hz=frequency_scale_hz,
            )
            lane_data.append((lane_key, cfd_spec, cmp_spec))
            spectra_for_xlim.extend((cfd_spec, cmp_spec))

        xlim = manual_xlim or _spectrum_focus_xlim(
            spectra_for_xlim,
            max_upper=(max_frequency_ratio if frequency_scale_hz is not None else None),
        )
        normalized_lane_data = [
            (
                lane_key,
                _area_normalized_spectrum(cfd_spec, xlim=xlim) if area_normalize else cfd_spec,
                _area_normalized_spectrum(cmp_spec, xlim=xlim) if area_normalize else cmp_spec,
            )
            for lane_key, cfd_spec, cmp_spec in lane_data
        ]
        scale_exclude = set(y_spectrum_scale_exclude) if signal_key == "y_over_D" else set()
        for lane_key, cfd_spec, cmp_spec in normalized_lane_data:
            spectra_for_lane_scale = (cfd_spec,) if lane_key in scale_exclude else (cfd_spec, cmp_spec)
            for spectrum in spectra_for_lane_scale:
                if spectrum is not None:
                    spectra_for_scale.append(_spectrum_y_values(spectrum[1], mode=spectrum_y_mode))

        height = _spectrum_height(spectra_for_scale)
        lane_gap = 1.35 * height
        offsets = lane_gap * np.arange(len(active_lanes) - 1, -1, -1, dtype=float)

        for offset, (lane_key, cfd_spec, cmp_spec) in zip(offsets, normalized_lane_data):
            cfd_style = style["cfd"]
            cmp_style = style.get(lane_key, {})
            comparator_label = _record_label(
                rollouts[lane_key], str(cmp_style.get("label", lane_key))
            )

            ax.axhline(offset, color="0.88", linewidth=0.6, zorder=0)
            if cfd_spec is not None:
                freqs, values = cfd_spec
                ax.plot(
                    freqs,
                    _spectrum_y_values(values, mode=spectrum_y_mode) + offset,
                    color=cfd_style.get("color", "black"),
                    linestyle=cfd_style.get("linestyle", "-"),
                    linewidth=cfd_style.get("linewidth", 1.6),
                    zorder=cfd_style.get("zorder", 4),
                )
            if cmp_spec is not None:
                freqs, values = cmp_spec
                ax.plot(
                    freqs,
                    _spectrum_y_values(values, mode=spectrum_y_mode) + offset,
                    color=cmp_style.get("color", "tab:blue"),
                    linestyle=cmp_style.get("linestyle", "-"),
                    linewidth=cmp_style.get("linewidth", 1.35),
                    zorder=cmp_style.get("zorder", 3),
                )
            text_transform = blended_transform_factory(ax.transAxes, ax.transData)
            ax.text(
                1.01,
                offset,
                comparator_label,
                transform=text_transform,
                ha="left",
                va="center",
                fontsize=base_font_size,
                color=cmp_style.get("color", "0.2"),
                clip_on=False,
            )
            progress_bar.update(f"{y_label}: {comparator_label}")

        ax.set_ylabel(y_label, rotation=0, labelpad=22, va="center")
        ax.set_yticks(offsets)
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
        if frequency_scale_hz is not None:
            ax.axvline(1.0, color="0.5", linewidth=0.8, alpha=0.35, zorder=0)
        _apply_block9_10_spectral_frame(ax)
        ax.set_ylim(-0.15 * height, offsets[0] + 1.15 * height)
        if xlim is not None:
            ax.set_xlim(*xlim)

        ax.set_xlabel(xlabel)

    _add_shared_model_legend(
        fig,
        rollouts,
        model_order=active_lanes,
        style=style,
        include_cfd=True,
        ncol=3,
    )

    saved_paths: list[Path] = []
    if save:
        out_name = basename or f"rollout_response_spectra_Ur_{_filename_ur(ur)}"
        saved_paths = _save_figure(fig, output_dir, out_name, dpi=dpi, formats=save_formats)
        progress_bar.update("saved files")
    fig._saved_paths = saved_paths  # type: ignore[attr-defined]

    if show:
        plt.show()
    return fig, axes


def plot_rollout_corrections_timeseries(
    rollouts: Mapping[str, Any],
    *,
    ur: float | str | None = None,
    output_dir: str | Path | None = None,
    basename: str | None = None,
    t_start: float | None = None,
    t_end: float | None = None,
    cycle_start: float | None = 0.0,
    cycle_end: float | None = 10.0,
    x_axis: str = "cycles",
    field_map: Mapping[str, str | Sequence[str]] | None = None,
    styles: Mapping[str, Mapping[str, Any]] | None = None,
    force_models: Sequence[str] = FORCE_CORRECTION_MODELS,
    frequency_models: Sequence[str] = FREQUENCY_CORRECTION_MODELS,
    x_origin: float = 0.0,
    figsize: tuple[float, float] = (5.85, 4.0),
    dpi: int = 300,
    base_font_size: float = 8.0,
    save: bool = True,
    save_formats: Sequence[str] = ("pdf", "png"),
    show: bool = False,
    progress: bool = False,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot learned correction quantities without CFD/VIVANA response curves."""
    aliases = _merged_field_aliases(field_map)
    style = _merged_styles(styles)
    _set_thesis_rcparams(base_font_size)

    x_canonical, start, end, x_label = _window_axis_settings(
        rollouts,
        aliases,
        x_axis=x_axis,
        t_start=t_start,
        t_end=t_end,
        cycle_start=cycle_start,
        cycle_end=cycle_end,
    )
    x_display_origin = float(x_origin) if x_canonical == "cycles" else 0.0

    fig, axes = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=True,
        constrained_layout=True,
    )
    axes = np.asarray(axes)

    xmins: list[float] = []
    xmaxs: list[float] = []

    correction_specs = (
        (axes[0], force_models, "delta_C_F", r"$\Delta C_F$"),
        (axes[1], frequency_models, "delta_freq", r"$\Delta \hat f$"),
    )
    progress_total = sum(len(model_keys) for _, model_keys, _, _ in correction_specs)
    progress_total += 1 if save and output_dir is not None else 0
    progress_bar = _ProgressBar(progress_total, "Corrections plot", enabled=progress)

    for ax, model_keys, correction_key, y_label in correction_specs:
        plotted = False
        for model_key in model_keys:
            if model_key not in rollouts:
                progress_bar.update(f"{y_label}: skipped {model_key}")
                continue
            if _resolve_field_name(
                rollouts[model_key], correction_key, aliases, required=False
            ) is None:
                progress_bar.update(f"{y_label}: skipped {model_key}")
                continue

            x, y = _windowed_xy(
                rollouts[model_key],
                correction_key,
                x_canonical=x_canonical,
                start=start,
                end=end,
                aliases=aliases,
            )
            x = x - x_display_origin
            model_style = style.get(model_key, {})
            label = _record_label(rollouts[model_key], str(model_style.get("label", model_key)))
            ax.plot(
                x,
                y,
                label=label,
                color=model_style.get("color", "tab:blue"),
                linestyle=model_style.get("linestyle", "-"),
                linewidth=model_style.get("linewidth", 1.45),
                zorder=model_style.get("zorder", 3),
            )
            xmins.append(float(np.nanmin(x)))
            xmaxs.append(float(np.nanmax(x)))
            plotted = True
            progress_bar.update(f"{y_label}: {label}")

        ax.axhline(0.0, color="0.75", linewidth=0.7, zorder=0)
        ax.set_ylabel(y_label, rotation=0, labelpad=16, va="center")
        ax.grid(axis="y", color="0.9", linewidth=0.5)
        ax.grid(axis="x", visible=False)
        _apply_subplot_border(ax)
        if not plotted:
            ax.text(
                0.5,
                0.5,
                "No matching correction field found",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="0.4",
                fontsize=base_font_size,
            )

    if xmins and xmaxs:
        axes[-1].set_xlim(min(xmins), max(xmaxs))
    axes[-1].set_xlabel(x_label)
    _ensure_endpoint_xticks(
        axes[-1],
        start=(None if start is None else float(start) - x_display_origin),
        end=(None if end is None else float(end) - x_display_origin),
        x_label=x_label,
    )

    legend_model_order: list[str] = []
    for model_key in (*force_models, *frequency_models):
        if model_key in rollouts and model_key not in legend_model_order:
            legend_model_order.append(model_key)
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=style.get(model_key, {}).get("color", "tab:blue"),
            linestyle=style.get(model_key, {}).get("linestyle", "-"),
            linewidth=style.get(model_key, {}).get("linewidth", 1.45),
            label=_record_label(rollouts[model_key], str(style.get(model_key, {}).get("label", model_key))),
        )
        for model_key in legend_model_order
    ]
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=len(legend_handles),
            frameon=False,
            borderaxespad=0.0,
        )

    saved_paths: list[Path] = []
    if save:
        out_name = basename or f"rollout_corrections_timeseries_Ur_{_filename_ur(ur)}"
        saved_paths = _save_figure(fig, output_dir, out_name, dpi=dpi, formats=save_formats)
        progress_bar.update("saved files")
    fig._saved_paths = saved_paths  # type: ignore[attr-defined]

    if show:
        plt.show()
    return fig, axes


def plot_rollout_corrections_spectra(
    rollouts: Mapping[str, Any],
    *,
    ur: float | str | None = None,
    output_dir: str | Path | None = None,
    basename: str | None = None,
    t_start: float | None = None,
    t_end: float | None = None,
    cycle_start: float | None = 0.0,
    cycle_end: float | None = 10.0,
    x_axis: str = "cycles",
    field_map: Mapping[str, str | Sequence[str]] | None = None,
    styles: Mapping[str, Mapping[str, Any]] | None = None,
    force_models: Sequence[str] = FORCE_CORRECTION_MODELS,
    frequency_models: Sequence[str] = FREQUENCY_CORRECTION_MODELS,
    area_normalize: bool = False,
    spectrum_full_series: bool = True,
    natural_frequency_hz: float | None = None,
    max_frequency_ratio: float | None = DEFAULT_SPECTRUM_MAX_F_OVER_FN,
    frequency_ratio_xlim: tuple[float, float] | None = DEFAULT_CORRECTION_SPECTRUM_F_OVER_FN_LIMITS,
    spectrum_y_mode: str = DEFAULT_SPECTRUM_Y_MODE,
    spectrum_zero_pad_factor: int = DEFAULT_SPECTRUM_ZERO_PAD_FACTOR,
    figsize: tuple[float, float] = (5.85, 4.0),
    dpi: int = 300,
    base_font_size: float = 8.0,
    save: bool = True,
    save_formats: Sequence[str] = ("pdf", "png"),
    show: bool = False,
    progress: bool = False,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot spectra of learned correction quantities."""
    aliases = _merged_field_aliases(field_map)
    style = _merged_styles(styles)
    _set_thesis_rcparams(base_font_size)
    frequency_scale_hz = _spectrum_frequency_scale_hz(
        rollouts,
        aliases,
        requested=natural_frequency_hz,
    )
    manual_xlim = _valid_xlim(frequency_ratio_xlim if frequency_scale_hz is not None else None)

    x_canonical, start, end, _ = _window_axis_settings(
        rollouts,
        aliases,
        x_axis=x_axis,
        t_start=t_start,
        t_end=t_end,
        cycle_start=cycle_start,
        cycle_end=cycle_end,
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=False,
        constrained_layout=True,
    )
    axes = np.asarray(axes)

    correction_specs = (
        (axes[0], force_models, "delta_C_F", r"$\Delta C_F$"),
        (axes[1], frequency_models, "delta_freq", r"$\Delta \hat f$"),
    )
    progress_total = sum(len(model_keys) for _, model_keys, _, _ in correction_specs)
    progress_total += 1 if save and output_dir is not None else 0
    progress_bar = _ProgressBar(progress_total, "Corrections spectra", enabled=progress)

    xlabel = r"$\omega/\omega_n$" if frequency_scale_hz is not None else r"$\omega$ [Hz]"
    for ax, model_keys, correction_key, y_label in correction_specs:
        plotted = False
        spectra_for_plot: list[tuple[str, tuple[np.ndarray, np.ndarray]]] = []
        spectra_for_xlim: list[tuple[np.ndarray, np.ndarray] | None] = []
        for model_key in model_keys:
            if model_key not in rollouts:
                progress_bar.update(f"{y_label}: skipped {model_key}")
                continue
            if _resolve_field_name(
                rollouts[model_key], correction_key, aliases, required=False
            ) is None:
                progress_bar.update(f"{y_label}: skipped {model_key}")
                continue

            t, y, axis_label = _spectrum_axis_y(
                rollouts[model_key],
                correction_key,
                window_x_canonical=x_canonical,
                start=start,
                end=end,
                aliases=aliases,
                use_full_series=spectrum_full_series,
            )
            if frequency_scale_hz is None:
                xlabel = axis_label
            spectrum = _scaled_spectrum_x(
                _single_sided_spectrum(
                    t,
                    y,
                    area_normalize=False,
                    zero_pad_factor=spectrum_zero_pad_factor,
                ),
                frequency_scale_hz=frequency_scale_hz,
            )
            spectra_for_xlim.append(spectrum)
            if spectrum is None:
                progress_bar.update(f"{y_label}: skipped {model_key}")
                continue
            spectra_for_plot.append((model_key, spectrum))

        xlim = manual_xlim or _spectrum_focus_xlim(
            spectra_for_xlim,
            max_upper=(max_frequency_ratio if frequency_scale_hz is not None else None),
        )
        for model_key, raw_spectrum in spectra_for_plot:
            spectrum = (
                _area_normalized_spectrum(raw_spectrum, xlim=xlim)
                if area_normalize
                else raw_spectrum
            )
            if spectrum is None:
                progress_bar.update(f"{y_label}: skipped {model_key}")
                continue
            freqs, values = spectrum
            model_style = style.get(model_key, {})
            label = _record_label(rollouts[model_key], str(model_style.get("label", model_key)))
            ax.plot(
                freqs,
                _spectrum_y_values(values, mode=spectrum_y_mode),
                label=label,
                color=model_style.get("color", "tab:blue"),
                linestyle=model_style.get("linestyle", "-"),
                linewidth=model_style.get("linewidth", 1.45),
                zorder=model_style.get("zorder", 3),
            )
            plotted = True
            progress_bar.update(f"{y_label}: {label}")

        if xlim is not None:
            ax.set_xlim(*xlim)
        ax.set_ylabel(y_label, rotation=0, labelpad=16, va="center")
        _apply_block9_10_spectral_frame(ax)
        ax.set_xlabel(xlabel)
        if not plotted:
            ax.text(
                0.5,
                0.5,
                "No matching correction spectrum found",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="0.4",
                fontsize=base_font_size,
            )

    legend_model_order: list[str] = []
    for model_key in (*force_models, *frequency_models):
        if model_key in rollouts and model_key not in legend_model_order:
            legend_model_order.append(model_key)
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=style.get(model_key, {}).get("color", "tab:blue"),
            linestyle=style.get(model_key, {}).get("linestyle", "-"),
            linewidth=style.get(model_key, {}).get("linewidth", 1.45),
            label=_record_label(rollouts[model_key], str(style.get(model_key, {}).get("label", model_key))),
        )
        for model_key in legend_model_order
    ]
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=len(legend_handles),
            frameon=False,
            borderaxespad=0.0,
        )

    saved_paths: list[Path] = []
    if save:
        out_name = basename or f"rollout_corrections_spectra_Ur_{_filename_ur(ur)}"
        saved_paths = _save_figure(fig, output_dir, out_name, dpi=dpi, formats=save_formats)
        progress_bar.update("saved files")
    fig._saved_paths = saved_paths  # type: ignore[attr-defined]

    if show:
        plt.show()
    return fig, axes


def plot_tau_sigma_sensitivity_timeseries(
    rollouts: Mapping[str, Any],
    *,
    ur: float | str | None = None,
    output_dir: str | Path | None = None,
    basename: str | None = None,
    t_start: float | None = None,
    t_end: float | None = None,
    cycle_start: float | None = 0.0,
    cycle_end: float | None = 10.0,
    x_axis: str = "cycles",
    field_map: Mapping[str, str | Sequence[str]] | None = None,
    styles: Mapping[str, Mapping[str, Any]] | None = None,
    center_on_lane_mean: bool = True,
    figsize: tuple[float, float] = (5.85, 4.4),
    dpi: int = 300,
    base_font_size: float = 8.0,
    save: bool = True,
    save_formats: Sequence[str] = ("pdf", "png"),
    show: bool = False,
    progress: bool = False,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot TD sigma hidden states in vertically offset tau/T_ref lanes."""
    if not rollouts:
        raise ValueError("rollouts must contain at least one tau sensitivity record.")

    aliases = _merged_field_aliases(field_map)
    style = _merged_styles(styles)
    _set_thesis_rcparams(base_font_size)

    x_canonical, start, end, x_label = _window_axis_settings(
        rollouts,
        aliases,
        x_axis=x_axis,
        t_start=t_start,
        t_end=t_end,
        cycle_start=cycle_start,
        cycle_end=cycle_end,
    )

    lane_keys = list(rollouts.keys())
    signals = (("sigma_dy", r"$\sigma_{\dot y}$"), ("sigma_ddy", r"$\sigma_{\ddot y}$"))
    progress_bar = _ProgressBar(
        len(signals) * len(lane_keys) + (1 if save and output_dir is not None else 0),
        "Tau sigma plot",
        enabled=progress,
    )

    fig, axes = plt.subplots(
        len(signals),
        1,
        figsize=figsize,
        sharex=True,
        constrained_layout=True,
    )
    axes = np.asarray(axes)

    xmins: list[float] = []
    xmaxs: list[float] = []

    for ax, (signal_key, y_label) in zip(axes, signals):
        lane_data: list[tuple[str, np.ndarray, np.ndarray, float]] = []
        centered_for_scale: list[np.ndarray] = []

        for lane_key in lane_keys:
            x, y = _windowed_xy(
                rollouts[lane_key],
                signal_key,
                x_canonical=x_canonical,
                start=start,
                end=end,
                aliases=aliases,
            )
            lane_center = float(np.nanmean(y)) if center_on_lane_mean else 0.0
            lane_data.append((lane_key, x, y, lane_center))
            centered_for_scale.append(y - lane_center)
            xmins.append(float(np.nanmin(x)))
            xmaxs.append(float(np.nanmax(x)))

        amp = _robust_amplitude(centered_for_scale)
        lane_gap = 3.0 * amp
        offsets = lane_gap * np.arange(len(lane_keys) - 1, -1, -1, dtype=float)

        for offset, (lane_key, x, y, lane_center) in zip(offsets, lane_data):
            lane_style = style.get(lane_key, {})
            label = _record_label(rollouts[lane_key], str(lane_style.get("label", lane_key)))
            ax.axhline(offset, color="0.88", linewidth=0.6, zorder=0)
            ax.plot(
                x,
                y - lane_center + offset,
                color=lane_style.get("color", "tab:blue"),
                linestyle=lane_style.get("linestyle", "-"),
                linewidth=lane_style.get("linewidth", 1.35),
                zorder=lane_style.get("zorder", 3),
            )
            text_transform = blended_transform_factory(ax.transAxes, ax.transData)
            ax.text(
                1.01,
                offset,
                label,
                transform=text_transform,
                ha="left",
                va="center",
                fontsize=base_font_size,
                color=lane_style.get("color", "0.2"),
                clip_on=False,
            )
            progress_bar.update(f"{y_label}: {label}")

        ax.set_ylabel(y_label, rotation=0, labelpad=24, va="center")
        ax.set_yticks(offsets)
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", color="0.9", linewidth=0.5)
        _apply_subplot_border(ax)
        ax.set_ylim(-1.45 * amp, offsets[0] + 1.45 * amp)

    if xmins and xmaxs:
        axes[-1].set_xlim(min(xmins), max(xmaxs))
    axes[-1].set_xlabel(x_label)
    _ensure_endpoint_xticks(axes[-1], start=start, end=end, x_label=x_label)

    saved_paths: list[Path] = []
    if save:
        out_name = basename or f"rollout_sigma_tau_sensitivity_Ur_{_filename_ur(ur)}"
        saved_paths = _save_figure(fig, output_dir, out_name, dpi=dpi, formats=save_formats)
        progress_bar.update("saved files")
    fig._saved_paths = saved_paths  # type: ignore[attr-defined]

    if show:
        plt.show()
    return fig, axes


def _dominant_frequency_from_time_signal(time_values: np.ndarray, signal_values: np.ndarray) -> float:
    time_arr = np.asarray(time_values, dtype=float).reshape(-1)
    signal_arr = np.asarray(signal_values, dtype=float).reshape(-1)
    n = min(time_arr.size, signal_arr.size)
    if n < 4:
        return float("nan")
    time_arr = time_arr[:n]
    signal_arr = signal_arr[:n]
    dt = float(np.nanmedian(np.diff(time_arr)))
    if not np.isfinite(dt) or dt <= 0.0:
        return float("nan")
    centered = signal_arr - float(np.nanmean(signal_arr))
    if np.allclose(centered, 0.0, equal_nan=False):
        return float("nan")
    freqs = np.fft.rfftfreq(centered.size, d=dt)
    power = np.abs(np.fft.rfft(centered)) ** 2
    if freqs.size < 2 or power.size < 2:
        return float("nan")
    power[0] = 0.0
    valid = np.isfinite(freqs) & np.isfinite(power) & (freqs > 0.0)
    if not np.any(valid):
        return float("nan")
    valid_indices = np.flatnonzero(valid)
    peak_index = int(valid_indices[int(np.argmax(power[valid]))])
    interpolated_index = float(peak_index)
    if 1 <= peak_index < power.size - 1:
        y_prev = float(power[peak_index - 1])
        y_peak = float(power[peak_index])
        y_next = float(power[peak_index + 1])
        denom = y_prev - 2.0 * y_peak + y_next
        if np.isfinite(denom) and abs(denom) > 1.0e-18:
            delta = 0.5 * (y_prev - y_next) / denom
            if np.isfinite(delta):
                interpolated_index += float(np.clip(delta, -1.0, 1.0))
    df = float(freqs[1] - freqs[0])
    return float(max(interpolated_index * df, 0.0)) if np.isfinite(df) and df > 0.0 else float(freqs[peak_index])


def _series_flow_speed(series: Mapping[str, Any], expected_length: int) -> np.ndarray:
    if "flow_speed" in series:
        flow = np.asarray(series["flow_speed"], dtype=float).reshape(-1)
    else:
        td_context = np.asarray(series["td_context"], dtype=float)
        if td_context.ndim != 2 or td_context.shape[1] < 5:
            raise ValueError("series['td_context'] must have shape (n, >=5) to recover flow speed.")
        flow = np.asarray(td_context[:, 4], dtype=float).reshape(-1)
    if flow.size == 1:
        return np.full((expected_length,), float(flow[0]), dtype=float)
    if flow.size != expected_length:
        raise ValueError(f"Expected flow speed of length {expected_length}, got {flow.size}.")
    return flow


def _force_per_m_to_coefficient(series: Mapping[str, Any], force_per_m: np.ndarray) -> np.ndarray:
    force_arr = np.asarray(force_per_m, dtype=float).reshape(-1)
    flow_speed = _series_flow_speed(series, force_arr.size)
    denom = 0.5 * float(series["rho"]) * float(series["diameter"]) * flow_speed * flow_speed
    coeff = np.full(force_arr.shape, np.nan, dtype=float)
    valid = np.isfinite(force_arr) & np.isfinite(denom) & (np.abs(denom) > 0.0)
    coeff[valid] = force_arr[valid] / denom[valid]
    return coeff


def _slice_series_from_index(series: Mapping[str, Any], start_idx: int) -> dict[str, Any]:
    time = np.asarray(series["time"], dtype=float).reshape(-1)
    idx = int(np.clip(start_idx, 0, max(time.size - 1, 0)))
    if time.size - idx < 2:
        raise ValueError(
            f"Cannot start rollout at index {idx}; only {time.size - idx} sample(s) remain."
        )
    out = dict(series)
    for key, value in series.items():
        try:
            arr = np.asarray(value)
        except Exception:
            continue
        if arr.shape[:1] == time.shape[:1]:
            out[key] = arr[idx:].copy()
    out["rollout_start_idx"] = int(idx)
    return out


def _rollout_key_from_label(label: str) -> str:
    if label in MODEL_LABEL_TO_ROLLOUT_KEY:
        return MODEL_LABEL_TO_ROLLOUT_KEY[label]
    lowered = label.lower()
    if "combined" in lowered:
        return "combined"
    if "standalone" in lowered or "latent" in lowered or "rnn" in lowered:
        return "standalone"
    if "freq" in lowered or "fhat" in lowered:
        return "frequency"
    if "force" in lowered or "mean" in lowered:
        return "force"
    return lowered.replace(" ", "_")


def _natural_frequency_hz_from_series(series: Mapping[str, Any]) -> float:
    for key in ("natural_frequency_hz", "structural_frequency_hz", "f_n_hz", "fn_hz"):
        if key not in series:
            continue
        values = np.asarray(series[key], dtype=float).reshape(-1)
        finite = values[np.isfinite(values)]
        if finite.size and float(finite[0]) > 0.0:
            return float(finite[0])

    stiffness = float(series.get("stiffness", np.nan))
    effective_mass = float(series.get("effective_mass", np.nan))
    if np.isfinite(stiffness) and stiffness > 0.0 and np.isfinite(effective_mass) and effective_mass > 0.0:
        return float(np.sqrt(stiffness / effective_mass) / (2.0 * np.pi))
    return float("nan")


def _spectrum_resolution_metadata(time_values: np.ndarray, natural_frequency_hz: float | None) -> dict[str, float]:
    time_arr = np.asarray(time_values, dtype=float).reshape(-1)
    finite = time_arr[np.isfinite(time_arr)]
    if finite.size < 2:
        duration = float("nan")
        dt = float("nan")
    else:
        finite = np.sort(finite)
        duration = float(finite[-1] - finite[0])
        diffs = np.diff(finite)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
        dt = float(np.nanmedian(diffs)) if diffs.size else float("nan")
    df_hz = 1.0 / duration if np.isfinite(duration) and duration > 0.0 else float("nan")
    fn = float(natural_frequency_hz) if natural_frequency_hz is not None else float("nan")
    df_over_fn = df_hz / fn if np.isfinite(df_hz) and np.isfinite(fn) and fn > 0.0 else float("nan")
    return {
        "spectrum_duration_s": duration,
        "spectrum_dt_s": dt,
        "spectrum_df_hz": df_hz,
        "spectrum_df_over_fn": df_over_fn,
    }


def _plot_record_from_rollout(
    *,
    label: str,
    series: Mapping[str, Any],
    time_values: np.ndarray,
    cycles: np.ndarray,
    displacement: np.ndarray,
    force_per_m: np.ndarray,
    delta_force_per_m: np.ndarray | None = None,
    delta_freq: np.ndarray | None = None,
    response_start_idx: int | None = None,
) -> dict[str, np.ndarray | str]:
    time_arr = np.asarray(time_values, dtype=float).reshape(-1)
    disp_arr = np.asarray(displacement, dtype=float).reshape(-1)
    force_arr = np.asarray(force_per_m, dtype=float).reshape(-1)
    if response_start_idx is not None:
        start_idx = int(response_start_idx)
        if start_idx > 0:
            prefix = slice(0, min(start_idx, disp_arr.size))
            disp_arr = disp_arr.copy()
            force_arr = force_arr.copy()
            disp_arr[prefix] = np.nan
            force_arr[prefix] = np.nan
    diameter = float(series["diameter"])
    if not np.isfinite(diameter) or diameter <= 0.0:
        raise ValueError("series['diameter'] must be positive and finite to compute y/D.")
    out: dict[str, np.ndarray | str] = {
        "label": label,
        "time": time_arr,
        "cycles": np.asarray(cycles, dtype=float).reshape(-1),
        "natural_frequency_hz": np.asarray([_natural_frequency_hz_from_series(series)], dtype=float),
        "y_over_D": disp_arr / diameter,
        "C_F": _force_per_m_to_coefficient(series, force_arr),
    }
    if delta_force_per_m is not None:
        out["delta_C_F"] = _force_per_m_to_coefficient(
            series,
            np.asarray(delta_force_per_m, dtype=float).reshape(-1),
        )
    if delta_freq is not None:
        out["delta_freq"] = np.asarray(delta_freq, dtype=float).reshape(-1)
    return out


def _load_block9_10_sources(*, device: str = "cpu") -> list[Any]:
    from vivana_cfd_data_pipeline.helpers.model_rollouts import load_trained_model_sources

    return load_trained_model_sources(
        [dict(spec) for spec in BLOCK_9_10_MODEL_SPECS],
        repo_root=ROOT,
        device=device,
    )


def _baseline_td_params() -> dict[str, float]:
    from vivana_cfd_data_pipeline.helpers.spectral_metrics import resolve_td_params

    return resolve_td_params(VIVANA_TD_BASELINE_PARAM_OVERRIDES)


def _baseline_td_params_from_overrides() -> dict[str, float]:
    return {
        "Cv": float(VIVANA_TD_BASELINE_PARAM_OVERRIDES["td_cv"]),
        "Cd": float(VIVANA_TD_BASELINE_PARAM_OVERRIDES["td_cd"]),
        "Ca": float(VIVANA_TD_BASELINE_PARAM_OVERRIDES["td_ca"]),
        "fhat0": float(VIVANA_TD_BASELINE_PARAM_OVERRIDES["td_fhat0"]),
        "fhat_min": float(VIVANA_TD_BASELINE_PARAM_OVERRIDES["td_fhat_min"]),
        "fhat_max": float(VIVANA_TD_BASELINE_PARAM_OVERRIDES["td_fhat_max"]),
        "n_memory": 500.0,
    }


def _select_real_series(
    *,
    dataset_root: Path,
    ur: float,
    case_index: int,
) -> tuple[dict[str, Any], Path, list[tuple[Path, dict[str, Any]]]]:
    from vivana_cfd_data_pipeline.scripts.training_npz_loader import iter_all_npz_files, load_series

    files = list(iter_all_npz_files(dataset_root))
    if not files:
        raise FileNotFoundError(f"No NPZ files found in {dataset_root}.")
    loaded = [(path, load_series(path)) for path in files]

    def raw_ur(item: tuple[Path, dict[str, Any]]) -> float:
        value = item[1].get("ur", item[1].get("ur_label", float("nan")))
        return float(np.asarray(value, dtype=float).reshape(-1)[0])

    matching = [
        item
        for item in loaded
        if np.isfinite(raw_ur(item)) and np.isclose(raw_ur(item), float(ur), rtol=0.0, atol=1.0e-8)
    ]
    if not matching:
        matching = sorted(
            loaded,
            key=lambda item: abs(float(item[1].get("ur_effective", raw_ur(item))) - float(ur)),
        )[:1]
    matching = sorted(matching, key=lambda item: item[0].name)
    if not (0 <= int(case_index) < len(matching)):
        raise IndexError(
            f"case_index={case_index} is out of range for U_r={ur:g}; "
            f"{len(matching)} matching case(s) are available."
        )
    selected_path, selected_series = matching[int(case_index)]
    return selected_series, selected_path, matching


def build_real_block9_10_rollouts(
    *,
    ur: float = DEFAULT_REAL_UR,
    case_index: int = 0,
    dataset_root: str | Path = DEFAULT_REAL_DATASET_ROOT,
    validation_reduction_factor: int = BLOCK_10_VALIDATION_REDUCTION_FACTOR,
    td_memory_tau_s: float | str | None = BLOCK_9_10_TD_MEMORY_TAU_SECONDS,
    mass_source: str = TD_MASS_SOURCE,
    device: str = "cpu",
    progress: bool = True,
) -> tuple[dict[str, dict[str, np.ndarray | str]], dict[str, Any]]:
    """Load real CFD/VIVANA/model rollouts matching Blocks 9 and 10.

    The default model checkpoints are the Block 9/10 best models:
    force correction, frequency correction, combined correction, and standalone.
    """
    dataset_root_path = Path(dataset_root)
    progress_bar = _ProgressBar(5 + len(BLOCK_9_10_MODEL_SPECS), "Real rollout data", enabled=progress)

    series, selected_path, matching_cases = _select_real_series(
        dataset_root=dataset_root_path,
        ur=float(ur),
        case_index=int(case_index),
    )
    progress_bar.update(f"loaded CFD case {selected_path.name}")

    sources = _load_block9_10_sources(device=device)
    progress_bar.update("loaded Block 9/10 best model checkpoints")

    from vivana_cfd_data_pipeline.helpers.model_rollouts import (
        _reduce_series_for_validation,
        simulate_checkpoint_series_rollout,
        simulate_vivana_td_stepwise,
    )

    baseline_td_params = _baseline_td_params()
    validation_series = _reduce_series_for_validation(
        series,
        reduce_time=True,
        reduction_factor=int(validation_reduction_factor),
        cut_start_seconds=0.0,
        td_params=(sources[0].base_td_params if sources else baseline_td_params),
        td_memory_cfg=(sources[0].td_memory_cfg if sources else None),
    )
    progress_bar.update(f"applied validation reduction factor {int(validation_reduction_factor)}")

    time_arr = np.asarray(validation_series["time"], dtype=float).reshape(-1)
    displacement_ref = np.asarray(validation_series["displacement"], dtype=float).reshape(-1)
    natural_frequency_hz = _natural_frequency_hz_from_series(validation_series)
    dominant_frequency_hz = _dominant_frequency_from_time_signal(time_arr, displacement_ref)
    if not np.isfinite(dominant_frequency_hz) or dominant_frequency_hz <= 0.0:
        dominant_frequency_hz = 1.0 / max(float(time_arr[-1] - time_arr[0]), 1.0)
    cycles = (time_arr - float(time_arr[0])) * float(dominant_frequency_hz)

    standalone_response_start_idx = 0
    standalone_response_start_cycle = float(cycles[0])
    standalone_rollouts: dict[str, dict[str, np.ndarray]] = {}
    standalone_source_ids: set[int] = set()
    for source in sources:
        key = _rollout_key_from_label(source.label)
        is_standalone = key == "standalone" or getattr(source, "kind", "") == "latent_rnn"
        if not is_standalone:
            continue
        model_rollout = simulate_checkpoint_series_rollout(
            source,
            validation_series,
            mass_source=mass_source,
            td_memory_tau_s=td_memory_tau_s,
        )
        standalone_rollouts[key] = model_rollout
        standalone_source_ids.add(id(source))
        response_start_idx = int(model_rollout.get("evaluation_start_idx", 0))
        standalone_response_start_idx = int(np.clip(response_start_idx, 0, max(cycles.size - 1, 0)))
        standalone_response_start_cycle = float(cycles[standalone_response_start_idx])
        progress_bar.update(f"ran {source.label} to locate encoder start")
        break

    rollout_series = _slice_series_from_index(validation_series, standalone_response_start_idx)
    rollout_time_arr = time_arr[standalone_response_start_idx:]
    rollout_cycles = cycles[standalone_response_start_idx:]
    progress_bar.update(
        f"aligned rollout start at cycle {standalone_response_start_cycle:g}"
    )

    rollouts: dict[str, dict[str, np.ndarray | str]] = {
        "cfd": _plot_record_from_rollout(
            label="CFD reference",
            series=rollout_series,
            time_values=rollout_time_arr,
            cycles=rollout_cycles,
            displacement=rollout_series["displacement"],
            force_per_m=rollout_series["force_per_m"],
        )
    }

    baseline_rollout = simulate_vivana_td_stepwise(
        rollout_series,
        td_params=baseline_td_params,
        mass_source=mass_source,
        td_memory_tau_s=td_memory_tau_s,
    )
    rollouts["vivana"] = _plot_record_from_rollout(
        label="VIVANA-TD baseline",
        series=rollout_series,
        time_values=rollout_time_arr,
        cycles=rollout_cycles,
        displacement=baseline_rollout["displacement_td"],
        force_per_m=baseline_rollout["force_td"],
    )
    progress_bar.update("ran VIVANA-TD baseline")

    for source in sources:
        key = _rollout_key_from_label(source.label)
        is_standalone = key == "standalone" or getattr(source, "kind", "") == "latent_rnn"
        if is_standalone:
            model_rollout = standalone_rollouts.get(key)
            if model_rollout is None:
                model_rollout = simulate_checkpoint_series_rollout(
                    source,
                    validation_series,
                    mass_source=mass_source,
                    td_memory_tau_s=td_memory_tau_s,
                )
            sl = slice(standalone_response_start_idx, None)
            displacement = np.asarray(model_rollout["displacement"], dtype=float).reshape(-1)[sl]
            force_per_m = np.asarray(model_rollout["force"], dtype=float).reshape(-1)[sl]
            delta_force = None
            delta_freq = None
        else:
            model_rollout = simulate_checkpoint_series_rollout(
                source,
                rollout_series,
                mass_source=mass_source,
                td_memory_tau_s=td_memory_tau_s,
            )
            displacement = model_rollout["displacement"]
            force_per_m = model_rollout["force"]
            delta_force = model_rollout.get("delta_force")
            delta_freq = model_rollout.get("delta_fhat")

        rollouts[key] = _plot_record_from_rollout(
            label=source.label,
            series=rollout_series,
            time_values=rollout_time_arr,
            cycles=rollout_cycles,
            displacement=displacement,
            force_per_m=force_per_m,
            delta_force_per_m=delta_force,
            delta_freq=delta_freq,
        )
        if id(source) not in standalone_source_ids:
            progress_bar.update(f"ran {source.label}")

    metadata = {
        "selected_path": selected_path,
        "matching_case_paths": [path for path, _ in matching_cases],
        "raw_ur": float(np.asarray(series.get("ur", np.nan), dtype=float).reshape(-1)[0]),
        "effective_ur": float(series.get("ur_effective", np.nan)),
        "dominant_frequency_hz": float(dominant_frequency_hz),
        "natural_frequency_hz": float(natural_frequency_hz),
        "validation_reduction_factor": int(validation_reduction_factor),
        "td_memory_tau_s": td_memory_tau_s,
        "mass_source": mass_source,
        "model_paths": [source.path for source in sources],
        "standalone_response_start_idx": standalone_response_start_idx,
        "standalone_response_start_cycle": standalone_response_start_cycle,
    }
    metadata.update(_spectrum_resolution_metadata(time_arr, natural_frequency_hz))
    return rollouts, metadata


def _tau_over_tref_key(value: float) -> str:
    return f"tau_{float(value):g}".replace(".", "p").replace("-", "m")


def _tau_over_tref_label(value: float) -> str:
    return rf"$\tau/T_{{ref}}={float(value):g}$"


def _tau_sigma_styles(tau_values: Sequence[float]) -> dict[str, dict[str, Any]]:
    cmap = plt.get_cmap("viridis")
    values = [float(value) for value in tau_values]
    n = max(1, len(values) - 1)
    return {
        _tau_over_tref_key(value): {
            "label": _tau_over_tref_label(value),
            "color": cmap(idx / n),
            "linestyle": "-",
            "linewidth": 1.35,
            "zorder": 3,
        }
        for idx, value in enumerate(values)
    }


def _sigma_record_from_td_rollout(
    *,
    label: str,
    time_values: np.ndarray,
    cycles: np.ndarray,
    rollout: Mapping[str, Any],
) -> dict[str, np.ndarray | str]:
    return {
        "label": label,
        "time": np.asarray(time_values, dtype=float).reshape(-1),
        "cycles": np.asarray(cycles, dtype=float).reshape(-1),
        "sigma_dy": np.asarray(rollout["sig_dy_td"], dtype=float).reshape(-1),
        "sigma_ddy": np.asarray(rollout["sig_ddy_td"], dtype=float).reshape(-1),
    }


def _reduce_series_time_for_plot(series: Mapping[str, Any], *, reduction_factor: int) -> dict[str, Any]:
    rf = max(1, int(reduction_factor))
    if rf <= 1:
        return dict(series)

    time = np.asarray(series["time"], dtype=float).reshape(-1)
    idx = np.arange(0, time.size, rf, dtype=int)
    if idx.size < 2:
        raise ValueError(f"Series {series.get('name', '<unknown>')} is too short after reduction.")

    reduced = dict(series)
    for key in (
        "time",
        "displacement",
        "velocity",
        "acceleration",
        "force_total",
        "force_per_m",
        "force_td_stored",
        "td_context",
    ):
        if key not in reduced:
            continue
        arr = np.asarray(reduced[key])
        if arr.shape[:1] == time.shape[:1]:
            reduced[key] = arr[idx].copy()
    reduced["validation_reduction_factor"] = int(rf)
    return reduced


def _recompute_sigma_history_for_tau_over_tref(
    series: Mapping[str, Any],
    *,
    tau_over_tref: float,
    td_params: Mapping[str, float],
) -> dict[str, np.ndarray]:
    time = np.asarray(series["time"], dtype=float).reshape(-1)
    velocity = np.asarray(series["velocity"], dtype=float).reshape(-1)
    acceleration = np.asarray(series["acceleration"], dtype=float).reshape(-1)
    td_context = np.asarray(series["td_context"], dtype=float)
    n = min(time.size, velocity.size, acceleration.size, td_context.shape[0])
    if n < 2:
        raise ValueError("Need at least two samples to recompute sigma history.")
    if td_context.ndim != 2 or td_context.shape[1] < 5:
        raise ValueError("series['td_context'] must have shape (n, >=5).")

    diameter = float(series["diameter"])
    fhat0 = float(td_params["fhat0"])
    tau_ratio = float(tau_over_tref)
    if not np.isfinite(diameter) or diameter <= 0.0:
        raise ValueError("series['diameter'] must be positive and finite.")
    if not np.isfinite(fhat0) or fhat0 <= 0.0:
        raise ValueError("td_params['fhat0'] must be positive and finite.")
    if not np.isfinite(tau_ratio) or tau_ratio <= 0.0:
        raise ValueError("tau_over_tref must be positive and finite.")

    sig_dy = np.empty((n,), dtype=float)
    sig_ddy = np.empty((n,), dtype=float)
    sig_dy[0] = float(td_context[0, 2])
    sig_ddy[0] = float(td_context[0, 3])

    for idx in range(n - 1):
        dt_step = float(time[idx + 1] - time[idx])
        flow_speed = float(td_context[idx, 4])
        if not np.isfinite(dt_step) or dt_step <= 0.0:
            raise ValueError("Time values must be strictly increasing to recompute sigma history.")
        if not np.isfinite(flow_speed) or abs(flow_speed) <= 0.0:
            raise ValueError("Need finite non-zero flow speed to resolve tau/T_ref.")
        tau_seconds = tau_ratio * diameter / (fhat0 * abs(flow_speed))
        n_memory = max(1.0, float(round(tau_seconds / dt_step)))

        speed_mag = np.sqrt(max(flow_speed * flow_speed + float(velocity[idx]) * float(velocity[idx]), 1.0e-12))
        projection = flow_speed / max(speed_mag, 1.0e-12)
        dy_r = float(velocity[idx]) * projection
        ddy_r = float(acceleration[idx]) * projection
        sig_dy[idx + 1] = np.sqrt(
            max(((n_memory - 1.0) / n_memory) * (sig_dy[idx] * sig_dy[idx]) + (dy_r * dy_r) / n_memory, 1.0e-12)
        )
        sig_ddy[idx + 1] = np.sqrt(
            max(((n_memory - 1.0) / n_memory) * (sig_ddy[idx] * sig_ddy[idx]) + (ddy_r * ddy_r) / n_memory, 1.0e-12)
        )

    return {"sig_dy_td": sig_dy, "sig_ddy_td": sig_ddy}


def build_real_tau_sigma_sensitivity_rollouts(
    *,
    ur: float = DEFAULT_TAU_SIGMA_UR,
    tau_over_tref_values: Sequence[float] = DEFAULT_TAU_OVER_TREF_VALUES,
    case_index: int = 0,
    dataset_root: str | Path = DEFAULT_REAL_DATASET_ROOT,
    validation_reduction_factor: int = BLOCK_10_VALIDATION_REDUCTION_FACTOR,
    mass_source: str = TD_MASS_SOURCE,
    progress: bool = True,
) -> tuple[dict[str, dict[str, np.ndarray | str]], dict[str, Any], dict[str, dict[str, Any]]]:
    """Build VIVANA-TD sigma hidden-state rollouts for several tau/T_ref values."""
    tau_values = tuple(float(value) for value in tau_over_tref_values)
    if not tau_values:
        raise ValueError("tau_over_tref_values must contain at least one value.")
    if any((not np.isfinite(value)) or value <= 0.0 for value in tau_values):
        raise ValueError("All tau/T_ref values must be positive and finite.")

    progress_bar = _ProgressBar(
        3 + len(tau_values),
        "Tau sigma data",
        enabled=progress,
    )
    series, selected_path, matching_cases = _select_real_series(
        dataset_root=Path(dataset_root),
        ur=float(ur),
        case_index=int(case_index),
    )
    progress_bar.update(f"loaded CFD case {selected_path.name}")

    baseline_td_params = _baseline_td_params_from_overrides()
    validation_series = _reduce_series_time_for_plot(
        series,
        reduction_factor=int(validation_reduction_factor),
    )
    progress_bar.update(f"applied validation reduction factor {int(validation_reduction_factor)}")

    time_arr = np.asarray(validation_series["time"], dtype=float).reshape(-1)
    displacement_ref = np.asarray(validation_series["displacement"], dtype=float).reshape(-1)
    natural_frequency_hz = _natural_frequency_hz_from_series(validation_series)
    dominant_frequency_hz = _dominant_frequency_from_time_signal(time_arr, displacement_ref)
    if not np.isfinite(dominant_frequency_hz) or dominant_frequency_hz <= 0.0:
        dominant_frequency_hz = 1.0 / max(float(time_arr[-1] - time_arr[0]), 1.0)
    cycles = (time_arr - float(time_arr[0])) * float(dominant_frequency_hz)
    progress_bar.update("resolved cycle axis")

    rollouts: dict[str, dict[str, np.ndarray | str]] = {}
    for tau_value in tau_values:
        sigma_history = _recompute_sigma_history_for_tau_over_tref(
            validation_series,
            tau_over_tref=float(tau_value),
            td_params=baseline_td_params,
        )
        key = _tau_over_tref_key(tau_value)
        rollouts[key] = _sigma_record_from_td_rollout(
            label=_tau_over_tref_label(tau_value),
            time_values=time_arr,
            cycles=cycles,
            rollout=sigma_history,
        )
        progress_bar.update(f"computed tau_over_tref:{tau_value:.12g}")

    metadata = {
        "selected_path": selected_path,
        "matching_case_paths": [path for path, _ in matching_cases],
        "raw_ur": float(np.asarray(series.get("ur", np.nan), dtype=float).reshape(-1)[0]),
        "effective_ur": float(series.get("ur_effective", np.nan)),
        "dominant_frequency_hz": float(dominant_frequency_hz),
        "natural_frequency_hz": float(natural_frequency_hz),
        "validation_reduction_factor": int(validation_reduction_factor),
        "tau_over_tref_values": tau_values,
        "mass_source": mass_source,
    }
    return rollouts, metadata, _tau_sigma_styles(tau_values)


def make_dummy_rollouts() -> dict[str, dict[str, np.ndarray | str]]:
    """Create synthetic data for testing the plotting layout.

    Replace these arrays with your real rollout arrays, for example:

    rollouts["cfd"]["time"] = cfd_time
    rollouts["cfd"]["cycles"] = cfd_cycles
    rollouts["cfd"]["y_over_D"] = cfd_y_over_D
    rollouts["cfd"]["C_F"] = cfd_force_coefficient
    """
    cycles = np.linspace(0.0, 30.0, 3000)
    time = cycles.copy()
    phase = 2.0 * np.pi * cycles

    cfd_y = 0.42 * np.sin(phase + 0.08) + 0.025 * np.sin(2.0 * phase)
    cfd_force = 1.15 * np.sin(phase + 0.72) + 0.12 * np.sin(2.0 * phase - 0.2)

    vivana_y = 0.36 * np.sin(0.985 * phase - 0.02)
    vivana_force = 0.95 * np.sin(0.985 * phase + 0.55)

    force_y = 0.39 * np.sin(0.995 * phase + 0.02)
    force_force = cfd_force - 0.16 * np.sin(phase - 0.4)

    frequency_y = 0.41 * np.sin(1.002 * phase + 0.06)
    frequency_force = 1.03 * np.sin(1.002 * phase + 0.64)

    combined_y = 0.415 * np.sin(1.0005 * phase + 0.07) + 0.018 * np.sin(2.0 * phase)
    combined_force = 1.10 * np.sin(1.0005 * phase + 0.69) + 0.10 * np.sin(2.0 * phase - 0.1)

    standalone_y = 0.40 * np.sin(0.998 * phase + 0.12) + 0.014 * np.sin(2.0 * phase + 0.5)
    standalone_force = 1.02 * np.sin(0.998 * phase + 0.80) + 0.08 * np.sin(2.0 * phase + 0.15)
    natural_frequency_hz = np.asarray([1.0], dtype=float)

    return {
        "cfd": {
            "label": "CFD reference",
            "time": time,
            "cycles": cycles,
            "natural_frequency_hz": natural_frequency_hz,
            "y_over_D": cfd_y,
            "C_F": cfd_force,
        },
        "vivana": {
            "label": "VIVANA-TD baseline",
            "time": time,
            "cycles": cycles,
            "natural_frequency_hz": natural_frequency_hz,
            "y_over_D": vivana_y,
            "C_F": vivana_force,
        },
        "force": {
            "label": "Force correction",
            "time": time,
            "cycles": cycles,
            "natural_frequency_hz": natural_frequency_hz,
            "y_over_D": force_y,
            "C_F": force_force,
            "delta_C_F": 0.10 * np.sin(phase - 0.4) + 0.025 * np.sin(2.0 * phase),
        },
        "frequency": {
            "label": "Frequency correction",
            "time": time,
            "cycles": cycles,
            "natural_frequency_hz": natural_frequency_hz,
            "y_over_D": frequency_y,
            "C_F": frequency_force,
            "delta_freq": 0.018 * np.sin(0.2 * phase) + 0.006 * np.cos(phase),
        },
        "combined": {
            "label": "Combined correction",
            "time": time,
            "cycles": cycles,
            "natural_frequency_hz": natural_frequency_hz,
            "y_over_D": combined_y,
            "C_F": combined_force,
            "delta_C_F": 0.075 * np.sin(phase - 0.25) + 0.015 * np.sin(2.0 * phase),
            "delta_freq": 0.012 * np.sin(0.2 * phase + 0.3) + 0.004 * np.cos(phase),
        },
        "standalone": {
            "label": "Standalone model",
            "time": time,
            "cycles": cycles,
            "natural_frequency_hz": natural_frequency_hz,
            "y_over_D": standalone_y,
            "C_F": standalone_force,
        },
    }


def _parse_example_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate thesis-style rollout time-series and spectrum plots. By default this uses the "
            "top-of-file PLOT_MODE configuration."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("rollout_timeseries", "tau_sigma_sensitivity"),
        default=None,
        help="Override the top-of-file PLOT_MODE configuration for this run.",
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Use synthetic dummy data instead of the real Block 9/10 CFD/model rollouts.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_REAL_DATASET_ROOT,
        help="Dataset root for real CFD NPZ files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where PDF/PNG files are written.",
    )
    parser.add_argument(
        "--ur",
        type=float,
        default=None,
        help=(
            "Raw reduced velocity to plot. Defaults to 10 for response/correction "
            "plots and 4 for --tau-sigma-sensitivity."
        ),
    )
    parser.add_argument(
        "--case-index",
        type=int,
        default=DEFAULT_CASE_INDEX,
        help="Index among matching CFD cases at the selected raw U_r.",
    )
    parser.add_argument(
        "--reduction-factor",
        type=int,
        default=BLOCK_10_VALIDATION_REDUCTION_FACTOR,
        help="Validation downsampling factor. Block 10 uses 20.",
    )
    parser.add_argument(
        "--cycle-start",
        type=float,
        default=DEFAULT_CYCLE_START,
        help="Start of the plotted cycle window.",
    )
    parser.add_argument(
        "--cycle-end",
        type=float,
        default=DEFAULT_CYCLE_END,
        help="End of the plotted cycle window.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device used when loading model checkpoints.",
    )
    parser.add_argument(
        "--tau-sigma-sensitivity",
        action="store_true",
        help="Compatibility shortcut for --mode tau_sigma_sensitivity.",
    )
    parser.add_argument(
        "--tau-values",
        type=float,
        nargs="+",
        default=DEFAULT_TAU_OVER_TREF_VALUES,
        help="tau/T_ref values for --tau-sigma-sensitivity.",
    )
    parser.add_argument(
        "--response-spectrum-xlim",
        type=float,
        nargs=2,
        default=DEFAULT_RESPONSE_SPECTRUM_F_OVER_FN_LIMITS,
        metavar=("LOW", "HIGH"),
        help="Frequency-ratio x-limits for displacement/force spectra.",
    )
    parser.add_argument(
        "--correction-spectrum-xlim",
        type=float,
        nargs=2,
        default=DEFAULT_CORRECTION_SPECTRUM_F_OVER_FN_LIMITS,
        metavar=("LOW", "HIGH"),
        help="Frequency-ratio x-limits for correction spectra.",
    )
    parser.add_argument(
        "--spectrum-y-mode",
        choices=("amplitude", "power"),
        default=DEFAULT_SPECTRUM_Y_MODE,
        help="Plot amplitude spectra sqrt(PSD) or power spectra PSD.",
    )
    parser.add_argument(
        "--spectrum-zero-pad-factor",
        type=int,
        default=DEFAULT_SPECTRUM_ZERO_PAD_FACTOR,
        help="FFT zero-padding factor for smoother spectrum curves. Does not improve true frequency resolution.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable terminal progress bars.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_example_args()
    mode = str(args.mode or PLOT_MODE).strip().lower().replace("-", "_")
    if bool(args.tau_sigma_sensitivity):
        mode = "tau_sigma_sensitivity"
    if mode not in {"rollout_timeseries", "tau_sigma_sensitivity"}:
        raise SystemExit(
            "PLOT_MODE must be one of: 'rollout_timeseries', 'tau_sigma_sensitivity'. "
            f"Got {PLOT_MODE!r}."
        )

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = DEFAULT_DUMMY_OUTPUT_DIR if args.dummy else DEFAULT_OUTPUT_DIR
    selected_ur = (
        DEFAULT_TAU_SIGMA_UR
        if args.ur is None and mode == "tau_sigma_sensitivity"
        else (DEFAULT_REAL_UR if args.ur is None else float(args.ur))
    )

    progress = not bool(args.no_progress)
    if mode == "tau_sigma_sensitivity":
        if args.dummy:
            raise SystemExit("PLOT_MODE='tau_sigma_sensitivity' requires real CFD NPZ data; do not combine it with --dummy.")
        rollouts, metadata, styles = build_real_tau_sigma_sensitivity_rollouts(
            ur=float(selected_ur),
            case_index=int(args.case_index),
            dataset_root=args.dataset_root,
            validation_reduction_factor=int(args.reduction_factor),
            tau_over_tref_values=args.tau_values,
            progress=progress,
        )
        sigma_fig, _ = plot_tau_sigma_sensitivity_timeseries(
            rollouts,
            ur=selected_ur,
            output_dir=output_dir,
            cycle_start=args.cycle_start,
            cycle_end=args.cycle_end,
            styles=styles,
            progress=progress,
        )
        saved_paths = list(getattr(sigma_fig, "_saved_paths", []))

        print("Generated real VIVANA-TD tau/T_ref sigma sensitivity figure:")
        selected_path = metadata.get("selected_path")
        if selected_path is not None:
            print(f"  CFD case: {selected_path}")
        print(
            f"  raw U_r={metadata.get('raw_ur', float('nan')):g}, "
            f"effective U_r={metadata.get('effective_ur', float('nan')):g}, "
            f"cycle frequency={metadata.get('dominant_frequency_hz', float('nan')):g} Hz, "
            f"natural frequency={metadata.get('natural_frequency_hz', float('nan')):g} Hz"
        )
        print(f"  validation reduction factor: {metadata.get('validation_reduction_factor')}")
        print(
            "  tau/T_ref values: "
            + ", ".join(f"{value:g}" for value in metadata.get("tau_over_tref_values", ()))
        )
        print("  saved figures:")
        for path in saved_paths:
            print(f"  {path}")
        plt.close("all")
        return

    if args.dummy:
        rollouts = make_dummy_rollouts()
        metadata = {
            "selected_path": None,
            "raw_ur": float(selected_ur),
            "effective_ur": float("nan"),
            "dominant_frequency_hz": 1.0,
            "natural_frequency_hz": 1.0,
        }
        metadata.update(_spectrum_resolution_metadata(np.asarray(rollouts["cfd"]["time"], dtype=float), 1.0))
    else:
        rollouts, metadata = build_real_block9_10_rollouts(
            ur=float(selected_ur),
            case_index=int(args.case_index),
            dataset_root=args.dataset_root,
            validation_reduction_factor=int(args.reduction_factor),
            device=str(args.device),
            progress=progress,
        )

    timeseries_cycle_start = float(args.cycle_start)
    timeseries_cycle_end = float(args.cycle_end)
    timeseries_cycle_origin = 0.0
    standalone_start_cycle = metadata.get("standalone_response_start_cycle")
    if standalone_start_cycle is not None and np.isfinite(float(standalone_start_cycle)):
        timeseries_cycle_start = float(standalone_start_cycle)
        timeseries_cycle_end = timeseries_cycle_start + float(TIMESERIES_CYCLES_AFTER_ENCODING)
        timeseries_cycle_origin = timeseries_cycle_start

    response_fig, _ = plot_rollout_response_timeseries(
        rollouts,
        ur=selected_ur,
        output_dir=output_dir,
        cycle_start=timeseries_cycle_start,
        cycle_end=timeseries_cycle_end,
        x_origin=timeseries_cycle_origin,
        progress=progress,
    )
    corrections_fig, _ = plot_rollout_corrections_timeseries(
        rollouts,
        ur=selected_ur,
        output_dir=output_dir,
        cycle_start=timeseries_cycle_start,
        cycle_end=timeseries_cycle_end,
        x_origin=timeseries_cycle_origin,
        progress=progress,
    )
    response_spectra_fig, _ = plot_rollout_response_spectra(
        rollouts,
        ur=selected_ur,
        output_dir=output_dir,
        cycle_start=args.cycle_start,
        cycle_end=args.cycle_end,
        frequency_ratio_xlim=tuple(args.response_spectrum_xlim) if args.response_spectrum_xlim is not None else None,
        spectrum_y_mode=str(args.spectrum_y_mode),
        spectrum_zero_pad_factor=int(args.spectrum_zero_pad_factor),
        progress=progress,
    )
    corrections_spectra_fig, _ = plot_rollout_corrections_spectra(
        rollouts,
        ur=selected_ur,
        output_dir=output_dir,
        cycle_start=args.cycle_start,
        cycle_end=args.cycle_end,
        frequency_ratio_xlim=tuple(args.correction_spectrum_xlim) if args.correction_spectrum_xlim is not None else None,
        spectrum_y_mode=str(args.spectrum_y_mode),
        spectrum_zero_pad_factor=int(args.spectrum_zero_pad_factor),
        progress=progress,
    )

    saved_paths = []
    saved_paths.extend(getattr(response_fig, "_saved_paths", []))
    saved_paths.extend(getattr(corrections_fig, "_saved_paths", []))
    saved_paths.extend(getattr(response_spectra_fig, "_saved_paths", []))
    saved_paths.extend(getattr(corrections_spectra_fig, "_saved_paths", []))

    if args.dummy:
        print("Generated dummy rollout time-series and spectrum figures:")
    else:
        print("Generated real Block 9/10 rollout time-series and spectrum figures:")
        selected_path = metadata.get("selected_path")
        if selected_path is not None:
            print(f"  CFD case: {selected_path}")
        print(
            f"  raw U_r={metadata.get('raw_ur', float('nan')):g}, "
            f"effective U_r={metadata.get('effective_ur', float('nan')):g}, "
            f"cycle frequency={metadata.get('dominant_frequency_hz', float('nan')):g} Hz, "
            f"natural frequency={metadata.get('natural_frequency_hz', float('nan')):g} Hz"
        )
        print(f"  validation reduction factor: {metadata.get('validation_reduction_factor')}")
        if metadata.get("standalone_response_start_cycle") is not None:
            print(
                f"  time-series cycle window: {timeseries_cycle_start:g} to "
                f"{timeseries_cycle_end:g} (after standalone encoding window)"
            )
        print(
            f"  spectrum duration={metadata.get('spectrum_duration_s', float('nan')):g} s, "
            f"df={metadata.get('spectrum_df_hz', float('nan')):g} Hz "
            f"({metadata.get('spectrum_df_over_fn', float('nan')):g} omega/omega_n)"
        )
        print("  model checkpoints:")
        for path in metadata.get("model_paths", []):
            print(f"    {path}")
        print("  saved figures:")
    for path in saved_paths:
        print(f"  {path}")

    plt.close("all")


if __name__ == "__main__":
    main()
