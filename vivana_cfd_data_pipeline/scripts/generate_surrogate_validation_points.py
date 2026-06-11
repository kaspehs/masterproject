"""Generate interpolation-based surrogate validation targets.

This utility builds reduced-velocity target points from existing CFD/TD .npz
exports. For each real anchor case it extracts the quantities used by the
aggregate rollout metrics:

  - displacement standard deviation
  - force standard deviation
  - dominant displacement frequency
  - dominant force frequency

It then interpolates these quantities between available reduced velocities and
also interpolates a minimal set of credible rollout initial-condition fields.

The generated points are surrogate/model-selection targets, not independent
validation data.
"""

from __future__ import annotations

import csv
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

try:
    from scipy.interpolate import Akima1DInterpolator, CubicSpline, PchipInterpolator, UnivariateSpline, make_interp_spline
except ImportError:  # pragma: no cover - scipy is optional
    Akima1DInterpolator = None
    CubicSpline = None
    PchipInterpolator = None
    UnivariateSpline = None
    make_interp_spline = None

_CACHE_ROOT = Path(tempfile.gettempdir()) / "masterproject_surrogate_validation_cache"
(_CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
(_CACHE_ROOT / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))


# ---------------------------------------------------------------------------
# Config for direct script use
# ---------------------------------------------------------------------------

INPUT_NPZ_GLOB = "vivana_cfd_data_pipeline/generated/cfd_npz_exports/*.npz"
OUTPUT_NPZ = Path("vivana_cfd_data_pipeline/outputs/analysis/surrogate_validation_points.npz")
OUTPUT_CSV = Path("vivana_cfd_data_pipeline/outputs/analysis/surrogate_validation_points.csv")
OUTPUT_PLOT = Path("vivana_cfd_data_pipeline/outputs/analysis/surrogate_validation_points_diagnostic.pdf")

# "label" usually gives the intended CFD case label, while "computed" uses the
# physically computed U_r stored in the export.
UR_SOURCE = "label"  # "label" | "computed" | "stored"
INTERPOLATION_UR_SOURCE = "computed"  # effective U_r with Ca=1 for these exports

# Metric targets can be computed from the current input file, or from the
# original full CFD export named by source_case_name. Use the latter when the
# input files have been burn-in trimmed but metric targets should describe the
# full pre-trim time series.
METRIC_TIMESERIES_SOURCE = "full_source"  # "full_source" | "current"
FULL_TIMESERIES_NPZ_DIR = Path("vivana_cfd_data_pipeline/generated/cfd_npz_exports")
METRIC_STD_USE_NONDIMENSIONAL = True
ROLLOUT_DISCARD_SECONDS = 50.0

# Leave empty to generate evenly spaced points inside every adjacent real-U_r
# interval. If set, these exact reduced velocities are generated instead.
TARGET_URS: Sequence[float] | None = None
POINTS_PER_INTERVAL = 1

# Exact interpolation keeps every real anchor fixed. "makima" is a modified
# Akima interpolator: smoother than PCHIP in many cases, but still anchor-exact.
INTERPOLATION_KIND = "pchip"  # "quadratic" | "makima" | "akima" | "pchip" | "cubic" | "linear" | "smoothing_spline"
SMOOTHING_STRENGTH = 0.05
INCLUDE_ANCHOR_POINTS = False

# Direct script default: no protected reduced velocities. Final-dataset builds
# should pass exclusions from build_final_training_dataset.py so the surrogate
# anchors stay coupled to the actual train/val_seen exclusions.
EXCLUDE_URS: Sequence[float] = ()
EXCLUDE_UR_ATOL = 1.0e-8

MAKE_DIAGNOSTIC_PLOT = True
SHOW_DIAGNOSTIC_PLOT = False
PLOT_DPI = 300


METRIC_KEYS = (
    "disp_std",
    "force_std",
    "disp_dominant_frequency_hz",
    "force_dominant_frequency_hz",
)

IC_SCALAR_KEYS = (
    "y0",
    "dy0",
    "ddy0",
    "force0",
    "force_td0",
    "sig_dy_td0",
    "sig_ddy_td0",
    "fhat_td0",
    "omega_vy_td0",
    "flow_speed0",
    "dt",
    "eval_duration_s",
    "rollout_duration_s",
    "diameter_m",
    "stiffness_n_m",
    "effective_mass_kg",
    "dry_mass_kg",
    "damping_c",
)

IC_ANGLE_KEYS = (
    "phi_td0",
    "theta_td0",
)

NONNEGATIVE_IC_SCALAR_KEYS = {
    "sig_dy_td0",
    "sig_ddy_td0",
    "fhat_td0",
    "omega_vy_td0",
    "flow_speed0",
    "dt",
    "eval_duration_s",
    "rollout_duration_s",
    "diameter_m",
    "stiffness_n_m",
    "effective_mass_kg",
    "dry_mass_kg",
}


@dataclass(frozen=True)
class AnchorPoint:
    path: str
    ur: float
    ur_label: float
    metric_source_path: str
    metrics: dict[str, float]
    ic_scalars: dict[str, float]
    ic_angles: dict[str, float]


def _first_present(data: Mapping[str, np.ndarray], keys: Sequence[str], *, path: Path) -> np.ndarray:
    for key in keys:
        if key in data:
            return np.asarray(data[key])
    raise KeyError(f"{path} is missing all of: {', '.join(keys)}")


def _optional_first_present(data: Mapping[str, np.ndarray], keys: Sequence[str]) -> np.ndarray | None:
    for key in keys:
        if key in data:
            return np.asarray(data[key])
    return None


def _finite_scalar(data: Mapping[str, np.ndarray], keys: Sequence[str], *, default: float = np.nan) -> float:
    for key in keys:
        if key not in data:
            continue
        values = np.asarray(data[key], dtype=float).reshape(-1)
        if values.size and np.isfinite(values[0]):
            return float(values[0])
    return float(default)


def _series_first(data: Mapping[str, np.ndarray], keys: Sequence[str], *, default: float = np.nan) -> float:
    arr = _optional_first_present(data, keys)
    if arr is None:
        return float(default)
    values = np.asarray(arr, dtype=float).reshape(-1)
    if values.size == 0:
        return float(default)
    return float(values[0])


def _string_first(data: Mapping[str, np.ndarray], keys: Sequence[str]) -> str | None:
    for key in keys:
        if key not in data:
            continue
        values = np.asarray(data[key]).reshape(-1)
        if values.size == 0:
            continue
        value = str(values[0]).strip()
        if value:
            return value
    return None


def _force_scale_inverse(data: Mapping[str, np.ndarray]) -> float:
    multiplier = _finite_scalar(data, ("force_scale",), default=np.nan)
    if np.isfinite(multiplier) and multiplier > 0.0:
        return float(multiplier)

    rho = _finite_scalar(data, ("python_rho_kg_m3", "rho_kg_m3"), default=np.nan)
    diameter = _finite_scalar(data, ("python_diameter_m", "diameter_m"), default=np.nan)
    flow_speed = _finite_scalar(
        data,
        ("python_flow_speed_m_s", "model_flow_speed_m_s", "training_flow_speed_m_s", "flow_speed_m_s"),
        default=np.nan,
    )
    f0 = 0.5 * float(rho) * float(diameter) * float(flow_speed) ** 2
    if np.isfinite(f0) and f0 > 0.0:
        return float(1.0 / f0)
    return float("nan")


def _load_time_displacement_force(data: Mapping[str, np.ndarray], *, path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.asarray(
        _first_present(data, ("time_dim", "a", "time"), path=path),
        dtype=float,
    ).reshape(-1)
    if METRIC_STD_USE_NONDIMENSIONAL:
        y_raw = _optional_first_present(data, ("y_disp_nd",))
        if y_raw is None:
            y_raw = _first_present(data, ("y_disp_dim", "b", "y"), path=path)
            diameter = _finite_scalar(data, ("python_diameter_m", "diameter_m"), default=1.0)
            if not np.isfinite(diameter) or diameter <= 0.0:
                raise ValueError(f"{path} has invalid diameter for y/D normalization: {diameter!r}.")
            y = np.asarray(y_raw, dtype=float).reshape(-1) / float(diameter)
        else:
            y = np.asarray(y_raw, dtype=float).reshape(-1)

        force_raw = _optional_first_present(data, ("y_force_nd", "force_coeff", "cf_force"))
        if force_raw is None:
            force_raw = _first_present(
                data,
                ("y_force_per_m_dim", "force_per_m_dim", "c", "F_total", "force_total", "force"),
                path=path,
            )
            force_multiplier = _force_scale_inverse(data)
            if not np.isfinite(force_multiplier) or force_multiplier <= 0.0:
                raise ValueError(f"{path} has invalid force scale for F/F0 normalization.")
            force = np.asarray(force_raw, dtype=float).reshape(-1) * float(force_multiplier)
        else:
            force = np.asarray(force_raw, dtype=float).reshape(-1)
    else:
        y = np.asarray(
            _first_present(data, ("y_disp_dim", "b", "y"), path=path),
            dtype=float,
        ).reshape(-1)
        force = np.asarray(
            _first_present(data, ("y_force_per_m_dim", "force_per_m_dim", "c", "F_total", "force_total", "force"), path=path),
            dtype=float,
        ).reshape(-1)
    n = min(t.size, y.size, force.size)
    if n < 4:
        raise ValueError(f"{path} is too short to compute surrogate validation metrics.")
    return t[:n], y[:n], force[:n]


def _resolve_metric_source_path(data: Mapping[str, np.ndarray], *, current_path: Path) -> Path:
    source_mode = str(METRIC_TIMESERIES_SOURCE).strip().lower()
    if source_mode == "current":
        return Path(current_path)
    if source_mode != "full_source":
        raise ValueError("METRIC_TIMESERIES_SOURCE must be one of: full_source, current.")

    source_case_name = _string_first(data, ("source_case_name",))
    if source_case_name is None:
        return Path(current_path)
    source_path = Path(FULL_TIMESERIES_NPZ_DIR) / f"{source_case_name}.npz"
    if not source_path.exists():
        raise FileNotFoundError(
            f"{current_path} points to source_case_name={source_case_name!r}, "
            f"but full metric source file does not exist: {source_path}"
        )
    return source_path


def _dominant_frequency(signal: np.ndarray, dt: float) -> float:
    """FFT peak frequency with the same basic behavior as training_utils."""
    if not np.isfinite(dt) or dt <= 0.0:
        return float("nan")
    signal = np.asarray(signal, dtype=float).reshape(-1)
    if signal.size < 4:
        return float("nan")
    centered = signal - float(np.mean(signal))
    if np.allclose(centered, 0.0):
        return float("nan")

    freqs = np.fft.rfftfreq(int(centered.size), d=float(dt))
    power = np.abs(np.fft.rfft(centered)) ** 2
    if freqs.size < 2 or power.size < 2:
        return float("nan")
    power[0] = 0.0
    mask = np.isfinite(freqs) & np.isfinite(power) & (freqs > 0.0)
    if not np.any(mask):
        return float("nan")

    valid_indices = np.flatnonzero(mask)
    peak_index = int(valid_indices[int(np.argmax(power[mask]))])
    interp_index = float(peak_index)
    if 1 <= peak_index < power.size - 1:
        y_prev = float(power[peak_index - 1])
        y_peak = float(power[peak_index])
        y_next = float(power[peak_index + 1])
        denom = y_prev - 2.0 * y_peak + y_next
        if np.isfinite(denom) and abs(denom) > 1.0e-18:
            delta = 0.5 * (y_prev - y_next) / denom
            if np.isfinite(delta):
                interp_index += float(np.clip(delta, -1.0, 1.0))

    df = float(freqs[1] - freqs[0])
    if not np.isfinite(df) or df <= 0.0:
        return float(freqs[peak_index])
    return float(max(interp_index * df, 0.0))


def _reduced_velocity(data: Mapping[str, np.ndarray], source: str, *, path: Path) -> float:
    source_key = str(source).strip().lower()
    if source_key == "label":
        keys = ("U_r_label_scalar", "label_ur", "U_r_label_series")
    elif source_key == "computed":
        keys = ("U_r_computed_scalar", "computed_ur", "U_r_computed_series")
    elif source_key == "stored":
        keys = ("U_r",)
    else:
        raise ValueError("ur_source must be one of: label, computed, stored.")
    values = np.asarray(_first_present(data, keys, path=path), dtype=float).reshape(-1)
    if values.size == 0 or not np.isfinite(values[0]):
        raise ValueError(f"{path} has no finite reduced velocity for source={source!r}.")
    return float(values[0])


def load_reduced_velocity(path: Path, *, ur_source: str = UR_SOURCE) -> float:
    """Read only the reduced velocity needed for include/exclude decisions."""
    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        return _reduced_velocity(data, ur_source, path=path)


def load_anchor_point(path: Path, *, ur_source: str = INTERPOLATION_UR_SOURCE) -> AnchorPoint:
    """Extract metric anchors and rollout initial-condition fields from one .npz."""
    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        metric_source_path = _resolve_metric_source_path(data, current_path=path)
        if metric_source_path == path:
            metric_t, metric_y, metric_force = _load_time_displacement_force(data, path=path)
        else:
            with np.load(metric_source_path, allow_pickle=True) as metric_data:
                metric_t, metric_y, metric_force = _load_time_displacement_force(metric_data, path=metric_source_path)

        t_ic = np.asarray(
            _first_present(data, ("time_dim", "a", "time"), path=path),
            dtype=float,
        ).reshape(-1)
        y_ic = np.asarray(
            _first_present(data, ("y_disp_dim", "b", "y"), path=path),
            dtype=float,
        ).reshape(-1)
        dy = np.asarray(
            _first_present(data, ("y_vel_dim", "dy", "e", "v"), path=path),
            dtype=float,
        ).reshape(-1)
        ddy = np.asarray(
            _first_present(data, ("y_acc_dim", "ddy"), path=path),
            dtype=float,
        ).reshape(-1)
        force_ic = np.asarray(
            _first_present(data, ("y_force_per_m_dim", "force_per_m_dim", "c", "F_total", "force_total", "force"), path=path),
            dtype=float,
        ).reshape(-1)

        n_ic = min(t_ic.size, y_ic.size, dy.size, ddy.size, force_ic.size)
        if n_ic < 2:
            raise ValueError(f"{path} is too short to extract rollout initial conditions.")
        t_ic = t_ic[:n_ic]
        y_ic = y_ic[:n_ic]
        dy = dy[:n_ic]
        ddy = ddy[:n_ic]
        force_ic = force_ic[:n_ic]
        metric_dt = float(metric_t[1] - metric_t[0])
        if not np.isfinite(metric_dt) or metric_dt <= 0.0:
            raise ValueError(f"{metric_source_path} has invalid metric dt={metric_dt!r}.")
        metric_duration_s = float(metric_t[-1] - metric_t[0] + metric_dt)
        if not np.isfinite(metric_duration_s) or metric_duration_s <= 0.0:
            raise ValueError(f"{metric_source_path} has invalid metric duration={metric_duration_s!r}.")
        ic_dt = float(t_ic[1] - t_ic[0])
        if not np.isfinite(ic_dt) or ic_dt <= 0.0:
            raise ValueError(f"{path} has invalid IC dt={ic_dt!r}.")

        force_td0 = _series_first(data, ("F_total_td_per_m", "F_total_td"), default=np.nan)
        ur = _reduced_velocity(data, ur_source, path=path)
        ur_label = _reduced_velocity(data, "label", path=path)
        flow_speed0 = _finite_scalar(
            data,
            ("python_flow_speed_m_s", "model_flow_speed_m_s", "training_flow_speed_m_s", "flow_speed_m_s"),
            default=np.nan,
        )

        metrics = {
            "disp_std": float(np.std(metric_y)),
            "force_std": float(np.std(metric_force)),
            "disp_dominant_frequency_hz": _dominant_frequency(metric_y, metric_dt),
            "force_dominant_frequency_hz": _dominant_frequency(metric_force, metric_dt),
        }
        ic_scalars = {
            "y0": float(y_ic[0]),
            "dy0": float(dy[0]),
            "ddy0": float(ddy[0]),
            "force0": float(force_ic[0]),
            "force_td0": float(force_td0),
            "sig_dy_td0": _series_first(data, ("sig_dy_loc_td", "sig_dy_td"), default=np.nan),
            "sig_ddy_td0": _series_first(data, ("sig_ddy_loc_td", "sig_ddy_td"), default=np.nan),
            "fhat_td0": _series_first(data, ("fhat_td",), default=np.nan),
            "omega_vy_td0": _series_first(data, ("omega_vy_td",), default=np.nan),
            "flow_speed0": float(flow_speed0),
            "dt": float(ic_dt),
            "eval_duration_s": float(metric_duration_s),
            "rollout_duration_s": float(metric_duration_s + float(ROLLOUT_DISCARD_SECONDS)),
            "diameter_m": _finite_scalar(data, ("python_diameter_m", "diameter_m"), default=1.0),
            "stiffness_n_m": _finite_scalar(data, ("python_stiffness_n_m", "model_stiffness_n_m", "training_stiffness_n_m", "stiffness_n_m")),
            "effective_mass_kg": _finite_scalar(data, ("python_effective_mass_kg", "model_effective_mass_kg", "training_effective_mass_kg", "effective_mass_kg")),
            "dry_mass_kg": _finite_scalar(data, ("python_dry_mass_kg", "python_mass_kg", "model_dry_mass_kg", "training_dry_mass_kg", "dry_mass_kg")),
            "damping_c": _finite_scalar(data, ("python_damping_c", "model_damping_c", "training_damping_c", "damping_c"), default=0.0),
        }
        ic_angles = {
            "phi_td0": _series_first(data, ("phi_vy_td", "phi_td"), default=np.nan),
            "theta_td0": _series_first(data, ("theta_td",), default=np.nan),
        }

    return AnchorPoint(
        path=str(path),
        ur=float(ur),
        ur_label=float(ur_label),
        metric_source_path=str(metric_source_path),
        metrics=metrics,
        ic_scalars=ic_scalars,
        ic_angles=ic_angles,
    )


def collect_anchor_points(
    paths: Iterable[Path],
    *,
    ur_source: str = UR_SOURCE,
    exclude_urs: Sequence[float] = EXCLUDE_URS,
    exclude_ur_atol: float = EXCLUDE_UR_ATOL,
) -> list[AnchorPoint]:
    """Load all anchor points, optionally excluding protected reduced velocities."""
    anchors: list[AnchorPoint] = []
    exclude = np.asarray(list(exclude_urs), dtype=float).reshape(-1)
    for path in sorted(Path(p) for p in paths):
        ur = load_reduced_velocity(path, ur_source=ur_source)
        if exclude.size and np.any(np.isclose(ur, exclude, rtol=0.0, atol=float(exclude_ur_atol))):
            continue
        anchor = load_anchor_point(path, ur_source=INTERPOLATION_UR_SOURCE)
        anchors.append(anchor)
    if len(anchors) < 2:
        raise ValueError("At least two anchor reduced velocities are required for interpolation.")
    return anchors


def _circular_mean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    s = float(np.mean(np.sin(arr)))
    c = float(np.mean(np.cos(arr)))
    if math_hypot(s, c) < 1.0e-8:
        return float(arr[0])
    return float(np.arctan2(s, c))


def math_hypot(a: float, b: float) -> float:
    return float(np.sqrt(float(a) * float(a) + float(b) * float(b)))


def _nanmean_or_nan(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def _group_anchor_points_by_ur(anchors: Sequence[AnchorPoint]) -> dict[str, np.ndarray]:
    urs = sorted({float(anchor.ur) for anchor in anchors})
    grouped: dict[str, list[float] | list[str]] = {
        "ur": [],
        "ur_label": [],
        "anchor_count": [],
        "source_paths": [],
        "metric_source_paths": [],
    }
    for key in METRIC_KEYS:
        grouped[key] = []
    for key in IC_SCALAR_KEYS:
        grouped[f"ic_{key}"] = []
    for key in IC_ANGLE_KEYS:
        grouped[f"ic_{key}"] = []

    for ur in urs:
        group = [anchor for anchor in anchors if float(anchor.ur) == float(ur)]
        grouped["ur"].append(float(ur))
        grouped["ur_label"].append(_nanmean_or_nan([anchor.ur_label for anchor in group]))
        grouped["anchor_count"].append(float(len(group)))
        grouped["source_paths"].append(";".join(anchor.path for anchor in group))
        grouped["metric_source_paths"].append(";".join(anchor.metric_source_path for anchor in group))

        for key in METRIC_KEYS:
            grouped[key].append(_nanmean_or_nan([anchor.metrics[key] for anchor in group]))
        for key in IC_SCALAR_KEYS:
            grouped[f"ic_{key}"].append(_nanmean_or_nan([anchor.ic_scalars[key] for anchor in group]))
        for key in IC_ANGLE_KEYS:
            grouped[f"ic_{key}"].append(_circular_mean([anchor.ic_angles[key] for anchor in group]))

    out: dict[str, np.ndarray] = {}
    for key, values in grouped.items():
        if key in {"source_paths", "metric_source_paths"}:
            out[key] = np.asarray(values, dtype=str)
        else:
            out[key] = np.asarray(values, dtype=float)
    return out


def _smoothing_parameter(y: np.ndarray, smoothing_strength: float) -> float:
    if smoothing_strength <= 0.0:
        return 0.0
    variance = float(np.nanvar(np.asarray(y, dtype=float)))
    if not np.isfinite(variance) or variance <= 0.0:
        return 0.0
    return float(smoothing_strength) * float(y.size) * variance


def _make_interpolator(
    x: np.ndarray,
    y: np.ndarray,
    kind: str,
    *,
    smoothing_strength: float = SMOOTHING_STRENGTH,
):
    finite = np.isfinite(x) & np.isfinite(y)
    x_f = np.asarray(x[finite], dtype=float)
    y_f = np.asarray(y[finite], dtype=float)
    if x_f.size < 2:
        return lambda x_new: np.full_like(np.asarray(x_new, dtype=float), np.nan, dtype=float)
    order = np.argsort(x_f)
    x_f = x_f[order]
    y_f = y_f[order]

    kind_key = str(kind).strip().lower()
    if kind_key in {"akima", "makima"} and Akima1DInterpolator is not None and x_f.size >= 3:
        interp = Akima1DInterpolator(x_f, y_f, method=kind_key)

        def _interp_akima(x_new):
            x_new_arr = np.asarray(x_new, dtype=float)
            values = np.asarray(interp(x_new_arr), dtype=float)
            outside = (x_new_arr < float(x_f[0])) | (x_new_arr > float(x_f[-1]))
            values = np.asarray(values, dtype=float)
            values[outside] = np.nan
            return values

        return _interp_akima
    if kind_key in {"smoothing_spline", "smooth_spline", "smooth"} and UnivariateSpline is not None and x_f.size >= 3:
        k = min(3, int(x_f.size) - 1)
        spline = UnivariateSpline(
            x_f,
            y_f,
            k=k,
            s=_smoothing_parameter(y_f, float(smoothing_strength)),
            ext=0,
        )

        def _interp_smooth(x_new):
            x_new_arr = np.asarray(x_new, dtype=float)
            values = np.asarray(spline(x_new_arr), dtype=float)
            outside = (x_new_arr < float(x_f[0])) | (x_new_arr > float(x_f[-1]))
            values = np.asarray(values, dtype=float)
            values[outside] = np.nan
            return values

        return _interp_smooth
    if kind_key == "pchip" and PchipInterpolator is not None:
        interp = PchipInterpolator(x_f, y_f, extrapolate=False)
        return lambda x_new: np.asarray(interp(x_new), dtype=float)
    if kind_key in {"quadratic", "square"} and make_interp_spline is not None and x_f.size >= 3:
        interp = make_interp_spline(x_f, y_f, k=2)

        def _interp_quadratic(x_new):
            x_new_arr = np.asarray(x_new, dtype=float)
            values = np.asarray(interp(x_new_arr), dtype=float)
            outside = (x_new_arr < float(x_f[0])) | (x_new_arr > float(x_f[-1]))
            values = np.asarray(values, dtype=float)
            values[outside] = np.nan
            return values

        return _interp_quadratic
    if kind_key == "cubic" and CubicSpline is not None and x_f.size >= 3:
        interp = CubicSpline(x_f, y_f, bc_type="natural", extrapolate=False)
        return lambda x_new: np.asarray(interp(x_new), dtype=float)
    if kind_key not in {"quadratic", "square", "makima", "akima", "smoothing_spline", "smooth_spline", "smooth", "pchip", "cubic", "linear"}:
        raise ValueError("interpolation_kind must be one of: quadratic, makima, akima, pchip, cubic, linear, smoothing_spline.")
    return lambda x_new: np.interp(np.asarray(x_new, dtype=float), x_f, y_f, left=np.nan, right=np.nan)


def _clip_nonnegative(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.where(np.isfinite(values), np.maximum(values, 0.0), values)


def _natural_frequency_hz(stiffness_n_m: np.ndarray, mass_kg: np.ndarray) -> np.ndarray:
    stiffness = np.asarray(stiffness_n_m, dtype=float)
    mass = np.asarray(mass_kg, dtype=float)
    out = np.full(np.broadcast_shapes(stiffness.shape, mass.shape), np.nan, dtype=float)
    stiffness_b = np.broadcast_to(stiffness, out.shape)
    mass_b = np.broadcast_to(mass, out.shape)
    valid = np.isfinite(stiffness_b) & np.isfinite(mass_b) & (stiffness_b > 0.0) & (mass_b > 0.0)
    out[valid] = np.sqrt(stiffness_b[valid] / mass_b[valid]) / (2.0 * np.pi)
    return out


def _stiffness_from_reduced_velocity(
    reduced_velocity: np.ndarray,
    *,
    effective_mass_kg: np.ndarray,
    flow_speed_m_s: np.ndarray,
    diameter_m: np.ndarray,
) -> np.ndarray:
    ur = np.asarray(reduced_velocity, dtype=float)
    mass = np.asarray(effective_mass_kg, dtype=float)
    flow_speed = np.asarray(flow_speed_m_s, dtype=float)
    diameter = np.asarray(diameter_m, dtype=float)
    shape = np.broadcast_shapes(ur.shape, mass.shape, flow_speed.shape, diameter.shape)
    ur_b = np.broadcast_to(ur, shape)
    mass_b = np.broadcast_to(mass, shape)
    flow_b = np.broadcast_to(flow_speed, shape)
    diameter_b = np.broadcast_to(diameter, shape)
    out = np.full(shape, np.nan, dtype=float)
    valid = (
        np.isfinite(ur_b)
        & np.isfinite(mass_b)
        & np.isfinite(flow_b)
        & np.isfinite(diameter_b)
        & (ur_b > 0.0)
        & (mass_b > 0.0)
        & (diameter_b > 0.0)
    )
    omega_n = np.zeros(shape, dtype=float)
    omega_n[valid] = 2.0 * np.pi * flow_b[valid] / (ur_b[valid] * diameter_b[valid])
    out[valid] = mass_b[valid] * omega_n[valid] ** 2
    return out


def _default_target_urs(anchor_urs: np.ndarray, points_per_interval: int) -> np.ndarray:
    points: list[float] = []
    n_between = max(0, int(points_per_interval))
    for left, right in zip(anchor_urs[:-1], anchor_urs[1:]):
        if n_between <= 0:
            continue
        segment = np.linspace(float(left), float(right), n_between + 2, endpoint=True)[1:-1]
        points.extend(float(value) for value in segment)
    return np.asarray(points, dtype=float)


def generate_surrogate_validation_points(
    anchors: Sequence[AnchorPoint],
    *,
    target_urs: Sequence[float] | None = TARGET_URS,
    points_per_interval: int = POINTS_PER_INTERVAL,
    interpolation_kind: str = INTERPOLATION_KIND,
    smoothing_strength: float = SMOOTHING_STRENGTH,
    include_anchor_points: bool = INCLUDE_ANCHOR_POINTS,
) -> dict[str, np.ndarray]:
    """Generate surrogate target metrics and interpolated initial conditions.

    Returns a dict of NumPy arrays suitable for saving as .npz. Rows with
    ``point_kind == "synthetic"`` are fictive/interpolated reduced velocities;
    rows with ``point_kind == "anchor"`` are real reduced velocities included
    only when ``include_anchor_points`` is true.
    """
    grouped = _group_anchor_points_by_ur(anchors)
    anchor_urs = np.asarray(grouped["ur"], dtype=float)
    order = np.argsort(anchor_urs)
    anchor_urs = anchor_urs[order]
    if np.unique(anchor_urs).size < 2:
        raise ValueError("At least two unique reduced velocities are required.")
    for key, values in list(grouped.items()):
        grouped[key] = values[order] if values.shape[:1] == order.shape else values

    if target_urs is None:
        synthetic_urs = _default_target_urs(anchor_urs, points_per_interval)
    else:
        synthetic_urs = np.asarray(target_urs, dtype=float).reshape(-1)
    synthetic_urs = np.asarray(
        [ur for ur in synthetic_urs if np.min(np.abs(anchor_urs - float(ur))) > 1.0e-10],
        dtype=float,
    )

    if include_anchor_points:
        all_urs = np.concatenate([anchor_urs, synthetic_urs])
        point_kind = np.asarray(["anchor"] * anchor_urs.size + ["synthetic"] * synthetic_urs.size, dtype=str)
    else:
        all_urs = synthetic_urs
        point_kind = np.asarray(["synthetic"] * synthetic_urs.size, dtype=str)
    row_order = np.argsort(all_urs)
    all_urs = all_urs[row_order]
    point_kind = point_kind[row_order]

    out: dict[str, np.ndarray] = {
        "ur": np.asarray(all_urs, dtype=float),
        "ur_effective": np.asarray(all_urs, dtype=float),
        "point_kind": point_kind,
        "anchor_ur": np.asarray(anchor_urs, dtype=float),
        "anchor_ur_effective": np.asarray(anchor_urs, dtype=float),
        "anchor_ur_label": np.asarray(grouped["ur_label"], dtype=float),
        "anchor_count": np.asarray(grouped["anchor_count"], dtype=float),
        "anchor_source_paths": np.asarray(grouped["source_paths"], dtype=str),
        "anchor_metric_source_paths": np.asarray(grouped["metric_source_paths"], dtype=str),
        "interpolation_kind": np.asarray(str(interpolation_kind)),
        "smoothing_strength": np.asarray(float(smoothing_strength)),
    }
    ur_label_interp = _make_interpolator(
        anchor_urs,
        np.asarray(grouped["ur_label"], dtype=float),
        "linear",
    )
    out["ur_label"] = np.asarray(ur_label_interp(all_urs), dtype=float)

    for key in METRIC_KEYS:
        interp = _make_interpolator(
            anchor_urs,
            np.asarray(grouped[key], dtype=float),
            interpolation_kind,
            smoothing_strength=smoothing_strength,
        )
        out[key] = _clip_nonnegative(np.asarray(interp(all_urs), dtype=float))
        out[f"anchor_{key}"] = np.asarray(grouped[key], dtype=float)

    for key in IC_SCALAR_KEYS:
        src_key = f"ic_{key}"
        scalar_interpolation_kind = "linear" if key in {"dt", "eval_duration_s", "rollout_duration_s"} else interpolation_kind
        interp = _make_interpolator(
            anchor_urs,
            np.asarray(grouped[src_key], dtype=float),
            scalar_interpolation_kind,
            smoothing_strength=smoothing_strength,
        )
        values = np.asarray(interp(all_urs), dtype=float)
        if key in NONNEGATIVE_IC_SCALAR_KEYS:
            values = _clip_nonnegative(values)
        out[src_key] = values
        out[f"anchor_{src_key}"] = np.asarray(grouped[src_key], dtype=float)

    calculated_stiffness = _stiffness_from_reduced_velocity(
        out["ur_effective"],
        effective_mass_kg=out["ic_effective_mass_kg"],
        flow_speed_m_s=out["ic_flow_speed0"],
        diameter_m=out["ic_diameter_m"],
    )
    calculated_anchor_stiffness = _stiffness_from_reduced_velocity(
        anchor_urs,
        effective_mass_kg=np.asarray(grouped["ic_effective_mass_kg"], dtype=float),
        flow_speed_m_s=np.asarray(grouped["ic_flow_speed0"], dtype=float),
        diameter_m=np.asarray(grouped["ic_diameter_m"], dtype=float),
    )
    out["ic_stiffness_n_m"] = calculated_stiffness
    out["anchor_ic_stiffness_n_m"] = calculated_anchor_stiffness
    out["rollout_discard_seconds"] = np.full_like(out["ur_effective"], float(ROLLOUT_DISCARD_SECONDS), dtype=float)
    out["rollout_steps"] = np.ceil(out["ic_rollout_duration_s"] / np.clip(out["ic_dt"], 1.0e-12, None)).astype(np.int64)
    out["eval_steps_after_discard"] = np.ceil(out["ic_eval_duration_s"] / np.clip(out["ic_dt"], 1.0e-12, None)).astype(np.int64)

    for key in IC_ANGLE_KEYS:
        src_key = f"ic_{key}"
        angles = np.asarray(grouped[src_key], dtype=float)
        sin_interp = _make_interpolator(
            anchor_urs,
            np.sin(angles),
            interpolation_kind,
            smoothing_strength=smoothing_strength,
        )
        cos_interp = _make_interpolator(
            anchor_urs,
            np.cos(angles),
            interpolation_kind,
            smoothing_strength=smoothing_strength,
        )
        sin_val = np.asarray(sin_interp(all_urs), dtype=float)
        cos_val = np.asarray(cos_interp(all_urs), dtype=float)
        out[src_key] = np.arctan2(sin_val, cos_val)
        out[f"anchor_{src_key}"] = angles

    out["td_context0"] = np.stack(
        [
            out["ic_ddy0"],
            out["ic_phi_td0"],
            out["ic_sig_dy_td0"],
            out["ic_sig_ddy_td0"],
            out["ic_flow_speed0"],
        ],
        axis=1,
    )
    out["z0_effective_mass"] = np.stack(
        [out["ic_y0"], out["ic_dy0"] * out["ic_effective_mass_kg"]],
        axis=1,
    )
    out["z0_dry_mass"] = np.stack(
        [out["ic_y0"], out["ic_dy0"] * out["ic_dry_mass_kg"]],
        axis=1,
    )
    return out


def save_surrogate_points(points: Mapping[str, np.ndarray], *, npz_path: Path, csv_path: Path | None = None) -> None:
    npz_path = Path(npz_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **points)

    if csv_path is None:
        return
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    scalar_keys = [
        "ur",
        "ur_effective",
        "ur_label",
        "point_kind",
        "rollout_discard_seconds",
        "rollout_steps",
        "eval_steps_after_discard",
        *METRIC_KEYS,
        *(f"ic_{key}" for key in IC_SCALAR_KEYS),
        *(f"ic_{key}" for key in IC_ANGLE_KEYS),
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys)
        writer.writeheader()
        n = int(np.asarray(points["ur"]).shape[0])
        for idx in range(n):
            row = {}
            for key in scalar_keys:
                value = np.asarray(points[key])[idx]
                row[key] = str(value) if isinstance(value, np.str_) else float(value)
            writer.writerow(row)


def save_diagnostic_plot(
    points: Mapping[str, np.ndarray],
    *,
    anchors: Sequence[AnchorPoint],
    output_path: Path = OUTPUT_PLOT,
    show: bool = SHOW_DIAGNOSTIC_PLOT,
    dpi: int = PLOT_DPI,
) -> None:
    """Plot real metric anchors and synthetic targets.

    Excluded/test anchors are intentionally not plotted.
    """
    try:
        import matplotlib

        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting is optional
        print(f"Skipping diagnostic plot because matplotlib could not be imported: {exc}")
        return

    grouped = _group_anchor_points_by_ur(anchors)
    anchor_ur = np.asarray(grouped["ur"], dtype=float)
    anchor_order = np.argsort(anchor_ur)
    anchor_ur = anchor_ur[anchor_order]

    synthetic_ur = np.asarray(points["ur"], dtype=float)
    point_kind = np.asarray(points["point_kind"]).astype(str)
    interpolation_kind = str(np.asarray(points.get("interpolation_kind", INTERPOLATION_KIND)).reshape(()))
    smoothing_strength = float(np.asarray(points.get("smoothing_strength", SMOOTHING_STRENGTH)).reshape(()))

    with plt.rc_context(
        {
            "font.family": "serif",
            "font.serif": ["Latin Modern Roman", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    ):
        _save_diagnostic_plot_figure(
            plt,
            points=points,
            grouped=grouped,
            output_path=Path(output_path),
            show=show,
            dpi=dpi,
            anchor_ur=anchor_ur,
            anchor_order=anchor_order,
            synthetic_ur=synthetic_ur,
            point_kind=point_kind,
            interpolation_kind=interpolation_kind,
            smoothing_strength=smoothing_strength,
        )


def _save_diagnostic_plot_figure(
    plt,
    *,
    points: Mapping[str, np.ndarray],
    grouped: Mapping[str, Sequence[object]],
    output_path: Path,
    show: bool,
    dpi: int,
    anchor_ur: np.ndarray,
    anchor_order: np.ndarray,
    synthetic_ur: np.ndarray,
    point_kind: np.ndarray,
    interpolation_kind: str,
    smoothing_strength: float,
) -> None:
    synthetic_mask = point_kind == "synthetic"
    anchor_row_mask = point_kind == "anchor"

    metric_ylabels = {
        "disp_std": r"$\sigma_{y/D}$",
        "force_std": r"$\sigma_{C_F}$",
        "disp_dominant_frequency_hz": r"$f_y$ [Hz]",
        "force_dominant_frequency_hz": r"$f_F$ [Hz]",
    }
    normalized_frequency_ylabels = {
        "disp_dominant_frequency_hz": r"$\omega_y/\omega_n$",
        "force_dominant_frequency_hz": r"$\omega_F/\omega_n$",
    }

    fig, axes = plt.subplots(2, 2, figsize=(5.85, 4.7), sharex=True)
    axes_flat = axes.reshape(-1)
    for ax, key in zip(axes_flat, METRIC_KEYS):
        anchor_metric = np.asarray(grouped[key], dtype=float)[anchor_order]
        curve_interp = _make_interpolator(
            anchor_ur,
            anchor_metric,
            interpolation_kind,
            smoothing_strength=smoothing_strength,
        )
        anchor_y = anchor_metric
        synthetic_y = np.asarray(points[key], dtype=float)
        plot_ur = np.unique(
            np.concatenate(
                [
                    np.linspace(float(anchor_ur[0]), float(anchor_ur[-1]), 400),
                    anchor_ur,
                    synthetic_ur[synthetic_mask],
                ]
            )
        )
        plot_y = np.asarray(curve_interp(plot_ur), dtype=float)
        plot_y = _clip_nonnegative(plot_y)
        ylabel = metric_ylabels[key]
        if key in {"disp_dominant_frequency_hz", "force_dominant_frequency_hz"}:
            anchor_mass = np.asarray(grouped["ic_effective_mass_kg"], dtype=float)[anchor_order]
            anchor_flow = np.asarray(grouped["ic_flow_speed0"], dtype=float)[anchor_order]
            anchor_diameter = np.asarray(grouped["ic_diameter_m"], dtype=float)[anchor_order]
            anchor_stiffness = _stiffness_from_reduced_velocity(
                anchor_ur,
                effective_mass_kg=anchor_mass,
                flow_speed_m_s=anchor_flow,
                diameter_m=anchor_diameter,
            )
            anchor_fn = _natural_frequency_hz(anchor_stiffness, anchor_mass)

            mass_interp = _make_interpolator(
                anchor_ur,
                anchor_mass,
                interpolation_kind,
                smoothing_strength=smoothing_strength,
            )
            flow_interp = _make_interpolator(
                anchor_ur,
                anchor_flow,
                interpolation_kind,
                smoothing_strength=smoothing_strength,
            )
            diameter_interp = _make_interpolator(
                anchor_ur,
                anchor_diameter,
                interpolation_kind,
                smoothing_strength=smoothing_strength,
            )
            plot_mass = mass_interp(plot_ur)
            plot_stiffness = _stiffness_from_reduced_velocity(
                plot_ur,
                effective_mass_kg=plot_mass,
                flow_speed_m_s=flow_interp(plot_ur),
                diameter_m=diameter_interp(plot_ur),
            )
            plot_fn = _natural_frequency_hz(plot_stiffness, plot_mass)
            synthetic_fn = _natural_frequency_hz(
                np.asarray(points["ic_stiffness_n_m"], dtype=float),
                np.asarray(points["ic_effective_mass_kg"], dtype=float),
            )
            finite_plot_ur = np.isfinite(plot_ur)
            can_normalize = (
                np.all(np.isfinite(anchor_fn))
                and np.all(anchor_fn > 0.0)
                and np.all(np.isfinite(plot_fn[finite_plot_ur]))
                and np.all(plot_fn[finite_plot_ur] > 0.0)
                and np.all(np.isfinite(synthetic_fn))
                and np.all(synthetic_fn > 0.0)
            )
            if can_normalize:
                anchor_y = anchor_y / anchor_fn
                plot_y = plot_y / plot_fn
                synthetic_y = synthetic_y / synthetic_fn
                ylabel = normalized_frequency_ylabels[key]

        line_mask = np.isfinite(plot_ur) & np.isfinite(plot_y)
        if np.count_nonzero(line_mask) >= 2:
            ax.plot(
                plot_ur[line_mask],
                plot_y[line_mask],
                color="0.65",
                linewidth=1.2,
                zorder=1,
                label="Surrogate curve",
            )

        ax.scatter(
            anchor_ur,
            anchor_y,
            facecolors="white",
            edgecolors="black",
            marker="o",
            s=34,
            linewidths=1.2,
            zorder=3,
            label="Training data anchors",
        )
        if np.any(synthetic_mask):
            ax.scatter(
                synthetic_ur[synthetic_mask],
                synthetic_y[synthetic_mask],
                color="tab:blue",
                marker="x",
                s=42,
                linewidths=1.6,
                zorder=4,
                label="Surrogate validation metrics",
            )
        if np.any(anchor_row_mask):
            ax.scatter(
                synthetic_ur[anchor_row_mask],
                synthetic_y[anchor_row_mask],
                facecolors="white",
                edgecolors="tab:green",
                marker="o",
                s=48,
                linewidths=1.4,
                zorder=4,
                label="Anchor rows in output",
            )
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0.0)
        ax.grid(True, color="0.88", linewidth=0.6)

    for ax in axes[-1, :]:
        ax.set_xlabel(r"Reduced velocity $U_r$")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        ncol=min(4, max(1, len(by_label))),
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=int(dpi), bbox_inches="tight", pad_inches=0.02)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    paths = sorted(Path(".").glob(INPUT_NPZ_GLOB))
    if not paths:
        raise FileNotFoundError(f"No files matched INPUT_NPZ_GLOB={INPUT_NPZ_GLOB!r}.")
    exclude = np.asarray(list(EXCLUDE_URS), dtype=float).reshape(-1)
    anchors: list[AnchorPoint] = []
    skipped_excluded = 0
    for path in paths:
        ur = load_reduced_velocity(path, ur_source=UR_SOURCE)
        if exclude.size and np.any(np.isclose(ur, exclude, rtol=0.0, atol=float(EXCLUDE_UR_ATOL))):
            skipped_excluded += 1
            continue
        anchors.append(load_anchor_point(path, ur_source=INTERPOLATION_UR_SOURCE))
    if len({float(anchor.ur) for anchor in anchors}) < 2:
        raise ValueError("At least two unique non-excluded reduced velocities are required for interpolation.")
    points = generate_surrogate_validation_points(
        anchors,
        target_urs=TARGET_URS,
        points_per_interval=POINTS_PER_INTERVAL,
        interpolation_kind=INTERPOLATION_KIND,
        smoothing_strength=SMOOTHING_STRENGTH,
        include_anchor_points=INCLUDE_ANCHOR_POINTS,
    )
    save_surrogate_points(points, npz_path=OUTPUT_NPZ, csv_path=OUTPUT_CSV)
    if MAKE_DIAGNOSTIC_PLOT:
        save_diagnostic_plot(
            points,
            anchors=anchors,
            output_path=OUTPUT_PLOT,
            show=SHOW_DIAGNOSTIC_PLOT,
            dpi=PLOT_DPI,
        )
    print(
        f"Wrote {points['ur'].shape[0]} surrogate row(s) from "
        f"{len(anchors)} anchor file(s) to {OUTPUT_NPZ}"
    )
    if OUTPUT_CSV is not None:
        print(f"Wrote CSV summary to {OUTPUT_CSV}")
    if MAKE_DIAGNOSTIC_PLOT:
        print(f"Wrote diagnostic plot to {OUTPUT_PLOT}")
    if skipped_excluded:
        print(f"Skipped {skipped_excluded} excluded/test file(s) before metric extraction.")


if __name__ == "__main__":
    main()
