from __future__ import annotations

import csv
import re
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

try:
    from scipy.signal import welch
except ImportError:
    welch = None


COLUMN_NAMES = (
    "time",
    "x_disp",
    "y_disp",
    "x_vel",
    "y_vel",
    "x_acc",
    "y_acc",
    "x_force",
    "y_force",
)
EXPECTED_COLUMNS = len(COLUMN_NAMES)
UR_PATTERN = re.compile(r"Ur([0-9]+(?:[A-Za-z0-9._-]*)?)", re.IGNORECASE)


@dataclass(frozen=True)
class CfdRecord:
    path: Path
    case_name: str
    ur_value: float | None
    data: np.ndarray
    skipped_lines: int

    @property
    def time(self) -> np.ndarray:
        return self.data[:, 0]

    @property
    def y_disp(self) -> np.ndarray:
        return self.data[:, 2]

    @property
    def y_vel(self) -> np.ndarray:
        return self.data[:, 4]

    @property
    def y_acc(self) -> np.ndarray:
        return self.data[:, 6]

    @property
    def y_force(self) -> np.ndarray:
        return self.data[:, 8]


def infer_ur_from_path(path: Path) -> float | None:
    for part in reversed(path.parts):
        match = UR_PATTERN.search(part)
        if match is None:
            continue
        raw = match.group(1)
        token = raw.replace("_", ".").replace("p", ".")
        if token.endswith("."):
            token = token[:-1]
        try:
            if "." not in token and raw.isdigit() and len(raw) > 2:
                return float(raw) / 100.0
            return float(token)
        except ValueError:
            continue
    return None


def load_dog_file(path: Path) -> tuple[np.ndarray, int]:
    rows: list[list[float]] = []
    skipped = 0
    with path.open("rb") as fh:
        for raw_line in fh:
            clean = raw_line.replace(b"\x00", b" ").strip()
            if not clean:
                continue
            text = clean.decode("utf-8", errors="ignore")
            parts = text.split()
            if len(parts) < EXPECTED_COLUMNS:
                skipped += 1
                continue
            try:
                row = [float(parts[idx]) for idx in range(EXPECTED_COLUMNS)]
            except ValueError:
                skipped += 1
                continue
            rows.append(row)
    if len(rows) < 2:
        raise ValueError(f"{path} did not contain at least two valid numeric rows.")
    data = np.asarray(rows, dtype=float)
    data = data[np.argsort(data[:, 0])]
    data = data[np.all(np.isfinite(data), axis=1)]
    if data.shape[0] < 2:
        raise ValueError(f"{path} became too short after filtering invalid rows.")
    return data, skipped


def dominant_frequency(signal: np.ndarray, dt: float) -> float:
    if signal.size < 2 or not np.isfinite(dt) or dt <= 0.0:
        return float("nan")
    centered = signal - np.mean(signal)
    if np.allclose(centered, 0.0):
        return float("nan")
    fft_vals = np.fft.rfft(centered)
    freqs = np.fft.rfftfreq(centered.size, d=dt)
    if freqs.size <= 1:
        return float("nan")
    mags = np.abs(fft_vals)
    mags[0] = 0.0
    idx = int(np.argmax(mags))
    if mags[idx] <= 0.0:
        return float("nan")
    return float(freqs[idx])


def mean_displacement_amplitude(signal: np.ndarray) -> float:
    signal = np.asarray(signal, dtype=float).reshape(-1)
    if signal.size == 0:
        return float("nan")
    centered = signal - float(np.mean(signal))
    return float(np.mean(np.abs(centered)))


def displacement_peak_amplitudes(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=float).reshape(-1)
    if signal.size < 3:
        return np.asarray([], dtype=float)
    centered = np.abs(signal - float(np.mean(signal)))
    peak_mask = (centered[1:-1] >= centered[:-2]) & (centered[1:-1] >= centered[2:])
    peaks = centered[1:-1][peak_mask]
    return np.asarray(peaks, dtype=float)


def mean_peak_displacement_amplitude(signal: np.ndarray) -> float:
    peaks = displacement_peak_amplitudes(signal)
    if peaks.size == 0:
        return float("nan")
    return float(np.mean(peaks))


def power_spectrum(
    signal: np.ndarray,
    dt: float,
    *,
    method: str = "welch",
    nperseg: int | None = None,
    nfft_factor: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    centered = np.asarray(signal, dtype=float).reshape(-1) - float(np.mean(signal))
    if centered.size < 8 or not np.isfinite(dt) or dt <= 0.0:
        return np.asarray([]), np.asarray([])
    if welch is not None and str(method).lower() == "welch":
        fs = 1.0 / float(dt)
        if nperseg is None:
            seg = centered.size
        else:
            seg = max(8, min(int(nperseg), centered.size))
        pad_factor = max(1, int(nfft_factor))
        base_nfft = 1 << int(np.ceil(np.log2(max(8, seg))))
        nfft = max(seg, pad_factor * base_nfft)
        overlap = 0 if seg < 16 else min(seg // 2, seg - 1)
        freqs, psd = welch(
            centered,
            fs=fs,
            window="boxcar",
            nperseg=seg,
            noverlap=overlap,
            nfft=nfft,
            detrend=False,
            scaling="density",
        )
        return np.asarray(freqs, dtype=float), np.asarray(psd, dtype=float)
    pad_factor = max(1, int(nfft_factor))
    base_nfft = 1 << int(np.ceil(np.log2(max(8, centered.size))))
    nfft = max(centered.size, pad_factor * base_nfft)
    freqs = np.fft.rfftfreq(nfft, d=float(dt))
    psd = np.abs(np.fft.rfft(centered, n=nfft)) ** 2
    return np.asarray(freqs, dtype=float), np.asarray(psd, dtype=float)


def psd_freq_limit(
    signal: np.ndarray,
    dt: float,
    freqs: np.ndarray,
    *,
    freq_multiplier: float = 3.0,
    min_fmax_hz: float = 0.25,
) -> float | None:
    if freqs.size == 0 or not np.isfinite(dt) or dt <= 0.0:
        return None
    nyquist = 0.5 / float(dt)
    dom = dominant_frequency(signal, dt)
    if np.isfinite(dom) and dom > 0.0:
        return float(min(nyquist, max(float(min_fmax_hz), float(freq_multiplier) * dom)))
    positive = freqs[np.isfinite(freqs) & (freqs > 0.0)]
    if positive.size == 0:
        return float(nyquist)
    return float(min(nyquist, max(float(min_fmax_hz), np.quantile(positive, 0.05))))


def record_summary(record: CfdRecord) -> dict[str, float | str]:
    time = record.time
    dt = float(np.median(np.diff(time)))
    return {
        "case_name": record.case_name,
        "path": str(record.path),
        "ur_value": float(record.ur_value) if record.ur_value is not None else float("nan"),
        "num_rows": int(record.data.shape[0]),
        "skipped_lines": int(record.skipped_lines),
        "dt": dt,
        "duration_s": float(time[-1] - time[0]),
        "y_disp_std": float(np.std(record.y_disp)),
        "y_disp_mean_amp": mean_displacement_amplitude(record.y_disp),
        "y_disp_mean_peak_amp": mean_peak_displacement_amplitude(record.y_disp),
        "y_disp_rms": float(np.sqrt(np.mean(record.y_disp * record.y_disp))),
        "y_vel_std": float(np.std(record.y_vel)),
        "y_force_std": float(np.std(record.y_force)),
        "y_force_rms": float(np.sqrt(np.mean(record.y_force * record.y_force))),
        "dominant_freq_y_disp_hz": dominant_frequency(record.y_disp, dt),
        "dominant_freq_y_force_hz": dominant_frequency(record.y_force, dt),
    }


def plot_case(
    record: CfdRecord,
    output_dir: Path | None = None,
    *,
    psd_method: str = "welch",
    psd_nperseg: int | None = None,
    psd_nfft_factor: int = 4,
    psd_freq_multiplier: float = 3.0,
    psd_min_fmax_hz: float = 0.25,
    figsize: tuple[float, float] = (14, 10),
    nondimensionalized: bool = False,
    nondim_time_label: str = "tau = omega_n t",
    nondim_freq_label: str = "omega / omega_n",
    nondim_freq_scale_factor: float = 1.0,
) -> None:
    time = record.time
    dt = float(np.median(np.diff(time)))
    freqs_disp, psd_disp = power_spectrum(
        record.y_disp,
        dt,
        method=psd_method,
        nperseg=psd_nperseg,
        nfft_factor=psd_nfft_factor,
    )
    freqs_force, psd_force = power_spectrum(
        record.y_force,
        dt,
        method=psd_method,
        nperseg=psd_nperseg,
        nfft_factor=psd_nfft_factor,
    )
    disp_fmax = psd_freq_limit(
        record.y_disp,
        dt,
        freqs_disp,
        freq_multiplier=psd_freq_multiplier,
        min_fmax_hz=psd_min_fmax_hz,
    )
    force_fmax = psd_freq_limit(
        record.y_force,
        dt,
        freqs_force,
        freq_multiplier=psd_freq_multiplier,
        min_fmax_hz=psd_min_fmax_hz,
    )

    fig, axes = plt.subplots(3, 2, figsize=figsize, constrained_layout=True)

    if nondimensionalized:
        disp_labels = ("x/D", "y/D")
        vel_labels = ("u/(omega_n D)", "v/(omega_n D)")
        acc_labels = ("a_x/(omega_n^2 D)", "a_y/(omega_n^2 D)")
        force_labels = ("F_x/f0", "F_y/f0")
        disp_title = "Nondimensionalized Displacement"
        vel_title = "Nondimensionalized Velocity"
        acc_title = "Nondimensionalized Acceleration"
        force_title = "Nondimensionalized Force"
        disp_psd_title = "PSD of y/D"
        force_psd_title = "PSD of F_y/f0"
        time_xlabel = nondim_time_label
        freq_xlabel = nondim_freq_label
        freqs_disp = freqs_disp * float(nondim_freq_scale_factor)
        freqs_force = freqs_force * float(nondim_freq_scale_factor)
        if disp_fmax is not None:
            disp_fmax = float(disp_fmax) * float(nondim_freq_scale_factor)
        if force_fmax is not None:
            force_fmax = float(force_fmax) * float(nondim_freq_scale_factor)
    else:
        disp_labels = ("x disp", "y disp")
        vel_labels = ("x vel", "y vel")
        acc_labels = ("x acc", "y acc")
        force_labels = ("x force", "y force")
        disp_title = "Displacement"
        vel_title = "Velocity"
        acc_title = "Acceleration"
        force_title = "Force"
        disp_psd_title = "Y-Displacement PSD"
        force_psd_title = "Y-Force PSD"
        time_xlabel = "time"
        freq_xlabel = "frequency [Hz]"

    axes[0, 0].plot(time, record.data[:, 1], label=disp_labels[0])
    axes[0, 0].plot(time, record.y_disp, label=disp_labels[1])
    axes[0, 0].set_title(disp_title)
    axes[0, 0].set_xlabel(time_xlabel)
    axes[0, 0].legend()

    axes[0, 1].plot(time, record.data[:, 3], label=vel_labels[0])
    axes[0, 1].plot(time, record.y_vel, label=vel_labels[1])
    axes[0, 1].set_title(vel_title)
    axes[0, 1].set_xlabel(time_xlabel)
    axes[0, 1].legend()

    axes[1, 0].plot(time, record.data[:, 5], label=acc_labels[0])
    axes[1, 0].plot(time, record.y_acc, label=acc_labels[1])
    axes[1, 0].set_title(acc_title)
    axes[1, 0].set_xlabel(time_xlabel)
    axes[1, 0].legend()

    axes[1, 1].plot(time, record.data[:, 7], label=force_labels[0])
    axes[1, 1].plot(time, record.y_force, label=force_labels[1])
    axes[1, 1].set_title(force_title)
    axes[1, 1].set_xlabel(time_xlabel)
    axes[1, 1].legend()

    if freqs_disp.size > 0:
        axes[2, 0].plot(freqs_disp, psd_disp)
        axes[2, 0].set_xlim(0.0, disp_fmax if disp_fmax is not None else float(freqs_disp[-1]))
    axes[2, 0].set_title(disp_psd_title)
    axes[2, 0].set_xlabel(freq_xlabel)
    axes[2, 0].set_ylabel("PSD")

    if freqs_force.size > 0:
        axes[2, 1].plot(freqs_force, psd_force)
        axes[2, 1].set_xlim(0.0, force_fmax if force_fmax is not None else float(freqs_force[-1]))
    axes[2, 1].set_title(force_psd_title)
    axes[2, 1].set_xlabel(freq_xlabel)
    axes[2, 1].set_ylabel("PSD")

    ur_text = "unknown" if record.ur_value is None else f"{record.ur_value:g}"
    fig.suptitle(f"{record.case_name} | U_r={ur_text} | rows={record.data.shape[0]}")

    if output_dir is not None:
        out_path = output_dir / "cases" / f"{record.case_name}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_summary(
    rows: list[dict[str, float | str]],
    output_dir: Path | None = None,
    *,
    figsize: tuple[float, float] = (12, 9),
    nondimensionalized: bool = False,
    nondim_freq_title: str = "Dominant (y/D) Frequency Ratio",
    nondim_freq_ylabel: str = "omega / omega_n",
    nondim_duration_title: str = "Nondimensional Duration",
    nondim_duration_ylabel: str = "t*",
    nondim_freq_scale_factor: float = 1.0,
    freq_reference_x: np.ndarray | None = None,
    freq_reference_y: np.ndarray | float | None = None,
    freq_reference_label: str = "Natural frequency",
    freq_reference_series: list[dict[str, object]] | None = None,
    x_values: np.ndarray | None = None,
    x_label: str = "Reduced velocity U_r",
) -> None:
    numeric_rows = [row for row in rows if np.isfinite(float(row["ur_value"]))]
    if not numeric_rows:
        return
    numeric_rows = sorted(numeric_rows, key=lambda row: (float(row["ur_value"]), str(row["case_name"])))
    ur = np.asarray([float(row["ur_value"]) for row in numeric_rows], dtype=float)
    if x_values is not None:
        ur = np.asarray(x_values, dtype=float)
    y_std = np.asarray([float(row["y_disp_std"]) for row in numeric_rows], dtype=float)
    y_freq = np.asarray([float(row["dominant_freq_y_disp_hz"]) for row in numeric_rows], dtype=float)
    f_std = np.asarray([float(row["y_force_std"]) for row in numeric_rows], dtype=float)
    duration = np.asarray([float(row["duration_s"]) for row in numeric_rows], dtype=float)
    if nondimensionalized:
        y_freq = y_freq * float(nondim_freq_scale_factor)

    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    if nondimensionalized:
        plots = (
            (y_std, "Std of y/D", "std(y/D)"),
            (y_freq, nondim_freq_title, nondim_freq_ylabel),
            (f_std, "Std of F_y/f0", "std(F_y/f0)"),
            (duration, nondim_duration_title, nondim_duration_ylabel),
        )
    else:
        plots = (
            (y_std, "Y-Displacement Std", "std"),
            (y_freq, "Dominant Y-Displacement Frequency", "Hz"),
            (f_std, "Y-Force Std", "std"),
            (duration, "Duration", "s"),
        )
    for ax_idx, (ax, (values, title, ylabel)) in enumerate(zip(axes.flat, plots)):
        ax.plot(ur, values, marker="o")
        if ax_idx == 1:
            reference_series = list(freq_reference_series or [])
            if freq_reference_y is not None:
                reference_series.append(
                    {
                        "x": freq_reference_x,
                        "y": freq_reference_y,
                        "label": freq_reference_label,
                        "color": "tab:red",
                        "linestyle": "--",
                    }
                )
            plotted_reference = False
            for series_idx, series in enumerate(reference_series):
                ref_y_raw = series.get("y")
                if ref_y_raw is None:
                    continue
                label = str(series.get("label", f"Reference {series_idx + 1}"))
                color = str(series.get("color", f"C{series_idx + 1}"))
                linestyle = str(series.get("linestyle", "--"))
                ref_x_raw = series.get("x")
                if np.isscalar(ref_y_raw):
                    ax.axhline(float(ref_y_raw), color=color, linestyle=linestyle, label=label)
                    plotted_reference = True
                else:
                    ref_x = ur if ref_x_raw is None else np.asarray(ref_x_raw, dtype=float)
                    ref_y = np.asarray(ref_y_raw, dtype=float)
                    finite = np.isfinite(ref_x) & np.isfinite(ref_y)
                    if np.any(finite):
                        ax.plot(
                            ref_x[finite],
                            ref_y[finite],
                            color=color,
                            linestyle=linestyle,
                            marker=None,
                            label=label,
                        )
                        plotted_reference = True
            if plotted_reference:
                ax.legend()
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel)

    if output_dir is not None:
        out_path = output_dir / "summary_vs_ur.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def write_summary_csv(rows: list[dict[str, float | str]], output_dir: Path) -> None:
    if not rows:
        return
    out_path = output_dir / "summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze_cfd_data(
    input_dir: Path,
    output_dir: Path,
    pattern: str = "*.dog",
    max_files: int | None = None,
) -> tuple[list[CfdRecord], list[dict[str, float | str]], list[tuple[Path, str]]]:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    files = sorted(input_dir.rglob(pattern))
    if max_files is not None:
        files = files[: max(0, int(max_files))]
    if not files:
        raise RuntimeError(f"No files matched {pattern!r} under {input_dir}")

    records: list[CfdRecord] = []
    summary_rows: list[dict[str, float | str]] = []
    failed: list[tuple[Path, str]] = []
    for path in files:
        try:
            data, skipped = load_dog_file(path)
            ur_value = infer_ur_from_path(path)
            case_name = path.relative_to(input_dir).with_suffix("").as_posix().replace("/", "__")
            record = CfdRecord(
                path=path,
                case_name=case_name,
                ur_value=ur_value,
                data=data,
                skipped_lines=skipped,
            )
            records.append(record)
            summary_rows.append(record_summary(record))
            plot_case(record, output_dir=output_dir)
        except Exception as exc:
            failed.append((path, str(exc)))

    write_summary_csv(summary_rows, output_dir)
    plot_summary(summary_rows, output_dir=output_dir)

    if failed:
        failed_path = output_dir / "failed_files.csv"
        failed_path.parent.mkdir(parents=True, exist_ok=True)
        with failed_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["path", "error"])
            writer.writerows((str(path), err) for path, err in failed)

    return records, summary_rows, failed


def plot_timeseries_grid(
    records: list[CfdRecord],
    *,
    sort_by: str = "ur",
    max_cases: int | None = None,
    figwidth: float = 14,
    rowheight: float = 2.4,
    figure_title: str | None = None,
    nondimensionalized: bool = False,
    nondim_time_label: str = "tau = omega_n t",
    compact_titles: bool = False,
    show_side_histogram: bool = False,
    histogram_bins: int = 30,
    histogram_size: str = "18%",
    histogram_pad: float = 0.08,
) -> plt.Figure | None:
    selected_records = list(records)
    if sort_by == "ur":
        selected_records = sorted(
            selected_records,
            key=lambda rec: (
                float("inf") if rec.ur_value is None else float(rec.ur_value),
                rec.case_name,
            ),
        )
    else:
        selected_records = sorted(selected_records, key=lambda rec: rec.case_name)

    if max_cases is not None:
        selected_records = selected_records[: int(max_cases)]

    if not selected_records:
        print("No records to plot.")
        return None

    fig, axes = plt.subplots(
        nrows=len(selected_records),
        ncols=2,
        figsize=(figwidth, rowheight * len(selected_records)),
        constrained_layout=True,
        squeeze=False,
    )

    for row_idx, record in enumerate(selected_records):
        ur_text = "unknown" if record.ur_value is None else f"{record.ur_value:g}"
        disp_ax = axes[row_idx, 0]
        force_ax = axes[row_idx, 1]
        case_title = f"{record.case_name} (U_r={ur_text})"

        if nondimensionalized:
            disp_title = case_title if compact_titles else f"{case_title} | y/D"
            disp_ylabel = "y/D"
            force_title = case_title if compact_titles else f"{case_title} | C_F"
            force_ylabel = "C_F"
            time_xlabel = nondim_time_label
        else:
            disp_title = case_title if compact_titles else f"{case_title} | Y displacement"
            disp_ylabel = "y"
            force_title = case_title if compact_titles else f"{case_title} | Y force"
            force_ylabel = "force"
            time_xlabel = "time"

        disp_ax.plot(record.time, record.y_disp, linewidth=1.0)
        disp_ax.set_title(disp_title)
        disp_ax.set_xlabel(time_xlabel)
        disp_ax.set_ylabel(disp_ylabel)
        disp_ax.grid(True, alpha=0.3)
        if show_side_histogram:
            disp_hist_ax = make_axes_locatable(disp_ax).append_axes(
                "right",
                size=histogram_size,
                pad=histogram_pad,
                sharey=disp_ax,
            )
            disp_hist_ax.hist(
                record.y_disp,
                bins=int(histogram_bins),
                orientation="horizontal",
                color="tab:blue",
                alpha=0.25,
                edgecolor="tab:blue",
                linewidth=0.8,
            )
            disp_hist_ax.grid(False)
            disp_hist_ax.tick_params(axis="y", left=False, labelleft=False)
            disp_hist_ax.tick_params(axis="x", labelsize=8)
            disp_hist_ax.spines["left"].set_visible(False)
            disp_hist_ax.spines["right"].set_visible(False)
            disp_hist_ax.spines["top"].set_visible(False)
            disp_hist_ax.set_xlabel("count", fontsize=8)

        force_ax.plot(record.time, record.y_force, linewidth=1.0, color="tab:orange")
        force_ax.set_title(force_title)
        force_ax.set_xlabel(time_xlabel)
        force_ax.set_ylabel(force_ylabel)
        force_ax.grid(True, alpha=0.3)
        if show_side_histogram:
            force_hist_ax = make_axes_locatable(force_ax).append_axes(
                "right",
                size=histogram_size,
                pad=histogram_pad,
                sharey=force_ax,
            )
            force_hist_ax.hist(
                record.y_force,
                bins=int(histogram_bins),
                orientation="horizontal",
                color="tab:orange",
                alpha=0.25,
                edgecolor="tab:orange",
                linewidth=0.8,
            )
            force_hist_ax.grid(False)
            force_hist_ax.tick_params(axis="y", left=False, labelleft=False)
            force_hist_ax.tick_params(axis="x", labelsize=8)
            force_hist_ax.spines["left"].set_visible(False)
            force_hist_ax.spines["right"].set_visible(False)
            force_hist_ax.spines["top"].set_visible(False)
            force_hist_ax.set_xlabel("count", fontsize=8)

    if figure_title:
        fig.suptitle(figure_title)
    plt.show()
    return fig


def default_cleaning_manifest(records: list[CfdRecord]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record in sorted(
        records,
        key=lambda rec: (
            float("inf") if rec.ur_value is None else float(rec.ur_value),
            rec.case_name,
        ),
    ):
        time = record.time
        rows.append(
            {
                "case_name": record.case_name,
                "path": str(record.path),
                "ur_value": float(record.ur_value) if record.ur_value is not None else np.nan,
                "action": "keep",
                "start_time": np.nan,
                "end_time": np.nan,
                "full_start_time": float(time[0]),
                "full_end_time": float(time[-1]),
                "duration_s": float(time[-1] - time[0]),
                "num_rows": int(record.data.shape[0]),
                "notes": "",
            }
        )
    return pd.DataFrame(rows)


def build_cleaning_manifest(records: list[CfdRecord], manifest_path: Path) -> pd.DataFrame:
    manifest_path = Path(manifest_path)
    defaults = default_cleaning_manifest(records)
    editable_cols = ["action", "start_time", "end_time", "notes"]
    if manifest_path.exists():
        existing = pd.read_csv(manifest_path)
        if "case_name" not in existing.columns:
            raise ValueError(f"Manifest at {manifest_path} is missing a 'case_name' column.")
        existing = existing.copy()
        existing["case_name"] = existing["case_name"].astype(str)
        manifest = defaults.set_index("case_name")
        existing = existing.set_index("case_name")
        common_cases = manifest.index.intersection(existing.index)
        for col in editable_cols:
            if col in existing.columns:
                manifest.loc[common_cases, col] = existing.loc[common_cases, col]
        manifest = manifest.reset_index()
    else:
        manifest = defaults.copy()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    return manifest


def normalize_clean_action(action: object) -> str:
    if action is None or pd.isna(action):
        return "keep"
    text = str(action).strip().lower()
    if text in {"", "keep", "yes", "true", "1"}:
        return "keep"
    if text in {"drop", "remove", "exclude", "skip", "no"}:
        return "drop"
    if text in {"trim", "cut", "crop"}:
        return "trim"
    return text


def apply_cleaning_manifest(
    records: list[CfdRecord],
    manifest_df: pd.DataFrame,
) -> tuple[list[CfdRecord], pd.DataFrame]:
    decisions = manifest_df.set_index("case_name").to_dict(orient="index")
    cleaned_records: list[CfdRecord] = []
    applied_rows: list[dict[str, object]] = []

    for record in records:
        decision = decisions.get(record.case_name, {})
        action = normalize_clean_action(decision.get("action", "keep"))
        start_raw = decision.get("start_time", np.nan)
        end_raw = decision.get("end_time", np.nan)
        start_time = float(start_raw) if pd.notna(start_raw) else None
        end_time = float(end_raw) if pd.notna(end_raw) else None

        if action == "drop":
            applied_rows.append(
                {
                    "case_name": record.case_name,
                    "action_applied": "drop",
                    "num_rows_after": 0,
                    "start_time_after": np.nan,
                    "end_time_after": np.nan,
                }
            )
            continue

        data = record.data
        trimmed = data
        action_applied = "keep"
        if start_time is not None or end_time is not None:
            mask = np.ones(data.shape[0], dtype=bool)
            if start_time is not None:
                mask &= data[:, 0] >= start_time
            if end_time is not None:
                mask &= data[:, 0] <= end_time
            trimmed = data[mask]
            action_applied = "trim"

        if trimmed.shape[0] < 2:
            applied_rows.append(
                {
                    "case_name": record.case_name,
                    "action_applied": "drop_too_short",
                    "num_rows_after": int(trimmed.shape[0]),
                    "start_time_after": np.nan,
                    "end_time_after": np.nan,
                }
            )
            continue

        if action_applied == "trim":
            cleaned_record = replace(record, data=np.asarray(trimmed, dtype=float))
        else:
            cleaned_record = record
        cleaned_records.append(cleaned_record)
        time = cleaned_record.time
        applied_rows.append(
            {
                "case_name": cleaned_record.case_name,
                "action_applied": action_applied,
                "num_rows_after": int(cleaned_record.data.shape[0]),
                "start_time_after": float(time[0]),
                "end_time_after": float(time[-1]),
            }
        )

    return cleaned_records, pd.DataFrame(applied_rows)


def summarize_duplicate_timestamps(records: list[CfdRecord]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record in records:
        time = np.asarray(record.time, dtype=float)
        if time.size < 2:
            rows.append(
                {
                    "case_name": record.case_name,
                    "ur_value": float(record.ur_value) if record.ur_value is not None else np.nan,
                    "num_rows_before": int(time.size),
                    "duplicate_rows_removed": 0,
                    "num_rows_after": int(time.size),
                    "has_duplicate_timestamps": False,
                    "first_duplicate_time": np.nan,
                }
            )
            continue
        duplicate_mask = np.diff(time) <= 0.0
        duplicate_indices = np.where(duplicate_mask)[0]
        duplicate_count = int(duplicate_indices.size)
        first_duplicate_time = float(time[duplicate_indices[0] + 1]) if duplicate_count > 0 else np.nan
        rows.append(
            {
                "case_name": record.case_name,
                "ur_value": float(record.ur_value) if record.ur_value is not None else np.nan,
                "num_rows_before": int(time.size),
                "duplicate_rows_removed": duplicate_count,
                "num_rows_after": int(time.size - duplicate_count),
                "has_duplicate_timestamps": bool(duplicate_count > 0),
                "first_duplicate_time": first_duplicate_time,
            }
        )
    return pd.DataFrame(rows)


def remove_duplicate_timestamps(
    records: list[CfdRecord],
) -> tuple[list[CfdRecord], pd.DataFrame]:
    cleaned_records: list[CfdRecord] = []
    report_rows: list[dict[str, object]] = []

    for record in records:
        data = np.asarray(record.data, dtype=float)
        time = data[:, 0]
        keep_mask = np.ones(data.shape[0], dtype=bool)
        duplicate_mask = np.diff(time) <= 0.0
        if np.any(duplicate_mask):
            keep_mask[1:] = ~duplicate_mask
        deduped = data[keep_mask]
        duplicate_indices = np.where(duplicate_mask)[0]
        duplicate_count = int(duplicate_indices.size)
        first_duplicate_time = float(time[duplicate_indices[0] + 1]) if duplicate_count > 0 else np.nan
        if deduped.shape[0] < 2:
            raise ValueError(
                f"Case {record.case_name} became too short after duplicate timestamp removal."
            )
        if duplicate_count > 0:
            cleaned_record = replace(record, data=np.asarray(deduped, dtype=float))
        else:
            cleaned_record = record
        cleaned_records.append(cleaned_record)
        report_rows.append(
            {
                "case_name": record.case_name,
                "ur_value": float(record.ur_value) if record.ur_value is not None else np.nan,
                "num_rows_before": int(data.shape[0]),
                "duplicate_rows_removed": duplicate_count,
                "num_rows_after": int(deduped.shape[0]),
                "has_duplicate_timestamps": bool(duplicate_count > 0),
                "first_duplicate_time": first_duplicate_time,
            }
        )

    return cleaned_records, pd.DataFrame(report_rows)


def default_nondim_metadata(
    records: list[CfdRecord],
    *,
    flow_speed_m_s: float | None,
    diameter_m: float,
    structural_frequency_hz: float | None,
    rho_kg_m3: float,
    span_m: float,
    stiffness_n_m: float | None = None,
    effective_mass_kg: float | None = None,
    dry_mass_kg: float | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record in sorted(
        records,
        key=lambda rec: (
            float("inf") if rec.ur_value is None else float(rec.ur_value),
            rec.case_name,
        ),
    ):
        dt = np.nan
        if record.time.size > 1:
            dt = float(np.median(np.diff(record.time)))
        rows.append(
            {
                "case_name": record.case_name,
                "path": str(record.path),
                "ur_value": float(record.ur_value) if record.ur_value is not None else np.nan,
                "dt": dt,
                "flow_speed_m_s": (
                    float(flow_speed_m_s)
                    if flow_speed_m_s is not None and np.isfinite(flow_speed_m_s)
                    else np.nan
                ),
                "diameter_m": float(diameter_m),
                "structural_frequency_hz": (
                    float(structural_frequency_hz)
                    if structural_frequency_hz is not None and np.isfinite(structural_frequency_hz)
                    else np.nan
                ),
                "stiffness_n_m": (
                    float(stiffness_n_m)
                    if stiffness_n_m is not None and np.isfinite(stiffness_n_m)
                    else np.nan
                ),
                "effective_mass_kg": (
                    float(effective_mass_kg)
                    if effective_mass_kg is not None and np.isfinite(effective_mass_kg)
                    else np.nan
                ),
                "dry_mass_kg": (
                    float(dry_mass_kg)
                    if dry_mass_kg is not None and np.isfinite(dry_mass_kg)
                    else np.nan
                ),
                "rho_kg_m3": float(rho_kg_m3),
                "span_m": float(span_m),
                "notes": "",
            }
        )
    return pd.DataFrame(rows)


def build_nondim_metadata(
    records: list[CfdRecord],
    metadata_path: Path,
    *,
    flow_speed_m_s: float | None,
    diameter_m: float,
    structural_frequency_hz: float | None,
    rho_kg_m3: float,
    span_m: float,
    stiffness_n_m: float | None = None,
    effective_mass_kg: float | None = None,
    dry_mass_kg: float | None = None,
) -> pd.DataFrame:
    metadata_path = Path(metadata_path)
    defaults = default_nondim_metadata(
        records,
        flow_speed_m_s=flow_speed_m_s,
        diameter_m=diameter_m,
        structural_frequency_hz=structural_frequency_hz,
        rho_kg_m3=rho_kg_m3,
        span_m=span_m,
        stiffness_n_m=stiffness_n_m,
        effective_mass_kg=effective_mass_kg,
        dry_mass_kg=dry_mass_kg,
    )
    editable_cols = [
        "flow_speed_m_s",
        "diameter_m",
        "structural_frequency_hz",
        "stiffness_n_m",
        "effective_mass_kg",
        "dry_mass_kg",
        "rho_kg_m3",
        "span_m",
        "notes",
    ]
    if metadata_path.exists():
        existing = pd.read_csv(metadata_path)
        if "case_name" not in existing.columns:
            raise ValueError(f"Nondim metadata at {metadata_path} is missing a 'case_name' column.")
        existing = existing.copy()
        existing["case_name"] = existing["case_name"].astype(str)
        metadata = defaults.set_index("case_name")
        existing = existing.set_index("case_name")
        common_cases = metadata.index.intersection(existing.index)
        for col in editable_cols:
            if col in existing.columns:
                metadata.loc[common_cases, col] = existing.loc[common_cases, col]
        metadata = metadata.reset_index()
    else:
        metadata = defaults.copy()
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(metadata_path, index=False)
    return metadata


def resolve_structural_frequency_hz(metadata_row: dict[str, object], case_name: str) -> float:
    fn_raw = metadata_row.get("structural_frequency_hz", np.nan)
    if pd.notna(fn_raw):
        fn = float(fn_raw)
        if np.isfinite(fn) and fn > 0.0:
            return fn

    stiffness_raw = metadata_row.get("stiffness_n_m", np.nan)
    mass_raw = metadata_row.get("effective_mass_kg", np.nan)
    if pd.notna(stiffness_raw) and pd.notna(mass_raw):
        stiffness = float(stiffness_raw)
        mass = float(mass_raw)
        if np.isfinite(stiffness) and stiffness > 0.0 and np.isfinite(mass) and mass > 0.0:
            return float(np.sqrt(stiffness / mass) / (2.0 * np.pi))

    raise ValueError(
        f"Case {case_name} needs either a positive structural_frequency_hz or both stiffness_n_m and effective_mass_kg."
    )


def resolve_dry_structural_frequency_hz(metadata_row: dict[str, object], case_name: str) -> float:
    stiffness_n_m = resolve_stiffness_n_m(
        metadata_row,
        case_name,
        structural_frequency_hz=resolve_structural_frequency_hz(metadata_row, case_name),
    )
    dry_mass_raw = metadata_row.get("dry_mass_kg", np.nan)
    if pd.notna(dry_mass_raw):
        dry_mass_kg = float(dry_mass_raw)
        if np.isfinite(dry_mass_kg) and dry_mass_kg > 0.0:
            return float(np.sqrt(stiffness_n_m / dry_mass_kg) / (2.0 * np.pi))
    raise ValueError(f"Case {case_name} needs a positive dry_mass_kg value to compute dry natural frequency.")


def resolve_flow_speed_m_s(metadata_row: dict[str, object], case_name: str) -> float:
    flow_speed_raw = metadata_row.get("flow_speed_m_s", np.nan)
    if pd.notna(flow_speed_raw):
        flow_speed = float(flow_speed_raw)
        if np.isfinite(flow_speed) and flow_speed > 0.0:
            return flow_speed
    raise ValueError(f"Case {case_name} needs a positive flow_speed_m_s value.")


def resolve_computed_ur_value(metadata_row: dict[str, object], case_name: str) -> float:
    flow_speed_m_s = resolve_flow_speed_m_s(metadata_row, case_name)
    diameter_raw = metadata_row.get("diameter_m", np.nan)
    if not pd.notna(diameter_raw):
        raise ValueError(f"Case {case_name} needs a positive diameter_m value.")
    diameter_m = float(diameter_raw)
    if not np.isfinite(diameter_m) or diameter_m <= 0.0:
        raise ValueError(f"Case {case_name} needs a positive diameter_m value.")
    structural_frequency_hz = resolve_structural_frequency_hz(metadata_row, case_name)
    return float(flow_speed_m_s / (structural_frequency_hz * diameter_m))


def resolve_stiffness_n_m(
    metadata_row: dict[str, object],
    case_name: str,
    *,
    structural_frequency_hz: float | None = None,
) -> float:
    stiffness_raw = metadata_row.get("stiffness_n_m", np.nan)
    if pd.notna(stiffness_raw):
        stiffness = float(stiffness_raw)
        if np.isfinite(stiffness) and stiffness > 0.0:
            return stiffness

    mass_raw = metadata_row.get("effective_mass_kg", np.nan)
    if structural_frequency_hz is not None and pd.notna(mass_raw):
        mass = float(mass_raw)
        if np.isfinite(mass) and mass > 0.0:
            return float(((2.0 * np.pi * structural_frequency_hz) ** 2) * mass)

    raise ValueError(
        f"Case {case_name} needs a positive stiffness_n_m, or effective_mass_kg together with structural_frequency_hz."
    )


def nondimensionalize_record(
    record: CfdRecord,
    *,
    flow_speed_m_s: float,
    diameter_m: float,
    structural_frequency_hz: float,
    rho_kg_m3: float,
    span_m: float,
    time_scale_factor: float | None = None,
) -> CfdRecord:
    diameter_m = float(diameter_m)
    flow_speed_m_s = float(flow_speed_m_s)
    structural_frequency_hz = float(structural_frequency_hz)
    rho_kg_m3 = float(rho_kg_m3)
    span_m = float(span_m)
    omega_n = 2.0 * np.pi * structural_frequency_hz
    if time_scale_factor is None:
        time_scale_factor = omega_n
    time_scale_factor = float(time_scale_factor)

    if not np.isfinite(diameter_m) or diameter_m <= 0.0:
        raise ValueError("diameter_m must be a positive finite number.")
    if not np.isfinite(flow_speed_m_s) or flow_speed_m_s <= 0.0:
        raise ValueError("flow_speed_m_s must be a positive finite number.")
    if not np.isfinite(structural_frequency_hz) or structural_frequency_hz <= 0.0:
        raise ValueError("structural_frequency_hz must be a positive finite number.")
    if not np.isfinite(time_scale_factor) or time_scale_factor <= 0.0:
        raise ValueError("time_scale_factor must be a positive finite number.")
    if not np.isfinite(rho_kg_m3) or rho_kg_m3 <= 0.0:
        raise ValueError("rho_kg_m3 must be a positive finite number.")
    if not np.isfinite(span_m) or span_m <= 0.0:
        raise ValueError("span_m must be a positive finite number.")

    force_scale = 0.5 * rho_kg_m3 * (diameter_m**2) * span_m * (flow_speed_m_s**2)
    vel_scale = omega_n * diameter_m
    acc_scale = (omega_n**2) * diameter_m

    if force_scale == 0.0:
        raise ValueError(f"Record {record.case_name} produced zero force scale.")

    data_total = np.asarray(record.data, dtype=float).copy()
    # The raw CFD .dog force columns are treated as force per unit span.
    # Convert to total force over the modeled span before nondimensionalization.
    data_total[:, 7:9] *= span_m
    data_nd = data_total.copy()
    data_nd[:, 0] *= time_scale_factor
    data_nd[:, 1:3] /= diameter_m
    data_nd[:, 3:5] /= vel_scale
    data_nd[:, 5:7] /= acc_scale
    data_nd[:, 7:9] /= force_scale
    return replace(record, data=data_nd)


def nondimensionalize_records(
    records: list[CfdRecord],
    *,
    flow_speed_m_s: float,
    diameter_m: float,
    structural_frequency_hz: float,
    rho_kg_m3: float,
    span_m: float,
    time_scale_factor: float | None = None,
) -> list[CfdRecord]:
    return [
        nondimensionalize_record(
            record,
            flow_speed_m_s=flow_speed_m_s,
            diameter_m=diameter_m,
            structural_frequency_hz=structural_frequency_hz,
            rho_kg_m3=rho_kg_m3,
            span_m=span_m,
            time_scale_factor=time_scale_factor,
        )
        for record in records
    ]


def nondimensionalize_records_from_metadata(
    records: list[CfdRecord],
    metadata_df: pd.DataFrame,
    *,
    time_scale_mode: str = "fn",
    reference_stiffness_n_m: float | None = None,
    preserve_dimensional_time: bool = False,
) -> list[CfdRecord]:
    if time_scale_mode not in {"fn", "k_ref"}:
        raise ValueError("time_scale_mode must be either 'fn' or 'k_ref'.")
    ref_stiffness = None
    if time_scale_mode == "k_ref":
        if reference_stiffness_n_m is None:
            raise ValueError("reference_stiffness_n_m is required when time_scale_mode='k_ref'.")
        ref_stiffness = float(reference_stiffness_n_m)
        if not np.isfinite(ref_stiffness) or ref_stiffness <= 0.0:
            raise ValueError("reference_stiffness_n_m must be a positive finite number.")

    metadata = metadata_df.set_index("case_name").to_dict(orient="index")
    nondim_records: list[CfdRecord] = []
    for record in records:
        row = metadata.get(record.case_name)
        if row is None:
            raise ValueError(f"Missing nondimensionalization metadata for case {record.case_name}.")
        structural_frequency_hz = resolve_structural_frequency_hz(row, record.case_name)
        if time_scale_mode == "fn":
            time_scale_factor = float(2.0 * np.pi * structural_frequency_hz)
        else:
            stiffness_n_m = resolve_stiffness_n_m(
                row,
                record.case_name,
                structural_frequency_hz=structural_frequency_hz,
            )
            time_scale_factor = float(np.sqrt(stiffness_n_m / ref_stiffness))
        if preserve_dimensional_time:
            time_scale_factor = 1.0
        nondim_records.append(
            nondimensionalize_record(
                record,
                flow_speed_m_s=resolve_flow_speed_m_s(row, record.case_name),
                diameter_m=float(row["diameter_m"]),
                structural_frequency_hz=structural_frequency_hz,
                rho_kg_m3=float(row["rho_kg_m3"]),
                span_m=float(row["span_m"]),
                time_scale_factor=time_scale_factor,
            )
        )
    return nondim_records


__all__ = [
    "CfdRecord",
    "analyze_cfd_data",
    "apply_cleaning_manifest",
    "build_nondim_metadata",
    "build_cleaning_manifest",
    "default_nondim_metadata",
    "default_cleaning_manifest",
    "dominant_frequency",
    "displacement_peak_amplitudes",
    "infer_ur_from_path",
    "load_dog_file",
    "mean_displacement_amplitude",
    "mean_peak_displacement_amplitude",
    "resolve_computed_ur_value",
    "resolve_dry_structural_frequency_hz",
    "resolve_flow_speed_m_s",
    "resolve_stiffness_n_m",
    "nondimensionalize_records_from_metadata",
    "nondimensionalize_record",
    "nondimensionalize_records",
    "normalize_clean_action",
    "plot_case",
    "plot_summary",
    "plot_timeseries_grid",
    "power_spectrum",
    "psd_freq_limit",
    "record_summary",
    "remove_duplicate_timestamps",
    "summarize_duplicate_timestamps",
    "write_summary_csv",
]
