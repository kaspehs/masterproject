from __future__ import annotations

import os
from pathlib import Path
import re
import sys
import warnings

_MPLCONFIGDIR_DEFAULT = Path("/tmp/matplotlib")
_MPLCONFIGDIR_DEFAULT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR_DEFAULT))
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import numpy as np

try:
    import Experimental_Data.analyze_experimental_data as analysis
except ModuleNotFoundError as exc:
    if getattr(exc, "name", "") != "Experimental_Data":
        raise
    current_dir = Path(__file__).resolve().parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    import analyze_experimental_data as analysis


# Dataset location / selection
NPZ_DATASET_ROOT = Path("Experimental_Data/corrected_dataset")
NPZ_SPLITS = ["train", "val"]  # Set empty list to scan NPZ_GLOB recursively from root.
NPZ_GLOB = "*.npz"
RECURSIVE_WHEN_SPLITS_EMPTY = True

# Exclude files by test number parsed from filename, e.g. series_003_test3005.npz -> 3005.
EXCLUDE_TEST_NUMBERS: list[int] = list(getattr(analysis, "EXCLUDE_TEST_NUMBERS", []))

# Time-window plotting behavior (mirrors analyze_experimental_data style).
USE_RELATIVE_TIME = bool(getattr(analysis, "USE_RELATIVE_TIME", True))
FIRST_WINDOW_SECONDS = float(getattr(analysis, "FIRST_WINDOW_SECONDS", 10.0))
# Start/end trimming is intentionally not applied here; exported NPZ files are expected
# to be pretrimmed in export_corrected_dataset.py when needed.

# When true, prepend split to labels for easier visual grouping, e.g. "train:test3005".
PREFIX_LABEL_WITH_SPLIT = False

# Keep only one chunk per test for plotting/analysis.
KEEP_ONLY_ONE_CHUNK_PER_TEST = True
# Which chunk to keep when multiple are available for a test:
# - "first": earliest chunk (recommended)
# - "middle": middle chunk
# - "last": latest chunk
CHUNK_SELECTION_PER_TEST = "first"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_path_in_project(path: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    return (_project_root() / p).resolve()


def _extract_test_number(path: Path) -> int | None:
    stem = str(path.stem).lower()
    match = re.search(r"test(\d+)", stem)
    if match is None:
        return None
    return int(match.group(1))


def _filter_excluded_tests(paths: list[Path]) -> list[Path]:
    excluded_raw = list(EXCLUDE_TEST_NUMBERS)
    if not excluded_raw:
        return paths
    excluded = {int(v) for v in excluded_raw}
    kept: list[Path] = []
    for p in paths:
        test_no = _extract_test_number(Path(p))
        if test_no is not None and test_no in excluded:
            continue
        kept.append(Path(p))
    return kept


def _resolve_npz_files() -> list[Path]:
    root = _resolve_path_in_project(Path(NPZ_DATASET_ROOT))
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    files: list[Path] = []
    split_dirs = [str(s) for s in NPZ_SPLITS if str(s).strip()]
    if split_dirs:
        for split in split_dirs:
            split_dir = root / split
            if not split_dir.exists():
                continue
            files.extend(sorted(split_dir.glob(str(NPZ_GLOB))))
    else:
        if bool(RECURSIVE_WHEN_SPLITS_EMPTY):
            files.extend(sorted(root.rglob(str(NPZ_GLOB))))
        else:
            files.extend(sorted(root.glob(str(NPZ_GLOB))))

    files = [Path(p) for p in files if Path(p).is_file()]
    files = _filter_excluded_tests(files)

    unique: list[Path] = []
    seen: set[Path] = set()
    for p in files:
        if p in seen:
            continue
        seen.add(p)
        unique.append(p)
    if not unique:
        raise FileNotFoundError("No NPZ files selected. Check NPZ_DATASET_ROOT/NPZ_SPLITS/NPZ_GLOB.")

    by_test: dict[int, list[Path]] = {}
    for p in unique:
        test_no = _extract_test_number(p)
        if test_no is None:
            continue
        by_test.setdefault(int(test_no), []).append(p)
    dup_tests = {k: sorted(v) for k, v in by_test.items() if len(v) > 1}
    if dup_tests:
        warnings.warn(
            "Multiple NPZ files found for test numbers: "
            + ", ".join(f"{k} ({len(v)})" for k, v in sorted(dup_tests.items()))
            + "."
        )
        if bool(KEEP_ONLY_ONE_CHUNK_PER_TEST):
            mode = str(CHUNK_SELECTION_PER_TEST).strip().lower()
            selected: list[Path] = []
            for test_no, paths in sorted(by_test.items()):
                sorted_paths = sorted(paths)
                if mode == "first":
                    picked = sorted_paths[0]
                elif mode == "middle":
                    picked = sorted_paths[len(sorted_paths) // 2]
                elif mode == "last":
                    picked = sorted_paths[-1]
                else:
                    raise ValueError("CHUNK_SELECTION_PER_TEST must be one of: first, middle, last")
                selected.append(picked)
            others = [p for p in unique if _extract_test_number(p) is None]
            unique = sorted(selected + others)
    return unique


def _load_npz_array(npz_obj: np.lib.npyio.NpzFile, keys: list[str], *, role: str) -> np.ndarray:
    for key in keys:
        if key in npz_obj.files:
            return np.asarray(npz_obj[key], dtype=float).reshape(-1)
    raise KeyError(f"Missing NPZ key for {role}. Tried keys: {keys}")


def _load_optional_npz_array(npz_obj: np.lib.npyio.NpzFile, keys: list[str]) -> np.ndarray | None:
    for key in keys:
        if key in npz_obj.files:
            return np.asarray(npz_obj[key], dtype=float).reshape(-1)
    return None


def _fit_vector_length(values: np.ndarray, *, n: int, role: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == n:
        return arr
    if arr.size == 1:
        return np.full(n, float(arr[0]), dtype=float)
    if arr.size < n:
        raise ValueError(f"{role}: length {arr.size} is shorter than required length {n}.")
    return arr[:n]


def _label_from_path(npz_path: Path) -> str:
    test_no = _extract_test_number(npz_path)
    if test_no is not None:
        base = f"test{int(test_no)}"
    else:
        base = npz_path.stem
    split = npz_path.parent.name
    if bool(PREFIX_LABEL_WITH_SPLIT):
        return f"{split}:{base}"
    return base


def _process_npz(npz_path: Path) -> dict[str, object]:
    with np.load(npz_path, allow_pickle=False) as npz_obj:
        time = _load_npz_array(npz_obj, ["time", "a"], role="time")
        ypos = _load_npz_array(npz_obj, ["y", "b"], role="corrected displacement")
        fy_combined = _load_npz_array(
            npz_obj,
            ["calculated_force", "F_total", "c"],
            role="corrected calculated force",
        )
        yvel_opt = _load_optional_npz_array(npz_obj, ["dy", "e"])
        yacc_opt = _load_optional_npz_array(npz_obj, ["ddy", "d2y", "yacc"])
        ur_opt = _load_optional_npz_array(npz_obj, ["U_r", "u_r", "ur"])

    n = int(min(time.size, ypos.size, fy_combined.size))
    if n < 2:
        raise ValueError(f"{npz_path.name}: not enough samples for analysis.")
    time = analysis._fill_nonfinite_1d(time[:n], role=f"{npz_path.name}: time")
    ypos = analysis._fill_nonfinite_1d(ypos[:n], role=f"{npz_path.name}: corrected displacement")
    fy_combined = analysis._fill_nonfinite_1d(fy_combined[:n], role=f"{npz_path.name}: corrected calculated force")

    dt = analysis._infer_dt(time, role=f"{npz_path.name}: time")
    fs = 1.0 / dt
    nt = int(n)

    if yvel_opt is None:
        yvel = np.gradient(ypos, dt)
    else:
        yvel = analysis._fill_nonfinite_1d(
            _fit_vector_length(yvel_opt, n=nt, role=f"{npz_path.name}: velocity"),
            role=f"{npz_path.name}: velocity",
        )
    if yacc_opt is None:
        yacc = np.gradient(yvel, dt)
    else:
        yacc = analysis._fill_nonfinite_1d(
            _fit_vector_length(yacc_opt, n=nt, role=f"{npz_path.name}: acceleration"),
            role=f"{npz_path.name}: acceleration",
        )

    if ur_opt is None:
        warnings.warn(f"{npz_path.name}: missing U_r; using NaN.")
        ur_inst = np.full(nt, np.nan, dtype=float)
    else:
        ur_inst = analysis._fill_nonfinite_1d(
            _fit_vector_length(ur_opt, n=nt, role=f"{npz_path.name}: reduced velocity"),
            role=f"{npz_path.name}: reduced velocity",
        )

    ur = float(np.mean(ur_inst))
    umean = float(ur * analysis.FN * analysis.D)
    fdrag = np.zeros_like(fy_combined)

    if analysis._use_mean_u_plus_dy2_norm():
        coeff_norm_mode_used = "mean_u_plus_dy2"
        q_ref_vec = 0.5 * analysis.RUO * analysis.L * analysis.D * ((umean**2) + np.asarray(yvel, dtype=float) ** 2)
        q_ref_vec = np.maximum(q_ref_vec, float(analysis.COEFF_NORM_EPS))
    else:
        coeff_norm_mode_used = "mean_u"
        q_ref_scalar = max(0.5 * analysis.RUO * analysis.L * analysis.D * (umean**2), float(analysis.COEFF_NORM_EPS))
        q_ref_vec = np.full_like(np.asarray(ypos, dtype=float), q_ref_scalar, dtype=float)
    cdrag_coeff = np.zeros_like(fy_combined)
    cd = float(np.nanmean(cdrag_coeff))

    finite_y = np.isfinite(ypos)
    y_mean = float(np.nanmean(ypos)) if np.any(finite_y) else 0.0
    y_nd = (ypos - y_mean) / analysis.D

    m_added, m_inertia_removed = analysis._cf_inertia_masses()
    if m_inertia_removed != 0.0:
        f_inertia = m_inertia_removed * yacc
        fy_combined = fy_combined + f_inertia
    fy_combined = float(analysis.CF_SIGN) * fy_combined
    cfy_coeff = fy_combined / q_ref_vec

    if analysis._use_raw_force_signals():
        cfy = fy_combined
        cdrag = fdrag
    else:
        cfy = cfy_coeff
        cdrag = cdrag_coeff

    spcy, fhiy, nhiy = analysis._spec(ypos, fs)
    if analysis.NORMALIZE_SPECTRA:
        spcy = analysis._normalize_spectrum(spcy, eps=analysis.SPECTRUM_NORM_EPS)
    freq = np.asarray(fhiy, dtype=float)
    spec = np.asarray(spcy, dtype=float)
    ydomfreq, ydomfreq_std = analysis._dominant_frequency_and_spread(
        freq,
        spec,
        fmin=analysis.DOM_FREQ_MIN_HZ,
        fmax=analysis.DOM_FREQ_MAX_HZ,
    )
    phase_cfy_y_deg = analysis._phase_lag_deg_at_frequency(y_nd, cfy, fs=fs, target_hz=ydomfreq)
    phase_cdrag_y_deg = analysis._phase_lag_deg_at_frequency(y_nd, cdrag, fs=fs, target_hz=ydomfreq)

    time_plot = time - float(time[0]) if bool(USE_RELATIVE_TIME) else time
    t_end = float(FIRST_WINDOW_SECONDS) if bool(USE_RELATIVE_TIME) else float(time[0]) + float(FIRST_WINDOW_SECONDS)
    mask_early = np.asarray(time_plot <= t_end, dtype=bool)

    spc_cfy, f_cfy, n_cfy = analysis._spec(cfy, fs)
    spc_cdrag, f_cdrag, n_cdrag = analysis._spec(cdrag, fs)
    spc_ur, f_ur, n_ur = analysis._spec(ur_inst, fs)
    if analysis.NORMALIZE_SPECTRA:
        spc_cfy = analysis._normalize_spectrum(spc_cfy, eps=analysis.SPECTRUM_NORM_EPS)
        spc_cdrag = analysis._normalize_spectrum(spc_cdrag, eps=analysis.SPECTRUM_NORM_EPS)
        spc_ur = analysis._normalize_spectrum(spc_ur, eps=analysis.SPECTRUM_NORM_EPS)

    amp_stats = analysis._displacement_amplitude_stats(y_nd)

    return {
        "path": npz_path,
        "label": _label_from_path(npz_path),
        "time": np.asarray(time, dtype=float),
        "y": np.asarray(ypos, dtype=float),
        "yacc": np.asarray(yacc, dtype=float),
        "cfy_force": np.asarray(fy_combined, dtype=float),
        "q_ref_vec": np.asarray(q_ref_vec, dtype=float),
        "is_corrected_input": True,
        "time_plot": time_plot,
        "ur_inst": ur_inst,
        "y_nd": y_nd,
        "yvel": yvel,
        "cdrag": cdrag,
        "cfy": cfy,
        "mask_early": mask_early,
        "sp_disp": (fhiy, spcy, nhiy),
        "sp_cfy": (f_cfy, spc_cfy, n_cfy),
        "sp_cdrag": (f_cdrag, spc_cdrag, n_cdrag),
        "sp_ur": (f_ur, spc_ur, n_ur),
        "summary": {
            "umean": umean,
            "ur": ur,
            "cd": cd,
            "ydomfreq": ydomfreq,
            "ydomfreq_std": ydomfreq_std,
            "phase_cfy_y_deg": phase_cfy_y_deg,
            "phase_cdrag_y_deg": phase_cdrag_y_deg,
            "amp_mean": amp_stats["amp_mean"],
            "amp_std": amp_stats["amp_std"],
            "amp_min": amp_stats["amp_min"],
            "amp_max": amp_stats["amp_max"],
            "nt": nt,
            "dt": dt,
            "fs": fs,
            "m_added": m_added,
            "m_inertia_removed": float(m_inertia_removed),
            "coeff_norm_mode": coeff_norm_mode_used,
        },
    }


def _add_legends(axes) -> None:
    for ax in np.asarray(axes).reshape(-1):
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best", fontsize="small")


def _show_or_close() -> None:
    if "agg" in str(plt.get_backend()).lower():
        plt.close("all")
        return
    plt.show()


def main() -> None:
    npz_files = _resolve_npz_files()
    entries = [_process_npz(p) for p in npz_files]

    for entry in entries:
        s = entry["summary"]
        assert isinstance(s, dict)
        print(
            f"{entry['label']}: Umean={s['umean']:.6f} m/s, Ur={s['ur']:.6f}, "
            f"Cd={s['cd']:.6f}, Ydomfreq={s['ydomfreq']:.6f} +/- {s['ydomfreq_std']:.6f} Hz, "
            f"nt={s['nt']}, dt={s['dt']:.6f} s, Fs={s['fs']:.6f} Hz, "
            f"coeff_norm={s['coeff_norm_mode']}"
        )

    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(entries), 2)))
    n_rows = len(entries)

    # Figure 1: full timeseries, small multiples (rows=file, cols=metrics)
    fig1, axes1 = plt.subplots(n_rows, 5, figsize=(17, max(1.1 * n_rows, 2.0)), sharex="col")
    axes1 = np.atleast_2d(axes1)
    col_titles_full = [
        "Measured reduced velocity U_r",
        "Measured CF displacement y/D (mean removed)",
        "Measured CF acceleration y_ddot",
        f"Measured {analysis._drag_name()}",
        f"Measured {analysis._cf_name()}",
    ]
    col_titles_full[4] = f"Measured {analysis._cf_name()} ({analysis._cf_force_mode_label()})"
    for j, title in enumerate(col_titles_full):
        axes1[0, j].set_title(title)

    for i, entry in enumerate(entries):
        color = colors[i % len(colors)]
        label = str(entry["label"])
        t = np.asarray(entry["time_plot"])
        ax_ur, ax_y, ax_acc, ax_cd, ax_cf = axes1[i, 0], axes1[i, 1], axes1[i, 2], axes1[i, 3], axes1[i, 4]
        ax_ur.plot(t, np.asarray(entry["ur_inst"]), color=color)
        ax_y.plot(t, np.asarray(entry["y_nd"]), color=color)
        ax_acc.plot(t, np.asarray(entry["yacc"]), color=color)
        ax_cd.plot(t, np.asarray(entry["cdrag"]), color=color)
        ax_cf.plot(t, np.asarray(entry["cfy"]), color=color)
        for ax in (ax_ur, ax_y, ax_acc, ax_cd, ax_cf):
            ax.grid(True)
        ax_ur.set_ylabel(f"{label}\nU_r (-)")
        ax_y.set_ylabel("y/D (-)")
        ax_acc.set_ylabel("y_ddot (m/s^2)")
        ax_cd.set_ylabel(analysis._drag_label())
        ax_cf.set_ylabel(analysis._cf_label())

    for j in range(5):
        axes1[-1, j].set_xlabel("Time (s)")
    fig1.tight_layout()

    # Figure 3: phase + spectra overlay
    fig3, axes3 = plt.subplots(2, 2, figsize=(11, 8))
    ax_phase = axes3[0, 0]
    for i, entry in enumerate(entries):
        color = colors[i % len(colors)]
        label = str(entry["label"])
        y_nd = np.asarray(entry["y_nd"])
        yvel = np.asarray(entry["yvel"])
        ax_phase.plot(y_nd, yvel, color=color, alpha=0.35, label=label)
        ax_phase.scatter(y_nd[0], yvel[0], color=color, s=22, zorder=3)
    ax_phase.grid(True)
    ax_phase.set_xlabel("CF displacement y/D (mean removed, -)")
    ax_phase.set_ylabel("CF velocity dy/dt (m/s)")
    ax_phase.set_title("CF phase diagram")
    _add_legends([ax_phase])

    ax_fy = axes3[0, 1]
    ax_fd = axes3[1, 0]
    ax_ur = axes3[1, 1]
    for i, entry in enumerate(entries):
        color = colors[i % len(colors)]
        label = str(entry["label"])
        f_cfy, sp_cfy, n_cfy = entry["sp_cfy"]
        f_cdrag, sp_cdrag, n_cdrag = entry["sp_cdrag"]
        f_ur, sp_ur, n_ur = entry["sp_ur"]
        ax_fy.plot(np.asarray(f_cfy)[: int(n_cfy)], np.asarray(sp_cfy)[: int(n_cfy)], color=color, label=label)
        ax_fd.plot(
            np.asarray(f_cdrag)[: int(n_cdrag)],
            np.asarray(sp_cdrag)[: int(n_cdrag)],
            color=color,
            label=label,
        )
        ax_ur.plot(np.asarray(f_ur)[: int(n_ur)], np.asarray(sp_ur)[: int(n_ur)], color=color, label=label)
    ax_fy.set_xlim(0.1, analysis.SPECTRUM_PLOT_MAX_HZ)
    ax_fy.grid(True)
    ax_fy.set_xlabel("Frequency (Hz)")
    ax_fy.set_ylabel("Normalized spectrum" if analysis.NORMALIZE_SPECTRA else "Spectrum")
    ax_fy.set_title(f"{analysis._cf_name()} spectrum ({analysis._cf_force_mode_label()})")
    ax_fd.set_xlim(0.1, analysis.SPECTRUM_PLOT_MAX_HZ)
    ax_fd.grid(True)
    ax_fd.set_xlabel("Frequency (Hz)")
    ax_fd.set_ylabel("Normalized spectrum" if analysis.NORMALIZE_SPECTRA else "Spectrum")
    ax_fd.set_title(f"{analysis._drag_name()} spectrum")
    ax_ur.set_xlim(0.1, analysis.SPECTRUM_PLOT_MAX_HZ)
    ax_ur.grid(True)
    ax_ur.set_xlabel("Frequency (Hz)")
    ax_ur.set_ylabel("Normalized spectrum" if analysis.NORMALIZE_SPECTRA else "Spectrum")
    ax_ur.set_title("Reduced velocity spectrum")
    _add_legends([ax_fy, ax_fd, ax_ur])
    fig3.tight_layout()

    # Figure 4: first N seconds, small multiples (rows=file, cols=metrics)
    fig4, axes4 = plt.subplots(n_rows, 5, figsize=(17, max(1.0 * n_rows, 2.0)), sharex="col")
    axes4 = np.atleast_2d(axes4)
    col_titles = [
        f"Reduced velocity U_r (first {FIRST_WINDOW_SECONDS:g} s)",
        f"CF displacement y/D (mean removed, first {FIRST_WINDOW_SECONDS:g} s)",
        f"CF acceleration y_ddot (first {FIRST_WINDOW_SECONDS:g} s)",
        f"{analysis._drag_name()} (first {FIRST_WINDOW_SECONDS:g} s)",
        f"{analysis._cf_name()} (first {FIRST_WINDOW_SECONDS:g} s)",
    ]
    col_titles[4] = f"{analysis._cf_name()} ({analysis._cf_force_mode_label()}, first {FIRST_WINDOW_SECONDS:g} s)"
    for j, title in enumerate(col_titles):
        axes4[0, j].set_title(title)

    for i, entry in enumerate(entries):
        color = colors[i % len(colors)]
        label = str(entry["label"])
        mask = np.asarray(entry["mask_early"], dtype=bool)
        ax_ur, ax_y, ax_acc, ax_cd, ax_cf = axes4[i, 0], axes4[i, 1], axes4[i, 2], axes4[i, 3], axes4[i, 4]
        if np.any(mask):
            t = np.asarray(entry["time_plot"])[mask]
            ax_ur.plot(t, np.asarray(entry["ur_inst"])[mask], color=color)
            ax_y.plot(t, np.asarray(entry["y_nd"])[mask], color=color)
            ax_acc.plot(t, np.asarray(entry["yacc"])[mask], color=color)
            ax_cd.plot(t, np.asarray(entry["cdrag"])[mask], color=color)
            ax_cf.plot(t, np.asarray(entry["cfy"])[mask], color=color)
        for ax in (ax_ur, ax_y, ax_acc, ax_cd, ax_cf):
            ax.grid(True)
        ax_ur.set_ylabel(f"{label}\nU_r (-)")
        ax_y.set_ylabel("y/D (-)")
        ax_acc.set_ylabel("y_ddot (m/s^2)")
        ax_cd.set_ylabel(analysis._drag_label())
        ax_cf.set_ylabel(analysis._cf_label())

    for j in range(5):
        axes4[-1, j].set_xlabel("Time (s)")
    fig4.tight_layout()

    # Figure 5: summary trends vs mean reduced velocity
    entries_sorted = sorted(entries, key=lambda e: float(e["summary"]["ur"]))
    ur_vals = np.array([float(e["summary"]["ur"]) for e in entries_sorted], dtype=float)
    ydom_vals = np.array([float(e["summary"]["ydomfreq"]) for e in entries_sorted], dtype=float)
    ydom_std_vals = np.array([float(e["summary"]["ydomfreq_std"]) for e in entries_sorted], dtype=float)
    amp_mean_vals = np.array([float(e["summary"]["amp_mean"]) for e in entries_sorted], dtype=float)
    amp_std_vals = np.array([float(e["summary"]["amp_std"]) for e in entries_sorted], dtype=float)
    amp_min_vals = np.array([float(e["summary"]["amp_min"]) for e in entries_sorted], dtype=float)
    amp_max_vals = np.array([float(e["summary"]["amp_max"]) for e in entries_sorted], dtype=float)
    labels_sorted = [str(e["label"]) for e in entries_sorted]

    fig5, axes5 = plt.subplots(1, 2, figsize=(12, 4.6))
    ax_f, ax_a = axes5

    m_added_ref = float(analysis.REF_CA_FOR_FN_LINE) * 0.25 * np.pi * analysis.RUO * analysis.D * analysis.D * analysis.L
    f_n_ref = (1.0 / (2.0 * np.pi)) * np.sqrt(analysis.K / (analysis.M + m_added_ref))

    freq_mask = np.isfinite(ur_vals) & np.isfinite(ydom_vals) & np.isfinite(ydom_std_vals)
    ur_freq = ur_vals[freq_mask]
    ydom_freq = ydom_vals[freq_mask]
    ydom_std_freq = ydom_std_vals[freq_mask]
    labels_freq = [name for name, keep in zip(labels_sorted, freq_mask) if keep]

    ax_f.fill_between(
        ur_freq,
        ydom_freq - ydom_std_freq,
        ydom_freq + ydom_std_freq,
        color="tab:blue",
        alpha=0.30,
        edgecolor="none",
        linewidth=0.0,
        label="±1σ spectral spread",
    )
    ax_f.errorbar(
        ur_freq,
        ydom_freq,
        yerr=ydom_std_freq,
        fmt="o",
        linestyle="none",
        capsize=3.0,
        color="tab:blue",
        label="Dominant frequency",
    )
    ax_f.axhline(
        f_n_ref,
        color="black",
        linewidth=1.5,
        linestyle="-",
        label=f"Natural frequency (C_a={analysis.REF_CA_FOR_FN_LINE:.1f})",
    )
    for x, yv, name in zip(ur_freq, ydom_freq, labels_freq):
        ax_f.annotate(name, (x, yv), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax_f.grid(True)
    ax_f.set_xlabel("Mean reduced velocity U_r (-)")
    ax_f.set_ylabel("Dominant CF oscillation frequency (Hz)")
    ax_f.set_title("Mean oscillation frequency vs mean reduced velocity")
    ax_f.legend(loc="best", fontsize="small")

    amp_mask = (
        np.isfinite(ur_vals)
        & np.isfinite(amp_mean_vals)
        & np.isfinite(amp_std_vals)
        & np.isfinite(amp_min_vals)
        & np.isfinite(amp_max_vals)
    )
    ur_amp = ur_vals[amp_mask]
    amp_mean = amp_mean_vals[amp_mask]
    amp_std = amp_std_vals[amp_mask]
    amp_min = amp_min_vals[amp_mask]
    amp_max = amp_max_vals[amp_mask]

    ax_a.errorbar(
        ur_amp,
        amp_mean,
        yerr=amp_std,
        fmt="o",
        linestyle="none",
        capsize=3.0,
        color="tab:orange",
        label="Mean amplitude",
    )
    ax_a.fill_between(
        ur_amp,
        amp_min,
        amp_max,
        color="tab:orange",
        alpha=0.30,
        edgecolor="none",
        linewidth=0.0,
        label="Min-Max band",
    )
    ax_a.fill_between(
        ur_amp,
        amp_mean - amp_std,
        amp_mean + amp_std,
        color="tab:orange",
        alpha=0.45,
        edgecolor="none",
        linewidth=0.0,
        label="±1σ band",
    )
    ax_a.grid(True)
    ax_a.set_xlabel("Mean reduced velocity U_r (-)")
    ax_a.set_ylabel("Displacement amplitude |y/D| (-)")
    ax_a.set_title("Displacement amplitude vs mean reduced velocity")
    ax_a.legend(loc="best", fontsize="small")
    fig5.tight_layout()

    # Figure 6: phase-binned mean phase portrait with uncertainty
    fig6, axes6 = plt.subplots(1, 3, figsize=(14, 4.8))
    ax_phase_mean, ax_y_phase, ax_v_phase = axes6
    for i, entry in enumerate(entries):
        color = colors[i % len(colors)]
        label = str(entry["label"])
        try:
            phase_stats = analysis._phase_binned_phase_portrait(
                np.asarray(entry["y_nd"]),
                np.asarray(entry["yvel"]),
                n_bins=int(analysis.PHASE_MEAN_BINS),
                min_samples_per_bin=int(analysis.PHASE_MIN_SAMPLES_PER_BIN),
            )
        except ValueError as exc:
            warnings.warn(f"{label}: skipped mean phase plot ({exc})")
            continue

        theta = np.asarray(phase_stats["theta"], dtype=float)
        mean_y = np.asarray(phase_stats["mean_y"], dtype=float)
        mean_v = np.asarray(phase_stats["mean_v"], dtype=float)
        std_y = np.asarray(phase_stats["std_y"], dtype=float)
        std_v = np.asarray(phase_stats["std_v"], dtype=float)
        upper_y = np.asarray(phase_stats["upper_y"], dtype=float)
        upper_v = np.asarray(phase_stats["upper_v"], dtype=float)
        lower_y = np.asarray(phase_stats["lower_y"], dtype=float)
        lower_v = np.asarray(phase_stats["lower_v"], dtype=float)

        poly_x = np.concatenate([upper_y, lower_y[::-1]])
        poly_y = np.concatenate([upper_v, lower_v[::-1]])
        ax_phase_mean.fill(poly_x, poly_y, color=color, alpha=0.28, edgecolor="none")
        ax_phase_mean.plot(mean_y, mean_v, color=color, linewidth=1.8, label=label)

        phase_norm = theta / (2.0 * np.pi)
        ax_y_phase.fill_between(
            phase_norm,
            mean_y - std_y,
            mean_y + std_y,
            color=color,
            alpha=0.28,
            edgecolor="none",
        )
        ax_y_phase.plot(phase_norm, mean_y, color=color, linewidth=1.8, label=label)

        ax_v_phase.fill_between(
            phase_norm,
            mean_v - std_v,
            mean_v + std_v,
            color=color,
            alpha=0.28,
            edgecolor="none",
        )
        ax_v_phase.plot(phase_norm, mean_v, color=color, linewidth=1.8, label=label)

    ax_phase_mean.grid(True)
    ax_phase_mean.set_xlabel("CF displacement y/D (mean removed, -)")
    ax_phase_mean.set_ylabel("CF velocity dy/dt (m/s)")
    ax_phase_mean.set_title("Mean phase portrait with ±1σ tube")
    ax_y_phase.grid(True)
    ax_y_phase.set_xlabel("Phase / 2π (-)")
    ax_y_phase.set_ylabel("CF displacement y/D (mean removed, -)")
    ax_y_phase.set_title("Mean displacement vs phase")
    ax_v_phase.grid(True)
    ax_v_phase.set_xlabel("Phase / 2π (-)")
    ax_v_phase.set_ylabel("CF velocity dy/dt (m/s)")
    ax_v_phase.set_title("Mean velocity vs phase")
    _add_legends([ax_phase_mean])
    fig6.tight_layout()

    # Figure 7: hysteresis loops
    fig7, (ax_h_cf, ax_h_cd) = plt.subplots(1, 2, figsize=(12, 4.8))
    for i, entry in enumerate(entries):
        color = colors[i % len(colors)]
        label = str(entry["label"])
        y_nd = np.asarray(entry["y_nd"], dtype=float)
        cfy = np.asarray(entry["cfy"], dtype=float)
        cdrag = np.asarray(entry["cdrag"], dtype=float)
        mask_cf = np.isfinite(y_nd) & np.isfinite(cfy)
        mask_cd = np.isfinite(y_nd) & np.isfinite(cdrag)
        ax_h_cf.plot(y_nd[mask_cf], cfy[mask_cf], color=color, alpha=0.55, linewidth=1.0, label=label)
        ax_h_cd.plot(y_nd[mask_cd], cdrag[mask_cd], color=color, alpha=0.55, linewidth=1.0, label=label)
    ax_h_cf.grid(True)
    ax_h_cf.set_xlabel("Displacement y/D (mean removed, -)")
    ax_h_cf.set_ylabel(analysis._cf_label())
    ax_h_cf.set_title(f"Hysteresis loop: {analysis._cf_name()} vs y/D")
    ax_h_cd.grid(True)
    ax_h_cd.set_xlabel("Displacement y/D (mean removed, -)")
    ax_h_cd.set_ylabel(analysis._drag_label())
    ax_h_cd.set_title(f"Hysteresis loop: {analysis._drag_name()} vs y/D")
    _add_legends([ax_h_cf, ax_h_cd])
    fig7.tight_layout()

    # Figure 8: phase lag vs reduced velocity
    phase_cfy_vals = np.array([float(e["summary"]["phase_cfy_y_deg"]) for e in entries_sorted], dtype=float)
    phase_cdrag_vals = np.array([float(e["summary"]["phase_cdrag_y_deg"]) for e in entries_sorted], dtype=float)

    fig8, (ax_p_cf, ax_p_cd) = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)

    phase_cf_mask = np.isfinite(ur_vals) & np.isfinite(phase_cfy_vals)
    ur_phase_cf = ur_vals[phase_cf_mask]
    phase_cf = phase_cfy_vals[phase_cf_mask]
    labels_phase_cf = [name for name, keep in zip(labels_sorted, phase_cf_mask) if keep]
    ax_p_cf.scatter(
        ur_phase_cf,
        phase_cf,
        color="tab:blue",
        s=42,
        label=f"Phase({analysis._cf_name()}) - Phase(y)",
    )
    ax_p_cf.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    for x, yv, name in zip(ur_phase_cf, phase_cf, labels_phase_cf):
        ax_p_cf.annotate(name, (x, yv), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax_p_cf.grid(True)
    ax_p_cf.set_xlabel("Mean reduced velocity U_r (-)")
    ax_p_cf.set_ylabel("Phase lag (deg)")
    ax_p_cf.set_title(f"Phase lag: {analysis._cf_name()} relative to y")
    ax_p_cf.set_ylim(-185.0, 185.0)
    ax_p_cf.legend(loc="best", fontsize="small")

    phase_cd_mask = np.isfinite(ur_vals) & np.isfinite(phase_cdrag_vals)
    ur_phase_cd = ur_vals[phase_cd_mask]
    phase_cd = phase_cdrag_vals[phase_cd_mask]
    labels_phase_cd = [name for name, keep in zip(labels_sorted, phase_cd_mask) if keep]
    ax_p_cd.scatter(
        ur_phase_cd,
        phase_cd,
        color="tab:orange",
        s=42,
        label=f"Phase({analysis._drag_name()}) - Phase(y)",
    )
    ax_p_cd.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    for x, yv, name in zip(ur_phase_cd, phase_cd, labels_phase_cd):
        ax_p_cd.annotate(name, (x, yv), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax_p_cd.grid(True)
    ax_p_cd.set_xlabel("Mean reduced velocity U_r (-)")
    ax_p_cd.set_title(f"Phase lag: {analysis._drag_name()} relative to y")
    ax_p_cd.legend(loc="best", fontsize="small")
    fig8.tight_layout()

    _show_or_close()


if __name__ == "__main__":
    main()
