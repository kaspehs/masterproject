from __future__ import annotations

from pathlib import Path
import re
import sys
from typing import Any

import numpy as np


def bootstrap_import_paths(script_file: str | Path) -> tuple[Path, Path]:
    script_dir = Path(script_file).resolve().parent
    project_root = script_dir.parent
    for p in (project_root, script_dir):
        p_str = str(p)
        if p_str not in sys.path:
            sys.path.insert(0, p_str)
    return script_dir, project_root


def import_analysis_and_phase(script_file: str | Path):
    bootstrap_import_paths(script_file)
    try:
        import Experimental_Data.analyze_experimental_data as analysis
        import Experimental_Data.phase_analysis as phase
    except ModuleNotFoundError:
        import analyze_experimental_data as analysis
        import phase_analysis as phase
    return analysis, phase


def _build_extracted_fallback(analysis_module):
    class _ExtractedFallback:
        DATA_VARIABLE = analysis_module.DATA_VARIABLE
        USE_RELATIVE_TIME = bool(analysis_module.USE_RELATIVE_TIME)
        PLOT_FIRST_SECONDS = float(getattr(analysis_module, "FIRST_WINDOW_SECONDS", 10.0))
        FIGURE_WIDTH = 12.0

        @classmethod
        def _sync_analysis_config(cls) -> None:
            analysis_module.DATA_VARIABLE = cls.DATA_VARIABLE
            analysis_module.USE_RELATIVE_TIME = bool(cls.USE_RELATIVE_TIME)
            analysis_module.FIRST_WINDOW_SECONDS = float(cls.PLOT_FIRST_SECONDS)

        @classmethod
        def _process_file(cls, mat_file: Path) -> dict[str, object]:
            cls._sync_analysis_config()
            entry = analysis_module._process_file(mat_file)

            t = np.asarray(entry.get("time_plot", entry.get("time", [])), dtype=float).reshape(-1)
            ur = np.asarray(entry.get("ur_inst", []), dtype=float).reshape(-1)
            y = np.asarray(entry.get("y", []), dtype=float).reshape(-1)
            yacc = np.asarray(entry.get("yacc", []), dtype=float).reshape(-1)
            fy1 = np.asarray(entry.get("fy1_component", []), dtype=float).reshape(-1)
            fy2 = np.asarray(entry.get("fy2_component", []), dtype=float).reshape(-1)
            mask = np.asarray(entry.get("mask_early", []), dtype=bool).reshape(-1)

            n = int(min(t.size, ur.size, y.size, yacc.size, fy1.size, fy2.size))
            if n < 2:
                raise ValueError(f"{Path(mat_file).name}: not enough aligned samples in fallback phase loader.")

            t = t[:n]
            ur = ur[:n]
            y = y[:n]
            yacc = yacc[:n]
            fy1 = fy1[:n]
            fy2 = fy2[:n]
            if mask.size != n:
                mask = np.ones(n, dtype=bool)
            else:
                mask = mask[:n]

            fy_diff = np.asarray(fy2 - fy1, dtype=float)
            dt = float(np.median(np.diff(t)))
            if not np.isfinite(dt) or dt <= 0.0:
                dt = float(entry.get("summary", {}).get("dt", np.nan))
            if not np.isfinite(dt) or dt <= 0.0:
                raise ValueError(f"{Path(mat_file).name}: invalid dt in fallback phase loader.")

            channels = [
                ("Reduced velocity U_r (-)", np.asarray(ur, dtype=float)),
                ("Fy1 (scaled LB/LA, N)", np.asarray(fy1, dtype=float)),
                ("Fy2 (scaled LB/LA, N)", np.asarray(fy2, dtype=float)),
                ("Fy2 - Fy1 (scaled LB/LA, N)", np.asarray(fy_diff, dtype=float)),
                ("Displacement y (m)", np.asarray(y, dtype=float)),
                ("Acceleration y_ddot (m/s^2)", np.asarray(yacc, dtype=float)),
            ]

            return {
                "path": Path(entry.get("path", mat_file)),
                "label": str(entry.get("label", Path(mat_file).stem)),
                "t": np.asarray(t, dtype=float),
                "mask": np.asarray(mask, dtype=bool),
                "dt": float(dt),
                "channels": channels,
                "_fallback_from_analysis": True,
            }

        @staticmethod
        def _compute_derivatives(values: np.ndarray, *, dt: float) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
            y = np.asarray(values, dtype=float).reshape(-1)
            if y.size < 2:
                zeros = np.zeros_like(y)
                return zeros, zeros, {"mode": "fallback"}
            y_dot = np.gradient(y, float(dt))
            y_ddot = np.gradient(y_dot, float(dt))
            return np.asarray(y_dot, dtype=float), np.asarray(y_ddot, dtype=float), {"mode": "fallback"}

    return _ExtractedFallback


def import_analysis_and_extracted(
    script_file: str | Path,
    *,
    allow_extracted_fallback: bool = True,
    print_fallback_message: bool = True,
) -> tuple[Any, Any, bool]:
    bootstrap_import_paths(script_file)

    try:
        import Experimental_Data.analyze_experimental_data as analysis
    except ModuleNotFoundError:
        import analyze_experimental_data as analysis

    extracted = None
    try:
        import Experimental_Data.plot_extracted_channels as extracted  # type: ignore
    except ModuleNotFoundError:
        try:
            import plot_extracted_channels as extracted  # type: ignore
        except ModuleNotFoundError:
            extracted = None

    used_fallback = False
    if extracted is None and bool(allow_extracted_fallback):
        extracted = _build_extracted_fallback(analysis)
        used_fallback = True
        if bool(print_fallback_message):
            print(
                "phase_analysis: 'plot_extracted_channels.py' not found; "
                "using analyze_experimental_data fallback for processing/derivatives."
            )

    return analysis, extracted, used_fallback


def extract_test_number(path: Path) -> int | None:
    stem = str(path.stem).lower()
    match = re.search(r"test(\d+)", stem)
    if match is None:
        return None
    return int(match.group(1))


def filter_excluded_tests(paths: list[Path], excluded_numbers: list[int] | tuple[int, ...]) -> list[Path]:
    excluded_raw = list(excluded_numbers)
    if not excluded_raw:
        return [Path(p) for p in paths]
    excluded = {int(v) for v in excluded_raw}
    kept: list[Path] = []
    for p in paths:
        test_no = extract_test_number(Path(p))
        if test_no is not None and test_no in excluded:
            continue
        kept.append(Path(p))
    return kept


def resolve_existing_dir(path_like: Path | str, *, script_file: str | Path) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        if p.exists() and p.is_dir():
            return p
        raise FileNotFoundError(f"Directory does not exist: {p}")

    # Try relative to current working directory first.
    cand_cwd = (Path.cwd() / p).resolve()
    if cand_cwd.exists() and cand_cwd.is_dir():
        return cand_cwd

    # Then try relative to the script directory.
    script_dir = Path(script_file).resolve().parent
    cand_script = (script_dir / p).resolve()
    if cand_script.exists() and cand_script.is_dir():
        return cand_script

    raise FileNotFoundError(
        f"Directory does not exist: {p} (tried {cand_cwd} and {cand_script})"
    )

