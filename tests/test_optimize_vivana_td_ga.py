from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from CFD_Data import optimize_vivana_td_ga as opt


def _dominant_frequency(signal: np.ndarray, dt: float) -> float:
    if dt <= 0.0:
        return float("nan")
    signal_arr = np.asarray(signal, dtype=float).reshape(-1)
    if signal_arr.size < 4:
        return float("nan")
    centered = signal_arr - np.mean(signal_arr)
    if np.allclose(centered, 0.0):
        return float("nan")
    freqs = np.fft.rfftfreq(int(centered.size), d=float(dt))
    power = np.abs(np.fft.rfft(centered)) ** 2
    if freqs.size < 2 or power.size < 2:
        return float("nan")
    power = np.asarray(power, dtype=float)
    power[0] = 0.0
    mask = np.isfinite(freqs) & np.isfinite(power) & (freqs > 0.0)
    if np.count_nonzero(mask) < 1:
        return float("nan")
    valid_indices = np.flatnonzero(mask)
    peak_index = int(valid_indices[int(np.argmax(power[mask]))])
    peak_power = float(power[peak_index])
    if not np.isfinite(peak_power) or peak_power <= 0.0:
        return float("nan")

    interpolated_index = float(peak_index)
    if 1 <= peak_index < (power.size - 1):
        y_prev = float(power[peak_index - 1])
        y_peak = float(power[peak_index])
        y_next = float(power[peak_index + 1])
        denom = y_prev - 2.0 * y_peak + y_next
        if np.isfinite(denom) and abs(denom) > 1.0e-18:
            delta = 0.5 * (y_prev - y_next) / denom
            if np.isfinite(delta):
                interpolated_index += float(np.clip(delta, -1.0, 1.0))

    df = float(freqs[1] - freqs[0])
    if not np.isfinite(df) or df <= 0.0:
        return float(freqs[peak_index])
    return float(max(interpolated_index * df, 0.0))


def _relative_error(model_value: float, true_value: float, eps: float = 1e-12) -> float:
    if not np.isfinite(true_value) or not np.isfinite(model_value):
        return float("nan")
    denom = abs(true_value)
    if denom <= eps:
        return float("nan")
    return float((model_value - true_value) / (denom + eps))


class OptimizeVivanaTdGaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset_root = opt.REPO_ROOT / "CFD_Data" / "npz_exports_td_burnin_trimmed"
        cls.metric_patch = mock.patch.object(
            opt.spectral_helpers,
            "_get_hnn_metric_helpers",
            return_value=(_dominant_frequency, _relative_error),
        )
        cls.metric_patch.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.metric_patch.stop()

    def test_reduce_grouped_metric_values_per_ur_mean(self) -> None:
        grouped = {
            2.0: [1.0, 3.0],
            4.0: [2.0],
        }
        reduced = opt.reduce_grouped_metric_values(grouped, "per_ur_mean")
        self.assertAlmostEqual(reduced, 2.0)

    def test_reduce_grouped_metric_values_per_case_mean(self) -> None:
        grouped = {
            2.0: [1.0, 3.0],
            4.0: [2.0],
        }
        reduced = opt.reduce_grouped_metric_values(grouped, "per_case_mean")
        self.assertAlmostEqual(reduced, 2.0)

    def test_combine_metric_aggregates_mae_and_mse(self) -> None:
        metric_aggregates = {
            opt.METRIC_LABELS[0]: 1.0,
            opt.METRIC_LABELS[1]: 2.0,
            opt.METRIC_LABELS[2]: 3.0,
            opt.METRIC_LABELS[3]: 4.0,
        }
        self.assertAlmostEqual(opt.combine_metric_aggregates(metric_aggregates, "mae"), 2.5)
        self.assertAlmostEqual(opt.combine_metric_aggregates(metric_aggregates, "mse"), 7.5)

    def test_repair_candidate_clamps_and_orders_fhat_values(self) -> None:
        bounds = {
            "Cv": (0.8, 1.6),
            "Cd": (0.4, 1.4),
            "Ca": (0.6, 1.4),
            "fhat0": (0.10, 0.22),
            "fhat_min": (0.05, 0.18),
            "fhat_max": (0.16, 0.32),
        }
        candidate = np.asarray([2.0, -1.0, np.nan, 0.17, 0.17, 0.17], dtype=float)
        repaired = opt.repair_candidate(candidate, bounds)
        self.assertGreaterEqual(repaired[0], 0.8)
        self.assertLessEqual(repaired[0], 1.6)
        self.assertGreaterEqual(repaired[1], 0.4)
        self.assertLessEqual(repaired[1], 1.4)
        self.assertGreaterEqual(repaired[2], 0.6)
        self.assertLessEqual(repaired[2], 1.4)
        self.assertLess(repaired[4], repaired[3])
        self.assertLess(repaired[3], repaired[5])
        self.assertGreaterEqual(repaired[5], 0.16)

    def test_smoke_run_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with mock.patch.multiple(
                opt,
                CV_BOUNDS=(1.0, 2.0),
                CD_BOUNDS=(1.0, 2.0),
                CA_BOUNDS=(0.5, 2.0),
                FHAT0_BOUNDS=(0.05, 0.30),
                FHAT_MIN_BOUNDS=(0.02, 0.30),
                FHAT_MAX_BOUNDS=(0.05, 0.40),
            ):
                result = opt.run_configured_optimization(
                    dataset_root=self.dataset_root,
                    output_dir=Path(tmp_dir),
                    max_cases=2,
                    population_size=4,
                    generations=1,
                    elite_count=1,
                    tournament_size=2,
                    random_seed=0,
                )
            self.assertTrue(np.isfinite(result.best_evaluation.objective))
            self.assertFalse(result.best_evaluation.failure)
            for filename in ("best_result.json", "history.csv", "run_config.json", "summary.txt"):
                self.assertTrue((Path(tmp_dir) / filename).exists(), msg=filename)
            best_result = json.loads((Path(tmp_dir) / "best_result.json").read_text())
            self.assertIn("best_params", best_result)
            self.assertTrue(np.isfinite(float(best_result["best_objective"])))

    def test_baseline_matches_manual_block8_style_aggregation(self) -> None:
        context = opt.build_evaluation_context(
            dataset_root=self.dataset_root,
            max_cases=3,
            excluded_ur_values=[],
        )
        evaluated = opt.evaluate_vivana_td_params(
            dict(opt.BASELINE_PARAMS),
            context=context,
            aggregate_weighting="per_ur_mean",
            loss_mode="mae",
            mass_source="dry",
            td_memory_tau_s="tau_over_tref:4",
        )
        self.assertFalse(evaluated.failure)

        manual_grouped = {label: {} for label in opt.METRIC_LABELS}
        continuation_state = None
        for reference_idx, reference_series in enumerate(context.reference_series):
            if reference_idx == 0:
                rollout_duration_s = context.first_case_duration_s
                rollout_transient_seconds = context.first_case_transient_s
            else:
                rollout_duration_s = context.continuation_duration_s
                rollout_transient_seconds = context.continuation_transient_s
            target_ur = opt.reference_ur_value(reference_series)
            ur_key = float(round(target_ur, 6))
            viv_rollout = opt.spectral_helpers.generate_vivana_summary_rollout(
                reference_series,
                target_ur,
                generation_dt=context.summary_generation_dt,
                generation_duration_s=rollout_duration_s,
                transient_seconds=rollout_transient_seconds,
                td_params=dict(opt.BASELINE_PARAMS),
                mass_source="dry",
                initial_state=continuation_state,
                td_memory_tau_s="tau_over_tref:4",
            )
            continuation_state = dict(viv_rollout["final_state"])
            aligned = opt.align_reference_series_to_rollout(reference_series, viv_rollout)
            self.assertIsNotNone(aligned)
            metrics = opt.spectral_helpers.compute_validation_style_error_metrics(
                time=aligned["time"],
                y_true=aligned["y_true"],
                y_pred=aligned["y_pred"],
                force_true=aligned["force_true"],
                force_pred=aligned["force_pred"],
            )
            for metric_key, metric_label in opt.ACTIVE_VALIDATION_ERROR_KEYS:
                value = float(metrics.get(metric_key, float("nan")))
                if np.isfinite(value):
                    manual_grouped[metric_label].setdefault(ur_key, []).append(value)

        manual_metric_aggregates: dict[str, float] = {}
        for metric_label in opt.METRIC_LABELS:
            ur_means = []
            for ur_key in sorted(manual_grouped[metric_label].keys()):
                values = np.asarray(manual_grouped[metric_label][ur_key], dtype=float).reshape(-1)
                values = values[np.isfinite(values)]
                if values.size == 0:
                    continue
                ur_means.append(float(np.mean(values)))
            manual_metric_aggregates[metric_label] = float(np.mean(np.asarray(ur_means, dtype=float)))
        manual_objective = float(np.mean(np.asarray([manual_metric_aggregates[label] for label in opt.METRIC_LABELS], dtype=float)))

        for metric_label in opt.METRIC_LABELS:
            self.assertAlmostEqual(
                evaluated.metric_aggregates[metric_label],
                manual_metric_aggregates[metric_label],
                places=10,
            )
        self.assertAlmostEqual(evaluated.objective, manual_objective, places=10)


if __name__ == "__main__":
    unittest.main()
