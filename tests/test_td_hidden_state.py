from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - test environment dependent
    torch = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Data_Gen.prepare_cfd_npz_for_td_training import _trim_payload_arrays
from Data_Gen.td_hidden_state import (
    compute_theta_series,
    detect_burnin_start_index,
    initial_hidden_sigmas,
    initial_phi_dy,
    replay_hidden_state_with_cfd_motion,
    wrap_phase,
)
if torch is not None:
    from HNN_helper import td_baseline_step_torch
else:  # pragma: no cover - test environment dependent
    td_baseline_step_torch = None


class DetectBurninStartIndexTests(unittest.TestCase):
    def test_detects_first_valid_regular_interval(self) -> None:
        time_dim = np.asarray([0.0, 0.5, 1.0, 1.5, 2.0], dtype=float)
        rel_std = np.asarray([2.0e-4, 5.0e-5, 4.0e-5, 5.0e-5, 4.0e-5], dtype=float)
        idx = detect_burnin_start_index(time_dim, rel_std, threshold=1.0e-4, persistence_seconds=1.0)
        self.assertEqual(idx, 1)

    def test_uses_irregular_time_spacing(self) -> None:
        time_dim = np.asarray([0.0, 0.3, 0.9, 1.4, 1.8], dtype=float)
        rel_std = np.asarray([2.0e-4, 5.0e-5, 6.0e-5, 5.0e-5, 5.0e-5], dtype=float)
        idx = detect_burnin_start_index(time_dim, rel_std, threshold=1.0e-4, persistence_seconds=1.0)
        self.assertEqual(idx, 1)

    def test_returns_none_when_threshold_not_sustained(self) -> None:
        time_dim = np.asarray([0.0, 0.5, 1.0, 1.5], dtype=float)
        rel_std = np.asarray([2.0e-4, 5.0e-5, 2.0e-4, 5.0e-5], dtype=float)
        idx = detect_burnin_start_index(time_dim, rel_std, threshold=1.0e-4, persistence_seconds=1.0)
        self.assertIsNone(idx)


class TrimPayloadTests(unittest.TestCase):
    def test_trim_payload_slices_only_time_aligned_arrays(self) -> None:
        payload = {
            "time_dim": np.asarray([10.0, 11.0, 12.0, 13.0], dtype=float),
            "y_disp_dim": np.asarray([1.0, 2.0, 3.0, 4.0], dtype=float),
            "U_r_computed_series": np.asarray([5.0, 5.0, 5.0, 5.0], dtype=float),
            "num_rows": np.asarray(4, dtype=int),
            "flow_speed_m_s": np.asarray(2.0, dtype=float),
        }
        trimmed = _trim_payload_arrays(payload, 2)
        np.testing.assert_allclose(trimmed["time_dim"], np.asarray([12.0, 13.0], dtype=float))
        np.testing.assert_allclose(trimmed["y_disp_dim"], np.asarray([3.0, 4.0], dtype=float))
        np.testing.assert_allclose(trimmed["U_r_computed_series"], np.asarray([5.0, 5.0], dtype=float))
        self.assertEqual(int(np.asarray(trimmed["num_rows"]).reshape(())), 2)
        self.assertEqual(float(np.asarray(trimmed["flow_speed_m_s"]).reshape(())), 2.0)


class RepresentativeReplayTests(unittest.TestCase):
    def test_theta0_zero_yields_zero_initial_theta(self) -> None:
        time = np.asarray([0.0, 0.1, 0.2, 0.3], dtype=float)
        dy = np.asarray([0.2, 0.2, 0.2, 0.2], dtype=float)
        ddy = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=float)
        case_like = {
            "dy_dim": dy,
            "ddy_dim": ddy,
            "dt_dim": np.asarray(0.1, dtype=float),
        }
        sig_dy_loc0, sig_ddy_loc0 = initial_hidden_sigmas(
            case_like=case_like,
            start_idx=0,
            flow_speed_m_s=1.0,
            n_memory=4,
            mode="local_rms",
            window_seconds=None,
        )
        phi_dy0 = initial_phi_dy(
            dy0=float(dy[0]),
            ddy0=float(ddy[0]),
            sig_dy_loc0=sig_dy_loc0,
            sig_ddy_loc0=sig_ddy_loc0,
            flow_speed_m_s=1.0,
        )
        phi_vy0 = float(wrap_phase(np.asarray([phi_dy0]))[0])
        sim = replay_hidden_state_with_cfd_motion(
            time=time,
            y=np.zeros_like(time),
            dy=dy,
            ddy=ddy,
            flow_speed_m_s=1.0,
            rho_kg_m3=1000.0,
            diameter_m=1.0,
            params={
                "Cv": 1.2,
                "Cd": 1.1,
                "Ca": 1.0,
                "C": 1.0e-4,
                "fhat_min": 0.11,
                "fhat0": 0.18,
                "fhat_max": 0.26,
            },
            phi_vy0=phi_vy0,
            sig_dy_loc0=sig_dy_loc0,
            sig_ddy_loc0=sig_ddy_loc0,
            n_memory=4,
        )
        theta = compute_theta_series(
            sim["dy"],
            sim["ddy"],
            sim["phi_vy"],
            sim["sig_dy_loc"],
            sim["sig_ddy_loc"],
            flow_speed_m_s=1.0,
        )
        self.assertAlmostEqual(float(theta[0]), 0.0, places=12)

    def test_torch_td_step_matches_numpy_reference(self) -> None:
        if torch is None:
            self.skipTest("torch is not installed in this test environment")
        params = {
            "Cv": 1.2,
            "Cd": 1.1,
            "Ca": 1.0,
            "C": 1.0e-4,
            "fhat_min": 0.11,
            "fhat0": 0.18,
            "fhat_max": 0.26,
        }
        sim = replay_hidden_state_with_cfd_motion(
            time=np.asarray([0.0, 0.1], dtype=float),
            y=np.asarray([0.0, 0.01], dtype=float),
            dy=np.asarray([0.2, 0.21], dtype=float),
            ddy=np.asarray([0.3, 0.31], dtype=float),
            flow_speed_m_s=1.0,
            rho_kg_m3=1000.0,
            diameter_m=1.0,
            params=params,
            phi_vy0=0.05,
            sig_dy_loc0=0.2,
            sig_ddy_loc0=0.3,
            n_memory=4,
        )
        force_next, td_context_next = td_baseline_step_torch(
            velocity=torch.tensor([[0.2]], dtype=torch.float32),
            acceleration=torch.tensor([[0.3]], dtype=torch.float32),
            td_context=torch.tensor([[0.3, 0.05, 0.2, 0.3, 1.0]], dtype=torch.float32),
            dt=0.1,
            rho=1000.0,
            diameter=1.0,
            params={
                "Cv": params["Cv"],
                "Cd": params["Cd"],
                "Ca": params["Ca"],
                "fhat0": params["fhat0"],
                "fhat_min": params["fhat_min"],
                "fhat_max": params["fhat_max"],
                "n_memory": 4.0,
            },
        )
        self.assertAlmostEqual(float(force_next[0, 0]), float(sim["F_total"][1]), places=5)
        self.assertAlmostEqual(float(td_context_next[0, 1]), float(sim["phi_vy"][1]), places=5)
        self.assertAlmostEqual(float(td_context_next[0, 2]), float(sim["sig_dy_loc"][1]), places=5)
        self.assertAlmostEqual(float(td_context_next[0, 3]), float(sim["sig_ddy_loc"][1]), places=5)


if __name__ == "__main__":
    unittest.main()
