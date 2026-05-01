from __future__ import annotations

import unittest
from unittest import mock

try:
    import torch
    from HNN_helper import PHVIV, tanh_bounded_output_torch
except ModuleNotFoundError:
    torch = None
    PHVIV = None
    tanh_bounded_output_torch = None


@unittest.skipIf(torch is None or PHVIV is None, "torch not available")
class TanhBoundedOutputTests(unittest.TestCase):
    def test_tanh_bound_preserves_identity_near_zero_and_caps_magnitude(self) -> None:
        raw = torch.tensor([[-0.5], [0.0], [0.5], [10.0]], dtype=torch.float32)
        bounded = tanh_bounded_output_torch(raw, 2.0)
        expected = 2.0 * torch.tanh(raw / 2.0)
        self.assertTrue(torch.allclose(bounded, expected))
        self.assertLessEqual(float(torch.max(torch.abs(bounded))), 2.0)

    def test_phviv_learned_force_coeff_applies_bound_in_coefficient_mode(self) -> None:
        model = PHVIV(
            dt=0.1,
            force_output="coefficient",
            coefficient_output_bound=2.0,
            use_reduced_velocity=False,
            use_stochastic_process_noise=False,
        )
        raw = torch.tensor([[10.0]], dtype=torch.float32)
        with mock.patch.object(model, "_force_net_raw", return_value=raw):
            coeff = model.learned_force_coeff(torch.zeros((1, 2), dtype=torch.float32))
        self.assertTrue(torch.allclose(coeff, 2.0 * torch.tanh(raw / 2.0)))


if __name__ == "__main__":
    unittest.main()
