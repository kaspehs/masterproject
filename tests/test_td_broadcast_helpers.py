from __future__ import annotations

import unittest

try:
    import torch
    from HNN_helper import _broadcast_td_hidden_param_torch
except ModuleNotFoundError:
    torch = None
    _broadcast_td_hidden_param_torch = None


@unittest.skipUnless(torch is not None and _broadcast_td_hidden_param_torch is not None, "torch is required")
class TdBroadcastHelperTests(unittest.TestCase):
    def test_broadcasts_batch_vector_over_rollout_time_axis(self) -> None:
        like = torch.zeros(256, 251, 1)
        structural_mass = torch.arange(1, 257, dtype=like.dtype)

        out = _broadcast_td_hidden_param_torch(
            structural_mass,
            like=like,
            name="structural_mass",
        )

        self.assertEqual(tuple(out.shape), (256, 251, 1))
        self.assertTrue(torch.equal(out[:, 0, 0], structural_mass))
        self.assertTrue(torch.equal(out[:, -1, 0], structural_mass))

    def test_rejects_non_broadcastable_shapes(self) -> None:
        like = torch.zeros(256, 251, 1)
        bad_mass = torch.zeros(255)

        with self.assertRaises(ValueError):
            _broadcast_td_hidden_param_torch(bad_mass, like=like, name="structural_mass")
