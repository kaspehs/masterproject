from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

try:
    import torch
    from HNN_helper import parse_config
    from methods.hnn.trainer import (
        _build_td_correction_hnn_loaders,
        _displacement_std_error_torch,
        _dominant_frequency_error_torch,
        _psd_error_torch,
        _resolve_td_rollout_loss_settings,
        _td_correction_rollout_losses_from_batch,
    )
except ModuleNotFoundError:
    torch = None
    parse_config = None
    _build_td_correction_hnn_loaders = None
    _displacement_std_error_torch = None
    _dominant_frequency_error_torch = None
    _psd_error_torch = None
    _resolve_td_rollout_loss_settings = None
    _td_correction_rollout_losses_from_batch = None


def _dummy_td_model() -> SimpleNamespace:
    return SimpleNamespace(D=1.0, nn_p_scale=1.0, input_scaling_mode="current")


def _make_rollout_batch(z_traj: torch.Tensor) -> tuple[torch.Tensor, ...]:
    dtype = z_traj.dtype
    device = z_traj.device
    batch_size, steps, _state_dim = z_traj.shape
    z0 = z_traj[:, 0, :]
    t_seq = torch.arange(steps, dtype=dtype, device=device).view(1, -1).expand(batch_size, -1)
    ur0 = torch.ones((batch_size, 1), dtype=dtype, device=device)
    td_context0 = torch.zeros((batch_size, 5), dtype=dtype, device=device)
    mass0 = torch.ones(batch_size, dtype=dtype, device=device)
    damping0 = torch.ones(batch_size, dtype=dtype, device=device)
    stiffness0 = torch.ones(batch_size, dtype=dtype, device=device)
    return (z0, t_seq, z_traj, ur0, td_context0, mass0, damping0, stiffness0)


def _rollout_return_tuple(z_pred: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, steps, _state_dim = z_pred.shape
    aux_shape = (batch_size, max(steps - 1, 0), 1)
    zeros = torch.zeros(aux_shape, dtype=z_pred.dtype, device=z_pred.device)
    return z_pred, zeros, zeros, zeros, zeros


def _make_td_traj(length: int, *, ur: float) -> dict[str, np.ndarray]:
    t = np.arange(length, dtype=np.float32)
    y = np.linspace(0.0, 1.0, length, dtype=np.float32)
    dy = np.linspace(0.1, 0.2, length, dtype=np.float32)
    td_context = np.zeros((length, 5), dtype=np.float32)
    td_context[:, 4] = np.float32(ur)
    return {
        "y": y,
        "dy": dy,
        "t": t,
        "ur": np.full((length,), ur, dtype=np.float32),
        "force_per_m": np.linspace(0.0, 0.5, length, dtype=np.float32),
        "td_context": td_context,
        "dry_mass_kg": np.asarray(1.0, dtype=np.float32),
        "effective_mass_kg": np.asarray(1.5, dtype=np.float32),
        "damping_c": np.asarray(0.1, dtype=np.float32),
        "stiffness_n_m": np.asarray(2.0, dtype=np.float32),
    }


def _manual_band_mask(freqs: np.ndarray, true_psd: np.ndarray, peak_rel_bandwidth: float) -> np.ndarray:
    base_mask = np.isfinite(freqs) & (freqs > 0.0)
    if peak_rel_bandwidth <= 0.0:
        return base_mask
    pos_freqs = freqs[base_mask]
    peak_freq = float(pos_freqs[int(np.argmax(true_psd[base_mask]))])
    if freqs.size > 1:
        freq_res = float(freqs[1] - freqs[0])
    else:
        freq_res = float("nan")
    min_half_width = max(0.5 * freq_res if np.isfinite(freq_res) and freq_res > 0.0 else 0.0, 1.0e-12)
    half_width = max(peak_freq * float(peak_rel_bandwidth), min_half_width)
    band_mask = base_mask & (np.abs(freqs - peak_freq) <= half_width)
    if np.count_nonzero(band_mask) < 2:
        return base_mask
    return band_mask


def _manual_spectrum(signal: np.ndarray, *, dt: float, use_hann_window: bool) -> tuple[np.ndarray, np.ndarray]:
    centered = np.asarray(signal, dtype=float).reshape(-1) - float(np.mean(signal))
    if use_hann_window:
        centered = centered * np.hanning(centered.size)
    freqs = np.fft.rfftfreq(centered.size, d=float(dt))
    psd = np.abs(np.fft.rfft(centered)) ** 2
    return freqs, psd


def _manual_psd_loss(
    true_signal: np.ndarray,
    pred_signal: np.ndarray,
    *,
    dt: float,
    peak_rel_bandwidth: float,
    use_hann_window: bool,
    relative: bool,
    eps: float = 1.0e-12,
) -> float:
    freqs, true_psd = _manual_spectrum(true_signal, dt=dt, use_hann_window=use_hann_window)
    _freqs_pred, pred_psd = _manual_spectrum(pred_signal, dt=dt, use_hann_window=use_hann_window)
    band_mask = _manual_band_mask(freqs, true_psd, peak_rel_bandwidth)
    true_amp = np.sqrt(np.clip(true_psd[band_mask], a_min=0.0, a_max=None) + eps)
    pred_amp = np.sqrt(np.clip(pred_psd[band_mask], a_min=0.0, a_max=None) + eps)
    loss = float(np.mean((pred_amp - true_amp) ** 2))
    if relative:
        loss /= float(np.mean(true_amp ** 2) + eps)
    return loss


def _manual_dominant_frequency_loss(
    true_signal: np.ndarray,
    pred_signal: np.ndarray,
    *,
    dt: float,
    peak_rel_bandwidth: float,
    use_hann_window: bool,
    relative: bool,
    power: float,
    alpha: float,
    eps: float = 1.0e-12,
) -> float:
    freqs, true_psd = _manual_spectrum(true_signal, dt=dt, use_hann_window=use_hann_window)
    _freqs_pred, pred_psd = _manual_spectrum(pred_signal, dt=dt, use_hann_window=use_hann_window)
    band_mask = _manual_band_mask(freqs, true_psd, peak_rel_bandwidth)
    freqs_band = freqs[band_mask]
    true_weights = np.power(np.clip(true_psd[band_mask], a_min=0.0, a_max=None), float(alpha))
    pred_weights = np.power(np.clip(pred_psd[band_mask], a_min=0.0, a_max=None), float(alpha))
    true_dom = float(np.sum(freqs_band * true_weights) / np.sum(true_weights))
    pred_dom = float(np.sum(freqs_band * pred_weights) / np.sum(pred_weights))
    loss = abs(pred_dom - true_dom) ** float(power)
    if relative:
        loss /= abs(true_dom) ** float(power) + eps
    return float(loss)


@unittest.skipUnless(
    torch is not None
    and parse_config is not None
    and _build_td_correction_hnn_loaders is not None
    and _resolve_td_rollout_loss_settings is not None
    and _td_correction_rollout_losses_from_batch is not None,
    "torch-backed PHNN rollout helpers are required",
)
class HnnTdRolloutLossTests(unittest.TestCase):
    def test_deterministic_rollout_trajectory_loss_excludes_initial_state_and_supports_relative_mode(self) -> None:
        z_true = torch.tensor(
            [[[100.0, 100.0], [1.0, 2.0], [2.0, 4.0]]],
            dtype=torch.float64,
        )
        z_pred = torch.tensor(
            [[[0.0, 0.0], [2.0, 4.0], [4.0, 8.0]]],
            dtype=torch.float64,
        )
        batch = _make_rollout_batch(z_true)

        with mock.patch(
            "methods.hnn.trainer._td_correction_state_rollout",
            return_value=_rollout_return_tuple(z_pred),
        ):
            losses_abs = _td_correction_rollout_losses_from_batch(
                model=_dummy_td_model(),
                batch=batch,
                device=torch.device("cpu"),
                non_blocking=False,
                td_params={},
                td_memory_cfg={},
                mean_active=False,
                sigma_active=False,
                fhat_active=False,
                td_force_input_source="none",
                fhat_bound_multiplier=1.0,
                force_zero_output=False,
                rollout_loss_mode="deterministic",
                rollout_stochastic_samples=1,
                rollout_noise_scale=1.0,
                trajectory_relative=False,
            )
            losses_rel = _td_correction_rollout_losses_from_batch(
                model=_dummy_td_model(),
                batch=batch,
                device=torch.device("cpu"),
                non_blocking=False,
                td_params={},
                td_memory_cfg={},
                mean_active=False,
                sigma_active=False,
                fhat_active=False,
                td_force_input_source="none",
                fhat_bound_multiplier=1.0,
                force_zero_output=False,
                rollout_loss_mode="deterministic",
                rollout_stochastic_samples=1,
                rollout_noise_scale=1.0,
                trajectory_relative=True,
            )

        self.assertAlmostEqual(float(losses_abs["trajectory_loss"].item()), 12.5, places=8)
        self.assertAlmostEqual(float(losses_rel["trajectory_loss"].item()), 1.0, places=8)

    def test_displacement_std_loss_supports_p_and_relative_modes(self) -> None:
        true_signal = torch.tensor([[0.0, 1.0, 0.0, -1.0]], dtype=torch.float64)
        pred_signal = torch.tensor([[0.0, 2.0, 0.0, -2.0]], dtype=torch.float64)
        true_std = float(np.std(true_signal.numpy()))
        pred_std = float(np.std(pred_signal.numpy()))
        diff = abs(pred_std - true_std)

        loss_l1 = _displacement_std_error_torch(
            true_signal=true_signal,
            pred_signal=pred_signal,
            relative=False,
            power=1.0,
        )
        loss_l2_rel = _displacement_std_error_torch(
            true_signal=true_signal,
            pred_signal=pred_signal,
            relative=True,
            power=2.0,
        )

        expected_l1 = diff
        expected_l2_rel = (diff**2) / ((abs(true_std) + 1.0e-6) ** 2)
        self.assertAlmostEqual(float(loss_l1.item()), expected_l1, places=8)
        self.assertAlmostEqual(float(loss_l2_rel.item()), expected_l2_rel, places=8)

    def test_td_correction_rollout_loaders_keep_train_and_val_splits_separate(self) -> None:
        train_trajs = [_make_td_traj(5, ur=7.0)]
        val_trajs = [_make_td_traj(4, ur=8.0)]

        train_loader, val_loader, train_rollout_loader, val_rollout_loader = _build_td_correction_hnn_loaders(
            train_trajs=train_trajs,
            val_trajs=val_trajs,
            mass_source="effective",
            input_scaling_mode="current",
            diameter=1.0,
            batch_size=2,
            rollout_batch_size=2,
            rollout_steps=2,
            num_workers=0,
            pin_memory=False,
        )

        self.assertEqual(len(train_loader.dataset), 4)
        self.assertEqual(len(val_loader.dataset), 3)
        self.assertEqual(len(train_rollout_loader.dataset), 3)
        self.assertEqual(len(val_rollout_loader.dataset), 2)

    def test_psd_loss_supports_absolute_relative_and_narrowband_modes(self) -> None:
        dt = 0.05
        time = np.arange(256, dtype=float) * dt
        true_signal_np = np.sin(2.0 * np.pi * 1.0 * time)
        pred_signal_np = 1.2 * np.sin(2.0 * np.pi * 1.0 * time) + 0.35 * np.sin(2.0 * np.pi * 3.0 * time)
        true_signal = torch.tensor(true_signal_np, dtype=torch.float64).view(1, -1)
        pred_signal = torch.tensor(pred_signal_np, dtype=torch.float64).view(1, -1)

        loss_abs = _psd_error_torch(
            true_signal=true_signal,
            pred_signal=pred_signal,
            dt=dt,
            peak_rel_bandwidth=0.0,
            use_hann_window=True,
            relative=False,
        )
        loss_rel_band = _psd_error_torch(
            true_signal=true_signal,
            pred_signal=pred_signal,
            dt=dt,
            peak_rel_bandwidth=0.10,
            use_hann_window=True,
            relative=True,
        )

        expected_abs = _manual_psd_loss(
            true_signal_np,
            pred_signal_np,
            dt=dt,
            peak_rel_bandwidth=0.0,
            use_hann_window=True,
            relative=False,
        )
        expected_rel_band = _manual_psd_loss(
            true_signal_np,
            pred_signal_np,
            dt=dt,
            peak_rel_bandwidth=0.10,
            use_hann_window=True,
            relative=True,
        )
        self.assertAlmostEqual(float(loss_abs.item()), expected_abs, places=6)
        self.assertAlmostEqual(float(loss_rel_band.item()), expected_rel_band, places=6)

    def test_dominant_frequency_loss_supports_p_and_alpha(self) -> None:
        dt = 0.05
        time = np.arange(256, dtype=float) * dt
        true_signal_np = np.sin(2.0 * np.pi * 1.0 * time)
        pred_signal_np = 0.9 * np.sin(2.0 * np.pi * 1.0 * time) + 0.75 * np.sin(2.0 * np.pi * 1.5 * time)
        true_signal = torch.tensor(true_signal_np, dtype=torch.float64).view(1, -1)
        pred_signal = torch.tensor(pred_signal_np, dtype=torch.float64).view(1, -1)

        loss_p1_alpha1 = _dominant_frequency_error_torch(
            true_signal=true_signal,
            pred_signal=pred_signal,
            dt=dt,
            peak_rel_bandwidth=0.0,
            use_hann_window=True,
            relative=False,
            power=1.0,
            alpha=1.0,
        )
        loss_p2_alpha12_rel = _dominant_frequency_error_torch(
            true_signal=true_signal,
            pred_signal=pred_signal,
            dt=dt,
            peak_rel_bandwidth=0.0,
            use_hann_window=True,
            relative=True,
            power=2.0,
            alpha=12.0,
        )

        expected_p1_alpha1 = _manual_dominant_frequency_loss(
            true_signal_np,
            pred_signal_np,
            dt=dt,
            peak_rel_bandwidth=0.0,
            use_hann_window=True,
            relative=False,
            power=1.0,
            alpha=1.0,
        )
        expected_p2_alpha12_rel = _manual_dominant_frequency_loss(
            true_signal_np,
            pred_signal_np,
            dt=dt,
            peak_rel_bandwidth=0.0,
            use_hann_window=True,
            relative=True,
            power=2.0,
            alpha=12.0,
        )

        self.assertAlmostEqual(float(loss_p1_alpha1.item()), expected_p1_alpha1, places=6)
        self.assertAlmostEqual(float(loss_p2_alpha12_rel.item()), expected_p2_alpha12_rel, places=6)
        self.assertLess(float(loss_p2_alpha12_rel.item()), float(loss_p1_alpha1.item()))

    def test_rollout_relative_loss_settings_resolve_global_and_legacy_modes(self) -> None:
        legacy_cfg = parse_config(
            {
                "loss": {
                    "rollout_relative_losses": None,
                    "rollout_det_amplitude_normalized_mse": True,
                    "rollout_disp_std_normalize_by_true": None,
                }
            }
        )
        legacy = _resolve_td_rollout_loss_settings(legacy_cfg.loss)
        self.assertIsNone(legacy["rollout_relative_losses"])
        self.assertTrue(legacy["trajectory_relative"])
        self.assertTrue(legacy["disp_std_relative"])
        self.assertFalse(legacy["disp_psd_relative"])
        self.assertTrue(legacy["disp_freq_relative"])

        force_false_cfg = parse_config(
            {
                "loss": {
                    "rollout_relative_losses": False,
                    "rollout_det_amplitude_normalized_mse": True,
                    "rollout_disp_std_normalize_by_true": True,
                    "rollout_disp_std_p": 1.5,
                    "rollout_disp_freq_p": 2.0,
                    "rollout_disp_freq_alpha": 9.0,
                }
            }
        )
        force_false = _resolve_td_rollout_loss_settings(force_false_cfg.loss)
        self.assertFalse(force_false["trajectory_relative"])
        self.assertFalse(force_false["disp_std_relative"])
        self.assertFalse(force_false["disp_psd_relative"])
        self.assertFalse(force_false["disp_freq_relative"])
        self.assertAlmostEqual(force_false["disp_std_p"], 1.5, places=8)
        self.assertAlmostEqual(force_false["disp_freq_p"], 2.0, places=8)
        self.assertAlmostEqual(force_false["disp_freq_alpha"], 9.0, places=8)

        force_true_cfg = parse_config({"loss": {"rollout_relative_losses": True}})
        force_true = _resolve_td_rollout_loss_settings(force_true_cfg.loss)
        self.assertTrue(force_true["trajectory_relative"])
        self.assertTrue(force_true["disp_std_relative"])
        self.assertTrue(force_true["disp_psd_relative"])
        self.assertTrue(force_true["disp_freq_relative"])

    def test_stochastic_nll_trajectory_loss_ignores_relative_toggle(self) -> None:
        z_true = torch.tensor(
            [[[5.0, 7.0], [1.0, 2.0], [2.0, 4.0]]],
            dtype=torch.float64,
        )
        z_pred_samples = torch.tensor(
            [
                [[5.0, 7.0], [1.2, 1.8], [1.9, 4.2]],
                [[5.0, 7.0], [0.8, 2.2], [2.1, 3.8]],
            ],
            dtype=torch.float64,
        )
        batch = _make_rollout_batch(z_true)

        with mock.patch(
            "methods.hnn.trainer._td_correction_state_rollout",
            return_value=_rollout_return_tuple(z_pred_samples),
        ):
            losses_abs = _td_correction_rollout_losses_from_batch(
                model=_dummy_td_model(),
                batch=batch,
                device=torch.device("cpu"),
                non_blocking=False,
                td_params={},
                td_memory_cfg={},
                mean_active=False,
                sigma_active=True,
                fhat_active=False,
                td_force_input_source="none",
                fhat_bound_multiplier=1.0,
                force_zero_output=False,
                rollout_loss_mode="stochastic_nll",
                rollout_stochastic_samples=2,
                rollout_noise_scale=1.0,
                trajectory_relative=False,
            )
            losses_rel = _td_correction_rollout_losses_from_batch(
                model=_dummy_td_model(),
                batch=batch,
                device=torch.device("cpu"),
                non_blocking=False,
                td_params={},
                td_memory_cfg={},
                mean_active=False,
                sigma_active=True,
                fhat_active=False,
                td_force_input_source="none",
                fhat_bound_multiplier=1.0,
                force_zero_output=False,
                rollout_loss_mode="stochastic_nll",
                rollout_stochastic_samples=2,
                rollout_noise_scale=1.0,
                trajectory_relative=True,
            )

        self.assertAlmostEqual(
            float(losses_abs["trajectory_loss"].item()),
            float(losses_rel["trajectory_loss"].item()),
            places=8,
        )


if __name__ == "__main__":
    unittest.main()
