import unittest

import numpy as np

from photon_simulator import (
    Detector,
    Events,
    apply_detector,
    cfar_amplitude_threshold,
    fourier_probe,
    simulate_analytic,
    simulate_gaussian_pulse_train,
    simulate_media,
    simulate_piecewise_constant,
)


class PhotonSimulatorTests(unittest.TestCase):
    def test_analytic_events_are_sorted_and_in_range(self):
        result = simulate_analytic(
            lambda t: np.full_like(t, 2_000.0),
            duration_s=0.1,
            rate_upper_bound_hz=2_000.0,
            rng=np.random.default_rng(1),
        )
        self.assertGreater(len(result.source_events), 0)
        self.assertTrue(np.all(np.diff(result.source_events.times_s) >= 0.0))
        self.assertGreaterEqual(result.source_events.times_s.min(), 0.0)
        self.assertLessEqual(result.source_events.times_s.max(), 0.1)

    def test_analytic_rejects_an_invalid_upper_bound(self):
        with self.assertRaises(ValueError):
            simulate_analytic(
                lambda t: np.full_like(t, 2_000.0),
                duration_s=0.1,
                rate_upper_bound_hz=1_000.0,
                rng=np.random.default_rng(2),
            )

    def test_piecewise_zero_bins_have_no_arrivals(self):
        result = simulate_piecewise_constant(
            [0.0, 100_000.0, 0.0],
            bin_width_s=0.01,
            rng=np.random.default_rng(3),
        )
        self.assertTrue(np.all(result.source_events.times_s >= 0.01))
        self.assertTrue(np.all(result.source_events.times_s < 0.02))
        self.assertGreater(len(np.unique(result.source_events.times_s)), 1)

    def test_narrow_gaussian_pulse_train_is_practical(self):
        result = simulate_gaussian_pulse_train(
            duration_s=0.001,
            mean_rate_hz=500_000.0,
            repetition_rate_hz=20_000_000.0,
            pulse_fwhm_s=8e-12,
            rng=np.random.default_rng(31),
        )
        self.assertGreater(len(result.source_events), 350)
        self.assertLess(len(result.source_events), 650)
        phase = np.mod(result.source_events.times_s, 1.0 / 20_000_000.0)
        distance_to_pulse = np.minimum(phase, 1.0 / 20_000_000.0 - phase)
        self.assertLess(np.quantile(distance_to_pulse, 0.99), 2e-11)

    def test_quantization_is_separate_from_source_sampling(self):
        source = Events(np.array([0.011, 0.026, 0.074]))
        detected = apply_detector(
            source,
            duration_s=0.1,
            detector=Detector(timestamp_resolution_s=0.01),
            rng=np.random.default_rng(4),
        )
        np.testing.assert_allclose(detected.times_s, [0.01, 0.03, 0.07])

    def test_nonparalyzable_dead_time(self):
        source = Events(np.array([0.00, 0.05, 0.11, 0.15, 0.22]))
        detected = apply_detector(
            source,
            duration_s=0.3,
            detector=Detector(dead_time_s=0.1, dead_time_model="nonparalyzable"),
            rng=np.random.default_rng(5),
        )
        np.testing.assert_allclose(detected.times_s, [0.00, 0.11, 0.22])

    def test_paralyzable_dead_time_extends_on_rejected_events(self):
        source = Events(np.array([0.00, 0.05, 0.11, 0.15, 0.22]))
        detected = apply_detector(
            source,
            duration_s=0.3,
            detector=Detector(dead_time_s=0.1, dead_time_model="paralyzable"),
            rng=np.random.default_rng(6),
        )
        np.testing.assert_allclose(detected.times_s, [0.00])

    def test_fourier_probe_has_expected_dc_value(self):
        times = np.array([0.1, 0.2, 0.8])
        coefficients = fourier_probe(times, [0.0], exposure_s=1.0)
        self.assertAlmostEqual(coefficients[0].real, 3.0)
        self.assertAlmostEqual(coefficients[0].imag, 0.0)

    def test_cfar_closed_form(self):
        threshold = cfar_amplitude_threshold(100, 2.0, np.exp(-4.0))
        self.assertAlmostEqual(threshold, 10.0)

    def test_media_coordinates_follow_nonzero_pixels(self):
        frames = np.array([[[1.0, 0.0], [0.0, 0.0]]])
        result = simulate_media(
            frames,
            frame_period_s=0.1,
            peak_rate_hz=10_000.0,
            rng=np.random.default_rng(7),
        )
        self.assertGreater(len(result.detected_events), 0)
        self.assertTrue(np.all(result.detected_events.rows == 0))
        self.assertTrue(np.all(result.detected_events.cols == 0))


if __name__ == "__main__":
    unittest.main()
