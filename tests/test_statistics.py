"""Seeded distribution checks; tolerances allow ordinary sampling variation."""

import unittest

import numpy as np

from photon_simulator import Detector, simulate_piecewise_constant


class StatisticsTests(unittest.TestCase):
    def test_poisson_count_mean_and_variance(self):
        result = simulate_piecewise_constant(np.full(5000, 1000.), .01,
                                             rng=np.random.default_rng(15))
        counts, _ = np.histogram(result.source_events.times_s, np.arange(5001)*.01)
        self.assertAlmostEqual(counts.mean(), 10, delta=.25)
        self.assertAlmostEqual(counts.var(), 10, delta=.9)

    def test_held_flux_has_uniform_continuous_arrivals_inside_bins(self):
        result = simulate_piecewise_constant(np.full(1000, 5000.), .01,
                                             rng=np.random.default_rng(16))
        phase = result.source_events.times_s/.01 % 1
        self.assertAlmostEqual(phase.mean(), .5, delta=.01)
        self.assertAlmostEqual(phase.var(), 1/12, delta=.004)
        self.assertEqual(np.count_nonzero(phase == 0), 0)

    def test_constant_flux_nonparalyzable_count_rate(self):
        rate, dead_time, duration = 10000., .0001, 10.
        result = simulate_piecewise_constant(
            [rate], duration,
            Detector(dead_time_s=dead_time, dead_time_model="nonparalyzable"),
            np.random.default_rng(17),
        )
        expected_rate = rate/(1+rate*dead_time)
        self.assertAlmostEqual(len(result.detected_events)/duration, expected_rate,
                               delta=.02*expected_rate)


if __name__ == "__main__":
    unittest.main()
