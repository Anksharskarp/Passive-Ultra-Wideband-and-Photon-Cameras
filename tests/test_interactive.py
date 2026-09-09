import json
import unittest
from dataclasses import replace

import numpy as np

from experiment import Flux, create_experiment, spectrum, summary, view
from photon_simulator import Detector, Events, apply_detector


class InteractiveTests(unittest.TestCase):
    def test_detector_changes_preserve_incident_realization(self):
        baseline = create_experiment({"seed": 11})
        changed = create_experiment({"seed": 11, "qe": .2, "dark_rate": 200,
                                     "dead_time": 2e-5, "dead_model": "nonparalyzable"})
        np.testing.assert_array_equal(baseline.result.source_events.times_s,
                                      changed.result.source_events.times_s)
        self.assertLess(len(changed.result.detected_events), len(baseline.result.detected_events))

    def test_view_changes_do_not_resimulate(self):
        run = create_experiment({})
        original = run.result.detected_events.times_s.copy()
        one = view(run, {"start": 0, "stop": .03, "bins": 32})
        two = view(run, {"start": 0, "stop": .03, "bins": 256})
        self.assertEqual(sum(one["counts"]), sum(two["counts"]))
        np.testing.assert_array_equal(original, run.result.detected_events.times_s)

    def test_display_integrates_narrow_pulses(self):
        config = {"kind": "pulse", "duration": .001, "frequency": 20e6,
                  "pulse_width": 80e-12, "mean_rate": 200000, "background": 0}
        exact = create_experiment(config)
        averaged = create_experiment({**config, "sampling": "average", "source_step": 50e-9})
        midpoint = create_experiment({**config, "sampling": "midpoint", "source_step": 50e-9})
        self.assertAlmostEqual(summary(exact)["expected_incident"], 200, places=6)
        self.assertAlmostEqual(summary(averaged)["expected_incident"], 200, places=6)
        self.assertLess(summary(midpoint)["expected_incident"], 1e-6)
        displayed = view(exact)
        integrated = np.dot(displayed["source"], np.diff(displayed["edges"]))
        self.assertAlmostEqual(integrated, 200, places=6)

    def test_reference_integral_matches_sine_closed_form(self):
        flux = Flux("sine", 1, 1000, (3,), (.8,))
        self.assertAlmostEqual(float(flux.integral(1)), 1000)
        self.assertAlmostEqual(float(flux.integral(1/6)), 1000/6+800/(3*np.pi))

    def test_sparse_fast_frequency_detected(self):
        run = create_experiment({})
        spec = spectrum(run)
        strongest = int(np.argmax(spec["amplitudes"]))
        self.assertAlmostEqual(spec["frequencies"][strongest], 8000)
        self.assertIn(strongest, spec["selected"])
        # A positive-frequency coefficient is half the real sinusoid amplitude.
        expected = 3000*.85/2
        self.assertLess(abs(spec["amplitudes"][strongest]-expected), 4*np.sqrt(600)/.2)

    def test_spectrum_uses_integer_exposure_grid(self):
        run = create_experiment({"probe_step": 13, "duration": .2})
        spec = spectrum(run)
        scaled = np.array(spec["frequencies"])*run.result.duration_s
        np.testing.assert_allclose(scaled, np.round(scaled))

    def test_dark_counts_and_zero_flux(self):
        run = create_experiment({"mean_rate": 0, "dark_rate": 300})
        self.assertEqual(len(run.result.source_events), 0)
        self.assertGreater(len(run.result.detected_events), 0)
        json.dumps(summary(run), allow_nan=False)
        json.dumps(spectrum(run), allow_nan=False)

    def test_jitter_changes_timestamps_without_censoring_detections(self):
        incident = Events(np.linspace(0, .999, 1000))
        measured = apply_detector(incident, 1, Detector(jitter_std_s=1), np.random.default_rng(8))
        self.assertEqual(len(measured), len(incident))
        self.assertTrue(np.any(measured.times_s < 0))
        self.assertTrue(np.any(measured.times_s >= 1))

    def test_each_pixel_has_its_own_dead_time(self):
        incident = Events([0, .01, .05, .06], [0, 1, 0, 1], [0, 0, 0, 0])
        measured = apply_detector(incident, .2,
                                 Detector(dead_time_s=.1, dead_time_model="nonparalyzable"))
        np.testing.assert_allclose(measured.times_s, [0, .01])
        np.testing.assert_array_equal(measured.rows, [0, 1])

    def test_exposure_end_is_excluded_from_histogram_not_export(self):
        run = create_experiment({"duration": .2})
        run.result = replace(run.result, detected_events=Events([-.001, 0, .1, .2, .201]))
        data = view(run)
        self.assertEqual(sum(data["counts"]), 2)
        self.assertEqual(data["detected_visible"], 2)
        self.assertEqual(summary(run)["outside"], 3)
        self.assertEqual(spectrum(run)["dc"], 5/.2)

    def test_unrepresentably_small_view_is_rejected(self):
        run = create_experiment({})
        with self.assertRaisesRegex(ValueError, "too small"):
            view(run, {"start": .1, "stop": np.nextafter(.1, 1)})

    def test_media_counts_and_selected_pixel(self):
        frames = [[[0., 1.], [0., .5]], [[1., 0.], [.5, 0.]]]
        run = create_experiment({"kind": "media", "frames": frames, "fps": 20, "mean_rate": 1000})
        self.assertEqual(int(run.frame_counts.sum()), len(run.result.detected_events))
        self.assertTrue(np.all(np.diff(run.result.detected_events.times_s) >= 0))
        pixel = view(run, {"row": 0, "col": 1})
        self.assertTrue(all(t < .05 for t in pixel["incident"]))
        self.assertEqual(pixel["media"]["image"], frames[0])

    def test_derived_exposures_reproduce_from_saved_config(self):
        configs = [
            {"kind": "media", "frames": [[[1.]]] * 8, "fps": 1, "mean_rate": 10},
            {"kind": "sampled", "rates": [5000], "source_step": 1e-12},
        ]
        for config in configs:
            with self.subTest(kind=config["kind"]):
                first = create_experiment(config)
                repeated = create_experiment(json.loads(json.dumps(first.config)))
                self.assertEqual(first.result.duration_s, repeated.result.duration_s)
                np.testing.assert_array_equal(first.result.detected_events.times_s,
                                               repeated.result.detected_events.times_s)
        for extra in ({}, {"duration": 24}):
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, "Media exposure"):
                create_experiment({"kind": "media", "frames": [[[1.]]] * 24,
                                   "fps": 1, "mean_rate": 1, **extra})

    def test_invalid_and_excessive_inputs_rejected(self):
        for config in ({"mean_rate": float("nan")}, {"duration": 10, "mean_rate": 1e8},
                       {"sampling": "average", "source_step": 1e-12},
                       {"kind": "sampled", "rates": [-1]},
                       {"kind": "media", "frames": [[[2]]]}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                create_experiment(config)
        with self.assertRaises(ValueError):
            Detector(jitter_std_s=float("nan"))
        with self.assertRaises(ValueError):
            Events([0], [.5], [0])


if __name__ == "__main__":
    unittest.main()
