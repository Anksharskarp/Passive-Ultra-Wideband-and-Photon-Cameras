"""State and plot data for the interactive photon experiments.

All times/rates crossing the API use seconds and photons/second. Display bins
integrate the source analytically, including Gaussian pulses much narrower than
a screen pixel. Changing a view never draws a new photon realization.
"""

from dataclasses import dataclass
import math

import numpy as np

from photon_simulator import (
    Detector, Events, SimulationResult, apply_detector, cfar_amplitude_threshold,
    fourier_probe, simulate_analytic, simulate_gaussian_pulse_train,
    simulate_media, simulate_piecewise_constant,
)
from simulate import gaussian_pulse_train_rate, sinusoidal_rate


MAX_EXPECTED_EVENTS = 250_000
MAX_SOURCE_BINS = 100_000
MAX_PROBES = 2000


def number(config, key, default, low, high):
    value = float(config.get(key, default))
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{key} must be between {low:g} and {high:g}.")
    return value


@dataclass
class Flux:
    kind: str
    duration: float
    mean: float = 0.0
    frequencies: tuple = ()
    modulations: tuple = ()
    pulse_width: float = 0.0
    background: float = 0.0
    rates: np.ndarray | None = None
    step: float = 0.0

    def rate(self, times):
        if self.rates is not None:
            indices = np.clip(np.floor(times / self.step).astype(int), 0, len(self.rates)-1)
            return self.rates[indices]
        if self.mean == 0:
            return np.zeros_like(times, dtype=float)
        if self.kind == "pulse":
            function, _ = gaussian_pulse_train_rate(
                self.mean, self.frequencies[0], self.pulse_width, self.background
            )
        else:
            function, _ = sinusoidal_rate(self.mean, self.frequencies, self.modulations)
        return function(times)

    def integral(self, times):
        """Integral from zero through each time; no temporal quadrature grid."""
        times = np.asarray(times, dtype=float)
        if self.rates is not None:
            times = np.clip(times, 0, self.duration)
            cumulative = np.r_[0, np.cumsum(self.rates * self.step)]
            indices = np.minimum((times / self.step).astype(int), len(self.rates)-1)
            return cumulative[indices] + self.rates[indices] * (times-indices*self.step)
        if self.kind == "pulse":
            period = 1 / self.frequencies[0]
            sigma = self.pulse_width / math.sqrt(8 * math.log(2))
            cycles = np.floor((times + period/2) / period)
            local = times - cycles*period
            erf = np.fromiter((math.erf(float(t)/(sigma*math.sqrt(2)))
                               for t in local.flat), float).reshape(times.shape)
            norm = math.erf(period/(2*sigma*math.sqrt(2)))
            return (self.mean * self.background * times
                    + self.mean*(1-self.background)*period*(cycles + erf/(2*norm)))
        result = self.mean * times
        for frequency, modulation in zip(self.frequencies, self.modulations):
            if frequency:
                result = result + self.mean*modulation*(
                    1-np.cos(2*np.pi*frequency*times)) / (2*np.pi*frequency)
        return result

    def averages(self, edges):
        return np.maximum(0, np.diff(self.integral(edges)) / np.diff(edges))


@dataclass
class Experiment:
    config: dict
    result: SimulationResult
    reference: Flux
    source: Flux
    detector: Detector
    frames: np.ndarray | None = None
    frame_counts: np.ndarray | None = None

    def pixel(self, row=0, col=0):
        if self.frames is None:
            return self.reference, self.source, self.result.source_events.times_s, \
                self.result.detected_events.times_s
        height, width = self.frames.shape[1:]
        if not (0 <= row < height and 0 <= col < width):
            raise ValueError("Selected pixel is outside the image.")
        flux = Flux("sampled", self.result.duration_s,
                    rates=self.frames[:, row, col]*self.config["mean_rate"],
                    step=1/self.config["fps"])
        incident, detected = self.result.source_events, self.result.detected_events
        return (flux, flux,
                incident.times_s[(incident.rows == row) & (incident.cols == col)],
                detected.times_s[(detected.rows == row) & (detected.cols == col)])


def create_experiment(raw_config):
    config = dict(raw_config)
    kind = config.get("kind", "sine")
    if kind not in {"sine", "mixture", "pulse", "sampled", "media"}:
        raise ValueError("Unknown source type.")
    # Sampled/media exposure is derived from the data, not a stale UI field.
    duration = (number(config, "duration", .2, 1e-7, 10)
                if kind not in {"sampled", "media"} else 0.0)
    mean = number(config, "mean_rate", 3000, 0, 1e8)
    seed = number(config, "seed", 7, 0, 2**32-1)
    if int(seed) != seed:
        raise ValueError("Seed must be an integer.")
    detector = Detector(
        quantum_efficiency=number(config, "qe", 1, 0, 1),
        dark_rate_hz=number(config, "dark_rate", 0, 0, 1e7),
        dead_time_s=number(config, "dead_time", 0, 0, 1),
        dead_time_model=config.get("dead_model", "none"),
        jitter_std_s=number(config, "jitter", 0, 0, .1),
        timestamp_resolution_s=number(config, "quantization", 0, 0, .1),
    )
    source_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0]))
    detector_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 1]))
    frames = None
    sensor_shape = None
    if kind == "media":
        frames = np.asarray(config.get("frames", []), dtype=float)
        if (frames.ndim != 3 or not all(frames.shape) or frames.size > 64*64*32
                or frames.shape[0] > 32 or max(frames.shape[1:]) > 64):
            raise ValueError("Use 1–32 frames, at most 64 × 64 pixels each.")
        if not np.all(np.isfinite(frames)) or np.any((frames < 0) | (frames > 1)):
            raise ValueError("Media intensities must be finite values in [0, 1].")
        fps = number(config, "fps", 24, .1, 1000)
        config["fps"] = fps
        duration = len(frames)/fps
        if duration > 10:
            raise ValueError("Media exposure must be at most 10 seconds. Increase frame rate or use fewer frames.")
        sensor_shape = frames.shape[1:]
        expected = float(frames.sum()*mean/fps)
        reference = source = Flux("sampled", duration, rates=frames[:, 0, 0]*mean, step=1/fps)
    elif kind == "sampled":
        rates = np.asarray(config.get("rates", []), dtype=float)
        if rates.ndim != 1 or not 1 <= len(rates) <= MAX_SOURCE_BINS:
            raise ValueError("Provide a non-empty one-dimensional list of rates.")
        if not np.all(np.isfinite(rates)) or np.any(rates < 0):
            raise ValueError("Sampled rates must be finite, non-negative photons/s.")
        step = number(config, "source_step", .001, 1e-12, 10)
        duration = len(rates)*step
        if duration > 10:
            raise ValueError("Sampled exposure must be at most 10 seconds.")
        reference = source = Flux(kind, duration, rates=rates, step=step)
        expected = float(rates.sum()*step)
    else:
        frequency = number(config, "frequency", 8000, 1e-6, 1e10)
        modulation = number(config, "modulation", .85, 0, 1)
        frequencies = (frequency,)
        modulations = (modulation,)
        if kind == "mixture":
            slow = number(config, "slow_frequency", 50, 1e-6, 1e10)
            frequencies = (slow, frequency)
            modulations = (.45*modulation, .55*modulation)
        width = number(config, "pulse_width", 80e-12, 1e-13, 1)
        background = number(config, "background", .05, 0, .99)
        if kind == "pulse" and width > 1/(3*frequency):
            raise ValueError("Pulse FWHM must be at most one third of the period.")
        reference = Flux(kind, duration, mean, frequencies, modulations, width, background)
        source = reference
        sampling = config.get("sampling", "continuous")
        if sampling not in {"continuous", "midpoint", "average"}:
            raise ValueError("Unknown source sampling method.")
        if sampling != "continuous":
            step = number(config, "source_step", .0001, 1e-12, 10)
            bins = math.ceil(duration/step)
            if bins > MAX_SOURCE_BINS:
                raise ValueError("Source grid exceeds 100,000 bins; increase its bin width.")
            step = duration/bins
            edges = np.linspace(0, duration, bins+1)
            rates = (reference.averages(edges) if sampling == "average"
                     else reference.rate((edges[:-1]+edges[1:])/2))
            source = Flux("sampled", duration, rates=rates, step=step)
        expected = float(source.integral(duration))

    pixels = math.prod(sensor_shape) if sensor_shape else 1
    expected_dark = duration*detector.dark_rate_hz*pixels
    if expected + expected_dark > MAX_EXPECTED_EVENTS:
        raise ValueError("This run exceeds 250,000 expected events. Reduce flux or exposure.")
    if frames is not None:
        raw = simulate_media(frames, 1/config["fps"], mean, rng=source_rng)
    elif source.rates is not None:
        raw = simulate_piecewise_constant(source.rates, source.step, rng=source_rng)
    elif kind == "pulse" and mean > 0:
        if duration*source.frequencies[0] > 2_000_000:
            raise ValueError("Interactive pulse runs support up to two million periods.")
        raw = simulate_gaussian_pulse_train(
            duration, mean, source.frequencies[0], source.pulse_width,
            source.background, rng=source_rng,
        )
    else:
        bound = mean*(1+sum(abs(m) for m in source.modulations))
        raw = simulate_analytic(source.rate, duration, max(bound, 1e-9), rng=source_rng)
    detected = apply_detector(raw.source_events, duration, detector, detector_rng, sensor_shape)
    result = SimulationResult(raw.source_events, detected, duration, raw.source_description)
    config.update(duration=duration, mean_rate=mean, seed=int(seed), kind=kind)
    experiment = Experiment(config, result, reference, source, detector, frames)
    if frames is not None:
        frame_counts = np.zeros(frames.shape, dtype=int)
        frame = np.floor(detected.times_s*config["fps"]).astype(int)
        valid = (frame >= 0) & (frame < len(frames))
        np.add.at(frame_counts, (frame[valid], detected.rows[valid], detected.cols[valid]), 1)
        experiment.frame_counts = frame_counts
    return experiment


def spectrum(experiment, row=0, col=0):
    _, _, _, times = experiment.pixel(row, col)
    config, duration = experiment.config, experiment.result.duration_s
    maximum = number(config, "probe_max", 10000, 1e-6, 1e11)
    requested_step = number(config, "probe_step", 0, 0, 1e11)
    # Integer Fourier-series bins keep a constant flux orthogonal to non-DC probes.
    stride = max(1, math.ceil(maximum*duration/MAX_PROBES), round(requested_step*duration))
    step = stride/duration
    count = math.floor(maximum/step + 1e-9)
    if count*len(times) > 30_000_000:
        raise ValueError("Spectrum is too expensive. Increase probe step or reduce photon count.")
    frequencies = np.arange(1, count+1)*step
    coefficients = fourier_probe(times, frequencies, duration)
    alpha = number(config, "alpha", .001, 1e-10, .1)
    threshold = cfar_amplitude_threshold(len(times), duration, alpha)
    keep = np.flatnonzero((abs(coefficients) > threshold) & (len(times) > 0))
    selected = keep[np.argsort(abs(coefficients[keep]))[::-1]][:40]
    return {
        "frequencies": frequencies.tolist(), "amplitudes": abs(coefficients).tolist(),
        "real": coefficients.real.tolist(), "imag": coefficients.imag.tolist(),
        "selected": selected.tolist(), "threshold": threshold, "alpha": alpha,
        "passed": len(keep), "dc": len(times)/duration, "step": step,
        "sparse": stride > 1, "expected_false_alarms": len(frequencies)*alpha,
    }


def view(experiment, options=None):
    options = options or {}
    row = int(number(options, "row", 0, 0, 63))
    col = int(number(options, "col", 0, 0, 63))
    reference, source, incident, detected = experiment.pixel(row, col)
    duration = experiment.result.duration_s
    start = number(options, "start", 0, 0, duration)
    stop = number(options, "stop", duration, 0, duration)
    if stop <= start:
        raise ValueError("View end must be after view start.")
    bins = int(number(options, "bins", 100, 8, 512))
    edges = np.linspace(start, stop, 801)
    if np.any(np.diff(edges) <= 0):
        raise ValueError("View window is too small at this absolute time; zoom out slightly.")
    hist_edges = np.linspace(start, stop, bins+1)
    visible_incident = incident[(incident >= start) & (incident < stop)]
    visible_detected = detected[(detected >= start) & (detected < stop)]
    counts, _ = np.histogram(visible_detected, hist_edges)
    output = {
        "start": start, "stop": stop, "edges": edges.tolist(),
        "reference": reference.averages(edges).tolist(),
        "source": source.averages(edges).tolist(),
        "hist_edges": hist_edges.tolist(), "counts": counts.tolist(),
        "hist_expected": (source.averages(hist_edges)*experiment.detector.quantum_efficiency
                          + experiment.detector.dark_rate_hz).tolist(),
        "incident": visible_incident[:6000].tolist(),
        "detected": visible_detected[:6000].tolist(),
        "incident_visible": len(visible_incident),
        "detected_visible": len(visible_detected),
        "pixel": [row, col],
    }
    if experiment.frames is not None:
        frame = min(len(experiment.frames)-1, int(number(options, "frame", 0, 0, 31)))
        output["media"] = {
            "frame": frame, "total_frames": len(experiment.frames),
            "image": experiment.frames[frame].tolist(),
            "counts": experiment.frame_counts[frame].tolist(),
            "count_max": int(experiment.frame_counts.max(initial=1)),
        }
    return output


def summary(experiment):
    config, result, detector = experiment.config, experiment.result, experiment.detector
    if experiment.frames is None:
        expected = float(experiment.source.integral(result.duration_s))
        pixels = 1
    else:
        expected = float(experiment.frames.sum()*config["mean_rate"]/config["fps"])
        pixels = math.prod(experiment.frames.shape[1:])
    recorded = result.detected_events.times_s
    outside = int(np.count_nonzero((recorded < 0) | (recorded >= result.duration_s)))
    notices = []
    if detector.dead_time_s and detector.dead_time_model != "none":
        notices.append("Dead time is active. The spectrum and reconstruction use uncorrected "
                       "detections; the low-flux CFAR line is only a reference here.")
    if experiment.source.rates is not None:
        notices.append("Arrivals are continuous within each held source bin. Samples constrain "
                       "the assumed flux; smaller timestamp bins cannot recover missing source detail.")
    if outside:
        notices.append(f"{outside} reported timestamps moved outside the exposure through jitter "
                       "or rounding. They remain in exports and Fourier probes; histograms exclude them.")
    if len(recorded) < 30:
        notices.append("Very few detections: the Gaussian/CFAR approximation can be unreliable.")
    return {
        "incident": len(result.source_events), "detected": len(result.detected_events),
        "expected_incident": expected, "duration": result.duration_s,
        "recorded_rate": len(recorded)/result.duration_s/pixels,
        "expected_dark": detector.dark_rate_hz*result.duration_s*pixels,
        "outside": outside, "notices": notices,
        "source_step": experiment.source.step,
        "nyquist": 1/(2*detector.timestamp_resolution_s)
                    if detector.timestamp_resolution_s else None,
        "pixels": pixels,
    }
