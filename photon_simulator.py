"""Photon arrival simulation for continuous and sampled flux functions.

The source model and detector model are deliberately separate.  A sampled
source is piecewise constant in this implementation, but its photon arrival
times are still continuous until the detector quantizes them.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from statistics import NormalDist
from typing import Callable, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
DeadTimeModel = Literal["none", "nonparalyzable", "paralyzable"]


@dataclass(frozen=True)
class Detector:
    """Simple SPAD detector model.

    ``dark_rate_hz`` is per pixel. Timestamp jitter and quantization happen
    after detection; dead time acts on the unjittered avalanche times.
    """

    quantum_efficiency: float = 1.0
    dark_rate_hz: float = 0.0
    dead_time_s: float = 0.0
    dead_time_model: DeadTimeModel = "none"
    jitter_std_s: float = 0.0
    timestamp_resolution_s: float = 0.0

    def __post_init__(self) -> None:
        numeric = (
            self.quantum_efficiency, self.dark_rate_hz, self.dead_time_s,
            self.jitter_std_s, self.timestamp_resolution_s,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("detector settings must be finite")
        if not 0.0 <= self.quantum_efficiency <= 1.0:
            raise ValueError("quantum_efficiency must be between 0 and 1")
        if self.dark_rate_hz < 0.0:
            raise ValueError("dark_rate_hz cannot be negative")
        if self.dead_time_s < 0.0:
            raise ValueError("dead_time_s cannot be negative")
        if self.dead_time_model not in {"none", "nonparalyzable", "paralyzable"}:
            raise ValueError(f"unknown dead-time model: {self.dead_time_model}")
        if self.jitter_std_s < 0.0:
            raise ValueError("jitter_std_s cannot be negative")
        if self.timestamp_resolution_s < 0.0:
            raise ValueError("timestamp_resolution_s cannot be negative")


@dataclass(frozen=True)
class Events:
    """An event list. Rows and columns are present for image/video data."""

    times_s: FloatArray
    rows: IntArray | None = None
    cols: IntArray | None = None

    def __post_init__(self) -> None:
        times = np.asarray(self.times_s, dtype=np.float64)
        if times.ndim != 1 or not np.all(np.isfinite(times)):
            raise ValueError("event times must be a finite one-dimensional array")
        object.__setattr__(self, "times_s", times)
        if (self.rows is None) != (self.cols is None):
            raise ValueError("rows and cols must either both be present or both be absent")
        if self.rows is not None and self.cols is not None:
            for coordinates in (self.rows, self.cols):
                values = np.asarray(coordinates)
                if (values.ndim != 1 or not np.all(np.isfinite(values))
                        or np.any(values < 0) or np.any(values != np.floor(values))):
                    raise ValueError("pixel coordinates must be non-negative integers")
            rows = np.asarray(self.rows, dtype=np.int64)
            cols = np.asarray(self.cols, dtype=np.int64)
            if not (len(times) == len(rows) == len(cols)):
                raise ValueError("time, row, and column arrays must have the same length")
            object.__setattr__(self, "rows", rows)
            object.__setattr__(self, "cols", cols)

    def __len__(self) -> int:
        return len(self.times_s)

    @property
    def is_spatial(self) -> bool:
        return self.rows is not None


@dataclass(frozen=True)
class SimulationResult:
    source_events: Events
    detected_events: Events
    duration_s: float
    source_description: str

    @property
    def detection_fraction(self) -> float:
        # Includes dark counts, so this is a count ratio and may exceed one.
        if len(self.source_events) == 0:
            return float("nan")
        return len(self.detected_events) / len(self.source_events)


def _rng(rng: np.random.Generator | None) -> np.random.Generator:
    return np.random.default_rng() if rng is None else rng


def simulate_analytic(
    rate_fn: Callable[[FloatArray], ArrayLike],
    duration_s: float,
    rate_upper_bound_hz: float,
    detector: Detector = Detector(),
    rng: np.random.Generator | None = None,
) -> SimulationResult:
    """Simulate an inhomogeneous Poisson process by rejection thinning.

    ``rate_upper_bound_hz`` must bound ``rate_fn`` over the full exposure.
    This is exact apart from floating-point evaluation of the supplied rate.
    """

    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    if not math.isfinite(rate_upper_bound_hz) or rate_upper_bound_hz <= 0.0:
        raise ValueError("rate_upper_bound_hz must be positive")

    generator = _rng(rng)
    candidate_count = generator.poisson(rate_upper_bound_hz * duration_s)
    candidate_times = np.sort(generator.uniform(0.0, duration_s, candidate_count))
    candidate_rates = np.asarray(rate_fn(candidate_times), dtype=np.float64)
    candidate_rates = np.broadcast_to(candidate_rates, candidate_times.shape)

    if np.any(~np.isfinite(candidate_rates)) or np.any(candidate_rates < 0.0):
        raise ValueError("rate_fn must return finite, non-negative rates")
    tolerance = rate_upper_bound_hz * 1e-12
    if np.any(candidate_rates > rate_upper_bound_hz + tolerance):
        largest = float(candidate_rates.max())
        raise ValueError(
            f"rate_upper_bound_hz={rate_upper_bound_hz:g} is below an evaluated "
            f"rate of {largest:g}"
        )

    keep = generator.random(candidate_count) < candidate_rates / rate_upper_bound_hz
    source = Events(candidate_times[keep])
    detected = apply_detector(source, duration_s, detector, generator)
    return SimulationResult(source, detected, duration_s, "continuous analytical rate")


def simulate_piecewise_constant(
    rates_hz: ArrayLike,
    bin_width_s: float,
    detector: Detector = Detector(),
    rng: np.random.Generator | None = None,
) -> SimulationResult:
    """Simulate continuous arrivals from a sampled, zero-order-hold rate.

    A Poisson count is drawn for every source bin. Conditional on that count,
    arrival times are uniform inside the bin. This permits multiple arrivals
    per bin and does not snap source arrivals to the sampling grid.
    """

    rates = np.asarray(rates_hz, dtype=np.float64)
    if rates.ndim != 1 or rates.size == 0:
        raise ValueError("rates_hz must be a non-empty one-dimensional array")
    if np.any(~np.isfinite(rates)) or np.any(rates < 0.0):
        raise ValueError("rates_hz must contain finite, non-negative values")
    if not math.isfinite(bin_width_s) or bin_width_s <= 0.0:
        raise ValueError("bin_width_s must be positive")

    generator = _rng(rng)
    counts = generator.poisson(rates * bin_width_s)
    bin_indices = np.repeat(np.arange(rates.size, dtype=np.int64), counts)
    times = (bin_indices + generator.random(bin_indices.size)) * bin_width_s
    times.sort()

    duration_s = rates.size * bin_width_s
    source = Events(times)
    detected = apply_detector(source, duration_s, detector, generator)
    return SimulationResult(source, detected, duration_s, "sampled zero-order-hold rate")


def simulate_gaussian_pulse_train(
    duration_s: float,
    mean_rate_hz: float,
    repetition_rate_hz: float,
    pulse_fwhm_s: float,
    background_fraction: float = 0.0,
    detector: Detector = Detector(),
    rng: np.random.Generator | None = None,
) -> SimulationResult:
    """Efficiently simulate a periodic, truncated Gaussian pulse train.

    This avoids the poor acceptance rate that ordinary thinning has for very
    narrow pulses. Pulses are centered at integer multiples of the period and
    truncated halfway between neighboring pulse centers.
    """

    if not all(math.isfinite(v) and v > 0 for v in
               (duration_s, mean_rate_hz, repetition_rate_hz, pulse_fwhm_s)):
        raise ValueError("duration, mean rate, and repetition rate must be positive")
    if pulse_fwhm_s <= 0.0:
        raise ValueError("pulse_fwhm_s must be positive")
    if not 0.0 <= background_fraction < 1.0:
        raise ValueError("background_fraction must be in [0, 1)")

    period = 1.0 / repetition_rate_hz
    if pulse_fwhm_s > period / 3.0:
        raise ValueError("pulse FWHM must be no more than one third of the pulse period")
    sigma = pulse_fwhm_s / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    half_period_z = period / (2.0 * sigma)
    full_pulse_probability = math.erf(half_period_z / math.sqrt(2.0))
    pulse_area_factor = sigma * math.sqrt(2.0 * math.pi) * full_pulse_probability
    background_rate = mean_rate_hz * background_fraction
    pulse_peak_rate = (
        mean_rate_hz * (1.0 - background_fraction) * period / pulse_area_factor
    )

    generator = _rng(rng)
    pieces: list[FloatArray] = []
    background_count = generator.poisson(background_rate * duration_s)
    if background_count:
        pieces.append(generator.uniform(0.0, duration_s, background_count))

    first_center_index = math.floor(-0.5)
    last_center_index = math.ceil(duration_s / period + 0.5)
    for center_index in range(first_center_index, last_center_index + 1):
        center = center_index * period
        lower = max(-period / 2.0, -center)
        upper = min(period / 2.0, duration_s - center)
        if lower >= upper:
            continue

        cdf_lower = _normal_cdf(lower / sigma)
        cdf_upper = _normal_cdf(upper / sigma)
        probability_mass = cdf_upper - cdf_lower
        if probability_mass <= 1e-15:
            continue
        expected_count = (
            pulse_peak_rate
            * sigma
            * math.sqrt(2.0 * math.pi)
            * probability_mass
        )
        count = generator.poisson(expected_count)
        if count:
            offsets = _sample_truncated_normal(
                count, lower, upper, sigma, generator
            )
            pieces.append(center + offsets)

    times = (
        np.sort(np.concatenate(pieces))
        if pieces
        else np.empty(0, dtype=np.float64)
    )
    source = Events(times)
    detected = apply_detector(source, duration_s, detector, generator)
    return SimulationResult(source, detected, duration_s, "continuous Gaussian pulse train")


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _sample_truncated_normal(
    count: int,
    lower: float,
    upper: float,
    sigma: float,
    rng: np.random.Generator,
) -> FloatArray:
    distribution = NormalDist()
    cdf_lower = distribution.cdf(lower / sigma)
    cdf_upper = distribution.cdf(upper / sigma)
    probabilities = cdf_lower + rng.random(count) * (cdf_upper - cdf_lower)
    probabilities = np.clip(
        probabilities,
        np.nextafter(0.0, 1.0),
        np.nextafter(1.0, 0.0),
    )
    return sigma * np.fromiter(
        (distribution.inv_cdf(float(value)) for value in probabilities),
        dtype=np.float64,
        count=count,
    )


def simulate_media(
    frames: ArrayLike,
    frame_period_s: float,
    peak_rate_hz: float,
    detector: Detector = Detector(),
    rng: np.random.Generator | None = None,
) -> SimulationResult:
    """Simulate per-pixel arrivals from a grayscale image or video.

    ``frames`` may have shape ``(height, width)`` or
    ``(frame, height, width)`` and is expected to be normalized to [0, 1].
    Each video frame is treated as a piecewise-constant exposure interval.
    """

    values = np.asarray(frames, dtype=np.float64)
    if values.ndim == 2:
        values = values[np.newaxis, ...]
    if values.ndim != 3 or 0 in values.shape:
        raise ValueError(
            "frames must have shape (height, width) or (frame, height, width)"
        )
    if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("frame values must be finite and normalized to [0, 1]")
    if not math.isfinite(frame_period_s) or frame_period_s <= 0.0:
        raise ValueError("frame_period_s must be positive")
    if not math.isfinite(peak_rate_hz) or peak_rate_hz < 0.0:
        raise ValueError("peak_rate_hz cannot be negative")

    generator = _rng(rng)
    counts = generator.poisson(values * peak_rate_hz * frame_period_s)
    flat_indices = np.repeat(np.arange(values.size, dtype=np.int64), counts.ravel())
    frame, row, col = np.unravel_index(flat_indices, values.shape)
    times = (frame + generator.random(flat_indices.size)) * frame_period_s
    order = np.argsort(times, kind="stable")
    source = Events(times[order], row[order], col[order])

    duration_s = values.shape[0] * frame_period_s
    detected = apply_detector(
        source,
        duration_s,
        detector,
        generator,
        sensor_shape=values.shape[1:],
    )
    return SimulationResult(source, detected, duration_s, "sampled image/video rate")


def apply_detector(
    source_events: Events,
    duration_s: float,
    detector: Detector,
    rng: np.random.Generator | None = None,
    sensor_shape: tuple[int, int] | None = None,
) -> Events:
    """Apply quantum efficiency, dark counts, dead time, jitter, and TDC bins.

    Acquisition gates true arrival times in [0, duration_s). Reported timestamps
    may move outside that interval through jitter/rounding; they are retained.
    This avoids silently losing detections at the acquisition boundaries.
    """

    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    if np.any((source_events.times_s < 0) | (source_events.times_s >= duration_s)):
        raise ValueError("incident events must lie in [0, duration_s)")
    generator = _rng(rng)

    qe_keep = generator.random(len(source_events)) < detector.quantum_efficiency
    times = source_events.times_s[qe_keep]
    rows = source_events.rows[qe_keep] if source_events.rows is not None else None
    cols = source_events.cols[qe_keep] if source_events.cols is not None else None

    if source_events.is_spatial:
        if sensor_shape is None:
            if detector.dark_rate_hz:
                raise ValueError("sensor_shape is required for spatial dark counts")
            assert rows is not None and cols is not None
            height = int(source_events.rows.max()) + 1 if len(source_events) else 0
            width = int(source_events.cols.max()) + 1 if len(source_events) else 0
            sensor_shape = (height, width)
        height, width = sensor_shape
        if height < 0 or width < 0 or int(height) != height or int(width) != width:
            raise ValueError("sensor dimensions must be non-negative integers")
        if (np.any(source_events.rows >= height)
                or np.any(source_events.cols >= width)):
            raise ValueError("event coordinates lie outside sensor_shape")
        times, rows, cols = _add_spatial_dark_counts(
            times, rows, cols, sensor_shape, duration_s, detector.dark_rate_hz, generator
        )
    else:
        dark_count = generator.poisson(detector.dark_rate_hz * duration_s)
        if dark_count:
            times = np.concatenate((times, generator.uniform(0.0, duration_s, dark_count)))

    if source_events.is_spatial:
        assert rows is not None and cols is not None
        order = np.lexsort((times, cols, rows))
        times, rows, cols = times[order], rows[order], cols[order]
        keep = _dead_time_mask_spatial(times, rows, cols, detector)
        times, rows, cols = times[keep], rows[keep], cols[keep]
    else:
        times.sort()
        keep = _dead_time_mask(times, detector)
        times = times[keep]

    if detector.jitter_std_s:
        times = times + generator.normal(0.0, detector.jitter_std_s, times.size)
    if detector.timestamp_resolution_s:
        q = detector.timestamp_resolution_s
        times = np.floor(times / q + 0.5) * q

    if rows is not None and cols is not None:
        order = np.argsort(times, kind="stable")
        return Events(times[order], rows[order], cols[order])
    return Events(np.sort(times))


def _add_spatial_dark_counts(
    times: FloatArray,
    rows: IntArray | None,
    cols: IntArray | None,
    sensor_shape: tuple[int, int],
    duration_s: float,
    dark_rate_hz: float,
    rng: np.random.Generator,
) -> tuple[FloatArray, IntArray, IntArray]:
    assert rows is not None and cols is not None
    height, width = sensor_shape
    if height <= 0 or width <= 0 or dark_rate_hz == 0.0:
        return times, rows, cols
    dark_count = rng.poisson(dark_rate_hz * duration_s * height * width)
    if dark_count == 0:
        return times, rows, cols
    return (
        np.concatenate((times, rng.uniform(0.0, duration_s, dark_count))),
        np.concatenate((rows, rng.integers(0, height, dark_count, dtype=np.int64))),
        np.concatenate((cols, rng.integers(0, width, dark_count, dtype=np.int64))),
    )


def _dead_time_mask(times: FloatArray, detector: Detector) -> NDArray[np.bool_]:
    keep = np.ones(times.size, dtype=bool)
    if detector.dead_time_model == "none" or detector.dead_time_s == 0.0:
        return keep

    ready_at = -np.inf
    for index, arrival in enumerate(times):
        if arrival >= ready_at:
            ready_at = arrival + detector.dead_time_s
        else:
            keep[index] = False
            if detector.dead_time_model == "paralyzable":
                ready_at = arrival + detector.dead_time_s
    return keep


def _dead_time_mask_spatial(
    times: FloatArray,
    rows: IntArray,
    cols: IntArray,
    detector: Detector,
) -> NDArray[np.bool_]:
    keep = np.ones(times.size, dtype=bool)
    if detector.dead_time_model == "none" or detector.dead_time_s == 0.0:
        return keep

    ready_at: dict[tuple[int, int], float] = {}
    for index, (arrival, row, col) in enumerate(zip(times, rows, cols, strict=True)):
        pixel = (int(row), int(col))
        ready = ready_at.get(pixel, -np.inf)
        if arrival >= ready:
            ready_at[pixel] = arrival + detector.dead_time_s
        else:
            keep[index] = False
            if detector.dead_time_model == "paralyzable":
                ready_at[pixel] = arrival + detector.dead_time_s
    return keep


def fourier_probe(
    timestamps_s: ArrayLike,
    frequencies_hz: ArrayLike,
    exposure_s: float,
    chunk_size: int = 4096,
) -> NDArray[np.complex128]:
    """Compute (1/T) sum exp(-j 2 pi f tau) with bounded temporary memory."""

    times = np.asarray(timestamps_s, dtype=np.float64)
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    if times.ndim != 1 or frequencies.ndim != 1:
        raise ValueError("timestamps and frequencies must be one-dimensional")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(frequencies)):
        raise ValueError("timestamps and frequencies must be finite")
    if not math.isfinite(exposure_s) or exposure_s <= 0.0:
        raise ValueError("exposure_s must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    result = np.empty(frequencies.size, dtype=np.complex128)
    max_matrix_elements = 2_000_000
    for start in range(0, frequencies.size, chunk_size):
        stop = min(start + chunk_size, frequencies.size)
        frequency_block = frequencies[start:stop]
        time_chunk_size = max(1, max_matrix_elements // max(1, frequency_block.size))
        total = np.zeros(frequency_block.size, dtype=np.complex128)
        for time_start in range(0, times.size, time_chunk_size):
            time_stop = min(time_start + time_chunk_size, times.size)
            phase = (
                -2j
                * np.pi
                * frequency_block[:, np.newaxis]
                * times[time_start:time_stop]
            )
            total += np.exp(phase).sum(axis=1)
        result[start:stop] = total / exposure_s
    return result


def cfar_amplitude_threshold(
    photon_count: int,
    exposure_s: float,
    false_alarm_probability: float,
) -> float:
    """CFAR amplitude threshold from Eq. 7 of Wei et al."""

    if photon_count < 0:
        raise ValueError("photon_count cannot be negative")
    if not math.isfinite(exposure_s) or exposure_s <= 0.0:
        raise ValueError("exposure_s must be positive")
    if not 0.0 < false_alarm_probability < 1.0:
        raise ValueError("false_alarm_probability must be between 0 and 1")
    return float(
        np.sqrt(-np.log(false_alarm_probability) * photon_count / exposure_s**2)
    )


def load_grayscale_media(
    path: str | Path,
    max_frames: int | None = None,
    spatial_stride: int = 1,
) -> FloatArray:
    """Load a still image, video, or NumPy array as normalized grayscale frames."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    if spatial_stride <= 0:
        raise ValueError("spatial_stride must be positive")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive")

    if source.suffix.lower() == ".npy":
        values = np.asarray(np.load(source), dtype=np.float64)
        if values.size == 0:
            raise ValueError("the NumPy array is empty")
        if values.max() > 1.0:
            values = values / values.max()
        if values.ndim == 2:
            values = values[np.newaxis, ...]
        if values.ndim != 3:
            raise ValueError("NumPy media must have shape (H, W) or (T, H, W)")
        values = values[:max_frames, ::spatial_stride, ::spatial_stride]
        return np.clip(values, 0.0, 1.0)

    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "OpenCV is required to read image/video files; install opencv-python"
        ) from error

    image = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
    if image is not None:
        return (
            image[np.newaxis, ::spatial_stride, ::spatial_stride].astype(np.float64)
            / 255.0
        )

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"could not read media file: {source}")
    frames: list[FloatArray] = []
    while max_frames is None or len(frames) < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(
            gray[::spatial_stride, ::spatial_stride].astype(np.float64) / 255.0
        )
    capture.release()
    if not frames:
        raise ValueError(f"video contains no readable frames: {source}")
    return np.stack(frames)


def save_events(path: str | Path, result: SimulationResult) -> None:
    """Save source and detected event lists in a compact NumPy archive."""

    data: dict[str, ArrayLike] = {
        "source_times_s": result.source_events.times_s,
        "detected_times_s": result.detected_events.times_s,
        "duration_s": np.array(result.duration_s),
        "source_description": np.array(result.source_description),
    }
    if result.source_events.rows is not None and result.source_events.cols is not None:
        data["source_rows"] = result.source_events.rows
        data["source_cols"] = result.source_events.cols
    if result.detected_events.rows is not None and result.detected_events.cols is not None:
        data["detected_rows"] = result.detected_events.rows
        data["detected_cols"] = result.detected_events.cols
    np.savez_compressed(path, **data)
