#!/usr/bin/env python3
"""Command-line experiments for photon_simulator.py."""

from __future__ import annotations

import argparse
import math
import os
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np

from photon_simulator import (
    Detector,
    SimulationResult,
    cfar_amplitude_threshold,
    fourier_probe,
    load_grayscale_media,
    save_events,
    simulate_analytic,
    simulate_gaussian_pulse_train,
    simulate_media,
    simulate_piecewise_constant,
)


def comma_separated_floats(text: str) -> list[float]:
    try:
        return [float(item.strip()) for item in text.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from error


def add_detector_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--quantum-efficiency", type=float, default=1.0)
    parser.add_argument("--dark-rate", type=float, default=0.0, help="dark counts/s/pixel")
    parser.add_argument("--dead-time", type=float, default=0.0, help="seconds")
    parser.add_argument(
        "--dead-time-model",
        choices=("none", "nonparalyzable", "paralyzable"),
        default="none",
    )
    parser.add_argument("--jitter", type=float, default=0.0, help="timestamp std. dev., seconds")
    parser.add_argument("--timestamp-resolution", type=float, default=0.0, help="seconds")
    parser.add_argument("--seed", type=int, default=7)


def detector_from_args(args: argparse.Namespace) -> Detector:
    return Detector(
        quantum_efficiency=args.quantum_efficiency,
        dark_rate_hz=args.dark_rate,
        dead_time_s=args.dead_time,
        dead_time_model=args.dead_time_model,
        jitter_std_s=args.jitter,
        timestamp_resolution_s=args.timestamp_resolution,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulate continuous or sampled photon flux and SPAD timestamps."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analytic = subparsers.add_parser("analytic", help="sum-of-sinusoids analytical flux")
    analytic.add_argument(
        "--waveform", choices=("sinusoid", "pulse-train"), default="sinusoid"
    )
    analytic.add_argument("--duration", type=float, default=0.1, help="seconds")
    analytic.add_argument("--mean-rate", type=float, default=50_000.0, help="photons/s")
    analytic.add_argument(
        "--frequencies", type=comma_separated_floats, default=[70.0, 900.0, 8_000.0]
    )
    analytic.add_argument(
        "--modulations",
        type=comma_separated_floats,
        default=[0.25, 0.20, 0.15],
        help="fractions of mean rate; absolute values must sum to at most 1",
    )
    analytic.add_argument("--pulse-frequency", type=float, default=2_000.0, help="Hz")
    analytic.add_argument("--pulse-fwhm", type=float, default=20e-6, help="seconds")
    analytic.add_argument(
        "--background-fraction",
        type=float,
        default=0.05,
        help="pulse-train background as a fraction of the mean rate",
    )
    analytic.add_argument(
        "--source-step",
        type=float,
        default=0.0,
        help="sample the analytical source at this interval; 0 keeps it continuous",
    )
    analytic.add_argument("--histogram-bins", type=int, default=300)
    analytic.add_argument("--probe-max", type=float, default=10_000.0, help="Hz")
    analytic.add_argument(
        "--probe-step", type=float, default=0.0, help="Hz; 0 uses 0.6/exposure"
    )
    analytic.add_argument("--false-alarm", type=float, default=1e-4)
    analytic.add_argument("--output", type=Path, default=Path("analytic_simulation.png"))
    analytic.add_argument("--events", type=Path, help="optional output .npz event list")
    analytic.add_argument("--show", action="store_true")
    add_detector_arguments(analytic)

    discrete = subparsers.add_parser("discrete", help="one-dimensional sampled flux")
    discrete.add_argument("input", type=Path, help=".npy or text/CSV rates in photons/s")
    discrete.add_argument("--sample-period", type=float, required=True, help="seconds")
    discrete.add_argument("--histogram-bins", type=int, default=200)
    discrete.add_argument("--output", type=Path, default=Path("discrete_simulation.png"))
    discrete.add_argument("--events", type=Path, help="optional output .npz event list")
    discrete.add_argument("--show", action="store_true")
    add_detector_arguments(discrete)

    media = subparsers.add_parser("media", help="per-pixel image or video flux")
    media.add_argument("input", type=Path, help="image, video, or (T,H,W) .npy file")
    media.add_argument("--fps", type=float, default=30.0)
    media.add_argument("--peak-rate", type=float, default=60.0, help="photons/s/pixel")
    media.add_argument("--max-frames", type=int, default=30)
    media.add_argument("--spatial-stride", type=int, default=1)
    media.add_argument("--output", type=Path, default=Path("media_simulation.png"))
    media.add_argument("--events", type=Path, default=Path("media_events.npz"))
    media.add_argument("--show", action="store_true")
    add_detector_arguments(media)

    sweep = subparsers.add_parser(
        "sweep", help="compare continuous flux to increasingly coarse source sampling"
    )
    sweep.add_argument("--duration", type=float, default=0.1)
    sweep.add_argument("--mean-rate", type=float, default=50_000.0)
    sweep.add_argument("--frequency", type=float, default=900.0)
    sweep.add_argument("--modulation", type=float, default=0.8)
    sweep.add_argument(
        "--samples-per-period",
        type=comma_separated_floats,
        default=[64, 32, 16, 8, 4, 2, 1],
    )
    sweep.add_argument("--trials", type=int, default=100)
    sweep.add_argument("--output", type=Path, default=Path("discretization_sweep.png"))
    sweep.add_argument("--show", action="store_true")
    sweep.add_argument("--seed", type=int, default=7)
    return parser


def sinusoidal_rate(
    mean_rate: float, frequencies: Sequence[float], modulations: Sequence[float]
):
    frequencies_array = np.asarray(frequencies, dtype=np.float64)
    modulations_array = np.asarray(modulations, dtype=np.float64)
    if frequencies_array.size != modulations_array.size:
        raise ValueError("frequencies and modulations must have the same length")
    if mean_rate <= 0.0 or np.any(frequencies_array < 0.0):
        raise ValueError("mean rate must be positive and frequencies cannot be negative")
    if np.sum(np.abs(modulations_array)) > 1.0 + 1e-12:
        raise ValueError("absolute modulation fractions must sum to at most 1")

    def rate_fn(times: np.ndarray) -> np.ndarray:
        phase = 2.0 * np.pi * frequencies_array[:, np.newaxis] * np.atleast_1d(times)
        return mean_rate * (1.0 + (modulations_array[:, np.newaxis] * np.sin(phase)).sum(axis=0))

    upper_bound = mean_rate * (1.0 + np.sum(np.abs(modulations_array)))
    return rate_fn, upper_bound


def gaussian_pulse_train_rate(
    mean_rate: float,
    pulse_frequency: float,
    pulse_fwhm_s: float,
    background_fraction: float,
):
    if mean_rate <= 0.0 or pulse_frequency <= 0.0 or pulse_fwhm_s <= 0.0:
        raise ValueError("mean rate, pulse frequency, and pulse FWHM must be positive")
    if not 0.0 <= background_fraction < 1.0:
        raise ValueError("background fraction must be in [0, 1)")
    period = 1.0 / pulse_frequency
    if pulse_fwhm_s > period / 3.0:
        raise ValueError("pulse FWHM must be no more than one third of the pulse period")

    sigma = pulse_fwhm_s / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    gaussian_mean = (
        sigma
        * math.sqrt(2.0 * math.pi)
        * math.erf(period / (2.0 * math.sqrt(2.0) * sigma))
        / period
    )
    background = mean_rate * background_fraction
    peak = mean_rate * (1.0 - background_fraction) / gaussian_mean

    def rate_fn(times: np.ndarray) -> np.ndarray:
        wrapped = (np.atleast_1d(times) + period / 2.0) % period - period / 2.0
        return background + peak * np.exp(-0.5 * (wrapped / sigma) ** 2)

    return rate_fn, background + peak


def run_analytic(args: argparse.Namespace) -> None:
    if args.waveform == "pulse-train":
        rate_fn, upper_bound = gaussian_pulse_train_rate(
            args.mean_rate,
            args.pulse_frequency,
            args.pulse_fwhm,
            args.background_fraction,
        )
    else:
        rate_fn, upper_bound = sinusoidal_rate(
            args.mean_rate, args.frequencies, args.modulations
        )
    detector = detector_from_args(args)
    generator = np.random.default_rng(args.seed)

    sampled_times = None
    sampled_rates = None
    if args.source_step > 0.0:
        bin_count = int(np.ceil(args.duration / args.source_step))
        actual_step = args.duration / bin_count
        sampled_times = (np.arange(bin_count) + 0.5) * actual_step
        sampled_rates = rate_fn(sampled_times)
        result = simulate_piecewise_constant(sampled_rates, actual_step, detector, generator)
    elif args.waveform == "pulse-train":
        result = simulate_gaussian_pulse_train(
            args.duration,
            args.mean_rate,
            args.pulse_frequency,
            args.pulse_fwhm,
            args.background_fraction,
            detector,
            generator,
        )
    else:
        result = simulate_analytic(rate_fn, args.duration, upper_bound, detector, generator)

    print_summary(result)
    if args.events:
        save_events(args.events, result)
        print(f"events:   {args.events}")
    plot_temporal_result(
        result,
        rate_fn,
        args.histogram_bins,
        args.probe_max,
        args.probe_step,
        args.false_alarm,
        args.output,
        args.show,
        sampled_times,
        sampled_rates,
    )


def run_discrete(args: argparse.Namespace) -> None:
    if args.input.suffix.lower() == ".npy":
        rates = np.atleast_1d(np.asarray(np.load(args.input), dtype=np.float64).squeeze())
    else:
        delimiter = "," if args.input.suffix.lower() == ".csv" else None
        rates = np.atleast_1d(np.loadtxt(args.input, delimiter=delimiter))
    if rates.ndim != 1:
        raise ValueError("the discrete rate input must be one-dimensional")
    result = simulate_piecewise_constant(
        rates,
        args.sample_period,
        detector_from_args(args),
        np.random.default_rng(args.seed),
    )
    print_summary(result)
    if args.events:
        save_events(args.events, result)
        print(f"events:   {args.events}")

    def held_rate(times: np.ndarray) -> np.ndarray:
        indices = np.minimum((times / args.sample_period).astype(int), rates.size - 1)
        return rates[indices]

    plot_temporal_result(
        result,
        held_rate,
        args.histogram_bins,
        probe_max=0.5 / args.sample_period,
        probe_step=1.0 / result.duration_s,
        false_alarm=1e-4,
        output=args.output,
        show=args.show,
        sampled_times=(np.arange(rates.size) + 0.5) * args.sample_period,
        sampled_rates=rates,
    )


def run_media(args: argparse.Namespace) -> None:
    frames = load_grayscale_media(args.input, args.max_frames, args.spatial_stride)
    result = simulate_media(
        frames,
        1.0 / args.fps,
        args.peak_rate,
        detector_from_args(args),
        np.random.default_rng(args.seed),
    )
    print_summary(result)
    save_events(args.events, result)
    print(f"events:   {args.events}")
    plot_media_result(frames, result, args.output, args.show)


def run_sweep(args: argparse.Namespace) -> None:
    if args.frequency <= 0.0 or args.duration <= 0.0 or args.trials <= 0:
        raise ValueError("frequency, duration, and trials must be positive")
    if not 0.0 <= args.modulation <= 1.0:
        raise ValueError("modulation must be between 0 and 1")
    samples_per_period = np.asarray(args.samples_per_period, dtype=np.float64)
    if np.any(samples_per_period <= 0.0):
        raise ValueError("samples per period must be positive")

    rate_fn, _ = sinusoidal_rate(args.mean_rate, [args.frequency], [args.modulation])
    generator = np.random.default_rng(args.seed)
    dense_times = np.linspace(0.0, args.duration, 50_000, endpoint=False)
    ground_truth = rate_fn(dense_times)
    model_rmse = []
    probe_amplitude_mean = []
    probe_amplitude_std = []

    for spp in samples_per_period:
        requested_step = 1.0 / (args.frequency * spp)
        bin_count = max(1, int(np.ceil(args.duration / requested_step)))
        step = args.duration / bin_count
        centers = (np.arange(bin_count) + 0.5) * step
        rates = rate_fn(centers)
        held = rates[np.minimum((dense_times / step).astype(int), bin_count - 1)]
        model_rmse.append(np.sqrt(np.mean((held - ground_truth) ** 2)) / args.mean_rate)

        amplitudes = []
        for _ in range(args.trials):
            result = simulate_piecewise_constant(rates, step, rng=generator)
            coefficient = fourier_probe(
                result.detected_events.times_s, [args.frequency], args.duration
            )[0]
            amplitudes.append(2.0 * abs(coefficient) / args.mean_rate)
        probe_amplitude_mean.append(np.mean(amplitudes))
        probe_amplitude_std.append(np.std(amplitudes))

    print("samples/period  source NRMSE  recovered modulation")
    for spp, error, amplitude in zip(
        samples_per_period, model_rmse, probe_amplitude_mean, strict=True
    ):
        print(f"{spp:14g}  {error:12.5f}  {amplitude:20.5f}")
    plot_sweep(
        samples_per_period,
        np.asarray(model_rmse),
        np.asarray(probe_amplitude_mean),
        np.asarray(probe_amplitude_std),
        args.modulation,
        args.output,
        args.show,
    )


def print_summary(result: SimulationResult) -> None:
    print(f"source:   {result.source_description}")
    print(f"duration: {result.duration_s:.6g} s")
    print(f"incident: {len(result.source_events):,}")
    print(f"detected: {len(result.detected_events):,}")
    print(f"detected / incident: {result.detection_fraction:.4f} (includes dark counts)")


def prepare_matplotlib(show: bool):
    cache_dir = Path(tempfile.gettempdir()) / "photon-simulator-matplotlib"
    cache_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_temporal_result(
    result: SimulationResult,
    rate_fn,
    histogram_bins: int,
    probe_max: float,
    probe_step: float,
    false_alarm: float,
    output: Path,
    show: bool,
    sampled_times: np.ndarray | None = None,
    sampled_rates: np.ndarray | None = None,
) -> None:
    if histogram_bins <= 0 or probe_max <= 0.0:
        raise ValueError("histogram bins and probe maximum must be positive")
    if probe_step <= 0.0:
        probe_step = 0.6 / result.duration_s

    plt = prepare_matplotlib(show)
    figure, axes = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True)
    display_times = np.linspace(0.0, result.duration_s, 5000, endpoint=False)
    axes[0].plot(
        display_times,
        rate_fn(display_times),
        color="black",
        lw=1.4,
        label="underlying rate",
    )
    if sampled_times is not None and sampled_rates is not None:
        axes[0].step(
            sampled_times,
            sampled_rates,
            where="mid",
            color="#df6c36",
            lw=1.0,
            alpha=0.9,
            label="sampled / held rate",
        )
    axes[0].set(xlabel="time (s)", ylabel="photons/s", title="Source flux")
    axes[0].legend(loc="upper right")

    times = result.detected_events.times_s
    in_exposure = times[(times >= 0) & (times < result.duration_s)]
    counts, edges = np.histogram(
        in_exposure,
        bins=histogram_bins,
        range=(0.0, result.duration_s),
    )
    rate_estimate = counts / np.diff(edges)
    axes[1].stairs(rate_estimate, edges, color="#2a6fbb", fill=True, alpha=0.65)
    visible_events = in_exposure[:3000]
    event_line_height = max(rate_estimate.max(initial=1.0) * 0.04, 1.0)
    axes[1].vlines(
        visible_events, 0.0, event_line_height, color="black", lw=0.25
    )
    axes[1].set(
        xlabel="time (s)",
        ylabel="counts / bin width",
        title="Detected timestamps and a conventional histogram",
    )

    frequencies = np.arange(0.0, probe_max + 0.5 * probe_step, probe_step)
    coefficients = fourier_probe(
        result.detected_events.times_s, frequencies, result.duration_s
    )
    amplitudes = np.abs(coefficients)
    threshold = cfar_amplitude_threshold(
        len(result.detected_events), result.duration_s, false_alarm
    )
    axes[2].plot(frequencies, amplitudes, color="#2a6fbb", lw=0.8, label="|Fourier probe|")
    axes[2].axhline(threshold, color="#c73e4d", ls="--", label=f"CFAR, alpha={false_alarm:g}")
    axes[2].set(
        xlabel="frequency (Hz)",
        ylabel="probe amplitude (photons/s)",
        title="Absolute-timestamp Fourier probing",
    )
    axes[2].legend(loc="upper right")

    figure.suptitle(
        f"{len(result.detected_events):,} detections from {len(result.source_events):,} arrivals"
    )
    figure.savefig(output, dpi=170)
    print(f"plot:     {output}")
    if show:
        plt.show()
    plt.close(figure)


def plot_media_result(
    frames: np.ndarray,
    result: SimulationResult,
    output: Path,
    show: bool,
) -> None:
    plt = prepare_matplotlib(show)
    rows = result.detected_events.rows
    cols = result.detected_events.cols
    assert rows is not None and cols is not None
    count_image = np.zeros(frames.shape[1:], dtype=np.int64)
    np.add.at(count_image, (rows, cols), 1)

    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    axes[0].imshow(frames[0], cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("First input frame")
    axes[1].imshow(frames.mean(axis=0), cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title("Mean input flux")
    image = axes[2].imshow(count_image, cmap="magma")
    axes[2].set_title("Detected photons")
    figure.colorbar(image, ax=axes[2], shrink=0.8)
    for axis in axes:
        axis.axis("off")
    figure.savefig(output, dpi=170)
    print(f"plot:     {output}")
    if show:
        plt.show()
    plt.close(figure)


def plot_sweep(
    samples_per_period: np.ndarray,
    model_rmse: np.ndarray,
    amplitude_mean: np.ndarray,
    amplitude_std: np.ndarray,
    true_modulation: float,
    output: Path,
    show: bool,
) -> None:
    plt = prepare_matplotlib(show)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    axes[0].plot(samples_per_period, model_rmse, "o-", color="#df6c36")
    axes[0].set(
        xscale="log",
        xlabel="source samples per signal period",
        ylabel="source NRMSE",
        title="Model error from zero-order hold",
    )
    axes[0].invert_xaxis()
    axes[0].grid(alpha=0.25)

    axes[1].errorbar(
        samples_per_period,
        amplitude_mean,
        yerr=amplitude_std,
        fmt="o-",
        color="#2a6fbb",
        capsize=3,
    )
    axes[1].axhline(true_modulation, color="black", ls="--", label="continuous truth")
    axes[1].set(
        xscale="log",
        xlabel="source samples per signal period",
        ylabel="recovered modulation",
        title="Fourier probe across photon trials",
    )
    axes[1].invert_xaxis()
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.savefig(output, dpi=170)
    print(f"plot:     {output}")
    if show:
        plt.show()
    plt.close(figure)


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "analytic":
        run_analytic(args)
    elif args.command == "discrete":
        run_discrete(args)
    elif args.command == "media":
        run_media(args)
    elif args.command == "sweep":
        run_sweep(args)


if __name__ == "__main__":
    main()
