# Notes on Passive Ultra-Wideband Single-Photon Imaging

William Zhang  
September 4, 2026

## The question I am trying to answer

Suppose a pixel receives a time-varying photon flux \(\phi(t)\), in photons per second. I want to simulate the timestamps a single-photon detector would produce.

There are two versions of this problem:

1. \(\phi(t)\) is an actual function that I can evaluate at any time.
2. I only have samples: a one-dimensional array, an image, or frames from a video.

The second case is what we will normally have. The important question is not just how to generate random counts. It is what continuous-time model the samples are supposed to represent, and which information was already lost when those samples were made.

The Passive Ultra-Wideband paper is a useful way to think about this because it does almost the reverse operation. It starts with a sparse list of absolute photon timestamps and asks how much of the continuous flux can be recovered.

## Passive Ultra-Wideband Single-Photon Imaging

### What the setup changes

A standard transient or single-photon lidar measurement is active. A laser emits a pulse, the detector receives a synchronization signal, and every photon is timestamped relative to that pulse. Repeating the experiment builds a histogram over one laser period. This is excellent for picosecond time of flight, but the wrapping throws away slow behavior and anything that is not synchronized with the laser.

This paper removes that synchronization. A free-running SPAD records monotonically increasing timestamps from its own clock:

\[
\mathcal T = \{\tau_1,\tau_2,\ldots,\tau_N\}.
\]

These are absolute timestamps from the beginning of the exposure. Nothing wraps back into one pulse period. Light from a bulb, projector, moving object, and several unrelated lasers can all be in the same stream.

At first this looks worse than histogramming. At picosecond resolution, a one-second exposure has an enormous number of possible bins, almost all of which contain zero photons. The useful idea is to never construct that dense array.

### Photon arrivals as a point process

In the low-flux case, the incident flux \(\phi(t)\) is the rate of an inhomogeneous Poisson process. For a short interval \([t,t+dt)\),

\[
P(\text{one arrival}) \approx \phi(t)dt.
\]

The accumulated number of photons is a counting process \(N(t)\). The paper writes it as

\[
N(t)=\int_0^t \phi(u)\,du + M(t),
\]

where \(M(t)\) is martingale noise. In less formal terms: cumulative photon counts are a noisy observation of cumulative flux, and the future noise has zero conditional mean given what has already happened.

This is the step that avoids the empty-bin problem. The raw list of event times is the representation. A trillion-element histogram is not needed.

### Flux probing

For a known probing function \(p(t)\), evaluate it only at the photon timestamps and add the answers:

\[
p(\mathcal T)=\sum_{\tau\in\mathcal T}p(\tau).
\]

The expectation of this random sum is the inner product with the unknown flux:

\[
E[p(\mathcal T)] = \int_0^{t_{\mathrm{exp}}}p(t)\phi(t)\,dt.
\]

So the timestamps let us ask questions about \(\phi\) without first estimating it at every time. For Fourier probing,

\[
p_f(t)=\frac{1}{t_{\mathrm{exp}}}e^{-j2\pi ft}, \qquad
p_f(\mathcal T)=\frac{1}{t_{\mathrm{exp}}}\sum_{\tau\in\mathcal T}e^{-j2\pi f\tau}.
\]

Each timestamp votes with a phase. A real frequency makes those votes line up; at a frequency that is not present, they mostly cancel. This also explains the initially surprising result that the mean gap between photons is not the maximum time resolution. Individual photons may be far apart, but their phases across a long exposure can still carry evidence of a much faster periodic signal.

The supplementary proves that a general probe is approximately normal with

\[
E[p(\mathcal T)]=\langle p,\phi\rangle,
\qquad
\operatorname{Var}[p(\mathcal T)]=\langle p^2,\phi\rangle.
\]

For complex Fourier probes, the real and imaginary parts get a corresponding covariance model. The authors turn that into a constant false alarm rate (CFAR) test. In the simplified form used by their algorithm, frequency \(f\) is kept when

\[
|p_f(\mathcal T)|^2 \geq
\frac{\operatorname{CDF}^{-1}_{\chi^2_2}(1-\alpha)N(t_{\mathrm{exp}})}
{2t_{\mathrm{exp}}^2}.
\]

Here \(\alpha\) is the probability of accepting a nonexistent frequency. This is more useful than taking a normal FFT and guessing where the noise floor ends.

### Where the bandwidth limit actually comes from

If timestamps are quantized in steps of \(Q\), frequencies above

\[
f_{\max}=\frac{1}{2Q}
\]

alias to lower frequencies. A 16 ps effective resolution gives 31.25 GHz. This is a reconstructability limit, not a promise that any signal below 31.25 GHz will be visible. Detectability still depends on photon count, modulation strength, jitter, dead time, exposure length, and how many candidate frequencies are tested.

Finite exposure matters too. Observing for \(t_{\mathrm{exp}}\) windows the signal, so each spectral line is spread by a sinc-shaped lobe. The supplementary uses a scan spacing of about \(0.6/t_{\mathrm{exp}}\) to avoid stepping over a lobe. A longer exposure collects more photons and also gives finer frequency resolution. It does not create the normal temporal blur of a single integrated camera frame because the absolute timestamps are kept.

### The reconstruction algorithm

The basic low-flux algorithm is:

1. Scan candidate frequencies from DC to \(f_{\max}\).
2. Compute the complex Fourier probe directly from the timestamp list.
3. Reject probes below the CFAR threshold.
4. Reconstruct a continuous sum of the surviving sinusoids.

The result is sparse in frequency, not sampled densely in time. That is how the method can represent both ordinary motion and picosecond laser pulses in the same reconstruction.

The computational cost is still serious. A one-hertz scan out to gigahertz frequencies means billions of probes, and each probe touches the timestamps. The paper points to nonuniform FFTs and sketching, but does not make that cost disappear. The public 1D pipeline also warns that intermediate probing data can take tens of gigabytes.

### Dead time is not just fewer photons

After a SPAD detects a photon, it is blind for its dead-time interval. Detections are then dependent, so the detected stream is no longer Poisson. Bright parts of the flux are suppressed more strongly than dark parts, which changes amplitude and can change apparent contrast.

The supplementary handles this by changing the probing integral from the entire exposure to the union of intervals during which the detector was active. The Fourier basis is no longer orthogonal over those irregular intervals. After frequency detection, the algorithm builds a matrix of pairwise probe inner products over active time and solves a complex least-squares system for corrected Fourier coefficients.

This is worth preserving in a simulator. Simply deleting events and then pretending the surviving stream came from a lower Poisson rate is wrong when dead time is significant. The simulator in this folder includes both nonparalyzable dead time (ignored arrivals do nothing) and paralyzable dead time (an ignored arrival extends the blind period). The paper's detector model is the nonparalyzable case.

### What they demonstrate

The main 1D experiment recovers flux from several unsynchronized sources over roughly nine orders of magnitude, including projector behavior, 900 Hz bulb modulation, and picosecond lasers. The reconstruction uses about 77,000 timestamps rather than a dense time array.

Other experiments are useful for understanding what “one pixel” can contain:

- A raster-scanning laser projector turns a video into a one-dimensional temporal flux. A SPAD looking at indirect light from a diffuse surface recovers passive non-line-of-sight video from that flux.
- A scanned single-pixel experiment renders a spinning fan at ordinary rates and at hundreds of billions of frames per second, showing both object motion and light propagation.
- The method is also applied independently to pixels of a 32 by 32 SPAD array, using the dead-time-aware version.

One qualification matters when explaining the fan and bottle results. Synchronization was used to align separately scanned spatial positions for the final video. It was not used for the per-pixel flux reconstruction itself. The flux probing claim is sync-free; the mechanical construction of a coherent 2D demonstration needed alignment.

## Transient Neural Radiance Fields for Lidar View Synthesis

This paper also uses single-photon measurements, but it is solving a different problem in a different acquisition regime.

### What a “single-photon scan” means here

The system has a pulsed picosecond laser, a coaxial single-pixel SPAD, scanning mirrors, and a time-correlated single-photon counter. At one scan position, the laser pulse is repeated many times. Photon times are measured relative to the laser pulse and accumulated into a histogram. The mirrors raster over a 512 by 512 grid, and the object is rotated to collect multiple viewpoints.

The captured dataset contains six scenes and 20 views per scene. Each view took 20 minutes. The raw histograms have 4096 bins at 4 ps; training uses a cropped/downsampled version with 1500 bins at 8 ps. The total system impulse response is about 70 ps. Measurements stay in the low-flux regime, below about a 5% detection probability per emitted pulse, to avoid pile-up distortion.

This is almost the opposite choice from passive ultra-wideband acquisition:

- Passive ultra-wideband: no controlled illumination, no sync, absolute timestamps, all timescales remain available.
- Transient NeRF lidar: controlled repeated pulses, sync-relative timestamps, deliberately accumulated histograms, detailed time of flight within the pulse window.

Both begin with individual detections, but the representation and the information kept by the acquisition are different.

### Why keep the raw histogram instead of making a point cloud?

A conventional lidar pipeline turns each histogram into one depth, then gives the resulting point cloud to later algorithms. That is an early, lossy decision. A noisy or multi-peaked histogram may not have one unambiguous depth. At a depth edge, the finite laser/sensor footprint can see two surfaces and produce two peaks.

Transient NeRF trains on the full photon-count histograms. Geometry from all views must jointly explain the original measurements. The model can therefore avoid committing to inconsistent per-view point estimates before reconstruction starts.

### The forward model

For an ideal surface at distance \(z\), a returned laser pulse is centered at round-trip delay \(2z/c\). The expected histogram integrates that shifted impulse response over the spatial pixel and temporal bin. Actual bin counts are Poisson variables. The mean includes the number of repeated laser pulses, detector efficiency, ambient photons, and detector dark counts.

The paper replaces the single-surface model with time-resolved volume rendering. Along a ray, the network predicts density \(\sigma\) and view-dependent radiance \(c\). A sample's contribution includes:

- ray termination density \(\sigma\),
- radiance \(c\),
- inverse-square radiometric falloff,
- squared transmittance \(T(t)^2\), because light travels from the sensor/laser to the point and back,
- assignment to a time bin based on propagation distance.

The ideal rendered transient is then convolved with the calibrated laser-pulse/sensor impulse response. This is not a cosmetic blur. If the impulse response is omitted, the optimizer can explain the temporal width of a measured return by inventing a thick cloud of geometry. Modeling the instrument response acts like doing the correct deconvolution during reconstruction.

### Representation and losses

The scene uses an Instant-NGP-style multiresolution hash grid and a small decoder that outputs density and radiance. Training renders a histogram for each sampled pixel and compares it with the captured histogram.

The lidar data has a large dynamic range, so the method predicts radiance in a strongly expanded positive parameterization and uses an L1 loss after \(\log(1+x)\). Otherwise a few bright returns dominate optimization.

There is also a space-carving loss. If a time bin is at or below expected background, density at the corresponding place on the ray is penalized. Without it, the network can put a dark, radiance-zero cloud in front of a real surface. That cloud changes transmittance without directly producing light, so it can fit the measurements with wrong geometry.

Finally, several rays are sampled per pixel using a Gaussian footprint. This matters at depth discontinuities, where one ideal center ray cannot explain a two-peak return.

### Results and limitations

On both simulated and captured data, training on raw transients produces better novel-view intensity and geometry than the compared methods trained on intensity images plus derived point clouds, especially with only two to five input views. The interesting result is not only a cleaner depth map. The trained model can render a complete time-resolved lidar scan from a new viewpoint.

The current model assumes coaxial, single-bounce direct returns. The real data contains richer multipath effects that the model does not explain. Captured reconstruction is also sensitive to roughly millimeter-scale calibration errors; when histogram bins correspond to picosecond timing, small spatial registration errors matter.

## Continuous flux, sampled flux, and discrete timestamps

These are three separate things and should not be mixed together.

### 1. Continuous analytical flux

If \(\phi(t)\) can be evaluated anywhere and has a known upper bound \(\phi_{\max}\), rejection thinning gives an exact inhomogeneous Poisson simulation:

1. Generate a homogeneous Poisson process at rate \(\phi_{\max}\).
2. Keep a candidate at time \(t\) with probability \(\phi(t)/\phi_{\max}\).

This does not use a time grid. It can be inefficient for an extremely narrow pulse because almost all candidates land where the flux is small. The ultra-wideband supplementary uses a cumulative-intensity method for its narrow pulse-train simulations for this reason. The simulator also has a specialized Gaussian pulse-train path that draws the Poisson number of pulse photons and samples their truncated Gaussian offsets directly, avoiding that rejection problem.

### 2. Discrete source samples

Suppose the input is rates \(\phi_k\) at interval width \(\Delta_s\). We have to choose an interpolation model. The simulator uses a zero-order hold:

\[
\phi(t)=\phi_k, \qquad t\in[k\Delta_s,(k+1)\Delta_s).
\]

This model can still be simulated exactly. For bin \(k\), draw

\[
N_k\sim\operatorname{Poisson}(\phi_k\Delta_s),
\]

then place those \(N_k\) photons uniformly at random inside the interval. The input rate is discrete, but the generated arrival times are continuous.

An image is the same model with one constant rate per pixel during an exposure. A video adds a frame dimension; each pixel gets a piecewise-constant rate over frame intervals. A video value might represent an instantaneous sample or an average over the camera's shutter. Treating it as constant is an assumption, not something guaranteed by the file.

### 3. Detector timestamp quantization

Only after arrivals hit the detector do we apply quantum efficiency, dark counts, dead time, timing jitter, and time-to-digital quantization. A source frame period of 1/30 s and a SPAD timestamp step of 16 ps describe completely different discretizations. Keeping both controls separate is necessary.

## What happens when the flux is discretized too aggressively?

“Too much” can mean bins that are too coarse or a grid that is unnecessarily fine.

With coarse source bins:

- Frequencies above \(1/(2\Delta_s)\) cannot be uniquely represented by ordinary samples and may alias lower.
- If samples are exposure averages, the exposure is a box filter with a sinc frequency response. High frequencies are attenuated even before Nyquist is reached.
- A pulse much narrower than a source bin loses its position and peak height. Preserving total photons gives a short, tall pulse and a long, low block the same bin average.
- Zero-order hold introduces sharp artificial edges and therefore extra harmonics. Linear interpolation would introduce a different bias; it cannot restore information that was never sampled.
- Dead time reacts nonlinearly to local peaks. Replacing a narrow peak by its bin average can predict the total incident count correctly while predicting the detected count incorrectly.

With an extremely fine source grid:

- Most \(\phi_k\Delta_s\) values are much less than one, so a dense count array is almost entirely zeros. Event lists are a better representation.
- Approximating each bin by a Bernoulli trial forbids multiple photons and is biased unless \(\phi_k\Delta_s\ll1\). Drawing a Poisson count per bin avoids that approximation.
- Memory and runtime grow with the number of bins even though the physical information is capped by jitter, detector quantization, and photon count.
- Eventually floating-point time precision becomes another practical limit for very long exposures with extremely small steps.

There are therefore two Nyquist-like limits in a sampled simulation:

\[
f_{\max,\,source}\approx\frac{1}{2\Delta_s},
\qquad
f_{\max,\,timestamp}=\frac{1}{2Q}.
\]

The first expression is an identifiability limit for an unknown band-limited signal before uniform sampling, not a hard spectral cutoff of the held function. A zero-order hold introduces discontinuities and therefore higher-frequency components of its own. Detecting those components does not recover the original within-bin variation. For recovering an unknown original source, the usable bandwidth is constrained by both sampling and timestamp resolution, and usually further reduced by jitter, finite exposure, low photon count, and dead time. The ultra-wideband result removes the average inter-photon interval as a hard bandwidth limit; it does not remove limits introduced by the source data itself.

## Simulator in this folder

The event-generation library is in [photon_simulator.py](photon_simulator.py), with command-line experiments in [simulate.py](simulate.py). There is now an interactive browser version too: run `python interactive.py` and open `http://127.0.0.1:8765`. [README.md](README.md) has the setup and a short walkthrough for presenting it.

The browser plots reference flux, the source actually simulated, incident and recorded timestamps, a count histogram, and full-exposure Fourier probes together. Image/video mode adds frame playback and pixel selection. Source and detector settings generate a new run; zooming, panning, and changing histogram bins reuse the same events.

The comparison I would start with is the 900 Hz source sampled into 1 ms bins. Midpoint sampling aliases it to 100 Hz. Bin averaging preserves the integrated flux in each bin, but still loses the within-bin timing. Both generate continuous arrival times inside the held bins. That is the distinction this simulator is meant to make concrete.

The picosecond preset integrates the source exactly over display bins so a narrow pulse cannot disappear just because no plotting sample landed on it. It uses known pulse harmonics and an illustrative 40-term reconstruction, not the paper's full blind recovery procedure. Dead-time compensation and calibrated jitter deconvolution remain unimplemented; the displayed reconstruction uses raw recorded events.

The library includes:

- continuous analytical-rate simulation by rejection thinning,
- exact piecewise-constant simulation for a sampled 1D rate,
- per-pixel image/video simulation,
- quantum efficiency and per-pixel dark counts,
- nonparalyzable or paralyzable dead time,
- Gaussian timestamp jitter and timestamp quantization,
- direct Fourier probing and the paper's simplified CFAR threshold,
- compressed event-list output rather than dense empty time bins.

Install the small dependency set:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Run a continuous sum-of-sinusoids example:

```bash
python3 simulate.py analytic \
  --duration 0.1 \
  --mean-rate 50000 \
  --frequencies 70,900,8000 \
  --modulations 0.25,0.20,0.15 \
  --probe-max 10000
```

Turn the exact same analytical function into a sampled source with 100 microsecond bins:

```bash
python3 simulate.py analytic --source-step 1e-4
```

Run a Gaussian laser pulse train:

```bash
python3 simulate.py analytic \
  --waveform pulse-train \
  --duration 0.01 \
  --mean-rate 200000 \
  --pulse-frequency 20000 \
  --pulse-fwhm 2e-6 \
  --probe-max 200000
```

Add detector effects similar in kind to the ultra-wideband setup:

```bash
python3 simulate.py analytic \
  --quantum-efficiency 0.35 \
  --dark-rate 100 \
  --dead-time 2.31e-7 \
  --dead-time-model nonparalyzable \
  --jitter 16e-12 \
  --timestamp-resolution 4e-12
```

Compare source sampling rates over repeated photon trials:

```bash
python3 simulate.py sweep \
  --frequency 900 \
  --samples-per-period 64,32,16,8,4,2,1 \
  --trials 100
```

For a saved one-dimensional rate array:

```bash
python3 simulate.py discrete rates.npy --sample-period 1e-5
```

For an image or video, grayscale values are normalized to \([0,1]\) and multiplied by `--peak-rate`. The default frame interval is \(1/30\) s. Spatial stride is useful before attempting a large video:

```bash
python3 simulate.py media input.mp4 \
  --fps 30 \
  --peak-rate 60 \
  --max-frames 30 \
  --spatial-stride 4
```

The media command writes an `.npz` event list with time, row, and column for every source and detected event. This is the photon-generation layer that a later transient renderer or reconstruction method can consume. It is not itself a Transient NeRF scene renderer: it does not trace geometry, compute round-trip flight time, or convolve returns with a calibrated laser impulse response.

## Checks I would run before using simulated data for a new method

1. Verify that the mean count approaches \(\int\phi(t)dt\) over repeated trials.
2. Check that photon times within a constant-rate bin are uniform, not snapped to the bin center.
3. Sweep source sample period and detector timestamp resolution independently.
4. Plot expected incident counts and detected counts versus flux when dead time is enabled.
5. Verify the Fourier probe at known frequencies and measure false alarms away from them.
6. Repeat every result over seeds. One sparse photon realization can look convincing by accident.
7. For video, state whether each frame means an instantaneous sample, a shutter average, or a constant rate over the whole frame.
8. For lidar, compare both raw timestamp/histogram statistics and the final depth. A simulator can match one while getting the other wrong.

## Possible next steps

The most useful next extension would be a shared forward model that accepts a time-resolved per-pixel flux cube and can emit either absolute timestamps or sync-relative lidar histograms. That would make the acquisition choice a toggle while holding the underlying light transport fixed.

After that, I would add a calibrated temporal impulse response, wavelength-dependent detection efficiency, afterpulsing, and the exact active-interval least-squares correction from the ultra-wideband supplementary. For a Transient NeRF-style experiment, the source flux should come from a renderer that produces round-trip time-of-flight radiance before Poisson sampling. For the passive case, the same ideal flux should remain on the absolute time axis without wrapping.

## Sources

- [Passive Ultra-Wideband project page](https://www.dgp.toronto.edu/projects/ultra-wideband/)
- [Passive Ultra-Wideband paper](<Passive Ultra-Wideband Single-Photon Imaging Paper (Nousias).pdf>)
- [Passive Ultra-Wideband supplementary](<Passive Ultra-Wideband Supplementary.pdf>)
- [Official Passive Ultra-Wideband 1D code](https://github.com/t-cig/UWB1D)
- [Transient NeRF project page](https://anaghmalik.com/TransientNeRF/)
- [Transient NeRF paper](<Transient Neural Radiance Fields for Lidar View Synthesis and 3D Reconstruction (Malik and Nousias).pdf>)
- [Official Transient NeRF code](https://github.com/anaghmalik/TransientNeRF)
