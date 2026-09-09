# Photon arrival lab

A small simulator for the question: if this is the light arriving at a pixel, what timestamps would a single-photon detector record?

The browser interface puts the source, arrivals, count histogram, and Fourier probes next to each other. The main point is to keep three different things separate: sampling the source, generating photons, and measuring their timestamps.

## Run it

From this folder, with Python 3.10 or newer:

```bash
python3 -m venv .venv             # only if the environment does not already exist
source .venv/bin/activate
python -m pip install -r requirements.txt
python interactive.py
```

Open **http://127.0.0.1:8765**. Stop it with Ctrl+C. If that port is occupied, use `python interactive.py --port 8766`.

The interface uses NumPy and Python's standard library. No frontend build step, account, or internet connection is needed. Image/video decoding happens in the browser; sampled intensities go only to the local Python server. Matplotlib and OpenCV in the existing requirements are for the command-line tools.

## A useful order to try things

1. **Sparse, fast signal.** An 8 kHz sinusoid with a mean arrival rate of 3,000 photons/s. The histogram is sparse, but the full-exposure Fourier probe can pick out the modulation. Click **One period**, then **Full exposure**. Neither operation regenerates photons.
2. **Coarse source bins.** Sample a 900 Hz source at 1 kHz. The held source has a 100 Hz alias. Switch from bin-center sampling to bin averaging, then reduce the source bin width. Averaging preserves the expected count in each bin; it does not preserve the original timing within that bin.
3. **Detector saturation.** Turn the 30 µs dead time off and on. The seed keeps the incident stream identical, so the difference comes from the detector. The Fourier reconstruction here is uncorrected; this is where the paper's active-interval correction would matter.
4. **Picosecond pulses.** 80 ps pulses repeating at 20 MHz, observed over 1 ms. The display integrates narrow pulses instead of hoping a plotting sample lands on one. This preset probes the *known* harmonics, not a blind 10 GHz search. Its truncated reconstruction is not a claim of recovering the exact pulse shape.
5. **Moving image.** Play the frames, then click either image to select a pixel. The temporal plots and spectrum now follow that pixel. Counts are displayed on one fixed scale across all frames. Playback is slowed to 4 frames/s for inspection; the source frame rate is still the one in the settings.

Change settings to update automatically, or uncheck that option and use **Simulate arrivals**. **New photons** increments the seed. Try several seeds before drawing conclusions from a sparse run.

## What the controls mean

| Control | What changes |
| --- | --- |
| Source bin width | The assumed flux itself; arrivals remain continuous inside each held bin. |
| Timestamp step Q | Rounds reported arrival times to the detector's clock grid. |
| Histogram bins, zoom, pan | Only the view of an existing realization. |
| Detection efficiency | Independently retains incident photons before detector dead time. |
| Dark counts | Adds independent candidate events per second, per pixel. |
| Dead time | Rejects events while the detector is inactive. Rejected candidates extend the inactive interval only in the paralyzable model. |
| Timing jitter σ | Adds Gaussian timing error after dead-time filtering. |
| Scan maximum / spacing | Which positive Fourier frequencies are tested over the full exposure. |

Drag across the flux plot to zoom, or enter a start and window in seconds. Double-click resets the view. The source and detector controls show their own units: exposure in ms, source/dead-time intervals in µs, and pulse width/jitter/timestamp step in ps. The Python API and exports use seconds and Hz throughout.

The incident and detector stages have separate random streams. Changing detector settings with the same source and seed preserves incident events. Changing source representation draws a realization from the new source model; those events are not paired one-to-one.

## Sampled input

**CSV / text:** one non-negative rate per bin, in photons/s, separated by commas, whitespace, or semicolons. No header or time column. A single rate is fine. Exposure is the number of rates times the source bin width.

**Image / video:** resize to at most 64 pixels on the longest side; video uses up to the first second and at most 32 frames. The frame-rate control sets the sampling rate and held-frame intervals. A still image is held for one such interval. The checkbox approximately converts decoded sRGB values to linear light. Resizing happens first in the browser, so this is not a calibrated radiometric pipeline. Transparent pixels are composited against black. Values are not stretched to make the brightest pixel white.

Peak flux means the incident rate assigned to intensity 1, per pixel. An ordinary video cannot supply picosecond light transport that was never recorded. Its frames are just the assumed source for this experiment.

For an arbitrary callable, use `simulate_analytic` in [photon_simulator.py](photon_simulator.py). It takes a rate function, exposure, and a valid upper bound for rejection thinning. The browser offers sinusoids, a two-frequency mixture, and Gaussian pulse trains rather than executing arbitrary uploaded code. Existing CLI examples are in [Ultra-Wideband.md](Ultra-Wideband.md#simulator-in-this-folder).

## Reading the Fourier plot

Probes evaluate `sum(exp(-2π i f t)) / T` directly from timestamps. The zero-frequency value is `N/T` and is reported separately. Positive-frequency coefficients are complex; a sinusoid's coefficient magnitude is half its real modulation amplitude. The time-domain estimate adds DC and twice the real part of the accepted positive-frequency terms, at most 40 terms. It is not clipped at zero, so truncation and noise can produce negative lobes.

Frequencies are integer multiples of `1/T`. Auto spacing tests at most 2,000 candidates and can skip a real signal between those candidates. Off-grid frequencies can leak across probes. A larger scan maximum does not guarantee better recovery. The displayed α is a per-probe false-alarm probability under the low-flux approximation, not a confidence level for the entire scan.

The pulse preset makes the paper's basic point visible: average photon spacing is not the same thing as timing resolution. It does **not** reproduce the full paper pipeline. There is no blind frequency refinement, active-interval least-squares correction, calibrated jitter deconvolution, or spatial reconstruction. With strong dead time or very few events, the CFAR line is only a reference. The interface calls these limitations out next to the plots.

Source events lie in `[0, T)`, and the detector starts active at zero. Jitter/rounding can move a reported timestamp outside that interval. Those detections are retained in exports and Fourier probes, not silently discarded; time histograms and frame counts exclude them. Their number is shown in a notice. Dead time acts on pre-jitter candidate times, not on rounded timestamps.

## Save and reproduce a run

**Export run** downloads `photons.npz`: source and detected timestamp arrays, pixel coordinates for spatial data, duration, description, and `config_json` including seed and sampled input. **PNG** saves a labeled plot for a slide. The local server keeps only the last three runs in memory; exported files persist after shutdown.

To regenerate an exported browser run with the same simulator/NumPy version:

```python
import json
import numpy as np
from experiment import create_experiment

with np.load("photons.npz", allow_pickle=False) as data:
    run = create_experiment(json.loads(str(data["config_json"])))
    np.testing.assert_array_equal(
        run.result.detected_events.times_s, data["detected_times_s"]
    )
```

## Checks

```bash
python -m unittest discover -s tests -v
node --check web/app.js
```

The tests cover photon-count statistics, uniform arrivals within held bins, independent pixel dead time, a known Fourier peak, exact narrow-pulse integrals, detector/source seed separation, timestamp boundaries, and view changes without regeneration.

An optional real-browser smoke test is in `tests/browser_smoke.cjs`. With Playwright and Chrome installed, run it while the server is running:

```bash
node tests/browser_smoke.cjs
```

If Playwright is installed elsewhere, set `PLAYWRIGHT_MODULE` to its module directory. `PHOTON_TEST_VIDEO` can point to a short browser-decodable `.mp4` to include video upload. `PHOTON_TEST_URL` changes the server URL; `PHOTON_TEST_OUTPUT` changes the screenshot/download directory (default `/tmp/photon-browser-test`). Playwright is a test dependency, not needed to use the simulator.

Interactive runs are bounded to 10 seconds of exposure, 250,000 expected source-plus-dark events, 100,000 source bins, 2,000 probes, and 30 million event–frequency products per pixel. Large experiments belong in the Python library, with deliberately chosen memory/runtime budgets.

## Files

- [Ultra-Wideband.md](Ultra-Wideband.md): notes on the passive paper, supplementary, Transient NeRF, and discretization.
- `photon_simulator.py`: event generation, detector models, Fourier probes, exports.
- `experiment.py`: reproducible browser experiments and numerically integrated plot data.
- `interactive.py` + `web/`: local server and interactive interface.
- `simulate.py`: original CLI examples and discretization sweep.
- `tests/`: numerical regression checks and browser smoke test.
