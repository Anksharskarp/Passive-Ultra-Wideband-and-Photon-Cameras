/* Display code only. Photon generation and statistical probes live in Python. */
"use strict";

const $ = (id) => document.getElementById(id);
// Canvas plots and HTML legends share the same theme tokens.
const theme = getComputedStyle(document.documentElement);
const themeColor = (name) => theme.getPropertyValue(`--${name}`).trim();
const colors = Object.fromEntries(
  ["teal", "amber", "violet", "blue", "red"].map((name) => [
    name,
    themeColor(name),
  ]),
);
const plotStyle = {
  background: themeColor("plot-bg"),
  grid: themeColor("plot-grid"),
  text: themeColor("plot-text"),
  foreground: themeColor("ink"),
  font: `10px ${theme.getPropertyValue("--mono").trim()}`,
};
const units = {
  duration: 1e-3,
  source_step: 1e-6,
  dead_time: 1e-6,
  pulse_width: 1e-12,
  jitter: 1e-12,
  quantization: 1e-12,
};
const numericKeys = [
  "duration",
  "mean_rate",
  "frequency",
  "slow_frequency",
  "modulation",
  "pulse_width",
  "background",
  "source_step",
  "fps",
  "qe",
  "dark_rate",
  "dead_time",
  "jitter",
  "quantization",
  "probe_max",
  "probe_step",
  "alpha",
  "seed",
];
const base = {
  kind: "sine",
  duration: 0.2,
  mean_rate: 3000,
  frequency: 8000,
  slow_frequency: 50,
  modulation: 0.85,
  pulse_width: 80e-12,
  background: 0.05,
  sampling: "continuous",
  source_step: 0.0001,
  fps: 24,
  qe: 1,
  dark_rate: 0,
  dead_time: 0,
  dead_model: "none",
  jitter: 0,
  quantization: 0,
  probe_max: 10000,
  probe_step: 0,
  alpha: 0.001,
  seed: 7,
};
const presets = {
  sparse: {
    config: {},
    window: 0.01,
    title: "Sparse arrivals, fast modulation",
    body: "The source varies at 8 kHz while only about 3,000 photons arrive per second. The Fourier probe uses absolute timing across the whole exposure; enough repeated phase information can reveal this fast signal even when the histogram is mostly noise.",
  },
  alias: {
    config: {
      frequency: 900,
      mean_rate: 12000,
      sampling: "midpoint",
      source_step: 0.001,
      probe_max: 2500,
    },
    window: 0.025,
    title: "Aliasing from coarse source bins",
    body: "A 900 Hz sinusoid sampled at 1,000 samples per second aliases to 100 Hz. Compare bin-center sampling with bin averaging, then reduce source bin width. Arrivals are continuous inside each held bin, but the original within-bin variation has already been lost.",
  },
  dead: {
    config: {
      frequency: 250,
      mean_rate: 40000,
      dead_time: 30e-6,
      dead_model: "nonparalyzable",
      probe_max: 2500,
    },
    window: 0.02,
    title: "Dead-time distortion",
    body: "After each detection the SPAD is blind for 30 µs. Compare the recorded histogram with the expected rate before dead time, then turn dead time off with the same seed. The incident photons stay identical. The displayed Fourier estimate does not compensate for this nonlinear distortion.",
  },
  pulse: {
    config: {
      kind: "pulse",
      duration: 0.001,
      mean_rate: 200000,
      frequency: 20000000,
      jitter: 16e-12,
      quantization: 4e-12,
      dead_time: 231e-9,
      dead_model: "nonparalyzable",
      probe_max: 1e10,
      probe_step: 2e7,
    },
    window: 50e-9,
    title: "Picosecond pulse train",
    body: "A 20 MHz train of 80 ps pulses is observed for 1 ms. This example probes the known 20 MHz harmonics out to 10 GHz; it does not discover the pulse repetition rate through a full blind scan. The reconstruction keeps only 40 harmonics, so it cannot reproduce the full narrow pulse profile. Zoom out to see why preserving absolute timestamps matters.",
  },
  media: {
    config: { kind: "media", mean_rate: 120, probe_max: 300 },
    window: 1,
    title: "Independent per-pixel arrivals",
    body: "Play the frames and click a pixel to inspect its source flux, arrivals, and spectrum. Each frame is a held rate, with an independent detector at each pixel. Brightness sets expected counts; it does not force a fixed number of photons into a frame.",
  },
};

let result = null,
  requestVersion = 0,
  viewVersion = 0,
  timer,
  playback = null;
let selectedPixel = [0, 0],
  selectedFrame = 0,
  pendingWindow = 0.01;
let mediaFrames = null,
  mediaFile = null,
  mediaVersion = 0,
  mediaLoading = false;
const charts = new Map();

function format(value, unit = "", digits = 3) {
  if (!Number.isFinite(value)) return "—";
  if (value === 0) return `0${unit ? " " + unit : ""}`;
  const scales = [
    [1e12, "T"],
    [1e9, "G"],
    [1e6, "M"],
    [1e3, "k"],
    [1, ""],
    [1e-3, "m"],
    [1e-6, "µ"],
    [1e-9, "n"],
    [1e-12, "p"],
  ];
  const [scale, prefix] = scales.find(
    ([s]) => Math.abs(value) >= s * 0.999999,
  ) || [1e-12, "p"];
  return `${Number((value / scale).toPrecision(digits))}${unit ? " " : ""}${prefix}${unit}`;
}

function showError(error) {
  $("error").hidden = false;
  $("error").textContent = error.message || String(error);
  $("status").textContent = result
    ? "Settings not applied — plots still show the previous run."
    : "Check settings and try again.";
}

function setControls(config) {
  for (const [key, value] of Object.entries(config)) {
    if ($(key))
      $(key).value =
        typeof value === "number"
          ? Number((value / (units[key] || 1)).toPrecision(12))
          : value;
  }
  syncFields();
}

function syncFields() {
  const kind = $("kind").value;
  document.querySelectorAll("[data-source]").forEach((node) => {
    node.hidden = !node.dataset.source.split(" ").includes(kind);
    node.querySelectorAll("input,select,textarea").forEach((control) => {
      control.disabled = node.hidden;
    });
  });
  $("source-step-label").hidden =
    kind === "media" ||
    (kind !== "sampled" && $("sampling").value === "continuous");
  $("source_step").disabled = $("source-step-label").hidden;
  $("duration").disabled = kind === "media" || kind === "sampled";
  $("mean_rate").disabled = kind === "sampled";
  $("rate-label").firstChild.textContent =
    kind === "media" ? "Peak flux " : "Mean flux ";
  $("qe-value").value = `${Math.round(Number($("qe").value) * 100)}%`;
  $("modulation-value").value = Number($("modulation").value).toFixed(2);
}

function collectConfig() {
  const config = {};
  for (const key of numericKeys) {
    const value = Number($(key).value) * (units[key] || 1);
    if (!Number.isFinite(value) || $(key).value.trim() === "")
      throw new Error(`Enter a number for ${key.replaceAll("_", " ")}.`);
    config[key] = value;
  }
  for (const key of ["kind", "sampling", "dead_model"])
    config[key] = $(key).value;
  if (config.kind === "sampled") {
    if (!$("rates").value.trim())
      throw new Error("Enter at least one source rate in photons/s.");
    config.rates = $("rates")
      .value.trim()
      .split(/[\s,;]+/)
      .map(Number);
    if (
      !config.rates.length ||
      config.rates.some((v) => !Number.isFinite(v) || v < 0)
    )
      throw new Error(
        "Use only non-negative rates in photons/s, without a CSV header.",
      );
    config.duration = config.rates.length * config.source_step;
  }
  if (config.kind === "media") {
    if (mediaLoading) throw new Error("Wait for the media to finish loading.");
    if (!mediaFrames) mediaFrames = makeDemo(Number($("media-size").value));
    config.frames = mediaFrames;
    config.duration = mediaFrames.length / config.fps;
    config.media_name =
      mediaFile?.name || "Built-in moving target (linear intensity)";
    config.linearized = !mediaFile || $("linearize").checked;
    config.resize_longest_side = Number($("media-size").value);
  }
  return config;
}

async function post(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok)
    throw new Error(data.error || `Request failed (${response.status}).`);
  return data;
}

function markDirty(custom = true) {
  clearTimeout(timer);
  requestVersion++;
  document.body.classList.remove("busy");
  document.body.classList.add("stale");
  $("export").disabled = true;
  $("status").textContent = "Settings changed — run to update the plots.";
  if (custom) {
    document
      .querySelectorAll("[data-preset]")
      .forEach((button) => button.classList.remove("active"));
    $("interpret-title").textContent = "Custom experiment";
    $("interpret-body").textContent =
      "Compare reference flux, the source actually simulated, and recorded detections. Source bins remove within-bin information; histogram bins only change the view. Fourier probes use the complete exposure. Keep the seed fixed to compare detector settings, then try other seeds before drawing conclusions.";
  }
  if ($("auto-run").checked) timer = setTimeout(simulate, 400);
}

async function simulate() {
  clearTimeout(timer);
  if (mediaLoading) return;
  stopPlayback();
  syncFields();
  const version = ++requestVersion;
  viewVersion++;
  $("export").disabled = true;
  document.body.classList.remove("busy");
  document.body.classList.add("stale");
  let config;
  try {
    config = collectConfig();
  } catch (error) {
    showError(error);
    return;
  }
  const duration = config.duration;
  const previousSpan = result ? Number($("view-span").value) : pendingWindow;
  let span = Math.min(duration, pendingWindow ?? previousSpan);
  let start =
    pendingWindow !== null
      ? 0
      : Math.min(Number($("view-start").value), duration - span);
  if (!(span > 0)) span = duration;
  config.view = {
    start: Math.max(0, start),
    stop: Math.max(0, start) + span,
    bins: Number($("bins").value),
    row: selectedPixel[0],
    col: selectedPixel[1],
  };
  if (config.kind !== "media") config.view.row = config.view.col = 0;
  $("error").hidden = true;
  $("status").textContent = "Simulating arrivals and probing frequencies…";
  document.body.classList.add("busy");
  const began = performance.now();
  try {
    const data = await post("/api/run", config);
    if (version !== requestVersion) return;
    result = data;
    if (["media", "sampled"].includes(config.kind))
      $("duration").value = Number(
        (data.summary.duration / units.duration).toPrecision(12),
      );
    selectedPixel = data.view.pixel;
    selectedFrame = 0;
    pendingWindow = null;
    $("export").disabled = false;
    document.body.classList.remove("stale");
    $("status").textContent =
      `Run ready · seed ${config.seed} · ${format(data.summary.duration, "s")} exposure`;
    $("timing").textContent =
      `${((performance.now() - began) / 1000).toFixed(2)} s`;
    updateMetrics();
    updateViewControls();
    renderAll();
  } catch (error) {
    if (version === requestVersion) showError(error);
  } finally {
    if (version === requestVersion) document.body.classList.remove("busy");
  }
}

async function changeView(options = {}) {
  if (!result) return;
  const version = ++viewVersion,
    runId = result.id;
  const start = Number($("view-start").value),
    span = Number($("view-span").value);
  if (!Number.isFinite(start) || !Number.isFinite(span) || span <= 0) return;
  const end = Math.min(result.summary.duration, start + span);
  try {
    const data = await post("/api/view", {
      id: runId,
      start,
      stop: end,
      bins: Number($("bins").value),
      row: selectedPixel[0],
      col: selectedPixel[1],
      frame: selectedFrame,
      ...options,
      // A frame/zoom request can overtake a pending pixel request. Keep its
      // spectrum attached to the same pixel as the returned time plots.
      spectrum:
        options.spectrum ||
        selectedPixel.some((value, i) => value !== result.view.pixel[i]),
    });
    if (version !== viewVersion || result.id !== runId) return;
    result.view = data.view;
    if (data.spectrum) result.spectrum = data.spectrum;
    $("error").hidden = true;
    updateViewControls();
    renderAll();
  } catch (error) {
    if (version === viewVersion) {
      stopPlayback();
      showError(error);
    }
  }
}

function updateViewControls() {
  const data = result.view;
  $("view-start").value = Number(data.start.toPrecision(10));
  $("view-span").value = Number((data.stop - data.start).toPrecision(10));
  $("pan").value = Math.round(
    (1000 * data.start) /
      (result.summary.duration - (data.stop - data.start) || 1),
  );
  $("media-panel").hidden = !data.media;
  $("zoom-period").disabled = ["sampled", "media"].includes(result.config.kind);
  if (data.media) {
    $("play").disabled = data.media.total_frames < 2;
    $("frame").max = data.media.total_frames - 1;
    $("frame").value = data.media.frame;
    $("frame-label").value =
      `${data.media.frame + 1} / ${data.media.total_frames}`;
    $("pixel-label").textContent = `Pixel (${data.pixel.join(", ")})`;
  }
}

function updateMetrics() {
  const s = result.summary;
  $("incident").textContent = s.incident.toLocaleString();
  $("detected").textContent = s.detected.toLocaleString();
  $("expected").textContent =
    `${format(s.expected_incident)} expected from source`;
  $("dark-count").textContent =
    `Dark events: ${format(s.expected_dark)} expected`;
  $("rate").textContent = format(s.recorded_rate);
  $("nyquist").textContent =
    s.nyquist === null ? "Ideal timing" : format(s.nyquist, "Hz");
  $("nyquist-note").textContent =
    s.nyquist === null
      ? "No timestamp quantization"
      : "Quantization bound, not guaranteed recovery";
}

function renderAll() {
  if (!result) return;
  drawFlux();
  drawEvents();
  drawHistogram();
  drawSpectrum();
  drawMedia();
  const notes = [...result.summary.notices];
  if (result.view.media) {
    const pixelCount = Math.round(result.spectrum.dc * result.summary.duration);
    notes.push(
      `Top counts are sensor-wide. Temporal plots use pixel (${result.view.pixel.join(", ")}), with ${pixelCount} recorded events across the exposure.`,
    );
    if (pixelCount < 30)
      notes.push(
        "This pixel has very few detections; its Gaussian/CFAR approximation can be unreliable.",
      );
  }
  if (result.spectrum.sparse)
    notes.push(
      `Sparse frequency scan at ${format(result.spectrum.step, "Hz")} spacing. Frequencies between probes were not tested.`,
    );
  if (result.spectrum.passed > 40)
    notes.push(
      `${result.spectrum.passed} probes passed the threshold; the reconstruction displays only the strongest 40.`,
    );
  if (result.summary.source_step)
    notes.push(
      `Actual source bin width: ${format(result.summary.source_step, "s")}. Equal bins span the complete exposure.`,
    );
  if (
    result.summary.nyquist &&
    result.config.probe_max > result.summary.nyquist
  )
    notes.push(
      "The scan extends above the timestamp Nyquist limit. Peaks above that limit have indistinguishable aliases.",
    );
  $("notices").replaceChildren(
    ...notes.map((text) => {
      const p = document.createElement("p");
      p.textContent = text;
      return p;
    }),
  );
}

function setupCanvas(id) {
  const canvas = $(id),
    rect = canvas.getBoundingClientRect();
  const width = Math.max(100, rect.width),
    height = Math.max(60, rect.height),
    ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  ctx.fillStyle = plotStyle.background;
  ctx.fillRect(0, 0, width, height);
  return { canvas, ctx, width, height };
}

function chart(id, x0, x1, y0, y1, xUnit, yLabel, log = false) {
  const plot = setupCanvas(id),
    { ctx, width, height } = plot;
  const area = { left: 65, right: width - 22, top: 24, bottom: height - 36 };
  if (y1 <= y0) y1 = y0 + 1;
  if (x1 <= x0) x1 = x0 + 1;
  const transform = log ? (v) => Math.log10(Math.max(v, x0)) : (v) => v;
  const x = (value) =>
    area.left +
    ((transform(value) - transform(x0)) / (transform(x1) - transform(x0))) *
      (area.right - area.left);
  const y = (value) =>
    area.bottom - ((value - y0) / (y1 - y0)) * (area.bottom - area.top);
  const unx = (pixel) => {
    const t = (pixel - area.left) / (area.right - area.left);
    return log
      ? 10 ** (Math.log10(x0) + t * (Math.log10(x1) - Math.log10(x0)))
      : x0 + t * (x1 - x0);
  };
  ctx.lineWidth = 1;
  ctx.font = plotStyle.font;
  for (let i = 0; i <= 4; i++) {
    const value = y0 + ((y1 - y0) * i) / 4,
      pixel = y(value);
    ctx.strokeStyle = plotStyle.grid;
    ctx.beginPath();
    ctx.moveTo(area.left, pixel);
    ctx.lineTo(area.right, pixel);
    ctx.stroke();
    ctx.fillStyle = plotStyle.text;
    ctx.textAlign = "right";
    ctx.fillText(format(value), area.left - 9, pixel + 3);
  }
  const ticks = width < 430 ? 3 : 5;
  for (let i = 0; i <= ticks; i++) {
    const value = log
      ? 10 ** (Math.log10(x0) + ((Math.log10(x1) - Math.log10(x0)) * i) / ticks)
      : x0 + ((x1 - x0) * i) / ticks;
    ctx.fillStyle = plotStyle.text;
    ctx.textAlign = i === 0 ? "left" : i === ticks ? "right" : "center";
    ctx.fillText(format(value, xUnit), x(value), height - 13);
  }
  ctx.fillStyle = plotStyle.text;
  ctx.textAlign = "left";
  ctx.fillText(yLabel, area.left, 12);
  Object.assign(plot, { area, x, y, unx, x0, x1, y0, y1 });
  charts.set(id, plot);
  return plot;
}

function line(plot, xs, ys, color, dashed = false, width = 1.5) {
  const { ctx, area, x, y } = plot;
  ctx.save();
  ctx.beginPath();
  ctx.rect(area.left, area.top, area.right - area.left, area.bottom - area.top);
  ctx.clip();
  ctx.beginPath();
  ctx.lineWidth = width;
  ctx.strokeStyle = color;
  ctx.setLineDash(dashed ? [4, 4] : []);
  for (let i = 0; i < xs.length; i++) {
    if (i === 0) ctx.moveTo(x(xs[i]), y(ys[i]));
    else ctx.lineTo(x(xs[i]), y(ys[i]));
  }
  ctx.stroke();
  ctx.restore();
}

function reconstructed(edges) {
  const spec = result.spectrum,
    output = Array(edges.length - 1).fill(spec.dc);
  for (const index of spec.selected) {
    const f = spec.frequencies[index],
      re = spec.real[index],
      im = spec.imag[index];
    for (let i = 0; i < output.length; i++) {
      const center = (edges[i] + edges[i + 1]) / 2,
        d = Math.PI * f * (edges[i + 1] - edges[i]);
      const attenuation = Math.abs(d) < 1e-10 ? 1 : Math.sin(d) / d;
      const phase = 2 * Math.PI * f * center;
      output[i] +=
        2 * (re * Math.cos(phase) - im * Math.sin(phase)) * attenuation;
    }
  }
  return output;
}

function drawFlux() {
  const data = result.view,
    xs = data.edges.slice(1).map((end, i) => (end + data.edges[i]) / 2);
  const reconstruction = $("reconstruction").checked
    ? reconstructed(data.edges)
    : [];
  const high =
    Math.max(1, ...data.reference, ...data.source, ...reconstruction) * 1.1;
  const low = Math.min(0, ...reconstruction) * 1.1;
  const plot = chart(
    "flux",
    data.start,
    data.stop,
    low,
    high,
    "s",
    "photons / s",
  );
  line(plot, xs, data.reference, colors.teal, false, 1.8);
  line(plot, xs, data.source, colors.amber, true, 1.4);
  if (reconstruction.length)
    line(plot, xs, reconstruction, colors.violet, false, 1.25);
  plot.xs = xs;
  plot.ys = data.source;
  plot.unit = "photons/s";
  $("flux-note").textContent =
    `Flux averaged over ${format((data.stop - data.start) / 800, "s")} display bins. Fourier estimate is in detected-rate units.`;
}

function drawEvents() {
  const data = result.view,
    plot = setupCanvas("events"),
    { ctx, width, height } = plot;
  const left = 65,
    right = width - 22,
    scale = (t) =>
      left + ((t - data.start) / (data.stop - data.start)) * (right - left);
  ctx.font = plotStyle.font;
  [
    [data.incident, colors.amber, 27, "Incident"],
    [data.detected, colors.blue, 55, "Recorded"],
  ].forEach(([times, color, y, label]) => {
    ctx.textAlign = "right";
    ctx.fillStyle = plotStyle.text;
    ctx.fillText(label, left - 9, y + 3);
    ctx.strokeStyle = plotStyle.grid;
    ctx.beginPath();
    ctx.moveTo(left, y);
    ctx.lineTo(right, y);
    ctx.stroke();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.beginPath();
    times.forEach((t) => {
      ctx.moveTo(scale(t), y - 6);
      ctx.lineTo(scale(t), y + 6);
    });
    ctx.stroke();
  });
  ctx.fillStyle = plotStyle.text;
  ctx.textAlign = "left";
  const capped = data.incident_visible > 6000 || data.detected_visible > 6000;
  const counts = `${data.incident_visible.toLocaleString()} incident · ${data.detected_visible.toLocaleString()} recorded`;
  const detail = capped
    ? "First 6,000 ticks per row"
    : "Each tick is one event";
  const caption = `${counts} in this window · ${detail.toLowerCase()}`;
  if (ctx.measureText(caption).width <= right - left) {
    ctx.fillText(caption, left, height - 9);
  } else {
    ctx.fillText(counts, left, height - 21, right - left);
    ctx.fillText(detail, left, height - 7, right - left);
  }
}

function drawHistogram() {
  const data = result.view,
    edges = data.hist_edges;
  const rates = data.counts.map((n, i) => n / (edges[i + 1] - edges[i]));
  const high = Math.max(1, ...rates, ...data.hist_expected) * 1.1;
  const plot = chart(
    "histogram",
    data.start,
    data.stop,
    0,
    high,
    "s",
    "counts / bin width (s)",
  );
  const { ctx, x, y } = plot;
  ctx.fillStyle = colors.blue;
  ctx.globalAlpha = 0.65;
  rates.forEach((rate, i) =>
    ctx.fillRect(
      x(edges[i]),
      y(rate),
      Math.max(0.3, x(edges[i + 1]) - x(edges[i]) - 1),
      y(0) - y(rate),
    ),
  );
  const xs = edges.slice(1).map((end, i) => (end + edges[i]) / 2);
  ctx.globalAlpha = 1;
  line(plot, xs, data.hist_expected, colors.amber, true);
  plot.xs = xs;
  plot.ys = rates;
  plot.unit = "counts/s";
}

function drawSpectrum() {
  const spec = result.spectrum,
    fs = spec.frequencies,
    amplitudes = spec.amplitudes;
  const max = Math.max(1, spec.threshold, ...amplitudes) * 1.1;
  const log = $("log-frequency").checked && fs.length > 1;
  const plot = chart(
    "spectrum",
    log ? fs[0] : 0,
    fs.at(-1) || result.config.probe_max,
    0,
    max,
    "Hz",
    "|coefficient| (photons/s)",
    log,
  );
  line(plot, fs, amplitudes, colors.violet, false, 1.1);
  line(
    plot,
    [plot.x0, plot.x1],
    [spec.threshold, spec.threshold],
    colors.red,
    true,
    1,
  );
  const { ctx, x, y } = plot;
  ctx.fillStyle = colors.violet;
  spec.selected.forEach((i) => {
    ctx.beginPath();
    ctx.arc(x(fs[i]), y(amplitudes[i]), 2.5, 0, 2 * Math.PI);
    ctx.fill();
  });
  const nyquist = result.summary.nyquist;
  if (nyquist && nyquist > plot.x0 && nyquist < plot.x1) {
    ctx.save();
    ctx.strokeStyle = colors.red;
    ctx.setLineDash([3, 4]);
    ctx.beginPath();
    ctx.moveTo(x(nyquist), plot.area.top);
    ctx.lineTo(x(nyquist), plot.area.bottom);
    ctx.stroke();
    ctx.restore();
  }
  plot.xs = fs;
  plot.ys = amplitudes;
  plot.unit = "photons/s";
  $("spectrum-note").textContent =
    `Whole exposure · DC ${format(spec.dc)} photons/s · ${fs.length} probes at ${format(spec.step, "Hz")} spacing · ≈${Number(spec.expected_false_alarms.toPrecision(2))} expected false alarms under the approximation.`;
  const nodes = spec.selected.slice(0, 5).map((i) => {
    const node = document.createElement("div"),
      freq = document.createElement("span"),
      amp = document.createElement("span");
    freq.textContent = format(fs[i], "Hz");
    amp.textContent = `${format(amplitudes[i])} photons/s`;
    node.append(freq, amp);
    return node;
  });
  $("peaks").replaceChildren(...nodes);
  if (!nodes.length)
    $("peaks").textContent = "No non-DC probes exceeded the threshold.";
}

function drawMedia() {
  const media = result.view.media;
  if (!media) return;
  const height = media.image.length,
    width = media.image[0].length;
  [
    ["source-image", media.image, 1, false],
    ["count-image", media.counts, Math.max(1, media.count_max), true],
  ].forEach(([id, data, maximum, counts]) => {
    const canvas = $(id);
    canvas.width = width * 8;
    canvas.height = height * 8;
    canvas.style.aspectRatio = `${width}/${height}`;
    const ctx = canvas.getContext("2d");
    for (let row = 0; row < height; row++)
      for (let col = 0; col < width; col++) {
        const value = Math.min(1, data[row][col] / maximum);
        const color = counts
          ? [
              Math.round(21 + 215 * value),
              Math.round(38 + 191 * value),
              Math.round(34 + 120 * value),
            ]
          : [0, 0, 0].map(() => Math.round(255 * Math.pow(value, 1 / 2.2)));
        ctx.fillStyle = `rgb(${color.join(",")})`;
        ctx.fillRect(col * 8, row * 8, 8, 8);
      }
    ctx.strokeStyle = counts ? colors.violet : colors.amber;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(
      selectedPixel[1] * 8 + 0.7,
      selectedPixel[0] * 8 + 0.7,
      6.6,
      6.6,
    );
  });
  $("count-caption").textContent =
    `Detected photons / frame · fixed scale 0–${media.count_max}`;
}

function makeDemo(side = 32) {
  return Array.from({ length: 24 }, (_, frame) =>
    Array.from({ length: side }, (_, row) =>
      Array.from({ length: side }, (_, col) => {
        const x = col / side,
          y = row / side,
          cx = 0.25 + (0.5 * frame) / 23;
        const moving =
          0.85 * Math.exp(-((x - cx) ** 2 + (y - 0.48) ** 2) / 0.012);
        const bar = y > 0.76 && y < 0.83 && x > 0.15 && x < 0.85 ? 0.25 : 0;
        return Math.min(1, 0.015 + moving + bar);
      }),
    ),
  );
}

function pixelsFromElement(element, width, height) {
  const side = Number($("media-size").value),
    scale = Math.min(1, side / Math.max(width, height));
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(width * scale));
  canvas.height = Math.max(1, Math.round(height * scale));
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(element, 0, 0, canvas.width, canvas.height);
  const rgba = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  const linear = $("linearize").checked;
  const channel = (v) => {
    v /= 255;
    return linear
      ? v <= 0.04045
        ? v / 12.92
        : ((v + 0.055) / 1.055) ** 2.4
      : v;
  };
  return Array.from({ length: canvas.height }, (_, row) =>
    Array.from({ length: canvas.width }, (_, col) => {
      const i = 4 * (row * canvas.width + col);
      return (
        ((0.2126 * channel(rgba[i]) +
          0.7152 * channel(rgba[i + 1]) +
          0.0722 * channel(rgba[i + 2])) *
          rgba[i + 3]) /
        255
      );
    }),
  );
}

function mediaEvent(element, name, action) {
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(
      () =>
        finish(
          new Error("Media decoding timed out. Try a shorter supported video."),
        ),
      10000,
    );
    const done = () => finish(),
      fail = () =>
        finish(new Error("This browser could not decode the media file."));
    function finish(error) {
      clearTimeout(timeout);
      element.removeEventListener(name, done);
      element.removeEventListener("error", fail);
      error ? reject(error) : resolve();
    }
    element.addEventListener(name, done, { once: true });
    element.addEventListener("error", fail, { once: true });
    try {
      action();
    } catch (error) {
      finish(error);
    }
  });
}

async function loadMedia(file) {
  const version = ++mediaVersion;
  mediaLoading = true;
  requestVersion++;
  clearTimeout(timer);
  stopPlayback();
  $("status").textContent = "Decoding and sampling media locally…";
  $("export").disabled = true;
  const url = URL.createObjectURL(file);
  try {
    let frames = [];
    if (file.type.startsWith("image/")) {
      const image = new Image();
      await mediaEvent(image, "load", () => {
        image.src = url;
      });
      frames = [
        pixelsFromElement(image, image.naturalWidth, image.naturalHeight),
      ];
    } else if (file.type.startsWith("video/")) {
      const video = document.createElement("video");
      video.muted = true;
      video.preload = "auto";
      await mediaEvent(video, "loadeddata", () => {
        video.src = url;
        video.load();
      });
      if (!Number.isFinite(video.duration) || video.duration <= 0)
        throw new Error("Video duration is unavailable.");
      const fps = Number($("fps").value);
      if (!(fps > 0)) throw new Error("Set a positive frame rate.");
      const count = Math.min(
        32,
        Math.max(1, Math.ceil(Math.min(1, video.duration) * fps)),
      );
      for (let frame = 0; frame < count; frame++) {
        if (version !== mediaVersion) return;
        if (frame > 0)
          await mediaEvent(video, "seeked", () => {
            video.currentTime = frame / fps;
          });
        frames.push(
          pixelsFromElement(video, video.videoWidth, video.videoHeight),
        );
      }
      video.removeAttribute("src");
      video.load();
    } else throw new Error("Choose a browser-supported image or video.");
    if (version !== mediaVersion) return;
    mediaFrames = frames;
    mediaFile = file;
    selectedPixel = [
      Math.floor(frames[0].length / 2),
      Math.floor(frames[0][0].length / 2),
    ];
    selectedFrame = 0;
    $("media-info").textContent =
      `${file.name} · ${frames.length} frame(s) · ${frames[0][0].length} × ${frames[0].length}. Resized intensities are treated as held source flux.`;
    $("error").hidden = true;
    pendingWindow = frames.length / Number($("fps").value);
  } catch (error) {
    if (version === mediaVersion) showError(error);
    return;
  } finally {
    URL.revokeObjectURL(url);
    if (version === mediaVersion) mediaLoading = false;
  }
  markDirty();
}

function stopPlayback() {
  if (playback) {
    clearInterval(playback);
    playback = null;
  }
  $("play").textContent = "Play";
}

function usePreset(name) {
  stopPlayback();
  mediaVersion++;
  mediaLoading = false;
  const preset = presets[name];
  setControls({ ...base, ...preset.config });
  selectedPixel = [0, 0];
  selectedFrame = 0;
  pendingWindow = preset.window;
  if (name === "media") {
    const side = Number($("media-size").value);
    mediaFrames = makeDemo(side);
    mediaFile = null;
    selectedPixel = [Math.floor(side / 2), Math.floor(side / 2)];
    $("media-file").value = "";
    $("media-info").textContent =
      `Built-in moving target · 24 frames · ${side} × ${side}`;
  }
  $("interpret-title").textContent = preset.title;
  $("interpret-body").textContent = preset.body;
  document
    .querySelectorAll("[data-preset]")
    .forEach((button) =>
      button.classList.toggle("active", button.dataset.preset === name),
    );
  simulate();
}

$("settings").addEventListener("submit", (event) => {
  event.preventDefault();
  simulate();
});
$("settings").addEventListener("input", (event) => {
  if (["qe", "modulation"].includes(event.target.id)) syncFields();
});
$("settings").addEventListener("change", (event) => {
  if (
    ["media-file", "csv-file", "media-size", "linearize", "auto-run"].includes(
      event.target.id,
    )
  )
    return;
  if (event.target.id === "kind") {
    mediaVersion++;
    mediaLoading = false;
    selectedPixel = [0, 0];
    selectedFrame = 0;
    pendingWindow = null;
  }
  if (event.target.id === "fps" && mediaFile?.type.startsWith("video/")) {
    loadMedia(mediaFile);
    return;
  }
  syncFields();
  markDirty();
});
$("auto-run").addEventListener("change", () => {
  if ($("auto-run").checked) markDirty(false);
  else clearTimeout(timer);
});
$("new-seed").addEventListener("click", () => {
  $("seed").value = (Number($("seed").value) + 1) >>> 0;
  simulate();
});
$("reset").addEventListener("click", () => usePreset("sparse"));
document
  .querySelectorAll("[data-preset]")
  .forEach((button) =>
    button.addEventListener("click", () => usePreset(button.dataset.preset)),
  );
$("export").addEventListener("click", () => {
  if (result) location.href = `/api/export?id=${result.id}`;
});
$("media-file").addEventListener("change", (event) => {
  if (event.target.files[0]) loadMedia(event.target.files[0]);
});
for (const id of ["media-size", "linearize"])
  $(id).addEventListener("change", () => {
    if (mediaFile) loadMedia(mediaFile);
    else {
      mediaFrames = makeDemo(Number($("media-size").value));
      selectedPixel = [0, 0];
      markDirty();
    }
  });
$("demo-media").addEventListener("click", () => usePreset("media"));
$("csv-file").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  if (file.size > 1e6) {
    showError(new Error("Use a CSV smaller than 1 MB."));
    return;
  }
  $("rates").value = await file.text();
  markDirty();
});
$("reconstruction").addEventListener("change", () => {
  if (result) drawFlux();
});
$("log-frequency").addEventListener("change", () => {
  if (result) drawSpectrum();
});
$("zoom-reset").addEventListener("click", () => {
  if (result) {
    $("view-start").value = 0;
    $("view-span").value = result.summary.duration;
    changeView();
  }
});
$("zoom-period").addEventListener("click", () => {
  if (result) {
    $("view-start").value = 0;
    $("view-span").value = Math.min(
      result.summary.duration,
      1 / result.config.frequency,
    );
    changeView();
  }
});
for (const id of ["view-start", "view-span", "bins"])
  $(id).addEventListener("change", () => changeView());
$("pan").addEventListener("input", () => {
  if (!result) return;
  $("view-start").value =
    (Number($("pan").value) / 1000) *
    Math.max(0, result.summary.duration - Number($("view-span").value));
});
$("pan").addEventListener("change", () => changeView());
$("frame").addEventListener("input", () => {
  selectedFrame = Number($("frame").value);
  changeView();
});
$("play").addEventListener("click", () => {
  if (playback) {
    stopPlayback();
    return;
  }
  if (!result?.view.media) return;
  $("play").textContent = "Pause";
  let loading = false;
  playback = setInterval(async () => {
    if (loading) return;
    loading = true;
    selectedFrame = (selectedFrame + 1) % result.view.media.total_frames;
    await changeView();
    loading = false;
  }, 250);
});
for (const id of ["source-image", "count-image"])
  $(id).addEventListener("click", (event) => {
    if (!result?.view.media) return;
    const rect = $(id).getBoundingClientRect(),
      image = result.view.media.image;
    selectedPixel = [
      Math.min(
        image.length - 1,
        Math.floor(((event.clientY - rect.top) / rect.height) * image.length),
      ),
      Math.min(
        image[0].length - 1,
        Math.floor(
          ((event.clientX - rect.left) / rect.width) * image[0].length,
        ),
      ),
    ];
    changeView({ spectrum: true });
  });

for (const id of ["flux", "histogram", "spectrum"]) {
  $(id).addEventListener("pointermove", (event) => {
    const plot = charts.get(id);
    if (!plot?.xs?.length) return;
    const rect = $(id).getBoundingClientRect(),
      value = plot.unx(event.clientX - rect.left);
    if (value < plot.x0 || value > plot.x1) {
      $("tooltip").hidden = true;
      return;
    }
    let lo = 0,
      hi = plot.xs.length - 1;
    while (lo < hi) {
      const mid = Math.floor((lo + hi) / 2);
      if (plot.xs[mid] < value) lo = mid + 1;
      else hi = mid;
    }
    $("tooltip").textContent =
      `${format(plot.xs[lo], id === "spectrum" ? "Hz" : "s")} · ${format(plot.ys[lo])} ${plot.unit}`;
    $("tooltip").style.left =
      `${Math.min(window.innerWidth - 235, event.clientX + 12)}px`;
    $("tooltip").style.top = `${event.clientY - 38}px`;
    $("tooltip").hidden = false;
  });
  $(id).addEventListener("pointerleave", () => {
    $("tooltip").hidden = true;
  });
}
let dragStart = null;
$("flux").addEventListener("pointerdown", (event) => {
  const plot = charts.get("flux");
  if (!plot) return;
  dragStart = Math.max(plot.x0, Math.min(plot.x1, plot.unx(event.offsetX)));
  $("flux").setPointerCapture(event.pointerId);
});
$("flux").addEventListener("pointerup", (event) => {
  if (dragStart === null) return;
  const plot = charts.get("flux"),
    end = Math.max(plot.x0, Math.min(plot.x1, plot.unx(event.offsetX)));
  if (Math.abs(end - dragStart) > (plot.x1 - plot.x0) * 0.02) {
    $("view-start").value = Math.min(dragStart, end);
    $("view-span").value = Math.abs(end - dragStart);
    changeView();
  }
  dragStart = null;
});
$("flux").addEventListener("pointercancel", () => {
  dragStart = null;
});
$("flux").addEventListener("dblclick", () => $("zoom-reset").click());
document.querySelectorAll("[data-png]").forEach((button) =>
  button.addEventListener("click", () => {
    if (!result) return;
    const id = button.dataset.png,
      original = $(id),
      canvas = document.createElement("canvas");
    canvas.width = original.width;
    canvas.height = original.height + 95;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = plotStyle.background;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = plotStyle.foreground;
    ctx.font = "18px sans-serif";
    ctx.fillText(`Photon simulator · ${id}`, 25, 29);
    ctx.font = "12px sans-serif";
    ctx.fillText(
      `Seed ${result.config.seed} · exposure ${format(result.summary.duration, "s")} · ${result.summary.detected} recorded events`,
      25,
      52,
    );
    ctx.fillText(
      id === "flux"
        ? "Teal: reference · amber: source · violet: uncorrected Fourier estimate"
        : id === "spectrum"
          ? "Violet: |coefficient| · red: approximate CFAR"
          : "Blue: counts/bin width · amber: expected rate before dead time",
      25,
      74,
    );
    ctx.drawImage(original, 0, 95);
    const link = document.createElement("a");
    link.download = `photons-${id}.png`;
    link.href = canvas.toDataURL();
    link.click();
  }),
);
new ResizeObserver(() => {
  if (result) renderAll();
}).observe(document.querySelector("main"));
usePreset("sparse");
