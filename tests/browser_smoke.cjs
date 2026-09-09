// Run with a local server and Playwright installed: node tests/browser_smoke.cjs
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || "playwright");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

(async () => {
  const output = process.env.PHOTON_TEST_OUTPUT || "/tmp/photon-browser-test";
  fs.mkdirSync(output, { recursive: true });
  const browser = await chromium.launch({ channel: "chrome", headless: true });
  try {
    const page = await browser.newPage({
      viewport: { width: 1440, height: 1060 },
      deviceScaleFactor: 1,
    });
    const errors = [];
    page.on("pageerror", (error) => errors.push(error.message));
    const ready = () =>
      page.waitForFunction(
        () =>
          document.querySelector("#status").textContent.startsWith("Run ready"),
        null,
        { timeout: 30000 },
      );
    const runAfter = async (action) => {
      const response = page.waitForResponse((r) =>
        r.url().endsWith("/api/run"),
      );
      await action();
      const res = await response;
      assert.equal(res.status(), 200, await res.text());
      await ready();
    };
    const viewAfter = async (action) => {
      const response = page.waitForResponse((r) =>
        r.url().endsWith("/api/view"),
      );
      await action();
      assert.equal((await response).status(), 200);
    };
    await page.goto(process.env.PHOTON_TEST_URL || "http://127.0.0.1:8765");
    await ready();
    const appearance = await page.evaluate(() => {
      const style = getComputedStyle(document.documentElement);
      const pixel = document
        .querySelector("#flux")
        .getContext("2d")
        .getImageData(0, 0, 1, 1).data;
      const canvasColor =
        "#" +
        [...pixel]
          .slice(0, 3)
          .map((v) => v.toString(16).padStart(2, "0"))
          .join("");
      return {
        scheme: style.colorScheme,
        canvasColor,
        plotColor: style.getPropertyValue("--plot-bg").trim(),
      };
    });
    assert.equal(appearance.scheme, "dark");
    assert.equal(
      appearance.canvasColor,
      appearance.plotColor,
      "canvas follows the CSS theme",
    );
    await page.screenshot({
      path: path.join(output, "desktop.png"),
      fullPage: true,
    });
    assert.equal(await page.locator("#incident").textContent(), "606");
    // Viewing bins and zoom must not regenerate events.
    const incident = await page.locator("#incident").textContent();
    await page.locator("#view-span").fill("0.0002");
    await viewAfter(() => page.locator("#view-span").dispatchEvent("change"));
    assert.equal(await page.locator("#incident").textContent(), incident);
    await viewAfter(() => page.locator("#zoom-reset").click());
    for (const preset of ["alias", "dead", "pulse", "media"]) {
      await runAfter(() => page.locator(`[data-preset="${preset}"]`).click());
      await page.screenshot({
        path: path.join(output, `${preset}.png`),
        fullPage: true,
      });
    }
    await viewAfter(() =>
      page.locator("#source-image").click({ position: { x: 120, y: 100 } }),
    );
    await page.waitForFunction(
      () =>
        document.querySelector("#pixel-label").textContent !== "Pixel (0, 0)",
    );
    await page.locator("#play").click();
    await page.waitForFunction(
      () => document.querySelector("#frame").value !== "0",
    );
    await page.locator("#play").click();
    const download = page.waitForEvent("download");
    await page.locator("#export").click();
    await (await download).saveAs(path.join(output, "export.npz"));
    const pngDownload = page.waitForEvent("download");
    await page.locator('[data-png="spectrum"]').click();
    await (await pngDownload).saveAs(path.join(output, "spectrum-export.png"));
    // A frame update must not discard the spectrum of a pending pixel change.
    const overlapping = await page.evaluate(async () => {
      selectedPixel = [10, 12];
      const pixelRequest = changeView({ spectrum: true });
      selectedFrame = 2;
      const frameRequest = changeView();
      await Promise.all([pixelRequest, frameRequest]);
      return {
        id: result.id,
        pixel: result.view.pixel,
        amplitudes: result.spectrum.amplitudes,
      };
    });
    assert.deepEqual(overlapping.pixel, [10, 12]);
    const expectedPixel = await page.request.post(
      new URL("/api/view", page.url()).href,
      {
        data: {
          id: overlapping.id,
          row: 10,
          col: 12,
          spectrum: true,
        },
      },
    );
    assert.deepEqual(
      overlapping.amplitudes,
      (await expectedPixel.json()).spectrum.amplitudes,
    );
    // Readable error with the previous successful plots retained.
    await page.locator("#auto-run").uncheck();
    await page.locator("#mean_rate").fill("10000000");
    const failed = page.waitForResponse((r) => r.url().endsWith("/api/run"));
    await page.locator("#run").click();
    assert.equal((await failed).status(), 400);
    await page.locator("#error").waitFor({ state: "visible" });
    assert.match(await page.locator("#error").textContent(), /250,000/);
    await runAfter(() => page.locator('[data-preset="sparse"]').click());
    // An actual CSV upload, including a single rate.
    await page.locator("#kind").selectOption("sampled");
    await page.locator("#rates").fill("");
    await page.locator("#run").click();
    await page.locator("#error").waitFor({ state: "visible" });
    assert.match(await page.locator("#error").textContent(), /at least one/);
    await page.locator("#csv-file").setInputFiles({
      name: "rates.csv",
      mimeType: "text/csv",
      buffer: Buffer.from("5000"),
    });
    await runAfter(() => page.locator("#run").click());
    // Browser image decoding path.
    const png = await page.evaluate(() => {
      const c = document.createElement("canvas");
      c.width = 12;
      c.height = 8;
      const ctx = c.getContext("2d");
      ctx.fillStyle = "#a0c080";
      ctx.fillRect(0, 0, 12, 8);
      return c.toDataURL().split(",")[1];
    });
    await page.locator("#kind").selectOption("media");
    await page.locator("#mean_rate").fill("120");
    await page.locator("#media-file").setInputFiles({
      name: "image.png",
      mimeType: "image/png",
      buffer: Buffer.from(png, "base64"),
    });
    await page.waitForFunction(() =>
      document.querySelector("#media-info").textContent.startsWith("image.png"),
    );
    await runAfter(() => page.locator("#run").click());
    assert.match(await page.locator("#media-info").textContent(), /12 × 8/);
    if (process.env.PHOTON_TEST_VIDEO) {
      await page
        .locator("#media-file")
        .setInputFiles(process.env.PHOTON_TEST_VIDEO);
      await page.waitForFunction(() =>
        document.querySelector("#media-info").textContent.includes(".mp4"),
      );
      await runAfter(() => page.locator("#run").click());
      assert.match(
        await page.locator("#media-info").textContent(),
        /24 frame\(s\)/,
      );
      console.log("Actual MP4 upload and decoding passed.");
    }
    await runAfter(() => page.locator('[data-preset="sparse"]').click());
    await page.setViewportSize({ width: 390, height: 844 });
    await page.screenshot({
      path: path.join(output, "mobile.png"),
      fullPage: true,
    });
    await page.locator(".mobile-setup").first().click();
    await page.screenshot({ path: path.join(output, "mobile-controls.png") });
    assert.ok(await page.locator("#kind").isVisible());
    assert.ok(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth,
      ),
      "mobile overflow",
    );
    assert.deepEqual(errors, [], "browser errors");
    console.log(
      `Browser controls, five presets, zoom, playback, export, uploads, errors, and mobile layout passed. Screenshots: ${output}`,
    );
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
