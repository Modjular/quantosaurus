<img width="100%" alt="quantosaurus_demo1" src="https://github.com/user-attachments/assets/1518774f-786f-4226-bfd7-e5054a0b07df" />

# 🦖 Quantosaurus

An open-source, browser-based pixel-classifier for counting cells in microscopy images. Like [Ilastik](https://github.com/ilastik/ilastik), but without the download or setup. It's GPU-accelerated and completely client side.

> [!TIP]
> **👋 Just here to count cells?** Head to https://quantosaur.us/. No signup, no downloads, no installation. Get started as soon as the page loads! Everything below is for people building, deploying, or extending the tool itself.

## About

Quantosaurus loads a microscopy image (`.tif`, `.png`, `.jpg`) and counts the cells in it — entirely in your browser, on your GPU, with nothing uploaded to a server. It offers three tools spanning an easy-to-advanced range, so you can match the method to the image:

- **Threshold** (easy) — a single brightness cutoff, chosen automatically with Otsu or by hand. Instant, no training.
- **Pixel Classifier** (medium) — paint a few example pixels and a random forest learns to classify the rest, live. For images where a plain threshold fails.
- **Cellpose** (advanced) — the pretrained cyto3 deep-learning model, running in-browser on WebGPU, for crowded or touching cells. No training, no GPU server.

A [landing page](https://quantosaur.us/) helps you pick; each tool is its own page (`classifier.html`, `threshold.html`, `cellpose.html`).

Everything — filter computation, thresholding, random-forest training/inference, the Cellpose network, and connected-component labeling — runs in the browser via WebGPU (with an automatic WebGL2 fallback for the threshold and classifier tools). There is no backend, no server-side processing, and your image data never leaves the machine.

### Features

- Three segmentation methods: Otsu **thresholding**, a trainable **pixel classifier**, and the **Cellpose** cyto3 model
- `.tif` (incl. 16-bit and multichannel), `.png`, and `.jpg` support
- Live preview as you threshold or paint, thanks to GPU acceleration
- Object counts via GPU connected-component labeling (or Cellpose instance labels, which keep touching cells separate)
- Segmentation-mask, probability-map, instance-label, and per-object CSV exports
- Figma-style canvas navigation (pan/zoom)

### Planned

- [ ] 3D support
- [x] ~~Contrast sliders~~
- [x] ~~`.ilp` export~~
- [x] ~~Thresholding & Cellpose tools~~
- [ ] `.ilp` import
- [ ] Bulk inference.

### Design stance
A good tool should do one thing well — here, counting cells — and stay simple, local, and dependency-light. The three tools share one small, framework-free codebase; the Cellpose network is hand-written WGSL with its weights vendored into the repo, not a cloud API or an npm dependency. There is no plan to go beyond the most common 2D bio-imaging formats (see the roadmap), or to add server-side processing, accounts, or telemetry. Suggestions are welcome — keep the [Design Goals](#Design-goals) in mind.

### Architecture

- **No backend, no database.** State lives in memory in the browser tab for the session; nothing is persisted or transmitted except what you explicitly export.
- **Three apps, one shared pipeline.** Each tool is its own HTML page over a shared per-image GPU pipeline. All three write a per-pixel **probability buffer**, and everything downstream — the composite overlay, connected-component labeling, per-object stats, and exports — is method-agnostic. The classifier fills that buffer with a `FlatRandomForest` evaluated over an 8-channel Gaussian-derivative feature bank; thresholding fills it from an Otsu (or manual) cutoff; Cellpose fills it from a hand-written WGSL forward pass of the cyto3 network and uploads instance labels directly (bypassing connected-components, so touching cells stay separate).
- **Two interchangeable GPU backends** (`js/backends/webgpu.js`, `js/backends/webgl2.js`) implement the same interface; WebGPU is preferred, with WebGL2 as a fallback. Cellpose is WebGPU-only (its WGSL network has no WebGL2 port); thresholding and the classifier run on either.
- **Shared chrome, method-specific UI.** `js/app.js` and `js/chrome.js` provide the bootstrap and app-agnostic UI (theme, feedback, cheatsheet, file ingest, nav) every page reuses; each page adds only its own controls.
- **Intensities stay in native units.** Loaded pixel values keep the source image's raw range (e.g. 0–65535 for 16-bit TIFFs) rather than being normalized to 0–1, so contrast windowing happens on the GPU from real values.

For the full internals — data flow, the GPU backend contract, random-forest buffer layout, stats struct layout — see [`CLAUDE.md`](./CLAUDE.md); it's written as engineering documentation, not just AI-assistant config.

### Design goals

- **Accessible** — no complicated UI, hidden affordances, or hotkeys to memorize for users; a flat, readable file structure for developers, with no `npm` or install step so you can `git clone` and start hacking immediately.
- **Local** — no active internet connection required to use it; no `node_modules` to dig through to build it. If a third-party library is genuinely necessary, it gets vendored into the repo rather than pulled in as a dependency.
- **Simple** — does one thing well; readability was preferred over cleverness, and DIY was explored before reaching for a library.

## Developer Setup

There's no build step, no bundler, and no package manager — the code is plain ES modules served directly to the browser.

```sh
git clone https://github.com/Modjular/quantosaurus.git
cd quantosaurus
python3 -m http.server 8000   # or any static file server
```

Then open `http://localhost:8000` in a browser. ES modules require serving over `http(s)://`, not `file://`. Edits to `js/*.js`, `index.html`, or `style.css` take effect on reload — no rebuild step.

A WebGPU-capable browser (recent Chrome/Edge) gets the full pipeline; other browsers fall back to WebGL2 automatically.

### Tests

Core CPU logic (random forest, filter math, connected-component reference implementation, config invariants) has dependency-free unit tests colocated with each module (`*.test.mjs`), run with plain Node — no framework, no install step:

```sh
node js/rf.test.mjs               # FlatRandomForest: training, flat-buffer layout, Gini/purity
node js/io.test.mjs               # intensityToRGBA normalization
node js/backends/webgl2.test.mjs  # CCL + stats reference implementations
node js/config.test.mjs           # NUM_CLASSES / DEFAULT_LABEL_COLORS invariants
node js/settings.test.mjs         # sanitizeSettings validation
node js/training.test.mjs         # decodeObjects (stats -> centroids)
node js/ilp.test.mjs              # buildIlpProject HDF5 output
node js/objects.test.mjs          # object-stats decode + CSV
node js/threshold.test.mjs        # Otsu threshold
```

The Cellpose WGSL network is validated separately, against a PyTorch reference, in its upstream repo (see `js/vendor/cellpose/`), not by these Node tests.

GPU-dependent code (actual draw/dispatch calls in the WebGPU/WebGL2 backends) isn't unit-testable this way and needs manual verification in a browser. There is no CI configured yet.

### Configuration

Quantosaurus has no environment variables, API keys, or external services to configure — it's a static, self-contained app. Tunable constants (forest size, feature count, training debounce, camera zoom bounds, stats buffer layout) live in [`js/config.js`](./js/config.js) and are read by both the JS and the WGSL/GLSL shaders; change values there rather than inlining them elsewhere.

## Contributing

Bug reports and pull requests are welcome. Before opening a PR:

1. Run the test suite above (`js/rf.test.mjs`, `js/io.test.mjs`, `js/backends/webgl2.test.mjs`, `js/config.test.mjs`) and confirm everything passes.
2. Manually exercise the change in a browser — GPU code paths in particular can't be caught by the unit tests.
3. Keep new app-logic code dependency-free and in the existing style: functions take `state` explicitly rather than closing over globals, and exported functions carry JSDoc comments. See [`CLAUDE.md`](./CLAUDE.md) for the full conventions.

If a third-party library is truly necessary, vendor it into `js/vendor/` rather than introducing `npm`/a build step — see how `itk-wasm-image-io` and `jszip` are handled there.

## License

[MIT](./LICENSE)

## Acknowledgments

Thanks to Ilastik for inspiration.
Thanks to ITK-Wasm for file handling.
