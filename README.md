<img width="100%" alt="quantosaurus_demo1" src="https://github.com/user-attachments/assets/1518774f-786f-4226-bfd7-e5054a0b07df" />

# 🦖 Quantosaurus

A free, open-source, browser-based pixel-classifier for counting cells in microscopy images — like [Ilastik](https://github.com/ilastik/ilastik), but runs entirely client-side on the GPU.

> **Just want to count cells?** Quantosaurus needs no account, install, or server — open `index.html` from a checkout (or a hosted copy, if your lab has one deployed) and load an image. Everything below is for people building, deploying, or extending the tool itself.

## About

Quantosaurus loads a microscopy image (`.tif`, `.png`, `.jpg`), and as you paint a few pixel labels per class, it trains a random forest on GPU-computed filter features and live-previews the classification across the whole image. Once you're happy with the result, GPU connected-component labeling turns the classified pixels into discrete object counts per class.

Everything — filter computation, random-forest training/inference, and connected-component labeling — runs in the browser via WebGPU, with an automatic WebGL2 fallback for browsers without WebGPU support. There is no backend, no server-side processing, and your image data never leaves the machine.

**When to reach for it:** after you've outgrown ImageJ's threshold tool, but before you need something heavier like Ilastik's Object Classifiers or Cellpose.

### Features

- `.tif`, `.png`, and `.jpg` support
- Live preview of classification as you label, thanks to GPU acceleration
- Per-class object counts via GPU connected-component labeling
- Segmentation, probability, and label exports
- Figma-style canvas navigation (pan/zoom)

### Architecture

- **No backend, no database.** State lives in memory in the browser tab for the session; nothing is persisted or transmitted except what you explicitly export.
- **Per-image GPU pipeline:** a separable Gaussian-derivative filter bank computes 8 per-pixel features → a `FlatRandomForest` (trained on the CPU from your labels, uploaded as a GPU buffer) classifies every pixel in a compute pass → connected-component labeling + a stats pass turn classified regions into per-object counts and metrics → a composite pass renders the overlay.
- **Two interchangeable GPU backends** (`js/backends/webgpu.js`, `js/backends/webgl2.js`) implement the same interface; WebGPU is preferred, with WebGL2 as a fallback for browsers that lack it.
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
```

GPU-dependent code (actual draw/dispatch calls in the WebGPU/WebGL2 backends) isn't unit-testable this way and needs manual verification in a browser. There is no CI configured yet.

### Configuration

Quantosaurus has no environment variables, API keys, or external services to configure — it's a static, self-contained app. Tunable constants (forest size, feature count, training debounce, camera zoom bounds, stats buffer layout) live in [`js/config.js`](./js/config.js) and are read by both the JS and the WGSL/GLSL shaders; change values there rather than inlining them elsewhere.

## Contributing

Bug reports and pull requests are welcome. Before opening a PR:

1. Run the test suite above (`js/rf.test.mjs`, `js/io.test.mjs`, `js/backends/webgl2.test.mjs`, `js/config.test.mjs`) and confirm everything passes.
2. Manually exercise the change in a browser — GPU code paths in particular can't be caught by the unit tests.
3. Keep new app-logic code dependency-free and in the existing style: functions take `state` explicitly rather than closing over globals, and exported functions carry JSDoc comments. See [`CLAUDE.md`](./CLAUDE.md) for the full conventions.

If a third-party library is truly necessary, vendor it into `js/vendor/` rather than introducing `npm`/a build step — see how `itk-wasm-image-io` and `jszip` are handled there.

## Roadmap

- [ ] 3D support
- [x] ~~Contrast sliders~~
- [ ] Export trained models as Ilastik-compatible `.ilp` files
- [ ] Multi-file batching

## License

[MIT](./LICENSE)

## Acknowledgments

Built with 🦖 for Dr. Chen.
Thanks to Ilastik for inspiration.
Thanks to ITK-Wasm for file handling.
