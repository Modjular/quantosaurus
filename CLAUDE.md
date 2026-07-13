# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Quantosaurus is a browser-based pixel-classification tool (inspired by Ilastik) for counting cells in
microscopy images. Everything — filter computation, random-forest training/inference, and connected-component
labeling — runs client-side on the GPU (WebGPU, with a WebGL2 fallback). There is no backend and no build step.

## Development

No build step, no bundler, no package manager. Just serve the repo root over localhost and open it
(ES modules require `http(s)://`, not `file://`):

```sh
python3 -m http.server 8000   # or any static file server
```

Open `index.html` via that server. Edits to `js/*.js`/`index.html`/`style.css` take effect on reload.

### Tests

Core CPU logic has dependency-free unit tests colocated with each module (`*.test.mjs`), run with plain
Node — no framework, no install step, no test runner config:

```sh
node js/rf.test.mjs               # FlatRandomForest: training, flat-buffer layout, Gini/purity
node js/io.test.mjs               # intensityToRGBA normalization
node js/backends/webgl2.test.mjs  # CCL + stats reference implementations
node js/config.test.mjs           # NUM_CLASSES / DEFAULT_LABEL_COLORS invariants
```

Each test file is self-contained: an `assert(cond, msg)` helper logs `ok`/`FAIL` lines and the script exits
non-zero on any failure. When adding CPU-only logic (pure functions, no `document`/WebGL/WebGPU calls),
add a matching `*.test.mjs` next to it rather than introducing a test framework. GPU-dependent code (actual
`WebGpuBackend`/`WebGl2Backend` draw/dispatch calls) is not unit-testable this way — `webgl2.test.mjs` only
exercises `cclLabel`/`accumulateStats`, the pure reference implementations it can import safely under Node.

There is no CI configured — verify by running the three commands above and by manually exercising the app
in a browser (see the `run`/`verify` skills for driving a browser check).

## Architecture

### Data flow

`Quantosaurus` (`js/quantosaurus.js`) is the app core: a class (extending `EventTarget`) that owns the single
shared `state` object (images array, the `FlatRandomForest` instance, current tool/class, brush size, camera
transform) and attaches to a board element. It's the deliberate class exception alongside the GPU backends and
`FlatRandomForest`; every other module is still a set of functions that take `state` (or a specific image's
entry in `state.images`) as an explicit argument. The class is a thin wrapper: its methods call those module
functions and then dispatch a `CustomEvent` for the change. All UI is a *subscriber* — nothing reaches into
the app core's internals. `index.html`'s inline module constructs one instance, wires the taskbar controls to
its methods, and subscribes to its events; `js/ui.js:bindChrome(q)` subscribes the count badges / save
indicator / training cursor the same way. This is what makes the board embeddable (e.g. in a notebook): a host
page can `new Quantosaurus(div)` and `addEventListener` for results without any of the built-in chrome.

**Event catalog** (all `CustomEvent`s on the instance; see the header of `quantosaurus.js` for payload shapes):
`imageloadstart`/`imageadded`/`imageloaderror`/`imageremoved`/`imagereordered`, `labelschanged`, `dirtychange`,
`toolchanged`/`brushsizechanged`/`sigmachanged`/`classchanged`, `classcolorchanged`/`classnamechanged`/
`markersvisibilitychanged`, `trainingstart`/`trainingcomplete`, and `statscomputed` (carries per-class counts +
per-image detected objects). Modules dispatch through `state.events?.dispatchEvent(...)`, optional-chained so
the CPU unit tests can import `images.js`/`training.js` under Node without a dispatch target.

Per-image state (`state.images[i]`) carries: the loaded `intensityArray` (raw pixel values, not normalized),
its GPU `backend` instance, `labels` (sparse `{x, y, cls}` painted by the user), the display `windowLo`/`windowHi`
contrast bounds, and the DOM nodes for its canvas tile (the tile carries its own hover controls — reorder /
contrast / delete; there is no separate sidebar).

The core loop: user paints labels (`images.js:paint`) → `training.js:scheduleTraining` debounces
(`TRAIN_DEBOUNCE_MS`) → `trainAndPredictAll` gathers per-pixel features for every labeled pixel across all
images (`backend.gatherFeaturesForTraining`), trains one `FlatRandomForest` shared across all images, then
reruns `backend.runInference` per image, then recomputes per-class object counts via connected-component
labeling + stats, emitted as `statscomputed` for the count chip.

### GPU backend interface

`js/backends/webgpu.js` (`WebGpuBackend`) and `js/backends/webgl2.js` (`WebGl2Backend`) are interchangeable
implementations of the same interface, selected at image-load time in `images.js:initializeBackend` (WebGPU
preferred, WebGL2 fallback, alert+throw if neither is available). Never assume a specific backend elsewhere;
new callers should only depend on this shared surface:

```
initialize, allocateImage, updateFeatures, downloadFeatures, gatherFeaturesForTraining,
runInference, renderComposite, downloadProbabilities, computeConnectedComponents,
computeStats, downloadStats, downloadLabels, setWindow, setColors, destroy
```

Each backend, per image, runs entirely on-GPU: a separable Gaussian-derivative filter bank produces
`NUM_FEATURES` (8) per-pixel features → the trained `FlatRandomForest` is uploaded and evaluated as a compute
pass to get per-class probabilities → connected-component labeling (atomic parallel union-find) + a stats
accumulation pass produce per-object metrics → a composite pass renders the argmax overlay with the current
contrast window. Data only crosses to the CPU through the `download*`/`gatherFeaturesForTraining` methods.

WebGPU compute pipelines are cached per-image in `_pipelineCache` keyed by a stable pass id, since only
image-lifetime constants (width/height/label count/scale) are baked into the WGSL; anything that varies
per-call (e.g. label count for stats) goes through a uniform buffer instead so passes compile once and are
reused across retrains rather than rebuilt on every call. Preserve this when touching either backend — don't
reintroduce per-call shader recompilation on hot paths like dragging the contrast slider or picking a class
color (`setColors`, like `setWindow`, only writes a uniform and repaints — no recompile).

`NUM_CLASSES` (`js/config.js`) is fixed at 4, not just as a default. WebGL2 packs every class's probability
into a single RGBA32F texture, one channel per class, and its shaders unpack it as a hardcoded 4-element
array (`webgl2.js`'s composite/RF-inference passes). WebGPU's probability buffer has no such limit, but
raising the class count past 4 would need a WebGL2 multi-render-target refactor (multiple probability
textures + a gather step) — don't bump `NUM_CLASSES` without doing that first.

### `FlatRandomForest` (`js/rf.js`)

A CPU-trained random forest whose nodes are packed into a flat buffer (`Int32Array`/`Float32Array` twin views
over one `ArrayBuffer`) so it can be uploaded directly as a GPU storage buffer and walked by the inference
shaders in both backends using the identical per-node layout:

```
slot 0  feature_index (i32)  — feature to test; -1 marks a leaf
slot 1  threshold     (f32)
slot 2  left_child    (i32)
slot 3  right_child   (i32) — OR, on a leaf, -(classId + 1)
```

`treeRoots[t]` gives each tree's starting node index in the shared buffer. If you change this layout, update
the WGSL/GLSL inference shaders in both `webgpu.js` and `webgl2.js` and the CPU reference walker
(`predictSingle`) in lockstep — they must stay bit-for-bit compatible. `rf.test.mjs` exercises this contract.

### Stats layout (`config.js: STATS_LAYOUT`)

Per-object stats (area, summed intensity, summed x/y for centroids, min/max intensity) are produced by both
backends' stats-accumulation pass and consumed by `training.js` (object counts) and `export.js` (label
metadata export, currently disabled — see TODO below). Summed fields are 64-bit, split into two u32 words
(`lo`/`hi`) since WGSL has no `atomic<u64>`; reassemble as `hi * 2**32 + lo`. `STATS_LAYOUT` is the single
source of truth for the struct's field count/order — keep it in sync with the WGSL/GLSL structs in both
backends if you change it.

### Image loading (`js/io.js`) and intensity units

Images are loaded via the vendored `itk-wasm-image-io` (TIFF) or `createImageBitmap` (PNG/JPG, converted to
luma). Loaded intensities are **raw pixel values in the source's native range** (e.g. 0–65535 for a uint16
TIFF), not normalized to 0–1 — features, stats, and the contrast control all operate in real units, and
display windowing (`windowLo`/`windowHi`) happens on the GPU from these raw values. Float-dtype images get a
fixed-point `scale` (`config.js` / `range.scale`) applied before the integer stats accumulator so fractional
intensities survive; descale by dividing by `range.scale` when reading stats back out (see `export.js`).
Only 2D single-channel images are supported; higher-dimensional loads are rejected in `images.js:addImage`.

### Export (`js/export.js`)

Builds ITK images (via the same vendored `itk-wasm-image-io`) for segmentation masks and/or per-class
probability maps, then zips them off the main thread in an inline `Worker` (constructed from a `Blob` URL,
importing the vendored `jszip.min.js`) so the UI stays responsive during large exports and progress can be
reported. Label/centroid stats export is present but gated behind `exportLabels = false` pending an upstream
itk-wasm bug (see the TODO/link in that file) — don't re-enable it without checking that issue.

### Camera (`js/camera.js`)

Figma-style pan/zoom on `#canvas-board` via a CSS transform driven by `state.camera {x, y, scale}`: Ctrl/Cmd+wheel
zooms toward the cursor, plain wheel pans, and Space-held (or the grab tool) + drag pans. This is purely a CSS
transform on the board container — it does not touch any per-image canvas or backend state.

### Vendored code (`js/vendor/`)

`itk-wasm-image-io.min.js` (+ its `tiff-read-image`/`tiff-write-image` WASM pipelines) and `jszip.min.js` are
vendored third-party builds, not app code — don't hand-edit them. Both `io.js` and `export.js` point
`setPipelinesBaseUrl` at the vendored pipelines directory instead of the default jsDelivr CDN, and both force
`webWorker: false` on itk-wasm calls because the bundled worker runs from a `data:` URL (opaque origin) that
needs CORS to fetch the vendored WASM even same-origin — running on the main thread sidesteps that.

## Pull Request Template

- Always include a section titled **Summary** that covers all commits in the PR in one concise paragraph — not a
  per-commit bullet list.
- Always include a section titled **Notes** that mentions any bugs of interest fixed along the way (incidental fixes
  outside the PR's main purpose) and any recommended areas of improvement that were noticed but not touched.

## Conventions

- No frameworks, no bundler, no TypeScript — plain ES modules loaded directly by the browser
  (`<script type="module">` in `index.html`). Keep new code dependency-free unless a vendored library already
  covers the need.
- Functions take `state` explicitly rather than closing over module-level globals; keep new app-logic code
  in this style rather than introducing classes/singletons (the GPU backends and `FlatRandomForest` are the
  deliberate exceptions, since they own real GPU/buffer resources).
- Tunable constants (forest size, feature count, debounce timing, camera zoom bounds, stats struct layout)
  live in `js/config.js` — add new ones there rather than inlining magic numbers, especially anything that
  must stay in sync between JS and the WGSL/GLSL shaders.
- JSDoc comments on exported functions are the norm throughout `js/*.js`; match that style for new exports.
