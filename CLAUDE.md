# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Quantosaurus is a browser-based tool for counting cells in microscopy images, offering three segmentation
methods across a difficulty range: **Otsu thresholding** (easy), a trainable **pixel classifier** (medium,
inspired by Ilastik), and the pretrained **Cellpose cyto3** deep-learning model (advanced). Everything —
filter computation, thresholding, random-forest training/inference, the Cellpose network, and
connected-component labeling — runs client-side on the GPU (WebGPU, with a WebGL2 fallback for the threshold
and classifier tools; Cellpose is WebGPU-only). There is no backend and no build step.

Each method is its own page over a **shared per-image GPU pipeline** (see the probability-buffer interchange
point below):

| Page | Tool |
|---|---|
| `index.html` | Landing page / tool chooser (SEO surface; the logo links here from every app). |
| `classifier.html` | Pixel classifier (paint labels → random forest). |
| `threshold.html` | Otsu / manual thresholding. |
| `cellpose.html` | Cellpose cyto3 (WebGPU-only). |

## Development

No build step, no bundler, no package manager. Just serve the repo root over localhost and open it
(ES modules require `http(s)://`, not `file://`):

```sh
python3 -m http.server 8000   # or any static file server
```

Open `index.html` (the landing page) via that server, or go straight to a tool page. Edits to
`js/*.js`/`*.html`/`style.css` take effect on reload.

### Tests

Core CPU logic has dependency-free unit tests colocated with each module (`*.test.mjs`), run with plain
Node — no framework, no install step, no test runner config:

```sh
node js/rf.test.mjs               # FlatRandomForest: training, flat-buffer layout, Gini/purity
node js/io.test.mjs               # intensityToRGBA normalization
node js/backends/webgl2.test.mjs  # CCL + stats reference implementations
node js/config.test.mjs           # NUM_CLASSES / DEFAULT_LABEL_COLORS invariants
node js/settings.test.mjs         # sanitizeSettings validation of untrusted localStorage
node js/training.test.mjs         # decodeObjects: dense stats buffer -> centroids/areas
node js/ilp.test.mjs              # buildIlpProject: HDF5 signature, superblock EOF, ASCII markers
node js/objects.test.mjs          # decodeObjectStats + buildObjectCsv (export decode/CSV)
node js/threshold.test.mjs        # computeOtsu (Otsu threshold on a histogram)
```

The vendored Cellpose network (`js/vendor/cellpose/`) is **not** covered by these Node tests — it's validated
upstream against a PyTorch reference (WGSL forward + dynamics, grayscale and 2-channel; see that directory's
provenance). Don't try to reproduce that here.

Each test file is self-contained: an `assert(cond, msg)` helper logs `ok`/`FAIL` lines and the script exits
non-zero on any failure. When adding CPU-only logic (pure functions, no `document`/WebGL/WebGPU calls),
add a matching `*.test.mjs` next to it rather than introducing a test framework. GPU-dependent code (actual
`WebGpuBackend`/`WebGl2Backend` draw/dispatch calls) is not unit-testable this way — `webgl2.test.mjs` only
exercises `cclLabel`/`accumulateStats`, the pure reference implementations it can import safely under Node.

There is no CI configured — verify by running the commands above and by manually exercising the app
in a browser (see the `run`/`verify` skills for driving a browser check).

## Architecture

### Data flow

**Three apps, one shared pipeline.** Each tool page (`classifier.html`, `threshold.html`, `cellpose.html`) has
its own inline `<script type="module">` that owns a `state` object, but the shared bootstrap and chrome live in
two modules every page reuses:

- `js/app.js` — `createBaseState(overrides)` builds the state fields common to all apps (images, camera,
  tool/pointer flags, `labelColors`, `brushSize`); `initApp(state, {page})` wires the shared chrome (camera,
  the `beforeunload` guard) and calls into `js/chrome.js`.
- `js/chrome.js` — the app-agnostic UI wiring: `setupTheme`, `setupFeedback`, `setupCheatsheet`,
  `setupFileIngest` (file input + drag/drop → `addFiles`), `setupNav` (top-bar highlight), plus `downloadBlob`
  / `dateStamp`. Every DOM lookup here is guarded so a page can omit any piece.

Every other module is a set of functions that take `state` (or a specific image's entry in `state.images`) as
an explicit argument — no classes or singletons for app logic, only for the GPU backends, the
`FlatRandomForest`, and the `CellposeWebGPU` instance (owned by the Cellpose worker, see below).

**Behavior hooks decouple `images.js`/`ui.js` from any one method.** `images.js` no longer imports the
classifier's `training.js`; instead it calls optional hooks each app sets on its `state`:

- `state.onLabelsChanged?.(state)` — a finished paint stroke (or deleting a labeled image). Classifier →
  debounced retrain; the other apps don't paint, so they leave it unset.
- `state.onImagesChanged?.(state)` — the image set changed (add/delete). Threshold → re-threshold + recount;
  Cellpose → repopulate channel selectors.
- `state.computeFeatures` — gate on `updateFeatures` (the 8-channel Gaussian bank). Only the classifier sets it;
  Threshold/Cellpose skip that GPU work.
- `state.saveIndicator` — gate on the unsaved-changes title/badge (classifier's `.ilp` flow only).
- `state.splitChannels` — default true (classifier/threshold split a multichannel file into one image per
  channel). Cellpose sets it false: one image per file, all channels kept together (`img.channels`,
  `img.displayChannel`; switch the shown channel via `images.js:setDisplayChannel`).

Per-image state (`state.images[i]`) carries: the loaded `intensityArray` (raw pixel values of the displayed
channel), its GPU `backend` instance, `labels` (sparse `{x, y, cls}`), the display `windowLo`/`windowHi`
contrast bounds, optionally `channels`/`displayChannel` (non-split loads), and the DOM nodes for its tile/row.

**The seam that makes one pipeline serve three methods is the per-pixel probability buffer.** Whatever fills it,
everything downstream — the composite overlay, connected-component labeling, per-object stats, counts,
centroids, and exports — is method-agnostic. The three methods fill it differently:

- Classifier: paint labels → `training.js:scheduleTraining` debounces → `trainAndPredictAll` gathers features
  for every labeled pixel, trains one shared `FlatRandomForest`, reruns `backend.runInference` per image.
- Threshold (`js/threshold.js`): `computeOtsu` (or the manual slider) → `applyThreshold` writes a binary
  foreground map via `backend.setProbabilities`.
- Cellpose (`js/cellpose.js`): `runCellpose` → `segmentImage` (WGSL network, run in the Cellpose worker) →
  `backend.setLabels` uploads the instance labels directly (bypassing CCL, so touching cells stay separate)
  plus a foreground overlay via `setProbabilities`.

Counting then works the same for all three: connected-components (or the uploaded labels) + stats → object
counts. `training.js:updateObjectCounts` does this for the classifier's per-class badges; `threshold.js` and
`cellpose.js` do the single-count analogue via `ui.js:animateCount`.

### GPU backend interface

`js/backends/webgpu.js` (`WebGpuBackend`) and `js/backends/webgl2.js` (`WebGl2Backend`) are interchangeable
implementations of the same interface, selected at image-load time in `images.js:initializeBackend` (WebGPU
preferred, WebGL2 fallback, alert+throw if neither is available). Backends are **per image, not per app**:
every loaded image constructs its own backend, and `WebGpuBackend.initialize` requests its own adapter +
`GPUDevice` (with the adapter's maximum `maxBufferSize`/`maxStorageBufferBindingSize`). So there is one
WebGPU device per image, and GPU resources are never shared between images — anything that must be shared
across images has to either live on the CPU or be uploaded once per image. Never assume a specific backend
elsewhere; new callers should only depend on this shared surface:

```
initialize, allocateImage, updateFeatures, downloadFeatures, gatherFeaturesForTraining,
runInference, renderComposite, downloadProbabilities, computeConnectedComponents,
computeStats, downloadStats, downloadLabels, setProbabilities, setLabels,
setWindow, setColors, destroy
```

`setProbabilities(probs)` and `setLabels(labels)` are the write-side entry points that let a non-classifier
method drive the shared pipeline. `setProbabilities` overwrites the probability buffer directly (Threshold's
binary map; Cellpose's foreground overlay) — same layout as `runInference`/`downloadProbabilities`. `setLabels`
uploads a precomputed instance-label map straight into the label buffer, bypassing connected-component
labeling (Cellpose: `computeStats` then reads those ids directly, preserving each cell rather than renumbering
to union-find roots). For this, the WebGPU `labelBuffer` is allocated with `COPY_DST`; the WebGL2 label store
is a CPU array, so its `setLabels` just replaces `this.labels`. Cellpose is WebGPU-only, so in practice only
WebGPU's `setLabels` runs — WebGL2's exists for interface parity.

Each backend, per image, runs entirely on-GPU: a separable Gaussian-derivative filter bank produces
`NUM_FEATURES` (8) per-pixel features → the trained `FlatRandomForest` is uploaded and evaluated as a compute
pass to get per-class probabilities → connected-component labeling (atomic parallel union-find) + a stats
accumulation pass produce per-object metrics → a composite pass renders the argmax overlay with the current
contrast window. Data only crosses to the CPU through the `download*`/`gatherFeaturesForTraining` methods.

The probability buffer carries a **`-1.0` "no overlay" sentinel**: `allocateImage` seeds every probability to
`-1.0`, and the composite shader tracks the argmax across classes and returns the bare (contrast-windowed)
raw pixel when that maximum is still negative, instead of tinting. That's what makes a freshly-loaded,
never-trained image render clean rather than flooded with class 0's color. Real RF output is a per-class vote
fraction summing to 1, so it never trips the sentinel. Keep this in mind when writing probabilities from any
new code path: `-1.0` means "draw nothing here", `0.0` does **not** — a zero-filled probability buffer renders
as a fully-tinted image. Both backends' composite passes implement this, so preserve it in both.

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
backends' stats-accumulation pass and consumed by `training.js` (object counts), `export.js` (per-object CSV +
label export), and `js/objects.js`. Summed fields are 64-bit, split into two u32 words (`lo`/`hi`) since WGSL
has no `atomic<u64>`; reassemble as `hi * 2**32 + lo`. `STATS_LAYOUT` is the single source of truth for the
struct's field count/order — keep it in sync with the WGSL/GLSL structs in both backends if you change it.
`js/objects.js` (`decodeObjectStats`, `buildObjectCsv`) is the pure, testable decoder over this layout,
consumed by every counting path and the CSV export.

### Image loading (`js/io.js`) and intensity units

Images are loaded via the vendored `itk-wasm-image-io` (TIFF) or `createImageBitmap` (PNG/JPG, converted to
luma). Loaded intensities are **raw pixel values in the source's native range** (e.g. 0–65535 for a uint16
TIFF), not normalized to 0–1 — features, stats, and the contrast control all operate in real units, and
display windowing (`windowLo`/`windowHi`) happens on the GPU from these raw values. Float-dtype images get a
fixed-point `scale` (`config.js` / `range.scale`) applied before the integer stats accumulator so fractional
intensities survive; descale by dividing by `range.scale` when reading stats back out (see `export.js`).

Only **2D** images are supported: `images.js:addImage` rejects anything with `shape.length > 2` (3D volumes /
z-stacks) with an alert, not just a console warning — a silently vanishing drop is worse than a refusal. 2D
**multichannel** images *are* supported. By default each channel becomes its own independent image entry named
`file.tif [chN]` (no channel-aware image type, just N single-channel images, each with its own row/backend/
labels/contrast). Cellpose instead sets `state.splitChannels = false`, keeping a file's channels together in
one entry (`img.channels`) so a cyto+nucleus pair segments as one image. Either way, dedup (`addFiles`) keys on
the *source* file (`sourceName` + size), not the channel-suffixed display name, so re-dropping doesn't re-add.

### Export (`js/export.js`)

`zipImages(images, opts, progressCallback)` takes an **options object** `{seg, prob, labels, csv,
instanceLabels, classNames, instanceClass}` (not positional booleans) so each app requests only the outputs its
method produces (classifier: seg+prob+csv; threshold: seg+csv; cellpose: seg+labels+csv). It builds ITK images
via the vendored `itk-wasm-image-io`, then zips off the main thread in an inline `Worker` (from a `Blob` URL,
importing the vendored `jszip.min.js`) so the UI stays responsive and progress is reported.

- **seg** — uint8 argmax mask over the probability buffer (0 = bg, else class+1).
- **prob** — float32 per-class probability map (classifier only; meaningless for a binary threshold).
- **labels** — uint16 instance-label image, requires `instanceLabels` (the label buffer already holds compact
  `1..N` — Cellpose). uint16, not uint32: itk-wasm's uint32 write is broken
  ([#1544](https://github.com/InsightSoftwareConsortium/ITK-Wasm/issues/1544), uint16 works), so images with
  >65535 objects are skipped with a warning. Not offered for the classifier/threshold, whose CCL labels are
  union-find roots, not `1..N`.
- **csv** — one combined per-object CSV across images (via `js/objects.js`), enumerating objects per class by
  connected-components, or from the uploaded instance labels when `instanceLabels`.

### ilastik project export (`js/ilp.js`)

`buildIlpProject(state, options)` serializes the whole session — one lane per loaded image, each image's raw
pixels, the painted labels (as one bounding-box block per image), class names/colors, and a starter feature
selection — into a `.ilp` ilastik Pixel Classification project, returned as `Uint8Array` file bytes.

The notable part: **it hand-writes HDF5 from first principles, with no HDF5 library.** That's viable because
`.ilp` needs only a small, well-defined subset (contiguous unchunked unfiltered datasets, fixed-length strings,
classic superblock-v0 symbol-table groups) that libhdf5 reads tolerantly. Group naming and the `zyx` axis
convention mirror ilastik's own serializers, so multi-image projects round-trip. Every image's pixels are
embedded via ilastik's "copy into project" representation (`location: "ProjectInternal"` + `inner_path`)
rather than a filesystem reference, since browsers never expose a real path for an uploaded file.

Deliberately **not** included: the trained classifier — Quantosaurus's `FlatRandomForest` is trained on its own
GPU feature bank and isn't interchangeable with ilastik's, so ilastik retrains on open. `ilp.test.mjs` covers
it as a black box (HDF5 signature, superblock EOF vs. actual byte length, known ASCII markers) since there's no
HDF5 parser available to assert against. The module docs at the top of the file are the real reference here.

### Settings persistence (`js/settings.js`)

`loadSettings`/`saveSettings` persist a deliberately narrow slice to `localStorage` (`SETTINGS_KEY`): only
per-class overlay colors and class names — things a user customizes and expects to survive a reload. Anything
image-specific (contrast window, camera, labels), too large (images, features), or useless without its inputs
(the trained forest) is intentionally excluded. Theme is separate (`quantosaurus-theme`), since it's applied to
`<body>` before `state` exists.

`sanitizeSettings(raw, numClasses)` is a pure validator kept separate from the `localStorage` wrappers so it's
testable under Node (`settings.test.mjs`). Treat stored data as untrusted — it may be corrupt, hand-edited, or
stale from a build with a different class count; it clamps both arrays to exactly `numClasses`, nulling invalid
entries so callers fall back per-slot.

### Contrast (`js/contrast.js`)

One shared popover reused for whichever image row is active (napari-style per-layer contrast limits), editing
that image's display-only `windowLo`/`windowHi` via `backend.setWindow`. This is presentation only: it re-runs
just the composite pass — no feature recompute, no retrain, classifier untouched. `oninput` bursts from
dragging a range input are coalesced to one `setWindow` + redraw per frame via `requestAnimationFrame`.

### Camera (`js/camera.js`)

Figma-style pan/zoom on `#canvas-board` via a CSS transform driven by `state.camera {x, y, scale}`: Ctrl/Cmd+wheel
zooms toward the cursor, plain wheel pans, and Space-held (or the grab tool) + drag pans. This is purely a CSS
transform on the board container — it does not touch any per-image canvas or backend state.

### Thresholding (`js/threshold.js`)

`computeOtsu(intensityArray, range)` is a pure 256-bin Otsu threshold in raw intensity units (unit-tested).
`applyThreshold` writes a binary foreground map (`setProbabilities`) — class 0 = `1.0` where a pixel passes,
`-1.0` sentinel elsewhere. The method is global but ranges are per-image: in Auto mode each image gets its own
Otsu; in Manual mode the one normalized `[0,1]` slider value maps through each image's `range` (see
`thresholdForImage`). `runThreshold` applies + recounts across all images. `invert` flips bright/dark
foreground for brightfield.

### Cellpose (`js/cellpose.js` + `js/cellpose.worker.js` + `js/vendor/cellpose/`)

`runCellpose` segments every image with the pretrained cyto3 network, uploads the instance labels via
`backend.setLabels` (bypassing CCL — the whole point is that touching cells stay separate), paints a foreground
overlay, and counts.

**The whole segmentation runs in a dedicated Web Worker (`js/cellpose.worker.js`).** The heavy work — the WGSL
forward pass, the GPU flow dynamics, and especially the *synchronous CPU* mask post-processing
(`cellpose_core.js:_getMasks`/`_maskFlowErrors`/`_filterMasks`) — would otherwise block the main thread and
freeze the tab for seconds, and it scales badly with cell count: a small `diameter` upscales the working
resolution (~3× the GPU work) and finds ~4× the cells. So `js/cellpose.js` is a thin **main-thread client** and
the worker owns the **single `CellposeWebGPU` instance** (its own WebGPU device, weights loaded once). WebGPU,
`fetch`, the Cache API, and `DecompressionStream` are all available in a module worker, so weight loading lives
in the worker too (`ensureCellposeLoaded` posts a `load` message; download progress comes back as `progress`
messages). Only small payloads cross the thread boundary: a grayscale plane in (copied via `.slice()` +
transfer, so the app's own `intensityArray` isn't detached), an instance-label `Int32Array` out (transferred).
A separate device is free anyway since Cellpose's output crosses back to the CPU — nothing GPU-side is shared
with the per-image backends.

Cellpose is **WebGPU-only** (`cellposeSupported()` gates the UI). Note `segmentImage`'s arg order is
`(gray, H, W)` — height before width — and it takes raw intensities (it percentile-normalizes internally),
with the optional nuclear channel as `opts.chan2`. `segmentImage` also caps the working resolution
(`practicalCap`/device-limit clamp) so a too-small diameter can't blow past GPU memory.

### Vendored code (`js/vendor/`)

`itk-wasm-image-io.min.js` (+ its `tiff-read-image`/`tiff-write-image` WASM pipelines) and `jszip.min.js` are
vendored third-party builds, not app code — don't hand-edit them. Both `io.js` and `export.js` point
`setPipelinesBaseUrl` at the vendored pipelines directory instead of the default jsDelivr CDN, and both force
`webWorker: false` on itk-wasm calls because the bundled worker runs from a `data:` URL (opaque origin) that
needs CORS to fetch the vendored WASM even same-origin — running on the main thread sidesteps that.

`js/vendor/cellpose/` holds the Cellpose port: `cellpose_core.js` (the `CellposeWebGPU` class — hand-written
WGSL forward pass + flow dynamics, environment-agnostic) plus `cyto3_weights.bin.gz` (~23.4 MB gzip; ~26 MB raw,
BatchNorm folded — see the ilp/gzip note above) and `cyto3_manifest.json`. **This repo is the source of truth —
there is no external upstream to sync from, so edit `cellpose_core.js` here directly** (it just deliberately
does **not** follow this repo's `state`/JSDoc conventions, and stays environment-agnostic so it can run
headless). It's validated against a PyTorch reference (grayscale + 2-channel cyto/nucleus, AP@0.5 = 1.000 on the
WGSL forward + masks), which is why the Node tests here don't cover it — so if you change the forward/dynamics,
re-validate against that reference. **`js/vendor/cellpose/README.md` documents the port's capabilities and
limits** (WebGPU-only, 2D-only, cyto3-only, the whole-image working-resolution cap, and CPU mask
post-processing). The known ceiling is the **whole-image working-resolution cap** (no tiling): small diameters
lose accuracy and large images can't run at full res. The freeze that a big run used to cause is handled by
**chunking the GPU submits** — `forwardFromInput` and `computeMasksGPU` each split their work into several
submits with a `queue.onSubmittedWorkDone()` yield between, so the GPU process can paint a compositor frame
instead of the tab freezing for the whole multi-second pass (don't re-collapse these into one submit). The
larger fix — **tiling** (overlapping tiles + taper-blended stitch), which would remove the cap, restore
small-cell accuracy, make submits inherently small, and enable out-of-core/`.zarr` inputs — is deferred future
work (noted in the README and at the cap site in `cellpose_core.js`).

## Pull Request Template

- Always include a section titled **Summary** that covers all commits in the PR in one concise paragraph — not a
  per-commit bullet list.
- Always include a section titled **Notes** that mentions any bugs of interest fixed along the way (incidental fixes
  outside the PR's main purpose) and any recommended areas of improvement that were noticed but not touched.

## Conventions

- No frameworks, no bundler, no TypeScript — plain ES modules loaded directly by the browser
  (`<script type="module">` in each app page). Keep new code dependency-free unless a vendored library already
  covers the need.
- Functions take `state` explicitly rather than closing over module-level globals; keep new app-logic code
  in this style rather than introducing classes/singletons (the GPU backends and `FlatRandomForest` are the
  deliberate exceptions, since they own real GPU/buffer resources).
- Tunable constants (forest size, feature count, debounce timing, camera zoom bounds, stats struct layout)
  live in `js/config.js` — add new ones there rather than inlining magic numbers, especially anything that
  must stay in sync between JS and the WGSL/GLSL shaders.
- JSDoc comments on exported functions are the norm throughout `js/*.js`; match that style for new exports.
