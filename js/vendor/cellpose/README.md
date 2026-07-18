# Cellpose cyto3 on WebGPU

A from-scratch, dependency-free port of the Cellpose **cyto3** cell-segmentation model
that runs entirely in the browser on WebGPU — the WGSL forward pass and flow dynamics
are hand-written; there is no PyTorch, ONNX runtime, WASM, or server involved.

This directory is the **source of truth** for that port — there is no external upstream
repo to sync from. Edit `cellpose_core.js` here directly. (It's still "vendored" in the
sense that it deliberately does **not** follow the app's `state`/JSDoc conventions and is
kept environment-agnostic so it can run headless as well as in the browser.)

## Files

| File | What it is |
|---|---|
| `cellpose_core.js` | The `CellposeWebGPU` class: model load, WGSL UNet forward pass, GPU Euler flow dynamics, CPU mask post-processing. Environment-agnostic (browser + headless). |
| `cyto3_weights.bin.gz` | The cyto3 weights, **BatchNorm folded** into the conv weights, **gzip-compressed** (~23.4 MB; ~26 MB raw). Decompressed client-side — see `js/cellpose.worker.js`. |
| `cyto3_manifest.json` | Per-tensor byte offsets/shapes into the weight blob, plus architecture params (`nbase`, `sz`, `nout`, `diam_mean`). 188 tensors. |

Consumed by `js/cellpose.worker.js` (owns the instance, off the main thread) via
`js/cellpose.js` (the app client). See the "Cellpose" section of the repo `CLAUDE.md`.

## Capabilities

- **cyto3 model**, the same weights the Cellpose desktop tool ships with — no training.
- **Grayscale** segmentation and **2-channel** cyto+nucleus segmentation (`opts.chan2`).
  Grayscale is a first-class mode (Cellpose `channels=[0,0]`), not a degraded one.
- **Full pipeline** from a raw intensity plane: percentile normalization → diameter
  rescale → WGSL UNet (encoder / GPU style vector / decoder / output head) → resize the
  flow + cellprob fields back to native resolution → GPU Euler flow dynamics → CPU mask
  assembly (seed/grow, oversize filter, flow-consistency QC, min-size) → compact `1..N`
  instance labels, with **touching cells kept separate** (the reason to use Cellpose).
- **Validated** against a PyTorch reference (AP@0.5 = 1.000 on the WGSL forward + masks,
  grayscale and 2-channel). This is why the repo's Node unit tests in `js/` do **not**
  cover this directory — it's GPU-dependent and validated separately, not against a
  reference that ships here.
- **Responsive under load.** The forward pass and the dynamics integration are each split
  into several GPU submissions with a yield between them (`forwardFromInput`,
  `computeMasksGPU`). A single all-in-one submit runs as one multi-second GPU task that
  monopolizes the browser's GPU process and freezes the tab; chunking lets the compositor
  paint between chunks. See those functions' comments.

## Limits

- **WebGPU only.** No WebGL2 / WASM / CPU fallback. `cellposeSupported()` gates the UI.
- **2D only.** No z-stacks / 3D volumes.
- **cyto3 only.** No other Cellpose model variants are vendored.
- **No diameter estimation.** Cellpose's size model isn't ported — the caller supplies
  `diameter` (default 30 = cyto3's `diam_mean`). A wrong diameter degrades results.
- **Whole-image, no tiling → a working-resolution cap.** The image is processed in a
  single forward pass, so it's downscaled to fit `min(GPU buffer limit, ~3M px)`. This is
  the main quality/scale ceiling:
  - **Small diameters lose accuracy.** A small diameter upscales the image (so cells reach
    the model's ~30 px training size); past the cap it's clamped back down, so small/dense
    cells are segmented at less-than-ideal scale.
  - **Large images can't run at full resolution.**
  - Runtime is still multiple seconds for large/upscaled inputs — chunking removes the
    *freeze*, not the total compute time.
- **CPU mask post-processing is single-threaded** and scales with cell count (runs in the
  worker, so it doesn't block the UI, but it is part of the per-image time).
- **Instance-label export is uint16** downstream (`js/export.js`) — images with >65535
  objects are skipped there (itk-wasm uint32 write is broken). Not a limit of this core,
  but relevant when many small cells are found.

## Future work: tiling (and out-of-core / `.zarr`)

The single biggest improvement would be to **tile** the image: run overlapping tiles
through the network and stitch the flow/cellprob fields with a taper/Hann-window blend on
the overlaps (as upstream Cellpose does for large images), then run the dynamics once on
the assembled full-resolution field. That would:

- **remove the working-resolution cap** (each tile is small regardless of image size),
- **restore small-cell accuracy** (no whole-image downscale),
- **make each GPU submit inherently small** (so the freeze fix would be a side effect, not
  a hand-rolled chunking of one big submit), and
- **enable out-of-core / `.zarr` inputs** — process tiles streamed in without ever
  materializing the whole array.

It's deferred because it's a real feature (tile extraction with overlap, blended
stitching, and re-validation against the PyTorch reference), not a patch. Until then,
chunking keeps the tab responsive and the resolution cap bounds memory/runtime.
