// Cellpose (cyto3) segmentation wrapper. Owns a single module-level CellposeWebGPU
// instance with its OWN WebGPU device, distinct from the per-image backends: each
// image constructs its own device (see images.js:initializeBackend), so sharing
// one image's device would re-upload the 26MB of weights per image. Cellpose's
// output crosses back to the CPU as an Int32Array anyway, so a separate device is
// free — the labels are then uploaded into each image's backend via setLabels.
//
// Vendored core is validated upstream (grayscale + 2-channel, AP@0.5 = 1.000 vs
// PyTorch — see js/vendor/cellpose/). This module handles lazy/cached weight
// loading (the prototype had neither) and the app orchestration.
import { CellposeWebGPU } from './vendor/cellpose/cellpose_core.js';
import { NUM_CLASSES } from './config.js';
import { decodeObjectStats } from './objects.js';
import { setCentroids, renderCentroids } from './images.js';
import { animateCount, setClassBadgesLoading } from './ui.js';

const WEIGHTS_URL = new URL('./vendor/cellpose/cyto3_weights.bin', import.meta.url).href;
const MANIFEST_URL = new URL('./vendor/cellpose/cyto3_manifest.json', import.meta.url).href;
// Bump the version suffix if the vendored weights ever change, to invalidate the
// old cached blob.
const CACHE_NAME = 'quantosaurus-cellpose-v1';
const WEIGHTS_BYTES = 26403372; // fallback for progress when content-length is absent
const FOREGROUND_CLASS = 0;     // Cellpose fills the single foreground/overlay class

let _cp = null;          // CellposeWebGPU (owns its device); null until loaded
let _loadPromise = null; // in-flight load shared by concurrent callers

/** Whether this browser can run Cellpose (WebGPU-only; no WebGL2 fallback). */
export function cellposeSupported() {
    return !!navigator.gpu;
}

/**
 * Lazily create the Cellpose instance and load its weights (~26MB), fetched once
 * and cached via the Cache API so later visits are instant. The first fetch is
 * streamed and reported through onProgress.
 * @param {(fraction:number, loaded:number, total:number)=>void} [onProgress]
 * @returns {Promise<CellposeWebGPU>}
 */
export async function ensureCellposeLoaded(onProgress) {
    if (_cp) return _cp;
    if (_loadPromise) return _loadPromise;
    _loadPromise = (async () => {
        const cp = await CellposeWebGPU.create();
        const [manifest, weights] = await Promise.all([
            fetch(MANIFEST_URL).then(r => r.json()),
            fetchWeightsCached(onProgress),
        ]);
        cp.loadWeights(manifest, weights);
        _cp = cp;
        return cp;
    })();
    try {
        return await _loadPromise;
    } catch (err) {
        _loadPromise = null; // allow a retry after a failed load
        throw err;
    }
}

/** Fetch the weight blob, from the Cache API if present, streaming for progress. */
async function fetchWeightsCached(onProgress) {
    const cache = ('caches' in self) ? await caches.open(CACHE_NAME) : null;
    let resp = cache ? await cache.match(WEIGHTS_URL) : null;
    const cached = !!resp;
    if (!resp) {
        resp = await fetch(WEIGHTS_URL);
        if (!resp.ok) throw new Error(`Failed to fetch Cellpose weights (${resp.status})`);
        if (cache) await cache.put(WEIGHTS_URL, resp.clone());
    }

    const total = Number(resp.headers.get('content-length')) || WEIGHTS_BYTES;
    const reader = resp.body.getReader();
    const chunks = [];
    let loaded = 0;
    for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        loaded += value.length;
        onProgress?.(Math.min(1, loaded / total), loaded, total);
    }
    const out = new Uint8Array(loaded);
    let off = 0;
    for (const c of chunks) { out.set(c, off); off += c.length; }
    onProgress?.(1, loaded, total);
    void cached;
    return out.buffer;
}

/**
 * Segments every loaded image with Cellpose, uploads the instance labels into each
 * backend (bypassing CCL so touching cells stay separate), paints a foreground
 * overlay, and reports the total cell count. Must be called after
 * ensureCellposeLoaded. This is Cellpose's analogue of trainAndPredictAll.
 * @param {Object} state - Shared app state (reads state.images, state.cellpose).
 * @param {Object} [opts]
 * @param {(index:number, count:number, name:string)=>void} [opts.onImageStart]
 *   Fired before each image is segmented (for per-image progress).
 * @returns {Promise<number>} Total cells across all images.
 */
export async function runCellpose(state, { onImageStart } = {}) {
    if (!_cp) throw new Error('Cellpose weights not loaded — call ensureCellposeLoaded first.');
    const { cytoChannel = 0, nucChannel = -1, diameter = 30 } = state.cellpose;
    setClassBadgesLoading();

    let total = 0;
    for (let i = 0; i < state.images.length; i++) {
        const img = state.images[i];
        onImageStart?.(i, state.images.length, img.name);

        // Resolve channels for this image, clamped to what it actually has (loaded
        // files may differ in channel count). channels is null for a plain
        // single-channel image — use its lone intensity array.
        const chans = img.channels;
        const cytoIdx = chans ? Math.min(cytoChannel, chans.length - 1) : 0;
        const cyto = chans ? chans[cytoIdx].intensityArray : img.intensityArray;
        const nuc = (chans && nucChannel >= 0 && nucChannel < chans.length)
            ? chans[nucChannel].intensityArray : null;

        // segmentImage takes (gray, H, W, opts) — height before width.
        const { labels } = await _cp.segmentImage(cyto, img.height, img.width,
            { diameter, chan2: nuc });

        img.backend.setLabels(labels);
        img.backend.setProbabilities(foregroundProbs(labels, img.width, img.height));

        // Count instances + get centroids from the uploaded labels (computeStats
        // reads them directly; no CCL, preserving each cell's id).
        const statsResult = await img.backend.computeStats();
        const objectsByClass = Array.from({ length: NUM_CLASSES }, () => []);
        if (statsResult !== null) {
            const stats = await img.backend.downloadStats();
            const objects = decodeObjectStats(stats, img.range?.scale ?? 1);
            objectsByClass[FOREGROUND_CLASS] = objects;
            total += objects.length;
        }
        setCentroids(img, objectsByClass);
        renderCentroids(state, img);
    }

    animateCount(total);
    return total;
}

/**
 * Build a foreground/background probability map from an instance-label image:
 * class 0 = 1.0 where labelled (any cell), -1.0 sentinel elsewhere so the
 * composite leaves the background untinted.
 */
function foregroundProbs(labels, w, h) {
    const probs = new Float32Array(w * h * NUM_CLASSES).fill(-1.0);
    for (let p = 0; p < labels.length; p++) {
        if (labels[p] > 0) probs[p * NUM_CLASSES + FOREGROUND_CLASS] = 1.0;
    }
    return probs;
}
