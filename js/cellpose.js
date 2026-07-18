// Cellpose (cyto3) segmentation — main-thread client. The actual work (weight
// loading, the WGSL forward pass, GPU flow dynamics, and the synchronous CPU mask
// post-processing) runs in a dedicated Web Worker (js/cellpose.worker.js) so it never
// freezes the UI: small diameters push ~3x the GPU work and ~4x the cells, and the
// mask post-processing is single-threaded CPU that would otherwise block the tab for
// seconds. This module just spawns the worker, relays load progress, and — once the
// worker hands back an instance-label map — uploads it into each image's own backend
// (setLabels bypasses CCL so touching cells stay separate) and counts.
//
// Vendored core is validated upstream (grayscale + 2-channel, AP@0.5 = 1.000 vs
// PyTorch — see js/vendor/cellpose/). The worker owns the single CellposeWebGPU
// instance (one device, weights loaded once); labels cross back as a transferable
// Int32Array, so nothing GPU-side is shared with the per-image backends.
import { NUM_CLASSES } from './config.js';
import { decodeObjectStats } from './objects.js';
import { setCentroids, renderCentroids } from './images.js';
import { animateCount, setClassBadgesLoading } from './ui.js';

const FOREGROUND_CLASS = 0; // Cellpose fills the single foreground/overlay class

let _worker = null;         // the segmentation worker; spawned lazily
let _loaded = false;        // true once the worker reports weights loaded
let _loadPromise = null;    // in-flight load shared by concurrent callers
let _loadWaiters = null;    // { resolve, reject } for the in-flight load
let _onProgress = null;     // download-progress callback for the in-flight load
let _reqSeq = 0;            // monotonic id pairing segment requests with results
const _pending = new Map(); // reqId -> { resolve, reject }

/** Whether this browser can run Cellpose (WebGPU-only; no WebGL2 fallback). */
export function cellposeSupported() {
    return !!navigator.gpu;
}

/** Spawn (once) the module worker and route its messages to the pending promises. */
function getWorker() {
    if (_worker) return _worker;
    _worker = new Worker(new URL('./cellpose.worker.js', import.meta.url), { type: 'module' });
    _worker.onmessage = (e) => {
        const msg = e.data;
        switch (msg.type) {
            case 'progress':
                _onProgress?.(msg.frac, msg.loaded, msg.total);
                break;
            case 'loaded':
                _loaded = true;
                _loadWaiters?.resolve();
                _loadWaiters = null;
                break;
            case 'load-error':
                _loadWaiters?.reject(new Error(msg.message));
                _loadWaiters = null;
                break;
            case 'seg-progress': {
                _pending.get(msg.reqId)?.onProgress?.(msg.frac);
                break;
            }
            case 'result': {
                const p = _pending.get(msg.reqId);
                if (p) { _pending.delete(msg.reqId); p.resolve(msg.labels); }
                break;
            }
            case 'seg-error': {
                const p = _pending.get(msg.reqId);
                if (p) { _pending.delete(msg.reqId); p.reject(new Error(msg.message)); }
                break;
            }
        }
    };
    // A worker-level crash (e.g. WebGPU device lost) rejects everything outstanding.
    _worker.onerror = (e) => {
        const err = new Error(e.message || 'Cellpose worker crashed');
        _loadWaiters?.reject(err); _loadWaiters = null;
        _loadPromise = null;
        for (const p of _pending.values()) p.reject(err);
        _pending.clear();
    };
    return _worker;
}

/**
 * Lazily load the Cellpose weights (~26MB) in the worker, fetched once and cached via
 * the Cache API so later visits are instant. The first fetch is streamed and reported
 * through onProgress.
 * @param {(fraction:number, loaded:number, total:number)=>void} [onProgress]
 * @returns {Promise<void>}
 */
export function ensureCellposeLoaded(onProgress) {
    if (_loaded) return Promise.resolve();
    if (onProgress) _onProgress = onProgress;
    if (_loadPromise) return _loadPromise;
    const worker = getWorker();
    _loadPromise = new Promise((resolve, reject) => {
        _loadWaiters = { resolve, reject };
        worker.postMessage({ type: 'load' });
    }).catch((err) => {
        _loadPromise = null; // allow a retry after a failed load
        throw err;
    });
    return _loadPromise;
}

/** Hand one grayscale plane (+ optional nuclear channel) to the worker; resolves to its labels. */
function segmentInWorker(gray, H, W, diameter, chan2, onProgress) {
    const worker = getWorker();
    const reqId = ++_reqSeq;
    // Copy into fresh buffers we can transfer, so the app's own intensityArray (still
    // needed for display/export) isn't detached. slice() copies once; the transfer
    // then moves that copy without a second structured-clone copy.
    const grayCopy = gray.slice();
    const chan2Copy = chan2 ? chan2.slice() : null;
    const transfer = [grayCopy.buffer];
    if (chan2Copy) transfer.push(chan2Copy.buffer);
    return new Promise((resolve, reject) => {
        _pending.set(reqId, { resolve, reject, onProgress });
        worker.postMessage({ type: 'segment', reqId, gray: grayCopy, H, W, diameter, chan2: chan2Copy }, transfer);
    });
}

/**
 * Segments every loaded image with Cellpose, uploads the instance labels into each
 * backend (bypassing CCL so touching cells stay separate), paints a foreground
 * overlay, and reports the total cell count. This is Cellpose's analogue of
 * trainAndPredictAll. Loads the weights first if a caller skipped ensureCellposeLoaded.
 * @param {Object} state - Shared app state (reads state.images, state.cellpose).
 * @param {Object} [opts]
 * @param {(index:number, count:number, name:string)=>void} [opts.onImageStart]
 *   Fired before each image is segmented (for per-image progress).
 * @param {(index:number, frac:number)=>void} [opts.onImageProgress]
 *   Fired repeatedly while an image segments, with a 0..1 fraction (chunk-driven).
 * @returns {Promise<number>} Total cells across all images.
 */
export async function runCellpose(state, { onImageStart, onImageProgress } = {}) {
    await ensureCellposeLoaded();
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
        const labels = await segmentInWorker(cyto, img.height, img.width, diameter, nuc,
            (frac) => onImageProgress?.(i, frac));

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
