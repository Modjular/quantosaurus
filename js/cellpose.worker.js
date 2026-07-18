// Cellpose segmentation worker. Runs the whole pipeline — the WGSL forward pass, the
// GPU flow dynamics, AND the synchronous CPU mask post-processing — off the main
// thread, so a heavy run (small diameters do ~3x the GPU work and ~4x the cells) never
// freezes the UI. WebGPU, fetch, the Cache API, and DecompressionStream are all
// available in a dedicated module worker, so weight loading lives here too; only the
// small payloads (a grayscale plane in, an instance-label map out) cross the boundary.
//
// The vendored core (cellpose_core.js) is environment-agnostic — it touches only
// navigator.gpu — so it runs here unchanged. See js/cellpose.js for the client.
import { CellposeWebGPU } from './vendor/cellpose/cellpose_core.js';

// Resolved relative to this worker's own URL (js/), so they match the paths the
// main-thread module used — including the Cache API key, so a blob cached by an
// earlier build (same CACHE_NAME + URL) is reused rather than re-downloaded.
const WEIGHTS_URL = new URL('./vendor/cellpose/cyto3_weights.bin.gz', import.meta.url).href;
const MANIFEST_URL = new URL('./vendor/cellpose/cyto3_manifest.json', import.meta.url).href;
const CACHE_NAME = 'quantosaurus-cellpose-v2';
const WEIGHTS_GZ_BYTES = 24576400; // compressed size; progress fallback when content-length is absent

let cp = null;          // CellposeWebGPU (owns its device); null until loaded
let loadPromise = null; // in-flight load shared by concurrent messages

self.onmessage = async (e) => {
    const msg = e.data;
    if (msg.type === 'load') {
        try {
            await ensureLoaded();
            self.postMessage({ type: 'loaded' });
        } catch (err) {
            self.postMessage({ type: 'load-error', message: err.message });
        }
    } else if (msg.type === 'segment') {
        try {
            await ensureLoaded();
            const { labels } = await cp.segmentImage(msg.gray, msg.H, msg.W, {
                diameter: msg.diameter, chan2: msg.chan2,
                onProgress: (frac) => self.postMessage({ type: 'seg-progress', reqId: msg.reqId, frac }),
            });
            // Transfer the freshly-allocated label buffer back (no copy).
            self.postMessage({ type: 'result', reqId: msg.reqId, labels }, [labels.buffer]);
        } catch (err) {
            self.postMessage({ type: 'seg-error', reqId: msg.reqId, message: err.message });
        }
    }
};

/** Lazily create the instance and load its weights, shared across concurrent callers. */
async function ensureLoaded() {
    if (cp) return;
    if (!loadPromise) {
        loadPromise = (async () => {
            const instance = await CellposeWebGPU.create();
            const [manifest, weights] = await Promise.all([
                fetch(MANIFEST_URL).then(r => r.json()),
                fetchWeightsCached(),
            ]);
            instance.loadWeights(manifest, weights);
            cp = instance;
        })();
    }
    try {
        await loadPromise;
    } catch (err) {
        loadPromise = null; // allow a retry after a failed load
        throw err;
    }
}

/**
 * Fetch the gzip-compressed weight blob (from the Cache API if present), posting
 * download progress to the main thread, and inflate it with DecompressionStream.
 * Reads the compressed body ourselves so progress fires as bytes arrive — cache.put
 * must not run first, since it drains the whole body before it resolves.
 */
async function fetchWeightsCached() {
    const cache = ('caches' in self) ? await caches.open(CACHE_NAME) : null;
    let resp = cache ? await cache.match(WEIGHTS_URL) : null;
    const fromCache = !!resp;
    if (!resp) {
        resp = await fetch(WEIGHTS_URL);
        if (!resp.ok) throw new Error(`Failed to fetch Cellpose weights (${resp.status})`);
    }

    const total = Number(resp.headers.get('content-length')) || WEIGHTS_GZ_BYTES;
    const reader = resp.body.getReader();
    const chunks = [];
    let loaded = 0;
    for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        loaded += value.length;
        self.postMessage({ type: 'progress', frac: Math.min(1, loaded / total), loaded, total });
    }
    self.postMessage({ type: 'progress', frac: 1, loaded, total });

    const compressed = new Blob(chunks);
    if (!fromCache && cache) await cache.put(WEIGHTS_URL, new Response(compressed, { headers: resp.headers }));

    const inflated = compressed.stream().pipeThrough(new DecompressionStream('gzip'));
    return await new Response(inflated).arrayBuffer();
}
