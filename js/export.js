import { writeImage, setPipelinesBaseUrl } from './vendor/itk-wasm-image-io.min.js';
import { NUM_CLASSES } from './config.js';
import { buildIlpProject, stripExtension } from './ilp.js';
import { decodeObjectStats, buildObjectCsv } from './objects.js';

// Instance labels are written as uint16 (one distinct ID per object), so at most
// this many objects per image can be exported as a label image. itk-wasm's uint32
// path is broken (issue #1544), and uint16 is plenty for cell counts.
const MAX_UINT16_LABEL = 65535;

/**
 * Serializes one 2D ITK image to a `.tif` and appends it to `filesToZip`.
 * Factors out the boilerplate shared by the segmentation / probability / label
 * exports (they differ only in name, pixel type, component count, and data).
 * @param {Array<{name:string,data:ArrayBuffer}>} filesToZip - Accumulator, mutated.
 * @param {Object} spec - {name, w, h, pixelType, componentType, components, data}.
 */
async function addItkImage(filesToZip, { name, w, h, pixelType, componentType, components, data }) {
    const itkImage = {
        imageType: { dimension: 2, pixelType, componentType, components },
        name,
        origin: [0.0, 0.0],
        spacing: [1.0, 1.0],
        direction: new Float64Array([1.0, 0.0, 0.0, 1.0]),
        size: [w, h],
        metadata: new Map(),
        data,
    };
    const filename = `${name}.tif`;
    // webWorker: false — see io.js for why (opaque-origin worker needs CORS).
    const { serializedImage } = await writeImage(itkImage, filename, { webWorker: false });
    filesToZip.push({ name: filename, data: serializedImage.data.buffer });
}

// itk-wasm fetches its WASM pipelines relative to this URL at runtime; point it at the
// vendored copy instead of the jsDelivr CDN default.
setPipelinesBaseUrl(new URL('./vendor/itk-wasm-image-io-pipelines', import.meta.url).href);


/**
 * Generates ITK images / a CSV from each image's GPU results and batches them
 * into a ZIP. The outputs are opt-in via `opts`, so each app requests only what
 * its method actually produces:
 *   - seg    uint8 segmentation mask (argmax of the probability buffer; 0 = bg,
 *            else class+1). Every app.
 *   - prob   float32 per-class probability map. Classifier only (meaningless for
 *            Threshold, whose "probabilities" are just ±1 constants).
 *   - labels uint16 instance-label image (one ID per object). Requires
 *            `instanceLabels` — the label buffer must already hold compact 1..N
 *            instances (Cellpose), since CCL roots aren't 1..N. Skipped, with a
 *            warning, for any image with more than 65535 objects.
 *   - csv    one combined per-object CSV across all images (image, class, label,
 *            centroid, area, intensity min/mean/max). Every app.
 *
 * @param {Array<Object>} images - Images to export.
 * @param {Object} opts
 * @param {boolean} [opts.seg]
 * @param {boolean} [opts.prob]
 * @param {boolean} [opts.labels]
 * @param {boolean} [opts.csv]
 * @param {boolean} [opts.instanceLabels] - Label buffer already holds 1..N
 *   instances (Cellpose): enumerate objects from it directly instead of running
 *   connected-components per class. Required for `labels`.
 * @param {string[]} [opts.classNames] - Per-class names for the CSV class column
 *   (classifier). Defaults to "Class N".
 * @param {string} [opts.instanceClass] - Class label for instance-mode CSV rows
 *   (default "cell").
 * @param {Function} [progressCallback] - Zip progress callback.
 * @returns {Promise<Blob|undefined>} The ZIP blob, or undefined if nothing was produced.
 */
export async function zipImages(images, opts = {}, progressCallback) {
    const {
        seg = false, prob = false, labels = false, csv = false,
        instanceLabels = false, classNames = [], instanceClass = 'cell',
    } = opts;
    const filesToZip = [];
    const csvRows = [];
    const callback = progressCallback || ((m) => console.log(m));

    for (let i = 0; i < images.length; i++) {
        const img = images[i];
        const w = img.width;
        const h = img.height;
        const baseName = img.name ? stripExtension(img.name) : `image_${i}`;

        if (seg || prob) {
            const probs = await img.backend.downloadProbabilities();
            const numClasses = (probs.length / (w * h));

            if (seg) {
                const dataUint8Array = new Uint8Array(w * h);
                for (let j = 0; j < w * h; j++) {
                    let max_p = -1.0;
                    let best_class = -1;
                    for (let c = 0; c < numClasses; c++) {
                        const p = probs[j * numClasses + c];
                        if (p > max_p) {
                            max_p = p;
                            best_class = c;
                        }
                    }
                    dataUint8Array[j] = max_p < 0 ? 0 : best_class + 1;
                }
                await addItkImage(filesToZip, {
                    name: `${baseName}_segmentation`, w, h,
                    pixelType: 'Scalar', componentType: 'uint8', components: 1,
                    data: dataUint8Array,
                });
            }

            if (prob) {
                await addItkImage(filesToZip, {
                    name: `${baseName}_probabilities`, w, h,
                    pixelType: 'Vector', componentType: 'float32', components: numClasses,
                    data: new Float32Array(probs),
                });
            }
        }

        if (labels || csv) {
            const scale = img.range?.scale ?? 1;

            if (instanceLabels) {
                // Cellpose: the label buffer already holds compact 1..N instances.
                // computeStats reads it directly (no CCL, which would renumber to
                // union-find roots and destroy the instance ids).
                const statsResult = await img.backend.computeStats();
                if (statsResult !== null) {
                    if (csv) {
                        const stats = await img.backend.downloadStats();
                        for (const o of decodeObjectStats(stats, scale)) {
                            csvRows.push({ image: img.name ?? baseName, class: instanceClass, ...o });
                        }
                    }
                    if (labels) {
                        const raw = await img.backend.downloadLabels(); // Uint32Array, 1..N
                        let maxLabel = 0;
                        for (let k = 0; k < raw.length; k++) if (raw[k] > maxLabel) maxLabel = raw[k];
                        if (maxLabel > MAX_UINT16_LABEL) {
                            console.warn(`Skipping label image for "${img.name}": ${maxLabel} objects ` +
                                `exceeds the uint16 limit (${MAX_UINT16_LABEL}).`);
                        } else {
                            const u16 = new Uint16Array(raw.length);
                            for (let k = 0; k < raw.length; k++) u16[k] = raw[k];
                            await addItkImage(filesToZip, {
                                name: `${baseName}_labels`, w, h,
                                pixelType: 'Scalar', componentType: 'uint16', components: 1,
                                data: u16,
                            });
                        }
                    }
                }
            } else if (csv) {
                // Classifier / Threshold: no pre-existing instance map, so enumerate
                // objects per class via connected-components (labels export isn't
                // offered here — CCL roots aren't the compact 1..N a label image needs).
                for (let cls = 0; cls < NUM_CLASSES; cls++) {
                    await img.backend.computeConnectedComponents(cls);
                    const statsResult = await img.backend.computeStats();
                    if (statsResult === null) continue;
                    const stats = await img.backend.downloadStats();
                    const className = classNames[cls] ?? `Class ${cls + 1}`;
                    for (const o of decodeObjectStats(stats, scale)) {
                        csvRows.push({ image: img.name ?? baseName, class: className, ...o });
                    }
                }
            }
        }
    }

    if (csv && csvRows.length > 0) {
        const bytes = new TextEncoder().encode(buildObjectCsv(csvRows));
        filesToZip.push({ name: 'objects.csv', data: bytes.buffer });
    }

    if (filesToZip.length > 0) {
        /**
         * To prevent the main thread from locking during the zip we do it in a webworker.
         * This allows a progress callback to update regularly.
         */
        return new Promise((resolve, reject) => {
            const workerCode = `
                importScripts(new URL('/js/vendor/jszip.min.js', self.location.origin).href);

                self.onmessage = async function(e) {
                    const files = e.data;
                    const zip = new JSZip();

                    for (const file of files) {
                        zip.file(file.name, file.data);
                    }

                    try {
                        const content = await zip.generateAsync({ type: "uint8array" }, (metadata) => {
                            self.postMessage({ type: 'progress', metadata });
                        });
                        self.postMessage({ type: 'done', content: content.buffer }, [content.buffer]);
                    } catch (error) {
                        self.postMessage({ type: 'error', error: error.message });
                    }
                };
            `;

            const workerBlob = new Blob([workerCode], { type: 'application/javascript' });
            const workerUrl = URL.createObjectURL(workerBlob);
            const worker = new Worker(workerUrl);

            worker.onmessage = (e) => {
                const { type, metadata, content, error } = e.data;

                if (type === 'progress') {
                    callback(metadata);
                }
                else if (type === 'done') {
                    worker.terminate();
                    URL.revokeObjectURL(workerUrl); // Cleanup
                    const blob = new Blob([content], { type: 'application/zip' });
                    resolve(blob);
                }
                else if (type === 'error') {
                    worker.terminate();
                    URL.revokeObjectURL(workerUrl);
                    reject(new Error(`Zip creation failed: ${error}`));
                }
            };

            // Extract ArrayBuffers to transfer ownership (Zero-copy for better performance)
            const transferables = filesToZip.map(f => f.data);
            worker.postMessage(filesToZip, transferables);
        });
    }
}

/**
 * Builds a downloadable, fully self-contained ilastik Pixel Classification
 * project (`.ilp`) reflecting the current app state: one lane per loaded
 * image with its raw pixels embedded directly in the file (ilastik's
 * "Project Internal" storage — browsers can't expose a real filesystem path
 * for an uploaded file, so this avoids depending on the original image files
 * still being around), the user's painted labels as one bounding-box block
 * per image, label names/colors, and a starter feature selection. No trained
 * classifier is embedded — Quantosaurus trains on a custom on-GPU feature
 * bank that doesn't correspond to the vigra features desktop ilastik
 * computes itself, so ilastik retrains from the exported labels on open
 * instead (see js/ilp.js for the full rationale).
 * @param {Object} state - Shared app state.
 * @param {Object} [options] - Forwarded to buildIlpProject (e.g. classNames).
 * @returns {Blob} The `.ilp` file, ready for download.
 */
export function exportIlp(state, options = {}) {
    const bytes = buildIlpProject(state, options);
    return new Blob([bytes], { type: 'application/octet-stream' });
}
