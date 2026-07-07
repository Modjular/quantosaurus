import { writeImage, setPipelinesBaseUrl } from './vendor/itk-wasm-image-io.min.js';

// itk-wasm fetches its WASM pipelines relative to this URL at runtime; point it at the
// vendored copy instead of the jsDelivr CDN default.
setPipelinesBaseUrl(new URL('./vendor/itk-wasm-image-io-pipelines', import.meta.url).href);


/**
 * Handles generating ITK images and batching files into ZIPs.
 * @param {Array<Object>} images - Array of image objects to export.
 * @param {boolean} exportSeg - Whether to export segmentation masks.
 * @param {boolean} exportProb - Whether to export probability maps.
 * @param {Function} progressCallback - Optional callback for UI progress updates.
 * @returns {Promise<Blob>} A promise that resolves to the generated ZIP Blob.
 */
export async function zipImages(images, exportSeg, exportProb, progressCallback) {
    // Array to temporarily hold file data if we are zipping
    const filesToZip = [];
    const callback = progressCallback || ((m) => console.log(m));

    for (let i = 0; i < images.length; i++) {
        const img = images[i];
        const probs = await img.backend.downloadProbabilities();
        const w = img.width;
        const h = img.height;
        const baseName = img.name ? img.name.replace(/\.[^/.]+$/, "") : `image_${i}`;
        const numClasses = (probs.length / (w * h));

        if (exportSeg) {
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

            const itkImage = {
                imageType: {
                    dimension: 2,
                    pixelType: 'Scalar',
                    componentType: 'uint8',
                    components: 1
                },
                name: `${baseName}_segmentation`,
                origin: [0.0, 0.0],
                spacing: [1.0, 1.0],
                direction: new Float64Array([1.0, 0.0, 0.0, 1.0]),
                size: [w, h],
                metadata: new Map(),
                data: dataUint8Array
            };

            const filename = `${itkImage.name}.tif`;
            // webWorker: false — see io.js for why (opaque-origin worker needs CORS).
            const { serializedImage } = await writeImage(itkImage, filename, { webWorker: false });

            filesToZip.push({ name: filename, data: serializedImage.data.buffer });
        }

        if (exportProb) {
            const dataFloat32Array = new Float32Array(probs);

            const itkImage = {
                imageType: {
                    dimension: 2,
                    pixelType: 'Vector',
                    componentType: 'float32',
                    components: numClasses
                },
                name: `${baseName}_probabilities`,
                origin: [0.0, 0.0],
                spacing: [1.0, 1.0],
                direction: new Float64Array([1.0, 0.0, 0.0, 1.0]),
                size: [w, h],
                metadata: new Map(),
                data: dataFloat32Array
            };

            const filename = `${itkImage.name}.tif`;
            // webWorker: false — see io.js for why (opaque-origin worker needs CORS).
            const { serializedImage } = await writeImage(itkImage, filename, { webWorker: false });

            filesToZip.push({ name: filename, data: serializedImage.data.buffer });
        }

        const exportLabels = false; // TODO: re-enable once itk-wasm label export bug is fixed
        if (exportLabels) {
            // Before export, make sure labels are fully connected, then run stats

            performance.mark('exportLabels')
            await img.backend.computeConnectedComponents(1);
            await img.backend.computeStats()

            const labels = await img.backend.downloadLabels()
            const data = await img.backend.downloadStats()

            /**
             * struct DenseMetrics {
             *     label: u32,
             *     area: u32,
             *     total_intensity: u32,
             *     sum_x: u32,
             *     sum_y: u32,
             *     min_intensity: u32,
             *     max_intensity: u32
             * };
             */

            const statsStructCount = 7;
            const numLabels = data.length / statsStructCount
            console.log("unique vs numLabels", new Set(labels).size, numLabels)

            for (let i = 0; i < 10 * statsStructCount; i += statsStructCount) {
                const label = data[i + 0]
                const area = data[i + 1];

                if (area === 0) continue; // Label not present

                const totalIntensity = data[i + 2];
                const sumX = data[i + 3];
                const sumY = data[i + 4];
                const minIntensityRaw = data[i + 5];
                const maxIntensityRaw = data[i + 6];

                // Compute the averages
                const centroidX = sumX / area;
                const centroidY = sumY / area;
                const avgIntensityRaw = totalIntensity / area;

                // If you scaled by 10000.0 in WGSL, scale back down:
                const minIntensity = minIntensityRaw / 10000.0;
                const maxIntensity = maxIntensityRaw / 10000.0;
                const avgIntensity = avgIntensityRaw / 10000.0;

                console.log(`Label ${label}: Centroid(${centroidX.toFixed(2)}, ${centroidY.toFixed(2)}), Intensity[Min: ${minIntensity}, Max: ${maxIntensity}, Avg: ${avgIntensity.toFixed(2)}]`);
            }

            // TODO: Waiting on bugfix from itk-wasm
            // TODO: https://github.com/InsightSoftwareConsortium/ITK-Wasm/issues/1544

            // const labels = await img.backend.downloadLabels()

            // const itkImage = {
            //     imageType: {
            //         dimension: 2,
            //         pixelType: 'Scalar',
            //         componentType: 'uint32',
            //         components: 1
            //     },
            //     name: `${baseName}_labels`,
            //     origin: [0.0, 0.0],
            //     spacing: [1.0, 1.0],
            //     direction: new Float64Array([1.0, 0.0, 0.0, 1.0]),
            //     // size: [w, h],
            //     size: [100, 100],
            //     metadata: new Map(),
            //     data: new Uint32Array(100 * 100),
            // };

            // const filename = `${itkImage.name}.tif`;
            // const { serializedImage } = await writeImage(itkImage, filename);

            // filesToZip.push({ name: filename, data: serializedImage.data.buffer });
        }
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
