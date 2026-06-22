import { readImage, writeImage } from "https://cdn.jsdelivr.net/npm/@itk-wasm/image-io@1.6.0/dist/bundle/index-worker-embedded.min.js";


/**
 * Loads an image file into a typed array, handling both TIFFs and standard web formats.
 * @param {File} file - The image file to load.
 * @returns {Promise<{intensityArray: Float32Array, rgba: Uint8Array|Uint8ClampedArray, w: number, h: number}>} An object containing the normalized intensity array, the raw RGBA array, and the dimensions.
 */
export async function loadFileIntoArray(file) {
  let data, rgba, w, h;

  if (file.name.endsWith('.tif') || file.name.endsWith('.tiff')) {
    const { image } = await readImage(file)

    w = image.size[0]
    h = image.size[1]
    rgba = new Uint8Array(image.data.length * 4)
    data = image.data
  } else {
    const img = await createImageBitmap(file);
    w = img.width;
    h = img.height;
    const off = new OffscreenCanvas(w, h);
    const ctx = off.getContext('2d');
    ctx.drawImage(img, 0, 0);
    rgba = ctx.getImageData(0, 0, w, h).data;
    data = new Float32Array(w * h)

    // Convert RGBA to intensity
    for (let i = 0; i < w * h; i++) {
        const r = rgba[i * 4] / 255;
        const g = rgba[i * 4 + 1] / 255;
        const b = rgba[i * 4 + 2] / 255;
        data[i] = 0.299 * r + 0.587 * g + 0.114 * b;
    }
  }

  const intensityArray = new Float32Array(w * h);
  intensityToRGBA(data, rgba, intensityArray);

  return { intensityArray, rgba, w, h }
}

/**
 * Normalizes an intensity array and converts it to RGBA.
 * @param {Float32Array|Array} data - The input data to normalize.
 * @param {Uint8Array|Uint8ClampedArray} rgba - Array to write RGBA values.
 * @param {Float32Array} [intensityArray] - Optional array to write normalized intensities.
 */
export function intensityToRGBA(data, rgba, intensityArray) {
    let [min, max] = [Infinity, -Infinity];
    const n = data.length;
    for (let i = 0; i < n; i++) {
        if (data[i] < min) min = data[i];
        if (data[i] > max) max = data[i];
    }

    const range = (max - min) > 0 ? (max - min) : 255;
    for (let i = 0; i < n; i++) {
        const norm = (data[i] - min) / range;
        if (intensityArray) intensityArray[i] = norm;

        const val8 = norm * 255;
        rgba[i * 4] = val8;
        rgba[i * 4 + 1] = val8;
        rgba[i * 4 + 2] = val8;
        rgba[i * 4 + 3] = 255;
    }
}

/**
 * Verifies or requests permission to access a file or directory handle.
 * @param {FileSystemHandle} fileHandle - The file or directory handle to verify.
 * @param {boolean} readWrite - Whether to request read-write permission (true) or just read permission (false).
 * @returns {Promise<boolean>} True if permission was granted, false otherwise.
 */
export async function verifyPermission(fileHandle, readWrite) {
  const options = {};
  if (readWrite) {
    options.mode = 'readwrite';
  }
  if ((await fileHandle.queryPermission(options)) === 'granted') {
    return true;
  }
  if ((await fileHandle.requestPermission(options)) === 'granted') {
    return true;
  }
  return false;
}

/**
 * Handles writing a file to disk using the File System API
 * @param {FileSystemDirectoryHandle} folderHandle 
 * @param {String} filename 
 * @param {Blob} blob 
 */
export async function writeFile(folderHandle, filename, blob) {
  const fileHandle = await folderHandle.getFileHandle(filename, { create: true });
  const writable = await fileHandle.createWritable();
  await writable.write(blob);
  await writable.close();
}

/**
 * Aggregates features and labels across all images into 1D typed arrays for Random Forest training.
 * @param {Array<Object>} images - Array of image objects containing labels and a backend to gather features.
 * @param {number} totalLabels - The total number of labels across all images.
 * @returns {Promise<{combinedX: Float32Array, yArray: Int32Array}>} An object containing the concatenated features (combinedX) and their corresponding labels (yArray).
 */
export async function buildTrainingDataset(images, totalLabels) {
    const allX = [];
    const yArray = new Int32Array(totalLabels);
    let currentLabelOffset = 0;

    for (const img of images) {
        const numLabels = img.labels.length;
        if (numLabels === 0) continue;

        const indicesArray = new Uint32Array(numLabels);
        for (let i = 0; i < numLabels; i++) {
            const l = img.labels[i];
            indicesArray[i] = l.y * img.width + l.x;
            yArray[currentLabelOffset + i] = l.cls;
        }

        const X_img = await img.backend.gatherFeaturesForTraining(indicesArray);
        allX.push(X_img);
        currentLabelOffset += numLabels;
    }

    const totalFeatureLength = allX.reduce((sum, arr) => sum + arr.length, 0);
    const combinedX = new Float32Array(totalFeatureLength);
    let xOffset = 0;
    
    for (const arr of allX) {
        combinedX.set(arr, xOffset);
        xOffset += arr.length;
    }

    return { combinedX, yArray };
}

/**
 * Handles generating ITK images, requesting file permissions, and batching files into ZIPs or directories.
 * @param {Array<Object>} images - Array of image objects to export.
 * @param {FileSystemDirectoryHandle} outputDirHandle - Optional handle to the output directory. If omitted, exports to a ZIP.
 * @param {boolean} exportSeg - Whether to export segmentation masks.
 * @param {boolean} exportProb - Whether to export probability maps.
 * @param {Function} progressCallback - Optional callback for UI progress updates.
 * @returns {Promise<void>}
 */
export async function exportImagesData(images, outputDirHandle, exportProb, exportSeg, progressCallback) {
    // Array to temporarily hold file data if we are zipping
    const filesToZip = []; 

    if (outputDirHandle) {
        const permission = await verifyPermission(outputDirHandle, true);
        if (!permission) {
            throw new Error("Permission to write to output directory was denied.");
        }
    }

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

            const filename = `${baseName}_segmentation.tif`;
            const { serializedImage } = await writeImage(itkImage, `${itkImage.name}.tif`);
            
            if (outputDirHandle) {
                const blob = new Blob([serializedImage.data], { type: 'image/tiff' });
                await writeFile(outputDirHandle, filename, blob);
            } else {
                // Store the raw ArrayBuffer for the worker
                filesToZip.push({ name: filename, data: serializedImage.data.buffer });
            }
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

            const filename = `${baseName}_probabilities.tif`;
            const { serializedImage } = await writeImage(itkImage, `${itkImage.name}.tif`);
            
            if (outputDirHandle) {
                const blob = new Blob([serializedImage.data], { type: 'image/tiff' });
                await writeFile(outputDirHandle, filename, blob);
            } else {
                // Store the raw ArrayBuffer for the worker
                filesToZip.push({ name: filename, data: serializedImage.data.buffer });
            }
        }
    }

    if (!outputDirHandle && filesToZip.length > 0) {
        /**
         * To prevent the main thread from locking during the zip we do it in a webworker.
         * This allows a progress callback to update regularly.
         */
        return new Promise((resolve, reject) => {
            const workerCode = `
                importScripts('https://cdn.jsdelivr.net/npm/jszip@3.10.1/dist/jszip.min.js');
                
                self.onmessage = async function(e) {
                    const files = e.data;
                    const zip = new JSZip();
                    
                    for (const file of files) {
                        zip.file(file.name, file.data);
                    }
                    
                    try {
                        const content = await zip.generateAsync({ type: "blob" }, (metadata) => {
                            self.postMessage({ type: 'progress', metadata });
                        });
                        self.postMessage({ type: 'done', content });
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
                    progressCallback(metadata.percent, metadata.currentFile);
                } 
                else if (type === 'done') {
                    const link = document.createElement('a');
                    const yyyymmdd = new Date().toISOString().slice(0,10).replace(/-/g,"");
                    link.href = URL.createObjectURL(content);
                    link.download = `ilastik_export_${yyyymmdd}.zip`;
                    link.click();
                    URL.revokeObjectURL(link.href);
                    
                    worker.terminate();
                    URL.revokeObjectURL(workerUrl); // Cleanup
                    resolve();
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
 * Returns an array of pixel coordinates that fall within a given radius of a center point.
 * @param {number} cx - The x-coordinate of the center.
 * @param {number} cy - The y-coordinate of the center.
 * @param {number} radius - The radius of the circle.
 * @param {number} width - The width of the bounding canvas/image.
 * @param {number} height - The height of the bounding canvas/image.
 * @returns {Array<{x: number, y: number}>} Array of point objects containing the coordinates within the radius.
 */
export function getPixelsInRadius(cx, cy, radius, width, height) {
    const pixels = [];
    
    // Exact 1-pixel brush
    if (radius === 1) {
        if (cx >= 0 && cx < width && cy >= 0 && cy < height) {
            pixels.push({ x: cx, y: cy });
        }
        return pixels;
    }
    
    const rSq = radius * radius;
    const rInt = Math.ceil(radius);
    
    // Check a bounding box around the center
    for (let y = cy - rInt; y <= cy + rInt; y++) {
        for (let x = cx - rInt; x <= cx + rInt; x++) {
            // Keep it inside canvas bounds
            if (x >= 0 && x < width && y >= 0 && y < height) {
                const dx = x - cx;
                const dy = y - cy;
                
                // If distance squared is within radius squared, it's inside the circle
                if (dx * dx + dy * dy <= rSq) {
                    pixels.push({ x, y });
                }
            }
        }
    }
    
    return pixels;
}

/**
 * Given a multi-dimensional image shape array (e.g. [10, 512, 768] for a
 * Z-stack), return the two axis indices that should be used as the display XY plane.
 * 
 * Strategy: prefer the last two dimensions (NumPy / TIFF channel-first
 * convention: [Z, Y, X] or [C, Y, X]), which is almost always correct.
 * The "two largest" heuristic would mis-select a large Z-stack as X/Y, so
 * we intentionally avoid it.
 * 
 * For 2-D images (shape.length === 2) this returns [0, 1] and callers should
 * hide the axis selector entirely — no-op selection.
 * 
 * @param {Array<number>} shape - The dimensions of the image.
 * @returns {{ axisY: number, axisX: number }} Object containing the indices for the Y and X display axes.
 */
export function inferAxes(shape) {
    if (!Array.isArray(shape) || shape.length < 2) {
        throw new Error(`inferAxes: shape must have at least 2 dimensions, got ${JSON.stringify(shape)}`);
    }
    const n = shape.length;
    return { axisY: n - 2, axisX: n - 1 };
}

/**
 * Extract a 2-D slice from a flat typed array representing an n-D image.
 * 
 * This is intentionally simple: it iterates in JS rather than using GPU tricks.
 * For large stacks the caller should call this once at slice-select time and
 * cache the result, not on every frame.
 * 
 * @param {Float32Array|TypedArray} ndarray - Flat typed array of the full image data.
 * @param {Array<number>} shape - Image dimensions (e.g., [10, 512, 768]).
 * @param {Object} axes - Which dimensions to treat as Y and X.
 * @param {number} axes.axisY - Index of the Y axis in the shape array.
 * @param {number} axes.axisX - Index of the X axis in the shape array.
 * @param {Array<number>} sliceIndices - Index to use for each non-display axis
 *        (length must equal shape.length - 2, in dimension order excluding axisY and axisX).
 * @returns {Float32Array} The extracted 2-D slice of length shape[axisY] * shape[axisX].
 */
export function pickSlice(ndarray, shape, { axisY, axisX }, sliceIndices) {
    const height = shape[axisY];
    const width  = shape[axisX];
    const result = new Float32Array(height * width);

    // Build stride table (row-major / C order)
    const strides = new Array(shape.length);
    strides[shape.length - 1] = 1;
    for (let i = shape.length - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    // Map non-display axes → their fixed slice index
    const fixedAxes = [];
    let sliceIdx = 0;
    for (let dim = 0; dim < shape.length; dim++) {
        if (dim !== axisY && dim !== axisX) {
            fixedAxes.push({ dim, idx: sliceIndices[sliceIdx++] });
        }
    }

    // Compute base offset from fixed axes
    let baseOffset = 0;
    for (const { dim, idx } of fixedAxes) {
        baseOffset += idx * strides[dim];
    }

    // Fill result
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const srcIdx = baseOffset + y * strides[axisY] + x * strides[axisX];
            result[y * width + x] = ndarray[srcIdx];
        }
    }

    return result;
}
