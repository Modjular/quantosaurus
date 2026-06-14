import { readImage, writeImage } from "https://cdn.jsdelivr.net/npm/@itk-wasm/image-io@1.6.0/dist/bundle/index-worker-embedded.min.js";


export async function loadFileIntoArray(file) {
  let data, rgba, w, h;

  if (file.name.endsWith('.tif') || file.name.endsWith('.tiff')) {
    const buffer = await file.arrayBuffer();
    const { image } = await readImage(file)

    w = image.size[0]
    h = image.size[1]
    rgba = new Int8Array(image.data.length * 4)
    data = image.data
  } else {
    const img = await createImageBitmap(file);
    w = img.width;
    h = img.height;
    const off = new OffscreenCanvas(w, h);
    const ctx = off.getContext('2d');
    ctx.drawImage(img, 0, 0);
    rgba = ctx.getImageData(0, 0, w, h).data;

    // Convert RGBA to intensity
    for (let i = 0; i < w * h; i++) {
        const r = rgba[i * 4] / 255;
        const g = rgba[i * 4 + 1] / 255;
        const b = rgba[i * 4 + 2] / 255;
        data[i] = 0.299 * r + 0.587 * g + 0.114 * b;
    }
  }

  const intensityArray = new Float32Array(w * h);

  // Calculate min/max for normalization
  let [min, max] = [Infinity, -Infinity]
  for (let i = 0; i < w * h; i++) {
    if (data[i] < min) min = data[i];
    if (data[i] > max) max = data[i];
  }

  const range = (max - min) > 0 ? (max - min) : 255;
  for (let i = 0; i < w * h; i++) {
    const norm = (data[i] - min) / range;
    intensityArray[i] = norm;

    const val8 = norm * 255;
    rgba[i * 4] = val8;
    rgba[i * 4 + 1] = val8;
    rgba[i * 4 + 2] = val8;
    rgba[i * 4 + 3] = 255;
  }

  return { intensityArray, rgba, w, h }
}

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

export async function writeFile(folderHandle, filename, image) {
  const { serializedImage } = await writeImage(image, `${image.name}.tif`);
  const blob = new Blob([serializedImage.data], { type: 'image/tiff' });
  const fileHandle = await folderHandle.getFileHandle(filename, { create: true });
  const writable = await fileHandle.createWritable();
  await writable.write(blob);
  await writable.close();
}

/**
 * Aggregates features and labels across all images into 1D typed arrays for Random Forest training.
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
 * Handles generating ITK images, requesting file permissions, and batching files into ZIPs or directories.
 */
export async function exportImagesData(images, rf, options) {
    const { exportSeg, exportProb, outputDirHandle, verifyPermission, writeFile } = options;
    
    let zip = null;
    if (!outputDirHandle) {
        const JSZip = (await import('https://esm.sh/jszip@3.10.1')).default;
        zip = new JSZip();
    } else {
        const permission = await verifyPermission(outputDirHandle, true);
        if (!permission) {
            throw new Error("Permission to write to output directory was denied.");
        }
    }

    for (let i = 0; i < images.length; i++) {
        const img = images[i];
        
        // Run inference to ensure textures/buffers are updated
        await img.backend.runInference(rf);

        const probs = await img.backend.downloadProbabilities();
        const w = img.width;
        const h = img.height;
        const baseName = img.name ? img.name.replace(/\.[^/.]+$/, "") : `image_${i}`;

        if (exportSeg) {
            const dataUint8Array = new Uint8Array(w * h);
            for (let j = 0; j < probs.length; j++) {
                const c = probs[j];
                dataUint8Array[j] = c < 0 ? 0 : c + 1;
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
            let blob; // Assuming createTiffBlob will be implemented or imported here eventually
            
            if (outputDirHandle) {
                await writeFile(outputDirHandle, filename, itkImage);
            } else {
                zip.file(filename, blob || itkImage.data);
            }
        }

        if (exportProb) {
            const numClasses = rf.numClasses || 2;
            const dataFloat32Array = new Float32Array(w * h * numClasses);
            for (let j = 0; j < probs.length; j++) {
                const c = probs[j];
                if (c >= 0 && c < numClasses) {
                    dataFloat32Array[j * numClasses + c] = 1.0;
                }
            }

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
            let blob; // Assuming createTiffBlob will be implemented or imported here eventually
            
            if (outputDirHandle) {
                await writeFile(outputDirHandle, filename, itkImage);
            } else {
                zip.file(filename, blob || itkImage.data);
            }
        }
    }

    if (zip) {
        const content = await zip.generateAsync({ type: "blob" });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(content);
        link.download = "ilastik_export.zip";
        link.click();
        URL.revokeObjectURL(link.href);
    }
}

// ---------------------------------------------------------------------------
// inferAxes(shape)
//
// Given a multi-dimensional image shape array (e.g. [10, 512, 768] for a
// Z-stack), return the two axis indices that should be used as the display
// XY plane.
//
// Strategy: prefer the last two dimensions (NumPy / TIFF channel-first
// convention: [Z, Y, X] or [C, Y, X]), which is almost always correct.
// The "two largest" heuristic would mis-select a large Z-stack as X/Y, so
// we intentionally avoid it.
//
// For 2-D images (shape.length === 2) this returns [0, 1] and callers should
// hide the axis selector entirely — no-op selection.
//
// Returns: { axisY: number, axisX: number }
// ---------------------------------------------------------------------------
export function inferAxes(shape) {
    if (!Array.isArray(shape) || shape.length < 2) {
        throw new Error(`inferAxes: shape must have at least 2 dimensions, got ${JSON.stringify(shape)}`);
    }
    const n = shape.length;
    return { axisY: n - 2, axisX: n - 1 };
}

// ---------------------------------------------------------------------------
// pickSlice(ndarray, shape, axes, sliceIndices)
//
// Extract a 2-D slice from a flat typed array representing an n-D image.
//
// Parameters:
//   ndarray      – flat Float32Array (or similar) of the full image data
//   shape        – Array<number>  e.g. [10, 512, 768]
//   axes         – { axisY, axisX }  which dims to treat as Y and X
//   sliceIndices – Array<number>  index to use for each non-display axis
//                  (length must equal shape.length - 2, in dimension order
//                  excluding axisY and axisX)
//
// Returns a Float32Array of length shape[axisY] * shape[axisX].
//
// This is intentionally simple: it iterates in JS rather than using GPU
// tricks. For large stacks the caller should call this once at slice-select
// time and cache the result, not on every frame.
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// isDuplicateFile(file, existingImages)
//
// Returns true if a file with the same name and byte-size is already in the
// loaded image list. Used to silently deduplicate drag-and-drop re-drops.
// ---------------------------------------------------------------------------
export function isDuplicateFile(file, existingImages) {
    return existingImages.some(
        img => img.name === file.name && img.fileSize === file.size
    );
}