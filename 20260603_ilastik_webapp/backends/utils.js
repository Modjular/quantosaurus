import { readImage, writeImage } from "https://cdn.jsdelivr.net/npm/@itk-wasm/image-io@1.6.0/dist/bundle/index-worker-embedded.min.js";

/**
 * Official ilastik feature identifiers in the order they are concatenated
 * in the output buffer for a single scale.
 */
export const FEATURE_IDS = [
  "GaussianSmoothing",
  "LaplacianOfGaussian",
  "GaussianGradientMagnitude",
  "DifferenceOfGaussians",
  "StructureTensorEigenvalues",   // 2 channels (Largest, Smallest)
  "HessianOfGaussianEigenvalues"  // 2 channels (Largest, Smallest)
];

/**
 * Generates a 1D Gaussian kernel or its analytical derivatives.
 * Replicates Vigra's dynamic radius and strict normalization requirements.
 * @param {number} scale - Sigma value (e.g. 0.3, 1.0, 3.5).
 * @param {number} order - 0 (Smoothing), 1 (1st Derivative), or 2 (2nd Derivative).
 * @returns {Float32Array} The right-half of the symmetric/anti-symmetric kernel [0...radius].
 */
export function gaussian_kernel(scale, order = 0) {
  if (scale <= 0) throw new Error("scale should be greater than 0");

  // Vigra dynamic radius: ensures energy isn't truncated at high orders/scales
  const radius = Math.ceil((3.0 + 0.5 * order) * scale);
  const kernel = new Float32Array(radius + 1);
  const twoSigmaSq = 2.0 * scale * scale;
  const fullSize = 2 * radius + 1;
  const fullKernel = new Float32Array(fullSize);

  // 1. Calculate raw analytical values
  let sum = 0;
  for (let i = -radius; i <= radius; i++) {
    const x = i;
    let val = Math.exp(-(x * x) / twoSigmaSq);
    if (order === 1) val = (-x / (scale * scale)) * val;
    else if (order === 2) val = (((x * x) / (scale * scale * scale * scale)) - (1.0 / (scale * scale))) * val;
    fullKernel[i + radius] = val;
    if (order === 0) sum += val;
  }

  // 2. Strict Normalization (required for mathematical identicality with Vigra)
  if (order === 0) {
    // Smoothing sums to 1.0
    for (let i = 0; i < fullSize; i++) fullKernel[i] /= sum;
  }
  else if (order === 1) {
    // 1st Derivative: 1st moment must be exactly 1.0 (forces correlation behavior)
    let sumX = 0;
    for (let i = -radius; i <= radius; i++) sumX += i * fullKernel[i + radius];
    for (let i = 0; i < fullSize; i++) fullKernel[i] /= sumX;
  }
  else if (order === 2) {
    // 2nd Derivative: strict zero-mean and variance normalization
    let sum0 = 0;
    for (let i = -radius; i <= radius; i++) sum0 += fullKernel[i + radius];
    const mean = sum0 / fullSize;
    for (let i = 0; i < fullSize; i++) fullKernel[i] -= mean;
    let sumX2 = 0;
    for (let i = -radius; i <= radius; i++) sumX2 += 0.5 * i * i * fullKernel[i + radius];
    for (let i = 0; i < fullSize; i++) fullKernel[i] /= sumX2;
  }

  // Return the half-kernel to save GPU uniform space
  for (let i = 0; i <= radius; i++) kernel[i] = fullKernel[radius + i];
  return kernel;
}

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

/**
 * Filters out labels that fall within the given eraser radius (in-place optimized).
 */
export function eraseLabelsInRadius(labels, x, y, radius) {
    const r2 = radius * radius;
    let writeIdx = 0;
    
    for (let i = 0; i < labels.length; i++) {
        const l = labels[i];
        const dx = l.x - x;
        const dy = l.y - y;
        
        // If the point is OUTSIDE the brush radius, keep it
        if ((dx * dx + dy * dy) > r2) {
            labels[writeIdx++] = l;
        }
    }
    
    labels.length = writeIdx; // Truncate the array
    return labels;
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
                dataUint8Array[j] = probs[j] >= 0.5 ? 255 : 0;
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
            const dataFloat32Array = new Float32Array(w * h * 2);
            for (let j = 0; j < probs.length; j++) {
                const p = Math.max(0, Math.min(1, probs[j]));
                dataFloat32Array[j * 2] = p;
                dataFloat32Array[j * 2 + 1] = 1.0 - p;
            }

            const itkImage = {
                imageType: {
                    dimension: 2,
                    pixelType: 'Vector',
                    componentType: 'float32',
                    components: 2
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
