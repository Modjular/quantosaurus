import { readImage } from "https://cdn.jsdelivr.net/npm/@itk-wasm/image-io@1.6.0/dist/bundle/index-worker-embedded.min.js";


/**
 * Loads an image file into a typed array, handling both TIFFs and standard web formats.
 * @param {File} file - The image file to load.
 * @returns {Promise<{intensityArray: Float32Array, rgba: Uint8Array|Uint8ClampedArray, w: number, h: number}>} An object containing the normalized intensity array, the raw RGBA array, and the dimensions.
 */
export async function loadFileIntoArray(file) {
  let data, rgba, w, h, shape;

  if (file.name.endsWith('.tif') || file.name.endsWith('.tiff')) {
    const { image } = await readImage(file)

    w = image.size[0]
    h = image.size[1]
    rgba = new Uint8Array(image.data.length * 4)
    data = image.data
    shape = image.size
  } else {
    const img = await createImageBitmap(file);
    w = img.width;
    h = img.height;
    const off = new OffscreenCanvas(w, h);
    const ctx = off.getContext('2d');
    ctx.drawImage(img, 0, 0);
    rgba = ctx.getImageData(0, 0, w, h).data;
    data = new Float32Array(w * h)
    shape = [w, h]

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

  return { intensityArray, rgba, w, h, shape}
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
