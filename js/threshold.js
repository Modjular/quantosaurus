// Threshold segmentation: Otsu auto-threshold + manual override, feeding the
// same GPU pipeline (probability buffer -> composite/CCL/stats) the classifier
// uses. computeOtsu is a pure CPU function (unit-tested in threshold.test.mjs);
// the rest orchestrates the shared backend/UI the same way training.js does for
// the classifier, but with one foreground class and a single headline count.
//
// Foreground convention: bright pixels (>= threshold) are objects, matching
// fluorescence/DAPI. `invert` flips this for brightfield (dark objects).
import { NUM_CLASSES } from './config.js';
import { setCentroids, renderCentroids, clearCentroids } from './images.js';
import { decodeObjectStats } from './objects.js';
import { animateCount, setClassBadgesLoading } from './ui.js';

const OTSU_BINS = 256;
// Class 0 is the foreground ("objects") channel; the composite overlays it.
const FOREGROUND_CLASS = 0;

/**
 * Otsu's method: the intensity threshold maximizing between-class variance of a
 * 256-bin histogram over [range.dataMin, range.dataMax]. Pure CPU. Returns the
 * threshold in the image's raw intensity units (pixels >= it are foreground).
 * @param {Float32Array|number[]} intensityArray - Raw pixel intensities.
 * @param {{dataMin:number, dataMax:number}} range - Image data range.
 * @returns {number} Threshold in raw intensity units.
 */
export function computeOtsu(intensityArray, range) {
    const lo = range.dataMin, hi = range.dataMax;
    const span = hi - lo;
    if (!(span > 0)) return lo; // constant image: nothing to separate

    const hist = new Float64Array(OTSU_BINS);
    const n = intensityArray.length;
    for (let i = 0; i < n; i++) {
        let b = Math.floor(((intensityArray[i] - lo) / span) * OTSU_BINS);
        if (b < 0) b = 0; else if (b >= OTSU_BINS) b = OTSU_BINS - 1;
        hist[b]++;
    }

    let sumAll = 0;
    for (let b = 0; b < OTSU_BINS; b++) sumAll += b * hist[b];

    let wB = 0, sumB = 0, maxVar = -1, threshBin = 0;
    for (let b = 0; b < OTSU_BINS; b++) {
        wB += hist[b];
        if (wB === 0) continue;
        const wF = n - wB;
        if (wF === 0) break;
        sumB += b * hist[b];
        const mB = sumB / wB;
        const mF = (sumAll - sumB) / wF;
        const between = wB * wF * (mB - mF) * (mB - mF);
        if (between > maxVar) { maxVar = between; threshBin = b; }
    }

    // The split sits between threshBin and threshBin+1; map that upper edge back
    // to raw units so "intensity >= threshold" reproduces the histogram split.
    return lo + ((threshBin + 1) / OTSU_BINS) * span;
}

/**
 * Resolve the raw-unit threshold for one image from the app's threshold state:
 * per-image Otsu in Auto mode, or the one global normalized [0,1] slider value
 * mapped through this image's own data range in Manual mode. (The method is
 * global but ranges are per-image, so Manual maps per image.)
 * @param {Object} img - Image state entry.
 * @param {Object} threshold - state.threshold {auto, value, invert}.
 * @returns {number} Threshold in raw intensity units.
 */
export function thresholdForImage(img, threshold) {
    if (threshold.auto) return computeOtsu(img.intensityArray, img.range);
    return img.range.dataMin + threshold.value * (img.range.dataMax - img.range.dataMin);
}

/**
 * Write a binary foreground/background probability map for one image and repaint.
 * Class 0 gets 1.0 where the pixel is foreground, and every channel is the -1.0
 * "no overlay" sentinel elsewhere so the composite leaves background untinted.
 * @param {Object} img - Image state entry.
 * @param {number} thresholdRaw - Threshold in raw units.
 * @param {boolean} [invert=false] - Treat dark pixels (<= threshold) as foreground.
 */
export function applyThreshold(img, thresholdRaw, invert = false) {
    const n = img.width * img.height;
    const probs = new Float32Array(n * NUM_CLASSES).fill(-1.0);
    const arr = img.intensityArray;
    for (let i = 0; i < n; i++) {
        const fg = invert ? arr[i] <= thresholdRaw : arr[i] >= thresholdRaw;
        if (fg) probs[i * NUM_CLASSES + FOREGROUND_CLASS] = 1.0;
    }
    img.backend.setProbabilities(probs);
}

/**
 * Applies the current threshold to every loaded image, then recomputes the
 * single object count (connected components + stats on the foreground class) and
 * repaints per-object centroid markers. This is Threshold's analogue of
 * training.js:trainAndPredictAll — same shared backend calls, one class.
 * @param {Object} state - Shared app state (reads state.images, state.threshold).
 */
export async function runThreshold(state) {
    if (state.images.length === 0) {
        animateCount(0);
        return;
    }
    setClassBadgesLoading();

    for (const img of state.images) {
        applyThreshold(img, thresholdForImage(img, state.threshold), state.threshold.invert);
    }
    await countObjects(state);
}

/**
 * Connected-component count of the foreground class across all images, feeding
 * the single-count readout and the centroid overlay.
 * @param {Object} state - Shared app state.
 */
async function countObjects(state) {
    let total = 0;
    for (const img of state.images) {
        await img.backend.computeConnectedComponents(FOREGROUND_CLASS);
        const statsResult = await img.backend.computeStats();
        const objectsByClass = Array.from({ length: NUM_CLASSES }, () => []);
        // WebGpuBackend.computeStats returns null when the class is empty; skip the
        // download then, or we'd read stale stats back (see training.js).
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
}
