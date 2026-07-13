import { MIN_LABELS_TO_TRAIN, NUM_FEATURES, RF_CONFIG, TRAIN_DEBOUNCE_MS, STATS_LAYOUT } from './config.js';
import { setCentroids, renderCentroids, clearCentroids } from './images.js';

// Dispatches a CustomEvent on the owning Quantosaurus instance (state.events).
// Optional-chained so state objects without a dispatch target (unit tests,
// plain-object fixtures) keep working.
function emit(state, name, detail) {
    state.events?.dispatchEvent(new CustomEvent(name, { detail }));
}

// Fields per object in the dense stats struct backends return from downloadStats
// (see STATS_LAYOUT): label, area, total_intensity {lo,hi}, sum_x {lo,hi},
// sum_y {lo,hi}, min_intensity, max_intensity.
const STATS_FIELDS_PER_OBJECT = STATS_LAYOUT.denseCount;

/**
 * Decodes a dense stats buffer (downloadStats) into per-object centroids + areas.
 * The 64-bit summed x/y are split across lo/hi u32 words (see STATS_LAYOUT); we
 * reassemble them and divide by area for the centroid. Centroids are pixel
 * coordinates and need no range.scale descaling (only intensity fields do).
 * @param {Uint32Array} stats - Flat dense stats, STATS_FIELDS_PER_OBJECT u32 per object.
 * @returns {Array<{cx: number, cy: number, area: number}>}
 */
export function decodeObjects(stats) {
    const n = STATS_FIELDS_PER_OBJECT;
    const u64 = (lo, hi) => hi * 2 ** 32 + lo;
    const out = [];
    for (let i = 0; i < stats.length; i += n) {
        const area = stats[i + 1];
        if (area === 0) continue;
        out.push({
            cx: u64(stats[i + 4], stats[i + 5]) / area,
            cy: u64(stats[i + 6], stats[i + 7]) / area,
            area,
        });
    }
    return out;
}

/**
 * Aggregates features and labels across all images into 1D typed arrays for Random Forest training.
 * @param {Array<Object>} images - Array of image objects containing labels and a backend to gather features.
 * @param {number} totalLabels - The total number of labels across all images.
 * @returns {Promise<{combinedX: Float32Array, yArray: Int32Array}>} An object containing the concatenated features (combinedX) and their corresponding labels (yArray).
 */
async function buildTrainingDataset(images, totalLabels) {
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

let _trainTimer = null;
/**
 * Debounced trigger for a full retrain. Coalesces rapid label edits into a
 * single trainAndPredictAll call after TRAIN_DEBOUNCE_MS of quiet.
 * @param {Object} state - Shared app state.
 */
export function scheduleTraining(state) {
    clearTimeout(_trainTimer);
    _trainTimer = setTimeout(() => trainAndPredictAll(state), TRAIN_DEBOUNCE_MS);
}

/**
 * Retrains the forest on all current labels, then runs inference on every image
 * and refreshes the per-class object counts. No-ops (emitting zeroed stats) if
 * there are fewer than MIN_LABELS_TO_TRAIN labels total. Emits trainingstart /
 * trainingcomplete around the run so chrome can show progress affordances.
 * @param {Object} state - Shared app state; reads state.images and mutates state.rf.
 */
export async function trainAndPredictAll(state) {
    const totalLabels = state.images.reduce((s, i) => s + i.labels.length, 0);
    if (totalLabels < MIN_LABELS_TO_TRAIN) {
        const zeros = new Array(RF_CONFIG.numClasses).fill(0);
        state.lastCounts = zeros;
        state.images.forEach(clearCentroids);
        emit(state, 'statscomputed', {
            counts: zeros, perImage: [], animate: false, reason: 'insufficient-labels',
        });
        return;
    }

    state.isTraining = true;
    emit(state, 'trainingstart', {});

    let trained = false;
    try {
        const { combinedX, yArray } = await buildTrainingDataset(state.images, totalLabels);
        state.rf.train(combinedX, yArray, NUM_FEATURES);

        for (const img of state.images) {
            await img.backend.runInference(state.rf);
        }

        await updateObjectCounts(state);
        trained = true;
    } finally {
        state.isTraining = false;
        emit(state, 'trainingcomplete', { trained });
    }
}

/**
 * Runs connected-component labeling + stats per class, per image, caches the
 * results on state, and emits statscomputed with the summed per-class counts
 * plus each image's detected objects. Each object is one dense stats struct,
 * so the count is stats.length / STATS_FIELDS_PER_OBJECT.
 * @param {Object} state - Shared app state.
 */
async function updateObjectCounts(state) {
    const numClasses = RF_CONFIG.numClasses;
    const counts = new Array(numClasses).fill(0);
    const perImage = [];

    for (const img of state.images) {
        const objectsByClass = Array.from({ length: numClasses }, () => []);
        for (let cls = 0; cls < numClasses; cls++) {
            await img.backend.computeConnectedComponents(cls);
            const statsResult = await img.backend.computeStats();
            // WebGpuBackend.computeStats returns null (and leaves its stats
            // buffers untouched) when this class had no pixels above threshold.
            // Skip the download in that case, or we'd read the previous class's
            // stale counts back out.
            if (statsResult === null) continue;

            const stats = await img.backend.downloadStats();
            const objects = decodeObjects(stats);
            objectsByClass[cls] = objects;
            counts[cls] += objects.length;
        }
        // Cache and repaint this image's centroid markers from the freshly detected objects.
        setCentroids(img, objectsByClass);
        renderCentroids(state, img);
        perImage.push({ id: img.id, objectsByClass });
    }

    state.lastCounts = counts;
    emit(state, 'statscomputed', { counts, perImage, animate: true });
}
