import { MIN_LABELS_TO_TRAIN, RF_CONFIG, TRAIN_DEBOUNCE_MS } from './state.js';

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
export function scheduleTraining(state) {
    clearTimeout(_trainTimer);
    _trainTimer = setTimeout(() => trainAndPredictAll(state), TRAIN_DEBOUNCE_MS);
}

export async function trainAndPredictAll(state) {
    const totalLabels = state.images.reduce((s, i) => s + i.labels.length, 0);
    if (totalLabels < MIN_LABELS_TO_TRAIN) return;

    const { combinedX, yArray } = await buildTrainingDataset(state.images, totalLabels);
    state.rf.train(combinedX, yArray, RF_CONFIG.numTrees);

    for (const img of state.images) {
        await img.backend.runInference(state.rf);
    }
}
