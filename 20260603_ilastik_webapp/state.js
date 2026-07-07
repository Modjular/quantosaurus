import { FlatRandomForest } from './rf.js';

export const LABEL_COLORS = [
    '#ff595e',
    '#ffca3a',
    '#8ac926',
    '#1982c4',

];
export const RF_CONFIG = { numTrees: 8, maxDepth: 8, numClasses: LABEL_COLORS.length };
export const MIN_LABELS_TO_TRAIN = 5;
export const CAMERA_ZOOM_MIN = 0.1;
export const CAMERA_ZOOM_MAX = 10;
export const CAMERA_ZOOM_SENSITIVITY = 0.01;
export const TRAIN_DEBOUNCE_MS = 300;

export const state = {
    // Images — ordered array; visual order == export order
    images: [],

    // ML
    rf: new FlatRandomForest(RF_CONFIG),

    // Drawing / Tools
    currentClass: 0,
    isDrawing: false,
    activeImageId: null,
    toolMode: 'grab', // 'grab', 'paint', 'erase'

    // Features
    sigma: 1.0,

    // Live update
    liveUpdate: true,

    // Camera
    camera: { x: 0, y: 0, scale: 1 },
    isSpaceDown: false,
    isPanning: false,
};
