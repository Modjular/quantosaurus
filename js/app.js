// Shared app bootstrap for every Quantosaurus app. Each page builds its state
// from createBaseState (adding its own method-specific fields), then calls
// initApp to wire up the chrome common to all three apps.
//
// Deliberately does NOT wire method-specific UI (paint tools, threshold slider,
// cellpose params) or classifier-only concerns like loadSettings/.ilp save —
// each page owns those. Keeping this module method-agnostic is what lets the
// three apps share one bootstrap without dragging the random forest into the
// threshold and cellpose pages.
import { DEFAULT_LABEL_COLORS } from './config.js';
import { setupCamera } from './camera.js';
import { setupTheme, setupFeedback, setupCheatsheet, setupFileIngest, setupNav } from './chrome.js';

/**
 * Builds the state fields every app shares (images, camera, tool/pointer flags,
 * overlay colors, brush size). Method-specific fields and behavior hooks
 * (onLabelsChanged, onImagesChanged, computeFeatures, saveIndicator, rf, …) are
 * supplied by the caller via `overrides`.
 * @param {Object} [overrides] - App-specific state merged over the base.
 * @returns {Object} The initial shared app state.
 */
export function createBaseState(overrides = {}) {
    return {
        // Images — ordered array; visual order == export order.
        images: [],

        // Per-class overlay colors, seeded from config defaults. Used by every
        // backend's composite pass; the classifier lets users recolor/persist
        // these, the other apps just use the defaults (class 0 = foreground).
        labelColors: [...DEFAULT_LABEL_COLORS],

        // Drawing / tools.
        currentClass: 0,
        isDrawing: false,
        activeImageId: null,
        toolMode: 'grab', // 'grab', 'paint', 'erase'
        brushSize: 5,

        // Camera.
        camera: { x: 0, y: 0, scale: 1 },
        isSpaceDown: false,
        isPanning: false,
        activePointerCount: 0,

        // Save/dirty tracking. Only the classifier acts on `dirty` (its .ilp save
        // flow); the others still set it harmlessly. `saveIndicator` gates the
        // unsaved-changes title/badge (see ui.js:updateSaveIndicator).
        dirty: false,

        ...overrides,
    };
}

/**
 * Wires the chrome shared by every app: camera, theme, feedback, cheatsheet,
 * file ingest, top-bar nav, and the unsaved-changes guard. App-specific wiring
 * (tools, method controls, export) is the caller's responsibility.
 * @param {Object} state - Shared app state.
 * @param {Object} [opts]
 * @param {string} [opts.page] - Page id ('classifier' | 'threshold' | 'cellpose')
 *   for nav highlight and the cheatsheet's per-page first-visit flag.
 */
export function initApp(state, { page } = {}) {
    const viewport = document.getElementById('viewport');
    const board    = document.getElementById('canvas-board');
    setupCamera(state, viewport, board);

    setupTheme();
    setupFeedback(state);
    setupCheatsheet(page);
    setupFileIngest(state);
    setupNav(page);

    // Guard against accidentally discarding work (loaded images + any labels,
    // none of which is persisted). state.dirty tracks unsaved changes precisely.
    window.addEventListener('beforeunload', (e) => {
        if (state.dirty) {
            e.preventDefault();
            e.returnValue = '';
        }
    });
}
