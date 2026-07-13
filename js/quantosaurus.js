// The embeddable app core: one Quantosaurus instance owns the shared `state`
// object (images, forest, tools, camera) and a pan/zoom board element, and is
// itself an EventTarget — every state change the app makes flows out as a
// CustomEvent (see the event catalog below), so any chrome (the built-in
// taskbar, or a host page embedding the board in a notebook) can subscribe
// instead of being hard-wired in.
//
// This class and the GPU backends are the deliberate exceptions to the
// "functions take state explicitly" convention (see CLAUDE.md): the modules it
// orchestrates (images.js, training.js, camera.js, contrast.js, settings.js,
// export.js) keep their fn(state, ...) signatures and know nothing about the
// class beyond the optional `state.events` dispatch target.
//
// Event catalog (all CustomEvents on the instance; payload in event.detail):
//   imageloadstart          { id, name }
//   imageadded              { id, name, width, height, index }
//   imageloaderror          { id, name, reason, message? }   reason: 'load-failed' | 'unsupported-dimensions' | 'no-backend'
//   imageremoved            { id, index }
//   imagereordered          { id, fromIndex, toIndex }
//   labelschanged           { id, labelCount, totalLabels }
//   dirtychange             { dirty }
//   toolchanged             { tool }
//   brushsizechanged        { size }
//   sigmachanged            { sigma }
//   classchanged            { index }
//   classcolorchanged       { index, color }
//   classnamechanged        { index, name }
//   markersvisibilitychanged{ index, visible }
//   trainingstart           { }
//   trainingcomplete        { trained }
//   statscomputed           { counts, perImage, animate, reason? }

import { DEFAULT_LABEL_COLORS, RF_CONFIG, BRUSH_SIZE_DEFAULT, clampBrushSize } from './config.js';
import { FlatRandomForest } from './rf.js';
import { addFiles, reorderImage, deleteImage, redrawLabels, renderCentroids } from './images.js';
import { scheduleTraining } from './training.js';
import { setupCamera } from './camera.js';
import { openContrastPopover, closeContrastPopover } from './contrast.js';
import { loadSettings, saveSettings } from './settings.js';
import { zipImages, exportIlp } from './export.js';

const TOOLS = ['grab', 'paint', 'erase'];

export class Quantosaurus extends EventTarget {
    /**
     * Attaches a new app instance to a board element. The board is the
     * pan/zoom transform target that image tiles are appended to; the viewport
     * is the clipping element that receives wheel/pointer gestures (camera.js).
     * @param {HTMLElement} boardEl - Element the image tiles live in (transformed by the camera).
     * @param {{viewportEl?: HTMLElement}} [options] - Viewport element; defaults to the board's parent.
     */
    constructor(boardEl, { viewportEl = boardEl.parentElement } = {}) {
        super();

        /**
         * The shared mutable app state threaded through every module. Exposed
         * so existing fn(state, ...) modules and chrome can keep working with
         * it directly; embedders should prefer the methods/events on this class.
         */
        this.state = {
            // Images — ordered array; visual order == export order
            images: [],

            // ML
            rf: new FlatRandomForest(RF_CONFIG),

            // Presentation — per-class overlay colors, indexed by class. Owned by the
            // UI (mutable), seeded from config defaults. The class *count* comes from
            // rf.numClasses; colors are a separate concern that can change at runtime.
            labelColors: [...DEFAULT_LABEL_COLORS],

            // Per-class display names, indexed by class. Like labelColors, this is a
            // UI-owned presentation concern (renamable at runtime, persisted to
            // localStorage); the class *count* still comes from rf.numClasses.
            classNames: Array.from({ length: RF_CONFIG.numClasses }, (_, i) => `Class ${i + 1}`),

            // Whether each class's centroid markers are drawn on canvas. Session-only —
            // like camera/labels, not persisted.
            classMarkersVisible: new Array(RF_CONFIG.numClasses).fill(true),

            // Drawing / Tools
            currentClass: 0,
            isDrawing: false,
            activeImageId: null,
            toolMode: 'grab', // 'grab', 'paint', 'erase'
            brushSize: BRUSH_SIZE_DEFAULT,

            // Features
            sigma: 1.0,

            // Camera
            camera: { x: 0, y: 0, scale: 1 },
            isSpaceDown: false,
            isPanning: false,
            activePointerCount: 0,

            // True while a debounced retrain+inference is running.
            isTraining: false,

            // Whether there are changes since the last successful save (set on
            // image/label mutations in images.js, cleared by markSaved()).
            dirty: false,

            // Last per-class object counts from the stats pass (see statscomputed).
            lastCounts: new Array(RF_CONFIG.numClasses).fill(0),

            // DOM the instance owns / dispatch target for module-emitted events.
            board: boardEl,
            viewport: viewportEl,
            events: this,
        };

        loadSettings(this.state); // restore persisted colors/names before any chrome renders
        if (viewportEl) setupCamera(this.state, viewportEl, boardEl);
    }

    /** @returns {'grab'|'paint'|'erase'} The active tool. */
    get tool() { return this.state.toolMode; }

    /** @returns {number} The active brush radius in image pixels. */
    get brushSize() { return this.state.brushSize; }

    /** @returns {number} The active class index. */
    get currentClass() { return this.state.currentClass; }

    /** @returns {number} The current feature scale (sigma). */
    get sigma() { return this.state.sigma; }

    /** @returns {boolean} Whether there are unsaved changes. */
    get dirty() { return this.state.dirty; }

    /** @returns {number} How many classes exist (fixed — see NUM_CLASSES in config.js). */
    get numClasses() { return this.state.rf.numClasses; }

    /** Dispatches a CustomEvent on this instance. @param {string} name @param {Object} [detail] */
    _emit(name, detail = {}) {
        this.dispatchEvent(new CustomEvent(name, { detail }));
    }

    // ---------------------------------------------------------------- images

    /**
     * Imports a batch of files onto the board, skipping unsupported types and
     * duplicates. Emits imageloadstart/imageadded/imageloaderror per file.
     * @param {Iterable<File>} files
     */
    async addFiles(files) {
        await addFiles(this.state, files);
    }

    /**
     * Moves an image one slot up or down in board/export order.
     * @param {string} imgId @param {-1|1} direction
     */
    reorderImage(imgId, direction) {
        reorderImage(this.state, imgId, direction);
    }

    /**
     * Removes an image and frees its GPU resources (confirms first if it
     * carries labels). Emits imageremoved.
     * @param {string} imgId
     */
    deleteImage(imgId) {
        deleteImage(this.state, imgId);
    }

    /**
     * Lightweight descriptors of the loaded images, in board/export order.
     * @returns {Array<{id: string, name: string, width: number, height: number, labelCount: number}>}
     */
    getImageInfo() {
        return this.state.images.map(({ id, name, width, height, labels }) =>
            ({ id, name, width, height, labelCount: labels.length }));
    }

    // ----------------------------------------------------------------- tools

    /**
     * Switches the active tool and updates the body cursor class.
     * @param {'grab'|'paint'|'erase'} tool
     */
    setTool(tool) {
        if (!TOOLS.includes(tool) || tool === this.state.toolMode) return;
        this.state.toolMode = tool;
        document.body.classList.remove(...TOOLS.map(t => `mode-${t}`));
        document.body.classList.add(`mode-${tool}`);
        this._emit('toolchanged', { tool });
    }

    /**
     * Sets the brush radius, clamped to the valid range (see config.js).
     * @param {number} size
     */
    setBrushSize(size) {
        const clamped = clampBrushSize(size);
        if (clamped === this.state.brushSize) return;
        this.state.brushSize = clamped;
        this._emit('brushsizechanged', { size: clamped });
    }

    /**
     * Selects the class that painting assigns.
     * @param {number} index
     */
    setClass(index) {
        if (!Number.isInteger(index) || index < 0 || index >= this.numClasses) return;
        if (index === this.state.currentClass) return;
        this.state.currentClass = index;
        this._emit('classchanged', { index });
    }

    /**
     * Sets the feature scale, recomputes every image's filter bank at the new
     * sigma, and schedules a retrain. Resolves once features are updated.
     * @param {number} sigma
     */
    async setSigma(sigma) {
        this.state.sigma = sigma;
        this._emit('sigmachanged', { sigma });
        for (const img of this.state.images) {
            await img.backend.updateFeatures(img.intensityArray, sigma);
        }
        scheduleTraining(this.state);
    }

    // --------------------------------------------------- class presentation

    /**
     * Recolors a class's overlay everywhere (composite, painted labels,
     * centroid markers) and persists the palette.
     * @param {number} index @param {string} color - #rrggbb hex.
     */
    setClassColor(index, color) {
        if (index < 0 || index >= this.numClasses) return;
        this.state.labelColors[index] = color;
        this.state.images.forEach(img => {
            img.backend.setColors(this.state.labelColors);
            redrawLabels(this.state, img);
            renderCentroids(this.state, img);
        });
        saveSettings(this.state);
        this._emit('classcolorchanged', { index, color });
    }

    /**
     * Renames a class and persists the name. Empty/whitespace names fall back
     * to the default so a blank label is never rendered or persisted.
     * @param {number} index @param {string} name
     * @returns {string|undefined} The normalized name actually stored.
     */
    setClassName(index, name) {
        if (index < 0 || index >= this.numClasses) return undefined;
        const normalized = String(name ?? '').trim() || `Class ${index + 1}`;
        this.state.classNames[index] = normalized;
        saveSettings(this.state);
        this._emit('classnamechanged', { index, name: normalized });
        return normalized;
    }

    /**
     * Shows/hides a class's on-canvas centroid markers. Cheap: repaints the 2D
     * overlay from cached objects, no GPU or stats work.
     * @param {number} index @param {boolean} visible
     */
    setMarkersVisible(index, visible) {
        if (index < 0 || index >= this.numClasses) return;
        this.state.classMarkersVisible[index] = !!visible;
        this.state.images.forEach(img => renderCentroids(this.state, img));
        this._emit('markersvisibilitychanged', { index, visible: !!visible });
    }

    /** @returns {string[]} A copy of the per-class overlay colors. */
    getClassColors() { return [...this.state.labelColors]; }

    /** @returns {string[]} A copy of the per-class display names. */
    getClassNames() { return [...this.state.classNames]; }

    /** @returns {boolean[]} A copy of the per-class marker-visibility flags. */
    getMarkersVisible() { return [...this.state.classMarkersVisible]; }

    // -------------------------------------------------------------- contrast

    /**
     * Opens (or toggles shut) the per-image contrast popover anchored beside
     * the given element.
     * @param {string} imgId @param {HTMLElement} anchorEl
     */
    openContrast(imgId, anchorEl) {
        openContrastPopover(this.state, imgId, anchorEl);
    }

    /** Closes the contrast popover, if open. */
    closeContrast() {
        closeContrastPopover();
    }

    // --------------------------------------------------------------- results

    /**
     * The most recent per-class object counts (summed across images), as last
     * reported by a statscomputed event.
     * @returns {number[]}
     */
    getCounts() { return [...this.state.lastCounts]; }

    /**
     * The most recent detected objects per image, grouped by class — each
     * object carries its centroid (image-pixel coords) and area.
     * @returns {Array<{id: string, objectsByClass: Array<Array<{cx: number, cy: number, area: number}>>}>}
     */
    getObjects() {
        return this.state.images
            .filter(img => img._centroids)
            .map(img => ({ id: img.id, objectsByClass: img._centroids }));
    }

    // --------------------------------------------------------- export / save

    /**
     * Builds a zip of segmentation masks and/or probability maps for every
     * loaded image (see export.js).
     * @param {{seg?: boolean, prob?: boolean, onProgress?: (metadata: Object) => void}} [options]
     * @returns {Promise<Blob|undefined>} The zip blob (undefined when there is nothing to export).
     */
    async exportZip({ seg = true, prob = false, onProgress } = {}) {
        return zipImages(this.state.images, seg, prob, onProgress);
    }

    /**
     * Builds a self-contained ilastik .ilp project blob (images + labels
     * embedded). Persisting it (file pickers, downloads) is the caller's job —
     * call markSaved() after a successful write.
     * @param {{classNames?: string[]}} [options]
     * @returns {Blob}
     */
    exportIlp(options = {}) {
        return exportIlp(this.state, { classNames: this.state.classNames, ...options });
    }

    /** Clears the dirty flag after a successful save and emits dirtychange. */
    markSaved() {
        if (!this.state.dirty) return;
        this.state.dirty = false;
        this._emit('dirtychange', { dirty: false });
    }

    // --------------------------------------------------------------- cleanup

    /**
     * Frees every image's GPU resources and removes their tiles from the
     * board. The instance should not be used afterward.
     */
    destroy() {
        closeContrastPopover();
        for (const img of this.state.images) {
            img.backend.destroy();
            img.container.remove();
        }
        this.state.images.length = 0;
    }
}
