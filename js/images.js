import { RF_CONFIG, CENTROID_OVERLAY } from './config.js';
import { WebGpuBackend } from './backends/webgpu.js';
import { WebGl2Backend } from './backends/webgl2.js';
import { loadFileIntoArray } from './io.js';
import { scheduleTraining } from './training.js';
import { openContrastPopover } from './contrast.js';

// Dispatches a CustomEvent on the owning Quantosaurus instance (state.events).
// Optional-chained so state objects without a dispatch target (unit tests,
// plain-object fixtures) keep working — events are strictly additive here.
function emit(state, name, detail) {
    state.events?.dispatchEvent(new CustomEvent(name, { detail }));
}

// Flips the dirty flag, emitting dirtychange only on the false->true
// transition — paint() is the hottest path in the app (fires on every
// mousemove while dragging), and dirty stays true for the rest of a typical
// painting session.
function markDirty(state) {
    if (state.dirty) return;
    state.dirty = true;
    emit(state, 'dirtychange', { dirty: true });
}


/**
 * Creates a compute/render backend for one image canvas, preferring WebGPU and
 * falling back to WebGL2. Both implement the same backend interface (see
 * backends/webgpu.js and backends/webgl2.js).
 * @param {HTMLCanvasElement} canvas - The per-image GPU canvas to render into.
 * @param {Array<string>} labelColors - CSS color per class.
 * @returns {Promise<WebGpuBackend|WebGl2Backend>} An initialized backend.
 * @throws If neither WebGPU nor WebGL2 is available.
 */
async function initializeBackend(canvas, labelColors) {
  // 1. Try WebGPU first
  if (navigator.gpu) {
      try {
          const backend = new WebGpuBackend(labelColors);
          await backend.initialize(canvas);
          console.log("WebGPU initialized successfully.");
          return backend;
      } catch (e) {
          console.warn("WebGPU initialization failed, falling back to WebGL2:", e);
      }
  }

  // 2. Fallback to WebGL2
  try {
      const backend = new WebGl2Backend(labelColors);
      await backend.initialize(canvas);
      console.log("WebGL2 initialized successfully.");
      return backend;
  } catch (e) {
      console.error("WebGL2 initialization failed:", e);
  }

  // 3. Final error handling
  alert("Your browser does not support the required graphics APIs (WebGPU or WebGL2).");
  throw new Error("No compatible rendering backend found.");
}

/**
 * Builds the floating controls overlaid on one image tile: reorder left/right,
 * contrast, and delete. Buttons call the image-lifecycle functions directly
 * with `state` (the tile owns them, like the paint handlers) — the contrast
 * button anchors its popover to itself.
 * @param {Object} state - Shared app state.
 * @param {string} imgId - Id of the image this tile represents.
 * @returns {HTMLDivElement} The controls overlay (not yet attached).
 */
function buildTileControls(state, imgId) {
    const wrap = document.createElement('div');
    wrap.className = 'tile-controls';

    const mk = (cls, glyph, title, onClick) => {
        const b = document.createElement('button');
        b.className   = `tile-btn ${cls}`;
        b.textContent = glyph;
        b.title       = title;
        b.onclick     = (e) => { e.stopPropagation(); onClick(b); };
        return b;
    };

    wrap.appendChild(mk('tile-reorder', '‹', 'Move earlier', () => reorderImage(state, imgId, -1)));
    wrap.appendChild(mk('tile-reorder', '›', 'Move later',   () => reorderImage(state, imgId, +1)));
    wrap.appendChild(mk('tile-contrast', '◐', 'Adjust contrast', (btn) => openContrastPopover(state, imgId, btn)));
    wrap.appendChild(mk('tile-delete', '✕', 'Remove image', () => deleteImage(state, imgId)));

    return wrap;
}

/**
 * Adds a batch of dropped/selected files, skipping unsupported types and
 * duplicates (matched by name + size).
 * @param {Object} state - Shared app state.
 * @param {Iterable<File>} files - Files to import.
 */
export async function addFiles(state, files) {
    for (const file of files) {
        const validFileTypes = ['.tif', '.tiff', '.png', '.jpg', '.jpeg'];
        const isValidFileType = validFileTypes.some((filetype) => file.name.toLowerCase().endsWith(filetype));
        const isDuplicate = state.images.some(img => img.name === file.name && img.fileSize === file.size);

        if (!isDuplicate && isValidFileType) {
            await addImage(state, file);
        }
    }
}

/**
 * Loads a single image file, builds its canvas tile, spins up a backend,
 * computes initial features, and registers the paint handlers. Emits
 * imageloadstart at the top, then either imageadded on success or
 * imageloaderror (cleaning up any partial DOM) on load failure, non-2D
 * images, or backend-init failure — chrome tracks load progress off these.
 * @param {Object} state - Shared app state; the new image is pushed to state.images.
 * @param {File} file - The image file to load.
 */
export async function addImage(state, file) {
    const imgId = crypto.randomUUID();
    emit(state, 'imageloadstart', { id: imgId, name: file.name });

    let loaded;
    try {
        loaded = await loadFileIntoArray(file);
    } catch (err) {
        console.error(err);
        emit(state, 'imageloaderror', {
            id: imgId, name: file.name, reason: 'load-failed',
            message: `Failed to load ${file.name}: ${err.message}`,
        });
        return;
    }

    if (loaded.shape.length > 2) {
      console.warn(`Only 2D images are currently supported. ${file.name} has shape (${loaded.shape})`);
      emit(state, 'imageloaderror', { id: imgId, name: file.name, reason: 'unsupported-dimensions' });
      return;
    }

    const { intensityArray, w, h, range } = loaded;

    const container = document.createElement('div');
    container.className = 'image-container';
    container.style.width  = w + 'px';
    container.style.height = h + 'px';

    const gpuCanvas   = document.createElement('canvas');
    gpuCanvas.className = 'gpu-canvas';
    gpuCanvas.width = w; gpuCanvas.height = h;

    const labelCanvas = document.createElement('canvas');
    labelCanvas.className = 'label-canvas';
    labelCanvas.width = w; labelCanvas.height = h;

    // Centroid markers layer: same w×h backing store as the others so it aligns
    // 1:1 in image-pixel space. Sits above the label canvas with pointer-events
    // disabled (see .overlay-canvas in style.css) so it never intercepts painting.
    const overlayCanvas = document.createElement('canvas');
    overlayCanvas.className = 'overlay-canvas';
    overlayCanvas.width = w; overlayCanvas.height = h;

    const tileLabel = document.createElement('div');
    tileLabel.className   = 'image-tile-label';
    tileLabel.textContent = `${file.name} ${w}x${h}`;

    // Per-tile controls (reorder ‹ ›, contrast ◐, delete ✕) — revealed on hover
    // (or always, on coarse pointers; see .tile-controls in style.css). These
    // replace the old sidebar row's buttons now that the board is the only view.
    const controls = buildTileControls(state, imgId);

    container.appendChild(gpuCanvas);
    container.appendChild(labelCanvas);
    container.appendChild(overlayCanvas);
    container.appendChild(tileLabel);
    container.appendChild(controls);
    state.board.appendChild(container);

    let backend = null;

    try {
        backend = await initializeBackend(gpuCanvas, state.labelColors)
    } catch (err) {
        console.error(err);
        container.remove();
        emit(state, 'imageloaderror', { id: imgId, name: file.name, reason: 'no-backend' });
        return;
    }
    await backend.allocateImage(w, h, intensityArray, range);
    await backend.updateFeatures(intensityArray, state.sigma);

    const imgState = {
        id: imgId,
        name: file.name,
        fileSize: file.size,
        backend,
        width: w, height: h,
        intensityArray,
        range,
        // Contrast window (black/white points) in the image's raw intensity units.
        // Defaults to the data's min/max, reproducing the previous auto-stretch look;
        // allocateImage seeds the backend with the same values.
        windowLo: range.dataMin,
        windowHi: range.dataMax,
        labels: [],
        gpuCanvas, labelCanvas, overlayCanvas, container,
        _cachedRect: null,
    };
    state.images.push(imgState);
    markDirty(state);

    // Pointer events unify mouse/touch/pen. Only the primary pointer paints
    // (the first touch contact, or the mouse) — a second simultaneous touch is
    // always a pinch/pan gesture (see camera.js), never a second brush.
    labelCanvas.addEventListener('pointerdown', (e) => {
        if (!e.isPrimary) return;
        if (state.isSpaceDown || state.toolMode === 'grab') return;
        labelCanvas.setPointerCapture(e.pointerId);
        imgState._cachedRect = labelCanvas.getBoundingClientRect();
        state.isDrawing    = true;
        state.activeImageId = imgId;
        paint(state, imgState, e, state.brushSize);
    });
    labelCanvas.addEventListener('pointermove', (e) => {
        if (!e.isPrimary) return;
        if (state.isSpaceDown) return;
        if (state.activePointerCount > 1) return; // a second finger joined — pause the stroke
        if (state.isDrawing && state.activeImageId === imgId && state.toolMode !== 'grab') {
            paint(state, imgState, e, state.brushSize);
        }
    });
    function endStroke(e) {
        if (!e.isPrimary) return;
        if (state.activeImageId === imgId) {
            state.isDrawing    = false;
            state.activeImageId = null;
            emit(state, 'labelschanged', {
                id: imgId,
                labelCount: imgState.labels.length,
                totalLabels: state.images.reduce((s, i) => s + i.labels.length, 0),
            });
            scheduleTraining(state);
        }
    }
    labelCanvas.addEventListener('pointerup', endStroke);
    labelCanvas.addEventListener('pointercancel', endStroke);

    const ro = new ResizeObserver(() => { imgState._cachedRect = null; });
    ro.observe(labelCanvas);

    emit(state, 'imageadded', {
        id: imgId, name: file.name, width: w, height: h,
        index: state.images.length - 1,
    });
}

/**
 * Moves an image one slot up or down, keeping state.images and the canvas
 * tiles in the same visual order (which is also export order). Emits
 * imagereordered so chrome tracking image order can follow.
 * @param {Object} state - Shared app state.
 * @param {string} imgId - Id of the image to move.
 * @param {-1|1} direction - -1 to move up, +1 to move down.
 */
export function reorderImage(state, imgId, direction) {
    const idx = state.images.findIndex(i => i.id === imgId);
    if (idx === -1) return;
    const newIdx = idx + direction;
    if (newIdx < 0 || newIdx >= state.images.length) return;

    [state.images[idx], state.images[newIdx]] =
    [state.images[newIdx], state.images[idx]];
    markDirty(state); // changes export lane order

    const tiles = [...state.board.children];
    const a = tiles[idx];
    const b = tiles[newIdx];
    if (direction === -1) state.board.insertBefore(a, b);
    else                  state.board.insertBefore(b, a);

    emit(state, 'imagereordered', { id: imgId, fromIndex: idx, toIndex: newIdx });
}

/**
 * Removes an image and frees its backend/GPU resources. Prompts for
 * confirmation first if it carries labels, and retrains afterward if so.
 * @param {Object} state - Shared app state.
 * @param {string} imgId - Id of the image to delete.
 */
export function deleteImage(state, imgId) {
    const idx = state.images.findIndex(i => i.id === imgId);
    if (idx === -1) return;

    const imgState = state.images[idx];
    const labelCount = imgState.labels.length;

    if (labelCount > 0) {
        const ok = confirm(
            `"${imgState.name}" has ${labelCount} label${labelCount !== 1 ? 's' : ''}.\n` +
            `Deleting it will remove those labels and retrain the model. Continue?`
        );
        if (!ok) return;
    }

    imgState.backend.destroy();
    imgState.container.remove();
    state.images.splice(idx, 1);
    markDirty(state);

    emit(state, 'imageremoved', { id: imgId, index: idx });

    if (labelCount > 0) {
        emit(state, 'labelschanged', {
            id: imgId,
            labelCount: 0,
            totalLabels: state.images.reduce((s, i) => s + i.labels.length, 0),
        });
        scheduleTraining(state);
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
function getPixelsInRadius(cx, cy, radius, width, height) {
    const pixels = [];
    const inside = (x, y, w, h) => x >= 0 && x < w && y >= 0 && y < h

    // Exact 1-pixel brush
    if (radius === 1) {
        if (inside(cx, cy, width, height)) {
            return [{ x: cx, y: cy }];
        }
    }

    const rSq = radius * radius;
    const rInt = Math.ceil(radius);

    // Check a bounding box around the center
    for (let y = cy - rInt; y <= cy + rInt; y++) {
        for (let x = cx - rInt; x <= cx + rInt; x++) {
            // Keep it inside canvas bounds
            if (inside(x, y, width, height)) {
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
 * Applies the active brush at a pointer event's location: maps client coords to
 * image pixels, paints or erases the brush footprint on the label canvas, and
 * updates imgState.labels accordingly (dropping any labels the brush overwrites).
 * @param {Object} state - Shared app state; reads toolMode, currentClass, labelColors.
 * @param {Object} imgState - The target image's state entry.
 * @param {PointerEvent} e - The triggering pointer event (mouse, touch, or pen).
 */
/**
 * Redraws every painted pixel on an image's label canvas using the current
 * state.labelColors palette. paint() bakes each pixel's color in at paint
 * time (a plain 2D canvas fillRect), so unlike the GPU composite overlay it
 * doesn't pick up palette changes on its own — call this after a color edit.
 * @param {Object} state - Shared app state; reads labelColors.
 * @param {Object} imgState - The target image's state entry; reads labels.
 */
export function redrawLabels(state, imgState) {
    const ctx = imgState.labelCanvas.getContext('2d');
    ctx.clearRect(0, 0, imgState.width, imgState.height);
    for (const lbl of imgState.labels) {
        ctx.fillStyle = state.labelColors[lbl.cls] ?? state.labelColors[0];
        ctx.fillRect(lbl.x, lbl.y, 1, 1);
    }
}

/**
 * Caches an image's freshly detected objects (grouped by class) without painting.
 * Split out from renderCentroids so a marker-visibility toggle can repaint from the
 * existing cache without a new stats download/retrain.
 * @param {Object} imgState - The target image's state entry.
 * @param {Array<Array<{cx: number, cy: number, area: number}>>} objectsByClass -
 *   Detected objects grouped by class index; each object carries its centroid and area.
 */
export function setCentroids(imgState, objectsByClass) {
    imgState._centroids = objectsByClass;
}

/**
 * Redraws the centroid-marker overlay for one image from its cached objects
 * (see setCentroids): a class-colored circle over each detected object's centroid,
 * sized by area on a log scale (CENTROID_OVERLAY). Classes flagged hidden in
 * state.classMarkersVisible are skipped. Fully clears and repaints on every call, so
 * it's cheap to call after a color change or a visibility toggle — no GPU/stats work.
 * @param {Object} state - Shared app state; reads labelColors, classMarkersVisible.
 * @param {Object} imgState - The target image's state entry; reads the _centroids cache.
 */
export function renderCentroids(state, imgState) {
    const ctx = imgState.overlayCanvas.getContext('2d');
    ctx.clearRect(0, 0, imgState.width, imgState.height);
    const objectsByClass = imgState._centroids;
    if (!objectsByClass) return;
    for (let cls = 0; cls < objectsByClass.length; cls++) {
        if (state.classMarkersVisible?.[cls] === false) continue;
        const objs = objectsByClass[cls];
        if (!objs?.length) continue;
        const color = state.labelColors[cls] ?? state.labelColors[0];
        for (const o of objs) {
            const r = CENTROID_OVERLAY.minRadius + CENTROID_OVERLAY.logScale * Math.log(o.area);
            ctx.beginPath();
            ctx.arc(o.cx, o.cy, r, 0, Math.PI * 2);
            // Dark halo first, then the class-color ring on top — keeps the marker
            // legible even over a composite region tinted in the same class color.
            ctx.lineWidth   = CENTROID_OVERLAY.lineWidth * 2;
            ctx.strokeStyle = `rgba(0,0,0,${CENTROID_OVERLAY.haloAlpha})`;
            ctx.stroke();
            ctx.lineWidth   = CENTROID_OVERLAY.lineWidth;
            ctx.strokeStyle = color;
            ctx.stroke();
        }
    }
}

/**
 * Clears an image's centroid-marker overlay. Used when there aren't enough labels
 * to train, so stale markers from a previous forest don't linger.
 * @param {Object} imgState - The target image's state entry.
 */
export function clearCentroids(imgState) {
    imgState._centroids = null;
    imgState.overlayCanvas.getContext('2d').clearRect(0, 0, imgState.width, imgState.height);
}

export function paint(state, imgState, e, radius) {
    const rect   = imgState._cachedRect || imgState.labelCanvas.getBoundingClientRect();
    const scaleX = imgState.width  / rect.width;
    const scaleY = imgState.height / rect.height;
    const x      = Math.floor((e.clientX - rect.left) * scaleX);
    const y      = Math.floor((e.clientY - rect.top)  * scaleY);
    const ctx    = imgState.labelCanvas.getContext('2d');

    const pixels = getPixelsInRadius(x, y, radius, imgState.width, imgState.height);
    if (pixels.length === 0) return;
    markDirty(state);

    const pixelSet = new Set(pixels.map(p => `${p.x},${p.y}`));
    imgState.labels = imgState.labels.filter(lbl => !pixelSet.has(`${lbl.x},${lbl.y}`));

    if (state.toolMode === 'paint') {
        ctx.fillStyle = state.labelColors[state.currentClass] ?? state.labelColors[0];
        pixels.forEach(p => ctx.fillRect(p.x, p.y, 1, 1));
        pixels.forEach(p => imgState.labels.push({ x: p.x, y: p.y, cls: state.currentClass }));
    } else if (state.toolMode === 'erase') {
        pixels.forEach(p => ctx.clearRect(p.x, p.y, 1, 1));
    } else {
        console.error("Unrecognized tool mode:", state.toolMode);
    }
}
