import { CENTROID_OVERLAY } from './config.js';
import { WebGpuBackend } from './backends/webgpu.js';
import { WebGl2Backend } from './backends/webgl2.js';
import { loadFileIntoArray } from './io.js';
import { createImageRow, syncUI, updateSaveIndicator } from './ui.js';
import { openContrastPopover } from './contrast.js';


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
 * Adds a batch of dropped/selected files, skipping unsupported types and
 * duplicates (matched by name + size).
 * @param {Object} state - Shared app state.
 * @param {Iterable<File>} files - Files to import.
 */
export async function addFiles(state, files) {
    for (const file of files) {
        const validFileTypes = ['.tif', '.tiff', '.png', '.jpg', '.jpeg'];
        const isValidFileType = validFileTypes.some((filetype) => file.name.toLowerCase().endsWith(filetype));
        // Dedup on the source file (name + size), not the display name: a multichannel
        // file expands into several images whose display names are channel-suffixed, but
        // they all share sourceName === file.name.
        const isDuplicate = state.images.some(img => img.sourceName === file.name && img.fileSize === file.size);

        if (!isDuplicate && isValidFileType) {
            await addImage(state, file);
        }
    }
}

/**
 * Builds a sidebar row wired to this app's reorder/delete/contrast handlers for a
 * given image id.
 * @param {Object} state - Shared app state.
 * @param {string} id - The image id the row's controls act on.
 * @param {string} name - Display name for the row.
 * @returns {HTMLElement} The sidebar row element (not yet attached).
 */
function createRowFor(state, id, name) {
    return createImageRow(id, name, {
        onReorder:  (dir)    => reorderImage(state, id, dir),
        onDelete:   ()       => deleteImage(state, id),
        onContrast: (anchor) => openContrastPopover(state, id, anchor),
    });
}

/**
 * Loads a single image file and materializes one displayed image per channel: a 2D
 * multichannel TIFF becomes C independent single-channel images (each with its own
 * sidebar row, canvas tile, backend, contrast window, and labels), while single-channel
 * TIFFs and PNG/JPG yield one. Bails out (cleaning up the partial DOM) on load failure
 * or non-2D images.
 * @param {Object} state - Shared app state; each new image is pushed to state.images.
 * @param {File} file - The image file to load.
 */
export async function addImage(state, file) {
    const placeholderId = crypto.randomUUID();
    const placeholderRow = createRowFor(state, placeholderId, file.name);
    placeholderRow.classList.add('loading');
    document.getElementById('img-empty').style.display = 'none';
    document.getElementById('image-list').appendChild(placeholderRow);

    let loaded;
    try {
        loaded = await loadFileIntoArray(file);
    } catch (err) {
        console.error(err);
        alert(`Failed to load ${file.name}: ${err.message}`);
        placeholderRow.remove();
        syncUI(state);
        return;
    }

    if (loaded.shape.length > 2) {
      const dims = loaded.shape.join('×');
      console.warn(`Only 2D images are currently supported. ${file.name} has shape (${loaded.shape})`);
      // Surface the rejection to the user — otherwise a dropped z-stack just silently
      // vanishes with only a console message they won't see.
      alert(`Only 2D images are supported. "${file.name}" is ${loaded.shape.length}D (${dims}) — 3D volumes / z-stacks aren't supported yet.`);
      placeholderRow.remove();
      syncUI(state);
      return;
    }

    const { channels, w, h } = loaded;
    const multichannel = channels.length > 1;

    // Cellpose keeps a file's channels together in one image (it segments a
    // cyto+nucleus pair), so it sets state.splitChannels = false. The classifier
    // and threshold apps leave it unset (defaulting to true) and split a
    // multichannel file into one independent single-channel image per channel.
    const split = state.splitChannels !== false;

    if (!split) {
        // One entry per file, holding every channel. The displayed channel starts
        // at 0; the caller (Cellpose) picks which channels are cyto/nucleus.
        await buildImageEntry(state, {
            id: placeholderId, name: file.name, sourceName: file.name, fileSize: file.size,
            intensityArray: channels[0].intensityArray, w, h, range: channels[0].range,
            row: placeholderRow, channels,
        });
        syncUI(state);
        state.onImagesChanged?.(state);
        return;
    }

    // Single channel reuses the loading placeholder row; multichannel gives each channel
    // its own row (with a channel-suffixed name), so drop the placeholder first.
    if (multichannel) placeholderRow.remove();

    for (let c = 0; c < channels.length; c++) {
        const { intensityArray, range } = channels[c];
        const name = multichannel ? `${file.name} [ch${c}]` : file.name;
        const id   = multichannel ? crypto.randomUUID() : placeholderId;
        let row = placeholderRow;
        if (multichannel) {
            row = createRowFor(state, id, name);
            row.classList.add('loading');
            document.getElementById('image-list').appendChild(row);
        }
        await buildImageEntry(state, {
            id, name, sourceName: file.name, fileSize: file.size,
            intensityArray, w, h, range, row,
        });
    }

    syncUI(state);
    // Let a method that produces results without user input (Threshold) run on the
    // freshly loaded image(s). The classifier leaves this unset — a new image has no
    // labels of its own and inference reruns on the next paint.
    state.onImagesChanged?.(state);
}

/**
 * Builds the canvas tile, GPU backend, per-image state entry, and paint handlers for a
 * single already-loaded single-channel intensity plane, and pushes it to state.images.
 * On backend-init failure it cleans up its own DOM (tile + sidebar row) and returns false.
 * @param {Object} state - Shared app state.
 * @param {Object} args
 * @param {string} args.id - Unique image id.
 * @param {string} args.name - Display name (channel-suffixed for multichannel sources).
 * @param {string} args.sourceName - Original source file name (shared across a file's channels; used for dedup).
 * @param {number} args.fileSize - Source file size in bytes (used for dedup).
 * @param {Float32Array} args.intensityArray - Raw single-channel intensities, length w*h.
 * @param {number} args.w - Image width.
 * @param {number} args.h - Image height.
 * @param {Object} args.range - Display range metadata {dataMin, dataMax, dtypeMax, scale}.
 * @param {HTMLElement} args.row - The sidebar row already attached for this image.
 * @returns {Promise<boolean>} True on success, false if the backend failed to initialize.
 */
async function buildImageEntry(state, { id, name, sourceName, fileSize, intensityArray, w, h, range, row, channels = null }) {
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
    tileLabel.textContent = `${name} ${w}x${h}`;

    container.appendChild(gpuCanvas);
    container.appendChild(labelCanvas);
    container.appendChild(overlayCanvas);
    container.appendChild(tileLabel);
    document.getElementById('canvas-board').appendChild(container);

    let backend = null;

    try {
        backend = await initializeBackend(gpuCanvas, state.labelColors)
    } catch (err) {
        console.error(err);
        container.remove();
        row.remove();
        return false;
    }
    await backend.allocateImage(w, h, intensityArray, range);
    // The 8-channel Gaussian-derivative feature bank is only consumed by the
    // random-forest classifier. Threshold and Cellpose never read it, so they
    // leave state.computeFeatures falsy and skip the GPU work + memory here.
    if (state.computeFeatures) await backend.updateFeatures(intensityArray, state.sigma);

    const imgState = {
        id,
        name,
        // Original source file name, shared by all channels of one file. Dedup keys on this
        // (see addFiles) so re-dropping a multichannel file doesn't re-add its channels.
        sourceName,
        fileSize,
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
        // All raw channels of the source file, kept together only in non-split
        // (Cellpose) loads; null for the classifier/threshold single-channel images.
        // intensityArray above is channels[displayChannel] when this is set.
        channels,
        displayChannel: 0,
        gpuCanvas, labelCanvas, overlayCanvas, container,
        _cachedRect: null,
        _sidebarRow: row,
    };
    state.images.push(imgState);
    state.dirty = true;

    // Pointer events unify mouse/touch/pen. Only the primary pointer paints
    // (the first touch contact, or the mouse) — a second simultaneous touch is
    // always a pinch/pan gesture (see camera.js), never a second brush.
    labelCanvas.addEventListener('pointerdown', (e) => {
        if (!e.isPrimary) return;
        if (state.isSpaceDown || state.toolMode === 'grab') return;
        labelCanvas.setPointerCapture(e.pointerId);
        imgState._cachedRect = labelCanvas.getBoundingClientRect();
        state.isDrawing    = true;
        state.activeImageId = id;
        paint(state, imgState, e, state.brushSize);
    });
    labelCanvas.addEventListener('pointermove', (e) => {
        if (!e.isPrimary) return;
        if (state.isSpaceDown) return;
        if (state.activePointerCount > 1) return; // a second finger joined — pause the stroke
        if (state.isDrawing && state.activeImageId === id && state.toolMode !== 'grab') {
            paint(state, imgState, e, state.brushSize);
        }
    });
    function endStroke(e) {
        if (!e.isPrimary) return;
        if (state.activeImageId === id) {
            state.isDrawing    = false;
            state.activeImageId = null;
            // Classifier hooks this to a debounced retrain; Threshold/Cellpose,
            // which don't learn from labels, simply leave it unset.
            state.onLabelsChanged?.(state);
        }
    }
    labelCanvas.addEventListener('pointerup', endStroke);
    labelCanvas.addEventListener('pointercancel', endStroke);

    const ro = new ResizeObserver(() => { imgState._cachedRect = null; });
    ro.observe(labelCanvas);

    row.classList.remove('loading');
    return true;
}

/**
 * Switches which raw channel a non-split (Cellpose) image displays: re-points its
 * intensityArray/range at that channel and re-allocates the backend so the canvas,
 * contrast window, and stats all operate on it. Re-allocating also resets the
 * probability buffer to the "no overlay" state, clearing any prior segmentation.
 * No-op for split images (channels === null).
 * @param {Object} img - Image state entry.
 * @param {number} idx - Channel index into img.channels.
 */
export async function setDisplayChannel(img, idx) {
    if (!img.channels || idx === img.displayChannel) return;
    const ch = img.channels[idx];
    img.intensityArray = ch.intensityArray;
    img.range = ch.range;
    img.displayChannel = idx;
    img.windowLo = ch.range.dataMin;
    img.windowHi = ch.range.dataMax;
    await img.backend.allocateImage(img.width, img.height, ch.intensityArray, ch.range);
    img.backend.renderComposite();
}

/**
 * Moves an image one slot up or down, keeping state.images, the canvas tiles,
 * and the sidebar rows in the same visual order (which is also export order).
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
    state.dirty = true; // changes export lane order
    updateSaveIndicator(state);

    const board = document.getElementById('canvas-board');
    const tiles = [...board.children];
    const a = tiles[idx];
    const b = tiles[newIdx];
    if (direction === -1) board.insertBefore(a, b);
    else                  board.insertBefore(b, a);

    const list = document.getElementById('image-list');
    const rows = [...list.children];
    const ra = rows[idx];
    const rb = rows[newIdx];
    if (direction === -1) list.insertBefore(ra, rb);
    else                  list.insertBefore(rb, ra);
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
    imgState._sidebarRow.remove();
    state.images.splice(idx, 1);
    state.dirty = true;

    // Deleting a labeled image changes the classifier's training set; deleting any
    // image changes the set the counts are computed over (Threshold/Cellpose).
    if (labelCount > 0) state.onLabelsChanged?.(state);
    state.onImagesChanged?.(state);

    syncUI(state);
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
    // Only touch the DOM on the false->true transition — paint() is the
    // hottest path in the app (fires on every mousemove while dragging), and
    // dirty stays true for the rest of a typical painting session.
    if (!state.dirty) { state.dirty = true; updateSaveIndicator(state); }

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
