import { LABEL_COLORS, RF_CONFIG, CONTRAST_DEFAULT } from './config.js';
import { WebGpuBackend } from './backends/webgpu.js';
import { WebGl2Backend } from './backends/webgl2.js';
import { loadFileIntoArray } from './io.js';
import { createImageRow, syncUI } from './ui.js';
import { openContrastPopover } from './contrast.js';
import { scheduleTraining } from './training.js';


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
        const isDuplicate = state.images.some(img => img.name === file.name && img.fileSize === file.size);

        if (!isDuplicate && isValidFileType) {
            await addImage(state, file);
        }
    }
}

/**
 * Loads a single image file, builds its sidebar row and canvas tile, spins up a
 * backend, computes initial features, and registers the paint handlers. Bails
 * out (cleaning up the partial DOM) on load failure or non-2D images.
 * @param {Object} state - Shared app state; the new image is pushed to state.images.
 * @param {File} file - The image file to load.
 */
export async function addImage(state, file) {
    const imgId = crypto.randomUUID();
    const row   = createImageRow(imgId, file.name, {
        onReorder: (dir)      => reorderImage(state, imgId, dir),
        onDelete:  ()         => deleteImage(state, imgId),
        onContrast: (anchor)  => openContrastPopover(state, imgId, anchor),
    });
    row.classList.add('loading');
    document.getElementById('img-empty').style.display = 'none';
    document.getElementById('image-list').appendChild(row);

    let loaded;
    try {
        loaded = await loadFileIntoArray(file);
    } catch (err) {
        console.error(err);
        alert(`Failed to load ${file.name}: ${err.message}`);
        row.remove();
        syncUI(state);
        return;
    }

    if (loaded.shape.length > 2) {
      console.warn(`Only 2D images are currently supported. ${file.name} has shape (${loaded.shape})`);
      row.remove();
      syncUI(state);
      return;
    }

    const { intensityArray, rgba, w, h } = loaded;

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

    const tileLabel = document.createElement('div');
    tileLabel.className   = 'image-tile-label';
    tileLabel.textContent = `${file.name} ${w}x${h}`;

    container.appendChild(gpuCanvas);
    container.appendChild(labelCanvas);
    container.appendChild(tileLabel);
    document.getElementById('canvas-board').appendChild(container);

    let backend = null;
  
    try {
        backend = await initializeBackend(gpuCanvas, LABEL_COLORS)
    } catch (err) {
        console.error(err);
        container.remove();
        row.remove();
        syncUI(state);
        return;
    }
    await backend.allocateImage(w, h, rgba);
    await backend.updateFeatures(intensityArray, state.sigma);

    const imgState = {
        id: imgId,
        name: file.name,
        fileSize: file.size,
        backend,
        width: w, height: h,
        intensityArray,
        windowLo: CONTRAST_DEFAULT.lo,
        windowHi: CONTRAST_DEFAULT.hi,
        labels: [],
        gpuCanvas, labelCanvas, container,
        _cachedRect: null,
        _sidebarRow: row,
    };
    state.images.push(imgState);

    labelCanvas.addEventListener('mousedown', (e) => {
        if (state.isSpaceDown || state.toolMode === 'grab') return;
        imgState._cachedRect = labelCanvas.getBoundingClientRect();
        state.isDrawing    = true;
        state.activeImageId = imgId;
        paint(state, imgState, e);
    });
    labelCanvas.addEventListener('mousemove', (e) => {
        if (state.isSpaceDown) return
        if (state.isDrawing && state.activeImageId === imgId && state.toolMode !== 'grab') {
            paint(state, imgState, e);
        }
    });
    labelCanvas.addEventListener('mouseup', () => {
        if (state.activeImageId === imgId) {
            state.isDrawing    = false;
            state.activeImageId = null;
            scheduleTraining(state);
        }
    });

    const ro = new ResizeObserver(() => { imgState._cachedRect = null; });
    ro.observe(labelCanvas);

    row.classList.remove('loading');

    syncUI(state);
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

    if (labelCount > 0) scheduleTraining(state);

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

    // Exact 1-pixel brush
    if (radius === 1) {
        if (cx >= 0 && cx < width && cy >= 0 && cy < height) {
            pixels.push({ x: cx, y: cy });
        }
        return pixels;
    }

    const rSq = radius * radius;
    const rInt = Math.ceil(radius);

    // Check a bounding box around the center
    for (let y = cy - rInt; y <= cy + rInt; y++) {
        for (let x = cx - rInt; x <= cx + rInt; x++) {
            // Keep it inside canvas bounds
            if (x >= 0 && x < width && y >= 0 && y < height) {
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
 * Applies the active brush at a mouse event's location: maps client coords to
 * image pixels, paints or erases the brush footprint on the label canvas, and
 * updates imgState.labels accordingly (dropping any labels the brush overwrites).
 * @param {Object} state - Shared app state; reads toolMode and currentClass.
 * @param {Object} imgState - The target image's state entry.
 * @param {MouseEvent} e - The triggering mouse event.
 */
export function paint(state, imgState, e) {
    const rect   = imgState._cachedRect || imgState.labelCanvas.getBoundingClientRect();
    const scaleX = imgState.width  / rect.width;
    const scaleY = imgState.height / rect.height;
    const x      = Math.floor((e.clientX - rect.left) * scaleX);
    const y      = Math.floor((e.clientY - rect.top)  * scaleY);
    const radius = parseInt(document.getElementById('brushSizeRange').value, 10);
    const ctx    = imgState.labelCanvas.getContext('2d');

    const pixels = getPixelsInRadius(x, y, radius, imgState.width, imgState.height);
    if (pixels.length === 0) return;
    
    const pixelSet = new Set(pixels.map(p => `${p.x},${p.y}`));
    imgState.labels = imgState.labels.filter(lbl => !pixelSet.has(`${lbl.x},${lbl.y}`));

    if (state.toolMode === 'paint') {
        ctx.fillStyle = LABEL_COLORS[state.currentClass] ?? LABEL_COLORS[0];
        pixels.forEach(p => ctx.fillRect(p.x, p.y, 1, 1));
        pixels.forEach(p => imgState.labels.push({ x: p.x, y: p.y, cls: state.currentClass }));
    } else if (state.toolMode === 'erase') {
        pixels.forEach(p => ctx.clearRect(p.x, p.y, 1, 1));
    } else {
        console.error("Unrecognized tool mode:", state.toolMode);
    }
}
