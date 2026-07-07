import { state, LABEL_COLORS, MIN_LABELS_TO_TRAIN, RF_CONFIG, TRAIN_DEBOUNCE_MS } from './state.js';
import { WebGpuBackend } from './backends/webgpu.js';
import { WebGl2Backend } from './backends/webgl2.js';
import { loadFileIntoArray } from './io.js';
import { createImageRow, syncUI, updateImageStateBadge } from './ui.js';

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
export function scheduleTrainAndPredictAll() {
    clearTimeout(_trainTimer);
    _trainTimer = setTimeout(() => trainAndPredictAll(), TRAIN_DEBOUNCE_MS);
}

export async function trainAndPredictAll() {
    const totalLabels = state.images.reduce((s, i) => s + i.labels.length, 0);
    if (totalLabels < MIN_LABELS_TO_TRAIN) return;

    const { combinedX, yArray } = await buildTrainingDataset(state.images, totalLabels);
    state.rf.train(combinedX, yArray, RF_CONFIG.numTrees);

    for (const img of state.images) {
        await img.backend.runInference(state.rf);
    }
}

export async function handleFiles(files) {
    for (const file of files) {
        const validFileTypes = ['.tif', '.tiff', '.png', '.jpg', '.jpeg'];
        const isValidFileType = validFileTypes.some((filetype) => file.name.toLowerCase().endsWith(filetype));
        const isDuplicate = state.images.some(img => img.name === file.name && img.fileSize === file.size);

        if (!isDuplicate && isValidFileType) {
            await addImage(file);
        }
    }
}

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

export async function addImage(file) {
    const imgId = crypto.randomUUID();
    const row   = createImageRow(imgId, file.name, {
        onReorder: (dir) => reorderImage(imgId, dir),
        onDelete:  ()    => deleteImage(imgId),
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
        syncUI();
        return;
    }

    if (loaded.shape.length > 2) {
      console.warn(`Only 2D images are currently supported. ${file.name} has shape (${loaded.shape})`);
      row.remove();
      syncUI();
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
        syncUI();
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
        paint(e, imgState);
    });
    labelCanvas.addEventListener('mousemove', (e) => {
        if (state.isSpaceDown) return
        if (state.isDrawing && state.activeImageId === imgId && state.toolMode !== 'grab') {
            paint(e, imgState);
        }
    });
    labelCanvas.addEventListener('mouseup', () => {
        if (state.activeImageId === imgId) {
            state.isDrawing    = false;
            state.activeImageId = null;
            scheduleTrainAndPredictAll();
        }
    });

    const ro = new ResizeObserver(() => { imgState._cachedRect = null; });
    ro.observe(labelCanvas);

    row.classList.remove('loading');
    updateImageStateBadge(imgState);

    syncUI();
}

export function reorderImage(imgId, direction) {
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

export function deleteImage(imgId) {
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

    if (labelCount > 0) scheduleTrainAndPredictAll();

    syncUI();
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

export function paint(e, imgState) {
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

    updateImageStateBadge(imgState);
}
