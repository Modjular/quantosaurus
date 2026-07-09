/**
 * Refreshes all image-count-driven chrome (status line, export button, empty
 * state) in one call. Invoke after any change to state.images.
 * @param {Object} state - Shared app state.
 */
export function syncUI(state) {
    updateStatus(state);
    updateExportButtonCount(state);
    refreshEmptyState(state);
}

/** Updates the status line with the loaded-image count. @param {Object} state */
export function updateStatus(state) {
    const n = state.images.length;
    document.getElementById('status').innerText =
        n === 0 ? 'No images loaded. Drag & drop onto the canvas.'
                : `${n} image${n !== 1 ? 's' : ''} loaded.`;
}

/** Enables/labels the "Export Loaded Images" button by image count. @param {Object} state */
export function updateExportButtonCount(state) {
    const btn = document.getElementById('btnExportAll');
    const n   = state.images.length;
    btn.disabled  = n === 0;
    btn.innerText = n > 0 ? `Export Loaded Images (${n})` : 'Export Loaded Images';

    const ilpBtn = document.getElementById('btnExportIlp');
    if (ilpBtn) ilpBtn.disabled = n === 0;
}

/** Shows/hides the image-list empty-state placeholder. @param {Object} state */
export function refreshEmptyState(state) {
    document.getElementById('img-empty').style.display =
        state.images.length === 0 ? '' : 'none';
}

/**
 * Writes per-class detected-object counts into the class-selector badges.
 * @param {Array<number>} counts - counts[classIdx] = objects for that class,
 *   summed across all loaded images.
 */
export function updateClassStatBadges(counts) {
    counts.forEach((count, index) => {
        const badge = document.getElementById(`stat-class-${index}`);
        if (badge) badge.textContent = count;
    });
}

/**
 * Builds the sidebar DOM row for one image. The controller supplies the
 * callbacks so this view stays decoupled from the image-lifecycle module.
 * @param {string} imgId - Image id, stored on the row's dataset.
 * @param {string} name - Display name.
 * @param {{onReorder: (direction: -1|1) => void, onDelete: () => void}} handlers
 * @returns {HTMLDivElement} The constructed row element (not yet attached).
 */
export function createImageRow(imgId, name, { onReorder, onDelete, onContrast }) {
    const row = document.createElement('div');
    row.className  = 'img-row';
    row.dataset.id = imgId;

    const loadBar = document.createElement('div');
    loadBar.className = 'img-loading-bar';

    const header = document.createElement('div');
    header.className = 'img-row-header';

    const reorder = document.createElement('div');
    reorder.className = 'img-reorder';
    const btnUp   = document.createElement('button');
    const btnDown = document.createElement('button');
    btnUp.textContent   = '▲';
    btnDown.textContent = '▼';
    btnUp.title   = 'Move up';
    btnDown.title = 'Move down';
    btnUp.onclick   = () => onReorder(-1);
    btnDown.onclick = () => onReorder(+1);
    reorder.appendChild(btnUp);
    reorder.appendChild(btnDown);

    const nameEl = document.createElement('div');
    nameEl.className   = 'img-name';
    nameEl.textContent = name;
    nameEl.title       = name;

    const btnContrast = document.createElement('button');
    btnContrast.className   = 'img-contrast';
    btnContrast.textContent = '◐';
    btnContrast.title       = 'Adjust contrast';
    btnContrast.onclick     = (e) => { e.stopPropagation(); onContrast?.(btnContrast); };

    const btnDel = document.createElement('button');
    btnDel.className   = 'img-delete';
    btnDel.textContent = '✕';
    btnDel.title       = 'Remove image';
    btnDel.onclick     = () => onDelete();

    header.appendChild(reorder);
    header.appendChild(nameEl);
    header.appendChild(btnContrast);
    header.appendChild(btnDel);

    row.appendChild(loadBar);
    row.appendChild(header);

    return row;
}
