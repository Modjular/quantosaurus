/**
 * Refreshes all image-count-driven chrome (status line, export button, empty
 * state) in one call. Invoke after any change to state.images.
 * @param {Object} state - Shared app state.
 */
export function syncUI(state) {
    updateStatus(state);
    updateExportButtonCount(state);
    refreshEmptyState(state);
    updateSaveIndicator(state);
}

/** Updates the status line with the loaded-image count. @param {Object} state */
export function updateStatus(state) {
    const n = state.images.length;
    document.getElementById('status').innerText =
        n === 0 ? 'No images loaded. Drag & drop onto the canvas.'
                : `${n} image${n !== 1 ? 's' : ''} loaded.`;
}

/**
 * Toggles the global "training in progress" affordance: a progress cursor (via a
 * body class) plus a status-line message. On turn-off it restores the normal
 * image-count status. Gives feedback while the overlay/probabilities recompute,
 * which is otherwise a silent await.
 * @param {Object} state - Shared app state (for restoring the status text).
 * @param {boolean} on - Whether training is currently running.
 */
export function setTrainingIndicator(state, on) {
    document.body.classList.toggle('training', on);
    if (on) {
        document.getElementById('status').innerText = 'Training…';
    } else {
        updateStatus(state);
    }
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

// Handle of the in-flight count-up animation frame, so a new retrain (or a no-op
// badge write) can cancel a still-running count-up before it clobbers fresh values.
let _countupRaf = null;

/** Cancels any in-flight count-up animation. */
export function cancelCountup() {
    if (_countupRaf !== null) {
        cancelAnimationFrame(_countupRaf);
        _countupRaf = null;
    }
}

/**
 * Reflects .ilp save status in the document title (a leading dot while
 * there are unsaved changes, mirroring how every desktop editor marks an
 * unsaved document) and the sidebar's save-status line: a dirty marker while
 * state.dirty is true, a success checkmark once saved, nothing before any
 * project exists. Call after anything that changes state.dirty.
 * @param {Object} state
 */
export function updateSaveIndicator(state) {
    const hasImages = state.images.length > 0;
    const dirty = hasImages && state.dirty;

    document.title = dirty ? '● Quantosaurus' : 'Quantosaurus';

    const el = document.getElementById('ilpSaveStatus');
    if (!el) return;
    if (!hasImages) {
        el.textContent = '';
        el.className = 'save-status';
    } else if (dirty) {
        el.textContent = '● Unsaved changes';
        el.className = 'save-status dirty';
    } else {
        el.textContent = '✓ Saved';
        el.className = 'save-status saved';
    }
}

/**
 * Writes per-class detected-object counts into the class-selector badges.
 * @param {Array<number>} counts - counts[classIdx] = objects for that class,
 *   summed across all loaded images.
 */
export function updateClassStatBadges(counts) {
    cancelCountup();
    counts.forEach((count, index) => {
        const badge = document.getElementById(`stat-class-${index}`);
        if (badge) {
            badge.classList.remove('loading');
            badge.textContent = count;
        }
    });
}

/**
 * Wipes the count badges to a pulsing "…" placeholder, signalling that fresh counts
 * are being computed. Called the moment a retrain starts so the stale numbers don't
 * just sit there until the new ones pop in.
 */
export function setClassBadgesLoading() {
    cancelCountup();
    document.querySelectorAll('.class-count-number').forEach(badge => {
        badge.classList.add('loading');
        badge.textContent = '…';
    });
}

/**
 * Animates each class badge counting up from 0 to its final value, as if the objects
 * are being tallied live. Clears the loading placeholder. A single rAF loop drives all
 * badges; a subsequent retrain cancels it via cancelCountup (see setClassBadgesLoading).
 * @param {Array<number>} counts - Final per-class object counts.
 * @param {{duration?: number}} [opts] - Animation duration in ms (default 500).
 */
export function animateClassStatBadges(counts, { duration = 500 } = {}) {
    cancelCountup();
    const badges = counts.map((_, i) => document.getElementById(`stat-class-${i}`));
    badges.forEach(b => b && b.classList.remove('loading'));

    const easeOutCubic = (t) => 1 - Math.pow(1 - t, 3);
    const start = performance.now();
    const step = (now) => {
        const t = Math.min(1, (now - start) / duration);
        const eased = easeOutCubic(t);
        counts.forEach((count, i) => {
            const badge = badges[i];
            if (badge) badge.textContent = Math.round(count * eased);
        });
        if (t < 1) {
            _countupRaf = requestAnimationFrame(step);
        } else {
            _countupRaf = null;
        }
    };
    _countupRaf = requestAnimationFrame(step);
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
