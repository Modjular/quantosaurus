/**
 * Wires the floating chrome onto a Quantosaurus instance's events: the count
 * badges, the training indicator, the save indicator, the export-button enable
 * state, and image-load errors. The chrome is a pure subscriber — every
 * mutation goes back through methods on `q`, so this whole layer can be swapped
 * out (or omitted, in an embed) without touching the app core. The taskbar
 * controls themselves are wired separately (see index.html's setupUI).
 * @param {import('./quantosaurus.js').Quantosaurus} q - The app instance.
 */
export function bindChrome(q) {
    const state = q.state;

    q.addEventListener('imageadded', () => syncUI(state));
    q.addEventListener('imageremoved', () => syncUI(state));

    q.addEventListener('imageloaderror', (e) => {
        if (e.detail.message) alert(e.detail.message);
    });

    q.addEventListener('dirtychange', () => updateSaveIndicator(state));

    q.addEventListener('trainingstart', () => {
        // Give immediate feedback for the otherwise-silent await: a progress
        // cursor while the overlay recomputes, and the count badges wiped to a
        // pulsing placeholder so the stale numbers don't just sit there.
        setTrainingIndicator(state, true);
        setClassBadgesLoading();
    });

    q.addEventListener('trainingcomplete', () => setTrainingIndicator(state, false));

    q.addEventListener('statscomputed', (e) => {
        const { counts, animate } = e.detail;
        if (animate) animateClassStatBadges(counts);
        else         updateClassStatBadges(counts);
    });

    syncUI(state); // seed the initial empty/export state
}

/**
 * Refreshes all image-count-driven chrome (export button, empty state, save
 * indicator) in one call. Invoke after any change to state.images.
 * @param {Object} state - Shared app state.
 */
export function syncUI(state) {
    updateExportButtonCount(state);
    refreshEmptyState(state);
    updateSaveIndicator(state);
}

/**
 * Toggles the global "training in progress" affordance: a progress cursor via a
 * body class. Gives feedback while the overlay/probabilities recompute, which
 * is otherwise a silent await.
 * @param {Object} _state - Shared app state (unused; kept for call-site symmetry).
 * @param {boolean} on - Whether training is currently running.
 */
export function setTrainingIndicator(_state, on) {
    document.body.classList.toggle('training', on);
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

/** Shows/hides the centered empty-state drop target. @param {Object} state */
export function refreshEmptyState(state) {
    const el = document.getElementById('empty-state');
    if (el) el.style.display = state.images.length === 0 ? '' : 'none';
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
        // badge.textContent = '....';
    });
}

/**
 * Animates each class badge counting up from 0 to its final value, as if the objects
 * are being tallied live. Clears the loading placeholder. A single rAF loop drives all
 * badges; a subsequent retrain cancels it via cancelCountup (see setClassBadgesLoading).
 * @param {Array<number>} counts - Final per-class object counts.
 */
export function animateClassStatBadges(counts) {
    cancelCountup();
    const badges = counts.map((_, i) => document.getElementById(`stat-class-${i}`));
    badges.forEach(b => b && b.classList.remove('loading'));

    // Bigger counts should take longer to count to.
    const durations = counts.map((c) => Math.log10(c === 0 ? 1 : c) * 200)
    const start = performance.now();
    const easeOutCubic = (t) => 1 - Math.pow(1 - t, 3);

    const step = (now) => {
        let keepAnimating = false;

        counts.forEach((count, i) => {
            const t = Math.min(1, (now - start) / durations[i]);
            const ease = easeOutCubic(t)
            const badge = badges[i];
            if (badge) badge.textContent = Math.round(count * ease);

            // If any badge is still going, we need to request another frame
            if (t < 1) {
                keepAnimating = true;
            }
        });

        // Only call requestAnimationFrame ONCE per tick
        if (keepAnimating) {
            _countupRaf = requestAnimationFrame(step);
        } else {
            _countupRaf = null;
        }
    };

    _countupRaf = requestAnimationFrame(step);
}
