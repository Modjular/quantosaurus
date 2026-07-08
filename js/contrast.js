// A single shared popover reused for whichever image row is active — napari-style
// per-layer contrast limits. The popover edits the image's display-only window
// (black/white points) via backend.setWindow, which only re-runs the composite
// pass: no feature recompute, no retrain, classifier untouched.

let popoverEl = null;   // the floating panel, created lazily
let activeImgId = null; // image the popover currently targets, or null when closed

// oninput fires on every pointer-move tick while dragging a range input, which can
// outpace the display's refresh rate. backend.setWindow is now cheap (a uniform
// write + redraw, no shader recompile — see webgpu.js), but there's still no reason
// to redraw more than once per frame, so coalesce bursts of oninput events down to
// one setWindow/render per animation frame via rAF.
let _pendingRaf = null;
let _pendingImg = null;

function scheduleSetWindow(img) {
    _pendingImg = img;
    if (_pendingRaf !== null) return; // a redraw is already queued for this frame
    _pendingRaf = requestAnimationFrame(() => {
        _pendingRaf = null;
        _pendingImg.backend.setWindow(_pendingImg.windowLo, _pendingImg.windowHi);
        _pendingImg = null;
    });
}

function cancelPendingSetWindow() {
    if (_pendingRaf !== null) { cancelAnimationFrame(_pendingRaf); _pendingRaf = null; _pendingImg = null; }
}

/**
 * Opens (or moves) the contrast popover next to the given image's row and binds
 * its controls to that image. Calling it again for the same image toggles it shut.
 * @param {Object} state - Shared app state (reads state.images).
 * @param {string} imgId - Id of the image to edit.
 * @param {HTMLElement} anchorEl - Row element/button to anchor the popover beside.
 */
export function openContrastPopover(state, imgId, anchorEl) {
    // Toggle off if the same row's control is clicked again.
    if (activeImgId === imgId && popoverEl && popoverEl.style.display !== 'none') {
        closeContrastPopover();
        return;
    }

    const img = state.images.find(i => i.id === imgId);
    if (!img) return;

    // Switching images without closing first (clicking another row's contrast
    // button) could otherwise leave a coalesced redraw pending against the image
    // being switched away from.
    cancelPendingSetWindow();

    if (!popoverEl) popoverEl = buildPopover();
    activeImgId = imgId;

    const loInput   = popoverEl.querySelector('.contrast-lo');
    const hiInput   = popoverEl.querySelector('.contrast-hi');
    const loVal     = popoverEl.querySelector('.contrast-lo-val');
    const hiVal     = popoverEl.querySelector('.contrast-hi-val');

    // The handles span the image's real range (e.g. 0–65535 for uint16) rather than a
    // normalized 0–1, so black/white points read in the source's true intensity units.
    const { dtypeMax } = img.range;
    const isInteger = Number.isInteger(dtypeMax) && dtypeMax >= 255;
    const step = isInteger ? 1 : dtypeMax / 1000;
    for (const input of [loInput, hiInput]) {
        input.min = 0;
        input.max = dtypeMax;
        input.step = step;
    }
    const fmt = (v) => isInteger ? Math.round(v).toString() : v.toFixed(3);

    const render = () => {
        loVal.textContent = fmt(img.windowLo);
        hiVal.textContent = fmt(img.windowHi);
        loInput.value = img.windowLo;
        hiInput.value = img.windowHi;
    };

    // Keep lo <= hi (napari clamps the handles so they can't cross). The readout
    // updates immediately (cheap DOM text); the actual GPU redraw is coalesced to
    // once per animation frame via scheduleSetWindow, however fast events fire.
    loInput.oninput = () => {
        img.windowLo = Math.min(parseFloat(loInput.value), img.windowHi);
        scheduleSetWindow(img);
        render();
    };
    hiInput.oninput = () => {
        img.windowHi = Math.max(parseFloat(hiInput.value), img.windowLo);
        scheduleSetWindow(img);
        render();
    };

    render();
    positionPopover(popoverEl, anchorEl);
    popoverEl.style.display = 'block';
}

/** Hides the shared popover, if open. */
export function closeContrastPopover() {
    if (popoverEl) popoverEl.style.display = 'none';
    cancelPendingSetWindow();
    activeImgId = null;
}

// Builds the popover DOM once and wires dismissal (click-outside / Escape).
function buildPopover() {
    const el = document.createElement('div');
    el.className = 'contrast-popover heavy-panel';
    el.innerHTML = `
        <div class="control-group">
            <span class="control-label">Contrast Limits</span>
            <div class="contrast-slider">
                <input type="range" class="contrast-lo" min="0" max="1" step="0.01" value="0">
                <input type="range" class="contrast-hi" min="0" max="1" step="0.01" value="1">
            </div>
            <div class="contrast-readout">
                <span>Black <b class="contrast-lo-val">0.00</b></span>
                <span>White <b class="contrast-hi-val">1.00</b></span>
            </div>
        </div>`;
    // Clicks inside the popover must not bubble to the document dismiss handler.
    el.addEventListener('mousedown', (e) => e.stopPropagation());
    document.body.appendChild(el);

    document.addEventListener('mousedown', () => closeContrastPopover());
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeContrastPopover();
    });

    return el;
}

// Anchors the popover beside the row button and vertically centers it on that
// button so the caret lines up. Defaults to the button's right (the sidebar is a
// fixed-width column, so "right" floats over the viewport); flips left if it would
// overrun the window edge.
const GAP = 10;      // space between button and popover for the caret
const CARET_HALF = 8; // half the caret's height (see .contrast-popover::before)

function positionPopover(el, anchorEl) {
    const rect = anchorEl.getBoundingClientRect();
    el.style.visibility = 'hidden';
    el.style.display = 'block';
    const pw = el.offsetWidth;
    const ph = el.offsetHeight;

    // Horizontal: right of the button, or flipped to the left if it won't fit.
    let left = rect.right + GAP;
    const onLeft = left + pw > window.innerWidth - 8;
    if (onLeft) left = rect.left - pw - GAP;
    el.classList.toggle('on-left', onLeft);

    // Vertical: center on the button, clamped to stay on screen.
    const buttonCenterY = rect.top + rect.height / 2;
    let top = buttonCenterY - ph / 2;
    top = Math.max(8, Math.min(top, window.innerHeight - ph - 8));

    // Point the caret tip at the button center even after clamping.
    el.style.setProperty('--caret-top', (buttonCenterY - top - CARET_HALF) + 'px');
    el.style.left = Math.max(8, left) + 'px';
    el.style.top  = top + 'px';
    el.style.visibility = 'visible';
}
