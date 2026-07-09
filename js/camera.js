import { CAMERA_ZOOM_MIN, CAMERA_ZOOM_MAX, CAMERA_ZOOM_SENSITIVITY } from './config.js';

/**
 * Wires up Figma-style pan/zoom on the canvas board. Mutates `state.camera`
 * ({ x, y, scale }) and applies it as a CSS transform on `board`.
 *
 * Interactions:
 *  - Ctrl/Cmd + wheel: zoom toward the cursor.
 *  - Plain wheel / trackpad: pan.
 *  - Space held (or grab tool) + drag: pan (mouse or single touch).
 *  - Two-finger touch: pinch to zoom toward the pinch midpoint, drag to pan —
 *    works regardless of tool, since touch has no modifier key to gate a
 *    dedicated zoom gesture behind.
 *
 * All pointer types (mouse, touch, pen) go through the Pointer Events API so
 * mouse and touch share one code path. `state.activePointerCount` tracks how
 * many pointers are currently down so other modules (images.js's painting)
 * can tell when a second touch joins mid-gesture and back off.
 *
 * @param {Object} state - Shared app state; reads/writes state.camera, state.isSpaceDown, state.isPanning, state.toolMode, state.activePointerCount.
 * @param {HTMLElement} viewport - The clipping element that receives wheel/pointer events.
 * @param {HTMLElement} board - The transformed content element.
 */
export function setupCamera(state, viewport, board) {
    function updateBoard() {
        board.style.transform =
            `translate(${state.camera.x}px, ${state.camera.y}px) ` +
            `scale(${state.camera.scale})`;
    }

    // Re-anchors the camera so that the board-space point under (beforeX, beforeY)
    // ends up under (afterX, afterY) once scale changes to newScale. A plain zoom
    // (wheel, or a pinch's zoom component) passes the same point for before/after;
    // a pinch also passes its moving midpoint to combine pan with zoom in one step.
    function applyZoomPan(beforeX, beforeY, afterX, afterY, newScale) {
        const rect = viewport.getBoundingClientRect();
        const boardX = (beforeX - rect.left - state.camera.x) / state.camera.scale;
        const boardY = (beforeY - rect.top  - state.camera.y) / state.camera.scale;
        state.camera.scale = newScale;
        state.camera.x = (afterX - rect.left) - boardX * state.camera.scale;
        state.camera.y = (afterY - rect.top)  - boardY * state.camera.scale;
    }

    viewport.addEventListener('wheel', (e) => {
        e.preventDefault();
        if (e.ctrlKey || e.metaKey) {
            const delta    = -e.deltaY * CAMERA_ZOOM_SENSITIVITY;
            const newScale = Math.min(
                Math.max(CAMERA_ZOOM_MIN, state.camera.scale * Math.exp(delta)),
                CAMERA_ZOOM_MAX
            );
            applyZoomPan(e.clientX, e.clientY, e.clientX, e.clientY, newScale);
        } else {
            state.camera.x -= e.deltaX;
            state.camera.y -= e.deltaY;
        }
        updateBoard();
    }, { passive: false });

    window.addEventListener('keydown', (e) => {
        if (e.code === 'Space' && !state.isSpaceDown) {
            state.isSpaceDown = true;
            document.body.classList.add('space-down');
        }
    });
    window.addEventListener('keyup', (e) => {
        if (e.code === 'Space') {
            state.isSpaceDown = false;
            state.isPanning   = false;
            document.body.classList.remove('space-down', 'panning');
        }
    });

    // Tracks every pointer currently down (mouse click, or each touch contact),
    // keyed by pointerId, in viewport-relative client coordinates. Drives both
    // single-pointer pan and two-pointer pinch-zoom below.
    const activePointers = new Map();
    let pinchLastDist = null;
    let pinchLastMid  = null;

    function distance(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }
    function midpoint(a, b) { return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }; }

    viewport.addEventListener('pointerdown', (e) => {
        activePointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
        state.activePointerCount = activePointers.size;

        if (activePointers.size === 1) {
            if (state.isSpaceDown || state.toolMode === 'grab') {
                state.isPanning = true;
                document.body.classList.add('panning');
                viewport.setPointerCapture(e.pointerId);
            }
        } else if (activePointers.size === 2) {
            // A second touch always means "start pinching", regardless of tool —
            // cancel any single-pointer pan in favor of the pinch gesture.
            state.isPanning = false;
            document.body.classList.remove('panning');
            const [p1, p2] = [...activePointers.values()];
            pinchLastDist = distance(p1, p2);
            pinchLastMid  = midpoint(p1, p2);
        }
    });

    window.addEventListener('pointermove', (e) => {
        if (!activePointers.has(e.pointerId)) return;
        const prev = activePointers.get(e.pointerId);
        activePointers.set(e.pointerId, { x: e.clientX, y: e.clientY });

        if (activePointers.size === 2) {
            const [p1, p2] = [...activePointers.values()];
            const dist = distance(p1, p2);
            const mid  = midpoint(p1, p2);

            if (pinchLastDist != null) {
                const scaleRatio = dist / pinchLastDist;
                const newScale   = Math.min(
                    Math.max(CAMERA_ZOOM_MIN, state.camera.scale * scaleRatio),
                    CAMERA_ZOOM_MAX
                );
                applyZoomPan(pinchLastMid.x, pinchLastMid.y, mid.x, mid.y, newScale);
                updateBoard();
            }
            pinchLastDist = dist;
            pinchLastMid  = mid;
        } else if (activePointers.size === 1 && state.isPanning) {
            state.camera.x += e.clientX - prev.x;
            state.camera.y += e.clientY - prev.y;
            updateBoard();
        }
    });

    function releasePointer(e) {
        if (!activePointers.has(e.pointerId)) return;
        activePointers.delete(e.pointerId);
        state.activePointerCount = activePointers.size;

        if (activePointers.size === 0) {
            state.isPanning = false;
            document.body.classList.remove('panning');
        } else if (activePointers.size === 1) {
            // Dropped from a pinch back to one finger — stop pinch tracking,
            // but don't auto-resume a single-finger pan, that would jump.
            pinchLastDist = null;
            pinchLastMid  = null;
        }
    }
    window.addEventListener('pointerup', releasePointer);
    window.addEventListener('pointercancel', releasePointer);
}
