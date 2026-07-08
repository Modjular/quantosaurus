import { CAMERA_ZOOM_MIN, CAMERA_ZOOM_MAX, CAMERA_ZOOM_SENSITIVITY } from './config.js';

/**
 * Wires up Figma-style pan/zoom on the canvas board. Mutates `state.camera`
 * ({ x, y, scale }) and applies it as a CSS transform on `board`.
 *
 * Interactions:
 *  - Ctrl/Cmd + wheel: zoom toward the cursor.
 *  - Plain wheel / trackpad: pan.
 *  - Space held (or grab tool) + drag: pan.
 *
 * @param {Object} state - Shared app state; reads/writes state.camera, state.isSpaceDown, state.isPanning, state.toolMode.
 * @param {HTMLElement} viewport - The clipping element that receives wheel/mouse events.
 * @param {HTMLElement} board - The transformed content element.
 */
export function setupCamera(state, viewport, board) {
    function updateBoard() {
        board.style.transform =
            `translate(${state.camera.x}px, ${state.camera.y}px) ` +
            `scale(${state.camera.scale})`;
    }

    viewport.addEventListener('wheel', (e) => {
        e.preventDefault();
        if (e.ctrlKey || e.metaKey) {
            const delta    = -e.deltaY * CAMERA_ZOOM_SENSITIVITY;
            const newScale = Math.min(
                Math.max(CAMERA_ZOOM_MIN, state.camera.scale * Math.exp(delta)),
                CAMERA_ZOOM_MAX
            );
            // Zoom toward the cursor: find the board-space point under the mouse
            // before scaling, then re-solve the translation so that same point
            // stays under the mouse after scaling.
            const rect   = viewport.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;
            const boardX = (mouseX - state.camera.x) / state.camera.scale;
            const boardY = (mouseY - state.camera.y) / state.camera.scale;
            state.camera.scale = newScale;
            state.camera.x = mouseX - boardX * state.camera.scale;
            state.camera.y = mouseY - boardY * state.camera.scale;
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

    viewport.addEventListener('mousedown', () => {
        if (state.isSpaceDown || state.toolMode === 'grab') {
            state.isPanning = true;
            document.body.classList.add('panning');
        }
    });
    window.addEventListener('mousemove', (e) => {
        if (state.isPanning) {
            state.camera.x += e.movementX;
            state.camera.y += e.movementY;
            updateBoard();
        }
    });
    window.addEventListener('mouseup', () => {
        state.isPanning = false;
        document.body.classList.remove('panning');
    });
}
