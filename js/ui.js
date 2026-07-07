import { state } from './state.js';

export function syncUI() {
    updateStatus();
    updateExportButtonCount();
    refreshEmptyState();
}

export function updateStatus() {
    const n = state.images.length;
    document.getElementById('status').innerText =
        n === 0 ? 'No images loaded. Drag & drop onto the canvas.'
                : `${n} image${n !== 1 ? 's' : ''} loaded.`;
}

export function updateExportButtonCount() {
    const btn = document.getElementById('btnExportAll');
    const n   = state.images.length;
    btn.disabled  = n === 0;
    btn.innerText = n > 0 ? `Export Loaded Images (${n})` : 'Export Loaded Images';
}

export function refreshEmptyState() {
    document.getElementById('img-empty').style.display =
        state.images.length === 0 ? '' : 'none';
}

export function updateImageStateBadge(imgState) {
    const badge = imgState._sidebarRow.querySelector('.img-label-badge');
    const count = imgState.labels.length;

    if (!badge) return;

    badge.textContent = count === 0 ? 'no labels' : `${count} label${count !== 1 ? 's' : ''}`;
    badge.classList.toggle('has-labels', count > 0);
}

// createImageRow — builds the DOM for one entry.
// The controller supplies { onReorder(direction), onDelete() } so this view
// stays decoupled from the image-lifecycle module.
export function createImageRow(imgId, name, { onReorder, onDelete }) {
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

    const badge = document.createElement('div');
    badge.className = 'img-label-badge';
    badge.dataset.badge = imgId;

    const btnDel = document.createElement('button');
    btnDel.className   = 'img-delete';
    btnDel.textContent = '✕';
    btnDel.title       = 'Remove image';
    btnDel.onclick     = () => onDelete();

    header.appendChild(reorder);
    header.appendChild(nameEl);
    header.appendChild(badge);
    header.appendChild(btnDel);

    row.appendChild(loadBar);
    row.appendChild(header);

    return row;
}
