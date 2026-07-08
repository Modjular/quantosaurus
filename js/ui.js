export function syncUI(state) {
    updateStatus(state);
    updateExportButtonCount(state);
    refreshEmptyState(state);
}

export function updateStatus(state) {
    const n = state.images.length;
    document.getElementById('status').innerText =
        n === 0 ? 'No images loaded. Drag & drop onto the canvas.'
                : `${n} image${n !== 1 ? 's' : ''} loaded.`;
}

export function updateExportButtonCount(state) {
    const btn = document.getElementById('btnExportAll');
    const n   = state.images.length;
    btn.disabled  = n === 0;
    btn.innerText = n > 0 ? `Export Loaded Images (${n})` : 'Export Loaded Images';
}

export function refreshEmptyState(state) {
    document.getElementById('img-empty').style.display =
        state.images.length === 0 ? '' : 'none';
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

    const btnDel = document.createElement('button');
    btnDel.className   = 'img-delete';
    btnDel.textContent = '✕';
    btnDel.title       = 'Remove image';
    btnDel.onclick     = () => onDelete();

    header.appendChild(reorder);
    header.appendChild(nameEl);
    header.appendChild(btnDel);

    row.appendChild(loadBar);
    row.appendChild(header);

    return row;
}
