import { state } from './state.js';
import { deleteImage, reorderImage, onAxisChange, onSliceChange } from './imageManager.js';

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
export function createImageRow(imgId, name) {
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
    btnUp.onclick   = () => reorderImage(imgId, -1);
    btnDown.onclick = () => reorderImage(imgId, +1);
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
    btnDel.onclick     = () => deleteImage(imgId);

    header.appendChild(reorder);
    header.appendChild(nameEl);
    header.appendChild(badge);
    header.appendChild(btnDel);

    row.appendChild(loadBar);
    row.appendChild(header);

    return row;
}

// Called after load completes — attaches axis / slice UI placeholders.
export function finaliseImageRow(row, imgState) {
    const axesEl = document.createElement('div');
    axesEl.className    = 'img-axes';
    axesEl.dataset.axes = imgState.id;

    const sliceEl = document.createElement('div');
    sliceEl.className     = 'img-slice';
    sliceEl.dataset.slice = imgState.id;

    updateImageStateBadge(imgState);

    row.appendChild(axesEl);
    row.appendChild(sliceEl);
}

// showAxesUI — populates and reveals axis selectors + slice sliders
export function showAxesUI(row, imgState) {
    const { shape, axes, nonDisplayDims, sliceIndices } = imgState;
    const axesEl = row.querySelector('.img-axes');
    const sliceEl = row.querySelector('.img-slice');
    axesEl.innerHTML = '';
    sliceEl.innerHTML = '';

    const dimLabels = shape.map((size, i) => `Dim ${i} (${size})`);

    function makeAxisSelect(role, currentAxisIdx) {
        const wrapper = document.createElement('div');
        wrapper.className = 'img-axes-row';
        const lbl = document.createElement('label');
        lbl.textContent = role;
        const sel = document.createElement('select');
        shape.forEach((size, i) => {
            const opt = document.createElement('option');
            opt.value = i;
            opt.textContent = dimLabels[i];
            if (i === currentAxisIdx) opt.selected = true;
            sel.appendChild(opt);
        });
        sel.onchange = () => onAxisChange(imgState, role, parseInt(sel.value, 10));
        wrapper.appendChild(lbl);
        wrapper.appendChild(sel);
        return wrapper;
    }

    axesEl.appendChild(makeAxisSelect('Y', axes.axisY));
    axesEl.appendChild(makeAxisSelect('X', axes.axisX));
    axesEl.classList.add('visible');

    nonDisplayDims.forEach(({ i: dimIdx, size }, arrIdx) => {
        const wrapper = document.createElement('div');
        wrapper.className = 'img-slice-row';

        const lbl = document.createElement('label');
        lbl.textContent = `D${dimIdx}`;

        const slider = document.createElement('input');
        slider.type  = 'range';
        slider.min   = 0;
        slider.max   = size - 1;
        slider.value = sliceIndices[arrIdx];
        slider.dataset.dimIdx = dimIdx;

        const valSpan = document.createElement('span');
        valSpan.textContent = `${sliceIndices[arrIdx]}/${size - 1}`;

        slider.oninput = () => {
            const v = parseInt(slider.value, 10);
            imgState.sliceIndices[arrIdx] = v;
            valSpan.textContent = `${v}/${size - 1}`;
            onSliceChange(imgState);
        };

        wrapper.appendChild(lbl);
        wrapper.appendChild(slider);
        wrapper.appendChild(valSpan);
        sliceEl.appendChild(wrapper);
    });
    sliceEl.classList.add('visible');
}
