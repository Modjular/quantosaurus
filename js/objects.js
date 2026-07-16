// Pure, dependency-free helpers for turning a backend's dense per-object stats
// buffer into object records and a CSV. Kept separate from export.js (which
// imports vendored itk-wasm/jszip and so can't be imported under Node) so this
// logic stays unit-testable — see objects.test.mjs.
//
// Following the app convention these are pure functions with no DOM/GPU access.
import { STATS_LAYOUT } from './config.js';

/**
 * Decode a dense per-object stats buffer (backend.downloadStats) into object
 * records. The dense struct is STATS_LAYOUT.denseCount u32 per object:
 *   [label, area, total_lo, total_hi, sumx_lo, sumx_hi, sumy_lo, sumy_hi,
 *    min_intensity, max_intensity]
 * Summed fields are 64-bit split across lo/hi u32 words; reassemble as
 * hi*2^32 + lo. Intensities are fixed-point (raw * range.scale) so they're
 * divided by `scale` back to real units; centroids are pixel coords (no descale).
 * Objects with area 0 (absent labels) are skipped.
 * @param {Uint32Array} data - Dense stats buffer.
 * @param {number} [scale=1] - range.scale used by the fixed-point accumulator.
 * @returns {Array<{label:number, area:number, cx:number, cy:number,
 *   minIntensity:number, meanIntensity:number, maxIntensity:number}>}
 */
export function decodeObjectStats(data, scale = 1) {
    const n = STATS_LAYOUT.denseCount;
    const u64 = (lo, hi) => hi * 2 ** 32 + lo;
    const s = scale || 1;
    const out = [];
    for (let i = 0; i < data.length; i += n) {
        const area = data[i + 1];
        if (area === 0) continue;
        const total = u64(data[i + 2], data[i + 3]);
        const sumX = u64(data[i + 4], data[i + 5]);
        const sumY = u64(data[i + 6], data[i + 7]);
        out.push({
            label: data[i + 0],
            area,
            cx: sumX / area,
            cy: sumY / area,
            minIntensity: data[i + 8] / s,
            meanIntensity: (total / area) / s,
            maxIntensity: data[i + 9] / s,
        });
    }
    return out;
}

const CSV_COLUMNS = [
    'image', 'class', 'label', 'centroid_x', 'centroid_y',
    'area', 'min_intensity', 'mean_intensity', 'max_intensity',
];

/** Quote a CSV field if it contains a comma, quote, or newline. */
function csvEscape(v) {
    const s = String(v);
    return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
}

/**
 * Build a per-object CSV (one row per object) from records produced by
 * decodeObjectStats, each augmented with `image` and `class` labels by the
 * caller. Centroids get 2 decimals, mean intensity 4; area/min/max stay integral
 * when they are. Always ends with a trailing newline.
 * @param {Array<Object>} rows - Records with image, class, label, cx, cy, area,
 *   minIntensity, meanIntensity, maxIntensity.
 * @returns {string} CSV text (header + rows).
 */
export function buildObjectCsv(rows) {
    const num = (v, dp) => Number.isInteger(v) ? String(v) : v.toFixed(dp);
    const lines = [CSV_COLUMNS.join(',')];
    for (const r of rows) {
        lines.push([
            csvEscape(r.image),
            csvEscape(r.class),
            r.label,
            r.cx.toFixed(2),
            r.cy.toFixed(2),
            r.area,
            num(r.minIntensity, 4),
            r.meanIntensity.toFixed(4),
            num(r.maxIntensity, 4),
        ].join(','));
    }
    return lines.join('\n') + '\n';
}
