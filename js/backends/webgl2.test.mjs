// Tests for the pure CCL + stats helpers in webgl2.js (cclLabel, accumulateStats).
// Dependency-free — run with: node js/backends/webgl2.test.mjs
//
// webgl2.js also contains browser-only code (document, WebGL), but only at
// call time; importing it under plain Node to reach these two pure functions
// is safe since this file never calls WebGl2Backend's methods.
import { cclLabel, accumulateStats } from './webgl2.js';

let failures = 0;
function assert(cond, msg) {
    if (cond) { console.log(`  ok  ${msg}`); }
    else { failures++; console.error(`FAIL  ${msg}`); }
}

function maskFromRows(rows) {
    const h = rows.length, w = rows[0].length;
    const mask = new Uint8Array(w * h);
    for (let y = 0; y < h; y++)
        for (let x = 0; x < w; x++)
            mask[y * w + x] = rows[y][x] === '#' ? 1 : 0;
    return { mask, w, h };
}

function componentInfo(labels) {
    const comps = new Map(); // label -> pixel indices
    labels.forEach((l, i) => {
        if (l === 0) return;
        if (!comps.has(l)) comps.set(l, []);
        comps.get(l).push(i);
    });
    return comps;
}

function checkMinIndexLabeling(labels, name) {
    const comps = componentInfo(labels);
    for (const [label, idxs] of comps) {
        const minIdx = Math.min(...idxs);
        assert(label === minIdx + 1, `${name}: component label ${label} equals min pixel index + 1 (${minIdx + 1})`);
    }
    return comps;
}

// ---- Test 1: spiral (single component, long winding path) ----
{
    // Single inward-winding path: (0,0) right along row0, down col8, left along
    // row6, up col0, right along row2, down col6, left along row4 ending at (2,4).
    const { mask, w, h } = maskFromRows([
        '#########',
        '........#',
        '#######.#',
        '#.....#.#',
        '#.#####.#',
        '#.......#',
        '#########',
    ]);
    const labels = cclLabel(mask, w, h);
    const comps = checkMinIndexLabeling(labels, 'spiral');
    assert(comps.size === 1, `spiral: exactly 1 component (got ${comps.size})`);
    const fg = mask.reduce((a, b) => a + b, 0);
    assert([...comps.values()][0].length === fg, `spiral: component covers all ${fg} foreground pixels`);
}

// ---- Test 2: full-image serpentine line (worst case for propagation CCL) ----
{
    const w = 64, h = 64;
    const mask = new Uint8Array(w * h);
    // boustrophedon path: full rows connected alternately at right/left edges
    for (let y = 0; y < h; y += 2) {
        for (let x = 0; x < w; x++) mask[y * w + x] = 1;
        if (y + 1 < h) mask[(y + 1) * w + ((y / 2) % 2 === 0 ? w - 1 : 0)] = 1;
    }
    const labels = cclLabel(mask, w, h);
    const comps = checkMinIndexLabeling(labels, 'serpentine');
    assert(comps.size === 1, `serpentine: exactly 1 component (got ${comps.size})`);
}

// ---- Test 3: diagonal-only touch stays 2 components (4-connectivity) ----
{
    const { mask, w, h } = maskFromRows([
        '#.',
        '.#',
    ]);
    const labels = cclLabel(mask, w, h);
    const comps = checkMinIndexLabeling(labels, 'diagonal');
    assert(comps.size === 2, `diagonal: 2 components under 4-connectivity (got ${comps.size})`);
}

// ---- Test 4: small blobs + stats ----
{
    const { mask, w, h } = maskFromRows([
        '##..#',
        '##..#',
        '.....',
        '..##.',
    ]);
    const labels = cclLabel(mask, w, h);
    const comps = checkMinIndexLabeling(labels, 'blobs');
    assert(comps.size === 3, `blobs: 3 components (got ${comps.size})`);

    // intensity: pixel index as fixed-point value, easy to reason about
    const intensity = new Uint32Array(w * h);
    for (let i = 0; i < w * h; i++) intensity[i] = i * 10;

    const stats = accumulateStats(labels, intensity, w, h);
    const SC = 7;
    assert(stats.length === 3 * SC, `blobs: stats has 3 dense structs (got ${stats.length / SC})`);

    const byLabel = new Map();
    for (let s = 0; s < stats.length; s += SC) byLabel.set(stats[s], stats.slice(s, s + SC));

    // 2x2 blob at (0,0)-(1,1): label 1, area 4, pixels 0,1,5,6
    let [label, area, total, sumX, sumY, min, max] = byLabel.get(1);
    assert(area === 4, `blob1: area 4 (got ${area})`);
    assert(total === (0 + 1 + 5 + 6) * 10, `blob1: total intensity (got ${total})`);
    assert(sumX === 0 + 1 + 0 + 1 && sumY === 0 + 0 + 1 + 1, `blob1: centroid sums (got ${sumX},${sumY})`);
    assert(min === 0 && max === 60, `blob1: min/max intensity (got ${min},${max})`);

    // 1x2 vertical bar at x=4: label 5 (min index 4), area 2, pixels 4,9
    [label, area, total, sumX, sumY, min, max] = byLabel.get(5);
    assert(area === 2, `blob2: area 2 (got ${area})`);
    assert(total === (4 + 9) * 10, `blob2: total intensity (got ${total})`);

    // 2x1 bar at row 3, x=2..3: pixels 17,18 -> label 18
    [label, area, total, sumX, sumY, min, max] = byLabel.get(18);
    assert(area === 2, `blob3: area 2 (got ${area})`);
    assert(sumX === 2 + 3 && sumY === 3 + 3, `blob3: centroid sums (got ${sumX},${sumY})`);
}

// ---- Test 5: empty mask ----
{
    const labels = cclLabel(new Uint8Array(16), 4, 4);
    assert(labels.every(l => l === 0), 'empty: all labels 0');
    const stats = accumulateStats(labels, new Uint32Array(16), 4, 4);
    assert(stats.length === 0, 'empty: stats empty');
}

// ---- Test 6: randomized cross-check against BFS flood fill ----
{
    for (let trial = 0; trial < 50; trial++) {
        const w = 37, h = 29;
        const mask = new Uint8Array(w * h);
        for (let i = 0; i < w * h; i++) mask[i] = Math.random() < 0.55 ? 1 : 0;

        const labels = cclLabel(mask, w, h);

        // reference: BFS flood fill, 4-connectivity, min-index labels
        const ref = new Uint32Array(w * h);
        for (let start = 0; start < w * h; start++) {
            if (!mask[start] || ref[start]) continue;
            const queue = [start];
            ref[start] = start + 1;
            while (queue.length) {
                const i = queue.pop();
                const x = i % w, y = (i / w) | 0;
                for (const [nx, ny] of [[x - 1, y], [x + 1, y], [x, y - 1], [x, y + 1]]) {
                    if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
                    const j = ny * w + nx;
                    if (mask[j] && !ref[j]) { ref[j] = start + 1; queue.push(j); }
                }
            }
        }
        // BFS in raster order gives min-index labels too; compare exactly
        let same = true;
        for (let i = 0; i < w * h; i++) if (labels[i] !== ref[i]) { same = false; break; }
        if (!same) { failures++; console.error(`FAIL random trial ${trial}: labels differ from BFS reference`); break; }
    }
    if (failures === 0) console.log('  ok  random: 50 trials match BFS flood-fill reference exactly');
}

console.log(failures === 0 ? '\nALL TESTS PASSED' : `\n${failures} FAILURES`);
process.exit(failures === 0 ? 0 : 1);
