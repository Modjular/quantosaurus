// Tests for decodeObjectStats + buildObjectCsv — the pure CPU helpers that turn a
// backend's dense stats buffer into object records and a CSV. Dependency-free —
// run with: node js/objects.test.mjs
import { decodeObjectStats, buildObjectCsv } from './objects.js';
import { STATS_LAYOUT } from './config.js';

let failures = 0;
function assert(cond, msg) {
    if (cond) { console.log(`  ok  ${msg}`); }
    else { failures++; console.error(`FAIL  ${msg}`); }
}

const N = STATS_LAYOUT.denseCount; // 10

// Build one dense object record. Summed fields are split lo/hi (hi*2^32 + lo).
function makeObject({ label, area, total, sumx, sumy, min, max }) {
    const lo = (v) => v >>> 0;
    const hi = (v) => Math.floor(v / 2 ** 32) >>> 0;
    return [label, area, lo(total), hi(total), lo(sumx), hi(sumx),
            lo(sumy), hi(sumy), min, max];
}

// --- decodeObjectStats ---
{
    // Two objects; one with a 64-bit sum that exceeds 2^32 to exercise lo/hi.
    const big = 5_000_000_000; // > 2^32, spans both words
    const a = makeObject({ label: 1, area: 100, total: 2000, sumx: 1000, sumy: 3000, min: 5, max: 40 });
    const b = makeObject({ label: 2, area: 2, total: big, sumx: big, sumy: 20, min: 1, max: 9 });
    const data = Uint32Array.from([...a, ...b]);

    const objs = decodeObjectStats(data, 1);
    assert(objs.length === 2, 'decodes two objects');
    assert(objs[0].label === 1 && objs[0].area === 100, 'object 1 label/area');
    assert(objs[0].cx === 10 && objs[0].cy === 30, 'object 1 centroid = sum/area');
    assert(objs[0].meanIntensity === 20, 'object 1 mean = total/area');
    assert(objs[0].minIntensity === 5 && objs[0].maxIntensity === 40, 'object 1 min/max');
    assert(objs[1].cx === big / 2, 'object 2 centroid uses reassembled 64-bit sum');
    assert(objs[1].meanIntensity === big / 2, 'object 2 mean uses 64-bit total');
}

// area 0 => absent label, skipped
{
    const present = makeObject({ label: 1, area: 10, total: 100, sumx: 50, sumy: 50, min: 1, max: 20 });
    const absent  = makeObject({ label: 2, area: 0, total: 0, sumx: 0, sumy: 0, min: 0, max: 0 });
    const objs = decodeObjectStats(Uint32Array.from([...present, ...absent]), 1);
    assert(objs.length === 1 && objs[0].label === 1, 'area-0 objects are skipped');
}

// scale descales intensity fields only, not centroids/area
{
    const o = makeObject({ label: 1, area: 4, total: 400, sumx: 20, sumy: 40, min: 100, max: 200 });
    const objs = decodeObjectStats(Uint32Array.from(o), 10); // scale 10 (e.g. float image)
    assert(objs[0].minIntensity === 10 && objs[0].maxIntensity === 20, 'intensities divided by scale');
    assert(objs[0].meanIntensity === 10, 'mean intensity divided by scale (400/4/10)');
    assert(objs[0].cx === 5 && objs[0].cy === 10 && objs[0].area === 4, 'centroid/area not descaled');
}

// scale of 0 is treated as 1 (defensive)
{
    const o = makeObject({ label: 1, area: 2, total: 20, sumx: 2, sumy: 2, min: 4, max: 8 });
    const objs = decodeObjectStats(Uint32Array.from(o), 0);
    assert(objs[0].maxIntensity === 8, 'scale 0 falls back to 1 (no divide-by-zero)');
}

// --- buildObjectCsv ---
{
    const rows = [
        { image: 'a.tif', class: 'cell', label: 1, cx: 10.5, cy: 20.25, area: 100,
          minIntensity: 5, meanIntensity: 20, maxIntensity: 40 },
        { image: 'weird,name.tif', class: 'Class 1', label: 2, cx: 1, cy: 2, area: 3,
          minIntensity: 0, meanIntensity: 1.23456, maxIntensity: 9 },
    ];
    const csv = buildObjectCsv(rows);
    const lines = csv.trimEnd().split('\n');
    assert(lines[0] === 'image,class,label,centroid_x,centroid_y,area,min_intensity,mean_intensity,max_intensity',
        'header row is correct');
    assert(lines.length === 3, 'header + 2 data rows');
    assert(lines[1] === 'a.tif,cell,1,10.50,20.25,100,5,20.0000,40', 'row 1 formatting (centroid 2dp, mean 4dp)');
    assert(lines[2].startsWith('"weird,name.tif"'), 'field with comma is quoted');
    assert(csv.endsWith('\n'), 'ends with trailing newline');
}

// empty input => header only
{
    const csv = buildObjectCsv([]);
    assert(csv.trimEnd().split('\n').length === 1, 'empty rows yields header-only CSV');
}

console.log(failures === 0 ? '\nALL PASS' : `\n${failures} FAILURE(S)`);
process.exit(failures === 0 ? 0 : 1);
