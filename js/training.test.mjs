// Unit tests for decodeObjects — the CPU-side decoder that turns the backends'
// dense stats buffer (downloadStats) into per-object centroids + areas for the
// centroid-marker overlay. Run with plain Node: `node js/training.test.mjs`.
import { decodeObjects } from './training.js';
import { STATS_LAYOUT } from './config.js';

let failures = 0;
function assert(cond, msg) {
    if (cond) {
        console.log(`  ok  ${msg}`);
    } else {
        console.log(`  FAIL ${msg}`);
        failures++;
    }
}

const N = STATS_LAYOUT.denseCount; // 10 u32 per dense struct

// Build one dense struct: [label, area, total_lo, total_hi, sumx_lo, sumx_hi,
// sumy_lo, sumy_hi, min, max]. Splits 64-bit sums into lo/hi u32 words.
function struct({ label = 0, area, sumX, sumY, total = 0, min = 0, max = 0 }) {
    const lo = (v) => v >>> 0;
    const hi = (v) => Math.floor(v / 2 ** 32) >>> 0;
    return [label, area, lo(total), hi(total), lo(sumX), hi(sumX), lo(sumY), hi(sumY), min, max];
}

// --- basic centroid + area ---
{
    const stats = Uint32Array.from(struct({ area: 4, sumX: 8, sumY: 12 }));
    const objs = decodeObjects(stats);
    assert(objs.length === 1, `single struct decodes to one object (got ${objs.length})`);
    assert(objs[0].area === 4, `area passes through (got ${objs[0].area})`);
    assert(objs[0].cx === 2, `centroid x = sumX/area (got ${objs[0].cx})`);
    assert(objs[0].cy === 3, `centroid y = sumY/area (got ${objs[0].cy})`);
}

// --- 64-bit reassembly: sums that overflow a single u32 ---
{
    const area = 100000;
    const sumX = 30000 * area; // 3.0e9 < 2^32 — fits in the lo word
    const bigSumY = 60000 * area; // 6.0e9 > 2^32 — forces the hi word
    const stats = Uint32Array.from(struct({ area, sumX, sumY: bigSumY }));
    const objs = decodeObjects(stats);
    assert(bigSumY > 2 ** 32, `sumY exceeds 2^32 (got ${bigSumY})`);
    assert(objs[0].cx === 30000, `x centroid reassembled (got ${objs[0].cx})`);
    assert(objs[0].cy === 60000, `y centroid reassembled from hi word (got ${objs[0].cy})`);
}

// --- multiple objects ---
{
    const stats = Uint32Array.from([
        ...struct({ label: 1, area: 2, sumX: 6, sumY: 2 }),
        ...struct({ label: 2, area: 10, sumX: 100, sumY: 50 }),
    ]);
    const objs = decodeObjects(stats);
    assert(objs.length === 2, `two structs decode to two objects (got ${objs.length})`);
    assert(objs[0].cx === 3 && objs[0].cy === 1, `object 0 centroid (got ${objs[0].cx},${objs[0].cy})`);
    assert(objs[1].cx === 10 && objs[1].cy === 5, `object 1 centroid (got ${objs[1].cx},${objs[1].cy})`);
}

// --- area 0 is skipped (guards divide-by-zero / absent labels) ---
{
    const stats = Uint32Array.from([
        ...struct({ area: 0, sumX: 0, sumY: 0 }),
        ...struct({ area: 4, sumX: 8, sumY: 8 }),
    ]);
    const objs = decodeObjects(stats);
    assert(objs.length === 1, `area-0 struct is dropped (got ${objs.length})`);
    assert(objs[0].cx === 2 && objs[0].cy === 2, `remaining object decodes (got ${objs[0].cx},${objs[0].cy})`);
}

// --- empty buffer ---
{
    const objs = decodeObjects(new Uint32Array(0));
    assert(Array.isArray(objs) && objs.length === 0, `empty stats -> empty array (got ${objs.length})`);
}

// sanity: struct width matches the layout consumers rely on
assert(N === 10, `dense struct is 10 u32 (got ${N})`);

console.log(failures === 0 ? '\nALL TESTS PASSED' : `\n${failures} TEST(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
