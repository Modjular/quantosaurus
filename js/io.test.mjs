// Tests for the pure computeMinMax and deinterleaveChannels helpers in io.js.
// Dependency-free — run with: node js/io.test.mjs
//
// io.js also contains browser/itk-bound code (loadFileIntoArray), but the
// top-level vendor import has no side effects that touch the DOM, so importing
// it under plain Node to reach these pure helpers is safe. loadFileIntoArray
// itself needs itk-wasm/DOM and is exercised in the browser end-to-end instead.
import { computeMinMax, deinterleaveChannels } from './io.js';

let failures = 0;
function assert(cond, msg) {
    if (cond) { console.log(`  ok  ${msg}`); }
    else { failures++; console.error(`FAIL  ${msg}`); }
}

// ---- Test 1: raw values are reported unchanged (no 0–1 stretch) ----
{
    // A uint16-style range: the helper must return the true magnitudes so the
    // contrast slider and stats can work in real units.
    const { dataMin, dataMax } = computeMinMax(new Float32Array([0, 4000, 65535]));
    assert(dataMin === 0, `raw min is 0 (got ${dataMin})`);
    assert(dataMax === 65535, `raw max is 65535, not normalized (got ${dataMax})`);
}

// ---- Test 2: negative values ----
{
    const { dataMin, dataMax } = computeMinMax(new Float32Array([-10, 0, 10]));
    assert(dataMin === -10 && dataMax === 10, `negatives: min -10 / max 10 (got ${dataMin}/${dataMax})`);
}

// ---- Test 3: constant image ----
{
    const { dataMin, dataMax } = computeMinMax(new Float32Array([7, 7, 7, 7]));
    assert(dataMin === 7 && dataMax === 7, `constant: min === max === 7 (got ${dataMin}/${dataMax})`);
}

// ---- Test 4: empty array falls back to 0/0 (no Infinity) ----
{
    const { dataMin, dataMax } = computeMinMax(new Float32Array(0));
    assert(dataMin === 0 && dataMax === 0, `empty: 0/0 fallback, no Infinity (got ${dataMin}/${dataMax})`);
}

// ---- Test 5: single channel is a zero-copy identity ----
{
    const buf = new Float32Array([1, 2, 3, 4]);
    const planes = deinterleaveChannels(buf, 2, 2, 1);
    assert(planes.length === 1, `C=1 yields one plane (got ${planes.length})`);
    assert(planes[0] === buf, `C=1 returns the same Float32Array (no copy)`);
}

// ---- Test 6: non-Float32 single channel is widened to Float32Array ----
{
    const buf = Uint16Array.from([0, 4000, 65535, 7]);
    const planes = deinterleaveChannels(buf, 2, 2, 1);
    assert(planes[0] instanceof Float32Array, `C=1 widens integer input to Float32Array`);
    assert(planes[0].length === 4 && planes[0][2] === 65535, `C=1 values preserved (got ${planes[0][2]})`);
}

// ---- Test 7: interleaved RGB de-interleaves into stride-separated planes ----
{
    // 2x1 image, 3 channels, pixel-interleaved: [r0,g0,b0, r1,g1,b1].
    const buf = new Float32Array([10, 20, 30,  11, 21, 31]);
    const planes = deinterleaveChannels(buf, 2, 1, 3);
    assert(planes.length === 3, `C=3 yields three planes (got ${planes.length})`);
    assert(planes[0][0] === 10 && planes[0][1] === 11, `plane 0 = [10, 11] (got [${planes[0]}])`);
    assert(planes[1][0] === 20 && planes[1][1] === 21, `plane 1 = [20, 21] (got [${planes[1]}])`);
    assert(planes[2][0] === 30 && planes[2][1] === 31, `plane 2 = [30, 31] (got [${planes[2]}])`);
    assert(planes.every(p => p.length === 2), `each plane has w*h = 2 values`);
    // Per-channel min/max are independent — what buildImageEntry seeds each contrast window from.
    assert(computeMinMax(planes[2]).dataMax === 31, `plane 2 max is 31 (got ${computeMinMax(planes[2]).dataMax})`);
}

console.log(failures === 0 ? '\nALL TESTS PASSED' : `\n${failures} FAILURES`);
process.exit(failures === 0 ? 0 : 1);
