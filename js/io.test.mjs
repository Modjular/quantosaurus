// Tests for the pure intensityToRGBA helper in io.js.
// Dependency-free — run with: node js/io.test.mjs
//
// io.js also contains browser/itk-bound code (loadFileIntoArray), but the
// top-level vendor import has no side effects that touch the DOM, so importing
// it under plain Node to reach intensityToRGBA is safe.
import { intensityToRGBA } from './io.js';

let failures = 0;
function assert(cond, msg) {
    if (cond) { console.log(`  ok  ${msg}`); }
    else { failures++; console.error(`FAIL  ${msg}`); }
}

// ---- Test 1: min/max normalization ----
{
    const data = new Float32Array([0, 5, 10]); // min 0, max 10
    const rgba = new Uint8Array(3 * 4);
    const intensity = new Float32Array(3);
    intensityToRGBA(data, rgba, intensity);

    assert(Math.abs(intensity[0] - 0) < 1e-6, 'norm: min maps to 0');
    assert(Math.abs(intensity[1] - 0.5) < 1e-6, 'norm: midpoint maps to 0.5');
    assert(Math.abs(intensity[2] - 1) < 1e-6, 'norm: max maps to 1');

    // rgba: grayscale (r=g=b), alpha opaque
    assert(rgba[0] === 0 && rgba[1] === 0 && rgba[2] === 0 && rgba[3] === 255, 'rgba: min pixel is black, opaque');
    assert(rgba[8] === 255 && rgba[9] === 255 && rgba[10] === 255 && rgba[11] === 255, 'rgba: max pixel is white, opaque');
    assert(rgba[4] === rgba[5] && rgba[5] === rgba[6], 'rgba: channels are equal (grayscale)');
    // val8 = 0.5 * 255 = 127.5; Uint8Array assignment truncates toward zero -> 127.
    assert(rgba[4] === 127, `rgba: midpoint truncates to 127 (got ${rgba[4]})`);
}

// ---- Test 2: negative values normalize correctly ----
{
    const data = new Float32Array([-10, 0, 10]); // range 20
    const rgba = new Uint8Array(3 * 4);
    const intensity = new Float32Array(3);
    intensityToRGBA(data, rgba, intensity);
    assert(Math.abs(intensity[0] - 0) < 1e-6 && Math.abs(intensity[1] - 0.5) < 1e-6 && Math.abs(intensity[2] - 1) < 1e-6,
        'negatives: -10/0/10 -> 0/0.5/1');
}

// ---- Test 3: constant image (max === min) uses the range=255 fallback ----
{
    const data = new Float32Array([7, 7, 7, 7]);
    const rgba = new Uint8Array(4 * 4);
    const intensity = new Float32Array(4);
    intensityToRGBA(data, rgba, intensity);

    let finite = true;
    for (let i = 0; i < intensity.length; i++) if (!Number.isFinite(intensity[i])) finite = false;
    for (let i = 0; i < rgba.length; i++) if (!Number.isFinite(rgba[i])) finite = false;
    assert(finite, 'constant: no NaN/Infinity when max === min');
    // norm = (7 - 7) / 255 = 0 for every pixel
    assert(intensity.every(v => v === 0), 'constant: all intensities 0 under fallback range');
    assert(rgba[3] === 255, 'constant: alpha still opaque');
}

// ---- Test 4: intensityArray is optional ----
{
    const data = new Float32Array([1, 2, 3]);
    const rgba = new Uint8Array(3 * 4);
    let threw = false;
    try { intensityToRGBA(data, rgba); } catch { threw = true; }
    assert(!threw, 'optional: omitting intensityArray does not throw');
    assert(rgba[0] === 0 && rgba[8] === 255, 'optional: rgba still written correctly');
}

console.log(failures === 0 ? '\nALL TESTS PASSED' : `\n${failures} FAILURES`);
process.exit(failures === 0 ? 0 : 1);
