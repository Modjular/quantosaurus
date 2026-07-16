// Tests for computeOtsu — the pure CPU Otsu threshold. Dependency-free —
// run with: node js/threshold.test.mjs
import { computeOtsu } from './threshold.js';

let failures = 0;
function assert(cond, msg) {
    if (cond) { console.log(`  ok  ${msg}`); }
    else { failures++; console.error(`FAIL  ${msg}`); }
}

// Clean bimodal: half the pixels at 10, half at 240 (range 0..255). Otsu should
// land the threshold strictly between the two modes.
{
    const arr = new Float32Array(2000);
    for (let i = 0; i < 1000; i++) arr[i] = 10;
    for (let i = 1000; i < 2000; i++) arr[i] = 240;
    const t = computeOtsu(arr, { dataMin: 0, dataMax: 255 });
    assert(t > 10 && t < 240, `bimodal threshold ${t.toFixed(1)} lies between the modes`);
    // Applying it separates the two populations exactly.
    let fgLow = 0, fgHigh = 0;
    for (const v of arr) { if (v >= t) { (v === 240 ? fgHigh++ : fgLow++); } }
    assert(fgLow === 0 && fgHigh === 1000, 'threshold cleanly separates the two modes');
}

// Uniform data inside a wider range: no real split exists, so every pixel must
// land on the same side of the threshold (never a spurious partial split).
{
    const arr = new Float32Array(100).fill(42);
    const t = computeOtsu(arr, { dataMin: 0, dataMax: 255 });
    let fg = 0;
    for (const v of arr) if (v >= t) fg++;
    assert(fg === 0 || fg === 100, `uniform data all on one side of threshold ${t.toFixed(2)} (fg=${fg})`);
}

// Zero-span range (dataMin == dataMax): guarded, returns dataMin.
{
    const arr = new Float32Array([5, 5, 5]);
    const t = computeOtsu(arr, { dataMin: 5, dataMax: 5 });
    assert(t === 5, 'zero-span range returns dataMin without error');
}

// Skewed foreground: a small bright blob (5%) on a dark background (95%).
// The threshold must fall above the background mode so the blob is foreground.
{
    const arr = new Float32Array(1000);
    for (let i = 0; i < 950; i++) arr[i] = 20;   // background
    for (let i = 950; i < 1000; i++) arr[i] = 200; // bright blob
    const t = computeOtsu(arr, { dataMin: 0, dataMax: 255 });
    let fg = 0;
    for (const v of arr) if (v >= t) fg++;
    assert(t > 20 && t < 200, `skewed threshold ${t.toFixed(1)} between background and blob`);
    assert(fg === 50, 'only the 50 bright pixels are foreground');
}

// 16-bit range: values in native uint16 units bin correctly over 0..65535.
{
    const arr = new Float32Array(2000);
    for (let i = 0; i < 1000; i++) arr[i] = 1000;
    for (let i = 1000; i < 2000; i++) arr[i] = 60000;
    const t = computeOtsu(arr, { dataMin: 0, dataMax: 65535 });
    assert(t > 1000 && t < 60000, `16-bit threshold ${t.toFixed(0)} between the modes`);
}

console.log(failures === 0 ? '\nALL PASS' : `\n${failures} FAILURE(S)`);
process.exit(failures === 0 ? 0 : 1);
