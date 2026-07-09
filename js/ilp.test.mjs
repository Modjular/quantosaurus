// Tests for buildIlpProject — pure CPU code (no browser/GPU dependencies), but
// note it does use the Web Crypto `crypto.randomUUID()` global, which Node
// also provides. Dependency-free — run with: node js/ilp.test.mjs
//
// This writer hand-serializes HDF5, and we have no HDF5 parser available
// here, so these are black-box checks on the public API: internal
// self-consistency (the superblock's declared EOF matches the actual byte
// length), the fixed HDF5 signature, and literal ASCII text we know must
// appear verbatim in the output (dataset/group names, JSON attribute values,
// the computed blockSlice string) — strong enough to catch structural or
// label-bounding-box regressions without needing to parse HDF5 ourselves.
// Full spec-compliance is verified out-of-band against ILP_PixelClassification_verify.py.
import { buildIlpProject } from './ilp.js';

let failures = 0;
function assert(cond, msg) {
    if (cond) { console.log(`  ok  ${msg}`); }
    else { failures++; console.error(`FAIL  ${msg}`); }
}
function throws(fn, msg) {
    try { fn(); failures++; console.error(`FAIL  ${msg} (did not throw)`); }
    catch { console.log(`  ok  ${msg}`); }
}

function containsAscii(buf, str) {
    const needle = new TextEncoder().encode(str);
    outer:
    for (let i = 0; i <= buf.length - needle.length; i++) {
        for (let j = 0; j < needle.length; j++) {
            if (buf[i + j] !== needle[j]) continue outer;
        }
        return true;
    }
    return false;
}

function fakeImage({ name, width, height, labels }) {
    return { name, width, height, labels };
}

function fakeState(images, overrides = {}) {
    return {
        images,
        rf: { numClasses: 4, numTrees: 8, ...overrides.rf },
        labelColors: overrides.labelColors ?? ['#ff595e', '#ffca3a', '#8ac926', '#1982c4'],
    };
}

// ---- Test 1: empty project (no images) is still well-formed ----
{
    const bytes = buildIlpProject(fakeState([]), { time: 'Wed Aug 23 10:29:38 2023' });
    const sig = [0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a];
    assert(sig.every((b, i) => bytes[i] === b), 'starts with the HDF5 signature');

    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const eofAddr = view.getBigUint64(40, true);
    assert(eofAddr === BigInt(bytes.length), 'superblock EOF address matches actual byte length');

    for (const s of ['ilastikVersion', 'workflowName', 'Pixel Classification', 'FeatureSelections',
        'PixelClassification', 'Input Data', 'LabelColors', 'ClassifierFactory', 'Bookmarks']) {
        assert(containsAscii(bytes, s), `contains expected literal "${s}"`);
    }
}

// ---- Test 2: one image with no labels still gets a lane + empty label set ----
{
    const img = fakeImage({ name: 'cells.tif', width: 100, height: 80, labels: [] });
    const bytes = buildIlpProject(fakeState([img]));
    for (const s of ['lane0000', 'labels000', 'Raw Data', 'cells.tif', 'cells']) {
        assert(containsAscii(bytes, s), `one-image project contains "${s}"`);
    }
    // shape is stored as int64 [1, height, width] — zyx convention
    assert(containsAscii(bytes, '"key": "z"') && containsAscii(bytes, '"key": "y"') && containsAscii(bytes, '"key": "x"'),
        'Raw Data axistags encode z, y, x axes');
}

// ---- Test 3: label bounding box is computed correctly end-to-end ----
{
    // Labels scattered across a known rectangle: x in [2,6], y in [7,9] -> bbox
    // is x:[2,7), y:[7,10) -> blockSlice "[7:10,2:7,0:1]" (y,x,c order).
    const labels = [
        { x: 2, y: 7, cls: 0 },
        { x: 6, y: 9, cls: 1 },
        { x: 4, y: 8, cls: 2 },
    ];
    const img = fakeImage({ name: 'scan.png', width: 50, height: 50, labels });
    const bytes = buildIlpProject(fakeState([img]));
    assert(containsAscii(bytes, '[7:10,2:7,0:1]'), 'blockSlice matches the labels\' bounding box');
    assert(containsAscii(bytes, 'block0000'), 'labeled image gets a block0000 dataset');
}

// ---- Test 4: multiple images produce correctly zero-padded lane/label-set names ----
{
    const images = Array.from({ length: 3 }, (_, i) =>
        fakeImage({ name: `img${i}.tif`, width: 10, height: 10, labels: [{ x: 0, y: 0, cls: 0 }] }));
    const bytes = buildIlpProject(fakeState(images));
    for (const n of ['lane0000', 'lane0001', 'lane0002', 'labels000', 'labels001', 'labels002']) {
        assert(containsAscii(bytes, n), `multi-image project contains "${n}"`);
    }
}

// ---- Test 5: label names/colors come from options / state, with sane defaults ----
{
    const img = fakeImage({ name: 'x.tif', width: 5, height: 5, labels: [] });
    const bytes = buildIlpProject(fakeState([img]), { classNames: ['Microglia', 'Background'] });
    assert(containsAscii(bytes, 'Microglia') && containsAscii(bytes, 'Background'), 'custom classNames are used');

    const defaultBytes = buildIlpProject(fakeState([img]));
    assert(containsAscii(defaultBytes, 'Class 1') && containsAscii(defaultBytes, 'Class 2'),
        'default classNames fall back to "Class N"');
}

// ---- Test 6: group-capacity guard throws instead of silently corrupting ----
{
    const tooMany = Array.from({ length: 100 }, (_, i) =>
        fakeImage({ name: `img${i}.tif`, width: 4, height: 4, labels: [] }));
    throws(() => buildIlpProject(fakeState(tooMany)), 'exceeding the single-SNOD capacity throws a clear error');
}

if (failures > 0) {
    console.error(`\n${failures} failure(s)`);
    process.exit(1);
} else {
    console.log('\nAll ilp.js tests passed.');
}
