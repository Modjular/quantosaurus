// Tests for FlatRandomForest — pure CPU code, no browser/GPU dependencies.
// Dependency-free — run with: node js/rf.test.mjs
//
// Covers the train/predict round-trip, the flat forest-buffer layout that the
// WebGPU/WebGL inference shaders also consume (twin i32/f32 views; leaf encoding
// feature_index = -1, right_child = -(classId + 1)), the Gini/purity/majority
// helpers, and edge cases.
import { FlatRandomForest } from './rf.js';

let failures = 0;
function assert(cond, msg) {
    if (cond) { console.log(`  ok  ${msg}`); }
    else { failures++; console.error(`FAIL  ${msg}`); }
}
function throws(fn, msg) {
    try { fn(); failures++; console.error(`FAIL  ${msg} (did not throw)`); }
    catch { console.log(`  ok  ${msg}`); }
}

// Build a flat row-major Float32Array from rows of feature values.
function flatX(rows) {
    const numFeatures = rows[0].length;
    const X = new Float32Array(rows.length * numFeatures);
    for (let i = 0; i < rows.length; i++)
        for (let f = 0; f < numFeatures; f++) X[i * numFeatures + f] = rows[i][f];
    return X;
}

// Decode the flat forest buffer the way predictSingle / the shaders do.
function i32View(rf) {
    return new Int32Array(rf.forestBuffer.buffer, rf.forestBuffer.byteOffset, rf.forestBuffer.length);
}

// ---- Test 1: single-tree train/predict round-trip on separable data ----
{
    // Deterministic path: numTrees === 1 uses all samples in order (no bootstrap).
    // 1 feature cleanly separates class 0 (<5) from class 1 (>5).
    const rows = [[1], [2], [3], [8], [9], [10]];
    const y = new Int32Array([0, 0, 0, 1, 1, 1]);
    const rf = new FlatRandomForest({ numTrees: 1, maxDepth: 5, numClasses: 2 });
    rf.train(flatX(rows), y, 1);

    let allCorrect = true;
    for (let i = 0; i < rows.length; i++)
        if (rf.predictSingle(rows[i]) !== y[i]) allCorrect = false;
    assert(allCorrect, 'separable: every training sample predicts its own label');
    assert(rf.predictSingle([0]) === 0 && rf.predictSingle([100]) === 1,
        'separable: unseen points classify by the learned threshold');
}

// ---- Test 2: two features, deterministic single tree ----
{
    // Quadrant labels: class = (x>5)?1:0 combined with feature 1 as noise-free signal.
    const rows = [[1, 1], [2, 1], [9, 9], [8, 9], [1, 8], [2, 9]];
    const y = new Int32Array([0, 0, 1, 1, 0, 0]);
    const rf = new FlatRandomForest({ numTrees: 1, maxDepth: 6, numClasses: 2 });
    rf.train(flatX(rows), y, 2);
    let allCorrect = true;
    for (let i = 0; i < rows.length; i++)
        if (rf.predictSingle(rows[i]) !== y[i]) allCorrect = false;
    assert(allCorrect, '2-feature: perfect fit on training samples');
}

// ---- Test 3: flat-buffer layout invariants ----
{
    const rows = [[1], [2], [8], [9]];
    const y = new Int32Array([0, 0, 1, 1]);
    const rf = new FlatRandomForest({ numTrees: 1, maxDepth: 5, numClasses: 2 });
    rf.train(flatX(rows), y, 1);

    const i32 = i32View(rf);
    const nodeCount = rf.forestBuffer.length / 4;
    assert(Number.isInteger(nodeCount) && nodeCount > 0,
        `layout: buffer length is a whole number of 4-slot nodes (got ${nodeCount})`);
    assert(rf.treeRoots.length === 1 && rf.treeRoots[0] === 0,
        'layout: single tree root at offset 0');

    let leaves = 0, internal = 0, badLeaf = 0, badPtr = 0;
    for (let n = 0; n < nodeCount; n++) {
        const s = n * 4;
        const feat = i32[s + 0];
        if (feat === -1) {
            leaves++;
            const classId = -i32[s + 3] - 1;           // decode leaf class
            if (!(classId >= 0 && classId < rf.numClasses)) badLeaf++;
            if (i32[s + 2] !== -1) badLeaf++;           // left child unused on leaf
        } else {
            internal++;
            const l = i32[s + 2], r = i32[s + 3];
            if (!(l >= 0 && l < nodeCount && r >= 0 && r < nodeCount)) badPtr++;
        }
    }
    assert(leaves > 0 && internal > 0, `layout: tree has both leaves and internal nodes (${leaves}/${internal})`);
    assert(badLeaf === 0, 'layout: every leaf encodes a valid class id and unused left child');
    assert(badPtr === 0, 'layout: every internal node points to in-range child nodes');
}

// ---- Test 4: pure dataset collapses to a single leaf ----
{
    const rows = [[1], [2], [3]];
    const y = new Int32Array([2, 2, 2]);
    const rf = new FlatRandomForest({ numTrees: 1, maxDepth: 5, numClasses: 4 });
    rf.train(flatX(rows), y, 1);
    assert(rf.forestBuffer.length / 4 === 1, 'pure: exactly one node');
    const i32 = i32View(rf);
    assert(i32[0] === -1 && -i32[3] - 1 === 2, 'pure: root is a leaf predicting class 2');
    assert(rf.predictSingle([42]) === 2, 'pure: predicts the only class regardless of input');
}

// ---- Test 5: maxDepth caps tree growth ----
{
    // Cleanly separable, but capped at depth 1: root splits once, both children
    // become leaves -> exactly 3 nodes (root + 2 leaves).
    const rows = [[1], [2], [3], [8], [9], [10]];
    const y = new Int32Array([0, 0, 0, 1, 1, 1]);
    const rf = new FlatRandomForest({ numTrees: 1, maxDepth: 1, numClasses: 2 });
    rf.train(flatX(rows), y, 1);
    assert(rf.forestBuffer.length / 4 === 3, 'depth1: root + two leaf children (3 nodes)');
    const i32 = i32View(rf);
    assert(i32[0] !== -1, 'depth1: root is a split node');
    const l = i32[2], r = i32[3];
    assert(i32[l * 4] === -1 && i32[r * 4] === -1, 'depth1: both children are leaves');
    assert(rf.predictSingle([1]) === 0 && rf.predictSingle([10]) === 1, 'depth1: still classifies separable data');
}

// ---- Test 6: pure-helper unit checks ----
{
    const rf = new FlatRandomForest({ numClasses: 2 });
    const y = new Int32Array([0, 0, 1, 1]);
    const all = [0, 1, 2, 3];
    assert(Math.abs(rf._calculateGini(y, all) - 0.5) < 1e-9, 'gini: 50/50 two-class -> 0.5');
    assert(rf._calculateGini(y, [0, 1]) === 0, 'gini: pure subset -> 0');
    assert(rf._checkPurity(y, [0, 1]) === true, 'purity: same-class indices -> true');
    assert(rf._checkPurity(y, [0, 2]) === false, 'purity: mixed indices -> false');
    assert(rf._getMajorityClass(new Int32Array([1, 1, 0]), [0, 1, 2]) === 1, 'majority: picks most frequent class');

    // _findBestSplit returns featureIdx = -1 when no split reduces impurity
    // (all feature values identical -> no candidate thresholds).
    const X = flatX([[5], [5], [5], [5]]);
    const split = rf._findBestSplit(X, y, 1, all);
    assert(split.featureIdx === -1, 'findBestSplit: no useful split -> featureIdx -1');
}

// ---- Test 7: multi-tree buffer stays well-formed (random bootstrap path) ----
{
    const rows = [[1], [2], [3], [8], [9], [10]];
    const y = new Int32Array([0, 0, 0, 1, 1, 1]);
    const rf = new FlatRandomForest({ numTrees: 8, maxDepth: 5, numClasses: 2 });
    rf.train(flatX(rows), y, 1);
    assert(rf.treeRoots.length === 8, 'multitree: 8 tree roots recorded');
    let monotonic = true;
    for (let t = 1; t < 8; t++) if (rf.treeRoots[t] < rf.treeRoots[t - 1]) monotonic = false;
    assert(monotonic, 'multitree: tree roots are non-decreasing offsets');
    const nodeCount = rf.forestBuffer.length / 4;
    assert(rf.treeRoots[7] < nodeCount, 'multitree: last root is within the buffer');
    // Prediction still returns a valid class index.
    const p = rf.predictSingle([2]);
    assert(p >= 0 && p < 2, 'multitree: predictSingle returns an in-range class');
}

// ---- Test 8: feature-stride contract ----
{
    // 2 features; feature 0 is constant (no split possible) so only feature 1
    // can separate the classes. A wrong numFeatures stride would read the wrong
    // column. Documents the numFeatures argument contract that trainAndPredictAll
    // relies on (config.NUM_FEATURES).
    const rows = [[5, 1], [5, 2], [5, 8], [5, 9]];
    const y = new Int32Array([0, 0, 1, 1]);
    const rf = new FlatRandomForest({ numTrees: 1, maxDepth: 5, numClasses: 2 });
    rf.train(flatX(rows), y, 2);
    const i32 = i32View(rf);
    assert(i32[0] === 1, 'stride: root splits on informative feature 1, not constant feature 0');
    assert(rf.predictSingle([5, 1]) === 0 && rf.predictSingle([5, 9]) === 1, 'stride: predicts via feature 1');
    assert(rf.predictSingle([999, 1]) === 0, 'stride: varying the non-split feature does not change prediction');
}

// ---- Test 9: empty labels throws ----
{
    const rf = new FlatRandomForest({ numTrees: 1, numClasses: 2 });
    throws(() => rf.train(new Float32Array(0), new Int32Array(0), 1), 'empty: train([]) throws');
    throws(() => rf.train(new Float32Array(0), null, 1), 'empty: train(null y) throws');
}

console.log(failures === 0 ? '\nALL TESTS PASSED' : `\n${failures} FAILURES`);
process.exit(failures === 0 ? 0 : 1);
