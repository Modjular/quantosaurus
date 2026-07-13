// Tests for the pure config constants and helpers in config.js.
// Dependency-free — run with: node js/config.test.mjs
import {
    NUM_CLASSES, DEFAULT_LABEL_COLORS, RF_CONFIG,
    BRUSH_SIZE_MIN, BRUSH_SIZE_MAX, BRUSH_SIZE_DEFAULT, clampBrushSize,
} from './config.js';

let failures = 0;
function assert(cond, msg) {
    if (cond) { console.log(`  ok  ${msg}`); }
    else { failures++; console.error(`FAIL  ${msg}`); }
}

// ---- Test 1: enough default colors for every class ----
{
    assert(
        DEFAULT_LABEL_COLORS.length >= NUM_CLASSES,
        `DEFAULT_LABEL_COLORS has at least NUM_CLASSES entries (${DEFAULT_LABEL_COLORS.length} >= ${NUM_CLASSES})`
    );
}

// ---- Test 2: every default color is #rrggbb hex, as <input type="color"> requires ----
{
    const hex = /^#[0-9a-f]{6}$/i;
    DEFAULT_LABEL_COLORS.forEach((color, i) => {
        assert(hex.test(color), `DEFAULT_LABEL_COLORS[${i}] is #rrggbb hex (got "${color}")`);
    });
}

// ---- Test 3: RF_CONFIG.numClasses stays in sync with NUM_CLASSES ----
{
    assert(
        RF_CONFIG.numClasses === NUM_CLASSES,
        `RF_CONFIG.numClasses matches NUM_CLASSES (got ${RF_CONFIG.numClasses} vs ${NUM_CLASSES})`
    );
}

// ---- Test 4: brush-size bounds are a sane, ordered range containing the default ----
{
    assert(
        BRUSH_SIZE_MIN <= BRUSH_SIZE_DEFAULT && BRUSH_SIZE_DEFAULT <= BRUSH_SIZE_MAX,
        `BRUSH_SIZE_DEFAULT within [min, max] (${BRUSH_SIZE_MIN} <= ${BRUSH_SIZE_DEFAULT} <= ${BRUSH_SIZE_MAX})`
    );
}

// ---- Test 5: clampBrushSize clamps, rounds, and rejects garbage ----
{
    assert(clampBrushSize(BRUSH_SIZE_DEFAULT) === BRUSH_SIZE_DEFAULT,
        `in-range size passes through (got ${clampBrushSize(BRUSH_SIZE_DEFAULT)})`);
    assert(clampBrushSize(0) === BRUSH_SIZE_MIN,
        `below-min clamps to BRUSH_SIZE_MIN (got ${clampBrushSize(0)})`);
    assert(clampBrushSize(-5) === BRUSH_SIZE_MIN,
        `negative clamps to BRUSH_SIZE_MIN (got ${clampBrushSize(-5)})`);
    assert(clampBrushSize(BRUSH_SIZE_MAX + 100) === BRUSH_SIZE_MAX,
        `above-max clamps to BRUSH_SIZE_MAX (got ${clampBrushSize(BRUSH_SIZE_MAX + 100)})`);
    assert(clampBrushSize(4.6) === 5,
        `fractional sizes round to the nearest integer (got ${clampBrushSize(4.6)})`);
    assert(clampBrushSize('12') === 12,
        `numeric strings coerce (got ${clampBrushSize('12')})`);
    assert(clampBrushSize('garbage') === BRUSH_SIZE_DEFAULT,
        `non-numeric input falls back to the default (got ${clampBrushSize('garbage')})`);
    assert(clampBrushSize(NaN) === BRUSH_SIZE_DEFAULT,
        `NaN falls back to the default (got ${clampBrushSize(NaN)})`);
}

console.log(failures === 0 ? '\nALL TESTS PASSED' : `\n${failures} FAILURES`);
process.exit(failures === 0 ? 0 : 1);
