// Tests for the pure config constants in config.js.
// Dependency-free — run with: node js/config.test.mjs
import { NUM_CLASSES, DEFAULT_LABEL_COLORS, RF_CONFIG } from './config.js';

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

console.log(failures === 0 ? '\nALL TESTS PASSED' : `\n${failures} FAILURES`);
process.exit(failures === 0 ? 0 : 1);
