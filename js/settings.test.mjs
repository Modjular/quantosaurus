// Tests for the pure sanitizeSettings validator in settings.js.
// Dependency-free (no localStorage/DOM needed) — run with: node js/settings.test.mjs
import { sanitizeSettings } from './settings.js';

let failures = 0;
function assert(cond, msg) {
    if (cond) { console.log(`  ok  ${msg}`); }
    else { failures++; console.error(`FAIL  ${msg}`); }
}

// ---- Test 1: valid input round-trips, arrays clamped to numClasses ----
{
    const raw = {
        labelColors: ['#ff0000', '#00ff00', '#0000ff', '#ffffff'],
        classNames: ['Nucleus', 'Cytoplasm', 'Background', 'Debris'],
    };
    const { labelColors, classNames } = sanitizeSettings(raw, 4);
    assert(labelColors.length === 4, `labelColors clamped to numClasses (got ${labelColors.length})`);
    assert(classNames.length === 4, `classNames clamped to numClasses (got ${classNames.length})`);
    assert(labelColors[0] === '#ff0000' && labelColors[3] === '#ffffff', 'valid colors preserved');
    assert(classNames[0] === 'Nucleus' && classNames[2] === 'Background', 'valid names preserved');
}

// ---- Test 2: arrays longer than numClasses are truncated ----
{
    const raw = {
        labelColors: ['#ff0000', '#00ff00', '#0000ff', '#ffffff', '#123456', '#abcdef'],
        classNames: ['a', 'b', 'c', 'd', 'e', 'f'],
    };
    const { labelColors, classNames } = sanitizeSettings(raw, 4);
    assert(labelColors.length === 4, `extra colors truncated (got ${labelColors.length})`);
    assert(classNames.length === 4, `extra names truncated (got ${classNames.length})`);
}

// ---- Test 3: arrays shorter than numClasses are padded with null ----
{
    const raw = { labelColors: ['#ff0000'], classNames: ['a'] };
    const { labelColors, classNames } = sanitizeSettings(raw, 4);
    assert(labelColors.length === 4 && classNames.length === 4, 'short arrays padded to numClasses');
    assert(labelColors[0] === '#ff0000' && labelColors[1] === null && labelColors[3] === null,
        'missing color slots are null');
    assert(classNames[0] === 'a' && classNames[1] === null, 'missing name slots are null');
}

// ---- Test 4: non-hex / malformed colors are rejected to null ----
{
    const raw = {
        labelColors: ['red', '#fff', '#GGGGGG', '#12345', 123, null],
        classNames: [],
    };
    const { labelColors } = sanitizeSettings(raw, 4);
    assert(labelColors.every((c) => c === null), 'all invalid colors coerced to null');
}

// ---- Test 5: names are coerced to trimmed strings; blank/whitespace -> null ----
{
    const raw = {
        labelColors: [],
        classNames: ['  Padded  ', '', '   ', 42],
    };
    const { classNames } = sanitizeSettings(raw, 4);
    assert(classNames[0] === 'Padded', `name trimmed (got ${JSON.stringify(classNames[0])})`);
    assert(classNames[1] === null, 'empty-string name -> null');
    assert(classNames[2] === null, 'whitespace-only name -> null');
    assert(classNames[3] === '42', `non-string name coerced (got ${JSON.stringify(classNames[3])})`);
}

// ---- Test 6: garbage / non-object input yields all-null arrays, no throw ----
{
    for (const bad of [null, undefined, 42, 'string', [], { labelColors: 'x', classNames: 7 }]) {
        const out = sanitizeSettings(bad, 4);
        assert(
            out.labelColors.length === 4 && out.classNames.length === 4 &&
            out.labelColors.every((c) => c === null) && out.classNames.every((c) => c === null),
            `garbage input ${JSON.stringify(bad)} -> 4 null slots each`
        );
    }
}

// ---- Test 7: non-positive / invalid numClasses yields empty arrays ----
{
    for (const bad of [0, -1, 1.5, NaN, undefined]) {
        const out = sanitizeSettings({ labelColors: ['#ff0000'], classNames: ['a'] }, bad);
        assert(out.labelColors.length === 0 && out.classNames.length === 0,
            `numClasses=${bad} -> empty arrays`);
    }
}

console.log(failures === 0 ? '\nALL TESTS PASSED' : `\n${failures} FAILURES`);
process.exit(failures === 0 ? 0 : 1);
