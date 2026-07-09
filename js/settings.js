// Persistence of global app settings to localStorage.
//
// Scope is deliberately narrow: only the settings a user customizes and reasonably
// expects to survive a reload — per-class overlay colors and class names. Anything
// tied to a specific image (contrast window, camera, labels) or too large for
// localStorage (images, features) is intentionally not persisted here, and neither
// is the trained forest (useless without the images/features it was trained on).
//
// Theme keeps its own separate `quantosaurus-theme` key (see index.html); it's
// applied to <body> before `state` exists, so it doesn't belong in this blob.
//
// Following the app convention, these functions take `state` explicitly and hold no
// module-level state. The localStorage-touching wrappers stay thin; the validation
// logic (`sanitizeSettings`) is pure so it can be unit-tested under Node, where
// localStorage doesn't exist (see settings.test.mjs).

export const SETTINGS_KEY = 'quantosaurus-settings';

const HEX_COLOR_RE = /^#[0-9a-fA-F]{6}$/;

/**
 * Validate/normalize a parsed settings object read from localStorage. The input is
 * untrusted: it may be corrupt, stale from a build with a different class count, or
 * hand-edited. Returns a clean `{ labelColors, classNames }` whose arrays are exactly
 * `numClasses` long — invalid or missing entries are dropped to `null` so callers can
 * fall back to their own defaults per-slot.
 *
 * @param {*} raw - Parsed JSON (any shape, including null/non-object).
 * @param {number} numClasses - Class count to clamp both arrays to.
 * @returns {{ labelColors: (string|null)[], classNames: (string|null)[] }}
 */
export function sanitizeSettings(raw, numClasses) {
    const n = Number.isInteger(numClasses) && numClasses > 0 ? numClasses : 0;
    const src = raw && typeof raw === 'object' ? raw : {};

    const rawColors = Array.isArray(src.labelColors) ? src.labelColors : [];
    const rawNames = Array.isArray(src.classNames) ? src.classNames : [];

    const labelColors = [];
    const classNames = [];
    for (let i = 0; i < n; i++) {
        const c = rawColors[i];
        labelColors.push(typeof c === 'string' && HEX_COLOR_RE.test(c) ? c : null);

        const name = rawNames[i];
        // Coerce to a trimmed string; blank/whitespace-only names become null so the
        // caller keeps its default (we never want to render an empty class label).
        const trimmed = name == null ? '' : String(name).trim();
        classNames.push(trimmed.length > 0 ? trimmed : null);
    }

    return { labelColors, classNames };
}

/**
 * Read persisted settings and apply them onto `state.labelColors` / `state.classNames`
 * in place, per-slot, leaving existing defaults for any slot with no valid saved value.
 * Fails soft: any read/parse error leaves `state` untouched.
 *
 * @param {object} state - The shared app state (uses `state.rf.numClasses`).
 */
export function loadSettings(state) {
    let parsed;
    try {
        const raw = localStorage.getItem(SETTINGS_KEY);
        if (!raw) return;
        parsed = JSON.parse(raw);
    } catch {
        return; // unavailable storage or malformed JSON — keep defaults
    }

    const { labelColors, classNames } = sanitizeSettings(parsed, state.rf.numClasses);
    labelColors.forEach((c, i) => { if (c !== null) state.labelColors[i] = c; });
    classNames.forEach((name, i) => { if (name !== null) state.classNames[i] = name; });
}

/**
 * Persist the current colors and class names. Fails soft: quota-exceeded or
 * private-mode errors are swallowed (mirroring how theme writes fail-soft).
 *
 * @param {object} state - The shared app state.
 */
export function saveSettings(state) {
    try {
        localStorage.setItem(SETTINGS_KEY, JSON.stringify({
            labelColors: state.labelColors,
            classNames: state.classNames,
        }));
    } catch {
        // ignore — persistence is best-effort
    }
}
