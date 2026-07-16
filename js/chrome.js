// App-agnostic UI chrome shared by every Quantosaurus app (classifier, threshold,
// cellpose): theme toggle, feedback modal, cheatsheet, file ingest, top-bar nav,
// and a couple of download helpers. None of this is specific to a segmentation
// method, so it lives here rather than being copy-pasted into each page.
//
// Following the app convention, functions take `state` explicitly and hold no
// module-level state. Every DOM lookup is guarded so an app can omit any given
// piece of chrome (e.g. a page without a cheatsheet) without this throwing.
//
// The three apps share one theme key and one feedback endpoint but get their own
// per-page first-visit flag so each tool can auto-open its own cheatsheet once.
import { addFiles } from './images.js';

/** Today's date as 'YYYYMMDD', for export filenames. */
export function dateStamp() {
    return new Date().toISOString().slice(0, 10).replace(/-/g, '');
}

/** Triggers a browser download of `blob` named `filename`. */
export function downloadBlob(blob, filename) {
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    link.click();
    URL.revokeObjectURL(link.href);
}

/**
 * Wires the light/dark theme toggle. Theme is persisted under its own
 * `quantosaurus-theme` key (shared across apps); the initial class is applied to
 * <body> inline in each page before modules load, so this only wires the button.
 */
export function setupTheme() {
    const btnToggle = document.getElementById('btnThemeToggle');
    if (!btnToggle) return;

    const savedTheme = localStorage.getItem('quantosaurus-theme') || 'dark';
    if (savedTheme === 'light') {
        document.body.classList.add('theme-light');
        btnToggle.innerHTML = '🌙&nbsp;&nbsp;Dark Mode';
    }

    btnToggle.onclick = () => {
        const isLight = document.body.classList.toggle('theme-light');
        btnToggle.innerHTML = isLight ? '🌙&nbsp;&nbsp;Dark Mode' : '☀️&nbsp;&nbsp;Light Mode';
        localStorage.setItem('quantosaurus-theme', isLight ? 'light' : 'dark');
    };
}

/**
 * Wires the feedback modal (open/close/submit → Feedstick). Reads a little
 * app-state context into the debug payload when the user opts in.
 * @param {Object} state - Shared app state (for the optional debug context).
 */
export function setupFeedback(state) {
    const modal = document.getElementById('feedbackModal');
    const btnOpen = document.getElementById('btnFeedback');
    if (!modal || !btnOpen) return;
    const btnClose = document.getElementById('btnCloseFeedback');
    const btnSubmit = document.getElementById('btnSubmitFeedback');
    const textarea = document.getElementById('feedbackText');
    const emailInput = document.getElementById('feedbackEmail');
    const chkDebug = document.getElementById('chkDebug');

    btnOpen.onclick = () => {
        modal.style.display = 'flex';
        textarea.value = '';
        emailInput.value = '';
        textarea.focus();
    };

    const closeModal = () => modal.style.display = 'none';
    if (btnClose) btnClose.onclick = closeModal;
    // Close modal if clicked outside content card
    modal.onclick = (e) => { if (e.target === modal) closeModal(); };

    btnSubmit.onclick = async () => {
        const content = textarea.value.trim();
        const email = emailInput.value.trim();

        if (!content) {
            alert('Please enter your feedback before submitting.');
            textarea.focus();
            return;
        }

        // Pure regex validation for optional email
        if (email) {
            const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
            if (!emailRegex.test(email)) {
                alert('Please enter a valid email address, or leave it blank.');
                emailInput.focus();
                return;
            }
        }

        btnSubmit.disabled = true;
        btnSubmit.innerText = 'Sending…';

        try {
            const payload = { content: content };
            if (email) payload.email = email;

            // Append technical context if the user opted in
            if (chkDebug && chkDebug.checked) {
                payload.context = {
                    userAgent: navigator.userAgent,
                    webGPU: !!navigator.gpu,
                    windowSize: `${window.innerWidth}x${window.innerHeight}`,
                    cores: navigator.hardwareConcurrency || 'unknown',
                    memoryGB: navigator.deviceMemory || 'unknown',
                    appState_imagesLoaded: state.images ? state.images.length : 0,
                    appState_activeTool: state.toolMode || 'unknown',
                };
            }

            const response = await fetch("https://a.feedstick.app/feedback", {
                method: "POST",
                headers: {
                    "X-Feedstick-Key": "pk_live_4af6a5acafc943c18f763a863f12615f",
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(payload),
            });

            if (!response.ok) throw new Error('Network response failure.');

            alert('Feedback submitted successfully! Thank you.');
            closeModal();
        } catch (err) {
            console.error(err);
            alert(`Could not send feedback: ${err.message}`);
        } finally {
            btnSubmit.disabled = false;
            btnSubmit.innerText = 'Submit';
        }
    };
}

/**
 * Wires the cheatsheet modal (open/close/print) and auto-opens it on the user's
 * first visit to this specific app. Each page has its own cheatsheet content but
 * the same control ids, so this wiring is shared; the first-visit flag is keyed
 * per page so each tool introduces itself once.
 * @param {string} [page] - Page id, used for the per-page first-visit key.
 */
export function setupCheatsheet(page = 'app') {
    const modal = document.getElementById('cheatsheetModal');
    if (!modal) return;
    const btnOpen = document.getElementById('btnCheatsheet');
    const btnClose = document.getElementById('btnCloseCheatsheet');
    const btnPrint = document.getElementById('btnPrintCheatsheet');

    modal.onclick = (e) => { if (e.target === modal) modal.style.display = 'none'; };
    if (btnOpen) btnOpen.onclick = () => modal.style.display = 'flex';
    if (btnClose) btnClose.onclick = () => modal.style.display = 'none';
    if (btnPrint) btnPrint.onclick = () => window.print();

    const visitedKey = `quantosaurus-visited-${page}`;
    if (!localStorage.getItem(visitedKey)) {
        localStorage.setItem(visitedKey, 'true');
        modal.style.display = 'flex';
    }
}

/**
 * Wires the file <input> and viewport drag-and-drop to addFiles. Both are the
 * same across apps — images load identically regardless of segmentation method.
 * @param {Object} state - Shared app state.
 */
export function setupFileIngest(state) {
    const fileInput = document.getElementById('fileInput');
    if (fileInput) {
        fileInput.onchange = (e) => addFiles(state, e.target.files).catch(console.error);
    }

    const viewport = document.getElementById('viewport');
    if (viewport) {
        viewport.ondragover  = (e) => { e.preventDefault(); viewport.classList.add('drag-active'); };
        viewport.ondragleave = (e) => { e.preventDefault(); viewport.classList.remove('drag-active'); };
        viewport.ondrop      = (e) => {
            e.preventDefault();
            viewport.classList.remove('drag-active');
            addFiles(state, e.dataTransfer.files).catch(console.error);
        };
    }
}

/**
 * Highlights the current app in the top-bar nav (links between the three apps).
 * The nav markup lives in each page; this only marks the active link so a page
 * doesn't have to hardcode its own highlighted state.
 * @param {string} [page] - Page id matching a nav link's data-page attribute.
 */
export function setupNav(page) {
    if (!page) return;
    document.querySelectorAll('[data-page]').forEach(el => {
        el.classList.toggle('active', el.dataset.page === page);
    });
}
