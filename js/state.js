export const state = {
    // Images — ordered array; visual order == export order
    images: [],

    // ML
    rf: null,

    // Drawing / Tools
    currentClass: 0,
    isDrawing: false,
    activeImageId: null,
    toolMode: 'grab', // 'grab', 'paint', 'erase'

    // Features
    sigma: 1.0,

    // Camera
    camera: { x: 0, y: 0, scale: 1 },
    isSpaceDown: false,
    isPanning: false,
};
