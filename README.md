# 🧮 Quantosaurus
Count cells in your browser using a pixel-classifier. Inspired by Ilastik.

### Why
A friend needed help counting cells, but I couldn't find any simple, browser-accessible tools. I also wanted to see how good WebGPU could be for a non-trivial task like pixel-classification. Filter-computation, random-forest inference and component-connection is all done on the GPU as well.

### Features
 - Live preview of your labels thanks to GPU-acceleration.
 - Segmentations, probabilities, labels, and points exporting.
 - Figma-style canvas navigation.

# Development
Because there's no backend, just pull the repo and serve over localhost (to enable WebGPU access).

### TODO:
 - [ ] 3D support
 - [ ] Contrast sliders
 - [ ] Export an Ilastik-compatible `.ilp`

### Tests
Core CPU logic has dependency-free unit tests colocated with each module (`*.test.mjs`), run with plain Node — no framework or install step:

```sh
node js/rf.test.mjs               # FlatRandomForest: training, flat-buffer layout, Gini/purity
node js/io.test.mjs               # intensityToRGBA normalization
node js/backends/webgl2.test.mjs  # CCL + stats reference implementations
```


