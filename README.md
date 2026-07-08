<img width="1204" height="805" alt="Screenshot 2026-07-08 at 1 34 15 PM" src="https://github.com/user-attachments/assets/549f7c67-51da-448c-b92d-bc6323660ba3" />


# 🦖 Quantosaurus
Use machine learning to count your cells, all inside the browser. Completely local, no internet connection required. Inspired by Ilastik.

### Why
A friend needed help counting cells in her lab, but I couldn't find any simple, accessible tools. I also wanted to see how good WebGPU could be for a non-trivial task like pixel-classification. Filter-computation, random-forest inference and component-connection is all done via WebGPU

### Features
 - `.tif`, `.png`, and `.jpg` support.
 - Live preview of your labels thanks to GPU-acceleration.
 - Segmentations, probabilities, labels, and points exporting.
 - Figma-style canvas navigation.

# Development
No backend, no build step. Just pull the repo, and serve over localhost.

### TODO:
 - [ ] 3D support
 - [x] ~Contrast sliders~
 - [ ] Export trained models as Ilastik-compatible `.ilp` files
 - [ ] Multi-file batching

### Tests
Core CPU logic has dependency-free unit tests colocated with each module (`*.test.mjs`), run with plain Node — no framework or install step:

```sh
node js/rf.test.mjs               # FlatRandomForest: training, flat-buffer layout, Gini/purity
node js/io.test.mjs               # intensityToRGBA normalization
node js/backends/webgl2.test.mjs  # CCL + stats reference implementations
```


