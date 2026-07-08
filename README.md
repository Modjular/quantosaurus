# Quanto
Count cells in your browser using a pixel-classifier. Inspired by Ilastik.

### Features
 - Live preview of your labels thanks to GPU-acceleration.
 - Segmentations, probabilities, labels, and points exporting.
 - Figma-style canvas navigation.

### TODO:
 [ ] 3D support
 [ ] Contrast sliders
 [ ] Export an Ilastik-compatible `.ilp`

### Tests
Core CPU logic has dependency-free unit tests colocated with each module (`*.test.mjs`), run with plain Node — no framework or install step:

```sh
node js/rf.test.mjs               # FlatRandomForest: training, flat-buffer layout, Gini/purity
node js/io.test.mjs               # intensityToRGBA normalization
node js/backends/webgl2.test.mjs  # CCL + stats reference implementations
```

### Why
This started out as an exploration into how performant WebGPU could be for a more mid-level task like pixel-classification. Filter-computation and random-forest computation is all done on the GPU. The component-connecting for labeling is also done on the GPU.


