<img width="1204" height="805" alt="Screenshot 2026-07-08 at 1 34 15 PM" src="https://github.com/user-attachments/assets/549f7c67-51da-448c-b92d-bc6323660ba3" />


# 🦖 Quantosaurus
Use a pixel-classifier to count your cells. Like [Ilastik](https://github.com/ilastik/ilastik) but in the browser.


### When
Use Quantosaurus only after you've tried ImageJ's threshold tool, but before you reach for something heavier duty like Ilastik's Object Classifiers or Cellpose.


### Why
When a friend needed help counting cells in her lab, I couldn't find any online, simple, accessible tools. At the same time, I wanted to learn more about WebGPU by using it for a non-trivial task like pixel-classification.


### Features
 - `.tif`, `.png`, and `.jpg` support.
 - Live preview of your labels thanks to GPU-acceleration.
 - Segmentations, probabilities, labels, and points exporting.
 - Figma-style canvas navigation.


### Goals
 - **Accessible**: For users, accessible means an enjoyable, browser first tool, with no complicated UI, hidden affordances or fancy hotkeys. For developers, code should be readable and file structure flat. No `npm` or install means you can `git clone` and start hacking as soon as possible.
 - **Local**: For users, local means no need for an active internet connection. For developers, it means no digging through `node_modules`; it's all here, and you (or your AI) can read it all in a sitting. If you find a lib truly necessary, vendor it.
 - **Simple**: For users, simple means the tool does one thing, and one thing well. For developers, it means readability was preferred over code-golf. DIY was explored before reaching for 3rd party libraries.


### Roadmap:
 - [ ] 3D support
 - [x] ~Contrast sliders~
 - [ ] Export trained models as Ilastik-compatible `.ilp` files
 - [ ] Multi-file batching


# For Developers
Because there's no backend or build step, just pull the repo, and serve over localhost.

```sh
    git clone https://github.com/Modjular/quantosaurus.git
    cd quantosaurus
    python -m http.server # or any server
```

### Tests
Core CPU logic has dependency-free unit tests colocated with each module (`*.test.mjs`), run with plain Node — no framework or install step:

```sh
node js/rf.test.mjs               # FlatRandomForest: training, flat-buffer layout, Gini/purity
node js/io.test.mjs               # intensityToRGBA normalization
node js/backends/webgl2.test.mjs  # CCL + stats reference implementations
```

# Acknowledgments
Built with 🦖 for Dr. Chen.
Thanks to Ilastik for inspiration.
Thanks to ITK-Wasm for file handling.
