/**
 * WebGL2 compute/render backend — the fallback path when WebGPU is unavailable.
 * Interchangeable with WebGpuBackend; both expose the same interface used by the
 * image lifecycle and training code:
 *
 *   initialize, allocateImage, updateFeatures, downloadFeatures,
 *   gatherFeaturesForTraining, runInference, renderComposite,
 *   downloadProbabilities, computeConnectedComponents, computeStats,
 *   downloadStats, downloadLabels, setWindow, setColors, destroy.
 *
 * Pipeline: a separable Gaussian-derivative filter bank (two fragment passes,
 * horizontal then vertical) writes 8 per-pixel features across two RGBA float
 * textures; a Random Forest inference pass turns features into per-class
 * probabilities; a composite pass overlays the argmax class on the image. WebGL2
 * has no compute shaders/atomics, so connected-component labeling and stats run
 * on the CPU (see cclLabel/accumulateStats) with the same output contract as the
 * WebGPU backend.
 */
import { STATS_LAYOUT } from '../config.js';

export class WebGl2Backend {
  constructor(labelColors) {
      this.gl = null;
      this.width = 0;
      this.height = 0;
      // Raw single-channel intensity (R32F), shared by the stats pass and the
      // display window/level in the composite pass. Holds real pixel magnitudes.
      this.rawIntensityTexture = null;
      // Fixed-point multiplier applied to raw intensity before integer accumulation
      // in computeStats (1 for integer dtypes, larger for float). Set per image.
      this.intensityScale = 1;

      // FBOs and Textures
      this.horizFbo = null;
      this.horizTexture = null;
      this.featFbo = null;
      this.featTexture0 = null; // Features 0-3
      this.featTexture1 = null; // Features 4-7
      this.probFbo = null;
      this.probTexture = null;

      // CPU-side CCL results (see computeConnectedComponents/computeStats)
      this.labels = null;
      this.denseStats = null;

      // Display-only contrast window (black/white points) in the image's raw
      // intensity units. Seeded to the data range in allocateImage; see setWindow.
      this.windowLo = 0.0;
      this.windowHi = 1.0;

      this.labelColors = labelColors || [
          'rgba(255,0,0,1.0)',
          'rgba(0,255,0,1.0)',
          'rgba(0,0,255,1.0)',
      ];
      // Per-class overlay colors packed as a flat rgba (0–1) Float32Array, uploaded
      // to the composite pass as the u_colors uniform. Recoloring a class only
      // repacks this + repaints (see setColors) — no shader recompile.
      this._colorData = null;

      // Shader Programs
      this.progHoriz = null;
      this.progVert = null;
      this.progRF = null;
      this.progComposite = null;
      this.quadVao = null;
  }

  /**
   * Acquires the WebGL2 context (requires the float-render extension), sets up
   * the fullscreen quad, and compiles all shader programs.
   * @param {HTMLCanvasElement} canvas
   * @throws If WebGL2 or EXT_color_buffer_float is unavailable.
   */
  async initialize(canvas) {
      this.gl = canvas.getContext('webgl2', { antialias: false });
      if (!this.gl) throw new Error("WebGL2 not supported");

      const ext = this.gl.getExtension('EXT_color_buffer_float');
      if (!ext) throw new Error("EXT_color_buffer_float not supported. Cannot render to float textures.");

      this._setupQuad();
      this._compileShaders();
  }

  /**
   * (Re)allocates all per-image textures and framebuffers for a new image,
   * freeing any previous ones, and seeds the probability buffer to -1
   * (the "unclassified" sentinel). Also seeds the contrast window and stats
   * fixed-point scale from the image's range metadata.
   * @param {number} width
   * @param {number} height
   * @param {Float32Array} intensityArray - Raw single-channel intensities.
   * @param {{dataMin: number, dataMax: number, dtypeMax: number, scale: number}} range
   */
  async allocateImage(width, height, intensityArray, range) {
      this.width = width;
      this.height = height;
      this.intensityScale = range.scale;
      // Default the display window to the data range → reproduces the auto-stretch look.
      this.windowLo = range.dataMin;
      this.windowHi = range.dataMax;
      const gl = this.gl;

      // Free previous per-image resources: allocateImage is called repeatedly
      // (once per slice), so recreating without deleting leaks GPU memory.
      [this.rawIntensityTexture, this.horizTexture, this.featTexture0,
       this.featTexture1, this.probTexture].forEach(t => { if (t) gl.deleteTexture(t); });
      [this.horizFbo, this.featFbo, this.probFbo].forEach(f => { if (f) gl.deleteFramebuffer(f); });
      this.labels = null;
      this.denseStats = null;

      // R32F holds raw intensity; sampled with NEAREST (float textures aren't
      // linearly filterable without OES_texture_float_linear, which we don't need).
      this.rawIntensityTexture = this._createTexture(width, height, gl.R32F, gl.RED, gl.FLOAT, intensityArray);

      this.horizTexture = this._createTexture(width, height, gl.RGBA32F, gl.RGBA, gl.FLOAT, null);
      this.horizFbo = this._createFbo([this.horizTexture]);

      this.featTexture0 = this._createTexture(width, height, gl.RGBA32F, gl.RGBA, gl.FLOAT, null);
      this.featTexture1 = this._createTexture(width, height, gl.RGBA32F, gl.RGBA, gl.FLOAT, null);
      this.featFbo = this._createFbo([this.featTexture0, this.featTexture1]);

      this.probTexture = this._createTexture(width, height, gl.RGBA32F, gl.RGBA, gl.FLOAT, null);
      this.probFbo = this._createFbo([this.probTexture]);
      
      // Initialize probs to -1.0
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.probFbo);
      gl.clearColor(-1.0, -1.0, -1.0, -1.0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  /**
   * Recomputes the feature textures for the given intensity image at the given
   * sigma, then repaints the composite view.
   * @param {Float32Array} intensityArray - Normalized single-channel intensities.
   * @param {number} sigma - Gaussian scale for the filter bank.
   */
  async updateFeatures(intensityArray, sigma) {
      this._extractFeatures(intensityArray, sigma);
      this.renderComposite();
  }

  // Runs the two-pass separable filter bank, writing 8 features into
  // featTexture0 (0-3) and featTexture1 (4-7). `scale` is the Gaussian sigma.
  _extractFeatures(data, scale) {
      const gl = this.gl;
      const k0 = gaussian_kernel(scale, 0);
      const k1 = gaussian_kernel(scale, 1);
      const k2 = gaussian_kernel(scale, 2);
      const k0sub = gaussian_kernel(scale * 0.66, 0);

      // Upload input intensity as R32F texture
      const inputTex = this._createTexture(this.width, this.height, gl.R32F, gl.RED, gl.FLOAT, data);

      gl.viewport(0, 0, this.width, this.height);
      gl.bindVertexArray(this.quadVao);

      // --- Pass 1: Horizontal ---
      gl.useProgram(this.progHoriz);
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.horizFbo);
      
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, inputTex);
      gl.uniform1i(gl.getUniformLocation(this.progHoriz, "u_input"), 0);
      gl.uniform2f(gl.getUniformLocation(this.progHoriz, "u_texelSize"), 1.0 / this.width, 1.0 / this.height);
      
      this._setKernelUniforms(this.progHoriz, k0, k1, k2, k0sub);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

      // --- Pass 2: Vertical ---
      gl.useProgram(this.progVert);
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.featFbo);
      gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]); // Enable MRT

      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.horizTexture);
      gl.uniform1i(gl.getUniformLocation(this.progVert, "u_horiz"), 0);
      gl.uniform2f(gl.getUniformLocation(this.progVert, "u_texelSize"), 1.0 / this.width, 1.0 / this.height);
      
      this._setKernelUniforms(this.progVert, k0, k1, k2, k0sub);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

      gl.deleteTexture(inputTex);
  }

  _setKernelUniforms(prog, k0, k1, k2, k0sub) {
      const gl = this.gl;
      // Pad arrays to 33 to match shader limits (maxRadius = 32 + 1)
      const pad = (arr) => {
          const out = new Float32Array(33);
          out.set(arr);
          return out;
      };
      gl.uniform1fv(gl.getUniformLocation(prog, "u_k0"), pad(k0));
      gl.uniform1fv(gl.getUniformLocation(prog, "u_k1"), pad(k1));
      gl.uniform1fv(gl.getUniformLocation(prog, "u_k2"), pad(k2));
      gl.uniform1fv(gl.getUniformLocation(prog, "u_k0sub"), pad(k0sub));
      gl.uniform1i(gl.getUniformLocation(prog, "u_r0"), k0.length - 1);
      gl.uniform1i(gl.getUniformLocation(prog, "u_r1"), k1.length - 1);
      gl.uniform1i(gl.getUniformLocation(prog, "u_r2"), k2.length - 1);
      gl.uniform1i(gl.getUniformLocation(prog, "u_r0sub"), k0sub.length - 1);
  }

  /**
   * Computes features and reads all 8 channels back to the CPU, interleaved as
   * `[f0..f7]` per pixel.
   * @param {Float32Array} intensityArray
   * @param {number} scale - Gaussian sigma.
   * @returns {Promise<Float32Array>} width*height*8 features.
   */
  async downloadFeatures(intensityArray, scale) {
      this._extractFeatures(intensityArray, scale);
      const gl = this.gl;
      
      const pixels0 = new Float32Array(this.width * this.height * 4);
      const pixels1 = new Float32Array(this.width * this.height * 4);
      
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.featFbo);
      gl.readBuffer(gl.COLOR_ATTACHMENT0);
      gl.readPixels(0, 0, this.width, this.height, gl.RGBA, gl.FLOAT, pixels0);
      gl.readBuffer(gl.COLOR_ATTACHMENT1);
      gl.readPixels(0, 0, this.width, this.height, gl.RGBA, gl.FLOAT, pixels1);

      const out = new Float32Array(this.width * this.height * 8);
      for (let i = 0; i < this.width * this.height; i++) {
          out[i * 8 + 0] = pixels0[i * 4 + 0];
          out[i * 8 + 1] = pixels0[i * 4 + 1];
          out[i * 8 + 2] = pixels0[i * 4 + 2];
          out[i * 8 + 3] = pixels0[i * 4 + 3];
          out[i * 8 + 4] = pixels1[i * 4 + 0];
          out[i * 8 + 5] = pixels1[i * 4 + 1];
          out[i * 8 + 6] = pixels1[i * 4 + 2];
          out[i * 8 + 7] = pixels1[i * 4 + 3];
      }
      return out;
  }

  /**
   * Gathers the 8-feature vectors for a set of labeled pixels, to feed
   * FlatRandomForest.train. Returns them row-major as `numLabels * 8` floats.
   * @param {Uint32Array} indicesArray - Flat pixel indices (y * width + x).
   * @returns {Promise<Float32Array>} Features for each labeled pixel.
   */
  async gatherFeaturesForTraining(indicesArray) {
      // In WebGL2, reading full texture to CPU and extracting there is usually
      // cleaner/faster than drawing point primitives to scatter/gather.
      const gl = this.gl;
      const numLabels = indicesArray.length;
      
      const pixels0 = new Float32Array(this.width * this.height * 4);
      const pixels1 = new Float32Array(this.width * this.height * 4);
      
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.featFbo);
      gl.readBuffer(gl.COLOR_ATTACHMENT0);
      gl.readPixels(0, 0, this.width, this.height, gl.RGBA, gl.FLOAT, pixels0);
      gl.readBuffer(gl.COLOR_ATTACHMENT1);
      gl.readPixels(0, 0, this.width, this.height, gl.RGBA, gl.FLOAT, pixels1);

      const features = new Float32Array(numLabels * 8);
      for (let i = 0; i < numLabels; i++) {
          const pIdx = indicesArray[i];
          features[i * 8 + 0] = pixels0[pIdx * 4 + 0];
          features[i * 8 + 1] = pixels0[pIdx * 4 + 1];
          features[i * 8 + 2] = pixels0[pIdx * 4 + 2];
          features[i * 8 + 3] = pixels0[pIdx * 4 + 3];
          features[i * 8 + 4] = pixels1[pIdx * 4 + 0];
          features[i * 8 + 5] = pixels1[pIdx * 4 + 1];
          features[i * 8 + 6] = pixels1[pIdx * 4 + 2];
          features[i * 8 + 7] = pixels1[pIdx * 4 + 3];
      }
      return features;
  }

  /**
   * Runs the trained forest over every pixel, writing per-class probabilities
   * into the probability texture, then repaints the composite. The forest is
   * uploaded as a float texture; see the packing note below for why each node
   * field is converted by its true type.
   * @param {FlatRandomForest} rf - A trained forest (max 8 trees).
   * @throws If the forest has more than 8 trees.
   */
  async runInference(rf) {
      const gl = this.gl;

      const numTrees = rf.treeRoots.length;
      if (numTrees > 8) {
          throw new Error(
              `runInference: ${numTrees} trees provided, but only up to 8 are supported ` +
              `by the u_treeRoots uniform layout. Reduce the forest size or widen u_treeRoots.`
          );
      }

      // Pack the forest into a Float32 texture (floats represent ints exactly up
      // to 16.7M). forestBuffer bytes are mixed-type per node slot: feat_idx,
      // left and right are raw i32 bits while threshold is f32 (see rf.js), so
      // each field must be converted by its actual type — reading the i32 fields
      // through the Float32Array view yields denormal garbage that truncates to 0.
      const forestNodes = rf.forestBuffer.byteLength / 16;
      const texWidth = Math.ceil(Math.sqrt(forestNodes));
      const texHeight = Math.ceil(forestNodes / texWidth);

      const i32Forest = new Int32Array(rf.forestBuffer.buffer, rf.forestBuffer.byteOffset, rf.forestBuffer.length);
      const floatForest = new Float32Array(texWidth * texHeight * 4);
      for (let node = 0; node < forestNodes; node++) {
          const o = node * 4;
          floatForest[o + 0] = i32Forest[o + 0];      // feat_idx
          floatForest[o + 1] = rf.forestBuffer[o + 1]; // threshold
          floatForest[o + 2] = i32Forest[o + 2];      // left
          floatForest[o + 3] = i32Forest[o + 3];      // right
      }

      const forestTex = this._createTexture(texWidth, texHeight, gl.RGBA32F, gl.RGBA, gl.FLOAT, floatForest);

      gl.viewport(0, 0, this.width, this.height);
      gl.useProgram(this.progRF);
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.probFbo);
      gl.bindVertexArray(this.quadVao);

      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.featTexture0);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.featTexture1);
      gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, forestTex);
      
      gl.uniform1i(gl.getUniformLocation(this.progRF, "u_feat0"), 0);
      gl.uniform1i(gl.getUniformLocation(this.progRF, "u_feat1"), 1);
      gl.uniform1i(gl.getUniformLocation(this.progRF, "u_forest"), 2);
      gl.uniform1i(gl.getUniformLocation(this.progRF, "u_forestWidth"), texWidth);

      // Unused root slots must be -1: 0 is a valid node index, not an "empty slot"
      // sentinel, so zero-filled slots would silently re-run tree 0's traversal and
      // add phantom votes (same fix as WebGpuBackend.runInference).
      const paddedRoots = new Int32Array(8).fill(-1);
      paddedRoots.set(rf.treeRoots);
      gl.uniform1iv(gl.getUniformLocation(this.progRF, "u_treeRoots"), paddedRoots);
      gl.uniform1i(gl.getUniformLocation(this.progRF, "u_numTrees"), numTrees);
      // Real traversal bound instead of a hardcoded depth that truncated deep trees.
      gl.uniform1i(gl.getUniformLocation(this.progRF, "u_maxDepth"), rf.maxDepth ?? 24);

      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

      gl.deleteTexture(forestTex);
      this.renderComposite();
  }

  /** Paints the canvas: the original image with the argmax class overlaid where classified. */
  renderComposite() {
      if (!this.rawIntensityTexture) return;
      const gl = this.gl;

      gl.viewport(0, 0, this.width, this.height);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null); // Render to canvas
      gl.useProgram(this.progComposite);
      gl.bindVertexArray(this.quadVao);

      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.rawIntensityTexture);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.probTexture);
      
      gl.uniform1i(gl.getUniformLocation(this.progComposite, "u_original"), 0);
      gl.uniform1i(gl.getUniformLocation(this.progComposite, "u_probs"), 1);
      gl.uniform1f(gl.getUniformLocation(this.progComposite, "u_winLo"), this.windowLo);
      gl.uniform1f(gl.getUniformLocation(this.progComposite, "u_winHi"), this.windowHi);
      gl.uniform4fv(gl.getUniformLocation(this.progComposite, "u_colors"), this._colorData);

      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  /**
   * Sets the display-only contrast window (black point `lo`, white point `hi`,
   * in the image's raw intensity units) and repaints. This only affects the
   * composite pass — it does not touch the intensity data fed to feature
   * extraction, so classification is unchanged and no retrain is triggered.
   * @param {number} lo - Black point; pixels <= lo render black.
   * @param {number} hi - White point; pixels >= hi render white.
   */
  setWindow(lo, hi) {
      this.windowLo = lo;
      this.windowHi = hi;
      this.renderComposite();
  }

  /**
   * Updates the per-class overlay colors and repaints. Only repacks the u_colors
   * uniform and re-records the draw with the existing program — no shader
   * recompile — so a color edit is cheap. Recolors the classes the image was
   * allocated with (the class *count* is fixed for the image's lifetime); any
   * extra entries in `colors` are ignored, and any missing ones leave that class's
   * current color unchanged.
   * @param {string[]} colors - CSS color strings indexed by class.
   */
  setColors(colors) {
      for (let i = 0; i < this.labelColors.length && i < colors.length; i++) {
          this.labelColors[i] = colors[i];
      }
      this._colorData = packColors(this.labelColors);
      this.renderComposite();
  }

  /**
   * Reads the probability map back to the CPU as `numColors` channels per pixel,
   * packed sequentially.
   * @returns {Promise<Float32Array>} width*height*numColors probabilities.
   */
  async downloadProbabilities() {
      const gl = this.gl;
      const pixels = new Float32Array(this.width * this.height * 4);
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.probFbo);
      gl.readPixels(0, 0, this.width, this.height, gl.RGBA, gl.FLOAT, pixels);
      
      // Output format expected: numColors channels packed sequentially 
      // For simplicity, returning the raw RGBA buffer (assuming max 4 labels)
      const numColors = this.labelColors.length;
      const out = new Float32Array(this.width * this.height * numColors);
      for(let i = 0; i < this.width * this.height; i++) {
          for(let c = 0; c < numColors; c++) {
              out[i * numColors + c] = pixels[i * 4 + c];
          }
      }
      return out;
  }

  /**
   * Connected Component Labeling over the live probability map.
   *
   * WebGL2 has no compute shaders or atomics, so the WebGPU backend's GPU
   * union-find can't be ported directly, and an iterative ping-pong label
   * propagation shader would need O(component diameter) passes (failing on
   * spirals/long cells with any fixed pass count). Since this backend is the
   * fallback path, run the union-find on the CPU instead — same output
   * contract as WebGpuBackend: 4-connectivity, background = 0, and each
   * component labeled with its minimum pixel index + 1.
   */
  async computeConnectedComponents(targetClassIdx, threshold = 0.5) {
      const gl = this.gl;
      const n = this.width * this.height;

      // Probabilities live in the RGBA channels of probTexture (max 4 classes)
      const pixels = new Float32Array(n * 4);
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.probFbo);
      gl.readPixels(0, 0, this.width, this.height, gl.RGBA, gl.FLOAT, pixels);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);

      const mask = new Uint8Array(n);
      for (let i = 0; i < n; i++) {
          if (pixels[i * 4 + targetClassIdx] >= threshold) mask[i] = 1;
      }

      this.labels = cclLabel(mask, this.width, this.height);
      return this.labels;
  }

  /**
   * Compiles area and cumulative intensity metric profiles per label ID.
   * Produces the same dense structs as WebGpuBackend.computeStats (see STATS_LAYOUT).
   */
  async computeStats() {
      if (!this.labels) return null;
      const gl = this.gl;
      const n = this.width * this.height;

      // Read the raw intensity texture (R32F) back for the red-channel intensity.
      const fbo = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.rawIntensityTexture, 0);
      const raw = new Float32Array(n * 4);
      gl.readPixels(0, 0, this.width, this.height, gl.RGBA, gl.FLOAT, raw);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.deleteFramebuffer(fbo);

      // Raw value -> fixed-point integer (scale 1 for integer dtypes, exact),
      // matching u32(raw * SCALE) in the WebGPU stats shader.
      const intensity = new Float64Array(n);
      for (let i = 0; i < n; i++) {
          intensity[i] = Math.floor(raw[i * 4] * this.intensityScale);
      }

      this.denseStats = accumulateStats(this.labels, intensity, this.width, this.height);
  }

  /**
   * Returns the dense 7-field-per-object stats from the last computeStats call
   * (empty if none). @returns {Promise<Uint32Array>}
   */
  async downloadStats() {
      return this.denseStats ?? new Uint32Array(0);
  }

  /**
   * Returns a copy of the last computed component labels (one u32 per pixel;
   * zeros if none computed yet). @returns {Promise<Uint32Array>}
   */
  async downloadLabels() {
      return this.labels ? this.labels.slice() : new Uint32Array(this.width * this.height);
  }

  // --- WebGL2 Helpers ---

  _setupQuad() {
      const gl = this.gl;
      this.quadVao = gl.createVertexArray();
      gl.bindVertexArray(this.quadVao);
      const vbo = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
          -1, -1,  1, -1,  -1, 1,  1, 1
      ]), gl.STATIC_DRAW);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
      gl.bindVertexArray(null);
  }

  _createTexture(w, h, internalFormat, format, type, data) {
      const gl = this.gl;
      const tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, w, h, 0, format, type, data);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      return tex;
  }

  _createFbo(textures) {
      const gl = this.gl;
      const fbo = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      const drawBuffers = [];
      for (let i = 0; i < textures.length; i++) {
          gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, textures[i], 0);
          drawBuffers.push(gl.COLOR_ATTACHMENT0 + i);
      }
      gl.drawBuffers(drawBuffers);
      if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
          throw new Error("FBO incomplete");
      }
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      return fbo;
  }

  /** Frees all GPU textures, framebuffers, programs, and the quad VAO. */
  destroy() {
      const gl = this.gl;
      if (!gl) return;
      [this.rawIntensityTexture, this.horizTexture, this.featTexture0,
       this.featTexture1, this.probTexture].forEach(t => { if (t) gl.deleteTexture(t); });
      [this.horizFbo, this.featFbo, this.probFbo].forEach(f => { if (f) gl.deleteFramebuffer(f); });
      [this.progHoriz, this.progVert, this.progRF, this.progComposite].forEach(p => { if (p) gl.deleteProgram(p); });
      if (this.quadVao) gl.deleteVertexArray(this.quadVao);
  }

  _compileShaders() {
      const gl = this.gl;
      const createProgram = (fsSource) => {
          const vs = gl.createShader(gl.VERTEX_SHADER);
          gl.shaderSource(vs, VS_QUAD); gl.compileShader(vs);
          const fs = gl.createShader(gl.FRAGMENT_SHADER);
          gl.shaderSource(fs, fsSource); gl.compileShader(fs);
          if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(fs));
          const prog = gl.createProgram();
          gl.attachShader(prog, vs); gl.attachShader(prog, fs);
          gl.linkProgram(prog);
          return prog;
      };

      this.progHoriz = createProgram(FS_HORIZ);
      this.progVert = createProgram(FS_VERT);
      
      const numColors = this.labelColors.length;
      this.progRF = createProgram(FS_RF_INFERENCE.replace(/{{NUM_COLORS}}/g, numColors));

      this.progComposite = createProgram(FS_COMPOSITE.replace(/{{NUM_COLORS}}/g, numColors));
      this._colorData = packColors(this.labelColors);
  }
}

/**
 * Parses any valid CSS color string (hex3/6/8, rgb/rgba, hsl/hsla, named colors,
 * etc.) into normalized [r, g, b, a] floats in 0–1, by letting the browser's own
 * CSS color parser do the work via a 1x1 canvas instead of hand-rolled regex.
 * Mirrors parseColorToRGBA in webgpu.js.
 * @param {string} colorStr - Any CSS color string.
 * @returns {[number, number, number, number]} rgba components in 0–1.
 */
let _colorParseCtx = null;
function parseColorToRGBA(colorStr) {
    if (!_colorParseCtx) {
        const canvas = document.createElement('canvas');
        canvas.width = 1;
        canvas.height = 1;
        _colorParseCtx = canvas.getContext('2d', { willReadFrequently: true });
    }
    const ctx = _colorParseCtx;
    ctx.fillStyle = '#ff0000'; // fallback color if the string below fails to parse
    ctx.fillStyle = colorStr;  // no-op (silently ignored) if colorStr isn't valid CSS
    ctx.clearRect(0, 0, 1, 1);
    ctx.fillRect(0, 0, 1, 1);
    const [r, g, b, a] = ctx.getImageData(0, 0, 1, 1).data;
    return [r / 255, g / 255, b / 255, a / 255];
}

/**
 * Packs an array of CSS color strings into a flat rgba (0–1) Float32Array — one
 * vec4 per class — for upload as the composite pass's u_colors uniform.
 * @param {string[]} colors - CSS color strings indexed by class.
 * @returns {Float32Array} length colors.length * 4.
 */
function packColors(colors) {
    const data = new Float32Array(colors.length * 4);
    for (let i = 0; i < colors.length; i++) {
        const [r, g, b, a] = parseColorToRGBA(colors[i]);
        data[i * 4 + 0] = r;
        data[i * 4 + 1] = g;
        data[i * 4 + 2] = b;
        data[i * 4 + 3] = a;
    }
    return data;
}

/**
 * CPU union-find CCL over a binary mask. 4-connectivity; background pixels get
 * label 0; every component is labeled with its minimum pixel index + 1,
 * matching the labeling semantics of the WebGPU backend's CCL_SHADER.
 * Exported for testing outside the browser.
 */
export function cclLabel(mask, width, height) {
    const n = width * height;
    const parent = new Int32Array(n).fill(-1); // -1 = background, else parent pixel index
    for (let i = 0; i < n; i++) {
        if (mask[i]) parent[i] = i;
    }

    function find(i) {
        let root = i;
        while (parent[root] !== root) root = parent[root];
        while (parent[i] !== root) { // path compression
            const next = parent[i];
            parent[i] = root;
            i = next;
        }
        return root;
    }

    // Union by minimum index, so a component's final root is its minimum pixel index
    function union(a, b) {
        const ra = find(a);
        const rb = find(b);
        if (ra === rb) return;
        if (ra < rb) parent[rb] = ra;
        else parent[ra] = rb;
    }

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const i = y * width + x;
            if (parent[i] < 0) continue;
            if (x + 1 < width && parent[i + 1] >= 0) union(i, i + 1);
            if (y + 1 < height && parent[i + width] >= 0) union(i, i + width);
        }
    }

    const labels = new Uint32Array(n); // background stays 0
    for (let i = 0; i < n; i++) {
        if (parent[i] >= 0) labels[i] = find(i) + 1;
    }
    return labels;
}

/**
 * Accumulates per-label metrics into the same dense structs as
 * WebGpuBackend.downloadStats (see STATS_LAYOUT): label, area, total_intensity
 * {lo,hi}, sum_x {lo,hi}, sum_y {lo,hi}, min_intensity, max_intensity. The summed
 * fields are 64-bit, split into two u32 words to match the WebGPU accumulator's
 * paired-atomic layout; sums are computed exactly in JS (f64 is exact to 2^53) and
 * then split. Exported for testing outside the browser.
 */
export function accumulateStats(labels, intensity, width, height) {
    const structCount = STATS_LAYOUT.denseCount;
    const rowIndex = new Map(); // label -> index into rows
    const rows = [];

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const i = y * width + x;
            const label = labels[i];
            if (label === 0) continue;

            let s = rowIndex.get(label);
            if (s === undefined) {
                s = rows.length;
                rowIndex.set(label, s);
                rows.push({ label, area: 0, total: 0, sumX: 0, sumY: 0, min: 0xFFFFFFFF, max: 0 });
            }

            const r = rows[s];
            const v = intensity[i];
            r.area++;
            r.total += v;
            r.sumX += x;
            r.sumY += y;
            if (v < r.min) r.min = v;
            if (v > r.max) r.max = v;
        }
    }

    // Split a JS number into low/high 32-bit words (little-endian: [lo, hi]).
    const lo = (v) => v >>> 0;
    const hi = (v) => Math.floor(v / 2 ** 32);

    const out = new Uint32Array(rows.length * structCount);
    for (let s = 0; s < rows.length; s++) {
        const r = rows[s];
        const o = s * structCount;
        out[o]     = r.label;
        out[o + 1] = r.area;
        out[o + 2] = lo(r.total); out[o + 3] = hi(r.total);
        out[o + 4] = lo(r.sumX);  out[o + 5] = hi(r.sumX);
        out[o + 6] = lo(r.sumY);  out[o + 7] = hi(r.sumY);
        out[o + 8] = r.min;
        out[o + 9] = r.max;
    }
    return out;
}

// 1D Gaussian kernel (or its 1st/2nd analytical derivative), returned as the
// right half [0..radius]. Kept byte-for-byte identical to gaussian_kernel in
// webgpu.js so both backends produce the same features; see that copy for the
// fully-documented version.
function gaussian_kernel(scale, order = 0) {
if (scale <= 0) throw new Error("scale should be greater than 0");
const radius = Math.ceil((3.0 + 0.5 * order) * scale);
const kernel = new Float32Array(radius + 1);
const twoSigmaSq = 2.0 * scale * scale;
const fullSize = 2 * radius + 1;
const fullKernel = new Float32Array(fullSize);

let sum = 0;
for (let i = -radius; i <= radius; i++) {
  const x = i;
  let val = Math.exp(-(x * x) / twoSigmaSq);
  if (order === 1) val = (-x / (scale * scale)) * val;
  else if (order === 2) val = (((x * x) / (scale * scale * scale * scale)) - (1.0 / (scale * scale))) * val;
  fullKernel[i + radius] = val;
  if (order === 0) sum += val;
}

if (order === 0) {
  for (let i = 0; i < fullSize; i++) fullKernel[i] /= sum;
} else if (order === 1) {
  let sumX = 0;
  for (let i = -radius; i <= radius; i++) sumX += i * fullKernel[i + radius];
  for (let i = 0; i < fullSize; i++) fullKernel[i] /= sumX;
} else if (order === 2) {
  let sum0 = 0;
  for (let i = -radius; i <= radius; i++) sum0 += fullKernel[i + radius];
  const mean = sum0 / fullSize;
  for (let i = 0; i < fullSize; i++) fullKernel[i] -= mean;
  let sumX2 = 0;
  for (let i = -radius; i <= radius; i++) sumX2 += 0.5 * i * i * fullKernel[i + radius];
  for (let i = 0; i < fullSize; i++) fullKernel[i] /= sumX2;
}

for (let i = 0; i <= radius; i++) kernel[i] = fullKernel[radius + i];
return kernel;
}


const VS_QUAD = `#version 300 es
layout(location = 0) in vec2 a_pos;
out vec2 v_uv;
void main() {
  v_uv = a_pos * 0.5 + 0.5;
  gl_Position = vec4(a_pos, 0.0, 1.0);
}`;

const FS_HORIZ = `#version 300 es
precision highp float;
in vec2 v_uv;
uniform sampler2D u_input;
uniform vec2 u_texelSize;

uniform float u_k0[33];
uniform float u_k1[33];
uniform float u_k2[33];
uniform float u_k0sub[33];
uniform int u_r0; uniform int u_r1; uniform int u_r2; uniform int u_r0sub;

out vec4 out_horiz;

float get_val(vec2 uv, int dx) {
  return texture(u_input, uv + vec2(float(dx) * u_texelSize.x, 0.0)).r;
}

void main() {
  float c = get_val(v_uv, 0);
  
  float h0 = u_k0[0] * c;
  for(int i = 1; i <= u_r0; i++) h0 += u_k0[i] * (get_val(v_uv, -i) + get_val(v_uv, i));
  
  float h1 = 0.0;
  for(int i = 1; i <= u_r1; i++) h1 += u_k1[i] * (get_val(v_uv, i) - get_val(v_uv, -i));
  
  float h2 = u_k2[0] * c;
  for(int i = 1; i <= u_r2; i++) h2 += u_k2[i] * (get_val(v_uv, -i) + get_val(v_uv, i));
  
  float h0s = u_k0sub[0] * c;
  for(int i = 1; i <= u_r0sub; i++) h0s += u_k0sub[i] * (get_val(v_uv, -i) + get_val(v_uv, i));

  out_horiz = vec4(h0, h1, h2, h0s);
}`;

const FS_VERT = `#version 300 es
precision highp float;
in vec2 v_uv;
uniform sampler2D u_horiz;
uniform vec2 u_texelSize;

uniform float u_k0[33];
uniform float u_k1[33];
uniform float u_k2[33];
uniform float u_k0sub[33];
uniform int u_r0; uniform int u_r1; uniform int u_r2; uniform int u_r0sub;

layout(location = 0) out vec4 out_feat0;
layout(location = 1) out vec4 out_feat1;

vec4 get_h(vec2 uv, int dy) {
  return texture(u_horiz, uv + vec2(0.0, float(dy) * u_texelSize.y));
}

void main() {
  vec4 c = get_h(v_uv, 0);

  vec4 l_vec = c * u_k0[0];
  for(int i = 1; i <= u_r0; i++) l_vec += (get_h(v_uv, -i) + get_h(v_uv, i)) * u_k0[i];
  float L = l_vec.x; float Lx = l_vec.y; float Lxx = l_vec.z;

  vec2 ly_vec = vec2(0.0);
  for(int i = 1; i <= u_r1; i++) ly_vec += (get_h(v_uv, i) - get_h(v_uv, -i)).xy * u_k1[i];
  float Ly = ly_vec.x; float Lxy = ly_vec.y;

  float lyy = c.x * u_k2[0];
  for(int i = 1; i <= u_r2; i++) lyy += (get_h(v_uv, -i).x + get_h(v_uv, i).x) * u_k2[i];

  float lsub_vec = c.w * u_k0sub[0];
  for(int i = 1; i <= u_r0sub; i++) lsub_vec += (get_h(v_uv, -i).w + get_h(v_uv, i).w) * u_k0sub[i];

  out_feat0 = vec4(L, Lxx + lyy, sqrt(Lx*Lx + Ly*Ly), L - lsub_vec);

  float s_a = Lx*Lx; float s_b = Lx*Ly; float s_c = Ly*Ly;
  float s_term = sqrt((s_a - s_c)*(s_a - s_c) * 0.25 + s_b*s_b);
  float st_min = (s_a + s_c) * 0.5 - s_term; // Largest and smallest swap intentionally based on your wgsl
  float st_max = (s_a + s_c) * 0.5 + s_term;
  
  float h_term = sqrt((Lxx - lyy)*(Lxx - lyy) * 0.25 + Lxy*Lxy);
  float h_min = (Lxx + lyy) * 0.5 - h_term;
  float h_max = (Lxx + lyy) * 0.5 + h_term;

  out_feat1 = vec4(st_max, st_min, h_max, h_min);
}`;

const FS_RF_INFERENCE = `#version 300 es
precision highp float;
in vec2 v_uv;

uniform sampler2D u_feat0;
uniform sampler2D u_feat1;
uniform sampler2D u_forest;
uniform int u_forestWidth;
uniform int u_treeRoots[8]; // unused slots are -1
uniform int u_numTrees;
uniform int u_maxDepth;

out vec4 out_probs;

vec4 getNode(int idx) {
  int y = idx / u_forestWidth;
  int x = idx % u_forestWidth;
  return texelFetch(u_forest, ivec2(x, y), 0);
}

float getFeat(int f_idx) {
  if (f_idx < 4) {
      vec4 f = texture(u_feat0, v_uv);
      if(f_idx == 0) return f.x; if(f_idx == 1) return f.y; 
      if(f_idx == 2) return f.z; return f.w;
  } else {
      vec4 f = texture(u_feat1, v_uv);
      if(f_idx == 4) return f.x; if(f_idx == 5) return f.y; 
      if(f_idx == 6) return f.z; return f.w;
  }
}

void main() {
  // Fixed size 4 (RGBA channel cap on labels elsewhere in this backend), not
  // {{NUM_COLORS}}: GLSL ES 3.00 rejects a constant out-of-bounds array index
  // even inside a branch guarded by "if ({{NUM_COLORS}} > 3)" below, so a
  // votes[{{NUM_COLORS}}] array failed to compile whenever NUM_COLORS <= 3.
  float votes[4];
  for(int i=0; i<4; i++) votes[i] = 0.0;
  
  for (int t = 0; t < u_numTrees; t++) {
      int node_idx = u_treeRoots[t];
      if (node_idx < 0) continue;

      for (int depth = 0; depth < u_maxDepth; depth++) {
          vec4 node = getNode(node_idx);
          int feat_idx = int(node.x);
          float threshold = node.y;
          int left = int(node.z);
          int right = int(node.w);

          if (feat_idx == -1) {
              int class_id = -right - 1;
              if (class_id >= 0 && class_id < {{NUM_COLORS}}) {
                  votes[class_id] += 1.0;
              }
              break;
          }
          
          float val = getFeat(feat_idx);
          if (val < threshold) {
              node_idx = left;
          } else {
              node_idx = right;
          }
      }
  }

  float denom = max(float(u_numTrees), 1.0);
  vec4 final_probs = vec4(0.0);
  if ({{NUM_COLORS}} > 0) final_probs.r = votes[0] / denom;
  if ({{NUM_COLORS}} > 1) final_probs.g = votes[1] / denom;
  if ({{NUM_COLORS}} > 2) final_probs.b = votes[2] / denom;
  if ({{NUM_COLORS}} > 3) final_probs.a = votes[3] / denom;
  
  out_probs = final_probs;
}`;

const FS_COMPOSITE = `#version 300 es
precision highp float;
in vec2 v_uv;
uniform sampler2D u_original; // R32F raw intensity in the red channel
uniform sampler2D u_probs;
uniform float u_winLo; // display contrast black point (raw intensity units)
uniform float u_winHi; // display contrast white point (raw intensity units)
// Per-class overlay colors as a uniform (not baked shader constants) so recoloring
// a class only needs a uniform4fv upload, not a shader recompile (see setColors).
uniform vec4 u_colors[{{NUM_COLORS}}];

out vec4 fragColor;

void main() {
  // Read raw intensity (red channel) and apply the display-only contrast window
  // [lo,hi] -> [0,1] in raw units, broadcast to gray.
  float raw_val = texture(u_original, v_uv).r;
  float v = clamp((raw_val - u_winLo) / max(u_winHi - u_winLo, 1e-4), 0.0, 1.0);
  vec4 raw = vec4(v, v, v, 1.0);
  vec4 probs = texture(u_probs, v_uv);
  
  float max_p = -1.0;
  int best_class = -1;
  
  float p_arr[4] = float[](probs.r, probs.g, probs.b, probs.a);
  
  for(int c = 0; c < {{NUM_COLORS}}; c++) {
      if (p_arr[c] > max_p) {
          max_p = p_arr[c];
          best_class = c;
      }
  }
  
  float alpha = 0.4;
  if (max_p < 0.0) {
      fragColor = vec4(raw.rgb, 1.0);
      return;
  }

  vec4 overlay = vec4(0.0, 0.0, 0.0, alpha);
  if (best_class >= 0 && best_class < {{NUM_COLORS}}) {
      overlay = u_colors[best_class];
      overlay.a = alpha;
  }

  fragColor = vec4(mix(raw.rgb, overlay.rgb, alpha), 1.0);
}`;
