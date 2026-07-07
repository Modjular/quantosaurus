export class WebGl2Backend {
  constructor(labelColors) {
      this.gl = null;
      this.width = 0;
      this.height = 0;
      this.originalTexture = null;
      
      // FBOs and Textures
      this.horizFbo = null;
      this.horizTexture = null;
      this.featFbo = null;
      this.featTexture0 = null; // Features 0-3
      this.featTexture1 = null; // Features 4-7
      this.probFbo = null;
      this.probTexture = null;

      this.labelColors = labelColors || [
          'rgba(255,0,0,1.0)',
          'rgba(0,255,0,1.0)',
          'rgba(0,0,255,1.0)',
      ];

      // Shader Programs
      this.progHoriz = null;
      this.progVert = null;
      this.progRF = null;
      this.progComposite = null;
      this.quadVao = null;
  }

  async initialize(canvas) {
      this.gl = canvas.getContext('webgl2', { antialias: false });
      if (!this.gl) throw new Error("WebGL2 not supported");

      const ext = this.gl.getExtension('EXT_color_buffer_float');
      if (!ext) throw new Error("EXT_color_buffer_float not supported. Cannot render to float textures.");

      this._setupQuad();
      this._compileShaders();
  }

  async allocateImage(width, height, rgbaData) {
      this.width = width;
      this.height = height;
      const gl = this.gl;

      this.originalTexture = this._createTexture(width, height, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, rgbaData);
      
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

  async updateFeatures(intensityArray, sigma) {
      this._extractFeatures(intensityArray, sigma);
      this.renderComposite();
  }

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

  async runInference(rf) {
      const gl = this.gl;

      // Pack Int32 struct into Float32 Texture (floats perfectly represent ints up to 16.7M)
      const forestNodes = rf.forestBuffer.byteLength / 16;
      const texWidth = Math.ceil(Math.sqrt(forestNodes));
      const texHeight = Math.ceil(forestNodes / texWidth);
      
      const paddedForest = new Int32Array(texWidth * texHeight * 4);
      paddedForest.set(new Int32Array(rf.forestBuffer));
      const floatForest = new Float32Array(paddedForest); // Cast bits to float

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

      const paddedRoots = new Int32Array(8);
      paddedRoots.set(rf.treeRoots);
      gl.uniform1iv(gl.getUniformLocation(this.progRF, "u_treeRoots"), paddedRoots);

      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

      gl.deleteTexture(forestTex);
      this.renderComposite();
  }

  renderComposite() {
      if (!this.originalTexture) return;
      const gl = this.gl;

      gl.viewport(0, 0, this.width, this.height);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null); // Render to canvas
      gl.useProgram(this.progComposite);
      gl.bindVertexArray(this.quadVao);

      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.originalTexture);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.probTexture);
      
      gl.uniform1i(gl.getUniformLocation(this.progComposite, "u_original"), 0);
      gl.uniform1i(gl.getUniformLocation(this.progComposite, "u_probs"), 1);

      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

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

  destroy() {
      const gl = this.gl;
      if (!gl) return;
      [this.originalTexture, this.horizTexture, this.featTexture0,
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

      const parseColor = (c) => {
          let m = c.match(/rgba?\((\d+),\s*(\d+),\s*(\d+),?\s*([\d.]+)?\)/);
          if (m) return `vec4(${m[1]/255}, ${m[2]/255}, ${m[3]/255}, ${m[4] || '0.8'})`;
          m = c.match(/^#([0-9a-f]{6})$/i)[1];
          if (m) return `vec4(${parseInt(m.substr(0,2),16)/255}, ${parseInt(m.substr(2,2),16)/255}, ${parseInt(m.substr(4,2),16)/255}, 0.8)`;
          return "vec4(1.0, 0.0, 0.0, 1.0)";
      };
      const colorsGLSL = this.labelColors.map(parseColor).join(',\n    ');
      this.progComposite = createProgram(FS_COMPOSITE
          .replace(/{{NUM_COLORS}}/g, numColors)
          .replace(/{{COLORS_ARRAY}}/g, colorsGLSL));
  }
}

// Keep the exact same gaussian_kernel JS function here.
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
uniform int u_treeRoots[8];

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
  float votes[{{NUM_COLORS}}];
  for(int i=0; i<{{NUM_COLORS}}; i++) votes[i] = 0.0;
  
  float num_trees = 8.0;

  for (int t = 0; t < 8; t++) {
      int node_idx = u_treeRoots[t];
      if (node_idx < 0) continue;

      for (int depth = 0; depth < 10; depth++) {
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

  vec4 final_probs = vec4(0.0);
  if ({{NUM_COLORS}} > 0) final_probs.r = votes[0] / num_trees;
  if ({{NUM_COLORS}} > 1) final_probs.g = votes[1] / num_trees;
  if ({{NUM_COLORS}} > 2) final_probs.b = votes[2] / num_trees;
  if ({{NUM_COLORS}} > 3) final_probs.a = votes[3] / num_trees;
  
  out_probs = final_probs;
}`;

const FS_COMPOSITE = `#version 300 es
precision highp float;
in vec2 v_uv;
uniform sampler2D u_original;
uniform sampler2D u_probs;

out vec4 fragColor;

void main() {
  vec4 raw = texture(u_original, v_uv);
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
  
  vec4 colors[{{NUM_COLORS}}] = vec4[](
      {{COLORS_ARRAY}}
  );

  vec4 overlay = vec4(0.0, 0.0, 0.0, alpha);
  if (best_class >= 0 && best_class < {{NUM_COLORS}}) {
      overlay = colors[best_class];
      overlay.a = alpha;
  }

  fragColor = vec4(mix(raw.rgb, overlay.rgb, alpha), 1.0);
}`;
