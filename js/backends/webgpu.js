/**
 * WebGPU compute/render backend — the preferred path. Interchangeable with
 * WebGl2Backend; both expose the same interface used by the image lifecycle and
 * training code:
 *
 *   initialize, allocateImage, updateFeatures, downloadFeatures,
 *   gatherFeaturesForTraining, runInference, renderComposite,
 *   downloadProbabilities, computeConnectedComponents, computeStats,
 *   downloadStats, downloadLabels, destroy.
 *
 * Everything runs on-GPU via compute shaders: a separable Gaussian-derivative
 * filter bank produces 8 per-pixel features; a Random Forest pass turns them
 * into per-class probabilities; connected-component labeling uses an atomic
 * parallel union-find; a stats pass accumulates per-object metrics and compacts
 * them; a render pass composites the argmax overlay. Data is only read back to
 * the CPU on demand through the download* methods.
 */
export class WebGpuBackend {
    constructor(labelColors) {
        this.device = null;
        this.context = null;
        this.format = null;
        this.width = 0;
        this.height = 0;
        this.originalTexture = null;
        this.featureBuffer = null;
        this.probBuffer = null;
        this.labelBuffer = null;
        this.statsBuffer = null;
        this.statsCounterBuffer = null;

        this.labelColors = labelColors || [
            'rgba(255,0,0,1.0)',
            'rgba(0,255,0,1.0)',
            'rgba(0,0,255,1.0)',
        ];
    }

    /**
     * Requests the GPU adapter/device (raising buffer-size limits to the
     * adapter maximum) and configures the canvas context.
     * @param {HTMLCanvasElement} canvas
     * @throws If WebGPU is unavailable.
     */
    async initialize(canvas) {
        if (!navigator.gpu) throw new Error("WebGPU not supported");
        const adapter = await navigator.gpu.requestAdapter();

        const requiredLimits = {};
        if (adapter.limits.maxBufferSize) requiredLimits.maxBufferSize = adapter.limits.maxBufferSize;
        if (adapter.limits.maxStorageBufferBindingSize) requiredLimits.maxStorageBufferBindingSize = adapter.limits.maxStorageBufferBindingSize;

        this.device = await adapter.requestDevice({ requiredLimits });
        this.context = canvas.getContext('webgpu');
        this.format = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({ device: this.device, format: this.format });
    }

    /**
     * Runs a single compute pass on a caller-supplied encoder. Centralizes the
     * "compile module -> create pipeline -> create bind group -> record pass" boilerplate.
     *
     * Multiple calls can share one encoder (e.g. multi-pass pipelines like CCL),
     * or each call can use its own encoder + submit if passes must be sequenced
     * around a CPU readback in between.
     */
    _addComputePass(encoder, { code, entryPoint, bindings, dispatchX, dispatchY = 1, dispatchZ = 1 }) {
        const module = this.device.createShaderModule({ code });
        const pipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: { module, entryPoint }
        });
        const bindGroup = this.device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: bindings
        });

        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
        pass.end();
    }

    /**
     * Copies a GPU buffer back to system RAM: creates a staging buffer, copies,
     * submits, maps, reads, and cleans up.
     */
    async _readBuffer(sourceBuffer, byteLength, ArrayType = Float32Array) {
        const staging = this.device.createBuffer({
            size: byteLength,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });

        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(sourceBuffer, 0, staging, 0, byteLength);
        this.device.queue.submit([encoder.finish()]);

        await staging.mapAsync(GPUMapMode.READ);
        const data = new ArrayType(staging.getMappedRange().slice());
        staging.unmap();
        staging.destroy();

        return data;
    }

    /**
     * (Re)allocates the per-image GPU resources (original texture, probability
     * and label buffers), freeing any previous ones, and seeds probabilities to
     * -1 (the "unclassified" sentinel). Feature/stats buffers are cleared and
     * rebuilt lazily on the next updateFeatures/computeStats.
     * @param {number} width
     * @param {number} height
     * @param {Uint8Array|Uint8ClampedArray} rgbaData - Source RGBA pixels.
     */
    async allocateImage(width, height, rgbaData) {
        this.width = width;
        this.height = height;

        if (this.originalTexture) this.originalTexture.destroy();
        this.originalTexture = this.device.createTexture({
            size: [width, height],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
        });
        this.device.queue.writeTexture({ texture: this.originalTexture }, rgbaData, { bytesPerRow: width * 4 }, [width, height]);

        const numColors = this.labelColors.length;
        if (this.probBuffer) this.probBuffer.destroy();
        this.probBuffer = this.device.createBuffer({
            size: width * height * numColors * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
        });

        const initialProbs = new Float32Array(width * height * numColors).fill(-1.0);
        this.device.queue.writeBuffer(this.probBuffer, 0, initialProbs);

        // Allocate/Reset Component Label Buffers (u32 per pixel)
        if (this.labelBuffer) this.labelBuffer.destroy();
        this.labelBuffer = this.device.createBuffer({
            size: width * height * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });

        if (this.featureBuffer) { this.featureBuffer.destroy(); this.featureBuffer = null; }
        if (this.statsBuffer) { this.statsBuffer.destroy(); this.statsBuffer = null; }
        if (this.statsCounterBuffer) { this.statsCounterBuffer.destroy(); this.statsCounterBuffer = null; }
    }

    /**
     * Recomputes the persistent feature buffer for the given intensity image at
     * the given sigma, then repaints the composite view.
     * @param {Float32Array} intensityArray - Normalized single-channel intensities.
     * @param {number} sigma - Gaussian scale for the filter bank.
     */
    async updateFeatures(intensityArray, sigma) {
        if (this.featureBuffer) this.featureBuffer.destroy();
        this.featureBuffer = await this._extractFeatures(intensityArray, sigma);
        this.renderComposite();
    }

    /**
     * Multi-pass Connected Component Labeling Pipeline
     * Processes the live probability buffer entirely on-GPU.
     */
    async computeConnectedComponents(targetClassIdx, threshold = 0.5) {
        const baseCode = CCL_SHADER
            .replace(/{{WIDTH}}/g, this.width)
            .replace(/{{HEIGHT}}/g, this.height)
            .replace(/{{NUM_COLORS}}/g, this.labelColors.length)
            .replace(/{{TARGET_CLASS}}/g, targetClassIdx)
            .replace(/{{THRESHOLD}}/g, threshold.toFixed(4));

        const dispatchX = Math.ceil(this.width / 16);
        const dispatchY = Math.ceil(this.height / 16);

        const enc = this.device.createCommandEncoder();

        // Pass 1: Initialize values based on raw inference thresholds
        this._addComputePass(enc, {
            code: baseCode,
            entryPoint: 'init_labels',
            bindings: [
                { binding: 0, resource: { buffer: this.probBuffer } },
                { binding: 1, resource: { buffer: this.labelBuffer } }
            ],
            dispatchX, dispatchY
        });

        // Pass 2: Neighborhood boundary checking and equivalence linking
        this._addComputePass(enc, {
            code: baseCode,
            entryPoint: 'merge_neighbors',
            bindings: [
                { binding: 1, resource: { buffer: this.labelBuffer } }
            ],
            dispatchX, dispatchY
        });

        // Pass 3: Path relaxation (Tree flattening) to find absolute roots
        this._addComputePass(enc, {
            code: baseCode,
            entryPoint: 'flatten_paths',
            bindings: [
                { binding: 1, resource: { buffer: this.labelBuffer } }
            ],
            dispatchX, dispatchY
        });

        this.device.queue.submit([enc.finish()]);
        return this.labelBuffer;
    }

    /**
     * Compiles area and cumulative intensity metric profiles per label ID.
     */
    async computeStats() {
        const dispatchX = Math.ceil(this.width / 16);
        const dispatchY = Math.ceil(this.height / 16);

        // =========================================================
        // PASS 1: Find the Maximum Label
        // =========================================================

        const maxLabelBuffer = this.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(maxLabelBuffer, 0, new Uint32Array([0]));

        const maxCode = FIND_MAX_LABEL_SHADER
            .replace(/{{WIDTH}}/g, this.width)
            .replace(/{{HEIGHT}}/g, this.height);

        const maxEncoder = this.device.createCommandEncoder();
        this._addComputePass(maxEncoder, {
            code: maxCode,
            entryPoint: 'main',
            bindings: [
                { binding: 0, resource: { buffer: this.labelBuffer } },
                { binding: 1, resource: { buffer: maxLabelBuffer } }
            ],
            dispatchX, dispatchY
        });
        this.device.queue.submit([maxEncoder.finish()]);

        // --- CPU/GPU SYNC POINT: Halt and wait for the max label ---
        const maxLabelArray = await this._readBuffer(maxLabelBuffer, 4, Uint32Array);
        const maxLabel = maxLabelArray[0];
        maxLabelBuffer.destroy();

        // If the max label is 0, the image is empty.
        if (maxLabel === 0) return null;

        // Labels are 0-indexed, so the required size is maxLabel + 1
        const maxExpectedLabels = maxLabel + 1;

        // =========================================================
        // PASS 2: Accumulate Statistics
        // =========================================================

        const statsStructCount = 6;
        const statsSize = maxExpectedLabels * statsStructCount * 4;

        if (this.statsBuffer) this.statsBuffer.destroy();
        this.statsBuffer = this.device.createBuffer({
            size: statsSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });

        const initData = new Uint32Array(maxExpectedLabels * statsStructCount);
        for (let i = 0; i < maxExpectedLabels; i++) {
            initData[i * statsStructCount + 4] = 0xFFFFFFFF; // Set min_intensity to max u32
        }
        this.device.queue.writeBuffer(this.statsBuffer, 0, initData);

        const statsCode = STATS_ACCUMULATOR_SHADER
            .replace(/{{WIDTH}}/g, this.width)
            .replace(/{{HEIGHT}}/g, this.height)
            .replace(/{{MAX_LABELS}}/g, maxExpectedLabels);

        const statsEncoder = this.device.createCommandEncoder();
        this._addComputePass(statsEncoder, {
            code: statsCode,
            entryPoint: 'main',
            bindings: [
                { binding: 0, resource: { buffer: this.labelBuffer } },
                { binding: 1, resource: this.originalTexture.createView() },
                { binding: 2, resource: { buffer: this.statsBuffer } }
            ],
            dispatchX, dispatchY
        });
        this.device.queue.submit([statsEncoder.finish()]);

        // =========================================================
        // PASS 3: Stream Compaction (Sparse -> Dense)
        // =========================================================

        const counterBuffer = this.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(counterBuffer, 0, new Uint32Array([0]));

        const denseStructCount = 7;
        const maxDenseSize = maxExpectedLabels * denseStructCount * 4;
        const compactStatsBuffer = this.device.createBuffer({
            size: maxDenseSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });

        const compactCode = COMPACT_STATS_SHADER
            .replace(/{{WIDTH}}/g, this.width)
            .replace(/{{HEIGHT}}/g, this.height)
            .replace(/{{MAX_LABELS}}/g, maxExpectedLabels);

        const compactEncoder = this.device.createCommandEncoder();
        this._addComputePass(compactEncoder, {
            code: compactCode,
            entryPoint: 'main',
            bindings: [
                { binding: 0, resource: { buffer: this.statsBuffer } },   // The sparse buffer
                { binding: 1, resource: { buffer: compactStatsBuffer } }, // The dense buffer
                { binding: 2, resource: { buffer: counterBuffer } }       // The atomic counter
            ],
            dispatchX, dispatchY
        });
        this.device.queue.submit([compactEncoder.finish()]);

        if (this.statsBuffer) this.statsBuffer.destroy();
        if (this.statsCounterBuffer) this.statsCounterBuffer.destroy();
        this.statsBuffer = compactStatsBuffer;
        this.statsCounterBuffer = counterBuffer;
    }

    /**
     * Reads back the compacted per-object stats from the last computeStats call
     * as dense 7-field structs (label, area, total_intensity, sum_x, sum_y, min,
     * max). Empty if no objects were found.
     * @returns {Promise<Uint32Array>}
     */
    async downloadStats() {
        const counterData = await this._readBuffer(this.statsCounterBuffer, 4, Uint32Array);
        const numObjects = counterData[0];

        // If no objects were found, bail early!
        if (numObjects === 0) return new Uint32Array(0);

        const denseStructCount = 7; // label, area, total_intensity, sum_x, sum_y, min, max
        const bytesToCopy = numObjects * denseStructCount * 4;

        return this._readBuffer(this.statsBuffer, bytesToCopy, Uint32Array);
    }

    /**
     * Reads back the connected-component labels (one u32 per pixel; 0 = background).
     * @returns {Promise<Uint32Array>}
     */
    async downloadLabels() {
        const outSize = this.width * this.height;
        return this._readBuffer(this.labelBuffer, outSize * 4, Uint32Array);
    }

    /**
     * Internal multi-pass WGSL pipeline for feature extraction.
     */
    async _extractFeatures(data, scale) {
        const NUM_CHANNELS = 8;
        const outSize = NUM_CHANNELS * this.width * this.height;
        const maxRadius = 32;

        const k0 = gaussian_kernel(scale, 0);
        const k1 = gaussian_kernel(scale, 1);
        const k2 = gaussian_kernel(scale, 2);
        const k0sub = gaussian_kernel(scale * 0.66, 0); // For Difference of Gaussians

        for (const k of [k0, k1, k2, k0sub]) {
            const radius = k.length - 1;
            if (radius > maxRadius) {
                throw new Error(
                    `_extractFeatures: sigma=${scale} requires a kernel radius of ${radius}, ` +
                    `which exceeds the supported maximum of ${maxRadius}. Reduce sigma, or raise ` +
                    `maxRadius here and the matching WGSL Kernels array sizes.`
                );
            }
        }

        // --- GPU Resource Allocation ---
        const kernelBuffer = this.device.createBuffer({
            size: (maxRadius * 4 * 4) + 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const kernelData = new Float32Array(maxRadius * 4 + 4);
        kernelData.set(k0, 0);
        kernelData.set(k1, maxRadius);
        kernelData.set(k2, maxRadius * 2);
        kernelData.set(k0sub, maxRadius * 3);
        const radiiView = new Int32Array(kernelData.buffer, maxRadius * 4 * 4);
        radiiView[0] = k0.length - 1;
        radiiView[1] = k1.length - 1;
        radiiView[2] = k2.length - 1;
        radiiView[3] = k0sub.length - 1;
        this.device.queue.writeBuffer(kernelBuffer, 0, kernelData);

        let gpuInput;
        if (data instanceof GPUBuffer) {
            gpuInput = data;
        } else {
            gpuInput = this.device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
            this.device.queue.writeBuffer(gpuInput, 0, data);
        }

        const gpuHoriz = this.device.createBuffer({ size: this.width * this.height * 4 * 4, usage: GPUBufferUsage.STORAGE });
        const gpuOutput = this.device.createBuffer({ size: outSize * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });

        // --- Pipeline Setup ---
        const commonWGSL = `
            struct Kernels {
                k0: array<vec4<f32>, 8>,
                k1: array<vec4<f32>, 8>,
                k2: array<vec4<f32>, 8>,
                k0sub: array<vec4<f32>, 8>,
                r0: i32, r1: i32, r2: i32, r0sub: i32,
            }
            @group(0) @binding(2) var<uniform> kernels: Kernels;
            fn get_k(k_idx: u32, i: i32) -> f32 {
                if (k_idx == 0u) { return kernels.k0[u32(i)/4u][u32(i)%4u]; }
                if (k_idx == 1u) { return kernels.k1[u32(i)/4u][u32(i)%4u]; }
                if (k_idx == 2u) { return kernels.k2[u32(i)/4u][u32(i)%4u]; }
                return kernels.k0sub[u32(i)/4u][u32(i)%4u];
            }
        `;

        const horizShader = `
            ${commonWGSL}
            @group(0) @binding(0) var<storage, read> input_data: array<f32>;
            @group(0) @binding(1) var<storage, read_write> horiz_data: array<f32>;

            fn get_val(ix: i32, iy: i32) -> f32 {
                return input_data[u32(clamp(iy, 0, ${this.height - 1})) * ${this.width}u + u32(clamp(ix, 0, ${this.width - 1}))];
            }

            @compute @workgroup_size(16, 16)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                let x = i32(id.x); let y = i32(id.y);
                if (x >= ${this.width} || y >= ${this.height}) { return; }

                var h0 = get_k(0u, 0) * get_val(x, y);
                for (var i = 1; i <= kernels.r0; i++) { h0 += get_k(0u, i) * (get_val(x - i, y) + get_val(x + i, y)); }

                var h1 = 0.0;
                for (var i = 1; i <= kernels.r1; i++) { h1 += get_k(1u, i) * (get_val(x + i, y) - get_val(x - i, y)); }

                var h2 = get_k(2u, 0) * get_val(x, y);
                for (var i = 1; i <= kernels.r2; i++) { h2 += get_k(2u, i) * (get_val(x - i, y) + get_val(x + i, y)); }

                var h0s = get_k(3u, 0) * get_val(x, y);
                for (var i = 1; i <= kernels.r0sub; i++) { h0s += get_k(3u, i) * (get_val(x - i, y) + get_val(x + i, y)); }

                let out_idx = (u32(y) * ${this.width}u + u32(x)) * 4u;
                horiz_data[out_idx] = h0; horiz_data[out_idx + 1] = h1; horiz_data[out_idx + 2] = h2; horiz_data[out_idx + 3] = h0s;
            }
        `;

        const vertShader = `
            ${commonWGSL}
            @group(0) @binding(0) var<storage, read> horiz_data: array<f32>;
            @group(0) @binding(1) var<storage, read_write> output_data: array<f32>;

            fn get_h(ix: i32, iy: i32) -> vec4<f32> {
                let idx = (u32(clamp(iy, 0, ${this.height - 1})) * ${this.width}u + u32(clamp(ix, 0, ${this.width - 1}))) * 4u;
                return vec4<f32>(horiz_data[idx], horiz_data[idx+1], horiz_data[idx+2], horiz_data[idx+3]);
            }

            @compute @workgroup_size(16, 16)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                let x = i32(id.x); let y = i32(id.y);
                if (x >= ${this.width} || y >= ${this.height}) { return; }

                var l_vec = get_h(x, y) * get_k(0u, 0);
                for (var i = 1; i <= kernels.r0; i++) { l_vec += (get_h(x, y-i) + get_h(x, y+i)) * get_k(0u, i); }
                let L = l_vec.x; let Lx = l_vec.y; let Lxx = l_vec.z;

                var ly_vec = vec2<f32>(0.0);
                for (var i = 1; i <= kernels.r1; i++) { ly_vec += (get_h(x, y+i) - get_h(x, y-i)).xy * get_k(1u, i); }
                let Ly = ly_vec.x; let Lxy = ly_vec.y;

                var lyy = get_h(x, y).x * get_k(2u, 0);
                for (var i = 1; i <= kernels.r2; i++) { lyy += (get_h(x, y-i).x + get_h(x, y+i).x) * get_k(2u, i); }

                var lsub_vec = get_h(x, y).w * get_k(3u, 0);
                for (var i = 1; i <= kernels.r0sub; i++) { lsub_vec += (get_h(x, y-i).w + get_h(x, y+i).w) * get_k(3u, i); }

                let out_idx = (u32(y) * ${this.width}u + u32(x)) * 8u;
                output_data[out_idx] = L;                               // GaussianSmoothing
                output_data[out_idx + 1] = Lxx + lyy;                   // LaplacianOfGaussian
                output_data[out_idx + 2] = sqrt(Lx*Lx + Ly*Ly);         // GaussianGradientMagnitude
                output_data[out_idx + 3] = L - lsub_vec;                // DifferenceOfGaussians

                let s_a = Lx*Lx; let s_b = Lx*Ly; let s_c = Ly*Ly;
                let s_term = sqrt((s_a - s_c)*(s_a - s_c) * 0.25 + s_b*s_b);
                output_data[out_idx + 4] = (s_a + s_c) * 0.5 + s_term;  // StructureTensorEigenvalues (smallest)
                output_data[out_idx + 5] = (s_a + s_c) * 0.5 - s_term;  // StructureTensorEigenvalues (largest)
                let h_term = sqrt((Lxx - lyy)*(Lxx - lyy) * 0.25 + Lxy*Lxy);
                output_data[out_idx + 6] = (Lxx + lyy) * 0.5 + h_term;  // HessianOfGaussianEigenvalues (smallest)
                output_data[out_idx + 7] = (Lxx + lyy) * 0.5 - h_term;  // HessianOfGaussianEigenvalues (largest)
            }
        `;

        const enc = this.device.createCommandEncoder();
        const dispatchX = Math.ceil(this.width / 16);
        const dispatchY = Math.ceil(this.height / 16);

        this._addComputePass(enc, {
            code: horizShader,
            entryPoint: 'main',
            bindings: [
                { binding: 0, resource: { buffer: gpuInput } },
                { binding: 1, resource: { buffer: gpuHoriz } },
                { binding: 2, resource: { buffer: kernelBuffer } }
            ],
            dispatchX, dispatchY
        });

        this._addComputePass(enc, {
            code: vertShader,
            entryPoint: 'main',
            bindings: [
                { binding: 0, resource: { buffer: gpuHoriz } },
                { binding: 1, resource: { buffer: gpuOutput } },
                { binding: 2, resource: { buffer: kernelBuffer } }
            ],
            dispatchX, dispatchY
        });

        this.device.queue.submit([enc.finish()]);

        if (!(data instanceof GPUBuffer)) gpuInput.destroy();
        gpuHoriz.destroy();
        kernelBuffer.destroy();

        return gpuOutput;
    }

    /**
     * Computes features and reads all 8 channels back to the CPU, interleaved as
     * `[f0..f7]` per pixel.
     * @param {Float32Array} intensityArray
     * @param {number} scale - Gaussian sigma.
     * @returns {Promise<Float32Array>} width*height*8 features.
     */
    async downloadFeatures(intensityArray, scale) {
        const gpuOutput = await this._extractFeatures(intensityArray, scale);
        const outSize = 8 * this.width * this.height;
        const res = await this._readBuffer(gpuOutput, outSize * 4, Float32Array);
        gpuOutput.destroy();
        return res;
    }

    /**
     * Gathers the 8-feature vectors for a set of labeled pixels (via a compute
     * scatter/gather pass) to feed FlatRandomForest.train. Returns them
     * row-major as `numLabels * 8` floats.
     * @param {Uint32Array} indicesArray - Flat pixel indices (y * width + x).
     * @returns {Promise<Float32Array>} Features for each labeled pixel.
     */
    async gatherFeaturesForTraining(indicesArray) {
        const numLabels = indicesArray.length;
        const indicesBuffer = this.device.createBuffer({
            size: indicesArray.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(indicesBuffer, 0, indicesArray);

        const gatherDstSize = numLabels * 8 * 4;
        const gatherDstBuffer = this.device.createBuffer({
            size: gatherDstSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });

        const enc = this.device.createCommandEncoder();
        this._addComputePass(enc, {
            code: GATHER_FEATURES_SHADER,
            entryPoint: 'main',
            bindings: [
                { binding: 0, resource: { buffer: this.featureBuffer } },
                { binding: 1, resource: { buffer: indicesBuffer } },
                { binding: 2, resource: { buffer: gatherDstBuffer } }
            ],
            dispatchX: Math.ceil(numLabels / 64)
        });
        this.device.queue.submit([enc.finish()]);

        const features = await this._readBuffer(gatherDstBuffer, gatherDstSize, Float32Array);

        indicesBuffer.destroy();
        gatherDstBuffer.destroy();

        return features;
    }

    /**
     * Runs the trained forest over every pixel, writing per-class probabilities
     * into a fresh probability buffer, then repaints the composite. Tree roots,
     * tree count, and max depth are passed as uniform metadata (see the fix
     * notes below for the sentinel/normalization pitfalls this handles).
     * @param {FlatRandomForest} rf - A trained forest (max 8 trees).
     * @throws If the forest has more than 8 trees.
     */
    async runInference(rf) {
        const forestBuffer = this.device.createBuffer({
            size: rf.forestBuffer.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(forestBuffer, 0, rf.forestBuffer);

        const numTrees = rf.treeRoots.length;
        if (numTrees > 8) {
            throw new Error(
                `runInference: ${numTrees} trees provided, but only up to 8 are supported ` +
                `by the forest_meta uniform layout. Reduce the forest size or widen forest_meta.`
            );
        }

        // FIX: previously the tree-roots buffer was zero-filled, and 0 is a valid node
        // index, not an "empty slot" sentinel -- so forests with fewer than 8 trees had
        // unused slots silently re-run tree 0's traversal and add phantom votes. Unused
        // slots must be -1. Also, num_trees was hardcoded to 8 in the shader regardless
        // of the real forest size, which both biased and mis-normalized every vote.
        // Both are now sourced from rf.treeRoots.length instead of a hardcoded constant.
        //
        // FIX: tree traversal depth was hardcoded to 10 in the shader; any tree deeper
        // than that silently stopped without reaching a leaf, contributing no vote while
        // the normalization denominator didn't adjust. max_depth is now real metadata
        // (falls back to 24 if rf doesn't provide it -- ideally rf.maxDepth should be
        // set from the true trained depth by whatever builds the forest buffer).
        const maxDepth = rf.maxDepth ?? 24;

        const forestMeta = new Int32Array(12).fill(-1); // 3x vec4<i32> = 48 bytes
        forestMeta.set(rf.treeRoots, 0);
        forestMeta[8] = numTrees;
        forestMeta[9] = maxDepth;

        const metaBuffer = this.device.createBuffer({
            size: forestMeta.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(metaBuffer, 0, forestMeta);

        const numColors = this.labelColors.length;

        if (this.probBuffer) this.probBuffer.destroy();
        this.probBuffer = this.device.createBuffer({
            size: this.width * this.height * numColors * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });

        const code = RF_INFERENCE_SHADER
            .replace(/{{WIDTH}}/g, this.width)
            .replace(/{{HEIGHT}}/g, this.height)
            .replace(/{{NUM_COLORS}}/g, numColors);

        const enc = this.device.createCommandEncoder();
        this._addComputePass(enc, {
            code,
            entryPoint: 'main',
            bindings: [
                { binding: 0, resource: { buffer: this.featureBuffer } },
                { binding: 1, resource: { buffer: forestBuffer } },
                { binding: 2, resource: { buffer: metaBuffer } },
                { binding: 3, resource: { buffer: this.probBuffer } }
            ],
            dispatchX: Math.ceil(this.width / 16),
            dispatchY: Math.ceil(this.height / 16)
        });
        this.device.queue.submit([enc.finish()]);

        forestBuffer.destroy();
        metaBuffer.destroy();

        this.renderComposite();
    }

    /** Paints the canvas: the original image with the argmax class overlaid where classified. */
    renderComposite() {
        if (!this.originalTexture || !this.probBuffer) return;

        const colors = this.labelColors;
        const colorsWGSL = colors.map(parseColorToWGSL).join(',\n            ');

        const code = COMPOSITE_SHADER
            .replace(/{{WIDTH}}/g, this.width)
            .replace(/{{HEIGHT}}/g, this.height)
            .replace(/{{NUM_COLORS}}/g, colors.length)
            .replace(/{{COLORS_ARRAY}}/g, colorsWGSL);

        const module = this.device.createShaderModule({ code });
        const pipeline = this.device.createRenderPipeline({
            layout: 'auto',
            vertex: { module, entryPoint: 'vs_main' },
            fragment: { module, entryPoint: 'fs_main', targets: [{ format: this.format }] },
            primitive: { topology: 'triangle-strip' }
        });

        const bindGroup = this.device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: this.device.createSampler() },
                { binding: 1, resource: this.originalTexture.createView() },
                { binding: 2, resource: { buffer: this.probBuffer } }
            ]
        });

        const enc = this.device.createCommandEncoder();
        const pass = enc.beginRenderPass({
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                clearValue: { r: 0, g: 0, b: 0, a: 1 },
                loadOp: 'clear', storeOp: 'store'
            }]
        });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.draw(4);
        pass.end();
        this.device.queue.submit([enc.finish()]);
    }

    /**
     * Reads the probability buffer back to the CPU as `numColors` channels per
     * pixel, packed sequentially.
     * @returns {Promise<Float32Array>} width*height*numColors probabilities.
     */
    async downloadProbabilities() {
        const numColors = this.labelColors.length;
        const outSize = this.width * this.height * numColors;
        return this._readBuffer(this.probBuffer, outSize * 4, Float32Array);
    }

    /** Frees all GPU textures and buffers held by this backend. */
    destroy() {
        if (this.originalTexture)    { this.originalTexture.destroy();    this.originalTexture = null; }
        if (this.featureBuffer)      { this.featureBuffer.destroy();      this.featureBuffer = null; }
        if (this.probBuffer)         { this.probBuffer.destroy();         this.probBuffer = null; }
        if (this.labelBuffer)        { this.labelBuffer.destroy();        this.labelBuffer = null; }
        if (this.statsBuffer)        { this.statsBuffer.destroy();        this.statsBuffer = null; }
        if (this.statsCounterBuffer) { this.statsCounterBuffer.destroy(); this.statsCounterBuffer = null; }
    }
}


/**
 * Official ilastik feature identifiers in the order they are concatenated
 * in the output buffer for a single scale.
 */
const FEATURE_IDS = [
  "GaussianSmoothing",
  "LaplacianOfGaussian",
  "GaussianGradientMagnitude",
  "DifferenceOfGaussians",
  "StructureTensorEigenvalues",   // 2 channels (Largest, Smallest)
  "HessianOfGaussianEigenvalues"  // 2 channels (Largest, Smallest)
];

/**
 * Generates a 1D Gaussian kernel or its analytical derivatives.
 * Replicates Vigra's dynamic radius and strict normalization requirements.
 * @param {number} scale - Sigma value (e.g. 0.3, 1.0, 3.5).
 * @param {number} order - 0 (Smoothing), 1 (1st Derivative), or 2 (2nd Derivative).
 * @returns {Float32Array} The right-half of the symmetric/anti-symmetric kernel [0...radius].
 */
function gaussian_kernel(scale, order = 0) {
  if (scale <= 0) throw new Error("scale should be greater than 0");

  // Vigra dynamic radius: ensures energy isn't truncated at high orders/scales
  const radius = Math.ceil((3.0 + 0.5 * order) * scale);
  const kernel = new Float32Array(radius + 1);
  const twoSigmaSq = 2.0 * scale * scale;
  const fullSize = 2 * radius + 1;
  const fullKernel = new Float32Array(fullSize);

  // 1. Calculate raw analytical values
  let sum = 0;
  for (let i = -radius; i <= radius; i++) {
    const x = i;
    let val = Math.exp(-(x * x) / twoSigmaSq);
    if (order === 1) val = (-x / (scale * scale)) * val;
    else if (order === 2) val = (((x * x) / (scale * scale * scale * scale)) - (1.0 / (scale * scale))) * val;
    fullKernel[i + radius] = val;
    if (order === 0) sum += val;
  }

  // 2. Strict Normalization (required for mathematical identicality with Vigra)
  if (order === 0) {
    // Smoothing sums to 1.0
    for (let i = 0; i < fullSize; i++) fullKernel[i] /= sum;
  }
  else if (order === 1) {
    // 1st Derivative: 1st moment must be exactly 1.0 (forces correlation behavior)
    let sumX = 0;
    for (let i = -radius; i <= radius; i++) sumX += i * fullKernel[i + radius];
    for (let i = 0; i < fullSize; i++) fullKernel[i] /= sumX;
  }
  else if (order === 2) {
    // 2nd Derivative: strict zero-mean and variance normalization
    let sum0 = 0;
    for (let i = -radius; i <= radius; i++) sum0 += fullKernel[i + radius];
    const mean = sum0 / fullSize;
    for (let i = 0; i < fullSize; i++) fullKernel[i] -= mean;
    let sumX2 = 0;
    for (let i = -radius; i <= radius; i++) sumX2 += 0.5 * i * i * fullKernel[i + radius];
    for (let i = 0; i < fullSize; i++) fullKernel[i] /= sumX2;
  }

  // Return the half-kernel to save GPU uniform space
  for (let i = 0; i <= radius; i++) kernel[i] = fullKernel[radius + i];
  return kernel;
}

/**
 * Converts any valid CSS color string (hex3/6/8, rgb/rgba, hsl/hsla, named
 * colors, etc.) into a WGSL vec4<f32> literal, by letting the browser's own
 * CSS color parser do the work via a 1x1 canvas instead of hand-rolled regex.
 */
let _colorParseCtx = null;
function parseColorToWGSL(colorStr) {
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
    return `vec4<f32>(${(r / 255).toFixed(3)}, ${(g / 255).toFixed(3)}, ${(b / 255).toFixed(3)}, ${(a / 255).toFixed(3)})`;
}

/**
* === GLOBAL COMPUTE AND RENDER WGSL SOURCE GLUE ===
*/

const GATHER_FEATURES_SHADER = `
@group(0) @binding(0) var<storage, read> src_features: array<f32>;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst_features: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let label_idx = id.x;
    let num_labels = arrayLength(&indices);
    if (label_idx >= num_labels) { return; }

    let pixel_idx = indices[label_idx];
    let src_offset = pixel_idx * 8u;
    let dst_offset = label_idx * 8u;

    for (var i = 0u; i < 8u; i++) {
        dst_features[dst_offset + i] = src_features[src_offset + i];
    }
}
`;

const RF_INFERENCE_SHADER = `
struct Node {
    feat_idx: i32,
    threshold: f32,
    left: i32,
    right: i32,
};

@group(0) @binding(0) var<storage, read> features: array<f32>;
@group(0) @binding(1) var<storage, read> forest: array<Node>;
// forest_meta layout: [0..7] = tree root node indices (-1 = empty slot),
// [8] = num_trees (actual tree count), [9] = max_depth (real traversal bound), [10..11] = reserved.
@group(0) @binding(2) var<uniform> forest_meta: array<vec4<i32>, 3>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

fn tree_root(t: u32) -> i32 {
    return forest_meta[t / 4u][t % 4u];
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x; let y = id.y;
    let w = u32({{WIDTH}});
    let h = u32({{HEIGHT}});
    if (x >= w || y >= h) { return; }

    let pixel_idx = y * w + x;
    let feat_offset = pixel_idx * 8u;

    let num_trees = u32(forest_meta[2].x);
    let max_depth = u32(forest_meta[2].y);

    var votes = array<f32, {{NUM_COLORS}}>();

    for (var t = 0u; t < num_trees; t++) {
        var node_idx = tree_root(t);
        if (node_idx < 0) { continue; }

        for (var depth = 0u; depth < max_depth; depth++) {
            let node = forest[u32(node_idx)];
            if (node.feat_idx == -1) {
                let class_id = -node.right - 1;
                if (class_id >= 0 && class_id < {{NUM_COLORS}}) {
                    votes[class_id] += 1.0;
                }
                break;
            }
            let val = features[feat_offset + u32(node.feat_idx)];
            if (val < node.threshold) {
                node_idx = node.left;
            } else {
                node_idx = node.right;
            }
        }
    }

    let denom = max(f32(num_trees), 1.0);
    for (var c = 0; c < {{NUM_COLORS}}; c++) {
        output[pixel_idx * {{NUM_COLORS}}u + u32(c)] = votes[c] / denom;
    }
}
`;

const COMPOSITE_SHADER = `
struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    var pos = array<vec2<f32>, 4>(vec2(-1,1), vec2(1,1), vec2(-1,-1), vec2(1,-1));
    var uv = array<vec2<f32>, 4>(vec2(0,0), vec2(1,0), vec2(0,1), vec2(1,1));
    var out: VertexOutput;
    out.pos = vec4(pos[idx], 0, 1);
    out.uv = uv[idx];
    return out;
}

@group(0) @binding(0) var s: sampler;
@group(0) @binding(1) var t_raw: texture_2d<f32>;
@group(0) @binding(2) var<storage, read> p_map: array<f32>;

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let raw = textureSample(t_raw, s, uv);
    let w = u32({{WIDTH}});
    let h = u32({{HEIGHT}});
    let x = u32(uv.x * f32(w));
    let y = u32(uv.y * f32(h));

    let base_idx = clamp(y * w + x, 0u, w * h - 1u) * {{NUM_COLORS}}u;
    var max_p: f32 = -1.0;
    var best_class: i32 = -1;

    for (var c = 0u; c < {{NUM_COLORS}}u; c++) {
        let p = p_map[base_idx + c];
        if (p > max_p) {
            max_p = p;
            best_class = i32(c);
        }
    }

    var alpha: f32 = 0.4;
    if (max_p < 0.0) { return vec4(raw.rgb, 1.0); }

    var colors = array<vec4<f32>, {{NUM_COLORS}}>(
        {{COLORS_ARRAY}}
    );

    var overlay = vec4<f32>(0.0, 0.0, 0.0, alpha);
    if (best_class >= 0 && best_class < {{NUM_COLORS}}) {
        overlay = colors[best_class];
        overlay.a = alpha;
    }

    return vec4(mix(raw.rgb, overlay.rgb, alpha), 1.0);
}
`;

/**
* Three-pass Atomic Parallel Union-Find Engine
*/
const CCL_SHADER = `
@group(0) @binding(0) var<storage, read> probabilities: array<f32>;
@group(0) @binding(1) var<storage, read_write> labels: array<atomic<u32>>;

// Pass 1: Threshold and allocate individual spatial component coordinates
@compute @workgroup_size(16, 16)
fn init_labels(@builtin(global_invocation_id) id: vec3<u32>) {
    let w = u32({{WIDTH}}); let h = u32({{HEIGHT}});
    if (id.x >= w || id.y >= h) { return; }

    let pixel_idx = id.y * w + id.x;
    let base_idx = pixel_idx * {{NUM_COLORS}}u;
    let p = probabilities[base_idx + {{TARGET_CLASS}}u];

    if (p >= {{THRESHOLD}}) {
        atomicStore(&labels[pixel_idx], pixel_idx + 1u); // 1-based index (0 is bg)
    } else {
        atomicStore(&labels[pixel_idx], 0u);
    }
}

// Pass 2: Connect neighborhood groups using atomic minimizations
@compute @workgroup_size(16, 16)
fn merge_neighbors(@builtin(global_invocation_id) id: vec3<u32>) {
    let w = u32({{WIDTH}}); let h = u32({{HEIGHT}});
    if (id.x >= w || id.y >= h) { return; }

    let pixel_idx = id.y * w + id.x;
    let self_label = atomicLoad(&labels[pixel_idx]);
    if (self_label == 0u) { return; }

    // Connect to right neighbor
    if (id.x + 1u < w) {
        let r_idx = id.y * w + (id.x + 1u);
        let r_label = atomicLoad(&labels[r_idx]);
        if (r_label != 0u) {
            merge_roots(pixel_idx, r_idx);
        }
    }

    // Connect to down neighbor
    if (id.y + 1u < h) {
        let d_idx = (id.y + 1u) * w + id.x;
        let d_label = atomicLoad(&labels[d_idx]);
        if (d_label != 0u) {
            merge_roots(pixel_idx, d_idx);
        }
    }
}

fn find_root(start_idx: u32) -> u32 {
    var curr = start_idx;
    var parent = atomicLoad(&labels[curr]);
    while (parent != 0u && parent != curr + 1u) {
        curr = parent - 1u;
        parent = atomicLoad(&labels[curr]);
    }
    return curr + 1u;
}

fn merge_roots(idx_a: u32, idx_b: u32) {
    var root_a = find_root(idx_a);
    var root_b = find_root(idx_b);

    while (root_a != root_b) {
        if (root_a < root_b) {
            let prev = atomicMin(&labels[root_b - 1u], root_a);
            if (prev == root_b) { break; }
            root_b = find_root(prev - 1u);
        } else {
            let prev = atomicMin(&labels[root_a - 1u], root_b);
            if (prev == root_a) { break; }
            root_a = find_root(prev - 1u);
        }
    }
}

// Pass 3: Flatten all tree structures to direct pointers
@compute @workgroup_size(16, 16)
fn flatten_paths(@builtin(global_invocation_id) id: vec3<u32>) {
    let w = u32({{WIDTH}}); let h = u32({{HEIGHT}});
    if (id.x >= w || id.y >= h) { return; }

    let pixel_idx = id.y * w + id.x;
    let self_label = atomicLoad(&labels[pixel_idx]);
    if (self_label == 0u) { return; }

    let root = find_root(pixel_idx);
    atomicStore(&labels[pixel_idx], root);
}
`;

const FIND_MAX_LABEL_SHADER = `
@group(0) @binding(0) var<storage, read> labels: array<u32>;
@group(0) @binding(1) var<storage, read_write> global_max: atomic<u32>;

var<workgroup> wg_max: atomic<u32>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32
) {
    let w = u32({{WIDTH}});
    let h = u32({{HEIGHT}});

    // 1. Initialize the workgroup's shared max variable
    if (local_idx == 0u) {
        atomicStore(&wg_max, 0u);
    }
    workgroupBarrier();

    // 2. Each thread finds the max in its assigned pixel
    if (global_id.x < w && global_id.y < h) {
        let pixel_idx = global_id.y * w + global_id.x;
        atomicMax(&wg_max, labels[pixel_idx]);
    }
    workgroupBarrier();

    // 3. One thread per workgroup pushes the local max to the global max
    if (local_idx == 0u) {
        atomicMax(&global_max, atomicLoad(&wg_max));
    }
}
`;

/**
* High-Speed Topology Statistics Accumulator
*/
const STATS_ACCUMULATOR_SHADER = `
struct Metrics {
    area: atomic<u32>,
    total_intensity: atomic<u32>,
    sum_x: atomic<u32>,
    sum_y: atomic<u32>,
    min_intensity: atomic<u32>,
    max_intensity: atomic<u32>
};

@group(0) @binding(0) var<storage, read> labels: array<u32>;
@group(0) @binding(1) var raw_intensity: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> stats: array<Metrics>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let w = u32({{WIDTH}}); let h = u32({{HEIGHT}});
    if (id.x >= w || id.y >= h) { return; }

    let pixel_idx = id.y * w + id.x;
    let label = labels[pixel_idx];

    if (label == 0u || label >= {{MAX_LABELS}}u) { return; }

    // Read directly from the texture using the X/Y coordinates
    let color = textureLoad(raw_intensity, vec2<i32>(id.xy), 0);
    let intensity_f = color.r; // Grab the red channel for intensity
    // Scale standard normalized float values to fixed-point integer spaces
    let intensity_u = u32(intensity_f * 10000.0);

    // Accumulate sums
    atomicAdd(&stats[label].area, 1u);
    atomicAdd(&stats[label].total_intensity, intensity_u);
    atomicAdd(&stats[label].sum_x, id.x);
    atomicAdd(&stats[label].sum_y, id.y);

    // Evaluate Min/Max
    atomicMin(&stats[label].min_intensity, intensity_u);
    atomicMax(&stats[label].max_intensity, intensity_u);
}
`;

const COMPACT_STATS_SHADER = `
struct SparseMetrics {
    area: atomic<u32>,
    total_intensity: atomic<u32>,
    sum_x: atomic<u32>,
    sum_y: atomic<u32>,
    min_intensity: atomic<u32>,
    max_intensity: atomic<u32>
};

struct DenseMetrics {
    label: u32,
    area: u32,
    total_intensity: u32,
    sum_x: u32,
    sum_y: u32,
    min_intensity: u32,
    max_intensity: u32
};

@group(0) @binding(0) var<storage, read_write> sparse_stats: array<SparseMetrics>;
@group(0) @binding(1) var<storage, read_write> compact_stats: array<DenseMetrics>;
@group(0) @binding(2) var<storage, read_write> counter: atomic<u32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let w = {{WIDTH}}u;
    let h = {{HEIGHT}}u;

    // Check texture bounds
    if (id.x >= w || id.y >= h) { return; }

    // Map 2D coordinates to the 1D label index
    let label_idx = id.y * w + id.x;

    // Check against the maximum expected labels found in Pass 1
    if (label_idx >= {{MAX_LABELS}}u) { return; }
    if (label_idx == 0u) { return; }

    let area = atomicLoad(&sparse_stats[label_idx].area);
    if (area > 0u) {
        let write_idx = atomicAdd(&counter, 1u);

        compact_stats[write_idx].label = label_idx;
        compact_stats[write_idx].area = area;
        compact_stats[write_idx].total_intensity = atomicLoad(&sparse_stats[label_idx].total_intensity);
        compact_stats[write_idx].sum_x = atomicLoad(&sparse_stats[label_idx].sum_x);
        compact_stats[write_idx].sum_y = atomicLoad(&sparse_stats[label_idx].sum_y);
        compact_stats[write_idx].min_intensity = atomicLoad(&sparse_stats[label_idx].min_intensity);
        compact_stats[write_idx].max_intensity = atomicLoad(&sparse_stats[label_idx].max_intensity);
    }
}
`;