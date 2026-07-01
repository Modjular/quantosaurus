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
        
        // New buffers for Topology and Analysis
        this.labelBuffer = null;
        this.statsBuffer = null;

        this.labelColors = labelColors || [
            'rgba(255,0,0,1.0)',
            'rgba(0,255,0,1.0)',
            'rgba(0,0,255,1.0)',
        ];
    }

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
    }

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
        const numPixels = this.width * this.height;

        const baseCode = CCL_SHADER
            .replace(/{{WIDTH}}/g, this.width)
            .replace(/{{HEIGHT}}/g, this.height)
            .replace(/{{NUM_COLORS}}/g, this.labelColors.length)
            .replace(/{{TARGET_CLASS}}/g, targetClassIdx)
            .replace(/{{THRESHOLD}}/g, threshold.toFixed(4));

        const module = this.device.createShaderModule({ code: baseCode });
        
        const pipeInit = this.device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'init_labels' } });
        const pipeMerge = this.device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'merge_neighbors' } });
        const pipeFlatten = this.device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'flatten_paths' } });

        const bindGroupInit = this.device.createBindGroup({
            layout: pipeInit.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.probBuffer } },
                { binding: 1, resource: { buffer: this.labelBuffer } }
            ]
        });

        const bindGroupMerge = this.device.createBindGroup({
            layout: pipeMerge.getBindGroupLayout(0),
            entries: [
                { binding: 1, resource: { buffer: this.labelBuffer } }
            ]
        });

        const bindGroupFlatten = this.device.createBindGroup({
            layout: pipeFlatten.getBindGroupLayout(0),
            entries: [
                { binding: 1, resource: { buffer: this.labelBuffer } }
            ]
        });

        const enc = this.device.createCommandEncoder();
        const dispatchX = Math.ceil(this.width / 16);
        const dispatchY = Math.ceil(this.height / 16);

        // Pass 1: Initialize values based on raw inference thresholds
        const p1 = enc.beginComputePass();
        p1.setPipeline(pipeInit);
        p1.setBindGroup(0, bindGroupInit);
        p1.dispatchWorkgroups(dispatchX, dispatchY);
        p1.end();

        // Pass 2: Neighborhood boundary checking and equivalence linking
        const p2 = enc.beginComputePass();
        p2.setPipeline(pipeMerge);
        p2.setBindGroup(0, bindGroupMerge);
        p2.dispatchWorkgroups(dispatchX, dispatchY);
        p2.end();

        // Pass 3: Path relaxation (Tree flattening) to find absolute roots
        const p3 = enc.beginComputePass();
        p3.setPipeline(pipeFlatten);
        p3.setBindGroup(0, bindGroupFlatten);
        p3.dispatchWorkgroups(dispatchX, dispatchY);
        p3.end();

        this.device.queue.submit([enc.finish()]);
        return this.labelBuffer;
    }

    /**
     * Compiles area and cumulative intensity metric profiles per label ID.
     */
    async computeStats() {
        const totalPixels = this.width * this.height;

        // =========================================================
        // PASS 1: Find the Maximum Label
        // =========================================================
        
        // Create a 4-byte buffer to hold the single maximum label value
        const maxLabelBuffer = this.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });

        this.device.queue.writeBuffer(maxLabelBuffer, 0, new Uint32Array([0]));

        const maxCode = FIND_MAX_LABEL_SHADER
                    .replace(/{{WIDTH}}/g, this.width)
                    .replace(/{{HEIGHT}}/g, this.height);

        const maxModule = this.device.createShaderModule({ code: maxCode });
        const maxPipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: { module: maxModule, entryPoint: 'main' }
        });

        const maxBindGroup = this.device.createBindGroup({
            layout: maxPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.labelBuffer } },
                { binding: 1, resource: { buffer: maxLabelBuffer } }
            ]
        });

        // We also need a staging buffer to read the result back to the CPU
        const maxStagingBuffer = this.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });

        const maxEncoder = this.device.createCommandEncoder();
        const maxPass = maxEncoder.beginComputePass();
        maxPass.setPipeline(maxPipeline);
        maxPass.setBindGroup(0, maxBindGroup);
        // Use a 2D dispatch matching the image dimensions
        maxPass.dispatchWorkgroups(
            Math.ceil(this.width / 16), 
            Math.ceil(this.height / 16)
        );
        maxPass.end();
        
        // Copy the result to the staging buffer
        maxEncoder.copyBufferToBuffer(maxLabelBuffer, 0, maxStagingBuffer, 0, 4);
        this.device.queue.submit([maxEncoder.finish()]);

        // --- CPU/GPU SYNC POINT: Halt and wait for the max label ---
        await maxStagingBuffer.mapAsync(GPUMapMode.READ);
        const maxLabelArray = new Uint32Array(maxStagingBuffer.getMappedRange());
        const maxLabel = maxLabelArray[0];
        maxStagingBuffer.unmap();
        
        // Clean up the temporary max buffers
        maxLabelBuffer.destroy();
        maxStagingBuffer.destroy();

        // If the max label is 0, the image is empty. Bail out early!
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

        // Initialize CPU-side array to handle the min_intensity starting value
        const initData = new Uint32Array(maxExpectedLabels * statsStructCount);
        for (let i = 0; i < maxExpectedLabels; i++) {
            initData[i * statsStructCount + 4] = 0xFFFFFFFF; // Set min_intensity to max u32
        }
        
        this.device.queue.writeBuffer(this.statsBuffer, 0, initData);

        const statsCode = STATS_ACCUMULATOR_SHADER
            .replace(/{{WIDTH}}/g, this.width)
            .replace(/{{HEIGHT}}/g, this.height)
            .replace(/{{MAX_LABELS}}/g, maxExpectedLabels);

        const statsModule = this.device.createShaderModule({ code: statsCode });
        const statsPipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: { module: statsModule, entryPoint: 'main' }
        });

        const statsBindGroup = this.device.createBindGroup({
            layout: statsPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.labelBuffer } },
                { binding: 1, resource: this.originalTexture.createView() },
                { binding: 2, resource: { buffer: this.statsBuffer } }
            ]
        });

        const statsEncoder = this.device.createCommandEncoder();
        const statsPass = statsEncoder.beginComputePass();
        statsPass.setPipeline(statsPipeline);
        statsPass.setBindGroup(0, statsBindGroup);
        statsPass.dispatchWorkgroups(Math.ceil(this.width / 16), Math.ceil(this.height / 16));
        statsPass.end();
        this.device.queue.submit([statsEncoder.finish()]);

        return this.statsBuffer;
    }

    async downloadStats() {
        const byteSize = this.statsBuffer.size;
        const stagingBuffer = this.device.createBuffer({
            size: byteSize,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });

        const copyEncoder = this.device.createCommandEncoder();
        copyEncoder.copyBufferToBuffer(
            this.statsBuffer, 0, // Source buffer & offset
            stagingBuffer, 0,    // Destination buffer & offset
            byteSize             // Number of bytes to copy
        );
        this.device.queue.submit([copyEncoder.finish()]);

        await stagingBuffer.mapAsync(GPUMapMode.READ);
        const data = new Uint32Array(stagingBuffer.getMappedRange()).slice();

        stagingBuffer.unmap();
        stagingBuffer.destroy();

        return data
    }

    /**
     * Downloads computed features back to system RAM.
     */
    async downloadLabels() {
        const outSize = this.width * this.height;
        const rb = this.device.createBuffer({
            size: outSize * 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });

        const copyEncoder = this.device.createCommandEncoder();
        copyEncoder.copyBufferToBuffer(this.labelBuffer, 0, rb, 0, outSize * 4);
        this.device.queue.submit([copyEncoder.finish()]);

        await rb.mapAsync(GPUMapMode.READ);
        const data = new Uint32Array(rb.getMappedRange().slice());
        rb.unmap();
        rb.destroy();
        
        return data;
    }

    /**
     * Internal multi-pass WGSL pipeline for feature extraction.
     */
    async _extractFeatures(data, scale) {
        const NUM_CHANNELS = 8;
        const outSize = NUM_CHANNELS * this.width * this.height;

        const k0 = gaussian_kernel(scale, 0);
        const k1 = gaussian_kernel(scale, 1);
        const k2 = gaussian_kernel(scale, 2);
        const k0sub = gaussian_kernel(scale * 0.66, 0); // For Difference of Gaussians

        // --- GPU Resource Allocation ---
        const maxRadius = 32;
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

        const modH = this.device.createShaderModule({ code: horizShader });
        const modV = this.device.createShaderModule({ code: vertShader });
        const pipeH = this.device.createComputePipeline({ layout: 'auto', compute: { module: modH, entryPoint: 'main' } });
        const pipeV = this.device.createComputePipeline({ layout: 'auto', compute: { module: modV, entryPoint: 'main' } });

        const bgH = this.device.createBindGroup({
            layout: pipeH.getBindGroupLayout(0),
            entries: [{ binding: 0, resource: { buffer: gpuInput } }, { binding: 1, resource: { buffer: gpuHoriz } }, { binding: 2, resource: { buffer: kernelBuffer } }]
        });
        const bgV = this.device.createBindGroup({
            layout: pipeV.getBindGroupLayout(0),
            entries: [{ binding: 0, resource: { buffer: gpuHoriz } }, { binding: 1, resource: { buffer: gpuOutput } }, { binding: 2, resource: { buffer: kernelBuffer } }]
        });

        const enc = this.device.createCommandEncoder();
        const p1 = enc.beginComputePass(); p1.setPipeline(pipeH); p1.setBindGroup(0, bgH); p1.dispatchWorkgroups(Math.ceil(this.width/16), Math.ceil(this.height/16)); p1.end();
        const p2 = enc.beginComputePass(); p2.setPipeline(pipeV); p2.setBindGroup(0, bgV); p2.dispatchWorkgroups(Math.ceil(this.width/16), Math.ceil(this.height/16)); p2.end();
        this.device.queue.submit([enc.finish()]);

        if (!(data instanceof GPUBuffer)) gpuInput.destroy();
        gpuHoriz.destroy();
        kernelBuffer.destroy();

        return gpuOutput;
    }

    async downloadFeatures(intensityArray, scale) {
        const gpuOutput = await this._extractFeatures(intensityArray, scale);
        const outSize = 8 * this.width * this.height;
        const rb = this.device.createBuffer({ size: outSize * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

        const enc = this.device.createCommandEncoder();
        enc.copyBufferToBuffer(gpuOutput, 0, rb, 0, outSize * 4);
        this.device.queue.submit([enc.finish()]);

        await rb.mapAsync(GPUMapMode.READ);
        const res = new Float32Array(rb.getMappedRange().slice());
        rb.unmap();
        rb.destroy();
        gpuOutput.destroy();
        
        return res;
    }

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

        const stagingBuffer = this.device.createBuffer({
            size: gatherDstSize,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });

        const module = this.device.createShaderModule({ code: GATHER_FEATURES_SHADER });
        const pipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: { module, entryPoint: 'main' }
        });

        const bindGroup = this.device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.featureBuffer } },
                { binding: 1, resource: { buffer: indicesBuffer } },
                { binding: 2, resource: { buffer: gatherDstBuffer } }
            ]
        });

        const enc = this.device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(numLabels / 64));
        pass.end();

        enc.copyBufferToBuffer(gatherDstBuffer, 0, stagingBuffer, 0, gatherDstSize);
        this.device.queue.submit([enc.finish()]);

        await stagingBuffer.mapAsync(GPUMapMode.READ);
        const features = new Float32Array(stagingBuffer.getMappedRange().slice());
        stagingBuffer.unmap();

        indicesBuffer.destroy();
        gatherDstBuffer.destroy();
        stagingBuffer.destroy();

        return features;
    }

    async runInference(rf) {
        const forestBuffer = this.device.createBuffer({
            size: rf.forestBuffer.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(forestBuffer, 0, rf.forestBuffer);

        const rootsBuffer = this.device.createBuffer({
            size: 32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        const paddedRoots = new Int32Array(8);
        paddedRoots.set(rf.treeRoots);
        this.device.queue.writeBuffer(rootsBuffer, 0, paddedRoots);

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

        const module = this.device.createShaderModule({ code });
        const pipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: { module, entryPoint: "main" }
        });

        const bindGroup = this.device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.featureBuffer } },
                { binding: 1, resource: { buffer: forestBuffer } },
                { binding: 2, resource: { buffer: rootsBuffer } },
                { binding: 3, resource: { buffer: this.probBuffer } }
            ]
        });

        const enc = this.device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(this.width / 16), Math.ceil(this.height / 16));
        pass.end();
        this.device.queue.submit([enc.finish()]);

        this.renderComposite();
    }

    renderComposite() {
        if (!this.originalTexture || !this.probBuffer) return;

        const colors = this.labelColors;

        function parseColor(c) {
            // rgba
            let m = c.match(/rgba?\((\d+),\s*(\d+),\s*(\d+),?\s*([\d.]+)?\)/);
            if (m) {
                const r = (m[1]/255).toFixed(3);
                const g = (m[2]/255).toFixed(3);
                const b = (m[3]/255).toFixed(3);
                return `vec4<f32>(${r}, ${g}, ${b}, ${m[4] || '0.8'})`;
            }

            // hex
            m = c.match(/^#([0-9a-f]{6})$/i)[1];
            if(m) {
                const r = (parseInt(m.substr(0,2),16)/255).toFixed(3);
                const g = (parseInt(m.substr(2,2),16)/255).toFixed(3);
                const b = (parseInt(m.substr(4,2),16)/255).toFixed(3);
                return `vec4<f32>(${r}, ${g}, ${b}, 0.8)`;
            }
            return "vec4<f32>(1.0, 0.0, 0.0, 1.0)";
        }
        
        const colorsWGSL = colors.map(parseColor).join(',\n            ');

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

    async downloadProbabilities() {
        const numColors = this.labelColors.length;
        const outSize = this.width * this.height * numColors;
        const rb = this.device.createBuffer({
            size: outSize * 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });

        const enc = this.device.createCommandEncoder();
        enc.copyBufferToBuffer(this.probBuffer, 0, rb, 0, outSize * 4);
        this.device.queue.submit([enc.finish()]);

        await rb.mapAsync(GPUMapMode.READ);
        const res = new Float32Array(rb.getMappedRange().slice());
        rb.unmap();
        rb.destroy();
        
        return res;
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
@group(0) @binding(2) var<uniform> tree_roots: array<vec4<i32>, 2>; 
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x; let y = id.y;
    let w = u32({{WIDTH}});
    let h = u32({{HEIGHT}});
    if (x >= w || y >= h) { return; }
    
    let pixel_idx = y * w + x;
    let feat_offset = pixel_idx * 8u;
    
    var votes = array<f32, {{NUM_COLORS}}>();
    var num_trees = 8u; 

    for (var t = 0u; t < num_trees; t++) {
        var node_idx = tree_roots[t/4u][t%4u];
        if (node_idx < 0) { continue; }

        for (var depth = 0; depth < 10; depth++) {
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

    for (var c = 0; c < {{NUM_COLORS}}; c++) {
        output[pixel_idx * {{NUM_COLORS}}u + u32(c)] = votes[c] / f32(num_trees);
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