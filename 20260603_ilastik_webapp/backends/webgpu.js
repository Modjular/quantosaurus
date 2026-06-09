import { gaussian_kernel } from './utils.js';
import * as shaders from './webgpu-shaders.js';

export class WebGpuBackend {
    constructor() {
        this.device = null;
        this.context = null;
        this.format = null;
        this.width = 0;
        this.height = 0;
        this.originalTexture = null;
        this.featureBuffer = null;
        this.probBuffer = null;
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

        if (this.probBuffer) this.probBuffer.destroy();
        this.probBuffer = this.device.createBuffer({
            size: width * height * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
        });

        const initialProbs = new Float32Array(width * height).fill(-1.0);
        this.device.queue.writeBuffer(this.probBuffer, 0, initialProbs);
    }

    async updateFeatures(intensityArray, sigma) {
        if (this.featureBuffer) this.featureBuffer.destroy();
        this.featureBuffer = await this._extractFeatures(intensityArray, sigma);
        this.renderComposite();
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

    /**
     * Optional utility to pull the feature data back to system RAM.
     * Useful for CPU-side unit testing.
     */
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

        const module = this.device.createShaderModule({ code: shaders.GATHER_FEATURES_SHADER });
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

        if (this.probBuffer) this.probBuffer.destroy();
        this.probBuffer = this.device.createBuffer({
            size: this.width * this.height * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });

        const code = shaders.RF_INFERENCE_SHADER
            .replace(/{{WIDTH}}/g, this.width)
            .replace(/{{HEIGHT}}/g, this.height);

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

        const code = shaders.COMPOSITE_SHADER
            .replace(/{{WIDTH}}/g, this.width)
            .replace(/{{HEIGHT}}/g, this.height);

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
        const outSize = this.width * this.height;
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