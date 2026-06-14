export const GATHER_FEATURES_SHADER = `
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

export const RF_INFERENCE_SHADER = `
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

        var max_votes = -1.0;
        var best_class = -1.0;
        for (var c = 0; c < {{NUM_COLORS}}; c++) {
            if (votes[c] > max_votes) {
                max_votes = votes[c];
                best_class = f32(c);
            }
        }

        output[pixel_idx] = best_class;
    }
`;

export const COMPOSITE_SHADER = `
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
        let p = p_map[clamp(y * w + x, 0u, w * h - 1u)];
        
        var alpha: f32 = 0.4;
        if (p < 0.0) { return vec4(raw.rgb, 1.0); }
        
        var colors = array<vec4<f32>, {{NUM_COLORS}}>(
            {{COLORS_ARRAY}}
        );

        let class_idx = i32(p + 0.5);
        var overlay = vec4<f32>(0.0, 0.0, 0.0, alpha);
        if (class_idx >= 0 && class_idx < {{NUM_COLORS}}) {
            overlay = colors[class_idx];
            overlay.a = alpha;
        }

        return vec4(mix(raw.rgb, overlay.rgb, alpha), 1.0);
    }
`;