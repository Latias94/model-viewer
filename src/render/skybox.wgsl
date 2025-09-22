struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) ndc_xy: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    var out: VsOut;
    let p = positions[vid];
    out.pos = vec4<f32>(p, 0.0, 1.0);
    out.ndc_xy = p; // clip-space xy for reconstruction
    return out;
}

struct Uniforms {
    view_proj: mat4x4<f32>,
    model: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
    view_proj_inv: mat4x4<f32>,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

// Environment: use prefiltered env as skybox
@group(1) @binding(2) var tex_prefilter: texture_cube<f32>;
@group(1) @binding(3) var samp_prefilter: sampler;

fn tonemap_aces(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_main(input: VsOut) -> @location(0) vec4<f32> {
    // Reconstruct world ray direction using inverse(view_proj)
    let ndc = vec2<f32>(input.ndc_xy.x, input.ndc_xy.y);
    let p0 = uniforms.view_proj_inv * vec4<f32>(ndc, 0.0, 1.0);
    let p1 = uniforms.view_proj_inv * vec4<f32>(ndc, 1.0, 1.0);
    let w0 = p0.xyz / max(p0.w, 1e-5);
    let w1 = p1.xyz / max(p1.w, 1e-5);
    let dir = normalize(w1 - w0);

    let color = textureSample(tex_prefilter, samp_prefilter, dir).rgb;
    let mapped = tonemap_aces(color);
    return vec4<f32>(mapped, 1.0);
}
