// Fullscreen triangle VS
struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) ndc_xy: vec2<f32>,
}

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> VsOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    var out: VsOut;
    let p = positions[vid];
    out.pos = vec4<f32>(p, 0.0, 1.0);
    out.ndc_xy = p;
    return out;
}

// Common params
struct Params {
    mode: u32,         // 0 = irradiance, 1 = prefilter
    face_index: u32,   // 0..5
    roughness: f32,    // for prefilter
    sample_count: u32, // samples per pixel
}

@group(0) @binding(0) var tex_equirect: texture_2d<f32>;
@group(0) @binding(1) var samp_equirect: sampler;
@group(1) @binding(0) var<uniform> params: Params;

fn dir_from_face_uv(face: u32, uv: vec2<f32>) -> vec3<f32> {
    // uv in [-1,1]
    let u = uv.x;
    let v = -uv.y; // flip to match cube convention
    if (face == 0u) { // +X
        return normalize(vec3<f32>( 1.0,    v,   -u));
    } else if (face == 1u) { // -X
        return normalize(vec3<f32>(-1.0,    v,    u));
    } else if (face == 2u) { // +Y
        return normalize(vec3<f32>(   u,  1.0,    v));
    } else if (face == 3u) { // -Y
        return normalize(vec3<f32>(   u, -1.0,   -v));
    } else if (face == 4u) { // +Z
        return normalize(vec3<f32>(   u,    v,  1.0));
    } else { // -Z
        return normalize(vec3<f32>(  -u,    v, -1.0));
    }
}

fn equirect_uv(dir: vec3<f32>) -> vec2<f32> {
    let d = normalize(dir);
    let phi = atan2(d.z, d.x);
    let theta = acos(clamp(d.y, -1.0, 1.0));
    let u = phi / (2.0 * 3.14159265) + 0.5;
    let v = theta / 3.14159265;
    return vec2<f32>(u, v);
}

fn basis_from_normal(n: vec3<f32>) -> mat3x3<f32> {
    let up = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 1.0, 0.0), abs(n.y) > 0.9);
    let t = normalize(cross(up, n));
    let b = cross(n, t);
    return mat3x3<f32>(t, b, n);
}

fn hammersley(i: u32, n: u32) -> vec2<f32> {
    var bits = i;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    let r2 = f32(bits) * 2.3283064365386963e-10; // / 0x100000000
    return vec2<f32>(f32(i) / f32(n), r2);
}

fn cosine_hemisphere(u: vec2<f32>) -> vec3<f32> {
    let r = sqrt(u.x);
    let phi = 2.0 * 3.14159265 * u.y;
    let x = r * cos(phi);
    let y = r * sin(phi);
    let z = sqrt(max(0.0, 1.0 - x * x - y * y));
    return vec3<f32>(x, y, z);
}

@fragment
fn fs_convolve(input: VsOut) -> @location(0) vec4<f32> {
    // Map ndc to [-1,1]
    let uv = input.ndc_xy;
    let dir = dir_from_face_uv(params.face_index, uv);

    if (params.mode == 0u) {
        // Irradiance: cosine-weighted hemisphere blur
        let basis = basis_from_normal(dir);
        let N = max(params.sample_count, 1u);
        var sum = vec3<f32>(0.0, 0.0, 0.0);
        var wsum = 0.0;
        for (var i: u32 = 0u; i < N; i = i + 1u) {
            let u2 = hammersley(i, N);
            let l = cosine_hemisphere(u2);
            let lw = (basis * l);
            let w = max(lw.z, 0.0);
            let st = equirect_uv(lw);
            let c = textureSampleLevel(tex_equirect, samp_equirect, st, 0.0).rgb;
            sum += c * w;
            wsum += w;
        }
        let col = sum / max(wsum, 1e-4);
        return vec4<f32>(col, 1.0);
    } else {
        // Prefilter: approximate blur with cone-angle from roughness
        let basis = basis_from_normal(dir);
        let N = max(params.sample_count, 1u);
        let R = clamp(params.roughness, 0.0, 1.0);
        var sum = vec3<f32>(0.0, 0.0, 0.0);
        var wsum = 0.0;
        for (var i: u32 = 0u; i < N; i = i + 1u) {
            let u2 = hammersley(i, N);
            // Distribute theta within [0, R * pi/2]
            let maxTheta = 1.57079632 * R;
            let theta = u2.x * maxTheta;
            let phi = 6.2831853 * u2.y;
            let xs = sin(theta) * cos(phi);
            let ys = sin(theta) * sin(phi);
            let zs = cos(theta);
            let l = vec3<f32>(xs, ys, zs);
            let lw = normalize(basis * l);
            let w = zs; // weight by cos(theta)
            let st = equirect_uv(lw);
            let c = textureSampleLevel(tex_equirect, samp_equirect, st, 0.0).rgb;
            sum += c * w;
            wsum += w;
        }
        let col = sum / max(wsum, 1e-4);
        return vec4<f32>(col, 1.0);
    }
}

