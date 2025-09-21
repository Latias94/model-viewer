// Fullscreen triangle VS
struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
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
    out.uv = vec2<f32>(0.5 * p.x + 0.5, 0.5 * p.y + 0.5);
    return out;
}

fn radical_inverse_vdc(bits: u32) -> f32 {
    var b = bits;
    b = (b << 16u) | (b >> 16u);
    b = ((b & 0x55555555u) << 1u) | ((b & 0xAAAAAAAAu) >> 1u);
    b = ((b & 0x33333333u) << 2u) | ((b & 0xCCCCCCCCu) >> 2u);
    b = ((b & 0x0F0F0F0Fu) << 4u) | ((b & 0xF0F0F0F0u) >> 4u);
    b = ((b & 0x00FF00FFu) << 8u) | ((b & 0xFF00FF00u) >> 8u);
    return f32(b) * 2.3283064365386963e-10; // / 0x100000000
}

fn hammersley(i: u32, n: u32) -> vec2<f32> {
    return vec2<f32>(f32(i) / f32(n), radical_inverse_vdc(i));
}

fn geometry_smith(nDotV: f32, nDotL: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    let g1 = nDotV / (nDotV * (1.0 - k) + k + 1e-5);
    let g2 = nDotL / (nDotL * (1.0 - k) + k + 1e-5);
    return g1 * g2;
}

@fragment
fn fs_brdf(input: VsOut) -> @location(0) vec4<f32> {
    let nDotV = clamp(input.uv.x, 0.0, 1.0);
    let roughness = clamp(input.uv.y, 0.0, 1.0);
    let V = vec3<f32>(sqrt(1.0 - nDotV * nDotV), 0.0, nDotV);
    var A = 0.0;
    var B = 0.0;
    let N = vec3<f32>(0.0, 0.0, 1.0);
    let SAMPLE_COUNT: u32 = 128u;
    for (var i: u32 = 0u; i < SAMPLE_COUNT; i = i + 1u) {
        let xi = hammersley(i, SAMPLE_COUNT);
        // Importance sample GGX
        let a = roughness * roughness;
        let phi = 6.2831853 * xi.x;
        let cosTheta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
        let sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
        let H = vec3<f32>(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
        let L = normalize(2.0 * dot(V, H) * H - V);
        let nDotL = clamp(L.z, 0.0, 1.0);
        let nDotH = clamp(H.z, 0.0, 1.0);
        let vDotH = clamp(dot(V, H), 0.0, 1.0);
        if (nDotL > 0.0) {
            let G = geometry_smith(nDotV, nDotL, roughness);
            let Gv = G * vDotH / max(nDotH * nDotV, 1e-5);
            let Fc = pow(1.0 - vDotH, 5.0);
            A += (1.0 - Fc) * Gv;
            B += Fc * Gv;
        }
    }
    A /= f32(SAMPLE_COUNT);
    B /= f32(SAMPLE_COUNT);
    return vec4<f32>(A, B, 0.0, 1.0);
}

