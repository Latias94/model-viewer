// Vertex shader

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tex_coords: vec2<f32>,
    @location(3) tangent: vec4<f32>,
    @location(4) tex_coords1: vec2<f32>,
    @location(5) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) tex_coords: vec2<f32>,
    @location(3) world_tangent: vec3<f32>,
    @location(4) tangent_sign: f32,
    @location(5) tex_coords1: vec2<f32>,
    @location(6) color: vec4<f32>,
}

struct Uniforms {
    view_proj: mat4x4<f32>,
    model: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let world_position = uniforms.model * vec4<f32>(input.position, 1.0);
    out.world_position = world_position.xyz;
    out.clip_position = uniforms.view_proj * world_position;

    // Transform normal to world space
    out.world_normal = (uniforms.normal_matrix * vec4<f32>(input.normal, 0.0)).xyz;

    out.tex_coords = input.tex_coords;
    out.tex_coords1 = input.tex_coords1;
    out.color = input.color;

    // Transform tangent to world space
    out.world_tangent = (uniforms.normal_matrix * vec4<f32>(input.tangent.xyz, 0.0)).xyz;
    out.tangent_sign = input.tangent.w;

    return out;
}

// Fragment shader

struct FragmentInput {
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) tex_coords: vec2<f32>,
    @location(3) world_tangent: vec3<f32>,
    @location(4) tangent_sign: f32,
    @location(5) tex_coords1: vec2<f32>,
    @location(6) color: vec4<f32>,
}

struct LightingData {
    light_position: vec3<f32>,
    light_color: vec3<f32>,
    view_position: vec3<f32>,
    ambient_strength: f32,
    lighting_intensity: f32,
}

@group(1) @binding(0)
var<uniform> lighting: LightingData;

struct MaterialParams {
    base_color_factor: vec4<f32>,
    emissive_factor: vec3<f32>,
    occlusion_strength: f32,
    metallic_factor: f32,
    roughness_factor: f32,
    ao_uv_index: u32,
    base_uv_index: u32,
    normal_uv_index: u32,
    mr_uv_index: u32,
    emissive_uv_index: u32,
    normal_scale: f32,
    alpha_cutoff: f32,
    alpha_mode: u32,
    _pad0: u32,
    base_uv_transform: mat4x4<f32>,
    normal_uv_transform: mat4x4<f32>,
    mr_uv_transform: mat4x4<f32>,
    emissive_uv_transform: mat4x4<f32>,
    ao_uv_transform: mat4x4<f32>,
}

@group(2) @binding(0) var tex_base_color: texture_2d<f32>;
@group(2) @binding(1) var samp_base_color: sampler;
@group(2) @binding(2) var tex_normal: texture_2d<f32>;
@group(2) @binding(3) var samp_normal: sampler;
@group(2) @binding(4) var tex_metal_rough: texture_2d<f32>;
@group(2) @binding(5) var samp_metal_rough: sampler;
@group(2) @binding(6) var tex_occlusion: texture_2d<f32>;
@group(2) @binding(7) var samp_occlusion: sampler;
@group(2) @binding(8) var tex_emissive: texture_2d<f32>;
@group(2) @binding(9) var samp_emissive: sampler;
@group(2) @binding(10) var<uniform> material: MaterialParams;

fn fresnel_schlick(cos_theta: f32, F0: vec3<f32>) -> vec3<f32> {
    return F0 + (vec3<f32>(1.0) - F0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn distribution_ggx(N: vec3<f32>, H: vec3<f32>, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let NdotH = max(dot(N, H), 0.0);
    let NdotH2 = NdotH * NdotH;
    let denom = (NdotH2 * (a2 - 1.0) + 1.0);
    return a2 / (3.14159265 * denom * denom + 1e-5);
}

fn geometry_smith(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    let NdotV = max(dot(N, V), 0.0);
    let NdotL = max(dot(N, L), 0.0);
    let ggx1 = NdotV / (NdotV * (1.0 - k) + k + 1e-5);
    let ggx2 = NdotL / (NdotL * (1.0 - k) + k + 1e-5);
    return ggx1 * ggx2;
}

@fragment
fn fs_main(input: FragmentInput) -> @location(0) vec4<f32> {
    // Flip Y coordinate to fix texture orientation
    let uv0_raw = input.tex_coords;
    let uv1_raw = input.tex_coords1;
    let uv0_base = (material.base_uv_transform * vec4<f32>(uv0_raw.x, uv0_raw.y, 0.0, 1.0)).xy;
    let uv1_base = (material.base_uv_transform * vec4<f32>(uv1_raw.x, uv1_raw.y, 0.0, 1.0)).xy;
    let uv0_norm = (material.normal_uv_transform * vec4<f32>(uv0_raw.x, uv0_raw.y, 0.0, 1.0)).xy;
    let uv1_norm = (material.normal_uv_transform * vec4<f32>(uv1_raw.x, uv1_raw.y, 0.0, 1.0)).xy;
    let uv0_mr = (material.mr_uv_transform * vec4<f32>(uv0_raw.x, uv0_raw.y, 0.0, 1.0)).xy;
    let uv1_mr = (material.mr_uv_transform * vec4<f32>(uv1_raw.x, uv1_raw.y, 0.0, 1.0)).xy;
    let uv0_em = (material.emissive_uv_transform * vec4<f32>(uv0_raw.x, uv0_raw.y, 0.0, 1.0)).xy;
    let uv1_em = (material.emissive_uv_transform * vec4<f32>(uv1_raw.x, uv1_raw.y, 0.0, 1.0)).xy;
    let uv_base0 = select(uv0_base, uv1_base, material.base_uv_index == 1u);
    let uv_norm0 = select(uv0_norm, uv1_norm, material.normal_uv_index == 1u);
    let uv_mr0 = select(uv0_mr, uv1_mr, material.mr_uv_index == 1u);
    let uv_em0 = select(uv0_em, uv1_em, material.emissive_uv_index == 1u);
    // Flip Y to match texture orientation
    let uv_base = vec2<f32>(uv_base0.x, 1.0 - uv_base0.y);
    let uv_norm = vec2<f32>(uv_norm0.x, 1.0 - uv_norm0.y);
    let uv_mr = vec2<f32>(uv_mr0.x, 1.0 - uv_mr0.y);
    let uv_em = vec2<f32>(uv_em0.x, 1.0 - uv_em0.y);

    // Sample base color (sRGB -> linear handled by texture format)
    let base_sample = textureSample(tex_base_color, samp_base_color, uv_base);
    var base = base_sample.rgb * material.base_color_factor.rgb * input.color.rgb;
    var alpha = base_sample.a * material.base_color_factor.a * input.color.a;

    // Metallic-Roughness (linear)
    let mr = textureSample(tex_metal_rough, samp_metal_rough, uv_mr).rgb;
    let metallic = clamp(mr.b * material.metallic_factor, 0.0, 1.0);
    let roughness = clamp(mr.g * material.roughness_factor, 0.04, 1.0);

    // Normal mapping
    let N = normalize(input.world_normal);
    let T = normalize(input.world_tangent);
    let B = normalize(cross(N, T) * input.tangent_sign);
    let tbn = mat3x3<f32>(T, B, N);
    var nmap = textureSample(tex_normal, samp_normal, uv_norm).xyz * 2.0 - vec3<f32>(1.0, 1.0, 1.0);
    nmap = vec3<f32>(nmap.x * material.normal_scale, nmap.y * material.normal_scale, nmap.z);
    let Nn = normalize(tbn * nmap);

    // Lighting vectors
    let V = normalize(lighting.view_position - input.world_position);
    let L = normalize(lighting.light_position - input.world_position);
    let H = normalize(V + L);

    let NdotL = max(dot(Nn, L), 0.0);
    let NdotV = max(dot(Nn, V), 0.0);

    // Fresnel base reflectance
    let F0 = mix(vec3<f32>(0.04, 0.04, 0.04), base, metallic);
    let F = fresnel_schlick(max(dot(H, V), 0.0), F0);
    let D = distribution_ggx(Nn, H, roughness);
    let G = geometry_smith(Nn, V, L, roughness);

    let numerator = D * G * F;
    let denom = 4.0 * NdotV * NdotL + 1e-5;
    let specular = numerator / denom;

    let kS = F;
    let kD = (vec3<f32>(1.0, 1.0, 1.0) - kS) * (1.0 - metallic);
    let diffuse = kD * base / 3.14159265;
    let radiance = lighting.light_color;
    var color = (diffuse + specular) * radiance * NdotL;

    // Ambient + AO
    let uv0_ao = (material.ao_uv_transform * vec4<f32>(uv0_raw.x, uv0_raw.y, 0.0, 1.0)).xy;
    let uv1_ao = (material.ao_uv_transform * vec4<f32>(uv1_raw.x, uv1_raw.y, 0.0, 1.0)).xy;
    let ao_uv0 = select(uv0_ao, uv1_ao, material.ao_uv_index == 1u);
    let ao_uv = vec2<f32>(ao_uv0.x, 1.0 - ao_uv0.y);
    let ao = textureSample(tex_occlusion, samp_occlusion, ao_uv).r * material.occlusion_strength;
    let ambient = lighting.ambient_strength * base;
    color = ambient * ao + color * lighting.lighting_intensity;

    // Alpha mask (cutout)
    if (material.alpha_mode == 1u && alpha < material.alpha_cutoff) {
        discard;
    }

    // Emissive
    let emissive = textureSample(tex_emissive, samp_emissive, uv_em).rgb * material.emissive_factor;
    color += emissive;

    // Alpha from base color a
    return vec4<f32>(color, alpha);
}

// Simple fragment shader for models without textures
@fragment
fn fs_main_simple(input: FragmentInput) -> @location(0) vec4<f32> {
    let object_color = vec3<f32>(0.8, 0.6, 0.4); // Default orange color

    // Ambient lighting
    let ambient = lighting.ambient_strength * lighting.light_color;

    // Diffuse lighting
    let norm = normalize(input.world_normal);
    let light_dir = normalize(lighting.light_position - input.world_position);
    let diff = max(dot(norm, light_dir), 0.0);
    let diffuse = diff * lighting.light_color;

    // Specular lighting
    let view_dir = normalize(lighting.view_position - input.world_position);
    let reflect_dir = reflect(-light_dir, norm);
    let spec = pow(max(dot(view_dir, reflect_dir), 0.0), 32.0);
    let specular = 0.5 * spec * lighting.light_color;

    let lit = (diffuse + specular) * lighting.lighting_intensity;
    let result = (ambient + lit) * object_color;
    return vec4<f32>(result, 1.0);
}

// Visualize normals as colors
@fragment
fn fs_show_normals(input: FragmentInput) -> @location(0) vec4<f32> {
    let n = normalize(input.world_normal);
    let color = 0.5 * (n + vec3<f32>(1.0, 1.0, 1.0));
    return vec4<f32>(color, 1.0);
}
