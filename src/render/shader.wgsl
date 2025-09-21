// Vertex shader

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tex_coords: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) tex_coords: vec2<f32>,
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

    return out;
}

// Fragment shader

struct FragmentInput {
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) tex_coords: vec2<f32>,
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

@group(2) @binding(0)
var texture_diffuse: texture_2d<f32>;
@group(2) @binding(1)
var texture_sampler: sampler;

@fragment
fn fs_main(input: FragmentInput) -> @location(0) vec4<f32> {
    let object_color = textureSample(texture_diffuse, texture_sampler, input.tex_coords).rgb;

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

    // Apply lighting intensity to diffuse + specular
    let lit = (diffuse + specular) * lighting.lighting_intensity;
    let result = (ambient + lit) * object_color;
    return vec4<f32>(result, 1.0);
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
