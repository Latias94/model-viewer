use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use wgpu::{Device, util::DeviceExt};

use crate::{camera::Camera, model::Vertex};

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Uniforms {
    pub view_proj: [[f32; 4]; 4],
    pub model: [[f32; 4]; 4],
    pub normal_matrix: [[f32; 4]; 4],
    pub view_proj_inv: [[f32; 4]; 4],
}

impl Uniforms {
    pub fn new() -> Self {
        Self {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            model: Mat4::IDENTITY.to_cols_array_2d(),
            normal_matrix: Mat4::IDENTITY.to_cols_array_2d(),
            view_proj_inv: Mat4::IDENTITY.to_cols_array_2d(),
        }
    }

    pub fn update_view_proj(&mut self, camera: &Camera, config: &wgpu::SurfaceConfiguration) {
        let view = camera.view_matrix();
        let proj = Mat4::perspective_rh(
            camera.zoom.to_radians(),
            config.width as f32 / config.height as f32,
            0.1,
            100.0,
        );
        let vp = proj * view;
        self.view_proj = vp.to_cols_array_2d();
        self.view_proj_inv = vp.inverse().to_cols_array_2d();
    }
}

pub const MAX_LIGHTS: usize = 8;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct LightItemStd140 {
    pub position: [f32; 4],        // xyz + pad
    pub color_kind: [f32; 4],      // rgb + kind(as f32)
    pub direction_range: [f32; 4], // xyz + range
    pub params: [f32; 4],          // intensity, inner_cos, outer_cos, pad
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct LightingData {
    pub viewpos_ambient: [f32; 4], // view.xyz + ambient
    pub counts_pad: [f32; 4],      // light_count in x, others pad
    pub lights: [LightItemStd140; MAX_LIGHTS],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct MaterialParams {
    pub base_color_factor: [f32; 4],
    pub emissive_occlusion: [f32; 4], // emissive.xyz, occlusion_strength
    pub mr_factors: [f32; 4],         // metallic, roughness, normal_scale, alpha_cutoff
    pub uv_indices: [u32; 4],         // ao, base, normal, mr
    pub misc: [u32; 4],               // emissive_uv_index, alpha_mode, pad, pad
    pub base_uv_transform: [[f32; 4]; 4],
    pub normal_uv_transform: [[f32; 4]; 4],
    pub mr_uv_transform: [[f32; 4]; 4],
    pub emissive_uv_transform: [[f32; 4]; 4],
    pub ao_uv_transform: [[f32; 4]; 4],
}

impl Default for MaterialParams {
    fn default() -> Self {
        Self {
            base_color_factor: [1.0, 1.0, 1.0, 1.0],
            emissive_occlusion: [0.0, 0.0, 0.0, 1.0],
            mr_factors: [1.0, 1.0, 1.0, 0.5],
            uv_indices: [0, 0, 0, 0],
            misc: [0, 0, 0, 0],
            base_uv_transform: glam::Mat4::IDENTITY.to_cols_array_2d(),
            normal_uv_transform: glam::Mat4::IDENTITY.to_cols_array_2d(),
            mr_uv_transform: glam::Mat4::IDENTITY.to_cols_array_2d(),
            emissive_uv_transform: glam::Mat4::IDENTITY.to_cols_array_2d(),
            ao_uv_transform: glam::Mat4::IDENTITY.to_cols_array_2d(),
        }
    }
}

impl Default for LightingData {
    fn default() -> Self {
        Self {
            viewpos_ambient: [0.0, 0.0, 3.0, 0.3],
            counts_pad: [0.0, 0.0, 0.0, 0.0],
            lights: [LightItemStd140 {
                position: [0.0, 0.0, 0.0, 0.0],
                color_kind: [1.0, 1.0, 1.0, 0.0],
                direction_range: [0.0, -1.0, 0.0, -1.0],
                params: [1.0, 0.9, 0.75, 0.0],
            }; MAX_LIGHTS],
        }
    }
}

pub struct ModelRenderPipeline {
    pub pipeline_solid: wgpu::RenderPipeline,
    pub pipeline_solid_double: wgpu::RenderPipeline,
    pub pipeline_alpha: wgpu::RenderPipeline,
    pub pipeline_alpha_double: wgpu::RenderPipeline,
    pub pipeline_normals: wgpu::RenderPipeline,
    pub pipeline_wireframe: Option<wgpu::RenderPipeline>,
    pub pipeline_skybox: wgpu::RenderPipeline,
    pub uniform_buffer: wgpu::Buffer,
    pub lighting_buffer: wgpu::Buffer,
    pub uniform_bind_group: wgpu::BindGroup,
    pub lighting_bind_group: wgpu::BindGroup,
    pub material_bind_group_layout: wgpu::BindGroupLayout,
    pub environment_bind_group_layout: wgpu::BindGroupLayout,
    pub environment_bind_group: wgpu::BindGroup,
    pub environment: crate::render::environment::Environment,
    pub environment_mip_count: u32,
    pub uniforms: Uniforms,
    pub lighting_data: LightingData,
    pub exposure: f32,
}

impl ModelRenderPipeline {
    pub fn new(
        device: &Device,
        queue: &wgpu::Queue,
        config: &wgpu::SurfaceConfiguration,
        wireframe_supported: bool,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Model Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });
        let skybox_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Skybox Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("skybox.wgsl").into()),
        });

        // Create uniform buffers
        let uniforms = Uniforms::new();
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let lighting_data = LightingData::default();
        let lighting_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Lighting Buffer"),
            contents: bytemuck::cast_slice(&[lighting_data]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create bind group layouts
        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("uniform_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let lighting_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("lighting_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let material_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("material_bind_group_layout"),
                entries: &[
                    // baseColor texture + sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // normal texture + sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // metallic-roughness texture + sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // occlusion texture + sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // emissive texture + sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 8,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 9,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // material params uniform
                    wgpu::BindGroupLayoutEntry {
                        binding: 10,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // skin matrices storage buffer (read-only), used in vertex shader
                    wgpu::BindGroupLayoutEntry {
                        binding: 11,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        // Environment (IBL) bind group layout
        let environment_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("environment_bind_group_layout"),
                entries: &[
                    // irradiance cube + sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::Cube,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // prefiltered cube + sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::Cube,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // BRDF LUT 2D + sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        // Extend material layout with skin matrices (binding 11)
        // NOTE: We add after existing entries so all material bind groups include a skin buffer.
        // This buffer may be a 1-matrix identity for non-skinned meshes.

        // Create bind groups
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("uniform_bind_group"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let lighting_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lighting_bind_group"),
            layout: &lighting_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: lighting_buffer.as_entire_binding(),
            }],
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Model Render Pipeline Layout"),
            bind_group_layouts: &[
                &uniform_bind_group_layout,
                &lighting_bind_group_layout,
                &material_bind_group_layout,
                &environment_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });
        let skybox_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Skybox Pipeline Layout"),
            bind_group_layouts: &[&uniform_bind_group_layout, &environment_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Helper to create pipeline variants
        let make_pipeline = |polygon_mode: wgpu::PolygonMode,
                             fragment_entry: &str,
                             cull: Option<wgpu::Face>,
                             blend: Option<wgpu::BlendState>| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Model Render Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Vertex::desc()],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(fragment_entry),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format,
                        blend,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: cull,
                    polygon_mode,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
                cache: None,
            })
        };

        // Pipelines
        let opaque_blend = Some(wgpu::BlendState::REPLACE);
        let alpha_blend = Some(wgpu::BlendState::ALPHA_BLENDING);

        let pipeline_solid = make_pipeline(
            wgpu::PolygonMode::Fill,
            "fs_main",
            Some(wgpu::Face::Back),
            opaque_blend,
        );
        let pipeline_solid_double =
            make_pipeline(wgpu::PolygonMode::Fill, "fs_main", None, opaque_blend);
        let pipeline_alpha = make_pipeline(
            wgpu::PolygonMode::Fill,
            "fs_main",
            Some(wgpu::Face::Back),
            alpha_blend,
        );
        let pipeline_alpha_double =
            make_pipeline(wgpu::PolygonMode::Fill, "fs_main", None, alpha_blend);
        let pipeline_normals = make_pipeline(
            wgpu::PolygonMode::Fill,
            "fs_show_normals",
            Some(wgpu::Face::Back),
            opaque_blend,
        );
        let pipeline_wireframe = if wireframe_supported {
            Some(make_pipeline(
                wgpu::PolygonMode::Line,
                "fs_main",
                Some(wgpu::Face::Back),
                opaque_blend,
            ))
        } else {
            None
        };
        let pipeline_skybox = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Skybox Pipeline"),
            layout: Some(&skybox_layout),
            vertex: wgpu::VertexState {
                module: &skybox_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &skybox_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // Environment resources and bind group (HDR -> IBL generation, fallback to dummy)
        // Try a few common asset locations; if none exist, fall back to dummy env.
        let try_paths = [
            "assets/ibl/default.hdr",
            "assets/ibl/pisa.hdr",
            "assets/ibl/venice_sunset_1k.hdr",
            "assets/ibl/venice_sunset_1k.exr",
        ];
        let mut environment = None;
        for p in try_paths {
            if std::path::Path::new(p).exists() {
                match crate::render::environment::Environment::from_hdr(device, queue, p) {
                    Ok(env) => {
                        log::info!("Loaded IBL HDR: {}", p);
                        environment = Some(env);
                        break;
                    }
                    Err(e) => {
                        log::warn!("Failed to generate IBL from {}: {}", p, e);
                    }
                }
            }
        }
        let environment = environment.unwrap_or_else(|| {
            log::warn!("No HDR found under assets/ibl. Using dummy environment.");
            crate::render::environment::Environment::create_dummy(device, queue)
        });
        let environment_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("environment_bind_group"),
            layout: &environment_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&environment.irradiance_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&environment.irradiance_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&environment.prefiltered_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&environment.prefiltered_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&environment.brdf_lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&environment.brdf_lut_sampler),
                },
            ],
        });

        // Use actual prefilter mip count from environment
        let environment_mip_count: u32 = environment.prefilter_mips;

        Self {
            pipeline_solid,
            pipeline_solid_double,
            pipeline_alpha,
            pipeline_alpha_double,
            pipeline_normals,
            pipeline_wireframe,
            pipeline_skybox,
            uniform_buffer,
            lighting_buffer,
            uniform_bind_group,
            lighting_bind_group,
            material_bind_group_layout,
            environment_bind_group_layout,
            environment_bind_group,
            environment,
            environment_mip_count,
            uniforms,
            lighting_data,
            exposure: 1.0,
        }
    }

    pub fn update_uniforms(
        &mut self,
        camera: &Camera,
        config: &wgpu::SurfaceConfiguration,
        queue: &wgpu::Queue,
        lighting_enabled: bool,
        light_intensity: f32,
        scene_lights: Option<&[crate::model::LightInfo]>,
        output_mode: u32,
    ) {
        self.uniforms.update_view_proj(camera, config);
        self.lighting_data.viewpos_ambient = [
            camera.position.x,
            camera.position.y,
            camera.position.z,
            if lighting_enabled { 0.3 } else { 1.0 },
        ];

        // Fill lights
        let mut count = 0usize;
        if lighting_enabled {
            if let Some(lights) = scene_lights {
                for l in lights.iter().take(MAX_LIGHTS) {
                    let kind_f = match l.kind {
                        crate::model::LightKind::Directional => 0.0,
                        crate::model::LightKind::Point => 1.0,
                        crate::model::LightKind::Spot => 2.0,
                    };
                    self.lighting_data.lights[count] = LightItemStd140 {
                        position: [l.position[0], l.position[1], l.position[2], 0.0],
                        color_kind: [l.color[0], l.color[1], l.color[2], kind_f],
                        direction_range: [l.direction[0], l.direction[1], l.direction[2], l.range],
                        params: [l.intensity * light_intensity, l.inner_cos, l.outer_cos, 0.0],
                    };
                    count += 1;
                }
            }
        }
        self.lighting_data.counts_pad[0] = count as f32;
        self.lighting_data.counts_pad[1] = output_mode as f32;
        self.lighting_data.counts_pad[2] = self.environment_mip_count as f32;
        self.lighting_data.counts_pad[3] = self.exposure as f32;

        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.uniforms]),
        );
        queue.write_buffer(
            &self.lighting_buffer,
            0,
            bytemuck::cast_slice(&[self.lighting_data]),
        );
    }

    fn create_default_texture(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> crate::render::Texture {
        // Create a 1x1 white texture as default
        let size = wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("default_texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Write white pixel data
        let white_pixel = [255u8, 255u8, 255u8, 255u8];
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &white_pixel,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            size,
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        crate::render::Texture {
            texture,
            view,
            sampler,
        }
    }
}
