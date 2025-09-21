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
}

impl Uniforms {
    pub fn new() -> Self {
        Self {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            model: Mat4::IDENTITY.to_cols_array_2d(),
            normal_matrix: Mat4::IDENTITY.to_cols_array_2d(),
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
        self.view_proj = (proj * view).to_cols_array_2d();
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct LightingData {
    pub light_position: [f32; 3],
    pub _padding1: f32,
    pub light_color: [f32; 3],
    pub _padding2: f32,
    pub view_position: [f32; 3],
    pub ambient_strength: f32,
    pub lighting_intensity: f32,
    pub _padding3: [f32; 3],
    pub alpha_cutoff: f32,
    pub alpha_mode: u32,
    pub _padding4: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct MaterialParams {
    pub base_color_factor: [f32; 4],
    pub emissive_factor: [f32; 3],
    pub occlusion_strength: f32,
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub ao_uv_index: u32,
    pub _pad: [f32; 5],
}

impl Default for MaterialParams {
    fn default() -> Self {
        Self {
            base_color_factor: [1.0, 1.0, 1.0, 1.0],
            emissive_factor: [0.0, 0.0, 0.0],
            occlusion_strength: 1.0,
            metallic_factor: 1.0,
            roughness_factor: 1.0,
            ao_uv_index: 0,
            _pad: [0.0; 5],
        }
    }
}

impl Default for LightingData {
    fn default() -> Self {
        Self {
            light_position: [2.0, 2.0, 2.0],
            _padding1: 0.0,
            light_color: [1.0, 1.0, 1.0],
            _padding2: 0.0,
            view_position: [0.0, 0.0, 3.0],
            ambient_strength: 0.3,
            lighting_intensity: 1.0,
            _padding3: [0.0; 3],
            alpha_cutoff: 0.5,
            alpha_mode: 0,
            _padding4: [0; 2],
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
    pub uniform_buffer: wgpu::Buffer,
    pub lighting_buffer: wgpu::Buffer,
    pub uniform_bind_group: wgpu::BindGroup,
    pub lighting_bind_group: wgpu::BindGroup,
    pub material_bind_group_layout: wgpu::BindGroupLayout,
    pub uniforms: Uniforms,
    pub lighting_data: LightingData,
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
                    visibility: wgpu::ShaderStages::VERTEX,
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
                ],
            });

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
            ],
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

        Self {
            pipeline_solid,
            pipeline_solid_double,
            pipeline_alpha,
            pipeline_alpha_double,
            pipeline_normals,
            pipeline_wireframe,
            uniform_buffer,
            lighting_buffer,
            uniform_bind_group,
            lighting_bind_group,
            material_bind_group_layout,
            uniforms,
            lighting_data,
        }
    }

    pub fn update_uniforms(
        &mut self,
        camera: &Camera,
        config: &wgpu::SurfaceConfiguration,
        queue: &wgpu::Queue,
        lighting_enabled: bool,
        light_intensity: f32,
        alpha_mask: bool,
        alpha_cutoff: f32,
    ) {
        self.uniforms.update_view_proj(camera, config);
        self.lighting_data.view_position = camera.position.to_array();
        if lighting_enabled {
            self.lighting_data.ambient_strength = 0.3;
            self.lighting_data.lighting_intensity = light_intensity;
        } else {
            self.lighting_data.ambient_strength = 1.0;
            self.lighting_data.lighting_intensity = 0.0;
        }
        self.lighting_data.alpha_mode = if alpha_mask { 1 } else { 0 };
        self.lighting_data.alpha_cutoff = alpha_cutoff;

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
