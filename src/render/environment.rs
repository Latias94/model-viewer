pub struct Environment {
    pub irradiance_tex: wgpu::Texture,
    pub irradiance_view: wgpu::TextureView,
    pub irradiance_sampler: wgpu::Sampler,
    pub prefiltered_tex: wgpu::Texture,
    pub prefiltered_view: wgpu::TextureView,
    pub prefiltered_sampler: wgpu::Sampler,
    pub brdf_lut_tex: wgpu::Texture,
    pub brdf_lut_view: wgpu::TextureView,
    pub brdf_lut_sampler: wgpu::Sampler,
    pub prefilter_mips: u32,
    // For debug: keep equirect source
    pub equirect_tex: wgpu::Texture,
    pub equirect_view: wgpu::TextureView,
    pub equirect_sampler: wgpu::Sampler,
}

impl Environment {
    pub fn from_hdr(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        hdr_path: &str,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // 1) Load HDR equirect texture as RGBA32F
        let img = image::open(hdr_path)?; // hdr feature enabled
        let rgb32f = img.to_rgb32f();
        let (w, h) = rgb32f.dimensions();
        let hdr_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("hdr_equirect"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        // Expand RGB32F to RGBA16F with A=1.0
        let mut data: Vec<u16> = Vec::with_capacity((w * h * 4) as usize);
        for p in rgb32f.pixels() {
            data.push(half::f16::from_f32(p[0]).to_bits());
            data.push(half::f16::from_f32(p[1]).to_bits());
            data.push(half::f16::from_f32(p[2]).to_bits());
            data.push(half::f16::from_f32(1.0).to_bits());
        }
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &hdr_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 8),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        let hdr_view = hdr_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let hdr_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // 2) Prepare target cube textures
        let irradiance_size = 32u32;
        let prefilter_size = 128u32;
        let prefilter_mips = 1 + (prefilter_size as f32).log2().floor() as u32;
        let irradiance_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("irradiance_cube"),
            size: wgpu::Extent3d {
                width: irradiance_size,
                height: irradiance_size,
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let prefiltered_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("prefiltered_cube"),
            size: wgpu::Extent3d {
                width: prefilter_size,
                height: prefilter_size,
                depth_or_array_layers: 6,
            },
            mip_level_count: prefilter_mips,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let irradiance_view = irradiance_tex.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });
        let prefiltered_view = prefiltered_tex.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // 3) Pipelines for convolution from equirect
        let conv_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ibl_convolution"),
            source: wgpu::ShaderSource::Wgsl(include_str!("ibl_convolution.wgsl").into()),
        });
        let params_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ibl_params_layout"),
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
        let src_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ibl_src_hdr_layout"),
                entries: &[
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
                ],
            });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ibl_conv_pipeline_layout"),
            bind_group_layouts: &[&src_bind_group_layout, &params_bind_group_layout],
            push_constant_ranges: &[],
        });
        let conv_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ibl_conv_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &conv_shader,
                entry_point: Some("vs_fullscreen"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &conv_shader,
                entry_point: Some("fs_convolve"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let src_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ibl_src_hdr_bg"),
            layout: &src_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&hdr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&hdr_sampler),
                },
            ],
        });

        // params buffer we update per draw
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct ParamsStd140 {
            mode: u32,
            face: u32,
            roughness: f32,
            sample_count: u32,
        }
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ibl_params_buf"),
            size: std::mem::size_of::<ParamsStd140>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ibl_params_bg"),
            layout: &params_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            }],
        });

        // 4) Render irradiance from HDR
        {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("irradiance_encoder"),
            });
            for face in 0..6u32 {
                let p = ParamsStd140 {
                    mode: 0,
                    face,
                    roughness: 0.0,
                    sample_count: 64,
                };
                queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&p));
                let view = irradiance_tex.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("irr_face"),
                    format: Some(wgpu::TextureFormat::Rgba16Float),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    aspect: wgpu::TextureAspect::All,
                    base_mip_level: 0,
                    mip_level_count: Some(1),
                    base_array_layer: face,
                    array_layer_count: Some(1),
                    ..Default::default()
                });
                let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("irradiance_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                });
                rpass.set_pipeline(&conv_pipeline);
                rpass.set_bind_group(0, &src_bind_group, &[]);
                rpass.set_bind_group(1, &params_bg, &[]);
                rpass.draw(0..3, 0..1);
            }
            queue.submit(Some(encoder.finish()));
        }

        // 5) Render prefiltered mips from HDR
        {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("prefilter_encoder"),
            });
            let max_mip = prefilter_mips - 1;
            for mip in 0..prefilter_mips {
                let rough = if max_mip > 0 {
                    (mip as f32) / (max_mip as f32)
                } else {
                    0.0
                };
                for face in 0..6u32 {
                    let p = ParamsStd140 {
                        mode: 1,
                        face,
                        roughness: rough,
                        sample_count: 64,
                    };
                    queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&p));
                    let view = prefiltered_tex.create_view(&wgpu::TextureViewDescriptor {
                        label: Some("preface"),
                        format: Some(wgpu::TextureFormat::Rgba16Float),
                        dimension: Some(wgpu::TextureViewDimension::D2),
                        aspect: wgpu::TextureAspect::All,
                        base_mip_level: mip,
                        mip_level_count: Some(1),
                        base_array_layer: face,
                        array_layer_count: Some(1),
                        ..Default::default()
                    });
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("prefilter_pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        occlusion_query_set: None,
                        timestamp_writes: None,
                    });
                    rpass.set_pipeline(&conv_pipeline);
                    rpass.set_bind_group(0, &src_bind_group, &[]);
                    rpass.set_bind_group(1, &params_bg, &[]);
                    rpass.draw(0..3, 0..1);
                }
            }
            queue.submit(Some(encoder.finish()));
        }

        // 6) BRDF LUT
        let brdf_lut_size = 256u32;
        let brdf_lut_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("brdf_lut"),
            size: wgpu::Extent3d {
                width: brdf_lut_size,
                height: brdf_lut_size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let brdf_lut_view = brdf_lut_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let brdf_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("brdf_lut_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("brdf_lut.wgsl").into()),
        });
        let brdf_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("brdf_lut_pipeline"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &brdf_shader,
                entry_point: Some("vs_fullscreen"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &brdf_shader,
                entry_point: Some("fs_brdf"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("brdf_lut_encoder"),
            });
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("brdf_lut_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &brdf_lut_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            rpass.set_pipeline(&brdf_pipeline);
            rpass.draw(0..3, 0..1);
            drop(rpass);
            queue.submit(Some(encoder.finish()));
        }

        let brdf_lut_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Ok(Self {
            irradiance_tex,
            irradiance_view,
            irradiance_sampler: sampler.clone(),
            prefiltered_tex,
            prefiltered_view,
            prefiltered_sampler: sampler,
            brdf_lut_tex,
            brdf_lut_view,
            brdf_lut_sampler,
            prefilter_mips,
            equirect_tex: hdr_tex,
            equirect_view: hdr_view,
            equirect_sampler: hdr_sampler,
        })
    }
    pub fn create_dummy(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        // Create small cube textures and fill with simple gradients to avoid validation issues
        let cube_size = 32u32;
        let mip_count = 1u32; // keep simple for now
        let cube_desc = wgpu::TextureDescriptor {
            label: Some("env_cube"),
            size: wgpu::Extent3d {
                width: cube_size,
                height: cube_size,
                depth_or_array_layers: 6,
            },
            mip_level_count: mip_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        };
        let irradiance_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("irradiance_cube"),
            ..cube_desc
        });
        let prefiltered_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("prefiltered_cube"),
            ..cube_desc
        });

        // Fill faces with colors (sky-like)
        for face in 0..6u32 {
            let mut data = vec![0u8; (cube_size * cube_size * 4) as usize];
            for y in 0..cube_size {
                for x in 0..cube_size {
                    let idx = ((y * cube_size + x) * 4) as usize;
                    let t = 1.0 - (y as f32 / (cube_size as f32 - 1.0));
                    let top = [13u8, 20, 33];
                    let bot = [51u8, 56, 64];
                    let r = (bot[0] as f32 * (1.0 - t) + top[0] as f32 * t) as u8;
                    let g = (bot[1] as f32 * (1.0 - t) + top[1] as f32 * t) as u8;
                    let b = (bot[2] as f32 * (1.0 - t) + top[2] as f32 * t) as u8;
                    data[idx] = r;
                    data[idx + 1] = g;
                    data[idx + 2] = b;
                    data[idx + 3] = 255;
                }
            }
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &irradiance_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: face,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                &data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(cube_size * 4),
                    rows_per_image: Some(cube_size),
                },
                wgpu::Extent3d {
                    width: cube_size,
                    height: cube_size,
                    depth_or_array_layers: 1,
                },
            );
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &prefiltered_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: face,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                &data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(cube_size * 4),
                    rows_per_image: Some(cube_size),
                },
                wgpu::Extent3d {
                    width: cube_size,
                    height: cube_size,
                    depth_or_array_layers: 1,
                },
            );
        }

        let irradiance_view = irradiance_tex.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });
        let prefiltered_view = prefiltered_tex.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // BRDF LUT 256x256 RG16F equivalent -> use Rgba16Float for simplicity
        let lut_size = 256u32;
        let brdf_lut_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("brdf_lut"),
            size: wgpu::Extent3d {
                width: lut_size,
                height: lut_size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        // Fill with neutral (approximate Fresnel integrate ~ small ramp)
        let mut lut = vec![0u8; (lut_size * lut_size * 4) as usize];
        for y in 0..lut_size {
            for x in 0..lut_size {
                let idx = ((y * lut_size + x) * 4) as usize;
                let v = (x as f32 / (lut_size as f32 - 1.0) * 255.0) as u8;
                lut[idx] = v;
                lut[idx + 1] = v;
                lut[idx + 2] = v;
                lut[idx + 3] = 255;
            }
        }
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &brdf_lut_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &lut,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(lut_size * 4),
                rows_per_image: Some(lut_size),
            },
            wgpu::Extent3d {
                width: lut_size,
                height: lut_size,
                depth_or_array_layers: 1,
            },
        );
        let brdf_lut_view = brdf_lut_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let brdf_lut_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        // Minimal equirect placeholder (1x1 white)
        let eq_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("equirect_dummy"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let white = [255u8, 255, 255, 255];
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &eq_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &white,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let eq_view = eq_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let eq_sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());

        Self {
            irradiance_tex,
            irradiance_view,
            irradiance_sampler: sampler.clone(),
            prefiltered_tex,
            prefiltered_view,
            prefiltered_sampler: sampler,
            brdf_lut_tex,
            brdf_lut_view,
            brdf_lut_sampler,
            prefilter_mips: 1,
            equirect_tex: eq_tex,
            equirect_view: eq_view,
            equirect_sampler: eq_sampler,
        }
    }
}
