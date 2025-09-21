use std::sync::Arc;
use winit::{dpi::PhysicalSize, window::Window};

use crate::{
    camera::Camera,
    model::Model,
    render::{
        ModelRenderPipeline,
        passes::{
            PassCtx, RenderPassExec, opaque::OpaquePass, transparent::TransparentPass,
            two_sided::TwoSidedPass,
        },
    },
    ui::Ui,
};

pub struct Renderer {
    pub surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    pub size: PhysicalSize<u32>,
    pub depth_texture: wgpu::Texture,
    pub depth_view: wgpu::TextureView,
    pub model: Option<Model>,
    pub pipeline: ModelRenderPipeline,
    pub background: wgpu::Color,
    pub wireframe_supported: bool,
}

impl Renderer {
    pub async fn new(
        window: Arc<Window>,
        model_path: Option<String>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let size = window.inner_size();

        // Create wgpu instance
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::default(),
            ..Default::default()
        });

        // Create surface
        let surface = instance.create_surface(window.clone())?;

        // Request adapter
        let adapter = match instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
        {
            Ok(adapter) => adapter,
            Err(e) => return Err(format!("Failed to find an appropriate adapter: {:?}", e).into()),
        };
        let info = adapter.get_info();
        log::info!(
            "Adapter: {} ({:?}, {:?}), Driver: {}",
            info.name,
            info.backend,
            info.device_type,
            info.driver
        );

        // Select features
        let adapter_features = adapter.features();
        let wireframe_supported = adapter_features.contains(wgpu::Features::POLYGON_MODE_LINE);
        let mut required_features = wgpu::Features::empty();
        if wireframe_supported {
            required_features |= wgpu::Features::POLYGON_MODE_LINE;
        }

        // Request device and queue
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features,
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: Default::default(),
            })
            .await?;

        // Configure surface
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        // Prefer vsync to avoid stutter/tearing
        let present_mode = surface_caps
            .present_modes
            .iter()
            .copied()
            .find(|m| *m == wgpu::PresentMode::Fifo)
            .unwrap_or(surface_caps.present_modes[0]);

        log::info!("Present mode: {:?}", present_mode);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&device, &config);

        // Create depth texture
        let (depth_texture, depth_view) = Self::create_depth_texture(&device, &config);

        // Create render pipeline
        let pipeline = ModelRenderPipeline::new(&device, &queue, &config, wireframe_supported);

        // Load model if path is provided
        let model = if let Some(path) = model_path {
            Some(Model::load(&device, &queue, &path, &pipeline.material_bind_group_layout).await?)
        } else {
            None
        };

        Ok(Self {
            surface,
            device,
            queue,
            config,
            size,
            depth_texture,
            depth_view,
            model,
            pipeline,
            background: wgpu::Color {
                r: 0.1,
                g: 0.1,
                b: 0.1,
                a: 1.0,
            },
            wireframe_supported,
        })
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);

            // Recreate depth texture
            let (depth_texture, depth_view) =
                Self::create_depth_texture(&self.device, &self.config);
            self.depth_texture = depth_texture;
            self.depth_view = depth_view;
        }
    }

    pub fn render(
        &mut self,
        camera: &Camera,
        ui: &mut Ui,
        window: &Window,
        delta_time: f32,
        fps: f32,
        frame_time_ms: f32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Apply UI-driven uniforms
        let scene_lights = self.model.as_ref().map(|m| m.lights.as_slice());

        self.pipeline.update_uniforms(
            camera,
            &self.config,
            &self.queue,
            ui.enable_lighting(),
            ui.light_intensity(),
            scene_lights,
            ui.output_mode(),
        );

        let output = match self.surface.get_current_texture() {
            Ok(frame) => frame,
            Err(err) => {
                match err {
                    wgpu::SurfaceError::Lost => {
                        // Reconfigure and retry next frame
                        self.surface.configure(&self.device, &self.config);
                        return Ok(());
                    }
                    wgpu::SurfaceError::OutOfMemory => return Err("Out of memory".into()),
                    wgpu::SurfaceError::Outdated | wgpu::SurfaceError::Timeout => return Ok(()),
                    wgpu::SurfaceError::Other => return Ok(()),
                }
            }
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // Multi-pass forward rendering
        if let Some(model) = &self.model {
            // 更新 UBO（包含多光源），并执行分 Pass 渲染
            let scene_lights = Some(model.lights.as_slice());
            self.pipeline.update_uniforms(
                camera,
                &self.config,
                &self.queue,
                ui.enable_lighting(),
                ui.light_intensity(),
                scene_lights,
                ui.output_mode(),
            );

            let mut ctx = PassCtx {
                device: &self.device,
                queue: &self.queue,
                encoder: &mut encoder,
                view: &view,
                depth_view: &self.depth_view,
            };
            if ui.show_normals() {
                // 法线可视化单 pass
                let mut rpass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Normals Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: ctx.view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(self.background),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: ctx.depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    occlusion_query_set: None,
                    timestamp_writes: None,
                });
                rpass.set_pipeline(&self.pipeline.pipeline_normals);
                rpass.set_bind_group(0, &self.pipeline.uniform_bind_group, &[]);
                rpass.set_bind_group(1, &self.pipeline.lighting_bind_group, &[]);
                rpass.set_bind_group(3, &self.pipeline.environment_bind_group, &[]);
                for mesh in &model.meshes {
                    if let Some(bg) = &mesh.material_bind_group {
                        rpass.set_bind_group(2, bg, &[]);
                    }
                    // Update per-mesh model & normal matrices for normals visualization
                    let model_mat = mesh.model_matrix;
                    let normal_mat = model_mat.inverse().transpose();
                    let mut u = self.pipeline.uniforms;
                    u.model = model_mat.to_cols_array_2d();
                    u.normal_matrix = normal_mat.to_cols_array_2d();
                    self.queue.write_buffer(
                        &self.pipeline.uniform_buffer,
                        0,
                        bytemuck::cast_slice(&[u]),
                    );
                    mesh.render(&mut rpass);
                }
                drop(rpass);
            } else {
                // Skybox first
                crate::render::passes::skybox::SkyboxPass.draw(
                    &mut ctx,
                    &self.pipeline,
                    model,
                    camera,
                    Some(self.background),
                );
                // Geometry
                OpaquePass.draw(&mut ctx, &self.pipeline, model, camera, None);
                TwoSidedPass.draw(&mut ctx, &self.pipeline, model, camera, None);
                // Nearly-opaque alpha materials without sorting
                crate::render::passes::opaque_transparent::OpaqueTransparentPass.draw(
                    &mut ctx,
                    &self.pipeline,
                    model,
                    camera,
                    None,
                );
                TransparentPass.draw(&mut ctx, &self.pipeline, model, camera, None);
            }
        }

        // Render UI
        ui.render(
            window,
            &mut encoder,
            &view,
            &self.device,
            &self.queue,
            self.model.as_ref(),
            camera,
            delta_time,
            fps,
            frame_time_ms,
        )?;
        if let Some(path) = ui.take_pending_open_path() {
            log::info!("Loading model from UI: {}", path);
            if let Err(e) = self.load_model_blocking(&path) {
                log::error!("Failed to load model {}: {}", path, e);
            }
        }

        // Submit commands
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    fn load_model_blocking(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let model = pollster::block_on(Model::load(
            &self.device,
            &self.queue,
            path,
            &self.pipeline.material_bind_group_layout,
        ))?;
        self.model = Some(model);
        Ok(())
    }

    fn create_depth_texture(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let size = wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        };

        let desc = wgpu::TextureDescriptor {
            label: Some("depth_texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };

        let texture = device.create_texture(&desc);
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        (texture, view)
    }
}
