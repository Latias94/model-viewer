use std::sync::Arc;
use winit::{dpi::PhysicalSize, window::Window};

use crate::{camera::Camera, model::Model, render::ModelRenderPipeline, ui::Ui};

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

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
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
            Some(Model::load(&device, &queue, &path, &pipeline.texture_bind_group_layout).await?)
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
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Apply UI-driven uniforms
        self.pipeline.update_uniforms(
            camera,
            &self.config,
            &self.queue,
            ui.enable_lighting(),
            ui.light_intensity(),
        );

        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // Render 3D scene
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.background),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            // Choose pipeline variant
            let use_normals = ui.show_normals();
            let use_wireframe = ui.show_wireframe() && self.wireframe_supported;
            if use_normals {
                render_pass.set_pipeline(&self.pipeline.pipeline_normals);
            } else if use_wireframe {
                if let Some(wire) = &self.pipeline.pipeline_wireframe {
                    render_pass.set_pipeline(wire);
                } else {
                    render_pass.set_pipeline(&self.pipeline.pipeline_solid);
                }
            } else {
                render_pass.set_pipeline(&self.pipeline.pipeline_solid);
            }
            render_pass.set_bind_group(0, &self.pipeline.uniform_bind_group, &[]);
            render_pass.set_bind_group(1, &self.pipeline.lighting_bind_group, &[]);
            // Render model with texture if available
            if let Some(model) = &self.model {
                for mesh in &model.meshes {
                    if let Some(bg) = &mesh.texture_bind_group {
                        render_pass.set_bind_group(2, bg, &[]);
                    } else {
                        render_pass.set_bind_group(
                            2,
                            &self.pipeline.default_texture_bind_group,
                            &[],
                        );
                    }
                    mesh.render(&mut render_pass);
                }
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
            &self.pipeline.texture_bind_group_layout,
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
