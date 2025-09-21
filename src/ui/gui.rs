use winit::{event::WindowEvent, window::Window};

use crate::camera::Camera;
use crate::model::Model;
use crate::render::Renderer;

pub struct Ui {
    context: egui::Context,
    state: egui_winit::State,
    renderer: egui_wgpu::Renderer,
    // UI state
    show_wireframe: bool,
    show_normals: bool,
    enable_lighting: bool,
    light_intensity: f32,
    show_performance: bool,
    pending_open_path: Option<String>,
    orbit_mode: bool,
    last_open_dir: Option<std::path::PathBuf>,
    alpha_mask: bool,
    alpha_cutoff: f32,
    output_mode_index: usize,
}

impl Ui {
    pub fn new(
        window: &Window,
        renderer: &Renderer,
        initial_dir: Option<std::path::PathBuf>,
    ) -> Self {
        let context = egui::Context::default();

        let egui_state = egui_winit::State::new(
            context.clone(),
            egui::viewport::ViewportId::ROOT,
            window,
            None,
            None,
            None,
        );

        let egui_renderer =
            egui_wgpu::Renderer::new(&renderer.device, renderer.config.format, None, 1, false);

        Self {
            context,
            state: egui_state,
            renderer: egui_renderer,
            show_wireframe: false,
            show_normals: false,
            enable_lighting: true,
            light_intensity: 1.0,
            show_performance: false,
            pending_open_path: None,
            orbit_mode: false,
            last_open_dir: initial_dir,
            alpha_mask: false,
            alpha_cutoff: 0.5,
            output_mode_index: 0,
        }
    }

    pub fn handle_event(&mut self, window: &Window, event: &WindowEvent) -> bool {
        let response = self.state.on_window_event(window, event);
        response.consumed
    }

    pub fn render(
        &mut self,
        window: &Window,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: Option<&Model>,
        camera: &Camera,
        delta_time: f32,
        fps: f32,
        frame_time_ms: f32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let raw_input = self.state.take_egui_input(window);

        // Capture ui state and potential actions
        let mut show_wireframe = self.show_wireframe;
        let mut show_normals = self.show_normals;
        let mut enable_lighting = self.enable_lighting;
        let mut light_intensity = self.light_intensity;
        let mut show_performance = self.show_performance;
        let mut open_path: Option<String> = None;

        let full_output = self.context.run(raw_input, |ctx| {
            // Main floating window
            egui::Window::new("Model Viewer")
                .default_width(320.0)
                .default_height(420.0)
                .default_pos([20.0, 20.0])
                .resizable(true)
                .collapsible(true)
                .open(&mut true)
                .show(ctx, |ui| {
                    egui::CollapsingHeader::new("Model Information")
                        .default_open(true)
                        .show(ui, |ui| {
                            if let Some(model) = model {
                                let total_vertices: usize =
                                    model.meshes.iter().map(|m| m.vertices.len()).sum();
                                let total_indices: usize =
                                    model.meshes.iter().map(|m| m.indices.len()).sum();
                                let total_triangles = total_indices / 3;
                                ui.label(format!("Meshes: {}", model.meshes.len()));
                                ui.label(format!("Vertices: {}", total_vertices));
                                ui.label(format!("Triangles: {}", total_triangles));
                            } else {
                                ui.label("No model loaded");
                                ui.small("Use: cargo run -- <model_path>");
                            }
                        });

                    if let Some(model) = model {
                        ui.separator();
                        egui::CollapsingHeader::new("Materials")
                            .default_open(false)
                            .show(ui, |ui| {
                                for (i, mesh) in model.meshes.iter().enumerate() {
                                    ui.group(|ui| {
                                        ui.label(format!("Mesh #{}", i));
                                        ui.label(format!("two_sided: {}", mesh.two_sided));
                                        ui.label(format!(
                                            "blend_mode: {}",
                                            match mesh.blend_mode {
                                                Some(asset_importer::material::BlendMode::Default) => "Default",
                                                Some(asset_importer::material::BlendMode::Additive) => "Additive",
                                                Some(_) => "Other",
                                                None => "None",
                                            }
                                        ));
                                        ui.label(format!("opaque_transparent: {}", mesh.opaque_transparent));
                                        if let Some(p) = &mesh.material_params_cpu {
                                            ui.separator();
                                            ui.label("MaterialParams:");
                                            ui.monospace(format!("base_color_factor = [{:.3}, {:.3}, {:.3}, {:.3}]",
                                                p.base_color_factor[0], p.base_color_factor[1], p.base_color_factor[2], p.base_color_factor[3]));
                                            ui.monospace(format!("metallic = {:.3}, roughness = {:.3}, normal_scale = {:.3}, alpha_cutoff = {:.3}",
                                                p.mr_factors[0], p.mr_factors[1], p.mr_factors[2], p.mr_factors[3]));
                                            ui.monospace(format!("uv_indices (ao, base, normal, mr) = [{}, {}, {}, {}]",
                                                p.uv_indices[0], p.uv_indices[1], p.uv_indices[2], p.uv_indices[3]));
                                            ui.monospace(format!("misc (emissive_uv, alpha_mode) = [{}, {}]",
                                                p.misc[0], p.misc[1]));
                                        }
                                    });
                                }
                            });
                    }

                    ui.separator();
                    egui::CollapsingHeader::new("Rendering Options")
                        .default_open(true)
                        .show(ui, |ui| {
                            ui.checkbox(&mut show_wireframe, "Wireframe mode");
                            ui.checkbox(&mut show_normals, "Show normals");
                            ui.checkbox(&mut enable_lighting, "Enable lighting");
                            ui.add(
                                egui::Slider::new(&mut light_intensity, 0.0..=3.0)
                                    .text("Light intensity"),
                            );
                            ui.separator();
                            ui.checkbox(&mut self.alpha_mask, "Alpha mask (cutout)");
                            ui.add_enabled(
                                self.alpha_mask,
                                egui::Slider::new(&mut self.alpha_cutoff, 0.0..=1.0)
                                    .text("Alpha cutoff"),
                            );
                            ui.separator();
                            let modes = [
                                "Final",
                                "BaseColor",
                                "Metallic",
                                "Roughness",
                                "Normal",
                                "Occlusion",
                                "Emissive",
                                "Alpha",
                                "TexCoord0",
                                "TexCoord1",
                            ];
                            egui::ComboBox::from_label("Output Mode")
                                .selected_text(modes[self.output_mode_index])
                                .show_ui(ui, |ui| {
                                    for (i, label) in modes.iter().enumerate() {
                                        ui.selectable_value(&mut self.output_mode_index, i, *label);
                                    }
                                });
                        });

                    ui.separator();
                    egui::CollapsingHeader::new("Camera")
                        .default_open(true)
                        .show(ui, |ui| {
                            ui.label(format!(
                                "Position: ({:.2}, {:.2}, {:.2})",
                                camera.position.x, camera.position.y, camera.position.z
                            ));
                            ui.label(format!("Yaw: {:.1}°", camera.yaw.to_degrees()));
                            ui.label(format!("Pitch: {:.1}°", camera.pitch.to_degrees()));
                            ui.label(format!("Zoom: {:.1}°", camera.zoom));
                            ui.label(format!("Speed: {:.1}", camera.movement_speed));
                            ui.separator();
                            ui.checkbox(&mut self.orbit_mode, "Orbit camera");
                        });

                    ui.separator();
                    if ui.button("Load Model").clicked() {
                        let mut dlg = rfd::FileDialog::new()
                            .add_filter("3D Models", &["gltf", "glb", "obj", "fbx", "dae"]);
                        if let Some(dir) = &self.last_open_dir {
                            dlg = dlg.set_directory(dir);
                        } else if let Ok(cwd) = std::env::current_dir() {
                            dlg = dlg.set_directory(cwd);
                        }
                        if let Some(path) = dlg.pick_file() {
                            if let Some(p) = path.to_str() {
                                open_path = Some(p.to_string());
                            }
                            if let Some(parent) = path.parent() {
                                self.last_open_dir = Some(parent.to_path_buf());
                            }
                        }
                    }

                    if ui.button("Export Screenshot").clicked() {
                        // TODO: Implement screenshot
                    }

                    ui.checkbox(&mut show_performance, "Show Performance");
                });

            if show_performance {
                egui::Window::new("Performance")
                    .default_width(200.0)
                    .show(ctx, |ui| {
                        ui.label(format!("FPS: {:.1}", fps));
                        ui.label(format!("Frame time: {:.2} ms", frame_time_ms));
                        ui.label(format!("Delta time: {:.4} s", delta_time));
                    });
            }
        });

        // Update state
        self.show_wireframe = show_wireframe;
        self.show_normals = show_normals;
        self.enable_lighting = enable_lighting;
        self.light_intensity = light_intensity;
        self.show_performance = show_performance;
        if self.pending_open_path.is_none() {
            self.pending_open_path = open_path;
        }

        self.state
            .handle_platform_output(window, full_output.platform_output);

        let paint_jobs = self
            .context
            .tessellate(full_output.shapes, full_output.pixels_per_point);
        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [window.inner_size().width, window.inner_size().height],
            pixels_per_point: window.scale_factor() as f32,
        };

        for (id, image_delta) in &full_output.textures_delta.set {
            self.renderer
                .update_texture(device, queue, *id, image_delta);
        }
        self.renderer
            .update_buffers(device, queue, encoder, &paint_jobs, &screen_descriptor);

        // Render egui on top
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("egui main render pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        let render_pass_static: &mut wgpu::RenderPass<'static> =
            unsafe { std::mem::transmute(&mut render_pass) };
        self.renderer
            .render(render_pass_static, &paint_jobs, &screen_descriptor);
        drop(render_pass);

        for id in &full_output.textures_delta.free {
            self.renderer.free_texture(id);
        }

        Ok(())
    }

    // Getters for render options
    pub fn show_wireframe(&self) -> bool {
        self.show_wireframe
    }
    pub fn show_normals(&self) -> bool {
        self.show_normals
    }
    pub fn enable_lighting(&self) -> bool {
        self.enable_lighting
    }
    pub fn light_intensity(&self) -> f32 {
        self.light_intensity
    }

    pub fn take_pending_open_path(&mut self) -> Option<String> {
        self.pending_open_path.take()
    }

    pub fn orbit_mode(&self) -> bool {
        self.orbit_mode
    }

    pub fn alpha_mask_enabled(&self) -> bool {
        self.alpha_mask
    }
    pub fn alpha_cutoff(&self) -> f32 {
        self.alpha_cutoff
    }

    pub fn output_mode(&self) -> u32 {
        self.output_mode_index as u32
    }
}
