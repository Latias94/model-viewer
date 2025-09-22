use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, ElementState, WindowEvent},
    event_loop::ActiveEventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

use crate::{camera::Camera, render::Renderer, ui::Ui};

pub struct App {
    pub window: Option<Arc<Window>>,
    pub renderer: Option<Renderer>,
    pub ui: Option<Ui>,
    pub camera: Camera,
    pub model_path: Option<String>,
    pub background_hex: String,
    pub last_frame: std::time::Instant,
    pub keys_pressed: std::collections::HashMap<KeyCode, bool>,
    pub first_mouse: bool,
    pub last_mouse_pos: (f64, f64),
    pub rotating: bool,
    pub panning: bool,
    pub initial_size: (u32, u32),
    pub fps: f32,
    pub frame_time_ms: f32,
    pub orbit_target: glam::Vec3,
    pub orbit_radius: f32,
    // animation & exposure
    pub anim_enabled: bool,
    pub anim_index: usize,
    pub anim_speed: f32,
    pub exposure: f32,
}

impl App {
    pub fn new(
        model_path: Option<String>,
        background_hex: String,
        initial_size: (u32, u32),
        anim_enabled: bool,
        anim_index: usize,
        anim_speed: f32,
        exposure: f32,
    ) -> Self {
        Self {
            window: None,
            renderer: None,
            ui: None,
            camera: Camera::new(),
            model_path,
            background_hex,
            last_frame: std::time::Instant::now(),
            keys_pressed: std::collections::HashMap::new(),
            first_mouse: true,
            last_mouse_pos: (400.0, 300.0),
            rotating: false,
            panning: false,
            initial_size,
            fps: 0.0,
            frame_time_ms: 0.0,
            orbit_target: glam::Vec3::ZERO,
            orbit_radius: 5.0,
            anim_enabled,
            anim_index,
            anim_speed,
            exposure,
        }
    }

    fn process_input(&mut self, delta_time: f32) {
        use crate::camera::CameraMovement;

        let mut dt = delta_time;
        // Shift to speed up
        if *self.keys_pressed.get(&KeyCode::ShiftLeft).unwrap_or(&false)
            || *self
                .keys_pressed
                .get(&KeyCode::ShiftRight)
                .unwrap_or(&false)
        {
            dt *= 3.0;
        }

        if *self.keys_pressed.get(&KeyCode::KeyW).unwrap_or(&false) {
            self.camera.process_keyboard(CameraMovement::Forward, dt);
        }
        if *self.keys_pressed.get(&KeyCode::KeyS).unwrap_or(&false) {
            self.camera.process_keyboard(CameraMovement::Backward, dt);
        }
        if *self.keys_pressed.get(&KeyCode::KeyA).unwrap_or(&false) {
            self.camera.process_keyboard(CameraMovement::Left, dt);
        }
        if *self.keys_pressed.get(&KeyCode::KeyD).unwrap_or(&false) {
            self.camera.process_keyboard(CameraMovement::Right, dt);
        }
        if *self.keys_pressed.get(&KeyCode::Space).unwrap_or(&false) {
            self.camera.process_keyboard(CameraMovement::Up, dt);
        }
        if *self
            .keys_pressed
            .get(&KeyCode::ControlLeft)
            .unwrap_or(&false)
            || *self
                .keys_pressed
                .get(&KeyCode::ControlRight)
                .unwrap_or(&false)
        {
            self.camera.process_keyboard(CameraMovement::Down, dt);
        }
    }

    fn render(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let current_frame = std::time::Instant::now();
        let delta_time = current_frame.duration_since(self.last_frame).as_secs_f32();
        self.last_frame = current_frame;

        // Smooth metrics (EMA)
        let alpha = 0.1f32;
        if delta_time.is_finite() && delta_time > 0.00001 {
            let inst_fps = 1.0 / delta_time;
            let inst_ms = delta_time * 1000.0;
            if self.fps == 0.0 {
                self.fps = inst_fps;
                self.frame_time_ms = inst_ms;
            } else {
                self.fps = self.fps * (1.0 - alpha) + inst_fps * alpha;
                self.frame_time_ms = self.frame_time_ms * (1.0 - alpha) + inst_ms * alpha;
            }
        }

        self.process_input(delta_time);

        if let (Some(renderer), Some(ui), Some(window)) =
            (&mut self.renderer, &mut self.ui, &self.window)
        {
            renderer.render(
                &self.camera,
                ui,
                window,
                delta_time,
                self.fps,
                self.frame_time_ms,
            )?;
            // Apply camera updates from UI
            self.camera.movement_speed = ui.camera_speed();
            if ui.take_reset_camera() {
                self.camera = Camera::new();
            }
        }

        Ok(())
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window_attributes = Window::default_attributes()
                .with_title("Model Viewer")
                .with_inner_size(winit::dpi::LogicalSize::new(
                    self.initial_size.0 as f64,
                    self.initial_size.1 as f64,
                ));

            let window = Arc::new(event_loop.create_window(window_attributes).unwrap());

            // Initialize renderer and UI
            let mut renderer = pollster::block_on(Renderer::new(
                Arc::clone(&window),
                self.model_path.clone(),
                self.anim_enabled,
                self.anim_index,
                self.anim_speed,
                self.exposure,
            ))
            .unwrap();
            // Apply background from CLI
            renderer.background = parse_hex_color(&self.background_hex).unwrap_or(wgpu::Color {
                r: 0.125,
                g: 0.125,
                b: 0.125,
                a: 1.0,
            });
            // Determine initial file dialog directory
            let initial_dir = if let Some(path) = &self.model_path {
                std::path::Path::new(path).parent().map(|p| p.to_path_buf())
            } else {
                std::env::current_dir().ok()
            };
            let ui = Ui::new(&window, &renderer, initial_dir);

            self.renderer = Some(renderer);
            self.ui = Some(ui);
            self.window = Some(window);
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        // Continuously redraw to drive rendering & egui
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        // Check if UI wants to handle the event, but don't consume it yet
        let ui_wants_event = if let Some(ui) = &mut self.ui {
            if let Some(window) = &self.window {
                ui.handle_event(window, &event)
            } else {
                false
            }
        } else {
            false
        };

        match event {
            WindowEvent::CloseRequested => {
                log::info!("Close was requested; stopping");
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                if let Some(renderer) = &mut self.renderer {
                    renderer.resize(size);
                }
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => {
                if let Err(e) = self.render() {
                    log::error!("Render error: {}", e);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(keycode) = event.physical_key {
                    match event.state {
                        ElementState::Pressed => {
                            self.keys_pressed.insert(keycode, true);
                            if keycode == KeyCode::Escape {
                                event_loop.exit();
                            }
                            if let Some(window) = &self.window {
                                window.request_redraw();
                            }
                        }
                        ElementState::Released => {
                            self.keys_pressed.insert(keycode, false);
                            if let Some(window) = &self.window {
                                window.request_redraw();
                            }
                        }
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if button == winit::event::MouseButton::Right {
                    self.rotating = state == ElementState::Pressed;
                    self.first_mouse = true;
                    if let Some(window) = &self.window {
                        let _ = window.set_cursor_grab(if self.rotating {
                            winit::window::CursorGrabMode::Locked
                        } else {
                            winit::window::CursorGrabMode::None
                        });
                        window.set_cursor_visible(!self.rotating);
                    }
                }
                if button == winit::event::MouseButton::Middle {
                    self.panning = state == ElementState::Pressed;
                    self.first_mouse = true;
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let (x, y) = (position.x, position.y);

                if self.first_mouse {
                    self.last_mouse_pos = (x, y);
                    self.first_mouse = false;
                }

                let (last_x, last_y) = self.last_mouse_pos;
                let xoffset = x - last_x;
                let yoffset = last_y - y; // reversed since y-coordinates go from bottom to top

                self.last_mouse_pos = (x, y);

                // Only handle camera movement if UI doesn't want the event
                let orbit_mode = if let Some(ui) = &self.ui {
                    ui.orbit_mode()
                } else {
                    false
                };
                if !ui_wants_event && self.rotating {
                    self.camera
                        .process_mouse_movement(xoffset as f32, yoffset as f32, true);
                    if orbit_mode {
                        // position is target - front * radius
                        self.camera.position =
                            self.orbit_target - self.camera.front * self.orbit_radius;
                    }
                }
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                // Only handle camera zoom if UI doesn't want the event
                if !ui_wants_event {
                    let orbit_mode = if let Some(ui) = &self.ui {
                        ui.orbit_mode()
                    } else {
                        false
                    };
                    match delta {
                        winit::event::MouseScrollDelta::LineDelta(_, y) => {
                            if orbit_mode {
                                self.orbit_radius =
                                    (self.orbit_radius - y * 0.25).clamp(0.2, 200.0);
                                self.camera.position =
                                    self.orbit_target - self.camera.front * self.orbit_radius;
                            } else {
                                self.camera.process_mouse_scroll(y);
                            }
                        }
                        winit::event::MouseScrollDelta::PixelDelta(pos) => {
                            let y = pos.y as f32 / 50.0;
                            if orbit_mode {
                                self.orbit_radius =
                                    (self.orbit_radius - y * 0.25).clamp(0.2, 200.0);
                                self.camera.position =
                                    self.orbit_target - self.camera.front * self.orbit_radius;
                            } else {
                                self.camera.process_mouse_scroll(y);
                            }
                        }
                    }
                }
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        // Use high-frequency raw mouse motion when rotating
        if let DeviceEvent::MouseMotion { delta } = event {
            let (dx, dy) = (delta.0 as f32, delta.1 as f32);
            let orbit_mode = if let Some(ui) = &self.ui {
                ui.orbit_mode()
            } else {
                false
            };
            if self.rotating {
                self.camera.process_mouse_movement(dx, -dy, true);
                if orbit_mode {
                    self.camera.position =
                        self.orbit_target - self.camera.front * self.orbit_radius;
                }
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            } else if self.panning {
                // Pan: translate along camera right/up
                let pan_sensitivity = 0.005;
                let delta_right = -dx * pan_sensitivity;
                let delta_up = dy * pan_sensitivity;
                let offset = self.camera.right * delta_right + self.camera.up * delta_up;
                if orbit_mode {
                    self.orbit_target += offset;
                }
                self.camera.position += offset;
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
        }
    }

    // removed duplicate stub device_event
}

fn parse_hex_color(hex: &str) -> Option<wgpu::Color> {
    let s = hex.trim().trim_start_matches('#');
    if s.len() != 6 {
        return None;
    }
    let r = u8::from_str_radix(&s[0..2], 16).ok()?;
    let g = u8::from_str_radix(&s[2..4], 16).ok()?;
    let b = u8::from_str_radix(&s[4..6], 16).ok()?;
    Some(wgpu::Color {
        r: r as f64 / 255.0,
        g: g as f64 / 255.0,
        b: b as f64 / 255.0,
        a: 1.0,
    })
}
