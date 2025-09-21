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
    pub initial_size: (u32, u32),
}

impl App {
    pub fn new(
        model_path: Option<String>,
        background_hex: String,
        initial_size: (u32, u32),
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
            initial_size,
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

        self.process_input(delta_time);

        if let (Some(renderer), Some(ui), Some(window)) =
            (&mut self.renderer, &mut self.ui, &self.window)
        {
            renderer.render(&self.camera, ui, window)?;
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
            let mut renderer =
                pollster::block_on(Renderer::new(Arc::clone(&window), self.model_path.clone()))
                    .unwrap();
            // Apply background from CLI
            renderer.background = parse_hex_color(&self.background_hex).unwrap_or(wgpu::Color {
                r: 0.125,
                g: 0.125,
                b: 0.125,
                a: 1.0,
            });
            let ui = Ui::new(&window, &renderer);

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
                if !ui_wants_event && self.rotating {
                    self.camera
                        .process_mouse_movement(xoffset as f32, yoffset as f32, true);
                }
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                // Only handle camera zoom if UI doesn't want the event
                if !ui_wants_event {
                    match delta {
                        winit::event::MouseScrollDelta::LineDelta(_, y) => {
                            self.camera.process_mouse_scroll(y);
                        }
                        winit::event::MouseScrollDelta::PixelDelta(pos) => {
                            self.camera.process_mouse_scroll(pos.y as f32 / 50.0);
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
        _event: DeviceEvent,
    ) {
        // Handle device events if needed
    }
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
