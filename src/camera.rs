use glam::{Mat4, Vec3};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum CameraMovement {
    Forward,
    Backward,
    Left,
    Right,
    Up,
    Down,
}

pub struct Camera {
    // Camera attributes
    pub position: Vec3,
    pub front: Vec3,
    pub up: Vec3,
    pub right: Vec3,
    pub world_up: Vec3,

    // Euler angles
    pub yaw: f32,
    pub pitch: f32,

    // Camera options
    pub movement_speed: f32,
    pub mouse_sensitivity: f32,
    pub zoom: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self::new()
    }
}

impl Camera {
    pub fn new() -> Self {
        let mut camera = Self {
            position: Vec3::new(0.0, 0.0, 3.0),
            front: Vec3::new(0.0, 0.0, -1.0),
            up: Vec3::ZERO,
            right: Vec3::ZERO,
            world_up: Vec3::Y,
            yaw: -90.0,
            pitch: 0.0,
            movement_speed: 2.5,
            mouse_sensitivity: 0.1,
            zoom: 45.0,
        };
        camera.update_camera_vectors();
        camera
    }

    pub fn new_with_position(position: Vec3) -> Self {
        let mut camera = Self {
            position,
            front: Vec3::new(0.0, 0.0, -1.0),
            up: Vec3::ZERO,
            right: Vec3::ZERO,
            world_up: Vec3::Y,
            yaw: -90.0,
            pitch: 0.0,
            movement_speed: 2.5,
            mouse_sensitivity: 0.1,
            zoom: 45.0,
        };
        camera.update_camera_vectors();
        camera
    }

    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.position + self.front, self.up)
    }

    pub fn process_keyboard(&mut self, direction: CameraMovement, delta_time: f32) {
        let velocity = self.movement_speed * delta_time;
        match direction {
            CameraMovement::Forward => self.position += self.front * velocity,
            CameraMovement::Backward => self.position -= self.front * velocity,
            CameraMovement::Left => self.position -= self.right * velocity,
            CameraMovement::Right => self.position += self.right * velocity,
            CameraMovement::Up => self.position += self.world_up * velocity,
            CameraMovement::Down => self.position -= self.world_up * velocity,
        }
    }

    pub fn process_mouse_movement(
        &mut self,
        mut xoffset: f32,
        mut yoffset: f32,
        constrain_pitch: bool,
    ) {
        xoffset *= self.mouse_sensitivity;
        yoffset *= self.mouse_sensitivity;

        self.yaw += xoffset;
        self.pitch += yoffset;

        if constrain_pitch {
            self.pitch = self.pitch.clamp(-89.0, 89.0);
        }

        self.update_camera_vectors();
    }

    pub fn process_mouse_scroll(&mut self, yoffset: f32) {
        self.zoom -= yoffset;
        self.zoom = self.zoom.clamp(1.0, 45.0);
    }

    fn update_camera_vectors(&mut self) {
        let front = Vec3::new(
            self.yaw.to_radians().cos() * self.pitch.to_radians().cos(),
            self.pitch.to_radians().sin(),
            self.yaw.to_radians().sin() * self.pitch.to_radians().cos(),
        );
        self.front = front.normalize();
        self.right = self.front.cross(self.world_up).normalize();
        self.up = self.right.cross(self.front).normalize();
    }
}
