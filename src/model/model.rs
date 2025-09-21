use crate::{model::Mesh, model::ModelLoader};

#[derive(Clone, Copy, Debug)]
pub enum LightKind {
    Directional = 0,
    Point = 1,
    Spot = 2,
}

#[derive(Clone, Copy, Debug)]
pub struct LightInfo {
    pub kind: LightKind,
    pub position: [f32; 3],
    pub direction: [f32; 3],
    pub color: [f32; 3],
    pub intensity: f32,
    pub range: f32,
    pub inner_cos: f32,
    pub outer_cos: f32,
}

pub struct Model {
    pub meshes: Vec<Mesh>,
    pub lights: Vec<LightInfo>,
}

impl Model {
    pub fn new() -> Self {
        Self {
            meshes: Vec::new(),
            lights: Vec::new(),
        }
    }

    pub async fn load(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        path: &str,
        texture_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        ModelLoader::load(device, queue, path, texture_bind_group_layout).await
    }

    pub fn render<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        for mesh in &self.meshes {
            mesh.render(render_pass);
        }
    }
}
