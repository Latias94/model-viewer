use crate::{model::Mesh, model::ModelLoader};

pub struct Model {
    pub meshes: Vec<Mesh>,
}

impl Model {
    pub fn new() -> Self {
        Self { meshes: Vec::new() }
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
