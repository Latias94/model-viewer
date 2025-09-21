use crate::{camera::Camera, model::Model, render::ModelRenderPipeline};

pub struct PassCtx<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub encoder: &'a mut wgpu::CommandEncoder,
    pub view: &'a wgpu::TextureView,
    pub depth_view: &'a wgpu::TextureView,
}

pub trait RenderPassExec {
    fn draw(
        &self,
        ctx: &mut PassCtx,
        pipeline: &ModelRenderPipeline,
        model: &Model,
        camera: &Camera,
        clear_color: Option<wgpu::Color>,
    );
}

pub mod opaque;
pub mod opaque_transparent;
pub mod skybox;
pub mod transparent;
pub mod two_sided;
