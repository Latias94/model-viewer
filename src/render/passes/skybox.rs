use super::{PassCtx, RenderPassExec};

pub struct SkyboxPass;

impl RenderPassExec for SkyboxPass {
    fn draw(
        &self,
        ctx: &mut PassCtx,
        pipeline: &crate::render::ModelRenderPipeline,
        _model: &crate::model::Model,
        _camera: &crate::camera::Camera,
        clear_color: Option<wgpu::Color>,
    ) {
        let load = if clear_color.is_some() {
            wgpu::LoadOp::Clear(clear_color.unwrap())
        } else {
            wgpu::LoadOp::Load
        };
        let mut rpass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Skybox Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: ctx.view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        rpass.set_pipeline(&pipeline.pipeline_skybox);
        // Bind camera uniforms (group 0) and environment (group 1)
        rpass.set_bind_group(0, &pipeline.uniform_bind_group, &[]);
        rpass.set_bind_group(1, &pipeline.environment_bind_group, &[]);
        rpass.draw(0..3, 0..1);
    }
}
