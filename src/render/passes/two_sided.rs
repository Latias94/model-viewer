use crate::{camera::Camera, model::Model, render::ModelRenderPipeline};

use super::{PassCtx, RenderPassExec};

pub struct TwoSidedPass;

impl RenderPassExec for TwoSidedPass {
    fn draw(
        &self,
        ctx: &mut PassCtx,
        pipeline: &ModelRenderPipeline,
        model: &Model,
        _camera: &Camera,
        _clear_color: Option<wgpu::Color>,
    ) {
        let mut rpass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("TwoSided Opaque Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: ctx.view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: ctx.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        rpass.set_bind_group(0, &pipeline.uniform_bind_group, &[]);
        rpass.set_bind_group(1, &pipeline.lighting_bind_group, &[]);
        rpass.set_bind_group(3, &pipeline.environment_bind_group, &[]);

        for mesh in &model.meshes {
            let is_alpha = matches!(
                mesh.blend_mode,
                Some(
                    asset_importer::material::BlendMode::Default
                        | asset_importer::material::BlendMode::Additive
                )
            );
            if is_alpha || !mesh.two_sided {
                continue;
            }

            rpass.set_pipeline(&pipeline.pipeline_solid_double);
            if let Some(bg) = &mesh.material_bind_group {
                rpass.set_bind_group(2, bg, &[]);
            }
            // material group (2) includes skin buffer at binding 11
            // Update per-mesh model & normal matrices
            // Use the mesh's model matrix for both skinned and non-skinned meshes
            let model_mat = mesh.model_matrix;
            let normal_mat = model_mat.inverse().transpose();
            let mut u = pipeline.uniforms;
            u.model = model_mat.to_cols_array_2d();
            u.normal_matrix = normal_mat.to_cols_array_2d();
            ctx.queue
                .write_buffer(&pipeline.uniform_buffer, 0, bytemuck::cast_slice(&[u]));
            rpass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
            rpass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            rpass.draw_indexed(0..mesh.indices.len() as u32, 0, 0..1);
        }
    }
}
