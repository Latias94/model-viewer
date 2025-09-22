use crate::{camera::Camera, model::Model, render::ModelRenderPipeline};

use super::{PassCtx, RenderPassExec};

pub struct OpaqueTransparentPass;

impl RenderPassExec for OpaqueTransparentPass {
    fn draw(
        &self,
        ctx: &mut PassCtx,
        pipeline: &ModelRenderPipeline,
        model: &Model,
        _camera: &Camera,
        _clear_color: Option<wgpu::Color>,
    ) {
        let mut rpass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("OpaqueTransparent Pass"),
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
            if !mesh.opaque_transparent {
                continue;
            }

            if mesh.two_sided {
                rpass.set_pipeline(&pipeline.pipeline_solid_double);
            } else {
                rpass.set_pipeline(&pipeline.pipeline_solid);
            }
            if let Some(bg) = &mesh.material_bind_group {
                rpass.set_bind_group(2, bg, &[]);
            }
            // Apply global model orientation correction if any
            let model_mat = pipeline.model_orientation * mesh.model_matrix;
            let normal_mat = model_mat.inverse().transpose();
            // Only update model and normal matrix, preserve view_proj
            ctx.queue.write_buffer(
                &pipeline.uniform_buffer,
                16 * 4, // Offset to model matrix (skip view_proj which is 16 floats)
                bytemuck::cast_slice(&model_mat.to_cols_array_2d()),
            );
            ctx.queue.write_buffer(
                &pipeline.uniform_buffer,
                32 * 4, // Offset to normal matrix
                bytemuck::cast_slice(&normal_mat.to_cols_array_2d()),
            );
            rpass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
            rpass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            rpass.draw_indexed(0..mesh.indices.len() as u32, 0, 0..1);
        }
    }
}
