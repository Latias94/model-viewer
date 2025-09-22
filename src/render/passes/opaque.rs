use crate::{camera::Camera, model::Model, render::ModelRenderPipeline};

use super::{PassCtx, RenderPassExec};

pub struct OpaquePass;

impl RenderPassExec for OpaquePass {
    fn draw(
        &self,
        ctx: &mut PassCtx,
        pipeline: &ModelRenderPipeline,
        model: &Model,
        _camera: &Camera,
        clear_color: Option<wgpu::Color>,
    ) {
        let load = if let Some(color) = clear_color {
            wgpu::LoadOp::Clear(color)
        } else {
            wgpu::LoadOp::Load
        };
        let mut rpass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Opaque Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: ctx.view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: ctx.depth_view,
                depth_ops: Some(wgpu::Operations {
                    // First geometry pass should clear depth
                    load: wgpu::LoadOp::Clear(1.0),
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

        let mut logged = false;

        for mesh in &model.meshes {
            let is_alpha = matches!(
                mesh.blend_mode,
                Some(
                    asset_importer::material::BlendMode::Default
                        | asset_importer::material::BlendMode::Additive
                )
            );
            if is_alpha || mesh.two_sided {
                continue;
            }

            rpass.set_pipeline(&pipeline.pipeline_solid);
            if let Some(bg) = &mesh.material_bind_group {
                rpass.set_bind_group(2, bg, &[]);
            }
            // material group (2) includes skin buffer at binding 11
            // Update per-mesh model & normal matrices
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
                32 * 4, // Offset to normal matrix (skip view_proj and model which are 32 floats)
                bytemuck::cast_slice(&normal_mat.to_cols_array_2d()),
            );
            if false && !logged && pipeline.frame_index <= 60 {
                let r = |m: &glam::Mat4, i: usize| -> (f32, f32, f32, f32) {
                    (m.x_axis[i], m.y_axis[i], m.z_axis[i], m.w_axis[i])
                };
                let (m0x, m0y, m0z, m0w) = r(&model_mat, 0);
                let (m1x, m1y, m1z, m1w) = r(&model_mat, 1);
                let (m2x, m2y, m2z, m2w) = r(&model_mat, 2);
                let (m3x, m3y, m3z, m3w) = r(&model_mat, 3);
                log::debug!(
                    "Model matrix rows:\n    [{:7.4}, {:7.4}, {:7.4}, {:7.4}]\n    [{:7.4}, {:7.4}, {:7.4}, {:7.4}]\n    [{:7.4}, {:7.4}, {:7.4}, {:7.4}]\n    [{:7.4}, {:7.4}, {:7.4}, {:7.4}]",
                    m0x,
                    m0y,
                    m0z,
                    m0w,
                    m1x,
                    m1y,
                    m1z,
                    m1w,
                    m2x,
                    m2y,
                    m2z,
                    m2w,
                    m3x,
                    m3y,
                    m3z,
                    m3w
                );
                logged = true;
            }
            rpass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
            rpass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            rpass.draw_indexed(0..mesh.indices.len() as u32, 0, 0..1);
        }
    }
}
