use crate::{camera::Camera, model::Model, render::ModelRenderPipeline};

use super::{PassCtx, RenderPassExec};

pub struct TransparentPass;

impl TransparentPass {
    fn build_sorted<'a>(model: &'a Model, cam_pos: glam::Vec3) -> Vec<&'a crate::model::Mesh> {
        let mut items: Vec<&crate::model::Mesh> = model
            .meshes
            .iter()
            .filter(|m| {
                matches!(
                    m.blend_mode,
                    Some(
                        asset_importer::material::BlendMode::Default
                            | asset_importer::material::BlendMode::Additive
                    )
                )
            })
            .collect();

        items.sort_by(|a, b| {
            let ac = Self::mesh_center(a);
            let bc = Self::mesh_center(b);
            let da = cam_pos.distance(ac);
            let db = cam_pos.distance(bc);
            db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
        });
        items
    }

    fn mesh_center(mesh: &crate::model::Mesh) -> glam::Vec3 {
        if mesh.vertices.is_empty() {
            return glam::Vec3::ZERO;
        }
        let mut sum = glam::Vec3::ZERO;
        for v in &mesh.vertices {
            sum += glam::Vec3::new(v.position[0], v.position[1], v.position[2]);
        }
        sum / (mesh.vertices.len() as f32)
    }
}

impl RenderPassExec for TransparentPass {
    fn draw(
        &self,
        ctx: &mut PassCtx,
        pipeline: &ModelRenderPipeline,
        model: &Model,
        camera: &Camera,
        _clear_color: Option<wgpu::Color>,
    ) {
        let mut rpass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Transparent Pass"),
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

        let cam_pos = camera.position;
        let items = Self::build_sorted(model, cam_pos);
        for mesh in items {
            if mesh.opaque_transparent {
                continue;
            }
            if mesh.two_sided {
                rpass.set_pipeline(&pipeline.pipeline_alpha_double);
            } else {
                rpass.set_pipeline(&pipeline.pipeline_alpha);
            }
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
