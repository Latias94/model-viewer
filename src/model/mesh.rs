use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::render::Texture;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tex_coords: [f32; 2],
    pub tangent: [f32; 4],
    pub tex_coords1: [f32; 2],
    pub color: [f32; 4],
}

impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 6] = wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x3,
        2 => Float32x2,
        3 => Float32x4,
        4 => Float32x2,
        5 => Float32x4
    ];

    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub textures: Vec<Texture>,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub material_bind_group: Option<wgpu::BindGroup>,
    pub material_params_buffer: Option<wgpu::Buffer>,
    pub model_matrix: glam::Mat4,
    pub two_sided: bool,
    pub blend_mode: Option<asset_importer::material::BlendMode>,
}

impl Mesh {
    pub fn new(
        device: &wgpu::Device,
        _material_bind_group_layout: &wgpu::BindGroupLayout,
        vertices: Vec<Vertex>,
        indices: Vec<u32>,
        textures: Vec<Texture>,
        material_bind_group: Option<wgpu::BindGroup>,
        material_params_buffer: Option<wgpu::Buffer>,
        model_matrix: glam::Mat4,
        two_sided: bool,
        blend_mode: Option<asset_importer::material::BlendMode>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Ok(Self {
            vertices,
            indices,
            textures,
            vertex_buffer,
            index_buffer,
            material_bind_group,
            material_params_buffer,
            model_matrix,
            two_sided,
            blend_mode,
        })
    }

    pub fn render<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        render_pass.draw_indexed(0..self.indices.len() as u32, 0, 0..1);
    }
}
