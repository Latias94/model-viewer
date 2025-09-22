use crate::{model::Mesh, model::ModelLoader};
use std::collections::HashMap;

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
    // Node hierarchy for animation
    pub nodes: Vec<NodeData>,
    pub name_to_index: HashMap<String, usize>,
    pub animation: Option<AnimationClip>,
    pub root_index: usize,
    pub root_bind_global: glam::Mat4,
    pub global_inverse_root_bind: glam::Mat4,
    pub root_current_global: glam::Mat4,
}

impl Model {
    pub fn new() -> Self {
        Self {
            meshes: Vec::new(),
            lights: Vec::new(),
            nodes: Vec::new(),
            name_to_index: HashMap::new(),
            animation: None,
            root_index: 0,
            root_bind_global: glam::Mat4::IDENTITY,
            global_inverse_root_bind: glam::Mat4::IDENTITY,
            root_current_global: glam::Mat4::IDENTITY,
        }
    }

    pub async fn load(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        path: &str,
        texture_bind_group_layout: &wgpu::BindGroupLayout,
        anim_index: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        ModelLoader::load(device, queue, path, texture_bind_group_layout, anim_index).await
    }

    pub fn render<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        for mesh in &self.meshes {
            mesh.render(render_pass);
        }
    }

    pub fn update_animation(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, time_sec: f32) {
        if self.nodes.is_empty() {
            return;
        }
        // Evaluate local transforms
        let mut local: Vec<glam::Mat4> = self.nodes.iter().map(|n| n.local_bind).collect();
        if let Some(anim) = &self.animation {
            let t_ticks = (time_sec as f64) * anim.ticks_per_second;
            let t = if anim.duration > 0.0 {
                t_ticks % anim.duration
            } else {
                t_ticks
            };
            for ch in &anim.channels {
                let pos = sample_vec3(&ch.position_keys, t);
                let rot = sample_quat(&ch.rotation_keys, t);
                let scl = sample_vec3(&ch.scaling_keys, t).unwrap_or(glam::Vec3::ONE);
                let m = glam::Mat4::from_scale_rotation_translation(
                    scl,
                    rot.unwrap_or(glam::Quat::IDENTITY),
                    pos.unwrap_or(glam::Vec3::ZERO),
                );
                local[ch.node_index] = m;
            }
        }
        // Compute global from root to leaves (nodes stored in pre-order traversal)
        let mut global: Vec<glam::Mat4> = vec![glam::Mat4::IDENTITY; self.nodes.len()];
        for (i, node) in self.nodes.iter().enumerate() {
            if let Some(p) = node.parent {
                global[i] = global[p] * local[i];
            } else {
                global[i] = local[i];
            }
        }
        // Update root current global
        self.root_current_global = global
            .get(self.root_index)
            .cloned()
            .unwrap_or(glam::Mat4::IDENTITY);
        // Update per-mesh model matrix and skin buffers
        for mesh in &mut self.meshes {
            // Update model matrix to current node global transform
            if let Some(mg) = global.get(mesh.mesh_node_index) {
                mesh.model_matrix = *mg;
            }
            if mesh.bone_count > 0 {
                // For skinned meshes, compute final bone matrices using LearnOpenGL standard formula
                let mut mats: Vec<[[f32; 4]; 4]> = Vec::with_capacity(mesh.bone_count as usize);

                for (bi, &node_idx) in mesh.bone_node_indices.iter().enumerate() {
                    let global_transform = global
                        .get(node_idx)
                        .cloned()
                        .unwrap_or(glam::Mat4::IDENTITY);
                    let offset_matrix = mesh
                        .bone_offset_mats
                        .get(bi)
                        .cloned()
                        .unwrap_or(glam::Mat4::IDENTITY);

                    // Use the standard LearnOpenGL formula: finalBoneMatrix = globalTransformation * offsetMatrix
                    let final_bone_matrix = global_transform * offset_matrix;

                    mats.push(final_bone_matrix.to_cols_array_2d());
                }
                if let Some(buf) = &mesh.skin_buffer {
                    queue.write_buffer(buf, 0, bytemuck::cast_slice(&mats));
                }
            }
        }
    }
}

pub struct NodeData {
    pub name: String,
    pub parent: Option<usize>,
    pub local_bind: glam::Mat4,
}

pub struct AnimationClip {
    pub duration: f64,
    pub ticks_per_second: f64,
    pub channels: Vec<AnimChannel>,
}

pub struct AnimChannel {
    pub node_index: usize,
    pub position_keys: Vec<(f64, glam::Vec3)>,
    pub rotation_keys: Vec<(f64, glam::Quat)>,
    pub scaling_keys: Vec<(f64, glam::Vec3)>,
}

fn sample_vec3(keys: &Vec<(f64, glam::Vec3)>, t: f64) -> Option<glam::Vec3> {
    if keys.is_empty() {
        return None;
    }
    if keys.len() == 1 {
        return Some(keys[0].1);
    }
    // find interval
    let mut prev = keys[0];
    for k in keys.iter().copied() {
        if k.0 >= t {
            // interpolate prev..k
            let dt = (k.0 - prev.0).max(1e-8);
            let a = ((t - prev.0) / dt) as f32;
            return Some(prev.1.lerp(k.1, a));
        }
        prev = k;
    }
    Some(keys.last().unwrap().1)
}

fn sample_quat(keys: &Vec<(f64, glam::Quat)>, t: f64) -> Option<glam::Quat> {
    if keys.is_empty() {
        return None;
    }
    if keys.len() == 1 {
        return Some(keys[0].1);
    }
    let mut prev = keys[0];
    for k in keys.iter().copied() {
        if k.0 >= t {
            let dt = (k.0 - prev.0).max(1e-8);
            let a = ((t - prev.0) / dt) as f32;
            return Some(prev.1.slerp(k.1, a));
        }
        prev = k;
    }
    Some(keys.last().unwrap().1)
}
