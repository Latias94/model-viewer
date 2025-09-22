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
    pub skeleton_roots: Vec<usize>,
    pub global_normalize: glam::Mat4,
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
            skeleton_roots: Vec::new(),
            global_normalize: glam::Mat4::IDENTITY,
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

    pub fn update_animation(
        &mut self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        time_sec: f32,
        freeze_root_motion: bool,
        skin_root_space: bool,
        freeze_skin_owner: bool,
        normalize_model: bool,
    ) {
        if self.nodes.is_empty() {
            return;
        }
        // Evaluate local transforms
        let mut local: Vec<glam::Mat4> = self.nodes.iter().map(|n| n.local_bind).collect();
        let mut near_wrap_flag = false;
        if let Some(anim) = &self.animation {
            if freeze_root_motion {
                log::debug!("Freeze root motion: ON");
            }
            // Some importers (e.g., glTF through Assimp) report 0 ticks/sec to mean "seconds".
            let tps = if anim.ticks_per_second <= 1e-6 {
                1.0
            } else {
                anim.ticks_per_second
            };
            let t_ticks = (time_sec as f64) * tps;
            let (t, near_wrap) = if anim.duration > 0.0 {
                let d = anim.duration;
                let mut tt = t_ticks % d;
                let eps = 1e-6;
                let near = tt < eps || tt > d - eps;
                if tt < eps {
                    tt = 0.0;
                }
                if tt > d - eps {
                    tt = d - eps;
                }
                (tt, near)
            } else {
                (t_ticks, false)
            };
            near_wrap_flag = near_wrap;
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
            // Optionally freeze root motion: keep translations on root chain at bind pose
            if freeze_root_motion {
                use std::collections::HashSet;
                if self.skeleton_roots.is_empty() {
                    self.skeleton_roots = self.compute_skeleton_roots();
                    log::debug!("Skeleton roots: {:?}", self.skeleton_roots);
                }
                let mut freeze_nodes: HashSet<usize> = HashSet::new();
                // Include skeleton roots
                for &ri in &self.skeleton_roots {
                    freeze_nodes.insert(ri);
                }
                // Include each mesh's node and its ancestors (armature/rig nodes)
                for mesh in &self.meshes {
                    let mut cur = Some(mesh.mesh_node_index);
                    while let Some(i) = cur {
                        if !freeze_nodes.insert(i) {
                            break;
                        }
                        cur = self.nodes[i].parent;
                    }
                }
                // Apply bind-pose translation on all collected nodes
                for i in freeze_nodes {
                    if let Some(m) = local.get_mut(i) {
                        let bind = self.nodes[i].local_bind;
                        m.w_axis = bind.w_axis;
                    }
                }
            }
            // Optionally freeze each skinned mesh owner node translation at bind pose
            if freeze_skin_owner {
                for mesh in &self.meshes {
                    if mesh.bone_count == 0 {
                        continue;
                    }
                    let ni = mesh.mesh_node_index;
                    if let Some(m) = local.get_mut(ni) {
                        let bind = self.nodes[ni].local_bind;
                        m.w_axis = bind.w_axis;
                    }
                }
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
            let mesh_global = global
                .get(mesh.mesh_node_index)
                .cloned()
                .unwrap_or(glam::Mat4::IDENTITY);
            let base_model = if skin_root_space && mesh.bone_count > 0 {
                glam::Mat4::IDENTITY
            } else {
                mesh_global
            };
            mesh.model_matrix = if normalize_model {
                self.global_normalize * base_model
            } else {
                base_model
            };
            if mesh.bone_count > 0 {
                // Compute final bone matrices in requested skin space
                let mut mats: Vec<[[f32; 4]; 4]> = Vec::with_capacity(mesh.bone_count as usize);
                let inv_mesh_global = mesh_global.inverse();

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

                    let mut final_bone_matrix = if skin_root_space {
                        // Root (LOGL): globalInverseRootBind * global(bone) * offset
                        self.global_inverse_root_bind * global_transform * offset_matrix
                    } else {
                        // Mesh-local: inv(mesh_global) * global(bone) * offset
                        inv_mesh_global * global_transform * offset_matrix
                    };
                    if !final_bone_matrix.is_finite() {
                        final_bone_matrix = glam::Mat4::IDENTITY;
                    }

                    mats.push(final_bone_matrix.to_cols_array_2d());
                }
                let reuse_prev = near_wrap_flag
                    && mesh.prev_skin_mats.as_ref().map(|v| v.len()).unwrap_or(0)
                        == (mesh.bone_count as usize);
                if reuse_prev {
                    if let Some(buf) = &mesh.skin_buffer {
                        let prev = mesh.prev_skin_mats.as_ref().unwrap();
                        let prev_arr: Vec<[[f32; 4]; 4]> =
                            prev.iter().map(|m| m.to_cols_array_2d()).collect();
                        queue.write_buffer(buf, 0, bytemuck::cast_slice(&prev_arr));
                    }
                } else {
                    if let Some(buf) = &mesh.skin_buffer {
                        queue.write_buffer(buf, 0, bytemuck::cast_slice(&mats));
                    }
                    mesh.prev_skin_mats = Some(
                        mats.iter()
                            .map(|m| glam::Mat4::from_cols_array_2d(m))
                            .collect(),
                    );
                }
            }
        }
    }

    fn compute_skeleton_roots(&self) -> Vec<usize> {
        use std::collections::HashSet;
        let mut out: HashSet<usize> = HashSet::new();
        for mesh in &self.meshes {
            if mesh.bone_node_indices.is_empty() {
                continue;
            }
            let set: HashSet<usize> = mesh.bone_node_indices.iter().cloned().collect();
            for &ni in &mesh.bone_node_indices {
                let mut is_root = true;
                if let Some(p) = self.nodes[ni].parent {
                    if set.contains(&p) {
                        is_root = false;
                    }
                }
                if is_root {
                    out.insert(ni);
                }
            }
        }
        out.into_iter().collect()
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
