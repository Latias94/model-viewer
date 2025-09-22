use crate::model::{Mesh, Model, Vertex, mesh::MaterialDebug};
use asset_importer::{Importer, postprocess::PostProcessSteps};
use wgpu::util::DeviceExt;

pub struct ModelLoader;

struct LoadedPbr {
    base_color: Option<crate::render::Texture>,
    normal: Option<crate::render::Texture>,
    metallic_roughness: Option<crate::render::Texture>,
    occlusion: Option<crate::render::Texture>,
    emissive: Option<crate::render::Texture>,
    base_color_factor: [f32; 4],
    metallic_factor: f32,
    roughness_factor: f32,
    emissive_factor: [f32; 3],
    occlusion_strength: f32,
    ao_uv_index: u32,
    base_uv_index: u32,
    normal_uv_index: u32,
    mr_uv_index: u32,
    emissive_uv_index: u32,
    base_uv_transform: glam::Mat4,
    normal_uv_transform: glam::Mat4,
    mr_uv_transform: glam::Mat4,
    emissive_uv_transform: glam::Mat4,
    ao_uv_transform: glam::Mat4,
    normal_scale: f32,
    alpha_mode: u32,
    alpha_cutoff: f32,
}

impl ModelLoader {
    pub async fn load(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        path: &str,
        material_bind_group_layout: &wgpu::BindGroupLayout,
        anim_index: usize,
    ) -> Result<Model, Box<dyn std::error::Error>> {
        log::info!("Loading model: {}", path);

        let importer = Importer::new();
        let scene = importer
            .read_file(path)
            .with_post_process(
                PostProcessSteps::TRIANGULATE
                    | PostProcessSteps::GEN_SMOOTH_NORMALS
                    | PostProcessSteps::CALC_TANGENT_SPACE
                    | PostProcessSteps::TRANSFORM_UV_COORDS,
            )
            .import_file(path)?;

        log::debug!("Scene loaded successfully!");
        log::debug!("  Meshes: {}", scene.num_meshes());
        log::debug!("  Materials: {}", scene.num_materials());
        log::debug!("  Textures: {}", scene.num_textures());

        let mut model = Model::new();

        // Build node hierarchy
        if let Some(root_node) = scene.root_node() {
            let mut nodes = Vec::new();
            let mut map = std::collections::HashMap::new();
            Self::collect_nodes(&root_node, None, &mut nodes, &mut map);
            // store in model
            model.name_to_index = map;
            model.nodes = nodes;
        }

        // Build animation (first clip)
        if scene.num_animations() > 0 {
            let pick = if anim_index < scene.num_animations() {
                anim_index
            } else {
                0
            };
            if let Some(anim) = scene.animation(pick) {
                let mut channels = Vec::new();
                for ch in anim.channels() {
                    let name = ch.node_name();
                    if let Some(&node_index) = model.name_to_index.get(&name) {
                        let mut pos_keys: Vec<(f64, glam::Vec3)> = ch
                            .position_keys()
                            .into_iter()
                            .map(|k| (k.time, glam::vec3(k.value.x, k.value.y, k.value.z)))
                            .collect();
                        let mut rot_keys: Vec<(f64, glam::Quat)> = ch
                            .rotation_keys()
                            .into_iter()
                            .map(|k| {
                                (
                                    k.time,
                                    glam::Quat::from_xyzw(
                                        k.value.x, k.value.y, k.value.z, k.value.w,
                                    ),
                                )
                            })
                            .collect();
                        let mut scl_keys: Vec<(f64, glam::Vec3)> = ch
                            .scaling_keys()
                            .into_iter()
                            .map(|k| (k.time, glam::vec3(k.value.x, k.value.y, k.value.z)))
                            .collect();
                        // Ensure quaternion continuity (same hemisphere) to avoid flips
                        if rot_keys.len() > 1 {
                            let mut prev = rot_keys[0].1;
                            for i in 1..rot_keys.len() {
                                let mut q = rot_keys[i].1;
                                if prev.dot(q) < 0.0 {
                                    q = -q;
                                }
                                rot_keys[i].1 = q.normalize();
                                prev = q;
                            }
                        }
                        // Append loop keys at duration to avoid a visible jump on wrap
                        let duration = anim.duration();
                        if duration > 0.0 {
                            if let Some(first) = pos_keys.first().map(|v| v.1) {
                                if let Some(last) = pos_keys.last().map(|v| v.1) {
                                    if (last - first).length() > 1e-4 {
                                        pos_keys.push((duration, first));
                                    }
                                }
                            }
                            if let Some(first) = scl_keys.first().map(|v| v.1) {
                                if let Some(last) = scl_keys.last().map(|v| v.1) {
                                    if (last - first).length() > 1e-4 {
                                        scl_keys.push((duration, first));
                                    }
                                }
                            }
                            if let Some(first) = rot_keys.first().map(|v| v.1) {
                                if let Some(last) = rot_keys.last().map(|v| v.1) {
                                    let mut loop_q = first;
                                    if last.dot(loop_q) < 0.0 {
                                        loop_q = -loop_q;
                                    }
                                    if last.dot(loop_q) < 0.999 {
                                        // not almost equal
                                        rot_keys.push((duration, loop_q.normalize()));
                                    }
                                }
                            }
                        }
                        channels.push(crate::model::AnimChannel {
                            node_index,
                            position_keys: pos_keys,
                            rotation_keys: rot_keys,
                            scaling_keys: scl_keys,
                        });
                    }
                }
                model.animation = Some(crate::model::AnimationClip {
                    duration: anim.duration(),
                    ticks_per_second: anim.ticks_per_second(),
                    channels,
                });
            }
        }

        // Precompute bind-pose globals and root info
        let mut globals: Vec<glam::Mat4> = vec![glam::Mat4::IDENTITY; model.nodes.len()];
        let mut root_index: usize = 0;
        for (i, n) in model.nodes.iter().enumerate() {
            if n.parent.is_none() {
                root_index = i;
                break;
            }
        }
        for i in 0..model.nodes.len() {
            if let Some(p) = model.nodes[i].parent {
                globals[i] = globals[p] * model.nodes[i].local_bind;
            } else {
                globals[i] = model.nodes[i].local_bind;
            }
        }
        model.root_index = root_index;
        model.root_bind_global = globals
            .get(root_index)
            .cloned()
            .unwrap_or(glam::Mat4::IDENTITY);
        model.global_inverse_root_bind = model.root_bind_global.inverse();
        model.root_current_global = model.root_bind_global;

        if let Some(root_node) = scene.root_node() {
            let root_transform = glam::Mat4::IDENTITY;
            Self::process_node(
                &mut model,
                device,
                queue,
                &root_node,
                &scene,
                path,
                material_bind_group_layout,
                &globals,
                root_transform,
            )
            .await?;
        }

        // Compute global normalization transform (unit cube at origin) from bind-pose meshes
        if !model.meshes.is_empty() {
            let mut min_v = glam::Vec3::splat(f32::INFINITY);
            let mut max_v = glam::Vec3::splat(f32::NEG_INFINITY);
            for mesh in &model.meshes {
                let m = mesh.model_matrix; // bind-pose global for this mesh
                for v in &mesh.vertices {
                    let p = glam::Vec3::new(v.position[0], v.position[1], v.position[2]);
                    let wp = (m * glam::Vec4::new(p.x, p.y, p.z, 1.0)).truncate();
                    min_v = min_v.min(wp);
                    max_v = max_v.max(wp);
                }
            }
            let center = (min_v + max_v) * 0.5;
            let extent = max_v - min_v;
            let max_extent = extent.x.max(extent.y.max(extent.z)).max(1e-5);
            let scale = 1.0 / max_extent; // fit in unit cube
            let t = glam::Mat4::from_translation(-center);
            let s = glam::Mat4::from_scale(glam::Vec3::splat(scale));
            model.global_normalize = s * t;
            log::info!(
                "Normalize: center=({:.3},{:.3},{:.3}) extent=({:.3},{:.3},{:.3}) scale={:.5}",
                center.x,
                center.y,
                center.z,
                extent.x,
                extent.y,
                extent.z,
                scale
            );
        }
        // Collect lights from scene (Assimp lights)
        let mut lights: Vec<crate::model::LightInfo> = Vec::new();
        for i in 0..scene.num_lights() {
            if let Some(l) = scene.light(i) {
                use asset_importer::light::LightType;
                let kind = match l.light_type() {
                    LightType::Directional => crate::model::LightKind::Directional,
                    LightType::Point => crate::model::LightKind::Point,
                    LightType::Spot => crate::model::LightKind::Spot,
                    _ => crate::model::LightKind::Point,
                };
                let pos = l.position();
                let dir = l.direction();
                let col = l.color_diffuse();
                let inner = l.angle_inner_cone();
                let outer = l.angle_outer_cone();
                let range = if l.attenuation_quadratic() > 0.0 {
                    // Rough heuristic to map attenuation to a reasonable range
                    (1.0 / l.attenuation_quadratic()).sqrt()
                } else {
                    -1.0
                };
                lights.push(crate::model::LightInfo {
                    kind,
                    position: [pos.x, pos.y, pos.z],
                    direction: [dir.x, dir.y, dir.z],
                    color: [col.x, col.y, col.z],
                    intensity: 1.0,
                    range,
                    inner_cos: inner.cos(),
                    outer_cos: outer.cos(),
                });
            }
        }
        // Fallback default light if none
        if lights.is_empty() {
            lights.push(crate::model::LightInfo {
                kind: crate::model::LightKind::Directional,
                position: [2.0, 2.0, 2.0],
                direction: [-1.0, -1.0, -1.0],
                color: [1.0, 1.0, 1.0],
                intensity: 1.0,
                range: -1.0,
                inner_cos: 0.9,
                outer_cos: 0.75,
            });
        }
        model.lights = lights;

        log::info!(
            "Model processing complete. Total meshes: {}",
            model.meshes.len()
        );
        Ok(model)
    }

    fn process_node<'a>(
        model: &'a mut Model,
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
        node: &'a asset_importer::node::Node,
        scene: &'a asset_importer::scene::Scene,
        base_path: &'a str,
        material_bind_group_layout: &'a wgpu::BindGroupLayout,
        bind_globals: &'a [glam::Mat4],
        parent_transform: glam::Mat4,
    ) -> futures::future::BoxFuture<'a, Result<(), Box<dyn std::error::Error>>> {
        Box::pin(async move {
            let local = node.transformation();
            let world = parent_transform * local;
            // current node index in hierarchy
            let node_name = node.name();
            let mesh_node_index = *model.name_to_index.get(&node_name).unwrap_or(&0usize);
            // Process all meshes in this node
            for mesh_index in node.mesh_indices() {
                if let Some(mesh) = scene.mesh(mesh_index) {
                    let processed_mesh = Self::process_mesh(
                        device,
                        queue,
                        &mesh,
                        scene,
                        base_path,
                        material_bind_group_layout,
                        model,
                        bind_globals,
                        mesh_node_index,
                        world,
                    )
                    .await?;
                    model.meshes.push(processed_mesh);
                }
            }

            // Process all child nodes
            for child in node.children() {
                Self::process_node(
                    model,
                    device,
                    queue,
                    &child,
                    scene,
                    base_path,
                    material_bind_group_layout,
                    bind_globals,
                    world,
                )
                .await?;
            }

            Ok(())
        })
    }

    async fn process_mesh(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mesh: &asset_importer::mesh::Mesh,
        scene: &asset_importer::scene::Scene,
        base_path: &str,
        material_bind_group_layout: &wgpu::BindGroupLayout,
        model: &Model,
        bind_globals: &[glam::Mat4],
        mesh_node_index: usize,
        world_transform: glam::Mat4,
    ) -> Result<Mesh, Box<dyn std::error::Error>> {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        // Process vertices
        let positions = mesh.vertices();
        let normals = mesh.normals().unwrap_or_default();
        let tex_coords = mesh.texture_coords(0).unwrap_or_default();
        let tangents = mesh.tangents().unwrap_or_default();
        let bitangents = mesh.bitangents().unwrap_or_default();
        let tex_coords1 = mesh.texture_coords(1).unwrap_or_default();

        for i in 0..positions.len() {
            // Compute tangent with handedness (w)
            let (tx, ty, tz, tw) =
                if i < tangents.len() && i < bitangents.len() && i < normals.len() {
                    let t = tangents[i];
                    let b = bitangents[i];
                    let n = normals[i];
                    let t3 = glam::Vec3::new(t.x, t.y, t.z).normalize_or_zero();
                    let n3 = glam::Vec3::new(n.x, n.y, n.z).normalize_or_zero();
                    let b3 = glam::Vec3::new(b.x, b.y, b.z);
                    let w = if n3.cross(t3).dot(b3) < 0.0 {
                        -1.0
                    } else {
                        1.0
                    };
                    (t3.x, t3.y, t3.z, w)
                } else {
                    (1.0, 0.0, 0.0, 1.0)
                };
            let vertex = Vertex {
                position: [positions[i].x, positions[i].y, positions[i].z],
                normal: if i < normals.len() {
                    [normals[i].x, normals[i].y, normals[i].z]
                } else {
                    [0.0, 1.0, 0.0]
                },
                tex_coords: if i < tex_coords.len() {
                    [tex_coords[i].x, tex_coords[i].y]
                } else {
                    [0.0, 0.0]
                },
                tangent: [tx, ty, tz, tw],
                tex_coords1: if i < tex_coords1.len() {
                    [tex_coords1[i].x, tex_coords1[i].y]
                } else {
                    [0.0, 0.0]
                },
                color: if let Some(cols) = mesh.vertex_colors(0) {
                    if i < cols.len() {
                        [cols[i].x, cols[i].y, cols[i].z, cols[i].w]
                    } else {
                        [1.0, 1.0, 1.0, 1.0]
                    }
                } else {
                    [1.0, 1.0, 1.0, 1.0]
                },
                joints: [0, 0, 0, 0],
                weights: [0.0, 0.0, 0.0, 0.0],
            };
            vertices.push(vertex);
        }

        // Debug: Print first few vertices to check for issues

        for (i, vertex) in vertices.iter().enumerate().take(5) {
            log::info!(
                "  Vertex {}: pos=[{:.3}, {:.3}, {:.3}] normal=[{:.3}, {:.3}, {:.3}]",
                i,
                vertex.position[0],
                vertex.position[1],
                vertex.position[2],
                vertex.normal[0],
                vertex.normal[1],
                vertex.normal[2]
            );
        }

        // Process indices
        for face in mesh.faces() {
            for &index in face.indices() {
                indices.push(index);
            }
        }

        // Skinning data (build before material so we can bind buffer in material group)
        let mut bone_count: u32 = 0;
        let mut bone_node_indices: Vec<usize> = Vec::new();
        let mut bone_offset_mats: Vec<glam::Mat4> = Vec::new();
        let mut skin_buffer: Option<wgpu::Buffer> = None;
        if mesh.has_bones() {
            log::info!(
                "🦴 Processing skinned mesh with {} bones",
                mesh.bones().count()
            );
            let mut per_vertex: Vec<Vec<(usize, f32)>> = vec![Vec::new(); vertices.len()];
            for (bi, b) in mesh.bones().enumerate() {
                let name = b.name();
                let offset = b.offset_matrix().transpose();
                // Transpose matrices from importer (row-major) to our column-major convention
                bone_offset_mats.push(offset);

                log::debug!("  Bone ...");
                log::info!(
                    "    [{:8.4}, {:8.4}, {:8.4}, {:8.4}]",
                    offset.x_axis.x,
                    offset.x_axis.y,
                    offset.x_axis.z,
                    offset.x_axis.w
                );
                log::info!(
                    "    [{:8.4}, {:8.4}, {:8.4}, {:8.4}]",
                    offset.y_axis.x,
                    offset.y_axis.y,
                    offset.y_axis.z,
                    offset.y_axis.w
                );
                log::info!(
                    "    [{:8.4}, {:8.4}, {:8.4}, {:8.4}]",
                    offset.z_axis.x,
                    offset.z_axis.y,
                    offset.z_axis.z,
                    offset.z_axis.w
                );
                log::info!(
                    "    [{:8.4}, {:8.4}, {:8.4}, {:8.4}]",
                    offset.w_axis.x,
                    offset.w_axis.y,
                    offset.w_axis.z,
                    offset.w_axis.w
                );

                if let Some(&ni) = model.name_to_index.get(&name) {
                    bone_node_indices.push(ni);
                    log::debug!("    -> Node index logged");
                } else {
                    bone_node_indices.push(0);
                    log::warn!(
                        "    -> Bone '{}' not found in node hierarchy, using root (0)",
                        name
                    );
                }

                let weight_count = b.weights().len();
                log::info!("    -> {} vertex weights", weight_count);
                for w in b.weights() {
                    let vid = w.vertex_id as usize;
                    if vid < per_vertex.len() {
                        per_vertex[vid].push((bi, w.weight));
                    }
                }
            }
            // Normalize weights and assign to vertices
            log::debug!("Assigning bone weights");
            let mut vertices_with_weights = 0;
            let mut debug_logged = 0;
            for (i, list) in per_vertex.iter_mut().enumerate() {
                if list.is_empty() {
                    continue;
                }
                list.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let mut joints = [0u32; 4];
                let mut weights = [0.0f32; 4];
                let limit = list.len().min(4);
                let mut sum = 0.0f32;
                for j in 0..limit {
                    joints[j] = list[j].0 as u32;
                    weights[j] = list[j].1;
                    sum += list[j].1;
                }
                if sum > 0.0 {
                    for j in 0..limit {
                        weights[j] /= sum;
                    }
                    vertices_with_weights += 1;
                    if debug_logged < 10 {
                        // Log first 10 vertices with weights for debugging
                        debug_logged += 1;
                        log::info!(
                            "  ✅ Vertex {}: joints=[{}, {}, {}, {}] weights=[{:.3}, {:.3}, {:.3}, {:.3}] (sum={:.3})",
                            i,
                            joints[0],
                            joints[1],
                            joints[2],
                            joints[3],
                            weights[0],
                            weights[1],
                            weights[2],
                            weights[3],
                            weights.iter().sum::<f32>()
                        );
                    }
                }
                vertices[i].joints = joints;
                vertices[i].weights = weights;
            }
            log::info!("  -> {} vertices have bone weights", vertices_with_weights);
            bone_count = bone_node_indices.len() as u32;
            // Initialize skin buffer with bind-pose matrices using LearnOpenGL formula
            let mut mats: Vec<[[f32; 4]; 4]> = Vec::with_capacity(bone_node_indices.len());
            log::info!(
                "🔧 Computing initial bone matrices for {} bones",
                bone_count
            );

            // Mesh-global at bind pose, so we can compute mesh-local skin matrices
            let mesh_bind_global = bind_globals
                .get(mesh_node_index)
                .cloned()
                .unwrap_or(glam::Mat4::IDENTITY);
            let inv_mesh_bind_global = mesh_bind_global.inverse();

            for (bi, &ni) in bone_node_indices.iter().enumerate() {
                let global_transform = bind_globals
                    .get(ni)
                    .cloned()
                    .unwrap_or(glam::Mat4::IDENTITY);
                let offset_matrix = bone_offset_mats[bi];
                // Skin matrix in mesh-local coordinates: inv(mesh_bind_global) * global(bone) * offset
                let final_bone_matrix = inv_mesh_bind_global * global_transform * offset_matrix;

                mats.push(final_bone_matrix.to_cols_array_2d());
            }
            let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("skin_buffer"),
                contents: bytemuck::cast_slice(&mats),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            skin_buffer = Some(buf);
        }

        // Load material textures and create bind group
        let mut textures: Vec<crate::render::Texture> = Vec::new();
        let mut material_bind_group: Option<wgpu::BindGroup> = None;
        let mut material_params_buffer: Option<wgpu::Buffer> = None;

        let mut opaque_transparent = false;
        let mut material_params_cpu: Option<crate::render::pipeline::MaterialParams> = None;
        let mut material_debug: Option<MaterialDebug> = None;
        if let Some(material) = scene.material(mesh.material_index()) {
            let loaded =
                Self::load_pbr_textures(device, queue, &material, base_path, scene).await?;

            // Create defaults if missing, keep strong ownership in `textures`
            let base_color = loaded.base_color.unwrap_or_else(|| {
                crate::render::Texture::from_color(
                    device,
                    queue,
                    [255, 255, 255, 255],
                    true,
                    Some("default_base_color"),
                )
            });
            let normal = loaded.normal.unwrap_or_else(|| {
                crate::render::Texture::from_color(
                    device,
                    queue,
                    [128, 128, 255, 255],
                    false,
                    Some("default_normal"),
                )
            });
            let mra = loaded.metallic_roughness.unwrap_or_else(|| {
                crate::render::Texture::from_color(
                    device,
                    queue,
                    [0, 255, 0, 255],
                    false,
                    Some("default_mr"),
                )
            });
            let occlusion = loaded.occlusion.unwrap_or_else(|| {
                crate::render::Texture::from_color(
                    device,
                    queue,
                    [255, 255, 255, 255],
                    false,
                    Some("default_ao"),
                )
            });
            let emissive = loaded.emissive.unwrap_or_else(|| {
                crate::render::Texture::from_color(
                    device,
                    queue,
                    [0, 0, 0, 255],
                    true,
                    Some("default_emissive"),
                )
            });

            let params = crate::render::pipeline::MaterialParams {
                base_color_factor: loaded.base_color_factor,
                emissive_occlusion: [
                    loaded.emissive_factor[0],
                    loaded.emissive_factor[1],
                    loaded.emissive_factor[2],
                    loaded.occlusion_strength,
                ],
                mr_factors: [
                    loaded.metallic_factor,
                    loaded.roughness_factor,
                    loaded.normal_scale,
                    loaded.alpha_cutoff,
                ],
                uv_indices: [
                    loaded.ao_uv_index,
                    loaded.base_uv_index,
                    loaded.normal_uv_index,
                    loaded.mr_uv_index,
                ],
                misc: [loaded.emissive_uv_index, loaded.alpha_mode, 0, 0],
                base_uv_transform: loaded.base_uv_transform.to_cols_array_2d(),
                normal_uv_transform: loaded.normal_uv_transform.to_cols_array_2d(),
                mr_uv_transform: loaded.mr_uv_transform.to_cols_array_2d(),
                emissive_uv_transform: loaded.emissive_uv_transform.to_cols_array_2d(),
                ao_uv_transform: loaded.ao_uv_transform.to_cols_array_2d(),
            };
            let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("material_params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
            // Heuristic: BLEND alpha mode but base alpha near 1.0 -> treat as opaque-transparent
            if loaded.alpha_mode == 2u32 && params.base_color_factor[3] >= 0.98 {
                opaque_transparent = true;
            }

            // Prepare skin buffer binding (kept alive at least during this call)
            let mut fallback_skin: Option<wgpu::Buffer> = None;
            let skin_binding = match &skin_buffer {
                Some(buf) => buf.as_entire_binding(),
                None => {
                    let m = glam::Mat4::IDENTITY.to_cols_array_2d();
                    let fb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("skin_buffer_fallback"),
                        contents: bytemuck::cast_slice(&[m]),
                        usage: wgpu::BufferUsages::STORAGE,
                    });
                    fallback_skin = Some(fb);
                    fallback_skin.as_ref().unwrap().as_entire_binding()
                }
            };

            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("material_bind_group"),
                layout: material_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&base_color.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&base_color.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&normal.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&normal.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(&mra.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::Sampler(&mra.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::TextureView(&occlusion.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: wgpu::BindingResource::Sampler(&occlusion.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: wgpu::BindingResource::TextureView(&emissive.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: wgpu::BindingResource::Sampler(&emissive.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 11,
                        resource: skin_binding,
                    },
                ],
            });

            material_bind_group = Some(bg);
            material_params_buffer = Some(params_buffer);
            material_params_cpu = Some(params);
            material_debug = Some(MaterialDebug {
                base: "default".to_string(),
                normal: "default".to_string(),
                mr: "default".to_string(),
                ao: "default".to_string(),
                emissive: "default".to_string(),
            });

            textures.push(emissive);
        } else {
            // Create a simple default material
            let base_color = crate::render::Texture::from_color(
                device,
                queue,
                [255, 255, 255, 255],
                true,
                Some("default_base_color"),
            );
            let normal = crate::render::Texture::from_color(
                device,
                queue,
                [128, 128, 255, 255],
                false,
                Some("default_normal"),
            );
            let mra = crate::render::Texture::from_color(
                device,
                queue,
                [0, 255, 0, 255],
                false,
                Some("default_mr"),
            );
            let occlusion = crate::render::Texture::from_color(
                device,
                queue,
                [255, 255, 255, 255],
                false,
                Some("default_ao"),
            );
            let emissive = crate::render::Texture::from_color(
                device,
                queue,
                [0, 0, 0, 255],
                true,
                Some("default_emissive"),
            );
            let params = crate::render::pipeline::MaterialParams::default();
            let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("material_params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

            // Prepare skin buffer binding for default-material path
            let mut fallback_skin2: Option<wgpu::Buffer> = None;
            let skin_binding2 = match &skin_buffer {
                Some(buf) => buf.as_entire_binding(),
                None => {
                    let m = glam::Mat4::IDENTITY.to_cols_array_2d();
                    let fb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("skin_buffer_fallback"),
                        contents: bytemuck::cast_slice(&[m]),
                        usage: wgpu::BufferUsages::STORAGE,
                    });
                    fallback_skin2 = Some(fb);
                    fallback_skin2.as_ref().unwrap().as_entire_binding()
                }
            };

            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("material_bind_group"),
                layout: material_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&base_color.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&base_color.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&normal.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&normal.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(&mra.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::Sampler(&mra.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::TextureView(&occlusion.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: wgpu::BindingResource::Sampler(&occlusion.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: wgpu::BindingResource::TextureView(&emissive.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: wgpu::BindingResource::Sampler(&emissive.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 11,
                        resource: skin_binding2,
                    },
                ],
            });

            material_bind_group = Some(bg);
            material_params_buffer = Some(params_buffer);
            material_params_cpu = Some(params);
            material_debug = Some(MaterialDebug {
                base: "default".to_string(),
                normal: "default".to_string(),
                mr: "default".to_string(),
                ao: "default".to_string(),
                emissive: "default".to_string(),
            });

            textures.push(emissive);
        }

        Mesh::new(
            device,
            material_bind_group_layout,
            vertices,
            indices,
            textures,
            material_bind_group,
            material_params_buffer,
            material_params_cpu,
            material_debug,
            world_transform,
            // material flags
            scene
                .material(mesh.material_index())
                .map(|m| m.is_two_sided())
                .unwrap_or(false),
            scene
                .material(mesh.material_index())
                .and_then(|m| m.blend_mode()),
            opaque_transparent,
            bone_count,
            bone_node_indices,
            bone_offset_mats,
            skin_buffer,
            mesh_node_index,
        )
    }

    async fn load_pbr_textures(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        material: &asset_importer::material::Material,
        base_path: &str,
        scene: &asset_importer::scene::Scene,
    ) -> Result<LoadedPbr, Box<dyn std::error::Error>> {
        // helper: map modes and transforms and create textures with custom samplers
        fn map_mode(m: asset_importer::material::TextureMapMode) -> wgpu::AddressMode {
            use asset_importer::material::TextureMapMode as M;
            match m {
                M::Wrap => wgpu::AddressMode::Repeat,
                M::Clamp => wgpu::AddressMode::ClampToEdge,
                M::Mirror => wgpu::AddressMode::MirrorRepeat,
                M::Decal => wgpu::AddressMode::ClampToEdge,
                _ => wgpu::AddressMode::Repeat,
            }
        }
        fn uv_transform_to_mat4(tr: &Option<asset_importer::material::UVTransform>) -> glam::Mat4 {
            if let Some(t) = tr {
                let trans =
                    glam::Mat4::from_translation(glam::vec3(t.translation.x, t.translation.y, 0.0));
                let rot = glam::Mat4::from_rotation_z(t.rotation);
                let scale = glam::Mat4::from_scale(glam::vec3(t.scaling.x, t.scaling.y, 1.0));
                trans * rot * scale
            } else {
                glam::Mat4::IDENTITY
            }
        }

        let load_tex = |ty: asset_importer::material::TextureType,
                        srgb: bool|
         -> Option<(
            crate::render::Texture,
            asset_importer::material::TextureInfo,
        )> {
            let count = material.texture_count(ty);
            if count == 0 {
                return None;
            }
            let info = material.texture(ty, 0)?;
            let texture_path = info.path.clone();
            let params = crate::render::texture::TextureSamplerParams {
                address_mode_u: map_mode(info.map_modes[0]),
                address_mode_v: map_mode(info.map_modes[1]),
                address_mode_w: wgpu::AddressMode::Repeat,
                ..Default::default()
            };
            if texture_path.starts_with('*') {
                if let Some(tex) = scene.embedded_texture_by_name(&texture_path) {
                    match tex.data() {
                        Ok(asset_importer::texture::TextureData::Compressed(bytes)) => {
                            if let Ok(img) = image::load_from_memory(&bytes) {
                                if let Ok(t) = crate::render::Texture::from_image_with_format_and_sampler_params(
                                    device, queue, &img, Some("embedded_tex"), srgb, &params,
                                ) { return Some((t, info)); }
                            }
                            None
                        }
                        Ok(asset_importer::texture::TextureData::Texels(texels)) => {
                            let (w, h) = tex.dimensions();
                            let mut rgba = Vec::with_capacity((w * h * 4) as usize);
                            for t in texels {
                                rgba.push(t.r);
                                rgba.push(t.g);
                                rgba.push(t.b);
                                rgba.push(t.a);
                            }
                            if let Ok(t) = crate::render::Texture::from_rgba8_with_params(
                                device,
                                queue,
                                &rgba,
                                w,
                                h,
                                srgb,
                                &params,
                                Some("embedded_tex"),
                            ) {
                                return Some((t, info));
                            }
                            None
                        }
                        Err(e) => {
                            log::warn!("Failed to read embedded texture {}: {}", texture_path, e);
                            None
                        }
                    }
                } else {
                    None
                }
            } else {
                let full_path = if std::path::Path::new(&texture_path).is_absolute() {
                    texture_path
                } else {
                    let base_dir = std::path::Path::new(base_path)
                        .parent()
                        .unwrap_or(std::path::Path::new(""));
                    base_dir.join(&texture_path).to_string_lossy().to_string()
                };
                match image::open(&full_path) {
                    Ok(img) => {
                        match crate::render::Texture::from_image_with_format_and_sampler_params(
                            device,
                            queue,
                            &img,
                            Some(&format!("texture_{}", full_path)),
                            srgb,
                            &params,
                        ) {
                            Ok(tex) => Some((tex, info)),
                            Err(e) => {
                                log::warn!("Failed to create texture {}: {}", full_path, e);
                                None
                            }
                        }
                    }
                    Err(e) => {
                        log::warn!("Failed to load texture {}: {}", full_path, e);
                        None
                    }
                }
            }
        };

        let base_loaded = load_tex(asset_importer::material::TextureType::BaseColor, true)
            .or_else(|| load_tex(asset_importer::material::TextureType::Diffuse, true));
        let normal_loaded = load_tex(asset_importer::material::TextureType::Normals, false);
        let mr_loaded = load_tex(
            asset_importer::material::TextureType::GltfMetallicRoughness,
            false,
        )
        .or_else(|| {
            load_tex(
                asset_importer::material::TextureType::DiffuseRoughness,
                false,
            )
        });
        let occlusion_loaded = load_tex(
            asset_importer::material::TextureType::AmbientOcclusion,
            false,
        );
        let emissive_loaded = load_tex(asset_importer::material::TextureType::Emissive, true);

        let (base_color, base_info) = match base_loaded {
            Some((t, i)) => (Some(t), Some(i)),
            None => (None, None),
        };
        let (normal, normal_info) = match normal_loaded {
            Some((t, i)) => (Some(t), Some(i)),
            None => (None, None),
        };
        let (metallic_roughness, mr_info) = match mr_loaded {
            Some((t, i)) => (Some(t), Some(i)),
            None => (None, None),
        };
        let (occlusion, occlusion_info) = match occlusion_loaded {
            Some((t, i)) => (Some(t), Some(i)),
            None => (None, None),
        };
        let (emissive, emissive_info) = match emissive_loaded {
            Some((t, i)) => (Some(t), Some(i)),
            None => (None, None),
        };

        let metallic_factor = material.metallic_factor().unwrap_or(1.0);
        let roughness_factor = material.roughness_factor().unwrap_or(1.0);
        let base_color_factor = material
            .base_color()
            .map(|c| [c.x, c.y, c.z, c.w])
            .unwrap_or([1.0, 1.0, 1.0, 1.0]);
        let emissive_strength = material.emissive_intensity().unwrap_or(1.0);
        let emissive_factor = material
            .emissive_color()
            .map(|c| {
                [
                    c.x * emissive_strength,
                    c.y * emissive_strength,
                    c.z * emissive_strength,
                ]
            })
            .unwrap_or([0.0, 0.0, 0.0]);
        let occlusion_strength = 1.0;

        let ao_uv_index = occlusion_info.as_ref().map(|i| i.uv_index).unwrap_or(0);
        let base_uv_index = base_info.as_ref().map(|i| i.uv_index).unwrap_or(0);
        let normal_uv_index = normal_info.as_ref().map(|i| i.uv_index).unwrap_or(0);
        let mr_uv_index = mr_info.as_ref().map(|i| i.uv_index).unwrap_or(0);
        let emissive_uv_index = emissive_info.as_ref().map(|i| i.uv_index).unwrap_or(0);

        let base_uv_transform = base_info
            .as_ref()
            .map(|i| uv_transform_to_mat4(&i.uv_transform))
            .unwrap_or(glam::Mat4::IDENTITY);
        let normal_uv_transform = normal_info
            .as_ref()
            .map(|i| uv_transform_to_mat4(&i.uv_transform))
            .unwrap_or(glam::Mat4::IDENTITY);
        let mr_uv_transform = mr_info
            .as_ref()
            .map(|i| uv_transform_to_mat4(&i.uv_transform))
            .unwrap_or(glam::Mat4::IDENTITY);
        let emissive_uv_transform = emissive_info
            .as_ref()
            .map(|i| uv_transform_to_mat4(&i.uv_transform))
            .unwrap_or(glam::Mat4::IDENTITY);
        let ao_uv_transform = occlusion_info
            .as_ref()
            .map(|i| uv_transform_to_mat4(&i.uv_transform))
            .unwrap_or(glam::Mat4::IDENTITY);

        let normal_scale = material.bump_scaling().unwrap_or(1.0);
        let has_opacity_tex = material.opacity_texture(0).is_some();
        let alpha_mode = if has_opacity_tex {
            1u32
        } else if matches!(
            material.blend_mode(),
            Some(
                asset_importer::material::BlendMode::Default
                    | asset_importer::material::BlendMode::Additive
            )
        ) {
            2u32
        } else {
            0u32
        };
        let alpha_cutoff = 0.5f32;

        Ok(LoadedPbr {
            base_color,
            normal,
            metallic_roughness,
            occlusion,
            emissive,
            base_color_factor,
            metallic_factor,
            roughness_factor,
            emissive_factor,
            occlusion_strength,
            ao_uv_index,
            base_uv_index,
            normal_uv_index,
            mr_uv_index,
            emissive_uv_index,
            base_uv_transform,
            normal_uv_transform,
            mr_uv_transform,
            emissive_uv_transform,
            ao_uv_transform,
            normal_scale,
            alpha_mode,
            alpha_cutoff,
        })
    }

    async fn load_texture_from_file(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        path: &str,
    ) -> Result<crate::render::Texture, Box<dyn std::error::Error>> {
        log::debug!("Loading texture: {}", path);

        let img = image::open(path).map_err(|e| {
            log::warn!("Failed to load texture: {}, error: {}", path, e);
            e
        })?;
        // Default to sRGB for legacy path
        crate::render::Texture::from_image_with_format(
            device,
            queue,
            &img,
            Some(&format!("texture_{}", path)),
            true,
        )
    }
}

impl ModelLoader {
    fn collect_nodes(
        node: &asset_importer::node::Node,
        parent: Option<usize>,
        out_nodes: &mut Vec<crate::model::NodeData>,
        out_map: &mut std::collections::HashMap<String, usize>,
    ) {
        let idx = out_nodes.len();
        let nd = crate::model::NodeData {
            name: node.name(),
            parent,
            // Use importer matrix directly; it matches our expected convention for node transforms
            local_bind: node.transformation(),
        };
        out_nodes.push(nd);
        out_map.insert(node.name(), idx);
        for child in node.children() {
            Self::collect_nodes(&child, Some(idx), out_nodes, out_map);
        }
    }
}
