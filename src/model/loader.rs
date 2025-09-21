use crate::model::{Mesh, Model, Vertex};
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
}

impl ModelLoader {
    pub async fn load(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        path: &str,
        material_bind_group_layout: &wgpu::BindGroupLayout,
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

        log::info!("Scene loaded successfully!");
        log::info!("  Meshes: {}", scene.num_meshes());
        log::info!("  Materials: {}", scene.num_materials());
        log::info!("  Textures: {}", scene.num_textures());

        let mut model = Model::new();

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
                root_transform,
            )
            .await?;
        }

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
        parent_transform: glam::Mat4,
    ) -> futures::future::BoxFuture<'a, Result<(), Box<dyn std::error::Error>>> {
        Box::pin(async move {
            let local = node.transformation();
            let world = parent_transform * local;
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
            };
            vertices.push(vertex);
        }

        // Process indices
        for face in mesh.faces() {
            for &index in face.indices() {
                indices.push(index);
            }
        }

        // Load material textures and create bind group
        let mut textures: Vec<crate::render::Texture> = Vec::new();
        let mut material_bind_group: Option<wgpu::BindGroup> = None;
        let mut material_params_buffer: Option<wgpu::Buffer> = None;

        if let Some(material) = scene.material(mesh.material_index()) {
            let loaded = Self::load_pbr_textures(device, queue, &material, base_path).await?;

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
                emissive_factor: loaded.emissive_factor,
                occlusion_strength: loaded.occlusion_strength,
                metallic_factor: loaded.metallic_factor,
                roughness_factor: loaded.roughness_factor,
                _pad: [0.0; 2],
            };
            let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("material_params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

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
                ],
            });

            material_bind_group = Some(bg);
            material_params_buffer = Some(params_buffer);

            // push strong references into mesh textures vector
            textures.push(base_color);
            textures.push(normal);
            textures.push(mra);
            textures.push(occlusion);
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
                ],
            });

            material_bind_group = Some(bg);
            material_params_buffer = Some(params_buffer);
            textures.push(base_color);
            textures.push(normal);
            textures.push(mra);
            textures.push(occlusion);
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
            world_transform,
        )
    }

    async fn load_pbr_textures(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        material: &asset_importer::material::Material,
        base_path: &str,
    ) -> Result<LoadedPbr, Box<dyn std::error::Error>> {
        // helper to resolve path and load with correct sRGB setting
        let load_tex = |ty: asset_importer::material::TextureType,
                        srgb: bool|
         -> Option<crate::render::Texture> {
            let count = material.texture_count(ty);
            if count == 0 {
                return None;
            }
            let info = material.texture(ty, 0)?;
            let texture_path = info.path.clone();
            let full_path = if std::path::Path::new(&texture_path).is_absolute() {
                texture_path
            } else {
                let base_dir = std::path::Path::new(base_path)
                    .parent()
                    .unwrap_or(std::path::Path::new(""));
                base_dir.join(&texture_path).to_string_lossy().to_string()
            };
            match image::open(&full_path) {
                Ok(img) => match crate::render::Texture::from_image_with_format(
                    device,
                    queue,
                    &img,
                    Some(&format!("texture_{}", full_path)),
                    srgb,
                ) {
                    Ok(t) => Some(t),
                    Err(e) => {
                        log::warn!("Failed to create texture {}: {}", full_path, e);
                        None
                    }
                },
                Err(e) => {
                    log::warn!("Failed to load texture {}: {}", full_path, e);
                    None
                }
            }
        };

        let base_color = load_tex(asset_importer::material::TextureType::BaseColor, true)
            .or_else(|| load_tex(asset_importer::material::TextureType::Diffuse, true));
        let normal = load_tex(asset_importer::material::TextureType::Normals, false);
        let metallic_roughness = load_tex(
            asset_importer::material::TextureType::GltfMetallicRoughness,
            false,
        )
        .or_else(|| {
            load_tex(
                asset_importer::material::TextureType::DiffuseRoughness,
                false,
            )
        });
        let occlusion = load_tex(
            asset_importer::material::TextureType::AmbientOcclusion,
            false,
        );
        let emissive = load_tex(asset_importer::material::TextureType::Emissive, true);

        let metallic_factor = material.metallic_factor().unwrap_or(1.0);
        let roughness_factor = material.roughness_factor().unwrap_or(1.0);
        // base_color_factor & emissive_factor accessors not found; default values
        let base_color_factor = [1.0, 1.0, 1.0, 1.0];
        let emissive_factor = [0.0, 0.0, 0.0];
        let occlusion_strength = 1.0;

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
