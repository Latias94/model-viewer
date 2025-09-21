use crate::model::{Mesh, Model, Vertex};
use asset_importer::{Importer, postprocess::PostProcessSteps};

pub struct ModelLoader;

impl ModelLoader {
    pub async fn load(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        path: &str,
        texture_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Result<Model, Box<dyn std::error::Error>> {
        log::info!("Loading model: {}", path);

        let importer = Importer::new();
        let scene = importer
            .read_file(path)
            .with_post_process(
                PostProcessSteps::TRIANGULATE
                    | PostProcessSteps::FLIP_UVS
                    | PostProcessSteps::GEN_SMOOTH_NORMALS
                    | PostProcessSteps::CALC_TANGENT_SPACE,
            )
            .import_file(path)?;

        log::info!("Scene loaded successfully!");
        log::info!("  Meshes: {}", scene.num_meshes());
        log::info!("  Materials: {}", scene.num_materials());
        log::info!("  Textures: {}", scene.num_textures());

        let mut model = Model::new();

        if let Some(root_node) = scene.root_node() {
            Self::process_node(
                &mut model,
                device,
                queue,
                &root_node,
                &scene,
                path,
                texture_bind_group_layout,
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
        texture_bind_group_layout: &'a wgpu::BindGroupLayout,
    ) -> futures::future::BoxFuture<'a, Result<(), Box<dyn std::error::Error>>> {
        Box::pin(async move {
            // Process all meshes in this node
            for mesh_index in node.mesh_indices() {
                if let Some(mesh) = scene.mesh(mesh_index) {
                    let processed_mesh = Self::process_mesh(
                        device,
                        queue,
                        &mesh,
                        scene,
                        base_path,
                        texture_bind_group_layout,
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
                    texture_bind_group_layout,
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
        texture_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Result<Mesh, Box<dyn std::error::Error>> {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        // Process vertices
        let positions = mesh.vertices();
        let normals = mesh.normals().unwrap_or_default();
        let tex_coords = mesh.texture_coords(0).unwrap_or_default();

        for i in 0..positions.len() {
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
            };
            vertices.push(vertex);
        }

        // Process indices
        for face in mesh.faces() {
            for &index in face.indices() {
                indices.push(index);
            }
        }

        // Load textures if material is available
        let textures = if let Some(material) = scene.material(mesh.material_index()) {
            Self::load_material_textures(device, queue, &material, base_path).await?
        } else {
            Vec::new()
        };

        Mesh::new(
            device,
            texture_bind_group_layout,
            vertices,
            indices,
            textures,
        )
    }

    async fn load_material_textures(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        material: &asset_importer::material::Material,
        base_path: &str,
    ) -> Result<Vec<crate::render::Texture>, Box<dyn std::error::Error>> {
        let mut textures = Vec::new();

        // Load diffuse textures
        let diffuse_count = material.texture_count(asset_importer::material::TextureType::Diffuse);
        for i in 0..diffuse_count {
            if let Some(texture_info) =
                material.texture(asset_importer::material::TextureType::Diffuse, i)
            {
                let texture_path = texture_info.path.clone();
                let full_path = if std::path::Path::new(&texture_path).is_absolute() {
                    texture_path
                } else {
                    let base_dir = std::path::Path::new(base_path)
                        .parent()
                        .unwrap_or(std::path::Path::new(""));
                    base_dir.join(&texture_path).to_string_lossy().to_string()
                };

                match Self::load_texture_from_file(device, queue, &full_path).await {
                    Ok(texture) => textures.push(texture),
                    Err(e) => log::warn!("Failed to load texture {}: {}", full_path, e),
                }
            }
        }

        Ok(textures)
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

        crate::render::Texture::from_image(device, queue, &img, Some(&format!("texture_{}", path)))
    }
}
