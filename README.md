# Model Viewer (asset-importer + wgpu)

This is a small model viewer built on top of the Assimp Rust binding [asset-importer](https://github.com/Latias94/asset-importer). It serves as a practical sample and a testbed to validate asset-importer across common formats.

![screenshot](screenshots/helmet.png)

## Highlights

- WGPU + Winit + EGUI (wgpu v25, winit v0.30, egui v0.32)
- Assimp-driven loading via asset-importer v0.4 (external and embedded textures)
- Physically-Based Rendering (metallic-roughness)
  - Base color, normal, metallic-roughness, occlusion, emissive
  - UV0/UV1, per-texture UV transforms, vertex colors
- Image-Based Lighting (IBL)
  - Generates irradiance, prefiltered environment (mips), and BRDF LUT from HDR
  - Cubemap skybox rendered from the environment
- Lights: multiple punctual lights imported from the scene
- Multi-pass forward renderer
  - Skybox → Opaque → Two-Sided → OpaqueTransparent → Transparent
  - Basic transparent sorting; alpha-mask and blend paths
- Debug views and GUI panel (output channels, material info, basic controls)

## Build & Run

```bash
cargo run --bin download-assets
cargo run --bin model-viewer -- "assets/glTF-Sample-Assets-main/Models/DamagedHelmet/glTF/DamagedHelmet.gltf"
```

Point it at any supported model file; the glTF-Sample-Assets Helmet is a good sanity check.

Notes:
- The downloader fetches both the glTF Sample Assets and a default IBL HDRI (venice_sunset_1k.exr) into `assets/ibl/` for PBR/IBL rendering.
- You can replace the EXR with your own HDR/EXR in `assets/ibl/`.

## CLI Options

Run `model-viewer -h` for the full list. Common flags:

- `--width <u32>` / `--height <u32>`: Window size (default 800x600)
- `--background <#RRGGBB>`: Clear color (default `#202020`)
- `--verbose`: Enable debug logging

Rendering:
- `--exposure <f32>`: Exposure for ACES tone mapping (default 1.0)

Animation:
- `--no-anim`: Disable animation playback
- `--anim-index <usize>`: Select which animation to play (0-based, default 0)
- `--anim-speed <f32>`: Playback speed multiplier (default 1.0)

Examples:

```bash
# Play animation 1 at 2x speed
cargo run --bin model-viewer -- "assets/glTF-Sample-Assets-main/Models/Fox/glTF/Fox.gltf" \
  --anim-index 1 --anim-speed 2.0 --exposure 1.25

# Disable animation
cargo run --bin model-viewer -- "assets/glTF-Sample-Assets-main/Models/CesiumMan/glTF/CesiumMan.gltf" --no-anim
```

## Notes

- This project is focused on feature coverage and correctness as a companion to asset-importer, not raw performance.
- Tested primarily on desktop (wgpu Vulkan/DirectX backends).
