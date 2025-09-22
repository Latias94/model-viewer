use clap::Parser;
use model_viewer::App;
use winit::event_loop::EventLoop;

#[derive(Parser)]
#[command(name = "model-viewer")]
#[command(about = "A 3D model viewer using Rust, wgpu, winit, and assimp")]
#[command(version = "0.1.0")]
struct Args {
    /// Path to the 3D model file to load
    #[arg(value_name = "MODEL_PATH")]
    model: Option<String>,

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,

    /// Window width
    #[arg(long, default_value = "800")]
    width: u32,

    /// Window height
    #[arg(long, default_value = "600")]
    height: u32,

    /// Enable wireframe mode
    #[arg(short, long)]
    wireframe: bool,

    /// Background color (hex format, e.g., #000000)
    #[arg(long, default_value = "#202020")]
    background: String,

    /// Disable animation playback
    #[arg(long)]
    no_anim: bool,

    /// Animation index (0-based)
    #[arg(long, default_value_t = 0)]
    anim_index: usize,

    /// Animation speed multiplier
    #[arg(long, default_value_t = 1.0)]
    anim_speed: f32,

    /// Exposure for tone mapping
    #[arg(long, default_value_t = 1.0)]
    exposure: f32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Initialize logging
    // - By default, keep our app at Info/Debug, but silence noisy GPU deps to Warn.
    // - If RUST_LOG is set, respect it entirely (no overrides here).
    let rust_log_set = std::env::var("RUST_LOG").is_ok();
    let mut builder = env_logger::Builder::from_default_env();
    if !rust_log_set {
        // App-wide default level
        if args.verbose {
            builder.filter_level(log::LevelFilter::Debug);
        } else {
            builder.filter_level(log::LevelFilter::Info);
        }

        // Turn down wgpu and related crates by default
        builder
            .filter_module("wgpu", log::LevelFilter::Warn)
            .filter_module("wgpu_core", log::LevelFilter::Warn)
            .filter_module("wgpu_hal", log::LevelFilter::Warn)
            .filter_module("naga", log::LevelFilter::Warn);
    }
    builder.init();

    log::info!("🎮 Model Viewer Starting");

    if let Some(ref model_path) = args.model {
        log::info!("📁 Loading model: {}", model_path);

        // Check if model file exists
        if !std::path::Path::new(model_path).exists() {
            log::error!("❌ Model file '{}' does not exist", model_path);
            std::process::exit(1);
        }
    } else {
        log::info!("🎯 No model specified, starting with empty scene");
    }

    log::info!("🎯 Controls:");
    log::info!("   WASD - Move camera");
    log::info!("   Mouse - Look around");
    log::info!("   Mouse wheel - Zoom");
    log::info!("   ESC - Exit");

    // Create event loop and application
    let event_loop = EventLoop::new()?;
    let mut app = App::new(
        args.model,
        args.background,
        (args.width, args.height),
        !args.no_anim,
        args.anim_index,
        args.anim_speed,
        args.exposure,
    );

    // Run the application
    event_loop.run_app(&mut app)?;

    log::info!("👋 Model Viewer Exiting");
    Ok(())
}
