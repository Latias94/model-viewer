use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

const GLTF_SAMPLE_ASSETS_URL: &str =
    "https://github.com/KhronosGroup/glTF-Sample-Assets/archive/refs/heads/main.zip";
// CC0 HDRI from Poly Haven (1k, EXR)
const IBL_EXR_URL: &str =
    "https://dl.polyhaven.org/file/ph-assets/HDRIs/exr/1k/venice_sunset_1k.exr";

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    println!("🚀 Lumia Asset Downloader");
    println!("Downloading glTF Sample Assets from KhronosGroup...");

    let assets_dir = get_assets_directory()?;
    let zip_path = assets_dir.join("gltf-sample-assets.zip");
    let extract_dir = assets_dir.join("glTF-Sample-Assets-main");
    let ibl_dir = assets_dir.join("ibl");
    let ibl_exr_path = ibl_dir.join("venice_sunset_1k.exr");

    // Create assets directory if it doesn't exist
    fs::create_dir_all(&assets_dir).context("Failed to create assets directory")?;

    // Ensure IBL dir exists and environment present (skip if already there)
    std::fs::create_dir_all(&ibl_dir).ok();
    if !ibl_exr_path.exists() {
        println!(
            "☀️  Downloading IBL (venice_sunset_1k.exr) to {}",
            ibl_exr_path.display()
        );
        download_with_resume_to(&ibl_exr_path, IBL_EXR_URL).await?;
        println!("✅ IBL saved: {}", ibl_exr_path.display());
    } else {
        println!("✅ IBL already exists: {}", ibl_exr_path.display());
    }

    // Check if already extracted
    if extract_dir.exists() {
        println!(
            "✅ glTF Sample Assets already exist at: {}",
            extract_dir.display()
        );
        println!("Delete the directory to re-download.");
        return Ok(());
    }

    // Download with resume support
    download_with_resume(&zip_path).await?;

    // Extract the zip file
    println!("📦 Extracting assets...");
    extract_zip(&zip_path, &assets_dir)?;

    // Clean up zip file
    if zip_path.exists() {
        fs::remove_file(&zip_path).context("Failed to remove zip file")?;
        println!("🧹 Cleaned up zip file");
    }

    println!("✅ Successfully downloaded and extracted glTF Sample Assets!");
    println!("📁 Assets location: {}", extract_dir.display());

    Ok(())
}

fn get_assets_directory() -> Result<PathBuf> {
    let current_dir = std::env::current_dir().context("Failed to get current directory")?;

    // Try to find the project root by looking for Cargo.toml
    let mut path = current_dir.clone();
    loop {
        if path.join("Cargo.toml").exists() {
            return Ok(path.join("assets"));
        }

        if let Some(parent) = path.parent() {
            path = parent.to_path_buf();
        } else {
            break;
        }
    }

    // Fallback to current directory + assets
    Ok(current_dir.join("assets"))
}

async fn download_with_resume(zip_path: &Path) -> Result<()> {
    const MAX_RETRIES: usize = 3;
    const RETRY_DELAY_SECS: u64 = 2;

    for attempt in 1..=MAX_RETRIES {
        match download_attempt(zip_path, attempt).await {
            Ok(()) => return Ok(()),
            Err(e) if attempt < MAX_RETRIES => {
                println!("⚠️  Download attempt {} failed: {}", attempt, e);
                println!("🔄 Retrying in {} seconds...", RETRY_DELAY_SECS);
                tokio::time::sleep(tokio::time::Duration::from_secs(RETRY_DELAY_SECS)).await;
            }
            Err(e) => return Err(e),
        }
    }

    unreachable!()
}

async fn download_with_resume_to(dest_path: &Path, url: &str) -> Result<()> {
    const MAX_RETRIES: usize = 3;
    const RETRY_DELAY_SECS: u64 = 2;

    for attempt in 1..=MAX_RETRIES {
        match download_attempt_to(dest_path, url, attempt).await {
            Ok(()) => return Ok(()),
            Err(e) if attempt < MAX_RETRIES => {
                println!("⚠️  Download attempt {} failed: {}", attempt, e);
                println!("🔄 Retrying in {} seconds...", RETRY_DELAY_SECS);
                tokio::time::sleep(tokio::time::Duration::from_secs(RETRY_DELAY_SECS)).await;
            }
            Err(e) => return Err(e),
        }
    }
    unreachable!()
}

async fn download_attempt_to(dest_path: &Path, url: &str, attempt: usize) -> Result<()> {
    let client = Client::builder()
        .timeout(tokio::time::Duration::from_secs(300))
        .build()
        .context("Failed to create HTTP client")?;

    let mut start_pos = 0u64;
    if dest_path.exists() {
        start_pos = dest_path
            .metadata()
            .context("Failed to get file metadata")?
            .len();
        if attempt == 1 {
            println!(
                "📄 Found partial download, resuming from byte {}",
                start_pos
            );
        }
    }

    let head_response = client
        .head(url)
        .send()
        .await
        .context("Failed to get file info")?;
    let total_size = head_response
        .headers()
        .get("content-length")
        .and_then(|h| h.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0);

    if start_pos >= total_size && total_size > 0 {
        println!("✅ File already fully downloaded");
        return Ok(());
    }

    let progress = ProgressBar::new(total_size);
    progress.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
            .unwrap()
            .progress_chars("#>-")
    );
    progress.set_position(start_pos);
    if attempt > 1 {
        progress.set_message(format!("Attempt {}", attempt));
    }

    let mut request = client.get(url);
    if start_pos > 0 {
        request = request.header("Range", format!("bytes={}-", start_pos));
    }
    let response = request.send().await.context("Failed to start download")?;
    if !response.status().is_success() && response.status().as_u16() != 206 {
        anyhow::bail!("Download failed with status: {}", response.status());
    }

    if let Some(parent) = dest_path.parent() {
        fs::create_dir_all(parent).ok();
    }
    let mut file = if start_pos > 0 {
        File::options()
            .append(true)
            .open(dest_path)
            .context("Failed to open file for resume")?
    } else {
        File::create(dest_path).context("Failed to create file")?
    };

    let mut stream = response.bytes_stream();
    use futures::StreamExt;
    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.context("Failed to read chunk")?;
        file.write_all(&chunk).context("Failed to write chunk")?;
        progress.inc(chunk.len() as u64);
    }
    progress.finish_with_message("Download completed!");
    Ok(())
}

async fn download_attempt(zip_path: &Path, attempt: usize) -> Result<()> {
    let client = Client::builder()
        .timeout(tokio::time::Duration::from_secs(300)) // 5 minute timeout
        .build()
        .context("Failed to create HTTP client")?;

    // Check if partial file exists
    let mut start_pos = 0u64;
    if zip_path.exists() {
        start_pos = zip_path
            .metadata()
            .context("Failed to get file metadata")?
            .len();
        if attempt == 1 {
            println!(
                "📄 Found partial download, resuming from byte {}",
                start_pos
            );
        }
    }

    // Get file size first
    let head_response = client
        .head(GLTF_SAMPLE_ASSETS_URL)
        .send()
        .await
        .context("Failed to get file info")?;

    let total_size = head_response
        .headers()
        .get("content-length")
        .and_then(|ct_len| ct_len.to_str().ok())
        .and_then(|ct_len| ct_len.parse::<u64>().ok())
        .unwrap_or(0);

    if start_pos >= total_size && total_size > 0 {
        println!("✅ File already fully downloaded");
        return Ok(());
    }

    // Setup progress bar
    let progress = ProgressBar::new(total_size);
    progress.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
            .unwrap()
            .progress_chars("#>-")
    );
    progress.set_position(start_pos);

    if attempt > 1 {
        progress.set_message(format!("Attempt {}", attempt));
    }

    // Create request with range header for resume
    let mut request = client.get(GLTF_SAMPLE_ASSETS_URL);
    if start_pos > 0 {
        request = request.header("Range", format!("bytes={}-", start_pos));
    }

    let response = request.send().await.context("Failed to start download")?;

    if !response.status().is_success() && response.status().as_u16() != 206 {
        anyhow::bail!("Download failed with status: {}", response.status());
    }

    // Open file for writing (append mode for resume)
    let mut file = if start_pos > 0 {
        File::options()
            .append(true)
            .open(zip_path)
            .context("Failed to open file for resume")?
    } else {
        File::create(zip_path).context("Failed to create file")?
    };

    // Download in chunks
    let mut stream = response.bytes_stream();
    use futures::StreamExt;

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.context("Failed to read chunk")?;
        file.write_all(&chunk).context("Failed to write chunk")?;
        progress.inc(chunk.len() as u64);
    }

    progress.finish_with_message("Download completed!");
    println!("✅ Download finished: {}", zip_path.display());

    Ok(())
}

fn extract_zip(zip_path: &Path, extract_to: &Path) -> Result<()> {
    let file = File::open(zip_path).context("Failed to open zip file")?;

    let mut archive = zip::ZipArchive::new(file).context("Failed to read zip archive")?;

    let progress = ProgressBar::new(archive.len() as u64);
    progress.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} files")
            .unwrap()
            .progress_chars("#>-"),
    );

    for i in 0..archive.len() {
        let mut file = archive
            .by_index(i)
            .context("Failed to get file from archive")?;

        let outpath = match file.enclosed_name() {
            Some(path) => extract_to.join(path),
            None => continue,
        };

        if file.name().ends_with('/') {
            // Directory
            fs::create_dir_all(&outpath).context("Failed to create directory")?;
        } else {
            // File
            if let Some(p) = outpath.parent() {
                if !p.exists() {
                    fs::create_dir_all(p).context("Failed to create parent directory")?;
                }
            }

            let mut outfile = File::create(&outpath).context("Failed to create output file")?;
            io::copy(&mut file, &mut outfile).context("Failed to extract file")?;
        }

        progress.inc(1);
    }

    progress.finish_with_message("Extraction completed!");
    Ok(())
}
