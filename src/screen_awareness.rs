use anyhow::{anyhow, Result};
use chrono::Utc;

#[derive(Debug, Clone)]
pub struct CapturedFrame {
    pub png_bytes: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub source: String,
}

pub fn capture_from_target(target: Option<&str>) -> Result<CapturedFrame> {
    capture_impl(target)
}

pub fn default_filename() -> String {
    format!("screen-{}.png", Utc::now().format("%Y%m%d-%H%M%S%.3f"))
}

#[cfg(any(target_os = "windows", target_os = "macos"))]
fn capture_impl(target: Option<&str>) -> Result<CapturedFrame> {
    use screenshots::Screen;

    let normalized = target.unwrap_or("focused_window").trim().to_ascii_lowercase();
    if let Some(id) = normalized.strip_prefix("monitor:") {
        let monitor_id: u32 = id
            .trim()
            .parse()
            .map_err(|_| anyhow!("invalid monitor id in target '{normalized}'"))?;
        return capture_monitor_by_id(monitor_id);
    }
    match normalized.as_str() {
        "primary" | "primary_monitor" => capture_primary_monitor(),
        // screenshots crate does not expose per-window capture; fallback to primary monitor.
        _ => capture_primary_monitor(),
    }
}

#[cfg(any(target_os = "windows", target_os = "macos"))]
fn capture_primary_monitor() -> Result<CapturedFrame> {
    use screenshots::Screen;

    let monitor = Screen::all()?
        .into_iter()
        .find(|m| m.display_info.is_primary)
        .or_else(|| Screen::all().ok().and_then(|mut all| all.pop()))
        .ok_or_else(|| anyhow!("no monitor available for capture"))?;
    capture_screen_image(&monitor, format!("monitor:{}:primary", monitor.display_info.id))
}

#[cfg(any(target_os = "windows", target_os = "macos"))]
fn capture_monitor_by_id(monitor_id: u32) -> Result<CapturedFrame> {
    use screenshots::Screen;

    let monitor = Screen::all()?
        .into_iter()
        .find(|m| m.display_info.id == monitor_id)
        .ok_or_else(|| anyhow!("monitor id {monitor_id} not found"))?;
    capture_screen_image(&monitor, format!("monitor:{monitor_id}"))
}

#[cfg(any(target_os = "windows", target_os = "macos"))]
fn capture_screen_image(monitor: &screenshots::Screen, source: String) -> Result<CapturedFrame> {
    let image = monitor.capture()?;
    encode_frame(image.as_raw(), image.width(), image.height(), source)
}

#[cfg(any(target_os = "windows", target_os = "macos"))]
fn encode_frame(raw_rgba: &[u8], width: u32, height: u32, source: String) -> Result<CapturedFrame> {
    use image::ColorType;
    use image::ImageEncoder;
    use image::codecs::png::PngEncoder;

    let mut out = Vec::new();
    let encoder = PngEncoder::new(&mut out);
    encoder.write_image(raw_rgba, width, height, ColorType::Rgba8.into())?;
    Ok(CapturedFrame {
        png_bytes: out,
        width,
        height,
        source,
    })
}

#[cfg(not(any(target_os = "windows", target_os = "macos")))]
fn capture_impl(_target: Option<&str>) -> Result<CapturedFrame> {
    Err(anyhow!(
        "screen capture is not enabled for this target build (requires macOS/Windows in this build profile)"
    ))
}
