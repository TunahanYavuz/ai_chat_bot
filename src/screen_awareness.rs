use anyhow::{anyhow, Result};
use chrono::Utc;
use image::ColorType;
use image::ImageEncoder;
use image::codecs::png::PngEncoder;
use xcap::{Monitor, Window};

#[derive(Debug, Clone)]
pub struct CapturedFrame {
    pub png_bytes: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub source: String,
}

pub fn capture_from_target(target: Option<&str>) -> Result<CapturedFrame> {
    let normalized = target.unwrap_or("focused_window").trim().to_ascii_lowercase();
    if let Some(title) = normalized.strip_prefix("window:") {
        return capture_window_by_title(title.trim());
    }
    if let Some(id) = normalized.strip_prefix("monitor:") {
        let monitor_id: u32 = id
            .trim()
            .parse()
            .map_err(|_| anyhow!("invalid monitor id in target '{normalized}'"))?;
        return capture_monitor_by_id(monitor_id);
    }
    match normalized.as_str() {
        "focused_window" => capture_focused_window().or_else(|_| capture_primary_monitor()),
        "primary" | "primary_monitor" => capture_primary_monitor(),
        _ => capture_focused_window().or_else(|_| capture_primary_monitor()),
    }
}

pub fn default_filename() -> String {
    format!("screen-{}.png", Utc::now().format("%Y%m%d-%H%M%S%.3f"))
}

fn capture_primary_monitor() -> Result<CapturedFrame> {
    let monitor = Monitor::all()?
        .into_iter()
        .find(|m| m.is_primary().unwrap_or(false))
        .or_else(|| Monitor::all().ok().and_then(|mut all| all.pop()))
        .ok_or_else(|| anyhow!("no monitor available for capture"))?;
    let source = format!(
        "monitor:{}:{}",
        monitor.id().unwrap_or_default(),
        monitor
            .friendly_name()
            .unwrap_or_else(|_| "primary".to_string())
    );
    capture_monitor_image(&monitor, source)
}

fn capture_monitor_by_id(monitor_id: u32) -> Result<CapturedFrame> {
    let monitor = Monitor::all()?
        .into_iter()
        .find(|m| m.id().ok() == Some(monitor_id))
        .ok_or_else(|| anyhow!("monitor id {monitor_id} not found"))?;
    let source = format!(
        "monitor:{}:{}",
        monitor_id,
        monitor
            .friendly_name()
            .unwrap_or_else(|_| "unknown".to_string())
    );
    capture_monitor_image(&monitor, source)
}

fn capture_monitor_image(monitor: &Monitor, source: String) -> Result<CapturedFrame> {
    let image = monitor.capture_image()?;
    encode_frame(image.as_raw(), image.width(), image.height(), source)
}

fn capture_focused_window() -> Result<CapturedFrame> {
    let window = Window::all()?
        .into_iter()
        .find(|w| w.is_focused().unwrap_or(false) && !w.is_minimized().unwrap_or(false))
        .ok_or_else(|| anyhow!("no focused window is available for capture"))?;
    capture_window_image(&window)
}

fn capture_window_by_title(partial_title: &str) -> Result<CapturedFrame> {
    let needle = partial_title.to_ascii_lowercase();
    let window = Window::all()?
        .into_iter()
        .find(|w| {
            !w.is_minimized().unwrap_or(false)
                && w.title()
                    .unwrap_or_default()
                    .to_ascii_lowercase()
                    .contains(&needle)
        })
        .ok_or_else(|| anyhow!("window containing '{partial_title}' was not found"))?;
    capture_window_image(&window)
}

fn capture_window_image(window: &Window) -> Result<CapturedFrame> {
    let title = window.title().unwrap_or_else(|_| "window".to_string());
    let image = window.capture_image()?;
    let source = format!("window:{}:{}", window.id().unwrap_or_default(), title);
    encode_frame(image.as_raw(), image.width(), image.height(), source)
}

fn encode_frame(raw_rgba: &[u8], width: u32, height: u32, source: String) -> Result<CapturedFrame> {
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
