use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use anyhow::Result;
use nvml_wrapper::Nvml;
use sysinfo::System;

/// Captures NVIDIA GPU VRAM details for one device.
#[derive(Debug, Clone)]
pub struct GpuTelemetry {
    pub index: u32,
    pub name: String,
    pub total_vram_bytes: u64,
    pub used_vram_bytes: u64,
    pub free_vram_bytes: u64,
}

/// Captures system-level telemetry used to inform model behavior.
#[derive(Debug, Clone)]
pub struct TelemetrySnapshot {
    pub total_memory_bytes: u64,
    pub available_memory_bytes: u64,
    pub cpu_usage_percent: f32,
    pub gpus: Vec<GpuTelemetry>,
    pub warnings: Vec<String>,
}

impl TelemetrySnapshot {
    /// Renders telemetry into a stable system-prompt section for the LLM payload.
    pub fn to_llm_system_context(&self) -> String {
        let mut lines = vec![
            "SYSTEM TELEMETRY:".to_string(),
            format!("- CPU usage (total): {:.2}%", self.cpu_usage_percent),
            format!("- Memory available: {}", format_bytes(self.available_memory_bytes)),
            format!("- Memory total: {}", format_bytes(self.total_memory_bytes)),
        ];

        if self.gpus.is_empty() {
            lines.push("- NVIDIA VRAM: unavailable (no supported NVIDIA GPU detected or NVML unavailable)".to_string());
        } else {
            lines.push("- NVIDIA VRAM:".to_string());
            for gpu in &self.gpus {
                lines.push(format!(
                    "  - GPU {} ({}): used {} / total {} (free {})",
                    gpu.index,
                    gpu.name,
                    format_bytes(gpu.used_vram_bytes),
                    format_bytes(gpu.total_vram_bytes),
                    format_bytes(gpu.free_vram_bytes)
                ));
            }
        }

        if !self.warnings.is_empty() {
            lines.push("- Telemetry warnings:".to_string());
            for warning in &self.warnings {
                lines.push(format!("  - {}", warning));
            }
        }

        lines.join("\n")
    }
}

/// Collects RAM + CPU telemetry with native sysinfo and GPU telemetry with NVML when available.
pub fn collect_telemetry() -> TelemetrySnapshot {
    let mut warnings = Vec::new();

    let mut system = System::new();
    system.refresh_memory();
    system.refresh_cpu_usage();

    let total_memory_bytes = system.total_memory();
    let available_memory_bytes = system.available_memory();
    let cpu_usage_percent = system.global_cpu_usage();

    let gpus = match collect_nvidia_gpu_telemetry() {
        Ok(items) => items,
        Err(err) => {
            warnings.push(format!("NVML collection failed: {err}"));
            Vec::new()
        }
    };

    TelemetrySnapshot {
        total_memory_bytes,
        available_memory_bytes,
        cpu_usage_percent,
        gpus,
        warnings,
    }
}

/// Returns a cached telemetry snapshot to avoid repeated heavy polling on each prompt dispatch.
///
/// The snapshot is refreshed at most once per `ttl`; otherwise a cloned cached value is returned.
/// This keeps prompt construction responsive while still giving recent hardware context.
pub fn collect_telemetry_cached(ttl: Duration) -> TelemetrySnapshot {
    #[derive(Clone)]
    struct CacheEntry {
        snapshot: TelemetrySnapshot,
        collected_at: Instant,
    }

    static CACHE: OnceLock<Mutex<Option<CacheEntry>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(None));

    if let Ok(mut guard) = cache.lock() {
        if let Some(entry) = guard.as_ref() {
            if entry.collected_at.elapsed() < ttl {
                return entry.snapshot.clone();
            }
        }

        let fresh = collect_telemetry();
        *guard = Some(CacheEntry {
            snapshot: fresh.clone(),
            collected_at: Instant::now(),
        });
        return fresh;
    }

    collect_telemetry()
}

fn collect_nvidia_gpu_telemetry() -> Result<Vec<GpuTelemetry>> {
    let nvml = Nvml::init()?;
    let count = nvml.device_count()?;

    let mut gpus = Vec::new();
    for index in 0..count {
        let device = nvml.device_by_index(index)?;
        let memory = device.memory_info()?;
        let name = device.name()?;

        gpus.push(GpuTelemetry {
            index,
            name,
            total_vram_bytes: memory.total,
            used_vram_bytes: memory.used,
            free_vram_bytes: memory.free,
        });
    }

    Ok(gpus)
}

fn format_bytes(bytes: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;

    let b = bytes as f64;
    if b >= GB {
        format!("{:.2} GiB", b / GB)
    } else if b >= MB {
        format!("{:.2} MiB", b / MB)
    } else if b >= KB {
        format!("{:.2} KiB", b / KB)
    } else {
        format!("{} B", bytes)
    }
}
