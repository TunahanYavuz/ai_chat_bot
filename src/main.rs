mod api;
mod app;
mod config;
mod db;
mod executor;
mod files;
mod parser;
mod rag_engine;
mod setup;
mod storage;
mod swarm;
mod telemetry;
mod web_engine;
mod models;
mod watcher;

use std::sync::Arc;

fn main() -> eframe::Result<()> {
    let rt = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => Arc::new(rt),
        Err(e) => {
            eprintln!("Failed to build Tokio runtime: {e}");
            return Ok(());
        }
    };

    let rt_clone = rt.clone();

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("AI Chat Bot")
            .with_inner_size([1200.0, 800.0])
            .with_min_inner_size([800.0, 600.0]),
        ..Default::default()
    };

    eframe::run_native(
        "AI Chat Bot",
        native_options,
        Box::new(move |cc| {
            egui_extras::install_image_loaders(&cc.egui_ctx);
            Ok(Box::new(app::ChatApp::new(cc, rt_clone.clone())))
        }),
    )
}
