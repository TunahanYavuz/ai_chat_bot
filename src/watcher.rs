use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};
use notify::{RecursiveMode, Watcher};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

/// Debounced workspace watcher loop.
///
/// - `workspace_root`: root folder to watch recursively.
/// - `refresh_tx`: debounced refresh signal channel to UI/app loop.
pub async fn run_workspace_watcher(
    workspace_root: String,
    refresh_tx: mpsc::Sender<()>,
) -> Result<()> {
    let watch_root = PathBuf::from(&workspace_root);
    let (fs_evt_tx, fs_evt_rx) = std::sync::mpsc::channel::<notify::Result<notify::Event>>();
    let (raw_signal_tx, raw_signal_rx) = mpsc::channel::<()>(512);

    let mut watcher = notify::recommended_watcher(move |res| {
        let _ = fs_evt_tx.send(res);
    })
    .context("failed to initialize notify watcher")?;

    watcher
        .watch(&watch_root, RecursiveMode::Recursive)
        .with_context(|| format!("failed to watch {}", watch_root.display()))?;

    let signal_bridge = tokio::task::spawn_blocking(move || {
        while let Ok(evt) = fs_evt_rx.recv() {
            if evt.is_ok() && raw_signal_tx.blocking_send(()).is_err() {
                break;
            }
        }
    });

    let mut stream = ReceiverStream::new(raw_signal_rx);
    while stream.next().await.is_some() {
        // Debounce event storms (cargo build, git checkout, branch switch, etc.).
        loop {
            let sleep = tokio::time::sleep(Duration::from_millis(200));
            tokio::pin!(sleep);
            tokio::select! {
                _ = &mut sleep => break,
                maybe_evt = stream.next() => {
                    if maybe_evt.is_none() {
                        break;
                    }
                }
            }
        }
        if refresh_tx.send(()).await.is_err() {
            break;
        }
    }

    signal_bridge.abort();
    drop(watcher);
    Ok(())
}
