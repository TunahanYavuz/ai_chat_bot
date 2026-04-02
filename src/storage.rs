use std::path::PathBuf;

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use tokio::fs;
use uuid::Uuid;

/// One persisted chat role.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StoredRole {
    System,
    User,
    Assistant,
}

/// One persisted message item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredMessage {
    pub role: StoredRole,
    pub content: String,
    pub timestamp: DateTime<Utc>,
}

/// Session payload persisted to local disk for cross-restart memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatSession {
    pub id: String,
    pub name: String,
    pub updated_at: DateTime<Utc>,
    pub messages: Vec<StoredMessage>,
}

/// Resolved host-specific storage paths for this app.
#[derive(Debug, Clone)]
pub struct StoragePaths {
    #[allow(dead_code)] // Kept for future settings/config file persistence expansion.
    pub config_dir: PathBuf,
    pub data_dir: PathBuf,
    pub latest_session_path: PathBuf,
    pub quota_cache_path: PathBuf,
}

impl StoragePaths {
    /// Resolve and eagerly create config/data directories.
    pub fn resolve() -> Result<Self> {
        let project_dirs = ProjectDirs::from("com", "TunahanYavuz", "ai-os")
            .context("failed to resolve OS project directories")?;

        let config_dir = project_dirs.config_dir().to_path_buf();
        let data_dir = project_dirs.data_dir().to_path_buf();

        std::fs::create_dir_all(&config_dir)
            .with_context(|| format!("failed to create config dir {}", config_dir.display()))?;
        std::fs::create_dir_all(&data_dir)
            .with_context(|| format!("failed to create data dir {}", data_dir.display()))?;

        let latest_session_path = data_dir.join("latest_session.json");
        let quota_cache_path = data_dir.join("quota_cache.json");
        Ok(Self {
            config_dir,
            data_dir,
            latest_session_path,
            quota_cache_path,
        })
    }
}

/// Thin async storage facade for session persistence.
#[derive(Debug, Clone)]
pub struct Storage {
    paths: StoragePaths,
}

#[derive(Debug, Clone)]
pub struct FileNode {
    pub name: String,
    pub path: String,
    pub is_dir: bool,
    pub children: Vec<FileNode>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StoredQuotaMetrics {
    pub tokens_used: u64,
    pub request_count: u64,
    pub max_tokens: u64,
    pub max_requests: u64,
}

impl Storage {
    pub fn new() -> Result<Self> {
        let paths = StoragePaths::resolve()?;
        Ok(Self { paths })
    }

    #[allow(dead_code)] // Kept for future diagnostics and path introspection in UI.
    pub fn paths(&self) -> &StoragePaths {
        &self.paths
    }

    /// Persist session atomically-ish via temp file + rename.
    pub async fn save_session(&self, session: &ChatSession) -> Result<()> {
        let serialized =
            serde_json::to_vec_pretty(session).context("failed to serialize chat session")?;
        let tmp_path = self
            .paths
            .data_dir
            .join(format!("latest_session_{}.json.tmp", Uuid::new_v4()));

        fs::write(&tmp_path, &serialized)
            .await
            .with_context(|| format!("failed to write temp session file {}", tmp_path.display()))?;
        fs::rename(&tmp_path, &self.paths.latest_session_path)
            .await
            .with_context(|| {
                format!(
                    "failed to replace latest session file {}",
                    self.paths.latest_session_path.display()
                )
            })?;
        Ok(())
    }

    /// Load most recent persisted session, if available.
    pub async fn load_latest_session(&self) -> Result<Option<ChatSession>> {
        if !self.paths.latest_session_path.exists() {
            return Ok(None);
        }

        let bytes = fs::read(&self.paths.latest_session_path)
            .await
            .with_context(|| {
                format!(
                    "failed to read latest session file {}",
                    self.paths.latest_session_path.display()
                )
            })?;
        let parsed = serde_json::from_slice::<ChatSession>(&bytes)
            .context("failed to parse session JSON")?;
        Ok(Some(parsed))
    }

    pub async fn save_quota_cache(
        &self,
        cache: &std::collections::HashMap<String, StoredQuotaMetrics>,
    ) -> Result<()> {
        let serialized =
            serde_json::to_vec_pretty(cache).context("failed to serialize quota cache")?;
        let tmp_path = self
            .paths
            .data_dir
            .join(format!("quota_cache_{}.json.tmp", Uuid::new_v4()));
        fs::write(&tmp_path, &serialized)
            .await
            .with_context(|| format!("failed to write temp quota cache {}", tmp_path.display()))?;
        fs::rename(&tmp_path, &self.paths.quota_cache_path)
            .await
            .with_context(|| {
                format!(
                    "failed to replace quota cache file {}",
                    self.paths.quota_cache_path.display()
                )
            })?;
        Ok(())
    }

    pub async fn load_quota_cache(
        &self,
    ) -> Result<Option<std::collections::HashMap<String, StoredQuotaMetrics>>> {
        if !self.paths.quota_cache_path.exists() {
            return Ok(None);
        }
        let bytes = fs::read(&self.paths.quota_cache_path)
            .await
            .with_context(|| {
                format!(
                    "failed to read quota cache file {}",
                    self.paths.quota_cache_path.display()
                )
            })?;
        let parsed =
            serde_json::from_slice::<std::collections::HashMap<String, StoredQuotaMetrics>>(&bytes)
                .context("failed to parse quota cache JSON")?;
        Ok(Some(parsed))
    }
}

const IGNORED_DIRS: &[&str] = &[".git", "target", "node_modules"];

fn should_ignore_dir(name: &str) -> bool {
    let normalized = name.to_ascii_lowercase();
    IGNORED_DIRS.iter().any(|d| *d == normalized)
}

fn file_name_or_path(path: &std::path::Path) -> String {
    path.file_name()
        .and_then(|s| s.to_str())
        .map(str::to_string)
        .unwrap_or_else(|| path.to_string_lossy().to_string())
}

pub async fn rebuild_file_cache(workspace_root: String) -> Result<FileNode> {
    tokio::task::spawn_blocking(move || build_file_tree_sync(std::path::Path::new(&workspace_root)))
        .await
        .context("workspace file cache task failed")?
}

fn build_file_tree_sync(path: &std::path::Path) -> Result<FileNode> {
    let metadata = std::fs::metadata(path)
        .with_context(|| format!("failed to read metadata for {}", path.display()))?;
    let is_dir = metadata.is_dir();

    let mut node = FileNode {
        name: file_name_or_path(path),
        path: path.to_string_lossy().to_string(),
        is_dir,
        children: Vec::new(),
    };

    if !is_dir {
        return Ok(node);
    }

    let mut entries: Vec<(std::path::PathBuf, bool, String)> = Vec::new();
    for entry in std::fs::read_dir(path)
        .with_context(|| format!("failed to read directory {}", path.display()))?
    {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };
        let entry_path = entry.path();
        let metadata = match entry.metadata() {
            Ok(m) => m,
            Err(_) => continue,
        };
        let is_dir = metadata.is_dir();
        let name = file_name_or_path(&entry_path);
        entries.push((entry_path, is_dir, name));
    }

    entries.sort_by(|a, b| match b.1.cmp(&a.1) {
        std::cmp::Ordering::Equal => a.2.cmp(&b.2),
        other => other,
    });

    for (entry_path, is_dir, name) in entries {
        if is_dir && should_ignore_dir(&name) {
            continue;
        }
        if let Ok(child) = build_file_tree_sync(&entry_path) {
            node.children.push(child);
        }
    }

    Ok(node)
}
