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
    pub config_dir: PathBuf,
    pub data_dir: PathBuf,
    pub latest_session_path: PathBuf,
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
        Ok(Self {
            config_dir,
            data_dir,
            latest_session_path,
        })
    }
}

/// Thin async storage facade for session persistence.
#[derive(Debug, Clone)]
pub struct Storage {
    paths: StoragePaths,
}

impl Storage {
    pub fn new() -> Result<Self> {
        let paths = StoragePaths::resolve()?;
        Ok(Self { paths })
    }

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
        fs::rename(&tmp_path, &self.paths.latest_session_path).await.with_context(|| {
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
        let parsed =
            serde_json::from_slice::<ChatSession>(&bytes).context("failed to parse session JSON")?;
        Ok(Some(parsed))
    }
}
