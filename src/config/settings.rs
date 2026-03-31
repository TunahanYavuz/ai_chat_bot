use serde::{Deserialize, Serialize};
use anyhow::Result;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    pub openai_api_key: String,
    pub openai_base_url: String,
    pub db_path: String,
    pub working_directory: String,
    pub shell_execution_enabled: bool,
    pub default_model: String,
}

impl Default for Settings {
    fn default() -> Self {
        let db_path = dirs::data_local_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("ai_chat_bot")
            .join("chat.db")
            .to_string_lossy()
            .to_string();

        Self {
            openai_api_key: String::new(),
            openai_base_url: "https://api.openai.com/v1".to_string(),
            db_path,
            working_directory: dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .to_string_lossy()
                .to_string(),
            shell_execution_enabled: false,
            default_model: "gpt-4o".to_string(),
        }
    }
}

fn config_path() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("ai_chat_bot")
        .join("config.json")
}

pub fn load_settings() -> Settings {
    let path = config_path();
    if path.exists() {
        match std::fs::read_to_string(&path) {
            Ok(content) => serde_json::from_str(&content).unwrap_or_default(),
            Err(_) => Settings::default(),
        }
    } else {
        Settings::default()
    }
}

pub fn save_settings(settings: &Settings) -> Result<()> {
    let path = config_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let content = serde_json::to_string_pretty(settings)?;
    std::fs::write(&path, content)?;
    Ok(())
}
