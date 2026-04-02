use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

pub const DEFAULT_MODEL_ID: &str = "gpt-4o";
const OLD_HUGGINGFACE_BASE_URL: &str = "https://api-inference.huggingface.co/v1";

// ─── API Provider ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Hash, Default)]
pub enum ApiProvider {
    #[default]
    OpenAI,
    Nvidia,
    OpenRouter,
    HuggingFace,
    Custom,
}

impl ApiProvider {
    /// Short key used in the HashMap and JSON storage.
    pub fn key(&self) -> &'static str {
        match self {
            Self::OpenAI => "openai",
            Self::Nvidia => "nvidia",
            Self::OpenRouter => "openrouter",
            Self::HuggingFace => "huggingface",
            Self::Custom => "custom",
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            Self::OpenAI => "OpenAI",
            Self::Nvidia => "NVIDIA NIM",
            Self::OpenRouter => "OpenRouter",
            Self::HuggingFace => "Hugging Face",
            Self::Custom => "Custom",
        }
    }

    pub fn default_base_url(&self) -> &'static str {
        match self {
            Self::OpenAI => "https://api.openai.com/v1",
            Self::Nvidia => "https://integrate.api.nvidia.com/v1",
            Self::OpenRouter => "https://openrouter.ai/api/v1",
            Self::HuggingFace => "https://router.huggingface.co/v1",
            Self::Custom => "https://",
        }
    }

    /// URL where the user can obtain an API key for this provider.
    pub fn api_key_url(&self) -> &'static str {
        match self {
            Self::OpenAI => "https://platform.openai.com/api-keys",
            Self::Nvidia => "https://build.nvidia.com/",
            Self::OpenRouter => "https://openrouter.ai/keys",
            Self::HuggingFace => "https://huggingface.co/settings/tokens",
            Self::Custom => "",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::OpenAI => "GPT-4o, o1, o3-mini and more. Industry-leading models.",
            Self::Nvidia => "Llama, Mistral, Gemma and more via NVIDIA NIM.",
            Self::OpenRouter => "100+ models: Claude, Gemini, Llama, DeepSeek, and more.",
            Self::HuggingFace => "Open-source models via the Hugging Face Inference API.",
            Self::Custom => "Any OpenAI-compatible API endpoint.",
        }
    }

    pub fn all() -> Vec<ApiProvider> {
        vec![
            Self::OpenAI,
            Self::Nvidia,
            Self::OpenRouter,
            Self::HuggingFace,
            Self::Custom,
        ]
    }
}

// ─── Per-provider credentials ─────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProviderConfig {
    pub api_key: String,
    pub base_url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ScheduledJobConfig {
    pub id: String,
    pub time_24h: String,
    pub prompt: String,
    pub enabled: bool,
}

// ─── Application Settings ─────────────────────────────────────────────────────

fn default_openai_url() -> String {
    "https://api.openai.com/v1".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    /// Currently active provider.
    #[serde(default)]
    pub selected_provider: ApiProvider,

    /// Credentials stored per provider key.
    #[serde(default)]
    pub provider_configs: HashMap<String, ProviderConfig>,

    // ── Legacy fields kept for backward-compat with existing config files ──
    #[serde(default)]
    pub openai_api_key: String,
    #[serde(default = "default_openai_url")]
    pub openai_base_url: String,

    // ── Shared settings ──────────────────────────────────────────────────
    pub db_path: String,
    pub working_directory: String,
    #[serde(default)]
    pub shell_execution_enabled: bool,
    pub default_model: String,
    /// Set to true once the setup wizard has been completed.
    #[serde(default)]
    pub setup_complete: bool,
    #[serde(default = "default_dark_mode")]
    pub dark_mode: bool,
    #[serde(default)]
    pub autonomous_schedules: Vec<ScheduledJobConfig>,
    #[serde(default)]
    pub screen_awareness_enabled: bool,
    #[serde(default)]
    pub mcp_enabled: bool,
    #[serde(default)]
    pub mcp_launch_command: String,
    #[serde(default)]
    pub mcp_launch_args: Vec<String>,
    #[serde(default = "default_rag_top_k_limit")]
    pub rag_top_k_limit: u64,
    #[serde(default = "default_rag_similarity_threshold")]
    pub rag_similarity_threshold: f32,
}

fn default_dark_mode() -> bool {
    true
}

fn default_rag_top_k_limit() -> u64 {
    5
}

fn default_rag_similarity_threshold() -> f32 {
    0.75
}

impl Settings {
    pub fn clamp_rag_settings(&mut self) {
        self.rag_top_k_limit = self.rag_top_k_limit.clamp(1, 20);
        self.rag_similarity_threshold = self.rag_similarity_threshold.clamp(0.0, 1.0);
    }

    /// API key for the currently selected provider.
    pub fn active_api_key(&self) -> String {
        if let Some(cfg) = self.provider_configs.get(self.selected_provider.key()) {
            if !cfg.api_key.is_empty() {
                return cfg.api_key.clone();
            }
        }
        // Fall back to the legacy OpenAI field so old config files keep working.
        if self.selected_provider == ApiProvider::OpenAI {
            return self.openai_api_key.clone();
        }
        String::new()
    }

    /// Base URL for the currently selected provider.
    pub fn active_base_url(&self) -> String {
        if let Some(cfg) = self.provider_configs.get(self.selected_provider.key()) {
            if !cfg.base_url.is_empty() {
                return cfg.base_url.clone();
            }
        }
        if self.selected_provider == ApiProvider::OpenAI && !self.openai_base_url.is_empty() {
            return self.openai_base_url.clone();
        }
        self.selected_provider.default_base_url().to_string()
    }

    /// Persist credentials for a specific provider.
    pub fn set_provider_config(&mut self, provider: &ApiProvider, api_key: &str, base_url: &str) {
        let url = if base_url.is_empty() || base_url == "https://" {
            provider.default_base_url().to_string()
        } else {
            base_url.to_string()
        };
        self.provider_configs.insert(
            provider.key().to_string(),
            ProviderConfig {
                api_key: api_key.to_string(),
                base_url: url.clone(),
            },
        );
        // Keep legacy fields in sync for OpenAI.
        if *provider == ApiProvider::OpenAI {
            self.openai_api_key = api_key.to_string();
            self.openai_base_url = url;
        }
    }

    /// Mutable access to a provider's config, inserting defaults if absent.
    pub fn provider_config_mut(&mut self, provider: &ApiProvider) -> &mut ProviderConfig {
        let default_url = provider.default_base_url().to_string();
        self.provider_configs
            .entry(provider.key().to_string())
            .or_insert_with(|| ProviderConfig {
                api_key: String::new(),
                base_url: default_url,
            })
    }

    /// Read-only API key for any provider (used in the settings dialog).
    pub fn get_provider_api_key(&self, provider: &ApiProvider) -> String {
        if let Some(cfg) = self.provider_configs.get(provider.key()) {
            if !cfg.api_key.is_empty() {
                return cfg.api_key.clone();
            }
        }
        if *provider == ApiProvider::OpenAI {
            return self.openai_api_key.clone();
        }
        String::new()
    }

    /// Read-only base URL for any provider (used in the settings dialog).
    pub fn get_provider_base_url(&self, provider: &ApiProvider) -> String {
        if let Some(cfg) = self.provider_configs.get(provider.key()) {
            if !cfg.base_url.is_empty() {
                return cfg.base_url.clone();
            }
        }
        provider.default_base_url().to_string()
    }
}

impl Default for Settings {
    fn default() -> Self {
        let db_path = dirs::data_local_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("ai_chat_bot")
            .join("chat.db")
            .to_string_lossy()
            .to_string();

        let mut provider_configs = HashMap::new();
        provider_configs.insert(
            ApiProvider::OpenAI.key().to_string(),
            ProviderConfig {
                api_key: String::new(),
                base_url: "https://api.openai.com/v1".to_string(),
            },
        );

        Self {
            selected_provider: ApiProvider::OpenAI,
            provider_configs,
            openai_api_key: String::new(),
            openai_base_url: "https://api.openai.com/v1".to_string(),
            db_path: db_path.clone(),
            working_directory: dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .to_string_lossy()
                .to_string(),
            shell_execution_enabled: false,
            default_model: DEFAULT_MODEL_ID.to_string(),
            setup_complete: false,
            dark_mode: true,
            autonomous_schedules: vec![],
            screen_awareness_enabled: false,
            mcp_enabled: false,
            mcp_launch_command: "npx".to_string(),
            mcp_launch_args: vec![
                "-y".to_string(),
                "mcp-server-sqlite".to_string(),
                "--db".to_string(),
                db_path.clone(),
            ],
            rag_top_k_limit: default_rag_top_k_limit(),
            rag_similarity_threshold: default_rag_similarity_threshold(),
        }
    }
}

// ─── Persistence ──────────────────────────────────────────────────────────────

fn config_path() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("ai_chat_bot")
        .join("config.json")
}

pub fn load_settings() -> Settings {
    let path = config_path();
    if path.exists() {
        if let Ok(content) = std::fs::read_to_string(&path) {
            if let Ok(mut s) = serde_json::from_str::<Settings>(&content) {
                // Migrate deprecated Hugging Face endpoint to the new router endpoint.
                if let Some(cfg) = s.provider_configs.get_mut(ApiProvider::HuggingFace.key()) {
                    if cfg.base_url.trim_end_matches('/') == OLD_HUGGINGFACE_BASE_URL {
                        cfg.base_url = ApiProvider::HuggingFace.default_base_url().to_string();
                    }
                }
                if s.mcp_launch_command.trim().is_empty() {
                    s.mcp_launch_command = "npx".to_string();
                }
                let uses_sqlite_server = s
                    .mcp_launch_args
                    .iter()
                    .any(|arg| arg.eq_ignore_ascii_case("mcp-server-sqlite"));
                if s.mcp_launch_args.is_empty() || uses_sqlite_server {
                    if !s
                        .mcp_launch_args
                        .iter()
                        .any(|arg| arg.eq_ignore_ascii_case("-y"))
                    {
                        s.mcp_launch_args.insert(0, "-y".to_string());
                    }
                    if !s
                        .mcp_launch_args
                        .iter()
                        .any(|arg| arg.eq_ignore_ascii_case("mcp-server-sqlite"))
                    {
                        s.mcp_launch_args.push("mcp-server-sqlite".to_string());
                    }
                    if !s
                        .mcp_launch_args
                        .iter()
                        .any(|arg| arg.eq_ignore_ascii_case("--db"))
                    {
                        s.mcp_launch_args.push("--db".to_string());
                        s.mcp_launch_args.push(s.db_path.clone());
                    } else {
                        let mut i = 0usize;
                        while i < s.mcp_launch_args.len() {
                            if s.mcp_launch_args[i].eq_ignore_ascii_case("--db") {
                                let missing_value = i + 1 >= s.mcp_launch_args.len()
                                    || s.mcp_launch_args[i + 1].trim().is_empty()
                                    || s.mcp_launch_args[i + 1].starts_with('-');
                                if missing_value {
                                    s.mcp_launch_args.insert(i + 1, s.db_path.clone());
                                }
                                break;
                            }
                            i += 1;
                        }
                    }
                }
                return s;
            }
        }
    }
    Settings::default()
}

pub fn save_settings(settings: &Settings) -> Result<()> {
    let path = config_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, serde_json::to_string_pretty(settings)?)?;
    Ok(())
}
