use serde::{Deserialize, Serialize};
use anyhow::{Result, anyhow};
use reqwest::Client;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThinkingMode {
    Disabled,
    Auto,
    Low,
    Medium,
    High,
}

impl ThinkingMode {
    pub fn as_reasoning_effort(&self) -> Option<&'static str> {
        match self {
            ThinkingMode::Disabled | ThinkingMode::Auto => None,
            ThinkingMode::Low => Some("low"),
            ThinkingMode::Medium => Some("medium"),
            ThinkingMode::High => Some("high"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub thinking_modes: Vec<ThinkingMode>,
}

impl ModelInfo {
    pub fn supports_thinking(&self) -> bool {
        !self.thinking_modes.is_empty()
    }
}

pub fn builtin_models() -> Vec<ModelInfo> {
    provider_models("openai")
}

pub fn provider_models(provider: &str) -> Vec<ModelInfo> {
    let p = provider.to_lowercase();
    if p == "nvidia" {
        return vec![
            ModelInfo {
                id: "meta/llama-3.1-70b-instruct".to_string(),
                name: "Llama 3.1 70B Instruct".to_string(),
                thinking_modes: vec![],
            },
            ModelInfo {
                id: "mistralai/mixtral-8x7b-instruct-v0.1".to_string(),
                name: "Mixtral 8x7B Instruct".to_string(),
                thinking_modes: vec![],
            },
        ];
    }
    if p == "openrouter" {
        return vec![
            ModelInfo {
                id: "openai/gpt-4o".to_string(),
                name: "OpenRouter GPT-4o".to_string(),
                thinking_modes: vec![],
            },
            ModelInfo {
                id: "anthropic/claude-3.5-sonnet".to_string(),
                name: "Claude 3.5 Sonnet".to_string(),
                thinking_modes: vec![],
            },
        ];
    }
    if p == "huggingface" {
        return vec![
            ModelInfo {
                id: "meta-llama/Llama-3.1-8B-Instruct".to_string(),
                name: "Llama 3.1 8B Instruct".to_string(),
                thinking_modes: vec![],
            },
        ];
    }

    vec![
        ModelInfo {
            id: "gpt-4o".to_string(),
            name: "GPT-4o".to_string(),
            thinking_modes: vec![],
        },
        ModelInfo {
            id: "gpt-4o-mini".to_string(),
            name: "GPT-4o Mini".to_string(),
            thinking_modes: vec![],
        },
        ModelInfo {
            id: "gpt-3.5-turbo".to_string(),
            name: "GPT-3.5 Turbo".to_string(),
            thinking_modes: vec![],
        },
        ModelInfo {
            id: "o1".to_string(),
            name: "O1".to_string(),
            thinking_modes: vec![
                ThinkingMode::Disabled,
                ThinkingMode::Auto,
                ThinkingMode::Low,
                ThinkingMode::Medium,
                ThinkingMode::High,
            ],
        },
        ModelInfo {
            id: "o1-mini".to_string(),
            name: "O1 Mini".to_string(),
            thinking_modes: vec![
                ThinkingMode::Disabled,
                ThinkingMode::Auto,
                ThinkingMode::Low,
                ThinkingMode::Medium,
                ThinkingMode::High,
            ],
        },
        ModelInfo {
            id: "o1-preview".to_string(),
            name: "O1 Preview".to_string(),
            thinking_modes: vec![
                ThinkingMode::Disabled,
                ThinkingMode::Auto,
                ThinkingMode::Low,
                ThinkingMode::Medium,
                ThinkingMode::High,
            ],
        },
        ModelInfo {
            id: "o3-mini".to_string(),
            name: "O3 Mini".to_string(),
            thinking_modes: vec![
                ThinkingMode::Disabled,
                ThinkingMode::Auto,
                ThinkingMode::Low,
                ThinkingMode::Medium,
                ThinkingMode::High,
            ],
        },
    ]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: serde_json::Value,
}

impl ChatMessage {
    pub fn text(role: &str, content: &str) -> Self {
        Self {
            role: role.to_string(),
            content: serde_json::Value::String(content.to_string()),
        }
    }

    pub fn with_cache_control(role: &str, content: &str) -> Self {
        Self {
            role: role.to_string(),
            content: serde_json::json!([{
                "type": "text",
                "text": content,
                "cache_control": {"type": "ephemeral"}
            }]),
        }
    }

    pub fn with_image(role: &str, text: &str, image_base64: &str, mime_type: &str) -> Self {
        Self {
            role: role.to_string(),
            content: serde_json::json!([
                {
                    "type": "text",
                    "text": text
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": format!("data:{};base64,{}", mime_type, image_base64)
                    }
                }
            ]),
        }
    }
}

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
}

#[derive(Debug, Deserialize)]
pub struct ChatChoice {
    pub message: ChatResponseMessage,
}

#[derive(Debug, Deserialize)]
pub struct ChatResponseMessage {
    pub role: String,
    pub content: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ChatResponse {
    pub choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
struct StreamDelta {
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct StreamChoice {
    delta: StreamDelta,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct StreamChunk {
    choices: Vec<StreamChoice>,
}

pub struct OpenAIClient {
    client: Client,
    api_key: String,
    base_url: String,
}

impl OpenAIClient {
    pub fn new(api_key: &str, base_url: &str) -> Self {
        Self {
            client: Client::new(),
            api_key: api_key.to_string(),
            base_url: base_url.trim_end_matches('/').to_string(),
        }
    }

    pub async fn chat_completion(
        &self,
        model: &str,
        messages: Vec<ChatMessage>,
        thinking_mode: Option<&ThinkingMode>,
        on_chunk: impl Fn(String) + Send + 'static,
    ) -> Result<String> {
        use futures_util::StreamExt;

        let is_thinking_model = model.starts_with("o1") || model.starts_with("o3");

        let request = ChatRequest {
            model: model.to_string(),
            messages,
            reasoning_effort: if is_thinking_model {
                thinking_mode
                    .and_then(|m| m.as_reasoning_effort())
                    .map(|s| s.to_string())
            } else {
                None
            },
            stream: Some(true),
            max_tokens: None,
        };

        let url = format!("{}/chat/completions", self.base_url);
        let response = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(anyhow!("API error {}: {}", status, body));
        }

        let mut stream = response.bytes_stream();
        let mut full_content = String::new();

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result?;
            let text = String::from_utf8_lossy(&chunk);

            for line in text.lines() {
                if let Some(data) = line.strip_prefix("data: ") {
                    if data.trim() == "[DONE]" {
                        break;
                    }
                    if let Ok(parsed) = serde_json::from_str::<StreamChunk>(data) {
                        for choice in parsed.choices {
                            if let Some(content) = choice.delta.content {
                                if !content.is_empty() {
                                    full_content.push_str(&content);
                                    on_chunk(content);
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(full_content)
    }

    pub async fn list_models(&self) -> Result<Vec<String>> {
        #[derive(Deserialize)]
        struct ModelItem {
            id: String,
        }
        #[derive(Deserialize)]
        struct ModelsResponse {
            data: Vec<ModelItem>,
        }

        let url = format!("{}/models", self.base_url);
        let response = self
            .client
            .get(&url)
            .bearer_auth(&self.api_key)
            .send()
            .await?;

        if !response.status().is_success() {
            return Ok(vec![]);
        }

        let models: ModelsResponse = response.json().await?;
        let mut ids: Vec<String> = models.data.into_iter().map(|m| m.id).collect();
        ids.sort();
        Ok(ids)
    }
}
