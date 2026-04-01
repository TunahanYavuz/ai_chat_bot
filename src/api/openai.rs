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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteModelInfo {
    pub id: String,
    pub gated: bool,
    pub premium: bool,
}

pub fn builtin_models() -> Vec<ModelInfo> {
    vec![ModelInfo {
        id: "gpt-4o".to_string(),
        name: "gpt-4o".to_string(),
        thinking_modes: vec![],
    }]
}

pub fn provider_models(provider: &str) -> Vec<ModelInfo> {
    let _ = provider;
    vec![]
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
        Self::text(role, content)
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

#[derive(Debug, Serialize, Clone)]
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

    async fn chat_completion_non_stream(&self, mut request: ChatRequest) -> Result<String> {
        request.stream = None;

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

        let parsed: ChatResponse = response.json().await?;
        let text = parsed
            .choices
            .into_iter()
            .next()
            .and_then(|choice| choice.message.content)
            .unwrap_or_default();
        Ok(text)
    }

    pub async fn chat_completion(
        &self,
        model: &str,
        messages: Vec<ChatMessage>,
        thinking_mode: Option<&ThinkingMode>,
        on_chunk: impl Fn(String) + Send + 'static,
    ) -> Result<String> {
        use futures_util::StreamExt;

        let is_thinking_model = model.contains("o1") || model.contains("o3");
        let supports_reasoning_effort = self
            .base_url
            .trim_end_matches('/')
            .starts_with("https://api.openai.com/v1");

        let request = ChatRequest {
            model: model.to_string(),
            messages,
            reasoning_effort: if is_thinking_model && supports_reasoning_effort {
                thinking_mode.and_then(|m| m.as_reasoning_effort()).map(str::to_string)
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

        let mut stream_finished = false;
        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result?;
            let text = String::from_utf8_lossy(&chunk);

            for line in text.lines() {
                if let Some(data) = line.strip_prefix("data: ") {
                    if data.trim() == "[DONE]" {
                        stream_finished = true;
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
                            if choice.finish_reason.is_some() {
                                stream_finished = true;
                            }
                        }
                    }
                }
            }
            if stream_finished {
                break;
            }
        }

        if full_content.trim().is_empty() && stream_finished {
            eprintln!("Streaming response completed empty; retrying once with non-stream request");
            let fallback_text = self.chat_completion_non_stream(request).await?;
            if !fallback_text.is_empty() {
                on_chunk(fallback_text.clone());
            }
            return Ok(fallback_text);
        }

        Ok(full_content)
    }

    pub async fn list_models(&self) -> Result<Vec<RemoteModelInfo>> {
        #[derive(Deserialize)]
        struct ModelItem {
            id: String,
            #[serde(default)]
            gated: Option<bool>,
            #[serde(default, rename = "is_gated")]
            is_gated: Option<bool>,
            #[serde(default)]
            premium: Option<bool>,
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
            return Ok(Vec::new());
        }

        let models: ModelsResponse = response.json().await?;
        let mut items: Vec<RemoteModelInfo> = models
            .data
            .into_iter()
            .map(|m| RemoteModelInfo {
                id: m.id,
                gated: m.gated.or(m.is_gated).unwrap_or(false),
                premium: m.premium.unwrap_or(false),
            })
            .collect();
        items.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(items)
    }
}
