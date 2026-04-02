use reqwest::header::HeaderMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Reasoning level used by models that support tiered thinking controls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThinkingMode {
    Low,
    Medium,
    High,
}

impl ThinkingMode {
    pub fn as_reasoning_effort(&self) -> &'static str {
        match self {
            ThinkingMode::Low => "low",
            ThinkingMode::Medium => "medium",
            ThinkingMode::High => "high",
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            ThinkingMode::Low => "Low",
            ThinkingMode::Medium => "Medium",
            ThinkingMode::High => "High",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningCapability {
    None,
    Binary,
    Tiered,
}

#[derive(Debug, Clone, Copy)]
pub struct ModelReasoningConfig {
    pub capability: ReasoningCapability,
    pub tiered_modes: &'static [ThinkingMode],
    pub binary_label: &'static str,
    pub tiered_label: &'static str,
}

const TIERED_DEFAULT_MODES: &[ThinkingMode] =
    &[ThinkingMode::Low, ThinkingMode::Medium, ThinkingMode::High];

const REASONING_NONE: ModelReasoningConfig = ModelReasoningConfig {
    capability: ReasoningCapability::None,
    tiered_modes: &[],
    binary_label: "Enable Reasoning / Thinking",
    tiered_label: "Thinking level:",
};

const REASONING_BINARY: ModelReasoningConfig = ModelReasoningConfig {
    capability: ReasoningCapability::Binary,
    tiered_modes: &[],
    binary_label: "Enable Reasoning / Thinking",
    tiered_label: "Thinking level:",
};

const REASONING_TIERED: ModelReasoningConfig = ModelReasoningConfig {
    capability: ReasoningCapability::Tiered,
    tiered_modes: TIERED_DEFAULT_MODES,
    binary_label: "Enable Reasoning / Thinking",
    tiered_label: "Thinking level:",
};

/// Central capability mapping used by the UI and request pipeline.
///
/// The mapping is intentionally explicit for known model families and then
/// falls back to conservative pattern-based matching for provider-prefixed IDs.
pub fn reasoning_config_for_model(model_name: &str) -> ModelReasoningConfig {
    let m = model_name.trim().to_lowercase();
    if m.is_empty() {
        return REASONING_NONE;
    }

    let known_mappings: &[(&str, ModelReasoningConfig)] = &[
        ("o1", REASONING_TIERED),
        ("o1-mini", REASONING_TIERED),
        ("o1-preview", REASONING_TIERED),
        ("o3", REASONING_TIERED),
        ("o3-mini", REASONING_TIERED),
        ("o3-pro", REASONING_TIERED),
        ("o4-mini", REASONING_TIERED),
        ("gpt-5", REASONING_TIERED),
        ("gpt-5-mini", REASONING_TIERED),
        ("gpt-5.1", REASONING_TIERED),
        ("deepseek-r1", REASONING_BINARY),
        ("deepseek-reasoner", REASONING_BINARY),
        ("qwen-reasoner", REASONING_BINARY),
        ("qwq", REASONING_BINARY),
        ("qvq", REASONING_BINARY),
        ("qwen3", REASONING_BINARY),
        ("qwen2.5", REASONING_BINARY),
        ("qwen2.5-coder", REASONING_BINARY),
        ("qwen-long", REASONING_BINARY),
    ];

    if let Some((_, cfg)) = known_mappings
        .iter()
        .find(|(prefix, _)| m.starts_with(prefix))
    {
        return *cfg;
    }

    if m.starts_with("o1")
        || m.starts_with("o3")
        || m.starts_with("o4")
        || m.starts_with("gpt-5")
        || m.contains("/o1")
        || m.contains("/o3")
        || m.contains("-o1")
        || m.contains("-o3")
    {
        return REASONING_TIERED;
    }

    if m.contains("deepseek-r1")
        || m.contains("deepseek-reasoner")
        || m.contains("qwen-reasoner")
        || m.contains("qwq")
        || m.contains("qvq")
        || m.contains("qwen2.5")
        || m.contains("qwen3")
    {
        return REASONING_BINARY;
    }

    if m.contains("deepseek")
        || m.contains("qwq")
        || m.contains("reasoner")
        || m.contains("thinking")
    {
        return REASONING_BINARY;
    }

    if m.contains("/o1")
        || m.contains("/o3")
        || m.contains("-o1")
        || m.contains("-o3")
        || m.contains("_o1")
        || m.contains("_o3")
    {
        return REASONING_TIERED;
    }

    REASONING_NONE
}

pub fn get_model_capability(model_name: &str) -> ReasoningCapability {
    reasoning_config_for_model(model_name).capability
}

#[derive(Debug, Clone, Default)]
pub struct ProviderRequestOptions {
    pub reasoning_effort: Option<String>,
    pub include_reasoning: Option<bool>,
    pub stream: bool,
    pub max_tokens: Option<u32>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProviderQuotaSnapshot {
    pub remaining_tokens: Option<u64>,
    pub limit_tokens: Option<u64>,
    pub remaining_requests: Option<u64>,
    pub limit_requests: Option<u64>,
}

pub trait ProviderAdapter: Send + Sync {
    fn chat_endpoint(&self, base_url: &str) -> String;
    fn models_endpoint(&self, base_url: &str) -> String;
    fn auth_headers(&self, api_key: &str) -> Vec<(&'static str, String)>;
    fn build_chat_payload(
        &self,
        model: &str,
        messages: &[serde_json::Value],
        options: &ProviderRequestOptions,
    ) -> serde_json::Value;
    fn stream_done(&self, data: &str) -> bool;
    fn stream_delta_text(&self, data: &str) -> Option<String>;
    fn parse_non_stream_text(&self, body: &serde_json::Value) -> Option<String>;
    fn parse_quota_from_headers(&self, headers: &HeaderMap) -> Option<ProviderQuotaSnapshot>;
    fn parse_quota_from_body(&self, body: &serde_json::Value) -> Option<ProviderQuotaSnapshot>;
    fn merge_quota(
        &self,
        header_quota: Option<ProviderQuotaSnapshot>,
        body_quota: Option<ProviderQuotaSnapshot>,
    ) -> Option<ProviderQuotaSnapshot> {
        let mut out = header_quota.or(body_quota.clone())?;
        if let Some(body) = body_quota {
            out.remaining_tokens = out.remaining_tokens.or(body.remaining_tokens);
            out.limit_tokens = out.limit_tokens.or(body.limit_tokens);
            out.remaining_requests = out.remaining_requests.or(body.remaining_requests);
            out.limit_requests = out.limit_requests.or(body.limit_requests);
        }
        Some(out)
    }
    fn fetch_account_status(
        &self,
        client: &reqwest::Client,
        api_key: &str,
        base_url: &str,
    ) -> futures_util::future::BoxFuture<'static, anyhow::Result<Option<ProviderQuotaSnapshot>>>;
}

#[derive(Debug, Default)]
pub struct OpenAiCompatibleAdapter;

impl ProviderAdapter for OpenAiCompatibleAdapter {
    fn chat_endpoint(&self, base_url: &str) -> String {
        format!("{}/chat/completions", base_url.trim_end_matches('/'))
    }

    fn models_endpoint(&self, base_url: &str) -> String {
        format!("{}/models", base_url.trim_end_matches('/'))
    }

    fn auth_headers(&self, api_key: &str) -> Vec<(&'static str, String)> {
        vec![("Authorization", format!("Bearer {api_key}"))]
    }

    fn build_chat_payload(
        &self,
        model: &str,
        messages: &[serde_json::Value],
        options: &ProviderRequestOptions,
    ) -> serde_json::Value {
        serde_json::json!({
            "model": model,
            "messages": messages,
            "reasoning_effort": options.reasoning_effort,
            "include_reasoning": options.include_reasoning,
            "stream": options.stream,
            "max_tokens": options.max_tokens
        })
    }

    fn stream_done(&self, data: &str) -> bool {
        if data.trim() == "[DONE]" {
            return true;
        }
        serde_json::from_str::<serde_json::Value>(data)
            .ok()
            .and_then(|v| {
                v.get("choices")
                    .and_then(|c| c.as_array())
                    .and_then(|arr| arr.first())
                    .and_then(|c| c.get("finish_reason"))
                    .and_then(|f| f.as_str())
                    .map(|s| !s.is_empty())
            })
            .unwrap_or(false)
    }

    fn stream_delta_text(&self, data: &str) -> Option<String> {
        let parsed = serde_json::from_str::<serde_json::Value>(data).ok()?;
        parsed
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|arr| arr.first())
            .and_then(|c| c.get("delta"))
            .and_then(|d| d.get("content"))
            .and_then(|c| c.as_str())
            .map(|s| s.to_string())
    }

    fn parse_non_stream_text(&self, body: &serde_json::Value) -> Option<String> {
        body.get("choices")
            .and_then(|c| c.as_array())
            .and_then(|arr| arr.first())
            .and_then(|choice| choice.get("message"))
            .and_then(|msg| msg.get("content"))
            .and_then(|content| content.as_str())
            .map(|s| s.to_string())
    }

    fn parse_quota_from_headers(&self, headers: &HeaderMap) -> Option<ProviderQuotaSnapshot> {
        fn u64_header(headers: &HeaderMap, key: &str) -> Option<u64> {
            headers
                .get(key)
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.trim().parse::<u64>().ok())
        }
        let remaining_tokens = u64_header(headers, "x-ratelimit-remaining-tokens")
            .or_else(|| u64_header(headers, "x-ratelimit-remaining-token"))
            .or_else(|| u64_header(headers, "x-ratelimit-remaining"));
        let limit_tokens = u64_header(headers, "x-ratelimit-limit-tokens")
            .or_else(|| u64_header(headers, "x-ratelimit-limit-token"))
            .or_else(|| u64_header(headers, "x-ratelimit-limit"));
        let remaining_requests = u64_header(headers, "x-ratelimit-remaining-requests")
            .or_else(|| u64_header(headers, "x-ratelimit-remaining-request"));
        let limit_requests = u64_header(headers, "x-ratelimit-limit-requests")
            .or_else(|| u64_header(headers, "x-ratelimit-limit-request"));
        if remaining_tokens.is_none()
            && limit_tokens.is_none()
            && remaining_requests.is_none()
            && limit_requests.is_none()
        {
            return None;
        }
        Some(ProviderQuotaSnapshot {
            remaining_tokens,
            limit_tokens,
            remaining_requests,
            limit_requests,
        })
    }

    fn parse_quota_from_body(&self, body: &serde_json::Value) -> Option<ProviderQuotaSnapshot> {
        let usage = body.get("usage");
        let rate = body.get("rate_limit").or_else(|| body.get("rateLimit"));
        let remaining_tokens = rate
            .and_then(|r| {
                r.get("remaining_tokens")
                    .or_else(|| r.get("remainingTokens"))
            })
            .and_then(|v| v.as_u64());
        let limit_tokens = rate
            .and_then(|r| r.get("limit_tokens").or_else(|| r.get("limitTokens")))
            .and_then(|v| v.as_u64());
        let remaining_requests = rate
            .and_then(|r| {
                r.get("remaining_requests")
                    .or_else(|| r.get("remainingRequests"))
            })
            .and_then(|v| v.as_u64());
        let limit_requests = rate
            .and_then(|r| r.get("limit_requests").or_else(|| r.get("limitRequests")))
            .and_then(|v| v.as_u64());
        if usage.is_none()
            && remaining_tokens.is_none()
            && limit_tokens.is_none()
            && remaining_requests.is_none()
            && limit_requests.is_none()
        {
            return None;
        }
        Some(ProviderQuotaSnapshot {
            remaining_tokens,
            limit_tokens,
            remaining_requests,
            limit_requests,
        })
    }

    fn fetch_account_status(
        &self,
        client: &reqwest::Client,
        api_key: &str,
        base_url: &str,
    ) -> futures_util::future::BoxFuture<'static, anyhow::Result<Option<ProviderQuotaSnapshot>>>
    {
        let api_key = api_key.to_string();
        let base_url = base_url.to_string();
        let client = client.clone();
        Box::pin(async move {
            let adapter = OpenAiCompatibleAdapter;
            let url = adapter.models_endpoint(&base_url);
            let mut request = client.get(&url);
            for (k, v) in adapter.auth_headers(&api_key) {
                request = request.header(k, v);
            }
            let response = request.send().await?;
            let header_quota = adapter.parse_quota_from_headers(response.headers());
            let body: serde_json::Value = response.json().await.unwrap_or_default();
            let body_quota = adapter.parse_quota_from_body(&body);
            Ok(adapter.merge_quota(header_quota, body_quota))
        })
    }
}

#[derive(Debug, Default)]
pub struct AnthropicAdapter;

impl ProviderAdapter for AnthropicAdapter {
    fn chat_endpoint(&self, base_url: &str) -> String {
        format!("{}/messages", base_url.trim_end_matches('/'))
    }

    fn models_endpoint(&self, base_url: &str) -> String {
        format!("{}/models", base_url.trim_end_matches('/'))
    }

    fn auth_headers(&self, api_key: &str) -> Vec<(&'static str, String)> {
        vec![
            ("x-api-key", api_key.to_string()),
            ("anthropic-version", "2023-06-01".to_string()),
        ]
    }

    fn build_chat_payload(
        &self,
        model: &str,
        messages: &[serde_json::Value],
        options: &ProviderRequestOptions,
    ) -> serde_json::Value {
        let mut system: Vec<String> = Vec::new();
        let mut non_system: Vec<serde_json::Value> = Vec::new();
        for msg in messages {
            let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or_default();
            if role == "system" {
                if let Some(content) = msg.get("content").and_then(|c| c.as_str()) {
                    if !content.trim().is_empty() {
                        system.push(content.to_string());
                    }
                }
            } else {
                non_system.push(msg.clone());
            }
        }
        serde_json::json!({
            "model": model,
            "system": if system.is_empty() { None } else { Some(system.join("\n\n")) },
            "messages": non_system,
            "stream": options.stream,
            "max_tokens": options.max_tokens.unwrap_or(1024)
        })
    }

    fn stream_done(&self, data: &str) -> bool {
        serde_json::from_str::<serde_json::Value>(data)
            .ok()
            .and_then(|v| {
                v.get("type")
                    .and_then(|t| t.as_str())
                    .map(|s| s == "message_stop")
            })
            .unwrap_or(false)
    }

    fn stream_delta_text(&self, data: &str) -> Option<String> {
        let parsed = serde_json::from_str::<serde_json::Value>(data).ok()?;
        let is_delta = parsed
            .get("type")
            .and_then(|t| t.as_str())
            .map(|t| t == "content_block_delta")
            .unwrap_or(false);
        if !is_delta {
            return None;
        }
        parsed
            .get("delta")
            .and_then(|d| d.get("text"))
            .and_then(|t| t.as_str())
            .map(|s| s.to_string())
    }

    fn parse_non_stream_text(&self, body: &serde_json::Value) -> Option<String> {
        body.get("content")
            .and_then(|c| c.as_array())
            .and_then(|arr| arr.first())
            .and_then(|part| part.get("text"))
            .and_then(|t| t.as_str())
            .map(|s| s.to_string())
    }

    fn parse_quota_from_headers(&self, headers: &HeaderMap) -> Option<ProviderQuotaSnapshot> {
        fn u64_header(headers: &HeaderMap, key: &str) -> Option<u64> {
            headers
                .get(key)
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.trim().parse::<u64>().ok())
        }
        let remaining_tokens = u64_header(headers, "anthropic-ratelimit-tokens-remaining");
        let limit_tokens = u64_header(headers, "anthropic-ratelimit-tokens-limit");
        let remaining_requests = u64_header(headers, "anthropic-ratelimit-requests-remaining");
        let limit_requests = u64_header(headers, "anthropic-ratelimit-requests-limit");
        if remaining_tokens.is_none()
            && limit_tokens.is_none()
            && remaining_requests.is_none()
            && limit_requests.is_none()
        {
            return None;
        }
        Some(ProviderQuotaSnapshot {
            remaining_tokens,
            limit_tokens,
            remaining_requests,
            limit_requests,
        })
    }

    fn parse_quota_from_body(&self, body: &serde_json::Value) -> Option<ProviderQuotaSnapshot> {
        let usage = body.get("usage");
        let rate = body.get("rate_limit").or_else(|| body.get("rateLimit"));
        let remaining_tokens = rate
            .and_then(|r| {
                r.get("tokens_remaining")
                    .or_else(|| r.get("remaining_tokens"))
            })
            .and_then(|v| v.as_u64());
        let limit_tokens = rate
            .and_then(|r| r.get("tokens_limit").or_else(|| r.get("limit_tokens")))
            .and_then(|v| v.as_u64());
        let remaining_requests = rate
            .and_then(|r| {
                r.get("requests_remaining")
                    .or_else(|| r.get("remaining_requests"))
            })
            .and_then(|v| v.as_u64());
        let limit_requests = rate
            .and_then(|r| r.get("requests_limit").or_else(|| r.get("limit_requests")))
            .and_then(|v| v.as_u64());
        if usage.is_none()
            && remaining_tokens.is_none()
            && limit_tokens.is_none()
            && remaining_requests.is_none()
            && limit_requests.is_none()
        {
            return None;
        }
        Some(ProviderQuotaSnapshot {
            remaining_tokens,
            limit_tokens,
            remaining_requests,
            limit_requests,
        })
    }

    fn fetch_account_status(
        &self,
        client: &reqwest::Client,
        api_key: &str,
        base_url: &str,
    ) -> futures_util::future::BoxFuture<'static, anyhow::Result<Option<ProviderQuotaSnapshot>>>
    {
        let api_key = api_key.to_string();
        let base_url = base_url.to_string();
        let client = client.clone();
        Box::pin(async move {
            let adapter = AnthropicAdapter;
            let url = adapter.models_endpoint(&base_url);
            let mut request = client.get(&url);
            for (k, v) in adapter.auth_headers(&api_key) {
                request = request.header(k, v);
            }
            let response = request.send().await?;
            let header_quota = adapter.parse_quota_from_headers(response.headers());
            let body: serde_json::Value = response.json().await.unwrap_or_default();
            let body_quota = adapter.parse_quota_from_body(&body);
            Ok(adapter.merge_quota(header_quota, body_quota))
        })
    }
}

pub fn provider_adapter_for(provider_key: &str) -> Arc<dyn ProviderAdapter> {
    match provider_key {
        "anthropic" => Arc::new(AnthropicAdapter),
        "deepseek" | "huggingface" | "openai" | "openrouter" | "nvidia" | "custom" => {
            Arc::new(OpenAiCompatibleAdapter)
        }
        _ => Arc::new(OpenAiCompatibleAdapter),
    }
}
