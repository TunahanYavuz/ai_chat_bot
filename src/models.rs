use serde::{Deserialize, Serialize};

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

    let known_mappings: [(&str, ModelReasoningConfig); 20] = [
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
        ("qwen1.5", REASONING_BINARY),
        ("qwen-long", REASONING_BINARY),
    ];

    if let Some((_, cfg)) = known_mappings.iter().find(|(prefix, _)| m.starts_with(prefix)) {
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

    REASONING_NONE
}

pub fn get_model_capability(model_name: &str) -> ReasoningCapability {
    reasoning_config_for_model(model_name).capability
}
