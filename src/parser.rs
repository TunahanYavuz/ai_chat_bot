use std::sync::OnceLock;

use regex::Regex;
use serde::Deserialize;

/// Parsed high-level response sections expected from the model.
#[derive(Debug, Clone)]
pub struct ParsedResponse {
    pub message: Option<String>,
    pub plan_items: Vec<String>,
    pub actions: Vec<AgentAction>,
    pub json_parse_error: Option<String>,
    pub fallback_text: String,
}

/// Top-level execution payload.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActionEnvelope {
    pub actions: Vec<AgentAction>,
}

/// One executable action emitted by the model.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgentAction {
    pub action: ActionKind,
    pub parameters: CommandParams,
}

/// Supported action names from the model JSON.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionKind {
    CreateFolder,
    CreateFile,
    EditFile,
    CreatePdf,
    RunCmd,
}

/// Unified parameter bag for all supported actions with strict unknown-field handling.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CommandParams {
    pub path: Option<String>,
    pub content: Option<String>,
    pub mode: Option<String>,
    pub title: Option<String>,
    pub command: Option<String>,
}

/// Parses a model response that may contain MESSAGE, PLAN, and fenced JSON actions.
pub fn parse_response(raw: &str) -> ParsedResponse {
    let json_block = extract_json_block(raw);
    let cleaned_text = strip_json_block(raw).trim().to_string();

    let (message, plan_items) = extract_message_and_plan(&cleaned_text);

    let (actions, json_parse_error) = if let Some(block) = json_block {
        match serde_json::from_str::<ActionEnvelope>(block) {
            Ok(envelope) => (envelope.actions, None),
            Err(err) => (Vec::new(), Some(format!("JSON parse failed: {err}"))),
        }
    } else {
        (Vec::new(), None)
    };

    ParsedResponse {
        message,
        plan_items,
        actions,
        json_parse_error,
        fallback_text: cleaned_text,
    }
}

fn extract_message_and_plan(text: &str) -> (Option<String>, Vec<String>) {
    let message_start = tag_matcher_message().find(text).map(|m| m.start());
    let plan_start = tag_matcher_plan().find(text).map(|m| m.start());

    match (message_start, plan_start) {
        (Some(m_start), Some(p_start)) if m_start < p_start => {
            let message = section_after_tag_until(text, m_start, p_start, tag_matcher_message())
                .map(clean_inline_section)
                .filter(|s| !s.is_empty());
            let plan_text = section_after_tag_until(text, p_start, text.len(), tag_matcher_plan())
                .unwrap_or_default();
            (message, parse_plan_items(plan_text))
        }
        (Some(m_start), Some(p_start)) => {
            let plan_text = section_after_tag_until(text, p_start, m_start, tag_matcher_plan())
                .unwrap_or_default();
            let message = section_after_tag_until(text, m_start, text.len(), tag_matcher_message())
                .map(clean_inline_section)
                .filter(|s| !s.is_empty());
            (message, parse_plan_items(plan_text))
        }
        (Some(m_start), None) => {
            let message = section_after_tag_until(text, m_start, text.len(), tag_matcher_message())
                .map(clean_inline_section)
                .filter(|s| !s.is_empty());
            (message, Vec::new())
        }
        (None, Some(p_start)) => {
            let plan_text = section_after_tag_until(text, p_start, text.len(), tag_matcher_plan())
                .unwrap_or_default();
            (None, parse_plan_items(plan_text))
        }
        (None, None) => {
            let fallback = text.trim();
            if fallback.is_empty() {
                (None, Vec::new())
            } else {
                (Some(fallback.to_string()), Vec::new())
            }
        }
    }
}

fn section_after_tag_until<'a>(
    text: &'a str,
    section_start: usize,
    section_end: usize,
    tag_regex: &Regex,
) -> Option<&'a str> {
    let range = text.get(section_start..section_end)?;
    let tag_match = tag_regex.find(range)?;
    let content_start = section_start + tag_match.end();
    text.get(content_start..section_end)
}

fn parse_plan_items(plan_text: &str) -> Vec<String> {
    let mut items = Vec::new();

    for line in plan_text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Accept common checklist/bullet formats and normalize to plain text.
        let normalized = trimmed
            .trim_start_matches("- [ ]")
            .trim_start_matches("- [x]")
            .trim_start_matches("- [X]")
            .trim_start_matches("- ")
            .trim_start_matches("* ")
            .trim();

        if !normalized.is_empty() {
            items.push(normalized.to_string());
        }
    }

    items
}

fn clean_inline_section(text: &str) -> String {
    text.trim().trim_matches('\n').trim().to_string()
}

fn extract_json_block(text: &str) -> Option<&str> {
    let captures = json_block_regex().captures(text)?;
    captures.name("json").map(|m| m.as_str())
}

fn strip_json_block(text: &str) -> String {
    json_block_regex().replace_all(text, "").to_string()
}

fn json_block_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"```json\s*(?P<json>\{[\s\S]*?\})\s*```")
            .expect("json block regex must be valid")
    })
}

fn tag_matcher_message() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?im)^\s*MESSAGE:\s*").expect("message regex must be valid"))
}

fn tag_matcher_plan() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?im)^\s*PLAN:\s*").expect("plan regex must be valid"))
}
