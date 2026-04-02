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
    pub json_schema_drift: bool,
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
    GenerateDocument,
    RunCmd,
    RunAndObserve,
    SearchWeb,
    ReadUrl,
    CaptureScreen,
    McpConnect,
    McpListTools,
    McpCallTool,
    McpDisconnect,
}

/// Unified parameter bag for all supported actions with strict unknown-field handling.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CommandParams {
    pub path: Option<String>,
    pub content: Option<String>,
    pub mode: Option<String>,
    pub title: Option<String>,
    pub format: Option<String>,
    pub markdown_content: Option<String>,
    pub command: Option<String>,
    pub delay_secs: Option<u64>,
    pub query: Option<String>,
    pub url: Option<String>,
    pub target: Option<String>,
    pub server_id: Option<String>,
    pub mcp_command: Option<String>,
    pub mcp_args: Option<Vec<String>>,
    pub tool: Option<String>,
    pub arguments: Option<serde_json::Value>,
}

/// Parses a model response that may contain MESSAGE, PLAN, and fenced JSON actions.
pub fn parse_response(raw: &str) -> ParsedResponse {
    let json_block = extract_json_block(raw);
    let cleaned_text = strip_json_block(raw).trim().to_string();

    let (message, plan_items) = extract_message_and_plan(&cleaned_text);

    let (actions, json_parse_error) = if let Some(block) = json_block {
        match serde_json::from_str::<ActionEnvelope>(&block) {
            Ok(envelope) => (envelope.actions, None),
            Err(err) => (Vec::new(), Some(format!("JSON parse failed: {err}"))),
        }
    } else {
        (Vec::new(), None)
    };

    let json_schema_drift = json_parse_error
        .as_deref()
        .map(is_schema_drift_error)
        .unwrap_or(false);

    ParsedResponse {
        message,
        plan_items,
        actions,
        json_parse_error,
        json_schema_drift,
        fallback_text: cleaned_text,
    }
}

pub fn parser_self_correction_feedback() -> &'static str {
    "[System Error]: Action parser failed. You outputted an invalid key. Remember, you must strictly use valid action keys like 'create_file', 'edit_file', 'generate_document', 'run_cmd', 'run_and_observe', 'search_web', 'read_url', 'capture_screen', 'mcp_connect', 'mcp_list_tools', 'mcp_call_tool', or 'mcp_disconnect'. Please correct your JSON and try again."
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
        let normalized = if let Some(rest) = trimmed.strip_prefix("- [ ]") {
            rest.trim()
        } else if let Some(rest) = trimmed.strip_prefix("- [x]") {
            rest.trim()
        } else if let Some(rest) = trimmed.strip_prefix("- [X]") {
            rest.trim()
        } else if let Some(rest) = trimmed.strip_prefix("- ") {
            rest.trim()
        } else if let Some(rest) = trimmed.strip_prefix("* ") {
            rest.trim()
        } else {
            trimmed
        };

        if !normalized.is_empty() {
            items.push(normalized.to_string());
        }
    }

    items
}

fn clean_inline_section(text: &str) -> String {
    text.trim().to_string()
}

fn extract_json_block(text: &str) -> Option<String> {
    let captures = json_fence_regex().captures(text)?;
    let fenced_body = captures.name("body")?.as_str();
    let sanitized = sanitize_fenced_json_body(fenced_body);
    extract_balanced_json_object(&sanitized).map(str::to_string)
}

fn strip_json_block(text: &str) -> String {
    json_fence_regex().replace_all(text, "").to_string()
}

fn json_fence_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?is)```json\s*(?P<body>[\s\S]*?)```").expect("json fence regex must be valid")
    })
}

fn tag_matcher_message() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?im)^\s*(?:\*\*)?\s*MESSAGE\s*(?:\*\*)?\s*:\s*")
            .expect("message regex must be valid")
    })
}

fn tag_matcher_plan() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?im)^\s*(?:\*\*)?\s*PLAN\s*(?:\*\*)?\s*:\s*")
            .expect("plan regex must be valid")
    })
}

fn sanitize_fenced_json_body(body: &str) -> String {
    let without_bom = body.trim().trim_start_matches('\u{feff}');
    let mut cleaned = String::new();

    for line in without_bom.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("```") {
            continue;
        }
        cleaned.push_str(line);
        cleaned.push('\n');
    }

    cleaned.trim().to_string()
}

fn extract_balanced_json_object(input: &str) -> Option<&str> {
    let bytes = input.as_bytes();
    let mut idx = 0usize;

    while idx < bytes.len() && bytes[idx].is_ascii_whitespace() {
        idx += 1;
    }
    if idx >= bytes.len() || bytes[idx] != b'{' {
        return None;
    }

    let start = idx;
    let mut depth = 1i32;
    let mut in_string = false;
    let mut escaped = false;

    for (offset, ch) in input[start + 1..].char_indices() {
        if in_string {
            if escaped {
                escaped = false;
                continue;
            }
            match ch {
                '\\' => escaped = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }

        match ch {
            '"' => in_string = true,
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    let end = start + 1 + offset + ch.len_utf8();
                    return input.get(start..end);
                }
            }
            _ => {}
        }
    }

    None
}

fn is_schema_drift_error(err: &str) -> bool {
    err.contains("unknown variant")
        || err.contains("unknown field")
        || err.contains("did not match any variant")
}
