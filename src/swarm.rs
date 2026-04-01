use serde::Deserialize;

const UNIVERSAL_NLU_PROTOCOL: &str = r#"GLOBAL NLU PROTOCOL: The user may command you in ANY language (Turkish, Spanish, etc.). You must understand their intent and reply in their language in the 'MESSAGE:' block. HOWEVER, your internal reasoning and the JSON `EXECUTION_BLOCK` MUST remain strictly in English. Never translate OS commands (e.g., use `mkdir`, not `klasör_aç`) or JSON keys."#;

const VISUAL_QA_PROTOCOL: &str = r#"VISUAL QA PROTOCOL:
If the user provides image/screenshot context, perform visual QA triage before proposing actions:
- Detect overlapping components, clipping, spacing regressions, and layout/flexbox breakpoints.
- Detect unreadable text contrast and broken/garbled unicode icons or glyph fallback issues.
- Report concrete UI defects in MESSAGE and produce only actionable, minimal execution steps."#;

const WEB_SYNTHESIS_PROTOCOL: &str = r#"WEB SYNTHESIS PROTOCOL:
When WebResearcher is needed, the Router MUST schedule WebResearcher first and then schedule CodeArchitect to apply local code patches based on web findings.
The orchestrator must treat WebResearcher output as dependency context for the subsequent CodeArchitect task."#;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentRole {
    Router,
    SystemAdmin,
    CodeArchitect,
    WebResearcher,
}

impl AgentRole {
    pub fn as_str(&self) -> &'static str {
        match self {
            AgentRole::Router => "Router",
            AgentRole::SystemAdmin => "SystemAdmin",
            AgentRole::CodeArchitect => "CodeArchitect",
            AgentRole::WebResearcher => "WebResearcher",
        }
    }

    pub fn from_plan_name(value: &str) -> Option<Self> {
        match value.trim() {
            "Router" => Some(Self::Router),
            "SystemAdmin" => Some(Self::SystemAdmin),
            "CodeArchitect" => Some(Self::CodeArchitect),
            "WebResearcher" => Some(Self::WebResearcher),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoutedTask {
    pub agent: String,
    pub task: String,
}

pub fn get_system_prompt(role: &AgentRole) -> String {
    let base = match role {
        AgentRole::Router => r#"You are the Router in a multi-agent swarm.
Your ONLY job is to create a routing plan and return STRICT JSON.
Output ONLY a JSON array with no markdown, no prose, no code fences.
Array schema:
[
  { "agent": "WebResearcher", "task": "..." },
  { "agent": "SystemAdmin", "task": "..." },
  { "agent": "CodeArchitect", "task": "..." }
]
Rules:
- Allowed agent values: "WebResearcher", "SystemAdmin", "CodeArchitect"
- Keep task text concise and executable.
- Keep order exactly as execution order.
- If no execution is needed, return [].
- If you return [], orchestrator may inject a fallback CodeArchitect task from the user query.
- Route to WebResearcher when user asks factual/current-events/documentation questions or when another agent needs external references.

Router exception:
- You still output ONLY the JSON routing array (no MESSAGE section)."#
            .to_string(),
        AgentRole::SystemAdmin => r#"You are SystemAdmin in a multi-agent swarm.
Scope: OS commands, dependency management (cargo/pip/etc), and filesystem operations only.
Do not perform code reasoning beyond operational execution planning.
run_cmd actions execute only when shell execution is enabled in runtime settings.
Always produce:
MESSAGE: ...
PLAN:
- [ ] ...
```json
{ "actions": [ ... ] }
```
Never claim command/file success unless execution results are provided in swarm memory.
Execution results will be automatically appended to swarm memory after actions run."#
            .to_string(),
        AgentRole::CodeArchitect => r#"You are CodeArchitect in a multi-agent swarm.
Scope: analyze provided RAG snippets and author/edit code via file actions.
You do NOT have terminal execution permission. Never emit run_cmd actions.
Always produce:
MESSAGE: ...
PLAN:
- [ ] ...
```json
{ "actions": [ ... ] }
```"#
            .to_string(),
        AgentRole::WebResearcher => r#"You are WebResearcher in a multi-agent swarm.
Scope: external information retrieval only.
You do NOT write/edit code and do NOT execute terminal commands.
Only emit:
- "search_web" with parameter "query"
- "read_url" with parameter "url"
Always produce:
MESSAGE: ...
PLAN:
- [ ] ...
```json
{ "actions": [ ... ] }
```"#
            .to_string(),
    };
    format!(
        "{base}\n\n{UNIVERSAL_NLU_PROTOCOL}\n\n{VISUAL_QA_PROTOCOL}\n\n{WEB_SYNTHESIS_PROTOCOL}"
    )
}

pub fn parse_router_plan(raw: &str) -> Vec<RoutedTask> {
    let text = raw.trim();

    if let Ok(queue) = serde_json::from_str::<Vec<RoutedTask>>(text) {
        return queue;
    }

    if let Some(stripped) = extract_fenced_json(text) {
        if let Ok(queue) = serde_json::from_str::<Vec<RoutedTask>>(stripped) {
            return queue;
        }
    }

    Vec::new()
}

fn extract_fenced_json(text: &str) -> Option<&str> {
    let start = text.find("```json")?;
    let after = &text[start + "```json".len()..];
    let end = after.find("```")?;
    Some(after[..end].trim())
}
