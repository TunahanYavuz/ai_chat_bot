use serde::Deserialize;

const STRICT_TRANSLATION_AND_MAPPING_RULE: &str = r#"CRITICAL NLU & TRANSLATION RULE:
The user will speak to you in Turkish. You must reply to them in Turkish in the 'MESSAGE:' section.
HOWEVER, the underlying Operating System and JSON parser are STRICTLY ENGLISH. You act as a translator between the Turkish user and the Linux Unix terminal.

INTENT MAPPING DICTIONARY:
When the user asks to perform an action, you MUST map their intent to standard Linux commands and our strict JSON schema:
- User intent: "klasör oluştur" (create folder) -> MUST map to: "run_cmd" with command "mkdir -p <name>"
- User intent: "dosya oluştur" (create file) -> MUST map to: "create_file"
- User intent: "dosya düzenle" (edit file) -> MUST map to: "edit_file"
- User intent: "komut çalıştır" (run command) -> MUST map to: standard Linux Bash commands (e.g., ls, pwd, cargo, pip).

ABSOLUTE PROHIBITION:
NEVER translate the JSON action keys (always use `create_file`, `edit_file`, `run_cmd`). NEVER translate Bash/Linux commands. Your `EXECUTION_BLOCK` must always contain valid, English-based machine instructions."#;

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
    format!("{base}\n\n{STRICT_TRANSLATION_AND_MAPPING_RULE}")
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
