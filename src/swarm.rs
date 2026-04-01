use serde::Deserialize;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentRole {
    Router,
    SystemAdmin,
    CodeArchitect,
}

impl AgentRole {
    pub fn as_str(&self) -> &'static str {
        match self {
            AgentRole::Router => "Router",
            AgentRole::SystemAdmin => "SystemAdmin",
            AgentRole::CodeArchitect => "CodeArchitect",
        }
    }

    pub fn from_plan_name(value: &str) -> Option<Self> {
        match value.trim() {
            "Router" => Some(Self::Router),
            "SystemAdmin" => Some(Self::SystemAdmin),
            "CodeArchitect" => Some(Self::CodeArchitect),
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
    match role {
        AgentRole::Router => r#"You are the Router in a multi-agent swarm.
Your ONLY job is to create a routing plan and return STRICT JSON.
Output ONLY a JSON array with no markdown, no prose, no code fences.
Array schema:
[
  { "agent": "SystemAdmin", "task": "..." },
  { "agent": "CodeArchitect", "task": "..." }
]
Rules:
- Allowed agent values: "SystemAdmin", "CodeArchitect"
- Keep task text concise and executable.
- Keep order exactly as execution order.
- If no execution is needed, return [].
- If you return [], orchestrator may inject a fallback CodeArchitect task from the user query."#
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
Never claim command/file success unless execution results are provided in swarm memory."#
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
    }
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
