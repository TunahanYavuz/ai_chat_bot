use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use base64::{engine::general_purpose, Engine as _};
use chrono::Utc;
use egui::text::LayoutJob;
use egui::{Color32, FontId, RichText, ScrollArea, TextEdit, TextFormat, Vec2};
use egui_commonmark::{CommonMarkCache, CommonMarkViewer};
use serde::Deserialize;
use uuid::Uuid;

use crate::api::{builtin_models, provider_models, ChatMessage, ModelInfo, RemoteModelInfo, ThinkingMode};
use crate::config::{load_settings, save_settings, ApiProvider, Settings, DEFAULT_MODEL_ID};
use crate::db::{Database, DbFileSnapshot, DbMessage, DbSession};
use crate::executor::{ActionExecutor, ExecutionStatus};
use crate::parser::{ActionKind, AgentAction, CommandParams};
use crate::rag_engine::{EmbeddingProvider, RagConfig, RagEngine};
use crate::setup::SetupWizard;
use crate::storage::{ChatSession, Storage, StoredMessage, StoredRole};
use crate::telemetry::collect_telemetry_cached;

const BG_PRIMARY: Color32 = Color32::from_rgb(0x1E, 0x1E, 0x1E);
const BG_SURFACE: Color32 = Color32::from_rgb(0x2D, 0x2D, 0x2D);
const BG_SURFACE_ALT: Color32 = Color32::from_rgb(0x36, 0x36, 0x36);
const ACCENT: Color32 = Color32::from_rgb(0x56, 0x9A, 0xD6);
const ACCENT_SOFT: Color32 = Color32::from_rgb(0x3D, 0x6A, 0x8C);
const TEXT_PRIMARY: Color32 = Color32::from_rgb(0xE0, 0xE0, 0xE0);
const TEXT_MUTED: Color32 = Color32::from_rgb(0xB5, 0xB5, 0xB5);
const TEXT_DARK: Color32 = Color32::from_rgb(0x10, 0x10, 0x10);
const BURGUNDY: Color32 = BG_PRIMARY;
const BURGUNDY_DARK: Color32 = BG_SURFACE;
const SKY_BLUE: Color32 = ACCENT;
const SKY_BLUE_DARK: Color32 = ACCENT_SOFT;
const GOLD: Color32 = Color32::from_rgb(0xD9, 0xD9, 0xD9);
const GOLD_DARK: Color32 = Color32::from_rgb(0x7A, 0x7A, 0x7A);
const WHITE: Color32 = TEXT_PRIMARY;
const DARK_TEXT: Color32 = TEXT_DARK;
const LIGHT_TEXT: Color32 = TEXT_PRIMARY;
const CUSTOM_MODEL_INPUT_RESERVED_WIDTH: f32 = 132.0;
const DELETE_CHAT_BUTTON_WIDTH: f32 = 36.0;
const MIN_CHAT_BUTTON_WIDTH: f32 = 80.0;
const TERMINAL_INPUT_RESERVED_WIDTH: f32 = 260.0;
/// Prepended as the first system prompt for API requests so responses include
/// a user-facing message plus a machine-readable ```json ... ``` execution block.
const CORE_OS_SYSTEM_PROMPT: &str = r#"You are 'CoreOS', an advanced local File System and Command Line Interface (CLI) Agent. Your purpose is to assist the user by generating precise, executable commands while maintaining a helpful, conversational tone.

YOUR WORKFLOW:
Whenever the user makes a request, you must output your response in EXACTLY two parts:
1. "MESSAGE": A friendly, brief, natural language response explaining what you are about to do.
2. "EXECUTION_BLOCK": A strictly formatted JSON block containing the system instructions. This block MUST be enclosed in ```json ... ``` tags.

AVAILABLE ACTIONS:
You can output the following actions inside the JSON array:
- "create_folder": Creates a new directory. (Requires: 'path')
- "create_file": Creates a new text-based file (.txt, .rs, .py, etc.). (Requires: 'path', 'content')
- "edit_file": Appends or overwrites text in an existing file. (Requires: 'path', 'mode' [overwrite/append], 'content')
- "create_pdf": Generates a PDF file. (Requires: 'path', 'title', 'content')
- "run_cmd": Executes a terminal command. (Requires: 'command')

JSON SCHEMA OBLIGATION:
Your JSON must strictly follow this format:
{
  "actions": [
    {
      "action": "action_name",
      "parameters": {
        "key": "value"
      }
    }
  ]
}

CRITICAL RULES:

Never output JSON outside the json block.

Always use valid, escaped characters inside the JSON payload.

GLOBAL NLU PROTOCOL: The user may speak to you in ANY language (e.g., Turkish, Spanish, etc.). You must understand and execute their intent. However, your structural output MUST remain strictly in English. The headers MUST be exactly 'MESSAGE:' and 'PLAN:'. The JSON keys and actions (e.g., 'create_file', 'run_cmd') MUST remain exactly as defined in the schema. You may translate the *content* of the MESSAGE and PLAN into the user's language, but NEVER translate the structural keys.

If a user asks a simple conversational question that does not require system execution, keep the actions array empty ([]).

THINK BEFORE YOU ACT: Ensure the commands are safe for a Linux/Unix environment."#;

#[derive(Debug)]
pub enum AppEvent {
    ChunkReceived(String, String),
    ResponseComplete(String),
    ResponseError(String, String),
    ModelsLoaded(Vec<RemoteModelInfo>),
    TerminalFinished {
        terminal_id: String,
        stdout: String,
        stderr: String,
        exit_code: i32,
    },
    InitialSessionsLoaded {
        sessions: Vec<Session>,
        error: Option<String>,
    },
    SessionMessagesLoaded {
        idx: usize,
        messages: Vec<Message>,
        error: Option<String>,
    },
    StoredSessionLoaded {
        session: Option<Session>,
        error: Option<String>,
    },
    SnapshotsLoaded {
        snapshots: Vec<DbFileSnapshot>,
        error: Option<String>,
    },
    SnapshotRestored {
        file_path: String,
        error: Option<String>,
    },
    DbError(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
    System,
}

impl Role {
    pub fn as_str(&self) -> &'static str {
        match self {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::System => "system",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Attachment {
    pub filename: String,
    pub text: Option<String>,
    pub image_base64: Option<String>,
    pub mime_type: String,
    pub raw_bytes: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct Message {
    pub id: String,
    pub role: Role,
    pub content: String,
    pub attachments: Vec<Attachment>,
    pub timestamp: chrono::DateTime<Utc>,
    pub is_streaming: bool,
}

#[derive(Debug, Clone)]
pub struct Session {
    pub id: String,
    pub name: String,
    pub messages: Vec<Message>,
}

impl Session {
    pub fn new(name: &str) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            name: name.to_string(),
            messages: vec![],
        }
    }
}

#[derive(Debug, Clone)]
pub enum PendingAction {
    WriteFile { path: String, content: String },
    ExecuteCommand { command: String },
    CreateFolder { path: String },
    EditFile {
        path: String,
        mode: String,
        content: String,
    },
    CreatePdf {
        path: String,
        title: String,
        content: String,
    },
}

#[derive(Debug, Clone)]
pub enum NotificationKind {
    Info,
    Error,
}

#[derive(Debug, Clone)]
pub struct Notification {
    pub id: String,
    pub text: String,
    pub kind: NotificationKind,
    pub ttl_secs: f32,
}

#[derive(Debug, Clone)]
pub struct AgentProfile {
    pub name: String,
    pub system_prompt: String,
    pub enabled: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TerminalOwner {
    AI,
    User,
}

#[derive(Debug, Clone)]
pub struct TerminalSession {
    pub id: String,
    pub name: String,
    pub owner: TerminalOwner,
    pub command: String,
    pub working_dir: String,
    pub output: String,
    pub running: bool,
    pub terminated: bool,
    pub exit_code: Option<i32>,
    pub pid: Option<u32>,
    pub linked_session_idx: Option<usize>,
    pub auto_continue_agent: bool,
    pub input_buffer: String,
}

#[derive(Debug, Clone)]
struct ActiveRequest {
    session_idx: usize,
    message_id: String,
    agent_name: String,
}

#[derive(Clone)]
struct OpenAIEmbeddingProvider {
    api_key: String,
    base_url: String,
    model: String,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingItem>,
}

#[derive(Deserialize)]
struct EmbeddingItem {
    embedding: Vec<f32>,
}

#[async_trait::async_trait]
impl EmbeddingProvider for OpenAIEmbeddingProvider {
    async fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        let client = reqwest::Client::new();
        let url = format!("{}/embeddings", self.base_url.trim_end_matches('/'));
        let response = client
            .post(url)
            .bearer_auth(&self.api_key)
            .json(&serde_json::json!({
                "model": self.model,
                "input": text,
            }))
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!("embedding API error {}: {}", status, body));
        }

        let parsed: EmbeddingResponse = response.json().await?;
        let embedding = parsed
            .data
            .into_iter()
            .next()
            .map(|d| d.embedding)
            .ok_or_else(|| anyhow::anyhow!("empty embedding response"))?;
        Ok(embedding)
    }
}

pub struct ChatApp {
    settings: Settings,
    show_settings: bool,
    setup_wizard: Option<SetupWizard>,

    sessions: Vec<Session>,
    current_session_idx: Option<usize>,

    models: Vec<ModelInfo>,
    selected_model_idx: usize,
    remote_models: Vec<RemoteModelInfo>,
    model_waiting_command_output: bool,
    model_access_outline_ok: Option<bool>,
    custom_model_id: String,
    selected_thinking_mode: ThinkingMode,

    input_text: String,
    pending_attachments: Vec<Attachment>,

    event_tx: tokio::sync::mpsc::Sender<AppEvent>,
    event_rx: tokio::sync::mpsc::Receiver<AppEvent>,
    tokio_rt: Arc<tokio::runtime::Runtime>,
    active_requests: HashMap<String, ActiveRequest>,

    db_path: String,
    storage: Option<Storage>,
    pending_action: Option<PendingAction>,
    pending_actions_queue: VecDeque<PendingAction>,
    pending_action_session_idx: Option<usize>,

    markdown_cache: CommonMarkCache,
    notifications: Vec<Notification>,
    activity_log: Vec<String>,

    show_snapshots: bool,
    snapshots: Vec<DbFileSnapshot>,

    agents: Vec<AgentProfile>,
    terminals: Vec<TerminalSession>,
    active_terminal_id: Option<String>,
    changed_files: Vec<String>,
    opened_files: Vec<String>,
    active_open_file: Option<String>,
    editor_buffers: HashMap<String, String>,
    editor_dirty: HashMap<String, bool>,
    show_left_sidebar: bool,
    show_activity_panel: bool,
}

impl ChatApp {
    fn stored_role_from_role(role: &Role) -> StoredRole {
        match role {
            Role::System => StoredRole::System,
            Role::User => StoredRole::User,
            Role::Assistant => StoredRole::Assistant,
        }
    }

    fn role_from_stored_role(role: &StoredRole) -> Role {
        match role {
            StoredRole::System => Role::System,
            StoredRole::User => Role::User,
            StoredRole::Assistant => Role::Assistant,
        }
    }

    fn session_to_stored(session: &Session) -> ChatSession {
        ChatSession {
            id: session.id.clone(),
            name: session.name.clone(),
            updated_at: Utc::now(),
            messages: session
                .messages
                .iter()
                .map(|m| StoredMessage {
                    role: Self::stored_role_from_role(&m.role),
                    content: m.content.clone(),
                    timestamp: m.timestamp,
                })
                .collect(),
        }
    }

    fn persist_session_snapshot_async(&self, session_idx: usize) {
        let Some(storage) = self.storage.clone() else {
            return;
        };
        let Some(session) = self.sessions.get(session_idx) else {
            return;
        };
        let payload = Self::session_to_stored(session);
        self.tokio_rt.spawn(async move {
            if let Err(e) = storage.save_session(&payload).await {
                eprintln!("Failed to persist latest session: {e}");
            }
        });
    }

    fn spawn_initial_data_load(&self) {
        let tx = self.event_tx.clone();
        let db_path = self.db_path.clone();
        self.tokio_rt.spawn(async move {
            let result = tokio::task::spawn_blocking(move || -> Result<Vec<Session>, String> {
                let db = Database::new(&db_path).map_err(|e| e.to_string())?;
                let sessions = db
                    .list_sessions()
                    .map_err(|e| e.to_string())?
                    .into_iter()
                    .map(|s| Session {
                        id: s.id,
                        name: s.name,
                        messages: vec![],
                    })
                    .collect();
                Ok(sessions)
            })
            .await;

            match result {
                Ok(Ok(sessions)) => {
                    let _ = tx.send(AppEvent::InitialSessionsLoaded { sessions, error: None }).await;
                }
                Ok(Err(e)) => {
                    let _ = tx
                        .send(AppEvent::InitialSessionsLoaded {
                            sessions: vec![],
                            error: Some(e),
                        })
                        .await;
                }
                Err(e) => {
                    let _ = tx
                        .send(AppEvent::InitialSessionsLoaded {
                            sessions: vec![],
                            error: Some(e.to_string()),
                        })
                        .await;
                }
            }
        });

        if let Some(storage) = self.storage.clone() {
            let tx = self.event_tx.clone();
            self.tokio_rt.spawn(async move {
                match storage.load_latest_session().await {
                    Ok(Some(stored)) => {
                        let restored_messages: Vec<Message> = stored
                            .messages
                            .into_iter()
                            .map(|m| Message {
                                id: Uuid::new_v4().to_string(),
                                role: Self::role_from_stored_role(&m.role),
                                content: m.content,
                                attachments: vec![],
                                timestamp: m.timestamp,
                                is_streaming: false,
                            })
                            .collect();
                        let session = Session {
                            id: stored.id,
                            name: stored.name,
                            messages: restored_messages,
                        };
                        let _ = tx
                            .send(AppEvent::StoredSessionLoaded {
                                session: Some(session),
                                error: None,
                            })
                            .await;
                    }
                    Ok(None) => {
                        let _ = tx
                            .send(AppEvent::StoredSessionLoaded {
                                session: None,
                                error: None,
                            })
                            .await;
                    }
                    Err(e) => {
                        let _ = tx
                            .send(AppEvent::StoredSessionLoaded {
                                session: None,
                                error: Some(e.to_string()),
                            })
                            .await;
                    }
                }
            });
        }
    }

    fn spawn_load_session_messages(&self, idx: usize, session_id: String) {
        let tx = self.event_tx.clone();
        let db_path = self.db_path.clone();
        self.tokio_rt.spawn(async move {
            let result = tokio::task::spawn_blocking(move || -> Result<Vec<Message>, String> {
                let db = Database::new(&db_path).map_err(|e| e.to_string())?;
                let messages = db.load_messages(&session_id).map_err(|e| e.to_string())?;
                let mapped: Vec<Message> = messages
                    .into_iter()
                    .map(|m| {
                        let message_id = m.id;
                        let attachments = db
                            .load_attachments(&message_id)
                            .unwrap_or_default()
                            .into_iter()
                            .map(|att| {
                                let ext = std::path::Path::new(&att.filename)
                                    .extension()
                                    .and_then(|s| s.to_str())
                                    .unwrap_or_default()
                                    .to_lowercase();
                                let is_image = matches!(ext.as_str(), "png" | "jpg" | "jpeg" | "gif" | "webp");
                                Attachment {
                                    filename: att.filename,
                                    text: if is_image {
                                        None
                                    } else {
                                        Some(String::from_utf8_lossy(&att.data).to_string())
                                    },
                                    image_base64: if is_image {
                                        Some(general_purpose::STANDARD.encode(&att.data))
                                    } else {
                                        None
                                    },
                                    mime_type: att.mime_type,
                                    raw_bytes: att.data,
                                }
                            })
                            .collect();
                        Message {
                            id: message_id.clone(),
                            role: match m.role.as_str() {
                                "user" => Role::User,
                                "assistant" => Role::Assistant,
                                _ => Role::System,
                            },
                            content: m.content,
                            attachments,
                            timestamp: m.created_at,
                            is_streaming: false,
                        }
                    })
                    .collect();
                Ok(mapped)
            })
            .await;

            match result {
                Ok(Ok(messages)) => {
                    let _ = tx
                        .send(AppEvent::SessionMessagesLoaded {
                            idx,
                            messages,
                            error: None,
                        })
                        .await;
                }
                Ok(Err(e)) => {
                    let _ = tx
                        .send(AppEvent::SessionMessagesLoaded {
                            idx,
                            messages: vec![],
                            error: Some(e),
                        })
                        .await;
                }
                Err(e) => {
                    let _ = tx
                        .send(AppEvent::SessionMessagesLoaded {
                            idx,
                            messages: vec![],
                            error: Some(e.to_string()),
                        })
                        .await;
                }
            }
        });
    }

    fn sanitized_command_stdout(stdout: &str, exit_code: i32) -> String {
        if stdout.trim().is_empty() {
            if exit_code == 0 {
                "[Command executed successfully with no output]".to_string()
            } else {
                format!("[Command produced no stdout output (exit code: {})]", exit_code)
            }
        } else {
            stdout.to_string()
        }
    }

    fn pending_actions_from_agent_actions(actions: Vec<crate::parser::AgentAction>) -> Vec<PendingAction> {
        actions
            .into_iter()
            .filter_map(|item| match item.action {
                crate::parser::ActionKind::CreateFolder => {
                    let path = item.parameters.path?.trim().to_string();
                    if path.is_empty() {
                        None
                    } else {
                        Some(PendingAction::CreateFolder { path })
                    }
                }
                crate::parser::ActionKind::CreateFile => {
                    let path = item.parameters.path?.trim().to_string();
                    if path.is_empty() {
                        None
                    } else {
                        Some(PendingAction::WriteFile {
                            path,
                            content: item.parameters.content.unwrap_or_default(),
                        })
                    }
                }
                crate::parser::ActionKind::EditFile => {
                    let path = item.parameters.path?.trim().to_string();
                    let mode = item.parameters.mode?.trim().to_string();
                    if path.is_empty() || (mode != "overwrite" && mode != "append") {
                        None
                    } else {
                        Some(PendingAction::EditFile {
                            path,
                            mode,
                            content: item.parameters.content.unwrap_or_default(),
                        })
                    }
                }
                crate::parser::ActionKind::CreatePdf => {
                    let path = item.parameters.path?.trim().to_string();
                    if path.is_empty() {
                        None
                    } else {
                        Some(PendingAction::CreatePdf {
                            path,
                            title: item.parameters.title.unwrap_or_default(),
                            content: item.parameters.content.unwrap_or_default(),
                        })
                    }
                }
                crate::parser::ActionKind::RunCmd => {
                    let command = item.parameters.command?.trim().to_string();
                    if command.is_empty() {
                        None
                    } else {
                        Some(PendingAction::ExecuteCommand { command })
                    }
                }
            })
            .collect()
    }

    fn selected_model_id(&self) -> String {
        if self.custom_model_id.trim().is_empty() {
            self.current_model()
                .map(|m| m.id.clone())
                .unwrap_or_else(|| DEFAULT_MODEL_ID.to_string())
        } else {
            self.custom_model_id.trim().to_string()
        }
    }

    fn refresh_model_access_outline(&mut self) {
        if self.remote_models.is_empty() {
            self.model_access_outline_ok = None;
            return;
        }
        let selected = self.selected_model_id();
        self.model_access_outline_ok = self
            .remote_models
            .iter()
            .find(|m| m.id == selected)
            .map(|m| !(m.gated || m.premium))
            .or(Some(false));
    }

    fn extract_tag_block<'a>(text: &'a str, open_tag: &str, close_tag: &str) -> Option<&'a str> {
        let start = text.find(open_tag)?;
        let content_start = start + open_tag.len();
        let end_rel = text[content_start..].find(close_tag)?;
        let end = content_start + end_rel;
        Some(&text[content_start..end])
    }

    fn parse_pending_action(message: &str) -> Option<PendingAction> {
        let trimmed = message.trim();

        if let Some(start) = trimmed.find("<write_file") {
            let open_end = trimmed[start..].find('>')?;
            let open_tag = &trimmed[start..start + open_end + 1];
            let path_key = "path=\"";
            let path_start = open_tag.find(path_key)? + path_key.len();
            let path_rest = &open_tag[path_start..];
            let path_end = path_rest.find('"')?;
            let path = path_rest[..path_end].trim();
            if path.is_empty() {
                return None;
            }
            let content_start = start + open_end + 1;
            let end_rel = trimmed[content_start..].find("</write_file>")?;
            let end = content_start + end_rel;
            let content = trimmed[content_start..end].to_string();
            return Some(PendingAction::WriteFile {
                path: path.to_string(),
                content,
            });
        }

        if let Some(command_block) =
            Self::extract_tag_block(trimmed, "<execute_command>", "</execute_command>")
        {
            let command = command_block.trim().to_string();
            if command.is_empty() {
                return None;
            }
            return Some(PendingAction::ExecuteCommand { command });
        }

        None
    }

    fn resolve_action_path(&self, raw_path: &str) -> std::path::PathBuf {
        let path = std::path::Path::new(raw_path);
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            std::path::Path::new(&self.settings.working_directory).join(path)
        }
    }

    fn ensure_changed_file_tracked(&mut self, path: String) {
        if !self.changed_files.iter().any(|p| p == &path) {
            self.changed_files.push(path);
        }
    }

    fn open_file_in_workspace(&mut self, path: String) {
        if !self.opened_files.iter().any(|p| p == &path) {
            self.opened_files.push(path.clone());
        }
        if !self.editor_buffers.contains_key(&path) {
            let content = Self::read_text_file_for_view(&path);
            self.editor_buffers.insert(path.clone(), content);
            self.editor_dirty.insert(path.clone(), false);
        }
        self.active_open_file = Some(path);
    }

    fn close_file_in_workspace(&mut self, path: &str) {
        self.opened_files.retain(|p| p != path);
        self.editor_buffers.remove(path);
        self.editor_dirty.remove(path);
        if self.active_open_file.as_deref() == Some(path) {
            self.active_open_file = self.opened_files.last().cloned();
        }
    }

    fn create_terminal(
        &mut self,
        name: String,
        owner: TerminalOwner,
        command: String,
        linked_session_idx: Option<usize>,
        auto_continue_agent: bool,
    ) -> String {
        let id = Uuid::new_v4().to_string();
        let term = TerminalSession {
            id: id.clone(),
            name,
            owner,
            command,
            working_dir: self.settings.working_directory.clone(),
            output: String::new(),
            running: false,
            terminated: false,
            exit_code: None,
            pid: None,
            linked_session_idx,
            auto_continue_agent,
            input_buffer: String::new(),
        };
        self.terminals.push(term);
        self.active_terminal_id = Some(id.clone());
        id
    }

    fn active_terminal_index(&self) -> Option<usize> {
        let active = self.active_terminal_id.as_ref()?;
        self.terminals.iter().position(|t| &t.id == active)
    }

    fn remove_terminal(&mut self, terminal_id: &str) {
        self.terminals.retain(|t| t.id != terminal_id);
        if self.active_terminal_id.as_deref() == Some(terminal_id) {
            self.active_terminal_id = self.terminals.last().map(|t| t.id.clone());
        }
    }

    fn terminate_terminal(&mut self, terminal_id: &str) {
        if let Some(term) = self.terminals.iter_mut().find(|t| t.id == terminal_id) {
            if let Some(pid) = term.pid {
                #[cfg(unix)]
                {
                    let _ = std::process::Command::new("kill")
                        .arg(pid.to_string())
                        .output();
                }
                #[cfg(windows)]
                {
                    let _ = std::process::Command::new("taskkill")
                        .args(["/PID", &pid.to_string(), "/T", "/F"])
                        .output();
                }
            }
            term.terminated = true;
            term.running = false;
            term.output.push_str("\n\n[terminated by user]\n");
        }
    }

    fn spawn_terminal_command(&mut self, terminal_id: &str, command: String) {
        let Some(idx) = self.terminals.iter().position(|t| t.id == terminal_id) else {
            return;
        };
        if self.terminals[idx].running || self.terminals[idx].terminated {
            return;
        }
        self.terminals[idx].command = command.clone();
        self.terminals[idx].running = true;
        self.terminals[idx].exit_code = None;
        self.terminals[idx]
            .output
            .push_str(&format!("$ {}\n", command));
        let wd = self.settings.working_directory.clone();
        let tx = self.event_tx.clone();
        let tid = terminal_id.to_string();
        self.tokio_rt.spawn(async move {
            let executor = ActionExecutor::new(wd);
            let action = AgentAction {
                action: ActionKind::RunCmd,
                parameters: CommandParams {
                    path: None,
                    content: None,
                    mode: None,
                    title: None,
                    command: Some(command),
                },
            };
            let outcome = executor.execute_action(action, true).await;
            match outcome {
                Ok(ExecutionStatus::Executed(report)) => {
                    let _ = tx
                        .send(AppEvent::TerminalFinished {
                            terminal_id: tid,
                            stdout: report.stdout,
                            stderr: report.stderr,
                            exit_code: report.exit_code,
                        })
                        .await;
                }
                Ok(ExecutionStatus::AwaitingApproval(_)) => {
                    let _ = tx
                        .send(AppEvent::TerminalFinished {
                            terminal_id: tid,
                            stdout: String::new(),
                            stderr: "Command awaiting approval".to_string(),
                            exit_code: -1,
                        })
                        .await;
                }
                Err(e) => {
                    let _ = tx
                        .send(AppEvent::TerminalFinished {
                            terminal_id: tid,
                            stdout: String::new(),
                            stderr: e.to_string(),
                            exit_code: -1,
                        })
                        .await;
                }
            }
        });
    }

    fn append_system_message(&mut self, session_idx: Option<usize>, content: String) {
        let idx = session_idx.or(self.current_session_idx);
        let mut to_save: Option<(String, Message)> = None;
        if let Some(i) = idx {
            if let Some(session) = self.sessions.get_mut(i) {
                let msg = Message {
                    id: Uuid::new_v4().to_string(),
                    role: Role::System,
                    content,
                    attachments: vec![],
                    timestamp: Utc::now(),
                    is_streaming: false,
                };
                let session_id = session.id.clone();
                session.messages.push(msg.clone());
                to_save = Some((session_id, msg));
            }
        }
        if let Some((session_id, msg)) = to_save {
            self.save_message(&session_id, &msg);
        }
        if let Some(i) = idx {
            self.persist_session_snapshot_async(i);
        }
    }

    fn spawn_save_message(&self, session_id: String, msg: Message) {
        let db_path = self.db_path.clone();
        let tx = self.event_tx.clone();
        self.tokio_rt.spawn(async move {
            let result = tokio::task::spawn_blocking(move || -> Result<(), String> {
                let db = Database::new(&db_path).map_err(|e| e.to_string())?;
                let db_msg = DbMessage {
                    id: msg.id.clone(),
                    session_id,
                    role: msg.role.as_str().to_string(),
                    content: msg.content.clone(),
                    created_at: msg.timestamp,
                };
                db.save_message(&db_msg).map_err(|e| e.to_string())?;

                for att in &msg.attachments {
                    let db_att = crate::db::DbAttachment {
                        id: Uuid::new_v4().to_string(),
                        message_id: msg.id.clone(),
                        filename: att.filename.clone(),
                        data: att.raw_bytes.clone(),
                        mime_type: att.mime_type.clone(),
                    };
                    db.save_attachment(&db_att).map_err(|e| e.to_string())?;
                }
                Ok(())
            })
            .await;

            if let Ok(Err(e)) = result {
                let _ = tx.send(AppEvent::DbError(format!("Failed to save message: {e}"))).await;
            } else if let Err(e) = result {
                let _ = tx.send(AppEvent::DbError(format!("Failed to save message: {e}"))).await;
            }
        });
    }

    fn spawn_create_session(&self, session: Session) {
        let db_path = self.db_path.clone();
        let tx = self.event_tx.clone();
        self.tokio_rt.spawn(async move {
            let result = tokio::task::spawn_blocking(move || -> Result<(), String> {
                let db = Database::new(&db_path).map_err(|e| e.to_string())?;
                let db_session = DbSession {
                    id: session.id,
                    name: session.name,
                    created_at: Utc::now(),
                };
                db.create_session(&db_session).map_err(|e| e.to_string())
            })
            .await;

            if let Ok(Err(e)) = result {
                let _ = tx.send(AppEvent::DbError(format!("Failed to create DB session: {e}"))).await;
            } else if let Err(e) = result {
                let _ = tx.send(AppEvent::DbError(format!("Failed to create DB session: {e}"))).await;
            }
        });
    }

    fn spawn_delete_session(&self, session_id: String) {
        let db_path = self.db_path.clone();
        let tx = self.event_tx.clone();
        self.tokio_rt.spawn(async move {
            let result = tokio::task::spawn_blocking(move || -> Result<(), String> {
                let db = Database::new(&db_path).map_err(|e| e.to_string())?;
                db.delete_session(&session_id).map_err(|e| e.to_string())
            })
            .await;

            if let Ok(Err(e)) = result {
                let _ = tx.send(AppEvent::DbError(format!("Failed to delete conversation: {e}"))).await;
            } else if let Err(e) = result {
                let _ = tx.send(AppEvent::DbError(format!("Failed to delete conversation: {e}"))).await;
            }
        });
    }

    fn spawn_save_snapshot(&self, snapshot: DbFileSnapshot) {
        let db_path = self.db_path.clone();
        let tx = self.event_tx.clone();
        self.tokio_rt.spawn(async move {
            let result = tokio::task::spawn_blocking(move || -> Result<(), String> {
                let db = Database::new(&db_path).map_err(|e| e.to_string())?;
                db.save_file_snapshot(&snapshot).map_err(|e| e.to_string())
            })
            .await;
            if let Ok(Err(e)) = result {
                let _ = tx
                    .send(AppEvent::DbError(format!("Failed to persist file snapshot: {e}")))
                    .await;
            } else if let Err(e) = result {
                let _ = tx
                    .send(AppEvent::DbError(format!("Failed to persist file snapshot: {e}")))
                    .await;
            }
        });
    }

    fn spawn_refresh_snapshots(&self) {
        let db_path = self.db_path.clone();
        let tx = self.event_tx.clone();
        self.tokio_rt.spawn(async move {
            let result = tokio::task::spawn_blocking(move || -> Result<Vec<DbFileSnapshot>, String> {
                let db = Database::new(&db_path).map_err(|e| e.to_string())?;
                db.list_file_snapshots(50).map_err(|e| e.to_string())
            })
            .await;
            match result {
                Ok(Ok(snapshots)) => {
                    let _ = tx
                        .send(AppEvent::SnapshotsLoaded {
                            snapshots,
                            error: None,
                        })
                        .await;
                }
                Ok(Err(e)) => {
                    let _ = tx
                        .send(AppEvent::SnapshotsLoaded {
                            snapshots: vec![],
                            error: Some(e),
                        })
                        .await;
                }
                Err(e) => {
                    let _ = tx
                        .send(AppEvent::SnapshotsLoaded {
                            snapshots: vec![],
                            error: Some(e.to_string()),
                        })
                        .await;
                }
            }
        });
    }

    fn spawn_restore_snapshot(&self, snapshot_id: String, file_path: String, fallback: Vec<u8>) {
        let db_path = self.db_path.clone();
        let tx = self.event_tx.clone();
        let file_path_for_write = file_path.clone();
        self.tokio_rt.spawn(async move {
            let result = tokio::task::spawn_blocking(move || -> Result<(), String> {
                let db = Database::new(&db_path).map_err(|e| e.to_string())?;
                let snapshot_data = db
                    .get_file_snapshot(&snapshot_id)
                    .map_err(|e| e.to_string())?
                    .map(|s| s.content)
                    .unwrap_or(fallback);
                std::fs::write(std::path::Path::new(&file_path_for_write), snapshot_data)
                    .map_err(|e| e.to_string())?;
                Ok(())
            })
            .await;
            match result {
                Ok(Ok(())) => {
                    let _ = tx
                        .send(AppEvent::SnapshotRestored {
                            file_path,
                            error: None,
                        })
                        .await;
                }
                Ok(Err(e)) => {
                    let _ = tx
                        .send(AppEvent::SnapshotRestored {
                            file_path,
                            error: Some(e),
                        })
                        .await;
                }
                Err(e) => {
                    let _ = tx
                        .send(AppEvent::SnapshotRestored {
                            file_path,
                            error: Some(e.to_string()),
                        })
                        .await;
                }
            }
        });
    }

    pub fn new(cc: &eframe::CreationContext, rt: Arc<tokio::runtime::Runtime>) -> Self {
        let (event_tx, event_rx) = tokio::sync::mpsc::channel(1024);
        let settings = load_settings();
        let db_path = settings.db_path.clone();
        Self::apply_theme(&cc.egui_ctx, settings.dark_mode);
        let storage = match Storage::new() {
            Ok(s) => Some(s),
            Err(e) => {
                eprintln!("Storage init error: {e}");
                None
            }
        };

        let models = provider_models(settings.selected_provider.key());
        let custom_model_id = settings.default_model.clone();
        let mut app = Self {
            setup_wizard: if settings.setup_complete {
                None
            } else {
                Some(SetupWizard::new(&settings))
            },
            settings,
            show_settings: false,
            sessions: vec![],
            current_session_idx: None,
            models,
            selected_model_idx: 0,
            remote_models: vec![],
            model_waiting_command_output: false,
            model_access_outline_ok: None,
            custom_model_id,
            selected_thinking_mode: ThinkingMode::Auto,
            input_text: String::new(),
            pending_attachments: vec![],
            event_tx,
            event_rx,
            tokio_rt: rt,
            active_requests: HashMap::new(),
            db_path,
            storage,
            pending_action: None,
            pending_actions_queue: VecDeque::new(),
            pending_action_session_idx: None,
            markdown_cache: CommonMarkCache::default(),
            notifications: vec![],
            activity_log: vec!["App started".to_string()],
            show_snapshots: false,
            snapshots: vec![],
            agents: vec![
                AgentProfile {
                    name: "General".to_string(),
                    system_prompt: "You are a helpful, concise assistant.".to_string(),
                    enabled: true,
                },
                AgentProfile {
                    name: "Reviewer".to_string(),
                    system_prompt: "You review ideas critically and point out risks.".to_string(),
                    enabled: false,
                },
                AgentProfile {
                    name: "Coder".to_string(),
                    system_prompt: "You focus on actionable implementation details.".to_string(),
                    enabled: false,
                },
            ],
            terminals: vec![],
            active_terminal_id: None,
            changed_files: vec![],
            opened_files: vec![],
            active_open_file: None,
            editor_buffers: HashMap::new(),
            editor_dirty: HashMap::new(),
            show_left_sidebar: true,
            show_activity_panel: false,
        };

        if let Some(idx) = app
            .models
            .iter()
            .position(|m| m.id == app.settings.default_model)
        {
            app.selected_model_idx = idx;
        }

        app.spawn_initial_data_load();
        app.load_models_from_provider();
        app
    }

    fn apply_theme(ctx: &egui::Context, dark_mode: bool) {
        if dark_mode {
            let mut style = (*ctx.style()).clone();
            style.visuals = egui::Visuals::dark();
            style.visuals.override_text_color = Some(TEXT_PRIMARY);
            style.visuals.faint_bg_color = BG_SURFACE;
            style.visuals.panel_fill = BG_PRIMARY;
            style.visuals.window_fill = BG_SURFACE;
            style.visuals.extreme_bg_color = BG_PRIMARY;
            style.visuals.code_bg_color = BG_SURFACE_ALT;
            style.visuals.widgets.noninteractive.bg_fill = BG_SURFACE;
            style.visuals.widgets.noninteractive.fg_stroke.color = TEXT_PRIMARY;
            style.visuals.widgets.inactive.bg_fill = BG_SURFACE_ALT;
            style.visuals.widgets.inactive.fg_stroke.color = TEXT_PRIMARY;
            style.visuals.widgets.hovered.bg_fill = BG_SURFACE_ALT;
            style.visuals.widgets.hovered.fg_stroke.color = TEXT_PRIMARY;
            style.visuals.widgets.active.bg_fill = ACCENT_SOFT;
            style.visuals.widgets.active.fg_stroke.color = TEXT_PRIMARY;
            style.visuals.widgets.open.bg_fill = BG_SURFACE;
            style.visuals.widgets.open.fg_stroke.color = TEXT_PRIMARY;
            style.visuals.selection.bg_fill = ACCENT_SOFT;
            style.visuals.selection.stroke.color = TEXT_PRIMARY;
            ctx.set_style(style);
        } else {
            ctx.set_visuals(egui::Visuals::light());
        }
    }

    fn notify(&mut self, text: impl Into<String>, kind: NotificationKind) {
        self.notifications.push(Notification {
            id: Uuid::new_v4().to_string(),
            text: text.into(),
            kind,
            ttl_secs: 4.0,
        });
    }

    fn current_model(&self) -> Option<&ModelInfo> {
        self.models.get(self.selected_model_idx)
    }

    fn current_session(&self) -> Option<&Session> {
        self.current_session_idx.and_then(|i| self.sessions.get(i))
    }

    fn load_models_from_provider(&mut self) {
        self.models = provider_models(self.settings.selected_provider.key());
        if self.models.is_empty() {
            self.models = builtin_models();
        }
        if let Some(idx) = self
            .models
            .iter()
            .position(|m| m.id == self.settings.default_model)
        {
            self.selected_model_idx = idx;
        } else {
            self.selected_model_idx = 0;
        }
        if self.custom_model_id.trim().is_empty() {
            if let Some(model) = self.current_model() {
                self.custom_model_id = model.id.clone();
            }
        }
        self.remote_models.clear();
        self.model_access_outline_ok = None;

        let api_key = self.settings.active_api_key();
        let base_url = self.settings.active_base_url();
        let tx = self.event_tx.clone();
        self.tokio_rt.spawn(async move {
            let client = crate::api::OpenAIClient::new(&api_key, &base_url);
            let model_ids = match client.list_models().await {
                Ok(ids) => ids,
                Err(_) => vec![],
            };
            let _ = tx.send(AppEvent::ModelsLoaded(model_ids)).await;
        });
    }

    fn dispatch_agent_requests(&mut self, session_idx: usize) {
        let model_id = if self.custom_model_id.trim().is_empty() {
            self.current_model()
                .map(|m| m.id.clone())
                .unwrap_or_else(|| DEFAULT_MODEL_ID.to_string())
        } else {
            self.custom_model_id.trim().to_string()
        };

        if !self.agents.iter().any(|a| a.enabled) {
            self.notify("Enable at least one agent", NotificationKind::Error);
            return;
        }

        for agent_idx in 0..self.agents.len() {
            if !self.agents[agent_idx].enabled {
                continue;
            }
            let agent_name = self.agents[agent_idx].name.clone();
            let agent_prompt = self.agents[agent_idx].system_prompt.clone();
            let request_id = Uuid::new_v4().to_string();
            let assistant_msg = Message {
                id: Uuid::new_v4().to_string(),
                role: Role::Assistant,
                content: String::new(),
                attachments: vec![],
                timestamp: Utc::now(),
                is_streaming: true,
            };
            let assistant_msg_id = assistant_msg.id.clone();
            if let Some(session) = self.sessions.get_mut(session_idx) {
                session.messages.push(assistant_msg);
            }

            self.active_requests.insert(
                request_id.clone(),
                ActiveRequest {
                    session_idx,
                    message_id: assistant_msg_id,
                    agent_name: agent_name.clone(),
                },
            );
            self.activity_log.push(format!("{} started response", agent_name));

            let mut api_messages: Vec<ChatMessage> = vec![];
            let system_prompt = format!(
                "You are agent '{}'. {} Working directory: {}. Shell execution: {}. \
Never claim file writes or shell commands succeeded unless they actually ran with user approval. \
If user asks to create/edit a file, respond ONLY with <write_file path=\"relative/or/absolute/path\">FILE_CONTENT</write_file>. \
If user asks to run a command and shell execution is enabled, respond ONLY with <execute_command>THE_COMMAND</execute_command>. \
Do not fabricate directory listings, command outputs, or success messages.",
                agent_name,
                agent_prompt,
                self.settings.working_directory,
                if self.settings.shell_execution_enabled {
                    "enabled"
                } else {
                    "disabled"
                }
            );
            let telemetry_context =
                collect_telemetry_cached(std::time::Duration::from_secs(2)).to_llm_system_context();
            let api_key = self.settings.active_api_key();
            let base_url = self.settings.active_base_url();
            let query = self
                .sessions
                .get(session_idx)
                .and_then(|s| {
                    s.messages
                        .iter()
                        .rev()
                        .find(|m| matches!(m.role, Role::User))
                        .map(|m| m.content.clone())
                })
                .unwrap_or_default();
            let working_dir = self.settings.working_directory.clone();

            if let Some(session) = self.sessions.get(session_idx) {
                for msg in &session.messages {
                    if msg.is_streaming {
                        continue;
                    }
                    let role = msg.role.as_str();
                    if msg.attachments.iter().any(|a| a.image_base64.is_some()) {
                        for att in &msg.attachments {
                            if let Some(b64) = &att.image_base64 {
                                api_messages.push(ChatMessage::with_image(role, &msg.content, b64, &att.mime_type));
                            }
                        }
                    } else {
                        let mut content = msg.content.clone();
                        for att in &msg.attachments {
                            if let Some(txt) = &att.text {
                                content.push_str(&format!("\n\n[File: {}]\n{}", att.filename, txt));
                            }
                        }
                        api_messages.push(ChatMessage::text(role, &content));
                    }
                }
            }

            let thinking_mode = if self.current_model().map(|m| m.supports_thinking()).unwrap_or(false) {
                Some(self.selected_thinking_mode.clone())
            } else {
                None
            };

            let event_tx = self.event_tx.clone();
            let req_id = request_id.clone();
            let model_id = model_id.clone();

            self.tokio_rt.spawn(async move {
                let mut rag_context = "LOCAL REPOSITORY CONTEXT:\n- No relevant snippets found.".to_string();
                if !api_key.is_empty() && !query.trim().is_empty() {
                    let provider = OpenAIEmbeddingProvider {
                        api_key: api_key.clone(),
                        base_url: base_url.clone(),
                        model: "text-embedding-3-small".to_string(),
                    };
                    let rag_cfg = RagConfig::with_workspace(working_dir, 1536);
                    if let Ok(engine) = RagEngine::new(rag_cfg, provider).await {
                        if let Ok(snippets) = engine.semantic_search(&query).await {
                            rag_context = RagEngine::<OpenAIEmbeddingProvider>::format_repository_context(&snippets);
                        }
                    }
                }

                let full_system_prompt = format!(
                    "{CORE_OS_SYSTEM_PROMPT}\n\n{telemetry_context}\n\n{rag_context}\n\n{system_prompt}"
                );
                api_messages.insert(0, ChatMessage::with_cache_control("system", &full_system_prompt));
                let client = crate::api::OpenAIClient::new(&api_key, &base_url);
                let req_id_clone = req_id.clone();
                let tx_clone = event_tx.clone();

                let result = client
                    .chat_completion(&model_id, api_messages, thinking_mode.as_ref(), move |chunk| {
                        let _ = tx_clone.blocking_send(AppEvent::ChunkReceived(req_id_clone.clone(), chunk));
                    })
                    .await;

                match result {
                    Ok(_) => {
                        let _ = event_tx.send(AppEvent::ResponseComplete(req_id)).await;
                    }
                    Err(e) => {
                        let _ = event_tx.send(AppEvent::ResponseError(req_id, e.to_string())).await;
                    }
                }
            });
        }
    }

    fn delete_session_by_index(&mut self, idx: usize) {
        let Some(session) = self.sessions.get(idx) else {
            return;
        };
        let session_id = session.id.clone();
        let session_name = session.name.clone();
        self.spawn_delete_session(session_id);

        self.sessions.remove(idx);
        self.current_session_idx = match self.current_session_idx {
            Some(current) if current == idx => None,
            Some(current) if current > idx => Some(current - 1),
            other => other,
        };
        self.activity_log.push(format!("Deleted conversation: {}", session_name));
        self.notify("Conversation deleted", NotificationKind::Info);
    }

    fn new_session(&mut self) {
        let session = Session::new("New Chat");
        self.spawn_create_session(session.clone());

        self.sessions.insert(0, session);
        self.current_session_idx = Some(0);
        self.activity_log.push("Created new conversation".to_string());
    }

    fn load_session_messages(&mut self, idx: usize) {
        let Some(session) = self.sessions.get(idx) else {
            return;
        };
        let session_id = session.id.clone();
        self.spawn_load_session_messages(idx, session_id);
    }

    fn save_message(&mut self, session_id: &str, msg: &Message) {
        self.spawn_save_message(session_id.to_string(), msg.clone());
    }

    fn save_snapshot_if_exists(&mut self, path: &str) {
        let full_path = std::path::Path::new(path);
        if !full_path.exists() {
            return;
        }
        let content = match std::fs::read(full_path) {
            Ok(content) => content,
            Err(e) => {
                self.notify(format!("Could not snapshot file before write: {e}"), NotificationKind::Error);
                return;
            }
        };
        let snapshot = DbFileSnapshot {
            id: Uuid::new_v4().to_string(),
            file_path: path.to_string(),
            content,
            created_at: Utc::now(),
        };
        self.spawn_save_snapshot(snapshot);
    }

    fn refresh_snapshots(&mut self) {
        self.spawn_refresh_snapshots();
    }

    fn read_text_file_for_view(path: &str) -> String {
        match crate::files::read_text_file(std::path::Path::new(path)) {
            Ok(c) => c,
            Err(e) => format!("Could not open file: {e}"),
        }
    }

    fn syntax_highlight_for_path(path: &str, code: &str) -> LayoutJob {
        fn is_escaped_closing_quote(buffer: &str, quote: char) -> bool {
            let mut chars = buffer.chars().rev();
            if chars.next() != Some(quote) {
                return true;
            }
            let mut slash_count = 0;
            for ch in chars {
                if ch == '\\' {
                    slash_count += 1;
                } else {
                    break;
                }
            }
            slash_count % 2 == 1
        }

        let ext = std::path::Path::new(path)
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_lowercase();

        let keyword_color = match ext.as_str() {
            "rs" => Color32::from_rgb(120, 210, 255),
            "py" => Color32::from_rgb(255, 220, 120),
            "md" => Color32::from_rgb(150, 215, 170),
            _ => Color32::from_rgb(180, 205, 255),
        };
        let string_color = Color32::from_rgb(230, 180, 120);
        let comment_color = Color32::from_rgb(130, 140, 150);
        let text_color = TEXT_PRIMARY;

        let mut job = LayoutJob::default();
        let mut word = String::new();
        let mut in_string: Option<char> = None;
        let base_text_format = TextFormat {
            font_id: FontId::monospace(13.0),
            color: text_color,
            ..Default::default()
        };
        let base_string_format = TextFormat {
            color: string_color,
            ..base_text_format.clone()
        };
        let base_comment_format = TextFormat {
            color: comment_color,
            ..base_text_format.clone()
        };

        let push_word = |layout_job: &mut LayoutJob, word_buffer: &mut String| {
            if word_buffer.is_empty() {
                return;
            }
            let mut format = base_text_format.clone();
            let is_keyword = match ext.as_str() {
                "rs" => matches!(
                    word_buffer.as_str(),
                    "fn" | "let" | "mut" | "pub" | "impl" | "struct" | "enum" | "match" | "if"
                        | "else" | "for" | "while" | "use" | "mod" | "return" | "const"
                        | "static" | "trait" | "type" | "async" | "await" | "unsafe" | "where"
                ),
                "py" => matches!(
                    word_buffer.as_str(),
                    "def" | "class" | "if" | "elif" | "else" | "for" | "while" | "import" | "from"
                        | "return" | "with" | "as" | "try" | "except" | "lambda" | "async"
                        | "await" | "yield" | "finally" | "raise" | "assert" | "pass"
                        | "break" | "continue" | "global" | "nonlocal" | "and" | "or" | "not"
                ),
                "md" => word_buffer.starts_with('#') || word_buffer == "```",
                _ => false,
            };
            if is_keyword {
                format.color = keyword_color;
            }
            layout_job.append(word_buffer, 0.0, format);
            word_buffer.clear();
        };

        for line in code.lines() {
            if ext == "py" {
                let trimmed = line.trim_start();
                if trimmed.starts_with('#') {
                    job.append(line, 0.0, base_comment_format.clone());
                    job.append("\n", 0.0, base_text_format.clone());
                    continue;
                }
            } else if let Some(comment_idx) = line.find("//") {
                let (left, comment) = line.split_at(comment_idx);
                for ch in left.chars() {
                    if in_string.is_none() && (ch == '"' || ch == '\'') {
                        push_word(&mut job, &mut word);
                        in_string = Some(ch);
                        word.push(ch);
                        continue;
                    }
                    if let Some(quote) = in_string {
                        word.push(ch);
                        if ch == quote && !is_escaped_closing_quote(&word, quote) {
                            job.append(&word, 0.0, base_string_format.clone());
                            word.clear();
                            in_string = None;
                        }
                        continue;
                    }
                    if ch.is_ascii_alphanumeric() || ch == '_' || (ext == "md" && ch == '#') {
                        word.push(ch);
                    } else {
                        push_word(&mut job, &mut word);
                        job.append(&ch.to_string(), 0.0, base_text_format.clone());
                    }
                }
                push_word(&mut job, &mut word);
                job.append(comment, 0.0, base_comment_format.clone());
                job.append("\n", 0.0, base_text_format.clone());
                continue;
            }

            for ch in line.chars() {
                if in_string.is_none() && (ch == '"' || ch == '\'') {
                    push_word(&mut job, &mut word);
                    in_string = Some(ch);
                    word.push(ch);
                    continue;
                }
                if let Some(quote) = in_string {
                    word.push(ch);
                    if ch == quote && !is_escaped_closing_quote(&word, quote) {
                        job.append(&word, 0.0, base_string_format.clone());
                        word.clear();
                        in_string = None;
                    }
                    continue;
                }
                if ch.is_ascii_alphanumeric() || ch == '_' || (ext == "md" && ch == '#') {
                    word.push(ch);
                } else {
                    push_word(&mut job, &mut word);
                    job.append(&ch.to_string(), 0.0, base_text_format.clone());
                }
            }
            push_word(&mut job, &mut word);
            job.append("\n", 0.0, base_text_format.clone());
        }

        job
    }

    fn save_active_editor_file(&mut self) {
        let Some(path) = self.active_open_file.clone() else {
            return;
        };
        let Some(content) = self.editor_buffers.get(&path).cloned() else {
            return;
        };
        let file_path = std::path::Path::new(&path);
        match crate::files::write_text_file(file_path, &content) {
            Ok(_) => {
                self.editor_dirty.insert(path.clone(), false);
                self.ensure_changed_file_tracked(path.clone());
                self.notify(format!("Saved {}", path), NotificationKind::Info);
            }
            Err(e) => self.notify(format!("Save failed for {}: {}", path, e), NotificationKind::Error),
        }
    }

    fn send_message(&mut self) {
        let text = self.input_text.trim().to_string();
        if text.is_empty() && self.pending_attachments.is_empty() {
            return;
        };

        if self.current_session_idx.is_none() {
            self.new_session();
        }
        let Some(session_idx) = self.current_session_idx else {
            self.notify("No active session", NotificationKind::Error);
            return;
        };

        let attachments = std::mem::take(&mut self.pending_attachments);
        let mut display_content = text.clone();
        for att in &attachments {
            display_content.push_str(&format!("\n[Attachment: {}]", att.filename));
        }

        let user_msg = Message {
            id: Uuid::new_v4().to_string(),
            role: Role::User,
            content: display_content,
            attachments: attachments.clone(),
            timestamp: Utc::now(),
            is_streaming: false,
        };

        let Some(session) = self.sessions.get(session_idx) else {
            self.notify("Session index out of bounds", NotificationKind::Error);
            return;
        };
        let session_id = session.id.clone();
        self.save_message(&session_id, &user_msg);

        if let Some(session) = self.sessions.get_mut(session_idx) {
            if session.messages.is_empty() && !text.is_empty() {
                session.name = text.chars().take(40).collect();
            }
            session.messages.push(user_msg);
        }
        self.input_text.clear();
        self.persist_session_snapshot_async(session_idx);

        self.dispatch_agent_requests(session_idx);
    }

    fn process_events(&mut self) {
        while let Ok(event) = self.event_rx.try_recv() {
            match event {
                AppEvent::ChunkReceived(req_id, chunk) => {
                    if let Some(active) = self.active_requests.get(&req_id).cloned() {
                        if let Some(session) = self.sessions.get_mut(active.session_idx) {
                            if let Some(msg) = session.messages.iter_mut().find(|m| m.id == active.message_id) {
                                msg.content.push_str(&chunk);
                            }
                        }
                    }
                }
                AppEvent::ResponseComplete(req_id) => {
                    if let Some(active) = self.active_requests.remove(&req_id) {
                        let mut to_save: Option<(String, Message)> = None;
                        let mut parse_error_notification: Option<String> = None;
                        if let Some(session) = self.sessions.get_mut(active.session_idx) {
                            let session_id = session.id.clone();
                            if let Some(msg) = session.messages.iter_mut().find(|m| m.id == active.message_id) {
                                msg.is_streaming = false;
                                let parsed = crate::parser::parse_response(&msg.content);
                                let mut rendered = String::new();
                                if let Some(message) = parsed.message.as_deref() {
                                    rendered.push_str(message);
                                } else if !parsed.fallback_text.trim().is_empty() {
                                    rendered.push_str(parsed.fallback_text.trim());
                                }

                                if !parsed.plan_items.is_empty() {
                                    if !rendered.is_empty() {
                                        rendered.push_str("\n\n");
                                    }
                                    rendered.push_str("PLAN:\n");
                                    for item in &parsed.plan_items {
                                        rendered.push_str("- [ ] ");
                                        rendered.push_str(item);
                                        rendered.push('\n');
                                    }
                                    rendered = rendered.trim_end().to_string();
                                }

                                if let Some(err) = parsed.json_parse_error.as_deref() {
                                    parse_error_notification =
                                        Some(format!("Execution block parse error: {err}"));
                                }

                                msg.content = rendered;

                                if self.pending_action.is_none() {
                                    let parsed_actions = Self::pending_actions_from_agent_actions(parsed.actions);
                                    if !parsed_actions.is_empty() {
                                        let mut iter = parsed_actions.into_iter();
                                        if let Some(action) = iter.next() {
                                            self.pending_action = Some(action);
                                            self.pending_action_session_idx = Some(active.session_idx);
                                            self.pending_actions_queue = iter.collect();
                                        }
                                    } else if let Some(action) = Self::parse_pending_action(&msg.content) {
                                        self.pending_action = Some(action);
                                        self.pending_action_session_idx = Some(active.session_idx);
                                    }
                                }
                                to_save = Some((session_id, msg.clone()));
                            }
                        }
                        if let Some(error_text) = parse_error_notification {
                            self.notify(error_text, NotificationKind::Error);
                        }
                        if let Some((session_id, msg)) = to_save {
                            self.save_message(&session_id, &msg);
                        }
                        self.persist_session_snapshot_async(active.session_idx);
                        self.activity_log.push(format!("{} finished response", active.agent_name));
                    }
                }
                AppEvent::ResponseError(req_id, err) => {
                    if let Some(active) = self.active_requests.remove(&req_id) {
                        if let Some(session) = self.sessions.get_mut(active.session_idx) {
                            if let Some(msg) = session.messages.iter_mut().find(|m| m.id == active.message_id) {
                                msg.content = format!("❌ Error: {}", err);
                                msg.is_streaming = false;
                            }
                        }
                        self.notify(format!("{} failed: {}", active.agent_name, err), NotificationKind::Error);
                    }
                }
                AppEvent::ModelsLoaded(model_ids) => {
                    self.remote_models = model_ids.clone();
                    if !model_ids.is_empty() {
                        self.models = model_ids
                            .iter()
                            .map(|m| {
                                let mut name = m.id.clone();
                                if m.premium {
                                    name.push_str("  [Premium]");
                                }
                                if m.gated {
                                    name.push_str("  [Gated]");
                                }
                                ModelInfo {
                                    id: m.id.clone(),
                                    name,
                                    thinking_modes: vec![],
                                }
                            })
                            .collect();
                        if let Some(idx) = self
                            .models
                            .iter()
                            .position(|m| m.id == self.settings.default_model)
                        {
                            self.selected_model_idx = idx;
                        } else {
                            self.selected_model_idx = 0;
                        }
                    }
                    self.refresh_model_access_outline();
                }
                AppEvent::TerminalFinished {
                    terminal_id,
                    stdout,
                    stderr,
                    exit_code,
                } => {
                    let mut continue_session: Option<usize> = None;
                    let mut system_result: Option<(usize, String)> = None;
                    let sanitized_stdout = Self::sanitized_command_stdout(&stdout, exit_code);
                    if let Some(term) = self.terminals.iter_mut().find(|t| t.id == terminal_id) {
                        term.running = false;
                        term.exit_code = Some(exit_code);
                        if !stdout.trim().is_empty() {
                            term.output.push_str(&stdout);
                            if !stdout.ends_with('\n') {
                                term.output.push('\n');
                            }
                        }
                        if !stderr.trim().is_empty() {
                            term.output.push_str("\n[stderr]\n");
                            term.output.push_str(&stderr);
                            if !stderr.ends_with('\n') {
                                term.output.push('\n');
                            }
                        }
                        term.output.push_str(&format!("\n[exit code: {}]\n", exit_code));
                        if term.owner == TerminalOwner::AI && self.model_waiting_command_output {
                            if let Some(session_idx) = term.linked_session_idx {
                                let mut result = format!(
                                    "✅ Command executed (approved)\nCommand: `{}`\nWorking directory: `{}`\nExit code: {}",
                                    term.command, term.working_dir, exit_code
                                );
                                result.push_str("\n\nstdout:\n```text\n");
                                result.push_str(&sanitized_stdout);
                                result.push_str("\n```");
                                if !stderr.trim().is_empty() {
                                    result.push_str("\n\nstderr:\n```text\n");
                                    result.push_str(&stderr);
                                    result.push_str("\n```");
                                }
                                system_result = Some((session_idx, result));
                                if term.auto_continue_agent {
                                    continue_session = Some(session_idx);
                                }
                            }
                        }
                    }
                    if let Some((session_idx, result)) = system_result {
                        self.append_system_message(Some(session_idx), result);
                        self.model_waiting_command_output = false;
                    }
                    if let Some(session_idx) = continue_session {
                        self.dispatch_agent_requests(session_idx);
                    }
                }
                AppEvent::InitialSessionsLoaded { sessions, error } => {
                    if let Some(err) = error {
                        self.notify(format!("DB load error: {err}"), NotificationKind::Error);
                    } else {
                        self.sessions = sessions;
                        if !self.sessions.is_empty() {
                            self.current_session_idx = Some(0);
                            self.load_session_messages(0);
                        }
                    }
                }
                AppEvent::SessionMessagesLoaded { idx, messages, error } => {
                    if let Some(err) = error {
                        self.notify(format!("Failed to load messages: {err}"), NotificationKind::Error);
                    } else if let Some(session) = self.sessions.get_mut(idx) {
                        session.messages = messages;
                    }
                }
                AppEvent::StoredSessionLoaded { session, error } => {
                    if let Some(err) = error {
                        self.activity_log.push(format!("Failed to load stored session: {err}"));
                    } else if self.sessions.is_empty() {
                        if let Some(stored) = session {
                            self.sessions.push(stored);
                            self.current_session_idx = Some(0);
                            self.activity_log
                                .push("Loaded latest chat session from local storage".to_string());
                        }
                    }
                }
                AppEvent::SnapshotsLoaded { snapshots, error } => {
                    if let Some(err) = error {
                        self.snapshots.clear();
                        self.notify(format!("Failed to load snapshots: {err}"), NotificationKind::Error);
                    } else {
                        self.snapshots = snapshots;
                    }
                }
                AppEvent::SnapshotRestored { file_path, error } => {
                    if let Some(err) = error {
                        self.notify(format!("Restore failed: {err}"), NotificationKind::Error);
                    } else {
                        self.notify(format!("Restored {}", file_path), NotificationKind::Info);
                    }
                }
                AppEvent::DbError(err) => {
                    self.notify(err, NotificationKind::Error);
                }
            }
        }
    }

    fn attach_file(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("All Files", &["*"])
            .add_filter("Images", &["png", "jpg", "jpeg", "gif", "webp"])
            .add_filter("Documents", &["pdf", "docx", "txt", "md"])
            .pick_file()
        {
            match crate::files::read_file(&path) {
                Ok(fc) => {
                    self.pending_attachments.push(Attachment {
                        filename: fc.filename,
                        text: fc.text,
                        image_base64: fc.image_base64,
                        mime_type: fc.mime_type,
                        raw_bytes: fc.raw_bytes,
                    });
                    self.notify("Attachment added", NotificationKind::Info);
                }
                Err(e) => self.notify(format!("Failed to read file: {e}"), NotificationKind::Error),
            }
        }
    }

    fn download_image(&mut self, att: &Attachment) {
        if let Some(path) = rfd::FileDialog::new().set_file_name(&att.filename).save_file() {
            if let Err(e) = std::fs::write(&path, &att.raw_bytes) {
                self.notify(format!("Failed to save image: {e}"), NotificationKind::Error);
            } else {
                self.notify("Image saved", NotificationKind::Info);
            }
        }
    }

    fn select_working_dir(&mut self) {
        if let Some(path) = rfd::FileDialog::new().pick_folder() {
            self.settings.working_directory = path.to_string_lossy().to_string();
            if let Err(e) = save_settings(&self.settings) {
                self.notify(format!("Failed to save settings: {e}"), NotificationKind::Error);
            }
        }
    }

    fn tick_notifications(&mut self, dt: f32) {
        for n in &mut self.notifications {
            n.ttl_secs -= dt;
        }
        self.notifications.retain(|n| n.ttl_secs > 0.0);
    }

    fn draw_notifications(&self, ctx: &egui::Context) {
        let mut offset = 16.0;
        for n in &self.notifications {
            let color = match n.kind {
                NotificationKind::Info => Color32::from_rgb(32, 128, 255),
                NotificationKind::Error => Color32::from_rgb(200, 60, 60),
            };

            egui::Area::new(egui::Id::new(format!("toast_{}", n.id)))
                .anchor(egui::Align2::RIGHT_TOP, [-16.0, offset])
                .show(ctx, |ui| {
                    egui::Frame::new()
                        .fill(color)
                        .corner_radius(6.0)
                        .inner_margin(egui::Margin::same(8))
                        .show(ui, |ui| {
                            ui.label(RichText::new(&n.text).color(Color32::WHITE));
                        });
                });
            offset += 44.0;
        }
    }

    fn render_snapshots_window(&mut self, ctx: &egui::Context) {
        if !self.show_snapshots {
            return;
        }

        let mut open = self.show_snapshots;
        egui::Window::new("Shadow Git Rollback")
            .open(&mut open)
            .resizable(true)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if ui.button("Refresh").clicked() {
                        self.refresh_snapshots();
                    }
                    ui.label(format!("{} snapshots", self.snapshots.len()));
                });
                ui.separator();

                let snapshots = self.snapshots.clone();
                ScrollArea::vertical().show(ui, |ui| {
                    for snap in snapshots {
                        ui.group(|ui| {
                            ui.label(RichText::new(&snap.file_path).strong());
                            ui.label(snap.created_at.format("%Y-%m-%d %H:%M:%S UTC").to_string());
                            if ui.button("Restore this snapshot").clicked() {
                                self.spawn_restore_snapshot(
                                    snap.id.clone(),
                                    snap.file_path.clone(),
                                    snap.content.clone(),
                                );
                            }
                        });
                    }
                });
            });
        self.show_snapshots = open;
    }

    fn render_file_workspace_panel(&mut self, ctx: &egui::Context) {
        egui::SidePanel::right("file_workspace")
            .resizable(true)
            .default_width(280.0)
            .min_width(200.0)
            .frame(egui::Frame::new().fill(BURGUNDY_DARK).inner_margin(egui::Margin::same(8)))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Editor").color(TEXT_PRIMARY).strong());
                    if ui.small_button("📂 Open File").clicked() {
                        if let Some(path) = rfd::FileDialog::new().pick_file() {
                            self.open_file_in_workspace(path.to_string_lossy().to_string());
                        }
                    }
                });
                ui.separator();
                ui.collapsing("Changed Files", |ui| {
                    ScrollArea::vertical().max_height(120.0).show(ui, |ui| {
                        let changed = self.changed_files.clone();
                        for path in changed {
                            ui.horizontal(|ui| {
                                if ui.small_button("Open").clicked() {
                                    self.open_file_in_workspace(path.clone());
                                }
                                ui.label(RichText::new(path.clone()).small().color(LIGHT_TEXT));
                            });
                        }
                        if self.changed_files.is_empty() {
                            ui.label(RichText::new("No changed files yet").small().color(TEXT_MUTED));
                        }
                    });
                });
                ui.separator();

                if self.opened_files.is_empty() {
                    ui.label(RichText::new("No file opened").small().color(TEXT_MUTED));
                    return;
                }

                ui.horizontal_wrapped(|ui| {
                    let opened = self.opened_files.clone();
                    for path in opened {
                        let selected = self.active_open_file.as_deref() == Some(path.as_str());
                        let is_dirty = self.editor_dirty.get(&path).copied().unwrap_or(false);
                        let filename = std::path::Path::new(&path)
                            .file_name()
                            .and_then(|s| s.to_str())
                            .unwrap_or(&path)
                            .to_string();
                        let label = if is_dirty {
                            format!("{} •", filename)
                        } else {
                            filename
                        };
                        let button = egui::Button::new(
                            RichText::new(label)
                                .small()
                                .color(if selected { DARK_TEXT } else { LIGHT_TEXT }),
                        )
                        .sense(egui::Sense::click())
                        .fill(if selected { GOLD } else { SKY_BLUE_DARK });
                        let response = ui
                            .add(button)
                            .on_hover_text(if is_dirty { "Unsaved changes" } else { "Saved" });
                        if response.clicked() {
                            self.active_open_file = Some(path.clone());
                        }
                        if ui.small_button("✕").clicked() {
                            self.close_file_in_workspace(&path);
                        }
                    }
                });
                ui.separator();

                if let Some(path) = self.active_open_file.clone() {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(path.clone()).small().color(TEXT_MUTED));
                        if ui.button("💾 Save").clicked() {
                            self.save_active_editor_file();
                        }
                    });
                    if let Some(buffer) = self.editor_buffers.get_mut(&path) {
                        let ext = std::path::Path::new(&path)
                            .extension()
                            .and_then(|s| s.to_str())
                            .unwrap_or("txt")
                            .to_lowercase();
                        ui.label(
                            RichText::new(format!("Language: {}", ext))
                                .small()
                                .color(TEXT_MUTED),
                        );
                        let mut layouter = |ui: &egui::Ui, text: &str, wrap_width: f32| {
                            let mut layout_job = Self::syntax_highlight_for_path(&path, text);
                            layout_job.wrap.max_width = wrap_width;
                            ui.fonts(|f| f.layout_job(layout_job))
                        };
                        ScrollArea::vertical().show(ui, |ui| {
                            let resp = ui.add(
                                TextEdit::multiline(buffer)
                                    .desired_width(ui.available_width())
                                    .desired_rows(24)
                                    .font(FontId::monospace(13.0))
                                    .layouter(&mut layouter),
                            );
                            if resp.changed() {
                                self.editor_dirty.insert(path.clone(), true);
                            }
                        });
                    }
                }
            });
    }

    fn render_bottom_terminal_panel(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("terminal_workspace")
            .resizable(true)
            .default_height(220.0)
            .min_height(120.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Terminals").strong().color(GOLD));
                    if ui.button("➕ New Terminal").clicked() {
                        let id = self.create_terminal(
                            "User Terminal".to_string(),
                            TerminalOwner::User,
                            String::new(),
                            None,
                            false,
                        );
                        self.active_terminal_id = Some(id);
                    }
                        if let Some(active_id) = self.active_terminal_id.clone() {
                        if ui.button("Terminate").on_hover_text("Stop the running process").clicked() {
                            self.terminate_terminal(&active_id);
                        }
                        if ui.button("Close").on_hover_text("Close this terminal tab").clicked() {
                            self.remove_terminal(&active_id);
                        }
                    }
                });
                ui.separator();

                ui.horizontal_wrapped(|ui| {
                    for term in self.terminals.clone() {
                        let selected = self.active_terminal_id.as_deref() == Some(term.id.as_str());
                        let label = format!(
                            "{} {}",
                            if term.owner == TerminalOwner::AI { "🤖" } else { "👤" },
                            term.name
                        );
                        let button = egui::Button::new(
                            RichText::new(label)
                                .small()
                                .color(if selected { DARK_TEXT } else { LIGHT_TEXT }),
                        )
                        .fill(if selected { GOLD } else { SKY_BLUE_DARK });
                        if ui.add(button).clicked() {
                            self.active_terminal_id = Some(term.id.clone());
                        }
                    }
                });
                ui.separator();

                if let Some(idx) = self.active_terminal_index() {
                    let mut run_cmd: Option<String> = None;
                    let term = &mut self.terminals[idx];
                    ui.horizontal(|ui| {
                        ui.label(
                            RichText::new(format!(
                                "{}  |  wd: {}  |  {}",
                                term.command,
                                term.working_dir,
                                if term.running { "running" } else { "idle" }
                            ))
                            .small()
                            .color(SKY_BLUE),
                        );
                    });
                    ScrollArea::vertical().max_height(120.0).show(ui, |ui| {
                        ui.add(
                            TextEdit::multiline(&mut term.output)
                                .desired_width(ui.available_width())
                                .desired_rows(8)
                                .interactive(false),
                        );
                    });
                    ui.horizontal(|ui| {
                        let term_input = ui.add(
                            TextEdit::multiline(&mut term.input_buffer)
                                .desired_rows(2)
                                .desired_width((ui.available_width() - TERMINAL_INPUT_RESERVED_WIDTH).max(120.0))
                                .hint_text("command"),
                        );
                        let submit_on_enter = term_input.has_focus()
                            && ctx.input(|i| {
                                i.key_pressed(egui::Key::Enter)
                                    && !i.modifiers.shift
                                    && !i.modifiers.command
                                    && !i.modifiers.ctrl
                                    && !i.modifiers.alt
                            });
                        if (ui.button("Run").clicked() || submit_on_enter)
                            && !term.input_buffer.trim().is_empty()
                        {
                            run_cmd = Some(term.input_buffer.trim().to_string());
                        }
                    });
                    if let Some(cmd) = run_cmd {
                        term.input_buffer.clear();
                        let terminal_id = term.id.clone();
                        self.model_waiting_command_output = false;
                        self.spawn_terminal_command(&terminal_id, cmd);
                    }
                } else {
                    ui.label(RichText::new("No terminal opened").small().color(SKY_BLUE));
                }
            });
    }

    fn render_message(&mut self, ui: &mut egui::Ui, msg: &Message) {
        let is_user = msg.role == Role::User;
        let width = ui.available_width();
        let max_bubble = (width * 0.75).min(700.0);

        ui.with_layout(
            if is_user {
                egui::Layout::right_to_left(egui::Align::TOP)
            } else {
                egui::Layout::left_to_right(egui::Align::TOP)
            },
            |ui| {
                ui.set_max_width(max_bubble);

                let (bg_color, text_color) = if is_user {
                    (GOLD_DARK, DARK_TEXT)
                } else {
                    (BURGUNDY_DARK, WHITE)
                };

                egui::Frame::new()
                    .fill(bg_color)
                    .corner_radius(8.0)
                    .inner_margin(egui::Margin::same(10))
                    .show(ui, |ui| {
                        ui.set_max_width(max_bubble - 20.0);

                        let role_label = if is_user { "You" } else { "Assistant" };
                        ui.label(
                            RichText::new(role_label)
                                .small()
                                .color(if is_user { BURGUNDY } else { SKY_BLUE })
                                .strong(),
                        );

                        if msg.is_streaming && msg.content.is_empty() {
                            ui.spinner();
                        } else {
                            let mut style = (*ui.ctx().style()).clone();
                            style.visuals.override_text_color = Some(text_color);
                            ui.scope(|ui| {
                                ui.set_style(style);
                                CommonMarkViewer::new().show(ui, &mut self.markdown_cache, &msg.content);
                            });
                        }

                        for att in &msg.attachments {
                            ui.separator();
                            if att.image_base64.is_some() {
                                if ui
                                    .button(
                                        RichText::new(format!("🖼 {} (click to download)", att.filename))
                                            .color(SKY_BLUE)
                                            .small(),
                                    )
                                    .clicked()
                                {
                                    self.download_image(att);
                                }
                            } else {
                                ui.label(RichText::new(format!("📄 {}", att.filename)).color(SKY_BLUE).small());
                            }
                        }

                        ui.label(
                            RichText::new(msg.timestamp.format("%H:%M").to_string())
                                .small()
                                .color(Color32::from_rgb(160, 160, 160)),
                        );
                    });
            },
        );
    }
}

impl eframe::App for ChatApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.process_events();
        self.tick_notifications(ctx.input(|i| i.stable_dt));
        if !self.active_requests.is_empty() {
            ctx.request_repaint();
        }

        if self.setup_wizard.is_some() {
            let mut done = false;
            if let Some(wizard) = self.setup_wizard.as_mut() {
                done = wizard.show(ctx, &mut self.settings);
            }
            if done {
                self.setup_wizard = None;
                self.load_models_from_provider();
                if let Err(e) = save_settings(&self.settings) {
                    self.notify(format!("Failed to save settings: {e}"), NotificationKind::Error);
                } else {
                    self.notify("Setup complete", NotificationKind::Info);
                }
            }
        }

        if let Some(action) = self.pending_action.clone() {
            let mut approved = false;
            let mut rejected = false;
            egui::Window::new("⚠ Action Approval Required")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, Vec2::ZERO)
                .show(ctx, |ui| {
                    ui.visuals_mut().override_text_color = Some(LIGHT_TEXT);
                    match &action {
                        PendingAction::WriteFile { path, content } => {
                            ui.label(format!("Write file: {path}"));
                            ui.separator();
                            ScrollArea::vertical().max_height(200.0).show(ui, |ui| {
                                ui.monospace(content);
                            });
                        }
                        PendingAction::ExecuteCommand { command } => {
                            ui.label("Execute command:");
                            ui.monospace(command);
                        }
                        PendingAction::CreateFolder { path } => {
                            ui.label(format!("Create folder: {path}"));
                        }
                        PendingAction::EditFile {
                            path,
                            mode,
                            content,
                        } => {
                            ui.label(format!("Edit file ({mode}): {path}"));
                            ui.separator();
                            ScrollArea::vertical().max_height(200.0).show(ui, |ui| {
                                ui.monospace(content);
                            });
                        }
                        PendingAction::CreatePdf {
                            path,
                            title,
                            content,
                        } => {
                            ui.label(format!("Create PDF: {path}"));
                            ui.label(format!("Title: {title}"));
                            ui.separator();
                            ScrollArea::vertical().max_height(200.0).show(ui, |ui| {
                                ui.monospace(content);
                            });
                        }
                    }
                    ui.separator();
                    ui.horizontal(|ui| {
                        if ui.button(RichText::new("✅ Approve").color(LIGHT_TEXT)).clicked() {
                            approved = true;
                        }
                        if ui.button(RichText::new("❌ Reject").color(LIGHT_TEXT)).clicked() {
                            rejected = true;
                        }
                    });
                });

            if approved {
                let action_session_idx = self.pending_action_session_idx;
                match &action {
                    PendingAction::WriteFile { path, content } => {
                        let full_path = self.resolve_action_path(path);
                        let full_path_str = full_path.to_string_lossy().to_string();
                        self.save_snapshot_if_exists(&full_path_str);
                        if let Err(e) = crate::files::write_text_file(&full_path, content) {
                            self.notify(format!("Write failed: {e}"), NotificationKind::Error);
                            self.append_system_message(
                                action_session_idx,
                                format!("❌ File write failed for `{}`: {}", full_path.display(), e),
                            );
                            if let Some(idx) = action_session_idx {
                                self.dispatch_agent_requests(idx);
                            }
                        } else {
                            self.notify(format!("File written: {}", full_path.display()), NotificationKind::Info);
                            self.ensure_changed_file_tracked(full_path_str.clone());
                            self.open_file_in_workspace(full_path_str.clone());
                            self.append_system_message(
                                action_session_idx,
                                format!("✅ File written (approved): `{}`", full_path.display()),
                            );
                            self.refresh_snapshots();
                            if let Some(idx) = action_session_idx {
                                self.dispatch_agent_requests(idx);
                            }
                        }
                    }
                    PendingAction::ExecuteCommand { command } => {
                        if !self.settings.shell_execution_enabled {
                            self.notify("Shell execution is disabled", NotificationKind::Error);
                            self.append_system_message(
                                action_session_idx,
                                "❌ Command not executed: shell execution is disabled in settings.".to_string(),
                            );
                            if let Some(idx) = action_session_idx {
                                self.dispatch_agent_requests(idx);
                            }
                        } else {
                            let terminal_name = if command.chars().count() > 32 {
                                let shortened: String = command.chars().take(32).collect();
                                format!("AI: {}…", shortened)
                            } else {
                                format!("AI: {}", command)
                            };
                            let tid = self.create_terminal(
                                terminal_name,
                                TerminalOwner::AI,
                                command.clone(),
                                action_session_idx,
                                true,
                            );
                            self.model_waiting_command_output = true;
                            self.spawn_terminal_command(&tid, command.clone());
                            self.notify("AI command started in dedicated terminal", NotificationKind::Info);
                        }
                    }
                    PendingAction::CreateFolder { path } => {
                        let full_path = self.resolve_action_path(path);
                        match std::fs::create_dir_all(&full_path) {
                            Ok(_) => {
                                self.notify(
                                    format!("Folder created: {}", full_path.display()),
                                    NotificationKind::Info,
                                );
                                self.append_system_message(
                                    action_session_idx,
                                    format!("✅ Folder created: `{}`", full_path.display()),
                                );
                                if let Some(idx) = action_session_idx {
                                    self.dispatch_agent_requests(idx);
                                }
                            }
                            Err(e) => {
                                self.notify(format!("Create folder failed: {e}"), NotificationKind::Error);
                                self.append_system_message(
                                    action_session_idx,
                                    format!("❌ Folder create failed for `{}`: {}", full_path.display(), e),
                                );
                                if let Some(idx) = action_session_idx {
                                    self.dispatch_agent_requests(idx);
                                }
                            }
                        }
                    }
                    PendingAction::EditFile {
                        path,
                        mode,
                        content,
                    } => {
                        let full_path = self.resolve_action_path(path);
                        let full_path_str = full_path.to_string_lossy().to_string();
                        self.save_snapshot_if_exists(&full_path_str);
                        let result = if mode == "append" {
                            if let Some(parent) = full_path.parent() {
                                let _ = std::fs::create_dir_all(parent);
                            }
                            std::fs::OpenOptions::new()
                                .create(true)
                                .append(true)
                                .open(&full_path)
                                .and_then(|mut f| {
                                    use std::io::Write;
                                    f.write_all(content.as_bytes())
                                })
                        } else {
                            crate::files::write_text_file(&full_path, content)
                                .map_err(std::io::Error::other)
                        };
                        match result {
                            Ok(_) => {
                                self.notify(format!("File edited: {}", full_path.display()), NotificationKind::Info);
                                self.ensure_changed_file_tracked(full_path_str.clone());
                                self.open_file_in_workspace(full_path_str.clone());
                                self.append_system_message(
                                    action_session_idx,
                                    format!("✅ File edited ({}): `{}`", mode, full_path.display()),
                                );
                                self.refresh_snapshots();
                                if let Some(idx) = action_session_idx {
                                    self.dispatch_agent_requests(idx);
                                }
                            }
                            Err(e) => {
                                self.notify(format!("Edit failed: {e}"), NotificationKind::Error);
                                self.append_system_message(
                                    action_session_idx,
                                    format!("❌ File edit failed for `{}`: {}", full_path.display(), e),
                                );
                                if let Some(idx) = action_session_idx {
                                    self.dispatch_agent_requests(idx);
                                }
                            }
                        }
                    }
                    PendingAction::CreatePdf {
                        path,
                        title,
                        content,
                    } => {
                        let full_path = self.resolve_action_path(path);
                        if full_path
                            .extension()
                            .and_then(|s| s.to_str())
                            .map(|s| !s.eq_ignore_ascii_case("pdf"))
                            .unwrap_or(true)
                        {
                            self.notify(
                                "Create PDF rejected: path must end with .pdf",
                                NotificationKind::Error,
                            );
                            self.append_system_message(
                                action_session_idx,
                                format!(
                                    "❌ PDF create failed for `{}`: file extension must be .pdf",
                                    full_path.display()
                                ),
                            );
                            if let Some(idx) = action_session_idx {
                                self.dispatch_agent_requests(idx);
                            }
                        } else {
                            let full_path_str = full_path.to_string_lossy().to_string();
                            self.save_snapshot_if_exists(&full_path_str);
                            let normalized = format!("{title}\n\n{content}")
                                .replace('\\', "\\\\")
                                .replace('(', "\\(")
                                .replace(')', "\\)");
                            let mut pdf = String::new();
                            pdf.push_str("%PDF-1.4\n");
                            let mut offsets = vec![0usize];
                            let obj1 = "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n";
                            offsets.push(pdf.len());
                            pdf.push_str(obj1);
                            let obj2 = "2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n";
                            offsets.push(pdf.len());
                            pdf.push_str(obj2);
                            let obj3 =
                                "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << >> >>\nendobj\n";
                            offsets.push(pdf.len());
                            pdf.push_str(obj3);
                            let stream_cmd = format!("BT /F1 12 Tf 72 720 Td ({normalized}) Tj ET\n");
                            let obj4 = format!(
                                "4 0 obj\n<< /Length {} >>\nstream\n{}endstream\nendobj\n",
                                stream_cmd.len(),
                                stream_cmd
                            );
                            offsets.push(pdf.len());
                            pdf.push_str(&obj4);
                            let xref_start = pdf.len();
                            pdf.push_str("xref\n0 5\n");
                            pdf.push_str("0000000000 65535 f \n");
                            for off in offsets.into_iter().skip(1) {
                                pdf.push_str(&format!("{off:010} 00000 n \n"));
                            }
                            pdf.push_str("trailer\n<< /Size 5 /Root 1 0 R >>\n");
                            pdf.push_str(&format!("startxref\n{xref_start}\n%%EOF\n"));
                            match std::fs::write(&full_path, pdf.as_bytes()) {
                                Ok(_) => {
                                    self.notify(
                                        format!("PDF file created: {}", full_path.display()),
                                        NotificationKind::Info,
                                    );
                                    self.ensure_changed_file_tracked(full_path_str.clone());
                                    self.open_file_in_workspace(full_path_str.clone());
                                    self.append_system_message(
                                        action_session_idx,
                                        format!("✅ PDF file created: `{}`", full_path.display()),
                                    );
                                    self.refresh_snapshots();
                                    if let Some(idx) = action_session_idx {
                                        self.dispatch_agent_requests(idx);
                                    }
                                }
                                Err(e) => {
                                    self.notify(format!("Create PDF failed: {e}"), NotificationKind::Error);
                                    self.append_system_message(
                                        action_session_idx,
                                        format!("❌ PDF create failed for `{}`: {}", full_path.display(), e),
                                    );
                                    if let Some(idx) = action_session_idx {
                                        self.dispatch_agent_requests(idx);
                                    }
                                }
                            }
                        }
                    }
                }
                self.pending_action = None;
                if let Some(next_action) = self.pending_actions_queue.pop_front() {
                    self.pending_action = Some(next_action);
                } else {
                    self.pending_action_session_idx = None;
                }
            }
            if rejected {
                self.append_system_message(
                    self.pending_action_session_idx,
                    "❌ Action rejected by user. No file write or command execution was performed.".to_string(),
                );
                if let Some(idx) = self.pending_action_session_idx {
                    self.dispatch_agent_requests(idx);
                }
                self.pending_action = None;
                self.pending_actions_queue.clear();
                self.pending_action_session_idx = None;
            }
        }

        if self.show_settings {
            let mut selected_provider = self.settings.selected_provider.clone();
            egui::Window::new("⚙ Settings")
                .collapsible(false)
                .resizable(true)
                .anchor(egui::Align2::CENTER_CENTER, Vec2::ZERO)
                .show(ctx, |ui| {
                    ui.set_min_width(420.0);
                    egui::Grid::new("settings_grid")
                        .num_columns(2)
                        .spacing([10.0, 8.0])
                        .show(ui, |ui| {
                            ui.label("Provider:");
                            egui::ComboBox::from_id_salt("provider_sel")
                                .selected_text(selected_provider.display_name())
                                .show_ui(ui, |ui| {
                                    for p in ApiProvider::all() {
                                        ui.selectable_value(&mut selected_provider, p.clone(), p.display_name());
                                    }
                                });
                            ui.end_row();

                            self.settings.selected_provider = selected_provider.clone();
                            let cfg = self.settings.provider_config_mut(&selected_provider);
                            ui.label("API Key:");
                            ui.add(TextEdit::singleline(&mut cfg.api_key).password(true));
                            ui.end_row();

                            ui.label("Base URL:");
                            ui.add(TextEdit::singleline(&mut cfg.base_url));
                            ui.end_row();

                            ui.label("DB Path:");
                            ui.add(TextEdit::singleline(&mut self.settings.db_path));
                            ui.end_row();

                            ui.label("Dark mode:");
                            ui.checkbox(&mut self.settings.dark_mode, "Enabled");
                            ui.end_row();
                        });

                    ui.separator();
                    ui.horizontal(|ui| {
                        if ui.button("Save").clicked() {
                            self.settings.set_provider_config(
                                &selected_provider,
                                &self.settings.get_provider_api_key(&selected_provider),
                                &self.settings.get_provider_base_url(&selected_provider),
                            );
                            self.settings.default_model = self.custom_model_id.trim().to_string();
                            if self.settings.default_model.is_empty() {
                                self.settings.default_model = self
                                    .current_model()
                                    .map(|m| m.id.clone())
                                    .unwrap_or_else(|| DEFAULT_MODEL_ID.to_string());
                            }
                            if let Err(e) = save_settings(&self.settings) {
                                self.notify(format!("Failed to save settings: {e}"), NotificationKind::Error);
                            } else {
                                self.notify("Settings saved", NotificationKind::Info);
                                Self::apply_theme(ctx, self.settings.dark_mode);
                                self.load_models_from_provider();
                                self.show_settings = false;
                            }
                        }
                        if ui.button("Cancel").clicked() {
                            self.show_settings = false;
                        }
                    });
                });
        }

        if ctx.input(|i| i.modifiers.command && i.key_pressed(egui::Key::S)) {
            self.save_active_editor_file();
        }

        egui::TopBottomPanel::top("layout_controls").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                if ui
                    .button(if self.show_left_sidebar {
                        "◀ Hide Sidebar"
                    } else {
                        "▶ Show Sidebar"
                    })
                    .clicked()
                {
                    self.show_left_sidebar = !self.show_left_sidebar;
                }
                if ui
                    .button(if self.show_activity_panel {
                        "◀ Hide Activity"
                    } else {
                        "▶ Show Activity"
                    })
                    .clicked()
                {
                    self.show_activity_panel = !self.show_activity_panel;
                }
            });
        });

        if self.show_left_sidebar {
            egui::SidePanel::left("sidebar")
                .resizable(true)
                .default_width(260.0)
                .min_width(180.0)
                .frame(egui::Frame::new().fill(BG_SURFACE).inner_margin(egui::Margin::same(8)))
                .show(ctx, |ui| {
                    ui.label(
                        RichText::new("🤖 AI Chat Bot")
                            .font(FontId::proportional(18.0))
                            .color(TEXT_PRIMARY)
                            .strong(),
                    );
                    ui.separator();

                    if ui
                        .add_sized(
                            [ui.available_width(), 32.0],
                            egui::Button::new(RichText::new("＋ New Chat").color(TEXT_DARK)).fill(GOLD),
                        )
                        .clicked()
                    {
                        self.new_session();
                    }

                    ui.separator();
                    ui.collapsing("Conversations", |ui| {
                        ScrollArea::vertical().max_height(200.0).show(ui, |ui| {
                            for idx in 0..self.sessions.len() {
                                let is_current = self.current_session_idx == Some(idx);
                                let name = self
                                    .sessions
                                    .get(idx)
                                    .map(|s| s.name.clone())
                                    .unwrap_or_else(|| "Conversation".to_string());
                                ui.horizontal(|ui| {
                                    let btn = egui::Button::new(
                                        RichText::new(format!("💬 {name}")).color(LIGHT_TEXT),
                                    )
                                    .fill(if is_current { GOLD_DARK } else { SKY_BLUE_DARK })
                                    .min_size(Vec2::new(
                                        (ui.available_width() - DELETE_CHAT_BUTTON_WIDTH)
                                            .max(MIN_CHAT_BUTTON_WIDTH),
                                        28.0,
                                    ));
                                    if ui.add(btn).clicked() {
                                        self.current_session_idx = Some(idx);
                                        if self
                                            .sessions
                                            .get(idx)
                                            .map(|s| s.messages.is_empty())
                                            .unwrap_or(false)
                                        {
                                            self.load_session_messages(idx);
                                        }
                                    }
                                    if ui.small_button("🗑").clicked() {
                                        self.delete_session_by_index(idx);
                                    }
                                });
                            }
                        });
                    });

                    ui.separator();
                    ui.collapsing("Model Selection", |ui| {
                        self.refresh_model_access_outline();
                        let selected_model_name = self
                            .models
                            .get(self.selected_model_idx)
                            .map(|m| m.name.clone())
                            .unwrap_or_else(|| "(none)".to_string());
                        let outline_color = match self.model_access_outline_ok {
                            Some(true) => Color32::from_rgb(0x4C, 0xAF, 0x50),
                            Some(false) => Color32::from_rgb(0xF2, 0xC9, 0x4C),
                            None => BG_SURFACE_ALT,
                        };
                        egui::Frame::new()
                            .fill(BG_SURFACE)
                            .stroke(egui::Stroke::new(1.0, outline_color))
                            .show(ui, |ui| {
                                egui::ComboBox::from_id_salt("model_selector")
                                    .selected_text(RichText::new(selected_model_name).color(TEXT_PRIMARY))
                                    .width(ui.available_width())
                                    .show_ui(ui, |ui| {
                                        for (i, m) in self.models.iter().enumerate() {
                                            ui.selectable_value(
                                                &mut self.selected_model_idx,
                                                i,
                                                RichText::new(&m.name).color(TEXT_PRIMARY),
                                            );
                                        }
                                        if !self.remote_models.is_empty() {
                                            ui.separator();
                                            ui.label(RichText::new("Remote model IDs:").color(TEXT_PRIMARY));
                                            for m in self.remote_models.iter().take(5) {
                                                let mut extra = String::new();
                                                if m.premium {
                                                    extra.push_str(" [Premium]");
                                                }
                                                if m.gated {
                                                    extra.push_str(" [Gated]");
                                                }
                                                ui.label(
                                                    RichText::new(format!("{}{}", m.id, extra))
                                                        .small()
                                                        .color(TEXT_PRIMARY),
                                                );
                                            }
                                        }
                                    });
                            });
                        ui.label(RichText::new("Custom model ID").small().color(TEXT_MUTED));
                        ui.horizontal(|ui| {
                            ui.add(
                                TextEdit::singleline(&mut self.custom_model_id)
                                    .desired_width(
                                        ui.available_width() - CUSTOM_MODEL_INPUT_RESERVED_WIDTH,
                                    )
                                    .hint_text("e.g. gpt-4o, meta-llama/Llama-3.1-8B-Instruct"),
                            );
                            if ui.button("Use").clicked() {
                                if self.custom_model_id.trim().is_empty() {
                                    self.custom_model_id = self
                                        .current_model()
                                        .map(|m| m.id.clone())
                                        .unwrap_or_else(|| DEFAULT_MODEL_ID.to_string());
                                }
                                self.settings.default_model = self.custom_model_id.trim().to_string();
                                let _ = save_settings(&self.settings);
                                self.refresh_model_access_outline();
                            }
                            if ui.button("From list").clicked() {
                                self.custom_model_id.clear();
                                self.settings.default_model = self
                                    .current_model()
                                    .map(|m| m.id.clone())
                                    .unwrap_or_else(|| DEFAULT_MODEL_ID.to_string());
                                let _ = save_settings(&self.settings);
                                self.refresh_model_access_outline();
                            }
                        });
                        if !self.remote_models.is_empty() {
                            let mut picked_model_id: Option<String> = None;
                            ui.horizontal_wrapped(|ui| {
                                ui.label(RichText::new("Quick pick:").small().color(TEXT_MUTED));
                                for model in self.remote_models.iter().take(8) {
                                    let mut label = model.id.clone();
                                    if model.premium {
                                        label.push_str(" [Premium]");
                                    }
                                    if model.gated {
                                        label.push_str(" [Gated]");
                                    }
                                    if ui.small_button(label).clicked() {
                                        picked_model_id = Some(model.id.clone());
                                    }
                                }
                            });
                            if let Some(model_id) = picked_model_id {
                                self.custom_model_id = model_id;
                                self.refresh_model_access_outline();
                            }
                        }

                        let supports_thinking = self
                            .current_model()
                            .map(|m| m.supports_thinking())
                            .unwrap_or(false);
                        if supports_thinking {
                            ui.horizontal(|ui| {
                                ui.label("Thinking mode:");
                                egui::ComboBox::from_id_salt("thinking_mode")
                                    .selected_text(match self.selected_thinking_mode {
                                        ThinkingMode::Disabled => "Disabled",
                                        ThinkingMode::Auto => "Auto",
                                        ThinkingMode::Low => "Low",
                                        ThinkingMode::Medium => "Medium",
                                        ThinkingMode::High => "High",
                                    })
                                    .show_ui(ui, |ui| {
                                        ui.selectable_value(
                                            &mut self.selected_thinking_mode,
                                            ThinkingMode::Disabled,
                                            "Disabled",
                                        );
                                        ui.selectable_value(
                                            &mut self.selected_thinking_mode,
                                            ThinkingMode::Auto,
                                            "Auto",
                                        );
                                        ui.selectable_value(
                                            &mut self.selected_thinking_mode,
                                            ThinkingMode::Low,
                                            "Low",
                                        );
                                        ui.selectable_value(
                                            &mut self.selected_thinking_mode,
                                            ThinkingMode::Medium,
                                            "Medium",
                                        );
                                        ui.selectable_value(
                                            &mut self.selected_thinking_mode,
                                            ThinkingMode::High,
                                            "High",
                                        );
                                    });
                            });
                        }
                    });

                    ui.separator();
                    ui.collapsing("Settings", |ui| {
                        for agent in &mut self.agents {
                            ui.checkbox(&mut agent.enabled, &agent.name);
                        }
                        ui.separator();
                        ui.checkbox(&mut self.settings.shell_execution_enabled, "Shell execution");
                        ui.checkbox(&mut self.settings.dark_mode, "Dark mode");
                        if ui.button("Apply theme").clicked() {
                            Self::apply_theme(ctx, self.settings.dark_mode);
                            let _ = save_settings(&self.settings);
                        }

                        ui.label(RichText::new("Working Dir").strong().color(TEXT_PRIMARY));
                        ui.label(RichText::new(&self.settings.working_directory).small());
                        if ui.button("📁 Change Directory").clicked() {
                            self.select_working_dir();
                        }
                        if ui.button("Shadow rollback").clicked() {
                            self.show_snapshots = true;
                            self.refresh_snapshots();
                        }
                        if ui.button("⚙ Advanced Settings").clicked() {
                            self.show_settings = true;
                        }
                    });
                });
        }

        self.render_file_workspace_panel(ctx);

        if self.show_activity_panel {
            egui::SidePanel::right("activity")
                .resizable(true)
                .default_width(220.0)
                .frame(egui::Frame::new().fill(BURGUNDY_DARK).inner_margin(egui::Margin::same(8)))
                .show(ctx, |ui| {
                    ui.collapsing("Activity Monitor", |ui| {
                        for req in self.active_requests.values() {
                            ui.label(
                                RichText::new(format!("⏳ {} is responding", req.agent_name))
                                    .color(SKY_BLUE),
                            );
                        }
                        ui.separator();
                        ScrollArea::vertical().show(ui, |ui| {
                            for line in self.activity_log.iter().rev().take(20) {
                                ui.label(RichText::new(line).small().color(WHITE));
                            }
                        });
                    });
                });
        }

        self.render_bottom_terminal_panel(ctx);

        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(BURGUNDY).inner_margin(egui::Margin::same(0)))
            .show(ctx, |ui| {
                egui::Frame::new()
                    .fill(BURGUNDY_DARK)
                    .inner_margin(egui::Margin::same(8))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            let session_name = self
                                .current_session()
                                .map(|s| s.name.clone())
                                .unwrap_or_else(|| "No conversation selected".to_string());
                            ui.label(
                                RichText::new(session_name)
                                    .font(FontId::proportional(16.0))
                                    .color(GOLD)
                                    .strong(),
                            );
                            if !self.active_requests.is_empty() {
                                ui.spinner();
                                ui.label(RichText::new("Agents thinking...").color(GOLD).italics());
                            }
                        });
                    });

                let input_height = 100.0;
                let msg_height = ui.available_height() - input_height;

                egui::Frame::new()
                    .fill(BURGUNDY)
                    .inner_margin(egui::Margin {
                        left: 12,
                        right: 12,
                        top: 8,
                        bottom: 4,
                    })
                    .show(ui, |ui| {
                        ScrollArea::vertical()
                            .id_salt("messages_scroll")
                            .max_height(msg_height)
                            .stick_to_bottom(true)
                            .show(ui, |ui| {
                                if let Some(idx) = self.current_session_idx {
                                    let msg_count = self
                                        .sessions
                                        .get(idx)
                                        .map(|s| s.messages.len())
                                        .unwrap_or(0);
                                    for msg_idx in 0..msg_count {
                                        let msg = self.sessions[idx].messages[msg_idx].clone();
                                        self.render_message(ui, &msg);
                                        ui.add_space(6.0);
                                    }
                                } else {
                                    ui.centered_and_justified(|ui| {
                                        ui.label(
                                            RichText::new("Select or create a conversation to begin")
                                                .color(Color32::from_rgb(180, 130, 135))
                                                .font(FontId::proportional(16.0)),
                                        );
                                    });
                                }
                            });
                    });

                egui::Frame::new()
                    .fill(BURGUNDY_DARK)
                    .inner_margin(egui::Margin::same(8))
                    .show(ui, |ui| {
                        if !self.pending_attachments.is_empty() {
                            ui.horizontal_wrapped(|ui| {
                                ui.label(RichText::new("📎 Attachments:").color(GOLD).small());
                                for att in &self.pending_attachments {
                                    ui.label(RichText::new(&att.filename).color(SKY_BLUE).small());
                                }
                                if ui.small_button("✕ Clear").clicked() {
                                    self.pending_attachments.clear();
                                }
                            });
                        }

                        ui.horizontal(|ui| {
                            let text_edit = TextEdit::multiline(&mut self.input_text)
                                .desired_rows(2)
                                .desired_width(ui.available_width() - 120.0)
                                .hint_text("Type a message…")
                                .frame(true);
                            let response = ui.add(text_edit);
                            if response.has_focus()
                                && ctx.input(|i| {
                                    i.key_pressed(egui::Key::Enter)
                                        && !i.modifiers.shift
                                        && !i.modifiers.command
                                        && !i.modifiers.ctrl
                                        && !i.modifiers.alt
                                })
                            {
                                self.send_message();
                            }

                            ui.vertical(|ui| {
                                if ui
                                    .add_sized(
                                        [110.0, 36.0],
                                        egui::Button::new(RichText::new("📤 Send").color(DARK_TEXT)).fill(GOLD),
                                    )
                                    .clicked()
                                {
                                    self.send_message();
                                }
                                if ui
                                    .add_sized(
                                        [110.0, 28.0],
                                        egui::Button::new(RichText::new("📎 Attach").color(LIGHT_TEXT)).fill(SKY_BLUE_DARK),
                                    )
                                    .clicked()
                                {
                                    self.attach_file();
                                }
                            });
                        });
                    });
            });

        self.render_snapshots_window(ctx);
        self.draw_notifications(ctx);
    }
}
