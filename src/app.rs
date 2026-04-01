use base64::{engine::general_purpose, Engine as _};
use chrono::Utc;
use egui::text::LayoutJob;
use egui::{Color32, FontId, RichText, ScrollArea, TextEdit, TextFormat, Vec2};
use egui_commonmark::{CommonMarkCache, CommonMarkViewer};
use serde::Deserialize;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use uuid::Uuid;
use chrono::Timelike;

use crate::api::{
    builtin_models, provider_models, ChatMessage, ModelInfo, RemoteModelInfo,
};
use crate::config::{load_settings, save_settings, ApiProvider, Settings, DEFAULT_MODEL_ID};
use crate::db::{Database, DbFileSnapshot, DbMessage, DbSession};
use crate::executor::{
    ActionExecutor, ApprovalDecision, ApprovalRequest, ExecutionPolicy, ExecutionStatus,
};
use crate::models::{
    get_model_capability, reasoning_config_for_model, ModelReasoningConfig, ReasoningCapability,
    ThinkingMode,
};
use crate::parser::ActionKind;
use crate::rag_engine::{EmbeddingProvider, RagConfig, RagEngine};
use crate::setup::SetupWizard;
use crate::storage::{ChatSession, Storage, StoredMessage, StoredRole};
use crate::swarm::{get_system_prompt, parse_router_plan, AgentRole, RoutedTask};
use crate::telemetry::collect_telemetry_cached;
use crate::watcher::run_workspace_watcher;
use qdrant_client::Qdrant;

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
const STATUS_SUCCESS: Color32 = Color32::from_rgb(0x66, 0xBB, 0x6A);
const STATUS_ERROR: Color32 = Color32::from_rgb(0xEF, 0x53, 0x50);
const CUSTOM_MODEL_INPUT_RESERVED_WIDTH: f32 = 132.0;
const DELETE_CHAT_BUTTON_WIDTH: f32 = 36.0;
const MIN_CHAT_BUTTON_WIDTH: f32 = 80.0;
const TERMINAL_INPUT_RESERVED_WIDTH: f32 = 260.0;
const EVENT_CHANNEL_CAPACITY: usize = 1024;
const WORKFLOW_STEP_DETAIL_MAX_CHARS: usize = 1200;
const TIMEOUT_EXIT_CODE: i32 = 124;
const TERMINAL_RUNNING_MARKER: &str = "[running...]\n";
const QDRANT_URL: &str = "http://127.0.0.1:6334";
const AUTONOMOUS_SCHEDULER_TICK_INTERVAL_SECS: u64 = 30;
const AUTONOMOUS_WORKFLOW_PREFIX: &str = "[AUTONOMOUS CRON-SWARM]";
const AUTONOMOUS_SESSION_NAME: &str = "Autonomous Briefing";
const AUTONOMOUS_TRIGGER_KEYWORDS: [&str; 6] = [
    "every day",
    "every morning",
    "schedule",
    "every weekday",
    "her gün",
    "zamanla",
];
const SWARM_SYNTH_ROLE_PROMPT: &str = r#"You are Synthesizer in a multi-agent swarm.
Output format:
MESSAGE: ...
PLAN: ```json ... ```
Rules:
- Produce a final user-facing summary in MESSAGE using only gathered execution data.
- Include key findings, errors, and blocked steps when relevant.
- If no reliable result is available, clearly state what is missing.
- Keep PLAN as an empty JSON object."#;
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
- "generate_document": Generates a PDF or DOCX from markdown. (Requires: 'format' [pdf/docx], 'path', 'markdown_content')
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
        streamed: bool,
    },
    TerminalChunk {
        terminal_id: String,
        line: String,
        is_stdout: bool,
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
    SwarmStatus {
        request_id: String,
        status: String,
    },
    SwarmWorkflowStep {
        request_id: String,
        title: String,
        details: String,
    },
    ExecutionApprovalRequested {
        request_id: String,
        approval_id: String,
        request: ApprovalRequest,
    },
    WorkspaceTreeLoaded {
        root: crate::storage::FileNode,
        error: Option<String>,
    },
    WorkspaceFileLoaded {
        path: String,
        content: String,
        error: Option<String>,
    },
    AutonomousScheduleTick,
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
    WriteFile {
        path: String,
        content: String,
    },
    ExecuteCommand {
        command: String,
    },
    CreateFolder {
        path: String,
    },
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WorkflowStepStatus {
    Running,
    Success,
    Failed,
}

#[derive(Debug, Clone)]
struct WorkflowStep {
    id: String,
    title: String,
    details: String,
    status: WorkflowStepStatus,
}

#[derive(Debug, Clone)]
struct WorkspaceNodeUiState {
    expanded: bool,
}

#[derive(Debug, Clone)]
struct PendingExecutionApproval {
    request_id: String,
    approval_id: String,
    request: ApprovalRequest,
}

#[derive(Debug, Clone)]
struct ClipboardAttachment {
    filename: String,
    text: String,
}

#[derive(Debug, Clone)]
struct AutonomousSchedule {
    id: String,
    time_24h: String,
    prompt: String,
    enabled: bool,
    last_run_date_utc: Option<String>,
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
    force_reasoning_controls: bool,
    selected_thinking_mode: ThinkingMode,
    binary_reasoning_enabled: bool,
    execution_policy: ExecutionPolicy,

    input_text: String,
    pending_attachments: Vec<Attachment>,

    event_tx: tokio::sync::mpsc::Sender<AppEvent>,
    event_rx: tokio::sync::mpsc::Receiver<AppEvent>,
    tokio_rt: Arc<tokio::runtime::Runtime>,
    active_requests: HashMap<String, ActiveRequest>,

    db_path: String,
    storage: Option<Storage>,
    rag_client: Option<Arc<Qdrant>>,
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
    swarm_status: String,
    swarm_workflow: Vec<WorkflowStep>,
    workflow_expanded: HashMap<String, bool>,
    workflow_request_id: Option<String>,
    workspace_tree: Option<crate::storage::FileNode>,
    workspace_node_state: HashMap<String, WorkspaceNodeUiState>,
    workspace_refresh_tx: tokio::sync::mpsc::Sender<()>,
    workspace_refresh_rx: tokio::sync::mpsc::Receiver<()>,
    workspace_watcher_task: Option<tokio::task::JoinHandle<()>>,
    policy_block_dialog: Option<String>,
    pending_execution_approval: Option<PendingExecutionApproval>,
    approval_waiters:
        Arc<Mutex<HashMap<String, tokio::sync::oneshot::Sender<ApprovalDecision>>>>,
    autonomous_schedules: Vec<AutonomousSchedule>,
    autonomous_time_input: String,
    autonomous_prompt_input: String,
}

impl ChatApp {
    fn ingest_dropped_files(&mut self, ctx: &egui::Context) {
        let dropped_files = ctx.input(|i| i.raw.dropped_files.clone());
        if dropped_files.is_empty() {
            return;
        }

        let mut added = 0usize;
        for file in dropped_files {
            if let Some(path) = file.path {
                match crate::files::read_file(&path) {
                    Ok(fc) => {
                        self.pending_attachments.push(Attachment {
                            filename: fc.filename,
                            text: fc.text,
                            image_base64: fc.image_base64,
                            mime_type: fc.mime_type,
                            raw_bytes: fc.raw_bytes,
                        });
                        added += 1;
                    }
                    Err(e) => {
                        self.notify(
                            format!("Failed to ingest dropped file: {e}"),
                            NotificationKind::Error,
                        );
                    }
                }
            } else if let Some(bytes) = file.bytes {
                let raw = bytes.to_vec();
                let filename = file
                    .name
                    .trim()
                    .split(std::path::MAIN_SEPARATOR)
                    .next_back()
                    .unwrap_or("pasted.bin")
                    .to_string();
                let mime_type = if file.mime.trim().is_empty() {
                    "application/octet-stream".to_string()
                } else {
                    file.mime.clone()
                };
                let text = std::str::from_utf8(&raw).ok().map(|s| s.to_string());
                let image_base64 = if mime_type.starts_with("image/") {
                    Some(general_purpose::STANDARD.encode(&raw))
                } else {
                    None
                };
                self.pending_attachments.push(Attachment {
                    filename,
                    text,
                    image_base64,
                    mime_type,
                    raw_bytes: raw,
                });
                added += 1;
            }
        }
        if added > 0 {
            self.notify(
                format!("Added {added} dropped attachment(s)"),
                NotificationKind::Info,
            );
        }
    }

    fn clipboard_attachment(&self) -> Option<ClipboardAttachment> {
        let mut clipboard = arboard::Clipboard::new().ok()?;
        let text = clipboard.get_text().ok()?;
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return None;
        }
        let lines: Vec<&str> = trimmed.lines().collect();
        let code_keyword_hits = ["fn ", "def ", "class ", "function ", "import ", "let ", "const "]
            .iter()
            .filter(|kw| trimmed.contains(**kw))
            .count();
        let punctuated_lines = lines
            .iter()
            .filter(|line| {
                let l = line.trim();
                l.ends_with(';') || l.ends_with('{') || l.ends_with('}')
            })
            .count();
        let looks_like_code = trimmed.contains("```")
            || code_keyword_hits >= 1
            || (lines.len() >= 3 && punctuated_lines >= 2);
        let filename = if trimmed.lines().count() > 1 || looks_like_code {
            "clipboard_code.txt".to_string()
        } else {
            "clipboard_text.txt".to_string()
        };
        Some(ClipboardAttachment {
            filename,
            text: trimmed.to_string(),
        })
    }

    fn attach_from_clipboard(&mut self) {
        let Some(att) = self.clipboard_attachment() else {
            self.notify(
                "Clipboard does not contain text attachment data",
                NotificationKind::Error,
            );
            return;
        };
        self.pending_attachments.push(Attachment {
            filename: att.filename,
            text: Some(att.text.clone()),
            image_base64: None,
            mime_type: "text/plain".to_string(),
            raw_bytes: att.text.into_bytes(),
        });
        self.notify("Clipboard attachment added", NotificationKind::Info);
    }

    fn request_workspace_refresh(&self) {
        let _ = self.workspace_refresh_tx.try_send(());
    }

    fn spawn_workspace_cache_rebuild(&self) {
        let workspace_root = self.settings.working_directory.clone();
        let tx = self.event_tx.clone();
        self.tokio_rt.spawn(async move {
            match crate::storage::rebuild_file_cache(workspace_root).await {
                Ok(root) => {
                    let _ = tx
                        .send(AppEvent::WorkspaceTreeLoaded { root, error: None })
                        .await;
                }
                Err(e) => {
                    let fallback = crate::storage::FileNode {
                        name: "workspace".to_string(),
                        path: String::new(),
                        is_dir: true,
                        children: vec![],
                    };
                    let _ = tx
                        .send(AppEvent::WorkspaceTreeLoaded {
                            root: fallback,
                            error: Some(e.to_string()),
                        })
                        .await;
                }
            }
        });
    }

    fn spawn_workspace_file_open(&self, path: String) {
        let tx = self.event_tx.clone();
        self.tokio_rt.spawn(async move {
            match tokio::fs::read_to_string(&path).await {
                Ok(content) => {
                    let _ = tx
                        .send(AppEvent::WorkspaceFileLoaded {
                            path,
                            content,
                            error: None,
                        })
                        .await;
                }
                Err(e) => {
                    let _ = tx
                        .send(AppEvent::WorkspaceFileLoaded {
                            path,
                            content: String::new(),
                            error: Some(e.to_string()),
                        })
                        .await;
                }
            }
        });
    }

    fn process_workspace_refresh_signals(&mut self) {
        let mut should_refresh = false;
        while self.workspace_refresh_rx.try_recv().is_ok() {
            should_refresh = true;
        }
        if should_refresh {
            self.spawn_workspace_cache_rebuild();
        }
    }

    fn start_workspace_watcher_task(&mut self) {
        if let Some(handle) = self.workspace_watcher_task.take() {
            handle.abort();
        }
        let workspace_root = self.settings.working_directory.clone();
        let tx = self.workspace_refresh_tx.clone();
        self.workspace_watcher_task = Some(self.tokio_rt.spawn(async move {
            if let Err(err) = run_workspace_watcher(workspace_root, tx).await {
                eprintln!("workspace watcher failed: {err}");
            }
        }));
    }

    fn render_workspace_tree_node(
        &mut self,
        ui: &mut egui::Ui,
        node: &crate::storage::FileNode,
        request_refresh: &mut bool,
    ) {
        let is_dir = node.is_dir;
        let node_marker = if is_dir { "[D]" } else { "[F]" };
        let label = format!("{node_marker} {}", node.name);

        if is_dir {
            let expanded_now = self
                .workspace_node_state
                .get(&node.path)
                .map(|s| s.expanded)
                .unwrap_or(false);
            let chevron = if expanded_now { "▾" } else { "▸" };
            let mut toggle = false;
            ui.horizontal(|ui| {
                if ui
                    .add(egui::Button::new(chevron).frame(false))
                    .clicked()
                {
                    toggle = true;
                }
                let resp = ui.selectable_label(false, RichText::new(label).color(TEXT_PRIMARY));
                resp.context_menu(|ui| {
                    if ui.button("Refresh").clicked() {
                        *request_refresh = true;
                        ui.close_menu();
                    }
                });
            });
            if toggle {
                let entry = self
                    .workspace_node_state
                    .entry(node.path.clone())
                    .or_insert(WorkspaceNodeUiState { expanded: false });
                entry.expanded = !entry.expanded;
            }
            let expanded_after = self
                .workspace_node_state
                .get(&node.path)
                .map(|s| s.expanded)
                .unwrap_or(false);
            if expanded_after {
                ui.indent(format!("indent_{}", node.path), |ui| {
                    for child in &node.children {
                        self.render_workspace_tree_node(ui, child, request_refresh);
                    }
                });
            }
            return;
        }

        let selected = self.active_open_file.as_deref() == Some(node.path.as_str());
        let resp = ui.selectable_label(
            selected,
            RichText::new(label).color(if selected { GOLD } else { TEXT_PRIMARY }),
        );
        if resp.double_clicked() {
            self.spawn_workspace_file_open(node.path.clone());
        }
        resp.context_menu(|ui| {
            if ui.button("Refresh").clicked() {
                *request_refresh = true;
                ui.close_menu();
            }
        });
    }

    fn sanitize_workflow_details(raw: &str) -> String {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return "No additional details.".to_string();
        }
        // Cap detail text to ~1200 chars so expanded blocks stay within a typical
        // laptop-height viewport and avoid pushing active steps out of view.
        if trimmed.len() <= WORKFLOW_STEP_DETAIL_MAX_CHARS {
            return trimmed.to_string();
        }
        let clipped: String = trimmed
            .chars()
            .take(WORKFLOW_STEP_DETAIL_MAX_CHARS)
            .collect();
        format!("{clipped}\n…")
    }

    fn active_reasoning_config(&self) -> ModelReasoningConfig {
        if self.force_reasoning_controls {
            ModelReasoningConfig {
                capability: ReasoningCapability::Tiered,
                tiered_modes: &[
                    ThinkingMode::Low,
                    ThinkingMode::Medium,
                    ThinkingMode::High,
                ],
                binary_label: "Enable Reasoning / Thinking",
                tiered_label: "Thinking level:",
            }
        } else {
            reasoning_config_for_model(&self.selected_model_id())
        }
    }

    fn ensure_thinking_mode_valid_for_model(&mut self) {
        let cfg = self.active_reasoning_config();
        if matches!(cfg.capability, ReasoningCapability::Tiered)
            && !cfg.tiered_modes.contains(&self.selected_thinking_mode)
        {
            if let Some(default_mode) = cfg.tiered_modes.first().copied() {
                self.selected_thinking_mode = default_mode;
            }
        }
    }

    fn activate_model_selection(&mut self) {
        self.settings.default_model = self.selected_model_id();
        let _ = save_settings(&self.settings);
        self.ensure_thinking_mode_valid_for_model();
        self.refresh_model_access_outline();
    }

    fn begin_workflow_step(&mut self, request_id: &str, title: String, details: String) {
        if self.workflow_request_id.as_deref() != Some(request_id) {
            self.workflow_request_id = Some(request_id.to_string());
            self.swarm_workflow.clear();
            self.workflow_expanded.clear();
        }

        if let Some(last_running) = self
            .swarm_workflow
            .iter_mut()
            .rev()
            .find(|step| matches!(step.status, WorkflowStepStatus::Running))
        {
            last_running.status = WorkflowStepStatus::Success;
        }

        let step_id = Uuid::new_v4().to_string();
        self.workflow_expanded.insert(step_id.clone(), false);
        self.swarm_workflow.push(WorkflowStep {
            id: step_id,
            title,
            details: Self::sanitize_workflow_details(&details),
            status: WorkflowStepStatus::Running,
        });
    }

    fn finalize_workflow(&mut self, request_id: &str, success: bool) {
        if self.workflow_request_id.as_deref() != Some(request_id) {
            return;
        }
        if let Some(last_running) = self
            .swarm_workflow
            .iter_mut()
            .rev()
            .find(|step| matches!(step.status, WorkflowStepStatus::Running))
        {
            last_running.status = if success {
                WorkflowStepStatus::Success
            } else {
                WorkflowStepStatus::Failed
            };
        }
    }

    fn render_workflow_visualizer(&mut self, ui: &mut egui::Ui) {
        egui::Frame::new()
            .fill(Color32::from_rgba_unmultiplied(0x22, 0x24, 0x2A, 0xB8))
            .corner_radius(8.0)
            .stroke(egui::Stroke::new(1.0, BG_SURFACE_ALT))
            .inner_margin(egui::Margin::same(10))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new("Swarm Workflow")
                            .strong()
                            .color(TEXT_PRIMARY)
                            .size(15.0),
                    );
                    ui.add_space(8.0);
                    if !self.active_requests.is_empty() {
                        ui.spinner();
                    }
                    ui.label(RichText::new(&self.swarm_status).small().color(TEXT_MUTED));
                });
                ui.add_space(8.0);

                if self.swarm_workflow.is_empty() {
                    ui.label(
                        RichText::new("No workflow steps yet. Send a request to start.")
                            .small()
                            .color(TEXT_MUTED),
                    );
                    return;
                }

                ScrollArea::vertical().max_height(180.0).show(ui, |ui| {
                    for idx in 0..self.swarm_workflow.len() {
                        let (step_id, step_title, step_details, step_status) = {
                            let step = &self.swarm_workflow[idx];
                            (
                                step.id.clone(),
                                step.title.clone(),
                                step.details.clone(),
                                step.status,
                            )
                        };
                        ui.horizontal(|ui| {
                            let expanded = self
                                .workflow_expanded
                                .entry(step_id.clone())
                                .or_insert(false);
                            let chevron = if *expanded { "▾" } else { "▸" };
                            if ui
                                .add(
                                    egui::Button::new(RichText::new(chevron).color(TEXT_PRIMARY))
                                        .frame(false),
                                )
                                .clicked()
                            {
                                *expanded = !*expanded;
                            }

                            match step_status {
                                WorkflowStepStatus::Running => ui.spinner(),
                                WorkflowStepStatus::Success => {
                                    ui.label(RichText::new("✅").color(STATUS_SUCCESS))
                                }
                                WorkflowStepStatus::Failed => {
                                    ui.label(RichText::new("❌").color(STATUS_ERROR))
                                }
                            };

                            ui.label(RichText::new(step_title).color(TEXT_PRIMARY));
                        });

                        if *self.workflow_expanded.get(&step_id).unwrap_or(&false) {
                            egui::Frame::new()
                                .fill(BG_SURFACE_ALT)
                                .corner_radius(6.0)
                                .inner_margin(egui::Margin::same(8))
                                .show(ui, |ui| {
                                    ui.label(
                                        RichText::new(step_details)
                                            .small()
                                            .monospace()
                                            .color(TEXT_MUTED),
                                    );
                                });
                        }
                        ui.add_space(4.0);
                    }
                });
            });
    }

    fn clean_copy_text(content: &str) -> String {
        let parsed = crate::parser::parse_response(content);
        let mut out = String::new();

        if let Some(message) = parsed.message.as_deref().map(str::trim).filter(|m| !m.is_empty()) {
            out.push_str(message);
        } else if !parsed.fallback_text.trim().is_empty() {
            out.push_str(parsed.fallback_text.trim());
        }

        if !parsed.plan_items.is_empty() {
            if !out.is_empty() {
                out.push_str("\n\n");
            }
            out.push_str("PLAN:\n");
            for item in parsed.plan_items {
                out.push_str("- [ ] ");
                out.push_str(item.trim());
                out.push('\n');
            }
            out = out.trim_end().to_string();
        }
        out.trim().to_string()
    }

    fn copy_to_clipboard(&mut self, raw_content: &str) {
        let text = Self::clean_copy_text(raw_content);
        if text.is_empty() {
            self.notify("Nothing to copy", NotificationKind::Info);
            return;
        }

        match arboard::Clipboard::new() {
            Ok(mut clipboard) => match clipboard.set_text(text) {
                Ok(_) => self.notify("Copied to clipboard", NotificationKind::Info),
                Err(err) => {
                    eprintln!("Clipboard set_text failed: {err}");
                    self.notify("Clipboard write failed", NotificationKind::Error);
                }
            },
            Err(err) => {
                eprintln!("Clipboard initialization failed: {err}");
                self.notify("Clipboard unavailable", NotificationKind::Error);
            }
        }
    }

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
                    let _ = tx
                        .send(AppEvent::InitialSessionsLoaded {
                            sessions,
                            error: None,
                        })
                        .await;
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
                                let is_image =
                                    matches!(ext.as_str(), "png" | "jpg" | "jpeg" | "gif" | "webp");
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
                format!(
                    "[Command produced no stdout output (exit code: {})]",
                    exit_code
                )
            }
        } else {
            stdout.to_string()
        }
    }

    fn pending_actions_from_agent_actions(
        actions: Vec<crate::parser::AgentAction>,
    ) -> Vec<PendingAction> {
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
                crate::parser::ActionKind::GenerateDocument => None,
                crate::parser::ActionKind::RunCmd => {
                    let command = item.parameters.command?.trim().to_string();
                    if command.is_empty() {
                        None
                    } else {
                        Some(PendingAction::ExecuteCommand { command })
                    }
                }
                // Web actions execute via the internal web engine and do not require pending
                // file/command approval UI prompts in this flow.
                crate::parser::ActionKind::SearchWeb | crate::parser::ActionKind::ReadUrl => None,
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

    fn current_reasoning_capability(&self) -> ReasoningCapability {
        if self.force_reasoning_controls {
            ReasoningCapability::Tiered
        } else {
            get_model_capability(&self.selected_model_id())
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
        self.terminals[idx].output.push_str(TERMINAL_RUNNING_MARKER);
        let wd = self.settings.working_directory.clone();
        let tx = self.event_tx.clone();
        let tid = terminal_id.to_string();
        self.tokio_rt.spawn(async move {
            let executor = ActionExecutor::new(wd);
            let tx_chunks = tx.clone();
            let tid_chunks = tid.clone();
            let outcome = executor
                .execute_command_streaming(&command, move |is_stdout, line| {
                    let _ = tx_chunks.blocking_send(AppEvent::TerminalChunk {
                        terminal_id: tid_chunks.clone(),
                        line,
                        is_stdout,
                    });
                })
                .await;
            match outcome {
                Ok(report) => {
                    let _ = tx
                        .send(AppEvent::TerminalFinished {
                            terminal_id: tid,
                            stdout: report.stdout,
                            stderr: report.stderr,
                            exit_code: report.exit_code,
                            streamed: true,
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
                            streamed: false,
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
                let _ = tx
                    .send(AppEvent::DbError(format!("Failed to save message: {e}")))
                    .await;
            } else if let Err(e) = result {
                let _ = tx
                    .send(AppEvent::DbError(format!("Failed to save message: {e}")))
                    .await;
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
                let _ = tx
                    .send(AppEvent::DbError(format!(
                        "Failed to create DB session: {e}"
                    )))
                    .await;
            } else if let Err(e) = result {
                let _ = tx
                    .send(AppEvent::DbError(format!(
                        "Failed to create DB session: {e}"
                    )))
                    .await;
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
                let _ = tx
                    .send(AppEvent::DbError(format!(
                        "Failed to delete conversation: {e}"
                    )))
                    .await;
            } else if let Err(e) = result {
                let _ = tx
                    .send(AppEvent::DbError(format!(
                        "Failed to delete conversation: {e}"
                    )))
                    .await;
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
                    .send(AppEvent::DbError(format!(
                        "Failed to persist file snapshot: {e}"
                    )))
                    .await;
            } else if let Err(e) = result {
                let _ = tx
                    .send(AppEvent::DbError(format!(
                        "Failed to persist file snapshot: {e}"
                    )))
                    .await;
            }
        });
    }

    fn spawn_refresh_snapshots(&self) {
        let db_path = self.db_path.clone();
        let tx = self.event_tx.clone();
        self.tokio_rt.spawn(async move {
            let result =
                tokio::task::spawn_blocking(move || -> Result<Vec<DbFileSnapshot>, String> {
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
        let (event_tx, event_rx) = tokio::sync::mpsc::channel(EVENT_CHANNEL_CAPACITY);
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
        let (workspace_refresh_tx, workspace_refresh_rx) =
            tokio::sync::mpsc::channel(EVENT_CHANNEL_CAPACITY);
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
            force_reasoning_controls: false,
            // Tiered-capable models expose Low/Medium/High only, so use Medium as neutral default.
            selected_thinking_mode: ThinkingMode::Medium,
            binary_reasoning_enabled: true,
            execution_policy: ExecutionPolicy::Manual,
            input_text: String::new(),
            pending_attachments: vec![],
            event_tx,
            event_rx,
            tokio_rt: rt,
            active_requests: HashMap::new(),
            db_path,
            storage,
            rag_client: crate::rag_engine::RagEngine::<OpenAIEmbeddingProvider>::init_shared_qdrant_client(QDRANT_URL).ok(),
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
            swarm_status: "✅ Ready".to_string(),
            swarm_workflow: vec![],
            workflow_expanded: HashMap::new(),
            workflow_request_id: None,
            workspace_tree: None,
            workspace_node_state: HashMap::new(),
            workspace_refresh_tx,
            workspace_refresh_rx,
            workspace_watcher_task: None,
            policy_block_dialog: None,
            pending_execution_approval: None,
            approval_waiters: Arc::new(Mutex::new(HashMap::new())),
            autonomous_schedules: vec![],
            autonomous_time_input: "08:00".to_string(),
            autonomous_prompt_input: String::new(),
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
        app.start_workspace_watcher_task();
        app.request_workspace_refresh();
        app.hydrate_autonomous_schedules_from_settings();
        app.start_autonomous_scheduler_task();
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
            style.visuals.window_stroke = egui::Stroke::new(1.0, BG_SURFACE_ALT);
            style.visuals.widgets.noninteractive.weak_bg_fill = BG_SURFACE;
            style.visuals.widgets.noninteractive.bg_fill = BG_SURFACE;
            style.visuals.widgets.noninteractive.fg_stroke.color = TEXT_PRIMARY;
            style.visuals.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0, BG_SURFACE_ALT);
            style.visuals.widgets.inactive.bg_fill = BG_SURFACE_ALT;
            style.visuals.widgets.inactive.fg_stroke.color = TEXT_PRIMARY;
            style.visuals.widgets.inactive.bg_stroke = egui::Stroke::new(1.0, BG_SURFACE_ALT);
            style.visuals.widgets.hovered.bg_fill = BG_SURFACE_ALT;
            style.visuals.widgets.hovered.fg_stroke.color = TEXT_PRIMARY;
            style.visuals.widgets.hovered.bg_stroke = egui::Stroke::new(1.0, ACCENT_SOFT);
            style.visuals.widgets.active.bg_fill = ACCENT_SOFT;
            style.visuals.widgets.active.fg_stroke.color = TEXT_PRIMARY;
            style.visuals.widgets.active.bg_stroke = egui::Stroke::new(1.0, ACCENT);
            style.visuals.widgets.open.bg_fill = BG_SURFACE;
            style.visuals.widgets.open.fg_stroke.color = TEXT_PRIMARY;
            style.visuals.selection.bg_fill = ACCENT_SOFT;
            style.visuals.selection.stroke.color = TEXT_PRIMARY;
            style.spacing.item_spacing = Vec2::new(8.0, 8.0);
            style.spacing.button_padding = Vec2::new(10.0, 6.0);
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

    fn resolve_pending_execution_approval(
        &mut self,
        approval_id: &str,
        decision: ApprovalDecision,
    ) {
        let sender = self
            .approval_waiters
            .lock()
            .ok()
            .and_then(|mut waiters| waiters.remove(approval_id));
        if let Some(sender) = sender {
            let _ = sender.send(decision);
        }
        self.pending_execution_approval = None;
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
            let model_ids: Vec<crate::api::openai::RemoteModelInfo> =
                client.list_models().await.unwrap_or_default();
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

        if !self
            .sessions
            .get(session_idx)
            .map(|s| s.messages.iter().any(|m| m.role == Role::User))
            .unwrap_or(false)
        {
            self.activity_log
                .push("Swarm dispatch skipped: no user message in session".to_string());
            return;
        }

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
                agent_name: "Swarm".to_string(),
            },
        );
        self.activity_log.push("Swarm started response".to_string());
        self.swarm_status = "🔄 Router is analyzing...".to_string();
        self.workflow_request_id = Some(request_id.clone());
        self.swarm_workflow.clear();
        self.workflow_expanded.clear();

        let mut api_messages: Vec<ChatMessage> = vec![];
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
        let shell_enabled = self.settings.shell_execution_enabled;

        if let Some(session) = self.sessions.get(session_idx) {
            for msg in &session.messages {
                if msg.is_streaming {
                    continue;
                }
                let role = msg.role.as_str();
                if msg.attachments.iter().any(|a| a.image_base64.is_some()) {
                    for att in &msg.attachments {
                        if let Some(b64) = &att.image_base64 {
                            let mut content = msg.content.clone();
                            Self::push_attachment_with_metadata(&mut content, att);
                            api_messages.push(ChatMessage::with_image(
                                role,
                                &content,
                                b64,
                                &att.mime_type,
                            ));
                        }
                    }
                } else {
                    let mut content = msg.content.clone();
                    for att in &msg.attachments {
                        Self::push_attachment_with_metadata(&mut content, att);
                    }
                    api_messages.push(ChatMessage::text(role, &content));
                }
            }
        }

        let reasoning_capability = self.current_reasoning_capability();
        let tiered_reasoning_level = if matches!(reasoning_capability, ReasoningCapability::Tiered) {
            Some(self.selected_thinking_mode)
        } else {
            None
        };
        let binary_reasoning_enabled = self.binary_reasoning_enabled;
        let execution_policy = self.execution_policy;

        let event_tx = self.event_tx.clone();
        let req_id = request_id.clone();
        let model_id = model_id.clone();
        let shared_rag_client = self.rag_client.clone();
        let approval_waiters = self.approval_waiters.clone();

        self.tokio_rt.spawn(async move {
            let mut rag_context = "LOCAL REPOSITORY CONTEXT:\n- No relevant snippets found.".to_string();
            let _ = event_tx
                .send(AppEvent::SwarmWorkflowStep {
                    request_id: req_id.clone(),
                    title: "Setting up environment".to_string(),
                    details: format!(
                        "provider_base_url={}\nworkspace={}\nmodel={}",
                        base_url, working_dir, model_id
                    ),
                })
                .await;
            if !api_key.is_empty() && !query.trim().is_empty() {
                let provider = OpenAIEmbeddingProvider {
                    api_key: api_key.clone(),
                    base_url: base_url.clone(),
                    model: "text-embedding-3-small".to_string(),
                };
                let rag_cfg = RagConfig::with_workspace(working_dir.clone(), 1536);
                if let Some(rag_client) = shared_rag_client {
                    if let Ok(engine) = RagEngine::new(rag_cfg, provider, rag_client).await {
                        if let Ok(snippets) = engine.semantic_search(&query).await {
                            rag_context = RagEngine::<OpenAIEmbeddingProvider>::format_repository_context(&snippets);
                        }
                    }
                }
            }

            let client = crate::api::OpenAIClient::new(&api_key, &base_url);
            let _ = event_tx
                .send(AppEvent::SwarmStatus {
                    request_id: req_id.clone(),
                    status: "🔄 Router is analyzing...".to_string(),
                })
                .await;

            let mut router_messages = api_messages.clone();
            let router_prompt = get_system_prompt(&AgentRole::Router);
            let router_system_prompt = format!(
                "{CORE_OS_SYSTEM_PROMPT}\n\n{telemetry_context}\n\n{rag_context}\n\n{router_prompt}\n\nUser request:\n{query}"
            );
            let _ = event_tx
                .send(AppEvent::SwarmWorkflowStep {
                    request_id: req_id.clone(),
                    title: "Router: planning workflow".to_string(),
                    details: router_system_prompt.clone(),
                })
                .await;
            router_messages.insert(
                0,
                ChatMessage::with_cache_control("system", &router_system_prompt),
            );

            let router_result = client
                .chat_completion(
                    &model_id,
                    router_messages,
                    reasoning_capability,
                    tiered_reasoning_level.as_ref(),
                    binary_reasoning_enabled,
                    |_| {},
                )
                .await;

            let mut queue = match router_result {
                Ok(raw) => parse_router_plan(&raw),
                Err(e) => {
                    let _ = event_tx
                        .send(AppEvent::ResponseError(
                            req_id,
                            format!("Router failed: {}", e),
                        ))
                        .await;
                    return;
                }
            };

            if queue.is_empty() {
                queue.push(RoutedTask {
                    agent: AgentRole::CodeArchitect.as_str().to_string(),
                    task: query.clone(),
                });
            }

            let mut swarm_memory = String::new();
            let mut final_parts: Vec<String> = Vec::new();
            let executor = ActionExecutor::new(working_dir);

            let mut auto_approve_enabled = false;

            for step in queue {
                let Some(role) = AgentRole::from_plan_name(&step.agent) else {
                    continue;
                };

                let status = match role {
                    AgentRole::Router => "🔄 Router is analyzing...".to_string(),
                    AgentRole::SystemAdmin => "🛠️ SystemAdmin is working...".to_string(),
                    AgentRole::CodeArchitect => "🧠 CodeArchitect is writing...".to_string(),
                    AgentRole::WebResearcher => "🌐 WebResearcher is researching...".to_string(),
                };
                let _ = event_tx
                    .send(AppEvent::SwarmStatus {
                        request_id: req_id.clone(),
                        status,
                    })
                    .await;
                let _ = event_tx
                    .send(AppEvent::SwarmWorkflowStep {
                        request_id: req_id.clone(),
                        title: format!("{}: analyze task", role.as_str()),
                        details: step.task.clone(),
                    })
                    .await;

                let mut role_messages = api_messages.clone();
                let role_prompt = get_system_prompt(&role);
                let memory_block = if swarm_memory.trim().is_empty() {
                    "No prior agent outputs.".to_string()
                } else {
                    swarm_memory.clone()
                };
                let role_system_prompt = format!(
                    "{CORE_OS_SYSTEM_PROMPT}\n\n{telemetry_context}\n\n{rag_context}\n\n{role_prompt}\n\nAssigned task:\n{}\n\nSwarm memory:\n{}",
                    step.task, memory_block
                );
                role_messages.insert(0, ChatMessage::with_cache_control("system", &role_system_prompt));

                let role_raw = match client
                    .chat_completion(
                        &model_id,
                        role_messages.clone(),
                        reasoning_capability,
                        tiered_reasoning_level.as_ref(),
                        binary_reasoning_enabled,
                        |_| {},
                    )
                    .await
                {
                    Ok(resp) => resp,
                    Err(e) => {
                        swarm_memory.push_str(&format!("\n[{} error]\n{}\n", role.as_str(), e));
                        continue;
                    }
                };

                let mut parsed = crate::parser::parse_response(&role_raw);
                if parsed.json_schema_drift {
                    swarm_memory.push_str(&format!(
                        "\n[{} parse error]\n{}\n",
                        role.as_str(),
                        parsed
                            .json_parse_error
                            .as_deref()
                            .unwrap_or("Unknown parser error")
                    ));
                    swarm_memory.push_str("\n[System feedback injected]\n");
                    swarm_memory.push_str(crate::parser::parser_self_correction_feedback());
                    swarm_memory.push('\n');

                    let mut retry_messages = role_messages.clone();
                    retry_messages.push(ChatMessage::text(
                        "system",
                        crate::parser::parser_self_correction_feedback(),
                    ));

                    match client
                        .chat_completion(
                            &model_id,
                            retry_messages,
                            reasoning_capability,
                            tiered_reasoning_level.as_ref(),
                            binary_reasoning_enabled,
                            |_| {},
                        )
                        .await
                    {
                        Ok(retry_raw) => {
                            let retry_parsed = crate::parser::parse_response(&retry_raw);
                            if retry_parsed.json_schema_drift {
                                swarm_memory.push_str(&format!(
                                    "\n[{} retry parse error]\n{}\n",
                                    role.as_str(),
                                    retry_parsed
                                        .json_parse_error
                                        .as_deref()
                                        .unwrap_or("Unknown parser error")
                                ));
                                continue;
                            }
                            parsed = retry_parsed;
                        }
                        Err(e) => {
                            swarm_memory
                                .push_str(&format!("\n[{} retry error]\n{}\n", role.as_str(), e));
                            continue;
                        }
                    }
                }
                let display_text = parsed
                    .message
                    .as_deref()
                    .filter(|m| !m.trim().is_empty())
                    .unwrap_or(parsed.fallback_text.as_str())
                    .trim()
                    .to_string();
                if !display_text.is_empty() {
                    final_parts.push(format!("{}: {}", role.as_str(), display_text));
                }

                if let Some(err) = parsed.json_parse_error.as_deref() {
                    swarm_memory.push_str(&format!("\n[{} parse error]\n{}\n", role.as_str(), err));
                }

                let filtered_actions: Vec<crate::parser::AgentAction> = parsed
                    .actions
                    .into_iter()
                    .filter(|a| match role {
                        AgentRole::CodeArchitect => !matches!(a.action, ActionKind::RunCmd),
                        AgentRole::SystemAdmin => !matches!(a.action, ActionKind::RunCmd) || shell_enabled,
                        AgentRole::WebResearcher => matches!(
                            a.action,
                            ActionKind::SearchWeb | ActionKind::ReadUrl
                        ),
                        AgentRole::Router => {
                            swarm_memory.push_str(
                                "\n[Router warning]\nRouter emitted actions unexpectedly; actions were ignored.\n",
                            );
                            false
                        }
                    })
                    .collect();

                if filtered_actions.is_empty() {
                    swarm_memory.push_str(&format!(
                        "\n[{}]\nNo executable actions.\n",
                        role.as_str()
                    ));
                    continue;
                }

                swarm_memory.push_str(&format!("\n[{} execution]\n", role.as_str()));
                for action in filtered_actions {
                    let request_id_for_approval = req_id.clone();
                    let approval_event_tx = event_tx.clone();
                    let approval_waiters = approval_waiters.clone();
                    match executor
                        .execute_action_with_permission(
                            action,
                            execution_policy,
                            auto_approve_enabled,
                            move |approval_request| {
                                let approval_event_tx = approval_event_tx.clone();
                                let request_id_for_approval = request_id_for_approval.clone();
                                let approval_waiters = approval_waiters.clone();
                                async move {
                                    let approval_id = Uuid::new_v4().to_string();
                                    let (tx, rx) = tokio::sync::oneshot::channel();
                                    let lock_ok = if let Ok(mut waiters) = approval_waiters.lock() {
                                        waiters.insert(approval_id.clone(), tx);
                                        true
                                    } else {
                                        false
                                    };
                                    if !lock_ok {
                                        let _ = approval_event_tx
                                            .send(AppEvent::SwarmWorkflowStep {
                                                request_id: request_id_for_approval.clone(),
                                                title: "Authorization callback failed".to_string(),
                                                details: "Approval lock acquisition failed; defaulting to deny."
                                                    .to_string(),
                                            })
                                            .await;
                                        return ApprovalDecision::Deny;
                                    }
                                    let _ = approval_event_tx
                                        .send(AppEvent::ExecutionApprovalRequested {
                                            request_id: request_id_for_approval.clone(),
                                            approval_id,
                                            request: approval_request,
                                        })
                                        .await;
                                    match rx.await {
                                        Ok(decision) => decision,
                                        Err(_) => {
                                            let _ = approval_event_tx
                                                .send(AppEvent::SwarmWorkflowStep {
                                                    request_id: request_id_for_approval,
                                                    title: "Authorization callback failed".to_string(),
                                                    details: "Approval channel closed; defaulting to deny."
                                                        .to_string(),
                                                })
                                                .await;
                                            ApprovalDecision::Deny
                                        }
                                    }
                                }
                            },
                        )
                        .await
                    {
                        Ok((status, next_auto_approve_state)) => {
                            auto_approve_enabled = next_auto_approve_state;
                            match status {
                                ExecutionStatus::Executed(report) => {
                                    let _ = event_tx
                                        .send(AppEvent::SwarmWorkflowStep {
                                            request_id: req_id.clone(),
                                            title: format!("{}: {}", role.as_str(), report.action),
                                            details: format!(
                                                "success={}\nexit_code={}\ntimed_out={}\nstdout:\n{}\nstderr:\n{}",
                                                report.success, report.exit_code, report.timed_out, report.stdout, report.stderr
                                            ),
                                        })
                                        .await;
                                    swarm_memory.push_str(&format!(
                                        "action={} success={} exit_code={} timed_out={}\nstdout:\n{}\nstderr:\n{}\n",
                                        report.action, report.success, report.exit_code, report.timed_out, report.stdout, report.stderr
                                    ));
                                    if report.timed_out {
                                        swarm_memory.push_str(
                                            "[synthesizer_timeout_hint]\nA command timed out. Analyze partial output, identify root cause, and propose an autonomous retry/fix strategy.\n",
                                        );
                                    }
                                }
                                ExecutionStatus::AuthorizationDenied { action, reason } => {
                                    let _ = event_tx
                                        .send(AppEvent::SwarmWorkflowStep {
                                            request_id: req_id.clone(),
                                            title: format!("{}: authorization denied", role.as_str()),
                                            details: format!("{reason}\naction={:?}", action.action),
                                        })
                                        .await;
                                    swarm_memory.push_str(&format!(
                                        "action authorization denied: {:?}\nreason: {}\n",
                                        action.action, reason
                                    ));
                                }
                                ExecutionStatus::AwaitingApproval(req) => {
                                    let _ = event_tx
                                        .send(AppEvent::SwarmWorkflowStep {
                                            request_id: req_id.clone(),
                                            title: format!("{}: awaiting approval", role.as_str()),
                                            details: req.reason,
                                        })
                                        .await;
                                }
                            }
                        }
                        Err(e) => {
                            let _ = event_tx
                                .send(AppEvent::SwarmWorkflowStep {
                                    request_id: req_id.clone(),
                                    title: format!("{}: execution error", role.as_str()),
                                    details: e.to_string(),
                                })
                                .await;
                            swarm_memory.push_str(&format!(
                                "\n[{} execution error]\n{}\n",
                                role.as_str(),
                                e
                            ));
                        }
                    }
                }
            }

            let unread_memory = swarm_memory.trim();
            if !unread_memory.is_empty() {
                let _ = event_tx
                    .send(AppEvent::SwarmStatus {
                        request_id: req_id.clone(),
                        status: "🧩 Synthesizer is composing final response...".to_string(),
                    })
                    .await;
                let _ = event_tx
                    .send(AppEvent::SwarmWorkflowStep {
                        request_id: req_id.clone(),
                        title: "Synthesizer: final response".to_string(),
                        details: "Generating final user-facing summary from swarm memory.".to_string(),
                    })
                    .await;

                let synth_system_prompt = format!(
                    "{CORE_OS_SYSTEM_PROMPT}\n\n{telemetry_context}\n\n{rag_context}\n\n{SWARM_SYNTH_ROLE_PROMPT}\n\nUser request:\n{query}\n\nSwarm memory:\n{swarm_memory}",
                    swarm_memory = swarm_memory
                );
                let mut synth_messages = api_messages.clone();
                synth_messages.insert(
                    0,
                    ChatMessage::with_cache_control("system", &synth_system_prompt),
                );
                match client
                    .chat_completion(
                        &model_id,
                        synth_messages,
                        reasoning_capability,
                        tiered_reasoning_level.as_ref(),
                        binary_reasoning_enabled,
                        |_stream_chunk| {},
                    )
                    .await
                {
                    Ok(synth_raw) => {
                        let synth_parsed = crate::parser::parse_response(&synth_raw);
                        let synth_message = synth_parsed
                            .message
                            .as_deref()
                            .filter(|m| !m.trim().is_empty())
                            .unwrap_or(synth_parsed.fallback_text.as_str())
                            .trim()
                            .to_string();
                        if !synth_message.is_empty() {
                            final_parts.push(format!("Synthesizer: {synth_message}"));
                        }
                    }
                    Err(e) => {
                        swarm_memory.push_str(&format!("\n[Synthesizer error]\n{}\n", e));
                    }
                }
            }

            let final_output = if final_parts.is_empty() {
                "Swarm completed. No additional assistant message.".to_string()
            } else {
                final_parts.join("\n\n")
            };

            let _ = event_tx
                .send(AppEvent::ChunkReceived(req_id.clone(), final_output))
                .await;
            let _ = event_tx
                .send(AppEvent::SwarmStatus {
                    request_id: req_id.clone(),
                    status: "✅ Swarm completed".to_string(),
                })
                .await;
            let _ = event_tx
                .send(AppEvent::SwarmWorkflowStep {
                    request_id: req_id.clone(),
                    title: "Workflow complete".to_string(),
                    details: "All planned swarm steps finished.".to_string(),
                })
                .await;
            let _ = event_tx.send(AppEvent::ResponseComplete(req_id)).await;
        });
    }

    fn start_autonomous_scheduler_task(&self) {
        let tx = self.event_tx.clone();
        self.tokio_rt.spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(
                AUTONOMOUS_SCHEDULER_TICK_INTERVAL_SECS,
            ));
            loop {
                interval.tick().await;
                let _ = tx.send(AppEvent::AutonomousScheduleTick).await;
            }
        });
    }

    fn hydrate_autonomous_schedules_from_settings(&mut self) {
        self.autonomous_schedules = self
            .settings
            .autonomous_schedules
            .iter()
            .map(|job| AutonomousSchedule {
                id: if job.id.trim().is_empty() {
                    Uuid::new_v4().to_string()
                } else {
                    job.id.clone()
                },
                time_24h: job.time_24h.clone(),
                prompt: job.prompt.clone(),
                enabled: job.enabled,
                last_run_date_utc: None,
            })
            .collect();
    }

    fn persist_autonomous_schedules_to_settings(&mut self) {
        self.settings.autonomous_schedules = self
            .autonomous_schedules
            .iter()
            .map(|job| crate::config::ScheduledJobConfig {
                id: job.id.clone(),
                time_24h: job.time_24h.clone(),
                prompt: job.prompt.clone(),
                enabled: job.enabled,
            })
            .collect();
        if let Err(e) = save_settings(&self.settings) {
            self.notify(
                format!("Failed to save autonomous schedules: {e}"),
                NotificationKind::Error,
            );
        }
    }

    fn parse_time_hhmm(value: &str) -> Option<(u32, u32)> {
        let trimmed = value.trim();
        let (h, m) = trimmed.split_once(':')?;
        let hour: u32 = h.parse().ok()?;
        let minute: u32 = m.parse().ok()?;
        if hour < 24 && minute < 60 {
            Some((hour, minute))
        } else {
            None
        }
    }

    fn try_extract_schedule_from_user_message(&self, text: &str) -> Option<(String, String)> {
        let lower = text.to_ascii_lowercase();
        if !AUTONOMOUS_TRIGGER_KEYWORDS.iter().any(|k| lower.contains(k)) {
            return None;
        }
        let bytes = text.as_bytes();
        for i in 0..bytes.len().saturating_sub(4) {
            if bytes[i].is_ascii_digit()
                && bytes[i + 1].is_ascii_digit()
                && bytes[i + 2] == b':'
                && bytes[i + 3].is_ascii_digit()
                && bytes[i + 4].is_ascii_digit()
            {
                let t = &text[i..i + 5];
                if Self::parse_time_hhmm(t).is_some() {
                    let prompt = text.trim().to_string();
                    return Some((t.to_string(), prompt));
                }
            }
        }
        None
    }

    fn add_autonomous_schedule(&mut self, time_24h: String, prompt: String) {
        if Self::parse_time_hhmm(&time_24h).is_none() {
            self.notify(
                "Invalid schedule time. Use HH:MM (24h).",
                NotificationKind::Error,
            );
            return;
        }
        if prompt.trim().is_empty() {
            self.notify("Schedule prompt cannot be empty.", NotificationKind::Error);
            return;
        }
        self.autonomous_schedules.push(AutonomousSchedule {
            id: Uuid::new_v4().to_string(),
            time_24h,
            prompt,
            enabled: true,
            last_run_date_utc: None,
        });
        self.persist_autonomous_schedules_to_settings();
        self.notify("Autonomous schedule added.", NotificationKind::Info);
    }

    fn run_autonomous_scheduler_tick(&mut self) {
        if self.autonomous_schedules.is_empty() {
            return;
        }
        let now = Utc::now();
        let today = now.format("%Y-%m-%d").to_string();
        let mut to_run: Vec<String> = Vec::new();
        for job in &mut self.autonomous_schedules {
            if !job.enabled {
                continue;
            }
            let Some((hour, minute)) = Self::parse_time_hhmm(&job.time_24h) else {
                continue;
            };
            if now.hour() == hour
                && now.minute() == minute
                && job.last_run_date_utc.as_deref() != Some(&today)
            {
                job.last_run_date_utc = Some(today.clone());
                to_run.push(job.prompt.clone());
            }
        }
        if to_run.is_empty() {
            return;
        }
        if self.current_session_idx.is_none() {
            self.new_session();
        }
        let Some(session_idx) = self.current_session_idx else {
            return;
        };
        for prompt in to_run {
            let user_msg = Message {
                id: Uuid::new_v4().to_string(),
                role: Role::User,
                content: format!("{AUTONOMOUS_WORKFLOW_PREFIX}\n{prompt}"),
                attachments: vec![],
                timestamp: Utc::now(),
                is_streaming: false,
            };
            if let Some(session) = self.sessions.get_mut(session_idx) {
                if session.messages.is_empty() {
                    session.name = AUTONOMOUS_SESSION_NAME.to_string();
                }
                let session_id = session.id.clone();
                session.messages.push(user_msg.clone());
                self.save_message(&session_id, &user_msg);
            }
            self.append_system_message(
                Some(session_idx),
                "🕒 Autonomous Cron-Swarm triggered scheduled workflow.".to_string(),
            );
        }
        self.persist_autonomous_schedules_to_settings();
        self.persist_session_snapshot_async(session_idx);
        self.dispatch_agent_requests(session_idx);
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
        self.activity_log
            .push(format!("Deleted conversation: {}", session_name));
        self.notify("Conversation deleted", NotificationKind::Info);
    }

    fn new_session(&mut self) {
        let session = Session::new("New Chat");
        self.spawn_create_session(session.clone());

        self.sessions.insert(0, session);
        self.current_session_idx = Some(0);
        self.activity_log
            .push("Created new conversation".to_string());
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
                self.notify(
                    format!("Could not snapshot file before write: {e}"),
                    NotificationKind::Error,
                );
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
                    "fn" | "let"
                        | "mut"
                        | "pub"
                        | "impl"
                        | "struct"
                        | "enum"
                        | "match"
                        | "if"
                        | "else"
                        | "for"
                        | "while"
                        | "use"
                        | "mod"
                        | "return"
                        | "const"
                        | "static"
                        | "trait"
                        | "type"
                        | "async"
                        | "await"
                        | "unsafe"
                        | "where"
                ),
                "py" => matches!(
                    word_buffer.as_str(),
                    "def"
                        | "class"
                        | "if"
                        | "elif"
                        | "else"
                        | "for"
                        | "while"
                        | "import"
                        | "from"
                        | "return"
                        | "with"
                        | "as"
                        | "try"
                        | "except"
                        | "lambda"
                        | "async"
                        | "await"
                        | "yield"
                        | "finally"
                        | "raise"
                        | "assert"
                        | "pass"
                        | "break"
                        | "continue"
                        | "global"
                        | "nonlocal"
                        | "and"
                        | "or"
                        | "not"
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
            Err(e) => self.notify(
                format!("Save failed for {}: {}", path, e),
                NotificationKind::Error,
            ),
        }
    }

    fn active_editor_language_label(&self) -> String {
        let Some(path) = self.active_open_file.as_ref() else {
            return "Language: txt".to_string();
        };
        let ext = std::path::Path::new(path)
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("txt")
            .to_lowercase();
        format!("Language: {}", ext)
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

        if let Some((time_24h, prompt)) = self.try_extract_schedule_from_user_message(&text) {
            self.add_autonomous_schedule(time_24h, prompt);
        }

        self.dispatch_agent_requests(session_idx);
    }

    fn process_events(&mut self) {
        while let Ok(event) = self.event_rx.try_recv() {
            match event {
                AppEvent::ChunkReceived(req_id, chunk) => {
                    if let Some(active) = self.active_requests.get(&req_id).cloned() {
                        if let Some(session) = self.sessions.get_mut(active.session_idx) {
                            if let Some(msg) = session
                                .messages
                                .iter_mut()
                                .find(|m| m.id == active.message_id)
                            {
                                msg.content.push_str(&chunk);
                            }
                        }
                    }
                }
                AppEvent::ResponseComplete(req_id) => {
                    self.finalize_workflow(&req_id, true);
                    if let Some(active) = self.active_requests.remove(&req_id) {
                        let mut to_save: Option<(String, Message)> = None;
                        let mut parse_error_notification: Option<String> = None;
                        if let Some(session) = self.sessions.get_mut(active.session_idx) {
                            let session_id = session.id.clone();
                            if let Some(msg) = session
                                .messages
                                .iter_mut()
                                .find(|m| m.id == active.message_id)
                            {
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
                                    let parsed_actions =
                                        Self::pending_actions_from_agent_actions(parsed.actions);
                                    if !parsed_actions.is_empty() {
                                        let mut iter = parsed_actions.into_iter();
                                        if let Some(action) = iter.next() {
                                            self.pending_action = Some(action);
                                            self.pending_action_session_idx =
                                                Some(active.session_idx);
                                            self.pending_actions_queue = iter.collect();
                                        }
                                    } else if let Some(action) =
                                        Self::parse_pending_action(&msg.content)
                                    {
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
                        self.activity_log
                            .push(format!("{} finished response", active.agent_name));
                        self.swarm_status = "✅ Ready".to_string();
                    }
                }
                AppEvent::ResponseError(req_id, err) => {
                    self.finalize_workflow(&req_id, false);
                    if let Some(active) = self.active_requests.remove(&req_id) {
                        if let Some(session) = self.sessions.get_mut(active.session_idx) {
                            if let Some(msg) = session
                                .messages
                                .iter_mut()
                                .find(|m| m.id == active.message_id)
                            {
                                msg.content = format!("❌ Error: {}", err);
                                msg.is_streaming = false;
                            }
                        }
                        self.notify(
                            format!("{} failed: {}", active.agent_name, err),
                            NotificationKind::Error,
                        );
                        self.swarm_status = "⚠️ Swarm failed".to_string();
                    }
                }
                AppEvent::SwarmStatus { request_id, status } => {
                    if self.active_requests.contains_key(&request_id) {
                        self.swarm_status = status;
                    }
                }
                AppEvent::SwarmWorkflowStep {
                    request_id,
                    title,
                    details,
                } => {
                    if self.active_requests.contains_key(&request_id) {
                        self.begin_workflow_step(&request_id, title, details);
                    }
                }
                AppEvent::ExecutionApprovalRequested {
                    request_id,
                    approval_id,
                    request,
                } => {
                    if self.active_requests.contains_key(&request_id) {
                        self.swarm_status = "⏸ Waiting for manual approval".to_string();
                        self.pending_execution_approval = Some(PendingExecutionApproval {
                            request_id,
                            approval_id,
                            request,
                        });
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
                    streamed,
                } => {
                    let mut continue_session: Option<usize> = None;
                    let mut system_result: Option<(usize, String)> = None;
                    let sanitized_stdout = Self::sanitized_command_stdout(&stdout, exit_code);
                    if let Some(term) = self.terminals.iter_mut().find(|t| t.id == terminal_id) {
                        term.running = false;
                        term.exit_code = Some(exit_code);
                        if term.output.ends_with(TERMINAL_RUNNING_MARKER) {
                            let len = term.output.len() - TERMINAL_RUNNING_MARKER.len();
                            term.output.truncate(len);
                        }
                        if !streamed && !stdout.trim().is_empty() {
                            term.output.push_str(&stdout);
                            if !stdout.ends_with('\n') {
                                term.output.push('\n');
                            }
                        }
                        if !streamed && !stderr.trim().is_empty() {
                            term.output.push_str("\n[stderr]\n");
                            term.output.push_str(&stderr);
                            if !stderr.ends_with('\n') {
                                term.output.push('\n');
                            }
                        }
                        term.output
                            .push_str(&format!("\n[exit code: {}]\n", exit_code));
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
                                if exit_code == TIMEOUT_EXIT_CODE
                                    || stderr.to_ascii_lowercase().contains("[timeout]")
                                {
                                    result.push_str(
                                        "\n\n[Synthesizer hint]\nThe command timed out. Analyze partial output, identify the blocker, and propose the next safe autonomous step.",
                                    );
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
                AppEvent::TerminalChunk {
                    terminal_id,
                    line,
                    is_stdout,
                } => {
                    if let Some(term) = self.terminals.iter_mut().find(|t| t.id == terminal_id) {
                        if is_stdout {
                            term.output.push_str(&line);
                            term.output.push('\n');
                        } else {
                            term.output.push_str("[stderr] ");
                            term.output.push_str(&line);
                            term.output.push('\n');
                        }
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
                AppEvent::SessionMessagesLoaded {
                    idx,
                    messages,
                    error,
                } => {
                    if let Some(err) = error {
                        self.notify(
                            format!("Failed to load messages: {err}"),
                            NotificationKind::Error,
                        );
                    } else if let Some(session) = self.sessions.get_mut(idx) {
                        session.messages = messages;
                    }
                }
                AppEvent::StoredSessionLoaded { session, error } => {
                    if let Some(err) = error {
                        self.activity_log
                            .push(format!("Failed to load stored session: {err}"));
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
                        self.notify(
                            format!("Failed to load snapshots: {err}"),
                            NotificationKind::Error,
                        );
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
                AppEvent::WorkspaceTreeLoaded { root, error } => {
                    self.workspace_tree = Some(root);
                    if let Some(err) = error {
                        self.notify(
                            format!("Workspace refresh failed: {err}"),
                            NotificationKind::Error,
                        );
                    }
                }
                AppEvent::WorkspaceFileLoaded { path, content, error } => {
                    if let Some(err) = error {
                        self.notify(
                            format!("Open failed for {}: {}", path, err),
                            NotificationKind::Error,
                        );
                        continue;
                    }
                    if !self.opened_files.iter().any(|p| p == &path) {
                        self.opened_files.push(path.clone());
                    }
                    self.editor_buffers.insert(path.clone(), content);
                    self.editor_dirty.insert(path.clone(), false);
                    self.active_open_file = Some(path);
                }
                AppEvent::AutonomousScheduleTick => {
                    self.run_autonomous_scheduler_tick();
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

    fn push_attachment_with_metadata(content: &mut String, att: &Attachment) {
        content.push_str(&format!(
            "\n\n[AttachmentMeta]\nname={}\nmime={}\nbytes={}\n",
            att.filename,
            att.mime_type,
            att.raw_bytes.len()
        ));
        if let Some(txt) = &att.text {
            content.push_str(txt);
        } else if att.image_base64.is_some() {
            content.push_str("[binary image attached]");
        } else {
            content.push_str("[binary attachment]");
        }
    }

    fn download_image(&mut self, att: &Attachment) {
        if let Some(path) = rfd::FileDialog::new()
            .set_file_name(&att.filename)
            .save_file()
        {
            if let Err(e) = std::fs::write(&path, &att.raw_bytes) {
                self.notify(
                    format!("Failed to save image: {e}"),
                    NotificationKind::Error,
                );
            } else {
                self.notify("Image saved", NotificationKind::Info);
            }
        }
    }

    fn select_working_dir(&mut self) {
        if let Some(path) = rfd::FileDialog::new().pick_folder() {
            self.settings.working_directory = path.to_string_lossy().to_string();
            if let Err(e) = save_settings(&self.settings) {
                self.notify(
                    format!("Failed to save settings: {e}"),
                    NotificationKind::Error,
                );
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
            .frame(
                egui::Frame::new()
                    .fill(BURGUNDY_DARK)
                    .inner_margin(egui::Margin::same(8)),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Editor").color(TEXT_PRIMARY).strong());
                    if ui.small_button("📂 Open File").clicked() {
                        if let Some(path) = rfd::FileDialog::new().pick_file() {
                            self.spawn_workspace_file_open(path.to_string_lossy().to_string());
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
                                    self.spawn_workspace_file_open(path.clone());
                                }
                                ui.label(RichText::new(path.clone()).small().color(LIGHT_TEXT));
                            });
                        }
                        if self.changed_files.is_empty() {
                            ui.label(
                                RichText::new("No changed files yet")
                                    .small()
                                    .color(TEXT_MUTED),
                            );
                        }
                    });
                });
                ui.separator();
                let panel_resp = ui.collapsing("Workspace Explorer", |ui| {
                    let max_h = (ui.available_height() * 0.30).max(120.0);
                    ScrollArea::vertical().max_height(max_h).show(ui, |ui| {
                        let mut request_refresh = false;
                        if let Some(root) = &self.workspace_tree {
                            let children = root.children.clone();
                            for child in &children {
                                self.render_workspace_tree_node(ui, child, &mut request_refresh);
                            }
                        } else {
                            ui.label(RichText::new("Loading workspace…").small().color(TEXT_MUTED));
                        }
                        if request_refresh {
                            self.request_workspace_refresh();
                        }
                    });
                });
                panel_resp.header_response.context_menu(|ui| {
                    if ui.button("Refresh").clicked() {
                        self.request_workspace_refresh();
                        ui.close_menu();
                    }
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
                        let button =
                            egui::Button::new(RichText::new(label).small().color(if selected {
                                DARK_TEXT
                            } else {
                                LIGHT_TEXT
                            }))
                            .sense(egui::Sense::click())
                            .fill(if selected {
                                GOLD
                            } else {
                                SKY_BLUE_DARK
                            });
                        let response = ui.add(button).on_hover_text(if is_dirty {
                            "Unsaved changes"
                        } else {
                            "Saved"
                        });
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
                    let language_label = self.active_editor_language_label();
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(path.clone()).small().color(TEXT_MUTED));
                        if ui.button("💾 Save").clicked() {
                            self.save_active_editor_file();
                        }
                    });
                    if let Some(buffer) = self.editor_buffers.get_mut(&path) {
                        ui.label(
                            RichText::new(language_label)
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
                        if ui
                            .button("Terminate")
                            .on_hover_text("Stop the running process")
                            .clicked()
                        {
                            self.terminate_terminal(&active_id);
                        }
                        if ui
                            .button("Close")
                            .on_hover_text("Close this terminal tab")
                            .clicked()
                        {
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
                            if term.owner == TerminalOwner::AI {
                                "🤖"
                            } else {
                                "👤"
                            },
                            term.name
                        );
                        let button =
                            egui::Button::new(RichText::new(label).small().color(if selected {
                                DARK_TEXT
                            } else {
                                LIGHT_TEXT
                            }))
                            .fill(if selected {
                                GOLD
                            } else {
                                SKY_BLUE_DARK
                            });
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
                                .desired_width(
                                    (ui.available_width() - TERMINAL_INPUT_RESERVED_WIDTH)
                                        .max(120.0),
                                )
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
                    .inner_margin(egui::Margin {
                        left: 12,
                        right: 12,
                        top: 10,
                        bottom: 10,
                    })
                    .show(ui, |ui| {
                        ui.with_layout(egui::Layout::left_to_right(egui::Align::TOP), |ui| {
                            ui.set_max_width(max_bubble - 24.0);

                            let role_label = if is_user { "You" } else { "Assistant" };
                            ui.horizontal(|ui| {
                                ui.label(
                                    RichText::new(role_label)
                                        .small()
                                        .color(if is_user { BURGUNDY } else { SKY_BLUE })
                                        .strong(),
                                );
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        if ui.small_button("Copy").clicked() {
                                            self.copy_to_clipboard(&msg.content);
                                        }
                                    },
                                );
                            });
                            ui.add_space(4.0);

                            if msg.is_streaming && msg.content.is_empty() {
                                ui.spinner();
                            } else {
                                let mut style = (*ui.ctx().style()).clone();
                                style.visuals.override_text_color = Some(text_color);
                                ui.scope(|ui| {
                                    ui.set_style(style);
                                    CommonMarkViewer::new().show(
                                        ui,
                                        &mut self.markdown_cache,
                                        &msg.content,
                                    );
                                });
                            }

                            for att in &msg.attachments {
                                ui.separator();
                                if att.image_base64.is_some() {
                                    if ui
                                        .button(
                                            RichText::new(format!(
                                                "🖼 {} (click to download)",
                                                att.filename
                                            ))
                                            .color(SKY_BLUE)
                                            .small(),
                                        )
                                        .clicked()
                                    {
                                        self.download_image(att);
                                    }
                                } else {
                                    ui.label(
                                        RichText::new(format!("📄 {}", att.filename))
                                            .color(SKY_BLUE)
                                            .small(),
                                    );
                                }
                            }

                            ui.add_space(6.0);
                            ui.horizontal(|ui| {
                                ui.add_space(2.0);
                                ui.label(
                                    RichText::new(msg.timestamp.format("%H:%M").to_string())
                                        .small()
                                        .color(TEXT_MUTED),
                                );
                            });
                        });
                    });
            },
        );
    }
}

impl eframe::App for ChatApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.process_events();
        self.process_workspace_refresh_signals();
        self.ingest_dropped_files(ctx);
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
                    self.notify(
                        format!("Failed to save settings: {e}"),
                        NotificationKind::Error,
                    );
                } else {
                    self.notify("Setup complete", NotificationKind::Info);
                }
            }
        }

        if let Some(pending) = self.pending_execution_approval.clone() {
            let mut approve_once = false;
            let mut grant_temporary = false;
            let mut deny = false;
            egui::Window::new("⚠ Action Approval Required")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, Vec2::ZERO)
                .show(ctx, |ui| {
                    ui.visuals_mut().override_text_color = Some(LIGHT_TEXT);
                    ui.label(RichText::new("Manual approval required").strong());
                    ui.label(format!("Request: {}", pending.request_id));
                    ui.separator();
                    match &pending.request.action.action {
                        ActionKind::RunCmd => {
                            ui.label("Execute command:");
                            ui.monospace(
                                pending
                                    .request
                                    .action
                                    .parameters
                                    .command
                                    .as_deref()
                                    .unwrap_or(""),
                            );
                        }
                        ActionKind::EditFile => {
                            let path = pending
                                .request
                                .action
                                .parameters
                                .path
                                .as_deref()
                                .unwrap_or("");
                            let mode = pending
                                .request
                                .action
                                .parameters
                                .mode
                                .as_deref()
                                .unwrap_or("overwrite");
                            let content = pending
                                .request
                                .action
                                .parameters
                                .content
                                .as_deref()
                                .unwrap_or("");
                            ui.label(format!("Edit file ({mode}): {path}"));
                            ui.separator();
                            ScrollArea::vertical().max_height(200.0).show(ui, |ui| {
                                ui.monospace(content);
                            });
                        }
                        ActionKind::GenerateDocument => {
                            let format = pending
                                .request
                                .action
                                .parameters
                                .format
                                .as_deref()
                                .unwrap_or("unknown");
                            let path = pending
                                .request
                                .action
                                .parameters
                                .path
                                .as_deref()
                                .unwrap_or("");
                            let markdown = pending
                                .request
                                .action
                                .parameters
                                .markdown_content
                                .as_deref()
                                .unwrap_or("");
                            ui.label(format!("Generate document ({format}): {path}"));
                            ui.separator();
                            ScrollArea::vertical().max_height(200.0).show(ui, |ui| {
                                ui.monospace(markdown);
                            });
                        }
                        _ => {
                            ui.label(format!("Action: {:?}", pending.request.action.action));
                        }
                    }
                    ui.separator();
                    ui.label(pending.request.reason.clone());
                    ui.separator();
                    ui.horizontal(|ui| {
                        if ui
                            .button(RichText::new("Approve Once").color(LIGHT_TEXT))
                            .clicked()
                        {
                            approve_once = true;
                        }
                        if ui
                            .button(RichText::new("Grant Temporary Access").color(LIGHT_TEXT))
                            .clicked()
                        {
                            grant_temporary = true;
                        }
                        if ui
                            .button(RichText::new("Deny").color(LIGHT_TEXT))
                            .clicked()
                        {
                            deny = true;
                        }
                    });
                });
            if approve_once {
                self.resolve_pending_execution_approval(
                    &pending.approval_id,
                    ApprovalDecision::ApproveOnce,
                );
                self.swarm_status = "▶ Approval granted once".to_string();
            } else if grant_temporary {
                self.resolve_pending_execution_approval(
                    &pending.approval_id,
                    ApprovalDecision::GrantTemporaryAccess,
                );
                self.swarm_status = "▶ Temporary access granted".to_string();
            } else if deny {
                self.resolve_pending_execution_approval(&pending.approval_id, ApprovalDecision::Deny);
                self.swarm_status = "⛔ Authorization denied".to_string();
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
                        if ui
                            .button(RichText::new("✅ Approve").color(LIGHT_TEXT))
                            .clicked()
                        {
                            approved = true;
                        }
                        if ui
                            .button(RichText::new("❌ Reject").color(LIGHT_TEXT))
                            .clicked()
                        {
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
                                format!(
                                    "❌ File write failed for `{}`: {}",
                                    full_path.display(),
                                    e
                                ),
                            );
                            if let Some(idx) = action_session_idx {
                                self.dispatch_agent_requests(idx);
                            }
                        } else {
                            self.notify(
                                format!("File written: {}", full_path.display()),
                                NotificationKind::Info,
                            );
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
                                "❌ Command not executed: shell execution is disabled in settings."
                                    .to_string(),
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
                            self.notify(
                                "AI command started in dedicated terminal",
                                NotificationKind::Info,
                            );
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
                                self.notify(
                                    format!("Create folder failed: {e}"),
                                    NotificationKind::Error,
                                );
                                self.append_system_message(
                                    action_session_idx,
                                    format!(
                                        "❌ Folder create failed for `{}`: {}",
                                        full_path.display(),
                                        e
                                    ),
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
                                self.notify(
                                    format!("File edited: {}", full_path.display()),
                                    NotificationKind::Info,
                                );
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
                                    format!(
                                        "❌ File edit failed for `{}`: {}",
                                        full_path.display(),
                                        e
                                    ),
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
                            let obj2 =
                                "2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n";
                            offsets.push(pdf.len());
                            pdf.push_str(obj2);
                            let obj3 =
                                "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << >> >>\nendobj\n";
                            offsets.push(pdf.len());
                            pdf.push_str(obj3);
                            let stream_cmd =
                                format!("BT /F1 12 Tf 72 720 Td ({normalized}) Tj ET\n");
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
                                    self.notify(
                                        format!("Create PDF failed: {e}"),
                                        NotificationKind::Error,
                                    );
                                    self.append_system_message(
                                        action_session_idx,
                                        format!(
                                            "❌ PDF create failed for `{}`: {}",
                                            full_path.display(),
                                            e
                                        ),
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
                    "❌ Action rejected by user. No file write or command execution was performed."
                        .to_string(),
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
                                        ui.selectable_value(
                                            &mut selected_provider,
                                            p.clone(),
                                            p.display_name(),
                                        );
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
                                self.notify(
                                    format!("Failed to save settings: {e}"),
                                    NotificationKind::Error,
                                );
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
                .max_width(420.0)
                .frame(
                    egui::Frame::new()
                        .fill(BG_SURFACE)
                        .inner_margin(egui::Margin::same(8)),
                )
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
                            egui::Button::new(RichText::new("＋ New Chat").color(TEXT_DARK))
                                .fill(GOLD),
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
                                let display_name: String = if name.chars().count() > 32 {
                                    let shortened: String = name.chars().take(29).collect();
                                    format!("{shortened}...")
                                } else {
                                    name.clone()
                                };
                                ui.horizontal(|ui| {
                                    let btn = egui::Button::new(
                                        RichText::new(format!("💬 {display_name}"))
                                            .color(LIGHT_TEXT),
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
                                let mut picked_model_idx: Option<usize> = None;
                                egui::ComboBox::from_id_salt("model_selector")
                                    .selected_text(
                                        RichText::new(selected_model_name).color(TEXT_PRIMARY),
                                    )
                                    .width(ui.available_width())
                                    .show_ui(ui, |ui| {
                                        for (i, m) in self.models.iter().enumerate() {
                                            if ui
                                                .selectable_label(
                                                    self.selected_model_idx == i,
                                                    RichText::new(&m.name).color(TEXT_PRIMARY),
                                                )
                                                .clicked()
                                            {
                                                // Defer activation until after ComboBox rendering
                                                // to avoid borrow checker conflicts: egui holds an
                                                // immutable borrow of `self.models` during iteration,
                                                // but activation requires mutable access to `self`.
                                                picked_model_idx = Some(i);
                                            }
                                        }
                                        if !self.remote_models.is_empty() {
                                            ui.separator();
                                            ui.label(
                                                RichText::new("Remote model IDs:")
                                                    .color(TEXT_PRIMARY),
                                            );
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
                                if let Some(idx) = picked_model_idx {
                                    self.selected_model_idx = idx;
                                    self.custom_model_id.clear();
                                    self.activate_model_selection();
                                }
                            });
                        ui.label(RichText::new("Custom model ID").small().color(TEXT_MUTED));
                        ui.horizontal(|ui| {
                            let custom_id_response = ui.add(
                                TextEdit::singleline(&mut self.custom_model_id)
                                    .desired_width(
                                        ui.available_width() - CUSTOM_MODEL_INPUT_RESERVED_WIDTH,
                                    )
                                    .hint_text(
                                        "e.g. gpt-4o, meta-llama/Llama-3.1-8B-Instruct",
                                    ),
                            );
                            let apply_custom_model = (custom_id_response.changed()
                                && custom_id_response.lost_focus())
                                || (custom_id_response.has_focus()
                                    && ui.input(|i| i.key_pressed(egui::Key::Enter)));
                            if apply_custom_model {
                                self.activate_model_selection();
                            }
                            if ui.button("From list").clicked() {
                                self.custom_model_id.clear();
                                self.activate_model_selection();
                            }
                        });
                        if ui
                            .checkbox(
                                &mut self.force_reasoning_controls,
                                "Force Enable Reasoning Controls",
                            )
                            .changed()
                        {
                            self.ensure_thinking_mode_valid_for_model();
                        }
                    });

                    ui.horizontal(|ui| {
                        ui.label("Execution policy:");
                        egui::ComboBox::from_id_salt("execution_policy")
                            .selected_text(match self.execution_policy {
                                ExecutionPolicy::Manual => "Manual",
                                ExecutionPolicy::ReadEdit => "ReadEdit",
                                ExecutionPolicy::Execute => "Execute",
                                ExecutionPolicy::FullAccess => "FullAccess",
                            })
                            .show_ui(ui, |ui| {
                                ui.selectable_value(
                                    &mut self.execution_policy,
                                    ExecutionPolicy::Manual,
                                    "Manual",
                                );
                                ui.selectable_value(
                                    &mut self.execution_policy,
                                    ExecutionPolicy::ReadEdit,
                                    "ReadEdit",
                                );
                                ui.selectable_value(
                                    &mut self.execution_policy,
                                    ExecutionPolicy::Execute,
                                    "Execute",
                                );
                                ui.selectable_value(
                                    &mut self.execution_policy,
                                    ExecutionPolicy::FullAccess,
                                    "FullAccess",
                                );
                            });
                    });

                    let reasoning_cfg = self.active_reasoning_config();
                    match reasoning_cfg.capability {
                        ReasoningCapability::None => {}
                        ReasoningCapability::Binary => {
                            ui.checkbox(
                                &mut self.binary_reasoning_enabled,
                                reasoning_cfg.binary_label,
                            );
                        }
                        ReasoningCapability::Tiered => {
                            ui.horizontal(|ui| {
                                ui.label(reasoning_cfg.tiered_label);
                                egui::ComboBox::from_id_salt("thinking_mode")
                                    .selected_text(self.selected_thinking_mode.display_name())
                                    .show_ui(ui, |ui| {
                                        for mode in reasoning_cfg.tiered_modes {
                                            ui.selectable_value(
                                                &mut self.selected_thinking_mode,
                                                *mode,
                                                mode.display_name(),
                                            );
                                        }
                                    });
                            });
                        }
                    }

                    ui.separator();
                    ui.collapsing("Settings", |ui| {
                        for agent in &mut self.agents {
                            ui.checkbox(&mut agent.enabled, &agent.name);
                            ui.label(
                                RichText::new(&agent.system_prompt)
                                    .small()
                                    .color(TEXT_MUTED),
                            );
                        }
                        ui.separator();
                        ui.checkbox(
                            &mut self.settings.shell_execution_enabled,
                            "Shell execution",
                        );
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
                    ui.separator();
                    ui.collapsing("Autonomous Cron-Swarm", |ui| {
                        ui.label(
                            RichText::new("Schedule autonomous workflows (daily HH:MM UTC)")
                                .small()
                                .color(TEXT_MUTED),
                        );
                        ui.horizontal(|ui| {
                            ui.label("Time:");
                            ui.add(TextEdit::singleline(&mut self.autonomous_time_input).desired_width(70.0));
                        });
                        ui.add(
                            TextEdit::multiline(&mut self.autonomous_prompt_input)
                                .desired_rows(3)
                                .desired_width(ui.available_width())
                                .hint_text("e.g. Every morning: research AI news, analyze latest repo changes, generate PDF briefing."),
                        );
                        if ui.button("➕ Add Schedule").clicked() {
                            let t = self.autonomous_time_input.trim().to_string();
                            let p = self.autonomous_prompt_input.trim().to_string();
                            self.add_autonomous_schedule(t, p);
                            self.autonomous_prompt_input.clear();
                        }
                        ui.separator();
                        let mut remove_id: Option<String> = None;
                        for job in &mut self.autonomous_schedules {
                            ui.group(|ui| {
                                ui.horizontal(|ui| {
                                    ui.checkbox(&mut job.enabled, "");
                                    ui.label(
                                        RichText::new(format!("{} — {}", job.time_24h, job.prompt))
                                            .small()
                                            .color(TEXT_PRIMARY),
                                    );
                                    if ui.small_button("🗑").clicked() {
                                        remove_id = Some(job.id.clone());
                                    }
                                });
                                if let Some(last) = &job.last_run_date_utc {
                                    ui.label(
                                        RichText::new(format!("Last run (UTC date): {last}"))
                                            .small()
                                            .color(TEXT_MUTED),
                                    );
                                }
                            });
                        }
                        if let Some(id) = remove_id {
                            self.autonomous_schedules.retain(|j| j.id != id);
                            self.persist_autonomous_schedules_to_settings();
                        }
                        if ui.button("💾 Save Schedule Changes").clicked() {
                            self.persist_autonomous_schedules_to_settings();
                            self.notify("Schedules saved.", NotificationKind::Info);
                        }
                    });
                });
        }

        self.render_file_workspace_panel(ctx);

        if let Some(reason) = self.policy_block_dialog.clone() {
            let mut close_dialog = false;
            egui::Window::new("🛡️ Policy Blocked Action")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, Vec2::ZERO)
                .show(ctx, |ui| {
                    ui.visuals_mut().override_text_color = Some(LIGHT_TEXT);
                    ui.label(RichText::new("Action blocked by execution policy").strong());
                    ui.separator();
                    ui.label(reason);
                    ui.separator();
                    if ui.button("OK").clicked() {
                        close_dialog = true;
                    }
                });
            if close_dialog {
                self.policy_block_dialog = None;
            }
        }

        if self.show_activity_panel {
            egui::SidePanel::right("activity")
                .resizable(true)
                .default_width(220.0)
                .min_width(180.0)
                .max_width(420.0)
                .frame(
                    egui::Frame::new()
                        .fill(BURGUNDY_DARK)
                        .inner_margin(egui::Margin::same(8)),
                )
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
            .frame(
                egui::Frame::new()
                    .fill(BURGUNDY)
                    .inner_margin(egui::Margin::same(0)),
            )
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
                                ui.label(RichText::new(&self.swarm_status).color(GOLD).italics());
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
                                            RichText::new(
                                                "Select or create a conversation to begin",
                                            )
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
                                        egui::Button::new(
                                            RichText::new("📤 Send").color(DARK_TEXT),
                                        )
                                        .fill(GOLD),
                                    )
                                    .clicked()
                                {
                                    self.send_message();
                                }
                                if ui
                                    .add_sized(
                                        [110.0, 28.0],
                                        egui::Button::new(
                                            RichText::new("📎 Attach").color(LIGHT_TEXT),
                                        )
                                        .fill(SKY_BLUE_DARK),
                                    )
                                    .clicked()
                                {
                                    self.attach_file();
                                }
                                if ui
                                    .add_sized(
                                        [110.0, 26.0],
                                        egui::Button::new(
                                            RichText::new("Paste Clip").color(LIGHT_TEXT),
                                        )
                                        .fill(BG_SURFACE_ALT),
                                    )
                                    .clicked()
                                {
                                    self.attach_from_clipboard();
                                }
                            });
                        });
                    });
            });

        egui::Window::new("Swarm Workflow")
            .title_bar(true)
            .resizable(true)
            .default_size(Vec2::new(360.0, 220.0))
            .default_pos(egui::pos2(290.0, 96.0))
            .frame(
                egui::Frame::new()
                    .fill(Color32::from_rgba_unmultiplied(0x16, 0x18, 0x1E, 0xCC))
                    .corner_radius(10.0)
                    .stroke(egui::Stroke::new(1.0, BG_SURFACE_ALT))
                    .inner_margin(egui::Margin::same(10)),
            )
            .show(ctx, |ui| {
                self.render_workflow_visualizer(ui);
            });

        self.render_snapshots_window(ctx);
        self.draw_notifications(ctx);
    }
}
