use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use chrono::Utc;
use egui::{Color32, FontId, RichText, ScrollArea, TextEdit, Vec2};
use egui_commonmark::{CommonMarkCache, CommonMarkViewer};
use uuid::Uuid;

use crate::api::{builtin_models, provider_models, ChatMessage, ModelInfo, ThinkingMode};
use crate::config::{load_settings, save_settings, ApiProvider, Settings, DEFAULT_MODEL_ID};
use crate::db::{Database, DbFileSnapshot, DbMessage, DbSession};
use crate::setup::SetupWizard;

const BURGUNDY: Color32 = Color32::from_rgb(26, 35, 50);
const BURGUNDY_LIGHT: Color32 = Color32::from_rgb(45, 58, 82);
const BURGUNDY_DARK: Color32 = Color32::from_rgb(17, 24, 37);
const SKY_BLUE: Color32 = Color32::from_rgb(176, 226, 255);
const SKY_BLUE_DARK: Color32 = Color32::from_rgb(86, 154, 214);
const GOLD: Color32 = Color32::from_rgb(255, 208, 92);
const GOLD_DARK: Color32 = Color32::from_rgb(214, 163, 50);
const WHITE: Color32 = Color32::WHITE;
const DARK_TEXT: Color32 = Color32::from_rgb(20, 10, 12);
const CUSTOM_MODEL_INPUT_RESERVED_WIDTH: f32 = 132.0;

#[derive(Debug)]
pub enum AppEvent {
    ChunkReceived(String, String),
    ResponseComplete(String),
    ResponseError(String, String),
    ModelsLoaded(Vec<String>),
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

#[derive(Debug, Clone)]
struct ActiveRequest {
    session_idx: usize,
    message_id: String,
    agent_name: String,
}

pub struct ChatApp {
    settings: Settings,
    show_settings: bool,
    setup_wizard: Option<SetupWizard>,

    sessions: Vec<Session>,
    current_session_idx: Option<usize>,

    models: Vec<ModelInfo>,
    selected_model_idx: usize,
    remote_models: Vec<String>,
    custom_model_id: String,
    selected_thinking_mode: ThinkingMode,

    input_text: String,
    pending_attachments: Vec<Attachment>,

    event_tx: std::sync::mpsc::Sender<AppEvent>,
    event_rx: std::sync::mpsc::Receiver<AppEvent>,
    tokio_rt: Arc<tokio::runtime::Runtime>,
    active_requests: HashMap<String, ActiveRequest>,

    db: Arc<Mutex<Option<Database>>>,
    pending_action: Option<PendingAction>,
    pending_action_session_idx: Option<usize>,

    markdown_cache: CommonMarkCache,
    notifications: Vec<Notification>,
    activity_log: Vec<String>,

    show_snapshots: bool,
    snapshots: Vec<DbFileSnapshot>,

    agents: Vec<AgentProfile>,
}

impl ChatApp {
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
    }

    pub fn new(cc: &eframe::CreationContext, rt: Arc<tokio::runtime::Runtime>) -> Self {
        let (event_tx, event_rx) = std::sync::mpsc::channel();
        let settings = load_settings();
        Self::apply_theme(&cc.egui_ctx, settings.dark_mode);

        let db = match Database::new(&settings.db_path) {
            Ok(d) => Some(d),
            Err(e) => {
                eprintln!("DB error: {e}");
                None
            }
        };
        let db = Arc::new(Mutex::new(db));

        let sessions = if let Ok(guard) = db.lock() {
            if let Some(database) = guard.as_ref() {
                database
                    .list_sessions()
                    .unwrap_or_default()
                    .into_iter()
                    .map(|s| Session {
                        id: s.id,
                        name: s.name,
                        messages: vec![],
                    })
                    .collect()
            } else {
                vec![]
            }
        } else {
            vec![]
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
            sessions,
            current_session_idx: None,
            models,
            selected_model_idx: 0,
            remote_models: vec![],
            custom_model_id,
            selected_thinking_mode: ThinkingMode::Auto,
            input_text: String::new(),
            pending_attachments: vec![],
            event_tx,
            event_rx,
            tokio_rt: rt,
            active_requests: HashMap::new(),
            db,
            pending_action: None,
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
        };

        if let Some(idx) = app
            .models
            .iter()
            .position(|m| m.id == app.settings.default_model)
        {
            app.selected_model_idx = idx;
        }

        app.load_models_from_provider();
        app
    }

    fn apply_theme(ctx: &egui::Context, dark_mode: bool) {
        if dark_mode {
            let mut style = (*ctx.style()).clone();
            style.visuals = egui::Visuals::dark();
            style.visuals.override_text_color = Some(WHITE);
            style.visuals.panel_fill = BURGUNDY;
            style.visuals.window_fill = BURGUNDY_DARK;
            style.visuals.extreme_bg_color = BURGUNDY_DARK;
            style.visuals.widgets.noninteractive.bg_fill = BURGUNDY;
            style.visuals.widgets.inactive.bg_fill = BURGUNDY_LIGHT;
            style.visuals.widgets.hovered.bg_fill = BURGUNDY_LIGHT;
            style.visuals.widgets.active.bg_fill = GOLD_DARK;
            style.visuals.selection.bg_fill = GOLD_DARK;
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

        let api_key = self.settings.active_api_key();
        let base_url = self.settings.active_base_url();
        let tx = self.event_tx.clone();
        self.tokio_rt.spawn(async move {
            let client = crate::api::OpenAIClient::new(&api_key, &base_url);
            let model_ids = match client.list_models().await {
                Ok(ids) => ids,
                Err(_) => vec![],
            };
            let _ = tx.send(AppEvent::ModelsLoaded(model_ids));
        });
    }

    fn new_session(&mut self) {
        let session = Session::new("New Chat");
        let db_session = DbSession {
            id: session.id.clone(),
            name: session.name.clone(),
            created_at: Utc::now(),
        };

        let db_err = if let Ok(guard) = self.db.lock() {
            if let Some(database) = guard.as_ref() {
                database.create_session(&db_session).err().map(|e| e.to_string())
            } else {
                None
            }
        } else {
            Some("Database lock poisoned".to_string())
        };
        if let Some(err) = db_err {
            self.notify(format!("Failed to create DB session: {err}"), NotificationKind::Error);
        }

        self.sessions.insert(0, session);
        self.current_session_idx = Some(0);
        self.activity_log.push("Created new conversation".to_string());
    }

    fn load_session_messages(&mut self, idx: usize) {
        let Some(session) = self.sessions.get(idx) else {
            return;
        };
        let session_id = session.id.clone();
        let messages = if let Ok(guard) = self.db.lock() {
            if let Some(database) = guard.as_ref() {
                database.load_messages(&session_id).unwrap_or_default()
            } else {
                vec![]
            }
        } else {
            vec![]
        };

        if let Some(s) = self.sessions.get_mut(idx) {
            s.messages = messages
                .into_iter()
                .map(|m| Message {
                    id: m.id,
                    role: match m.role.as_str() {
                        "user" => Role::User,
                        "assistant" => Role::Assistant,
                        _ => Role::System,
                    },
                    content: m.content,
                    attachments: vec![],
                    timestamp: m.created_at,
                    is_streaming: false,
                })
                .collect();
        }
    }

    fn save_message(&mut self, session_id: &str, msg: &Message) {
        let db_msg = DbMessage {
            id: msg.id.clone(),
            session_id: session_id.to_string(),
            role: msg.role.as_str().to_string(),
            content: msg.content.clone(),
            created_at: msg.timestamp,
        };

        let db_err = if let Ok(guard) = self.db.lock() {
            if let Some(database) = guard.as_ref() {
                database.save_message(&db_msg).err().map(|e| e.to_string())
            } else {
                None
            }
        } else {
            Some("Database lock poisoned".to_string())
        };
        if let Some(err) = db_err {
            self.notify(format!("Failed to save message: {err}"), NotificationKind::Error);
        }
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
        let db_err = if let Ok(guard) = self.db.lock() {
            if let Some(database) = guard.as_ref() {
                database.save_file_snapshot(&snapshot).err().map(|e| e.to_string())
            } else {
                None
            }
        } else {
            Some("Database lock poisoned".to_string())
        };
        if let Some(err) = db_err {
            self.notify(format!("Failed to persist file snapshot: {err}"), NotificationKind::Error);
        }
    }

    fn refresh_snapshots(&mut self) {
        let result = if let Ok(guard) = self.db.lock() {
            if let Some(database) = guard.as_ref() {
                database.list_file_snapshots(50)
            } else {
                Ok(vec![])
            }
        } else {
            Err(anyhow::anyhow!("Database lock poisoned"))
        };
        match result {
            Ok(list) => {
                self.snapshots = list;
            }
            Err(e) => {
                self.snapshots.clear();
                self.notify(format!("Failed to load snapshots: {e}"), NotificationKind::Error);
            }
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
            api_messages.push(ChatMessage::with_cache_control("system", &system_prompt));

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

            let api_key = self.settings.active_api_key();
            let base_url = self.settings.active_base_url();
            let event_tx = self.event_tx.clone();
            let req_id = request_id.clone();
            let model_id = model_id.clone();

            self.tokio_rt.spawn(async move {
                let client = crate::api::OpenAIClient::new(&api_key, &base_url);
                let req_id_clone = req_id.clone();
                let tx_clone = event_tx.clone();

                let result = client
                    .chat_completion(&model_id, api_messages, thinking_mode.as_ref(), move |chunk| {
                        let _ = tx_clone.send(AppEvent::ChunkReceived(req_id_clone.clone(), chunk));
                    })
                    .await;

                match result {
                    Ok(_) => {
                        let _ = event_tx.send(AppEvent::ResponseComplete(req_id));
                    }
                    Err(e) => {
                        let _ = event_tx.send(AppEvent::ResponseError(req_id, e.to_string()));
                    }
                }
            });
        }
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
                        if let Some(session) = self.sessions.get_mut(active.session_idx) {
                            let session_id = session.id.clone();
                            if let Some(msg) = session.messages.iter_mut().find(|m| m.id == active.message_id) {
                                msg.is_streaming = false;
                                if self.pending_action.is_none() {
                                    if let Some(action) = Self::parse_pending_action(&msg.content) {
                                        self.pending_action = Some(action);
                                        self.pending_action_session_idx = Some(active.session_idx);
                                    }
                                }
                                to_save = Some((session_id, msg.clone()));
                            }
                        }
                        if let Some((session_id, msg)) = to_save {
                            self.save_message(&session_id, &msg);
                        }
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
                    self.remote_models = model_ids;
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
                                match std::fs::write(std::path::Path::new(&snap.file_path), &snap.content) {
                                    Ok(_) => self.notify(format!("Restored {}", snap.file_path), NotificationKind::Info),
                                    Err(e) => self.notify(format!("Restore failed: {e}"), NotificationKind::Error),
                                }
                            }
                        });
                    }
                });
            });
        self.show_snapshots = open;
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
                    ui.visuals_mut().override_text_color = Some(DARK_TEXT);
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
                    }
                    ui.separator();
                    ui.horizontal(|ui| {
                        if ui.button(RichText::new("✅ Approve").color(DARK_TEXT)).clicked() {
                            approved = true;
                        }
                        if ui.button(RichText::new("❌ Reject").color(DARK_TEXT)).clicked() {
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
                        } else {
                            self.notify(format!("File written: {}", full_path.display()), NotificationKind::Info);
                            self.append_system_message(
                                action_session_idx,
                                format!("✅ File written (approved): `{}`", full_path.display()),
                            );
                            self.refresh_snapshots();
                        }
                    }
                    PendingAction::ExecuteCommand { command } => {
                        if !self.settings.shell_execution_enabled {
                            self.notify("Shell execution is disabled", NotificationKind::Error);
                            self.append_system_message(
                                action_session_idx,
                                "❌ Command not executed: shell execution is disabled in settings.".to_string(),
                            );
                        } else {
                            let wd = self.settings.working_directory.clone();
                            match crate::shell::execute_command(command, &wd) {
                                Ok(out) => {
                                    self.notify(format!("Command exited {}", out.exit_code), NotificationKind::Info);
                                    let mut result = format!(
                                        "✅ Command executed (approved)\nCommand: `{}`\nWorking directory: `{}`\nExit code: {}",
                                        command, wd, out.exit_code
                                    );
                                    if !out.stdout.trim().is_empty() {
                                        result.push_str("\n\nstdout:\n````text\n");
                                        result.push_str(&out.stdout);
                                        result.push_str("\n````");
                                    }
                                    if !out.stderr.trim().is_empty() {
                                        result.push_str("\n\nstderr:\n````text\n");
                                        result.push_str(&out.stderr);
                                        result.push_str("\n````");
                                    }
                                    self.append_system_message(action_session_idx, result);
                                }
                                Err(e) => {
                                    self.notify(format!("Command error: {e}"), NotificationKind::Error);
                                    self.append_system_message(
                                        action_session_idx,
                                        format!("❌ Command execution failed for `{}`: {}", command, e),
                                    );
                                }
                            }
                        }
                    }
                }
                self.pending_action = None;
                self.pending_action_session_idx = None;
            }
            if rejected {
                self.append_system_message(
                    self.pending_action_session_idx,
                    "❌ Action rejected by user. No file write or command execution was performed.".to_string(),
                );
                self.pending_action = None;
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

        egui::SidePanel::left("sidebar")
            .resizable(true)
            .default_width(260.0)
            .min_width(180.0)
            .frame(egui::Frame::new().fill(SKY_BLUE).inner_margin(egui::Margin::same(8)))
            .show(ctx, |ui| {
                ui.visuals_mut().override_text_color = Some(DARK_TEXT);
                ui.label(
                    RichText::new("🤖 AI Chat Bot")
                        .font(FontId::proportional(18.0))
                        .color(BURGUNDY_DARK)
                        .strong(),
                );
                ui.separator();

                if ui
                    .add_sized(
                        [ui.available_width(), 32.0],
                        egui::Button::new(RichText::new("＋ New Chat").color(DARK_TEXT)).fill(GOLD),
                    )
                    .clicked()
                {
                    self.new_session();
                }

                ui.separator();
                ui.label(RichText::new("Conversations").strong().color(BURGUNDY_DARK));
                ScrollArea::vertical().max_height(200.0).show(ui, |ui| {
                    for idx in 0..self.sessions.len() {
                        let is_current = self.current_session_idx == Some(idx);
                        let name = self
                            .sessions
                            .get(idx)
                            .map(|s| s.name.clone())
                            .unwrap_or_else(|| "Conversation".to_string());
                        let btn = egui::Button::new(RichText::new(format!("💬 {name}")).color(DARK_TEXT))
                            .fill(if is_current { GOLD_DARK } else { SKY_BLUE_DARK })
                            .min_size(Vec2::new(ui.available_width(), 28.0));
                        if ui.add(btn).clicked() {
                            self.current_session_idx = Some(idx);
                            if self.sessions.get(idx).map(|s| s.messages.is_empty()).unwrap_or(false) {
                                self.load_session_messages(idx);
                            }
                        }
                    }
                });

                ui.separator();
                ui.label(RichText::new("Model").strong().color(BURGUNDY_DARK));
                let selected_model_name = self
                    .models
                    .get(self.selected_model_idx)
                    .map(|m| m.name.clone())
                    .unwrap_or_else(|| "(none)".to_string());
                egui::ComboBox::from_id_salt("model_selector")
                    .selected_text(selected_model_name)
                    .width(ui.available_width())
                    .show_ui(ui, |ui| {
                        for (i, m) in self.models.iter().enumerate() {
                            ui.selectable_value(&mut self.selected_model_idx, i, &m.name);
                        }
                        if !self.remote_models.is_empty() {
                            ui.separator();
                            ui.label("Remote model IDs:");
                            for id in self.remote_models.iter().take(5) {
                                ui.label(RichText::new(id).small());
                            }
                        }
                    });
                ui.label(RichText::new("Custom model ID").small().color(DARK_TEXT));
                ui.horizontal(|ui| {
                    ui.add(
                        TextEdit::singleline(&mut self.custom_model_id)
                            .desired_width(ui.available_width() - CUSTOM_MODEL_INPUT_RESERVED_WIDTH)
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
                    }
                    if ui.button("From list").clicked() {
                        self.custom_model_id.clear();
                        self.settings.default_model = self
                            .current_model()
                            .map(|m| m.id.clone())
                            .unwrap_or_else(|| DEFAULT_MODEL_ID.to_string());
                        let _ = save_settings(&self.settings);
                    }
                });
                if !self.remote_models.is_empty() {
                    ui.horizontal_wrapped(|ui| {
                        ui.label(RichText::new("Quick pick:").small().color(DARK_TEXT));
                        for id in self.remote_models.iter().take(8) {
                            if ui.small_button(id).clicked() {
                                self.custom_model_id = id.clone();
                            }
                        }
                    });
                }

                let supports_thinking = self.current_model().map(|m| m.supports_thinking()).unwrap_or(false);
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
                                ui.selectable_value(&mut self.selected_thinking_mode, ThinkingMode::Disabled, "Disabled");
                                ui.selectable_value(&mut self.selected_thinking_mode, ThinkingMode::Auto, "Auto");
                                ui.selectable_value(&mut self.selected_thinking_mode, ThinkingMode::Low, "Low");
                                ui.selectable_value(&mut self.selected_thinking_mode, ThinkingMode::Medium, "Medium");
                                ui.selectable_value(&mut self.selected_thinking_mode, ThinkingMode::High, "High");
                            });
                    });
                }

                ui.separator();
                ui.label(RichText::new("Agents").strong().color(BURGUNDY_DARK));
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

                ui.label(RichText::new("Working Dir").strong().color(BURGUNDY_DARK));
                ui.label(RichText::new(&self.settings.working_directory).small());
                if ui.button("📁 Change Directory").clicked() {
                    self.select_working_dir();
                }

                if ui.button("Shadow rollback").clicked() {
                    self.show_snapshots = true;
                    self.refresh_snapshots();
                }
                if ui.button("⚙ Settings").clicked() {
                    self.show_settings = true;
                }
            });

        egui::SidePanel::right("activity")
            .resizable(true)
            .default_width(220.0)
            .frame(egui::Frame::new().fill(BURGUNDY_DARK).inner_margin(egui::Margin::same(8)))
            .show(ctx, |ui| {
                ui.label(RichText::new("Activity Monitor").color(GOLD).strong());
                ui.separator();
                for req in self.active_requests.values() {
                    ui.label(RichText::new(format!("⏳ {} is responding", req.agent_name)).color(SKY_BLUE));
                }
                ui.separator();
                ScrollArea::vertical().show(ui, |ui| {
                    for line in self.activity_log.iter().rev().take(20) {
                        ui.label(RichText::new(line).small().color(WHITE));
                    }
                });
            });

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
                            if response.has_focus() && ctx.input(|i| i.key_pressed(egui::Key::Enter) && i.modifiers.ctrl) {
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
                                        egui::Button::new(RichText::new("📎 Attach").color(DARK_TEXT)).fill(SKY_BLUE_DARK),
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
