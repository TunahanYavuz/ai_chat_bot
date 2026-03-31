use egui::{Color32, FontId, RichText, ScrollArea, TextEdit, Vec2};
use std::sync::{Arc, Mutex};
use chrono::Utc;
use uuid::Uuid;

use crate::api::{ChatMessage, ModelInfo, ThinkingLevel, builtin_models};
use crate::config::{Settings, load_settings, save_settings};
use crate::db::{Database, DbSession, DbMessage};

// ─── Color Palette ────────────────────────────────────────────────────────────
const BURGUNDY: Color32 = Color32::from_rgb(114, 47, 55);
const BURGUNDY_LIGHT: Color32 = Color32::from_rgb(140, 65, 75);
const BURGUNDY_DARK: Color32 = Color32::from_rgb(80, 30, 38);
const SKY_BLUE: Color32 = Color32::from_rgb(135, 206, 235);
const SKY_BLUE_DARK: Color32 = Color32::from_rgb(100, 170, 200);
const GOLD: Color32 = Color32::from_rgb(255, 215, 0);
const GOLD_DARK: Color32 = Color32::from_rgb(220, 180, 0);
const WHITE: Color32 = Color32::WHITE;
const DARK_TEXT: Color32 = Color32::from_rgb(20, 10, 12);

// ─── App message types ────────────────────────────────────────────────────────
#[derive(Debug)]
pub enum AppEvent {
    ChunkReceived(String, String), // (request_id, chunk)
    ResponseComplete(String),      // request_id
    ResponseError(String, String), // (request_id, error)
    ModelsLoaded(Vec<String>),
}

// ─── Chat data model ──────────────────────────────────────────────────────────
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

impl Message {
    pub fn new(role: Role, content: &str) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            role,
            content: content.to_string(),
            attachments: vec![],
            timestamp: Utc::now(),
            is_streaming: false,
        }
    }
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

// ─── Pending AI tool actions ──────────────────────────────────────────────────
#[derive(Debug, Clone)]
pub enum PendingAction {
    WriteFile { path: String, content: String },
    ExecuteCommand { command: String },
}

// ─── Main Application State ───────────────────────────────────────────────────
pub struct ChatApp {
    // Settings
    settings: Settings,
    show_settings: bool,

    // Sessions
    sessions: Vec<Session>,
    current_session_idx: Option<usize>,

    // Available models
    models: Vec<ModelInfo>,
    selected_model_idx: usize,
    remote_models: Vec<String>,

    // Thinking
    thinking_enabled: bool,
    selected_thinking_level: ThinkingLevel,

    // Input
    input_text: String,
    pending_attachments: Vec<Attachment>,

    // Async communication
    event_tx: std::sync::mpsc::Sender<AppEvent>,
    event_rx: std::sync::mpsc::Receiver<AppEvent>,
    tokio_rt: Arc<tokio::runtime::Runtime>,

    // Streaming state: maps request_id -> message_id
    active_request: Option<(String, String)>, // (request_id, message_id)

    // Database
    db: Arc<Mutex<Option<Database>>>,

    // Pending approvals
    pending_action: Option<PendingAction>,

    // Image download notification
    status_message: Option<String>,
    status_timer: f32,
}

impl ChatApp {
    pub fn new(cc: &eframe::CreationContext, rt: Arc<tokio::runtime::Runtime>) -> Self {
        let (event_tx, event_rx) = std::sync::mpsc::channel();
        let settings = load_settings();

        // Style
        let mut style = (*cc.egui_ctx.style()).clone();
        style.visuals.override_text_color = Some(WHITE);
        style.visuals.panel_fill = BURGUNDY;
        style.visuals.window_fill = BURGUNDY_DARK;
        style.visuals.extreme_bg_color = BURGUNDY_DARK;
        style.visuals.widgets.noninteractive.bg_fill = BURGUNDY;
        style.visuals.widgets.inactive.bg_fill = BURGUNDY_LIGHT;
        style.visuals.widgets.hovered.bg_fill = BURGUNDY_LIGHT;
        style.visuals.widgets.active.bg_fill = GOLD_DARK;
        style.visuals.selection.bg_fill = GOLD_DARK;
        cc.egui_ctx.set_style(style);

        // Open DB
        let db = match Database::new(&settings.db_path) {
            Ok(d) => Some(d),
            Err(e) => {
                eprintln!("DB error: {e}");
                None
            }
        };
        let db = Arc::new(Mutex::new(db));

        // Load sessions from DB
        let sessions = {
            let guard = db.lock().unwrap();
            if let Some(database) = guard.as_ref() {
                database.list_sessions().unwrap_or_default()
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
        };

        let models = builtin_models();

        let mut app = Self {
            settings,
            show_settings: false,
            sessions,
            current_session_idx: None,
            models,
            selected_model_idx: 0,
            remote_models: vec![],
            thinking_enabled: false,
            selected_thinking_level: ThinkingLevel::Medium,
            input_text: String::new(),
            pending_attachments: vec![],
            event_tx,
            event_rx,
            tokio_rt: rt,
            active_request: None,
            db,
            pending_action: None,
            status_message: None,
            status_timer: 0.0,
        };

        // Set default model from settings
        if let Some(idx) = app.models.iter().position(|m| m.id == app.settings.default_model) {
            app.selected_model_idx = idx;
        }

        app
    }

    fn current_model(&self) -> &ModelInfo {
        &self.models[self.selected_model_idx]
    }

    fn current_session(&self) -> Option<&Session> {
        self.current_session_idx.map(|i| &self.sessions[i])
    }

    fn current_session_mut(&mut self) -> Option<&mut Session> {
        self.current_session_idx.map(|i| &mut self.sessions[i])
    }

    fn new_session(&mut self) {
        let session = Session::new("New Chat");
        let db_session = DbSession {
            id: session.id.clone(),
            name: session.name.clone(),
            created_at: Utc::now(),
        };
        {
            let guard = self.db.lock().unwrap();
            if let Some(database) = guard.as_ref() {
                let _ = database.create_session(&db_session);
            }
        }
        self.sessions.insert(0, session);
        self.current_session_idx = Some(0);
    }

    fn load_session_messages(&mut self, idx: usize) {
        let session_id = self.sessions[idx].id.clone();
        let messages = {
            let guard = self.db.lock().unwrap();
            if let Some(database) = guard.as_ref() {
                database.load_messages(&session_id).unwrap_or_default()
            } else {
                vec![]
            }
        };
        self.sessions[idx].messages = messages
            .into_iter()
            .map(|m| {
                let role = match m.role.as_str() {
                    "user" => Role::User,
                    "assistant" => Role::Assistant,
                    _ => Role::System,
                };
                Message {
                    id: m.id,
                    role,
                    content: m.content,
                    attachments: vec![],
                    timestamp: m.created_at,
                    is_streaming: false,
                }
            })
            .collect();
    }

    fn save_message(&self, session_id: &str, msg: &Message) {
        let db_msg = DbMessage {
            id: msg.id.clone(),
            session_id: session_id.to_string(),
            role: msg.role.as_str().to_string(),
            content: msg.content.clone(),
            created_at: msg.timestamp,
        };
        let guard = self.db.lock().unwrap();
        if let Some(database) = guard.as_ref() {
            let _ = database.save_message(&db_msg);
        }
    }

    fn send_message(&mut self) {
        let text = self.input_text.trim().to_string();
        if text.is_empty() && self.pending_attachments.is_empty() {
            return;
        }

        let session_idx = match self.current_session_idx {
            Some(i) => i,
            None => {
                self.new_session();
                self.current_session_idx.unwrap()
            }
        };

        let attachments = std::mem::take(&mut self.pending_attachments);

        // Build user message content
        let mut display_content = text.clone();
        for att in &attachments {
            display_content.push_str(&format!("\n[Attachment: {}]", att.filename));
        }

        let user_msg = Message {
            id: Uuid::new_v4().to_string(),
            role: Role::User,
            content: display_content.clone(),
            attachments: attachments.clone(),
            timestamp: Utc::now(),
            is_streaming: false,
        };

        let session_id = self.sessions[session_idx].id.clone();
        self.save_message(&session_id, &user_msg);

        // Update session name if first message
        if self.sessions[session_idx].messages.is_empty() && !text.is_empty() {
            let name: String = text.chars().take(40).collect();
            self.sessions[session_idx].name = name;
        }

        self.sessions[session_idx].messages.push(user_msg);
        self.input_text.clear();

        // Build placeholder assistant message
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
        self.sessions[session_idx].messages.push(assistant_msg);
        self.active_request = Some((request_id.clone(), assistant_msg_id));

        // Build API messages
        let mut api_messages: Vec<ChatMessage> = vec![];

        // System prompt
        let system_prompt = format!(
            "You are a helpful AI assistant. Working directory: {}. Shell execution: {}.",
            self.settings.working_directory,
            if self.settings.shell_execution_enabled { "enabled" } else { "disabled" }
        );
        api_messages.push(ChatMessage::with_cache_control("system", &system_prompt));

        // History
        for msg in &self.sessions[session_idx].messages {
            if msg.is_streaming {
                continue;
            }
            let role = msg.role.as_str();
            if msg.attachments.iter().any(|a| a.image_base64.is_some()) {
                for att in &msg.attachments {
                    if let Some(b64) = &att.image_base64 {
                        api_messages.push(ChatMessage::with_image(
                            role,
                            &msg.content,
                            b64,
                            &att.mime_type,
                        ));
                    }
                }
            } else {
                let mut content = msg.content.clone();
                for att in &msg.attachments {
                    if let Some(text) = &att.text {
                        content.push_str(&format!("\n\n[File: {}]\n{}", att.filename, text));
                    }
                }
                api_messages.push(ChatMessage::text(role, &content));
            }
        }

        // Current user message (last)
        if let Some(last) = api_messages.last_mut() {
            if last.role == "user" {
                // Already added above
            }
        }

        // Spawn async task
        let api_key = self.settings.active_api_key();
        let base_url = self.settings.active_base_url();
        let model_id = self.current_model().id.clone();
        let thinking_level = if self.thinking_enabled && self.current_model().supports_thinking() {
            Some(self.selected_thinking_level.clone())
        } else {
            None
        };
        let event_tx = self.event_tx.clone();
        let req_id = request_id.clone();

        self.tokio_rt.spawn(async move {
            let client = crate::api::OpenAIClient::new(&api_key, &base_url);
            let req_id_clone = req_id.clone();
            let tx_clone = event_tx.clone();

            let result = client
                .chat_completion(
                    &model_id,
                    api_messages,
                    thinking_level.as_ref(),
                    move |chunk| {
                        let _ = tx_clone.send(AppEvent::ChunkReceived(req_id_clone.clone(), chunk));
                    },
                )
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

    fn process_events(&mut self) {
        while let Ok(event) = self.event_rx.try_recv() {
            match event {
                AppEvent::ChunkReceived(req_id, chunk) => {
                    if let Some((active_req, msg_id)) = &self.active_request {
                        if *active_req == req_id {
                            let msg_id = msg_id.clone();
                            if let Some(session) = self.current_session_mut() {
                                if let Some(msg) = session.messages.iter_mut().find(|m| m.id == msg_id) {
                                    msg.content.push_str(&chunk);
                                }
                            }
                        }
                    }
                }
                AppEvent::ResponseComplete(req_id) => {
                    if let Some((active_req, msg_id)) = self.active_request.take() {
                        if active_req == req_id {
                            if let Some(idx) = self.current_session_idx {
                                let session_id = self.sessions[idx].id.clone();
                                if let Some(msg) = self.sessions[idx].messages.iter_mut().find(|m| m.id == msg_id) {
                                    msg.is_streaming = false;
                                    let msg_clone = msg.clone();
                                    self.save_message(&session_id, &msg_clone);
                                }
                            }
                        }
                    }
                }
                AppEvent::ResponseError(req_id, err) => {
                    if let Some((active_req, msg_id)) = self.active_request.take() {
                        if active_req == req_id {
                            if let Some(session) = self.current_session_mut() {
                                if let Some(msg) = session.messages.iter_mut().find(|m| m.id == msg_id) {
                                    msg.content = format!("❌ Error: {}", err);
                                    msg.is_streaming = false;
                                }
                            }
                        }
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
                }
                Err(e) => {
                    self.status_message = Some(format!("Failed to read file: {}", e));
                    self.status_timer = 3.0;
                }
            }
        }
    }

    fn download_image(&mut self, att: &Attachment) {
        if let Some(path) = rfd::FileDialog::new()
            .set_file_name(&att.filename)
            .save_file()
        {
            if let Err(e) = std::fs::write(&path, &att.raw_bytes) {
                self.status_message = Some(format!("Failed to save image: {}", e));
                self.status_timer = 3.0;
            }
        }
    }

    fn select_working_dir(&mut self) {
        if let Some(path) = rfd::FileDialog::new().pick_folder() {
            self.settings.working_directory = path.to_string_lossy().to_string();
            let _ = save_settings(&self.settings);
        }
    }
}

impl eframe::App for ChatApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.process_events();

        // Decrement status timer
        if self.status_timer > 0.0 {
            self.status_timer -= ctx.input(|i| i.stable_dt);
            if self.status_timer <= 0.0 {
                self.status_message = None;
            }
        }

        // Request repaint while streaming
        if self.active_request.is_some() {
            ctx.request_repaint();
        }

        // Pending action dialog
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
                            ui.label(format!("Write file: {}", path));
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
                match &action {
                    PendingAction::WriteFile { path, content } => {
                        let _ = crate::files::write_text_file(
                            std::path::Path::new(path),
                            content,
                        );
                    }
                    PendingAction::ExecuteCommand { command } => {
                        let wd = self.settings.working_directory.clone();
                        match crate::shell::execute_command(command, &wd) {
                            Ok(out) => {
                                self.status_message = Some(format!(
                                    "Command exited {}: {}",
                                    out.exit_code,
                                    out.stdout.lines().next().unwrap_or("")
                                ));
                                self.status_timer = 5.0;
                            }
                            Err(e) => {
                                self.status_message = Some(format!("Command error: {}", e));
                                self.status_timer = 5.0;
                            }
                        }
                    }
                }
                self.pending_action = None;
            }
            if rejected {
                self.pending_action = None;
            }
        }

        // Settings panel
        if self.show_settings {
            egui::Window::new("⚙ Settings")
                .collapsible(false)
                .resizable(true)
                .anchor(egui::Align2::CENTER_CENTER, Vec2::ZERO)
                .show(ctx, |ui| {
                    ui.set_min_width(400.0);
                    egui::Grid::new("settings_grid")
                        .num_columns(2)
                        .spacing([10.0, 8.0])
                        .show(ui, |ui| {
                            ui.label("Provider:");
                            let provider_name = self.settings.selected_provider.display_name();
                            egui::ComboBox::from_id_salt("provider_sel")
                                .selected_text(provider_name)
                                .width(280.0)
                                .show_ui(ui, |ui| {
                                    for p in crate::config::ApiProvider::all() {
                                        let label = p.display_name();
                                        let mut cur = self.settings.selected_provider.clone();
                                        if ui.selectable_value(&mut cur, p.clone(), label).clicked() {
                                            self.settings.selected_provider = p;
                                        }
                                    }
                                });
                            ui.end_row();

                            let provider = self.settings.selected_provider.clone();
                            let cfg = self.settings.provider_config_mut(&provider);

                            ui.label("API Key:");
                            ui.add(
                                TextEdit::singleline(&mut cfg.api_key)
                                    .password(true)
                                    .desired_width(280.0),
                            );
                            ui.end_row();

                            ui.label("Base URL:");
                            ui.add(
                                TextEdit::singleline(&mut cfg.base_url)
                                    .desired_width(280.0),
                            );
                            ui.end_row();

                            ui.label("DB Path:");
                            ui.add(
                                TextEdit::singleline(&mut self.settings.db_path)
                                    .desired_width(280.0),
                            );
                            ui.end_row();
                        });

                    ui.separator();
                    ui.horizontal(|ui| {
                        if ui.button("Save").clicked() {
                            let _ = save_settings(&self.settings);
                            self.show_settings = false;
                        }
                        if ui.button("Cancel").clicked() {
                            self.show_settings = false;
                        }
                    });
                });
        }

        // ─── Left Sidebar ──────────────────────────────────────────────────────
        egui::SidePanel::left("sidebar")
            .resizable(true)
            .default_width(230.0)
            .min_width(160.0)
            .frame(egui::Frame::new().fill(SKY_BLUE).inner_margin(egui::Margin::same(8_i8)))
            .show(ctx, |ui| {
                ui.visuals_mut().override_text_color = Some(DARK_TEXT);

                // App title
                ui.add_space(4.0);
                ui.label(
                    RichText::new("🤖 AI Chat Bot")
                        .font(FontId::proportional(18.0))
                        .color(BURGUNDY_DARK)
                        .strong(),
                );
                ui.separator();

                // New Chat button
                if ui
                    .add_sized(
                        [ui.available_width(), 32.0],
                        egui::Button::new(RichText::new("＋ New Chat").color(DARK_TEXT))
                            .fill(GOLD),
                    )
                    .clicked()
                {
                    self.new_session();
                }

                ui.add_space(6.0);
                ui.label(RichText::new("Conversations").strong().color(BURGUNDY_DARK));
                ui.separator();

                // Session list
                let available_h = ui.available_height() - 280.0;
                ScrollArea::vertical()
                    .id_salt("sessions_scroll")
                    .max_height(available_h.max(80.0))
                    .show(ui, |ui| {
                        let session_count = self.sessions.len();
                        for idx in 0..session_count {
                            let is_current = self.current_session_idx == Some(idx);
                            let name = self.sessions[idx].name.clone();
                            let btn = egui::Button::new(
                                RichText::new(format!("💬 {}", name)).color(DARK_TEXT),
                            )
                            .fill(if is_current { GOLD_DARK } else { SKY_BLUE_DARK })
                            .min_size(Vec2::new(ui.available_width(), 28.0));

                            if ui.add(btn).clicked() {
                                self.current_session_idx = Some(idx);
                                if self.sessions[idx].messages.is_empty() {
                                    self.load_session_messages(idx);
                                }
                            }
                        }
                    });

                ui.separator();

                // Model selector
                ui.label(RichText::new("Model").strong().color(BURGUNDY_DARK));
                let model_names: Vec<String> =
                    self.models.iter().map(|m| m.name.clone()).collect();
                egui::ComboBox::from_id_salt("model_selector")
                    .selected_text(&model_names[self.selected_model_idx])
                    .width(ui.available_width())
                    .show_ui(ui, |ui| {
                        for (i, name) in model_names.iter().enumerate() {
                            ui.selectable_value(&mut self.selected_model_idx, i, name);
                        }
                    });

                ui.add_space(4.0);

                // Thinking mode
                let model_supports_thinking = self.models[self.selected_model_idx].supports_thinking();
                ui.horizontal(|ui| {
                    ui.checkbox(&mut self.thinking_enabled, "");
                    ui.label(RichText::new("Thinking Mode").color(BURGUNDY_DARK));
                });
                if !model_supports_thinking && self.thinking_enabled {
                    self.thinking_enabled = false;
                }

                if model_supports_thinking && self.thinking_enabled {
                    ui.horizontal(|ui| {
                        ui.label("Level:");
                        egui::ComboBox::from_id_salt("thinking_level")
                            .selected_text(match self.selected_thinking_level {
                                ThinkingLevel::Low => "Low",
                                ThinkingLevel::Medium => "Medium",
                                ThinkingLevel::High => "High",
                            })
                            .show_ui(ui, |ui| {
                                ui.selectable_value(
                                    &mut self.selected_thinking_level,
                                    ThinkingLevel::Low,
                                    "Low",
                                );
                                ui.selectable_value(
                                    &mut self.selected_thinking_level,
                                    ThinkingLevel::Medium,
                                    "Medium",
                                );
                                ui.selectable_value(
                                    &mut self.selected_thinking_level,
                                    ThinkingLevel::High,
                                    "High",
                                );
                            });
                    });
                }

                ui.separator();

                // Shell execution toggle
                ui.horizontal(|ui| {
                    ui.checkbox(&mut self.settings.shell_execution_enabled, "");
                    ui.label(RichText::new("Shell Execution").color(BURGUNDY_DARK));
                });

                ui.add_space(4.0);

                // Working directory
                ui.label(RichText::new("Working Dir").strong().color(BURGUNDY_DARK));
                let wd = &self.settings.working_directory;
                let wd_display: String = if wd.chars().count() > 28 {
                    let skip = wd.chars().count() - 28;
                    format!("…{}", wd.chars().skip(skip).collect::<String>())
                } else {
                    wd.clone()
                };
                ui.label(
                    RichText::new(format!("…{}", wd_display))
                        .small()
                        .color(BURGUNDY_DARK),
                );
                if ui.button("📁 Change Directory").clicked() {
                    self.select_working_dir();
                }

                ui.separator();

                // Settings button
                if ui
                    .add_sized(
                        [ui.available_width(), 28.0],
                        egui::Button::new(RichText::new("⚙ Settings").color(DARK_TEXT))
                            .fill(GOLD_DARK),
                    )
                    .clicked()
                {
                    self.show_settings = true;
                }
            });

        // ─── Main chat area ────────────────────────────────────────────────────
        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(BURGUNDY).inner_margin(egui::Margin::same(0_i8)))
            .show(ctx, |ui| {
                // Header bar
                egui::Frame::new()
                    .fill(BURGUNDY_DARK)
                    .inner_margin(egui::Margin::same(8_i8))
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

                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    if self.active_request.is_some() {
                                        ui.spinner();
                                        ui.label(
                                            RichText::new("Thinking…").color(GOLD).italics(),
                                        );
                                    }
                                    if let Some(msg) = &self.status_message.clone() {
                                        ui.label(RichText::new(msg).color(GOLD).small());
                                    }
                                },
                            );
                        });
                    });

                // Messages area
                let input_height = 90.0;
                let msg_height = ui.available_height() - input_height;

                egui::Frame::new()
                    .fill(BURGUNDY)
                    .inner_margin(egui::Margin { left: 12_i8, right: 12_i8, top: 8_i8, bottom: 4_i8 })
                    .show(ui, |ui| {
                        ScrollArea::vertical()
                            .id_salt("messages_scroll")
                            .max_height(msg_height)
                            .stick_to_bottom(true)
                            .show(ui, |ui| {
                                ui.set_width(ui.available_width());

                                if let Some(idx) = self.current_session_idx {
                                    let messages = self.sessions[idx].messages.clone();
                                    for msg in &messages {
                                        self.render_message(ui, msg);
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

                // Input area
                egui::Frame::new()
                    .fill(BURGUNDY_DARK)
                    .inner_margin(egui::Margin::same(8_i8))
                    .show(ui, |ui| {
                        ui.set_height(input_height);

                        // Pending attachments
                        if !self.pending_attachments.is_empty() {
                            ui.horizontal(|ui| {
                                ui.label(
                                    RichText::new("📎 Attachments:").color(GOLD).small(),
                                );
                                let att_names: Vec<String> = self
                                    .pending_attachments
                                    .iter()
                                    .map(|a| a.filename.clone())
                                    .collect();
                                for name in &att_names {
                                    ui.label(
                                        RichText::new(name).color(SKY_BLUE).small(),
                                    );
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

                            // Send on Ctrl+Enter
                            if response.has_focus()
                                && ctx.input(|i| {
                                    i.key_pressed(egui::Key::Enter)
                                        && i.modifiers.ctrl
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
                                    && self.active_request.is_none()
                                {
                                    self.send_message();
                                }

                                if ui
                                    .add_sized(
                                        [110.0, 28.0],
                                        egui::Button::new(
                                            RichText::new("📎 Attach").color(DARK_TEXT),
                                        )
                                        .fill(SKY_BLUE_DARK),
                                    )
                                    .clicked()
                                {
                                    self.attach_file();
                                }
                            });
                        });
                    });
            });
    }
}

impl ChatApp {
    fn render_message(&mut self, ui: &mut egui::Ui, msg: &Message) {
        let is_user = msg.role == Role::User;
        let width = ui.available_width();
        let max_bubble = (width * 0.75).min(600.0);

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
                    .corner_radius(8.0_f32)
                    .inner_margin(egui::Margin::same(10_i8))
                    .show(ui, |ui| {
                        ui.set_max_width(max_bubble - 20.0);

                        // Role label
                        let role_label = if is_user { "You" } else { "Assistant" };
                        ui.label(
                            RichText::new(role_label)
                                .small()
                                .color(if is_user { BURGUNDY } else { SKY_BLUE })
                                .strong(),
                        );

                        // Content
                        if msg.is_streaming && msg.content.is_empty() {
                            ui.spinner();
                        } else {
                            ui.label(
                                RichText::new(&msg.content).color(text_color),
                            );
                        }

                        // Attachments
                        for att in &msg.attachments {
                            ui.separator();
                            if att.image_base64.is_some() {
                                // Show image indicator — clickable to download
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
                                ui.label(
                                    RichText::new(format!("📄 {}", att.filename))
                                        .color(SKY_BLUE)
                                        .small(),
                                );
                            }
                        }

                        // Timestamp
                        ui.label(
                            RichText::new(
                                msg.timestamp.format("%H:%M").to_string(),
                            )
                            .small()
                            .color(Color32::from_rgb(160, 160, 160)),
                        );
                    });
            },
        );
    }
}
