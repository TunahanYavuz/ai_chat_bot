use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Context, Result};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Child;
use tokio::sync::mpsc;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tokio::time::{timeout, Duration};

use crate::parser::{ActionKind, AgentAction};
use crate::web_engine::{format_search_results, WebEngine};
use crate::{
    mcp_client::{McpManager, McpServerConfig},
    screen_awareness,
};

const EMPTY_COMMAND_SUCCESS_MSG: &str = "[System: Command executed successfully with no output]";
const SUPPORTED_DOCUMENT_FORMATS: [&str; 2] = ["pdf", "docx"];
const TERMINAL_ACTION_TIMEOUT_SECS: u64 = 90;
const FILE_ACTION_TIMEOUT_SECS: u64 = 15;
const TIMEOUT_EXIT_CODE: i32 = 124;
const SCRIPTS_DIR: &str = "scripts";
const EXPORTS_DIR: &str = "exports";
const LOGS_DIR: &str = "logs";
const SCREENSHOTS_DIR: &str = "assets/screenshots";
const CORE_ROOT_FILENAMES: &[&str] = &[
    "cargo.toml",
    "cargo.lock",
    "readme.md",
    "main.rs",
    "lib.rs",
    "build.rs",
];

/// Execution status emitted by the action pipeline.
#[derive(Debug, Clone)]
pub enum ExecutionStatus {
    /// The action was executed and produced a normalized report.
    Executed(ExecutionReport),
    /// Execution was intentionally paused because user confirmation is required.
    AwaitingApproval(ApprovalRequest),
    /// Execution was denied by user authorization decision.
    AuthorizationDenied {
        action: AgentAction,
        reason: String,
    },
}

/// Approval payload that the UI can render for approve/reject interactions.
#[derive(Debug, Clone)]
pub struct ApprovalRequest {
    pub action: AgentAction,
    pub reason: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApprovalDecision {
    ApproveOnce,
    GrantTemporaryAccess,
    Deny,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionPolicy {
    Manual,
    ReadEdit,
    Execute,
    FullAccess,
}

impl ExecutionPolicy {
    pub fn requires_manual_approval(self, action: &AgentAction) -> bool {
        if self != ExecutionPolicy::FullAccess && action_requires_delete_safety(action) {
            return true;
        }
        match self {
            ExecutionPolicy::Manual => true,
            ExecutionPolicy::ReadEdit => {
                matches!(action.action, ActionKind::RunCmd | ActionKind::RunAndObserve)
            }
            ExecutionPolicy::Execute => false,
            ExecutionPolicy::FullAccess => false,
        }
    }

    pub fn approval_reason(self, action: &AgentAction) -> String {
        if self != ExecutionPolicy::FullAccess && action_requires_delete_safety(action) {
            return "Delete safety warning: destructive delete command/file deletion requires explicit confirmation".to_string();
        }
        match self {
            ExecutionPolicy::Manual => {
                "ExecutionPolicy::Manual requires explicit confirmation for all actions".to_string()
            }
            ExecutionPolicy::ReadEdit => {
                "ExecutionPolicy::ReadEdit requires confirmation for command execution".to_string()
            }
            ExecutionPolicy::Execute => {
                "ExecutionPolicy::Execute auto-approves commands and file edits".to_string()
            }
            ExecutionPolicy::FullAccess => "ExecutionPolicy::FullAccess auto-approves all actions".to_string(),
        }
    }
}

/// Normalized execution result returned to upper layers / LLM relay.
#[derive(Debug, Clone)]
pub struct ExecutionReport {
    pub action: String,
    pub success: bool,
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
    pub timed_out: bool,
    pub screenshot_png_bytes: Option<Vec<u8>>,
    pub screenshot_source: Option<String>,
    pub screenshot_width: Option<u32>,
    pub screenshot_height: Option<u32>,
}

/// Enterprise-grade async action executor.
///
/// - Uses `tokio::process::Command` for terminal operations.
/// - Uses `tokio::fs` for file/folder operations.
/// - Explicitly handles auto-approval workflow.
/// - Normalizes empty successful command output.
pub struct ActionExecutor {
    workspace_root: PathBuf,
    web_engine: Option<WebEngine>,
    web_engine_init_error: Option<String>,
    mcp_manager: Option<Arc<McpManager>>,
}

impl ActionExecutor {
    pub fn new(workspace_root: impl Into<PathBuf>, mcp_manager: Option<Arc<McpManager>>) -> Self {
        let (web_engine, web_engine_init_error) = match WebEngine::new() {
            Ok(engine) => (Some(engine), None),
            Err(e) => (None, Some(e.to_string())),
        };
        Self {
            workspace_root: workspace_root.into(),
            web_engine,
            web_engine_init_error,
            mcp_manager,
        }
    }

    /// Executes one action or yields an approval request depending on execution policy.
    pub async fn execute_action(
        &self,
        action: AgentAction,
        policy: ExecutionPolicy,
    ) -> Result<ExecutionStatus> {
        if policy.requires_manual_approval(&action) {
            let reason = policy.approval_reason(&action);
            return Ok(ExecutionStatus::AwaitingApproval(ApprovalRequest {
                action,
                reason,
            }));
        }

        let report = self.execute_approved_action(action).await?;
        Ok(ExecutionStatus::Executed(report))
    }

    /// Executes one action with async permission callback support.
    ///
    /// Returns the action status and the potentially-updated auto-approve state.
    pub async fn execute_action_with_permission<F, Fut>(
        &self,
        action: AgentAction,
        policy: ExecutionPolicy,
        current_auto_approve_state: bool,
        mut request_permission: F,
    ) -> Result<(ExecutionStatus, bool)>
    where
        F: FnMut(ApprovalRequest) -> Fut,
        Fut: Future<Output = ApprovalDecision>,
    {
        if policy.requires_manual_approval(&action) {
            if current_auto_approve_state {
                let report = self.execute_approved_action(action).await?;
                return Ok((ExecutionStatus::Executed(report), current_auto_approve_state));
            }

            let request = ApprovalRequest {
                action: action.clone(),
                reason: policy.approval_reason(&action),
            };
            match request_permission(request).await {
                ApprovalDecision::ApproveOnce => {
                    let report = self.execute_approved_action(action).await?;
                    Ok((ExecutionStatus::Executed(report), current_auto_approve_state))
                }
                ApprovalDecision::GrantTemporaryAccess => {
                    let report = self.execute_approved_action(action).await?;
                    Ok((ExecutionStatus::Executed(report), true))
                }
                ApprovalDecision::Deny => Ok((
                    ExecutionStatus::AuthorizationDenied {
                        action: action.clone(),
                        reason: format!("Authorization Denied for {:?}", action.action),
                    },
                    current_auto_approve_state,
                )),
            }
        } else {
            let report = self.execute_approved_action(action).await?;
            Ok((ExecutionStatus::Executed(report), current_auto_approve_state))
        }
    }

    /// Executes a full action list in sequence.
    ///
    /// If policy requires approval, the function returns early with first pending approval.
    #[allow(dead_code)]
    pub async fn execute_actions(
        &self,
        actions: Vec<AgentAction>,
        policy: ExecutionPolicy,
    ) -> Result<Vec<ExecutionStatus>> {
        let mut statuses = Vec::with_capacity(actions.len());

        for action in actions {
            let status = self.execute_action(action, policy).await?;
            let needs_approval = matches!(status, ExecutionStatus::AwaitingApproval(_));
            statuses.push(status);
            if needs_approval {
                break;
            }
        }

        Ok(statuses)
    }

    async fn execute_approved_action(&self, action: AgentAction) -> Result<ExecutionReport> {
        match action.action {
            ActionKind::RunCmd => {
                let command = required_non_empty(action.parameters.command.as_deref(), "command")?;
                self.execute_command(&command).await
            }
            ActionKind::CreateFolder => {
                let path = required_non_empty(action.parameters.path.as_deref(), "path")?;
                let full_path = self.resolve_folder_path(&path);
                timeout(
                    Duration::from_secs(FILE_ACTION_TIMEOUT_SECS),
                    tokio::fs::create_dir_all(&full_path),
                )
                .await
                .map_err(|_| anyhow!("create_folder timed out after {}s", FILE_ACTION_TIMEOUT_SECS))?
                .with_context(|| format!("failed to create folder {}", full_path.display()))?;

                Ok(ExecutionReport {
                    action: "create_folder".to_string(),
                    success: true,
                    stdout: format!("[System: Folder created] {}", full_path.display()),
                    stderr: String::new(),
                    exit_code: 0,
                    timed_out: false,
                    screenshot_png_bytes: None,
                    screenshot_source: None,
                    screenshot_width: None,
                    screenshot_height: None,
                })
            }
            ActionKind::CreateFile => {
                let path = required_non_empty(action.parameters.path.as_deref(), "path")?;
                let content = action.parameters.content.unwrap_or_default();
                let full_path = self.resolve_auxiliary_file_path(&path);

                ensure_parent_dir(&full_path).await?;
                timeout(
                    Duration::from_secs(FILE_ACTION_TIMEOUT_SECS),
                    tokio::fs::write(&full_path, content),
                )
                .await
                .map_err(|_| anyhow!("create_file timed out after {}s", FILE_ACTION_TIMEOUT_SECS))?
                .with_context(|| format!("failed to write file {}", full_path.display()))?;

                Ok(ExecutionReport {
                    action: "create_file".to_string(),
                    success: true,
                    stdout: format!("[System: File created] {}", full_path.display()),
                    stderr: String::new(),
                    exit_code: 0,
                    timed_out: false,
                    screenshot_png_bytes: None,
                    screenshot_source: None,
                    screenshot_width: None,
                    screenshot_height: None,
                })
            }
            ActionKind::EditFile => {
                let path = required_non_empty(action.parameters.path.as_deref(), "path")?;
                let mode = required_non_empty(action.parameters.mode.as_deref(), "mode")?;
                let content = action.parameters.content.unwrap_or_default();
                let full_path = self.resolve_auxiliary_file_path(&path);

                ensure_parent_dir(&full_path).await?;

                match mode.as_str() {
                    "overwrite" => {
                        timeout(
                            Duration::from_secs(FILE_ACTION_TIMEOUT_SECS),
                            tokio::fs::write(&full_path, content),
                        )
                        .await
                        .map_err(|_| anyhow!("edit_file timed out after {}s", FILE_ACTION_TIMEOUT_SECS))?
                        .with_context(|| {
                            format!("failed to overwrite file {}", full_path.display())
                        })?;
                    }
                    "append" => {
                        let mut file = timeout(
                            Duration::from_secs(FILE_ACTION_TIMEOUT_SECS),
                            tokio::fs::OpenOptions::new()
                                .create(true)
                                .append(true)
                                .open(&full_path),
                        )
                        .await
                        .map_err(|_| anyhow!("edit_file timed out after {}s", FILE_ACTION_TIMEOUT_SECS))?
                        .with_context(|| {
                            format!("failed to open file for append {}", full_path.display())
                        })?;
                        file.write_all(content.as_bytes()).await.with_context(|| {
                            format!("failed to append file {}", full_path.display())
                        })?;
                    }
                    _ => {
                        return Err(anyhow!(
                            "unsupported edit mode '{}'; expected overwrite or append",
                            mode
                        ));
                    }
                }

                Ok(ExecutionReport {
                    action: "edit_file".to_string(),
                    success: true,
                    stdout: format!("[System: File edited] {}", full_path.display()),
                    stderr: String::new(),
                    exit_code: 0,
                    timed_out: false,
                    screenshot_png_bytes: None,
                    screenshot_source: None,
                    screenshot_width: None,
                    screenshot_height: None,
                })
            }
            ActionKind::CreatePdf => {
                let path = required_non_empty(action.parameters.path.as_deref(), "path")?;
                let title = action
                    .parameters
                    .title
                    .unwrap_or_else(|| "Untitled".to_string());
                let content = action.parameters.content.unwrap_or_default();
                let full_path = self.resolve_export_path(&path);

                ensure_parent_dir(&full_path).await?;

                // Minimal valid PDF generator for text payloads.
                let pdf = build_minimal_pdf(&title, &content);
                tokio::fs::write(&full_path, pdf)
                    .await
                    .with_context(|| format!("failed to create pdf {}", full_path.display()))?;

                Ok(ExecutionReport {
                    action: "create_pdf".to_string(),
                    success: true,
                    stdout: format!("[System: PDF created] {}", full_path.display()),
                    stderr: String::new(),
                    exit_code: 0,
                    timed_out: false,
                    screenshot_png_bytes: None,
                    screenshot_source: None,
                    screenshot_width: None,
                    screenshot_height: None,
                })
            }
            ActionKind::GenerateDocument => {
                let format = required_non_empty(action.parameters.format.as_deref(), "format")?
                    .to_ascii_lowercase();
                if !SUPPORTED_DOCUMENT_FORMATS.contains(&format.as_str()) {
                    return Err(anyhow!(
                        "unsupported document format '{}'; expected pdf or docx",
                        format
                    ));
                }
                let path = required_non_empty(action.parameters.path.as_deref(), "path")?;
                let markdown_content = required_non_empty(
                    action.parameters.markdown_content.as_deref(),
                    "markdown_content",
                )?;
                let mut full_path = self.resolve_export_path(&path);
                if full_path.extension().and_then(|e| e.to_str()).is_none() {
                    full_path.set_extension(&format);
                }
                ensure_parent_dir(&full_path).await?;
                self.generate_document_with_pandoc(&full_path, &format, &markdown_content)
                    .await
            }
            ActionKind::SearchWeb => {
                let query = required_non_empty(action.parameters.query.as_deref(), "query")?;
                let web_engine = self
                    .web_engine
                    .as_ref()
                    .ok_or_else(|| {
                        anyhow!(
                            "web engine initialization failed: {}",
                            self.web_engine_init_error
                                .as_deref()
                                .unwrap_or("unknown initialization error")
                        )
                    })?;
                let results = web_engine.search_web(&query).await?;
                Ok(ExecutionReport {
                    action: "search_web".to_string(),
                    success: true,
                    stdout: format_search_results(&results),
                    stderr: String::new(),
                    exit_code: 0,
                    timed_out: false,
                    screenshot_png_bytes: None,
                    screenshot_source: None,
                    screenshot_width: None,
                    screenshot_height: None,
                })
            }
            ActionKind::ReadUrl => {
                let url = required_non_empty(action.parameters.url.as_deref(), "url")?;
                let web_engine = self
                    .web_engine
                    .as_ref()
                    .ok_or_else(|| {
                        anyhow!(
                            "web engine initialization failed: {}",
                            self.web_engine_init_error
                                .as_deref()
                                .unwrap_or("unknown initialization error")
                        )
                    })?;
                let content = web_engine.read_url(&url).await?;
                Ok(ExecutionReport {
                    action: "read_url".to_string(),
                    success: true,
                    stdout: format!("[Web] Read URL: {url}\n\n{content}"),
                    stderr: String::new(),
                    exit_code: 0,
                    timed_out: false,
                    screenshot_png_bytes: None,
                    screenshot_source: None,
                    screenshot_width: None,
                    screenshot_height: None,
                })
            }
            ActionKind::CaptureScreen => {
                self.capture_screen_now(action.parameters.target.as_deref()).await
            }
            ActionKind::RunAndObserve => {
                let command = required_non_empty(action.parameters.command.as_deref(), "command")?;
                let delay_secs = action.parameters.delay_secs.unwrap_or(3).max(1);
                let target = action
                    .parameters
                    .target
                    .clone()
                    .unwrap_or_else(|| "focused_window".to_string());
                self.run_and_observe(&command, delay_secs, &target).await
            }
            ActionKind::McpConnect => {
                let server_id = required_non_empty(action.parameters.server_id.as_deref(), "server_id")?;
                let command = required_non_empty(action.parameters.mcp_command.as_deref(), "mcp_command")?;
                let args = action.parameters.mcp_args.unwrap_or_default();
                let manager = self
                    .mcp_manager
                    .as_ref()
                    .ok_or_else(|| anyhow!("mcp client manager is unavailable"))?;
                let connected = manager
                    .connect(McpServerConfig {
                        id: server_id,
                        command,
                        args,
                    })
                    .await?;
                Ok(ExecutionReport {
                    action: "mcp_connect".to_string(),
                    success: true,
                    stdout: format!("[MCP] connected server_id={connected}"),
                    stderr: String::new(),
                    exit_code: 0,
                    timed_out: false,
                    screenshot_png_bytes: None,
                    screenshot_source: None,
                    screenshot_width: None,
                    screenshot_height: None,
                })
            }
            ActionKind::McpListTools => {
                let server_id = required_non_empty(action.parameters.server_id.as_deref(), "server_id")?;
                let manager = self
                    .mcp_manager
                    .as_ref()
                    .ok_or_else(|| anyhow!("mcp client manager is unavailable"))?;
                let tools = manager.list_tools(&server_id).await?;
                let mut out = format!("[MCP] tools for {server_id}:\n");
                if tools.is_empty() {
                    out.push_str("- (none)\n");
                } else {
                    for t in tools {
                        out.push_str(&format!("- {}: {}\n", t.name, t.description));
                    }
                }
                Ok(ExecutionReport {
                    action: "mcp_list_tools".to_string(),
                    success: true,
                    stdout: out.trim_end().to_string(),
                    stderr: String::new(),
                    exit_code: 0,
                    timed_out: false,
                    screenshot_png_bytes: None,
                    screenshot_source: None,
                    screenshot_width: None,
                    screenshot_height: None,
                })
            }
            ActionKind::McpCallTool => {
                let server_id = required_non_empty(action.parameters.server_id.as_deref(), "server_id")?;
                let tool = required_non_empty(action.parameters.tool.as_deref(), "tool")?;
                let manager = self
                    .mcp_manager
                    .as_ref()
                    .ok_or_else(|| anyhow!("mcp client manager is unavailable"))?;
                let output = manager
                    .call_tool(&server_id, &tool, action.parameters.arguments)
                    .await?;
                Ok(ExecutionReport {
                    action: "mcp_call_tool".to_string(),
                    success: true,
                    stdout: format!("[MCP] call_tool {server_id}/{tool}\n{output}"),
                    stderr: String::new(),
                    exit_code: 0,
                    timed_out: false,
                    screenshot_png_bytes: None,
                    screenshot_source: None,
                    screenshot_width: None,
                    screenshot_height: None,
                })
            }
            ActionKind::McpDisconnect => {
                let server_id = required_non_empty(action.parameters.server_id.as_deref(), "server_id")?;
                let manager = self
                    .mcp_manager
                    .as_ref()
                    .ok_or_else(|| anyhow!("mcp client manager is unavailable"))?;
                manager.disconnect(&server_id).await?;
                Ok(ExecutionReport {
                    action: "mcp_disconnect".to_string(),
                    success: true,
                    stdout: format!("[MCP] disconnected server_id={server_id}"),
                    stderr: String::new(),
                    exit_code: 0,
                    timed_out: false,
                    screenshot_png_bytes: None,
                    screenshot_source: None,
                    screenshot_width: None,
                    screenshot_height: None,
                })
            }
        }
    }

    async fn run_and_observe(
        &self,
        cmd: &str,
        delay_secs: u64,
        target: &str,
    ) -> Result<ExecutionReport> {
        let command_for_spawn = cmd.to_string();
        let workspace = self.workspace_root.clone();
        let spawn_join = tokio::spawn(async move {
            let mut command = if cfg!(target_os = "windows") {
                let mut c = Command::new("cmd");
                c.args(["/C", &command_for_spawn]);
                c
            } else {
                let mut c = Command::new("sh");
                c.args(["-c", &command_for_spawn]);
                c
            };
            command
                .current_dir(workspace)
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()
                .map(|_| ())
                .map_err(|e| e.to_string())
        });
        match spawn_join.await {
            Ok(Ok(())) => {}
            Ok(Err(err)) => return Err(anyhow!("failed to spawn background command: {err}")),
            Err(err) => return Err(anyhow!("background task join error: {err}")),
        }

        tokio::time::sleep(Duration::from_secs(delay_secs)).await;
        let capture = self.capture_screen_now(Some(target)).await?;

        Ok(ExecutionReport {
            action: "run_and_observe".to_string(),
            success: true,
            stdout: format!(
                "[run_and_observe] spawned command in background\n[run_and_observe] waited={}s\n[Screen] source={}\n[Screen] size={}x{}",
                delay_secs,
                capture
                    .screenshot_source
                    .as_deref()
                    .unwrap_or("focused_window"),
                capture.screenshot_width.unwrap_or(0),
                capture.screenshot_height.unwrap_or(0),
            ),
            stderr: String::new(),
            exit_code: 0,
            timed_out: false,
            screenshot_png_bytes: capture.screenshot_png_bytes,
            screenshot_source: capture.screenshot_source,
            screenshot_width: capture.screenshot_width,
            screenshot_height: capture.screenshot_height,
        })
    }

    async fn capture_screen_now(&self, target: Option<&str>) -> Result<ExecutionReport> {
        let target = target.unwrap_or("focused_window").to_string();
        let captured =
            tokio::task::spawn_blocking(move || screen_awareness::capture_from_target(Some(&target)))
                .await
                .map_err(|e| anyhow!("screen capture task join error: {e}"))??;
        let filename = screen_awareness::default_filename();
        let full_path = self.resolve_screenshot_path(&filename);
        ensure_parent_dir(&full_path).await?;
        timeout(
            Duration::from_secs(FILE_ACTION_TIMEOUT_SECS),
            tokio::fs::write(&full_path, &captured.png_bytes),
        )
        .await
        .map_err(|_| anyhow!("capture_screen timed out after {}s", FILE_ACTION_TIMEOUT_SECS))?
        .with_context(|| format!("failed to save screenshot {}", full_path.display()))?;

        Ok(ExecutionReport {
            action: "capture_screen".to_string(),
            success: true,
            stdout: format!(
                "[Screen] source={}\n[Screen] size={}x{}\n[Screen] saved={}\n[Screen] staged=true",
                captured.source,
                captured.width,
                captured.height,
                full_path.display(),
            ),
            stderr: String::new(),
            exit_code: 0,
            timed_out: false,
            screenshot_png_bytes: Some(captured.png_bytes),
            screenshot_source: Some(captured.source),
            screenshot_width: Some(captured.width),
            screenshot_height: Some(captured.height),
        })
    }

    async fn execute_command(&self, cmd: &str) -> Result<ExecutionReport> {
        let normalized_cmd = self.normalize_command_for_safe_execution(cmd);
        let mut command = if cfg!(target_os = "windows") {
            let mut c = Command::new("cmd");
            c.args(["/C", &normalized_cmd]);
            c
        } else {
            let mut c = Command::new("sh");
            c.args(["-c", &normalized_cmd]);
            c
        };

        command
            .current_dir(&self.workspace_root)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        let timeout_enabled = !should_skip_timeout_for_ui_app(cmd);
        let output = if timeout_enabled {
            match timeout(
                Duration::from_secs(TERMINAL_ACTION_TIMEOUT_SECS),
                command.output(),
            )
            .await
            {
                Ok(result) => result
                    .with_context(|| format!("failed to execute command: {normalized_cmd}"))?,
                Err(_) => {
                    return Ok(ExecutionReport {
                        action: "run_cmd".to_string(),
                        success: false,
                        stdout: String::new(),
                        stderr: format!(
                            "[timeout] command exceeded {}s: {}",
                            TERMINAL_ACTION_TIMEOUT_SECS, normalized_cmd
                        ),
                        exit_code: TIMEOUT_EXIT_CODE,
                        timed_out: true,
                        screenshot_png_bytes: None,
                        screenshot_source: None,
                        screenshot_width: None,
                        screenshot_height: None,
                    });
                }
            }
        } else {
            command
                .output()
                .await
                .with_context(|| format!("failed to execute command: {normalized_cmd}"))?
        };

        let mut stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let exit_code = output.status.code().unwrap_or(-1);

        if exit_code == 0 && stdout.trim().is_empty() {
            stdout = EMPTY_COMMAND_SUCCESS_MSG.to_string();
        }

        Ok(ExecutionReport {
            action: "run_cmd".to_string(),
            success: exit_code == 0,
            stdout,
            stderr,
            exit_code,
            timed_out: false,
            screenshot_png_bytes: None,
            screenshot_source: None,
            screenshot_width: None,
            screenshot_height: None,
        })
    }

    pub async fn execute_command_streaming<F>(
        &self,
        cmd: &str,
        mut on_chunk: F,
    ) -> Result<ExecutionReport>
    where
        F: FnMut(bool, String) + Send + 'static,
    {
        let normalized_cmd = self.normalize_command_for_safe_execution(cmd);
        let mut command = if cfg!(target_os = "windows") {
            let mut c = Command::new("cmd");
            c.args(["/C", &normalized_cmd]);
            c
        } else {
            let mut c = Command::new("sh");
            c.args(["-c", &normalized_cmd]);
            c
        };

        command
            .current_dir(&self.workspace_root)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        let mut child = command
            .spawn()
            .with_context(|| format!("failed to start command: {normalized_cmd}"))?;
        let stdout_pipe = child.stdout.take();
        let stderr_pipe = child.stderr.take();
        let (chunk_tx, mut chunk_rx) = mpsc::unbounded_channel::<(bool, String)>();
        let stdout_tx = chunk_tx.clone();
        let stderr_tx = chunk_tx.clone();

        let stdout_task = tokio::spawn(async move {
            let mut output = String::new();
            if let Some(stdout) = stdout_pipe {
                let mut reader = BufReader::new(stdout).lines();
                loop {
                    match reader.next_line().await {
                        Ok(Some(line)) => {
                            if let Err(e) = stdout_tx.send((true, line.clone())) {
                                eprintln!("failed to send stdout chunk to terminal UI channel: {e}");
                            }
                            output.push_str(&line);
                            output.push('\n');
                        }
                        Ok(None) => break,
                        Err(e) => {
                            let msg = format!("[stdout read error] {e}");
                            if let Err(e) = stdout_tx.send((false, msg.clone())) {
                                eprintln!("failed to send stdout read error to terminal UI channel: {e}");
                            }
                            output.push_str(&msg);
                            output.push('\n');
                            break;
                        }
                    }
                }
            }
            output
        });

        let stderr_task = tokio::spawn(async move {
            let mut output = String::new();
            if let Some(stderr) = stderr_pipe {
                let mut reader = BufReader::new(stderr).lines();
                loop {
                    match reader.next_line().await {
                        Ok(Some(line)) => {
                            if let Err(e) = stderr_tx.send((false, line.clone())) {
                                eprintln!("failed to send stderr chunk to terminal UI channel: {e}");
                            }
                            output.push_str(&line);
                            output.push('\n');
                        }
                        Ok(None) => break,
                        Err(e) => {
                            let msg = format!("[stderr read error] {e}");
                            if let Err(e) = stderr_tx.send((false, msg.clone())) {
                                eprintln!("failed to send stderr read error to terminal UI channel: {e}");
                            }
                            output.push_str(&msg);
                            output.push('\n');
                            break;
                        }
                    }
                }
            }
            output
        });
        drop(chunk_tx);

        let timeout_enabled = !should_skip_timeout_for_ui_app(&normalized_cmd);
        let status_task = tokio::spawn(async move {
            wait_for_child_with_optional_timeout(&mut child, timeout_enabled).await
        });

        while let Some((is_stdout, line)) = chunk_rx.recv().await {
            on_chunk(is_stdout, line);
        }

        let status = status_task.await??;

        let mut stdout = stdout_task.await.unwrap_or_default();
        let stderr = stderr_task.await.unwrap_or_default();
        let timed_out = status.1;
        let exit_code = if timed_out {
            TIMEOUT_EXIT_CODE
        } else {
            status.0.code().unwrap_or(-1)
        };

        if exit_code == 0 && stdout.trim().is_empty() {
            stdout = EMPTY_COMMAND_SUCCESS_MSG.to_string();
        }
        let stderr = if timed_out {
            format!(
                "{}\n[timeout] command exceeded {}s: {}",
                stderr, TERMINAL_ACTION_TIMEOUT_SECS, normalized_cmd
            )
            .trim()
            .to_string()
        } else {
            stderr
        };

        Ok(ExecutionReport {
            action: "run_cmd".to_string(),
            success: exit_code == 0,
            stdout,
            stderr,
            exit_code,
            timed_out,
            screenshot_png_bytes: None,
            screenshot_source: None,
            screenshot_width: None,
            screenshot_height: None,
        })
    }

    fn resolve_folder_path(&self, raw: &str) -> PathBuf {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return self.workspace_root.clone();
        }
        if Path::new(trimmed).is_absolute() {
            return PathBuf::from(trimmed);
        }
        let normalized = normalize_rel_path(trimmed);
        if is_top_level_workspace_dir(&normalized) {
            return self.workspace_root.join(normalized);
        }
        self.workspace_root.join(SCRIPTS_DIR).join(normalized)
    }

    fn resolve_export_path(&self, raw: &str) -> PathBuf {
        self.resolve_into_named_bucket(raw, EXPORTS_DIR)
    }

    fn resolve_screenshot_path(&self, raw: &str) -> PathBuf {
        self.resolve_into_named_bucket(raw, SCREENSHOTS_DIR)
    }

    fn resolve_auxiliary_file_path(&self, raw: &str) -> PathBuf {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return self.workspace_root.join(SCRIPTS_DIR);
        }
        if Path::new(trimmed).is_absolute() {
            return PathBuf::from(trimmed);
        }
        let normalized = normalize_rel_path(trimmed);
        if is_allowed_bucketed_path(&normalized) {
            return self.workspace_root.join(normalized);
        }
        if is_core_root_file(&normalized) {
            return self.workspace_root.join(normalized);
        }
        match classify_auxiliary_path(&normalized) {
            BucketKind::Scripts => self.workspace_root.join(SCRIPTS_DIR).join(normalized),
            BucketKind::Exports => self.workspace_root.join(EXPORTS_DIR).join(normalized),
            BucketKind::Screenshots => self.workspace_root.join(SCREENSHOTS_DIR).join(normalized),
            BucketKind::Logs => self.workspace_root.join(LOGS_DIR).join(normalized),
        }
    }

    fn resolve_into_named_bucket(&self, raw: &str, bucket: &str) -> PathBuf {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return self.workspace_root.join(bucket);
        }
        if Path::new(trimmed).is_absolute() {
            return PathBuf::from(trimmed);
        }
        let normalized = normalize_rel_path(trimmed);
        if is_allowed_bucketed_path(&normalized) || is_core_root_file(&normalized) {
            return self.workspace_root.join(normalized);
        }
        self.workspace_root.join(bucket).join(normalized)
    }

    fn normalize_command_for_safe_execution(&self, cmd: &str) -> String {
        let maybe_non_interactive = make_install_command_non_interactive(cmd);
        self.normalize_generated_script_command(&maybe_non_interactive)
    }

    fn normalize_generated_script_command(&self, cmd: &str) -> String {
        let trimmed = cmd.trim();
        if trimmed.is_empty() {
            return trimmed.to_string();
        }
        let mut parts = trimmed.split_whitespace();
        let Some(first) = parts.next() else {
            return trimmed.to_string();
        };
        let first_l = first.to_ascii_lowercase();
        let rest: Vec<&str> = parts.collect();
        let script_runtimes = [
            "python", "python3", "bash", "sh", "node", "ruby", "perl", "php",
        ];
        if script_runtimes.contains(&first_l.as_str()) && !rest.is_empty() {
            let script = rest[0];
            if should_rewrite_to_scripts(script) {
                let mut rebuilt = vec![
                    first.to_string(),
                    format!("{SCRIPTS_DIR}/{}", script.trim_start_matches("./")),
                ];
                for arg in rest.iter().skip(1) {
                    rebuilt.push((*arg).to_string());
                }
                return rebuilt.join(" ");
            }
            return trimmed.to_string();
        }
        if should_rewrite_to_scripts(first) {
            let mut rebuilt = vec![format!("{SCRIPTS_DIR}/{}", first.trim_start_matches("./"))];
            for arg in rest {
                rebuilt.push(arg.to_string());
            }
            return rebuilt.join(" ");
        }
        trimmed.to_string()
    }

    async fn generate_document_with_pandoc(
        &self,
        output_path: &Path,
        format: &str,
        markdown_content: &str,
    ) -> Result<ExecutionReport> {
        let temp_name = format!(
            ".ai_os_doc_{}_{}.md",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis()
        );
        let temp_md_path = self.workspace_root.join(temp_name);
        tokio::fs::write(&temp_md_path, markdown_content)
            .await
            .with_context(|| format!("failed writing temporary markdown {}", temp_md_path.display()))?;

        let mut cmd = Command::new("pandoc");
        cmd.arg(&temp_md_path)
            .arg("-o")
            .arg(output_path)
            .arg("--from")
            .arg("markdown")
            .arg("--to")
            .arg(format)
            .current_dir(&self.workspace_root)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        let output = cmd.output().await;
        let _ = tokio::fs::remove_file(&temp_md_path).await;

        match output {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                let exit_code = output.status.code().unwrap_or(-1);
                if output.status.success() {
                    Ok(ExecutionReport {
                        action: "generate_document".to_string(),
                        success: true,
                        stdout: format!(
                            "[System: Document generated] {} ({})",
                            output_path.display(),
                            format
                        ),
                        stderr,
                        exit_code,
                        timed_out: false,
                        screenshot_png_bytes: None,
                        screenshot_source: None,
                        screenshot_width: None,
                        screenshot_height: None,
                    })
                } else {
                    Ok(ExecutionReport {
                        action: "generate_document".to_string(),
                        success: false,
                        stdout,
                        stderr: format!("pandoc failed (exit={exit_code}): {stderr}"),
                        exit_code,
                        timed_out: false,
                        screenshot_png_bytes: None,
                        screenshot_source: None,
                        screenshot_width: None,
                        screenshot_height: None,
                    })
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(ExecutionReport {
                action: "generate_document".to_string(),
                success: false,
                stdout: String::new(),
                stderr: "pandoc is not installed. Install pandoc and retry document generation."
                    .to_string(),
                exit_code: 127,
                timed_out: false,
                screenshot_png_bytes: None,
                screenshot_source: None,
                screenshot_width: None,
                screenshot_height: None,
            }),
            Err(e) => Err(anyhow!(e)).with_context(|| {
                format!(
                    "failed to execute pandoc for document generation at {}",
                    output_path.display()
                )
            }),
        }
    }
}

fn should_skip_timeout_for_ui_app(cmd: &str) -> bool {
    let normalized = cmd.to_ascii_lowercase();
    let ui_indicators = [
        " ui.py",
        "python ui.py",
        "python3 ui.py",
        "streamlit run",
        "gradio",
        "uvicorn",
        "flask run",
        "npm run dev",
    ];
    ui_indicators.iter().any(|pat| normalized.contains(pat))
}

fn make_install_command_non_interactive(cmd: &str) -> String {
    let trimmed = cmd.trim();
    if trimmed.is_empty() {
        return trimmed.to_string();
    }
    let lower = trimmed.to_ascii_lowercase();

    if is_pacman_install_command(trimmed) {
        let mut rebuilt = trimmed.to_string();
        if !contains_flag(trimmed, "--noconfirm") {
            rebuilt.push_str(" --noconfirm");
        }
        if !contains_flag(trimmed, "--needed") {
            rebuilt.push_str(" --needed");
        }
        return rebuilt;
    }

    if (lower.contains("apt-get install")
        || lower.contains("apt install")
        || lower.contains("dnf install")
        || lower.contains("yum install")
        || lower.contains("zypper install"))
        && !contains_flag(trimmed, "-y")
        && !contains_flag(trimmed, "--yes")
    {
        return format!("{trimmed} -y");
    }

    if lower.starts_with("npx ") && !contains_flag(trimmed, "--yes") {
        return format!("{trimmed} --yes");
    }

    trimmed.to_string()
}

fn contains_flag(cmd: &str, flag: &str) -> bool {
    cmd.split_whitespace().any(|part| part == flag)
}

fn is_pacman_install_command(cmd: &str) -> bool {
    let parts: Vec<&str> = cmd.split_whitespace().collect();
    if parts.is_empty() {
        return false;
    }
    let idx = if parts[0] == "sudo" {
        if parts.len() < 2 {
            return false;
        }
        1
    } else {
        0
    };
    let has_pacman_bin = parts.get(idx).is_some_and(|p| *p == "pacman");
    let has_sync_flag = parts
        .iter()
        .skip(idx + 1)
        .any(|arg| arg.eq_ignore_ascii_case("--sync") || is_pacman_short_sync_flag(arg));
    has_pacman_bin && has_sync_flag
}

fn is_pacman_short_sync_flag(arg: &str) -> bool {
    if !arg.starts_with("-S") {
        return false;
    }
    // Exclude query/list flags that are not install/upgrade operations.
    !(arg.starts_with("-Ss") || arg.starts_with("-Si") || arg.starts_with("-Sl"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BucketKind {
    Scripts,
    Exports,
    Screenshots,
    Logs,
}

fn normalize_rel_path(raw: &str) -> String {
    raw.replace('\\', "/").trim_start_matches("./").to_string()
}

fn is_top_level_workspace_dir(path: &str) -> bool {
    matches!(
        path,
        "src" | "tests" | "benches" | "examples" | "docs" | "assets" | "scripts" | "exports" | "logs" | ".github"
    ) || path.starts_with("src/")
        || path.starts_with("tests/")
        || path.starts_with("benches/")
        || path.starts_with("examples/")
        || path.starts_with("docs/")
        || path.starts_with(".github/")
}

fn is_allowed_bucketed_path(path: &str) -> bool {
    path.starts_with("scripts/")
        || path.starts_with("exports/")
        || path.starts_with("assets/screenshots/")
        || path.starts_with("logs/")
}

fn is_core_root_file(path: &str) -> bool {
    if path.contains('/') {
        return false;
    }
    let lower = path.to_ascii_lowercase();
    CORE_ROOT_FILENAMES.contains(&lower.as_str())
}

fn classify_auxiliary_path(path: &str) -> BucketKind {
    let lower = path.to_ascii_lowercase();
    if lower.ends_with(".png")
        || lower.ends_with(".jpg")
        || lower.ends_with(".jpeg")
        || lower.ends_with(".webp")
        || lower.ends_with(".bmp")
    {
        return BucketKind::Screenshots;
    }
    if lower.ends_with(".log") || lower.ends_with(".trace") || lower.ends_with(".out") {
        return BucketKind::Logs;
    }
    if lower.ends_with(".pdf")
        || lower.ends_with(".docx")
        || lower.ends_with(".md")
        || lower.ends_with(".txt")
        || lower.ends_with(".csv")
        || lower.ends_with(".json")
    {
        return BucketKind::Exports;
    }
    if lower.ends_with(".py")
        || lower.ends_with(".sh")
        || lower.ends_with(".bash")
        || lower.ends_with(".zsh")
        || lower.ends_with(".js")
        || lower.ends_with(".ts")
        || lower.ends_with(".sql")
    {
        return BucketKind::Scripts;
    }
    BucketKind::Scripts
}

fn should_rewrite_to_scripts(script_arg: &str) -> bool {
    let normalized = normalize_rel_path(script_arg);
    if normalized.is_empty()
        || normalized.starts_with('/')
        || normalized.starts_with("scripts/")
        || normalized.starts_with("src/")
        || normalized.starts_with("tests/")
        || normalized.starts_with("examples/")
        || normalized.starts_with("benches/")
        || normalized.contains('/')
    {
        return false;
    }
    let lower = normalized.to_ascii_lowercase();
    lower.ends_with(".py")
        || lower.ends_with(".sh")
        || lower.ends_with(".bash")
        || lower.ends_with(".zsh")
        || lower.ends_with(".js")
        || lower.ends_with(".ts")
        || lower.ends_with(".sql")
}

async fn wait_for_child_with_optional_timeout(
    child: &mut Child,
    timeout_enabled: bool,
) -> Result<(std::process::ExitStatus, bool)> {
    if timeout_enabled {
        match timeout(
            Duration::from_secs(TERMINAL_ACTION_TIMEOUT_SECS),
            child.wait(),
        )
        .await
        {
            Ok(wait_result) => Ok((
                wait_result.context("failed waiting for command process")?,
                false,
            )),
            Err(_) => {
                let _ = child.kill().await;
                let status = child
                    .wait()
                    .await
                    .context("failed waiting for timed out process after kill")?;
                Ok((status, true))
            }
        }
    } else {
        let status = child
            .wait()
            .await
            .context("failed waiting for command process")?;
        Ok((status, false))
    }
}

fn required_non_empty(value: Option<&str>, field: &str) -> Result<String> {
    let Some(value) = value else {
        return Err(anyhow!("missing required field: {field}"));
    };
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(anyhow!("field '{field}' cannot be empty"));
    }
    Ok(trimmed.to_string())
}

async fn ensure_parent_dir(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .with_context(|| format!("failed to create parent directory {}", parent.display()))?;
    }
    Ok(())
}

fn build_minimal_pdf(title: &str, content: &str) -> Vec<u8> {
    let escaped_title = escape_pdf_text(title);
    let escaped_content = escape_pdf_text(content);

    let mut pdf = String::new();
    pdf.push_str("%PDF-1.4\n");

    let mut offsets = Vec::new();
    offsets.push(0usize);

    offsets.push(pdf.len());
    pdf.push_str("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

    offsets.push(pdf.len());
    pdf.push_str("2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");

    offsets.push(pdf.len());
    pdf.push_str("3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n");

    let stream = format!(
        "BT /F1 12 Tf 72 740 Td ({}) Tj 0 -24 Td ({}) Tj ET",
        escaped_title, escaped_content
    );

    offsets.push(pdf.len());
    pdf.push_str(&format!(
        "4 0 obj\n<< /Length {} >>\nstream\n{}\nendstream\nendobj\n",
        stream.len(),
        stream
    ));

    offsets.push(pdf.len());
    pdf.push_str("5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n");

    let xref_start = pdf.len();
    pdf.push_str("xref\n0 6\n");
    pdf.push_str("0000000000 65535 f \n");
    for off in offsets.into_iter().skip(1) {
        // PDF xref entries are fixed-width records:
        // 10-digit offset + space + 5-digit generation + space + in-use flag + space + '\n'
        // which totals 20 bytes per line.
        pdf.push_str(&format!("{off:010} 00000 n \n"));
    }
    pdf.push_str("trailer\n<< /Size 6 /Root 1 0 R >>\n");
    pdf.push_str(&format!("startxref\n{}\n%%EOF\n", xref_start));

    pdf.into_bytes()
}

#[cfg(test)]
mod tests {
    use super::{is_pacman_install_command, make_install_command_non_interactive};

    #[test]
    fn keeps_empty_command_unchanged() {
        assert_eq!(make_install_command_non_interactive("   "), "");
    }

    #[test]
    fn appends_pacman_noninteractive_flags() {
        let cmd = "sudo pacman -S pandoc";
        let normalized = make_install_command_non_interactive(cmd);
        assert!(normalized.contains("--noconfirm"));
        assert!(normalized.contains("--needed"));
    }

    #[test]
    fn preserves_existing_pacman_flags() {
        let cmd = "pacman -S --noconfirm --needed nodejs";
        let normalized = make_install_command_non_interactive(cmd);
        assert_eq!(normalized, cmd);
    }

    #[test]
    fn appends_apt_yes_flag() {
        let cmd = "sudo apt-get install ripgrep";
        assert_eq!(
            make_install_command_non_interactive(cmd),
            "sudo apt-get install ripgrep -y"
        );
    }

    #[test]
    fn appends_npx_yes_flag() {
        let cmd = "npx @modelcontextprotocol/server-everything";
        assert_eq!(
            make_install_command_non_interactive(cmd),
            "npx @modelcontextprotocol/server-everything --yes"
        );
    }

    #[test]
    fn does_not_duplicate_npx_yes_flag() {
        let cmd = "npx --yes @modelcontextprotocol/server-everything";
        assert_eq!(make_install_command_non_interactive(cmd), cmd);
    }

    #[test]
    fn pacman_install_detection_supports_common_forms() {
        assert!(is_pacman_install_command("pacman -S pandoc"));
        assert!(is_pacman_install_command("sudo pacman -Syu"));
        assert!(is_pacman_install_command("pacman --sync python"));
        assert!(!is_pacman_install_command("pacman -Ss pandoc"));
        assert!(!is_pacman_install_command("pacman -h"));
    }
}

fn escape_pdf_text(input: &str) -> String {
    input
        .replace('\\', "\\\\")
        .replace('(', "\\(")
        .replace(')', "\\)")
        .replace('\n', " ")
}

fn action_requires_delete_safety(action: &AgentAction) -> bool {
    if matches!(action.action, ActionKind::RunCmd | ActionKind::RunAndObserve) {
        let cmd = action
            .parameters
            .command
            .as_deref()
            .unwrap_or_default()
            .to_ascii_lowercase();
        let mut tokens = cmd.split_whitespace();
        let first = tokens.next().unwrap_or_default();
        if first == "rm" || first == "del" || first == "rmdir" {
            return true;
        }
        return std::iter::once(first)
            .chain(tokens)
            .any(|tok| tok == "delete_file");
    }
    false
}
