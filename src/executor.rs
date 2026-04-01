use std::path::{Path, PathBuf};
use std::future::Future;

use anyhow::{anyhow, Context, Result};
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

use crate::parser::{ActionKind, AgentAction};
use crate::web_engine::{format_search_results, WebEngine};

const EMPTY_COMMAND_SUCCESS_MSG: &str = "[System: Command executed successfully with no output]";

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
            ExecutionPolicy::ReadEdit => matches!(action.action, ActionKind::RunCmd),
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
}

/// Enterprise-grade async action executor.
///
/// - Uses `tokio::process::Command` for terminal operations.
/// - Uses `tokio::fs` for file/folder operations.
/// - Explicitly handles auto-approval workflow.
/// - Normalizes empty successful command output.
pub struct ActionExecutor {
    workspace_root: PathBuf,
    web_engine: WebEngine,
}

impl ActionExecutor {
    pub fn new(workspace_root: impl Into<PathBuf>) -> Self {
        let web_engine = WebEngine::new().expect("web engine init must succeed");
        Self {
            workspace_root: workspace_root.into(),
            web_engine,
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
                let full_path = self.resolve_path(&path);
                tokio::fs::create_dir_all(&full_path)
                    .await
                    .with_context(|| format!("failed to create folder {}", full_path.display()))?;

                Ok(ExecutionReport {
                    action: "create_folder".to_string(),
                    success: true,
                    stdout: format!("[System: Folder created] {}", full_path.display()),
                    stderr: String::new(),
                    exit_code: 0,
                })
            }
            ActionKind::CreateFile => {
                let path = required_non_empty(action.parameters.path.as_deref(), "path")?;
                let content = action.parameters.content.unwrap_or_default();
                let full_path = self.resolve_path(&path);

                ensure_parent_dir(&full_path).await?;
                tokio::fs::write(&full_path, content)
                    .await
                    .with_context(|| format!("failed to write file {}", full_path.display()))?;

                Ok(ExecutionReport {
                    action: "create_file".to_string(),
                    success: true,
                    stdout: format!("[System: File created] {}", full_path.display()),
                    stderr: String::new(),
                    exit_code: 0,
                })
            }
            ActionKind::EditFile => {
                let path = required_non_empty(action.parameters.path.as_deref(), "path")?;
                let mode = required_non_empty(action.parameters.mode.as_deref(), "mode")?;
                let content = action.parameters.content.unwrap_or_default();
                let full_path = self.resolve_path(&path);

                ensure_parent_dir(&full_path).await?;

                match mode.as_str() {
                    "overwrite" => {
                        tokio::fs::write(&full_path, content)
                            .await
                            .with_context(|| {
                                format!("failed to overwrite file {}", full_path.display())
                            })?;
                    }
                    "append" => {
                        let mut file = tokio::fs::OpenOptions::new()
                            .create(true)
                            .append(true)
                            .open(&full_path)
                            .await
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
                })
            }
            ActionKind::CreatePdf => {
                let path = required_non_empty(action.parameters.path.as_deref(), "path")?;
                let title = action
                    .parameters
                    .title
                    .unwrap_or_else(|| "Untitled".to_string());
                let content = action.parameters.content.unwrap_or_default();
                let full_path = self.resolve_path(&path);

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
                })
            }
            ActionKind::SearchWeb => {
                let query = required_non_empty(action.parameters.query.as_deref(), "query")?;
                let results = self.web_engine.search_web(&query).await?;
                Ok(ExecutionReport {
                    action: "search_web".to_string(),
                    success: true,
                    stdout: format_search_results(&results),
                    stderr: String::new(),
                    exit_code: 0,
                })
            }
            ActionKind::ReadUrl => {
                let url = required_non_empty(action.parameters.url.as_deref(), "url")?;
                let content = self.web_engine.read_url(&url).await?;
                Ok(ExecutionReport {
                    action: "read_url".to_string(),
                    success: true,
                    stdout: format!("[Web] Read URL: {url}\n\n{content}"),
                    stderr: String::new(),
                    exit_code: 0,
                })
            }
        }
    }

    async fn execute_command(&self, cmd: &str) -> Result<ExecutionReport> {
        let mut command = if cfg!(target_os = "windows") {
            let mut c = Command::new("cmd");
            c.args(["/C", cmd]);
            c
        } else {
            let mut c = Command::new("sh");
            c.args(["-c", cmd]);
            c
        };

        command
            .current_dir(&self.workspace_root)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        let output = command
            .output()
            .await
            .with_context(|| format!("failed to execute command: {cmd}"))?;

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
        })
    }

    fn resolve_path(&self, raw: &str) -> PathBuf {
        let path = Path::new(raw);
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.workspace_root.join(path)
        }
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

fn escape_pdf_text(input: &str) -> String {
    input
        .replace('\\', "\\\\")
        .replace('(', "\\(")
        .replace(')', "\\)")
        .replace('\n', " ")
}

fn action_requires_delete_safety(action: &AgentAction) -> bool {
    if let ActionKind::RunCmd = action.action {
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
