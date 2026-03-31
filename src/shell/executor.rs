use anyhow::Result;
use std::process::Command;

pub struct CommandOutput {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
}

pub fn execute_command(cmd: &str, working_dir: &str) -> Result<CommandOutput> {
    let output = if cfg!(target_os = "windows") {
        Command::new("cmd")
            .args(["/C", cmd])
            .current_dir(working_dir)
            .output()?
    } else {
        Command::new("sh")
            .args(["-c", cmd])
            .current_dir(working_dir)
            .output()?
    };

    Ok(CommandOutput {
        stdout: String::from_utf8_lossy(&output.stdout).to_string(),
        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
        exit_code: output.status.code().unwrap_or(-1),
    })
}
