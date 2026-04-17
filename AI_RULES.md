# Nexus-Prime AI-OS: Core Directives

## 1. Tech Stack & Async Integrity
- Language: Rust. GUI: `egui`. Async Runtime: `tokio`. Vector DB: `qdrant`.
- Rule: NEVER block the async runtime. Wrap blocking I/O (file system, heavy parsing) in `tokio::task::spawn_blocking`.
- Rule: Do not use `std::sync::Mutex` in async contexts; use `tokio::sync::Mutex` or `parking_lot`.

## 2. OS-Awareness & Dependency Management
- Rule: Always dynamically check `std::env::consts::OS`.
- Linux (Manjaro/Arch): Use `sudo pacman -S --noconfirm` or `pamac` for missing packages (e.g., `base-devel`, `npm`).
- Windows: Use `winget` with `--silent --force --accept-package-agreements --accept-source-agreements`.
- Requirement: Autonomously verify and install missing tools before failing.

## 3. Tool Selection & MCP Protocol
- Standard CLI: Use the terminal execution tool (`run_cmd`) for standard system tools (winget, pacman, python).
- MCP Servers: Use `mcp_connect` ONLY for Model Context Protocol servers.
- SQLite Protocol: Use `npx -y mcp-server-sqlite --db <path>`. DO NOT write Python scripts for database reads. Render all structured output as Markdown Tables.

## 4. UI & Terminal Twinning (Glass-Box)
- Execution Visibility: All executed commands MUST stream their stdout/stderr live to the 'Terminals' UI tab using `tokio::process::Command` and piped `AsyncRead`. No freezing UI or "Running..." deadlock states.
- UI Layout constraints: Chat messages must be wrapped in `ScrollArea::vertical().stick_to_bottom(true)`. The Vision Staging panel must have a strict `max_height` and strictly not overlap inputs.

## 5. Self-Healing & Multilingual NLU
- Healing Loop: If a command fails (e.g., npm build error, 404 package), do NOT just report the error to the user. Autonomously search the web for the fix, install required dependencies, and retry the execution.
- NLU Protocol: If the user speaks a non-English language (e.g., Turkish), internally translate the technical intent to English for tool execution to prevent syntax issues, but ALWAYS reply to the user in their native language.

---
*For ALL future prompts and tasks in this project, silently read this file as primary context before generating any code or executing any action.*
