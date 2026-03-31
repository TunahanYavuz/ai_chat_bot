# ai_chat_bot

Desktop AI chat client built with Rust + egui.

## 1) Before launch (requirements)

### Required
- Rust toolchain (stable) with Cargo
- Internet connection (for cloud model APIs)
- An API key from one of:
  - OpenAI
  - NVIDIA NIM
  - OpenRouter
  - Hugging Face

### Linux notes
If your distro is missing GUI/system libraries, install the common desktop deps first (Wayland/X11/GTK and OpenSSL dev packages).

## 2) Build and run

From the project root:

```bash
cargo check
cargo run
```

The app window should open after compilation.

## 3) First execution (setup wizard)

On first launch, a 6-step setup wizard is shown.

### Step 1 — Welcome
- Click **Next** to begin.

### Step 2 — Provider selection
- Choose your provider: **OpenAI / NVIDIA NIM / OpenRouter / Hugging Face / Custom**.
- Click **Next**.

### Step 3 — API credentials
- Paste your **API key**.
- Confirm/edit **Base URL** (auto-filled by provider).
- Click **Next**.

### Step 4 — Default model
- Pick the model you want to use by default.
- Click **Next**.

### Step 5 — Workspace options
- Choose **Working Directory** (folder AI can read; file writes require approval).
- Toggle **Shell Execution** only if you want command execution (still approval-gated).
- Click **Next**.

### Step 6 — Finish
- Review your configuration.
- Click **Finish** to save setup and enter the main chat UI.

## 4) Basic usage after setup

- Click **+ New Chat** to start a conversation.
- Pick model and thinking mode from the sidebar.
- Use **Attach** to include PDF/DOCX/images/text files.
- Click image attachments in messages to download.
- Use **Settings** to change provider, API key, base URL, database path, and theme.
- Use **Shadow rollback** to restore file snapshots if needed.

## 5) Notes

- Settings are stored in your user config directory.
- Conversations and attachments are stored in SQLite.
- API keys are never hardcoded; they are loaded from saved settings.
