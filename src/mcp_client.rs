use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use rust_mcp_sdk::mcp_client::{ClientHandler, ClientRuntime, McpClientOptions, client_runtime};
use rust_mcp_sdk::schema::{
    CallToolRequestParams, ClientCapabilities, ContentBlock, Implementation, InitializeRequestParams,
    LATEST_PROTOCOL_VERSION,
};
use rust_mcp_sdk::{McpClient, StdioTransport, ToMcpClientHandler, TransportOptions};
use serde_json::{Map, Value};
use tokio::sync::Mutex;

#[derive(Clone, Debug)]
pub struct McpServerConfig {
    pub id: String,
    pub command: String,
    pub args: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct McpToolInfo {
    pub name: String,
    pub description: String,
    pub llm_tool_schema: Value,
}

#[derive(Default)]
struct EmptyClientHandler;

#[async_trait::async_trait]
impl ClientHandler for EmptyClientHandler {}

#[derive(Default)]
pub struct McpManager {
    sessions: Mutex<HashMap<String, Arc<ClientRuntime>>>,
    cached_tools: Mutex<HashMap<String, Vec<McpToolInfo>>>,
}

impl McpManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn connect(&self, config: McpServerConfig) -> Result<String> {
        if config.command.trim().is_empty() {
            return Err(anyhow!("mcp launch command cannot be empty"));
        }
        let transport = StdioTransport::create_with_server_launch(
            config.command.trim(),
            config.args.clone(),
            None,
            TransportOptions::default(),
        )
        .map_err(|e| anyhow!("failed to create MCP stdio transport: {e:?}"))?;

        let client_details = InitializeRequestParams {
            capabilities: ClientCapabilities::default(),
            client_info: Implementation {
                name: "ai-chat-bot-mcp-client".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
                title: Some("AI Chat Bot MCP Client".to_string()),
                description: Some("Embedded MCP client runtime".to_string()),
                icons: vec![],
                website_url: None,
            },
            protocol_version: LATEST_PROTOCOL_VERSION.into(),
            meta: None,
        };

        let handler = EmptyClientHandler;
        let client = client_runtime::create_client(McpClientOptions {
            client_details,
            transport,
            handler: handler.to_mcp_client_handler(),
            task_store: None,
            server_task_store: None,
            message_observer: None,
        });
        client
            .clone()
            .start()
            .await
            .map_err(|e| anyhow!("failed to start MCP client runtime: {e:?}"))?;

        let mut sessions = self.sessions.lock().await;
        sessions.insert(config.id.clone(), client);
        drop(sessions);
        let _ = self.refresh_tools(&config.id).await;
        Ok(config.id)
    }

    pub async fn disconnect(&self, server_id: &str) -> Result<()> {
        let client = {
            let mut sessions = self.sessions.lock().await;
            sessions.remove(server_id)
        };
        if let Some(client) = client {
            client
                .shut_down()
                .await
                .map_err(|e| anyhow!("failed while shutting down MCP client: {e:?}"))?;
            self.cached_tools.lock().await.remove(server_id);
            Ok(())
        } else {
            Err(anyhow!("mcp server '{server_id}' is not connected"))
        }
    }

    pub async fn list_tools(&self, server_id: &str) -> Result<Vec<McpToolInfo>> {
        self.refresh_tools(server_id).await?;
        Ok(self
            .cached_tools
            .lock()
            .await
            .get(server_id)
            .cloned()
            .unwrap_or_default())
    }

    pub async fn mcp_tool_prompt_context(&self) -> String {
        let cache = self.cached_tools.lock().await;
        if cache.is_empty() {
            return "MCP TOOL REGISTRY:\n- No connected MCP tools discovered yet.".to_string();
        }
        let mut server_ids: Vec<String> = cache.keys().cloned().collect();
        server_ids.sort();
        let mut out = String::from(
            "MCP TOOL REGISTRY (dynamic):\nUse these native MCP tools instead of writing scripts.\n",
        );
        for server_id in server_ids {
            let mut tools = cache.get(&server_id).cloned().unwrap_or_default();
            tools.sort_by(|a, b| a.name.cmp(&b.name));
            if tools.is_empty() {
                out.push_str(&format!("- server_id={server_id}: no tools\n"));
                continue;
            }
            out.push_str(&format!("- server_id={server_id}\n"));
            for tool in tools {
                out.push_str(&format!(
                    "  - MCP tool '{}': {}\n    LLM tool schema: {}\n    Invocation JSON action: {{\"action\":\"mcp_call_tool\",\"parameters\":{{\"server_id\":\"{}\",\"tool\":\"{}\",\"arguments\":<JSON object following input schema>}}}}\n",
                    tool.name,
                    tool.description,
                    tool.llm_tool_schema,
                    server_id,
                    tool.name
                ));
            }
        }
        out
    }

    async fn refresh_tools(&self, server_id: &str) -> Result<Vec<McpToolInfo>> {
        let client = self.client(server_id).await?;
        let listed = client
            .request_tool_list(None)
            .await
            .map_err(|e| anyhow!("failed to list tools for '{server_id}': {e:?}"))?;
        let tools: Vec<McpToolInfo> = listed
            .tools
            .into_iter()
            .map(|tool| {
                let schema = serde_json::to_value(&tool.input_schema).unwrap_or_else(|_| {
                    serde_json::json!({
                        "type": "object",
                        "properties": {},
                        "additionalProperties": true
                    })
                });
                let description = tool.description.unwrap_or_default();
                McpToolInfo {
                    llm_tool_schema: serde_json::json!({
                        "type": "function",
                        "function": {
                            "name": format!("mcp_{}__{}", sanitize_identifier(server_id), sanitize_identifier(&tool.name)),
                            "description": if description.trim().is_empty() {
                                format!("MCP tool '{}' on server '{}'", tool.name, server_id)
                            } else {
                                format!("MCP tool '{}' on server '{}': {}", tool.name, server_id, description)
                            },
                            "parameters": schema.clone()
                        }
                    }),
                    name: tool.name,
                    description,
                }
            })
            .collect();
        self.cached_tools
            .lock()
            .await
            .insert(server_id.to_string(), tools.clone());
        Ok(tools)
    }

    pub async fn call_tool(
        &self,
        server_id: &str,
        tool: &str,
        arguments: Option<Value>,
    ) -> Result<String> {
        let client = self.client(server_id).await?;
        let arguments = normalize_arguments(arguments)?;
        let result = client
            .request_tool_call(CallToolRequestParams {
                name: tool.to_string(),
                arguments,
                meta: None,
                task: None,
            })
            .await
            .map_err(|e| anyhow!("failed to call tool '{tool}' on '{server_id}': {e:?}"))?;

        let mut lines: Vec<String> = Vec::new();
        for block in result.content {
            lines.push(format_content_block(&block));
        }
        if lines.is_empty() {
            lines.push("[MCP] tool returned no content blocks".to_string());
        }
        if result.is_error.unwrap_or(false) {
            return Err(anyhow!(lines.join("\n")));
        }
        Ok(lines.join("\n"))
    }

    async fn client(&self, server_id: &str) -> Result<Arc<ClientRuntime>> {
        let sessions = self.sessions.lock().await;
        sessions
            .get(server_id)
            .cloned()
            .ok_or_else(|| anyhow!("mcp server '{server_id}' is not connected"))
    }
}

fn sanitize_identifier(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut prev_is_underscore = false;
    for ch in input.chars() {
        let normalized = if ch.is_ascii_alphanumeric() || ch == '_' {
            ch.to_ascii_lowercase()
        } else {
            '_'
        };
        if normalized == '_' {
            if !prev_is_underscore {
                out.push('_');
            }
            prev_is_underscore = true;
        } else {
            out.push(normalized);
            prev_is_underscore = false;
        }
    }
    out.trim_matches('_').to_string()
}

fn format_content_block(block: &ContentBlock) -> String {
    match block {
        ContentBlock::TextContent(text) => text.text.clone(),
        ContentBlock::ImageContent(image) => {
            format!("[image content mime={}]", image.mime_type)
        }
        ContentBlock::AudioContent(audio) => {
            format!("[audio content mime={}]", audio.mime_type)
        }
        ContentBlock::ResourceLink(resource) => {
            format!("[resource link uri={} name={}]", resource.uri, resource.name)
        }
        ContentBlock::EmbeddedResource(resource) => match &resource.resource {
            rust_mcp_sdk::schema::EmbeddedResourceResource::TextResourceContents(text) => format!(
                "[embedded text resource uri={} mime={}]",
                text.uri,
                text.mime_type.as_deref().unwrap_or("unknown")
            ),
            rust_mcp_sdk::schema::EmbeddedResourceResource::BlobResourceContents(blob) => format!(
                "[embedded blob resource uri={} mime={}]",
                blob.uri,
                blob.mime_type.as_deref().unwrap_or("unknown")
            ),
        },
    }
}

fn normalize_arguments(value: Option<Value>) -> Result<Option<Map<String, Value>>> {
    match value {
        None => Ok(None),
        Some(Value::Object(map)) => Ok(Some(map)),
        Some(other) => Err(anyhow!(
            "mcp_call_tool arguments must be a JSON object, got {}",
            other
        )),
    }
}
