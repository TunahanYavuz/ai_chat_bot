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
}

#[derive(Default)]
struct EmptyClientHandler;

#[async_trait::async_trait]
impl ClientHandler for EmptyClientHandler {}

#[derive(Default)]
pub struct McpManager {
    sessions: Mutex<HashMap<String, Arc<ClientRuntime>>>,
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
            Ok(())
        } else {
            Err(anyhow!("mcp server '{server_id}' is not connected"))
        }
    }

    pub async fn list_tools(&self, server_id: &str) -> Result<Vec<McpToolInfo>> {
        let client = self.client(server_id).await?;
        let tools = client
            .request_tool_list(None)
            .await
            .map_err(|e| anyhow!("failed to list tools for '{server_id}': {e:?}"))?;
        Ok(tools
            .tools
            .into_iter()
            .map(|tool| McpToolInfo {
                name: tool.name,
                description: tool.description.unwrap_or_default(),
            })
            .collect())
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
