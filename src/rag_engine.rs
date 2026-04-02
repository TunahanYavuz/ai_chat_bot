use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use async_trait::async_trait;
use ignore::WalkBuilder;
use std::sync::Arc;
use qdrant_client::qdrant::{
    vectors_config::Config, CreateCollection, Distance, PointStruct, SearchPoints, UpsertPoints,
    Value, VectorParams, VectorsConfig,
};
use qdrant_client::Qdrant;
use tokio::task::JoinSet;

/// Configuration for the repository-aware retrieval engine.
#[allow(dead_code)] // Kept for future indexing/chunking configuration.
#[derive(Debug, Clone)]
pub struct RagConfig {
    pub qdrant_url: String,
    pub collection_name: String,
    pub workspace_root: PathBuf,
    pub embedding_size: u64,
    pub chunk_size_chars: usize,
    pub chunk_overlap_chars: usize,
    pub top_k: u64,
    pub similarity_threshold: f32,
}

impl RagConfig {
    /// Sensible defaults for local development with a local Qdrant instance.
    pub fn with_workspace(workspace_root: impl Into<PathBuf>, embedding_size: u64) -> Self {
        Self {
            qdrant_url: "http://127.0.0.1:6334".to_string(),
            collection_name: "workspace_memory".to_string(),
            workspace_root: workspace_root.into(),
            embedding_size,
            chunk_size_chars: 1200,
            chunk_overlap_chars: 150,
            top_k: 5,
            similarity_threshold: 0.75,
        }
    }
}

/// Abstraction boundary for embedding generation.
///
/// This allows swapping providers (Ollama, HuggingFace, OpenAI-compatible endpoint)
/// without modifying retrieval or persistence logic.
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;
}

/// One workspace file loaded for indexing.
#[allow(dead_code)] // Kept for future workspace indexing pipeline.
#[derive(Debug, Clone)]
pub struct WorkspaceDocument {
    pub path: PathBuf,
    pub content: String,
}

/// A normalized chunk that is embedded + persisted to Qdrant.
#[allow(dead_code)] // Kept for future chunk persistence and tracing.
#[derive(Debug, Clone)]
pub struct ChunkRecord {
    pub id: u64,
    pub file_path: String,
    pub snippet: String,
}

/// Search result surfaced to prompt-building layers.
#[allow(dead_code)] // Kept for future typed retrieval outputs.
#[derive(Debug, Clone)]
pub struct RetrievedSnippet {
    pub file_path: String,
    pub score: f32,
    pub snippet: String,
}

/// Repository retrieval-augmented engine backed by Qdrant.
pub struct RagEngine<E>
where
    E: EmbeddingProvider,
{
    client: Arc<Qdrant>,
    config: RagConfig,
    embedder: E,
}

impl<E> RagEngine<E>
where
    E: EmbeddingProvider,
{
    /// Creates the engine with a pre-initialized shared Qdrant client and
    /// asynchronously initializes collection state.
    pub async fn new(config: RagConfig, embedder: E, client: Arc<Qdrant>) -> Result<Self> {
        let engine = Self {
            client,
            config,
            embedder,
        };

        engine.ensure_collection().await?;
        Ok(engine)
    }

    /// Creates a shared Qdrant client with compatibility checks disabled to
    /// avoid noisy version-check logs on startup.
    pub fn init_shared_qdrant_client(qdrant_url: &str) -> Result<Arc<Qdrant>> {
        let client = Qdrant::from_url(qdrant_url)
            .skip_compatibility_check()
            .build()
            .context("failed to build qdrant client")?;
        Ok(Arc::new(client))
    }

    /// Scans repository files while honoring .gitignore rules via `ignore` crate.
    ///
    /// Allowed extensions: .rs, .py, .md, .txt
    /// Explicitly skips: target/, .git/
    #[allow(dead_code)] // Kept for future full indexing flow from UI triggers.
    pub async fn scan_workspace(&self) -> Result<Vec<WorkspaceDocument>> {
        let root = self.config.workspace_root.clone();
        let file_paths = tokio::task::spawn_blocking(move || -> Result<Vec<PathBuf>> {
            let mut builder = WalkBuilder::new(&root);
            builder
                .hidden(false)
                .git_ignore(true)
                .git_global(true)
                .git_exclude(true)
                .filter_entry(|entry| {
                    let p = entry.path();
                    if p.components().any(|c| c.as_os_str() == "target") {
                        return false;
                    }
                    if p.components().any(|c| c.as_os_str() == ".git") {
                        return false;
                    }
                    true
                });

            let mut paths = Vec::new();
            for result in builder.build() {
                let dent = match result {
                    Ok(dent) => dent,
                    Err(_) => continue,
                };
                let path = dent.path();
                if !path.is_file() {
                    continue;
                }
                if !is_allowed_extension(path) {
                    continue;
                }
                paths.push(path.to_path_buf());
            }
            Ok(paths)
        })
        .await
        .context("workspace scanner task failed")??;

        let mut read_tasks = JoinSet::new();
        for path in file_paths {
            read_tasks.spawn(async move {
                let content = tokio::fs::read_to_string(&path).await;
                (path, content)
            });
        }

        let mut docs = Vec::new();
        while let Some(joined) = read_tasks.join_next().await {
            let (path, content_result) = joined.context("failed to join file read task")?;
            let content = match content_result {
                Ok(c) => c,
                Err(_) => continue,
            };
            docs.push(WorkspaceDocument { path, content });
        }

        Ok(docs)
    }

    /// Splits documents into overlap-preserving chunks and computes embeddings for each chunk.
    #[allow(dead_code)] // Kept for future full indexing flow from UI triggers.
    pub async fn chunk_and_embed(
        &self,
        docs: &[WorkspaceDocument],
    ) -> Result<Vec<(ChunkRecord, Vec<f32>)>> {
        let chunks = self.chunk_documents(docs);
        let mut result = Vec::with_capacity(chunks.len());

        for chunk in chunks {
            let vector = self
                .embedder
                .embed(&chunk.snippet)
                .await
                .with_context(|| format!("embedding failed for {}", chunk.file_path))?;
            result.push((chunk, vector));
        }

        Ok(result)
    }

    /// End-to-end indexing pipeline: scan -> chunk -> embed -> upsert.
    #[allow(dead_code)] // Kept for future full indexing flow from UI triggers.
    pub async fn index_workspace(&self) -> Result<usize> {
        let docs = self.scan_workspace().await?;
        let embedded_chunks = self.chunk_and_embed(&docs).await?;

        let points: Vec<PointStruct> = embedded_chunks
            .into_iter()
            .map(|(chunk, vector)| {
                let mut payload = std::collections::HashMap::new();
                payload.insert("file_path".to_string(), Value::from(chunk.file_path));
                payload.insert("snippet".to_string(), Value::from(chunk.snippet));

                PointStruct::new(chunk.id, vector, payload)
            })
            .collect();

        if points.is_empty() {
            return Ok(0);
        }

        let indexed_count = points.len();

        self.client
            .upsert_points(UpsertPoints {
                collection_name: self.config.collection_name.clone(),
                wait: Some(true),
                points,
                ordering: None,
                shard_key_selector: None,
                update_filter: None,
                timeout: None,
                update_mode: None,
            })
            .await
            .context("failed to upsert points into qdrant")?;

        Ok(indexed_count)
    }

    /// Executes semantic search and returns top-k snippets enriched with file path.
    pub async fn semantic_search(&self, query: &str) -> Result<Vec<String>> {
        let vector = self
            .embedder
            .embed(query)
            .await
            .context("failed to embed semantic search query")?;

        let response = self
            .client
            .search_points(SearchPoints {
                collection_name: self.config.collection_name.clone(),
                vector,
                limit: self.config.top_k,
                with_payload: Some(true.into()),
                with_vectors: None,
                score_threshold: None,
                offset: None,
                vector_name: None,
                filter: None,
                params: None,
                read_consistency: None,
                timeout: None,
                shard_key_selector: None,
                sparse_indices: None,
            })
            .await
            .context("failed to search qdrant")?;

        let snippets = response
            .result
            .into_iter()
            .filter(|point| point.score >= self.config.similarity_threshold)
            .map(|point| {
                let file_path = point
                    .payload
                    .get("file_path")
                    .and_then(payload_value_to_string)
                    .unwrap_or_else(|| "unknown-path".to_string());
                let snippet = point
                    .payload
                    .get("snippet")
                    .and_then(payload_value_to_string)
                    .unwrap_or_default();
                format!("[{}]\n{}", file_path, snippet)
            })
            .collect();

        Ok(snippets)
    }

    /// Utility formatter for direct system-prompt injection.
    pub fn format_repository_context(snippets: &[String]) -> String {
        if snippets.is_empty() {
            return "LOCAL REPOSITORY CONTEXT:\n- No relevant snippets found.".to_string();
        }

        let mut out = String::from("LOCAL REPOSITORY CONTEXT:\n");
        for (idx, snippet) in snippets.iter().enumerate() {
            out.push_str(&format!("{}. {}\n\n", idx + 1, snippet));
        }
        out.trim_end().to_string()
    }

    /// Stores one self-healing learning snippet into Qdrant for future retrieval.
    pub async fn upsert_learning_snippet(&self, title: &str, content: &str) -> Result<()> {
        let title = title.trim();
        let content = content.trim();
        if content.is_empty() {
            return Ok(());
        }
        let packed = format!("{title}\n{content}");
        let vector = self
            .embedder
            .embed(&packed)
            .await
            .context("failed to embed learning snippet")?;
        let mut payload = std::collections::HashMap::new();
        payload.insert("file_path".to_string(), Value::from(format!("[learned] {title}")));
        payload.insert("snippet".to_string(), Value::from(content.to_string()));
        self.client
            .upsert_points(UpsertPoints {
                collection_name: self.config.collection_name.clone(),
                wait: Some(true),
                points: vec![PointStruct::new(
                    deterministic_chunk_id("[learned]", 0, &packed),
                    vector,
                    payload,
                )],
                ordering: None,
                shard_key_selector: None,
                update_filter: None,
                timeout: None,
                update_mode: None,
            })
            .await
            .context("failed to upsert learning snippet into qdrant")?;
        Ok(())
    }

    async fn ensure_collection(&self) -> Result<()> {
        if self
            .client
            .collection_exists(&self.config.collection_name)
            .await
            .unwrap_or(false)
        {
            return Ok(());
        }

        self.client
            .create_collection(CreateCollection {
                collection_name: self.config.collection_name.clone(),
                vectors_config: Some(VectorsConfig {
                    config: Some(Config::Params(VectorParams {
                        size: self.config.embedding_size,
                        distance: Distance::Cosine.into(),
                        hnsw_config: None,
                        quantization_config: None,
                        on_disk: None,
                        datatype: None,
                        multivector_config: None,
                    })),
                }),
                shard_number: None,
                replication_factor: None,
                write_consistency_factor: None,
                on_disk_payload: None,
                hnsw_config: None,
                wal_config: None,
                optimizers_config: None,
                quantization_config: None,
                timeout: None,
                strict_mode_config: None,
                sharding_method: None,
                sparse_vectors_config: None,
                metadata: std::collections::HashMap::new(),
            })
            .await
            .context("failed to create qdrant collection")?;

        Ok(())
    }

    #[allow(dead_code)] // Kept for future full indexing flow from UI triggers.
    fn chunk_documents(&self, docs: &[WorkspaceDocument]) -> Vec<ChunkRecord> {
        let mut chunks = Vec::new();
        let size = self.config.chunk_size_chars.max(64);
        let overlap = self.config.chunk_overlap_chars.min(size / 2);

        for doc in docs {
            let path = doc.path.to_string_lossy().to_string();
            let chars: Vec<char> = doc.content.chars().collect();
            if chars.is_empty() {
                continue;
            }

            let mut start = 0usize;
            while start < chars.len() {
                let end = (start + size).min(chars.len());
                let snippet: String = chars[start..end].iter().collect();
                let id = deterministic_chunk_id(&path, start, &snippet);

                chunks.push(ChunkRecord {
                    id,
                    file_path: path.clone(),
                    snippet,
                });

                if end == chars.len() {
                    break;
                }

                let mut next_start = end.saturating_sub(overlap);
                if next_start <= start {
                    next_start = start + 1;
                }
                start = next_start;
            }
        }

        chunks
    }
}

#[allow(dead_code)] // Kept for future indexing scan usage.
fn is_allowed_extension(path: &Path) -> bool {
    let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
        return false;
    };

    matches!(
        ext.to_ascii_lowercase().as_str(),
        "rs" | "py" | "md" | "txt"
    )
}

fn payload_value_to_string(value: &Value) -> Option<String> {
    match &value.kind {
        Some(qdrant_client::qdrant::value::Kind::StringValue(s)) => Some(s.clone()),
        _ => None,
    }
}

#[allow(dead_code)] // Kept for future deterministic indexing IDs.
fn deterministic_chunk_id(file_path: &str, start_offset: usize, snippet: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    file_path.hash(&mut hasher);
    start_offset.hash(&mut hasher);
    snippet.hash(&mut hasher);
    hasher.finish()
}
