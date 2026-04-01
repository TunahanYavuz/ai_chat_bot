use anyhow::{Context, Result};
use reqwest::Client;
use scraper::{Html, Selector};
use std::sync::OnceLock;
use std::time::Duration;

const DEFAULT_TIMEOUT_SECS: u64 = 12;
const MAX_TEXT_CHARS: usize = 16_000;
const SEARCH_MAX_RESULTS: usize = 5;

#[derive(Debug, Clone)]
pub struct WebSearchResult {
    pub title: String,
    pub snippet: String,
    pub url: String,
}

#[derive(Debug, Clone)]
pub struct WebEngine {
    client: Client,
}

impl WebEngine {
    pub fn new() -> Result<Self> {
        static CLIENT: OnceLock<Client> = OnceLock::new();
        if let Some(client) = CLIENT.get() {
            return Ok(Self {
                client: client.clone(),
            });
        }

        let built = Client::builder()
            .timeout(Duration::from_secs(DEFAULT_TIMEOUT_SECS))
            .user_agent("ai_chat_bot-web-researcher/phase8")
            .build()
            .context("failed to build web client")?;
        let _ = CLIENT.set(built.clone());
        Ok(Self { client: built })
    }

    pub async fn search_web(&self, query: &str) -> Result<Vec<WebSearchResult>> {
        let q = query.trim();
        if q.is_empty() {
            return Ok(Vec::new());
        }

        let resp = self
            .client
            .get("https://duckduckgo.com/html/")
            .query(&[("q", q)])
            .send()
            .await
            .context("search request failed")?;
        let body = resp.text().await.context("failed to read search body")?;

        let doc = Html::parse_document(&body);
        let result_sel = Selector::parse("div.result").expect("valid selector");
        let title_sel = Selector::parse("a.result__a").expect("valid selector");
        let snippet_sel =
            Selector::parse("a.result__snippet, .result__snippet").expect("valid selector");

        let mut out = Vec::new();
        for node in doc.select(&result_sel) {
            let Some(a) = node.select(&title_sel).next() else {
                continue;
            };
            let title = collapse_ws(&a.text().collect::<Vec<_>>().join(" "));
            let href = a
                .value()
                .attr("href")
                .unwrap_or_default()
                .trim()
                .to_string();
            if href.is_empty() {
                continue;
            }
            let snippet = node
                .select(&snippet_sel)
                .next()
                .map(|s| collapse_ws(&s.text().collect::<Vec<_>>().join(" ")))
                .unwrap_or_default();
            out.push(WebSearchResult {
                title,
                snippet,
                url: href,
            });
            if out.len() >= SEARCH_MAX_RESULTS {
                break;
            }
        }

        Ok(out)
    }

    pub async fn read_url(&self, url: &str) -> Result<String> {
        let target = url.trim();
        if target.is_empty() {
            return Ok(String::new());
        }
        let resp = self
            .client
            .get(target)
            .send()
            .await
            .with_context(|| format!("failed to fetch url: {target}"))?;
        let body = resp.text().await.context("failed reading web page body")?;
        Ok(extract_clean_text(&body))
    }
}

pub fn format_search_results(results: &[WebSearchResult]) -> String {
    if results.is_empty() {
        return "[Web] No results found.".to_string();
    }
    let mut out = String::from("[Web] Search results:\n");
    for (i, r) in results.iter().enumerate() {
        out.push_str(&format!(
            "{}. {}\nURL: {}\nSnippet: {}\n\n",
            i + 1,
            r.title,
            r.url,
            r.snippet
        ));
    }
    out.trim_end().to_string()
}

fn extract_clean_text(html: &str) -> String {
    let doc = Html::parse_document(html);
    let selectors = [
        "main",
        "article",
        "[role='main']",
        "section",
        ".content",
        "#content",
        "body",
    ];
    let mut chunks = Vec::new();
    for s in selectors {
        if let Ok(sel) = Selector::parse(s) {
            for node in doc.select(&sel) {
                let text = collapse_ws(&node.text().collect::<Vec<_>>().join(" "));
                if text.len() >= 80 {
                    chunks.push(text);
                }
            }
        }
        if !chunks.is_empty() {
            break;
        }
    }

    let mut joined = if chunks.is_empty() {
        collapse_ws(&doc.root_element().text().collect::<Vec<_>>().join(" "))
    } else {
        chunks.join("\n\n")
    };
    if joined.len() > MAX_TEXT_CHARS {
        truncate_on_char_boundary(&mut joined, MAX_TEXT_CHARS);
    }
    joined
}

fn collapse_ws(input: &str) -> String {
    input.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn truncate_on_char_boundary(s: &mut String, max_bytes: usize) {
    if s.len() <= max_bytes {
        return;
    }
    let mut idx = max_bytes;
    while idx > 0 && !s.is_char_boundary(idx) {
        idx -= 1;
    }
    s.truncate(idx);
}
