use anyhow::Result;
use std::path::Path;
use base64::{engine::general_purpose, Engine as _};

pub struct FileContent {
    pub filename: String,
    pub text: Option<String>,
    pub image_base64: Option<String>,
    pub mime_type: String,
    pub raw_bytes: Vec<u8>,
}

pub fn read_file(path: &Path) -> Result<FileContent> {
    let filename = path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    let raw_bytes = std::fs::read(path)?;
    let ext = path
        .extension()
        .unwrap_or_default()
        .to_string_lossy()
        .to_lowercase();

    let mime_type = match ext.as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "pdf" => "application/pdf",
        "docx" => "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "txt" | "md" | "rs" | "py" | "js" | "ts" | "json" | "toml" | "yaml" | "yml" => {
            "text/plain"
        }
        _ => "application/octet-stream",
    }
    .to_string();

    let is_image = matches!(ext.as_str(), "png" | "jpg" | "jpeg" | "gif" | "webp");

    if is_image {
        let b64 = general_purpose::STANDARD.encode(&raw_bytes);
        return Ok(FileContent {
            filename,
            text: None,
            image_base64: Some(b64),
            mime_type,
            raw_bytes,
        });
    }

    let text = match ext.as_str() {
        "pdf" => extract_pdf_text(&raw_bytes),
        "docx" => extract_docx_text(&raw_bytes),
        _ => String::from_utf8_lossy(&raw_bytes).to_string(),
    };

    Ok(FileContent {
        filename,
        text: Some(text),
        image_base64: None,
        mime_type,
        raw_bytes,
    })
}

fn extract_pdf_text(data: &[u8]) -> String {
    // Simple PDF text extraction: look for BT...ET blocks and extract ASCII text
    let content = String::from_utf8_lossy(data);
    let mut result = String::new();
    let mut in_text = false;
    for line in content.lines() {
        if line.contains("BT") {
            in_text = true;
        }
        if in_text {
            if let Some(start) = line.find('(') {
                if let Some(end) = line.rfind(')') {
                    if end > start {
                        let text = &line[start + 1..end];
                        let cleaned: String = text.chars().filter(|c| c.is_ascii_graphic() || c.is_ascii_whitespace()).collect();
                        if !cleaned.trim().is_empty() {
                            result.push_str(&cleaned);
                            result.push(' ');
                        }
                    }
                }
            }
        }
        if line.contains("ET") {
            in_text = false;
            result.push('\n');
        }
    }
    if result.trim().is_empty() {
        format!("[PDF file - {} bytes, text extraction limited]", data.len())
    } else {
        result
    }
}

fn extract_docx_text(data: &[u8]) -> String {
    // DOCX is a ZIP file containing word/document.xml
    // Try to find XML content by looking for the word/document.xml entry
    use std::io::Read;
    let cursor = std::io::Cursor::new(data);
    let mut archive: zip::ZipArchive<std::io::Cursor<&[u8]>> = match zip::ZipArchive::new(cursor) {
        Ok(a) => a,
        Err(_) => return format!("[DOCX file - {} bytes, could not parse]", data.len()),
    };

    let mut xml_content = String::new();
    for i in 0..archive.len() {
        let mut file = match archive.by_index(i) {
            Ok(f) => f,
            Err(_) => continue,
        };
        if file.name().contains("word/document.xml") {
            let _ = file.read_to_string(&mut xml_content);
            break;
        }
    }

    if xml_content.is_empty() {
        return format!("[DOCX file - {} bytes, no content found]", data.len());
    }

    // Strip XML tags
    let mut result = String::new();
    let mut in_tag = false;
    for ch in xml_content.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => {
                in_tag = false;
                result.push(' ');
            }
            _ if !in_tag => result.push(ch),
            _ => {}
        }
    }
    // Collapse whitespace
    result.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub fn read_text_file(path: &Path) -> Result<String> {
    Ok(std::fs::read_to_string(path)?)
}

pub fn write_text_file(path: &Path, content: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, content)?;
    Ok(())
}
