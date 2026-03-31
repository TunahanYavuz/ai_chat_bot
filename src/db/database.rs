use anyhow::Result;
use rusqlite::{Connection, params};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DbSession {
    pub id: String,
    pub name: String,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DbMessage {
    pub id: String,
    pub session_id: String,
    pub role: String,
    pub content: String,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DbAttachment {
    pub id: String,
    pub message_id: String,
    pub filename: String,
    pub data: Vec<u8>,
    pub mime_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DbFileSnapshot {
    pub id: String,
    pub file_path: String,
    pub content: Vec<u8>,
    pub created_at: DateTime<Utc>,
}

pub struct Database {
    conn: Connection,
}

impl Database {
    pub fn new(path: &str) -> Result<Self> {
        if let Some(parent) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(path)?;
        let db = Self { conn };
        db.init()?;
        Ok(db)
    }

    fn init(&self) -> Result<()> {
        self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );
            CREATE TABLE IF NOT EXISTS attachments (
                id TEXT PRIMARY KEY,
                message_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                data BLOB NOT NULL,
                mime_type TEXT NOT NULL,
                FOREIGN KEY (message_id) REFERENCES messages(id)
            );
            CREATE TABLE IF NOT EXISTS file_snapshots (
                id TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                content BLOB NOT NULL,
                created_at TEXT NOT NULL
            );",
        )?;
        Ok(())
    }

    pub fn create_session(&self, session: &DbSession) -> Result<()> {
        self.conn.execute(
            "INSERT INTO sessions (id, name, created_at) VALUES (?1, ?2, ?3)",
            params![session.id, session.name, session.created_at.to_rfc3339()],
        )?;
        Ok(())
    }

    pub fn list_sessions(&self) -> Result<Vec<DbSession>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name, created_at FROM sessions ORDER BY created_at DESC",
        )?;
        let sessions = stmt.query_map([], |row| {
            let created_str: String = row.get(2)?;
            Ok(DbSession {
                id: row.get(0)?,
                name: row.get(1)?,
                created_at: created_str
                    .parse::<DateTime<Utc>>()
                    .unwrap_or_else(|_| Utc::now()),
            })
        })?
        .filter_map(|r| r.ok())
        .collect();
        Ok(sessions)
    }

    pub fn delete_session(&self, session_id: &str) -> Result<()> {
        let tx = self.conn.unchecked_transaction()?;
        tx.execute("DELETE FROM attachments WHERE message_id IN (SELECT id FROM messages WHERE session_id = ?1)", params![session_id])?;
        tx.execute("DELETE FROM messages WHERE session_id = ?1", params![session_id])?;
        tx.execute("DELETE FROM sessions WHERE id = ?1", params![session_id])?;
        tx.commit()?;
        Ok(())
    }

    pub fn save_message(&self, msg: &DbMessage) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO messages (id, session_id, role, content, created_at) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![msg.id, msg.session_id, msg.role, msg.content, msg.created_at.to_rfc3339()],
        )?;
        Ok(())
    }

    pub fn load_messages(&self, session_id: &str) -> Result<Vec<DbMessage>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, session_id, role, content, created_at FROM messages WHERE session_id = ?1 ORDER BY created_at ASC",
        )?;
        let messages = stmt.query_map(params![session_id], |row| {
            let created_str: String = row.get(4)?;
            Ok(DbMessage {
                id: row.get(0)?,
                session_id: row.get(1)?,
                role: row.get(2)?,
                content: row.get(3)?,
                created_at: created_str
                    .parse::<DateTime<Utc>>()
                    .unwrap_or_else(|_| Utc::now()),
            })
        })?
        .filter_map(|r| r.ok())
        .collect();
        Ok(messages)
    }

    pub fn save_attachment(&self, att: &DbAttachment) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO attachments (id, message_id, filename, data, mime_type) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![att.id, att.message_id, att.filename, att.data, att.mime_type],
        )?;
        Ok(())
    }

    pub fn load_attachments(&self, message_id: &str) -> Result<Vec<DbAttachment>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, message_id, filename, data, mime_type FROM attachments WHERE message_id = ?1",
        )?;
        let atts = stmt.query_map(params![message_id], |row| {
            Ok(DbAttachment {
                id: row.get(0)?,
                message_id: row.get(1)?,
                filename: row.get(2)?,
                data: row.get(3)?,
                mime_type: row.get(4)?,
            })
        })?
        .filter_map(|r| r.ok())
        .collect();
        Ok(atts)
    }

    pub fn save_file_snapshot(&self, snapshot: &DbFileSnapshot) -> Result<()> {
        self.conn.execute(
            "INSERT INTO file_snapshots (id, file_path, content, created_at) VALUES (?1, ?2, ?3, ?4)",
            params![
                snapshot.id,
                snapshot.file_path,
                snapshot.content,
                snapshot.created_at.to_rfc3339()
            ],
        )?;
        Ok(())
    }

    pub fn list_file_snapshots(&self, limit: usize) -> Result<Vec<DbFileSnapshot>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, file_path, content, created_at FROM file_snapshots ORDER BY created_at DESC LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![limit as i64], |row| {
            let created_str: String = row.get(3)?;
            Ok(DbFileSnapshot {
                id: row.get(0)?,
                file_path: row.get(1)?,
                content: row.get(2)?,
                created_at: created_str.parse::<DateTime<Utc>>().unwrap_or_else(|_| Utc::now()),
            })
        })?;
        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    pub fn get_file_snapshot(&self, snapshot_id: &str) -> Result<Option<DbFileSnapshot>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, file_path, content, created_at FROM file_snapshots WHERE id = ?1 LIMIT 1",
        )?;
        let mut rows = stmt.query(params![snapshot_id])?;
        if let Some(row) = rows.next()? {
            let created_str: String = row.get(3)?;
            return Ok(Some(DbFileSnapshot {
                id: row.get(0)?,
                file_path: row.get(1)?,
                content: row.get(2)?,
                created_at: created_str.parse::<DateTime<Utc>>().unwrap_or_else(|_| Utc::now()),
            }));
        }
        Ok(None)
    }
}
