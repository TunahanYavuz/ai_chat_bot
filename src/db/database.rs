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
        self.conn.execute("BEGIN", [])?;
        let result = (|| -> Result<()> {
            self.conn.execute("DELETE FROM attachments WHERE message_id IN (SELECT id FROM messages WHERE session_id = ?1)", params![session_id])?;
            self.conn.execute("DELETE FROM messages WHERE session_id = ?1", params![session_id])?;
            self.conn.execute("DELETE FROM sessions WHERE id = ?1", params![session_id])?;
            Ok(())
        })();
        match result {
            Ok(_) => { self.conn.execute("COMMIT", [])?; }
            Err(e) => { let _ = self.conn.execute("ROLLBACK", []); return Err(e); }
        }
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
}
