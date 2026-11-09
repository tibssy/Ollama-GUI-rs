use std::cell::RefCell;
use std::collections::HashMap;

use anyhow::{bail, Context, Result};
use rusqlite::{params, Connection, OptionalExtension};

use crate::api::StreamStats;
use crate::memory::{self, DateHint, MemoryEntry, MemoryPool};

#[derive(Debug, Clone)]
pub struct ConversationSummary {
    pub id: i64,
    pub title: String,
    pub model: String,
    /// Estimated tokens in the context after the latest assistant reply.
    pub context_used: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct StoredMessage {
    pub id: i64,
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredSummary {
    pub text: String,
    /// Id of the last message covered by the summary.
    pub upto_id: i64,
    pub tokens: u32,
}

pub struct Store {
    conn: Connection,
    memory_cache: RefCell<HashMap<(MemoryPool, String), Vec<MemoryEntry>>>,
}

impl Store {
    pub fn open() -> Result<Self> {
        let dir = crate::config::data_dir();
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("creating data directory {}", dir.display()))?;
        let path = dir.join("chats.db");
        let conn = Connection::open(&path)
            .with_context(|| format!("opening database {}", path.display()))?;
        restrict_database_permissions(&path);
        let store = Self::from_connection(conn)?;
        // WAL and SHM files are created by the first migration.
        restrict_database_permissions(&path);
        Ok(store)
    }

    fn from_connection(conn: Connection) -> Result<Self> {
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        let store = Self {
            conn,
            memory_cache: RefCell::new(HashMap::new()),
        };
        store.migrate()?;
        Ok(store)
    }

    #[cfg(test)]
    fn open_in_memory() -> Result<Self> {
        Self::from_connection(Connection::open_in_memory()?)
    }

    fn migrate(&self) -> Result<()> {
        self.conn.execute_batch(
            r#"
            PRAGMA journal_mode = WAL;
            PRAGMA foreign_keys = ON;

            CREATE TABLE IF NOT EXISTS conversations(
                id          INTEGER PRIMARY KEY,
                title       TEXT NOT NULL,
                model       TEXT NOT NULL,
                created_at  INTEGER NOT NULL DEFAULT (unixepoch()),
                updated_at  INTEGER NOT NULL DEFAULT (unixepoch())
            );

            CREATE TABLE IF NOT EXISTS messages(
                id              INTEGER PRIMARY KEY,
                conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                role            TEXT NOT NULL,
                content         TEXT NOT NULL,
                stats_json      TEXT,
                created_at      INTEGER NOT NULL DEFAULT (unixepoch())
            );

            CREATE INDEX IF NOT EXISTS idx_messages_conversation
                ON messages(conversation_id, id);

            CREATE TABLE IF NOT EXISTS memories(
                id              INTEGER PRIMARY KEY,
                pool            TEXT NOT NULL,
                embedding_model TEXT NOT NULL,
                dims            INTEGER NOT NULL,
                text            TEXT NOT NULL,
                vector          BLOB NOT NULL,
                model           TEXT NOT NULL,
                conversation_id INTEGER,
                created_at      INTEGER NOT NULL DEFAULT (unixepoch()),
                UNIQUE(pool, embedding_model, text)
            );

            CREATE INDEX IF NOT EXISTS idx_memories_lookup
                ON memories(pool, embedding_model, dims);
            "#,
        )?;
        self.ensure_column("conversations", "context_used", "INTEGER")?;
        self.ensure_column("conversations", "summary", "TEXT")?;
        self.ensure_column("conversations", "summary_upto_id", "INTEGER")?;
        self.ensure_column("conversations", "summary_tokens", "INTEGER")?;
        self.ensure_column("conversations", "memory_upto_id", "INTEGER")?;
        self.backfill_context_usage()?;
        Ok(())
    }

    fn ensure_column(&self, table: &str, column: &str, declaration: &str) -> Result<()> {
        let exists = {
            let mut stmt = self.conn.prepare(&format!("PRAGMA table_info({table})"))?;
            let mut rows = stmt.query([])?;
            let mut exists = false;
            while let Some(row) = rows.next()? {
                let name: String = row.get(1)?;
                if name == column {
                    exists = true;
                    break;
                }
            }
            exists
        };
        if !exists {
            self.conn.execute_batch(&format!(
                "ALTER TABLE {table} ADD COLUMN {column} {declaration}"
            ))?;
        }
        Ok(())
    }

    /// Existing databases predate the `context_used` column; derive it from the
    /// newest stored stats payload.
    fn backfill_context_usage(&self) -> Result<()> {
        let ids = {
            let mut stmt = self
                .conn
                .prepare("SELECT id FROM conversations WHERE context_used IS NULL")?;
            let rows = stmt.query_map([], |row| row.get::<_, i64>(0))?;
            rows.collect::<std::result::Result<Vec<_>, _>>()?
        };
        for id in ids {
            if let Some(used) = self.latest_stats_used(id)? {
                self.conn.execute(
                    "UPDATE conversations SET context_used = ?1 WHERE id = ?2",
                    params![used, id],
                )?;
            }
        }
        Ok(())
    }

    fn latest_stats_used(&self, conversation_id: i64) -> Result<Option<u32>> {
        let stats_json: Option<String> = self
            .conn
            .query_row(
                "SELECT stats_json FROM messages
                 WHERE conversation_id = ?1 AND stats_json IS NOT NULL
                 ORDER BY id DESC LIMIT 1",
                params![conversation_id],
                |row| row.get(0),
            )
            .optional()?;
        Ok(stats_json.and_then(|json| context_used_from_json(&json)))
    }

    pub fn conversation_context_used(&self, id: i64) -> Result<Option<u32>> {
        Ok(self
            .conn
            .query_row(
                "SELECT context_used FROM conversations WHERE id = ?1",
                params![id],
                |row| row.get(0),
            )
            .optional()?
            .flatten())
    }

    pub fn set_conversation_context_used(&self, id: i64, used: Option<u32>) -> Result<()> {
        self.conn.execute(
            "UPDATE conversations SET context_used = ?1 WHERE id = ?2",
            params![used, id],
        )?;
        Ok(())
    }

    pub fn conversation_summary(&self, id: i64) -> Result<Option<StoredSummary>> {
        let row = self
            .conn
            .query_row(
                "SELECT summary, summary_upto_id, summary_tokens
                 FROM conversations WHERE id = ?1",
                params![id],
                |row| {
                    Ok((
                        row.get::<_, Option<String>>(0)?,
                        row.get::<_, Option<i64>>(1)?,
                        row.get::<_, Option<u32>>(2)?,
                    ))
                },
            )
            .optional()?;

        Ok(row.and_then(|(text, upto_id, tokens)| {
            let text = text?;
            if text.trim().is_empty() {
                return None;
            }
            Some(StoredSummary {
                text,
                upto_id: upto_id.unwrap_or(0),
                tokens: tokens.unwrap_or(0),
            })
        }))
    }

    pub fn set_conversation_summary(&self, id: i64, summary: &StoredSummary) -> Result<()> {
        self.conn.execute(
            "UPDATE conversations
             SET summary = ?1, summary_upto_id = ?2, summary_tokens = ?3
             WHERE id = ?4",
            params![summary.text, summary.upto_id, summary.tokens, id],
        )?;
        Ok(())
    }

    pub fn clear_conversation_summary(&self, id: i64) -> Result<()> {
        self.conn.execute(
            "UPDATE conversations
             SET summary = NULL, summary_upto_id = NULL, summary_tokens = NULL,
                 updated_at = unixepoch()
             WHERE id = ?1",
            params![id],
        )?;
        Ok(())
    }

    /// Last message id already summarized into long-term memory.
    pub fn conversation_memory_checkpoint(&self, id: i64) -> Result<i64> {
        Ok(self
            .conn
            .query_row(
                "SELECT COALESCE(memory_upto_id, 0) FROM conversations WHERE id = ?1",
                params![id],
                |row| row.get(0),
            )
            .optional()?
            .unwrap_or(0))
    }

    pub fn set_conversation_memory_checkpoint(&self, id: i64, upto_id: i64) -> Result<()> {
        self.conn.execute(
            "UPDATE conversations SET memory_upto_id = ?1 WHERE id = ?2",
            params![upto_id, id],
        )?;
        Ok(())
    }

    pub fn latest_message_id(&self, conversation_id: i64) -> Result<Option<i64>> {
        Ok(self
            .conn
            .query_row(
                "SELECT id FROM messages
                 WHERE conversation_id = ?1 ORDER BY id DESC LIMIT 1",
                params![conversation_id],
                |row| row.get(0),
            )
            .optional()?)
    }

    pub fn memory_count(&self, pool: MemoryPool) -> Result<usize> {
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM memories WHERE pool = ?1",
            params![pool.as_str()],
            |row| row.get(0),
        )?;
        Ok(count as usize)
    }

    pub fn memory_counts(&self) -> Result<(usize, usize)> {
        Ok((
            self.memory_count(MemoryPool::Local)?,
            self.memory_count(MemoryPool::Cloud)?,
        ))
    }

    /// Today's local date in YYYY-MM-DD form.
    pub fn local_date(&self) -> Result<String> {
        Ok(self
            .conn
            .query_row("SELECT date('now','localtime')", [], |row| row.get(0))?)
    }

    /// Resolve a relative date from the user's query to an inclusive
    /// YYYY-MM-DD range using SQLite's local-time date arithmetic.
    pub fn resolve_date_hint(&self, hint: DateHint) -> Result<(String, String)> {
        match hint {
            DateHint::Exact(date) => {
                let normalized: Option<String> =
                    self.conn
                        .query_row("SELECT date(?1)", params![date], |row| row.get(0))?;
                match normalized {
                    Some(date) => Ok((date.clone(), date)),
                    None => bail!("invalid date"),
                }
            }
            DateHint::Today => {
                let date = self.local_date()?;
                Ok((date.clone(), date))
            }
            DateHint::Yesterday => self.date_offset(1),
            DateHint::DaysAgo(days) => self.date_offset(days),
            DateHint::LastWeekday(weekday) => {
                let modifier = format!("weekday {weekday}");
                let date: String = self.conn.query_row(
                    "SELECT date('now','localtime','-7 days', ?1)",
                    params![modifier],
                    |row| row.get(0),
                )?;
                Ok((date.clone(), date))
            }
            DateHint::LastWeek => Ok(self.conn.query_row(
                "SELECT date('now','localtime','-13 days'),
                        date('now','localtime','-7 days')",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )?),
            DateHint::LastMonth => Ok(self.conn.query_row(
                "SELECT date('now','localtime','start of month','-1 month'),
                        date('now','localtime','start of month','-1 day')",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )?),
            DateHint::LastYear => Ok(self.conn.query_row(
                "SELECT date('now','localtime','start of year','-1 year'),
                        date('now','localtime','start of year','-1 day')",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )?),
        }
    }

    fn date_offset(&self, days: u32) -> Result<(String, String)> {
        let modifier = format!("-{days} days");
        let date: String = self.conn.query_row(
            "SELECT date('now','localtime', ?1)",
            params![modifier],
            |row| row.get(0),
        )?;
        Ok((date.clone(), date))
    }

    /// Load the notes for one pool and embedding model. Results are cached
    /// until a note is added or memory is cleared.
    pub fn memory_entries(
        &self,
        pool: MemoryPool,
        embedding_model: &str,
    ) -> Result<Vec<MemoryEntry>> {
        let key = (pool, embedding_model.to_string());
        if let Some(entries) = self.memory_cache.borrow().get(&key) {
            return Ok(entries.clone());
        }

        let mut stmt = self.conn.prepare(
            "SELECT text, vector, date(created_at, 'unixepoch', 'localtime')
             FROM memories
             WHERE pool = ?1 AND embedding_model = ?2
             ORDER BY id ASC",
        )?;
        let rows = stmt.query_map(params![pool.as_str(), embedding_model], |row| {
            let bytes: Vec<u8> = row.get(1)?;
            Ok(MemoryEntry {
                text: row.get(0)?,
                vector: memory::decode_vector(&bytes),
                created: row.get(2)?,
            })
        })?;
        let mut entries = Vec::new();
        for row in rows {
            entries.push(row?);
        }

        self.memory_cache.borrow_mut().insert(key, entries.clone());
        Ok(entries)
    }

    pub fn add_memory(
        &self,
        pool: MemoryPool,
        embedding_model: &str,
        model: &str,
        conversation_id: Option<i64>,
        text: &str,
        vector: &[f32],
    ) -> Result<bool> {
        let bytes = memory::encode_vector(vector);
        let changed = self.conn.execute(
            "INSERT OR IGNORE INTO memories(
                 pool, embedding_model, dims, text, vector, model, conversation_id
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                pool.as_str(),
                embedding_model,
                vector.len() as i64,
                text,
                bytes,
                model,
                conversation_id
            ],
        )?;
        self.memory_cache
            .borrow_mut()
            .remove(&(pool, embedding_model.to_string()));
        Ok(changed > 0)
    }

    pub fn clear_memories(&self, pool: Option<MemoryPool>) -> Result<usize> {
        let changed = match pool {
            Some(pool) => self.conn.execute(
                "DELETE FROM memories WHERE pool = ?1",
                params![pool.as_str()],
            )?,
            None => self.conn.execute("DELETE FROM memories", [])?,
        };
        self.memory_cache.borrow_mut().clear();
        Ok(changed)
    }

    pub fn list_conversations(&self) -> Result<Vec<ConversationSummary>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, title, model, context_used FROM conversations
             ORDER BY updated_at DESC, id DESC",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(ConversationSummary {
                id: row.get(0)?,
                title: row.get(1)?,
                model: row.get(2)?,
                context_used: row.get(3)?,
            })
        })?;
        let mut conversations = Vec::new();
        for row in rows {
            conversations.push(row?);
        }
        Ok(conversations)
    }

    pub fn create_conversation(&self, model: &str, title: &str) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO conversations(title, model) VALUES (?1, ?2)",
            params![title, model],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    pub fn delete_conversation(&self, id: i64) -> Result<()> {
        self.conn
            .execute("DELETE FROM conversations WHERE id = ?1", params![id])?;
        Ok(())
    }

    #[allow(dead_code)]
    pub fn rename_conversation(&self, id: i64, title: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE conversations SET title = ?1, updated_at = unixepoch() WHERE id = ?2",
            params![title, id],
        )?;
        Ok(())
    }

    pub fn set_conversation_model(&self, id: i64, model: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE conversations SET model = ?1, updated_at = unixepoch() WHERE id = ?2",
            params![model, id],
        )?;
        Ok(())
    }

    pub fn conversation_model(&self, id: i64) -> Result<Option<String>> {
        Ok(self
            .conn
            .query_row(
                "SELECT model FROM conversations WHERE id = ?1",
                params![id],
                |row| row.get(0),
            )
            .optional()?)
    }

    pub fn conversation_title(&self, id: i64) -> Result<Option<String>> {
        Ok(self
            .conn
            .query_row(
                "SELECT title FROM conversations WHERE id = ?1",
                params![id],
                |row| row.get(0),
            )
            .optional()?)
    }

    pub fn add_message(
        &self,
        conversation_id: i64,
        role: &str,
        content: &str,
        stats: Option<&StreamStats>,
    ) -> Result<i64> {
        let stats_json = stats.and_then(|stats| {
            if stats.eval_count.is_none()
                && stats.eval_duration.is_none()
                && stats.prompt_eval_count.is_none()
            {
                return None;
            }
            serde_json::to_string(&serde_json::json!({
                "eval_count": stats.eval_count,
                "eval_duration": stats.eval_duration,
                "prompt_eval_count": stats.prompt_eval_count,
            }))
            .ok()
        });

        self.conn.execute(
            "INSERT INTO messages(conversation_id, role, content, stats_json)
             VALUES (?1, ?2, ?3, ?4)",
            params![conversation_id, role, content, stats_json],
        )?;

        let context_used = if role == "assistant" {
            stats.and_then(|stats| {
                let has_counts = stats.eval_count.is_some() || stats.prompt_eval_count.is_some();
                has_counts
                    .then(|| stats.eval_count.unwrap_or(0) + stats.prompt_eval_count.unwrap_or(0))
            })
        } else {
            None
        };

        self.conn.execute(
            "UPDATE conversations
             SET updated_at = unixepoch(), context_used = COALESCE(?2, context_used)
             WHERE id = ?1",
            params![conversation_id, context_used],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    pub fn messages(&self, conversation_id: i64) -> Result<Vec<StoredMessage>> {
        self.messages_after(conversation_id, 0)
    }

    pub fn messages_after(
        &self,
        conversation_id: i64,
        after_id: i64,
    ) -> Result<Vec<StoredMessage>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, role, content FROM messages
             WHERE conversation_id = ?1 AND id > ?2 ORDER BY id ASC",
        )?;
        let rows = stmt.query_map(params![conversation_id, after_id], |row| {
            Ok(StoredMessage {
                id: row.get(0)?,
                role: row.get(1)?,
                content: row.get(2)?,
            })
        })?;
        let mut messages = Vec::new();
        for row in rows {
            messages.push(row?);
        }
        Ok(messages)
    }

    /// Remove a message together with its counterpart in the same turn.
    ///
    /// Removing a user message also removes the assistant reply that follows
    /// it; removing an assistant message also removes the user prompt that
    /// precedes it. Returns the deleted message ids.
    pub fn delete_turn(&self, conversation_id: i64, message_id: i64) -> Result<Vec<i64>> {
        let messages = self.messages(conversation_id)?;
        let Some(index) = messages.iter().position(|message| message.id == message_id) else {
            return Ok(Vec::new());
        };

        let mut ids = vec![message_id];
        match messages[index].role.as_str() {
            "user" => {
                if let Some(next) = messages.get(index + 1) {
                    if next.role == "assistant" {
                        ids.push(next.id);
                    }
                }
            }
            "assistant" => {
                if let Some(previous) = index
                    .checked_sub(1)
                    .and_then(|previous_index| messages.get(previous_index))
                {
                    if previous.role == "user" {
                        ids.push(previous.id);
                    }
                }
            }
            _ => {}
        }

        for id in &ids {
            self.conn.execute(
                "DELETE FROM messages WHERE conversation_id = ?1 AND id = ?2",
                params![conversation_id, id],
            )?;
        }
        let watermark = self
            .conversation_summary(conversation_id)?
            .map(|summary| summary.upto_id);
        if watermark.is_some_and(|watermark| ids.iter().any(|id| *id <= watermark)) {
            self.clear_conversation_summary(conversation_id)?;
        }
        self.recalculate_context_used(conversation_id)?;
        Ok(ids)
    }

    /// Delete a message and everything after it. Used when regenerating an
    /// earlier reply, where all later turns no longer apply.
    pub fn delete_from_message(&self, conversation_id: i64, message_id: i64) -> Result<usize> {
        let deleted = self.conn.execute(
            "DELETE FROM messages WHERE conversation_id = ?1 AND id >= ?2",
            params![conversation_id, message_id],
        )?;
        let watermark = self
            .conversation_summary(conversation_id)?
            .map(|summary| summary.upto_id);
        if deleted > 0 && watermark.is_some_and(|watermark| message_id <= watermark) {
            self.clear_conversation_summary(conversation_id)?;
        }
        self.recalculate_context_used(conversation_id)?;
        Ok(deleted)
    }

    pub fn recalculate_context_used(&self, conversation_id: i64) -> Result<()> {
        let used = self.latest_stats_used(conversation_id)?;
        self.conn.execute(
            "UPDATE conversations
             SET context_used = ?1, updated_at = unixepoch()
             WHERE id = ?2",
            params![used, conversation_id],
        )?;
        Ok(())
    }
}

/// Chat history and memory notes contain personal data, so keep the database
/// readable only by the owner.
fn restrict_database_permissions(path: &std::path::Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            return;
        };
        for candidate in [
            path.to_path_buf(),
            path.with_file_name(format!("{name}-wal")),
            path.with_file_name(format!("{name}-shm")),
        ] {
            if let Ok(metadata) = std::fs::metadata(&candidate) {
                let mut permissions = metadata.permissions();
                permissions.set_mode(0o600);
                let _ = std::fs::set_permissions(&candidate, permissions);
            }
        }
    }
}

fn context_used_from_json(json: &str) -> Option<u32> {
    #[derive(serde::Deserialize)]
    struct StoredStats {
        #[serde(default)]
        eval_count: Option<u32>,
        #[serde(default)]
        prompt_eval_count: Option<u32>,
    }

    let stats: StoredStats = serde_json::from_str(json).ok()?;
    let has_counts = stats.eval_count.is_some() || stats.prompt_eval_count.is_some();
    has_counts.then(|| stats.eval_count.unwrap_or(0) + stats.prompt_eval_count.unwrap_or(0))
}

#[cfg(test)]
mod tests {
    use super::{context_used_from_json, Store, StoredSummary};
    use crate::api::StreamStats;
    use crate::memory::{DateHint, MemoryPool};
    use rusqlite::params;

    #[test]
    fn context_used_sums_prompt_and_generated_tokens() {
        let json = r#"{"eval_count":12,"eval_duration":100,"prompt_eval_count":34}"#;
        assert_eq!(context_used_from_json(json), Some(46));
    }

    #[test]
    fn context_used_handles_missing_counts() {
        assert_eq!(context_used_from_json("{}"), None);
        assert_eq!(context_used_from_json(r#"{"eval_count":5}"#), Some(5));
        assert_eq!(context_used_from_json("not json"), None);
    }

    #[test]
    fn add_message_tracks_context_usage() {
        let store = Store::open_in_memory().unwrap();
        let id = store.create_conversation("model", "title").unwrap();
        let stats = StreamStats {
            eval_count: Some(7),
            eval_duration: None,
            prompt_eval_count: Some(16),
        };
        store
            .add_message(id, "assistant", "hi", Some(&stats))
            .unwrap();
        assert_eq!(store.conversation_context_used(id).unwrap(), Some(23));

        let listed = store.list_conversations().unwrap();
        assert_eq!(listed[0].context_used, Some(23));

        // Messages without stats must not reset the recorded usage.
        store.add_message(id, "user", "hello", None).unwrap();
        assert_eq!(store.conversation_context_used(id).unwrap(), Some(23));
    }

    fn stats(eval_count: u32, prompt_eval_count: u32) -> StreamStats {
        StreamStats {
            eval_count: Some(eval_count),
            eval_duration: None,
            prompt_eval_count: Some(prompt_eval_count),
        }
    }

    #[test]
    fn delete_turn_removes_prompt_and_reply() {
        let store = Store::open_in_memory().unwrap();
        let id = store.create_conversation("model", "title").unwrap();
        let user1 = store.add_message(id, "user", "one", None).unwrap();
        let assistant1 = store
            .add_message(id, "assistant", "two", Some(&stats(2, 3)))
            .unwrap();
        let user2 = store.add_message(id, "user", "three", None).unwrap();
        let assistant2 = store
            .add_message(id, "assistant", "four", Some(&stats(4, 5)))
            .unwrap();
        assert_eq!(store.conversation_context_used(id).unwrap(), Some(9));

        let removed = store.delete_turn(id, user2).unwrap();
        assert_eq!(removed.len(), 2);
        assert!(removed.contains(&user2));
        assert!(removed.contains(&assistant2));
        assert_eq!(store.messages(id).unwrap().len(), 2);
        assert_eq!(store.conversation_context_used(id).unwrap(), Some(5));

        let removed = store.delete_turn(id, assistant1).unwrap();
        assert_eq!(removed.len(), 2);
        assert!(removed.contains(&user1));
        assert!(removed.contains(&assistant1));
        assert!(store.messages(id).unwrap().is_empty());
        assert_eq!(store.conversation_context_used(id).unwrap(), None);
    }

    #[test]
    fn delete_from_message_truncates_later_turns() {
        let store = Store::open_in_memory().unwrap();
        let id = store.create_conversation("model", "title").unwrap();
        let user1 = store.add_message(id, "user", "one", None).unwrap();
        let assistant1 = store
            .add_message(id, "assistant", "two", Some(&stats(2, 3)))
            .unwrap();
        store.add_message(id, "user", "three", None).unwrap();
        store
            .add_message(id, "assistant", "four", Some(&stats(4, 5)))
            .unwrap();

        let deleted = store.delete_from_message(id, assistant1).unwrap();
        assert_eq!(deleted, 3);
        let remaining = store.messages(id).unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].id, user1);
        assert_eq!(store.conversation_context_used(id).unwrap(), None);
    }

    #[test]
    fn stores_and_clears_conversation_summary() {
        let store = Store::open_in_memory().unwrap();
        let id = store.create_conversation("model", "title").unwrap();
        let message = store.add_message(id, "user", "one", None).unwrap();
        let summary = StoredSummary {
            text: "The user discussed one.".into(),
            upto_id: message,
            tokens: 9,
        };

        store.set_conversation_summary(id, &summary).unwrap();
        assert_eq!(store.conversation_summary(id).unwrap(), Some(summary));

        store.clear_conversation_summary(id).unwrap();
        assert_eq!(store.conversation_summary(id).unwrap(), None);
    }

    #[test]
    fn deleting_turns_invalidates_the_summary() {
        let store = Store::open_in_memory().unwrap();
        let id = store.create_conversation("model", "title").unwrap();
        let user1 = store.add_message(id, "user", "one", None).unwrap();
        let assistant1 = store
            .add_message(id, "assistant", "two", Some(&stats(2, 3)))
            .unwrap();
        let summary = StoredSummary {
            text: "summary".into(),
            upto_id: assistant1,
            tokens: 3,
        };

        store.set_conversation_summary(id, &summary).unwrap();
        store.delete_turn(id, user1).unwrap();
        assert_eq!(store.conversation_summary(id).unwrap(), None);
    }

    #[test]
    fn truncating_from_a_summarized_message_invalidates_the_summary() {
        let store = Store::open_in_memory().unwrap();
        let id = store.create_conversation("model", "title").unwrap();
        store.add_message(id, "user", "one", None).unwrap();
        let assistant1 = store
            .add_message(id, "assistant", "two", Some(&stats(2, 3)))
            .unwrap();
        store.add_message(id, "user", "three", None).unwrap();
        store
            .add_message(id, "assistant", "four", Some(&stats(4, 5)))
            .unwrap();

        let summary = StoredSummary {
            text: "summary".into(),
            upto_id: assistant1,
            tokens: 3,
        };
        store.set_conversation_summary(id, &summary).unwrap();
        store.delete_from_message(id, assistant1).unwrap();
        assert_eq!(store.conversation_summary(id).unwrap(), None);
    }

    #[test]
    fn deleting_recent_turns_keeps_the_summary() {
        let store = Store::open_in_memory().unwrap();
        let id = store.create_conversation("model", "title").unwrap();
        store.add_message(id, "user", "one", None).unwrap();
        let assistant1 = store
            .add_message(id, "assistant", "two", Some(&stats(2, 3)))
            .unwrap();
        let user2 = store.add_message(id, "user", "three", None).unwrap();
        store
            .add_message(id, "assistant", "four", Some(&stats(4, 5)))
            .unwrap();

        let summary = StoredSummary {
            text: "summary".into(),
            upto_id: assistant1,
            tokens: 3,
        };
        store.set_conversation_summary(id, &summary).unwrap();
        store.delete_turn(id, user2).unwrap();
        assert_eq!(store.conversation_summary(id).unwrap(), Some(summary));
    }

    #[test]
    fn stores_and_searches_memories_by_pool() {
        let store = Store::open_in_memory().unwrap();
        let id = store.create_conversation("model", "title").unwrap();

        assert!(store
            .add_memory(
                MemoryPool::Local,
                "all-minilm",
                "model",
                Some(id),
                "The user prefers dark mode.",
                &[1.0, 0.0],
            )
            .unwrap());
        assert!(store
            .add_memory(
                MemoryPool::Cloud,
                "all-minilm",
                "cloud-model",
                None,
                "The user's project is called NeuroPipe.",
                &[1.0, 0.0],
            )
            .unwrap());

        assert_eq!(store.memory_counts().unwrap(), (1, 1));
        let local = store
            .memory_entries(MemoryPool::Local, "all-minilm")
            .unwrap();
        assert_eq!(local.len(), 1);
        assert_eq!(local[0].text, "The user prefers dark mode.");
        let hits = crate::memory::search_entries(&local, &[1.0, 0.0], 4, 0.0, None);
        assert_eq!(hits.len(), 1);

        let cloud = store
            .memory_entries(MemoryPool::Cloud, "all-minilm")
            .unwrap();
        assert_eq!(cloud.len(), 1);
        assert_eq!(cloud[0].text, "The user's project is called NeuroPipe.");
    }

    #[test]
    fn memory_dedup_ignores_duplicate_text() {
        let store = Store::open_in_memory().unwrap();
        assert!(store
            .add_memory(
                MemoryPool::Local,
                "all-minilm",
                "model",
                None,
                "Same note.",
                &[1.0],
            )
            .unwrap());
        assert!(!store
            .add_memory(
                MemoryPool::Local,
                "all-minilm",
                "model",
                None,
                "Same note.",
                &[1.0],
            )
            .unwrap());
        assert_eq!(store.memory_count(MemoryPool::Local).unwrap(), 1);
    }

    #[test]
    fn memory_entries_are_scoped_by_embedding_model() {
        let store = Store::open_in_memory().unwrap();
        store
            .add_memory(
                MemoryPool::Local,
                "all-minilm",
                "model",
                None,
                "Note.",
                &[1.0, 0.0],
            )
            .unwrap();
        assert_eq!(
            store
                .memory_entries(MemoryPool::Local, "all-minilm")
                .unwrap()
                .len(),
            1
        );
        assert!(store
            .memory_entries(MemoryPool::Local, "nomic-embed-text")
            .unwrap()
            .is_empty());
    }

    #[test]
    fn clears_memory_pools() {
        let store = Store::open_in_memory().unwrap();
        for (pool, text) in [
            (MemoryPool::Local, "local note"),
            (MemoryPool::Cloud, "cloud note"),
        ] {
            store
                .add_memory(pool, "all-minilm", "model", None, text, &[1.0])
                .unwrap();
        }

        assert_eq!(store.clear_memories(Some(MemoryPool::Local)).unwrap(), 1);
        assert_eq!(store.memory_counts().unwrap(), (0, 1));
        assert!(store
            .memory_entries(MemoryPool::Local, "all-minilm")
            .unwrap()
            .is_empty());

        assert_eq!(store.clear_memories(None).unwrap(), 1);
        assert_eq!(store.memory_counts().unwrap(), (0, 0));
    }

    #[test]
    fn resolves_relative_date_hints() {
        let store = Store::open_in_memory().unwrap();
        let today = store.local_date().unwrap();
        assert_eq!(today.len(), 10);

        let (start, end) = store.resolve_date_hint(DateHint::Today).unwrap();
        assert_eq!(
            (start.as_str(), end.as_str()),
            (today.as_str(), today.as_str())
        );

        let (start, end) = store.resolve_date_hint(DateHint::LastWeekday(5)).unwrap();
        assert_eq!(start, end);
        let weekday: String = store
            .conn
            .query_row("SELECT strftime('%w', ?1)", params![start], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(weekday, "5");

        let (start, end) = store.resolve_date_hint(DateHint::LastWeek).unwrap();
        assert!(start <= end);
        let (start, end) = store.resolve_date_hint(DateHint::LastMonth).unwrap();
        assert!(start <= end);
        let (start, end) = store.resolve_date_hint(DateHint::LastYear).unwrap();
        assert!(start <= end);

        let (start, end) = store
            .resolve_date_hint(DateHint::Exact("2026-11-06".into()))
            .unwrap();
        assert_eq!((start.as_str(), end.as_str()), ("2026-11-06", "2026-11-06"));
        assert!(store
            .resolve_date_hint(DateHint::Exact("2026-99-99".into()))
            .is_err());
    }

    #[test]
    fn tracks_memory_checkpoint_and_messages_after() {
        let store = Store::open_in_memory().unwrap();
        let id = store.create_conversation("model", "title").unwrap();
        let first = store.add_message(id, "user", "one", None).unwrap();
        let second = store.add_message(id, "assistant", "two", None).unwrap();
        let third = store.add_message(id, "user", "three", None).unwrap();

        assert_eq!(store.conversation_memory_checkpoint(id).unwrap(), 0);
        assert_eq!(store.latest_message_id(id).unwrap(), Some(third));

        store
            .set_conversation_memory_checkpoint(id, second)
            .unwrap();
        assert_eq!(store.conversation_memory_checkpoint(id).unwrap(), second);

        let messages = store.messages_after(id, second).unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].id, third);
        assert_eq!(store.messages_after(id, first).unwrap().len(), 2);
    }
}
