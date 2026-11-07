use anyhow::{Context, Result};
use rusqlite::{params, Connection, OptionalExtension};

use crate::api::StreamStats;

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

pub struct Store {
    conn: Connection,
}

impl Store {
    pub fn open() -> Result<Self> {
        let dir = crate::config::data_dir();
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("creating data directory {}", dir.display()))?;
        let path = dir.join("chats.db");
        let conn = Connection::open(&path)
            .with_context(|| format!("opening database {}", path.display()))?;
        Self::from_connection(conn)
    }

    fn from_connection(conn: Connection) -> Result<Self> {
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        let store = Self { conn };
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
            "#,
        )?;
        self.ensure_column("conversations", "context_used", "INTEGER")?;
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
        let mut stmt = self.conn.prepare(
            "SELECT id, role, content FROM messages
             WHERE conversation_id = ?1 ORDER BY id ASC",
        )?;
        let rows = stmt.query_map(params![conversation_id], |row| {
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
    use super::{context_used_from_json, Store};
    use crate::api::StreamStats;

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
}
