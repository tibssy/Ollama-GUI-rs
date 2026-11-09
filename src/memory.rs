use crate::store::StoredMessage;

/// Minimum cosine similarity for a stored note to be injected.
pub const MEMORY_MIN_SCORE: f32 = 0.35;
/// Notes at or above this similarity are treated as duplicates.
pub const MEMORY_DUPLICATE_SCORE: f32 = 0.90;
/// Shorter summaries are considered empty.
pub const MEMORY_MIN_SUMMARY_CHARS: usize = 24;

/// Cap the injected memory context at this fraction of `num_ctx`.
const MEMORY_CONTEXT_RATIO: f64 = 0.08;
const MEMORY_CONTEXT_MIN_TOKENS: u32 = 128;
const MEMORY_CONTEXT_MAX_TOKENS: u32 = 1024;

const MEMORY_HEADER: &str = "Relevant long-term memory from prior sessions. Use it only when \
helpful and do not fabricate details.";

/// Lines or sentences that are extraction commentary rather than durable facts.
const META_PHRASES: &[&str] = &[
    "no durable",
    "no information",
    "no additional",
    "nothing worth",
    "did not provide",
    "didn't provide",
    "was not mentioned",
    "assistant responded",
    "assistant asked",
    "the assistant",
    "user asked",
    "user mentioned",
    "offer:",
    "offered",
    "ready to",
    "help with",
    "let me know",
    "would you like",
    "want me to",
    "for future reference",
    "memory note",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryPool {
    Local,
    Cloud,
}

impl MemoryPool {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::Cloud => "cloud",
        }
    }
}

#[derive(Debug, Clone)]
pub struct MemoryEntry {
    pub text: String,
    pub vector: Vec<f32>,
    /// Local date the note was captured, formatted by SQLite as YYYY-MM-DD.
    pub created: String,
}

#[derive(Debug, Clone)]
pub struct MemoryHit {
    pub text: String,
    pub created: String,
    pub score: f32,
}

pub fn encode_vector(vector: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(vector.len() * 4);
    for value in vector {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

pub fn decode_vector(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a.sqrt() * norm_b.sqrt())
    }
}

pub fn search_entries(
    entries: &[MemoryEntry],
    query: &[f32],
    top_k: usize,
    min_score: f32,
) -> Vec<MemoryHit> {
    if query.is_empty() || top_k == 0 {
        return Vec::new();
    }

    let mut hits: Vec<MemoryHit> = entries
        .iter()
        .filter(|entry| entry.vector.len() == query.len())
        .map(|entry| MemoryHit {
            text: entry.text.clone(),
            created: entry.created.clone(),
            score: cosine(query, &entry.vector),
        })
        .filter(|hit| hit.score >= min_score)
        .collect();
    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    hits.truncate(top_k);
    hits
}

pub fn max_similarity(entries: &[MemoryEntry], vector: &[f32]) -> f32 {
    entries
        .iter()
        .filter(|entry| entry.vector.len() == vector.len())
        .map(|entry| cosine(vector, &entry.vector))
        .fold(0.0, f32::max)
}

pub fn memory_context_budget(num_ctx: u32) -> u32 {
    let budget = (num_ctx as f64 * MEMORY_CONTEXT_RATIO) as u32;
    let ceiling = (num_ctx / 4).max(MEMORY_CONTEXT_MIN_TOKENS);
    budget
        .clamp(MEMORY_CONTEXT_MIN_TOKENS, MEMORY_CONTEXT_MAX_TOKENS)
        .min(ceiling)
}

pub fn format_memory_context(hits: &[MemoryHit], budget_tokens: u32) -> Option<String> {
    if hits.is_empty() || budget_tokens == 0 {
        return None;
    }

    let mut blocks = Vec::new();
    let mut used = estimate_tokens(MEMORY_HEADER);
    for hit in hits {
        let text = hit.text.trim();
        if text.is_empty() {
            continue;
        }
        let block = format!("[{}]\n{text}", hit.created);
        let tokens = estimate_tokens(&block).saturating_add(2);
        if used.saturating_add(tokens) > budget_tokens {
            break;
        }
        used = used.saturating_add(tokens);
        blocks.push(block);
    }

    if blocks.is_empty() {
        return None;
    }
    Some(format!("{MEMORY_HEADER}\n{}", blocks.join("\n\n")))
}

/// Keep only the bullet lines that state durable facts. Small models often add
/// meta-commentary such as "No durable facts were found"; that must not become
/// a memory.
pub fn sanitize_note(text: &str, max_chars: usize) -> String {
    let mut lines = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        let content = line
            .strip_prefix("- ")
            .or_else(|| line.strip_prefix("* "))
            .or_else(|| line.strip_prefix("• "))
            .map(str::trim)
            .filter(|content| content.chars().count() >= 3);
        let Some(content) = content else {
            continue;
        };
        let lower = content.to_lowercase();
        if META_PHRASES.iter().any(|phrase| lower.contains(phrase)) {
            continue;
        }
        lines.push(format!("- {content}"));
    }

    if !lines.is_empty() {
        return clamp_text(&lines.join("\n"), max_chars);
    }

    // Models occasionally ignore the bullet-list request. Keep a plain note
    // only when it does not look like extraction commentary.
    let lower = text.to_lowercase();
    if text.chars().count() >= MEMORY_MIN_SUMMARY_CHARS
        && !META_PHRASES.iter().any(|phrase| lower.contains(phrase))
    {
        clamp_text(text, max_chars)
    } else {
        String::new()
    }
}

pub fn summary_prompt(max_chars: usize) -> String {
    format!(
        "You are a memory extraction function. Read only the user's messages in the transcript \
         and output only a bullet list of durable facts and explicit preferences the user \
         stated about themselves that would help in future conversations. Ignore the \
         assistant's replies, suggestions, advice, questions, and offers of help. Do not \
         answer the conversation and do not include greetings, filler, transient details, or \
         raw code. Example:\n- The user's name is Alex.\n- The user prefers dark mode.\n\
         Max {max_chars} characters. If there is nothing durable, output nothing."
    )
}

pub fn clamp_text(text: &str, max_chars: usize) -> String {
    let text = text.trim();
    if max_chars == 0 {
        return String::new();
    }
    if text.chars().count() <= max_chars {
        return text.to_string();
    }

    let mut out: String = text.chars().take(max_chars.saturating_sub(3)).collect();
    out.push_str("...");
    out
}

/// Format the user's messages as a transcript, keeping the most recent ones
/// that fit within `max_tokens`. Assistant replies are intentionally excluded
/// so their suggestions and advice are not stored as user preferences.
pub fn build_transcript(messages: &[StoredMessage], max_tokens: u32) -> String {
    let mut lines = Vec::new();
    let mut used = 0u32;

    for message in messages.iter().rev() {
        if message.role != "user" {
            continue;
        }
        let line = format!("User: {}\n", message.content.trim());
        let tokens = estimate_tokens(&line);
        if !lines.is_empty() && used.saturating_add(tokens) > max_tokens {
            break;
        }
        used = used.saturating_add(tokens);
        lines.push(line);
    }

    lines.reverse();
    lines.join("")
}

pub fn model_is_cloud(model: &str, remote_host: Option<&str>, base_url: &str) -> bool {
    if remote_host.is_some_and(|host| !host.trim().is_empty()) {
        return true;
    }
    if model.ends_with(":cloud") || model.ends_with("-cloud") {
        return true;
    }
    base_url_is_remote(base_url)
}

pub fn model_names_match(a: &str, b: &str) -> bool {
    fn normalize(name: &str) -> &str {
        name.strip_suffix(":latest").unwrap_or(name)
    }
    normalize(a) == normalize(b)
}

fn base_url_is_remote(base_url: &str) -> bool {
    let Ok(url) = reqwest::Url::parse(base_url.trim()) else {
        return false;
    };
    match url.host_str() {
        Some(host) => !matches!(host, "localhost" | "127.0.0.1" | "::1" | "[::1]"),
        None => false,
    }
}

fn estimate_tokens(text: &str) -> u32 {
    let mut tokens = 0.0f64;
    for character in text.chars() {
        tokens += if character.is_ascii() { 0.25 } else { 1.0 };
    }
    tokens.ceil() as u32
}

#[cfg(test)]
mod tests {
    use super::{
        build_transcript, clamp_text, cosine, decode_vector, encode_vector, format_memory_context,
        max_similarity, memory_context_budget, model_is_cloud, model_names_match, sanitize_note,
        search_entries, MemoryEntry, MemoryHit, MEMORY_MIN_SCORE,
    };
    use crate::store::StoredMessage;

    fn entry(text: &str, vector: Vec<f32>, created: &str) -> MemoryEntry {
        MemoryEntry {
            text: text.to_string(),
            vector,
            created: created.to_string(),
        }
    }

    fn message(id: i64, role: &str, content: &str) -> StoredMessage {
        StoredMessage {
            id,
            role: role.to_string(),
            content: content.to_string(),
        }
    }

    #[test]
    fn vector_roundtrip_preserves_values() {
        let vector = vec![0.25, -1.5, 3.0];
        assert_eq!(decode_vector(&encode_vector(&vector)), vector);
        assert!(decode_vector(&[1, 2, 3]).is_empty());
    }

    #[test]
    fn cosine_ranks_closest_first() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 0.0001);
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 0.0001);
        assert_eq!(cosine(&[1.0], &[1.0, 0.0]), 0.0);
    }

    #[test]
    fn search_filters_by_score_and_limit() {
        let entries = vec![
            entry("close", vec![1.0, 0.0], "2026-11-09"),
            entry("far", vec![0.0, 1.0], "2026-11-08"),
            entry("also close", vec![0.9, 0.1], "2026-11-07"),
        ];
        let hits = search_entries(&entries, &[1.0, 0.0], 1, MEMORY_MIN_SCORE);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].text, "close");

        let hits = search_entries(&entries, &[1.0, 0.0], 10, MEMORY_MIN_SCORE);
        assert_eq!(hits.len(), 2);
        assert!(hits[0].score >= hits[1].score);
    }

    #[test]
    fn similarity_dedup_detects_near_duplicates() {
        let entries = vec![entry("note", vec![1.0, 0.0], "2026-11-09")];
        assert!(max_similarity(&entries, &[1.0, 0.0]) > 0.95);
        assert!(max_similarity(&entries, &[0.0, 1.0]) < 0.1);
    }

    #[test]
    fn memory_context_respects_budget() {
        assert_eq!(memory_context_budget(8192), 655);
        assert_eq!(memory_context_budget(512), 128);

        let hits = vec![
            MemoryHit {
                text: "The user prefers dark mode.".into(),
                created: "2026-11-09".into(),
                score: 0.9,
            },
            MemoryHit {
                text: "The user works on a Rust GTK app.".into(),
                created: "2026-11-08".into(),
                score: 0.8,
            },
        ];
        let context = format_memory_context(&hits, 1024).expect("context");
        assert!(context.contains("dark mode"));
        assert!(context.contains("Rust GTK"));
        assert!(format_memory_context(&hits, 4).is_none());
    }

    #[test]
    fn clamps_long_text() {
        assert_eq!(clamp_text("  short  ", 20), "short");
        assert_eq!(clamp_text("abcdefghij", 6), "abc...");
        assert_eq!(clamp_text("anything", 0), "");
    }

    #[test]
    fn sanitizes_extraction_commentary() {
        let note = "- Favorite editor: Neovim\n- Home city: Budapest\n\n- User mentioned preferences.\n\nNo additional durable facts were found.";
        assert_eq!(
            sanitize_note(note, 1200),
            "- Favorite editor: Neovim\n- Home city: Budapest"
        );
        assert_eq!(
            sanitize_note("- The user's favorite color is teal.", 1200),
            "- The user's favorite color is teal."
        );
        assert!(sanitize_note(
            "- The user asked about their favorite color but did not provide an answer.\nNo information was given.",
            1200
        )
        .is_empty());
    }

    #[test]
    fn transcript_keeps_recent_messages() {
        let messages = vec![
            message(1, "user", "old question"),
            message(2, "assistant", "old answer"),
            message(3, "user", "new question"),
            message(4, "assistant", "new answer"),
        ];
        let transcript = build_transcript(&messages, 8);
        assert!(!transcript.contains("old question"));
        assert!(transcript.contains("new question"));
        assert!(!transcript.contains("new answer"));
    }

    #[test]
    fn cloud_detection_uses_remote_host_suffix_and_url() {
        assert!(model_is_cloud(
            "gpt-oss:120b",
            Some("https://ollama.com"),
            "http://localhost:11434"
        ));
        assert!(model_is_cloud(
            "gpt-oss:120b-cloud",
            None,
            "http://localhost:11434"
        ));
        assert!(model_is_cloud("llama3", None, "https://ollama.example.com"));
        assert!(!model_is_cloud("llama3", None, "http://localhost:11434"));
        assert!(!model_is_cloud("llama3", None, "http://127.0.0.1:11434"));
    }

    #[test]
    fn model_names_match_latest_tag() {
        assert!(model_names_match("all-minilm", "all-minilm:latest"));
        assert!(model_names_match("all-minilm:latest", "all-minilm"));
        assert!(!model_names_match("all-minilm", "nomic-embed-text"));
    }
}
