use crate::store::StoredMessage;

/// Minimum cosine similarity for a stored note to be injected.
pub const MEMORY_MIN_SCORE: f32 = 0.35;
/// Notes at or above this similarity are treated as duplicates.
pub const MEMORY_DUPLICATE_SCORE: f32 = 0.90;
/// Shorter summaries are considered empty.
pub const MEMORY_MIN_SUMMARY_CHARS: usize = 24;
/// Ranking bonus for notes captured inside a date the user asked about.
const MEMORY_DATE_BOOST: f32 = 0.10;

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

/// A date expression found in a user query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DateHint {
    Exact(String),
    Today,
    Yesterday,
    DaysAgo(u32),
    /// SQLite weekday number: 0 = Sunday, 1 = Monday, ... 6 = Saturday.
    LastWeekday(u8),
    LastWeek,
    LastMonth,
    LastYear,
}

/// Find a date expression in a user query. Explicit YYYY-MM-DD dates win,
/// then weekdays (so "last Friday" wins over "last week"), then broader
/// phrases.
pub fn date_hint(query: &str) -> Option<DateHint> {
    let lower = query.to_lowercase();
    if let Some(date) = first_date(&lower) {
        return Some(DateHint::Exact(date.to_string()));
    }
    if lower.contains("day before yesterday") {
        return Some(DateHint::DaysAgo(2));
    }

    for (name, weekday) in [
        ("monday", 1),
        ("tuesday", 2),
        ("wednesday", 3),
        ("thursday", 4),
        ("friday", 5),
        ("saturday", 6),
        ("sunday", 0),
    ] {
        if lower.contains(&format!("last {name}")) || lower.contains(&format!("{name} last week")) {
            return Some(DateHint::LastWeekday(weekday));
        }
    }

    if lower.contains("yesterday") {
        return Some(DateHint::Yesterday);
    }
    if lower.contains("today") || lower.contains("tonight") || lower.contains("this morning") {
        return Some(DateHint::Today);
    }
    if lower.contains("last week") {
        return Some(DateHint::LastWeek);
    }
    if lower.contains("last month") {
        return Some(DateHint::LastMonth);
    }
    if lower.contains("last year") {
        return Some(DateHint::LastYear);
    }
    parse_days_ago(&lower).map(DateHint::DaysAgo)
}

fn parse_days_ago(lower: &str) -> Option<u32> {
    if lower.contains("a week ago") {
        return Some(7);
    }
    for (word, days) in [
        ("one", 1),
        ("two", 2),
        ("three", 3),
        ("four", 4),
        ("five", 5),
        ("six", 6),
        ("seven", 7),
        ("eight", 8),
        ("nine", 9),
        ("ten", 10),
    ] {
        if lower.contains(&format!("{word} days ago")) {
            return Some(days);
        }
    }

    let tokens: Vec<&str> = lower.split_whitespace().collect();
    for window in tokens.windows(3) {
        if window[1] == "days" && window[2] == "ago" {
            if let Ok(days) = window[0].parse::<u32>() {
                if (1..=31).contains(&days) {
                    return Some(days);
                }
            }
        }
    }
    None
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
    date_range: Option<(&str, &str)>,
) -> Vec<MemoryHit> {
    if query.is_empty() || top_k == 0 {
        return Vec::new();
    }

    let mut scored: Vec<(MemoryHit, bool)> = entries
        .iter()
        .filter(|entry| entry.vector.len() == query.len())
        .filter_map(|entry| {
            let semantic = cosine(query, &entry.vector);
            if semantic < min_score {
                return None;
            }
            let in_range = date_range.is_some_and(|(start, end)| {
                let capture_in_range =
                    entry.created.as_str() >= start && entry.created.as_str() <= end;
                capture_in_range || note_has_date_in_range(&entry.text, start, end)
            });
            Some((
                MemoryHit {
                    text: entry.text.clone(),
                    created: entry.created.clone(),
                    score: semantic + if in_range { MEMORY_DATE_BOOST } else { 0.0 },
                },
                in_range,
            ))
        })
        .collect();

    // When the user names a date, prefer notes captured on that date or notes
    // whose text explicitly mentions it. Fall back to all hits if nothing
    // matches, so a missing date does not hide relevant context.
    if date_range.is_some() && scored.iter().any(|(_, in_range)| *in_range) {
        scored.retain(|(_, in_range)| *in_range);
    }

    let mut hits: Vec<MemoryHit> = scored.into_iter().map(|(hit, _)| hit).collect();
    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    hits.truncate(top_k);
    hits
}

/// Find YYYY-MM-DD dates in note text and check whether any fall in the range.
fn note_has_date_in_range(text: &str, start: &str, end: &str) -> bool {
    text_dates(text).any(|candidate| candidate >= start && candidate <= end)
}

fn first_date(text: &str) -> Option<&str> {
    text_dates(text).next()
}

fn text_dates(text: &str) -> impl Iterator<Item = &str> {
    text.as_bytes().windows(10).filter_map(|window| {
        let candidate = std::str::from_utf8(window).ok()?;
        let bytes = candidate.as_bytes();
        let looks_like_date = bytes[4] == b'-'
            && bytes[7] == b'-'
            && bytes[..4].iter().all(u8::is_ascii_digit)
            && bytes[5..7].iter().all(u8::is_ascii_digit)
            && bytes[8..10].iter().all(u8::is_ascii_digit);
        looks_like_date.then_some(candidate)
    })
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

pub fn summary_prompt(max_chars: usize, date: &str) -> String {
    format!(
        "You are a memory extraction function. The conversation date is {date}. Read only the \
         user's messages in the transcript and output only a bullet list of durable facts and \
         explicit preferences the user stated about themselves that would help in future \
         conversations. Preserve dates and resolve relative dates such as 'yesterday' or \
         'last Friday' to absolute dates (YYYY-MM-DD) in the notes. Ignore the assistant's \
         replies, suggestions, advice, questions, and offers of help. Do not answer the \
         conversation and do not include greetings, filler, transient details, or raw code. \
         Example:\n- The user's name is Alex.\n- The user prefers dark mode.\n\
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
        build_transcript, clamp_text, cosine, date_hint, decode_vector, encode_vector,
        format_memory_context, max_similarity, memory_context_budget, model_is_cloud,
        model_names_match, sanitize_note, search_entries, DateHint, MemoryEntry, MemoryHit,
        MEMORY_MIN_SCORE,
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
        let hits = search_entries(&entries, &[1.0, 0.0], 1, MEMORY_MIN_SCORE, None);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].text, "close");

        let hits = search_entries(&entries, &[1.0, 0.0], 10, MEMORY_MIN_SCORE, None);
        assert_eq!(hits.len(), 2);
        assert!(hits[0].score >= hits[1].score);
    }

    #[test]
    fn search_prefers_notes_from_the_requested_date() {
        let entries = vec![
            entry("recipe from Friday", vec![1.0, 0.0], "2026-11-06"),
            entry("recipe from earlier", vec![1.0, 0.0], "2026-11-01"),
        ];
        let hits = search_entries(
            &entries,
            &[1.0, 0.0],
            2,
            0.0,
            Some(("2026-11-06", "2026-11-06")),
        );
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].created, "2026-11-06");
    }

    #[test]
    fn search_matches_dates_mentioned_in_note_text() {
        let entries = vec![
            entry("lasagna on 2026-11-06", vec![1.0, 0.0], "2026-11-09"),
            entry("pizza on 2026-10-30", vec![1.0, 0.0], "2026-10-30"),
        ];
        let hits = search_entries(
            &entries,
            &[1.0, 0.0],
            4,
            0.0,
            Some(("2026-11-06", "2026-11-06")),
        );
        assert_eq!(hits.len(), 1);
        assert!(hits[0].text.contains("lasagna"));
    }

    #[test]
    fn finds_relative_dates_in_queries() {
        assert_eq!(
            date_hint("Do you remember what I cooked last Friday?"),
            Some(DateHint::LastWeekday(5))
        );
        assert_eq!(
            date_hint("What did we talk about yesterday?"),
            Some(DateHint::Yesterday)
        );
        assert_eq!(
            date_hint("What did we discuss today?"),
            Some(DateHint::Today)
        );
        assert_eq!(date_hint("Show me last month"), Some(DateHint::LastMonth));
        assert_eq!(date_hint("three days ago"), Some(DateHint::DaysAgo(3)));
        assert_eq!(
            date_hint("the day before yesterday"),
            Some(DateHint::DaysAgo(2))
        );
        assert_eq!(
            date_hint("What did I cook on 2026-11-06?"),
            Some(DateHint::Exact("2026-11-06".into()))
        );
        assert_eq!(date_hint("hello there"), None);
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
