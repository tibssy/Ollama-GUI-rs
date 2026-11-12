pub mod stream;

use std::time::Duration;

use anyhow::{bail, Context, Result};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};

use stream::NdjsonBuffer;

#[derive(Debug, Clone, Deserialize)]
pub struct ModelSummary {
    pub name: String,
    #[allow(dead_code)]
    #[serde(default)]
    pub size: u64,
    #[allow(dead_code)]
    #[serde(default)]
    pub details: ModelDetails,
    /// Ollama 0.3+ reports what a model can do, e.g. "completion",
    /// "embedding", "decision", "vision", "tools".
    #[serde(default)]
    pub capabilities: Vec<String>,
    /// Set by Ollama for models hosted on ollama.com rather than pulled locally.
    #[serde(default)]
    pub remote_host: Option<String>,
}

impl ModelSummary {
    /// Chat-capable models report "completion" (or "chat" on some versions).
    /// Decision and embedding models are excluded. Models from older Ollama
    /// builds may not report capabilities at all; those are kept.
    pub fn is_chat_model(&self) -> bool {
        if self.capabilities.is_empty() {
            return true;
        }
        let mut chat_capable = false;
        for capability in &self.capabilities {
            match capability.as_str() {
                "completion" | "chat" => chat_capable = true,
                "decision" | "embedding" => return false,
                _ => {}
            }
        }
        chat_capable
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ModelDetails {
    #[serde(default)]
    pub parent_model: String,
    #[serde(default)]
    pub format: String,
    #[serde(default)]
    pub family: String,
    #[serde(default, deserialize_with = "deserialize_vec_or_null")]
    pub families: Vec<String>,
    #[serde(default)]
    pub parameter_size: String,
    #[serde(default)]
    pub quantization_level: String,
}

/// Parsed `/api/show` response used by the model details dialog.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ModelShow {
    #[serde(default)]
    pub details: ModelDetails,
    #[serde(default)]
    pub capabilities: Vec<String>,
    #[serde(default)]
    pub model_info: serde_json::Map<String, serde_json::Value>,
    #[serde(default)]
    pub parameters: Option<String>,
    #[serde(default)]
    pub license: Option<String>,
    #[serde(default)]
    pub modified_at: Option<String>,
    #[serde(default)]
    pub requires: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChatOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_ctx: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_predict: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    /// Ollama returns an id for each call and accepts it on the follow-up turn.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub function: ToolFunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolFunctionCall {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    /// Base64-encoded images for vision-capable Ollama models.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<String>,
    /// Tool calls made by the assistant. Sent back when continuing a tool loop.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    /// Set on `role: "tool"` messages so Ollama can match the result to the call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
}

impl ChatMessage {
    pub fn new(role: &str, content: impl Into<String>) -> Self {
        Self {
            role: role.to_string(),
            content: content.into(),
            images: Vec::new(),
            tool_calls: Vec::new(),
            tool_name: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum ThinkSetting {
    Bool(bool),
    Level(String),
}

#[derive(Debug, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<ChatOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub think: Option<ThinkSetting>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, Default)]
pub struct StreamStats {
    pub eval_count: Option<u32>,
    pub eval_duration: Option<u64>,
    pub prompt_eval_count: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThinkingControl {
    pub supports_off: bool,
    pub levels: Vec<String>,
    pub default: Option<String>,
}

impl ThinkingControl {
    pub fn is_level_only(&self) -> bool {
        !self.supports_off && !self.levels.is_empty()
    }

    pub fn lowest_level(&self) -> Option<&str> {
        self.levels.first().map(String::as_str)
    }

    pub fn preferred_level(&self, stored: &str) -> String {
        if self.levels.iter().any(|level| level == stored) {
            return stored.to_string();
        }
        if let Some(default) = self
            .default
            .as_deref()
            .filter(|default| self.levels.iter().any(|level| level == default))
        {
            return default.to_string();
        }
        self.levels.first().cloned().unwrap_or_default()
    }
}

#[derive(Debug, Clone)]
pub struct ChatReply {
    pub content: String,
    pub stats: StreamStats,
}

impl StreamStats {
    pub fn tokens_per_second(&self) -> Option<f64> {
        let count = self.eval_count? as f64;
        let seconds = self.eval_duration? as f64 / 1_000_000_000.0;
        (seconds > 0.0).then(|| count / seconds)
    }
}

/// One line of Ollama's streaming pull response.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PullProgress {
    pub status: String,
    pub digest: Option<String>,
    pub total: Option<u64>,
    pub completed: Option<u64>,
}

impl PullProgress {
    /// Fraction of the current download layer, when Ollama reports a total.
    pub fn fraction(&self) -> Option<f64> {
        let total = self.total?;
        if total == 0 {
            return None;
        }
        let completed = self.completed.unwrap_or(0);
        Some((completed as f64 / total as f64).clamp(0.0, 1.0))
    }
}

#[derive(Debug)]
pub enum StreamEvent {
    Delta(String),
    Thinking(String),
    ToolCalls(Vec<ToolCall>),
    Done(StreamStats),
    Error(String),
}

#[derive(Debug, Deserialize)]
struct ChatChunk {
    #[serde(default)]
    message: Option<ChunkMessage>,
    #[serde(default)]
    done: bool,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    eval_count: Option<u32>,
    #[serde(default)]
    eval_duration: Option<u64>,
    #[serde(default)]
    prompt_eval_count: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct ChunkMessage {
    #[serde(default)]
    content: String,
    #[serde(default)]
    thinking: String,
    #[serde(default)]
    tool_calls: Vec<ToolCall>,
}

#[derive(Clone)]
pub struct OllamaClient {
    http: reqwest::Client,
    base_url: String,
    api_key: String,
}

impl OllamaClient {
    pub fn new(base_url: &str, api_key: &str) -> Self {
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .build()
            .expect("failed to build HTTP client");
        Self {
            http,
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: api_key.trim().to_string(),
        }
    }

    fn request(&self, method: reqwest::Method, path: &str) -> reqwest::RequestBuilder {
        let builder = self
            .http
            .request(method, format!("{}{}", self.base_url, path));
        if self.api_key.is_empty() {
            builder
        } else {
            builder.bearer_auth(&self.api_key)
        }
    }

    #[allow(dead_code)]
    pub async fn version(&self) -> Result<String> {
        let response = self
            .request(reqwest::Method::GET, "/api/version")
            .send()
            .await
            .context("could not reach Ollama")?;
        if !response.status().is_success() {
            bail!("Ollama returned HTTP {}", response.status());
        }
        let body: serde_json::Value = response.json().await?;
        Ok(body
            .get("version")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string())
    }

    pub async fn list_models(&self) -> Result<Vec<ModelSummary>> {
        let response = self
            .request(reqwest::Method::GET, "/api/tags")
            .send()
            .await
            .context("could not reach Ollama")?;
        if !response.status().is_success() {
            bail!("Ollama returned HTTP {}", response.status());
        }
        let body: serde_json::Value = response.json().await?;
        let models = body
            .get("models")
            .cloned()
            .unwrap_or_else(|| serde_json::Value::Array(Vec::new()));
        serde_json::from_value(models).context("unexpected /api/tags payload")
    }

    /// Fetch the model description used by the details dialog.
    pub async fn show_model(&self, model: &str) -> Result<ModelShow> {
        let response = self
            .request(reqwest::Method::POST, "/api/show")
            .json(&serde_json::json!({ "model": model }))
            .send()
            .await
            .context("could not reach Ollama")?;
        let response = ensure_success(response, "show").await?;
        response
            .json()
            .await
            .context("unexpected /api/show payload")
    }

    /// Fetch the model's thinking controls from `/api/show`.
    pub async fn model_thinking(&self, model: &str) -> Result<Option<ThinkingControl>> {
        let response = self
            .request(reqwest::Method::POST, "/api/show")
            .json(&serde_json::json!({ "model": model }))
            .send()
            .await
            .context("could not reach Ollama")?;
        if !response.status().is_success() {
            bail!("Ollama returned HTTP {}", response.status());
        }
        let body: serde_json::Value = response.json().await?;
        Ok(parse_thinking_control(&body))
    }

    pub async fn chat(&self, request: &ChatRequest) -> Result<reqwest::Response> {
        let response = self
            .request(reqwest::Method::POST, "/api/chat")
            .json(request)
            .send()
            .await
            .context("could not reach Ollama")?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            let detail = body.trim();
            if detail.is_empty() {
                bail!("Ollama returned HTTP {status}");
            }
            bail!("Ollama returned HTTP {status}: {detail}");
        }
        Ok(response)
    }

    /// Run a non-streaming chat request. Used for background tasks such as
    /// conversation summaries.
    pub async fn chat_once(&self, request: &ChatRequest) -> Result<ChatReply> {
        let response = self.chat(request).await?;
        let chunk: ChatChunk = response
            .json()
            .await
            .context("parsing Ollama chat response")?;
        if let Some(error) = chunk.error {
            bail!(error);
        }
        Ok(ChatReply {
            content: chunk
                .message
                .map(|message| message.content)
                .unwrap_or_default(),
            stats: StreamStats {
                eval_count: chunk.eval_count,
                eval_duration: chunk.eval_duration,
                prompt_eval_count: chunk.prompt_eval_count,
            },
        })
    }

    /// Embed text with an Ollama embedding model via `/api/embed`.
    pub async fn embed(&self, model: &str, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let response = tokio::time::timeout(
            Duration::from_secs(30),
            self.request(reqwest::Method::POST, "/api/embed")
                .json(&serde_json::json!({ "model": model, "input": texts }))
                .send(),
        )
        .await
        .context("embedding request timed out")??;
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            let detail = body.trim();
            if detail.is_empty() {
                bail!("Ollama embedding endpoint returned HTTP {status}");
            }
            bail!("Ollama embedding endpoint returned HTTP {status}: {detail}");
        }
        let body: serde_json::Value = response
            .json()
            .await
            .context("parsing Ollama embedding response")?;
        parse_embeddings(&body)
    }

    /// Generate speech with an Ollama speech-capable model via the
    /// OpenAI-compatible `/v1/audio/speech` endpoint. Returns WAV bytes.
    pub async fn speech(&self, model: &str, input: &str, voice: &str) -> Result<Vec<u8>> {
        let response = self
            .request(reqwest::Method::POST, "/v1/audio/speech")
            .json(&serde_json::json!({
                "model": model,
                "input": input,
                "voice": voice,
                "response_format": "wav",
            }))
            .send()
            .await
            .context("could not reach the Ollama speech endpoint")?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            let detail = body.trim();
            if detail.is_empty() {
                bail!("Ollama speech endpoint returned HTTP {status}");
            }
            bail!("Ollama speech endpoint returned HTTP {status}: {detail}");
        }
        Ok(response
            .bytes()
            .await
            .context("reading speech audio")?
            .to_vec())
    }

    /// Transcribe a 16-bit PCM WAV with an Ollama model that has the `audio`
    /// capability, using the OpenAI-compatible `/v1/audio/transcriptions`
    /// endpoint.
    pub async fn transcribe(
        &self,
        model: &str,
        wav: Vec<u8>,
        language: Option<&str>,
    ) -> Result<String> {
        let part = reqwest::multipart::Part::bytes(wav)
            .file_name("audio.wav")
            .mime_str("audio/wav")?;
        let mut form = reqwest::multipart::Form::new()
            .part("file", part)
            .text("model", model.to_string())
            .text("response_format", "json");
        if let Some(language) = language {
            if !language.is_empty() && language != "auto" {
                form = form.text("language", language.to_string());
            }
        }

        let response = self
            .request(reqwest::Method::POST, "/v1/audio/transcriptions")
            .multipart(form)
            .send()
            .await
            .context("could not reach the Ollama transcription endpoint")?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            let detail = body.trim();
            if detail.is_empty() {
                bail!("Ollama transcription endpoint returned HTTP {status}");
            }
            bail!("Ollama transcription endpoint returned HTTP {status}: {detail}");
        }

        let body: serde_json::Value = response.json().await?;
        Ok(body
            .get("text")
            .and_then(|text| text.as_str())
            .unwrap_or_default()
            .trim()
            .to_string())
    }

    /// Ask Ollama to load a model without generating anything. An empty
    /// `/api/generate` request returns `done_reason: "load"` once the weights
    /// are ready, which lets the next chat request skip the cold start.
    pub async fn preload(&self, model: &str, num_ctx: u32, keep_alive: &str) -> Result<()> {
        let response = self
            .request(reqwest::Method::POST, "/api/generate")
            .json(&serde_json::json!({
                "model": model,
                "keep_alive": keep_alive,
                "stream": false,
                "options": { "num_ctx": num_ctx.max(1) },
            }))
            .send()
            .await
            .context("could not preload the model")?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            let detail = body.trim();
            if detail.is_empty() {
                bail!("Ollama returned HTTP {status}");
            }
            bail!("Ollama returned HTTP {status}: {detail}");
        }
        let _ = response.bytes().await?;
        Ok(())
    }

    /// Pull a model and report progress as Ollama streams it.
    pub async fn pull_model<F>(&self, model: &str, mut on_progress: F) -> Result<()>
    where
        F: FnMut(PullProgress) + Send,
    {
        let response = self
            .request(reqwest::Method::POST, "/api/pull")
            .json(&serde_json::json!({ "model": model, "stream": true }))
            .send()
            .await
            .context("could not reach Ollama")?;
        let response = ensure_success(response, "pull").await?;

        let mut stream = response.bytes_stream();
        let mut buffer = NdjsonBuffer::new();
        let mut success = false;

        while let Some(chunk) = stream.next().await {
            let bytes = chunk.context("error while reading the pull stream")?;
            for line in buffer.push(&bytes) {
                let progress = parse_pull_progress(&line)?;
                success |= progress.status == "success";
                on_progress(progress);
            }
        }
        if let Some(line) = buffer.finish() {
            let progress = parse_pull_progress(&line)?;
            success |= progress.status == "success";
            on_progress(progress);
        }

        if !success {
            bail!("model pull ended before Ollama reported success");
        }
        Ok(())
    }

    /// Copy a model to another name. Used to shorten Hugging Face model names.
    pub async fn copy_model(&self, source: &str, destination: &str) -> Result<()> {
        let response = self
            .request(reqwest::Method::POST, "/api/copy")
            .json(&serde_json::json!({
                "source": source,
                "destination": destination,
            }))
            .send()
            .await
            .context("could not reach Ollama")?;
        ensure_success(response, "copy").await.map(|_| ())
    }

    /// Remove a model from Ollama.
    pub async fn delete_model(&self, model: &str) -> Result<()> {
        let response = self
            .request(reqwest::Method::DELETE, "/api/delete")
            .json(&serde_json::json!({ "model": model }))
            .send()
            .await
            .context("could not reach Ollama")?;
        ensure_success(response, "delete").await.map(|_| ())
    }
}

fn deserialize_vec_or_null<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(Option::<Vec<String>>::deserialize(deserializer)?.unwrap_or_default())
}

async fn ensure_success(response: reqwest::Response, operation: &str) -> Result<reqwest::Response> {
    if response.status().is_success() {
        return Ok(response);
    }
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    let detail = body.trim();
    if detail.is_empty() {
        bail!("Ollama {operation} returned HTTP {status}");
    }
    bail!("Ollama {operation} returned HTTP {status}: {detail}")
}

fn parse_pull_progress(line: &[u8]) -> Result<PullProgress> {
    let value: serde_json::Value =
        serde_json::from_slice(line).context("could not parse the pull stream")?;
    if let Some(error) = value.get("error").and_then(|value| value.as_str()) {
        bail!("{error}");
    }
    Ok(PullProgress {
        status: value
            .get("status")
            .and_then(|value| value.as_str())
            .unwrap_or_default()
            .to_string(),
        digest: value
            .get("digest")
            .and_then(|value| value.as_str())
            .map(str::to_string),
        total: value.get("total").and_then(|value| value.as_u64()),
        completed: value.get("completed").and_then(|value| value.as_u64()),
    })
}

/// Drives a streaming chat request on the tokio runtime and forwards parsed
/// events to the GTK main loop over an async channel.
///
/// Returns an abort handle so the UI can stop generation.
pub fn spawn_chat_stream(
    runtime: &tokio::runtime::Runtime,
    client: OllamaClient,
    request: ChatRequest,
    tx: async_channel::Sender<StreamEvent>,
) -> tokio::task::AbortHandle {
    let task = runtime.spawn(async move {
        let response = match client.chat(&request).await {
            Ok(response) => response,
            Err(err) => {
                let _ = tx.send(StreamEvent::Error(format!("{err:#}"))).await;
                return;
            }
        };

        let mut stream = response.bytes_stream();
        let mut buffer = NdjsonBuffer::new();
        let mut stats = StreamStats::default();
        let mut finished = false;

        'stream: while let Some(chunk) = stream.next().await {
            let bytes = match chunk {
                Ok(bytes) => bytes,
                Err(err) => {
                    let _ = tx
                        .send(StreamEvent::Error(format!("stream error: {err}")))
                        .await;
                    return;
                }
            };

            for line in buffer.push(&bytes) {
                match handle_line(&line, &tx, &mut stats).await {
                    Ok(true) => {
                        finished = true;
                        break 'stream;
                    }
                    Ok(false) => {}
                    Err(err) => {
                        let _ = tx.send(StreamEvent::Error(err)).await;
                        return;
                    }
                }
            }
        }

        if !finished {
            if let Some(line) = buffer.finish() {
                match handle_line(&line, &tx, &mut stats).await {
                    Ok(_) => {}
                    Err(err) => {
                        let _ = tx.send(StreamEvent::Error(err)).await;
                        return;
                    }
                }
            }
        }

        let _ = tx.send(StreamEvent::Done(stats)).await;
    });

    task.abort_handle()
}

/// Returns Ok(true) when the stream signalled completion.
async fn handle_line(
    line: &[u8],
    tx: &async_channel::Sender<StreamEvent>,
    stats: &mut StreamStats,
) -> Result<bool, String> {
    let chunk: ChatChunk = serde_json::from_slice(line).map_err(|err| {
        let text = String::from_utf8_lossy(line);
        format!("could not parse stream chunk ({err}): {text}")
    })?;

    if let Some(error) = chunk.error {
        return Err(error);
    }

    if let Some(message) = chunk.message {
        if !message.thinking.is_empty() {
            let _ = tx.send(StreamEvent::Thinking(message.thinking)).await;
        }
        if !message.content.is_empty() {
            let _ = tx.send(StreamEvent::Delta(message.content)).await;
        }
        if !message.tool_calls.is_empty() {
            let _ = tx.send(StreamEvent::ToolCalls(message.tool_calls)).await;
        }
    }

    stats.eval_count = chunk.eval_count.or(stats.eval_count);
    stats.eval_duration = chunk.eval_duration.or(stats.eval_duration);
    stats.prompt_eval_count = chunk.prompt_eval_count.or(stats.prompt_eval_count);

    Ok(chunk.done)
}

fn parse_embeddings(body: &serde_json::Value) -> Result<Vec<Vec<f32>>> {
    let embeddings = body
        .get("embeddings")
        .and_then(serde_json::Value::as_array)
        .context("Ollama embedding response missing embeddings")?;

    let mut result = Vec::with_capacity(embeddings.len());
    for embedding in embeddings {
        let values = embedding
            .as_array()
            .context("Ollama embedding was not an array")?;
        let mut vector = Vec::with_capacity(values.len());
        for value in values {
            let number = value
                .as_f64()
                .context("Ollama embedding contained a non-number")?;
            vector.push(number as f32);
        }
        result.push(vector);
    }

    if result.is_empty() {
        bail!("Ollama returned no embeddings");
    }
    Ok(result)
}

fn parse_thinking_control(body: &serde_json::Value) -> Option<ThinkingControl> {
    let thinking = body.get("thinking")?;
    let mut supports_off = false;
    let mut levels = Vec::new();

    if let Some(values) = thinking.get("values").and_then(serde_json::Value::as_array) {
        for value in values {
            match value {
                serde_json::Value::Bool(false) => supports_off = true,
                serde_json::Value::String(level) => levels.push(level.clone()),
                _ => {}
            }
        }
    }

    let default = thinking
        .get("default")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string);

    Some(ThinkingControl {
        supports_off,
        levels,
        default,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_tokens_per_second() {
        let stats = StreamStats {
            eval_count: Some(100),
            eval_duration: Some(2_000_000_000),
            prompt_eval_count: None,
        };
        assert_eq!(stats.tokens_per_second(), Some(50.0));
    }

    #[test]
    fn pull_progress_computes_fraction() {
        let progress = PullProgress {
            completed: Some(25),
            total: Some(100),
            ..Default::default()
        };
        assert_eq!(progress.fraction(), Some(0.25));
        assert_eq!(
            PullProgress {
                completed: Some(5),
                total: Some(0),
                ..Default::default()
            }
            .fraction(),
            None
        );
    }

    #[test]
    fn parses_show_payload() {
        let show: ModelShow = serde_json::from_value(serde_json::json!({
            "details": {
                "format": "gguf",
                "family": "qwen35",
                "families": null,
                "parameter_size": "9.0B",
                "quantization_level": "Q4_K_M"
            },
            "capabilities": ["completion", "tools"],
            "parameters": "temperature 0.7",
            "license": "MIT",
            "modified_at": "2026-11-09T22:23:30.826158724Z",
            "requires": "0.30.9"
        }))
        .unwrap();
        assert_eq!(show.details.family, "qwen35");
        assert_eq!(show.details.format, "gguf");
        assert!(show.details.families.is_empty());
        assert_eq!(show.capabilities, vec!["completion", "tools"]);
        assert_eq!(show.parameters.as_deref(), Some("temperature 0.7"));
        assert_eq!(show.license.as_deref(), Some("MIT"));
    }

    #[test]
    fn parses_pull_progress_lines() {
        let progress = parse_pull_progress(
            br#"{"status":"pulling 797b70c4edf8","digest":"sha256:797b","total":10,"completed":4}"#,
        )
        .unwrap();
        assert_eq!(progress.status, "pulling 797b70c4edf8");
        assert_eq!(progress.digest.as_deref(), Some("sha256:797b"));
        assert_eq!(progress.fraction(), Some(0.4));

        let error = parse_pull_progress(br#"{"error":"model not found"}"#).unwrap_err();
        assert!(error.to_string().contains("model not found"));
    }

    #[test]
    fn serializes_tool_calls_and_definitions() {
        let call = ToolCall {
            id: Some("call_1".into()),
            function: ToolFunctionCall {
                name: "get_time".into(),
                arguments: serde_json::json!({}),
            },
        };
        let mut message = ChatMessage::new("assistant", "");
        message.tool_calls = vec![call];
        let json = serde_json::to_value(&message).unwrap();
        assert_eq!(json["tool_calls"][0]["id"], "call_1");
        assert_eq!(json["tool_calls"][0]["function"]["name"], "get_time");
        assert!(json.get("tool_name").is_none());

        let tool_message = ChatMessage {
            role: "tool".into(),
            content: "sunny".into(),
            images: Vec::new(),
            tool_calls: Vec::new(),
            tool_name: Some("get_weather".into()),
        };
        let json = serde_json::to_value(&tool_message).unwrap();
        assert_eq!(json["tool_name"], "get_weather");
        assert!(json.get("tool_calls").is_none());

        let request = ChatRequest {
            model: "llama3".into(),
            messages: vec![ChatMessage::new("user", "hi")],
            stream: true,
            options: None,
            keep_alive: None,
            think: None,
            tools: vec![serde_json::json!({
                "type": "function",
                "function": {"name": "get_time", "parameters": {"type": "object"}}
            })],
        };
        let json = serde_json::to_value(&request).unwrap();
        assert_eq!(json["tools"][0]["function"]["name"], "get_time");
    }

    #[test]
    fn parses_stream_tool_calls() {
        let chunk: ChatChunk = serde_json::from_value(serde_json::json!({
            "message": {
                "content": "",
                "tool_calls": [{
                    "id": "call_1",
                    "function": {
                        "index": 0,
                        "name": "calculator",
                        "arguments": {"expression": "2 + 2"}
                    }
                }]
            },
            "done": true
        }))
        .unwrap();
        let message = chunk.message.expect("message");
        assert_eq!(message.tool_calls.len(), 1);
        assert_eq!(message.tool_calls[0].function.name, "calculator");
        assert_eq!(
            message.tool_calls[0].function.arguments["expression"],
            "2 + 2"
        );
    }

    #[test]
    fn request_serializes_options() {
        let request = ChatRequest {
            model: "llama3".into(),
            messages: vec![ChatMessage::new("user", "hi")],
            stream: true,
            options: Some(ChatOptions {
                temperature: Some(0.5),
                num_ctx: Some(4096),
                ..Default::default()
            }),
            keep_alive: None,
            think: None,
            tools: Vec::new(),
        };
        let json = serde_json::to_value(&request).unwrap();
        assert_eq!(json["options"]["temperature"], 0.5);
        assert_eq!(json["options"]["num_ctx"], 4096);
        assert!(json["options"].get("seed").is_none());
    }

    #[test]
    fn chat_message_omits_empty_images() {
        let message = ChatMessage::new("user", "hi");
        let json = serde_json::to_value(&message).unwrap();
        assert!(json.get("images").is_none());

        let mut message = ChatMessage::new("user", "look");
        message.images = vec!["YWJj".into()];
        let json = serde_json::to_value(&message).unwrap();
        assert_eq!(json["images"][0], "YWJj");
    }

    #[test]
    fn parses_non_streaming_chat_reply() {
        let json = r#"{"model":"m","message":{"role":"assistant","content":"summary"},"done":true,"eval_count":5,"prompt_eval_count":11}"#;
        let chunk: ChatChunk = serde_json::from_str(json).unwrap();
        assert_eq!(chunk.message.unwrap().content, "summary");
        assert_eq!(chunk.eval_count, Some(5));
        assert_eq!(chunk.prompt_eval_count, Some(11));
        assert!(chunk.done);
    }

    #[test]
    fn parses_thinking_chunks() {
        let json = r#"{"message":{"role":"assistant","thinking":"step 1"},"done":false}"#;
        let chunk: ChatChunk = serde_json::from_str(json).unwrap();
        assert_eq!(chunk.message.unwrap().thinking, "step 1");
        assert!(!chunk.done);
    }

    #[test]
    fn parses_level_only_thinking_controls() {
        let body = serde_json::json!({
            "thinking": {"values": ["low", "medium", "high"], "default": "medium"}
        });
        let control = parse_thinking_control(&body).expect("thinking controls");
        assert!(!control.supports_off);
        assert!(control.is_level_only());
        assert_eq!(control.lowest_level(), Some("low"));
        assert_eq!(control.preferred_level("high"), "high");
        assert_eq!(control.preferred_level("unsupported"), "medium");
    }

    #[test]
    fn parses_boolean_thinking_controls() {
        let body = serde_json::json!({
            "thinking": {"values": [true, false], "default": true}
        });
        let control = parse_thinking_control(&body).expect("thinking controls");
        assert!(control.supports_off);
        assert!(!control.is_level_only());
    }

    #[test]
    fn serializes_think_levels() {
        let request = ChatRequest {
            model: "gpt-oss".into(),
            messages: vec![ChatMessage::new("user", "hi")],
            stream: true,
            options: None,
            keep_alive: None,
            think: Some(ThinkSetting::Level("low".into())),
            tools: Vec::new(),
        };
        let json = serde_json::to_value(&request).unwrap();
        assert_eq!(json["think"], "low");
    }

    #[test]
    fn chat_model_capability_filter() {
        let model = |capabilities: &[&str]| ModelSummary {
            name: "m".into(),
            size: 0,
            details: ModelDetails::default(),
            capabilities: capabilities.iter().map(|c| c.to_string()).collect(),
            remote_host: None,
        };

        assert!(model(&["completion"]).is_chat_model());
        assert!(model(&["completion", "vision", "tools"]).is_chat_model());
        assert!(model(&[]).is_chat_model());
        assert!(!model(&["decision", "vision"]).is_chat_model());
        assert!(!model(&["embedding"]).is_chat_model());
        assert!(!model(&["completion", "decision"]).is_chat_model());
    }

    #[test]
    fn parses_embeddings() {
        let body = serde_json::json!({
            "embeddings": [[0.1, 0.2, 0.3], [1.0, -2.0, 3.5]]
        });
        let vectors = parse_embeddings(&body).expect("embeddings");
        assert_eq!(vectors.len(), 2);
        assert_eq!(vectors[0], vec![0.1, 0.2, 0.3]);
        assert_eq!(vectors[1], vec![1.0, -2.0, 3.5]);
        assert!(parse_embeddings(&serde_json::json!({ "embeddings": [] })).is_err());
        assert!(parse_embeddings(&serde_json::json!({})).is_err());
    }

    #[test]
    fn reads_remote_host_from_tags() {
        let json = r#"{
            "name": "gpt-oss:120b-cloud",
            "size": 0,
            "capabilities": ["completion"],
            "remote_host": "https://ollama.com"
        }"#;
        let model: ModelSummary = serde_json::from_str(json).unwrap();
        assert_eq!(model.remote_host.as_deref(), Some("https://ollama.com"));
        assert!(crate::memory::model_is_cloud(
            &model.name,
            model.remote_host.as_deref(),
            "http://localhost:11434"
        ));
    }

    #[test]
    #[ignore = "requires a local Ollama embedding model"]
    fn live_embedding_returns_vectors() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let client = OllamaClient::new("http://localhost:11434", "");
            let vectors = client
                .embed("all-minilm", &["Hello from Ollama GUI.".to_string()])
                .await
                .expect("embedding request");
            assert_eq!(vectors.len(), 1);
            assert!(!vectors[0].is_empty());
        });
    }

    #[test]
    #[ignore = "requires a local Ollama speech model"]
    fn live_speech_returns_wav() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let client = OllamaClient::new("http://localhost:11434", "");
            let models = client.list_models().await.expect("list models");
            let model = models
                .iter()
                .find(|model| model.capabilities.iter().any(|item| item == "speech"))
                .map(|model| model.name.clone())
                .expect("no speech-capable models installed");

            let audio = client
                .speech(&model, "Hello from Ollama GUI.", "default")
                .await
                .expect("speech request");
            assert!(audio.starts_with(b"RIFF"), "response is not a WAV file");
            assert!(audio.len() > 1_000, "speech audio is suspiciously short");
        });
    }

    #[test]
    #[ignore = "requires a local Ollama daemon"]
    fn live_preloads_a_model() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let client = OllamaClient::new("http://localhost:11434", "");
            let models = client.list_models().await.expect("list models");
            let model = models
                .iter()
                .find(|model| model.is_chat_model())
                .map(|model| model.name.clone())
                .expect("no chat-capable models installed");
            client
                .preload(&model, 2048, "30s")
                .await
                .expect("preload model");
        });
    }

    #[test]
    #[ignore = "requires a local Ollama daemon"]
    fn live_chat_streams() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let client = OllamaClient::new("http://localhost:11434", "");
            let models = client.list_models().await.expect("list models");
            assert!(!models.is_empty(), "no models installed");

            let model = models
                .iter()
                .find(|model| {
                    model.is_chat_model() && !model.capabilities.iter().any(|item| item == "speech")
                })
                .or_else(|| models.iter().find(|model| model.is_chat_model()))
                .map(|model| model.name.clone())
                .expect("no chat-capable models installed");

            let request = ChatRequest {
                model,
                messages: vec![ChatMessage::new("user", "Reply with exactly: hello")],
                stream: true,
                options: Some(ChatOptions {
                    temperature: Some(0.0),
                    num_predict: Some(16),
                    ..Default::default()
                }),
                keep_alive: Some("30s".into()),
                think: Some(ThinkSetting::Bool(false)),
                tools: Vec::new(),
            };

            let (tx, rx) = async_channel::unbounded();
            let abort = spawn_chat_stream(&runtime, client, request, tx);

            let mut text = String::new();
            let mut done = false;
            while let Ok(event) = rx.recv().await {
                match event {
                    StreamEvent::Delta(delta) => text.push_str(&delta),
                    StreamEvent::Thinking(_) => {}
                    StreamEvent::ToolCalls(_) => {}
                    StreamEvent::Done(stats) => {
                        done = true;
                        println!("stats: {stats:?}");
                    }
                    StreamEvent::Error(err) => panic!("stream error: {err}"),
                }
            }
            abort.abort();

            assert!(done, "stream ended without a Done event");
            assert!(!text.is_empty(), "stream produced no text");
            println!("reply: {text}");
        });
    }
}
