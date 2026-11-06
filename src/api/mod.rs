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

#[allow(dead_code)]
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ModelDetails {
    #[serde(default)]
    pub parameter_size: String,
    #[serde(default)]
    pub quantization_level: String,
    #[serde(default)]
    pub family: String,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: &str, content: impl Into<String>) -> Self {
        Self {
            role: role.to_string(),
            content: content.into(),
        }
    }
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
}

#[derive(Debug, Clone, Default)]
pub struct StreamStats {
    pub eval_count: Option<u32>,
    pub eval_duration: Option<u64>,
    pub prompt_eval_count: Option<u32>,
}

impl StreamStats {
    pub fn tokens_per_second(&self) -> Option<f64> {
        let count = self.eval_count? as f64;
        let seconds = self.eval_duration? as f64 / 1_000_000_000.0;
        (seconds > 0.0).then(|| count / seconds)
    }
}

#[derive(Debug)]
pub enum StreamEvent {
    Delta(String),
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
        if !message.content.is_empty() {
            let _ = tx.send(StreamEvent::Delta(message.content)).await;
        }
    }

    stats.eval_count = chunk.eval_count.or(stats.eval_count);
    stats.eval_duration = chunk.eval_duration.or(stats.eval_duration);
    stats.prompt_eval_count = chunk.prompt_eval_count.or(stats.prompt_eval_count);

    Ok(chunk.done)
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
        };
        let json = serde_json::to_value(&request).unwrap();
        assert_eq!(json["options"]["temperature"], 0.5);
        assert_eq!(json["options"]["num_ctx"], 4096);
        assert!(json["options"].get("seed").is_none());
    }

    #[test]
    fn chat_model_capability_filter() {
        let model = |capabilities: &[&str]| ModelSummary {
            name: "m".into(),
            size: 0,
            details: ModelDetails::default(),
            capabilities: capabilities.iter().map(|c| c.to_string()).collect(),
        };

        assert!(model(&["completion"]).is_chat_model());
        assert!(model(&["completion", "vision", "tools"]).is_chat_model());
        assert!(model(&[]).is_chat_model());
        assert!(!model(&["decision", "vision"]).is_chat_model());
        assert!(!model(&["embedding"]).is_chat_model());
        assert!(!model(&["completion", "decision"]).is_chat_model());
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
                .find(|model| model.name == "lfm3:latest")
                .map(|model| model.name.clone())
                .unwrap_or_else(|| models[0].name.clone());

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
            };

            let (tx, rx) = async_channel::unbounded();
            let abort = spawn_chat_stream(&runtime, client, request, tx);

            let mut text = String::new();
            let mut done = false;
            while let Ok(event) = rx.recv().await {
                match event {
                    StreamEvent::Delta(delta) => text.push_str(&delta),
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
