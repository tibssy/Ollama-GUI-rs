//! Wake-word detection with the WakeHuBERT models from OpenVoiceOS.
//!
//! The pipeline is a tiny two-stage ONNX graph: a HuBERT featurizer turns a
//! 1.5 s window of 16 kHz audio into `[1, 75, 128]` features, and a per-word
//! GRU head turns those features into one calibrated logit. We score the
//! trailing window every 80 ms and fire when `sigmoid(logit)` crosses the
//! word's calibrated threshold.

use anyhow::{bail, Context, Result};
use ort::session::Session;
use ort::value::Tensor;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::config::data_dir;
use crate::stt;

/// Samples in the 1.5 s decision window.
pub const WINDOW_SAMPLES: usize = 24_000;
/// Samples between decisions (80 ms).
pub const HOP_SAMPLES: usize = 1_280;
/// Ignore repeated hits for this long after a detection.
const COOLDOWN: Duration = Duration::from_secs(2);
/// Silero VAD frame at 16 kHz.
const VAD_FRAME: usize = 512;
/// RMS that opens the scoring gate in a quiet room.
const GATE_RMS: f32 = 0.004;
/// Ceiling for the tracked noise floor.
const GATE_FLOOR_MAX: f32 = 0.02;
/// How long the gate stays open after the last loud frame.
const GATE_HANGOVER: Duration = Duration::from_millis(1_500);

const CATALOG_URL: &str =
    "https://huggingface.co/OpenVoiceOS/wakehubert-wakewords/raw/main/models.json";
const HEAD_BASE_URL: &str = "https://huggingface.co/OpenVoiceOS/wakehubert-wakewords/resolve/main";
const FEATURIZER_BASE_URL: &str = "https://huggingface.co/TigreGotico/wakehubert-tiny/resolve/main";

/// Words shipped by the upstream catalog, in menu order.
pub const WORDS: &[(&str, &str)] = &[
    ("jarvis", "en"),
    ("hey jarvis", "en"),
    ("alexa", "en"),
    ("hey marvin", "en"),
    ("marvin", "en"),
    ("hey mycroft", "en"),
    ("home assistant", "en"),
    ("okay nabu", "en"),
    ("hello nabu", "en"),
    ("hey chatterbox", "en"),
    ("hey floyd", "en"),
    ("hey rhasspy", "en"),
    ("hey robin", "en"),
    ("sheila", "en"),
    ("stop", "en"),
    ("android", "en"),
    ("hey computer", "en"),
    ("computer", "en"),
    ("hey k9", "en"),
    ("hey scout", "en"),
    ("hey ziggy", "en"),
    ("hey potato", "en"),
    ("hey stemcom", "en"),
    ("wake up", "en"),
    ("despierta", "es"),
    ("aufwachen", "de"),
    ("wakker worden", "nl"),
    ("sveglia", "it"),
];

#[derive(Debug, Clone, Deserialize)]
pub struct WakeWord {
    pub name: String,
    pub file: String,
    pub word: String,
    #[serde(default)]
    pub language: String,
    pub featurizer: String,
    #[serde(default)]
    pub default_threshold: f32,
    #[serde(default)]
    pub features: Option<String>,
    #[serde(default)]
    pub sha256: String,
}

impl WakeWord {
    /// Streaming heads use a different input contract, so skip them.
    pub fn is_streaming(&self) -> bool {
        self.features.as_deref() == Some("streamed") || self.name.ends_with("_streaming")
    }

    fn head_file(&self) -> PathBuf {
        Path::new(&self.file)
            .file_name()
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(format!("{}.onnx", self.name)))
    }

    fn featurizer_file(&self) -> &'static str {
        match self.featurizer.as_str() {
            "wakehubert" => "wakehubert.onnx",
            _ => "wakehubert_int8.onnx",
        }
    }
}

#[derive(Debug, Deserialize)]
struct Catalog {
    models: Vec<WakeWord>,
}

pub fn model_dir() -> PathBuf {
    data_dir().join("wake")
}

fn catalog_path() -> PathBuf {
    model_dir().join("models.json")
}

/// Capitalised label used by the settings row.
pub fn display_name(word: &str) -> String {
    let mut text = String::with_capacity(word.len());
    for part in word.split(' ') {
        if let Some(first) = part.chars().next() {
            text.extend(first.to_uppercase());
            text.push_str(&part[first.len_utf8()..]);
        }
        text.push(' ');
    }
    text.pop();
    text
}

/// True when the selected word's head and featurizer are on disk.
pub fn is_downloaded(word: &str) -> bool {
    let Ok(catalog) = read_catalog() else {
        return false;
    };
    catalog
        .iter()
        .filter(|entry| !entry.is_streaming())
        .find(|entry| entry.word == word)
        .is_some_and(|entry| {
            model_dir().join(entry.head_file()).exists()
                && model_dir().join(entry.featurizer_file()).exists()
        })
}

fn read_catalog() -> Result<Vec<WakeWord>> {
    let text = std::fs::read_to_string(catalog_path())
        .with_context(|| format!("reading {}", catalog_path().display()))?;
    let catalog: Catalog = serde_json::from_str(&text).context("parsing the wake word catalog")?;
    Ok(catalog
        .models
        .into_iter()
        .filter(|entry| !entry.is_streaming())
        .collect())
}

async fn ensure_catalog(client: &reqwest::Client) -> Result<Vec<WakeWord>> {
    if let Ok(words) = read_catalog() {
        if !words.is_empty() {
            return Ok(words);
        }
    }

    std::fs::create_dir_all(model_dir())
        .with_context(|| format!("creating {}", model_dir().display()))?;
    let text = client
        .get(CATALOG_URL)
        .send()
        .await
        .context("downloading the wake word catalog")?
        .error_for_status()
        .context("downloading the wake word catalog")?
        .text()
        .await
        .context("reading the wake word catalog")?;
    let catalog: Catalog = serde_json::from_str(&text).context("parsing the wake word catalog")?;
    std::fs::write(catalog_path(), &text)
        .with_context(|| format!("writing {}", catalog_path().display()))?;
    Ok(catalog
        .models
        .into_iter()
        .filter(|entry| !entry.is_streaming())
        .collect())
}

/// Download the selected word's models and load a detector for it.
pub async fn prepare(word_name: &str, threshold_override: f32) -> Result<WakeDetector> {
    std::fs::create_dir_all(model_dir())
        .with_context(|| format!("creating {}", model_dir().display()))?;
    let client = reqwest::Client::builder()
        .user_agent("ollama-gui")
        .build()
        .context("building the wake word download client")?;

    let catalog = ensure_catalog(&client).await?;
    let entry = catalog
        .into_iter()
        .find(|entry| entry.word == word_name)
        .with_context(|| format!("wake word {word_name:?} is not in the catalog"))?;

    download(
        &client,
        &format!("{FEATURIZER_BASE_URL}/{}", entry.featurizer_file()),
        &model_dir().join(entry.featurizer_file()),
        None,
    )
    .await?;
    download(
        &client,
        &format!("{HEAD_BASE_URL}/{}", entry.file),
        &model_dir().join(entry.head_file()),
        Some(entry.sha256.as_str()),
    )
    .await?;

    tokio::task::spawn_blocking(move || WakeDetector::load(&entry, threshold_override)).await?
}

async fn download(
    client: &reqwest::Client,
    url: &str,
    path: &Path,
    expected_sha256: Option<&str>,
) -> Result<()> {
    if path.exists() {
        let valid = match expected_sha256.filter(|sha| !sha.is_empty()) {
            Some(sha) => sha256_file(path).is_ok_and(|actual| actual == sha),
            None => true,
        };
        if valid {
            return Ok(());
        }
        tracing::warn!("wake word: discarding invalid {}", path.display());
        let _ = std::fs::remove_file(path);
    }

    tracing::info!("wake word: downloading {url}");
    let response = client
        .get(url)
        .send()
        .await
        .with_context(|| format!("downloading {url}"))?
        .error_for_status()
        .with_context(|| format!("downloading {url}"))?;
    let bytes = response
        .bytes()
        .await
        .with_context(|| format!("reading {url}"))?;
    if bytes.len() < 1024 {
        bail!("downloaded {url} is too small ({} bytes)", bytes.len());
    }

    let temp = path.with_extension("part");
    std::fs::write(&temp, &bytes).with_context(|| format!("writing {}", temp.display()))?;
    std::fs::rename(&temp, path)
        .with_context(|| format!("moving {} to {}", temp.display(), path.display()))?;
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 16 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

#[derive(Debug, Clone)]
pub enum WakeEvent {
    /// Score of a detection that crossed the threshold.
    Detected(f32),
    Error(String),
}

pub struct WakeSession {
    _stream: cpal::Stream,
    stop: Arc<AtomicBool>,
}

impl WakeSession {
    pub fn stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
    }
}

/// Open the microphone and start scoring the audio in a background thread.
pub fn start(
    detector: WakeDetector,
    events: async_channel::Sender<WakeEvent>,
) -> Result<WakeSession> {
    let error_events = events.clone();
    let capture = stt::start_capture(move |message| {
        let _ = error_events.try_send(WakeEvent::Error(message));
    })?;

    let mut resampler = stt::LinearResampler::new(capture.sample_rate);
    let mut gate = SpeechGate::new();
    let audio = capture.audio;
    let stop = Arc::new(AtomicBool::new(false));
    let thread_stop = Arc::clone(&stop);

    std::thread::Builder::new()
        .name("ollama-gui-wake".to_string())
        .spawn(move || {
            let mut detector = detector;
            let mut pending: Vec<f32> = Vec::new();
            while !thread_stop.load(Ordering::SeqCst) {
                let Ok(chunk) = audio.recv_blocking() else {
                    break;
                };
                let pcm = resampler.push(&chunk);
                if pcm.is_empty() {
                    continue;
                }
                pending.extend_from_slice(&pcm);

                let mut offset = 0;
                let mut fired = None;
                while pending.len() - offset >= VAD_FRAME {
                    let frame = &pending[offset..offset + VAD_FRAME];
                    let score = gate.update(frame);
                    let result = detector.push(frame, score);
                    offset += VAD_FRAME;
                    match result {
                        Ok(Some(score)) => {
                            fired = Some(score);
                            break;
                        }
                        Ok(None) => {}
                        Err(error) => {
                            let _ = events.try_send(WakeEvent::Error(format!("{error:#}")));
                            return;
                        }
                    }
                }
                if offset > 0 {
                    pending.drain(..offset);
                }
                if let Some(score) = fired {
                    tracing::info!("wake word: detected (score {score:.3})");
                    if events.try_send(WakeEvent::Detected(score)).is_err() {
                        break;
                    }
                }
            }
        })
        .context("starting the wake word thread")?;

    Ok(WakeSession {
        _stream: capture.stream,
        stop,
    })
}

pub struct WakeDetector {
    featurizer: Session,
    head: Session,
    threshold: f32,
    window: Vec<f32>,
    hop: Vec<f32>,
    cooldown_until: Option<Instant>,
    last_score: f32,
}

impl WakeDetector {
    pub fn load(word: &WakeWord, threshold_override: f32) -> Result<Self> {
        let dir = model_dir();
        let featurizer_path = dir.join(word.featurizer_file());
        let head_path = dir.join(word.head_file());

        let featurizer = session(&featurizer_path, "wake word featurizer")?;
        let head = session(&head_path, "wake word head")?;

        let threshold = if threshold_override > 0.0 {
            threshold_override
        } else if word.default_threshold > 0.0 {
            word.default_threshold
        } else {
            0.5
        };

        tracing::info!(
            "wake word: loaded {:?} [{}] with threshold {threshold:.2}",
            word.word,
            word.language
        );

        Ok(Self {
            featurizer,
            head,
            threshold,
            window: Vec::with_capacity(WINDOW_SAMPLES),
            hop: Vec::with_capacity(HOP_SAMPLES),
            cooldown_until: None,
            last_score: 0.0,
        })
    }

    /// Highest score produced by the most recent inference pass.
    #[cfg(test)]
    pub fn last_score(&self) -> f32 {
        self.last_score
    }

    /// Feed 16 kHz mono samples. Returns the score when a word is detected.
    ///
    /// `score` comes from the speech gate: the window is always updated, but
    /// inference is skipped while the audio is quiet, which is where almost
    /// all of the CPU savings come from.
    pub fn push(&mut self, samples: &[f32], score: bool) -> Result<Option<f32>> {
        for sample in samples {
            self.hop.push(*sample);
            if self.hop.len() < HOP_SAMPLES {
                continue;
            }

            self.window.extend_from_slice(&self.hop);
            self.hop.clear();
            let excess = self.window.len().saturating_sub(WINDOW_SAMPLES);
            if excess > 0 {
                self.window.drain(..excess);
            }
            if self.window.len() < WINDOW_SAMPLES || !score {
                continue;
            }

            let now = Instant::now();
            if self.cooldown_until.is_some_and(|until| now < until) {
                continue;
            }

            let score = self.score()?;
            if score >= self.threshold {
                self.cooldown_until = Some(now + COOLDOWN);
                return Ok(Some(score));
            }
        }
        Ok(None)
    }

    fn score(&mut self) -> Result<f32> {
        let waveform = Tensor::from_array(([1usize, self.window.len()], self.window.clone()))
            .map_err(|error| anyhow::anyhow!("building the wake word waveform: {error}"))?;
        let output = self
            .featurizer
            .run(ort::inputs!["waveform" => waveform])
            .map_err(|error| anyhow::anyhow!("running the wake word featurizer: {error}"))?;

        let (shape, data) = output["features"]
            .try_extract_tensor::<f32>()
            .map_err(|error| anyhow::anyhow!("reading wake word features: {error}"))?;
        let dims: Vec<i64> = shape.iter().copied().collect();
        let features = data.to_vec();
        drop(output);

        let frames = dims.get(1).copied().unwrap_or(0).max(0) as usize;
        if frames == 0 {
            bail!("the wake word featurizer returned no frames");
        }

        let tensor = Tensor::from_array(([1usize, frames, 128usize], features))
            .map_err(|error| anyhow::anyhow!("building the wake word features: {error}"))?;
        let output = self
            .head
            .run(ort::inputs!["features" => tensor])
            .map_err(|error| anyhow::anyhow!("running the wake word head: {error}"))?;
        let (_, logits) = output["logit_calibrated"]
            .try_extract_tensor::<f32>()
            .map_err(|error| anyhow::anyhow!("reading the wake word score: {error}"))?;
        let logit = logits.first().copied().unwrap_or(f32::NEG_INFINITY);
        self.last_score = 1.0 / (1.0 + (-logit).exp());
        Ok(self.last_score)
    }
}

fn session(path: &Path, label: &str) -> Result<Session> {
    let builder = Session::builder()
        .map_err(|error| anyhow::anyhow!("creating the {label} session: {error}"))?;
    let builder = builder
        .with_intra_threads(1)
        .map_err(|error| anyhow::anyhow!("configuring the {label} session: {error}"))?;
    // Detection is infrequent, so spin-waiting between runs would burn CPU
    // for no benefit.
    let builder = builder
        .with_inter_op_spinning(false)
        .map_err(|error| anyhow::anyhow!("configuring the {label} session: {error}"))?;
    let mut builder = builder
        .with_intra_op_spinning(false)
        .map_err(|error| anyhow::anyhow!("configuring the {label} session: {error}"))?;
    builder
        .commit_from_file(path)
        .with_context(|| format!("loading {}", path.display()))
}

/// Energy gate that keeps the expensive featurizer idle while the room is
/// quiet. It never decides *whether* a word was spoken, only whether it is
/// worth running the model, so a false open costs CPU and nothing else.
struct SpeechGate {
    floor: f32,
    active_until: Option<Instant>,
}

impl SpeechGate {
    fn new() -> Self {
        Self {
            floor: 0.001,
            active_until: None,
        }
    }

    fn update(&mut self, frame: &[f32]) -> bool {
        let energy = frame.iter().map(|sample| sample * sample).sum::<f32>();
        let rms = (energy / frame.len().max(1) as f32).sqrt();
        let now = Instant::now();
        let was_open = self.active_until.is_some_and(|until| now < until);
        let threshold = (self.floor * 2.5).max(GATE_RMS);
        if rms >= threshold {
            self.active_until = Some(now + GATE_HANGOVER);
        } else {
            self.floor = (self.floor * 0.98 + rms * 0.02).min(GATE_FLOOR_MAX);
        }
        let open = self.active_until.is_some_and(|until| now < until);
        if open != was_open {
            tracing::debug!(
                "wake word: gate {} (rms {rms:.4})",
                if open { "open" } else { "closed" }
            );
        }
        open
    }
}

#[cfg(test)]
mod tests {
    use super::{display_name, WakeWord, WORDS};

    fn entry(name: &str, features: Option<&str>) -> WakeWord {
        WakeWord {
            name: name.to_string(),
            file: format!("models/{name}.onnx"),
            word: "test".to_string(),
            language: "en".to_string(),
            featurizer: "wakehubert-int8".to_string(),
            default_threshold: 0.5,
            features: features.map(str::to_string),
            sha256: String::new(),
        }
    }

    #[test]
    fn catalog_keeps_non_streaming_heads() {
        assert!(entry("wakehubert_jarvis_streaming", Some("streamed")).is_streaming());
        assert!(entry("wakehubert_jarvis_streaming", None).is_streaming());
        assert!(!entry("wakehubert_jarvis", None).is_streaming());
    }

    #[test]
    fn head_file_drops_the_models_prefix() {
        assert_eq!(
            entry("wakehubert_jarvis", None).head_file(),
            std::path::PathBuf::from("wakehubert_jarvis.onnx")
        );
    }

    #[test]
    fn energy_gate_stays_closed_in_silence() {
        let mut gate = super::SpeechGate::new();
        let silence = vec![0.0f32; super::VAD_FRAME];
        for _ in 0..20 {
            assert!(!gate.update(&silence));
        }
    }

    #[test]
    fn energy_gate_opens_on_speech() {
        let mut gate = super::SpeechGate::new();
        let silence = vec![0.0f32; super::VAD_FRAME];
        gate.update(&silence);
        let speech = vec![0.1f32; super::VAD_FRAME];
        assert!(gate.update(&speech));
    }

    #[test]
    fn word_list_has_no_duplicates() {
        let mut words: Vec<&str> = WORDS.iter().map(|(word, _)| *word).collect();
        words.sort_unstable();
        words.dedup();
        assert_eq!(words.len(), WORDS.len());
        assert_eq!(WORDS.len(), 28);
    }

    #[test]
    fn display_names_are_capitalised() {
        assert_eq!(display_name("hey jarvis"), "Hey Jarvis");
        assert_eq!(display_name("wake up"), "Wake Up");
        assert_eq!(display_name("jarvis"), "Jarvis");
    }

    fn read_wav_16k(path: &str) -> Vec<f32> {
        let mut reader = hound::WavReader::open(path).expect("open wav");
        let spec = reader.spec();
        let channels = spec.channels as usize;
        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Int => reader
                .samples::<i16>()
                .map(|sample| sample.expect("sample") as f32 / 32_768.0)
                .collect(),
            hound::SampleFormat::Float => reader
                .samples::<f32>()
                .map(|sample| sample.expect("sample"))
                .collect(),
        };
        let mono: Vec<f32> = if channels <= 1 {
            samples
        } else {
            samples
                .chunks(channels)
                .map(|frame| frame.iter().sum::<f32>() / channels as f32)
                .collect()
        };
        if spec.sample_rate == 16_000 {
            return mono;
        }
        let step = spec.sample_rate as f64 / 16_000.0;
        let mut out = Vec::new();
        let mut position = 0.0;
        while position + 1.0 < mono.len() as f64 {
            let index = position.floor() as usize;
            let fraction = (position - index as f64) as f32;
            out.push(mono[index] + (mono[index + 1] - mono[index]) * fraction);
            position += step;
        }
        out
    }

    #[test]
    #[ignore = "requires downloaded wake word models and WAKE_TEST_WAV"]
    fn detects_spoken_wake_word() {
        let path = std::env::var("WAKE_TEST_WAV").expect("set WAKE_TEST_WAV");
        let target = std::env::var("WAKE_TEST_WORD").unwrap_or_else(|_| "hey jarvis".to_string());
        let words = super::read_catalog().expect("catalog");
        let word = words
            .iter()
            .find(|word| word.word == target)
            .unwrap_or_else(|| panic!("{target} head"));
        let mut detector = super::WakeDetector::load(word, 0.0).expect("load detector");

        let mut samples = vec![0.0f32; 16_000];
        samples.extend(read_wav_16k(&path));
        samples.extend(std::iter::repeat_n(0.0, 32_000));
        let mut detected = None;
        let mut best = 0.0f32;
        for chunk in samples.chunks(1_280) {
            if let Some(score) = detector.push(chunk, true).expect("score") {
                detected = Some(score);
                break;
            }
            best = best.max(detector.last_score());
        }
        best = best.max(detector.last_score());
        println!("detected={detected:?} best={best:.3}");
        assert!(detected.is_some(), "the wake word was not detected");
    }

    #[test]
    #[ignore = "requires downloaded wake word models and WAKE_TEST_NEGATIVE_WAV"]
    fn ignores_other_speech() {
        let path = std::env::var("WAKE_TEST_NEGATIVE_WAV").expect("set WAKE_TEST_NEGATIVE_WAV");
        let target = std::env::var("WAKE_TEST_WORD").unwrap_or_else(|_| "hey jarvis".to_string());
        let words = super::read_catalog().expect("catalog");
        let word = words
            .iter()
            .find(|word| word.word == target)
            .unwrap_or_else(|| panic!("{target} head"));
        let mut detector = super::WakeDetector::load(word, 0.0).expect("load detector");

        let mut samples = vec![0.0f32; 16_000];
        samples.extend(read_wav_16k(&path));
        samples.extend(std::iter::repeat_n(0.0, 32_000));

        let mut best = 0.0f32;
        for chunk in samples.chunks(1_280) {
            detector.push(chunk, true).expect("score");
            best = best.max(detector.last_score());
        }
        println!("negative best={best:.3}");
        assert!(
            best < 0.5,
            "unexpected detection in unrelated speech: {best}"
        );
    }
}
