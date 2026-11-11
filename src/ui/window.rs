use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use base64::Engine as _;
use gtk::prelude::*;
use gtk::{gdk, gio, glib};
use ksni::TrayMethods;
use libadwaita as adw;
use libadwaita::prelude::*;

use crate::api::{
    self, ChatMessage, ChatOptions, ChatRequest, ModelSummary, OllamaClient, StreamEvent,
    StreamStats, ThinkSetting, ThinkingControl,
};
use crate::autostart;
use crate::config::Settings;
use crate::feedback::{self, Cue};
use crate::memory::{self, MemoryPool};
use crate::store::{Store, StoredImage, StoredMessage, StoredSummary};
use crate::stt::{self, PreparedBackend, SttEvent};
use crate::tray::{StatusTray, TrayCommand};
use crate::ui::image;
use crate::wake::{self, WakeEvent};

/// Seconds without speech activity before local TTS/STT models are released.
const SPEECH_IDLE_UNLOAD_SECS: u32 = 60;

/// Compress older history once the context reaches this fraction of `num_ctx`.
const CONTEXT_COMPRESS_RATIO: f64 = 0.85;
/// Number of recent user turns kept verbatim.
const CONTEXT_KEEP_TURNS: usize = 5;
/// Cap the rolling summary at this fraction of `num_ctx`.
const CONTEXT_SUMMARY_RATIO: f64 = 0.10;
/// Feed summarization this fraction of `num_ctx` at a time.
const CONTEXT_CHUNK_RATIO: f64 = 0.45;
const SUMMARY_TEMPERATURE: f64 = 0.2;
/// Wait this long after a reply before extracting long-term memory.
const MEMORY_DEBOUNCE_MS: u64 = 3_000;
/// Let the graphical session and microphone settle before an autostart launch
/// arms the wake detector.
const AUTOSTART_WAKE_DELAY_SECS: u32 = 3;
/// How long Ollama keeps an unused model loaded.
const CHAT_KEEP_ALIVE: &str = "5m";
/// Feed memory extraction this fraction of `num_ctx` at a time.
const MEMORY_TRANSCRIPT_RATIO: f64 = 0.45;
const MEMORY_TEMPERATURE: f64 = 0.1;
const SUMMARY_SYSTEM_PROMPT: &str = "You maintain a compact summary of an ongoing chat. \
Merge the existing summary with the older messages provided. Preserve the user's goals, \
preferences, decisions, facts, names, numbers, file paths, code identifiers, and unresolved \
tasks. Remove greetings, filler, repetition, and raw code that is not essential. Output only \
the updated summary, without commentary.";

pub struct AppState {
    pub settings: RefCell<Settings>,
    pub store: RefCell<Store>,
    pub runtime: tokio::runtime::Runtime,
    pub client: RefCell<OllamaClient>,
    pub models: RefCell<Vec<ModelSummary>>,
    pub paradee: Arc<std::sync::Mutex<Option<crate::paradee::Paradee>>>,
    pub whistle: Arc<std::sync::Mutex<Option<crate::whistle::Whistle>>>,
    pub parakeet: crate::parakeet::SharedParakeet,
}

impl AppState {
    pub fn new() -> anyhow::Result<Self> {
        let settings = Settings::load();
        let client = OllamaClient::new(&settings.base_url, &settings.api_key);
        Ok(Self {
            settings: RefCell::new(settings),
            store: RefCell::new(Store::open()?),
            runtime: tokio::runtime::Runtime::new()?,
            client: RefCell::new(client),
            models: RefCell::new(Vec::new()),
            paradee: Arc::new(std::sync::Mutex::new(None)),
            whistle: Arc::new(std::sync::Mutex::new(None)),
            parakeet: Arc::new(std::sync::Mutex::new(None)),
        })
    }

    fn rebuild_client(&self) {
        let settings = self.settings.borrow();
        *self.client.borrow_mut() = OllamaClient::new(&settings.base_url, &settings.api_key);
    }
}

struct ActiveStream {
    abort: tokio::task::AbortHandle,
    row: gtk::Box,
    bubble: gtk::Box,
    view: gtk::TextView,
    content: Rc<RefCell<String>>,
    conv_id: i64,
    model: String,
    stats: Rc<RefCell<Option<StreamStats>>>,
}

struct PendingAttachment {
    id: u64,
    name: String,
    image: StoredImage,
    preview_data: Vec<u8>,
}

struct PlaybackChunk {
    generation: u64,
    samples: Vec<f32>,
    sample_rate: u32,
    channels: u16,
}

enum SpeechEvent {
    Playing,
    AudioUnavailable,
    Error(String),
    Finished,
}

struct CompressionPlan {
    model: String,
    num_ctx: u32,
    existing: Option<String>,
    upto_id: i64,
    chunks: Vec<String>,
}

struct CompressionResult {
    text: String,
    tokens: u32,
    upto_id: i64,
}

#[derive(Clone)]
struct GenerationRequest {
    conversation_id: i64,
    memory_context: Option<String>,
}

struct MemoryJob {
    conversation_id: i64,
    upto_id: i64,
    pool: MemoryPool,
    model: String,
    embedding_model: String,
    created_date: String,
    transcript: String,
    max_chars: usize,
    num_ctx: u32,
    epoch: u64,
}

impl MemoryJob {
    fn empty_result(&self) -> MemoryJobResult {
        MemoryJobResult {
            conversation_id: self.conversation_id,
            upto_id: self.upto_id,
            pool: self.pool,
            model: self.model.clone(),
            embedding_model: self.embedding_model.clone(),
            text: None,
            vector: None,
            epoch: self.epoch,
        }
    }
}

struct MemoryJobResult {
    conversation_id: i64,
    upto_id: i64,
    pool: MemoryPool,
    model: String,
    embedding_model: String,
    text: Option<String>,
    vector: Option<Vec<f32>>,
    epoch: u64,
}

#[derive(Clone)]
struct ThinkingUi {
    container: gtk::Box,
    label: gtk::Label,
    spinner: gtk::Spinner,
    view: gtk::TextView,
}

impl ThinkingUi {
    fn new(bubble: &gtk::Box) -> Self {
        let container = gtk::Box::new(gtk::Orientation::Vertical, 4);
        container.add_css_class("thinking-box");
        container.set_visible(false);

        let header = gtk::Box::new(gtk::Orientation::Horizontal, 6);
        let icon = gtk::Image::from_icon_name("applications-science-symbolic");
        icon.add_css_class("dim-label");
        icon.set_valign(gtk::Align::Center);
        header.append(&icon);

        let label = gtk::Label::new(Some("Thinking…"));
        label.set_xalign(0.0);
        label.set_hexpand(true);
        label.add_css_class("dim-label");
        header.append(&label);

        let spinner = gtk::Spinner::new();
        spinner.set_valign(gtk::Align::Center);
        spinner.set_visible(false);
        header.append(&spinner);

        let toggle_icon = gtk::Image::from_icon_name("pan-end-symbolic");
        let toggle = gtk::ToggleButton::new();
        toggle.set_child(Some(&toggle_icon));
        toggle.add_css_class("flat");
        toggle.set_valign(gtk::Align::Center);
        toggle.set_tooltip_text(Some("Show thinking"));
        header.append(&toggle);
        container.append(&header);

        let view = gtk::TextView::builder()
            .editable(false)
            .cursor_visible(false)
            .wrap_mode(gtk::WrapMode::WordChar)
            .build();
        view.add_css_class("thinking-text");

        let content = gtk::ScrolledWindow::builder()
            .hscrollbar_policy(gtk::PolicyType::Never)
            .vscrollbar_policy(gtk::PolicyType::Automatic)
            .max_content_height(260)
            .propagate_natural_height(true)
            .child(&view)
            .build();
        content.set_visible(false);
        container.append(&content);

        bubble.prepend(&container);

        {
            let content = content.clone();
            let toggle_icon = toggle_icon.clone();
            let toggle = toggle.clone();
            toggle.connect_toggled(move |button| {
                let expanded = button.is_active();
                content.set_visible(expanded);
                toggle_icon.set_icon_name(Some(if expanded {
                    "pan-down-symbolic"
                } else {
                    "pan-end-symbolic"
                }));
                button.set_tooltip_text(Some(if expanded {
                    "Hide thinking"
                } else {
                    "Show thinking"
                }));
            });
        }

        Self {
            container,
            label,
            spinner,
            view,
        }
    }

    fn show(&self) {
        if !self.container.get_visible() {
            self.container.set_visible(true);
            self.label.set_text("Thinking…");
            self.spinner.set_visible(true);
            self.spinner.start();
        }
    }

    fn append(&self, delta: &str) {
        let buffer = self.view.buffer();
        let mut end = buffer.end_iter();
        buffer.insert(&mut end, delta);
    }

    fn finish(&self) {
        if self.spinner.get_visible() {
            self.spinner.stop();
            self.spinner.set_visible(false);
            self.label.set_text("Thought");
            tracing::debug!(
                "thinking: captured {} characters",
                self.view.buffer().char_count()
            );
        }
    }
}

struct SpeechSynthesis {
    choice: TtsChoice,
    voice: String,
    generation: u64,
    generation_counter: Arc<AtomicU64>,
    sender: async_channel::Sender<PlaybackChunk>,
    ui_tx: async_channel::Sender<SpeechEvent>,
    client: OllamaClient,
    model: String,
    paradee: Arc<std::sync::Mutex<Option<crate::paradee::Paradee>>>,
}

impl SpeechSynthesis {
    fn alive(&self) -> bool {
        self.generation_counter.load(Ordering::SeqCst) == self.generation
    }

    async fn run(self, sentences: async_channel::Receiver<String>) {
        if self.choice == TtsChoice::Paradee {
            if let Err(err) = crate::paradee::ensure_downloaded().await {
                let _ = self
                    .ui_tx
                    .send(SpeechEvent::Error(format!("{err:#}")))
                    .await;
                let _ = self.ui_tx.send(SpeechEvent::Finished).await;
                return;
            }
        }

        let mut first_chunk = true;
        while let Ok(sentence) = sentences.recv().await {
            if !self.alive() {
                break;
            }

            match self.synthesize(&sentence).await {
                Ok((samples, sample_rate, channels)) => {
                    if !self.alive() {
                        break;
                    }
                    let chunk = PlaybackChunk {
                        generation: self.generation,
                        samples,
                        sample_rate,
                        channels,
                    };
                    if self.sender.send(chunk).await.is_err() {
                        let _ = self.ui_tx.send(SpeechEvent::AudioUnavailable).await;
                        break;
                    }
                    if first_chunk {
                        first_chunk = false;
                        let _ = self.ui_tx.send(SpeechEvent::Playing).await;
                    }
                }
                Err(err) => {
                    let _ = self
                        .ui_tx
                        .send(SpeechEvent::Error(format!("{err:#}")))
                        .await;
                    break;
                }
            }
        }

        let _ = self.ui_tx.send(SpeechEvent::Finished).await;
    }

    async fn synthesize(&self, sentence: &str) -> anyhow::Result<(Vec<f32>, u32, u16)> {
        match self.choice {
            TtsChoice::Builtin => {
                let bytes = self
                    .client
                    .speech(&self.model, sentence, &self.voice)
                    .await?;
                let (samples, sample_rate, channels) = decode_wav(&bytes)?;
                Ok((samples, sample_rate, channels))
            }
            TtsChoice::Paradee => {
                let engine = Arc::clone(&self.paradee);
                let voice = self.voice.clone();
                let sentence = sentence.to_string();
                match tokio::task::spawn_blocking(move || {
                    let mut guard = engine
                        .lock()
                        .map_err(|_| anyhow::anyhow!("Paradee engine lock failed"))?;
                    if guard.is_none() {
                        *guard = Some(crate::paradee::Paradee::load()?);
                    }
                    guard
                        .as_mut()
                        .expect("Paradee engine")
                        .synthesize(&sentence, &voice)
                })
                .await
                {
                    Ok(result) => {
                        let (samples, sample_rate) = result?;
                        Ok((samples, sample_rate, 1))
                    }
                    Err(err) => Err(anyhow::anyhow!("speech task failed: {err}")),
                }
            }
        }
    }
}

/// Accumulates streamed assistant text and emits sentences as soon as they are
/// complete. Fenced code blocks are skipped, and a partial final line is held
/// back until more text arrives or the stream finishes.
#[derive(Default)]
struct SpeechTextBuffer {
    pending: String,
    carry: String,
    fence: Option<&'static str>,
}

impl SpeechTextBuffer {
    fn clear(&mut self) {
        self.pending.clear();
        self.carry.clear();
        self.fence = None;
    }

    fn push(&mut self, delta: &str) -> Vec<String> {
        self.pending.push_str(delta);

        loop {
            if let Some(marker) = self.fence {
                let Some(position) = self.pending.find('\n') else {
                    break;
                };
                let line: String = self.pending.drain(..=position).collect();
                if line.trim_start().starts_with(marker) {
                    self.fence = None;
                }
                continue;
            }

            if let Some(position) = self.pending.find('\n') {
                let line: String = self.pending.drain(..=position).collect();
                let trimmed = line.trim_start();
                if trimmed.starts_with("```") {
                    self.fence = Some("```");
                    continue;
                }
                if trimmed.starts_with("~~~") {
                    self.fence = Some("~~~");
                    continue;
                }
                self.append(plain_text_for_speech(&line), true);
                continue;
            }

            let trimmed = self.pending.trim_start();
            if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
                break;
            }

            let (sentences, tail) = take_complete_sentences(&self.pending);
            self.pending = tail;
            for sentence in sentences {
                self.append(plain_text_for_speech(&sentence), false);
            }
            break;
        }

        let (sentences, rest) = take_complete_sentences(&self.carry);
        self.carry = rest;
        sentences
    }

    fn finish(&mut self) -> Vec<String> {
        if self.fence.is_none() {
            let plain = plain_text_for_speech(&self.pending);
            self.append(plain, false);
        }
        self.pending.clear();
        self.fence = None;

        let sentences = split_sentences(&self.carry);
        self.carry.clear();
        sentences
    }

    fn append(&mut self, plain: String, line_end: bool) {
        if !plain.is_empty() {
            if !self.carry.is_empty() && !self.carry.ends_with(char::is_whitespace) {
                self.carry.push(' ');
            }
            self.carry.push_str(&plain);
        }
        if line_end {
            self.carry.push('\n');
        }
    }
}

#[derive(Clone, Copy)]
struct TtsLanguage {
    code: &'static str,
    label: &'static str,
    builtin_suffix: &'static str,
}

const TTS_LANGUAGES: &[TtsLanguage] = &[
    TtsLanguage {
        code: "en-us",
        label: "English (US)",
        builtin_suffix: "american-english",
    },
    TtsLanguage {
        code: "en-gb",
        label: "English (UK)",
        builtin_suffix: "british-english",
    },
    TtsLanguage {
        code: "ar",
        label: "Arabic",
        builtin_suffix: "arabic",
    },
    TtsLanguage {
        code: "zh",
        label: "Chinese",
        builtin_suffix: "chinese",
    },
    TtsLanguage {
        code: "da",
        label: "Danish",
        builtin_suffix: "danish",
    },
    TtsLanguage {
        code: "nl",
        label: "Dutch",
        builtin_suffix: "dutch",
    },
    TtsLanguage {
        code: "fi",
        label: "Finnish",
        builtin_suffix: "finnish",
    },
    TtsLanguage {
        code: "fr",
        label: "French",
        builtin_suffix: "french",
    },
    TtsLanguage {
        code: "de",
        label: "German",
        builtin_suffix: "german",
    },
    TtsLanguage {
        code: "el",
        label: "Greek",
        builtin_suffix: "greek",
    },
    TtsLanguage {
        code: "he",
        label: "Hebrew",
        builtin_suffix: "hebrew",
    },
    TtsLanguage {
        code: "it",
        label: "Italian",
        builtin_suffix: "italian",
    },
    TtsLanguage {
        code: "ja",
        label: "Japanese",
        builtin_suffix: "japanese",
    },
    TtsLanguage {
        code: "ko",
        label: "Korean",
        builtin_suffix: "korean",
    },
    TtsLanguage {
        code: "pl",
        label: "Polish",
        builtin_suffix: "polish",
    },
    TtsLanguage {
        code: "pt",
        label: "Portuguese",
        builtin_suffix: "portuguese",
    },
    TtsLanguage {
        code: "es",
        label: "Spanish",
        builtin_suffix: "spanish",
    },
    TtsLanguage {
        code: "sv",
        label: "Swedish",
        builtin_suffix: "swedish",
    },
    TtsLanguage {
        code: "tr",
        label: "Turkish",
        builtin_suffix: "turkish",
    },
];

struct ParadeeVoice {
    id: &'static str,
    label: &'static str,
    language: &'static str,
}

const PARADEE_VOICES: &[ParadeeVoice] = &[
    ParadeeVoice {
        id: "af_heart",
        label: "Female (American)",
        language: "en-us",
    },
    ParadeeVoice {
        id: "am_puck",
        label: "Male (American)",
        language: "en-us",
    },
    ParadeeVoice {
        id: "bf_isabella",
        label: "Female (British)",
        language: "en-gb",
    },
    ParadeeVoice {
        id: "bm_fable",
        label: "Male (British)",
        language: "en-gb",
    },
];

#[derive(Clone, Debug, PartialEq, Eq)]
struct VoiceOption {
    id: String,
    label: String,
}

fn tts_language(code: &str) -> TtsLanguage {
    TTS_LANGUAGES
        .iter()
        .copied()
        .find(|language| language.code == code)
        .unwrap_or(TTS_LANGUAGES[0])
}

fn has_paradee_voice(code: &str) -> bool {
    PARADEE_VOICES.iter().any(|voice| voice.language == code)
}

fn paradee_voice_options(language: TtsLanguage) -> Vec<VoiceOption> {
    PARADEE_VOICES
        .iter()
        .filter(|voice| voice.language == language.code)
        .map(|voice| VoiceOption {
            id: voice.id.to_string(),
            label: voice.label.to_string(),
        })
        .collect()
}

fn builtin_voice_options(language: TtsLanguage) -> Vec<VoiceOption> {
    vec![
        VoiceOption {
            id: format!("female-{}", language.builtin_suffix),
            label: "Female".to_string(),
        },
        VoiceOption {
            id: format!("male-{}", language.builtin_suffix),
            label: "Male".to_string(),
        },
    ]
}

fn voice_options_for_choice(choice: TtsChoice, language: TtsLanguage) -> Vec<VoiceOption> {
    match choice {
        TtsChoice::Paradee => paradee_voice_options(language),
        TtsChoice::Builtin => builtin_voice_options(language),
    }
}

fn voice_options_for_backend(backend: &str, language: TtsLanguage) -> Vec<VoiceOption> {
    match backend {
        "paradee" => paradee_voice_options(language),
        "builtin" => builtin_voice_options(language),
        _ => {
            if has_paradee_voice(language.code) {
                paradee_voice_options(language)
            } else {
                builtin_voice_options(language)
            }
        }
    }
}

fn languages_for_backend(backend: &str) -> Vec<TtsLanguage> {
    if backend == "paradee" {
        TTS_LANGUAGES
            .iter()
            .copied()
            .filter(|language| has_paradee_voice(language.code))
            .collect()
    } else {
        TTS_LANGUAGES.to_vec()
    }
}

fn tts_backend_from_index(index: u32) -> &'static str {
    match index {
        1 => "paradee",
        2 => "builtin",
        _ => "auto",
    }
}

fn tts_backend_index(backend: &str) -> u32 {
    match backend {
        "paradee" => 1,
        "builtin" => 2,
        _ => 0,
    }
}

fn fill_language_row(
    row: &adw::ComboRow,
    backend: &str,
    current_code: &str,
    choices: &RefCell<Vec<TtsLanguage>>,
    suppress: &Cell<bool>,
) -> String {
    let items = languages_for_backend(backend);
    let labels: Vec<&str> = items.iter().map(|language| language.label).collect();
    let model = gtk::StringList::new(&labels);
    let selected = items
        .iter()
        .position(|language| language.code == current_code)
        .unwrap_or(0);
    let code = items
        .get(selected)
        .map(|language| language.code.to_string())
        .unwrap_or_else(|| "en-us".to_string());

    suppress.set(true);
    row.set_model(Some(&model));
    row.set_selected(selected as u32);
    suppress.set(false);

    *choices.borrow_mut() = items;
    code
}

#[derive(Clone, Copy)]
struct SttLanguage {
    code: &'static str,
    label: &'static str,
}

const STT_LANGUAGES: &[SttLanguage] = &[
    SttLanguage {
        code: "auto",
        label: "Auto detect",
    },
    SttLanguage {
        code: "en",
        label: "English",
    },
    SttLanguage {
        code: "de",
        label: "German",
    },
    SttLanguage {
        code: "fr",
        label: "French",
    },
    SttLanguage {
        code: "es",
        label: "Spanish",
    },
    SttLanguage {
        code: "it",
        label: "Italian",
    },
    SttLanguage {
        code: "nl",
        label: "Dutch",
    },
    SttLanguage {
        code: "pl",
        label: "Polish",
    },
    SttLanguage {
        code: "ar",
        label: "Arabic",
    },
    SttLanguage {
        code: "zh",
        label: "Chinese",
    },
    SttLanguage {
        code: "da",
        label: "Danish",
    },
    SttLanguage {
        code: "fi",
        label: "Finnish",
    },
    SttLanguage {
        code: "el",
        label: "Greek",
    },
    SttLanguage {
        code: "he",
        label: "Hebrew",
    },
    SttLanguage {
        code: "ja",
        label: "Japanese",
    },
    SttLanguage {
        code: "ko",
        label: "Korean",
    },
    SttLanguage {
        code: "pt",
        label: "Portuguese",
    },
    SttLanguage {
        code: "sv",
        label: "Swedish",
    },
    SttLanguage {
        code: "tr",
        label: "Turkish",
    },
];

fn stt_languages_for_provider(provider: &str) -> Vec<SttLanguage> {
    if provider == "parakeet" {
        STT_LANGUAGES
            .iter()
            .copied()
            .filter(|language| language.code == "auto")
            .collect()
    } else if provider == "whistle" {
        STT_LANGUAGES
            .iter()
            .copied()
            .filter(|language| {
                language.code == "auto"
                    || crate::whistle::WHISTLE_LANGUAGES.contains(&language.code)
            })
            .collect()
    } else {
        STT_LANGUAGES.to_vec()
    }
}

fn fill_stt_language_row(
    row: &adw::ComboRow,
    provider: &str,
    current_code: &str,
    choices: &RefCell<Vec<SttLanguage>>,
    suppress: &Cell<bool>,
) -> String {
    let items = stt_languages_for_provider(provider);
    let labels: Vec<&str> = items.iter().map(|language| language.label).collect();
    let model = gtk::StringList::new(&labels);
    let selected = items
        .iter()
        .position(|language| language.code == current_code)
        .unwrap_or(0);
    let code = items
        .get(selected)
        .map(|language| language.code.to_string())
        .unwrap_or_else(|| "auto".to_string());

    suppress.set(true);
    row.set_model(Some(&model));
    row.set_selected(selected as u32);
    suppress.set(false);

    *choices.borrow_mut() = items;
    code
}

fn stt_provider_from_index(index: u32) -> &'static str {
    match index {
        1 => "parakeet",
        2 => "whistle",
        3 => "builtin",
        _ => "auto",
    }
}

fn stt_provider_index(provider: &str) -> u32 {
    match provider {
        "parakeet" => 1,
        "whistle" => 2,
        "builtin" => 3,
        _ => 0,
    }
}

fn fill_voice_row(
    row: &adw::ComboRow,
    backend: &str,
    language: TtsLanguage,
    current_voice: &str,
    choices: &RefCell<Vec<VoiceOption>>,
    suppress: &Cell<bool>,
) -> String {
    let items = voice_options_for_backend(backend, language);
    let labels: Vec<&str> = items.iter().map(|voice| voice.label.as_str()).collect();
    let model = gtk::StringList::new(&labels);
    let selected = items
        .iter()
        .position(|voice| voice.id == current_voice)
        .unwrap_or(0);
    let id = items
        .get(selected)
        .map(|voice| voice.id.clone())
        .unwrap_or_default();

    suppress.set(true);
    row.set_model(Some(&model));
    row.set_selected(selected as u32);
    suppress.set(false);

    *choices.borrow_mut() = items;
    id
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TtsChoice {
    Paradee,
    Builtin,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SttChoice {
    Parakeet,
    Whistle,
    Builtin,
}

struct ChatWindow {
    window: adw::ApplicationWindow,
    state: Rc<AppState>,

    chat_list: gtk::ListBox,
    new_chat_button: gtk::Button,
    refresh_button: gtk::Button,
    model_dropdown: gtk::DropDown,
    capability_box: adw::WrapBox,
    status_label: gtk::Label,

    title_widget: adw::WindowTitle,
    main_stack: gtk::Stack,
    messages_box: gtk::Box,
    messages_clamp: adw::Clamp,
    messages_scroll: gtk::ScrolledWindow,
    input: gtk::TextView,
    attach_button: gtk::Button,
    attachments_box: gtk::Box,
    attachments: RefCell<Vec<PendingAttachment>>,
    next_attachment_id: Cell<u64>,
    send_button: gtk::Button,
    stop_button: gtk::Button,
    mic_button: gtk::Button,
    live_button: gtk::Button,
    think_button: gtk::ToggleButton,
    think_menu_button: gtk::MenuButton,
    think_menu_box: gtk::Box,
    think_bar: gtk::Box,
    think_control: RefCell<Option<ThinkingControl>>,
    thinking_model: RefCell<String>,
    context_label: gtk::Label,
    context_bar: gtk::LevelBar,
    toast_overlay: adw::ToastOverlay,
    menu_button: gtk::MenuButton,
    split_view: adw::OverlaySplitView,
    sidebar_button: gtk::Button,
    sidebar_animating: Cell<bool>,
    sidebar_animation_token: Cell<u64>,
    last_split_width: Cell<i32>,
    sidebar_width_stale: Cell<bool>,
    tray: RefCell<Option<ksni::Handle<StatusTray>>>,
    tray_live: Cell<bool>,
    tray_live_available: Cell<bool>,
    tray_wake_visible: Cell<bool>,
    tray_wake_listening: Cell<bool>,
    tray_wake_word: RefCell<String>,

    current_id: RefCell<Option<i64>>,
    active: RefCell<Option<ActiveStream>>,
    speech_buttons: RefCell<Vec<gtk::Button>>,
    message_pictures: RefCell<Vec<glib::WeakRef<gtk::Picture>>>,
    last_image_column: Cell<i32>,
    speech_player: RefCell<Option<async_channel::Sender<PlaybackChunk>>>,
    speech_active: Cell<bool>,
    speech_generation: Arc<AtomicU64>,
    speech_stream_tx: RefCell<Option<async_channel::Sender<String>>>,
    speech_text: RefCell<SpeechTextBuffer>,
    tts_idle_token: Arc<AtomicU64>,
    summary_notice: RefCell<Option<gtk::Box>>,
    compressing: Cell<bool>,
    pending_generation: RefCell<Option<GenerationRequest>>,
    memory_busy: Cell<bool>,
    memory_pending: RefCell<Option<(i64, String)>>,
    memory_token: Cell<u64>,
    memory_epoch: Cell<u64>,
    memory_lookup: Cell<bool>,
    memory_warned: Cell<bool>,
    stt_session: RefCell<Option<crate::stt::SttSession>>,
    stt_prefix: RefCell<String>,
    stt_listening: Cell<bool>,
    stt_preparing: Cell<bool>,
    stt_generation: Cell<u64>,
    stt_idle_token: Arc<AtomicU64>,
    stt_keep_alive: RefCell<Option<Arc<std::sync::atomic::AtomicBool>>>,
    live_mode: Cell<bool>,
    live_pending: Cell<bool>,
    /// True once the LLM was preloaded for the current live session.
    live_model_warmed: Cell<bool>,
    /// Whether the last live session ended with a sleep phrase.
    last_voice_end_goodbye: Cell<bool>,
    /// Set at wake start when the next user message should open a new chat.
    voice_new_chat_requested: Cell<bool>,
    wake_session: RefCell<Option<wake::WakeSession>>,
    wake_generation: Cell<u64>,
    wake_preparing: Cell<bool>,
    wake_listening: Cell<bool>,
    /// Session-only armed state. Wake detection never starts by itself.
    wake_active: Cell<bool>,
    /// True when the app was launched by its autostart entry.
    autostart: bool,
    /// Arms wake detection a few seconds after an autostart launch, unless the
    /// user changes the wake state first.
    autostart_wake_pending: Cell<bool>,
    wake_pending: Cell<bool>,
    wake_tx: async_channel::Sender<()>,
    wake_rx: RefCell<Option<async_channel::Receiver<()>>>,
    suppress_autoread: Cell<bool>,
    suppress_model_change: Cell<bool>,
}

pub fn build(app: &adw::Application, autostart: bool) -> anyhow::Result<adw::ApplicationWindow> {
    let state = Rc::new(AppState::new()?);
    let window = adw::ApplicationWindow::builder()
        .application(app)
        .title("Ollama GUI")
        .default_width(1080)
        .default_height(760)
        .build();

    let chat = ChatWindow::new(&window, state, autostart);
    chat.wire();
    chat.start();
    Ok(window)
}

impl ChatWindow {
    fn new(window: &adw::ApplicationWindow, state: Rc<AppState>, autostart: bool) -> Rc<Self> {
        let (wake_tx, wake_rx) = async_channel::unbounded();

        // ----- Sidebar -----------------------------------------------------
        let chat_list = gtk::ListBox::new();
        chat_list.add_css_class("navigation-sidebar");
        chat_list.set_selection_mode(gtk::SelectionMode::Single);
        chat_list.set_valign(gtk::Align::Start);

        let sidebar_scroll = gtk::ScrolledWindow::builder()
            .hscrollbar_policy(gtk::PolicyType::Never)
            .vexpand(true)
            .child(&chat_list)
            .build();

        let model_dropdown = gtk::DropDown::from_strings(&[]);
        model_dropdown.set_hexpand(true);
        model_dropdown.set_enable_search(true);
        model_dropdown.set_tooltip_text(Some("Model"));

        let capability_box = adw::WrapBox::new();
        capability_box.set_child_spacing(4);
        capability_box.set_line_spacing(4);
        capability_box.set_align(0.0);

        let status_label = gtk::Label::new(Some("Connecting…"));
        status_label.add_css_class("dim-label");
        status_label.set_xalign(0.0);
        status_label.set_wrap(true);
        status_label.set_tooltip_text(Some("Ollama connection status"));

        let footer = gtk::Box::new(gtk::Orientation::Vertical, 8);
        footer.set_margin_top(8);
        footer.set_margin_bottom(12);
        footer.set_margin_start(12);
        footer.set_margin_end(12);
        footer.append(&status_label);
        footer.append(&model_dropdown);
        footer.append(&capability_box);

        let sidebar_content = gtk::Box::new(gtk::Orientation::Vertical, 0);
        sidebar_content.append(&sidebar_scroll);
        sidebar_content.append(&footer);

        let sidebar_header = adw::HeaderBar::new();
        let sidebar_title = gtk::Label::new(Some("Chats"));
        sidebar_title.add_css_class("title");
        sidebar_header.set_title_widget(Some(&sidebar_title));

        let new_chat_button = gtk::Button::from_icon_name("list-add-symbolic");
        new_chat_button.set_tooltip_text(Some("New chat"));
        new_chat_button.add_css_class("flat");

        let refresh_button = gtk::Button::from_icon_name("view-refresh-symbolic");
        refresh_button.set_tooltip_text(Some("Refresh models"));
        refresh_button.add_css_class("flat");

        let sidebar_actions = gtk::Box::new(gtk::Orientation::Horizontal, 0);
        sidebar_actions.append(&refresh_button);
        sidebar_actions.append(&new_chat_button);
        sidebar_header.pack_end(&sidebar_actions);

        let sidebar_toolbar = adw::ToolbarView::new();
        sidebar_toolbar.add_top_bar(&sidebar_header);
        sidebar_toolbar.set_content(Some(&sidebar_content));

        // ----- Message area ------------------------------------------------
        let messages_box = gtk::Box::new(gtk::Orientation::Vertical, 12);
        messages_box.set_margin_top(18);
        messages_box.set_margin_bottom(18);
        messages_box.set_margin_start(18);
        messages_box.set_margin_end(18);

        let clamp = adw::Clamp::builder()
            .maximum_size(900)
            .tightening_threshold(900)
            .child(&messages_box)
            .build();

        let messages_scroll = gtk::ScrolledWindow::builder()
            .hscrollbar_policy(gtk::PolicyType::Never)
            .vexpand(true)
            .hexpand(true)
            .child(&clamp)
            .build();

        let empty_page = adw::StatusPage::builder()
            .title("Start a conversation")
            .description("Pick a model and say hello to your local Ollama.")
            .build();

        let main_stack = gtk::Stack::new();
        main_stack.set_transition_type(gtk::StackTransitionType::Crossfade);
        main_stack.add_named(&empty_page, Some("empty"));
        main_stack.add_named(&messages_scroll, Some("chat"));
        main_stack.set_visible_child_name("empty");
        main_stack.set_vexpand(true);

        // ----- Composer ----------------------------------------------------
        let input = gtk::TextView::new();
        input.set_wrap_mode(gtk::WrapMode::WordChar);
        input.set_accepts_tab(false);
        input.set_height_request(70);
        input.set_top_margin(4);
        input.set_bottom_margin(4);
        input.set_left_margin(6);
        input.set_right_margin(6);
        input.set_tooltip_text(Some(
            "Enter to send, Shift+Enter for a new line, Ctrl+L to focus, Ctrl+V to paste an image",
        ));

        let attach_button = gtk::Button::from_icon_name("insert-image-symbolic");
        attach_button.add_css_class("flat");
        attach_button.set_tooltip_text(Some("Attach images (Ctrl+Shift+A or Ctrl+V)"));
        attach_button.set_visible(false);

        let send_button = gtk::Button::from_icon_name("go-up-symbolic");
        send_button.add_css_class("suggested-action");
        send_button.add_css_class("circular");
        send_button.set_tooltip_text(Some("Send (Enter)"));

        let stop_button = gtk::Button::from_icon_name("media-playback-stop-symbolic");
        stop_button.add_css_class("destructive-action");
        stop_button.add_css_class("circular");
        stop_button.set_tooltip_text(Some("Stop generating"));
        stop_button.set_visible(false);

        let mic_button = gtk::Button::from_icon_name("audio-input-microphone-symbolic");
        mic_button.add_css_class("flat");
        mic_button.set_tooltip_text(Some("Dictate"));
        mic_button.set_visible(false);

        let live_button = gtk::Button::from_icon_name("media-playback-start-symbolic");
        live_button.add_css_class("flat");
        live_button.set_tooltip_text(Some("Start live voice conversation"));
        live_button.set_visible(false);

        let think_button = gtk::ToggleButton::with_label("Think");
        think_button.add_css_class("flat");
        think_button.add_css_class("think-toggle");
        think_button.set_tooltip_text(Some("Thinking mode"));
        think_button.set_active(state.settings.borrow().think_enabled);
        think_button.set_visible(false);

        let think_menu_box = gtk::Box::new(gtk::Orientation::Vertical, 2);
        think_menu_box.set_margin_top(6);
        think_menu_box.set_margin_bottom(6);
        think_menu_box.set_margin_start(6);
        think_menu_box.set_margin_end(6);
        let think_popover = gtk::Popover::new();
        think_popover.set_child(Some(&think_menu_box));

        let think_menu_button = gtk::MenuButton::builder().label("Think").build();
        think_menu_button.set_popover(Some(&think_popover));
        think_menu_button.add_css_class("flat");
        think_menu_button.add_css_class("think-toggle");
        think_menu_button.set_tooltip_text(Some("Thinking level"));
        think_menu_button.set_visible(false);

        let think_bar = gtk::Box::new(gtk::Orientation::Horizontal, 6);
        think_bar.set_margin_start(6);
        think_bar.set_margin_top(2);
        think_bar.append(&think_button);
        think_bar.append(&think_menu_button);
        think_bar.set_visible(false);

        let attachments_box = gtk::Box::new(gtk::Orientation::Horizontal, 6);
        attachments_box.add_css_class("attachment-strip");
        attachments_box.set_visible(false);

        let icon_row = gtk::Box::new(gtk::Orientation::Horizontal, 2);
        icon_row.set_halign(gtk::Align::End);
        icon_row.append(&attach_button);
        icon_row.append(&live_button);
        icon_row.append(&mic_button);

        let action_row = gtk::Box::new(gtk::Orientation::Horizontal, 2);
        action_row.set_halign(gtk::Align::End);
        action_row.append(&stop_button);
        action_row.append(&send_button);

        let composer_buttons = gtk::Box::new(gtk::Orientation::Vertical, 2);
        composer_buttons.set_valign(gtk::Align::Start);
        composer_buttons.set_halign(gtk::Align::End);
        composer_buttons.append(&icon_row);
        composer_buttons.append(&action_row);

        let input_row = gtk::Box::new(gtk::Orientation::Horizontal, 6);
        input_row.set_hexpand(true);
        input.set_hexpand(true);
        input_row.append(&input);
        input_row.append(&composer_buttons);

        let composer = gtk::Box::new(gtk::Orientation::Vertical, 2);
        composer.add_css_class("composer");
        composer.set_margin_top(6);
        composer.set_margin_bottom(12);
        composer.set_margin_start(12);
        composer.set_margin_end(12);
        composer.append(&think_bar);
        composer.append(&attachments_box);
        composer.append(&input_row);

        let content_box = gtk::Box::new(gtk::Orientation::Vertical, 0);

        let context_label = gtk::Label::new(Some("Context: 0 / 8,192 tokens · 0%"));
        context_label.add_css_class("dim-label");
        context_label.set_xalign(0.0);
        context_label.set_hexpand(true);
        context_label.set_tooltip_text(Some(
            "Estimated from the last Ollama reply: prompt tokens + generated tokens",
        ));

        let context_bar = gtk::LevelBar::for_interval(0.0, 1.0);
        context_bar.set_value(0.0);
        context_bar.set_size_request(140, -1);
        context_bar.set_valign(gtk::Align::Center);
        context_bar.add_css_class("context-level");

        let context_row = gtk::Box::new(gtk::Orientation::Horizontal, 10);
        context_row.set_margin_start(18);
        context_row.set_margin_end(18);
        context_row.set_margin_top(4);
        context_row.append(&context_label);
        context_row.append(&context_bar);

        content_box.append(&main_stack);
        content_box.append(&context_row);
        content_box.append(&composer);

        let title_widget = adw::WindowTitle::new("New Chat", "No model selected");
        let header = adw::HeaderBar::new();
        header.set_title_widget(Some(&title_widget));

        let sidebar_button = gtk::Button::from_icon_name("sidebar-show-symbolic");
        sidebar_button.add_css_class("flat");
        sidebar_button.set_tooltip_text(Some("Hide sidebar (Ctrl+B)"));
        header.pack_start(&sidebar_button);

        let menu_button = gtk::MenuButton::new();
        menu_button.set_icon_name("open-menu-symbolic");
        menu_button.set_tooltip_text(Some("Main menu"));
        header.pack_end(&menu_button);

        let content_toolbar = adw::ToolbarView::new();
        content_toolbar.add_top_bar(&header);
        content_toolbar.set_content(Some(&content_box));

        let sidebar_visible = state.settings.borrow().sidebar_visible;
        let split = adw::OverlaySplitView::new();
        split.set_sidebar(Some(&sidebar_toolbar));
        split.set_content(Some(&content_toolbar));
        split.set_min_sidebar_width(240.0);
        split.set_max_sidebar_width(320.0);
        split.set_show_sidebar(sidebar_visible);

        let toast_overlay = adw::ToastOverlay::new();
        toast_overlay.set_child(Some(&split));
        window.set_content(Some(&toast_overlay));

        Rc::new(Self {
            window: window.clone(),
            state,
            chat_list,
            new_chat_button,
            refresh_button,
            model_dropdown,
            capability_box,
            status_label,
            title_widget,
            main_stack,
            messages_box,
            messages_clamp: clamp,
            messages_scroll,
            input,
            attach_button,
            attachments_box,
            attachments: RefCell::new(Vec::new()),
            next_attachment_id: Cell::new(0),
            send_button,
            stop_button,
            mic_button,
            live_button,
            think_button,
            think_menu_button,
            think_menu_box,
            think_bar,
            think_control: RefCell::new(None),
            thinking_model: RefCell::new(String::new()),
            context_label,
            context_bar,
            toast_overlay,
            menu_button,
            split_view: split,
            sidebar_button,
            sidebar_animating: Cell::new(false),
            sidebar_animation_token: Cell::new(0),
            last_split_width: Cell::new(0),
            sidebar_width_stale: Cell::new(false),
            tray: RefCell::new(None),
            tray_live: Cell::new(false),
            tray_live_available: Cell::new(false),
            tray_wake_visible: Cell::new(false),
            tray_wake_listening: Cell::new(false),
            tray_wake_word: RefCell::new(String::new()),
            current_id: RefCell::new(None),
            active: RefCell::new(None),
            speech_buttons: RefCell::new(Vec::new()),
            message_pictures: RefCell::new(Vec::new()),
            last_image_column: Cell::new(0),
            speech_player: RefCell::new(None),
            speech_active: Cell::new(false),
            speech_generation: Arc::new(AtomicU64::new(0)),
            speech_stream_tx: RefCell::new(None),
            speech_text: RefCell::new(SpeechTextBuffer::default()),
            tts_idle_token: Arc::new(AtomicU64::new(0)),
            summary_notice: RefCell::new(None),
            compressing: Cell::new(false),
            pending_generation: RefCell::new(None),
            memory_busy: Cell::new(false),
            memory_pending: RefCell::new(None),
            memory_token: Cell::new(0),
            memory_epoch: Cell::new(0),
            memory_lookup: Cell::new(false),
            memory_warned: Cell::new(false),
            stt_session: RefCell::new(None),
            stt_prefix: RefCell::new(String::new()),
            stt_listening: Cell::new(false),
            stt_preparing: Cell::new(false),
            stt_generation: Cell::new(0),
            stt_idle_token: Arc::new(AtomicU64::new(0)),
            stt_keep_alive: RefCell::new(None),
            live_mode: Cell::new(false),
            live_pending: Cell::new(false),
            live_model_warmed: Cell::new(false),
            last_voice_end_goodbye: Cell::new(false),
            voice_new_chat_requested: Cell::new(false),
            wake_session: RefCell::new(None),
            wake_generation: Cell::new(0),
            wake_preparing: Cell::new(false),
            wake_listening: Cell::new(false),
            wake_active: Cell::new(false),
            autostart,
            autostart_wake_pending: Cell::new(false),
            wake_pending: Cell::new(false),
            wake_tx,
            wake_rx: RefCell::new(Some(wake_rx)),
            suppress_autoread: Cell::new(false),
            suppress_model_change: Cell::new(false),
        })
    }

    fn wire(self: &Rc<Self>) {
        self.sync_sidebar_visibility(self.split_view.shows_sidebar());
        {
            let win = self.clone();
            self.window.connect_visible_notify(move |_| {
                win.update_stt_ui();
            });
        }
        {
            let win = self.clone();
            glib::timeout_add_local(std::time::Duration::from_millis(250), move || {
                win.update_message_width();
                glib::ControlFlow::Continue
            });
        }
        {
            let win = self.clone();
            self.sidebar_button
                .connect_clicked(move |_| win.toggle_sidebar());
        }
        {
            let win = self.clone();
            self.attach_button
                .connect_clicked(move |_| win.choose_images());
        }
        {
            let win = self.clone();
            let drop_target =
                gtk::DropTarget::new(gdk::FileList::static_type(), gdk::DragAction::COPY);
            drop_target.connect_drop(move |_, value, _, _| {
                let Ok(files) = value.get::<gdk::FileList>() else {
                    return false;
                };
                let files = files.files();
                if files.is_empty() {
                    return false;
                }
                for file in files {
                    win.load_attachment(file);
                }
                true
            });
            self.input.add_controller(drop_target);
        }
        {
            let win = self.clone();
            let bypass = Rc::new(Cell::new(false));
            let bypass_in = Rc::clone(&bypass);
            self.input.connect_paste_clipboard(move |view| {
                if bypass_in.replace(false) {
                    // Re-entry from the file-list fallback below.
                    return;
                }

                let clipboard = view.clipboard();
                let formats = clipboard.formats();

                if formats_have_image(&formats) {
                    // The clipboard holds an image, not text. Keep the default
                    // handler from inserting the accompanying text/html.
                    glib::signal_stop_emission_by_name(view, "paste-clipboard");
                    if !win.selected_model_supports("vision") {
                        win.toast("The selected model does not support images.");
                        return;
                    }

                    let win = win.clone();
                    glib::spawn_future_local(async move {
                        match clipboard.read_texture_future().await {
                            Ok(Some(texture)) => {
                                let png = texture.save_to_png_bytes();
                                win.add_attachment_bytes("Pasted image", png.to_vec());
                            }
                            Ok(None) => win.toast("The clipboard does not contain an image."),
                            Err(err) => win.toast(&format!("Could not read the clipboard: {err}")),
                        }
                    });
                    return;
                }

                // File managers expose copied images as a file list instead.
                // Attach image files; if there are none, replay the paste so
                // non-image clipboard content keeps its normal behavior.
                if formats.contain_mime_type("text/uri-list") {
                    glib::signal_stop_emission_by_name(view, "paste-clipboard");
                    let view = view.clone();
                    let win = win.clone();
                    let bypass = Rc::clone(&bypass);
                    glib::spawn_future_local(async move {
                        let mut has_image = false;
                        if let Ok(value) = clipboard
                            .read_value_future(
                                gdk::FileList::static_type(),
                                glib::Priority::DEFAULT,
                            )
                            .await
                        {
                            if let Ok(files) = value.get::<gdk::FileList>() {
                                for file in files.files() {
                                    if file_is_image(&file) {
                                        has_image = true;
                                        win.load_attachment(file);
                                    }
                                }
                            }
                        }
                        if !has_image {
                            bypass.set(true);
                            view.emit_paste_clipboard();
                        }
                    });
                }
            });
        }
        {
            let win = self.clone();
            self.think_button.connect_toggled(move |button| {
                win.state.settings.borrow_mut().think_enabled = button.is_active();
                win.save_settings();
                win.update_think_button();
            });
        }
        {
            let win = self.clone();
            self.split_view.connect_show_sidebar_notify(move |split| {
                win.sync_sidebar_visibility(split.shows_sidebar());
            });
        }
        {
            let win = self.clone();
            self.send_button
                .connect_clicked(move |_| win.send_message());
        }
        {
            let win = self.clone();
            self.stop_button
                .connect_clicked(move |_| win.stop_streaming());
        }
        {
            let win = self.clone();
            self.mic_button.connect_clicked(move |_| win.toggle_stt());
        }
        {
            let win = self.clone();
            self.live_button.connect_clicked(move |_| win.toggle_live());
        }
        {
            let win = self.clone();
            let controller = gtk::EventControllerKey::new();
            controller.connect_key_pressed(move |_, key, _, modifier| {
                let enter = key == gdk::Key::Return || key == gdk::Key::KP_Enter;
                if enter && !modifier.contains(gdk::ModifierType::SHIFT_MASK) {
                    win.send_message();
                    glib::Propagation::Stop
                } else {
                    glib::Propagation::Proceed
                }
            });
            self.input.add_controller(controller);
        }
        {
            let win = self.clone();
            let controller = gtk::EventControllerKey::new();
            controller.connect_key_pressed(move |_, key, _, modifier| {
                let control = modifier.contains(gdk::ModifierType::CONTROL_MASK);
                let shift = modifier.contains(gdk::ModifierType::SHIFT_MASK);
                let character = key.to_unicode().map(|value| value.to_ascii_lowercase());
                if control && shift && character == Some('a') {
                    win.choose_images();
                    return glib::Propagation::Stop;
                }
                if control && !shift && character == Some('l') {
                    win.input.grab_focus();
                    return glib::Propagation::Stop;
                }
                glib::Propagation::Proceed
            });
            self.window.add_controller(controller);
        }
        {
            let input = self.input.clone();
            self.input.buffer().connect_changed(move |buffer| {
                let lines = buffer.line_count().clamp(1, 8);
                input.set_height_request(lines * 22 + 16);
            });
        }
        {
            let win = self.clone();
            self.chat_list.connect_row_activated(move |_, row| {
                if let Ok(id) = row.widget_name().parse::<i64>() {
                    win.load_conversation(id);
                }
            });
        }
        {
            let win = self.clone();
            self.new_chat_button
                .connect_clicked(move |_| win.new_chat());
        }
        {
            let win = self.clone();
            self.refresh_button
                .connect_clicked(move |_| win.refresh_models());
        }
        {
            let win = self.clone();
            self.model_dropdown.connect_selected_notify(move |_| {
                win.on_model_changed();
            });
        }

        // Menu + window actions
        let menu = gio::Menu::new();
        let section = gio::Menu::new();
        section.append(Some("New Chat"), Some("win.new-chat"));
        section.append(Some("Toggle Sidebar"), Some("win.toggle-sidebar"));
        section.append(Some("Preferences"), Some("win.preferences"));
        section.append(Some("About Ollama GUI"), Some("win.about"));
        section.append(Some("Quit"), Some("app.quit"));
        menu.append_section(None, &section);
        self.menu_button.set_menu_model(Some(&menu));

        {
            let action = gio::SimpleAction::new("new-chat", None);
            let win = self.clone();
            action.connect_activate(move |_, _| win.new_chat());
            self.window.add_action(&action);
        }
        {
            let action = gio::SimpleAction::new("toggle-sidebar", None);
            let win = self.clone();
            action.connect_activate(move |_, _| win.toggle_sidebar());
            self.window.add_action(&action);
        }
        {
            let action = gio::SimpleAction::new("preferences", None);
            let win = self.clone();
            action.connect_activate(move |_, _| win.show_settings());
            self.window.add_action(&action);
        }
        {
            let action = gio::SimpleAction::new("about", None);
            let win = self.clone();
            action.connect_activate(move |_, _| win.show_about());
            self.window.add_action(&action);
        }

        if let Some(app) = self.window.application() {
            app.set_accels_for_action("win.new-chat", &["<primary>n"]);
            app.set_accels_for_action("win.toggle-sidebar", &["<primary>b"]);
            app.set_accels_for_action("win.preferences", &["<primary>comma"]);
        }
    }

    fn start(self: &Rc<Self>) {
        self.apply_theme();
        self.reload_sidebar();

        // An autostart entry only makes sense with the tray, so keep the
        // background setting consistent even if it was edited externally.
        if autostart::is_enabled() && !self.state.settings.borrow().run_in_background {
            self.state.settings.borrow_mut().run_in_background = true;
            self.save_settings();
        }

        let first = self
            .state
            .store
            .borrow()
            .list_conversations()
            .unwrap_or_default()
            .into_iter()
            .next();

        match first {
            Some(conversation) => self.load_conversation(conversation.id),
            None => self.show_empty(),
        }

        self.refresh_models();
        self.input.grab_focus();
        self.update_stt_ui();
        self.apply_background_startup();

        // Autostart is the only launch path that arms wake detection on its
        // own. Delay it a little so the tray host and microphone are ready.
        if self.autostart && self.state.settings.borrow().wake_enabled {
            self.autostart_wake_pending.set(true);
            let win = self.clone();
            glib::timeout_add_seconds_local_once(AUTOSTART_WAKE_DELAY_SECS, move || {
                if win.autostart_wake_pending.replace(false) {
                    win.wake_active.set(true);
                    win.refresh_wake();
                }
            });
        }

        if let Some(rx) = self.wake_rx.borrow_mut().take() {
            let win = self.clone();
            glib::spawn_future_local(async move {
                while rx.recv().await.is_ok() {
                    win.refresh_wake();
                }
            });
        }
        self.refresh_wake();
    }

    // ----- Models ----------------------------------------------------------

    fn selected_model(&self) -> String {
        let index = self.model_dropdown.selected();
        if index == u32::MAX {
            return String::new();
        }
        self.model_dropdown
            .model()
            .and_downcast::<gtk::StringList>()
            .and_then(|list| list.string(index))
            .map(|value| value.to_string())
            .unwrap_or_default()
    }

    fn select_model_in_dropdown(self: &Rc<Self>, model: &str) {
        let Some(list) = self
            .model_dropdown
            .model()
            .and_downcast::<gtk::StringList>()
        else {
            return;
        };
        let mut found = None;
        for index in 0..list.n_items() {
            if list.string(index).map(|value| value.to_string()).as_deref() == Some(model) {
                found = Some(index);
                break;
            }
        }
        if let Some(index) = found {
            self.suppress_model_change.set(true);
            self.model_dropdown.set_selected(index);
            self.suppress_model_change.set(false);
        }
        self.update_capabilities();
    }

    fn update_capabilities(self: &Rc<Self>) {
        while let Some(child) = self.capability_box.first_child() {
            child.unparent();
        }
        self.refresh_thinking_control();
        self.update_attachment_ui();

        let selected = self.selected_model();
        if selected.is_empty() {
            self.update_think_button();
            self.update_speech_buttons();
            return;
        }

        let models = self.state.models.borrow();
        let Some(model) = models.iter().find(|model| model.name == selected) else {
            drop(models);
            self.update_think_button();
            self.update_speech_buttons();
            return;
        };

        for capability in &model.capabilities {
            let label = gtk::Label::new(Some(capability));
            label.add_css_class("capability-pill");
            if let Some(class) = capability_css_class(capability) {
                label.add_css_class(class);
            }
            label.set_tooltip_text(capability_description(capability));
            self.capability_box.append(&label);
        }
        self.update_think_button();
        self.update_speech_buttons();
        self.update_tray_state();
    }

    fn refresh_models(self: &Rc<Self>) {
        self.set_status("Connecting…", None);
        let client = self.state.client.borrow().clone();
        let (tx, rx) = async_channel::bounded(1);
        self.state.runtime.spawn(async move {
            let result = client.list_models().await;
            let _ = tx.send(result).await;
        });

        let win = self.clone();
        glib::spawn_future_local(async move {
            match rx.recv().await {
                Ok(Ok(models)) => win.apply_models(models),
                Ok(Err(err)) => win.set_offline(&format!("{err:#}")),
                Err(_) => {}
            }
        });
    }

    fn apply_models(self: &Rc<Self>, models: Vec<ModelSummary>) {
        let names: Vec<String> = models
            .iter()
            .filter(|model| model.is_chat_model())
            .map(|model| model.name.clone())
            .collect();
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let list = gtk::StringList::new(&refs);

        let previous = self.selected_model();
        let conversation_model = self.current_id.borrow().and_then(|id| {
            self.state
                .store
                .borrow()
                .conversation_model(id)
                .ok()
                .flatten()
        });
        let default_model = self.state.settings.borrow().default_model.clone();

        self.suppress_model_change.set(true);
        self.model_dropdown.set_model(Some(&list));
        let preferred = conversation_model
            .filter(|model| names.contains(model))
            .or_else(|| (!previous.is_empty() && names.contains(&previous)).then_some(previous))
            .or_else(|| {
                (!default_model.is_empty() && names.contains(&default_model))
                    .then_some(default_model)
            })
            .or_else(|| names.first().cloned());

        if let Some(model) = preferred {
            if let Some(index) = names.iter().position(|name| *name == model) {
                self.model_dropdown.set_selected(index as u32);
            }
        } else {
            self.model_dropdown.set_selected(u32::MAX);
        }
        self.suppress_model_change.set(false);

        *self.state.models.borrow_mut() = models;

        if names.is_empty() {
            self.set_status("Online · no chat models installed", Some(true));
        } else {
            let plural = if names.len() == 1 { "" } else { "s" };
            self.set_status(
                &format!("Online · {} chat model{plural}", names.len()),
                Some(true),
            );
        }

        let model = self.selected_model();
        if !model.is_empty() {
            self.title_widget.set_subtitle(&model);
        }
        self.update_capabilities();
        self.update_stt_ui();
    }

    fn on_model_changed(self: &Rc<Self>) {
        if self.suppress_model_change.get() {
            return;
        }
        let model = self.selected_model();
        if model.is_empty() {
            return;
        }
        self.title_widget.set_subtitle(&model);
        self.state.settings.borrow_mut().default_model = model.clone();
        self.save_settings();
        self.update_capabilities();
        if self.resolve_tts().is_none() {
            self.stop_speech();
        }
        if let Some(id) = *self.current_id.borrow() {
            let _ = self.state.store.borrow().set_conversation_model(id, &model);
            self.reload_sidebar();
        }
    }

    fn set_status(&self, text: &str, online: Option<bool>) {
        self.status_label.set_text(text);
        self.status_label.remove_css_class("status-online");
        self.status_label.remove_css_class("status-offline");
        match online {
            Some(true) => {
                self.status_label.remove_css_class("dim-label");
                self.status_label.add_css_class("status-online")
            }
            Some(false) => {
                self.status_label.remove_css_class("dim-label");
                self.status_label.add_css_class("status-offline")
            }
            None => self.status_label.add_css_class("dim-label"),
        }
    }

    fn set_offline(&self, detail: &str) {
        self.set_status("Offline · check Preferences", Some(false));
        self.status_label.set_tooltip_text(Some(detail));
        tracing::warn!("Ollama unavailable: {detail}");
    }

    // ----- Context usage ---------------------------------------------------

    fn update_context_indicator(&self, used: Option<u32>) {
        let total = self.state.settings.borrow().num_ctx.max(1);
        let used_tokens = used.unwrap_or(0);
        let fraction = used_tokens as f64 / total as f64;
        let percent = fraction * 100.0;

        self.context_label.set_text(&format!(
            "Context: {} / {} tokens · {}",
            group_digits(used_tokens),
            group_digits(total),
            format_percent(percent)
        ));
        self.context_bar.set_value(fraction.clamp(0.0, 1.0));
        apply_usage_class(&self.context_label, percent);
        self.context_label.set_tooltip_text(Some(&format!(
            "Estimated context usage: {} of {} tokens ({})",
            group_digits(used_tokens),
            group_digits(total),
            format_percent(percent)
        )));
    }

    fn refresh_context_indicator(&self) {
        let used = self.current_id.borrow().and_then(|id| {
            self.state
                .store
                .borrow()
                .conversation_context_used(id)
                .ok()
                .flatten()
        });
        self.update_context_indicator(used);
    }

    /// Show or hide the "earlier messages summarized" row above the chat.
    fn refresh_summary_notice(&self, conversation_id: i64) {
        if let Some(notice) = self.summary_notice.borrow_mut().take() {
            self.messages_box.remove(&notice);
        }
        if *self.current_id.borrow() != Some(conversation_id) {
            return;
        }

        let summary = self
            .state
            .store
            .borrow()
            .conversation_summary(conversation_id)
            .ok()
            .flatten();
        let Some(summary) = summary else {
            return;
        };

        let row = gtk::Box::new(gtk::Orientation::Horizontal, 8);
        row.add_css_class("card");
        row.set_margin_start(12);
        row.set_margin_end(12);
        row.set_margin_top(6);
        row.set_margin_bottom(6);

        let icon = gtk::Image::from_icon_name("document-open-recent-symbolic");
        icon.set_valign(gtk::Align::Center);
        row.append(&icon);

        let label = gtk::Label::new(Some(&format!(
            "Earlier messages summarized (~{} tokens)",
            group_digits(summary.tokens)
        )));
        label.set_xalign(0.0);
        label.set_hexpand(true);
        label.set_wrap(true);
        label.add_css_class("dim-label");
        row.append(&label);

        let button = gtk::Button::with_label("View");
        button.add_css_class("flat");
        button.set_valign(gtk::Align::Center);
        let text = summary.text.clone();
        button.connect_clicked(move |button| {
            let parent = button
                .root()
                .and_then(|root| root.downcast::<gtk::Window>().ok());
            show_summary_dialog(parent.as_ref(), &text);
        });
        row.append(&button);

        self.messages_box.prepend(&row);
        *self.summary_notice.borrow_mut() = Some(row);
    }

    // ----- Conversations ---------------------------------------------------

    fn reload_sidebar(self: &Rc<Self>) {
        while let Some(child) = self.chat_list.first_child() {
            self.chat_list.remove(&child);
        }

        let conversations = self
            .state
            .store
            .borrow()
            .list_conversations()
            .unwrap_or_default();
        let current = *self.current_id.borrow();
        let num_ctx = self.state.settings.borrow().num_ctx.max(1);

        for conversation in conversations {
            let action_row = adw::ActionRow::builder()
                .title(&conversation.title)
                .subtitle(&conversation.model)
                .activatable(true)
                .build();

            let used = conversation.context_used.unwrap_or(0);
            let percent = used as f64 / num_ctx as f64 * 100.0;
            let percent_label = gtk::Label::new(Some(&format_percent(percent)));
            percent_label.set_valign(gtk::Align::Center);
            percent_label.set_width_chars(5);
            percent_label.set_xalign(1.0);
            apply_usage_class(&percent_label, percent);
            percent_label.set_tooltip_text(Some(&format!(
                "Context: {} / {} tokens ({})",
                group_digits(used),
                group_digits(num_ctx),
                format_percent(percent)
            )));
            action_row.add_suffix(&percent_label);

            let delete_button = gtk::Button::from_icon_name("user-trash-symbolic");
            delete_button.add_css_class("flat");
            delete_button.set_tooltip_text(Some("Delete chat"));
            delete_button.set_valign(gtk::Align::Center);
            action_row.add_suffix(&delete_button);
            action_row.set_widget_name(&conversation.id.to_string());

            {
                let win = self.clone();
                let id = conversation.id;
                delete_button.connect_clicked(move |_| {
                    if let Err(err) = win.state.store.borrow().delete_conversation(id) {
                        win.toast(&format!("Could not delete chat: {err:#}"));
                        return;
                    }
                    if *win.current_id.borrow() == Some(id) {
                        win.new_chat();
                    }
                    win.reload_sidebar();
                });
            }

            self.chat_list.append(&action_row);

            if current == Some(conversation.id) {
                self.chat_list.select_row(Some(&action_row));
            }
        }
    }

    fn load_conversation(self: &Rc<Self>, id: i64) {
        self.stop_streaming();
        self.stop_speech();
        self.stop_stt();
        *self.pending_generation.borrow_mut() = None;
        *self.current_id.borrow_mut() = Some(id);

        let (title, model, messages, images) = {
            let store = self.state.store.borrow();
            (
                store
                    .conversation_title(id)
                    .ok()
                    .flatten()
                    .unwrap_or_else(|| "Chat".to_string()),
                store
                    .conversation_model(id)
                    .ok()
                    .flatten()
                    .unwrap_or_default(),
                store.messages(id).unwrap_or_default(),
                store.images_for_conversation(id).unwrap_or_default(),
            )
        };

        self.clear_messages();
        self.refresh_summary_notice(id);
        for message in &messages {
            let message_images = images
                .get(&message.id)
                .map(Vec::as_slice)
                .unwrap_or_default();
            self.add_message_widget(id, message, message_images);
        }

        self.title_widget.set_title(&title);
        if model.is_empty() {
            self.title_widget.set_subtitle("No model selected");
        } else {
            self.title_widget.set_subtitle(&model);
        }
        self.select_model_in_dropdown(&model);

        if messages.is_empty() {
            self.main_stack.set_visible_child_name("empty");
        } else {
            self.main_stack.set_visible_child_name("chat");
            self.scroll_to_bottom();
        }
        self.refresh_context_indicator();
    }

    fn show_empty(&self) {
        self.clear_messages();
        self.main_stack.set_visible_child_name("empty");
        self.title_widget.set_title("New Chat");
        let model = self.selected_model();
        if model.is_empty() {
            self.title_widget.set_subtitle("No model selected");
        } else {
            self.title_widget.set_subtitle(&model);
        }
        self.refresh_context_indicator();
    }

    fn new_chat(self: &Rc<Self>) {
        self.stop_streaming();
        self.stop_speech();
        self.stop_stt();
        *self.pending_generation.borrow_mut() = None;
        *self.current_id.borrow_mut() = None;
        self.chat_list.unselect_all();
        self.show_empty();
        self.input.grab_focus();
    }

    // ----- Messages --------------------------------------------------------

    fn clear_messages(&self) {
        while let Some(child) = self.messages_box.first_child() {
            self.messages_box.remove(&child);
        }
        *self.summary_notice.borrow_mut() = None;
        self.speech_buttons.borrow_mut().clear();
        self.message_pictures.borrow_mut().clear();
    }

    fn add_message_widget(
        self: &Rc<Self>,
        conversation_id: i64,
        message: &StoredMessage,
        images: &[StoredImage],
    ) {
        if message.role == "user" {
            self.add_user_message(conversation_id, Some(message.id), &message.content, images);
        } else {
            let (_row, bubble, view) = self.add_assistant_bubble();
            // A GtkTextView populated before its first allocation caches a zero
            // height. Render once it has a width, then request one more resize
            // after that allocation so wrapped markdown gets its real height.
            let rendered = Rc::new(Cell::new(false));
            let resized = Rc::new(Cell::new(false));
            let content = message.content.clone();
            {
                let rendered = Rc::clone(&rendered);
                let resized = Rc::clone(&resized);
                view.add_tick_callback(move |view, _clock| {
                    if view.width() == 0 {
                        return glib::ControlFlow::Continue;
                    }
                    if !rendered.replace(true) {
                        super::markdown::render(view, &content);
                        view.queue_resize();
                        return glib::ControlFlow::Continue;
                    }
                    if !resized.replace(true) {
                        view.queue_resize();
                    }
                    glib::ControlFlow::Break
                });
            }
            self.attach_message_actions(
                conversation_id,
                &bubble,
                "assistant",
                &message.content,
                &[],
                Some(message.id),
            );
        }
    }

    fn add_user_message(
        self: &Rc<Self>,
        conversation_id: i64,
        message_id: Option<i64>,
        text: &str,
        images: &[StoredImage],
    ) {
        let bubble = gtk::Box::new(gtk::Orientation::Vertical, 6);
        bubble.add_css_class("user-bubble");
        bubble.set_valign(gtk::Align::Start);
        bubble.set_halign(if images.is_empty() {
            gtk::Align::End
        } else {
            gtk::Align::Fill
        });

        if !images.is_empty() {
            let column = if self.message_column_width() > 0 {
                self.message_column_width()
            } else {
                900
            };
            for image in images {
                let widget = image::message_picture(image, self.window.upcast_ref());
                widget.set_halign(gtk::Align::Fill);
                widget.set_valign(gtk::Align::Start);
                bubble.append(&widget);

                let Some(picture) = widget.downcast_ref::<gtk::Picture>() else {
                    continue;
                };
                fit_message_picture(picture, column);
                self.message_pictures.borrow_mut().push(picture.downgrade());

                // Once the row is allocated, match the card exactly to the
                // bubble width instead of the estimate.
                let picture = picture.clone();
                glib::idle_add_local_once(move || fit_message_picture(&picture, column));
            }
        }

        if !text.is_empty() {
            let label = gtk::Label::new(Some(text));
            label.set_wrap(true);
            label.set_wrap_mode(gtk::pango::WrapMode::WordChar);
            label.set_max_width_chars(60);
            label.set_selectable(true);
            label.set_xalign(0.0);
            bubble.append(&label);
        }

        if images.is_empty() {
            let row = gtk::Box::new(gtk::Orientation::Horizontal, 0);
            row.set_halign(gtk::Align::End);
            row.append(&bubble);
            self.messages_box.append(&row);
        } else {
            // Two homogeneous columns keep the media bubble at 50% of the
            // message column, independent of the source image dimensions.
            let grid = gtk::Grid::new();
            grid.set_column_homogeneous(true);
            grid.set_hexpand(true);
            grid.set_halign(gtk::Align::Fill);
            grid.set_valign(gtk::Align::Start);
            let spacer = gtk::Box::new(gtk::Orientation::Horizontal, 0);
            spacer.set_hexpand(true);
            grid.attach(&spacer, 0, 0, 1, 1);
            bubble.set_hexpand(true);
            grid.attach(&bubble, 1, 0, 1, 1);
            self.messages_box.append(&grid);
        }

        self.attach_message_actions(conversation_id, &bubble, "user", text, images, message_id);
    }

    fn add_assistant_bubble(&self) -> (gtk::Box, gtk::Box, gtk::TextView) {
        let view = gtk::TextView::builder()
            .editable(false)
            .cursor_visible(false)
            .wrap_mode(gtk::WrapMode::WordChar)
            .build();
        view.add_css_class("message-text");
        view.set_can_focus(true);

        let bubble = gtk::Box::new(gtk::Orientation::Vertical, 4);
        bubble.add_css_class("assistant-bubble");
        bubble.set_halign(gtk::Align::Fill);
        bubble.set_hexpand(true);
        bubble.append(&view);

        let row = gtk::Box::new(gtk::Orientation::Horizontal, 0);
        row.set_halign(gtk::Align::Fill);
        row.set_hexpand(true);
        row.append(&bubble);
        self.messages_box.append(&row);

        (row, bubble, view)
    }

    fn attach_message_actions(
        self: &Rc<Self>,
        conversation_id: i64,
        bubble: &gtk::Box,
        role: &str,
        content: &str,
        images: &[StoredImage],
        message_id: Option<i64>,
    ) {
        let Some(message_id) = message_id else {
            return;
        };

        let actions = gtk::Box::new(gtk::Orientation::Horizontal, 2);
        actions.set_halign(gtk::Align::End);
        actions.set_margin_top(2);

        if role == "assistant" {
            let regenerate = message_action_button("view-refresh-symbolic", "Regenerate reply");
            {
                let win = self.clone();
                regenerate.connect_clicked(move |_| {
                    win.regenerate(conversation_id, message_id);
                });
            }
            actions.append(&regenerate);
        }

        if !content.trim().is_empty() || !images.is_empty() {
            let copy_image = content.trim().is_empty();
            let copy = message_action_button(
                "edit-copy-symbolic",
                if copy_image {
                    "Copy image"
                } else {
                    "Copy message"
                },
            );
            {
                let text = content.to_string();
                let image_data = images.first().map(|image| image.data.clone());
                copy.connect_clicked(move |button| {
                    if copy_image {
                        if let Some(data) = &image_data {
                            if let Some(texture) = image::texture(data) {
                                if let Some(display) = gdk::Display::default() {
                                    display.clipboard().set_texture(&texture);
                                }
                            }
                        }
                    } else if let Some(display) = gdk::Display::default() {
                        display.clipboard().set_text(&text);
                    }
                    button.set_icon_name("object-select-symbolic");
                    let button = button.clone();
                    glib::timeout_add_seconds_local_once(1, move || {
                        button.set_icon_name("edit-copy-symbolic");
                    });
                });
            }
            actions.append(&copy);
        }

        if role == "assistant" {
            let speech = message_action_button("audio-volume-high-symbolic", "Read aloud");
            speech.set_visible(self.resolve_tts().is_some());
            {
                let win = self.clone();
                let text = content.to_string();
                speech.connect_clicked(move |button| win.speak(&text, Some(button)));
            }
            actions.append(&speech);
            self.speech_buttons.borrow_mut().push(speech);
        }

        let remove = message_action_button("user-trash-symbolic", "Remove turn");
        {
            let win = self.clone();
            remove.connect_clicked(move |_| {
                win.remove_turn(conversation_id, message_id);
            });
        }
        actions.append(&remove);

        bubble.append(&actions);
        self.update_speech_buttons();
    }

    fn selected_model_supports(&self, capability: &str) -> bool {
        let selected = self.selected_model();
        self.state
            .models
            .borrow()
            .iter()
            .find(|model| model.name == selected)
            .map(|model| model.capabilities.iter().any(|item| item == capability))
            .unwrap_or(false)
    }

    /// Cloud models do not have local weights, so preloading them is useless.
    fn selected_model_is_remote(&self) -> bool {
        let model = self.selected_model();
        let remote_host = self
            .state
            .models
            .borrow()
            .iter()
            .find(|candidate| memory::model_names_match(&candidate.name, &model))
            .and_then(|candidate| candidate.remote_host.clone());
        let base_url = self.state.settings.borrow().base_url.clone();
        memory::model_is_cloud(&model, remote_host.as_deref(), &base_url)
    }

    fn refresh_thinking_control(self: &Rc<Self>) {
        let model = self.selected_model();
        if model.is_empty() || !self.selected_model_supports("thinking") {
            self.thinking_model.borrow_mut().clear();
            *self.think_control.borrow_mut() = None;
            self.rebuild_think_menu();
            self.update_think_button();
            return;
        }
        if *self.thinking_model.borrow() == model {
            self.update_think_button();
            return;
        }

        *self.thinking_model.borrow_mut() = model.clone();
        *self.think_control.borrow_mut() = None;
        self.update_think_button();

        let client = self.state.client.borrow().clone();
        let request_model = model.clone();
        let (tx, rx) = async_channel::bounded(1);
        self.state.runtime.spawn(async move {
            let result = client.model_thinking(&request_model).await;
            let _ = tx.send(result).await;
        });

        let win = self.clone();
        glib::spawn_future_local(async move {
            let Ok(result) = rx.recv().await else {
                return;
            };
            if win.selected_model() != model {
                return;
            }
            match result {
                Ok(control) => win.set_thinking_control(control),
                Err(err) => {
                    tracing::warn!("thinking controls unavailable: {err:#}");
                    win.set_thinking_control(None);
                }
            }
        });
    }

    fn set_thinking_control(self: &Rc<Self>, control: Option<ThinkingControl>) {
        if let Some(control) = &control {
            if control.is_level_only() {
                let stored = self.state.settings.borrow().think_level.clone();
                let preferred = control.preferred_level(&stored);
                if preferred != stored {
                    self.state.settings.borrow_mut().think_level = preferred;
                    self.save_settings();
                }
            }
        }

        *self.think_control.borrow_mut() = control;
        self.rebuild_think_menu();
        self.update_think_button();
    }

    fn rebuild_think_menu(self: &Rc<Self>) {
        while let Some(child) = self.think_menu_box.first_child() {
            self.think_menu_box.remove(&child);
        }

        let Some(control) = self.think_control.borrow().clone() else {
            return;
        };
        if !control.is_level_only() {
            return;
        }

        let selected = self.selected_think_level(&control);
        for level in &control.levels {
            let button = gtk::Button::new();
            button.add_css_class("flat");

            let row = gtk::Box::new(gtk::Orientation::Horizontal, 6);
            let check = gtk::Image::from_icon_name("object-select-symbolic");
            check.set_visible(*level == selected);
            row.append(&check);
            let label = gtk::Label::new(Some(&title_case(level)));
            label.set_xalign(0.0);
            label.set_hexpand(true);
            row.append(&label);
            button.set_child(Some(&row));

            let win = self.clone();
            let level = level.clone();
            button.connect_clicked(move |_| {
                win.state.settings.borrow_mut().think_level = level.clone();
                win.save_settings();
                win.think_menu_button.popdown();
                win.rebuild_think_menu();
                win.update_think_button();
            });
            self.think_menu_box.append(&button);
        }
    }

    fn selected_think_level(&self, control: &ThinkingControl) -> String {
        control.preferred_level(&self.state.settings.borrow().think_level)
    }

    fn think_setting(&self) -> Option<ThinkSetting> {
        if !self.selected_model_supports("thinking") {
            return None;
        }

        let live = self.live_mode.get() || self.live_pending.get();
        let control = self.think_control.borrow().clone();
        if let Some(control) = control.filter(|control| control.is_level_only()) {
            let level = if live {
                control.lowest_level().unwrap_or("low").to_string()
            } else {
                self.selected_think_level(&control)
            };
            return Some(ThinkSetting::Level(level));
        }

        let enabled = self.state.settings.borrow().think_enabled && !live;
        Some(ThinkSetting::Bool(enabled))
    }

    fn update_think_button(&self) {
        let supported = self.selected_model_supports("thinking");
        let live = self.live_mode.get() || self.live_pending.get();
        let control = self.think_control.borrow().clone();

        self.think_bar.set_visible(supported);
        if !supported {
            self.think_button.set_visible(false);
            self.think_menu_button.set_visible(false);
            return;
        }

        let level_only = control.as_ref().is_some_and(ThinkingControl::is_level_only);
        self.think_button.set_visible(!level_only);
        self.think_menu_button.set_visible(level_only);

        if let Some(control) = control.as_ref().filter(|control| control.is_level_only()) {
            let level = self.selected_think_level(control);
            self.think_menu_button
                .set_label(&format!("Think: {}", title_case(&level)));
            self.think_menu_button.set_sensitive(!live);
            self.think_menu_button.set_tooltip_text(Some(if live {
                "Live mode uses the lowest reasoning level"
            } else {
                "Reasoning level"
            }));
        } else {
            self.think_button.set_sensitive(!live);
            self.think_button.set_tooltip_text(Some(if live {
                "Thinking is disabled during live conversation"
            } else if self.think_button.is_active() {
                "Thinking mode is on"
            } else {
                "Thinking mode is off"
            }));
        }
    }

    fn update_speech_buttons(&self) {
        let available = self.resolve_tts().is_some();
        for button in self.speech_buttons.borrow().iter() {
            button.set_visible(available);
        }
    }

    fn remove_turn(self: &Rc<Self>, conversation_id: i64, message_id: i64) {
        if self.active.borrow().is_some() {
            self.toast("Wait for the current reply to finish.");
            return;
        }

        let removed = match self
            .state
            .store
            .borrow()
            .delete_turn(conversation_id, message_id)
        {
            Ok(ids) => ids,
            Err(err) => {
                self.toast(&format!("Could not remove turn: {err:#}"));
                return;
            }
        };
        if removed.is_empty() {
            return;
        }

        if *self.current_id.borrow() == Some(conversation_id) {
            self.load_conversation(conversation_id);
        }
        self.reload_sidebar();
        self.toast("Turn removed");
    }

    fn regenerate(self: &Rc<Self>, conversation_id: i64, assistant_id: i64) {
        if self.active.borrow().is_some() {
            self.toast("Wait for the current reply to finish.");
            return;
        }

        let removed = match self
            .state
            .store
            .borrow()
            .delete_from_message(conversation_id, assistant_id)
        {
            Ok(count) => count,
            Err(err) => {
                self.toast(&format!("Could not regenerate: {err:#}"));
                return;
            }
        };
        if removed == 0 {
            return;
        }

        if *self.current_id.borrow() == Some(conversation_id) {
            self.load_conversation(conversation_id);
        }
        self.reload_sidebar();
        self.start_generation(GenerationRequest {
            conversation_id,
            memory_context: None,
        });

        if removed > 1 {
            self.toast("Regenerating; later turns were removed.");
        }
    }

    fn speak(self: &Rc<Self>, text: &str, button: Option<&gtk::Button>) {
        self.cancel_tts_idle_unload();
        let Some(choice) = self.resolve_tts() else {
            self.toast("Speech is not available for this TTS setting and language.");
            return;
        };

        let (sentences, truncated) = sentences_for_speech(text, 4096);
        if sentences.is_empty() {
            return;
        }
        if truncated {
            self.toast("Speech is limited to the first 4,096 characters.");
        }

        let language = tts_language(&self.state.settings.borrow().tts_language);
        let voice = self.resolved_voice(choice, language);
        tracing::info!(
            "speech: queueing {} sentence(s) via {} with voice '{}'",
            sentences.len(),
            match choice {
                TtsChoice::Paradee => "Paradee",
                TtsChoice::Builtin => "Ollama",
            },
            voice
        );

        let Some(sender) = self.speech_sender() else {
            return;
        };
        let generation = self.speech_generation.fetch_add(1, Ordering::SeqCst) + 1;

        if let Some(button) = button {
            button.set_sensitive(false);
        }
        if choice == TtsChoice::Paradee && !crate::paradee::is_downloaded() {
            self.toast("Downloading Paradee voice model (8.5 MB)…");
        }
        let (ui_tx, ui_rx) = async_channel::bounded(8);
        let synthesis = SpeechSynthesis {
            choice,
            voice,
            generation,
            generation_counter: Arc::clone(&self.speech_generation),
            sender,
            ui_tx,
            client: self.state.client.borrow().clone(),
            model: self.selected_model(),
            paradee: Arc::clone(&self.state.paradee),
        };

        self.state.runtime.spawn(async move {
            let (sentence_tx, sentence_rx) = async_channel::unbounded();
            for sentence in sentences {
                let _ = sentence_tx.send(sentence).await;
            }
            drop(sentence_tx);
            synthesis.run(sentence_rx).await;
        });

        self.show_speech_events(ui_rx, button.cloned());
        if button.is_some() {
            self.toast("Generating speech…");
        }
    }

    fn show_speech_events(
        self: &Rc<Self>,
        rx: async_channel::Receiver<SpeechEvent>,
        button: Option<gtk::Button>,
    ) {
        let announce = button.is_some();
        let win = self.clone();
        glib::spawn_future_local(async move {
            while let Ok(event) = rx.recv().await {
                match event {
                    SpeechEvent::Playing => {
                        win.speech_active.set(true);
                        win.set_stt_keep_alive(true);
                        win.refresh_wake();
                        if announce {
                            win.toast("Speaking…");
                        }
                    }
                    SpeechEvent::AudioUnavailable => {
                        win.toast("Audio output is unavailable");
                        *win.speech_player.borrow_mut() = None;
                        win.speech_active.set(false);
                        win.set_stt_keep_alive(false);
                        win.refresh_wake();
                    }
                    SpeechEvent::Error(message) => {
                        win.toast(&format!("Speech failed: {message}"));
                        win.speech_active.set(false);
                        win.set_stt_keep_alive(false);
                        win.refresh_wake();
                    }
                    SpeechEvent::Finished => {
                        if let Some(button) = button.as_ref() {
                            button.set_sensitive(true);
                        }
                        win.speech_active.set(false);
                        win.set_stt_keep_alive(false);
                        win.refresh_wake();
                        win.schedule_tts_idle_unload();
                        break;
                    }
                }
            }
        });
    }

    fn resolve_tts(&self) -> Option<TtsChoice> {
        let (enabled, backend, language_code) = {
            let settings = self.state.settings.borrow();
            (
                settings.tts_enabled,
                settings.tts_backend.clone(),
                settings.tts_language.clone(),
            )
        };
        if !enabled {
            return None;
        }
        let language = tts_language(&language_code);

        match backend.as_str() {
            "paradee" => has_paradee_voice(language.code).then_some(TtsChoice::Paradee),
            "builtin" => self
                .selected_model_supports("speech")
                .then_some(TtsChoice::Builtin),
            _ => {
                if has_paradee_voice(language.code) {
                    Some(TtsChoice::Paradee)
                } else if self.selected_model_supports("speech") {
                    Some(TtsChoice::Builtin)
                } else {
                    None
                }
            }
        }
    }

    fn resolved_voice(&self, choice: TtsChoice, language: TtsLanguage) -> String {
        let stored = self.state.settings.borrow().tts_voice.clone();
        let options = voice_options_for_choice(choice, language);
        options
            .iter()
            .find(|voice| voice.id == stored)
            .map(|voice| voice.id.clone())
            .unwrap_or_else(|| {
                options
                    .first()
                    .map(|voice| voice.id.clone())
                    .unwrap_or_default()
            })
    }

    fn speech_sender(self: &Rc<Self>) -> Option<async_channel::Sender<PlaybackChunk>> {
        let existing = self.speech_player.borrow().as_ref().cloned();
        if let Some(sender) = existing {
            return Some(sender);
        }

        match spawn_playback_thread(Arc::clone(&self.speech_generation)) {
            Ok(sender) => {
                *self.speech_player.borrow_mut() = Some(sender.clone());
                Some(sender)
            }
            Err(err) => {
                self.toast(&format!("Audio output unavailable: {err:#}"));
                None
            }
        }
    }

    fn begin_streaming_speech(self: &Rc<Self>) -> bool {
        self.cancel_tts_idle_unload();
        self.end_streaming_speech();
        self.speech_text.borrow_mut().clear();

        let Some(choice) = self.resolve_tts() else {
            return false;
        };
        if choice == TtsChoice::Paradee && !crate::paradee::is_downloaded() {
            self.toast("Downloading Paradee voice model (8.5 MB)…");
        }

        let language = tts_language(&self.state.settings.borrow().tts_language);
        let voice = self.resolved_voice(choice, language);
        let Some(sender) = self.speech_sender() else {
            return false;
        };
        let generation = self.speech_generation.fetch_add(1, Ordering::SeqCst) + 1;
        let (ui_tx, ui_rx) = async_channel::bounded(8);
        let (sentence_tx, sentence_rx) = async_channel::unbounded();
        *self.speech_stream_tx.borrow_mut() = Some(sentence_tx);

        let synthesis = SpeechSynthesis {
            choice,
            voice,
            generation,
            generation_counter: Arc::clone(&self.speech_generation),
            sender,
            ui_tx,
            client: self.state.client.borrow().clone(),
            model: self.selected_model(),
            paradee: Arc::clone(&self.state.paradee),
        };
        self.state.runtime.spawn(synthesis.run(sentence_rx));
        self.show_speech_events(ui_rx, None);
        true
    }

    fn end_streaming_speech(&self) {
        self.speech_stream_tx.borrow_mut().take();
    }

    fn queue_streaming_sentence(&self, sentence: &str) {
        let sentence = sentence.trim();
        if sentence.is_empty() {
            return;
        }
        let sender = self.speech_stream_tx.borrow().as_ref().cloned();
        let Some(sender) = sender else {
            return;
        };
        if sender.try_send(sentence.to_string()).is_err() {
            *self.speech_stream_tx.borrow_mut() = None;
        }
    }

    /// Feed newly streamed assistant text into the speech pipeline. Complete
    /// sentences are queued for synthesis as soon as they arrive instead of
    /// waiting for the whole reply to finish.
    fn feed_streaming_speech(&self, delta: &str) {
        if self.speech_stream_tx.borrow().is_none() {
            return;
        }

        let sentences = self.speech_text.borrow_mut().push(delta);
        for sentence in sentences {
            self.queue_streaming_sentence(&sentence);
        }
    }

    /// Queue whatever text remains after the reply finished, including a final
    /// sentence without punctuation, then close the streaming speech session.
    fn flush_streaming_speech(&self) {
        let sentences = self.speech_text.borrow_mut().finish();
        for sentence in sentences {
            self.queue_streaming_sentence(&sentence);
        }
    }

    fn cancel_tts_idle_unload(&self) {
        self.tts_idle_token.fetch_add(1, Ordering::SeqCst);
    }

    /// Release the local text-to-speech model after a minute without speech.
    fn schedule_tts_idle_unload(self: &Rc<Self>) {
        if !self.state.settings.borrow().tts_enabled {
            return;
        }

        let token = self.tts_idle_token.fetch_add(1, Ordering::SeqCst) + 1;
        let win = self.clone();
        glib::timeout_add_seconds_local_once(SPEECH_IDLE_UNLOAD_SECS, move || {
            // A newer request or an active streaming session keeps the model loaded.
            if win.tts_idle_token.load(Ordering::SeqCst) != token
                || win.speech_stream_tx.borrow().is_some()
            {
                return;
            }
            win.unload_tts_engine(Some(token));
        });
    }

    fn stop_speech(&self) {
        self.end_streaming_speech();
        self.speech_text.borrow_mut().clear();
        self.speech_generation.fetch_add(1, Ordering::SeqCst);
        self.speech_active.set(false);
        self.set_stt_keep_alive(false);
        let _ = self.wake_tx.try_send(());
    }

    /// Drop the local TTS engine. When a timer token is supplied, the unload is
    /// skipped if speech started again while the blocking task was queued.
    fn unload_tts_engine(&self, token: Option<u64>) {
        let engine = Arc::clone(&self.state.paradee);
        let idle_token = Arc::clone(&self.tts_idle_token);
        drop(self.state.runtime.spawn_blocking(move || {
            let mut released = false;
            if let Ok(mut guard) = engine.lock() {
                let idle = token.is_none_or(|token| idle_token.load(Ordering::SeqCst) == token);
                if idle {
                    released = guard.take().is_some();
                }
            }
            if released {
                tracing::info!("TTS: released local engine");
            }
            trim_heap();
        }));
    }

    fn cancel_stt_idle_unload(&self) {
        self.stt_idle_token.fetch_add(1, Ordering::SeqCst);
    }

    /// Release Whistle/Parakeet after a minute without dictation or live turns.
    fn schedule_stt_idle_unload(self: &Rc<Self>) {
        if !self.state.settings.borrow().stt_enabled {
            return;
        }

        let token = self.stt_idle_token.fetch_add(1, Ordering::SeqCst) + 1;
        let win = self.clone();
        glib::timeout_add_seconds_local_once(SPEECH_IDLE_UNLOAD_SECS, move || {
            if win.stt_idle_token.load(Ordering::SeqCst) != token
                || win.stt_listening.get()
                || win.stt_preparing.get()
                || win.live_mode.get()
                || win.live_pending.get()
            {
                return;
            }
            win.unload_stt_engine(Some(token));
        });
    }

    /// Drop the local STT engines. When a timer token is supplied, the unload
    /// is skipped if listening started again while the task was queued.
    fn unload_stt_engine(&self, token: Option<u64>) {
        let whistle = Arc::clone(&self.state.whistle);
        let parakeet = Arc::clone(&self.state.parakeet);
        let idle_token = Arc::clone(&self.stt_idle_token);
        drop(self.state.runtime.spawn_blocking(move || {
            let idle = |expected: Option<u64>| {
                expected.is_none_or(|expected| idle_token.load(Ordering::SeqCst) == expected)
            };
            let mut released = false;
            if let Ok(mut guard) = whistle.lock() {
                if idle(token) {
                    released |= guard.take().is_some();
                }
            }
            if let Ok(mut guard) = parakeet.lock() {
                if idle(token) {
                    released |= guard.take().is_some();
                }
            }
            if released {
                tracing::info!("STT: released local engine");
            }
            trim_heap();
        }));
    }

    // ----- Speech to text --------------------------------------------------

    fn update_stt_ui(&self) {
        let (stt_enabled, tts_enabled) = {
            let settings = self.state.settings.borrow();
            (settings.stt_enabled, settings.tts_enabled)
        };
        let live_active = self.live_mode.get() || self.live_pending.get();

        self.mic_button.set_visible(stt_enabled);
        self.live_button.set_visible(stt_enabled && tts_enabled);

        if !stt_enabled {
            self.mic_button.set_sensitive(true);
            self.mic_button
                .set_icon_name("audio-input-microphone-symbolic");
            self.mic_button.remove_css_class("destructive-action");
            self.mic_button.set_tooltip_text(Some("Dictate"));
        } else if live_active {
            self.mic_button.set_sensitive(false);
            self.mic_button
                .set_icon_name("audio-input-microphone-symbolic");
            self.mic_button.remove_css_class("destructive-action");
            self.mic_button
                .set_tooltip_text(Some("Unavailable during live conversation"));
        } else if self.stt_preparing.get() {
            self.mic_button.set_sensitive(true);
            self.mic_button
                .set_icon_name("media-playback-stop-symbolic");
            self.mic_button.add_css_class("destructive-action");
            self.mic_button
                .set_tooltip_text(Some("Cancel speech recognition preparation"));
        } else if self.stt_listening.get() {
            self.mic_button.set_sensitive(true);
            self.mic_button
                .set_icon_name("media-playback-stop-symbolic");
            self.mic_button.add_css_class("destructive-action");
            self.mic_button.set_tooltip_text(Some("Stop dictation"));
        } else {
            self.mic_button.set_sensitive(true);
            self.mic_button
                .set_icon_name("audio-input-microphone-symbolic");
            self.mic_button.remove_css_class("destructive-action");
            self.mic_button.set_tooltip_text(Some("Dictate"));
        }

        self.update_think_button();
        self.update_tray_state();
        // Check the button's own visibility flag: is_visible() is also false
        // while the window is hidden, which would skip updating the icon.
        if !self.live_button.get_visible() {
            return;
        }
        if live_active {
            self.live_button.set_sensitive(true);
            self.live_button
                .set_icon_name("media-playback-stop-symbolic");
            self.live_button.add_css_class("destructive-action");
            self.live_button
                .set_tooltip_text(Some(if self.stt_preparing.get() {
                    "Cancel live conversation preparation"
                } else {
                    "Stop live voice conversation"
                }));
        } else {
            self.live_button
                .set_sensitive(!self.stt_listening.get() && !self.stt_preparing.get());
            self.live_button
                .set_icon_name("media-playback-start-symbolic");
            self.live_button.remove_css_class("destructive-action");
            self.live_button
                .set_tooltip_text(Some("Start live voice conversation"));
        }

        // Wake-word availability depends on dictation, TTS and the model list,
        // so re-evaluate it whenever the speech UI changes.
        let _ = self.wake_tx.try_send(());
    }

    fn toggle_stt(self: &Rc<Self>) {
        if self.stt_listening.get() || self.stt_preparing.get() {
            self.stop_stt();
        } else {
            self.start_stt(false);
        }
    }

    fn toggle_live(self: &Rc<Self>) {
        if self.live_mode.get() || self.live_pending.get() {
            self.stop_stt();
            self.stop_speech();
            self.update_stt_ui();
        } else {
            self.start_stt(true);
        }
    }

    fn start_stt(self: &Rc<Self>, live: bool) {
        self.start_stt_inner(live);
        // If the session could not start (missing TTS voice, disabled
        // dictation, …), hand the microphone back to the wake detector.
        if !self.stt_listening.get() && !self.stt_preparing.get() {
            self.wake_pending.set(false);
            self.live_model_warmed.set(false);
            self.voice_new_chat_requested.set(false);
            self.refresh_wake();
        }
    }

    fn start_stt_inner(self: &Rc<Self>, live: bool) {
        if self.stt_listening.get() || self.stt_preparing.get() {
            return;
        }
        // The wake detector owns the microphone; release it before dictation.
        self.stop_wake();
        self.cancel_stt_idle_unload();
        if live && self.resolve_tts().is_none() {
            self.toast("Live conversation needs an available text-to-speech voice.");
            return;
        }

        let wake_started = live && self.wake_pending.get();
        let (enabled, provider, language, start_secs, end_ms, wake_timeout) = {
            let settings = self.state.settings.borrow();
            (
                settings.stt_enabled,
                settings.stt_provider.clone(),
                settings.stt_language.clone(),
                settings.stt_start_timeout_secs,
                settings.stt_end_silence_ms,
                settings.wake_listen_timeout_secs,
            )
        };
        if !enabled {
            return;
        }

        let audio_model = self.selected_model_supports("audio");
        let language = if language == "auto" {
            None
        } else {
            Some(language)
        };
        let whistle_language = language
            .as_deref()
            .is_none_or(|code| crate::whistle::WHISTLE_LANGUAGES.contains(&code));

        let parakeet_model = self.parakeet_model();
        let parakeet_downloaded = crate::parakeet::is_downloaded(parakeet_model);
        let choice = match provider.as_str() {
            "parakeet" => SttChoice::Parakeet,
            "whistle" => SttChoice::Whistle,
            "builtin" => {
                if !audio_model {
                    self.toast("The selected model does not support speech input.");
                    return;
                }
                SttChoice::Builtin
            }
            _ => {
                if parakeet_downloaded {
                    SttChoice::Parakeet
                } else if crate::whistle::is_downloaded() && whistle_language {
                    SttChoice::Whistle
                } else if audio_model {
                    SttChoice::Builtin
                } else if whistle_language {
                    SttChoice::Whistle
                } else {
                    self.toast("No speech-to-text provider supports that language.");
                    return;
                }
            }
        };
        let use_whistle = choice == SttChoice::Whistle;

        let start_timeout = std::time::Duration::from_secs(start_secs.max(1) as u64);
        let end_silence = std::time::Duration::from_millis(end_ms.max(200) as u64);
        // Wake-triggered sessions stop themselves after this much silence so
        // the detector can take the microphone back.
        let idle_timeout =
            wake_started.then(|| std::time::Duration::from_secs(wake_timeout.max(5) as u64));
        tracing::info!(
            "STT: using {} for language {:?}{}",
            match choice {
                SttChoice::Parakeet => parakeet_model.display_name(),
                SttChoice::Whistle => "Whistle",
                SttChoice::Builtin => "Ollama",
            },
            language,
            if live { " (live conversation)" } else { "" }
        );

        if choice == SttChoice::Parakeet {
            let engine = Arc::clone(&self.state.parakeet);
            let loaded = engine
                .lock()
                .map(|guard| {
                    guard
                        .as_ref()
                        .is_some_and(|(loaded, _)| *loaded == parakeet_model)
                })
                .unwrap_or(false);
            if loaded && parakeet_downloaded {
                self.begin_listening(
                    PreparedBackend::Parakeet(engine),
                    None,
                    start_timeout,
                    end_silence,
                    live,
                    idle_timeout,
                );
                return;
            }

            self.stt_preparing.set(true);
            if live {
                self.live_pending.set(true);
            }
            let prep_id = self.stt_generation.get().saturating_add(1);
            self.stt_generation.set(prep_id);
            self.update_stt_ui();
            if parakeet_downloaded {
                self.toast(&format!(
                    "Preparing {} speech recognition…",
                    parakeet_model.display_name()
                ));
            } else {
                self.toast(&format!(
                    "Downloading {} model ({})…",
                    parakeet_model.display_name(),
                    parakeet_model.approx_size()
                ));
            }

            let (tx, rx) = async_channel::bounded(1);
            let prepare_engine = Arc::clone(&engine);
            self.state.runtime.spawn(async move {
                let result = stt::prepare_parakeet(prepare_engine, parakeet_model).await;
                let _ = tx.send(result).await;
            });

            let win = self.clone();
            glib::spawn_future_local(async move {
                match rx.recv().await {
                    Ok(Ok(())) => {
                        if win.stt_generation.get() != prep_id || !win.stt_preparing.get() {
                            return;
                        }
                        win.stt_preparing.set(false);
                        if !win.state.settings.borrow().stt_enabled {
                            win.live_pending.set(false);
                            win.update_stt_ui();
                            return;
                        }
                        if win.parakeet_model() != parakeet_model {
                            win.unload_stt_engine(None);
                            win.update_stt_ui();
                            return;
                        }
                        win.begin_listening(
                            PreparedBackend::Parakeet(engine),
                            None,
                            start_timeout,
                            end_silence,
                            live,
                            idle_timeout,
                        );
                    }
                    Ok(Err(error)) => {
                        if win.stt_generation.get() != prep_id {
                            return;
                        }
                        win.stt_preparing.set(false);
                        win.live_pending.set(false);
                        win.update_stt_ui();
                        win.schedule_stt_idle_unload();
                        win.toast(&format!(
                            "Could not prepare {}: {error:#}",
                            parakeet_model.display_name()
                        ));
                    }
                    Err(_) => {
                        if win.stt_generation.get() == prep_id {
                            win.stt_preparing.set(false);
                            win.live_pending.set(false);
                            win.update_stt_ui();
                            win.schedule_stt_idle_unload();
                        }
                    }
                }
            });
            return;
        }

        if use_whistle {
            if let Some(code) = language.as_deref() {
                if !crate::whistle::WHISTLE_LANGUAGES.contains(&code) {
                    self.toast("Whistle does not support that language.");
                    return;
                }
            }

            let engine = Arc::clone(&self.state.whistle);
            let loaded = engine.lock().map(|guard| guard.is_some()).unwrap_or(false);
            if loaded && crate::whistle::is_downloaded() {
                self.begin_listening(
                    PreparedBackend::Whistle(engine),
                    language,
                    start_timeout,
                    end_silence,
                    live,
                    idle_timeout,
                );
                return;
            }

            self.stt_preparing.set(true);
            if live {
                self.live_pending.set(true);
            }
            let prep_id = self.stt_generation.get().saturating_add(1);
            self.stt_generation.set(prep_id);
            self.update_stt_ui();
            if crate::whistle::is_downloaded() {
                self.toast("Preparing Whistle speech recognition…");
            } else {
                self.toast("Downloading Whistle model (~160 MB)…");
            }

            let (tx, rx) = async_channel::bounded(1);
            let prepare_engine = Arc::clone(&engine);
            self.state.runtime.spawn(async move {
                let result = stt::prepare_whistle(prepare_engine).await;
                let _ = tx.send(result).await;
            });

            let win = self.clone();
            glib::spawn_future_local(async move {
                match rx.recv().await {
                    Ok(Ok(())) => {
                        if win.stt_generation.get() != prep_id || !win.stt_preparing.get() {
                            return;
                        }
                        win.stt_preparing.set(false);
                        if !win.state.settings.borrow().stt_enabled {
                            win.live_pending.set(false);
                            win.update_stt_ui();
                            return;
                        }
                        win.begin_listening(
                            PreparedBackend::Whistle(engine),
                            language,
                            start_timeout,
                            end_silence,
                            live,
                            idle_timeout,
                        );
                    }
                    Ok(Err(error)) => {
                        if win.stt_generation.get() != prep_id {
                            return;
                        }
                        win.stt_preparing.set(false);
                        win.live_pending.set(false);
                        win.update_stt_ui();
                        win.schedule_stt_idle_unload();
                        win.toast(&format!("Could not prepare Whistle: {error:#}"));
                    }
                    Err(_) => {
                        if win.stt_generation.get() == prep_id {
                            win.stt_preparing.set(false);
                            win.live_pending.set(false);
                            win.update_stt_ui();
                            win.schedule_stt_idle_unload();
                        }
                    }
                }
            });
            return;
        }

        let model = self.selected_model();
        if model.is_empty() {
            self.toast("No model selected for speech recognition.");
            return;
        }
        let backend = PreparedBackend::Builtin {
            client: self.state.client.borrow().clone(),
            model,
        };
        self.begin_listening(
            backend,
            language,
            start_timeout,
            end_silence,
            live,
            idle_timeout,
        );
    }

    fn begin_listening(
        self: &Rc<Self>,
        backend: PreparedBackend,
        language: Option<String>,
        start_timeout: std::time::Duration,
        end_silence: std::time::Duration,
        live: bool,
        idle_timeout: Option<std::time::Duration>,
    ) {
        if !self.state.settings.borrow().stt_enabled {
            return;
        }

        let wake_started = live && idle_timeout.is_some();

        // Internal stop: refresh_wake() reopens the microphone, and we are
        // about to take it for dictation.
        self.stop_stt_session();
        self.cancel_stt_idle_unload();
        if live {
            self.input.buffer().set_text("");
            self.stt_prefix.borrow_mut().clear();
        } else {
            let buffer = self.input.buffer();
            *self.stt_prefix.borrow_mut() = buffer
                .text(&buffer.start_iter(), &buffer.end_iter(), true)
                .to_string();
        }

        let keep_alive = idle_timeout.map(|_| Arc::new(std::sync::atomic::AtomicBool::new(false)));
        *self.stt_keep_alive.borrow_mut() = keep_alive.clone();

        match stt::start_listening(
            &self.state.runtime,
            backend,
            stt::ListenConfig {
                language,
                start_timeout,
                end_silence,
                continuous: live,
                idle_timeout,
                keep_alive,
            },
        ) {
            Ok((session, events)) => {
                *self.stt_session.borrow_mut() = Some(session);
                self.live_mode.set(live);
                self.live_pending.set(false);
                self.wake_pending.set(false);
                self.stt_listening.set(true);
                self.stt_preparing.set(false);
                if live {
                    self.live_model_warmed.set(false);
                    self.voice_new_chat_requested
                        .set(should_start_new_voice_chat(
                            wake_started,
                            self.state.settings.borrow().wake_new_conversation,
                            self.last_voice_end_goodbye.get(),
                        ));
                } else {
                    self.voice_new_chat_requested.set(false);
                }
                self.update_stt_ui();
                if live {
                    self.toast("Live voice conversation started");
                }

                let id = self.stt_generation.get();
                let win = self.clone();
                glib::spawn_future_local(async move {
                    let mut finished = false;
                    while let Ok(event) = events.recv().await {
                        if win.stt_generation.get() != id {
                            return;
                        }
                        tracing::debug!("STT event: {event:?}");

                        if live {
                            match event {
                                SttEvent::SpeechStarted => {
                                    if win.active.borrow().is_some() {
                                        win.stop_streaming();
                                    }
                                    win.stop_speech();
                                    win.warm_live_model();
                                }
                                SttEvent::Partial(text) => win.apply_stt_text(&text),
                                SttEvent::Final(text) => {
                                    if text.trim().is_empty() {
                                        continue;
                                    }
                                    if is_sleep_command(&text) {
                                        win.stop_streaming();
                                        win.stop_speech();
                                        win.input.buffer().set_text("");
                                        win.stop_stt_goodbye();
                                        feedback::play(Cue::Sleep);
                                        win.toast("Live conversation ended");
                                        finished = true;
                                        break;
                                    }
                                    win.apply_stt_text(&text);
                                    if win.active.borrow().is_some() {
                                        win.stop_streaming();
                                    }
                                    win.send_message();
                                }
                                SttEvent::Timeout => {
                                    feedback::play(Cue::Timeout);
                                    win.toast("Live conversation ended");
                                    win.stop_stt();
                                    finished = true;
                                    break;
                                }
                                SttEvent::Error(error) => {
                                    win.toast(&format!("Speech recognition failed: {error}"));
                                    win.stop_stt();
                                    finished = true;
                                    break;
                                }
                                SttEvent::Stopped => {
                                    win.stop_stt();
                                    finished = true;
                                    break;
                                }
                            }
                            continue;
                        }

                        match event {
                            SttEvent::SpeechStarted => {
                                win.mic_button
                                    .set_tooltip_text(Some("Listening… click to stop"));
                            }
                            SttEvent::Partial(text) => win.apply_stt_text(&text),
                            SttEvent::Final(text) => {
                                win.apply_stt_text(&text);
                                let autosend = win.state.settings.borrow().stt_autosend;
                                let send = autosend && !text.trim().is_empty();
                                win.stop_stt();
                                if send {
                                    win.send_message();
                                }
                                finished = true;
                                break;
                            }
                            SttEvent::Timeout => {
                                win.toast("No speech detected");
                                win.stop_stt();
                                finished = true;
                                break;
                            }
                            SttEvent::Error(error) => {
                                win.toast(&format!("Speech recognition failed: {error}"));
                                win.stop_stt();
                                finished = true;
                                break;
                            }
                            SttEvent::Stopped => {
                                win.stop_stt();
                                finished = true;
                                break;
                            }
                        }
                    }
                    if !finished && win.stt_generation.get() == id {
                        win.stop_stt();
                    }
                });
            }
            Err(error) => {
                self.stt_listening.set(false);
                self.live_mode.set(false);
                self.live_pending.set(false);
                self.live_model_warmed.set(false);
                self.voice_new_chat_requested.set(false);
                self.update_stt_ui();
                self.toast(&format!("Could not start the microphone: {error:#}"));
            }
        }
    }

    fn apply_stt_text(&self, text: &str) {
        let text = text.trim();
        if text.is_empty() {
            return;
        }
        let prefix = self.stt_prefix.borrow().clone();
        let combined = if prefix.trim().is_empty() {
            text.to_string()
        } else if prefix.ends_with(char::is_whitespace) {
            format!("{prefix}{text}")
        } else {
            format!("{prefix} {text}")
        };

        let buffer = self.input.buffer();
        buffer.set_text(&combined);
        let end = buffer.end_iter();
        buffer.place_cursor(&end);
    }

    fn stop_stt(self: &Rc<Self>) {
        self.last_voice_end_goodbye.set(false);
        self.voice_new_chat_requested.set(false);
        self.stop_stt_session();
        self.wake_pending.set(false);
        self.refresh_wake();
    }

    fn stop_stt_goodbye(self: &Rc<Self>) {
        self.last_voice_end_goodbye.set(true);
        self.voice_new_chat_requested.set(false);
        self.stop_stt_session();
        self.wake_pending.set(false);
        self.refresh_wake();
    }

    fn stop_stt_session(self: &Rc<Self>) {
        self.stt_generation
            .set(self.stt_generation.get().saturating_add(1));
        if let Some(session) = self.stt_session.borrow_mut().take() {
            session.stop();
        }
        self.stt_listening.set(false);
        self.stt_preparing.set(false);
        self.live_mode.set(false);
        self.live_pending.set(false);
        self.live_model_warmed.set(false);
        *self.stt_keep_alive.borrow_mut() = None;
        self.update_stt_ui();
        self.schedule_stt_idle_unload();
    }

    /// Load the chat model in parallel with the first transcription. VAD has
    /// just detected speech, so this avoids warming the model for a wake that
    /// never turns into a conversation.
    fn warm_live_model(self: &Rc<Self>) {
        if self.live_model_warmed.get() {
            return;
        }
        let model = self.selected_model();
        if model.is_empty() || self.selected_model_is_remote() {
            return;
        }
        self.live_model_warmed.set(true);
        let num_ctx = self.state.settings.borrow().num_ctx.max(1);
        let client = self.state.client.borrow().clone();
        let generation = self.stt_generation.get();
        let (tx, rx) = async_channel::bounded(1);
        let log_model = model.clone();
        self.state.runtime.spawn(async move {
            let result = client.preload(&model, num_ctx, CHAT_KEEP_ALIVE).await;
            let _ = tx.send(result).await;
        });

        let win = self.clone();
        glib::spawn_future_local(async move {
            let Ok(result) = rx.recv().await else {
                return;
            };
            match result {
                Ok(()) => tracing::debug!("model: preloaded {log_model}"),
                Err(err) => {
                    tracing::debug!("model: could not preload {log_model}: {err:#}");
                    if win.stt_generation.get() == generation {
                        win.live_model_warmed.set(false);
                    }
                }
            }
        });
    }

    /// Tell the idle timer that a reply or its speech is still in progress.
    fn set_stt_keep_alive(&self, busy: bool) {
        if let Some(flag) = self.stt_keep_alive.borrow().as_ref() {
            flag.store(busy, Ordering::Relaxed);
        }
    }

    fn near_bottom(&self) -> bool {
        let adjustment = self.messages_scroll.vadjustment();
        adjustment.value() + adjustment.page_size() >= adjustment.upper() - 80.0
    }

    fn scroll_to_bottom(&self) {
        let adjustment = self.messages_scroll.vadjustment();
        glib::idle_add_local_once(move || {
            adjustment.set_value(adjustment.upper() - adjustment.page_size());
        });
    }

    // ----- Image attachments ------------------------------------------------

    fn update_attachment_ui(self: &Rc<Self>) {
        let supported = self.selected_model_supports("vision");
        self.attach_button.set_visible(supported);
        if !supported && !self.attachments.borrow().is_empty() {
            self.attachments.borrow_mut().clear();
            self.refresh_attachment_previews();
            self.toast("The selected model does not support images.");
        }
    }

    fn choose_images(self: &Rc<Self>) {
        let dialog = gtk::FileDialog::builder()
            .title("Attach images")
            .accept_label("Attach")
            .modal(true)
            .build();

        let filter = gtk::FileFilter::new();
        filter.set_name(Some("Images"));
        filter.add_mime_type("image/*");
        for pattern in ["*.png", "*.jpg", "*.jpeg", "*.webp", "*.gif", "*.bmp"] {
            filter.add_pattern(pattern);
        }
        let filters = gio::ListStore::new::<gtk::FileFilter>();
        filters.append(&filter);
        dialog.set_filters(Some(&filters));
        dialog.set_default_filter(Some(&filter));

        let win = self.clone();
        dialog.open_multiple(
            Some(&self.window),
            None::<&gio::Cancellable>,
            move |result| {
                let Ok(files) = result else {
                    tracing::warn!("image: file dialog failed");
                    return;
                };
                tracing::debug!("image: file dialog returned {} file(s)", files.n_items());
                for index in 0..files.n_items() {
                    if let Some(file) = files.item(index).and_downcast::<gio::File>() {
                        win.load_attachment(file);
                    }
                }
            },
        );
    }

    fn load_attachment(self: &Rc<Self>, file: gio::File) {
        let name = file
            .basename()
            .map(|path| path.to_string_lossy().to_string())
            .unwrap_or_else(|| "image".to_string());
        tracing::debug!("image: loading {name}");
        match file.load_contents(None::<&gio::Cancellable>) {
            Ok((contents, _)) => self.add_attachment_bytes(&name, contents.to_vec()),
            Err(err) => self.toast(&format!("Could not read {name}: {err}")),
        }
    }

    fn add_attachment_bytes(self: &Rc<Self>, name: &str, data: Vec<u8>) {
        if self.attachments.borrow().len() >= image::MAX_ATTACHMENTS {
            self.toast(&format!(
                "You can attach at most {} images.",
                image::MAX_ATTACHMENTS
            ));
            return;
        }

        tracing::debug!("image: processing {name} ({} bytes)", data.len());
        if data.len() > image::MAX_FILE_BYTES {
            self.toast(&format!(
                "{name} is too large. The limit is {} MB.",
                image::MAX_FILE_BYTES / (1024 * 1024)
            ));
            return;
        }

        let processed = match image::process(&data) {
            Ok(processed) => processed,
            Err(err) => {
                self.toast(&format!("Could not attach {name}: {err:#}"));
                return;
            }
        };

        tracing::debug!(
            "image: attached {name} as {}x{} {} ({} bytes)",
            processed.width,
            processed.height,
            processed.mime,
            processed.data.len()
        );
        let id = self.next_attachment_id.get() + 1;
        self.next_attachment_id.set(id);
        self.attachments.borrow_mut().push(PendingAttachment {
            id,
            name: name.to_string(),
            image: StoredImage {
                mime: processed.mime,
                width: processed.width,
                height: processed.height,
                data: processed.data,
            },
            preview_data: processed.preview_data,
        });
        self.refresh_attachment_previews();
    }

    fn refresh_attachment_previews(self: &Rc<Self>) {
        while let Some(child) = self.attachments_box.first_child() {
            self.attachments_box.remove(&child);
        }

        let attachments = self.attachments.borrow();
        self.attachments_box.set_visible(!attachments.is_empty());
        for attachment in attachments.iter() {
            let overlay = gtk::Overlay::new();
            overlay.add_css_class("attachment-thumb");
            overlay.set_tooltip_text(Some(&attachment.name));

            overlay.set_child(Some(&image::preview_widget(&attachment.preview_data)));

            let remove = gtk::Button::from_icon_name("window-close-symbolic");
            remove.add_css_class("flat");
            remove.add_css_class("circular");
            remove.add_css_class("attachment-remove");
            remove.set_halign(gtk::Align::End);
            remove.set_valign(gtk::Align::Start);
            remove.set_tooltip_text(Some("Remove image"));

            let win = self.clone();
            let id = attachment.id;
            remove.connect_clicked(move |_| win.remove_attachment(id));
            overlay.add_overlay(&remove);

            self.attachments_box.append(&overlay);
        }
    }

    fn remove_attachment(self: &Rc<Self>, id: u64) {
        self.attachments.borrow_mut().retain(|item| item.id != id);
        self.refresh_attachment_previews();
    }

    fn clear_attachments(self: &Rc<Self>) {
        self.attachments.borrow_mut().clear();
        self.refresh_attachment_previews();
    }

    fn send_message(self: &Rc<Self>) {
        if self.active.borrow().is_some() || self.memory_lookup.get() {
            return;
        }

        let buffer = self.input.buffer();
        let text = buffer
            .text(&buffer.start_iter(), &buffer.end_iter(), true)
            .trim()
            .to_string();
        if text.is_empty() && self.attachments.borrow().is_empty() {
            return;
        }

        if self.stt_listening.get() && !self.live_mode.get() {
            self.stop_stt();
        }

        let model = self.selected_model();
        if model.is_empty() {
            self.toast("No chat model selected. Pull one with `ollama pull`, then refresh.");
            return;
        }

        let images: Vec<StoredImage> = self
            .attachments
            .borrow()
            .iter()
            .map(|attachment| attachment.image.clone())
            .collect();
        if !images.is_empty() && !self.selected_model_supports("vision") {
            self.toast("The selected model does not support images.");
            return;
        }

        let from_live = self.live_mode.get();
        let start_new = from_live && self.voice_new_chat_requested.replace(false);
        if start_new {
            *self.current_id.borrow_mut() = None;
            self.chat_list.unselect_all();
            self.show_empty();
        }
        self.last_voice_end_goodbye.set(false);
        if !from_live {
            self.voice_new_chat_requested.set(false);
        }

        let existing_id = *self.current_id.borrow();
        let conversation_id = match existing_id {
            Some(id) => {
                let _ = self.state.store.borrow().set_conversation_model(id, &model);
                id
            }
            None => {
                let title = if text.is_empty() {
                    "Image".to_string()
                } else {
                    title_from(&text)
                };
                let id = match self
                    .state
                    .store
                    .borrow()
                    .create_conversation(&model, &title)
                {
                    Ok(id) => id,
                    Err(err) => {
                        self.toast(&format!("Could not create chat: {err:#}"));
                        return;
                    }
                };
                *self.current_id.borrow_mut() = Some(id);
                self.title_widget.set_title(&title);
                id
            }
        };

        let user_id = match self.state.store.borrow().add_message_with_images(
            conversation_id,
            "user",
            &text,
            None,
            &images,
        ) {
            Ok(id) => id,
            Err(err) => {
                self.toast(&format!("Could not save message: {err:#}"));
                return;
            }
        };

        self.add_user_message(conversation_id, Some(user_id), &text, &images);
        self.main_stack.set_visible_child_name("chat");

        buffer.set_text("");
        self.clear_attachments();
        self.reload_sidebar();
        self.start_generation_with_memory(conversation_id, &text);
    }

    /// Look up relevant long-term notes before starting a reply. Embedding is
    /// asynchronous, so the send button stays disabled until the request starts.
    fn start_generation_with_memory(self: &Rc<Self>, conversation_id: i64, text: &str) {
        // An image-only message has no useful text to embed for memory lookup.
        if text.trim().is_empty() {
            self.start_generation(GenerationRequest {
                conversation_id,
                memory_context: None,
            });
            return;
        }

        let model = self.selected_model();
        if model.is_empty() {
            self.start_generation(GenerationRequest {
                conversation_id,
                memory_context: None,
            });
            return;
        }
        let Some(pool) = self.memory_pool_for(&model) else {
            self.start_generation(GenerationRequest {
                conversation_id,
                memory_context: None,
            });
            return;
        };

        let embedding_model = self.embedding_model();
        if !self.embedding_model_ready(&embedding_model) {
            self.memory_warning(
                "Long-term memory is paused: install an embedding model such as all-minilm.",
            );
            self.start_generation(GenerationRequest {
                conversation_id,
                memory_context: None,
            });
            return;
        }

        let entries = match self
            .state
            .store
            .borrow()
            .memory_entries(pool, &embedding_model)
        {
            Ok(entries) => entries,
            Err(err) => {
                tracing::warn!("memory: could not load notes: {err:#}");
                Vec::new()
            }
        };
        if entries.is_empty() {
            tracing::debug!("memory: no notes in the {} pool", pool.as_str());
            self.start_generation(GenerationRequest {
                conversation_id,
                memory_context: None,
            });
            return;
        }

        let (num_ctx, top_k) = {
            let settings = self.state.settings.borrow();
            (
                settings.num_ctx.max(1),
                settings.memory_retrieve_top_k.max(1) as usize,
            )
        };
        let budget = memory::memory_context_budget(num_ctx);
        let date_hint = memory::date_hint(text);
        let date_range = date_hint.clone().and_then(|hint| {
            self.state
                .store
                .borrow()
                .resolve_date_hint(hint)
                .map_err(|err| {
                    tracing::warn!("memory: could not resolve date hint: {err:#}");
                    err
                })
                .ok()
        });
        let mut query = text.to_string();
        if let Some((start, end)) = &date_range {
            if start == end {
                query.push_str(&format!("\nDate: {start}"));
            } else {
                query.push_str(&format!("\nDate range: {start} to {end}"));
            }
            tracing::debug!("memory: date hint {date_hint:?} resolved to {start}..{end}");
        }
        let client = self.state.client.borrow().clone();
        let model_for_check = model.clone();

        self.memory_lookup.set(true);
        self.send_button.set_sensitive(false);
        let (tx, rx) = async_channel::bounded(1);
        self.state.runtime.spawn(async move {
            let result = async {
                let mut vectors = client.embed(&embedding_model, &[query]).await?;
                let Some(query_vector) = vectors.pop() else {
                    anyhow::bail!("embedding response was empty");
                };
                let hits = memory::search_entries(
                    &entries,
                    &query_vector,
                    top_k,
                    memory::MEMORY_MIN_SCORE,
                    date_range
                        .as_ref()
                        .map(|(start, end)| (start.as_str(), end.as_str())),
                );
                tracing::debug!(
                    "memory: retrieved {} note(s) from the {} pool",
                    hits.len(),
                    pool.as_str()
                );
                Ok::<_, anyhow::Error>(memory::format_memory_context(&hits, budget))
            }
            .await;
            let _ = tx.send(result).await;
        });

        let win = self.clone();
        glib::spawn_future_local(async move {
            let result = rx.recv().await;
            win.memory_lookup.set(false);
            win.send_button.set_sensitive(win.active.borrow().is_none());
            if *win.current_id.borrow() != Some(conversation_id) {
                return;
            }
            let memory_context = match result {
                Ok(Ok(context)) => context,
                Ok(Err(err)) => {
                    win.memory_warning(&format!("Could not retrieve long-term memory: {err:#}"));
                    None
                }
                Err(_) => None,
            };
            // If the model changed while the query was embedding, the notes may
            // belong to the other pool. Drop them rather than crossing pools.
            let memory_context = if win.selected_model() == model_for_check
                && win.memory_pool_for(&model_for_check).is_some()
            {
                memory_context
            } else {
                None
            };
            win.start_generation(GenerationRequest {
                conversation_id,
                memory_context,
            });
        });
    }

    // ----- Context compression --------------------------------------------

    fn maybe_compress_context(self: &Rc<Self>, conversation_id: i64) {
        if self.compressing.get() {
            return;
        }
        if let Some(plan) = self.compression_plan(conversation_id, None) {
            self.start_compression(conversation_id, plan);
        }
    }

    fn compression_plan(
        &self,
        conversation_id: i64,
        memory_context: Option<&str>,
    ) -> Option<CompressionPlan> {
        let model = self.selected_model();
        if model.is_empty() {
            return None;
        }
        let (num_ctx, system_prompt) = {
            let settings = self.state.settings.borrow();
            (
                settings.num_ctx.max(1),
                settings.system_prompt.trim().to_string(),
            )
        };

        let (messages, summary, used) = {
            let store = self.state.store.borrow();
            (
                store.messages(conversation_id).ok()?,
                store.conversation_summary(conversation_id).ok().flatten(),
                store
                    .conversation_context_used(conversation_id)
                    .ok()
                    .flatten(),
            )
        };

        build_compression_plan(
            model,
            num_ctx,
            &system_prompt,
            memory_context,
            &messages,
            summary.as_ref(),
            used,
        )
    }

    fn start_compression(self: &Rc<Self>, conversation_id: i64, plan: CompressionPlan) {
        self.compressing.set(true);
        tracing::info!("context: summarizing earlier messages for conversation {conversation_id}");
        self.toast("Summarizing earlier messages…");

        let client = self.state.client.borrow().clone();
        let (tx, rx) = async_channel::bounded(1);
        self.state.runtime.spawn(async move {
            let result = run_compression(client, plan).await;
            let _ = tx.send(result).await;
        });

        let win = self.clone();
        glib::spawn_future_local(async move {
            let result = rx.recv().await;
            win.finish_compression(conversation_id, result);
        });
    }

    fn finish_compression(
        self: &Rc<Self>,
        conversation_id: i64,
        result: Result<anyhow::Result<CompressionResult>, async_channel::RecvError>,
    ) {
        self.compressing.set(false);
        let mut succeeded = false;

        match result {
            Ok(Ok(result)) => {
                let stored = StoredSummary {
                    text: result.text,
                    upto_id: result.upto_id,
                    tokens: result.tokens,
                };
                match self
                    .state
                    .store
                    .borrow()
                    .set_conversation_summary(conversation_id, &stored)
                {
                    Ok(()) => {
                        succeeded = true;
                        let memory_context = self
                            .pending_generation
                            .borrow()
                            .as_ref()
                            .and_then(|request| request.memory_context.clone());
                        let estimated = self
                            .estimated_request_tokens(conversation_id, memory_context.as_deref());
                        let _ = self
                            .state
                            .store
                            .borrow()
                            .set_conversation_context_used(conversation_id, Some(estimated));
                        tracing::info!(
                            "context: summarized older messages, about {estimated} tokens remain"
                        );
                    }
                    Err(err) => {
                        self.toast(&format!("Could not save the summary: {err:#}"));
                    }
                }
            }
            Ok(Err(err)) => {
                self.toast(&format!("Could not summarize earlier messages: {err:#}"));
            }
            Err(_) => {
                self.toast("Could not summarize earlier messages.");
            }
        }

        if *self.current_id.borrow() == Some(conversation_id) {
            self.refresh_context_indicator();
            self.refresh_summary_notice(conversation_id);
        }
        self.reload_sidebar();

        let pending = self.pending_generation.borrow_mut().take();
        if let Some(pending) = pending {
            if succeeded {
                self.start_generation(pending);
            } else {
                self.start_generation_now(pending);
            }
        }
    }

    fn estimated_request_tokens(&self, conversation_id: i64, memory_context: Option<&str>) -> u32 {
        let (system_prompt, messages, summary) = {
            let settings = self.state.settings.borrow();
            let store = self.state.store.borrow();
            (
                settings.system_prompt.trim().to_string(),
                store.messages(conversation_id).unwrap_or_default(),
                store.conversation_summary(conversation_id).ok().flatten(),
            )
        };
        let kept: Vec<&StoredMessage> = match &summary {
            Some(summary) => messages
                .iter()
                .filter(|message| message.id > summary.upto_id)
                .collect(),
            None => messages.iter().collect(),
        };
        estimate_request_tokens(&system_prompt, memory_context, summary.as_ref(), &kept)
    }

    fn start_generation(self: &Rc<Self>, request: GenerationRequest) {
        if self.active.borrow().is_some() {
            return;
        }
        if self.compressing.get() {
            *self.pending_generation.borrow_mut() = Some(request);
            return;
        }
        let conversation_id = request.conversation_id;
        if let Some(plan) =
            self.compression_plan(conversation_id, request.memory_context.as_deref())
        {
            *self.pending_generation.borrow_mut() = Some(request);
            self.start_compression(conversation_id, plan);
            return;
        }
        self.start_generation_now(request);
    }

    fn start_generation_now(self: &Rc<Self>, request: GenerationRequest) {
        if self.active.borrow().is_some() {
            return;
        }

        let conversation_id = request.conversation_id;
        let memory_context = request.memory_context;
        let model = self.selected_model();
        if model.is_empty() {
            self.toast("No chat model selected. Pull one with `ollama pull`, then refresh.");
            return;
        }
        let _ = self
            .state
            .store
            .borrow()
            .set_conversation_model(conversation_id, &model);

        let (history, images) = {
            let store = self.state.store.borrow();
            (
                store.messages(conversation_id),
                store
                    .images_for_conversation(conversation_id)
                    .unwrap_or_default(),
            )
        };
        let history = match history {
            Ok(messages) => messages,
            Err(err) => {
                self.toast(&format!("Could not load chat: {err:#}"));
                return;
            }
        };

        let (system_prompt, summary) = {
            let settings = self.state.settings.borrow();
            let store = self.state.store.borrow();
            (
                settings.system_prompt.trim().to_string(),
                store.conversation_summary(conversation_id).ok().flatten(),
            )
        };
        let (temperature, num_ctx) = {
            let settings = self.state.settings.borrow();
            (settings.temperature, settings.num_ctx.max(1))
        };
        let messages = build_request_messages(
            &system_prompt,
            memory_context.as_deref(),
            summary.as_ref(),
            &history,
            &images,
            num_ctx,
        );
        let think = self.think_setting();
        tracing::debug!("chat: thinking setting {think:?}");

        let request = ChatRequest {
            model: model.clone(),
            messages,
            stream: true,
            options: Some(ChatOptions {
                temperature: Some(temperature),
                num_ctx: Some(num_ctx),
                ..Default::default()
            }),
            keep_alive: Some(CHAT_KEEP_ALIVE.to_string()),
            think,
        };

        let (row, bubble, assistant_view) = self.add_assistant_bubble();
        let thinking = ThinkingUi::new(&bubble);
        let assistant_buffer = assistant_view.buffer();
        let content = Rc::new(RefCell::new(String::new()));
        let stats: Rc<RefCell<Option<StreamStats>>> = Rc::new(RefCell::new(None));
        let (tx, rx) = async_channel::unbounded();

        let client = self.state.client.borrow().clone();
        let abort = api::spawn_chat_stream(&self.state.runtime, client, request, tx);

        *self.active.borrow_mut() = Some(ActiveStream {
            abort,
            row,
            bubble,
            view: assistant_view,
            content: content.clone(),
            conv_id: conversation_id,
            model,
            stats: stats.clone(),
        });

        self.set_streaming_ui(true);
        self.scroll_to_bottom();

        let stream_speech = {
            let settings = self.state.settings.borrow();
            settings.tts_enabled && (settings.tts_autoread || self.live_mode.get())
        } && self.resolve_tts().is_some();
        if stream_speech {
            self.begin_streaming_speech();
        }

        let win = self.clone();
        glib::spawn_future_local(async move {
            let mut had_error = false;
            let mut final_stats = None;
            while let Ok(event) = rx.recv().await {
                match event {
                    StreamEvent::Thinking(delta) => {
                        let follow = win.near_bottom();
                        thinking.show();
                        thinking.append(&delta);
                        if follow {
                            win.scroll_to_bottom();
                        }
                    }
                    StreamEvent::Delta(delta) => {
                        let follow = win.near_bottom();
                        thinking.finish();
                        assistant_buffer.insert(&mut assistant_buffer.end_iter(), &delta);
                        content.borrow_mut().push_str(&delta);
                        win.feed_streaming_speech(&delta);
                        if follow {
                            win.scroll_to_bottom();
                        }
                    }
                    StreamEvent::Done(stream_stats) => {
                        final_stats = Some(stream_stats);
                    }
                    StreamEvent::Error(message) => {
                        had_error = true;
                        win.toast(&message);
                    }
                }
            }

            thinking.finish();
            if let Some(stream_stats) = final_stats {
                *stats.borrow_mut() = Some(stream_stats);
            }
            win.finish_stream(had_error);
        });
    }

    fn finish_stream(self: &Rc<Self>, had_error: bool) {
        let Some(active) = self.active.borrow_mut().take() else {
            return;
        };

        let content = active.content.borrow().clone();
        let stats = active.stats.borrow().clone();
        let mut assistant_id = None;

        if !content.trim().is_empty() {
            let conversation_exists = self
                .state
                .store
                .borrow()
                .conversation_model(active.conv_id)
                .ok()
                .flatten()
                .is_some();
            if conversation_exists {
                match self.state.store.borrow().add_message(
                    active.conv_id,
                    "assistant",
                    &content,
                    stats.as_ref(),
                ) {
                    Ok(id) => assistant_id = Some(id),
                    Err(err) => {
                        self.toast(&format!("Could not save reply: {err:#}"));
                    }
                }
            }
            if active.row.parent().is_some() {
                super::markdown::render(&active.view, &content);
            }
        } else if active.row.parent().is_some() {
            self.messages_box.remove(&active.row);
        }

        tracing::info!(
            "stream finished for conversation {} ({} chars)",
            active.conv_id,
            content.len()
        );

        if content.trim().is_empty() && had_error {
            self.toast("Generation failed. Check the Ollama server.");
        } else if content.trim().is_empty() {
            self.toast("The model returned an empty response.");
        }

        if let Some(stats) = stats {
            if let Some(tokens_per_second) = stats.tokens_per_second() {
                let label = gtk::Label::new(Some(&format!("{tokens_per_second:.1} tok/s")));
                label.add_css_class("dim-label");
                label.set_halign(gtk::Align::Start);
                label.set_margin_start(6);
                active.bubble.append(&label);
            }
        }

        if active.row.parent().is_some() {
            if let Some(assistant_id) = assistant_id {
                self.attach_message_actions(
                    active.conv_id,
                    &active.bubble,
                    "assistant",
                    &content,
                    &[],
                    Some(assistant_id),
                );
            }
        }

        self.set_streaming_ui(false);
        self.reload_sidebar();
        self.refresh_context_indicator();
        self.maybe_compress_context(active.conv_id);
        self.schedule_memory(active.conv_id, active.model.clone());
        self.scroll_to_bottom();

        let streaming_speech = self.speech_stream_tx.borrow().is_some();
        if streaming_speech {
            if !self.suppress_autoread.get() {
                self.flush_streaming_speech();
            }
            self.end_streaming_speech();
        }

        let suppressed = self.suppress_autoread.replace(false);
        let auto_read = {
            let settings = self.state.settings.borrow();
            settings.tts_enabled && (settings.tts_autoread || self.live_mode.get())
        };
        if !streaming_speech
            && auto_read
            && !suppressed
            && !content.trim().is_empty()
            && *self.current_id.borrow() == Some(active.conv_id)
            && self.resolve_tts().is_some()
        {
            self.speak(&content, None);
        }
    }

    fn stop_streaming(&self) {
        if let Some(active) = self.active.borrow().as_ref() {
            self.suppress_autoread.set(true);
            active.abort.abort();
        }
        self.stop_speech();
        self.set_streaming_ui(false);
    }

    fn set_streaming_ui(&self, streaming: bool) {
        self.send_button.set_sensitive(!streaming);
        self.stop_button.set_visible(streaming);
    }

    // ----- Long-term memory -----------------------------------------------

    fn model_pool(&self, model: &str) -> MemoryPool {
        let remote_host = self
            .state
            .models
            .borrow()
            .iter()
            .find(|candidate| memory::model_names_match(&candidate.name, model))
            .and_then(|candidate| candidate.remote_host.clone());
        let base_url = self.state.settings.borrow().base_url.clone();
        if memory::model_is_cloud(model, remote_host.as_deref(), &base_url) {
            MemoryPool::Cloud
        } else {
            MemoryPool::Local
        }
    }

    /// The pool to use for a model, if memory is enabled for that pool.
    fn memory_pool_for(&self, model: &str) -> Option<MemoryPool> {
        let pool = self.model_pool(model);
        let enabled = {
            let settings = self.state.settings.borrow();
            match pool {
                MemoryPool::Local => settings.memory_local_enabled,
                MemoryPool::Cloud => settings.memory_cloud_enabled,
            }
        };
        enabled.then_some(pool)
    }

    fn embedding_model(&self) -> String {
        self.state
            .settings
            .borrow()
            .memory_embedding_model
            .trim()
            .to_string()
    }

    /// When the model list is loaded, require a configured model that advertises
    /// the embedding capability. Before the first refresh, let the request fail
    /// naturally so memory is not blocked by a slow startup.
    fn embedding_model_ready(&self, model: &str) -> bool {
        if model.is_empty() {
            return false;
        }
        let models = self.state.models.borrow();
        if models.is_empty() {
            return true;
        }
        models.iter().any(|candidate| {
            memory::model_names_match(&candidate.name, model)
                && candidate
                    .capabilities
                    .iter()
                    .any(|capability| capability == "embedding")
        })
    }

    fn memory_warning(&self, message: &str) {
        if self.memory_warned.replace(true) {
            return;
        }
        tracing::warn!("memory: {message}");
        self.toast(message);
    }

    /// Debounce memory extraction so rapid turns are batched and a reply that
    /// is still streaming is not put under extra model load.
    fn schedule_memory(self: &Rc<Self>, conversation_id: i64, model: String) {
        let token = self.memory_token.get().wrapping_add(1);
        self.memory_token.set(token);

        let win = self.clone();
        glib::timeout_add_local_once(
            std::time::Duration::from_millis(MEMORY_DEBOUNCE_MS),
            move || {
                if win.memory_token.get() != token {
                    return;
                }
                if win.active.borrow().is_some() {
                    // The next completed reply will schedule again.
                    return;
                }
                if win.compressing.get() {
                    win.schedule_memory(conversation_id, model);
                    return;
                }
                win.enqueue_memory(conversation_id, &model);
            },
        );
    }

    fn advance_memory_checkpoint(&self, conversation_id: i64) {
        let latest = self
            .state
            .store
            .borrow()
            .latest_message_id(conversation_id)
            .ok()
            .flatten();
        if let Some(upto_id) = latest {
            let _ = self
                .state
                .store
                .borrow()
                .set_conversation_memory_checkpoint(conversation_id, upto_id);
        }
    }

    fn enqueue_memory(self: &Rc<Self>, conversation_id: i64, model: &str) {
        if model.is_empty() {
            return;
        }
        if self.memory_busy.get() {
            *self.memory_pending.borrow_mut() = Some((conversation_id, model.to_string()));
            return;
        }

        let Some(pool) = self.memory_pool_for(model) else {
            self.advance_memory_checkpoint(conversation_id);
            return;
        };

        let embedding_model = self.embedding_model();
        if !self.embedding_model_ready(&embedding_model) {
            self.memory_warning(
                "Long-term memory is paused: install an embedding model such as all-minilm.",
            );
            self.advance_memory_checkpoint(conversation_id);
            return;
        }

        let checkpoint = match self
            .state
            .store
            .borrow()
            .conversation_memory_checkpoint(conversation_id)
        {
            Ok(checkpoint) => checkpoint,
            Err(err) => {
                tracing::warn!("memory: could not read checkpoint: {err:#}");
                return;
            }
        };

        let messages = match self
            .state
            .store
            .borrow()
            .messages_after(conversation_id, checkpoint)
        {
            Ok(messages) => messages,
            Err(err) => {
                tracing::warn!("memory: could not read messages: {err:#}");
                return;
            }
        };
        let Some(upto_id) = messages.last().map(|message| message.id) else {
            return;
        };

        let num_ctx = self.state.settings.borrow().num_ctx.max(1);
        let transcript_budget = ((num_ctx as f64 * MEMORY_TRANSCRIPT_RATIO) as u32).max(256);
        let transcript = memory::build_transcript(&messages, transcript_budget);
        if transcript.trim().is_empty() {
            let _ = self
                .state
                .store
                .borrow()
                .set_conversation_memory_checkpoint(conversation_id, upto_id);
            return;
        }

        let max_chars = self.state.settings.borrow().memory_summary_chars.max(200) as usize;
        let created_date = self.state.store.borrow().local_date().unwrap_or_default();
        let job = MemoryJob {
            conversation_id,
            upto_id,
            pool,
            model: model.to_string(),
            embedding_model,
            created_date,
            transcript,
            max_chars,
            num_ctx,
            epoch: self.memory_epoch.get(),
        };
        self.memory_busy.set(true);

        let client = self.state.client.borrow().clone();
        let (tx, rx) = async_channel::bounded(1);
        self.state.runtime.spawn(async move {
            let result = run_memory_job(client, job).await;
            let _ = tx.send(result).await;
        });

        let win = self.clone();
        glib::spawn_future_local(async move {
            let result = rx.recv().await;
            win.finish_memory_job(result);
        });
    }

    fn finish_memory_job(
        self: &Rc<Self>,
        result: Result<anyhow::Result<MemoryJobResult>, async_channel::RecvError>,
    ) {
        self.memory_busy.set(false);

        match result {
            Ok(Ok(result)) => {
                if result.epoch == self.memory_epoch.get() {
                    let enabled = {
                        let settings = self.state.settings.borrow();
                        match result.pool {
                            MemoryPool::Local => settings.memory_local_enabled,
                            MemoryPool::Cloud => settings.memory_cloud_enabled,
                        }
                    };
                    if enabled {
                        self.store_memory_result(&result);
                    } else {
                        tracing::debug!("memory: pool disabled while a note was being extracted");
                    }
                } else {
                    tracing::debug!("memory: discarded a note from before memory was cleared");
                }
                let _ = self
                    .state
                    .store
                    .borrow()
                    .set_conversation_memory_checkpoint(result.conversation_id, result.upto_id);
            }
            Ok(Err(err)) => {
                self.memory_warning(&format!("Could not update long-term memory: {err:#}"));
            }
            Err(_) => {}
        }

        let pending = self.memory_pending.borrow_mut().take();
        if let Some((next_conversation, next_model)) = pending {
            self.enqueue_memory(next_conversation, &next_model);
        }
    }

    fn store_memory_result(&self, result: &MemoryJobResult) {
        let Some(text) = result.text.as_deref() else {
            return;
        };
        let Some(vector) = result.vector.as_deref() else {
            return;
        };

        let entries = match self
            .state
            .store
            .borrow()
            .memory_entries(result.pool, &result.embedding_model)
        {
            Ok(entries) => entries,
            Err(err) => {
                tracing::warn!("memory: could not load notes for dedup: {err:#}");
                Vec::new()
            }
        };
        let similarity = memory::max_similarity(&entries, vector);
        if similarity >= memory::MEMORY_DUPLICATE_SCORE {
            tracing::debug!(
                "memory: skipped a near-duplicate note ({:.3} similarity)",
                similarity
            );
            return;
        }

        match self.state.store.borrow().add_memory(
            result.pool,
            &result.embedding_model,
            &result.model,
            Some(result.conversation_id),
            text,
            vector,
        ) {
            Ok(true) => tracing::info!(
                "memory: saved a {} note from {} ({} chars)",
                result.pool.as_str(),
                result.model,
                text.chars().count()
            ),
            Ok(false) => tracing::debug!("memory: note already stored"),
            Err(err) => tracing::warn!("memory: could not save note: {err:#}"),
        }
    }

    fn clear_memory_pool(self: &Rc<Self>, pool: Option<MemoryPool>, row: &adw::ActionRow) {
        self.memory_epoch
            .set(self.memory_epoch.get().wrapping_add(1));
        match self.state.store.borrow().clear_memories(pool) {
            Ok(count) => {
                self.update_memory_counts_row(row);
                let message = match pool {
                    Some(MemoryPool::Local) => {
                        format!("Cleared {count} local long-term memories.")
                    }
                    Some(MemoryPool::Cloud) => {
                        format!("Cleared {count} cloud long-term memories.")
                    }
                    None => format!("Cleared {count} long-term memories."),
                };
                self.toast(&message);
            }
            Err(err) => self.toast(&format!("Could not clear long-term memory: {err:#}")),
        }
    }

    fn update_memory_counts_row(&self, row: &adw::ActionRow) {
        let (local, cloud) = self.state.store.borrow().memory_counts().unwrap_or((0, 0));
        row.set_subtitle(&format!(
            "{} local note{}, {} cloud note{}",
            local,
            if local == 1 { "" } else { "s" },
            cloud,
            if cloud == 1 { "" } else { "s" }
        ));
    }

    fn show_clear_memory_dialog(self: &Rc<Self>, row: &adw::ActionRow) {
        let (local, cloud) = self.state.store.borrow().memory_counts().unwrap_or((0, 0));
        if local == 0 && cloud == 0 {
            self.toast("Long-term memory is already empty.");
            return;
        }

        let dialog = gtk::Window::builder()
            .title("Clear long-term memory")
            .modal(true)
            .default_width(420)
            .build();
        dialog.set_transient_for(Some(&self.window));

        let header = gtk::HeaderBar::new();
        dialog.set_titlebar(Some(&header));

        let content = gtk::Box::new(gtk::Orientation::Vertical, 12);
        content.set_margin_top(18);
        content.set_margin_bottom(18);
        content.set_margin_start(18);
        content.set_margin_end(18);

        let label = gtk::Label::new(Some("This permanently deletes the selected memory notes."));
        label.set_wrap(true);
        label.set_xalign(0.0);
        content.append(&label);

        let buttons = gtk::Box::new(gtk::Orientation::Vertical, 8);
        content.append(&buttons);

        let add_clear_button = |label: &str, pool: Option<MemoryPool>| {
            let button = gtk::Button::with_label(label);
            button.add_css_class("destructive-action");
            let win = self.clone();
            let row = row.clone();
            let dialog = dialog.clone();
            button.connect_clicked(move |_| {
                win.clear_memory_pool(pool, &row);
                dialog.close();
            });
            buttons.append(&button);
        };

        if local > 0 && cloud > 0 {
            add_clear_button("Clear all", None);
        }
        if cloud > 0 {
            add_clear_button(&format!("Clear cloud ({cloud})"), Some(MemoryPool::Cloud));
        }
        if local > 0 {
            add_clear_button(&format!("Clear local ({local})"), Some(MemoryPool::Local));
        }

        let cancel = gtk::Button::with_label("Cancel");
        cancel.connect_clicked({
            let dialog = dialog.clone();
            move |_| dialog.close()
        });
        buttons.append(&cancel);

        let controller = gtk::EventControllerKey::new();
        let dialog_for_key = dialog.clone();
        controller.connect_key_pressed(move |_, key, _, _| {
            if key == gdk::Key::Escape {
                dialog_for_key.close();
                glib::Propagation::Stop
            } else {
                glib::Propagation::Proceed
            }
        });
        dialog.add_controller(controller);

        dialog.set_child(Some(&content));
        dialog.present();
        cancel.grab_focus();
    }

    // ----- Settings / theme / misc ----------------------------------------

    fn save_settings(&self) {
        if let Err(err) = self.state.settings.borrow().save() {
            self.toast(&format!("Could not save settings: {err:#}"));
        }
    }

    fn toggle_sidebar(self: &Rc<Self>) {
        let showing = !self.split_view.shows_sidebar();
        tracing::debug!(
            "sidebar: showing={showing} total={} clamp={} scroll={} messages={}",
            self.split_view.width(),
            self.messages_clamp.maximum_size(),
            self.messages_scroll.width(),
            self.messages_box.width()
        );

        if !showing {
            // Lock the message column to its current width before the sidebar
            // animates away, so hiding does not re-wrap every message.
            let width = self.message_column_width();
            if width > 0 {
                let limit = width.clamp(320, 900);
                self.messages_clamp.set_maximum_size(limit);
                self.messages_clamp.set_tightening_threshold(limit);
            }
        }

        self.sidebar_animating.set(true);
        self.split_view.set_show_sidebar(showing);

        let token = self.sidebar_animation_token.get().wrapping_add(1);
        self.sidebar_animation_token.set(token);
        let win = self.clone();
        glib::timeout_add_local_once(std::time::Duration::from_millis(1000), move || {
            if win.sidebar_animation_token.get() != token {
                return;
            }
            win.sidebar_animating.set(false);
            win.update_message_width();
        });
    }

    /// Keep the chat column at the width it has when the sidebar is visible, so
    /// toggling the sidebar does not re-wrap every message while it animates.
    fn update_message_width(&self) {
        if self.sidebar_animating.get() {
            return;
        }

        let total = self.split_view.width();
        if total <= 0 {
            return;
        }
        let previous_total = self.last_split_width.replace(total);
        let resized = previous_total != 0 && previous_total != total;

        if !self.split_view.shows_sidebar() {
            if resized {
                // The hidden width will be recalculated the next time the
                // sidebar is shown.
                self.sidebar_width_stale.set(true);
            }
            if self.messages_clamp.maximum_size() == 900 {
                // The app started with the sidebar hidden. Estimate the width
                // the content area will have once it is shown.
                let fraction = self.split_view.sidebar_width_fraction();
                let min = self.split_view.min_sidebar_width();
                let max = self.split_view.max_sidebar_width();
                let sidebar = (total as f64 * fraction).clamp(min, max);
                let estimate = (total as f64 - sidebar - 40.0).clamp(320.0, 900.0) as i32;
                self.messages_clamp.set_maximum_size(estimate);
                self.messages_clamp.set_tightening_threshold(estimate);
            }
            return;
        }

        if resized || self.sidebar_width_stale.replace(false) {
            // Let the clamp use the new available width; the next tick locks it
            // to the actual content width.
            self.messages_clamp.set_maximum_size(900);
            self.messages_clamp.set_tightening_threshold(900);
            return;
        }

        let width = self.message_column_width();
        if width <= 0 {
            return;
        }

        let limit = width.clamp(320, 900);
        if self.messages_clamp.maximum_size() != limit {
            tracing::debug!(
                "sidebar: clamp {limit} (box width={} total={})",
                self.messages_box.width(),
                total
            );
            self.messages_clamp.set_maximum_size(limit);
            self.messages_clamp.set_tightening_threshold(limit);
        }
        if self.last_image_column.replace(limit) != limit {
            // Fit after the new width has been allocated, otherwise the
            // pictures still report the previous card width.
            let pictures: Vec<gtk::Picture> = {
                let mut refs = self.message_pictures.borrow_mut();
                refs.retain(|weak| weak.upgrade().is_some());
                refs.iter().filter_map(|weak| weak.upgrade()).collect()
            };
            glib::idle_add_local_once(move || {
                for picture in pictures {
                    fit_message_picture(&picture, limit);
                }
            });
        }
    }

    fn message_column_width(&self) -> i32 {
        let width = self.messages_box.width();
        if width <= 0 {
            return 0;
        }
        let margins = self.messages_box.margin_start() + self.messages_box.margin_end();
        width + margins
    }

    fn sync_sidebar_visibility(&self, visible: bool) {
        tracing::debug!("sidebar: visibility changed to {visible}");
        self.sidebar_button.set_tooltip_text(Some(if visible {
            "Hide sidebar (Ctrl+B)"
        } else {
            "Show sidebar (Ctrl+B)"
        }));

        let changed = {
            let mut settings = self.state.settings.borrow_mut();
            if settings.sidebar_visible == visible {
                false
            } else {
                settings.sidebar_visible = visible;
                true
            }
        };
        if changed {
            self.save_settings();
        }
    }

    fn parakeet_model(&self) -> crate::parakeet::ParakeetModel {
        crate::parakeet::ParakeetModel::from_setting(
            &self.state.settings.borrow().stt_parakeet_model,
        )
    }

    fn update_parakeet_model_row(
        &self,
        row: &adw::ActionRow,
        download: &gtk::Button,
        delete: &gtk::Button,
    ) {
        let model = self.parakeet_model();
        let downloaded = crate::parakeet::is_downloaded(model);
        row.set_title(&format!("{} model", model.display_name()));
        row.set_subtitle(&format!(
            "{} · {} · CC-BY-4.0",
            if downloaded {
                "Downloaded"
            } else {
                "Not downloaded"
            },
            model.approx_size()
        ));
        download.set_label(if downloaded { "Downloaded" } else { "Download" });
        download.set_sensitive(!downloaded);
        delete.set_visible(downloaded);
    }

    fn update_paradee_model_row(
        &self,
        row: &adw::ActionRow,
        download: &gtk::Button,
        delete: &gtk::Button,
    ) {
        let downloaded = crate::paradee::is_downloaded();
        row.set_subtitle(if downloaded {
            "Downloaded · 8.5 MB · Apache-2.0"
        } else {
            "Not downloaded · 8.5 MB · Apache-2.0"
        });
        download.set_label(if downloaded { "Downloaded" } else { "Download" });
        download.set_sensitive(!downloaded);
        delete.set_visible(downloaded);
    }

    fn update_whistle_model_row(
        &self,
        row: &adw::ActionRow,
        download: &gtk::Button,
        delete: &gtk::Button,
    ) {
        let downloaded = crate::whistle::is_downloaded();
        row.set_subtitle(if downloaded {
            "Downloaded · ~160 MB · Apache-2.0"
        } else {
            "Not downloaded · ~160 MB · Apache-2.0"
        });
        download.set_label(if downloaded { "Downloaded" } else { "Download" });
        download.set_sensitive(!downloaded);
        delete.set_visible(downloaded);
    }

    fn sync_stt_model_rows(
        &self,
        provider: &str,
        parakeet_model_row: &adw::ComboRow,
        parakeet_download: &gtk::Button,
        whistle_download: &gtk::Button,
    ) {
        parakeet_model_row.set_visible(provider == "parakeet");
        parakeet_download.set_visible(matches!(provider, "auto" | "parakeet"));
        whistle_download.set_visible(matches!(provider, "auto" | "whistle"));
    }

    fn sync_tts_model_row(&self, backend: &str, paradee_download: &gtk::Button) {
        paradee_download.set_visible(matches!(backend, "auto" | "paradee"));
    }

    fn live_available(&self) -> bool {
        let enabled = {
            let settings = self.state.settings.borrow();
            settings.stt_enabled && settings.tts_enabled
        };
        enabled && self.resolve_tts().is_some()
    }

    // ----- Wake word -------------------------------------------------------

    fn wake_should_run(&self) -> bool {
        if !self.state.settings.borrow().wake_enabled || !self.wake_active.get() {
            return false;
        }
        if self.live_mode.get() || self.live_pending.get() {
            return false;
        }
        if self.stt_listening.get() || self.stt_preparing.get() {
            return false;
        }
        if self.speech_active.get() {
            return false;
        }
        self.live_available()
    }

    /// Start or stop wake-word detection to match the current app state.
    fn refresh_wake(self: &Rc<Self>) {
        if self.wake_should_run() {
            if self.wake_session.borrow().is_none() && !self.wake_preparing.get() {
                self.start_wake();
            }
        } else {
            self.stop_wake();
        }
        self.update_tray_state();
    }

    fn start_wake(self: &Rc<Self>) {
        if self.wake_preparing.get() || self.wake_session.borrow().is_some() {
            return;
        }
        self.wake_preparing.set(true);
        let generation = self.wake_generation.get().saturating_add(1);
        self.wake_generation.set(generation);

        let (word, threshold) = {
            let settings = self.state.settings.borrow();
            (settings.wake_word.clone(), settings.wake_threshold)
        };
        tracing::info!("wake word: preparing {word:?}");

        let (tx, rx) = async_channel::bounded(1);
        self.state.runtime.spawn(async move {
            let result = wake::prepare(&word, threshold).await;
            let _ = tx.send(result).await;
        });

        let win = self.clone();
        glib::spawn_future_local(async move {
            let Ok(result) = rx.recv().await else {
                return;
            };
            if win.wake_generation.get() != generation {
                return;
            }
            win.wake_preparing.set(false);
            match result {
                Ok(detector) => win.begin_wake(detector),
                Err(error) => {
                    tracing::warn!("wake word: {error:#}");
                    win.toast(&format!("Wake word unavailable: {error}"));
                    win.update_tray_state();
                }
            }
        });
    }

    fn begin_wake(self: &Rc<Self>, detector: wake::WakeDetector) {
        if !self.wake_should_run() {
            return;
        }

        let (tx, rx) = async_channel::unbounded();
        match wake::start(detector, tx) {
            Ok(session) => {
                *self.wake_session.borrow_mut() = Some(session);
                self.wake_listening.set(true);
                tracing::info!("wake word: listening");

                let id = self.wake_generation.get();
                let win = self.clone();
                glib::spawn_future_local(async move {
                    let Ok(event) = rx.recv().await else {
                        return;
                    };
                    if win.wake_generation.get() != id {
                        return;
                    }
                    match event {
                        WakeEvent::Detected(score) => {
                            tracing::debug!("wake word: score {score:.3}");
                            win.on_wake_detected();
                        }
                        WakeEvent::Error(error) => {
                            win.toast(&format!("Wake word failed: {error}"));
                            win.stop_wake();
                        }
                    }
                });
                self.update_tray_state();
            }
            Err(error) => {
                self.toast(&format!("Could not listen for the wake word: {error:#}"));
                self.update_tray_state();
            }
        }
    }

    fn on_wake_detected(self: &Rc<Self>) {
        if !self.wake_should_run() {
            return;
        }
        // Release the microphone before the dictation path opens it again.
        self.stop_wake();
        self.wake_pending.set(true);
        feedback::play(Cue::Wake);

        // Let the cue finish first, otherwise the live session picks it up
        // through the speakers and transcribes it as speech.
        let win = self.clone();
        glib::timeout_add_local_once(std::time::Duration::from_millis(260), move || {
            if win.wake_pending.get() {
                win.start_stt(true);
            }
        });
    }

    fn stop_wake(&self) {
        self.wake_generation
            .set(self.wake_generation.get().saturating_add(1));
        if let Some(session) = self.wake_session.borrow_mut().take() {
            session.stop();
        }
        if self.wake_listening.replace(false) {
            tracing::info!("wake word: stopped listening");
        }
        self.wake_preparing.set(false);
        self.update_tray_state();
    }

    fn start_tray(self: &Rc<Self>, row: Option<adw::SwitchRow>) {
        if let Some(handle) = self.tray.borrow().as_ref() {
            if !handle.is_closed() {
                self.window.set_hide_on_close(true);
                return;
            }
        }
        self.tray.borrow_mut().take();

        let (commands_tx, commands_rx) = async_channel::unbounded();
        let live = self.live_mode.get() || self.live_pending.get();
        let live_available = self.live_available();
        let (wake_visible, wake_word) = {
            let settings = self.state.settings.borrow();
            (settings.wake_enabled, settings.wake_word.clone())
        };
        self.tray_live.set(live);
        self.tray_live_available.set(live_available);
        self.tray_wake_visible.set(wake_visible);
        self.tray_wake_listening.set(self.wake_listening.get());
        *self.tray_wake_word.borrow_mut() = wake_word.clone();
        let watcher_online = Arc::new(AtomicBool::new(true));
        let autostart = self.autostart;
        let tray = StatusTray {
            commands: commands_tx,
            live,
            live_available,
            wake_visible,
            wake_listening: self.wake_listening.get(),
            wake_word,
            watcher_online: watcher_online.clone(),
        };
        let (result_tx, result_rx) = async_channel::bounded(1);
        self.state.runtime.spawn(async move {
            // An autostart launch may beat the tray host. Let ksni wait for it
            // instead of failing and showing the window immediately.
            let result = if autostart {
                tray.assume_sni_available(true).spawn().await
            } else {
                tray.spawn().await
            };
            let result = result.map_err(|err| format!("{err}"));
            let _ = result_tx.send(result).await;
        });

        let win = self.clone();
        glib::spawn_future_local(async move {
            match result_rx.recv().await {
                Ok(Ok(handle)) => {
                    let enabled = win.autostart || win.state.settings.borrow().run_in_background;
                    if !enabled {
                        win.state.runtime.spawn(async move {
                            handle.shutdown().await;
                        });
                        return;
                    }

                    *win.tray.borrow_mut() = Some(handle);
                    win.window.set_hide_on_close(true);
                    win.update_tray_state();

                    if win.autostart {
                        let win = win.clone();
                        let online = watcher_online.clone();
                        glib::timeout_add_seconds_local_once(15, move || {
                            if !online.load(Ordering::Relaxed) {
                                tracing::warn!(
                                    "system tray watcher did not appear; showing window"
                                );
                                win.toast("The system tray is unavailable; showing the window.");
                                win.window.set_hide_on_close(false);
                                win.window.present();
                            }
                        });
                    }

                    let win = win.clone();
                    glib::spawn_future_local(async move {
                        while let Ok(command) = commands_rx.recv().await {
                            match command {
                                TrayCommand::ToggleWindow(token) => {
                                    win.toggle_window_from_tray(token);
                                }
                                TrayCommand::ToggleLive => win.toggle_live(),
                                TrayCommand::ToggleWake => {
                                    win.autostart_wake_pending.set(false);
                                    let active = !win.wake_active.get();
                                    win.wake_active.set(active);
                                    win.refresh_wake();
                                    win.update_tray_state();
                                }
                                TrayCommand::Quit => win.quit_from_tray(),
                            }
                        }
                    });
                }
                Ok(Err(err)) => {
                    tracing::warn!("system tray unavailable: {err}");
                    win.window.set_hide_on_close(false);
                    if win.autostart {
                        win.toast("The system tray is unavailable; showing the window.");
                        if !win.window.is_visible() {
                            win.window.present();
                        }
                    } else {
                        win.toast("System tray is unavailable; background mode is disabled.");
                        win.state.settings.borrow_mut().run_in_background = false;
                        win.save_settings();
                        if !win.window.is_visible() {
                            win.window.present();
                        }
                        if let Some(row) = row {
                            row.set_active(false);
                        }
                    }
                }
                Err(_) => {}
            }
        });
    }

    fn stop_tray(&self) {
        let handle = self.tray.borrow_mut().take();
        if let Some(handle) = handle {
            self.state.runtime.spawn(async move {
                handle.shutdown().await;
            });
        }
    }

    fn update_tray_state(&self) {
        let live = self.live_mode.get() || self.live_pending.get();
        let live_available = self.live_available();
        let (wake_visible, wake_word) = {
            let settings = self.state.settings.borrow();
            (settings.wake_enabled, settings.wake_word.clone())
        };
        let wake_listening = self.wake_listening.get();

        if self.tray_live.get() == live
            && self.tray_live_available.get() == live_available
            && self.tray_wake_visible.get() == wake_visible
            && self.tray_wake_listening.get() == wake_listening
            && *self.tray_wake_word.borrow() == wake_word
        {
            return;
        }
        self.tray_live.set(live);
        self.tray_live_available.set(live_available);
        self.tray_wake_visible.set(wake_visible);
        self.tray_wake_listening.set(wake_listening);
        *self.tray_wake_word.borrow_mut() = wake_word.clone();

        let Some(handle) = self.tray.borrow().clone() else {
            return;
        };
        self.state.runtime.spawn(async move {
            let _ = handle
                .update(move |tray| {
                    tray.live = live;
                    tray.live_available = live_available;
                    tray.wake_visible = wake_visible;
                    tray.wake_listening = wake_listening;
                    tray.wake_word = wake_word.clone();
                })
                .await;
        });
    }

    fn set_background_enabled(self: &Rc<Self>, enabled: bool, row: Option<adw::SwitchRow>) {
        let changed = {
            let mut settings = self.state.settings.borrow_mut();
            if settings.run_in_background == enabled {
                false
            } else {
                settings.run_in_background = enabled;
                true
            }
        };
        if changed {
            self.save_settings();
        }

        if enabled {
            self.window.set_hide_on_close(true);
            self.start_tray(row);
        } else {
            self.stop_tray();
            self.window.set_hide_on_close(false);
        }
    }

    fn apply_background_startup(self: &Rc<Self>) {
        if self.autostart || self.state.settings.borrow().run_in_background {
            self.window.set_hide_on_close(true);
            self.start_tray(None);
        }
    }

    fn toggle_window_from_tray(&self, token: Option<String>) {
        if self.window.is_active() {
            self.window.set_visible(false);
        } else {
            if let Some(token) = token {
                self.window.set_startup_id(&token);
            }
            self.window.present();
        }
    }

    fn quit_from_tray(self: &Rc<Self>) {
        self.stop_stt();
        self.stop_speech();
        if let Some(app) = self.window.application() {
            app.quit();
        }
    }

    fn apply_theme(&self) {
        let theme = self.state.settings.borrow().theme.clone();
        let scheme = match theme.as_str() {
            "light" => adw::ColorScheme::ForceLight,
            "dark" => adw::ColorScheme::ForceDark,
            _ => adw::ColorScheme::Default,
        };
        adw::StyleManager::default().set_color_scheme(scheme);
    }

    fn show_settings(self: &Rc<Self>) {
        let page = adw::PreferencesPage::new();
        let prefs_dialog = adw::PreferencesDialog::builder()
            .title("Preferences")
            .build();

        let connection = adw::PreferencesGroup::new();
        connection.set_title("Connection");

        let url_row = adw::EntryRow::new();
        url_row.set_title("Server URL");
        url_row.set_text(&self.state.settings.borrow().base_url);
        connection.add(&url_row);

        let key_row = adw::PasswordEntryRow::new();
        key_row.set_title("API key (optional)");
        key_row.set_text(&self.state.settings.borrow().api_key);
        connection.add(&key_row);

        page.add(&connection);

        let application = adw::PreferencesGroup::new();
        application.set_title("Application");

        let background_row = adw::SwitchRow::new();
        background_row.set_title("Keep running in the background");
        background_row.set_subtitle("Closing the window hides it to the system tray");
        background_row.set_active(self.state.settings.borrow().run_in_background);
        application.add(&background_row);

        let autostart_row = adw::SwitchRow::new();
        autostart_row.set_title("Start automatically at login");
        autostart_row.set_subtitle(if self.state.settings.borrow().wake_enabled {
            "Starts hidden in the tray and listens for the wake word"
        } else {
            "Starts hidden in the system tray"
        });
        autostart_row.set_active(autostart::is_enabled());
        application.add(&autostart_row);
        page.add(&application);

        {
            let win = self.clone();
            let autostart_row = autostart_row.clone();
            background_row.connect_active_notify(move |row| {
                win.set_background_enabled(row.is_active(), Some(row.clone()));
                if !row.is_active() && autostart::is_enabled() {
                    match autostart::set_enabled(false) {
                        Ok(()) => autostart_row.set_active(false),
                        Err(err) => {
                            tracing::warn!("autostart: {err:#}");
                            win.toast(&format!("Could not disable autostart: {err}"));
                        }
                    }
                }
            });
        }
        {
            let win = self.clone();
            let background_row = background_row.clone();
            autostart_row.connect_active_notify(move |row| {
                let enabled = row.is_active();
                if enabled && !win.state.settings.borrow().run_in_background {
                    // A hidden window needs the tray, so background mode is
                    // implied by autostart.
                    background_row.set_active(true);
                    win.set_background_enabled(true, Some(background_row.clone()));
                }
                if let Err(err) = autostart::set_enabled(enabled) {
                    tracing::warn!("autostart: {err:#}");
                    row.set_active(!enabled);
                    win.toast(&format!("Could not update autostart: {err}"));
                }
            });
        }

        let generation = adw::PreferencesGroup::new();
        generation.set_title("Generation");

        let system_row = adw::EntryRow::new();
        system_row.set_title("System prompt");
        system_row.set_text(&self.state.settings.borrow().system_prompt);
        generation.add(&system_row);

        let temperature_row = adw::SpinRow::new(
            Some(&gtk::Adjustment::new(
                self.state.settings.borrow().temperature,
                0.0,
                2.0,
                0.05,
                0.1,
                0.0,
            )),
            0.05,
            2,
        );
        temperature_row.set_title("Temperature");
        temperature_row.set_subtitle("Higher values produce more varied replies");
        generation.add(&temperature_row);

        let context_row = adw::SpinRow::new(
            Some(&gtk::Adjustment::new(
                self.state.settings.borrow().num_ctx as f64,
                512.0,
                262_144.0,
                512.0,
                4096.0,
                0.0,
            )),
            1024.0,
            0,
        );
        context_row.set_title("Context size");
        context_row.set_subtitle("num_ctx passed to Ollama");
        generation.add(&context_row);

        page.add(&generation);

        let memory_group = adw::PreferencesGroup::new();
        memory_group.set_title("Long-term memory");
        memory_group.set_description(Some(
            "Stored locally in the chat database. Local and cloud models use separate pools.",
        ));

        let memory_local_row = adw::SwitchRow::new();
        memory_local_row.set_title("Use memory with local models");
        memory_local_row.set_subtitle("Read and write long-term memory with local models");
        memory_local_row.set_active(self.state.settings.borrow().memory_local_enabled);
        memory_group.add(&memory_local_row);

        let memory_cloud_row = adw::SwitchRow::new();
        memory_cloud_row.set_title("Use memory with cloud models");
        memory_cloud_row.set_subtitle("Read and write long-term memory with cloud models");
        memory_cloud_row.set_active(self.state.settings.borrow().memory_cloud_enabled);
        memory_group.add(&memory_cloud_row);

        let mut embedding_names: Vec<String> = self
            .state
            .models
            .borrow()
            .iter()
            .filter(|model| {
                model
                    .capabilities
                    .iter()
                    .any(|capability| capability == "embedding")
            })
            .map(|model| model.name.clone())
            .collect();
        embedding_names.sort();
        embedding_names.dedup();
        let configured_embedding = self.state.settings.borrow().memory_embedding_model.clone();
        if let Some(name) = embedding_names
            .iter()
            .find(|name| memory::model_names_match(name, &configured_embedding))
            .cloned()
        {
            if name != configured_embedding {
                self.state.settings.borrow_mut().memory_embedding_model = name;
                self.save_settings();
            }
        } else if !configured_embedding.is_empty() {
            embedding_names.insert(0, configured_embedding);
        }

        let memory_embedding_row = adw::ComboRow::new();
        memory_embedding_row.set_title("Embedding model");
        memory_embedding_row.set_subtitle(if embedding_names.is_empty() {
            "No embedding model installed. Pull one with `ollama pull all-minilm`."
        } else {
            "Installed embedding-capable model used for memory text. Changing it hides notes from the previous model."
        });
        let embedding_refs: Vec<&str> = embedding_names.iter().map(String::as_str).collect();
        let embedding_list = gtk::StringList::new(&embedding_refs);
        memory_embedding_row.set_model(Some(&embedding_list));
        if let Some(index) = embedding_names
            .iter()
            .position(|name| memory::model_names_match(name, &self.embedding_model()))
        {
            memory_embedding_row.set_selected(index as u32);
        }
        memory_group.add(&memory_embedding_row);

        let memory_top_row = adw::SpinRow::new(
            Some(&gtk::Adjustment::new(
                self.state.settings.borrow().memory_retrieve_top_k as f64,
                1.0,
                20.0,
                1.0,
                4.0,
                0.0,
            )),
            1.0,
            0,
        );
        memory_top_row.set_title("Retrieved memories");
        memory_top_row.set_subtitle("Maximum relevant notes included in each request");
        memory_group.add(&memory_top_row);

        let memory_length_row = adw::SpinRow::new(
            Some(&gtk::Adjustment::new(
                self.state.settings.borrow().memory_summary_chars as f64,
                200.0,
                10_000.0,
                100.0,
                500.0,
                0.0,
            )),
            100.0,
            0,
        );
        memory_length_row.set_title("Maximum note length");
        memory_length_row.set_subtitle("Characters retained in one memory note");
        memory_group.add(&memory_length_row);

        let memory_clear_row = adw::ActionRow::new();
        memory_clear_row.set_title("Clear long-term memory");
        self.update_memory_counts_row(&memory_clear_row);
        let memory_clear_button = gtk::Button::with_label("Clear…");
        memory_clear_button.add_css_class("flat");
        memory_clear_button.set_valign(gtk::Align::Center);
        memory_clear_row.add_suffix(&memory_clear_button);
        memory_group.add(&memory_clear_row);

        page.add(&memory_group);

        {
            let win = self.clone();
            memory_local_row.connect_active_notify(move |row| {
                win.state.settings.borrow_mut().memory_local_enabled = row.is_active();
                win.save_settings();
            });
        }
        {
            let win = self.clone();
            memory_cloud_row.connect_active_notify(move |row| {
                win.state.settings.borrow_mut().memory_cloud_enabled = row.is_active();
                win.save_settings();
            });
        }
        {
            let win = self.clone();
            let names = embedding_names.clone();
            memory_embedding_row.connect_selected_notify(move |row| {
                let index = row.selected() as usize;
                if let Some(name) = names.get(index) {
                    win.state.settings.borrow_mut().memory_embedding_model = name.clone();
                    win.save_settings();
                }
            });
        }
        {
            let win = self.clone();
            memory_top_row.connect_value_notify(move |row| {
                win.state.settings.borrow_mut().memory_retrieve_top_k = row.value().max(1.0) as u32;
                win.save_settings();
            });
        }
        {
            let win = self.clone();
            memory_length_row.connect_value_notify(move |row| {
                win.state.settings.borrow_mut().memory_summary_chars =
                    row.value().max(200.0) as u32;
                win.save_settings();
            });
        }
        {
            let win = self.clone();
            let row = memory_clear_row.clone();
            memory_clear_button.connect_clicked(move |_| {
                win.show_clear_memory_dialog(&row);
            });
        }

        let speech = adw::PreferencesGroup::new();
        speech.set_title("Speech");

        let tts_enable_row = adw::SwitchRow::new();
        tts_enable_row.set_title("Enable text-to-speech");
        tts_enable_row.set_subtitle("Show read-aloud buttons on assistant replies");
        tts_enable_row.set_active(self.state.settings.borrow().tts_enabled);
        speech.add(&tts_enable_row);

        let backend_row = adw::ComboRow::new();
        backend_row.set_title("Preferred TTS");
        backend_row.set_subtitle(
            "Automatic uses Paradee for English and the Ollama model for other languages",
        );
        let backends = gtk::StringList::new(&[
            "Automatic",
            "Paradee (local ONNX)",
            "Ollama model (built-in)",
        ]);
        backend_row.set_model(Some(&backends));
        backend_row.set_selected(tts_backend_index(&self.state.settings.borrow().tts_backend));
        speech.add(&backend_row);

        let language_row = adw::ComboRow::new();
        language_row.set_title("Language");
        language_row
            .set_subtitle("Paradee: English only. Built-in languages depend on the selected model");
        let language_choices: Rc<RefCell<Vec<TtsLanguage>>> = Rc::new(RefCell::new(Vec::new()));
        let suppress_language = Rc::new(Cell::new(false));
        {
            let settings = self.state.settings.borrow();
            let selected = fill_language_row(
                &language_row,
                &settings.tts_backend,
                &settings.tts_language,
                &language_choices,
                &suppress_language,
            );
            drop(settings);
            if self.state.settings.borrow().tts_language != selected {
                self.state.settings.borrow_mut().tts_language = selected;
                self.save_settings();
            }
        }
        speech.add(&language_row);

        let voice_row = adw::ComboRow::new();
        voice_row.set_title("Voice");
        voice_row.set_subtitle("Female or male, depending on the selected TTS and language");
        let voice_choices: Rc<RefCell<Vec<VoiceOption>>> = Rc::new(RefCell::new(Vec::new()));
        let suppress_voice = Rc::new(Cell::new(false));
        {
            let settings = self.state.settings.borrow();
            let language = tts_language(&settings.tts_language);
            let selected = fill_voice_row(
                &voice_row,
                &settings.tts_backend,
                language,
                &settings.tts_voice,
                &voice_choices,
                &suppress_voice,
            );
            drop(settings);
            if self.state.settings.borrow().tts_voice != selected {
                self.state.settings.borrow_mut().tts_voice = selected;
                self.save_settings();
            }
        }
        speech.add(&voice_row);

        let tts_autoread_row = adw::SwitchRow::new();
        tts_autoread_row.set_title("Read replies automatically");
        tts_autoread_row.set_subtitle("Speak each assistant reply as it streams in");
        tts_autoread_row.set_active(self.state.settings.borrow().tts_autoread);
        tts_autoread_row.set_sensitive(self.state.settings.borrow().tts_enabled);
        speech.add(&tts_autoread_row);

        let paradee_row = adw::ActionRow::new();
        paradee_row.set_title("Paradee model");
        let paradee_spinner = gtk::Spinner::new();
        paradee_spinner.set_valign(gtk::Align::Center);
        paradee_spinner.set_visible(false);
        paradee_row.add_suffix(&paradee_spinner);
        let paradee_button = gtk::Button::new();
        paradee_button.add_css_class("flat");
        paradee_button.set_valign(gtk::Align::Center);
        paradee_row.add_suffix(&paradee_button);
        let paradee_delete = gtk::Button::from_icon_name("user-trash-symbolic");
        paradee_delete.add_css_class("flat");
        paradee_delete.set_valign(gtk::Align::Center);
        paradee_delete.set_tooltip_text(Some("Delete model"));
        paradee_row.add_suffix(&paradee_delete);
        self.update_paradee_model_row(&paradee_row, &paradee_button, &paradee_delete);
        let tts_backend = self.state.settings.borrow().tts_backend.clone();
        self.sync_tts_model_row(&tts_backend, &paradee_button);
        speech.add(&paradee_row);

        page.add(&speech);

        let stt = adw::PreferencesGroup::new();
        stt.set_title("Speech to Text");

        let stt_enable_row = adw::SwitchRow::new();
        stt_enable_row.set_title("Enable dictation");
        stt_enable_row.set_subtitle("Show a microphone button in the composer");
        stt_enable_row.set_active(self.state.settings.borrow().stt_enabled);
        stt.add(&stt_enable_row);

        let stt_provider_row = adw::ComboRow::new();
        stt_provider_row.set_title("Preferred STT");
        stt_provider_row.set_subtitle(
            "Automatic prefers the selected Parakeet model, then Whistle, then an audio-capable Ollama model",
        );
        let stt_providers = gtk::StringList::new(&[
            "Automatic",
            "Parakeet (local ONNX)",
            "Whistle (local ONNX)",
            "Ollama model (built-in)",
        ]);
        stt_provider_row.set_model(Some(&stt_providers));
        stt_provider_row.set_selected(stt_provider_index(
            &self.state.settings.borrow().stt_provider,
        ));
        stt.add(&stt_provider_row);

        let parakeet_model_row = adw::ComboRow::new();
        parakeet_model_row.set_title("Parakeet model");
        parakeet_model_row
            .set_subtitle("Redux is faster on CPU; Ultra is more accurate but larger");
        let parakeet_models =
            gtk::StringList::new(&["Redux — fast · ~420 MB", "Ultra — accurate · ~720 MB"]);
        parakeet_model_row.set_model(Some(&parakeet_models));
        parakeet_model_row.set_selected(
            if self.parakeet_model() == crate::parakeet::ParakeetModel::Ultra {
                1
            } else {
                0
            },
        );
        stt.add(&parakeet_model_row);

        let stt_language_row = adw::ComboRow::new();
        stt_language_row.set_title("Language");
        stt_language_row.set_subtitle(
            "Parakeet auto-detects; Whistle supports seven languages; built-in depends on the model",
        );
        let stt_choices: Rc<RefCell<Vec<SttLanguage>>> = Rc::new(RefCell::new(Vec::new()));
        let suppress_stt_language = Rc::new(Cell::new(false));
        {
            let settings = self.state.settings.borrow();
            let selected = fill_stt_language_row(
                &stt_language_row,
                &settings.stt_provider,
                &settings.stt_language,
                &stt_choices,
                &suppress_stt_language,
            );
            drop(settings);
            if self.state.settings.borrow().stt_language != selected {
                self.state.settings.borrow_mut().stt_language = selected;
                self.save_settings();
            }
        }
        stt.add(&stt_language_row);

        let stt_start_row = adw::SpinRow::new(
            Some(&gtk::Adjustment::new(
                self.state.settings.borrow().stt_start_timeout_secs as f64,
                1.0,
                60.0,
                1.0,
                5.0,
                0.0,
            )),
            1.0,
            0,
        );
        stt_start_row.set_title("Wait for speech");
        stt_start_row.set_subtitle("Seconds before listening gives up when no voice is detected");
        stt.add(&stt_start_row);

        let stt_end_row = adw::SpinRow::new(
            Some(&gtk::Adjustment::new(
                self.state.settings.borrow().stt_end_silence_ms as f64 / 1000.0,
                0.3,
                5.0,
                0.1,
                0.5,
                0.0,
            )),
            0.1,
            1,
        );
        stt_end_row.set_title("End of speech silence");
        stt_end_row.set_subtitle("Seconds of silence before the transcript is finished");
        stt.add(&stt_end_row);

        let stt_autosend_row = adw::SwitchRow::new();
        stt_autosend_row.set_title("Send automatically");
        stt_autosend_row.set_subtitle("Send the transcript when speech ends");
        stt_autosend_row.set_active(self.state.settings.borrow().stt_autosend);
        stt.add(&stt_autosend_row);

        let parakeet_row = adw::ActionRow::new();
        let parakeet_spinner = gtk::Spinner::new();
        parakeet_spinner.set_valign(gtk::Align::Center);
        parakeet_spinner.set_visible(false);
        parakeet_row.add_suffix(&parakeet_spinner);
        let parakeet_button = gtk::Button::new();
        parakeet_button.add_css_class("flat");
        parakeet_button.set_valign(gtk::Align::Center);
        parakeet_row.add_suffix(&parakeet_button);
        let parakeet_delete = gtk::Button::from_icon_name("user-trash-symbolic");
        parakeet_delete.add_css_class("flat");
        parakeet_delete.set_valign(gtk::Align::Center);
        parakeet_delete.set_tooltip_text(Some("Delete model"));
        parakeet_row.add_suffix(&parakeet_delete);
        self.update_parakeet_model_row(&parakeet_row, &parakeet_button, &parakeet_delete);
        stt.add(&parakeet_row);

        let whistle_row = adw::ActionRow::new();
        whistle_row.set_title("Whistle model");
        let whistle_spinner = gtk::Spinner::new();
        whistle_spinner.set_valign(gtk::Align::Center);
        whistle_spinner.set_visible(false);
        whistle_row.add_suffix(&whistle_spinner);
        let whistle_button = gtk::Button::new();
        whistle_button.add_css_class("flat");
        whistle_button.set_valign(gtk::Align::Center);
        whistle_row.add_suffix(&whistle_button);
        let whistle_delete = gtk::Button::from_icon_name("user-trash-symbolic");
        whistle_delete.add_css_class("flat");
        whistle_delete.set_valign(gtk::Align::Center);
        whistle_delete.set_tooltip_text(Some("Delete model"));
        whistle_row.add_suffix(&whistle_delete);
        self.update_whistle_model_row(&whistle_row, &whistle_button, &whistle_delete);
        stt.add(&whistle_row);

        let stt_provider = self.state.settings.borrow().stt_provider.clone();
        self.sync_stt_model_rows(
            &stt_provider,
            &parakeet_model_row,
            &parakeet_button,
            &whistle_button,
        );

        page.add(&stt);

        let wake = adw::PreferencesGroup::new();
        wake.set_title("Wake word");
        wake.set_description(Some(
            "Start live conversation with a phrase. Use the tray menu to start and stop listening.",
        ));

        let wake_enable_row = adw::SwitchRow::new();
        wake_enable_row.set_title("Enable wake word");
        wake_enable_row.set_subtitle(
            "Adds a Wake word toggle to the tray; autostart launches arm it automatically",
        );
        wake_enable_row.set_active(self.state.settings.borrow().wake_enabled);
        wake.add(&wake_enable_row);

        let current_word = self.state.settings.borrow().wake_word.clone();
        let word_labels: Vec<String> = crate::wake::WORDS
            .iter()
            .map(|(word, language)| {
                let name = crate::wake::display_name(word);
                if *language == "en" {
                    name
                } else {
                    format!("{name} ({language})")
                }
            })
            .collect();
        let word_refs: Vec<&str> = word_labels.iter().map(String::as_str).collect();
        let word_list = gtk::StringList::new(&word_refs);

        let wake_word_row = adw::ComboRow::new();
        wake_word_row.set_title("Wake word");
        wake_word_row.set_subtitle("Phrase that starts the conversation");
        wake_word_row.set_model(Some(&word_list));
        wake_word_row.set_selected(
            crate::wake::WORDS
                .iter()
                .position(|(word, _)| *word == current_word)
                .unwrap_or(1) as u32,
        );
        wake.add(&wake_word_row);

        let wake_timeout_row = adw::ComboRow::new();
        wake_timeout_row.set_title("Listening timeout");
        wake_timeout_row.set_subtitle("Stop listening after this much silence");
        let timeout_list = gtk::StringList::new(&["15 seconds", "30 seconds", "60 seconds"]);
        wake_timeout_row.set_model(Some(&timeout_list));
        wake_timeout_row.set_selected(
            match self.state.settings.borrow().wake_listen_timeout_secs {
                15 => 0,
                60 => 2,
                _ => 1,
            },
        );
        wake.add(&wake_timeout_row);

        let wake_new_chat_row = adw::SwitchRow::new();
        wake_new_chat_row.set_title("New conversation on wake word");
        wake_new_chat_row
            .set_subtitle("Start a fresh chat only after the previous one ended with a goodbye");
        wake_new_chat_row.set_active(self.state.settings.borrow().wake_new_conversation);
        wake.add(&wake_new_chat_row);

        let wake_threshold_row = adw::SpinRow::new(
            Some(&gtk::Adjustment::new(
                self.state.settings.borrow().wake_threshold as f64,
                0.0,
                0.99,
                0.01,
                0.05,
                0.0,
            )),
            0.01,
            2,
        );
        wake_threshold_row.set_title("Detection threshold");
        wake_threshold_row.set_subtitle("0 uses the value calibrated for the selected word");
        wake.add(&wake_threshold_row);

        let wake_model_row = adw::ActionRow::new();
        wake_model_row.set_title("Wake word model");
        let wake_downloaded = crate::wake::is_downloaded(&current_word);
        wake_model_row.set_subtitle(if wake_downloaded {
            "Downloaded · ~2 MB · Apache-2.0"
        } else {
            "Not downloaded · ~2 MB · Apache-2.0"
        });
        let wake_model_button = gtk::Button::with_label(if wake_downloaded {
            "Downloaded"
        } else {
            "Download"
        });
        wake_model_button.add_css_class("flat");
        wake_model_button.set_valign(gtk::Align::Center);
        wake_model_button.set_sensitive(!wake_downloaded);
        wake_model_row.add_suffix(&wake_model_button);

        let wake_model_row = adw::ActionRow::new();
        wake_model_row.set_title("Wake word model");
        let wake_downloaded = crate::wake::is_downloaded(&current_word);
        wake_model_row.set_subtitle(if wake_downloaded {
            "Downloaded · ~2 MB · Apache-2.0"
        } else {
            "Not downloaded · ~2 MB · Apache-2.0"
        });
        let wake_model_button = gtk::Button::with_label(if wake_downloaded {
            "Downloaded"
        } else {
            "Download"
        });
        wake_model_button.add_css_class("flat");
        wake_model_button.set_valign(gtk::Align::Center);
        wake_model_button.set_sensitive(!wake_downloaded);
        wake_model_row.add_suffix(&wake_model_button);
        wake.add(&wake_model_row);

        page.add(&wake);

        {
            let win = self.clone();
            wake_enable_row.connect_active_notify(move |row| {
                win.state.settings.borrow_mut().wake_enabled = row.is_active();
                win.save_settings();
                win.autostart_wake_pending.set(false);
                if !row.is_active() {
                    win.wake_active.set(false);
                }
                win.stop_wake();
                win.refresh_wake();
                win.update_tray_state();
            });
        }
        {
            let win = self.clone();
            wake_word_row.connect_selected_notify(move |row| {
                let index = row.selected() as usize;
                let Some((word, _)) = crate::wake::WORDS.get(index) else {
                    return;
                };
                if win.state.settings.borrow().wake_word == *word {
                    return;
                }
                win.state.settings.borrow_mut().wake_word = (*word).to_string();
                win.save_settings();
                win.stop_wake();
                win.refresh_wake();
            });
        }
        {
            let win = self.clone();
            wake_timeout_row.connect_selected_notify(move |row| {
                let seconds = match row.selected() {
                    0 => 15,
                    2 => 60,
                    _ => 30,
                };
                win.state.settings.borrow_mut().wake_listen_timeout_secs = seconds;
                win.save_settings();
            });
        }
        {
            let win = self.clone();
            wake_new_chat_row.connect_active_notify(move |row| {
                win.state.settings.borrow_mut().wake_new_conversation = row.is_active();
                win.save_settings();
                win.voice_new_chat_requested.set(false);
                win.last_voice_end_goodbye.set(false);
            });
        }
        {
            let win = self.clone();
            wake_threshold_row.connect_value_notify(move |row| {
                win.state.settings.borrow_mut().wake_threshold = row.value() as f32;
                win.save_settings();
                if win.wake_listening.get() {
                    win.stop_wake();
                    win.refresh_wake();
                }
            });
        }
        {
            let win = self.clone();
            let row = wake_model_row.clone();
            let button = wake_model_button.clone();
            wake_model_button.connect_clicked(move |_| {
                button.set_sensitive(false);
                button.set_label("Downloading…");
                row.set_subtitle("Downloading the wake word model…");
                win.refresh_wake();

                let row = row.clone();
                let button = button.clone();
                let win = win.clone();
                glib::timeout_add_local_once(std::time::Duration::from_secs(2), move || {
                    if crate::wake::is_downloaded(&win.state.settings.borrow().wake_word) {
                        row.set_subtitle("Downloaded · ~2 MB · Apache-2.0");
                        button.set_label("Downloaded");
                    } else {
                        row.set_subtitle("Not downloaded · ~2 MB · Apache-2.0");
                        button.set_label("Download");
                        button.set_sensitive(true);
                    }
                });
            });
        }

        let appearance = adw::PreferencesGroup::new();
        appearance.set_title("Appearance");

        let theme_row = adw::ComboRow::new();
        theme_row.set_title("Theme");
        let themes = gtk::StringList::new(&["Follow system", "Light", "Dark"]);
        theme_row.set_model(Some(&themes));
        theme_row.set_selected(match self.state.settings.borrow().theme.as_str() {
            "light" => 1,
            "dark" => 2,
            _ => 0,
        });
        appearance.add(&theme_row);

        page.add(&appearance);

        {
            let win = self.clone();
            url_row.connect_changed(move |row| {
                win.state.settings.borrow_mut().base_url = row.text().to_string();
                win.save_settings();
                win.state.rebuild_client();
            });
        }
        {
            let win = self.clone();
            key_row.connect_changed(move |row| {
                win.state.settings.borrow_mut().api_key = row.text().to_string();
                win.save_settings();
                win.state.rebuild_client();
            });
        }
        {
            let win = self.clone();
            system_row.connect_changed(move |row| {
                win.state.settings.borrow_mut().system_prompt = row.text().to_string();
                win.save_settings();
            });
        }
        {
            let win = self.clone();
            temperature_row.connect_value_notify(move |row| {
                win.state.settings.borrow_mut().temperature = row.value();
                win.save_settings();
            });
        }
        {
            let win = self.clone();
            context_row.connect_value_notify(move |row| {
                win.state.settings.borrow_mut().num_ctx = row.value() as u32;
                win.save_settings();
                win.refresh_context_indicator();
                win.reload_sidebar();
            });
        }
        {
            let win = self.clone();
            theme_row.connect_selected_notify(move |row| {
                let theme = match row.selected() {
                    1 => "light",
                    2 => "dark",
                    _ => "system",
                };
                win.state.settings.borrow_mut().theme = theme.to_string();
                win.save_settings();
                win.apply_theme();
            });
        }
        {
            let win = self.clone();
            let autoread_row = tts_autoread_row.clone();
            tts_enable_row.connect_active_notify(move |row| {
                win.state.settings.borrow_mut().tts_enabled = row.is_active();
                win.save_settings();
                autoread_row.set_sensitive(row.is_active());
                win.update_speech_buttons();
                if !row.is_active() {
                    if win.live_mode.get() || win.live_pending.get() {
                        win.stop_stt();
                    }
                    win.stop_speech();
                    win.cancel_tts_idle_unload();
                    win.unload_tts_engine(None);
                }
                win.update_stt_ui();
            });
        }
        {
            let win = self.clone();
            tts_autoread_row.connect_active_notify(move |row| {
                win.state.settings.borrow_mut().tts_autoread = row.is_active();
                win.save_settings();
            });
        }
        {
            let win = self.clone();
            let language_row = language_row.clone();
            let choices = Rc::clone(&language_choices);
            let suppress = Rc::clone(&suppress_language);
            let voice_row = voice_row.clone();
            let voice_choices = Rc::clone(&voice_choices);
            let suppress_voice = Rc::clone(&suppress_voice);
            let paradee_button = paradee_button.clone();
            backend_row.connect_selected_notify(move |row| {
                let backend = tts_backend_from_index(row.selected());
                win.state.settings.borrow_mut().tts_backend = backend.to_string();
                win.sync_tts_model_row(backend, &paradee_button);
                let current = win.state.settings.borrow().tts_language.clone();
                let selected =
                    fill_language_row(&language_row, backend, &current, &choices, &suppress);
                let language = tts_language(&selected);
                win.state.settings.borrow_mut().tts_language = selected;

                let current_voice = win.state.settings.borrow().tts_voice.clone();
                let selected_voice = fill_voice_row(
                    &voice_row,
                    backend,
                    language,
                    &current_voice,
                    &voice_choices,
                    &suppress_voice,
                );
                win.state.settings.borrow_mut().tts_voice = selected_voice;

                win.save_settings();
                win.stop_speech();
                win.update_speech_buttons();
            });
        }
        {
            let win = self.clone();
            let choices = Rc::clone(&language_choices);
            let suppress = Rc::clone(&suppress_language);
            let voice_row = voice_row.clone();
            let voice_choices = Rc::clone(&voice_choices);
            let suppress_voice = Rc::clone(&suppress_voice);
            language_row.connect_selected_notify(move |row| {
                if suppress.get() {
                    return;
                }
                let index = row.selected() as usize;
                if let Some(language) = choices.borrow().get(index).copied() {
                    win.state.settings.borrow_mut().tts_language = language.code.to_string();
                    let backend = win.state.settings.borrow().tts_backend.clone();
                    let current_voice = win.state.settings.borrow().tts_voice.clone();
                    let selected_voice = fill_voice_row(
                        &voice_row,
                        &backend,
                        language,
                        &current_voice,
                        &voice_choices,
                        &suppress_voice,
                    );
                    win.state.settings.borrow_mut().tts_voice = selected_voice;
                    win.save_settings();
                    win.stop_speech();
                    win.update_speech_buttons();
                }
            });
        }
        {
            let win = self.clone();
            let choices = Rc::clone(&voice_choices);
            let suppress = Rc::clone(&suppress_voice);
            voice_row.connect_selected_notify(move |row| {
                if suppress.get() {
                    return;
                }
                let index = row.selected() as usize;
                if let Some(voice) = choices.borrow().get(index) {
                    win.state.settings.borrow_mut().tts_voice = voice.id.clone();
                    win.save_settings();
                    win.stop_speech();
                }
            });
        }
        {
            let win = self.clone();
            let row = paradee_row.clone();
            let spinner = paradee_spinner.clone();
            let button = paradee_button.clone();
            let delete = paradee_delete.clone();
            paradee_button.connect_clicked(move |_| {
                if crate::paradee::is_downloaded() {
                    return;
                }
                spinner.set_visible(true);
                spinner.start();
                button.set_sensitive(false);
                delete.set_visible(false);
                row.set_subtitle("Downloading…");
                let (tx, rx) = async_channel::bounded(1);
                win.state.runtime.spawn(async move {
                    let result = crate::paradee::ensure_downloaded().await;
                    let _ = tx.send(result).await;
                });

                let win = win.clone();
                let row = row.clone();
                let spinner = spinner.clone();
                let button = button.clone();
                let delete = delete.clone();
                glib::spawn_future_local(async move {
                    match rx.recv().await {
                        Ok(Ok(())) => {
                            spinner.stop();
                            spinner.set_visible(false);
                            win.update_paradee_model_row(&row, &button, &delete);
                            win.toast("Paradee voice model downloaded");
                        }
                        Ok(Err(err)) => {
                            spinner.stop();
                            spinner.set_visible(false);
                            win.update_paradee_model_row(&row, &button, &delete);
                            win.toast(&format!("Could not download Paradee model: {err:#}"));
                        }
                        Err(_) => {
                            spinner.stop();
                            spinner.set_visible(false);
                        }
                    }
                });
            });
        }
        {
            let win = self.clone();
            let row = paradee_row.clone();
            let button = paradee_button.clone();
            let delete = paradee_delete.clone();
            paradee_delete.connect_clicked(move |_| {
                win.stop_speech();
                let engine = Arc::clone(&win.state.paradee);
                let (tx, rx) = async_channel::bounded(1);
                win.state.runtime.spawn(async move {
                    let result = tokio::task::spawn_blocking(move || {
                        if let Ok(mut guard) = engine.lock() {
                            *guard = None;
                        }
                        crate::paradee::delete()
                    })
                    .await
                    .map_err(|err| anyhow::anyhow!("model cleanup task failed: {err}"))
                    .and_then(|result| result);
                    let _ = tx.send(result).await;
                });

                let win = win.clone();
                let row = row.clone();
                let button = button.clone();
                let delete = delete.clone();
                glib::spawn_future_local(async move {
                    match rx.recv().await {
                        Ok(Ok(())) => {
                            win.update_paradee_model_row(&row, &button, &delete);
                            win.toast("Paradee model deleted");
                        }
                        Ok(Err(err)) => {
                            win.toast(&format!("Could not delete Paradee model: {err:#}"));
                        }
                        Err(_) => {}
                    }
                });
            });
        }

        {
            let win = self.clone();
            stt_enable_row.connect_active_notify(move |row| {
                win.state.settings.borrow_mut().stt_enabled = row.is_active();
                win.save_settings();
                if row.is_active() {
                    win.update_stt_ui();
                } else {
                    win.stop_stt();
                    win.unload_stt_engine(None);
                }
            });
        }
        {
            let win = self.clone();
            let language_row = stt_language_row.clone();
            let choices = Rc::clone(&stt_choices);
            let suppress = Rc::clone(&suppress_stt_language);
            let parakeet_model_row = parakeet_model_row.clone();
            let parakeet_button = parakeet_button.clone();
            let whistle_button = whistle_button.clone();
            stt_provider_row.connect_selected_notify(move |row| {
                let provider = stt_provider_from_index(row.selected());
                win.state.settings.borrow_mut().stt_provider = provider.to_string();
                win.sync_stt_model_rows(
                    provider,
                    &parakeet_model_row,
                    &parakeet_button,
                    &whistle_button,
                );
                let current = win.state.settings.borrow().stt_language.clone();
                let selected =
                    fill_stt_language_row(&language_row, provider, &current, &choices, &suppress);
                win.state.settings.borrow_mut().stt_language = selected;
                win.save_settings();
                win.stop_stt();
            });
        }
        {
            let win = self.clone();
            let choices = Rc::clone(&stt_choices);
            let suppress = Rc::clone(&suppress_stt_language);
            stt_language_row.connect_selected_notify(move |row| {
                if suppress.get() {
                    return;
                }
                let index = row.selected() as usize;
                if let Some(language) = choices.borrow().get(index).copied() {
                    win.state.settings.borrow_mut().stt_language = language.code.to_string();
                    win.save_settings();
                    win.stop_stt();
                }
            });
        }
        {
            let win = self.clone();
            stt_start_row.connect_value_notify(move |row| {
                win.state.settings.borrow_mut().stt_start_timeout_secs =
                    row.value().max(1.0) as u32;
                win.save_settings();
            });
        }
        {
            let win = self.clone();
            stt_end_row.connect_value_notify(move |row| {
                win.state.settings.borrow_mut().stt_end_silence_ms =
                    (row.value() * 1000.0).max(200.0) as u32;
                win.save_settings();
            });
        }
        {
            let win = self.clone();
            stt_autosend_row.connect_active_notify(move |row| {
                win.state.settings.borrow_mut().stt_autosend = row.is_active();
                win.save_settings();
            });
        }
        {
            let win = self.clone();
            let row = parakeet_row.clone();
            let button = parakeet_button.clone();
            let delete = parakeet_delete.clone();
            parakeet_model_row.connect_selected_notify(move |combo| {
                let model = if combo.selected() == 1 {
                    crate::parakeet::ParakeetModel::Ultra
                } else {
                    crate::parakeet::ParakeetModel::Redux
                };
                win.state.settings.borrow_mut().stt_parakeet_model = model.as_str().to_string();
                win.save_settings();
                if win.stt_listening.get() || win.stt_preparing.get() {
                    win.stop_stt();
                }
                win.unload_stt_engine(None);
                win.update_parakeet_model_row(&row, &button, &delete);
            });
        }
        {
            let win = self.clone();
            let row = parakeet_row.clone();
            let spinner = parakeet_spinner.clone();
            let button = parakeet_button.clone();
            let delete = parakeet_delete.clone();
            parakeet_button.connect_clicked(move |_| {
                let model = win.parakeet_model();
                if crate::parakeet::is_downloaded(model) {
                    return;
                }
                spinner.set_visible(true);
                spinner.start();
                button.set_sensitive(false);
                delete.set_visible(false);
                row.set_subtitle(&format!("Downloading {}…", model.display_name()));
                let (tx, rx) = async_channel::bounded(1);
                win.state.runtime.spawn(async move {
                    let result = crate::parakeet::ensure_downloaded(model).await;
                    let _ = tx.send(result).await;
                });

                let win = win.clone();
                let row = row.clone();
                let spinner = spinner.clone();
                let button = button.clone();
                let delete = delete.clone();
                glib::spawn_future_local(async move {
                    match rx.recv().await {
                        Ok(Ok(())) => {
                            spinner.stop();
                            spinner.set_visible(false);
                            win.update_parakeet_model_row(&row, &button, &delete);
                            win.toast(&format!("{} model downloaded", model.display_name()));
                        }
                        Ok(Err(err)) => {
                            spinner.stop();
                            spinner.set_visible(false);
                            win.update_parakeet_model_row(&row, &button, &delete);
                            win.toast(&format!(
                                "Could not download {}: {err:#}",
                                model.display_name()
                            ));
                        }
                        Err(_) => {
                            spinner.stop();
                            spinner.set_visible(false);
                        }
                    }
                });
            });
        }
        {
            let win = self.clone();
            let row = parakeet_row.clone();
            let button = parakeet_button.clone();
            let delete = parakeet_delete.clone();
            parakeet_delete.connect_clicked(move |_| {
                let model = win.parakeet_model();
                if !crate::parakeet::is_downloaded(model) {
                    return;
                }
                if win.stt_listening.get() || win.stt_preparing.get() {
                    win.stop_stt();
                }
                let engine = Arc::clone(&win.state.parakeet);
                let (tx, rx) = async_channel::bounded(1);
                win.state.runtime.spawn(async move {
                    let result = tokio::task::spawn_blocking(move || {
                        if let Ok(mut guard) = engine.lock() {
                            if guard.as_ref().is_some_and(|(loaded, _)| *loaded == model) {
                                *guard = None;
                            }
                        }
                        crate::parakeet::delete(model)
                    })
                    .await
                    .map_err(|err| anyhow::anyhow!("model cleanup task failed: {err}"))
                    .and_then(|result| result);
                    let _ = tx.send(result).await;
                });

                let win = win.clone();
                let row = row.clone();
                let button = button.clone();
                let delete = delete.clone();
                glib::spawn_future_local(async move {
                    match rx.recv().await {
                        Ok(Ok(())) => {
                            win.update_parakeet_model_row(&row, &button, &delete);
                            win.toast(&format!("{} model deleted", model.display_name()));
                        }
                        Ok(Err(err)) => {
                            win.toast(&format!(
                                "Could not delete {}: {err:#}",
                                model.display_name()
                            ));
                        }
                        Err(_) => {}
                    }
                });
            });
        }
        {
            let win = self.clone();
            let row = whistle_row.clone();
            let spinner = whistle_spinner.clone();
            let button = whistle_button.clone();
            let delete = whistle_delete.clone();
            whistle_button.connect_clicked(move |_| {
                if crate::whistle::is_downloaded() {
                    return;
                }
                spinner.set_visible(true);
                spinner.start();
                button.set_sensitive(false);
                delete.set_visible(false);
                row.set_subtitle("Downloading…");
                let (tx, rx) = async_channel::bounded(1);
                win.state.runtime.spawn(async move {
                    let result = crate::whistle::ensure_downloaded().await;
                    let _ = tx.send(result).await;
                });

                let win = win.clone();
                let row = row.clone();
                let spinner = spinner.clone();
                let button = button.clone();
                let delete = delete.clone();
                glib::spawn_future_local(async move {
                    match rx.recv().await {
                        Ok(Ok(())) => {
                            spinner.stop();
                            spinner.set_visible(false);
                            win.update_whistle_model_row(&row, &button, &delete);
                            win.toast("Whistle speech model downloaded");
                        }
                        Ok(Err(err)) => {
                            spinner.stop();
                            spinner.set_visible(false);
                            win.update_whistle_model_row(&row, &button, &delete);
                            win.toast(&format!("Could not download Whistle model: {err:#}"));
                        }
                        Err(_) => {
                            spinner.stop();
                            spinner.set_visible(false);
                        }
                    }
                });
            });
        }
        {
            let win = self.clone();
            let row = whistle_row.clone();
            let button = whistle_button.clone();
            let delete = whistle_delete.clone();
            whistle_delete.connect_clicked(move |_| {
                if !crate::whistle::is_downloaded() {
                    return;
                }
                if win.stt_listening.get() || win.stt_preparing.get() {
                    win.stop_stt();
                }
                let engine = Arc::clone(&win.state.whistle);
                let (tx, rx) = async_channel::bounded(1);
                win.state.runtime.spawn(async move {
                    let result = tokio::task::spawn_blocking(move || {
                        if let Ok(mut guard) = engine.lock() {
                            *guard = None;
                        }
                        crate::whistle::delete()
                    })
                    .await
                    .map_err(|err| anyhow::anyhow!("model cleanup task failed: {err}"))
                    .and_then(|result| result);
                    let _ = tx.send(result).await;
                });

                let win = win.clone();
                let row = row.clone();
                let button = button.clone();
                let delete = delete.clone();
                glib::spawn_future_local(async move {
                    match rx.recv().await {
                        Ok(Ok(())) => {
                            win.update_whistle_model_row(&row, &button, &delete);
                            win.toast("Whistle model deleted");
                        }
                        Ok(Err(err)) => {
                            win.toast(&format!("Could not delete Whistle model: {err:#}"));
                        }
                        Err(_) => {}
                    }
                });
            });
        }

        prefs_dialog.add(&page);
        prefs_dialog.present(Some(&self.window));
    }

    fn show_about(&self) {
        let about = adw::AboutDialog::builder()
            .application_name("Ollama GUI")
            .application_icon("io.github.ltux.OllamaGui")
            .developer_name("ltux")
            .version(env!("CARGO_PKG_VERSION"))
            .comments("A native GTK4/libadwaita client for Ollama.")
            .build();
        about.present(Some(&self.window));
    }

    fn toast(&self, message: &str) {
        self.toast_overlay.add_toast(adw::Toast::new(message));
    }
}

fn trim_heap() {
    #[cfg(target_os = "linux")]
    unsafe {
        libc::malloc_trim(0);
    }
}

/// True when the clipboard advertises image data we can decode.
fn formats_have_image(formats: &gdk::ContentFormats) -> bool {
    formats.contains_type(gdk::Texture::static_type())
        || [
            "image/png",
            "image/jpeg",
            "image/webp",
            "image/gif",
            "image/bmp",
            "image/tiff",
        ]
        .iter()
        .any(|mime| formats.contain_mime_type(mime))
}

/// Filter file-manager clipboard entries down to images before reading them.
fn file_is_image(file: &gio::File) -> bool {
    if let Ok(info) = file.query_info(
        "standard::content-type",
        gio::FileQueryInfoFlags::NONE,
        None::<&gio::Cancellable>,
    ) {
        if info
            .content_type()
            .is_some_and(|content_type| content_type.starts_with("image/"))
        {
            return true;
        }
    }

    let extension = file
        .basename()
        .and_then(|name| {
            std::path::Path::new(&name)
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext.to_ascii_lowercase())
        })
        .unwrap_or_default();
    matches!(
        extension.as_str(),
        "png" | "jpg" | "jpeg" | "webp" | "gif" | "bmp" | "tif" | "tiff"
    )
}

fn title_from(text: &str) -> String {
    let first_line = text.lines().next().unwrap_or("New Chat").trim();
    let mut title: String = first_line.chars().take(48).collect();
    if first_line.chars().count() > 48 {
        title.push('…');
    }
    if title.is_empty() {
        "New Chat".to_string()
    } else {
        title
    }
}

/// A wake session opens a new chat only when the setting is on and the last
/// live session ended with a closing phrase, not a timeout.
fn should_start_new_voice_chat(
    wake_started: bool,
    setting_enabled: bool,
    last_end_goodbye: bool,
) -> bool {
    wake_started && setting_enabled && last_end_goodbye
}

const SLEEP_PHRASES: &[&str] = &[
    "goodbye",
    "good bye",
    "bye",
    "bye bye",
    "see you",
    "see ya",
    "see you later",
    "talk to you later",
    "talk to you tomorrow",
    "thats all",
    "im done",
    "i am done",
    "stop listening",
    "end conversation",
    "go to sleep",
    "sleep",
    "good night",
    "goodnight",
];

const SLEEP_TAILS: &[&str] = &[
    "thanks",
    "thank you",
    "for now",
    "then",
    "everyone",
    "all",
    "now",
    "please",
    "have a nice day",
    "have a good day",
    "have a good night",
    "and thanks",
    "and thank you",
    "thanks for everything",
    "thanks for your help",
    "thank you for everything",
    "thank you for your help",
    "see you",
    "see you later",
    "see you tomorrow",
    "talk to you later",
    "tomorrow",
    "soon",
    "later",
];

/// Leading words that speech recognition often adds before a command.
const SLEEP_FILLERS: &[&str] = &[
    "all right",
    "alright",
    "okay",
    "ok",
    "well",
    "then",
    "hey",
    "um",
    "uh",
    "yeah",
];

fn normalize_sleep_text(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut pending_space = false;
    for ch in text.chars().flat_map(char::to_lowercase) {
        if ch == '\'' || ch == '’' {
            // "that's" -> "thats", "I'm" -> "im".
            continue;
        }
        if ch.is_alphanumeric() {
            if pending_space && !out.is_empty() {
                out.push(' ');
            }
            out.push(ch);
            pending_space = false;
        } else {
            pending_space = true;
        }
    }
    out
}

fn strip_sleep_fillers(mut text: &str) -> &str {
    loop {
        let mut changed = false;
        for filler in SLEEP_FILLERS {
            if let Some(rest) = text
                .strip_prefix(filler)
                .and_then(|rest| rest.strip_prefix(' '))
            {
                text = rest;
                changed = true;
                break;
            }
        }
        if !changed {
            return text;
        }
    }
}

/// Match phrases that end a live conversation. Only a short trailing courtesy
/// is allowed, so "goodbye, what's the weather?" is still a real question.
fn is_sleep_command(text: &str) -> bool {
    let normalized = normalize_sleep_text(text);
    let normalized = strip_sleep_fillers(&normalized);
    if normalized.is_empty() {
        return false;
    }
    SLEEP_PHRASES.iter().any(|phrase| {
        if normalized == *phrase {
            return true;
        }
        normalized
            .strip_prefix(phrase)
            .and_then(|rest| rest.strip_prefix(' '))
            .is_some_and(|rest| SLEEP_TAILS.contains(&rest))
    })
}

/// Size a message image to a 4:3 card that matches its bubble width.
fn fit_message_picture(picture: &gtk::Picture, column: i32) {
    let width = if picture.width() > 0 {
        picture.width()
    } else {
        (column / 2 - 28).max(80)
    };
    let height = ((width as f64) * 0.75).round() as i32;
    picture.set_size_request(-1, height.max(60));
}

fn group_digits(value: u32) -> String {
    let text = value.to_string();
    let mut out = String::with_capacity(text.len() + text.len() / 3);
    for (index, ch) in text.chars().enumerate() {
        if index > 0 && (text.len() - index).is_multiple_of(3) {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

fn title_case(value: &str) -> String {
    let mut chars = value.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

fn format_percent(percent: f64) -> String {
    if percent > 0.0 && percent < 10.0 {
        format!("{percent:.1}%")
    } else {
        format!("{percent:.0}%")
    }
}

fn apply_usage_class(label: &gtk::Label, percent: f64) {
    label.remove_css_class("dim-label");
    label.remove_css_class("context-warning");
    label.remove_css_class("context-high");
    if percent >= 90.0 {
        label.add_css_class("context-high");
    } else if percent >= 75.0 {
        label.add_css_class("context-warning");
    } else {
        label.add_css_class("dim-label");
    }
}

fn message_action_button(icon: &str, tooltip: &str) -> gtk::Button {
    let button = gtk::Button::from_icon_name(icon);
    button.add_css_class("flat");
    button.add_css_class("message-action");
    button.set_tooltip_text(Some(tooltip));
    button
}

const SENTENCE_ABBREVIATIONS: &[&str] = &[
    "mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "vs", "etc", "e.g", "i.e", "u.s", "u.k",
    "ph.d", "a.m", "p.m",
];

/// Punctuation that adds a pause when spoken. Everything else that is not a
/// letter, digit, or whitespace is dropped from speech input.
const SPEECH_PUNCTUATION: &[char] = &[
    '.', ',', '!', '?', ';', ':', '\'', '"', '-', '–', '—', '…', '‘', '’', '“', '”', '。', '，',
    '！', '？', '；', '：', '、',
];

/// Convert markdown-rich assistant output into plain text for speech engines.
///
/// Fenced code blocks are skipped, HTML tags and link targets are removed, and
/// symbols/emoji that TTS engines would read as noise are dropped. Letters and
/// digits from every language are kept, as are common sentence punctuation and
/// apostrophes.
fn plain_text_for_speech(text: &str) -> String {
    let stripped = strip_fenced_code_blocks(text);
    let mut lines = Vec::new();

    for line in stripped.lines() {
        let line = strip_html_tags(line);
        let line = replace_markdown_links(&line);
        let line = strip_line_prefixes(&line);
        let line = strip_bare_urls(&line);
        let line = speech_safe_characters(&line);
        let line = line.split_whitespace().collect::<Vec<_>>().join(" ");
        if line.chars().any(|character| character.is_alphanumeric()) {
            lines.push(line);
        }
    }

    lines.join("\n")
}

fn strip_fenced_code_blocks(text: &str) -> String {
    let mut result = String::new();
    let mut fence: Option<&str> = None;

    for line in text.lines() {
        let trimmed = line.trim_start();
        if let Some(marker) = fence {
            if trimmed.starts_with(marker) {
                fence = None;
            }
            continue;
        }
        if trimmed.starts_with("```") {
            fence = Some("```");
            continue;
        }
        if trimmed.starts_with("~~~") {
            fence = Some("~~~");
            continue;
        }
        result.push_str(line);
        result.push('\n');
    }

    result
}

fn strip_html_tags(line: &str) -> String {
    let chars: Vec<char> = line.chars().collect();
    let mut result = String::new();
    let mut index = 0;

    while index < chars.len() {
        if chars[index] == '<' {
            if let Some(close) = chars[index + 1..].iter().position(|c| *c == '>') {
                let tag: String = chars[index + 1..index + 1 + close].iter().collect();
                let name = tag
                    .trim_start_matches('/')
                    .split_whitespace()
                    .next()
                    .unwrap_or("");
                if name.eq_ignore_ascii_case("br") {
                    result.push('\n');
                }
                index += close + 2;
                continue;
            }
        }
        result.push(chars[index]);
        index += 1;
    }

    decode_html_entities(&result)
}

fn decode_html_entities(text: &str) -> String {
    text.replace("&amp;", " and ")
        .replace("&lt;", " ")
        .replace("&gt;", " ")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
        .replace("&nbsp;", " ")
}

fn replace_markdown_links(line: &str) -> String {
    let chars: Vec<char> = line.chars().collect();
    let mut result = String::new();
    let mut index = 0;

    while index < chars.len() {
        let image = chars[index] == '!' && chars.get(index + 1) == Some(&'[');
        let open = if image {
            index + 1
        } else if chars[index] == '[' {
            index
        } else {
            usize::MAX
        };

        if open != usize::MAX {
            if let Some(close) = chars[open + 1..].iter().position(|c| *c == ']') {
                let close = open + 1 + close;
                let mut end = close + 1;
                let mut has_target = false;
                if chars.get(end) == Some(&'(') {
                    if let Some(paren) = chars[end + 1..].iter().position(|c| *c == ')') {
                        end = end + 1 + paren + 1;
                        has_target = true;
                    }
                } else if chars.get(end) == Some(&'[') {
                    if let Some(bracket) = chars[end + 1..].iter().position(|c| *c == ']') {
                        end = end + 1 + bracket + 1;
                        has_target = true;
                    }
                }
                if has_target {
                    result.extend(&chars[open + 1..close]);
                    index = end;
                    continue;
                }
            }
        }

        result.push(chars[index]);
        index += 1;
    }

    result
}

fn strip_line_prefixes(line: &str) -> String {
    let mut text = line.trim_start();
    while let Some(rest) = text.strip_prefix('>') {
        text = rest.trim_start();
    }
    text = text.trim_start_matches('#').trim_start();
    for marker in ["- ", "* ", "+ "] {
        if let Some(rest) = text.strip_prefix(marker) {
            text = rest;
            break;
        }
    }
    for marker in ["[ ] ", "[x] ", "[X] "] {
        if let Some(rest) = text.strip_prefix(marker) {
            text = rest;
            break;
        }
    }
    text.to_string()
}

fn strip_bare_urls(line: &str) -> String {
    line.split_whitespace()
        .filter(|token| {
            let trimmed =
                token.trim_matches(|character: char| matches!(character, '(' | '[' | '<' | '"'));
            let lower = trimmed.to_ascii_lowercase();
            !(lower.starts_with("http://")
                || lower.starts_with("https://")
                || lower.starts_with("www."))
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn speech_safe_characters(line: &str) -> String {
    line.chars()
        .filter(|character| {
            character.is_alphanumeric()
                || character.is_whitespace()
                || SPEECH_PUNCTUATION.contains(character)
                || is_combining_mark(*character)
        })
        .collect()
}

fn is_combining_mark(character: char) -> bool {
    matches!(
        character as u32,
        0x0300..=0x036F
            | 0x1AB0..=0x1AFF
            | 0x1DC0..=0x1DFF
            | 0x20D0..=0x20FF
            | 0xFE20..=0xFE2F
    )
}

fn take_complete_sentences(text: &str) -> (Vec<String>, String) {
    let mut sentences = split_sentences(text);
    if sentences.is_empty() {
        return (sentences, String::new());
    }

    if text.ends_with('\n')
        || sentences
            .last()
            .is_some_and(|sentence| ends_sentence(sentence))
    {
        (sentences, String::new())
    } else {
        let tail = sentences.pop().unwrap_or_default();
        (sentences, tail)
    }
}

fn ends_sentence(text: &str) -> bool {
    let trimmed = text.trim_end_matches(&['"', '\'', ')', ']', '}', '»', '”'][..]);
    let Some(last) = trimmed.chars().last() else {
        return false;
    };
    if !matches!(last, '.' | '!' | '?' | '…' | '。' | '！' | '？') {
        return false;
    }
    if last != '.' {
        return true;
    }

    let token = trimmed
        .split_whitespace()
        .last()
        .unwrap_or("")
        .trim_matches(|character: char| !character.is_ascii_alphanumeric() && character != '.')
        .to_ascii_lowercase();
    let token = token.trim_end_matches('.');
    if SENTENCE_ABBREVIATIONS.contains(&token) {
        return false;
    }
    if token.chars().count() == 1
        && token
            .chars()
            .all(|character| character.is_ascii_alphabetic())
    {
        return false;
    }
    true
}

fn split_sentences(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut sentences = Vec::new();
    let mut start = 0usize;
    let mut index = 0usize;

    while index < chars.len() {
        let character = chars[index];
        let boundary = match character {
            '!' | '?' | '\n' | '。' | '！' | '？' => true,
            '.' => {
                let previous = index.checked_sub(1).map(|position| chars[position]);
                let next = chars.get(index + 1).copied();
                let ellipsis = previous == Some('.') || next == Some('.');
                let decimal = previous.is_some_and(|c| c.is_ascii_digit())
                    && next.is_some_and(|c| c.is_ascii_digit());
                let initial = previous.is_some_and(|c| c.is_ascii_uppercase())
                    && (index < 2
                        || chars[index - 2].is_whitespace()
                        || matches!(chars[index - 2], '"' | '\'' | '(' | '['));
                let after = chars[index + 1..]
                    .iter()
                    .find(|c| !c.is_whitespace())
                    .copied();
                let continues = after.is_some_and(|c| c.is_lowercase() || c.is_ascii_digit());
                let token = chars[start..index]
                    .iter()
                    .collect::<String>()
                    .split_whitespace()
                    .last()
                    .unwrap_or("")
                    .trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '.')
                    .to_ascii_lowercase();
                let abbreviation = SENTENCE_ABBREVIATIONS.contains(&token.as_str());
                !ellipsis && !decimal && !initial && !continues && !abbreviation
            }
            _ => false,
        };

        if boundary {
            let mut end = index;
            while let Some(next) = chars.get(end + 1) {
                if matches!(next, '"' | '\'' | ')' | ']' | '}' | '»' | '”') {
                    end += 1;
                } else {
                    break;
                }
            }
            let sentence: String = chars[start..=end].iter().collect();
            let sentence = sentence.trim();
            if !sentence.is_empty() {
                sentences.push(sentence.to_string());
            }
            start = end + 1;
            index = end + 1;
            continue;
        }

        index += 1;
    }

    if start < chars.len() {
        let tail: String = chars[start..].iter().collect();
        let tail = tail.trim();
        if !tail.is_empty() {
            sentences.push(tail.to_string());
        }
    }

    if sentences.is_empty() {
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            sentences.push(trimmed.to_string());
        }
    }

    sentences
}

/// Split text into sentences, limiting the total to Ollama's 4,096-character
/// speech input budget. Returns the chunks and whether text was dropped.
fn sentences_for_speech(text: &str, max_chars: usize) -> (Vec<String>, bool) {
    let text = plain_text_for_speech(text);
    let sentences = split_sentences(&text);
    let mut result = Vec::new();
    let mut used = 0usize;

    for sentence in sentences {
        let length = sentence.chars().count();
        if used + length <= max_chars {
            used += length;
            result.push(sentence);
            continue;
        }

        if result.is_empty() {
            let cut: String = sentence.chars().take(max_chars).collect();
            let cut = cut.trim_end();
            if !cut.is_empty() {
                result.push(cut.to_string());
            }
        }
        return (result, true);
    }

    (result, false)
}

/// Rough token estimate: ASCII text is about four characters per token, while
/// non-ASCII characters are counted as one token each.
fn estimate_tokens(text: &str) -> u32 {
    let mut tokens = 0.0f64;
    for character in text.chars() {
        tokens += if character.is_ascii() { 0.25 } else { 1.0 };
    }
    tokens.ceil() as u32
}

fn context_needs_compression(used: u32, num_ctx: u32) -> bool {
    used as f64 >= num_ctx as f64 * CONTEXT_COMPRESS_RATIO
}

fn estimate_request_tokens(
    system_prompt: &str,
    memory_context: Option<&str>,
    summary: Option<&StoredSummary>,
    messages: &[&StoredMessage],
) -> u32 {
    let mut total = estimate_tokens(system_prompt);
    if let Some(memory_context) = memory_context {
        total = total
            .saturating_add(estimate_tokens(memory_context))
            .saturating_add(4);
    }
    if let Some(summary) = summary {
        total = total
            .saturating_add(estimate_tokens(&summary.text))
            .saturating_add(8);
    }
    total.saturating_add(
        messages
            .iter()
            .map(|message| estimate_tokens(&message.content).saturating_add(4))
            .sum::<u32>(),
    )
}

/// Index where the last `keep_turns` user turns begin. Returns zero when there
/// are fewer turns than requested, meaning nothing is old enough to summarize.
fn kept_turn_start(messages: &[StoredMessage], keep_turns: usize) -> usize {
    if keep_turns == 0 {
        return messages.len();
    }

    let mut seen = 0usize;
    for (index, message) in messages.iter().enumerate().rev() {
        if message.role == "user" {
            seen += 1;
            if seen == keep_turns {
                return index;
            }
        }
    }
    0
}

fn chunk_messages(messages: &[&StoredMessage], max_tokens: u32) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();
    let mut used = 0u32;

    for message in messages {
        let role = match message.role.as_str() {
            "user" => "User",
            "assistant" => "Assistant",
            other => other,
        };
        let line = format!("{role}: {}\n", message.content.trim());
        let tokens = estimate_tokens(&line);
        if !current.is_empty() && used.saturating_add(tokens) > max_tokens {
            chunks.push(std::mem::take(&mut current));
            used = 0;
        }
        current.push_str(&line);
        used = used.saturating_add(tokens);
    }

    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

fn summary_budget(num_ctx: u32) -> u32 {
    ((num_ctx as f64 * CONTEXT_SUMMARY_RATIO) as u32).max(256)
}

fn summary_chunk_budget(num_ctx: u32, summary_budget: u32) -> u32 {
    let chunk = (num_ctx as f64 * CONTEXT_CHUNK_RATIO) as u32;
    let available = num_ctx
        .saturating_sub(summary_budget)
        .saturating_sub(128)
        .max(128);
    chunk.saturating_sub(256).max(128).min(available)
}

fn build_request_messages(
    system_prompt: &str,
    memory_context: Option<&str>,
    summary: Option<&StoredSummary>,
    history: &[StoredMessage],
    images: &HashMap<i64, Vec<StoredImage>>,
    num_ctx: u32,
) -> Vec<ChatMessage> {
    let kept: Vec<&StoredMessage> = match summary {
        Some(summary) => history
            .iter()
            .filter(|message| message.id > summary.upto_id)
            .collect(),
        None => history.iter().collect(),
    };

    // If the kept turns alone are too large, drop the oldest ones for this
    // request without touching the stored history.
    let budget = (num_ctx as f64 * CONTEXT_COMPRESS_RATIO) as u32;
    let min_start = kept
        .iter()
        .rposition(|message| message.role == "user")
        .unwrap_or_else(|| kept.len().saturating_sub(1));
    let mut start = 0;
    while start < min_start
        && estimate_request_tokens(system_prompt, memory_context, summary, &kept[start..]) > budget
    {
        start += 1;
        while start < min_start && kept[start].role != "user" {
            start += 1;
        }
    }
    let kept = &kept[start..];

    let mut messages = Vec::new();
    if !system_prompt.trim().is_empty() {
        messages.push(ChatMessage::new("system", system_prompt));
    }
    if let Some(memory_context) = memory_context.filter(|context| !context.trim().is_empty()) {
        messages.push(ChatMessage::new("system", memory_context));
    }
    if let Some(summary) = summary {
        messages.push(ChatMessage::new(
            "system",
            format!("Summary of earlier conversation:\n{}", summary.text),
        ));
    }
    messages.extend(kept.iter().map(|message| {
        let mut chat = ChatMessage::new(&message.role, message.content.clone());
        if let Some(message_images) = images.get(&message.id) {
            chat.images = message_images
                .iter()
                .map(|image| base64::engine::general_purpose::STANDARD.encode(&image.data))
                .collect();
        }
        chat
    }));
    messages
}

fn build_compression_plan(
    model: String,
    num_ctx: u32,
    system_prompt: &str,
    memory_context: Option<&str>,
    messages: &[StoredMessage],
    summary: Option<&StoredSummary>,
    used: Option<u32>,
) -> Option<CompressionPlan> {
    let cutoff = kept_turn_start(messages, CONTEXT_KEEP_TURNS);
    if cutoff == 0 {
        return None;
    }

    let watermark = summary.map(|summary| summary.upto_id).unwrap_or(0);
    let older: Vec<&StoredMessage> = messages[..cutoff]
        .iter()
        .filter(|message| message.id > watermark)
        .collect();
    if older.is_empty() {
        return None;
    }

    let relevant: Vec<&StoredMessage> = match summary {
        Some(summary) => messages
            .iter()
            .filter(|message| message.id > summary.upto_id)
            .collect(),
        None => messages.iter().collect(),
    };
    let estimated = estimate_request_tokens(system_prompt, memory_context, summary, &relevant);
    let used = used.map_or(estimated, |used| used.max(estimated));
    if !context_needs_compression(used, num_ctx) {
        return None;
    }

    let budget = summary_budget(num_ctx);
    let chunks = chunk_messages(&older, summary_chunk_budget(num_ctx, budget));
    if chunks.is_empty() {
        return None;
    }

    Some(CompressionPlan {
        model,
        num_ctx,
        existing: summary.map(|summary| summary.text.clone()),
        upto_id: older.last().map(|message| message.id)?,
        chunks,
    })
}

async fn run_memory_job(client: OllamaClient, job: MemoryJob) -> anyhow::Result<MemoryJobResult> {
    // Memory extraction wants the notes, not the model's reasoning. Ask
    // thinking models for the cheapest mode; leave non-thinking models alone.
    let think = match client.model_thinking(&job.model).await {
        Ok(Some(control)) if control.is_level_only() => control
            .lowest_level()
            .map(|level| ThinkSetting::Level(level.to_string())),
        Ok(Some(_)) => Some(ThinkSetting::Bool(false)),
        _ => None,
    };

    let request = ChatRequest {
        model: job.model.clone(),
        messages: vec![
            ChatMessage::new(
                "system",
                memory::summary_prompt(job.max_chars, &job.created_date),
            ),
            ChatMessage::new("user", format!("Transcript:\n{}", job.transcript)),
        ],
        stream: false,
        options: Some(ChatOptions {
            temperature: Some(MEMORY_TEMPERATURE),
            num_ctx: Some(job.num_ctx),
            // Keep the output inside the context window on small `num_ctx`
            // settings while allowing the configured note length otherwise.
            num_predict: Some(
                ((job.max_chars / 3).max(64) as u32).min((job.num_ctx / 4).max(64)) as i64,
            ),
            ..Default::default()
        }),
        keep_alive: Some(CHAT_KEEP_ALIVE.to_string()),
        think,
    };

    let reply = client.chat_once(&request).await?;
    let text = memory::sanitize_note(reply.content.trim(), job.max_chars);
    let mut result = job.empty_result();
    if text.chars().count() < memory::MEMORY_MIN_SUMMARY_CHARS {
        return Ok(result);
    }

    // Encode the capture date into the vector so date-flavoured queries can
    // match even when the note itself does not repeat the date.
    let embedding_text = format!("[{}] {text}", job.created_date);
    let mut vectors = client
        .embed(&job.embedding_model, std::slice::from_ref(&embedding_text))
        .await?;
    let vector = vectors.pop().filter(|vector| !vector.is_empty());
    if vector.is_none() {
        anyhow::bail!("embedding response was empty");
    }

    result.text = Some(text);
    result.vector = vector;
    Ok(result)
}

async fn run_compression(
    client: OllamaClient,
    plan: CompressionPlan,
) -> anyhow::Result<CompressionResult> {
    let budget = summary_budget(plan.num_ctx);
    let mut summary = plan.existing.unwrap_or_default();

    for chunk in &plan.chunks {
        let mut content = String::new();
        if !summary.trim().is_empty() {
            content.push_str("Existing summary:\n");
            content.push_str(summary.trim());
            content.push_str("\n\n");
        }
        content.push_str("Older messages to merge:\n");
        content.push_str(chunk);

        let request = ChatRequest {
            model: plan.model.clone(),
            messages: vec![
                ChatMessage::new("system", SUMMARY_SYSTEM_PROMPT),
                ChatMessage::new("user", content),
            ],
            stream: false,
            options: Some(ChatOptions {
                temperature: Some(SUMMARY_TEMPERATURE),
                num_ctx: Some(plan.num_ctx),
                num_predict: Some(budget as i64),
                ..Default::default()
            }),
            keep_alive: Some(CHAT_KEEP_ALIVE.to_string()),
            think: None,
        };

        let reply = client.chat_once(&request).await?;
        tracing::debug!(
            "context: summary chunk used {:?} prompt tokens and {:?} output tokens",
            reply.stats.prompt_eval_count,
            reply.stats.eval_count
        );
        let text = reply.content.trim();
        if text.is_empty() {
            anyhow::bail!("summarization returned an empty response");
        }
        summary = clamp_summary(text, budget);
    }

    if summary.trim().is_empty() {
        anyhow::bail!("summary is empty");
    }
    Ok(CompressionResult {
        text: summary.trim().to_string(),
        tokens: estimate_tokens(&summary),
        upto_id: plan.upto_id,
    })
}

fn clamp_summary(text: &str, max_tokens: u32) -> String {
    let mut used = 0.0f64;
    let mut end = 0usize;
    for (index, character) in text.char_indices() {
        used += if character.is_ascii() { 0.25 } else { 1.0 };
        if used > max_tokens as f64 {
            break;
        }
        end = index + character.len_utf8();
    }
    if end == 0 {
        return String::new();
    }

    let mut clamped = text[..end].trim_end().to_string();
    if end < text.len() {
        if let Some(position) = clamped.rfind(['.', '!', '?', '\n']) {
            clamped.truncate(position + 1);
        }
    }
    clamped
}

fn show_summary_dialog(parent: Option<&gtk::Window>, text: &str) {
    let dialog = gtk::Window::builder()
        .title("Conversation summary")
        .modal(true)
        .default_width(560)
        .default_height(420)
        .build();
    let header = gtk::HeaderBar::new();
    dialog.set_titlebar(Some(&header));
    if let Some(parent) = parent {
        dialog.set_transient_for(Some(parent));
    }

    let scrolled = gtk::ScrolledWindow::new();
    scrolled.set_policy(gtk::PolicyType::Never, gtk::PolicyType::Automatic);
    scrolled.set_hexpand(true);
    scrolled.set_vexpand(true);

    let label = gtk::Label::new(Some(text));
    label.set_wrap(true);
    label.set_xalign(0.0);
    label.set_selectable(true);
    label.set_margin_top(16);
    label.set_margin_bottom(16);
    label.set_margin_start(16);
    label.set_margin_end(16);
    scrolled.set_child(Some(&label));

    dialog.set_child(Some(&scrolled));
    dialog.present();
}

fn spawn_playback_thread(
    generation: Arc<AtomicU64>,
) -> anyhow::Result<async_channel::Sender<PlaybackChunk>> {
    let (tx, rx) = async_channel::bounded::<PlaybackChunk>(4);
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<(), String>>();

    std::thread::Builder::new()
        .name("ollama-gui-tts".to_string())
        .spawn(move || {
            let stream = match rodio::OutputStreamBuilder::open_default_stream() {
                Ok(stream) => {
                    let _ = ready_tx.send(Ok(()));
                    stream
                }
                Err(err) => {
                    let _ = ready_tx.send(Err(err.to_string()));
                    return;
                }
            };

            while let Ok(chunk) = rx.recv_blocking() {
                if chunk.generation != generation.load(Ordering::SeqCst) {
                    continue;
                }

                let sink = rodio::Sink::connect_new(stream.mixer());
                tracing::debug!(
                    "speech playback: {} samples at {} Hz, {} channel(s)",
                    chunk.samples.len(),
                    chunk.sample_rate,
                    chunk.channels
                );
                sink.append(rodio::buffer::SamplesBuffer::new(
                    chunk.channels,
                    chunk.sample_rate,
                    chunk.samples,
                ));

                while !sink.empty() {
                    if generation.load(Ordering::SeqCst) != chunk.generation {
                        sink.stop();
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(20));
                }
            }
        })?;

    match ready_rx.recv_timeout(std::time::Duration::from_secs(2)) {
        Ok(Ok(())) => Ok(tx),
        Ok(Err(err)) => anyhow::bail!("{err}"),
        Err(_) => anyhow::bail!("audio output did not initialize"),
    }
}

/// Decode the 16-bit PCM WAV returned by `/v1/audio/speech` into rodio samples.
fn decode_wav(bytes: &[u8]) -> anyhow::Result<(Vec<f32>, u32, u16)> {
    anyhow::ensure!(
        bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE",
        "speech endpoint did not return WAV audio"
    );

    let mut position = 12usize;
    let mut channels = 1u16;
    let mut sample_rate = 24_000u32;
    let mut bits = 16u16;
    let mut data: Option<&[u8]> = None;

    while position + 8 <= bytes.len() {
        let id = &bytes[position..position + 4];
        let size = u32::from_le_bytes(
            bytes[position + 4..position + 8]
                .try_into()
                .expect("chunk size"),
        ) as usize;
        let body_start = position + 8;
        let body_end = body_start.saturating_add(size).min(bytes.len());
        let body = &bytes[body_start..body_end];

        match id {
            b"fmt " if body.len() >= 16 => {
                channels = u16::from_le_bytes([body[0], body[1]]);
                sample_rate = u32::from_le_bytes([body[4], body[5], body[6], body[7]]);
                bits = u16::from_le_bytes([body[14], body[15]]);
            }
            b"data" => data = Some(body),
            _ => {}
        }

        position = body_start.saturating_add(size).saturating_add(size & 1);
    }

    let data = data.ok_or_else(|| anyhow::anyhow!("WAV file has no data chunk"))?;
    anyhow::ensure!(bits == 16, "unsupported WAV sample size: {bits} bits");
    anyhow::ensure!(channels > 0, "unsupported WAV channel count");

    let samples: Vec<f32> = data
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| i16::from_le_bytes(*pair) as f32 / 32_768.0)
        .collect();
    anyhow::ensure!(!samples.is_empty(), "WAV data chunk is empty");

    Ok((samples, sample_rate, channels))
}

fn capability_css_class(capability: &str) -> Option<&'static str> {
    match capability {
        "vision" => Some("cap-vision"),
        "tools" => Some("cap-tools"),
        "thinking" => Some("cap-thinking"),
        "audio" | "speech" => Some("cap-audio"),
        _ => None,
    }
}

fn capability_description(capability: &str) -> Option<&'static str> {
    match capability {
        "completion" => Some("Generates text completions"),
        "chat" => Some("Chat-capable model"),
        "vision" => Some("Can process images"),
        "tools" => Some("Supports tool calling"),
        "thinking" => Some("Emits reasoning/thinking output"),
        "audio" => Some("Can process audio input"),
        "speech" => Some("Can generate speech"),
        "embedding" => Some("Produces embeddings"),
        "decision" => Some("Decision/classification model"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        build_compression_plan, build_request_messages, chunk_messages, clamp_summary,
        context_needs_compression, decode_wav, estimate_tokens, file_is_image, format_percent,
        formats_have_image, group_digits, is_sleep_command, kept_turn_start, plain_text_for_speech,
        sentences_for_speech, should_start_new_voice_chat, split_sentences,
        take_complete_sentences, tts_language, voice_options_for_backend, SpeechTextBuffer,
    };
    use crate::store::{StoredImage, StoredMessage, StoredSummary};
    use gtk::{gdk, gio};
    use std::collections::HashMap;

    #[test]
    fn groups_thousands() {
        assert_eq!(group_digits(0), "0");
        assert_eq!(group_digits(999), "999");
        assert_eq!(group_digits(8_192), "8,192");
        assert_eq!(group_digits(1_234_567), "1,234,567");
    }

    #[test]
    fn recognizes_sleep_commands() {
        for text in [
            "Goodbye.",
            "Good bye!",
            "Okay, goodbye.",
            "All right, bye.",
            "Bye bye",
            "See you later",
            "See you tomorrow",
            "Talk to you later",
            "That's all for now.",
            "I'm done.",
            "Stop listening, please.",
            "Good night",
            "goodnight",
            "Go to sleep",
            "goodbye thanks",
            "okay bye",
            "okay then, goodbye",
            "goodbye, thanks for everything",
            "goodbye, have a good night",
            "goodbye have a nice day",
        ] {
            assert!(is_sleep_command(text), "did not recognize {text:?}");
        }
    }

    #[test]
    fn new_voice_chat_requires_a_goodbye_not_a_timeout() {
        assert!(should_start_new_voice_chat(true, true, true));
        assert!(!should_start_new_voice_chat(true, true, false));
        assert!(!should_start_new_voice_chat(true, false, true));
        assert!(!should_start_new_voice_chat(false, true, true));
    }

    #[test]
    fn ignores_sleep_words_inside_questions() {
        for text in [
            "I don't want to say goodbye.",
            "Goodbye, what's the weather?",
            "By the way, what time is it?",
            "See you at the meeting",
            "Don't stop listening to music.",
            "I am not done yet.",
            "Goodbye is a word I like",
        ] {
            assert!(!is_sleep_command(text), "falsely recognized {text:?}");
        }
    }

    #[test]
    fn detects_image_clipboard_formats() {
        assert!(formats_have_image(&gdk::ContentFormats::new(&[
            "image/png"
        ])));
        assert!(formats_have_image(&gdk::ContentFormats::new(&[
            "image/jpeg"
        ])));
        assert!(!formats_have_image(&gdk::ContentFormats::new(&[
            "text/plain",
            "text/html"
        ])));
        assert!(!formats_have_image(&gdk::ContentFormats::new(&[
            "text/uri-list"
        ])));
    }

    #[test]
    fn detects_image_files_from_a_file_list() {
        assert!(file_is_image(&gio::File::for_path("/tmp/photo.PNG")));
        assert!(file_is_image(&gio::File::for_path("/tmp/photo.webp")));
        assert!(!file_is_image(&gio::File::for_path("/tmp/notes.txt")));
        assert!(!file_is_image(&gio::File::for_path("/tmp/archive")));
    }

    #[test]
    fn formats_small_percentages_with_one_decimal() {
        assert_eq!(format_percent(0.0), "0%");
        assert_eq!(format_percent(0.28), "0.3%");
        assert_eq!(format_percent(9.9), "9.9%");
        assert_eq!(format_percent(15.4), "15%");
    }

    #[test]
    fn paradee_offers_two_voices_per_english_accent() {
        let us = voice_options_for_backend("paradee", tts_language("en-us"));
        assert_eq!(us.len(), 2);
        assert_eq!(us[0].id, "af_heart");
        assert_eq!(us[1].id, "am_puck");

        let gb = voice_options_for_backend("paradee", tts_language("en-gb"));
        assert_eq!(gb.len(), 2);
        assert_eq!(gb[0].id, "bf_isabella");
        assert_eq!(gb[1].id, "bm_fable");
    }

    #[test]
    fn builtin_offers_female_and_male_per_language() {
        let spanish = voice_options_for_backend("builtin", tts_language("es"));
        assert_eq!(spanish.len(), 2);
        assert_eq!(spanish[0].id, "female-spanish");
        assert_eq!(spanish[1].id, "male-spanish");

        let auto_english = voice_options_for_backend("auto", tts_language("en-us"));
        assert_eq!(auto_english[0].id, "af_heart");
        let auto_spanish = voice_options_for_backend("auto", tts_language("es"));
        assert_eq!(auto_spanish[0].id, "female-spanish");
    }

    #[test]
    fn splits_sentences_on_punctuation() {
        assert_eq!(
            split_sentences("First sentence. Second one! Third?"),
            vec!["First sentence.", "Second one!", "Third?"]
        );
        assert_eq!(
            split_sentences("Dr. Smith went home. He slept."),
            vec!["Dr. Smith went home.", "He slept."]
        );
        assert_eq!(
            split_sentences("Pi is 3.14 today."),
            vec!["Pi is 3.14 today."]
        );
        assert_eq!(split_sentences("Wait... what?"), vec!["Wait... what?"]);
    }

    #[test]
    fn streaming_speech_keeps_incomplete_tail() {
        let (sentences, tail) = take_complete_sentences("First. Second");
        assert_eq!(sentences, vec!["First."]);
        assert_eq!(tail, "Second");

        let (sentences, tail) = take_complete_sentences("First. Second.");
        assert_eq!(sentences, vec!["First.", "Second."]);
        assert_eq!(tail, "");

        let (sentences, tail) = take_complete_sentences("Dr. Smith");
        assert!(sentences.is_empty());
        assert_eq!(tail, "Dr. Smith");

        let (sentences, tail) = take_complete_sentences("A complete paragraph.\n");
        assert_eq!(sentences, vec!["A complete paragraph."]);
        assert_eq!(tail, "");

        let (sentences, tail) = take_complete_sentences("Dr.");
        assert!(sentences.is_empty());
        assert_eq!(tail, "Dr.");
    }

    #[test]
    fn streamed_text_emits_sentences_as_chunks_arrive() {
        let mut buffer = SpeechTextBuffer::default();
        assert!(buffer.push("Hello").is_empty());
        assert_eq!(buffer.push(" world."), vec!["Hello world."]);
        assert!(buffer.push(" Next").is_empty());
        assert_eq!(buffer.push(" one!"), vec!["Next one!"]);
        assert!(buffer.finish().is_empty());

        let mut buffer = SpeechTextBuffer::default();
        assert!(buffer.push("Dr. Smith").is_empty());
        assert_eq!(buffer.push(" arrived."), vec!["Dr. Smith arrived."]);
    }

    #[test]
    fn streamed_text_preserves_words_split_across_chunks() {
        let mut buffer = SpeechTextBuffer::default();
        assert!(buffer.push("Hel").is_empty());
        assert_eq!(buffer.push("lo."), vec!["Hello."]);

        let mut buffer = SpeechTextBuffer::default();
        assert!(buffer.push("**Bold").is_empty());
        assert_eq!(buffer.push(" text**."), vec!["Bold text."]);
    }

    #[test]
    fn streamed_text_skips_fences_and_flushes_tail() {
        let mut buffer = SpeechTextBuffer::default();
        assert_eq!(buffer.push("Intro.\n```rust\nfn main"), vec!["Intro."]);
        assert_eq!(buffer.push("() {}\n```\nAfter.\n"), vec!["After."]);
        assert!(buffer.finish().is_empty());

        let mut buffer = SpeechTextBuffer::default();
        assert!(buffer.push("A final thought").is_empty());
        assert_eq!(buffer.finish(), vec!["A final thought"]);
    }

    #[test]
    fn speech_budget_stops_before_oversized_sentence() {
        let (chunks, truncated) = sentences_for_speech("One. Two. Three.", 9);
        assert_eq!(chunks, vec!["One.", "Two."]);
        assert!(truncated);
    }

    #[test]
    fn strips_markdown_and_symbols_for_speech() {
        let text = "## Heading\n**Bold** and *italic* with `code`.\n- [x] done ✅\n\n```rust\nfn main() {}\n```\nSee [the docs](https://example.com) & 50%.";
        assert_eq!(
            plain_text_for_speech(text),
            "Heading\nBold and italic with code.\ndone\nSee the docs 50."
        );
        assert_eq!(plain_text_for_speech("Café — naïve → 🚀"), "Café — naïve");
        assert_eq!(
            sentences_for_speech("## Hi\n**There** — friend.", 4096).0,
            vec!["Hi", "There — friend."]
        );
    }

    #[test]
    fn decodes_16_bit_wav() {
        let wav = wav_bytes(&[0, 16384, -16384], 24_000);
        let (samples, sample_rate, channels) = decode_wav(&wav).unwrap();
        assert_eq!(sample_rate, 24_000);
        assert_eq!(channels, 1);
        assert_eq!(samples.len(), 3);
        assert!(samples[0].abs() < 0.001);
        assert!((samples[1] - 0.5).abs() < 0.001);
        assert!((samples[2] + 0.5).abs() < 0.001);
    }

    #[test]
    #[ignore = "requires an audio output device"]
    fn rodio_plays_silence() {
        let stream = rodio::OutputStreamBuilder::open_default_stream().expect("audio output");
        let sink = rodio::Sink::connect_new(stream.mixer());
        sink.append(rodio::buffer::SamplesBuffer::new(
            1,
            24_000,
            vec![0.0f32; 4_800],
        ));
        let started = std::time::Instant::now();
        while !sink.empty() {
            assert!(
                started.elapsed() < std::time::Duration::from_secs(3),
                "rodio playback did not finish"
            );
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }

    #[test]
    #[ignore = "requires a local Ollama speech model"]
    fn live_speech_decodes_to_samples() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let client = crate::api::OllamaClient::new("http://localhost:11434", "");
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
            let (samples, sample_rate, channels) = decode_wav(&audio).expect("decode wav");
            assert_eq!(channels, 1);
            assert_eq!(sample_rate, 24_000);
            assert!(samples.len() > 1_000);
        });
    }

    fn stored_message(id: i64, role: &str, content: &str) -> StoredMessage {
        StoredMessage {
            id,
            role: role.to_string(),
            content: content.to_string(),
        }
    }

    #[test]
    fn keeps_only_the_last_five_turns() {
        let mut messages = Vec::new();
        for turn in 0..7 {
            messages.push(stored_message(turn * 2 + 1, "user", &format!("u{turn}")));
            messages.push(stored_message(
                turn * 2 + 2,
                "assistant",
                &format!("a{turn}"),
            ));
        }

        assert_eq!(kept_turn_start(&messages, 5), 4);
        assert_eq!(kept_turn_start(&messages[..6], 5), 0);
        assert_eq!(kept_turn_start(&[], 5), 0);
    }

    #[test]
    fn builds_a_request_with_a_summary() {
        let history = vec![
            stored_message(1, "user", "old question"),
            stored_message(2, "assistant", "old answer"),
            stored_message(3, "user", "new question"),
            stored_message(4, "assistant", "new answer"),
        ];
        let summary = StoredSummary {
            text: "They talked about old things.".into(),
            upto_id: 2,
            tokens: 8,
        };

        let request = build_request_messages(
            "Be concise.",
            None,
            Some(&summary),
            &history,
            &HashMap::new(),
            8192,
        );
        assert_eq!(request.len(), 4);
        assert_eq!(request[0].role, "system");
        assert_eq!(request[0].content, "Be concise.");
        assert_eq!(request[1].role, "system");
        assert!(request[1].content.contains("old things"));
        assert_eq!(request[2].content, "new question");
        assert_eq!(request[3].content, "new answer");
    }

    #[test]
    fn builds_a_request_with_memory_context() {
        let history = vec![stored_message(1, "user", "new question")];
        let request = build_request_messages(
            "Be concise.",
            Some("Relevant long-term memory:\n- likes dark mode"),
            None,
            &history,
            &HashMap::new(),
            8192,
        );
        assert_eq!(request.len(), 3);
        assert_eq!(request[0].content, "Be concise.");
        assert_eq!(request[1].role, "system");
        assert!(request[1].content.contains("dark mode"));
        assert_eq!(request[2].content, "new question");
    }

    #[test]
    fn request_includes_message_images() {
        let history = vec![stored_message(1, "user", "what is this?")];
        let mut images = HashMap::new();
        images.insert(
            1,
            vec![StoredImage {
                mime: "image/png".into(),
                width: 1,
                height: 1,
                data: b"abc".to_vec(),
            }],
        );

        let request = build_request_messages("", None, None, &history, &images, 8192);
        assert_eq!(request.len(), 1);
        assert_eq!(request[0].images, vec!["YWJj".to_string()]);
    }

    #[test]
    fn estimates_tokens_for_ascii_and_cjk() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("hello world"), 3);
        assert_eq!(estimate_tokens("日本語"), 3);
    }

    #[test]
    fn triggers_compression_at_eighty_five_percent() {
        assert!(!context_needs_compression(8_499, 10_000));
        assert!(context_needs_compression(8_500, 10_000));
        assert!(context_needs_compression(8_501, 10_000));
    }

    #[test]
    fn request_trim_keeps_the_last_user_turn() {
        let history: Vec<StoredMessage> = (0..20)
            .map(|index| {
                stored_message(
                    index,
                    if index % 2 == 0 { "user" } else { "assistant" },
                    &"word ".repeat(200),
                )
            })
            .collect();

        let request = build_request_messages("", None, None, &history, &HashMap::new(), 512);
        assert_eq!(request.iter().filter(|m| m.role == "user").count(), 1);
        assert_eq!(request.last().map(|m| m.role.as_str()), Some("assistant"));
    }

    #[test]
    fn plans_compression_only_when_needed() {
        let mut messages = Vec::new();
        for turn in 0..7 {
            messages.push(stored_message(
                turn * 2 + 1,
                "user",
                &format!("question {turn} {}", "word ".repeat(50)),
            ));
            messages.push(stored_message(
                turn * 2 + 2,
                "assistant",
                &format!("answer {turn} {}", "word ".repeat(50)),
            ));
        }

        let plan =
            build_compression_plan("model".into(), 8192, "", None, &messages, None, Some(8_500))
                .expect("compression plan");
        assert_eq!(plan.upto_id, 4);
        assert!(!plan.chunks.is_empty());

        assert!(build_compression_plan(
            "model".into(),
            8192,
            "",
            None,
            &messages,
            None,
            Some(6_000)
        )
        .is_none());

        let summary = StoredSummary {
            text: "old".into(),
            upto_id: 4,
            tokens: 3,
        };
        assert!(build_compression_plan(
            "model".into(),
            8192,
            "",
            None,
            &messages,
            Some(&summary),
            Some(9_000),
        )
        .is_none());

        let short = &messages[..8];
        assert!(
            build_compression_plan("model".into(), 8192, "", None, short, None, Some(9_000))
                .is_none()
        );
    }

    #[test]
    fn clamps_long_summaries() {
        let clamped = clamp_summary("One. Two. Three. Four.", 2);
        assert_eq!(clamped, "One.");
    }

    #[test]
    fn splits_long_histories_into_chunks() {
        let messages: Vec<StoredMessage> = (0..6)
            .map(|index| stored_message(index, "user", &"word ".repeat(40)))
            .collect();
        let refs: Vec<&StoredMessage> = messages.iter().collect();
        assert!(chunk_messages(&refs, 40).len() > 1);
    }

    #[test]
    #[ignore = "requires a local Ollama daemon and a chat model"]
    fn compresses_older_turns_with_ollama() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let client = crate::api::OllamaClient::new("http://localhost:11434", "");
            let models = client.list_models().await.expect("list models");
            let model = models
                .iter()
                .find(|model| model.is_chat_model())
                .map(|model| model.name.clone())
                .expect("no chat-capable models installed");

            let plan = super::CompressionPlan {
                model,
                num_ctx: 4096,
                existing: None,
                upto_id: 42,
                chunks: vec![
                    "User: My name is Ada and I prefer tea.\nAssistant: Noted, Ada.\n".to_string(),
                ],
            };
            let result = super::run_compression(client, plan)
                .await
                .expect("summarize older turns");
            println!("summary: {}", result.text);
            assert!(!result.text.trim().is_empty());
            assert_eq!(result.upto_id, 42);
        });
    }

    fn wav_bytes(samples: &[i16], sample_rate: u32) -> Vec<u8> {
        let data_length = (samples.len() * 2) as u32;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + data_length).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes()); // PCM
        bytes.extend_from_slice(&1u16.to_le_bytes()); // mono
        bytes.extend_from_slice(&sample_rate.to_le_bytes());
        bytes.extend_from_slice(&(sample_rate * 2).to_le_bytes());
        bytes.extend_from_slice(&2u16.to_le_bytes());
        bytes.extend_from_slice(&16u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data_length.to_le_bytes());
        for sample in samples {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        bytes
    }
}
