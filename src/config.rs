use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub base_url: String,
    pub api_key: String,
    pub default_model: String,
    pub system_prompt: String,
    pub temperature: f64,
    pub num_ctx: u32,
    /// Ask thinking-capable models to emit their reasoning
    pub think_enabled: bool,
    /// Reasoning level for models that expose named thinking levels
    pub think_level: String,
    /// "system", "light" or "dark"
    pub theme: String,
    /// Whether the conversation sidebar is shown
    pub sidebar_visible: bool,
    /// "auto", "paradee" or "builtin"
    pub tts_backend: String,
    /// Language code such as "en-us" or "es"
    pub tts_language: String,
    /// Voice id for the selected TTS backend
    pub tts_voice: String,
    /// Enable text-to-speech buttons and auto-reading
    pub tts_enabled: bool,
    /// Read assistant replies automatically
    pub tts_autoread: bool,
    /// Enable the microphone/dictation button
    pub stt_enabled: bool,
    /// "auto", "whistle" or "builtin"
    pub stt_provider: String,
    /// "auto" or a language code
    pub stt_language: String,
    /// Seconds to wait for speech before stopping
    pub stt_start_timeout_secs: u32,
    /// Silence in milliseconds that ends an utterance
    pub stt_end_silence_ms: u32,
    /// Send the transcript to the model automatically
    pub stt_autosend: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:11434".to_string(),
            api_key: String::new(),
            default_model: String::new(),
            system_prompt: String::new(),
            temperature: 0.7,
            num_ctx: 8192,
            think_enabled: true,
            think_level: "medium".to_string(),
            theme: "system".to_string(),
            sidebar_visible: true,
            tts_backend: "auto".to_string(),
            tts_language: "en-us".to_string(),
            tts_voice: "af_heart".to_string(),
            tts_enabled: true,
            tts_autoread: false,
            stt_enabled: false,
            stt_provider: "auto".to_string(),
            stt_language: "auto".to_string(),
            stt_start_timeout_secs: 5,
            stt_end_silence_ms: 1000,
            stt_autosend: false,
        }
    }
}

pub fn config_dir() -> PathBuf {
    std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| home().join(".config"))
        .join("ollama-gui")
}

pub fn data_dir() -> PathBuf {
    std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| home().join(".local/share"))
        .join("ollama-gui")
}

fn home() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

impl Settings {
    fn path() -> PathBuf {
        config_dir().join("settings.json")
    }

    pub fn load() -> Self {
        let path = Self::path();
        match fs::read_to_string(&path) {
            Ok(contents) => serde_json::from_str(&contents).unwrap_or_else(|err| {
                tracing::warn!("invalid settings at {}: {err}", path.display());
                Self::default()
            }),
            Err(_) => Self::default(),
        }
    }

    pub fn save(&self) -> anyhow::Result<()> {
        let dir = config_dir();
        fs::create_dir_all(&dir)?;
        let path = Self::path();
        fs::write(&path, serde_json::to_string_pretty(self)?)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&path)?.permissions();
            perms.set_mode(0o600);
            fs::set_permissions(&path, perms)?;
        }
        Ok(())
    }
}
