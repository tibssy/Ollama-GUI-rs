use anyhow::{bail, Context, Result};
use ort::session::Session;
use ort::value::Tensor;
use std::collections::HashMap;
use std::path::Path;
use std::process::Command;

const MODEL_URL: &str =
    "https://huggingface.co/sahilmahendrakar/Paradee-8M-v1.2/resolve/main/onnx/paradee_int8.onnx";
const CONFIG_URL: &str =
    "https://huggingface.co/sahilmahendrakar/Paradee-8M-v1.2/resolve/main/config.json";
const MODEL_MIN_BYTES: u64 = 8_000_000;
const MAX_PHONEMES: usize = 510;

static DOWNLOAD_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// Kokoro/Paradee phoneme vocabulary mapping used to decode eSpeak output into
/// the phoneme spelling the model was trained on. Ported from Paradee's
/// `web/misaki.js` (Apache-2.0).
const E2M: &[(&str, &str)] = &[
    ("ʔˌn̩", "ʔn"),
    ("ʔn̩", "ʔn"),
    ("aɪ", "I"),
    ("aʊ", "W"),
    ("dʒ", "ʤ"),
    ("eɪ", "A"),
    ("e", "A"),
    ("tʃ", "ʧ"),
    ("ɔɪ", "Y"),
    ("ʲo", "jo"),
    ("ʲə", "jə"),
    ("ʲ", ""),
    ("ɚ", "əɹ"),
    ("r", "ɹ"),
    ("x", "k"),
    ("ç", "k"),
    ("ɬ", "l"),
    ("̃", ""),
];

#[derive(Debug, serde::Deserialize)]
struct ParadeeConfig {
    sample_rate: u32,
    voices: Vec<String>,
    vocab: HashMap<String, i64>,
}

pub fn model_dir() -> std::path::PathBuf {
    crate::config::data_dir().join("models").join("paradee")
}

fn model_path() -> std::path::PathBuf {
    model_dir().join("paradee_int8.onnx")
}

fn config_path() -> std::path::PathBuf {
    model_dir().join("config.json")
}

pub fn is_downloaded() -> bool {
    let model_ready = std::fs::metadata(model_path())
        .map(|meta| meta.len() >= MODEL_MIN_BYTES)
        .unwrap_or(false);
    model_ready && config_path().exists()
}

/// Remove the downloaded voice files. The caller must unload the engine first
/// when it may still be in use.
pub fn delete() -> Result<()> {
    let dir = model_dir();
    match std::fs::remove_dir_all(&dir) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err).with_context(|| format!("removing {}", dir.display())),
    }
}

pub async fn ensure_downloaded() -> Result<()> {
    let _guard = DOWNLOAD_LOCK.lock().await;
    if is_downloaded() {
        return Ok(());
    }

    let dir = model_dir();
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(180))
        .build()
        .context("creating download client")?;

    download(&client, MODEL_URL, &model_path(), MODEL_MIN_BYTES).await?;
    download(&client, CONFIG_URL, &config_path(), 100).await?;

    if !is_downloaded() {
        bail!("Paradee model download did not produce usable files");
    }
    Ok(())
}

async fn download(client: &reqwest::Client, url: &str, path: &Path, min_bytes: u64) -> Result<()> {
    if std::fs::metadata(path)
        .map(|meta| meta.len() >= min_bytes)
        .unwrap_or(false)
    {
        return Ok(());
    }

    let response = client
        .get(url)
        .send()
        .await
        .with_context(|| format!("downloading {url}"))?;
    if !response.status().is_success() {
        bail!("downloading {url} returned HTTP {}", response.status());
    }
    let bytes = response
        .bytes()
        .await
        .with_context(|| format!("reading {url}"))?;
    if bytes.len() as u64 > 0 && (bytes.len() as u64) < min_bytes {
        bail!(
            "downloaded {url} is too small ({} bytes, expected at least {min_bytes})",
            bytes.len()
        );
    }

    let temp = path.with_extension("part");
    std::fs::write(&temp, &bytes).with_context(|| format!("writing {}", temp.display()))?;
    std::fs::rename(&temp, path)
        .with_context(|| format!("moving {} to {}", temp.display(), path.display()))?;
    Ok(())
}

pub struct Paradee {
    session: Session,
    vocab: HashMap<char, i64>,
    voices: Vec<String>,
    sample_rate: u32,
}

impl Paradee {
    pub fn load() -> Result<Self> {
        let config: ParadeeConfig = serde_json::from_str(
            &std::fs::read_to_string(config_path())
                .with_context(|| format!("reading {}", config_path().display()))?,
        )
        .context("parsing Paradee config.json")?;

        let mut vocab = HashMap::new();
        for (key, id) in config.vocab {
            if let Some(character) = key.chars().next() {
                vocab.insert(character, id);
            }
        }

        let session = Session::builder()
            .map_err(|error| anyhow::anyhow!("creating ONNX session builder: {error}"))?
            .with_intra_threads(2)
            .map_err(|error| anyhow::anyhow!("configuring ONNX session: {error}"))?
            .commit_from_file(model_path())
            .with_context(|| format!("loading {}", model_path().display()))?;

        Ok(Self {
            session,
            vocab,
            voices: config.voices,
            sample_rate: config.sample_rate,
        })
    }

    pub fn synthesize(&mut self, text: &str, voice: &str) -> Result<(Vec<f32>, u32)> {
        let voice_index = self
            .voices
            .iter()
            .position(|candidate| candidate == voice)
            .ok_or_else(|| anyhow::anyhow!("Paradee has no voice named '{voice}'"))?;

        let phonemes = phonemize(text, voice.starts_with('b'))?;
        let mut audio = Vec::new();
        for chunk in chunk_phonemes(&phonemes, MAX_PHONEMES) {
            audio.extend(self.run_chunk(&chunk, voice_index as i64)?);
        }
        if audio.is_empty() {
            bail!("Paradee produced no audio for this text");
        }
        Ok((audio, self.sample_rate))
    }

    fn run_chunk(&mut self, phonemes: &str, voice: i64) -> Result<Vec<f32>> {
        let mut ids: Vec<i64> = Vec::with_capacity(phonemes.chars().count() + 2);
        ids.push(0);
        ids.extend(
            phonemes
                .chars()
                .filter(|character| *character == ' ' || self.vocab.contains_key(character))
                .filter_map(|character| self.vocab.get(&character).copied()),
        );
        ids.push(0);
        if ids.len() > MAX_PHONEMES + 2 {
            bail!("phoneme sequence is too long");
        }

        let input_ids = Tensor::from_array(([1usize, ids.len()], ids))?;
        let speed = Tensor::from_array(([1usize], vec![1.0f32]))?;
        let voice = Tensor::from_array(([1usize], vec![voice]))?;

        let outputs = self.session.run(ort::inputs![
            "input_ids" => input_ids,
            "speed" => speed,
            "voice" => voice,
        ])?;
        let (_, samples) = outputs[0].try_extract_tensor::<f32>()?;
        Ok(samples.to_vec())
    }
}

fn chunk_phonemes(phonemes: &str, max: usize) -> Vec<String> {
    let chars: Vec<char> = phonemes.chars().collect();
    let mut chunks = Vec::new();
    let mut start = 0usize;

    while start < chars.len() {
        let end = (start + max).min(chars.len());
        if end == chars.len() {
            let chunk: String = chars[start..].iter().collect();
            if !chunk.trim().is_empty() {
                chunks.push(chunk.trim().to_string());
            }
            break;
        }

        let cut = (start..end)
            .rev()
            .find(|index| chars[*index] == ' ')
            .unwrap_or(end);
        let cut = if cut > start { cut } else { end };
        let chunk: String = chars[start..cut].iter().collect();
        if !chunk.trim().is_empty() {
            chunks.push(chunk.trim().to_string());
        }
        start = cut;
        while start < chars.len() && chars[start] == ' ' {
            start += 1;
        }
    }

    chunks
}

fn phonemize(text: &str, british: bool) -> Result<String> {
    let language = if british { "en-gb" } else { "en-us" };
    let output = Command::new("espeak-ng")
        .args(["-q", "--ipa", "-v", language, text])
        .output()
        .context("espeak-ng is required for Paradee; install the espeak-ng package")?;

    if !output.status.success() {
        bail!(
            "espeak-ng failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }

    let raw = String::from_utf8_lossy(&output.stdout);
    let joined = raw.split_whitespace().collect::<Vec<_>>().join(" ");
    Ok(to_misaki(&joined, british))
}

fn to_misaki(phonemes: &str, british: bool) -> String {
    let mut result = phonemes.to_string();

    // British eSpeak writes "eə" where misaki uses ɛː; this must run before the
    // generic e -> A rule below.
    if british {
        result = result.replace("eə", "ɛː");
    }
    for (from, to) in E2M {
        result = result.replace(from, to);
    }

    // eSpeak's syllabic consonant marker is a combining vertical line below.
    // misaki uses ᵊ before the consonant instead.
    let chars: Vec<char> = result.chars().collect();
    let mut folded = String::with_capacity(result.len());
    let mut index = 0usize;
    while index < chars.len() {
        if index + 1 < chars.len()
            && chars[index + 1] == '\u{0329}'
            && !chars[index].is_whitespace()
        {
            folded.push('ᵊ');
            folded.push(chars[index]);
            index += 2;
        } else if chars[index] == '\u{0329}' {
            index += 1;
        } else {
            folded.push(chars[index]);
            index += 1;
        }
    }
    result = folded;

    if british {
        result = result.replace("iə", "ɪə").replace("əʊ", "Q");
    } else {
        result = result
            .replace("oʊ", "O")
            .replace("ɜːɹ", "ɜɹ")
            .replace("ɜː", "ɜɹ")
            .replace("ɪə", "iə")
            .replace('ː', "");
    }
    result = result.replace('o', "ɔ");
    result = result.replace('ɾ', "T").replace('ʔ', "t");

    let mut tokens: Vec<String> = result.split_whitespace().map(str::to_string).collect();
    for token in &mut tokens {
        // misaki writes a word-final syllabic l as ᵊl (naval -> nˈAvᵊl).
        if let Some(prefix) = token.strip_suffix("əl") {
            *token = format!("{prefix}ᵊl");
        }

        // "a" (and, in British, "an") keep their ɐ; other ɐ become ə.
        let stripped = token.trim_start_matches(['ˈ', 'ˌ']);
        let keep_article = if british {
            stripped == "ɐ" || stripped.starts_with("ɐn")
        } else {
            stripped == "ɐ"
        };
        if !keep_article && token.contains('ɐ') {
            *token = token.replace('ɐ', "ə");
        }
    }

    tokens.join(" ")
}

#[cfg(test)]
mod tests {
    use super::{chunk_phonemes, ensure_downloaded, to_misaki, Paradee};

    #[test]
    fn maps_american_vowels_to_misaki() {
        assert_eq!(to_misaki("həlˈoʊ wˈɜːld", false), "həlˈO wˈɜɹld");
        assert_eq!(to_misaki("ðɪs ɪz ɐ tˈɛst", false), "ðɪs ɪz ɐ tˈɛst");
    }

    #[test]
    fn maps_british_vowels_to_misaki() {
        assert_eq!(to_misaki("həlˈəʊ wˈɜːld", true), "həlˈQ wˈɜːld");
        assert_eq!(to_misaki("nˈeɪvəl", true), "nˈAvᵊl");
    }

    #[test]
    fn chunks_long_phoneme_runs_at_spaces() {
        let phonemes = "aa bb cc dd ee ff gg hh";
        let chunks = chunk_phonemes(phonemes, 10);
        assert!(chunks.len() > 1);
        assert!(chunks.iter().all(|chunk| chunk.chars().count() <= 10));
        assert_eq!(chunks.join(" "), phonemes);
    }

    #[test]
    #[ignore = "downloads the Paradee model and requires espeak-ng"]
    fn paradee_synthesizes_audio() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime
            .block_on(ensure_downloaded())
            .expect("download model");

        let mut paradee = Paradee::load().expect("load model");
        let (samples, sample_rate) = paradee
            .synthesize("Hello from Paradee.", "af_heart")
            .expect("synthesize");
        assert_eq!(sample_rate, 24_000);
        assert!(samples.len() > 4_000, "audio is suspiciously short");
        assert!(
            samples.iter().any(|sample| sample.abs() > 0.001),
            "audio is silent"
        );
    }
}
